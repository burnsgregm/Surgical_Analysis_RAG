import os
import subprocess
import cv2
import torch
import json
import uuid
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, pipeline
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# --- 1. GLOBAL CONFIGURATION & SETUP ---

STATIC_DIR = "static"
UPLOADS_DIR = "uploads"
TEXTBOOKS_DIR = "textbooks"
VECTOR_DB_DIR = "vector_db_enriched/"
NER_MODEL_PATH = "surgical_ner_model/"

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# --- FastAPI App Initialization ---

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Globals for AI Models (Lazy Loaded) ---

embedding_model = None
vector_db = None
llava_model = None
llava_processor = None
gemini_model_cache = {}
ner_pipeline = None

# --- 2. HELPER FUNCTIONS FOR LAZY-LOADING AI MODELS ---

def get_embedding_model():
    """Loads the sentence embedding model into memory, only once."""
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model...")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'}
        )
    return embedding_model

def get_vector_db():
    """Loads the Chroma vector database from disk, only once."""
    global vector_db
    if vector_db is None:
        print("Loading vector database...")
        from langchain_community.vectorstores import Chroma
        if not os.path.exists(VECTOR_DB_DIR):
            raise RuntimeError("Enriched vector database not found. Please run the ingestion script first.")
        vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=get_embedding_model()
        )
    return vector_db

def get_llava_model():
    """Loads the local LLaVA model and its processor, only once."""
    global llava_model, llava_processor
    if llava_model is None:
        print("Loading LLaVA model...")
        model_id = "llava-hf/llava-1.5-7b-hf"
        llava_model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True
        )
        llava_processor = AutoProcessor.from_pretrained(model_id)
    return llava_model, llava_processor

def get_gemini_model(system_prompt):
    """Initializes a Gemini model with a specific system prompt, caching it for reuse."""
    if system_prompt not in gemini_model_cache:
        print(f"Initializing new Gemini model for system prompt...")
        try:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            gemini_model_cache[system_prompt] = genai.GenerativeModel(
                'gemini-1.5-flash',
                system_instruction=system_prompt
            )
        except KeyError:
            raise RuntimeError("GOOGLE_API_KEY not set.")
    return gemini_model_cache[system_prompt]

def get_ner_pipeline():
    """Loads the fine-tuned NER model into a pipeline, only once."""
    global ner_pipeline
    if ner_pipeline is None:
        print("Loading fine-tuned NER model...")
        ner_pipeline = pipeline(
            "ner", 
            model=NER_MODEL_PATH, 
            aggregation_strategy="simple"
        )
    return ner_pipeline


# --- 3. CORE LOGIC FUNCTIONS ---

def transcribe_audio_gemini(audio_path):
    """Transcribes an audio file using the Gemini API."""
    print("Transcribing audio with Gemini...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        audio_file = genai.upload_file(path=audio_path)
        response = model.generate_content(["Please provide a verbatim transcript of this audio.", audio_file])
        genai.delete_file(audio_file.name)
        return response.text
    except Exception as e:
        print(f"Error during Gemini transcription: {e}")
        return None

def analyze_video_frames(video_path):
    """Analyzes video frames with the local LLaVA model."""
    print("Analyzing video frames with LLaVA...")
    model, processor = get_llava_model()
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 5)
    visual_descriptions = []
    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret: break
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            print(f"  - Analyzing frame at {timestamp:.2f}s...")
            try:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                prompt = "USER: <image>\nDescribe the surgical action in this frame in detail. What instruments are visible? ASSISTANT:"
                inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to("cuda", torch.float16)
                output = model.generate(**inputs, max_new_tokens=100)
                description = processor.decode(output[0], skip_special_tokens=True)
                cleaned_description = description.split("ASSISTANT:")[-1].strip()
                visual_descriptions.append({"timestamp": timestamp, "description": cleaned_description})
            except Exception as e:
                print(f"    - Error analyzing frame: {e}")
        frame_count += 1
    video.release()
    return visual_descriptions

def run_rag_analysis(transcript, visual_data, system_prompt, user_prompt):
    """Runs the RAG analysis by retrieving context and generating a final response."""
    print("Running RAG analysis with enriched metadata...")
    vector_db = get_vector_db()
    gemini_model = get_gemini_model(system_prompt)
    ner_pipeline = get_ner_pipeline()
    
    if not visual_data:
        raise HTTPException(status_code=400, detail="Could not extract visual information.")

    all_visual_descriptions = "\n".join([f"- At {item['timestamp']:.2f}s: {item['description']}" for item in visual_data])
    query = f"Visual Events Summary:\n{all_visual_descriptions}\n\nFull Audio Transcript:\n{transcript}"
    
    entities = ner_pipeline(query)
    filter_dict = {}
    if entities:
        for entity in entities:
            label = entity['entity_group']
            word = entity['word']
            if label not in filter_dict:
                filter_dict[label] = word
        print(f"Created metadata filter from NER: {filter_dict}")

    print("Retrieving documents from vector store using hybrid search...")
    retriever = vector_db.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": filter_dict
        }
    )
    retrieved_docs = retriever.invoke(query)
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    
    print("Generating final analysis with Gemini...")
    final_user_prompt = f"""
    **Textbook Context:**
    {context}

    **Observed Data:**
    - Visual Events Timeline: {all_visual_descriptions}
    - Full Audio Transcript: {transcript}
    
    **User's Task:**
    {user_prompt}
    """
    
    response = gemini_model.generate_content(final_user_prompt)
    serializable_docs = [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs
    ]
    
    return {
        "analysis": response.text,
        "retrieved_context": serializable_docs
    }

# --- 4. API ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serves the main HTML front-end file."""
    try:
        with open(os.path.join(STATIC_DIR, "index.html")) as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found.")

@app.post("/analyze_video")
async def analyze_video_endpoint(
    video_file: UploadFile = File(...), 
    system_prompt: str = Form(...),
    user_prompt: str = Form(...)
):
    """Main API endpoint to handle the full end-to-end analysis pipeline."""
    video_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOADS_DIR, f"{video_id}_{video_file.filename}")
    audio_path = os.path.join(UPLOADS_DIR, f"{video_id}.aac")
    
    with open(video_path, "wb") as buffer:
        buffer.write(await video_file.read())
        
    try:
        command = f"ffmpeg -i '{video_path}' -vn -acodec copy -y '{audio_path}'"
        subprocess.run(command, check=True, shell=True, capture_output=True)
        
        full_transcript = transcribe_audio_gemini(audio_path)
        if not full_transcript:
            raise HTTPException(status_code=500, detail="Audio transcription failed.")
            
        visual_descriptions = analyze_video_frames(video_path)
        
        rag_result = run_rag_analysis(full_transcript, visual_descriptions, system_prompt, user_prompt)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(video_path): os.remove(video_path)
        if os.path.exists(audio_path): os.remove(audio_path)
        
    return {
        "video_source": video_file.filename,
        "full_audio_transcript": full_transcript,
        "visual_timeline_events": visual_descriptions,
        "final_analysis": rag_result
    }

# --- 5. MAIN ENTRY POINT ---

if __name__ == "__main__":
    print("Starting Surgical Analysis Web App on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)