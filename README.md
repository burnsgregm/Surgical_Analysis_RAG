# Surgical Analysis AI — Proof of Concept (V2)

> **POC disclaimer:** This repository demonstrates a research prototype that analyzes a narrated surgical video against textbook knowledge to highlight potential inconsistencies. **Not for clinical use.** No PHI/PII; single-user demo.

---

## What it does

- **Upload a surgical video** via a simple web UI.
- **Extract audio** (ffmpeg) ➜ **transcribe** with **Gemini 1.5 Flash**.
- **Sample frames** (~every 5s) ➜ **caption** with local **LLaVA-1.5-7B (4-bit)** on GPU.
- **Enrich/Retrieve**: Build a query from transcript + visual timeline, **extract entities** with a fine-tuned NER model, **retrieve** top-k chunks from a **Chroma** vector DB (Sentence-BERT embeddings).
- **Compose analysis** with Gemini using a user-editable system prompt.
- **Return JSON** (transcript, timeline, retrieved context, final analysis) and render analysis as Markdown in the browser.

---

## Repo layout

```
.
├─ main_app.py                 # FastAPI app serving the web UI and /analyze_video
├─ index.html                  # Web UI (place under ./static/ for V2 runtime)
├─ 1_ingest_knowledge_base.py  # Build enriched Chroma DB from PDF textbooks
├─ 3_train_ner_model.py        # Fine-tune DistilBERT NER (INSTRUMENT/ANATOMY/ACTION/OBSERVATION)
```

> **Note:** `main_app.py` serves `./static/index.html`. If your `index.html` is in repo root, move it to `./static/index.html`.

---

## Requirements

- **Python** 3.10+  
- **GPU (CUDA)** recommended (e.g., RTX A5000)  
  - LLaVA-1.5-7B (4-bit) + embeddings require a CUDA-capable GPU; VRAM usage depends on drivers/libs.
- **ffmpeg** available on PATH  
- **Google Generative AI key**: `GOOGLE_API_KEY` environment variable (Gemini 1.5)

### Suggested Python deps

If you don’t keep a `requirements.txt`, install the following (adjust versions to your environment):

```bash
pip install fastapi uvicorn[standard] python-multipart
pip install google-generativeai
pip install transformers accelerate bitsandbytes
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick your CUDA
pip install sentence-transformers chromadb langchain langchain-community
pip install opencv-python-headless pillow pymupdf datasets evaluate seqeval
```

---

## Setup

1. **Add textbooks**  
   Put your PDFs under `./textbooks/`.

2. **(Optional) Train NER (DistilBERT)**  
   Prepare `ner_training_data.json` with entries like:
   ```json
   [
     {
       "text": "Incision made with scalpel at the midline...",
       "labels": [
         {"start": 0, "end": 8, "label": "ACTION"},
         {"start": 18, "end": 25, "label": "INSTRUMENT"}
       ]
     }
   ]
   ```
   Then:
   ```bash
   python 3_train_ner_model.py
   ```
   Output goes to `./surgical_ner_model/`.

3. **Build the enriched vector DB**  
   ```bash
   python 1_ingest_knowledge_base.py
   ```
   This loads PDFs (`pymupdf`), splits (size=1000, overlap=150), runs NER to attach entity metadata, embeds with `all-MiniLM-L6-v2` (GPU), and persists the Chroma DB to `./vector_db_enriched/`.

4. **Place the web UI**  
   Ensure the UI is at `./static/index.html` (the app serves `/` from there).

5. **Environment**  
   ```bash
   export GOOGLE_API_KEY="your_gemini_api_key"
   ffmpeg -version   # verify it's installed
   ```

---

## Run (V2)

```bash
python main_app.py
# → http://localhost:8000
```

- Upload a video, edit the **System Prompt** and **User Prompt**, and click **Analyze**.
- The app will:
  - Save the file under `./uploads/`
  - Extract audio (`ffmpeg`)
  - Transcribe via Gemini 1.5 Flash
  - Sample frames (~every 5s) and caption via LLaVA-1.5-7B (4-bit)
  - Pull context from `./vector_db_enriched/`
  - Ask Gemini to compose the final analysis
  - Return JSON; UI renders Markdown + raw JSON

---

## API

### `POST /analyze_video` (multipart/form-data)

**Fields**
- `video_file`: file (mp4/mov/etc.)
- `system_prompt`: string (sets Gemini system instruction)
- `user_prompt`: string (task instruction, e.g., “Compare narration against references and list inconsistencies with citations.”)

**200 OK — JSON**
```json
{
  "video_source": "sample.mp4",
  "full_audio_transcript": "Verbatim transcript ...",
  "visual_timeline_events": [
    {"timestamp": 0.0, "description": "Scalpel visible; incision at midline ..."},
    {"timestamp": 5.1, "description": "Forceps and retractor in view ..."}
  ],
  "final_analysis": {
    "analysis": "## Findings\\n- ...\\n\\n### Citations\\n- ...",
    "retrieved_context": [
      {"source": "Textbook A p.123", "chunk": "...", "metadata": {"INSTRUMENT": ["scalpel"], "ACTION": ["incision"]}}
    ]
  }
}
```

**curl example**
```bash
curl -sS -X POST http://localhost:8000/analyze_video \
  -F "video_file=@/path/to/sample.mp4" \
  -F 'system_prompt=You are a surgical QA assistant that only uses the provided references.' \
  -F 'user_prompt=Compare the narration to best practices. List inconsistencies and cite retrieved sources.'
```

---

## Optional: Lightweight RAG API (JSON-in/JSON-out)

A separate prototype service that accepts a transcript + visual events and returns a concise check.

Run:
```bash
python 2_rag_api_server.py
# → http://localhost:8000
```

Request (excerpt):
```json
{
  "video_source": "sample.mp4",
  "full_audio_transcript": "....",
  "visual_timeline_events": [
    {"timestamp": 5.1, "description": "Scalpel ..."}
  ]
}
```

---

## Configuration (key paths in V2)

- `main_app.py`
  - `STATIC_DIR = "static"`
  - `UPLOADS_DIR = "uploads"`
  - `VECTOR_DB_DIR = "vector_db_enriched/"`
  - `NER_MODEL_PATH = "surgical_ner_model/"`
  - LLaVA model: `llava-hf/llava-1.5-7b-hf` (4-bit load)
  - Embeddings: `all-MiniLM-L6-v2` (GPU)
  - Port: `8000`

---

## Troubleshooting

- **`GOOGLE_API_KEY not set`** → export the key, or set in your service env.
- **`Enriched vector database not found`** → run `python 1_ingest_knowledge_base.py`.
- **ffmpeg errors** → install ffmpeg and ensure it’s on PATH.
- **CUDA OOM / model load issues**
  - Close other GPU apps; ensure proper CUDA/PyTorch versions.
  - The app loads LLaVA in 4-bit (`bitsandbytes`) to reduce VRAM.
- **Non-JSON 500s** → check `./uploads/`, verify input video and size; check logs/console.

---

## Roadmap (V3 ideas — not in V2)

- `202 Accepted` + background jobs + status polling/SSE.
- Progress events and cancelation (AbortController).
- Sanitized Markdown render in UI.
- Unified config (one source of truth) for vector DB and models.
- Multi-event retrieval & structured evaluation metrics.

---

## Acknowledgments

- **Gemini 1.5** (Google Generative AI) for transcription & analysis.
- **LLaVA-1.5-7B** for frame captioning.
- **Sentence-Transformers** + **Chroma** for retrieval.
- **LangChain community** loaders/splitters for ingestion.
