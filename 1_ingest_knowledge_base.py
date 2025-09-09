import os
import json
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# --- 1. CONFIGURATION ---

PDF_SOURCE_DIRECTORY = "textbooks/"
VECTOR_DB_DIRECTORY = "vector_db_enriched/"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
NER_MODEL_PATH = "surgical_ner_model/" 

# --- 2. CORE LOGIC FUNCTIONS ---

def load_documents(directory_path):
    """Loads all PDF documents from a specified directory."""
    print(f"Loading documents from '{directory_path}'...")
    all_docs = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            path = os.path.join(directory_path, filename)
            loader = PyMuPDFLoader(path)
            all_docs.extend(loader.load())
    print(f"Loaded {len(all_docs)} pages from all PDF files.")
    return all_docs

def split_documents(documents):
    """Splits the loaded documents into smaller, manageable chunks."""
    print("Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunked_docs = text_splitter.split_documents(documents)
    print(f"Created {len(chunked_docs)} chunks from the documents.")
    return chunked_docs

def enrich_documents_with_ner(documents, ner_pipeline):
    """
    Analyzes each document chunk with the NER model and adds the
    extracted entities to its metadata. 
    """
    print("Enriching documents with NER metadata...")
    enriched_docs = []
    for i, doc in enumerate(documents):
        # Run NER on the document's text content
        entities = ner_pipeline(doc.page_content)
        
        # process the raw output from the NER pipeline
        if entities:
            # Group entities by their label
            grouped_entities = {}
            for entity in entities:
                label = entity['entity_group']
                word = entity['word']
                if label not in grouped_entities:
                    grouped_entities[label] = []
                # Avoid duplicates
                if word not in grouped_entities[label]:
                    grouped_entities[label].append(word)
            
            # Add the structured entities to the document's metadata
            for label, words in grouped_entities.items():
                # We store the list as a comma-separated string for compatibility
                doc.metadata[label] = ", ".join(words)

        enriched_docs.append(doc)
        
        if (i + 1) % 100 == 0:
            print(f"  - Processed {i + 1} / {len(documents)} chunks...")
            
    print("Document enrichment complete.")
    return enriched_docs

# --- 3. MAIN EXECUTION BLOCK ---

def main():
    """Main function to orchestrate the knowledge base ingestion pipeline."""
    # Step 1: Load documents
    documents = load_documents(PDF_SOURCE_DIRECTORY)
    if not documents:
        print(f"No PDF files found in '{PDF_SOURCE_DIRECTORY}'.")
        return

    # Step 2: Split documents into chunks
    chunked_documents = split_documents(documents)

    # Step 3: Load the fine-tuned NER model into a pipeline
    print(f"Loading fine-tuned NER model from '{NER_MODEL_PATH}'...")
    ner_pipeline = pipeline(
        "ner", 
        model=NER_MODEL_PATH, 
        aggregation_strategy="simple" # Groups sub-word tokens together (e.g., "lamino" + "tomy")
    )

    # Step 4: Enrich the chunks with metadata from the NER model
    enriched_documents = enrich_documents_with_ner(chunked_documents, ner_pipeline)

    # Step 5: Initialize the embedding model
    print("Initializing sentence embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'}
    )

    # Step 6: Create the new, enriched vector database
    print(f"Creating enriched vector database in '{VECTOR_DB_DIRECTORY}'...")
    vector_db = Chroma.from_documents(
        documents=enriched_documents,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_DIRECTORY
    )
    
    print("\n-----------------------------------------")
    print("Enriched knowledge base ingestion complete!")
    print(f"The new vector database is stored in '{VECTOR_DB_DIRECTORY}'.")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()
