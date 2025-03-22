from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
import os
from pathlib import Path

# Create directory for vector database
VECTOR_DB_PATH = "vectorstore/medical_faiss"
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# Load and process data
print("Loading data from PDF files...")
extracted_data = load_pdf("data/")
print(f"Loaded {len(extracted_data)} documents")

print("Splitting text into chunks...")
text_chunks = text_split(extracted_data)
print(f"Created {len(text_chunks)} chunks")

print("Loading embedding model...")
embeddings = download_hugging_face_embeddings()

# Create FAISS index from chunks and save locally
print("Creating vector database...")
vectorstore = FAISS.from_documents(text_chunks, embeddings)
vectorstore.save_local(VECTOR_DB_PATH)

print(f"Vector database created and saved at: {VECTOR_DB_PATH}")
print(f"Total chunks: {len(text_chunks)}")