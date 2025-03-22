from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import FAISS
import os
import argparse

def update_vectorstore(new_pdf_dir):
    """Update vector database with new documents"""
    VECTOR_DB_PATH = "vectorstore/medical_faiss"
    
    print(f"Loading data from {new_pdf_dir}...")
    new_data = load_pdf(new_pdf_dir)
    if not new_data:
        print(f"No PDF files found in {new_pdf_dir}")
        return
        
    print("Processing text...")
    text_chunks = text_split(new_data)
    print(f"Created {len(text_chunks)} chunks")
    
    print("Loading embedding model...")
    embeddings = download_hugging_face_embeddings()
    
    if os.path.exists(VECTOR_DB_PATH):
        print("Loading existing vector database...")
        vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings)
        
        print("Adding new data...")
        vectorstore.add_documents(text_chunks)
        
        print("Saving updated vector database...")
        vectorstore.save_local(VECTOR_DB_PATH)
        print(f"Updated vector database with {len(text_chunks)} new chunks")
    else:
        print("Existing vector database not found. Creating new one...")
        vectorstore = FAISS.from_documents(text_chunks, embeddings)
        vectorstore.save_local(VECTOR_DB_PATH)
        print(f"Created new vector database with {len(text_chunks)} chunks")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update vector database with new PDF documents")
    parser.add_argument("--pdf_dir", type=str, default="new_data", 
                        help="Directory containing new PDF files")
    args = parser.parse_args()
    
    update_vectorstore(args.pdf_dir)