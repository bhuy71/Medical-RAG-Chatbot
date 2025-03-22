import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from langchain_huggingface import HuggingFaceEmbeddings

# Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()
    
    # Thêm metadata
    for doc in documents:
        source_file = doc.metadata.get("source", "")
        doc.metadata["filename"] = source_file.split("/")[-1]
    
    return documents


# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


# Download embedding model
def download_hugging_face_embeddings():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True, 'batch_size': 16} 
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings