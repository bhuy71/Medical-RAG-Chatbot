from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompt import *
import os

app = Flask(__name__)

# Path to vector database
VECTOR_DB_PATH = "vectorstore/medical_faiss"

# Check if vector database exists
if not os.path.exists(VECTOR_DB_PATH):
    raise FileNotFoundError(f"Vector database not found at {VECTOR_DB_PATH}. Please run store_index.py first.")

print("Loading embedding model...")
embeddings = download_hugging_face_embeddings()

print("Loading vector database...")
docsearch = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

print("Configuring prompt template...")

PROMPT = ChatPromptTemplate.from_template(prompt_template)
chain_type_kwargs = {"prompt": PROMPT}

print("Loading language model...")
import logging
logging.basicConfig(level=logging.DEBUG)
try:
    llm = llm = LlamaCpp(
    model_path="model/llama-2-7b-chat.Q4_0.gguf",
    temperature=0.3,
    max_tokens=256,
    n_ctx=4096,
    n_gpu_layers=8,  
    n_batch=256,
    f16_kv=True,  
    verbose=False
)
    print("Language model loaded successfully")
except Exception as e:
    logging.error(f"Lỗi khi tải mô hình ngôn ngữ: {e}")

print("Setting up RetrievalQA chain...")
document_chain = create_stuff_documents_chain(llm=llm, prompt=PROMPT)
qa = create_retrieval_chain(
    retriever=docsearch.as_retriever(search_kwargs={'k': 3}),
    combine_docs_chain=document_chain
)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(f"Question: {input}")
    
    try:
        print("Running QA chain...")
        result = qa.invoke({"input": input})
        print("QA chain completed")
        answer = result["answer"]
        
        # Get source information
        sources = []
        for doc in result.get("source_documents", []):
            if "filename" in doc.metadata:
                sources.append(doc.metadata["filename"])
            elif "source" in doc.metadata:
                sources.append(doc.metadata["source"].split("/")[-1])
        
        # Add sources to the answer if available
        if sources:
            unique_sources = list(set(sources))
            if len(unique_sources) <= 2:  # If 2 sources or less, display all
                answer += f"\n\nSource: {', '.join(unique_sources)}"
            else:  # If more than 2 sources, display count
                answer += f"\n\nBased on {len(unique_sources)} medical document sources."
        
        print(f"Successfully answered")
        return answer
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Sorry, I encountered an issue processing your question. Please try again."


if __name__ == '__main__':
    print("Starting Medical Chatbot...")
    app.run(host="0.0.0.0", port=8080, debug=False)