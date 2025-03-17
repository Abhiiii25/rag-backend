from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import os

app = Flask(__name__)

# Load local model
llm = OllamaLLM(model="gemma:2b")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = None  # Placeholder for vector database

@app.route('/upload', methods=['POST'])
def upload_file():
    global db
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)
    
    # Load and process document
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif filename.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        return jsonify({"error": "Unsupported file format"}), 400
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    
    # Create Vector Database
    db = FAISS.from_documents(documents, embedding)
    return jsonify({"message": "File processed and vector database updated"})

@app.route('/query', methods=['POST'])
def query_rag():
    global db
    if db is None:
        return jsonify({"error": "No document uploaded. Please upload first."}), 400
    
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa.run(query)
    
    return jsonify({"response": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
