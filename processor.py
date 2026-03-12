import os
import json
import requests
import shutil
import urllib3
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader, 
    CSVLoader, 
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "chroma_db"

def call_llm(system_prompt, user_content):
    api_key = os.getenv('OPENROUTER_API_KEY')
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openrouter/free", 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    }
    
    try:
        # Disable warnings properly for both old and new urllib3 versions
        urllib3.disable_warnings() 
        
        # verify=False bypasses the SSL check
        response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

def ingest_documents(directory_path):
    all_documents = []
    
    # Process all files in the uploads folder together
    for filename in os.listdir(directory_path):
        path = os.path.join(directory_path, filename)
        
        if path.endswith(".pdf"): loader = PyPDFLoader(path)
        elif path.endswith(".csv"): loader = CSVLoader(path)
        elif path.endswith(".md"): loader = UnstructuredMarkdownLoader(path)
        else: continue
        
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = filename
        all_documents.extend(docs)

    if not all_documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(all_documents)

    # WINDOWS FIX: Clear the directory safely
    if os.path.exists(CHROMA_PATH):
        # We use a try-except because Chroma might still have a lock
        try:
            shutil.rmtree(CHROMA_PATH)
        except PermissionError:
            print("Database locked. Creating new collection inside existing DB.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vector_db

# --- Tool Functions (Keep these as they were in previous version) ---
def factual_qa_tool(query, vector_db):
    docs = vector_db.similarity_search(query, k=4)
    context = "\n\n".join([f"[{d.metadata.get('source_file')}]: {d.page_content}" for d in docs])
    prompt = "Answer strictly based on context. Provide a Confidence Score (High/Medium/Low)."
    return f"🛠️ Factual Q&A\n\n{call_llm(prompt, f'Context: {context}\nQuestion: {query}')}"

def comparative_tool(query, vector_db):
    docs = vector_db.similarity_search(query, k=8)
    context = "\n\n".join([f"Source: {d.metadata.get('source_file')}\nContent: {d.page_content}" for d in docs])
    prompt = "Compare and correlate data across different files. Provide a Confidence Score."
    return f"🛠️ Comparative Tool\n\n{call_llm(prompt, f'Context: {context}\nQuestion: {query}')}"

def summary_tool(query, vector_db):
    docs = vector_db.similarity_search(query, k=10)
    context = "\n".join([d.page_content for d in docs])
    prompt = "Summarize the key points for the provided context."
    return f"🛠️ Summary Tool\n\n{call_llm(prompt, f'Context: {context}\nQuestion: {query}')}"

def agent_dispatcher(user_query, vector_db):
    intent_prompt = "Reply with ONLY 'SUMMARY', 'COMPARATIVE', or 'QA' based on user query."
    intent = call_llm(intent_prompt, user_query).strip().upper()
    if "SUMMARY" in intent: return summary_tool(user_query, vector_db)
    elif "COMPARATIVE" in intent: return comparative_tool(user_query, vector_db)
    else: return factual_qa_tool(user_query, vector_db)