import os
import time
import threading
from typing import TypedDict, List
from flask import Flask, request, jsonify
import traceback

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. SETUP: API KEYS (FROM ENV VARS)
# ==========================================

# On Render, you will paste all your keys in one Environment Variable named "GOOGLE_API_KEYS"
# separated by commas. Example: key1,key2,key3
raw_keys = os.environ.get("GOOGLE_API_KEYS", "")
API_KEYS = [k.strip() for k in raw_keys.split(',') if k.strip()]

if not API_KEYS:
    # Fallback to checking individual keys if the list isn't set
    if os.environ.get("GOOGLE_API_KEY"):
        API_KEYS = [os.environ.get("GOOGLE_API_KEY")]
    else:
        print("❌ CRITICAL: No API Keys found in Environment Variables!")
        API_KEYS = []

print(f"✅ Loaded {len(API_KEYS)} API Keys.")

# ==========================================
# 2. SETUP: FOLDERS & DATABASE
# ==========================================

UPLOAD_FOLDER = "upload"
FAISS_PATH = "faiss_db"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# NOTE: On Render, you must push your PDF files to GitHub inside the 'upload' folder.
# We cannot wait for user input here.

# Use the first key for Embeddings
embeddings_key = API_KEYS[0] if API_KEYS else None
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=embeddings_key
)

def get_vectorstore():
    # If a DB already exists, load it to save time
    if os.path.exists(FAISS_PATH):
        try:
            print("🔄 Loading existing FAISS DB...")
            return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"⚠️ Failed to load FAISS DB: {e}")
            pass # If load fails, rebuild

    print("🔄 Building Database from PDFs...")
    try:
        # check if files exist
        if len(os.listdir(UPLOAD_FOLDER)) == 0:
            print("⚠️ No PDFs found in 'upload' folder.")
            return None

        loader = DirectoryLoader(UPLOAD_FOLDER, glob="*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        if not docs: 
            print("⚠️ No documents loaded from PDFs")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # REDUCED for memory
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Save locally (though it's ephemeral on Render free tier)
        vectorstore.save_local(FAISS_PATH)
        return vectorstore
    except Exception as e:
        print(f"❌ PDF Error: {e}")
        return None

# Initialize DB on startup
try:
    vectorstore = get_vectorstore()
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # REDUCED k for memory
        print("✅ Database Ready!")
    else:
        retriever = None
        print("⚠️ No vector database available")
except Exception as e:
    print(f"❌ Database initialization failed: {e}")
    retriever = None

# =========================================================
# 3. ADVANCED ROTATION LOGIC (FIXED VERSION)
# =========================================================

MASTER_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

class RotationManager:
    def __init__(self, keys):
        self.master_keys = keys
        self.active_keys = keys.copy()
        self.max_retries = len(keys) * 2  # Maximum retry attempts
        self.key_lock = threading.Lock()
        
    def get_response(self, prompt):
        if not self.master_keys:
            return "No API Keys configured. Please add GOOGLE_API_KEYS environment variable."
        
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            with self.key_lock:
                # If all keys blocked, restore them
                if len(self.active_keys) == 0:
                    print("🔄 All keys were blocked. Restoring all keys...")
                    self.active_keys = self.master_keys.copy()
                
                if len(self.active_keys) == 0:
                    return "All API keys are currently unavailable. Please try again later."
                
                current_key = self.active_keys[0]
                current_models = MASTER_MODELS.copy()
            
            key_failed = False
            
            # Try each model with current key
            for model in current_models:
                try:
                    print(f"🔄 Trying key ending in ...{current_key[-4:]} with model {model}")
                    
                    # Set timeout for LLM call
                    llm = ChatGoogleGenerativeAI(
                        model=model,
                        temperature=0.3,
                        google_api_key=current_key,
                        timeout=30,  # ADDED TIMEOUT
                        max_retries=2  # ADDED RETRIES
                    )
                    
                    response = llm.invoke(prompt)
                    
                    # If successful, rotate keys for next request
                    with self.key_lock:
                        if self.active_keys and self.active_keys[0] == current_key:
                            self.active_keys.append(self.active_keys.pop(0))
                    
                    return response
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    print(f"⚠️ Error on Key ...{current_key[-4:]} | {model}: {error_msg[:50]}")
                    last_error = e
                    
                    # Check if it's a quota/blocking error
                    if any(err in error_msg for err in ["quota", "blocked", "resource exhausted", "permission denied"]):
                        key_failed = True
                        break  # Don't try other models with this key
            
            # If key failed with all models or was blocked
            if key_failed:
                with self.key_lock:
                    if current_key in self.active_keys:
                        print(f"⛔ Removing blocked key ending in ...{current_key[-4:]}")
                        self.active_keys.remove(current_key)
            
            retry_count += 1
            time.sleep(1)  # Small delay between retries
        
        # If we get here, all retries failed
        error_msg = str(last_error) if last_error else "Unknown error"
        return f"All API attempts failed. Last error: {error_msg[:100]}"

llm_manager = RotationManager(API_KEYS)

def call_llm_with_rotation(prompt):
    try:
        return llm_manager.get_response(prompt)
    except Exception as e:
        print(f"❌ Rotation manager error: {e}")
        return f"System error: {str(e)[:100]}"

# ==========================================
# 4. PROCESSING NODE
# ==========================================

class ChatState(TypedDict):
    question: str
    history: List[str]
    answer: str

def processing_node(state: ChatState):
    question = state["question"]
    
    # Context
    context_text = ""
    if retriever:
        try:
            docs = retriever.invoke(question)
            context_text = "\n".join([d.page_content for d in docs][:2])  # LIMIT context
        except Exception as e:
            print(f"Retrieval Error: {e}")
            context_text = ""

    history_text = "\n".join(state["history"][-2:] if state["history"] else [])

    prompt = f"""
    You are a Cyber Security Advisor.
    
    CONTEXT: {context_text}
    HISTORY: {history_text}
    QUESTION: {question}
    
    INSTRUCTION:
    - If greeting, ignore context.
    - If technical, use context. Use emojis 🛡️.
    - Reply in same language (Burmese/English).
    - Keep response concise.
    """
    
    try:
        response = call_llm_with_rotation(prompt)
        if isinstance(response, str):
            state["answer"] = response
        else:
            state["answer"] = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"❌ Processing error: {traceback.format_exc()}")
        state["answer"] = f"System error. Please try again."
    
    state["history"].append(f"User: {question}")
    state["history"].append(f"AI: {state['answer']}")
    
    # Limit history size to prevent memory issues
    if len(state["history"]) > 10:
        state["history"] = state["history"][-10:]
    
    return state

graph = StateGraph(ChatState)
graph.add_node("process", processing_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)
chat_graph = graph.compile()

# ==========================================
# 5. FLASK SERVER (FIXED FOR RENDER)
# ==========================================
app = Flask(__name__)
server_memory = []

@app.route("/", methods=["GET"])
def health_check():
    return "<h1>Cyber Security Bot is Running 🚀</h1>"

@app.route("/chat", methods=["POST"])
def chat():
    global server_memory
    try:
        data = request.json
        if not data or "message" not in data:
            return jsonify({"response": "Please provide a 'message' in the request body"}), 400
        
        user_input = data.get("message", "").strip()
        if not user_input:
            return jsonify({"response": "Message cannot be empty"}), 400
        
        state = {"question": user_input, "history": server_memory, "answer": ""}
        result = chat_graph.invoke(state)
        server_memory = result["history"]
        
        return jsonify({"response": result["answer"]})
        
    except Exception as e:
        print(f"❌ Chat endpoint error: {traceback.format_exc()}")
        return jsonify({"response": f"Internal server error: {str(e)[:100]}"}), 500

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": "ok",
        "api_keys_loaded": len(API_KEYS),
        "database_ready": retriever is not None,
        "memory_usage": len(server_memory)
    })

if __name__ == "__main__":
    # Render assigns a port via the PORT environment variable
    # We must listen on 0.0.0.0
    port = int(os.environ.get("PORT", 10000))
    
    # For Render free tier, use single worker and lower memory settings
    app.run(
        host="0.0.0.0", 
        port=port,
        debug=False,
        threaded=True
    )

