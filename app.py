import os
import signal
import sys
import threading
import time
from typing import TypedDict, List, Literal

# Libraries
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import GoogleGenerativeAIEmbeddings 


SYSTEM_PROMPT = """
You are a friendly and professional Cyber Security Advisor.

FORMATTING RULES:
1. **Use Emojis:** You MUST use relevant emojis (e.g., 🛡️, ⚠️, 💻, 🔑) for every Title and Key Factor to make the response engaging.
2. **Bold Text:** You MUST use **Bold Text** for all Titles, Headers, and important Key Factors.
3. **Structure:** Use clear bullet points.

GUIDELINES:
1. Always be polite and professional.
2. If the user asks in Burmese, reply in Burmese.
3. If the user asks in English, reply in English.
4. Keep answers concise but informative.

LANGUAGE RULES:
- If the user asks in Burmese, reply in Burmese.
- If the user asks in English, reply in English.
"""

# Configuration & Setup
# -----------------------------
UPLOAD_FOLDER = "upload"
FAISS_PATH = "faiss_db"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# -----------------------------
# Vector Store Setup (CHANGED)
# -----------------------------

# 1. Get the API Key from Environment Variable
api_key = os.environ.get("GOOGLE_API_KEY_1") 

if not api_key:
    print("⚠️ Error: GOOGLE_API_KEY_1 not found in environment variables!")

# 2. Use Google Embeddings instead of HuggingFace
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", # This is the efficient Google model
    google_api_key=api_key
)

def get_vectorstore():
    # IMPORTANT: You must delete the old 'faiss_db' folder if you switched models!
    if os.path.exists(FAISS_PATH):
        print("Loading FAISS...")
        try:
            return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading FAISS: {e}")
            print("You might need to delete the 'faiss_db' folder and restart to rebuild it.")
            return None
    else:
        print("Scanning PDFs...")
        # Check if folder has files before loading
        if not os.listdir(UPLOAD_FOLDER):
            print("No PDF files found in upload folder.")
            return None
            
        loader = DirectoryLoader(UPLOAD_FOLDER, glob="*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        if not docs: return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Build vectorstore with Google Embeddings
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(FAISS_PATH)
        return vectorstore

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None


# -----------------------------
# 1. SETUP: 10 API KEYS (FROM OS ENVIRONMENT)
# -----------------------------

# This list comprehension looks for variables named "GOOGLE_API_KEY_1" to "GOOGLE_API_KEY_10"

API_KEYS = [os.environ.get(f"GOOGLE_API_KEY_{i}") for i in range(1, 6)]

# Remove any 'None' values (in case you set fewer than 10 keys)
API_KEYS = [key for key in API_KEYS if key is not None]

# Validation Check
if not API_KEYS:
    raise ValueError("❌ No API Keys found! Please set environment variables: GOOGLE_API_KEY_1 ... GOOGLE_API_KEY_10")

print(f"✅ Successfully loaded {len(API_KEYS)} API Keys from Environment.")

# -----------------------------
# 2. MODELS SETUP
# -----------------------------
# The two models to rotate
MODELS = ["gemini-1.5-flash", "gemini-2.5-flash"]

# Global Index tracker
current_key_index = 0

class ChatState(TypedDict):
    question: str
    history: List[str]
    context: List[str]
    answer: str

# Defined Schemas (Must be available globally for the helper)
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "general_llm"]

class RAGResponse(BaseModel):
    answer: str
    confidence: int

# -----------------------------
# HELPER: Rotation Logic
# -----------------------------
def call_llm_with_rotation(prompt, task_type="normal"):
    """
    Logic:
    1. Try Model A (Flash) on Current Key.
    2. If Limit -> Try Model B (Flash-Elite) on Current Key.
    3. If Both Limit -> Switch to Next API Key and repeat.
    """
    global current_key_index

    # We loop through the keys starting from the current one.
    # We use a range equal to the number of keys to ensure we try everyone once.
    for _ in range(len(API_KEYS)):
        current_api_key = API_KEYS[current_key_index]

        # Try both models on this specific key
        for model_name in MODELS:
            try:
                # Initialize LLM with current Key and Model
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0,
                    google_api_key=current_api_key
                )

                # Execute based on task type
                if task_type == "router":
                    structured_llm = llm.with_structured_output(RouteQuery)
                    return structured_llm.invoke(prompt)

                elif task_type == "rag":
                    structured_llm = llm.with_structured_output(RAGResponse)
                    return structured_llm.invoke(prompt)

                else: # normal direct generation
                    return llm.invoke(prompt)

            except Exception as e:
                error_msg = str(e).lower()
                # Check if it is a Quota/Rate Limit error
                if "429" in error_msg or "quota" in error_msg or "resource exhausted" in error_msg:
                    print(f"⚠️ Quota Limit: Key #{current_key_index+1} | Model {model_name}. Switching Model...")
                    continue # Try the next model in the inner loop
                else:
                    # If it's another error (e.g., code error), raise it
                    print(f"❌ Error (Not Quota): {e}")
                    raise e

        # If we finish the inner loop, it means BOTH models failed on this key.
        print(f"🚫 Key #{current_key_index+1} Exhausted (Both Models). Switching API Key...")

        # Rotate to next key
        current_key_index = (current_key_index + 1) % len(API_KEYS)

    # If we exit the outer loop, all 10 keys failed
    raise Exception("CRITICAL: All 10 API Keys and Models are exhausted.")

# -----------------------------
# Nodes (Updated to use Rotation)
# -----------------------------

def route_question(state: ChatState):
    print(f"\n[ROUTER] Question: {state['question']}")
    history_subset = state["history"][-4:] if state["history"] else []
    history_text = "\n".join(history_subset) if history_subset else "No history."

    prompt = f"""
    Decide routing.
    History: {history_text}
    Question: {state['question']}

    INSTRUCTION:
    - If user says "explain this", "tell me more", "translate", look at History.
    - If user ask about Cyber Law/Security, choose 'vectorstore'.
    - Otherwise 'general_llm'.
    """
    try:
        # CALL ROTATION HELPER
        source = call_llm_with_rotation(prompt, task_type="router")
        print(f"[ROUTER] Selected: {source.datasource}")
        return source.datasource
    except Exception as e:
        print(f"Router Error: {e}")
        return "general_llm"

def direct_answer_node(state: ChatState):
    print(f"[DIRECT] Generating... (History Size: {len(state['history'])})")
    history_subset = state["history"][-4:] if state["history"] else []
    history_text = "\n".join(history_subset) if history_subset else "No history."

    # Assumes SYSTEM_PROMPT is defined globally as in your original code
    prompt = f"""
    SYSTEM INSTRUCTION:
    {SYSTEM_PROMPT}

    Context/History:
    {history_text}

    User Request: {state['question']}

    SPECIFIC INSTRUCTION:
    - If user asks to "explain this" or "translate", explain the LAST TOPIC in History.
    - If asking for Burmese, translate the explanation to Myanmar language.
    """

    # CALL ROTATION HELPER
    msg = call_llm_with_rotation(prompt, task_type="normal")
    state["answer"] = msg.content
    return state

def retrieve_node(state: ChatState):
    print("[RAG] Retrieving...")
    if retriever:
        docs = retriever.invoke(state["question"])
        state["context"] = [d.page_content for d in docs]
    return state

def generate_rag_node(state: ChatState):
    print("[RAG] Generating Answer...")
    context_text = "\n".join(state["context"])
    print(context_text)
    prompt = f"""
    SYSTEM INSTRUCTION:
    {SYSTEM_PROMPT}

    RETRIEVED CONTEXT FROM DATABASE:
    {context_text}

    User Question: {state['question']}

    Answer the question based ONLY on the context provided above.
    """

    try:
        # CALL ROTATION HELPER
        res = call_llm_with_rotation(prompt, task_type="rag")
        state["answer"] = f"{res.answer}\n(Confidence: {res.confidence}/10)"
    except Exception as e:
        state["answer"] = f"Error in RAG generation: {str(e)}"
    return state

def update_memory_node(state: ChatState):
    state["history"].append(f"User: {state['question']}")
    state["history"].append(f"AI: {state['answer']}")
    print(f"[MEMORY] Updated. Total items: {len(state['history'])}")
    return state

# -----------------------------
# Build Graph (Same as before)
# -----------------------------
graph = StateGraph(ChatState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate_rag", generate_rag_node)
graph.add_node("direct_answer", direct_answer_node)
graph.add_node("update_memory", update_memory_node)

graph.add_conditional_edges(START, route_question,
                            {"vectorstore": "retrieve", "general_llm": "direct_answer"})

graph.add_edge("retrieve", "generate_rag")
graph.add_edge("generate_rag", "update_memory")
graph.add_edge("direct_answer", "update_memory")
graph.add_edge("update_memory", END)

chat_graph = graph.compile()

# ... (Keep the Flask App code below exactly as it was) ...
# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
server_memory = [] # Global Memory Store

@app.route("/chat", methods=["POST"])
def chat():
    global server_memory
    data = request.json
    user_input = data.get("message") or data.get("query")
    state = {
        "question": user_input,
        "history": server_memory, # Load History
        "context": [],
        "answer": ""
    }

    result = chat_graph.invoke(state)

    # Save History back to Global
    server_memory = result["history"]

    return jsonify({"response": result["answer"]})

# ... [KEEP YOUR CLASSES: ChatState, RouteQuery, RAGResponse HERE] ...
# ... [KEEP YOUR HELPER: call_llm_with_rotation HERE] ...
# ... [KEEP YOUR NODES: route_question, direct_answer_node, etc. HERE] ...
# ... [KEEP YOUR GRAPH BUILD CODE HERE] ...

# --- FLASK APP ---
app = Flask(__name__)
server_memory = []

@app.route("/chat", methods=["POST"])
def chat():
    global server_memory
    data = request.json
    user_input = data.get("message") or data.get("query")
    state = {
        "question": user_input,
        "history": server_memory,
        "context": [],
        "answer": ""
    }
    # Invoke Graph
    result = chat_graph.invoke(state)
    server_memory = result["history"]
    return jsonify({"response": result["answer"]})

@app.route("/")
def home():
    # Simple Health Check / Interface
    return "<h1>Cyber Advisor Bot is Running</h1>"

# --- PRODUCTION ENTRY POINT ---
if __name__ == "__main__":
    # This block only runs when you type 'python app.py' locally
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
