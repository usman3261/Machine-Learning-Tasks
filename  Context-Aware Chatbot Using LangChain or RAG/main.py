import streamlit as st
import os
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# --- PAGE CONFIG ---
st.set_page_config(page_title="Personal CV Assistant", layout="centered")
st.title("🤖 Local CV Chatbot")
st.markdown("Ask questions about the uploaded CV. Everything runs locally on your Mac.")

# --- CONFIGURATION ---
# Ensure this matches the filename in your folder exactly
MODEL_PATH = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
DB_DIR = "./local_db"

@st.cache_resource
def initialize_rag():
    # 1. Load the same embeddings used in your notebook
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Load the existing vector database
    if not os.path.exists(DB_DIR):
        st.error("❌ 'local_db' folder not found. Please run your task.ipynb first!")
        st.stop()
        
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # 3. Load the Local LLM (LlamaCpp)
    # n_gpu_layers=0 is critical for stability on Intel Iris Graphics
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_gpu_layers=0, 
        verbose=False,
        temperature=0.7
    )
    
    # 4. Create the Chat Chain
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

# Initialize the system
try:
    qa_chain = initialize_rag()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.info("Check if the .gguf file is in the same folder as this script.")
    st.stop()

# --- CHAT INTERFACE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input field
user_query = st.chat_input("Ask about Usman's experience...")

if user_query:
    with st.spinner("Thinking..."):
        # Get response from the chain
        response = qa_chain.invoke({
            "question": user_query, 
            "chat_history": st.session_state.chat_history
        })
        
        answer = response["answer"]
        
        # Update session history
        st.session_state.chat_history.append((user_query, answer))

# Display message history
for q, a in st.session_state.history_display if "history_display" in st.session_state else []:
    pass # Simple display loop below

for human, assistant in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(human)
    with st.chat_message("assistant"):
        st.write(assistant)