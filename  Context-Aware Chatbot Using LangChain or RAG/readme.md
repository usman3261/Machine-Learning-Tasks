🤖 Local RAG CV Assistant
An intelligent Document Assistant built with LangChain and Streamlit that allows users to chat with a PDF resume. The entire pipeline—from text embedding to response generation—runs 100% locally on an Intel-based MacBook.

🌟 Key Features
Privacy-First: No data ever leaves your machine. No OpenAI/Third-party API keys required.

Local RAG Pipeline: Uses Retrieval-Augmented Generation to provide context-aware answers based on the uploaded CV.

Optimized for Intel Macs: Specifically configured using LlamaCpp with CPU-only execution for stability on older macOS versions (12.7+).

🛠️ Technical Stack
LLM: Llama 3.2 1B (Quantized GGUF)

Orchestration: LangChain

Vector Store: ChromaDB

Embeddings: HuggingFace all-MiniLM-L6-v2

Frontend: Streamlit