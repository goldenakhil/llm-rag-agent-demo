# RAG + ReAct Agent Demo (FastAPI + OpenAI)

This project is a lightweight demo of a production-style
**RAG (Retrieval-Augmented Generation)** pipeline combined
with a simple **ReAct-based agent**, exposed as a FastAPI
microservice.

The agent:

- Embeds and indexes local markdown files from `./data`
- Uses a simple in-memory vector store with cosine similarity
- Executes a small ReAct reasoning loop:
  - Thought → Action → Observation → Answer
- Returns:
  - Final answer
  - Reasoning steps
  - Retrieved document chunks

---

## 🧱 Architecture

- **FastAPI** for the API server  
- **OpenAI API**:
  - `text-embedding-3-small` for embeddings  
  - `gpt-4o-mini` for reasoning  
- **VectorStore** using NumPy for similarity search  
- **RAG pipeline** for retrieving relevant file chunks  
- **ReAct agent** that decides when to call retrieval  

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
