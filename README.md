# RAG Engine

A modular Retrieval-Augmented Generation (RAG) engine built with FastAPI, LangChain, and FAISS.

## Features
- Modular ingestion (PDF/DOCX/TXT)
- Hybrid retrieval (FAISS + BM25)
- Semantic chunking
- FastAPI endpoint

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `uvicorn app.main:app --reload`
