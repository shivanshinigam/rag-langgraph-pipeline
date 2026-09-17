# rag-langgraph-pipeline

A step-by-step RAG (Retrieval-Augmented Generation) pipeline built with **LangChain** + **LangGraph**.

## Project Structure
```
rag-langgraph-pipeline/
├── pdfs/               ← Drop your PDF files here
├── utils/
│   ├── embeddings.py   ← Shared embedding setup
│   └── vectorstore.py  ← Shared ChromaDB client
│
├── 1_ingest.py         ← Task 1: Load PDFs → Embed → Store in ChromaDB
├── 2_query.py          ← Task 2: Query ChromaDB with a question
├── 3_langgraph_pipeline.py  ← Task 3: Full LangGraph RAG pipeline
├── requirements.txt
└── .env.example
```

## Setup
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Add PDFs
# Drop your PDF files inside the /pdfs folder
```

## Running Each Task

### Task 1 — Ingest PDFs into ChromaDB
```bash
python 1_ingest.py
```

### Task 2 — Query the Vector DB
```bash
python 2_query.py "What is this document about?"
```

### Task 3 — Run the full LangGraph Pipeline
```bash
python 3_langgraph_pipeline.py "What is this document about?"
```
