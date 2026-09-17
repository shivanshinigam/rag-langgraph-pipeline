# rag-langgraph-pipeline

A step-by-step RAG (Retrieval-Augmented Generation) pipeline built with **LangChain** + **LangGraph**.

## Project Structure
```
rag-langgraph-pipeline/
├── pdfs/                        # Drop your PDF files here
├── utils/
│   ├── embeddings.py            # Shared embedding model setup
│   └── vectorstore.py           # Shared ChromaDB client
│
├── 1_ingest.py                  # Task 1: Load PDFs → Embed → Store in ChromaDB
├── 2_query.py                   # Task 2: Query ChromaDB with a question
├── 3_langgraph_pipeline.py      # Task 3: Full LangGraph RAG pipeline
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
# Edit .env and add your OPENAI_API_KEY (or leave blank to use local HuggingFace embeddings)

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

---

## Task Outputs

### Task 1 — PDF Ingestion (`1_ingest.py`)

**PDF used:** `attention_is_all_you_need.pdf` (Vaswani et al., 2017 — the original Transformer paper)

**Console output:**
```
Starting Task 1: PDF Ingestion Pipeline
------------------------------------------------------------
Found 1 PDF(s) in './pdfs'

  Loading: attention_is_all_you_need.pdf
  Loaded 15 pages

Total pages loaded: 15

Splitting documents into chunks...
  chunk_size=1000, chunk_overlap=200
  Created 52 chunks from 15 pages

Embedding 52 chunks and storing in ChromaDB...
  Collection : rag_docs
  Persist dir: ./chroma_db
  Successfully stored 52 vectors in ChromaDB!

------------------------------------------------------------
Task 1 Complete!
  Vector database saved at : ./chroma_db/
  Collection name          : rag_docs
  Total vectors stored     : 52
```

**Run statistics:**

| Metric                  | Value                                      |
|-------------------------|--------------------------------------------|
| PDF pages loaded        | 15                                         |
| Chunks created          | 52                                         |
| Chunk size              | 1,000 characters                           |
| Chunk overlap           | 200 characters                             |
| Embedding model         | sentence-transformers/all-MiniLM-L6-v2     |
| Vector dimensions       | 384                                        |
| Vector store            | ChromaDB (local, on-disk)                  |
| Vectors stored          | 52                                         |

**Concepts covered:**

| Concept                          | Description                                                                   |
|----------------------------------|-------------------------------------------------------------------------------|
| `PyPDFLoader`                    | Reads a PDF and returns one `Document` object per page                        |
| `RecursiveCharacterTextSplitter` | Breaks pages into overlapping chunks to preserve context at boundaries        |
| `chunk_overlap`                  | Shared characters between adjacent chunks — prevents loss of context          |
| `HuggingFaceEmbeddings`          | Converts text into a 384-dimensional float vector representing its meaning    |
| `Chroma.from_documents()`        | Embeds all chunks and persists them to disk in a single call                  |
| `persist_directory`              | ChromaDB saves data locally — no external server or cloud account required    |
