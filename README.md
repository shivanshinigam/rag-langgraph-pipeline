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

---

### Task 2 — Vector DB Query (`2_query.py`)

Three queries were run against the ChromaDB collection built in Task 1.

---

**Query 1:** `"What is multi-head attention?"`

```
Query  : What is multi-head attention?
Results: 5 chunks retrieved

  Rank #1  |  Relevance: 64.5%  (distance: 0.7093)  |  Page: 4
  Content  : "...MultiHead(Q, K, V) = Concat(head1,...,headh)W^O
              where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)..."

  Rank #2  |  Relevance: 58.8%  (distance: 0.8246)  |  Page: 4
  Content  : "...In this work we employ h = 8 parallel attention layers, or heads.
              For each of these we use dk = dv = dmodel/h = 64..."

  Rank #3  |  Relevance: 50.2%  (distance: 0.9963)  |  Page: 5
  Content  : "...The Transformer uses multi-head attention in three different ways:
              In encoder-decoder attention layers, the queries come from the
              previous decoder layer..."

  Rank #4  |  Relevance: 48.7%  (distance: 1.0268)  |  Page: 13
  Rank #5  |  Relevance: 46.8%  (distance: 1.0631)  |  Page: 3
```

---

**Query 2:** `"What optimizer was used for training?"`

```
Query  : What optimizer was used for training?
Results: 5 chunks retrieved

  Rank #1  |  Relevance: 41.8%  (distance: 1.1650)  |  Page: 7
  Content  : "...We trained our models on one machine with 8 NVIDIA P100 GPUs.
              Each training step took about 0.4 seconds. We trained the base
              models for a total of 100,000 steps or 12 hours..."

  Rank #2  |  Relevance: 33.3%  (distance: 1.3348)  |  Page: 7
  Content  : "...We trained on the standard WMT 2014 English-German dataset
              consisting of about 4.5 million sentence pairs..."

  Rank #3  |  Relevance: 32.4%  (distance: 1.3513)  |  Page: 8
  Rank #4  |  Relevance: 31.1%  (distance: 1.3783)  |  Page: 8
  Rank #5  |  Relevance: 30.6%  (distance: 1.3879)  |  Page: 8
```

> Note: Lower relevance scores here indicate the query term "optimizer" does not
> appear verbatim in the paper — it uses "Adam" and "training regime" instead.
> This is expected behavior and is why the filter node in Task 3 is important.

---

**Query 3:** `"How does the encoder work?"`

```
Query  : How does the encoder work?
Results: 5 chunks retrieved

  Rank #1  |  Relevance: 63.8%  (distance: 0.7231)  |  Page: 2
  Content  : "...the encoder maps an input sequence of symbol representations
              (x1,...,xn) to a sequence of continuous representations z = (z1,...,zn).
              Given z, the decoder then generates an output sequence..."

  Rank #2  |  Relevance: 54.2%  (distance: 0.9161)  |  Page: 5
  Content  : "...The Transformer uses multi-head attention in three different ways:
              In encoder-decoder attention layers, the queries come from the
              previous decoder layer..."

  Rank #3  |  Relevance: 47.7%  (distance: 1.0455)  |  Page: 3
  Content  : "...The Transformer follows this overall architecture using stacked
              self-attention and point-wise, fully connected layers for both
              the encoder and decoder..."

  Rank #4  |  Relevance: 45.4%  (distance: 1.0911)  |  Page: 3
  Rank #5  |  Relevance: 43.6%  (distance: 1.1281)  |  Page: 5
```

---

**Score interpretation:**

| Relevance Range | Meaning                                     |
|-----------------|---------------------------------------------|
| 70% – 100%      | Highly relevant — strong semantic match     |
| 50% – 70%       | Moderately relevant — good contextual match |
| 30% – 50%       | Weak match — may contain noise              |
| Below 30%       | Not relevant — filtered out in Task 3       |

**Concepts covered:**

| Concept                         | Description                                                                             |
|---------------------------------|-----------------------------------------------------------------------------------------|
| `similarity_search_with_score()`| Embeds the query and finds the top-K nearest vectors in ChromaDB                       |
| Cosine distance                 | Measures angle between two vectors — 0.0 = identical, 2.0 = completely opposite        |
| Relevance %                     | Human-readable conversion: `(1 - distance / 2) * 100`                                  |
| `doc.metadata`                  | Each result carries its source filename and page number from the original PDF           |
| Score threshold (preview)       | Query 2 shows why low-relevance results need filtering — used in Task 3's filter node   |
