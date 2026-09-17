"""
1_ingest.py  —  TASK 1: Load PDFs → Chunk → Embed → Store in ChromaDB

HOW IT WORKS:
  1. Scan the /pdfs folder for all .pdf files
  2. Load each PDF page-by-page using PyPDFLoader
  3. Split the pages into smaller overlapping chunks (RecursiveCharacterTextSplitter)
  4. Embed each chunk using HuggingFace all-MiniLM-L6-v2 (local, no API key needed!)
  5. Store all embeddings in a local ChromaDB vector store

RUN:
  python 1_ingest.py
"""

import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from utils.embeddings import get_embedding_model

# ── Load .env variables ────────────────────────────────────────────────────────
load_dotenv()

# ── Config (reads from .env with fallback defaults) ───────────────────────────
PDF_FOLDER       = "./pdfs"
PERSIST_DIR      = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME  = os.getenv("CHROMA_COLLECTION_NAME", "rag_docs")
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", 200))


def load_all_pdfs(folder: str) -> list:
    """
    Step 1 & 2: Find all PDFs in the folder and load them.

    PyPDFLoader:
      - Reads each PDF page as a separate LangChain Document
      - Each Document has: page_content (text) + metadata (source, page number)
    """
    pdf_paths = glob.glob(os.path.join(folder, "**/*.pdf"), recursive=True)
    pdf_paths += glob.glob(os.path.join(folder, "*.pdf"))
    pdf_paths = list(set(pdf_paths))  # remove duplicates

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in '{folder}'\n"
            f"Please drop your PDF files into the /pdfs folder and try again."
        )

    all_documents = []
    print(f"\n📂 Found {len(pdf_paths)} PDF(s) in '{folder}'")
    print("=" * 60)

    for pdf_path in pdf_paths:
        print(f"\n📄 Loading: {os.path.basename(pdf_path)}")
        loader = PyPDFLoader(pdf_path)          # LangChain PDF Loader
        documents = loader.load()               # Returns list of Documents (1 per page)
        print(f"   ✅ Loaded {len(documents)} pages")
        all_documents.extend(documents)

    print(f"\n📊 Total pages loaded: {len(all_documents)}")
    return all_documents


def split_documents(documents: list) -> list:
    """
    Step 3: Split large pages into smaller overlapping chunks.

    RecursiveCharacterTextSplitter:
      - Tries to split on paragraph breaks first, then sentences, then words
      - chunk_size    = max characters per chunk
      - chunk_overlap = how many characters the next chunk shares with the previous
                        (overlap helps avoid losing context at chunk boundaries)
    """
    print(f"\n✂️  Splitting documents into chunks...")
    print(f"   chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],  # try paragraph → line → word → char
    )

    chunks = splitter.split_documents(documents)
    print(f"   ✅ Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


def embed_and_store(chunks: list) -> Chroma:
    """
    Step 4 & 5: Embed each chunk and store in ChromaDB.

    OpenAIEmbeddings:
      - Sends each chunk's text to OpenAI API
      - Returns a 1536-dimensional float vector (numerical representation)

    Chroma.from_documents():
      - Takes chunks + embedding model
      - Calls the embedding model for each chunk automatically
      - Saves vectors to disk at PERSIST_DIR
    """
    print(f"\n🧠 Embedding {len(chunks)} chunks and storing in ChromaDB...")
    print(f"   Collection : {COLLECTION_NAME}")
    print(f"   Persist dir: {PERSIST_DIR}")

    embedding_model = get_embedding_model()

    # Chroma.from_documents() handles embedding + storage in one call
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    print(f"   ✅ Successfully stored {len(chunks)} vectors in ChromaDB!")
    return vectorstore


def main():
    print("🚀 Starting Task 1: PDF Ingestion Pipeline")
    print("=" * 60)

    # Step 1 & 2: Load PDFs
    documents = load_all_pdfs(PDF_FOLDER)

    # Step 3: Split into chunks
    chunks = split_documents(documents)

    # Step 4 & 5: Embed and store
    vectorstore = embed_and_store(chunks)

    print("\n" + "=" * 60)
    print("🎉 Task 1 Complete!")
    print(f"   Your vector database is saved at: {PERSIST_DIR}/")
    print(f"   Collection name                 : {COLLECTION_NAME}")
    print(f"   Total vectors stored            : {len(chunks)}")
    print("\n   ➡️  Next step: Run 'python 2_query.py' to query your documents!")


if __name__ == "__main__":
    main()
