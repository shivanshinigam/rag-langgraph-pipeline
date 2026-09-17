"""
utils/vectorstore.py

Shared ChromaDB vector store client.
All scripts import get_vectorstore() from here so they always
connect to the same ChromaDB collection.
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from utils.embeddings import get_embedding_model

load_dotenv()


def get_vectorstore() -> Chroma:
    """
    Returns a LangChain Chroma vector store connected to the
    persisted local ChromaDB collection.

    ChromaDB stores data on disk at CHROMA_PERSIST_DIR.
    No external server required — it's just a local folder.
    """
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "rag_docs")

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=persist_dir,
    )
