"""
utils/embeddings.py

Shared embedding model setup — using HuggingFace sentence-transformers.
Runs 100% LOCALLY. No API key required!

Model: all-MiniLM-L6-v2
  - Lightweight and fast
  - Produces 384-dimensional vectors
  - Great for semantic similarity tasks
"""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns a local HuggingFace embedding model.
    First run will download the model (~90MB) from HuggingFace Hub.
    Subsequent runs use the cached version — no internet needed.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},   # use "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},  # normalize for cosine similarity
    )
