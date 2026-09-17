"""
utils/embeddings.py

Shared embedding model setup.
All scripts import from here so we always use the same embedding model.
Changing the model in ONE place updates the whole pipeline.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def get_embedding_model() -> OpenAIEmbeddings:
    """
    Returns a configured OpenAI embedding model.

    Model: text-embedding-3-small
      - Fast, cheap, and very good quality
      - Produces 1536-dimensional vectors
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found!\n"
            "Make sure you copied .env.example to .env and filled in your API key."
        )

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )
