"""
2_query.py  —  TASK 2: Query the ChromaDB Vector Store

HOW IT WORKS:
  1. Connect to the existing ChromaDB collection (built in Task 1)
  2. Accept a query string from the command line
  3. Convert the query into a vector using the same embedding model
  4. Search ChromaDB for the most similar chunk vectors
  5. Return results ranked by similarity score

WHAT IS A SIMILARITY SCORE?
  - ChromaDB returns a "distance" score (cosine distance)
  - Score closer to 0.0  → very similar  (high relevance)
  - Score closer to 2.0  → very different (low relevance)
  - We display it as a relevance % for readability

RUN:
  python 2_query.py "What is multi-head attention?"
  python 2_query.py "How does the encoder work?"
  python 2_query.py "What optimizer was used for training?"
"""

import os
import sys
from dotenv import load_dotenv

from utils.vectorstore import get_vectorstore

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", 5))   # how many results to return


def distance_to_relevance(distance: float) -> float:
    """
    Convert ChromaDB cosine distance → relevance percentage.

    ChromaDB uses cosine distance (0 = identical, 2 = opposite).
    We convert it to a 0-100% relevance score for readability:
      relevance = (1 - distance/2) * 100
    """
    return round((1 - distance / 2) * 100, 1)


def query_vectorstore(query: str) -> list[tuple]:
    """
    Step 1-5: Embed the query and search ChromaDB.

    similarity_search_with_score():
      - Takes a plain text query
      - Internally embeds it using the same model used during ingestion
      - Compares the query vector against all stored vectors
      - Returns the top-K most similar (Document, distance_score) pairs
    """
    print(f"\nConnecting to ChromaDB...")
    vectorstore = get_vectorstore()

    print(f"Running similarity search for: \"{query}\"")
    print(f"Returning top {TOP_K} results\n")

    # Returns: list of (Document, float) where float is cosine distance
    results = vectorstore.similarity_search_with_score(query, k=TOP_K)
    return results


def display_results(results: list[tuple], query: str) -> None:
    """
    Print results in a clean, readable format with scores and metadata.
    """
    print("=" * 70)
    print(f"  QUERY  : {query}")
    print(f"  RESULTS: {len(results)} chunks retrieved")
    print("=" * 70)

    if not results:
        print("\n  No results found. Make sure you have run 1_ingest.py first.")
        return

    for rank, (doc, distance) in enumerate(results, start=1):
        relevance = distance_to_relevance(distance)
        source    = doc.metadata.get("source", "unknown")
        page      = doc.metadata.get("page", "?")
        content   = doc.page_content.strip().replace("\n", " ")

        # Truncate long content for display
        preview = content[:300] + "..." if len(content) > 300 else content

        print(f"\n  Rank #{rank}")
        print(f"  Relevance : {relevance}%  (cosine distance: {distance:.4f})")
        print(f"  Source    : {os.path.basename(source)}  |  Page: {page + 1}")
        print(f"  Content   : {preview}")
        print(f"  {'-' * 66}")


def main():
    # Accept query from command line, or use a default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What is multi-head attention?"
        print(f"No query provided. Using default: \"{query}\"")

    results = query_vectorstore(query)
    display_results(results, query)

    print(f"\n  Next step: Run 'python 3_langgraph_pipeline.py' for the full RAG pipeline.")


if __name__ == "__main__":
    main()
