"""
3_langgraph_pipeline.py  —  TASK 3: Full LangGraph RAG Pipeline

WHAT IS LANGGRAPH?
  LangGraph is a library for building stateful, graph-based workflows.
  Instead of calling functions manually one by one, you define:
    - Nodes  : individual processing steps (functions)
    - Edges  : connections that define the order of execution
    - State  : a shared dictionary that flows through every node

PIPELINE GRAPH:
  [START]
     |
     v
  [retrieve_node]   — searches ChromaDB, returns top-K docs with scores
     |
     v
  [filter_node]     — drops docs whose distance score exceeds the threshold
     |
     v
  [generate_node]   — builds a prompt from the filtered docs and calls the LLM
     |
     v
  [END]

STATE:
  Every node reads from and writes to a shared RAGState dict:
    query         : the original question (set at START, never changed)
    raw_docs      : (Document, distance) pairs returned by retrieve_node
    filtered_docs : Documents that passed the score threshold
    answer        : the final LLM response (set by generate_node)

RUN:
  python 3_langgraph_pipeline.py "What is multi-head attention?"
  python 3_langgraph_pipeline.py "How does the encoder work?"
"""

import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END

from utils.vectorstore import get_vectorstore

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
TOP_K           = int(os.getenv("TOP_K", 10))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 1.0))    # cosine distance
                                                               # < 1.0 = > 50% relevance
                                                               # all-MiniLM-L6-v2 distances
                                                               # typically range 0.7 – 1.5


# ══════════════════════════════════════════════════════════════════════════════
# STATE SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class RAGState(TypedDict):
    """
    The shared state object that flows through every node in the graph.

    TypedDict makes state explicit and type-safe — every node knows exactly
    what keys are available and what type they hold.
    """
    query         : str                           # original user question
    raw_docs      : list[tuple[Document, float]]  # (doc, cosine_distance) pairs
    filtered_docs : list[Document]                # docs that passed threshold
    answer        : str                           # final response


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1: RETRIEVE
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_node(state: RAGState) -> dict:
    """
    Node 1: Query ChromaDB and retrieve the top-K most similar chunks.

    - Reads  : state["query"]
    - Writes : state["raw_docs"]  (list of (Document, cosine_distance) tuples)

    similarity_search_with_score():
      Internally embeds the query text using the same model used during ingestion,
      then computes cosine distance against all stored vectors and returns the
      closest matches ranked by distance (ascending — lower is better).
    """
    print("\n[Node 1: RETRIEVE]")
    print(f"  Query       : {state['query']}")
    print(f"  Fetching top {TOP_K} results from ChromaDB...")

    vectorstore = get_vectorstore()
    raw_docs = vectorstore.similarity_search_with_score(state["query"], k=TOP_K)

    print(f"  Retrieved   : {len(raw_docs)} documents")
    for i, (doc, score) in enumerate(raw_docs, 1):
        relevance = round((1 - score / 2) * 100, 1)
        page = doc.metadata.get("page", "?") + 1
        print(f"    #{i}  distance={score:.4f}  relevance={relevance}%  page={page}")

    return {"raw_docs": raw_docs}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2: FILTER
# ══════════════════════════════════════════════════════════════════════════════

def filter_node(state: RAGState) -> dict:
    """
    Node 2: Drop documents whose cosine distance exceeds SCORE_THRESHOLD.

    - Reads  : state["raw_docs"]
    - Writes : state["filtered_docs"]

    WHY FILTER?
      Not all retrieved documents are relevant — especially for queries that use
      different terminology than the source text. Without filtering, low-quality
      chunks pollute the LLM context and degrade answer quality.

    THRESHOLD LOGIC (cosine distance):
      distance < SCORE_THRESHOLD  →  KEEP  (relevant)
      distance >= SCORE_THRESHOLD →  DROP  (not relevant enough)

      e.g. SCORE_THRESHOLD = 0.80
        distance 0.72 → relevance 64% → KEEP
        distance 0.92 → relevance 54% → KEEP
        distance 1.16 → relevance 42% → DROP
    """
    print(f"\n[Node 2: FILTER]")
    print(f"  Threshold   : distance < {SCORE_THRESHOLD}  (relevance > {round((1 - SCORE_THRESHOLD / 2) * 100, 1)}%)")

    filtered = []
    dropped  = []

    for doc, distance in state["raw_docs"]:
        relevance = round((1 - distance / 2) * 100, 1)
        if distance < SCORE_THRESHOLD:
            filtered.append(doc)
            status = "KEEP"
        else:
            dropped.append(doc)
            status = "DROP"
        page = doc.metadata.get("page", "?") + 1
        print(f"  {status}  distance={distance:.4f}  relevance={relevance}%  page={page}")

    print(f"\n  Kept  : {len(filtered)} documents")
    print(f"  Dropped: {len(dropped)} documents")

    return {"filtered_docs": filtered}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3: GENERATE
# ══════════════════════════════════════════════════════════════════════════════

def generate_node(state: RAGState) -> dict:
    """
    Node 3: Build a prompt from the filtered context and call the LLM.

    - Reads  : state["query"], state["filtered_docs"]
    - Writes : state["answer"]

    If OPENAI_API_KEY is set  → calls ChatOpenAI (gpt-4o-mini)
    If no API key             → returns a structured summary of the context
                                (so the pipeline still runs end-to-end for learning)

    PROMPT STRUCTURE:
      System : You are a helpful assistant. Answer using only the provided context.
      Human  : Context: <filtered chunks joined>\n\nQuestion: <query>
    """
    print(f"\n[Node 3: GENERATE]")
    print(f"  Context docs: {len(state['filtered_docs'])}")

    if not state["filtered_docs"]:
        answer = (
            "No relevant documents found above the relevance threshold. "
            "Try rephrasing your question or lowering the SCORE_THRESHOLD."
        )
        print(f"  No context available — returning fallback answer.")
        return {"answer": answer}

    # Build the context string from filtered documents
    context_parts = []
    for i, doc in enumerate(state["filtered_docs"], 1):
        page = doc.metadata.get("page", "?") + 1
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        context_parts.append(
            f"[Source: {source}, Page {page}]\n{doc.page_content.strip()}"
        )
    context = "\n\n---\n\n".join(context_parts)

    api_key = os.getenv("OPENAI_API_KEY", "")

    if api_key and api_key.startswith("sk-"):
        # Real LLM call via OpenAI
        print("  LLM         : ChatOpenAI (gpt-4o-mini)")
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)

        messages = [
            SystemMessage(content=(
                "You are a helpful research assistant. "
                "Answer the question using ONLY the provided context. "
                "If the context does not contain the answer, say so clearly."
            )),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['query']}")
        ]

        response = llm.invoke(messages)
        answer   = response.content

    else:
        # No API key — show what the LLM would receive (educational output)
        print("  LLM         : No API key set — showing context that would be sent to LLM")
        answer = (
            f"[LLM would receive the following context for: \"{state['query']}\"]\n\n"
            f"{'=' * 60}\n"
            f"{context}\n"
            f"{'=' * 60}\n\n"
            f"Set OPENAI_API_KEY in your .env file to get a real generated answer."
        )

    return {"answer": answer}


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def build_graph() -> object:
    """
    Assemble the LangGraph StateGraph.

    StateGraph(RAGState):
      Creates a graph where every node shares the RAGState schema.

    add_node(name, function):
      Registers a function as a named node. The function must accept a RAGState
      dict and return a partial dict with the keys it wants to update.

    add_edge(from, to):
      Defines the execution order. LangGraph supports conditional edges too
      (e.g. branch based on state values) — used in more advanced pipelines.

    compile():
      Validates the graph (no disconnected nodes, valid edges) and returns
      a runnable object that can be invoked with .invoke(initial_state).
    """
    graph = StateGraph(RAGState)

    # Register nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("filter",   filter_node)
    graph.add_node("generate", generate_node)

    # Define execution order
    graph.add_edge(START,      "retrieve")
    graph.add_edge("retrieve", "filter")
    graph.add_edge("filter",   "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is multi-head attention?"

    print("=" * 60)
    print("  LangGraph RAG Pipeline")
    print("  START → retrieve → filter → generate → END")
    print("=" * 60)
    print(f"\n  Graph nodes : retrieve, filter, generate")
    print(f"  Threshold   : distance < {SCORE_THRESHOLD}")
    print(f"  Top-K       : {TOP_K}")

    # Build the compiled graph
    app = build_graph()

    # Define the initial state — only query is set; all other fields start empty
    initial_state: RAGState = {
        "query"         : query,
        "raw_docs"      : [],
        "filtered_docs" : [],
        "answer"        : "",
    }

    # Run the full graph — LangGraph automatically passes state between nodes
    final_state = app.invoke(initial_state)

    # Display the final answer
    print("\n" + "=" * 60)
    print("  FINAL ANSWER")
    print("=" * 60)
    print(f"\n  Query : {final_state['query']}")
    print(f"\n  Answer:\n")
    for line in final_state["answer"].split("\n"):
        print(f"  {line}")

    print("\n" + "=" * 60)
    print(f"  Pipeline complete.")
    print(f"  Docs retrieved : {len(final_state['raw_docs'])}")
    print(f"  Docs after filter: {len(final_state['filtered_docs'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
