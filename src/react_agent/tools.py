"""This module provides custom tools for the Airflow Error Analysis agent.

It includes a BM25S + ChromaDB hybrid search tool and a source code reading tool.
"""

import json
import logging
from typing import Any, Callable, List, Optional, cast

import bm25s
import chromadb
from langchain_core.tools import tool
from Stemmer import Stemmer

logger = logging.getLogger(__name__)

# Initialize clients (ensure these don't fail if indices are missing at import time)
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="airflow_errors")

    bm25_retriever = bm25s.BM25.load("./bm25s_index", load_corpus=True)
    with open("./bm25s_index/metadata.json", "r") as f:
        bm25_metadata = json.load(f)
    stemmer = Stemmer("english")
    indices_loaded = True
except Exception as e:
    logger.warning(
        f"Failed to load Hybrid Indices. Ensure vectore_store.py has been run. Error: {e}"
    )
    indices_loaded = False


def hybrid_search(query: str, top_k: int = 2) -> str:
    """Combines BM25S and ChromaDB vector search results."""
    if not indices_loaded:
        return "Indices not loaded. Please run vectore_store.py first."

    logger.info(f"🔎 hybrid_search started with top_k={top_k}")

    # 1. BM25S search
    query_tokens = bm25s.tokenize([query], stemmer=stemmer)
    bm25_docs, _ = bm25_retriever.retrieve(query_tokens, k=top_k)
    logger.info(f"📄 BM25S retrieved {len(bm25_docs[0])} documents")

    # 2. Vector search
    vector_results = collection.query(query_texts=[query], n_results=top_k)
    docs = vector_results.get("documents", [[]])[0]
    logger.info(f"📊 Vector DB retrieved {len(docs)} documents")

    # 3. Deduplicate and construct context
    unique_contexts = set()

    for doc in bm25_docs[0]:
        unique_contexts.add(f"[Keyword Match] {doc}")

    for doc in docs:
        unique_contexts.add(f"[Semantic Match] {doc}")

    result = "\\n\\n".join(list(unique_contexts))
    logger.info(f"✨ hybrid_search combined {len(unique_contexts)} unique results")
    return result


@tool
def search_error_guide(query: str) -> str:
    """
    Searches for relevant error guides from the 30 error clusters and knowledge base.
    Use this when classifying error types or seeking resolution steps.
    """
    logger.info(f"🔍 search_error_guide called with query: {query}")
    result = hybrid_search(query)
    logger.info(f"✅ search_error_guide returned result of length {len(result)}")
    return result


@tool
def read_failed_source_code(file_path: str, line_number: int) -> str:
    """
    Reads the source code around the specified line number of the given file.
    Use this to find logic flaws when the error log itself is insufficient.
    """
    logger.info(
        f"📂 read_failed_source_code called - file: {file_path}, line: {line_number}"
    )
    window = 10
    try:
        logger.debug(f"📖 Reading file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if line_number:
            start = max(0, line_number - window)
            end = min(len(lines), line_number + window)
            code_snippet = "".join(lines[start:end])
            logger.info(f"📝 Extracted code snippet from line {start} to {end}")
            return f"--- Code Snippet ({file_path}) ---\\n{code_snippet}"

        logger.info(f"📝 Returning first 100 lines of {file_path}")
        return "".join(lines[:100])
    except Exception as e:
        error_msg = f"Error reading file: {str(e)}"
        logger.error(error_msg)
        return error_msg


TOOLS: List[Callable[..., Any]] = [search_error_guide, read_failed_source_code]
