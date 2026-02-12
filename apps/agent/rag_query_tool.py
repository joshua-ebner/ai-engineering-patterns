from typing import Any, Dict
import os
import httpx
from langchain_core.tools import tool

# Configurable endpoint
RAG_API_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000/query"
)

@tool
def rag_query_tool(query: str) -> Dict[str, Any]:
    """
    Query the RAG API for grounded answers from AI engineering documentation.

    Returns a dict containing:
    - answer
    - refused (bool)
    - sources
    - refusal_reason
    """

    payload = {"query": query.strip()}

    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(RAG_API_URL, json=payload)
            resp.raise_for_status()
            return resp.json()

    except Exception as e:
        # Tool-level failure (different from RAG refusal)
        return {
            "answer": "RAG service unavailable.",
            "refused": True,
            "sources": [],
            "refusal_reason": "tool_error",
            "error": str(e),
        }
