# RAG Demo Apps

This folder contains runnable Retrieval-Augmented Generation (RAG) demos built on top of the repository’s ingestion and embedding pipelines.

These scripts demonstrate practical RAG patterns over a real documentation corpus (LangChain docs).

---

## Scripts

### `retrieve.py`

Minimal retrieval demo.

- Loads the persisted Chroma vector store
- Runs similarity search
- Prints top-k chunks with sources and distances

Use this to inspect raw retrieval behavior and debug chunk quality.

Run:

```bash
python apps/rag/retrieve.py
```

### retrieval_qa_v1.py

Baseline RAG QA system.

Implements:

- Retrieval with similarity scores
- Relevance threshold filtering
- Grounded generation (LLM answers using retrieved context only)
- Explicit refusal when context is insufficient
- Source citation with distances

This represents a clean v1 RAG pipeline suitable for evaluation and iteration.

Run:

```bash
python apps/rag/retrieval_qa_v1.py
```

## Notes
- Both scripts expect a populated vector store in data/vectorstore/
- Build the vector DB using the ingestion scripts in /scripts first
- These demos are intentionally simple and form the foundation for evaluation and more advanced RAG patterns