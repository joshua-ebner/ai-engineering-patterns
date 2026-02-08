# AI System Patterns

A collection of production-oriented patterns for building Retrieval-Augmented Generation (RAG) and agent-based AI systems.

This repository focuses on *how real AI systems are engineered*, not just how to call an LLM API.

The goal is to demonstrate practical system design decisions around retrieval, control flow, evaluation, and failure handling.

---

## What This Repository Demonstrates

This repo contains working examples and patterns for:

- Embeddings and vector-based retrieval  
- Token-aware chunking strategies  
- Deterministic ingestion pipelines  
- Agent routing and control flow  
- Prompt-driven decision logic  
- Tool selection and invocation  
- Confidence-based refusal behavior  
- Evaluation of retrieval quality and agent behavior  
- Testing with known inputs and expected outputs  
- Observing failure modes and edge cases  

The emphasis is on correctness, reasoning, and system design — not toy demos or UI wrappers.

---

## Project Goal

The system being built here is an **AI Engineering Assistant**:

A RAG/agent system grounded in real AI engineering documentation (LangChain, OpenAI, and related tooling) that can:

- Retrieve relevant technical knowledge  
- Answer grounded engineering questions  
- Refuse when context is insufficient  
- Support evaluation and testing of behavior  

At the same time, the repository serves as **external proof of RAG and agent engineering ability.**

---

## Current Status

This project is being built in phases.

### Phase 1 — RAG Foundations (In Progress)

Focus:

- Documentation ingestion  
- Token-aware chunking  
- Embedding + retrieval pipelines  
- Confidence-based refusal logic  
- Small evaluation set for retrieval quality  

Corpus sources:

- LangChain documentation  
- OpenAI documentation (embeddings and tool use)

Out of scope for Phase 1:

- Agents  
- LangGraph  
- LangSmith  
- Deployment infrastructure  

---
## Data Pipeline Overview

This repository uses a reproducible documentation pipeline
to build an AI Engineering knowledge base.

Raw documentation is not committed to the repo.
Instead, source documents are pulled programmatically.

To fetch the LangChain documentation corpus:

```bash
bash scripts/pull_langchain_docs.sh
```

This will populate:

```text
data/raw/langchain/
```

Subsequent ingestion and chunking steps operate on this data.

---

## RAG API v1

A simple Retrieval-Augmented Generation (RAG) API built with:

- FastAPI
- LangChain
- OpenAI embeddings
- Chroma vector store

This service retrieves relevant document chunks and generates grounded answers with source citations.

---

### Features

- Semantic retrieval over a local vector database
- Distance-threshold filtering
- Grounded answer generation
- Source attribution with snippets
- Refusal for low-relevance queries
- JSONL logging for analysis

---

### Running the API

Start the server:

```bash
uvicorn apps.rag.rag_api:app --reload
```

Open interactive docs:

`http://127.0.0.1:8000/docs`

Health check:
```bash
curl http://127.0.0.1:8000/health`
```

Query:
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is a retriever in LangChain?"}'
```

Response includes:
- answer
- refused flag (for low-relevance / out-of-scope queries)
- sources with distance + snippet


---


## Future Directions

Later phases introduce:

- Agent orchestration with LangGraph  
- Observability and tracing  
- Evaluation harnesses  
- Human-in-the-loop patterns  
- Broader AI system design patterns  

---

## Who This Is For

This repository is useful for:

- Teams building RAG systems  
- Engineers designing AI agents  
- Practitioners evaluating LLM reliability  
- Anyone interested in production-oriented AI system design  