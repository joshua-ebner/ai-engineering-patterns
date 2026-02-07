# apps/rag/retrieval_qa_v1.py


from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from typing import List, Tuple

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document



BASE_DIR = Path(__file__).resolve().parents[2]
PERSIST_DIR = str(BASE_DIR / "data/vectorstore/langchain_db")


# Heuristic relevance threshold.
# Lower distance = more similar.
# Tune later if needed.
MAX_DISTANCE = 1.2


def format_context(documents: List[Document]) -> str:
    formatted_chunks = []

    for doc_number, document in enumerate(documents, 1):
        source = document.metadata.get("source", "unknown")

        chunk_text = (
            f"[Chunk {doc_number} | Source: {source}]\n"
            f"{document.page_content}"
        )

        formatted_chunks.append(chunk_text)

    return "\n\n".join(formatted_chunks)



def retrieve_with_threshold(
    vectordb: Chroma,
    query: str,
    k: int = 5,
) -> List[Tuple[Document, float]]:
    
    results = vectordb.similarity_search_with_score(query, k=k)
    
    filtered = [(doc, score) for doc, score in results if score <= MAX_DISTANCE]

    return filtered


def main():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    query = input("\nQuestion: ").strip()


    results = retrieve_with_threshold(vectordb, query, k=5)


    if not results:
        print("\nAnswer:\n")
        print("I don't have enough relevant context to answer confidently.")
        print("\nSources: none (low relevance retrieval)")
        return

    docs = [doc for doc, _ in results]

    context = format_context(docs)

    prompt = f"""
You are an AI engineering assistant.

Answer the question using ONLY the context below.
If the answer is not contained in the context, say you don't know.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    print("\nAnswer:\n")
    print(response.content)

    print("\nSources:")
    for doc, score in results:
        print(f"- {doc.metadata.get('source', 'unknown')} (distance={score:.3f})")


if __name__ == "__main__":
    main()
