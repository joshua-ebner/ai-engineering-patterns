# apps/rag/retrieve.py

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = "data/vectorstore/langchain_db"

def main():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    query = input("\nEnter query: ")

    results = vectordb.similarity_search(query, k=5)

    print("\nTop Results:\n")

    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")

        print(f"--- Result {i} ---")
        print(f"Source: {source}")
        print(doc.page_content[:500])
        print()

if __name__ == "__main__":
    main()
