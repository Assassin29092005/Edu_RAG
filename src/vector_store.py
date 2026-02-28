"""
vector_store.py — Chroma operations: create index, add chunks, query.
Persists to disk so data survives app restarts.
"""

import os
import chromadb
from langchain_chroma import Chroma
from src.embeddings import get_embedding_model
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "notego_collection"

def get_vector_store() -> Chroma:
    """
    Get or create the Chroma vector store.
    """
    embedding_model = get_embedding_model()

    client = chromadb.CloudClient(
      api_key=os.environ.get("CHROMA_API_KEY"),
      tenant=os.environ.get("CHROMA_TENANT"),
      database=os.environ.get("CHROMA_DATABASE")
    )

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
    )
    return vector_store

def add_chunks_to_store(chunks: list[dict], vector_store: Chroma) -> int:
    """
    Add text chunks to the vector store and save to disk.
    
    Args:
        chunks: List of chunk dicts
        vector_store: The Chroma vector store instance
    
    Returns:
        Number of chunks added
    """
    if not chunks:
        return 0

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "type": chunk["type"],
        }
        for chunk in chunks
    ]
    ids = [chunk["chunk_id"] for chunk in chunks]

    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    
    return len(chunks)

def query_store(query: str, vector_store: Chroma, k: int = 5) -> list[dict]:
    """
    Search the vector store for chunks most relevant to the query.
    
    Args:
        query: The student's question
        vector_store: The Chroma vector store instance
        k: Number of top results to return
    
    Returns:
        List of dicts with keys: text, source, page
    """
    results = vector_store.similarity_search(query, k=k)

    return [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "?"),
        }
        for doc in results
    ]

def get_collection_stats(vector_store: Chroma) -> dict:
    """Get basic stats about what's stored in the vector store."""
    try:
        count = len(vector_store.get()["ids"])
        return {"total_chunks": count}
    except Exception:
        return {"total_chunks": 0}

from langchain_core.documents import Document

def get_all_documents(vector_store: Chroma) -> list[Document]:
    """Retrieve all documents currently stored in the vector database for BM25 indexing."""
    try:
        data = vector_store.get()
        docs = []
        if data and data.get("documents") and data.get("metadatas"):
            for text, meta in zip(data["documents"], data["metadatas"]):
                docs.append(Document(page_content=text, metadata=meta))
        return docs
    except Exception:
        return []
