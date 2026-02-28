"""
vector_store.py — Chroma operations: create index, add documents, query.
Persists to disk so data survives app restarts.
"""

import os
import chromadb
from langchain_chroma import Chroma
from src.embeddings import get_embedding_model
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

COLLECTION_NAME = "notego_collection"

def get_vector_store() -> Chroma:
    """Get or create the Chroma vector store for child chunks."""
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

def get_parent_document_retriever() -> ParentDocumentRetriever:
    """Get the ParentDocumentRetriever which maps child chunks to parent docs."""
    vector_store = get_vector_store()
    
    # Persistent store for Parent Documents
    docstore_path = os.path.join(os.path.dirname(__file__), "..", "data", "docstore")
    os.makedirs(docstore_path, exist_ok=True)
    
    fs = LocalFileStore(docstore_path)
    store = create_kv_docstore(fs)
    
    # Splitter for child chunks (these get embedded)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs={"k": 5} # Number of child chunks to match
    )
    
    return retriever

def add_documents_to_store(docs: list[dict]) -> int:
    """
    Add full parent documents to the retriever. It handles splitting and storing.
    
    Args:
        docs: List of document dicts {"text": "...", "source": "xyz", "page": 1, "type": "pdf"}
    
    Returns:
        Number of parent documents added
    """
    if not docs:
        return 0

    retriever = get_parent_document_retriever()
    
    lc_docs = []
    seen_ids = set()
    for doc in docs:
        # Create a unique ID for the parent document
        doc_id = f"{doc['source']}_obj{doc['page']}_{hash(doc['text']) % 10000}"
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            lc_docs.append(Document(
                page_content=doc["text"],
                metadata={"source": doc["source"], "page": doc["page"], "type": doc["type"], "doc_id": doc_id}
            ))

    ids = [d.metadata["doc_id"] for d in lc_docs]
    retriever.add_documents(lc_docs, ids=ids)
    
    return len(lc_docs)

def get_collection_stats() -> dict:
    """Get basic stats about what's stored."""
    vector_store = get_vector_store()
    try:
        count = len(vector_store.get()["ids"])
        
        # Approximate parent doc count
        docstore_path = os.path.join(os.path.dirname(__file__), "..", "data", "docstore")
        os.makedirs(docstore_path, exist_ok=True)
        fs = LocalFileStore(docstore_path)
        parent_count = len(list(fs.yield_keys()))

        return {"total_chunks": count, "parent_docs": parent_count}
    except Exception:
        return {"total_chunks": 0, "parent_docs": 0}

def get_all_documents() -> list[Document]:
    """Retrieve all parent documents currently stored. (for BM25 indexing)"""
    try:
        docstore_path = os.path.join(os.path.dirname(__file__), "..", "data", "docstore")
        fs = LocalFileStore(docstore_path)
        store = create_kv_docstore(fs)
        
        keys = list(store.yield_keys())
        if not keys: return []
        
        docs = store.mget(keys)
        return [d for d in docs if d is not None]
    except Exception:
        return []
