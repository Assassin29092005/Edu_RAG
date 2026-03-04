"""
vector_store.py — Chroma operations: create index, add documents, query.
Persists to disk so data survives app restarts.
Supports multiple collections (admin, student, formulas).
"""

import os
import logging
import chromadb
from langchain_chroma import Chroma
from src.embeddings import get_embedding_model
from src.config import child_chunk_size, child_chunk_overlap, retriever_k
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logger = logging.getLogger(__name__)

ADMIN_COLLECTION = "notego_admin"
STUDENT_COLLECTION = "notego_student"
FORMULA_COLLECTION = "notego_formulas"

# Legacy name for backward compat during migration
_LEGACY_COLLECTION = "notego_collection"

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


def get_vector_store(collection_name: str = ADMIN_COLLECTION) -> Chroma:
    """Get or create the Chroma vector store for child chunks."""
    embedding_model = get_embedding_model()

    db_path = os.path.join(_PROJECT_ROOT, "chroma_db")
    os.makedirs(db_path, exist_ok=True)

    client = chromadb.PersistentClient(path=db_path)

    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )
    return vector_store


def get_parent_document_retriever(collection_name: str = ADMIN_COLLECTION) -> ParentDocumentRetriever:
    """Get the ParentDocumentRetriever which maps child chunks to parent docs."""
    vector_store = get_vector_store(collection_name)

    role_suffix = collection_name.replace("notego_", "")
    docstore_path = os.path.join(_PROJECT_ROOT, "data", f"docstore_{role_suffix}")
    os.makedirs(docstore_path, exist_ok=True)

    fs = LocalFileStore(docstore_path)
    store = create_kv_docstore(fs)

    # Try semantic chunking, fall back to recursive character splitting
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        embedding_model = get_embedding_model()
        child_splitter = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )
        logger.info("Using SemanticChunker for collection '%s'", collection_name)
    except ImportError:
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size(),
            chunk_overlap=child_chunk_overlap()
        )
        logger.info("Using RecursiveCharacterTextSplitter for collection '%s'", collection_name)

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs={"k": retriever_k()}
    )

    return retriever


def add_documents_to_store(docs: list[dict], collection_name: str = ADMIN_COLLECTION) -> int:
    """
    Add full parent documents to the retriever. It handles splitting and storing.

    Args:
        docs: List of document dicts {"text": "...", "source": "xyz", "page": 1, "type": "pdf"}
        collection_name: Which collection to add to

    Returns:
        Number of parent documents added
    """
    if not docs:
        return 0

    retriever = get_parent_document_retriever(collection_name)

    lc_docs = []
    seen_ids = set()
    for doc in docs:
        safe_source = "".join([c if c.isalnum() else "_" for c in doc['source']])
        doc_id = f"{safe_source}_obj{doc['page']}_{hash(doc['text']) % 10000}"
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)

            # Contextual chunk header — makes embeddings topic-aware
            header = f"[Document: {doc['source']} | Page: {doc['page']}]\n\n"
            content = header + doc["text"]

            lc_docs.append(Document(
                page_content=content,
                metadata={
                    "source": doc["source"],
                    "page": doc["page"],
                    "type": doc["type"],
                    "doc_id": doc_id,
                    "collection": collection_name,
                    "has_context_header": True,
                }
            ))

    ids = [d.metadata["doc_id"] for d in lc_docs]
    retriever.add_documents(lc_docs, ids=ids)

    logger.info("Added %d parent documents to collection '%s'", len(lc_docs), collection_name)
    return len(lc_docs)


def get_collection_stats(collection_name: str = ADMIN_COLLECTION) -> dict:
    """Get basic stats about what's stored in a specific collection."""
    vector_store = get_vector_store(collection_name)
    try:
        count = len(vector_store.get()["ids"])

        role_suffix = collection_name.replace("notego_", "")
        docstore_path = os.path.join(_PROJECT_ROOT, "data", f"docstore_{role_suffix}")
        os.makedirs(docstore_path, exist_ok=True)
        fs = LocalFileStore(docstore_path)
        parent_count = len(list(fs.yield_keys()))

        return {"total_chunks": count, "parent_docs": parent_count}
    except Exception:
        return {"total_chunks": 0, "parent_docs": 0}


def get_all_documents(collection_name: str = ADMIN_COLLECTION) -> list[Document]:
    """Retrieve all parent documents currently stored. (for BM25 indexing)"""
    try:
        role_suffix = collection_name.replace("notego_", "")
        docstore_path = os.path.join(_PROJECT_ROOT, "data", f"docstore_{role_suffix}")
        fs = LocalFileStore(docstore_path)
        store = create_kv_docstore(fs)

        keys = list(store.yield_keys())
        if not keys:
            return []

        docs = store.mget(keys)
        return [d for d in docs if d is not None]
    except Exception:
        return []


def get_filtered_vector_retriever(collection_name: str, source_filter: str):
    """Return a retriever filtered to a specific source filename using ChromaDB where clause."""
    vector_store = get_vector_store(collection_name)
    return vector_store.as_retriever(
        search_kwargs={
            "k": retriever_k(),
            "filter": {"source": source_filter}
        }
    )


def get_formula_store() -> Chroma:
    """Get or create the formula-specific Chroma collection."""
    return get_vector_store(FORMULA_COLLECTION)


def clear_store(collection_name: str = ADMIN_COLLECTION):
    """Wipe a specific Chroma collection and its docstore/images."""
    try:
        store = get_vector_store(collection_name)
        store.delete_collection()
        logger.info("Cleared Chroma collection '%s'", collection_name)
    except Exception as e:
        logger.error("Error clearing Chroma DB collection '%s': %s", collection_name, e)

    try:
        import shutil
        role_suffix = collection_name.replace("notego_", "")

        docstore_path = os.path.join(_PROJECT_ROOT, "data", f"docstore_{role_suffix}")
        if os.path.exists(docstore_path):
            shutil.rmtree(docstore_path)

        images_path = os.path.join(_PROJECT_ROOT, "data", f"images_{role_suffix}")
        if os.path.exists(images_path):
            shutil.rmtree(images_path)

        uploads_path = os.path.join(_PROJECT_ROOT, "data", f"uploads_{role_suffix}")
        if os.path.exists(uploads_path):
            shutil.rmtree(uploads_path)

        # Clear ingestion tracker
        try:
            from src.ingestion_tracker import clear_tracker
            clear_tracker(collection_name)
        except ImportError:
            pass

        logger.info("Cleared docstore, images, and uploads for '%s'", collection_name)
    except Exception as e:
        logger.error("Error clearing docstore/images for '%s': %s", collection_name, e)
