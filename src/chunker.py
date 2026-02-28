"""
chunker.py — Splits extracted document text into overlapping chunks.
Uses LangChain's RecursiveCharacterTextSplitter for intelligent splitting.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents: list[dict], chunk_size: int = 500, chunk_overlap: int = 100) -> list[dict]:
    """
    Split documents into smaller, overlapping chunks.
    
    Args:
        documents: List of dicts with keys: text, source, page, type
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
    
    Returns:
        List of chunk dicts with keys: text, source, page, type, chunk_id
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    chunk_id = 0

    for doc in documents:
        split_texts = splitter.split_text(doc["text"])

        for text in split_texts:
            chunks.append({
                "text": text,
                "source": doc["source"],
                "page": doc["page"],
                "type": doc["type"],
                "chunk_id": f"{doc['source']}_p{doc['page']}_c{chunk_id}"
            })
            chunk_id += 1

    return chunks
