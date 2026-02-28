import os
from src.vector_store import get_vector_store

def view_embeddings():
    print("Loading vector store...")
    vector_store = get_vector_store()
    
    # Get all items from the collection, specifically requesting embeddings
    print("Fetching data...")
    collection_data = vector_store.get(include=["embeddings", "documents", "metadatas"])
    
    if not collection_data.get("ids") or len(collection_data["ids"]) == 0:
        print("The vector store is currently empty. Upload a document first.")
        return
        
    print(f"Total chunks in vector store: {len(collection_data['ids'])}\n")
    
    # Show the first chunk's embedding
    first_doc = collection_data["documents"][0]
    first_embedding = collection_data["embeddings"][0]
    
    print("--- FIRST CHUNK ---")
    print(f"Document Text Snippet:\n{first_doc[:150]}...\n")
    print(f"Embedding Vector Length: {len(first_embedding)} dimensions")
    print(f"Embedding Values (showing first 10 values of {len(first_embedding)}):")
    for i, val in enumerate(first_embedding[:100]):
        print(f"  [{i}]: {val}")
    print("  ...")
    
if __name__ == "__main__":
    view_embeddings()
