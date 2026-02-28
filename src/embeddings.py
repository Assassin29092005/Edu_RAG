"""
embeddings.py — Wrapper around Ollama's mxbai-embed-large model.
Provides a LangChain-compatible embedding class.
"""

from langchain_community.embeddings import OllamaEmbeddings


def get_embedding_model(model_name: str = "mxbai-embed-large") -> OllamaEmbeddings:
    """
    Returns an Ollama embedding model instance compatible with LangChain.
    
    Args:
        model_name: Ollama model to use for embeddings.
                    Default: mxbai-embed-large (best for English RAG).
                    Alternative: bge-m3 (for multilingual notes).
    """
    return OllamaEmbeddings(model=model_name)
