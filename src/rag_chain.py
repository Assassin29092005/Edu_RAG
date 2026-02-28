"""
rag_chain.py — RAG pipeline using LangChain Expression Language (LCEL).
Ties retrieval (ChromaDB) + generation (Ollama llama3.1) together.
"""

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from src.vector_store import get_vector_store, get_collection_stats, get_all_documents, get_parent_document_retriever

@st.cache_resource(show_spinner=False)
def get_bm25_retriever(collection_stats_hash):
    """
    Cache the BM25 retriever so it doesn't rebuild on every query.
    We pass a hash/count of the collection stats to invalidate the cache when new files are uploaded.
    """
    docs = get_all_documents()
    if not docs:
        return None
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5
    return bm25_retriever

# Prompt template that keeps the LLM grounded in course material
RAG_PROMPT_TEMPLATE = """You are a helpful teaching assistant. Answer the student's question 
using ONLY the following context from their course notes.

Rules:
- Answer accurately based on the context provided.
- If the answer is not in the context, say "I couldn't find this information in your uploaded notes."
- Cite the source file and page/slide number when possible.
- Keep your answer clear, well-structured, and student-friendly.

Context from course notes:
{context}

Chat History:
{chat_history}

Student's Question: {question}

Helpful Answer:"""

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


def format_docs(docs):
    """Format retrieved documents into a single context string."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {source}, Page/Slide: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def stream_rag_answer(question: str, chat_history: str = "", model_name: str = "llama3.1:8b"):
    """
    Ask a question and stream an answer grounded in the uploaded course notes.
    
    Args:
        question: The student's question
        chat_history: Previous conversation context, formatted as a string
        model_name: Ollama model name
    
    Returns:
        Tuple containing the generator stream and the source references.
    """
    # Added streaming=True
    llm = Ollama(model=model_name, temperature=0.1, streaming=True)
    vector_retriever = get_parent_document_retriever()

    # BM25 Keyword retriever
    stats = get_collection_stats()
    bm25_retriever = get_bm25_retriever(stats.get("parent_docs", stats["total_chunks"]))
    
    if bm25_retriever:
        base_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
    else:
        base_retriever = vector_retriever

    # Multi-Query Expansion
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    # Retrieve docs first (we need them for both the chain and source extraction)
    retrieved_docs = retriever.invoke(question)
    context_str = format_docs(retrieved_docs)

    # Build LCEL chain
    chain = (
        {
            "context": lambda x: context_str, 
            "chat_history": lambda x: chat_history, 
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    # Extract source references and exact text
    sources = []
    seen = set()
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": source, 
                "page": page,
                "text": doc.page_content
            })

    # Return the stream directly along with extracted sources
    return chain.stream(question), sources