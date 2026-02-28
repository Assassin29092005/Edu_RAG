"""
rag_chain.py — RAG pipeline using LangChain Expression Language (LCEL).
Ties retrieval (ChromaDB) + generation (Ollama llama3.1) together.
"""

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.vector_store import get_vector_store

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


def ask_question(question: str, model_name: str = "llama3.1:8b") -> dict:
    """
    Ask a question and get an answer grounded in the uploaded course notes.
    
    Args:
        question: The student's question
        model_name: Ollama model name
    
    Returns:
        Dict with keys: answer, sources
    """
    llm = Ollama(model=model_name, temperature=0.1)
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Retrieve docs first (we need them for both the chain and source extraction)
    retrieved_docs = retriever.invoke(question)

    # Build LCEL chain
    chain = (
        {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)

    # Extract source references
    sources = []
    seen = set()
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({"source": source, "page": page})

    return {
        "answer": answer,
        "sources": sources,
    }
