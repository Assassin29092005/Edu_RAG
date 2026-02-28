"""
NoteGo RAG — Course Notes Q&A System
Main Streamlit App: File upload + Chat interface
"""

import os
import streamlit as st
from src.file_parser import parse_file
from src.vector_store import get_vector_store, add_documents_to_store, get_collection_stats, clear_store
# Updated import here
from src.rag_chain import stream_rag_answer

# --- Page Config ---
st.set_page_config(
    page_title="NoteGo RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for premium look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0;
    }

    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    .source-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea44;
        border-radius: 8px;
        padding: 4px 10px;
        margin: 3px 4px 3px 0;
        font-size: 0.8rem;
        color: #667eea;
    }

    .stats-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        border: 1px solid #334155;
    }

    .stats-card h3 {
        color: #667eea;
        font-size: 1.8rem;
        margin: 0;
    }

    .stats-card p {
        color: #94a3b8;
        font-size: 0.85rem;
        margin: 5px 0 0 0;
    }

    .upload-success {
        background: linear-gradient(135deg, #065f4622, #10b98122);
        border: 1px solid #10b98144;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e293b);
    }

    .stChatMessage {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Upload directory ---
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "files_processed" not in st.session_state:
    st.session_state.files_processed = set()

# --- Sidebar: File Upload & Stats ---
with st.sidebar:
    st.markdown("### 📁 Upload Course Notes")
    st.caption("Supported: PDF, DOCX, PPTX")

    uploaded_files = st.file_uploader(
        "Drag & drop your files",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        vector_store = get_vector_store()

        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.files_processed:
                with st.spinner(f"Processing **{uploaded_file.name}**..."):
                    # Save to disk
                    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Parse and extract semantic parent documents
                    documents = parse_file(file_path)
                    num_added = add_documents_to_store(documents)

                    st.session_state.files_processed.add(uploaded_file.name)

                    st.markdown(
                        f'<div class="upload-success">'
                        f'✅ <strong>{uploaded_file.name}</strong><br>'
                        f'<small>{len(documents)} parent docs / sections indexed</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # Stats
    st.markdown("---")
    st.markdown("### 📊 Knowledge Base")
    stats = get_collection_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="stats-card"><h3>{stats.get("parent_docs", stats["total_chunks"])}</h3><p>Parent Docs Indexed</p></div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="stats-card"><h3>{len(st.session_state.files_processed)}</h3><p>Files Uploaded</p></div>',
            unsafe_allow_html=True
        )

    # Uploaded files list
    if st.session_state.files_processed:
        st.markdown("---")
        st.markdown("### 📄 Uploaded Files")
        for fname in sorted(st.session_state.files_processed):
            ext = os.path.splitext(fname)[1].lower()
            icon = {"pdf": "📕", "docx": "📘", "pptx": "📙"}.get(ext.strip("."), "📄")
            st.caption(f"{icon} {fname}")

    # Model info
    st.markdown("---")
    st.markdown("### ⚙️ Models")
    st.caption("🧠 LLM: llama3 (8B)")
    st.caption("📐 Embeddings: mxbai-embed-large")
    st.caption("💾 Vector DB: ChromaDB")

    # Database Management
    st.markdown("---")
    st.markdown("### 🛠️ Manage")
    if st.button("🗑️ Clear Database", use_container_width=True):
        with st.spinner("Deleting database..."):
            clear_store()
            st.session_state.files_processed.clear()
            st.session_state.chat_history.clear()
            st.success("Database cleared successfully!")
            st.rerun()

# --- Main Area: Chat ---
st.markdown('<h1 class="main-header">📚 NoteGo RAG</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your course notes, then ask any question about them</p>', unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("🔍 View Source Documents"):
                for s in msg["sources"]:
                    st.markdown(f"**📄 {s['source']} (Page {s['page']})**")
                    st.info(s.get("text", "Text unavailable."))

# Chat input
if prompt := st.chat_input("Ask a question about your course notes..."):
    # Check if any files are uploaded
    if stats["total_chunks"] == 0:
        st.warning("⚠️ Please upload some course notes first using the sidebar!")
    else:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Format chat history
        # We exclude the last message because it's the current prompt we just appended
        formatted_history = ""
        if len(st.session_state.chat_history) > 1:
            for msg in st.session_state.chat_history[:-1]:
                role_name = "Student" if msg["role"] == "user" else "Assistant"
                formatted_history += f"{role_name}: {msg['content']}\n"

        # Generate streamed answer
        with st.chat_message("assistant"):
            # The spinner only runs while fetching documents from the multi-query/ensemble retriever
            with st.spinner("🔍 Searching notes & generating answer..."):
                generator, sources = stream_rag_answer(prompt, chat_history=formatted_history)

            # Stream out the result. st.write_stream automatically returns the complete string.
            full_response = st.write_stream(generator)

            # Output the sources right below the typed response
            if sources:
                with st.expander("🔍 View Source Documents"):
                    for s in sources:
                        st.markdown(f"**📄 {s['source']} (Page {s['page']})**")
                        st.info(s.get("text", "Text unavailable."))

        # Save to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
        })