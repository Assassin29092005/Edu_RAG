"""
NoteGo RAG — Admin Panel
Upload course notes, manage knowledge base, and chat with RAG.
"""

import os
import re
import streamlit as st
from src.styles import inject_css
from src.file_parser import parse_file, parse_file_streaming
from src.vector_store import (
    add_documents_to_store, get_collection_stats, clear_store,
    get_all_documents, ADMIN_COLLECTION
)
from src.rag_chain import (
    stream_rag_answer, verify_answer_against_context, format_docs,
    detect_comparison_intent, stream_comparative_answer
)
from src.ingestion_tracker import (
    is_already_processed, mark_as_processed, get_all_processed_filenames
)
from src.formula_extractor import extract_formulas, is_formula_query
from src.quiz_chain import generate_quiz
from streamlit_pdf_viewer import pdf_viewer
from src.pdf_utils import get_pdf_annotations
from src.config import (
    llm_model, vision_model, embeddings_model,
    verification_enabled, memory_enabled, memory_max_token_limit,
    quiz_default_num_questions, quiz_default_type
)

inject_css()

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads_admin")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images_admin")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- Session state ---
if "admin_chat_history" not in st.session_state:
    st.session_state.admin_chat_history = []
if "admin_files_processed" not in st.session_state:
    st.session_state.admin_files_processed = set(get_all_processed_filenames(ADMIN_COLLECTION))

# --- Conversation Memory ---
if memory_enabled() and "admin_memory" not in st.session_state:
    try:
        from langchain.memory import ConversationSummaryBufferMemory
        from langchain_ollama import OllamaLLM
        memory_llm = OllamaLLM(model=llm_model(), temperature=0)
        st.session_state.admin_memory = ConversationSummaryBufferMemory(
            llm=memory_llm,
            max_token_limit=memory_max_token_limit(),
            return_messages=False,
        )
    except Exception:
        st.session_state.admin_memory = None

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📁 Upload Course Notes")
    st.caption("Supported: PDF, DOCX, PPTX")

    # Mode selector
    mode = st.radio("Mode", ["Chat", "Quiz"], horizontal=True, key="admin_mode")

    uploaded_files = st.file_uploader(
        "Drag & drop your files",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="admin_uploader",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

            # Save to disk first
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Check incremental ingestion
            if is_already_processed(file_path, ADMIN_COLLECTION):
                if uploaded_file.name not in st.session_state.admin_files_processed:
                    st.session_state.admin_files_processed.add(uploaded_file.name)
                st.info(f"'{uploaded_file.name}' is already indexed.")
                continue

            if uploaded_file.name not in st.session_state.admin_files_processed:
                # Progress bar for large files
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                if ext == ".pdf":
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    all_docs = []
                    for doc_dict, current_page, total_pages in parse_file_streaming(file_path, IMAGES_DIR):
                        all_docs.append(doc_dict)
                        progress_bar.progress(current_page / total_pages)
                        status_text.text(f"Processing page {current_page}/{total_pages}...")
                    progress_bar.empty()
                    status_text.empty()
                    documents = all_docs
                else:
                    with st.spinner(f"Processing **{uploaded_file.name}**..."):
                        documents = parse_file(file_path, IMAGES_DIR)

                num_added = add_documents_to_store(documents, ADMIN_COLLECTION)

                # Extract and store formulas
                for doc in documents:
                    formulas = extract_formulas(doc["text"], doc["source"], doc["page"])
                    if formulas:
                        from src.vector_store import add_documents_to_store as add_docs, FORMULA_COLLECTION
                        formula_docs = [{
                            "text": f"Formula: {f['formula']}\nContext: {f['context']}",
                            "source": f["source"],
                            "page": f["page"],
                            "type": "formula",
                        } for f in formulas]
                        add_docs(formula_docs, FORMULA_COLLECTION)

                mark_as_processed(file_path, ADMIN_COLLECTION)
                st.session_state.admin_files_processed.add(uploaded_file.name)

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
    stats = get_collection_stats(ADMIN_COLLECTION)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="stats-card"><h3>{stats.get("parent_docs", stats["total_chunks"])}</h3><p>Parent Docs Indexed</p></div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="stats-card"><h3>{len(st.session_state.admin_files_processed)}</h3><p>Files Uploaded</p></div>',
            unsafe_allow_html=True
        )

    # Uploaded files list
    if st.session_state.admin_files_processed:
        st.markdown("---")
        st.markdown("### 📄 Uploaded Files")
        for fname in sorted(st.session_state.admin_files_processed):
            ext = os.path.splitext(fname)[1].lower()
            icon = {"pdf": "📕", "docx": "📘", "pptx": "📙"}.get(ext.strip("."), "📄")
            st.caption(f"{icon} {fname}")

    # Model info
    st.markdown("---")
    st.markdown("### ⚙️ Models")
    st.caption(f"🧠 LLM: {llm_model()}")
    st.caption(f"👁️ Vision: {vision_model()}")
    st.caption(f"📐 Embeddings: {embeddings_model()}")
    st.caption("💾 Vector DB: ChromaDB")

    st.info("💡 **Tip for Images:** Before uploading PDFs/PPTXs with complex diagrams, ensure you've run `ollama pull llava:7b` in your terminal.")

    # Database Management
    st.markdown("---")
    st.markdown("### 🛠️ Manage")
    if st.button("🗑️ Clear Admin Database", use_container_width=True, key="admin_clear"):
        with st.spinner("Deleting admin database..."):
            clear_store(ADMIN_COLLECTION)
            st.session_state.admin_files_processed.clear()
            st.session_state.admin_chat_history.clear()
            if "admin_memory" in st.session_state and st.session_state.admin_memory:
                st.session_state.admin_memory.clear()
            st.success("Admin database cleared successfully!")
            st.rerun()

# --- Main Area ---
st.markdown('<h1 class="main-header">📚 NoteGo RAG — Admin Panel</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload course notes, then ask any question about them</p>', unsafe_allow_html=True)

if mode == "Quiz":
    # --- Quiz Mode ---
    st.markdown("### 🧪 Quiz Generator")

    quiz_topic = st.text_input("Topic (e.g., 'sorting algorithms', 'neural networks')", key="admin_quiz_topic")

    qcol1, qcol2 = st.columns(2)
    with qcol1:
        q_type = st.selectbox("Question Type", ["MCQ", "Short Answer"],
                              index=0 if quiz_default_type() == "mcq" else 1,
                              key="admin_q_type")
    with qcol2:
        num_q = st.slider("Number of Questions", 1, 10, quiz_default_num_questions(), key="admin_num_q")

    # Source filter
    source_filter = None
    if st.session_state.admin_files_processed:
        sources = sorted(st.session_state.admin_files_processed)
        selected_source = st.selectbox("Filter by source (optional)", ["All Sources"] + sources,
                                       key="admin_quiz_source")
        if selected_source != "All Sources":
            source_filter = selected_source

    if st.button("Generate Quiz", use_container_width=True, type="primary", key="admin_gen_quiz"):
        if not quiz_topic:
            st.warning("Please enter a topic.")
        elif stats["total_chunks"] == 0:
            st.warning("Please upload some course notes first!")
        else:
            with st.spinner("Generating quiz..."):
                q_type_key = "mcq" if q_type == "MCQ" else "short_answer"
                quiz_stream = generate_quiz(quiz_topic, num_q, q_type_key,
                                            ADMIN_COLLECTION, source_filter)
                full_quiz = st.write_stream(quiz_stream)

else:
    # --- Chat Mode ---
    # Display chat history
    for msg in st.session_state.admin_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Display verification badge
            if msg.get("verification"):
                v = msg["verification"]
                css_class = f"confidence-{v['verdict'].lower()}"
                score_pct = int(v["coverage_score"] * 100)
                st.markdown(f'<p class="{css_class}">Confidence: {v["verdict"]} ({score_pct}% of claims found in notes)</p>',
                            unsafe_allow_html=True)

            # Display relevant image
            displayed_image = False
            if msg.get("sources") and os.path.exists(IMAGES_DIR):
                for s in msg["sources"]:
                    if displayed_image:
                        break
                    safe_filename = "".join([c if c.isalnum() else "_" for c in s["source"]])
                    page_img_prefix = f"{safe_filename}_page{s['page']}_img"
                    for img_file in os.listdir(IMAGES_DIR):
                        if img_file.startswith(page_img_prefix):
                            st.image(os.path.join(IMAGES_DIR, img_file),
                                     caption=f"Relevant Diagram from {s['source']} (Page/Slide {s['page']})",
                                     use_container_width=True)
                            displayed_image = True
                            break

            if msg.get("sources"):
                with st.expander("🔍 View Source Documents"):
                    for s in msg["sources"]:
                        st.markdown(f"**📄 {s['source']} (Page {s['page']})**")
                        if s["source"].lower().endswith(".pdf"):
                            pdf_path = os.path.join(UPLOAD_DIR, s["source"])
                            if os.path.exists(pdf_path):
                                page_num = int(s["page"])
                                try:
                                    annotations = get_pdf_annotations(pdf_path, page_num, s.get("text", ""))
                                    pdf_viewer(input=pdf_path, width=700, annotations=annotations,
                                               pages_to_render=[page_num])
                                except Exception as e:
                                    st.error(f"Error viewing PDF: {e}")
                                    st.info(s.get("text", "Text unavailable."))
                            else:
                                st.info(s.get("text", "Text unavailable."))
                        else:
                            st.info(s.get("text", "Text unavailable."))

    # Chat input
    if prompt := st.chat_input("Ask a question about your course notes...", key="admin_chat_input"):
        if stats["total_chunks"] == 0:
            st.warning("⚠️ Please upload some course notes first using the sidebar!")
        else:
            st.session_state.admin_chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Build chat history from memory or manual
            if memory_enabled() and st.session_state.get("admin_memory"):
                memory_vars = st.session_state.admin_memory.load_memory_variables({})
                formatted_history = memory_vars.get("history", "")
            else:
                formatted_history = ""
                if len(st.session_state.admin_chat_history) > 1:
                    for msg in st.session_state.admin_chat_history[:-1]:
                        role_name = "Student" if msg["role"] == "user" else "Assistant"
                        formatted_history += f"{role_name}: {msg['content']}\n"

            # Check for comparison intent
            available_sources = list(st.session_state.admin_files_processed)
            is_compare, src1, src2 = detect_comparison_intent(prompt, available_sources)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching notes & generating answer..."):
                    if is_compare and src1 and src2:
                        generator, sources, retrieved_docs = stream_comparative_answer(
                            prompt, src1, src2, ADMIN_COLLECTION, formatted_history)
                    else:
                        # Check for formula queries
                        if is_formula_query(prompt):
                            from src.vector_store import get_all_documents, FORMULA_COLLECTION
                            formula_docs = get_all_documents(FORMULA_COLLECTION)
                            # Prepend formula context (handled within the regular chain)

                        generator, sources, retrieved_docs = stream_rag_answer(
                            prompt, ADMIN_COLLECTION, formatted_history)

                full_response = st.write_stream(generator)

                # Citation badge rendering
                def render_citations(text):
                    return re.sub(r'\[(\d+)\]', r'<span class="citation-badge">[\1]</span>', text)

                # Verification
                verification = None
                if verification_enabled() and retrieved_docs:
                    context_str = format_docs(retrieved_docs)
                    verification = verify_answer_against_context(full_response, context_str)
                    if verification["verdict"] != "GROUNDED":
                        css_class = f"confidence-{verification['verdict'].lower()}"
                        score_pct = int(verification["coverage_score"] * 100)
                        st.markdown(
                            f'<p class="{css_class}">Confidence: {verification["verdict"]} '
                            f'({score_pct}% of claims found in notes)</p>',
                            unsafe_allow_html=True
                        )

                # Display relevant images
                displayed_image = False
                if sources and os.path.exists(IMAGES_DIR):
                    for s in sources:
                        if displayed_image:
                            break
                        safe_filename = "".join([c if c.isalnum() else "_" for c in s["source"]])
                        page_img_prefix = f"{safe_filename}_page{s['page']}_img"
                        for img_file in os.listdir(IMAGES_DIR):
                            if img_file.startswith(page_img_prefix):
                                st.image(os.path.join(IMAGES_DIR, img_file),
                                         caption=f"Relevant Diagram from {s['source']} (Page/Slide {s['page']})",
                                         use_container_width=True)
                                displayed_image = True
                                break

                if sources:
                    with st.expander("🔍 View Source Documents"):
                        for s in sources:
                            st.markdown(f"**📄 {s['source']} (Page {s['page']})**")
                            if s["source"].lower().endswith(".pdf"):
                                pdf_path = os.path.join(UPLOAD_DIR, s["source"])
                                if os.path.exists(pdf_path):
                                    page_num = int(s["page"])
                                    try:
                                        annotations = get_pdf_annotations(pdf_path, page_num, s.get("text", ""))
                                        pdf_viewer(input=pdf_path, width=700, annotations=annotations,
                                                   pages_to_render=[page_num])
                                    except Exception as e:
                                        st.error(f"Error viewing PDF: {e}")
                                        st.info(s.get("text", "Text unavailable."))
                                else:
                                    st.info(s.get("text", "Text unavailable."))
                            else:
                                st.info(s.get("text", "Text unavailable."))

            # Save to history
            st.session_state.admin_chat_history.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
                "verification": verification,
            })

            # Save to memory
            if memory_enabled() and st.session_state.get("admin_memory"):
                st.session_state.admin_memory.save_context(
                    {"input": prompt}, {"output": full_response}
                )
