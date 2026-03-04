"""
NoteGo RAG — Course Notes Q&A System
Auth router: Login page + st.navigation to Admin/Student pages.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# --- Page Config (MUST be first Streamlit call) ---
st.set_page_config(
    page_title="NoteGo RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Ollama Health Check ---
try:
    import ollama
    ollama.list()
except Exception:
    st.error("⚠️ **Ollama is not running.** Please start it with `ollama serve` in your terminal.")
    st.stop()

# --- Auth helpers ---
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
STUDENT_PASSWORD = os.getenv("STUDENT_PASSWORD", "student123")


def logout():
    """Clear auth state and rerun."""
    for key in ["authenticated", "role"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def login_page():
    """Render the login page."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .login-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0;
        }
        .login-sub {
            text-align: center;
            color: #888;
            font-size: 1.1rem;
            margin-top: -5px;
            margin-bottom: 40px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="login-header">📚 NoteGo RAG</h1>', unsafe_allow_html=True)
    st.markdown('<p class="login-sub">Course Notes Q&A System</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        role = st.selectbox("Select your role", ["Admin", "Student"], key="login_role")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", use_container_width=True, type="primary"):
            if role == "Admin" and password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.session_state.role = "admin"
                st.rerun()
            elif role == "Student" and password == STUDENT_PASSWORD:
                st.session_state.authenticated = True
                st.session_state.role = "student"
                st.rerun()
            else:
                st.error("Invalid password. Please try again.")


# --- Main routing logic ---
if not st.session_state.get("authenticated"):
    login_page()
else:
    role = st.session_state.get("role", "student")

    logout_page = st.Page(logout, title="Logout", icon="🚪")

    if role == "admin":
        admin_page = st.Page("pages/admin.py", title="Admin Panel", icon="🔧", default=True)
        nav = st.navigation([admin_page, logout_page])
    else:
        student_page = st.Page("pages/student.py", title="Student Portal", icon="🎓", default=True)
        nav = st.navigation([student_page, logout_page])

    nav.run()
