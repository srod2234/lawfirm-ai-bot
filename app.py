# app.py – Secure multi-doc legal assistant (June 2025 compatible)
import os, fitz, streamlit as st
from dotenv import load_dotenv
import streamlit_authenticator as stauth
from llama_index.core import VectorStoreIndex, Document, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding

# ─────────────────────  ENV & AUTH  ─────────────────────
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
AUTH_USER = os.getenv("AUTH_USERNAME", "demo")
AUTH_HASH = os.getenv("AUTH_PASSWORD_HASH", "")

import openai
openai.api_key = OPENAI_KEY

# --- Authenticator setup ---
credentials = {
    "usernames": {
        AUTH_USER: {
            "name": "Legal-User",
            "password": AUTH_HASH,
        }
    }
}
authenticator = stauth.Authenticate(
    credentials,
    "lawbot_cookie", "abcdef", cookie_expiry_days=1
)

# 🚩 LOGIN (new API: returns only auth_status)
auth_status = authenticator.login(location="main")

if auth_status is False:
    st.error("Invalid credentials")
    st.stop()
elif auth_status is None:
    st.warning("Please enter username & password")
    st.stop()

# --- Get username for greeting (may be None)
user = getattr(authenticator, "username", None) or getattr(authenticator, "name", None)

# ─────────────────────  STREAMLIT UI  ─────────────────────
st.set_page_config(page_title="Legal PDF Assistant", layout="wide")
st.sidebar.title(f"👋 Hi {user or 'user'}")

# --- session state init
for key, default in {
    "docs": {},
    "chat": {},
    "last_doc": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Sidebar: upload & doc management
st.sidebar.header("📁 Documents")
uploaded = st.sidebar.file_uploader("Upload PDF", type="pdf")
label = st.sidebar.text_input("Label for this PDF")

def extract_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join(p.get_text() for p in doc)

if uploaded and st.sidebar.button("Save PDF"):
    with st.spinner("Indexing…"):
        text = extract_text(uploaded)
        docs = [Document(text=text)]
        idx = VectorStoreIndex.from_documents(
            docs,
            service_context=ServiceContext.from_defaults(embed_model=OpenAIEmbedding())
        )
        st.session_state.docs[label] = {"text": text, "index": idx}
        st.session_state.chat[label] = []
        st.success(f"Saved '{label}'")

# --- List & manage docs
for lbl in list(st.session_state.docs.keys()):
    with st.sidebar.expander(lbl):
        if st.button("👁 Preview", key=f"prev_{lbl}"):
            st.sidebar.text_area("Preview", st.session_state.docs[lbl]["text"][:800]+"…", height=180)
        if st.button("♻️ Reset Chat", key=f"reset_{lbl}"):
            st.session_state.chat[lbl] = []
            st.session_state.last_doc = lbl
            st.rerun()
        if st.button("🗑 Delete", key=f"del_{lbl}"):
            del st.session_state.docs[lbl]
            st.session_state.chat.pop(lbl, None)
            st.session_state.last_doc = None
            st.rerun()

# --- Select & chat with doc
if st.session_state.docs:
    selected = st.selectbox("Choose a document to chat with:", st.session_state.docs.keys())
    st.session_state.last_doc = selected
    doc_data = st.session_state.docs[selected]
    q_engine = doc_data["index"].as_query_engine(response_mode="compact", return_source=True)

    # Show chat history
    for q, a, src in st.session_state.chat[selected]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        with st.expander("Sources"):
            for node in src:
                st.code(node.node.get_text().strip(), language="markdown")
        st.markdown("---")

    # New question
    question = st.text_input("Ask a question:")
    if question:
        with st.spinner("Thinking…"):
            res = q_engine.query(question)
        answer, sources = res.response, res.source_nodes
        st.markdown(f"**You:** {question}")
        st.markdown(f"**Bot:** {answer}")
        with st.expander("Sources"):
            for node in sources:
                st.code(node.node.get_text().strip(), language="markdown")
        st.markdown("---")
        st.session_state.chat[selected].append((question, answer, sources))
else:
    st.info("Upload a PDF to begin.")
