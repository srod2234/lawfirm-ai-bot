import os
from dotenv import load_dotenv

import streamlit as st
import streamlit_authenticator as stauth
import fitz  # PyMuPDF
import openai

from sqlmodel import Session
from models import init_db, engine, Document as DocModel

from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.openai import OpenAIEmbedding

# ───── Page Configuration ─────
st.set_page_config(page_title="Legal PDF Assistant", layout="wide")

# ───── Load & Clean ENV VARS ─────
load_dotenv(override=True)
AUTH_USERNAME      = os.getenv("AUTH_USERNAME", "").strip()
AUTH_PASSWORD_HASH = os.getenv("AUTH_PASSWORD_HASH", "").strip()
COOKIE_KEY         = os.getenv("COOKIE_KEY", "").strip()
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "").strip()

# ───── Initialize Database ─────
init_db()

if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY not set in .env or Railway")
    st.stop()
openai.api_key = OPENAI_API_KEY

# ───── streamlit-authenticator Setup ─────
credentials = {
    "usernames": {
        AUTH_USERNAME: {"name": AUTH_USERNAME, "password": AUTH_PASSWORD_HASH}
    }
}
authenticator = stauth.Authenticate(
    credentials=credentials,
    cookie_name="law_firm_ai_session",
    key=COOKIE_KEY,
    cookie_expiry_days=1,
    preauthorized=[]
)

# ───── Login Flow ─────
authenticator.login(location="main")
auth_status = st.session_state.get("authentication_status")

if auth_status:
    st.sidebar.success(f"👋 Welcome, {st.session_state['name']}!")
    authenticator.logout(location="sidebar")
elif auth_status is False:
    st.error("❌ Username/password is incorrect")
    st.stop()
else:
    st.warning("ℹ️ Please enter your credentials")
    st.stop()

# ───── Session State Initialization ─────
for key, default in {"docs": {}, "chat": {}, "last_doc": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ───── Sidebar: Document Management ─────
st.sidebar.header("📁 Documents")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
label = st.sidebar.text_input("Label for this PDF")

def extract_text(stream) -> str:
    pdf = fitz.open(stream=stream.read(), filetype="pdf")
    return "".join(page.get_text() for page in pdf)

if uploaded_file and st.sidebar.button("Save PDF"):
    with st.spinner("Saving & indexing…"):
        # 1) Persist PDF to disk
        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{label}.pdf"
        with open(path, "wb") as f:
            f.write(uploaded_file.read())

        # 2) Insert Document row
        with Session(engine) as db:
            doc = DocModel(owner_id=1, label=label, file_path=path)
            db.add(doc)
            db.commit()
            db.refresh(doc)

        # 3) Build and cache your index using the new API signature
        text = extract_text(open(path, "rb"))
        idx = VectorStoreIndex.from_documents(
            [Document(text=text)],
            embed_model=OpenAIEmbedding()
        )

        # Store the index in session for now
        st.session_state.docs[label] = {"db_id": doc.id, "text": text, "index": idx}
        st.session_state.chat[label] = []

        st.success(f"📥 Saved '{label}' (db id={doc.id})")

# ───── Manage Saved Docs ─────
for lbl in list(st.session_state.docs.keys()):
    with st.sidebar.expander(lbl):
        if st.button("👁 Preview", key=f"prev_{lbl}"):
            st.sidebar.text_area(
                "Preview",
                st.session_state.docs[lbl]["text"][:800] + "…",
                height=180
            )
        if st.button("♻️ Reset Chat", key=f"reset_{lbl}"):
            st.session_state.chat[lbl] = []
            st.session_state.last_doc = lbl
            st.stop()
        if st.button("🗑 Delete", key=f"del_{lbl}"):
            del st.session_state.docs[lbl]
            st.session_state.chat.pop(lbl, None)
            st.session_state.last_doc = None
            st.stop()

# ───── Main: Chat Interface ─────
if st.session_state.docs:
    selected = st.selectbox(
        "Choose a document to chat with:",
        list(st.session_state.docs.keys())
    )
    st.session_state.last_doc = selected
    doc_data = st.session_state.docs[selected]
    q_engine = doc_data["index"].as_query_engine(
        response_mode="compact",
        return_source=True
    )

    # Render chat history
    for q, a, src in st.session_state.chat[selected]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        with st.expander("Sources"):
            for node in src:
                st.code(node.node.get_text().strip(), language="markdown")
        st.markdown("---")

    # New question
    question = st.text_input("Ask a question:", key="chat_input")
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
