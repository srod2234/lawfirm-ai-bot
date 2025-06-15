import os
from dotenv import load_dotenv

import streamlit as st
import streamlit_authenticator as stauth
import fitz  # PyMuPDF
import openai

from sqlmodel import Session
from models import init_db, engine, Document as DocModel, ChatMessage

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

# ───── Auth Setup ─────
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
authenticator.login(location="main")
if st.session_state.get("authentication_status") is False:
    st.error("❌ Username/password is incorrect"); st.stop()
if st.session_state.get("authentication_status") is None:
    st.warning("ℹ️ Please enter your credentials"); st.stop()
st.sidebar.success(f"👋 Welcome, {st.session_state['name']}!")
authenticator.logout(location="sidebar")

# ───── Session State Defaults ─────
st.session_state.setdefault("docs", {})
st.session_state.setdefault("chat", {})
st.session_state.setdefault("last_doc", None)

# ───── Helper to extract text from a PDF file path ─────
def extract_text_from_path(path: str) -> str:
    pdf = fitz.open(path)
    return "".join(page.get_text() for page in pdf)

# ───── 1) Load all documents from the DB on startup ─────
with Session(engine) as db:
    all_docs = db.query(DocModel).all()
for doc in all_docs:
    label = doc.label
    if label not in st.session_state.docs:
        text = extract_text_from_path(doc.file_path)
        idx = VectorStoreIndex.from_documents(
            [Document(text=text)],
            embed_model=OpenAIEmbedding()
        )
        st.session_state.docs[label] = {
            "db_id": doc.id,
            "text": text,
            "index": idx,
        }

# ───── 2) Load chat history for each doc ─────
with Session(engine) as db:
    for label, payload in st.session_state.docs.items():
        rows = db.query(ChatMessage).filter(ChatMessage.doc_id == payload["db_id"]).all()
        st.session_state.chat[label] = [(r.question, r.answer, []) for r in rows]

# ───── Sidebar: Upload & Manage ─────
st.sidebar.header("📁 Documents")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
label = st.sidebar.text_input("Label for this PDF")

if uploaded_file and label and st.sidebar.button("Save PDF"):
    with st.spinner("Saving & indexing…"):
        # Persist PDF
        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{label}.pdf"
        with open(path, "wb") as f:
            f.write(uploaded_file.read())

        # Insert into DB
        with Session(engine) as db:
            doc = DocModel(owner_id=1, label=label, file_path=path)
            db.add(doc); db.commit(); db.refresh(doc)

        # Index
        text = extract_text_from_path(path)
        idx = VectorStoreIndex.from_documents(
            [Document(text=text)],
            embed_model=OpenAIEmbedding()
        )

        st.session_state.docs[label] = {"db_id": doc.id, "text": text, "index": idx}
        st.session_state.chat[label] = []

        st.success(f"📥 Saved '{label}' (db id={doc.id})")

# ───── Sidebar: Manage Saved Docs ─────
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
            with Session(engine) as db:
                db.delete(db.query(DocModel).filter(DocModel.id == st.session_state.docs[lbl]["db_id"]).first())
                db.commit()
            try:
                os.remove(st.session_state.docs[lbl]["file_path"])
            except OSError:
                pass
            del st.session_state.docs[lbl]
            del st.session_state.chat[lbl]
            st.session_state.last_doc = None
            st.stop()

# ───── Main: Chat UI ─────
if st.session_state.docs:
    selected = st.selectbox("Choose a document to chat with:", list(st.session_state.docs.keys()))
    st.session_state.last_doc = selected
    payload = st.session_state.docs[selected]
    q_engine = payload["index"].as_query_engine(response_mode="compact", return_source=True)

    # Render history
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

        # Persist
        with Session(engine) as db:
            row = ChatMessage(doc_id=payload["db_id"], question=question, answer=answer)
            db.add(row); db.commit()

        # Update UI
        st.session_state.chat[selected].append((question, answer, sources))
        st.markdown(f"**You:** {question}")
        st.markdown(f"**Bot:** {answer}")
        with st.expander("Sources"):
            for node in sources:
                st.code(node.node.get_text().strip(), language="markdown")
        st.markdown("---")

else:
    st.info("Upload a PDF to begin.")
