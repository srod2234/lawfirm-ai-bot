import os
from dotenv import load_dotenv

import streamlit as st
import streamlit_authenticator as stauth
import fitz  # PyMuPDF
import openai

from sqlmodel import Session, select, delete
from models import init_db, engine, User, Document as DocModel, ChatMessage, Page
from ingest import ingest_pdf

from llama_index.core import VectorStoreIndex, Document as LIDoc
from llama_index.embeddings.openai import OpenAIEmbedding

from analytics import show_dashboard

# ───── Page Configuration ─────
st.set_page_config(page_title="Legal PDF Assistant", layout="wide")

# ───── Load ENV ─────
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY not set")
    st.stop()
openai.api_key = OPENAI_API_KEY

# ───── Init DB ─────
init_db()

# ───── Load Users for Auth ─────
with Session(engine) as db:
    users = db.exec(select(User)).all()
credentials = {"usernames": {}}
for u in users:
    credentials["usernames"][u.username] = {
        "name": u.username,
        "password": u.password_hash,
        "role": u.role,
    }

authenticator = stauth.Authenticate(
    credentials=credentials,
    cookie_name="law_firm_ai_session",
    key=os.getenv("COOKIE_KEY", ""),
    cookie_expiry_days=1,
    preauthorized=[],
)
authenticator.login("main")
status = st.session_state.get("authentication_status")
if status is False:
    st.error("❌ Username/password incorrect")
    st.stop()
if status is None:
    st.warning("ℹ️ Please enter your credentials")
    st.stop()

current_username = st.session_state["name"]
current_role = credentials["usernames"][current_username]["role"]

# ───── Clear‐state Logout Button ─────
st.sidebar.success(f"👋 Welcome, {current_username}!")
authenticator.logout("sidebar")
if st.sidebar.button("Logout"):
    # 1) Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # 2) Rerun or stop
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.stop()

# ───── Session State Defaults ─────
st.session_state.setdefault("docs", {})
st.session_state.setdefault("chat", {})
st.session_state.setdefault("last_doc", None)

# ───── Helper: Legacy Text Extract ─────
def extract_text_from_path(path: str) -> str:
    pdf = fitz.open(path)
    return "".join(page.get_text() for page in pdf)

# ───── 1) Load Docs (scoped by user if not admin) ─────
with Session(engine) as db:
    if current_role == "admin":
        all_docs = db.exec(select(DocModel)).all()
    else:
        all_docs = db.exec(
            select(DocModel)
            .join(User, DocModel.owner_id == User.id)
            .where(User.username == current_username)
        ).all()

for doc in all_docs:
    label = doc.label
    if label in st.session_state.docs:
        continue
    pages = db.exec(
        select(Page).where(Page.document_id == doc.id).order_by(Page.page_number)
    ).all()
    if not pages:
        text = extract_text_from_path(doc.file_path).strip()
        if text:
            pages = [Page(document_id=doc.id, page_number=1, text=text, is_scanned=False)]
    chunks = [LIDoc(text=pg.text) for pg in pages if pg.text.strip()]
    if not chunks:
        continue
    try:
        idx = VectorStoreIndex.from_documents(chunks, embed_model=OpenAIEmbedding())
    except ValueError:
        continue
    st.session_state.docs[label] = {"db_id": doc.id, "pages": pages, "index": idx}

# ───── 2) Load Chat History ─────
with Session(engine) as db:
    for label, payload in st.session_state.docs.items():
        q = select(ChatMessage).where(ChatMessage.doc_id == payload["db_id"])
        if current_role != "admin":
            q = q.join(User, ChatMessage.user_id == User.id).where(User.username == current_username)
        rows = db.exec(q).all()
        st.session_state.chat[label] = [(r.question, r.answer, []) for r in rows]

# ───── Sidebar: Upload / Manage ─────
st.sidebar.header("📁 Documents")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
label = st.sidebar.text_input("Label for this PDF")

if uploaded_file and label and st.sidebar.button("Save PDF"):
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{label}.pdf"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Parsing PDF & running OCR…"):
        owner = next(u.id for u in users if u.username == current_username)
        ingest_pdf(path, owner_id=owner)

    st.success(f"✅ Document '{label}' saved!")
    try:
        st.experimental_rerun()
    except AttributeError:
        st.stop()

# ───── Section Switcher ─────
options = ["Chat"] + (["Analytics"] if current_role == "admin" else [])
page = st.sidebar.radio("Go to", options)

if page == "Analytics":
    show_dashboard()
else:
    # ───── Sidebar: Manage Docs ─────
    for lbl in list(st.session_state.docs.keys()):
        with st.sidebar.expander(lbl):
            if st.button("👁 Preview", key=f"prev_{lbl}"):
                preview = "\n\n".join(pg.text for pg in st.session_state.docs[lbl]["pages"])
                st.sidebar.text_area("Preview", preview[:800] + "…", height=180)
            if st.button("♻️ Reset Chat", key=f"reset_{lbl}"):
                doc_id = st.session_state.docs[lbl]["db_id"]
                with Session(engine) as db:
                    db.exec(delete(ChatMessage).where(ChatMessage.doc_id == doc_id))
                    db.commit()
                st.session_state.chat[lbl] = []
                st.success("🔄 Chat cleared!")
            if st.button("🗑 Delete", key=f"del_{lbl}"):
                doc_id = st.session_state.docs[lbl]["db_id"]
                with Session(engine) as db:
                    db.exec(delete(Page).where(Page.document_id == doc_id))
                    db.exec(delete(ChatMessage).where(ChatMessage.doc_id == doc_id))
                    db.exec(delete(DocModel).where(DocModel.id == doc_id))
                    db.commit()
                try:
                    os.remove(st.session_state.docs[lbl]["pages"][0].document.file_path)
                except Exception:
                    pass
                del st.session_state.docs[lbl]
                del st.session_state.chat[lbl]
                st.success(f"🗑 Deleted '{lbl}'")

    # ───── Main: Chat UI ─────
    if st.session_state.docs:
        choice = st.selectbox("Choose a document to chat with:", list(st.session_state.docs.keys()))
        st.session_state.last_doc = choice
        payload = st.session_state.docs[choice]
        q_engine = payload["index"].as_query_engine(response_mode="compact", return_source=True)

        for q, a, src in st.session_state.chat[choice]:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
            with st.expander("Sources"):
                for node in src:
                    st.code(node.node.get_text().strip(), language="markdown")
            st.markdown("---")

        question = st.text_input("Ask a question:", key="chat_input")
        if question:
            with st.spinner("Thinking…"):
                res = q_engine.query(question)
            answer, sources = res.response, res.source_nodes
            with Session(engine) as db:
                row = ChatMessage(
                    doc_id=payload["db_id"],
                    user_id=next(u.id for u in users if u.username == current_username),
                    question=question,
                    answer=answer,
                )
                db.add(row)
                db.commit()
            st.session_state.chat[choice].append((question, answer, sources))
            st.markdown(f"**You:** {question}")
            st.markdown(f"**Bot:** {answer}")
            with st.expander("Sources"):
                for node in sources:
                    st.code(node.node.get_text().strip(), language="markdown")
            st.markdown("---")
    else:
        st.info("Upload a PDF to begin.")
