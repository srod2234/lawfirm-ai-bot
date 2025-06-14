import os
from dotenv import load_dotenv

import streamlit as st
import streamlit_authenticator as stauth
import fitz  # PyMuPDF
import openai

from llama_index.core import VectorStoreIndex, Document, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding

# ─────── Page Config ───────
st.set_page_config(page_title="Legal PDF Assistant", layout="wide")

# ─────── Load & Clean ENV ───────
load_dotenv(override=True)
AUTH_USERNAME      = os.getenv("AUTH_USERNAME", "demo").strip().strip("'\"")
AUTH_PASSWORD_HASH = os.getenv("AUTH_PASSWORD_HASH", "").strip().strip("'\"")
COOKIE_KEY         = os.getenv("COOKIE_KEY", "").strip().strip("'\"")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "").strip()

# ─────── Debug ENV ───────
st.write("🔍 ENV VARS", {
    "AUTH_USERNAME":      repr(AUTH_USERNAME),
    "AUTH_PASSWORD_HASH": repr(AUTH_PASSWORD_HASH),
    "COOKIE_KEY":         repr(COOKIE_KEY),
    "OPENAI_API_KEY_SET": bool(OPENAI_API_KEY)
})

if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY not set.")
    st.stop()

openai.api_key = OPENAI_API_KEY

# ─────── Build Authenticator ───────
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

# ─────── Login Flow ───────
auth_status = authenticator.login("main")

# ─────── Debug Login ───────
st.write("🔍 LOGIN STATE", {"auth_status": auth_status})

if auth_status:
    st.sidebar.success(f"Welcome, {AUTH_USERNAME}!")
elif auth_status is False:
    st.error("❌ Username/password is incorrect")
    st.stop()
else:
    # auth_status is None
    st.warning("ℹ️ Login returned None — form did rerun but we didn’t match credentials.")
    st.stop()

# ─────── The rest of your app follows unchanged ───────
# (document upload, RAG setup, chat UI, etc.)
