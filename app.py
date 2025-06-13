import streamlit as st
from llama_index.core import VectorStoreIndex, Document, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
import fitz  # PyMuPDF
import openai
import os

# 🔐 Load your OpenAI key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- STREAMLIT SETUP ---
st.set_page_config(page_title="Legal PDF Assistant", layout="wide")
st.title("📄🧠 Legal Document Assistant")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# --- EXTRACT PDF TEXT ---
def extract_text_from_uploaded_pdf(uploaded_file):
    if uploaded_file is not None:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "".join(page.get_text() for page in doc)
    return None

# --- MAIN LOGIC ---
if uploaded_file:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.spinner("Processing your PDF..."):
        pdf_text = extract_text_from_uploaded_pdf(uploaded_file)
        documents = [Document(text=pdf_text)]

        embed_model = OpenAIEmbedding()
        service_context = ServiceContext.from_defaults(embed_model=embed_model)

        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        query_engine = index.as_query_engine(response_mode="compact", return_source=True)

        st.success("✅ PDF processed. Ask questions below:")

        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
            st.markdown("---")

        question = st.text_input("Ask a new question about this PDF:")

        if question:
            response = query_engine.query(question)
            answer = response.response
            sources = response.source_nodes

            st.markdown(f"**You:** {question}")
            st.markdown(f"**Bot:** {answer}")
            st.markdown("**Source:**")
            for i, node in enumerate(sources):
                st.code(node.node.get_text().strip(), language="markdown")
            st.markdown("---")

            st.session_state.chat_history.append((question, answer))
else:
    st.info("👈 Upload a PDF file to get started.")
