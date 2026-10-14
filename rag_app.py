import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="Uzhavan AI | PDF Indexer", page_icon="🌾", layout="wide")

VECTOR_DIR = Path("scheme_vectorstore")
EMBEDDING_MODEL = "text-embedding-3-small"

st.markdown("""
<style>
.stApp { background:#F5F8F3; color:#20352A; }
.block-container { max-width:1150px; padding-top:2rem; }
section[data-testid="stSidebar"] { background:#EAF3E8; }
.hero { padding:28px 30px; border-radius:22px; color:white;
background:linear-gradient(120deg,#10452F,#176B45,#3C8C58); margin-bottom:22px; }
.hero h1 { color:white; margin:0 0 8px 0; }
.hero p { color:#E3F3E6; margin:0; }
div[data-testid="stMetric"] { background:white; border:1px solid #DCE9DA;
border-radius:14px; padding:14px; }
.stButton>button { border-radius:10px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div style="font-size:13px;letter-spacing:1px;">🌿 UZHAVAN AI · APP 1</div>
  <h1>Scheme PDF Indexer</h1>
  <p>Upload official scheme documents and build one searchable knowledge base for the Farmer Assistant.</p>
</div>
""", unsafe_allow_html=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is missing. Add it to .env or set it in PowerShell.")
    st.stop()

with st.sidebar:
    st.markdown("## 🌾 Uzhavan AI")
    st.caption("Tamil Nadu Farmer Assistance")
    st.markdown("### Indexing workflow")
    st.markdown("1. Upload PDFs\n2. Choose chunk settings\n3. Build the FAISS store\n4. Open the chatbot")
    st.info("Use official, text-readable PDFs where possible.")

uploaded_files = st.file_uploader(
    "Upload one or more government scheme PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

c1, c2 = st.columns(2)
with c1:
    chunk_size = st.slider("Chunk size (characters)", 400, 1800, 1000, 100)
with c2:
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, 50)

if chunk_overlap >= chunk_size:
    st.error("Chunk overlap must be smaller than chunk size.")
    st.stop()

if st.button("🌱 Build / Rebuild Vector Database", type="primary", use_container_width=True):
    if not uploaded_files:
        st.warning("Upload at least one PDF first.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    errors = []
    progress = st.progress(0)
    status = st.empty()

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, uploaded in enumerate(uploaded_files):
            safe_name = Path(uploaded.name).name
            status.write(f"Reading {safe_name} ({i + 1}/{len(uploaded_files)})")
            try:
                pdf_path = Path(temp_dir) / f"{i}_{safe_name}"
                pdf_path.write_bytes(uploaded.getvalue())
                pages = PyPDFLoader(str(pdf_path)).load()

                for page in pages:
                    page.metadata["source"] = safe_name

                file_chunks = splitter.split_documents(pages)
                if not file_chunks:
                    errors.append(f"{safe_name}: no selectable text found (may need OCR).")
                else:
                    chunks.extend(file_chunks)
            except Exception as exc:
                errors.append(f"{safe_name}: {exc}")
            progress.progress((i + 1) / len(uploaded_files))

    for error in errors:
        st.warning(error)

    if not chunks:
        st.error("No text chunks were created. Check the PDFs or OCR scanned documents first.")
        st.stop()

    try:
        status.write("Creating embeddings and building FAISS index…")
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
        store = FAISS.from_documents(chunks, embeddings)

        new_dir = Path("scheme_vectorstore_new")
        backup_dir = Path("scheme_vectorstore_backup")
        shutil.rmtree(new_dir, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)
        store.save_local(str(new_dir))

        if VECTOR_DIR.exists():
            VECTOR_DIR.rename(backup_dir)
        try:
            new_dir.rename(VECTOR_DIR)
        except Exception:
            if backup_dir.exists() and not VECTOR_DIR.exists():
                backup_dir.rename(VECTOR_DIR)
            raise
        shutil.rmtree(backup_dir, ignore_errors=True)

        status.empty()
        progress.empty()
        st.success("Your shared FAISS vector database is ready.")
        m1, m2, m3 = st.columns(3)
        m1.metric("PDFs processed", len({Path(d.metadata["source"]).name for d in chunks}))
        m2.metric("Chunks indexed", len(chunks))
        m3.metric("Embedding model", EMBEDDING_MODEL)
        st.markdown("### Indexed files")
        for name in sorted({d.metadata["source"] for d in chunks}):
            st.write(f"📄 {name}")
        st.info(f"Saved to `{VECTOR_DIR.resolve()}`. Now start App 2.")
    except Exception as exc:
        st.error(f"Indexing failed: {exc}")

if VECTOR_DIR.exists():
    st.divider()
    st.success("A vector database already exists on disk.")
    st.caption("Rebuild it only when you add or update source PDFs.")
