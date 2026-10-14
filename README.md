# Uzhavan AI — Tamil Nadu Farmer RAG

Two separate Streamlit apps:
- `rag_app.py`: upload multiple PDFs, chunk and embed them, save a shared FAISS store.
- `farmer_chatbot.py`: green agriculture-themed Tamil/English chatbot with source filename/page citations.
-  'download_goscheme.py': webscrapping the schemes and download them as pdf from URL's.

## 1. Setup (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenAI API key:

```text
OPENAI_API_KEY=your-openai-api-key
```

Do not commit or share `.env`. ChatGPT subscriptions do not automatically include API credits.

## 2. Build the vector database

```powershell
streamlit run rag_app.py
```

Upload multiple text-readable scheme PDFs and click **Build / Rebuild Vector Database**.
The app saves the combined index in `scheme_vectorstore/`.

## 3. Run the separate chatbot

Open a second PowerShell terminal in this folder, activate the same environment, then run:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run farmer_chatbot.py --server.port 8502
```

URLs are usually:
- Indexer: http://localhost:8501
- Chatbot: http://localhost:8502

## Notes
- Both apps must run from the same project folder so they can see `scheme_vectorstore/`.
- Page citations are converted from PyPDFLoader's zero-based page metadata to human page numbers.
- Scanned PDFs may need OCR before indexing.
- The chatbot is grounded in indexed PDFs, not live government updates. Verify current scheme rules with the relevant department.
- The FAISS loader enables pickle deserialization; only load the local index you created and trust.
- For best source accuracy, keep official filenames unique and index current official PDFs.
