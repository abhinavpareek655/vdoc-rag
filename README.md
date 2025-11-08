# VDoc RAG - MVP (Visually-Rich Document RAG)

This is a minimal, runnable MVP for a **VDoc RAG system** that supports:
- PDF / image upload  
- OCR-based text chunk extraction with bounding boxes  
- Table extraction (`pdfplumber`) and CSV saving  
- Sentence-Transformers embeddings (text)  
- Local **Chroma** vector index (DuckDB + Parquet backend)  
- **LLM Reader** (OpenAI or local HuggingFace) that returns answers with provenance  

---

## 🚀 Quickstart (Windows)

### 1️⃣ Install System Dependencies

Make sure the following tools are installed and added to your PATH:

- **Tesseract OCR**  
  Download and install from:  
  👉 https://github.com/UB-Mannheim/tesseract/wiki  
  Example installation path:  
  `C:\Program Files\Tesseract-OCR`  
  Add this folder to your **PATH** environment variable.  

- **Poppler for Windows** (required for `pdf2image`)  
  Download from:  
  👉 https://github.com/oschwartz10612/poppler-windows/releases/  
  After extracting, add the `/bin` folder to your **PATH**.  
  Example: `C:\poppler-24.02.0\Library\bin`

---

### 2️⃣ Create Virtual Environment and Install Dependencies

Open **PowerShell** or **Command Prompt** inside the project folder:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### 3️⃣ (Optional) Set OpenAI API Key

If you plan to use OpenAI models for answering:

```powershell
setx OPENAI_API_KEY "sk-your-key-here"
```

If you prefer to use a local HuggingFace model (e.g., `distilgpt2`), skip this step and set an environment variable:
```powershell
setx VDOCRAG_READER_PROVIDER "local"
```

---

### 4️⃣ Run the App

Start the FastAPI server with:
```powershell
uvicorn app.main:app --reload --port 8000
```

The API will start at:  
👉 http://127.0.0.1:8000  

Endpoints:
- `POST /upload` → Upload PDFs or images
- `GET /query?q=your+question` → Ask questions about uploaded documents

---

### 5️⃣ Example Usage

1. Upload a document:
   - Use tools like **Postman** or **curl**:
     ```bash
     curl -X POST "http://127.0.0.1:8000/upload" -F "file=@samples/sample_flyer.pdf"
     ```
2. Query it:
   ```bash
   curl "http://127.0.0.1:8000/query?q=When+is+the+event?"
   ```

---

### 📁 Project Structure

```
vdoc-rag-mvp/
│
├─ app/
│  ├─ main.py          # FastAPI endpoints (upload/query)
│  ├─ ingest.py        # PDF -> OCR -> chunking + table extraction
│  ├─ tables.py        # Table extraction & CSV saving
│  ├─ embeddings.py    # Sentence-Transformers wrapper
│  ├─ indexer.py       # Chroma index wrapper
│  ├─ reader.py        # LLM Reader (OpenAI/local)
│  └─ utils.py         # Helper utilities
│
├─ samples/
│  └─ sample_flyer.pdf
│
├─ requirements.txt
└─ README.md
```

---

### 🧠 Tips for Windows Users

- If `pdf2image` throws a **“poppler not found”** error, ensure the path to `poppler/bin` is correctly added to your system PATH and restart your terminal.  
- If `tesseract` fails, verify its path by running:
  ```powershell
  tesseract --version
  ```
- You can change temporary directories by editing:
  ```python
  TABLES_DIR = os.environ.get('VDOCRAG_TABLES_DIR', 'C:\\Temp\\vdoc_tables')
  ```

---
