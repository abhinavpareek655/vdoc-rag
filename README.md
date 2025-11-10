# 📄 VDoc-RAG (Visually-Rich Document Retrieval-Augmented Generation)

VDoc-RAG is an advanced multimodal system that answers questions from visually-rich documents (PDFs, reports, flyers) by combining OCR, table and chart reasoning, semantic embeddings, and LLMs.

---

## 🚀 Features

- 🧠 **RAG Pipeline** with persistent ChromaDB  
- 🪄 **OCR + Table + Chart understanding**  
- 📊 **Chart Reasoning** (Pix2Struct + OCR-based)  
- 🔐 **Environment-based API key handling**  
- 🧮 **Confidence Scoring** via cosine similarity  
- 🧾 **Feedback Loop** for self-improving embeddings  
- 📈 **Benchmark Dashboard** for evaluating embedding models  
- 💾 **Persistent Storage** (DuckDB + Parquet backend)

---

## ⚙️ Quickstart (Windows)

### 1️⃣ Install Dependencies

Install:
- **Tesseract OCR** → [Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki)
- **Poppler for Windows** → [Poppler Releases](https://github.com/oschwartz10612/poppler-windows/releases)

Add both to your system PATH.

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Run the App
```bash
uvicorn app.main:app --reload --port 8000
```

Open → [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🖥️ Web Interfaces

| Page | Route | Description |
|------|-------|--------------|
| `/` | Main Interface | Upload, query, visualize highlights |
| `/feedback_dashboard` | Feedback Loop | View stats, fine-tune model |
| `/benchmark_dashboard` | Benchmarking | Evaluate embeddings (Precision/Recall/MRR) |

---

## 📁 Project Structure

```
vdoc-rag-mvp/
├─ app/
│  ├─ ingest.py              # OCR, table & chart extraction
│  ├─ chart_reasoner.py      # Chart summarization and trend detection
│  ├─ indexer.py             # Persistent ChromaDB retrieval
│  ├─ reader.py              # LLM question answering
│  ├─ feedback_manager.py    # Feedback collection system
│  ├─ main.py                # FastAPI server + dashboards
│  └─ visual_highlight.py    # Highlight relevant regions
│
├─ models/vdoc_feedback_tuned/  # Fine-tuned embedding model
├─ storage/chroma_db/           # Persistent vector store
├─ notebooks/evaluate_embeddings.ipynb  # Benchmarking notebook
└─ templates/                   # HTML UIs (main, feedback, benchmark)
```

---

## 🧠 Models Used

| Type | Model | Purpose |
|------|--------|----------|
| Embedding | `all-MiniLM-L6-v2` (base), `multi-qa-MiniLM`, feedback-tuned variant | Semantic encoding |
| LLM Reader | Gemini / DistilGPT2 | Context-based answering |
| Chart Reasoning | Pix2Struct / OCR fallback | Visual trend analysis |
| Vector Store | ChromaDB (DuckDB + Parquet) | Persistent retrieval |
| Fine-tuning | SentenceTransformer + CosineLoss | Feedback-based learning |

---

## 🧩 Evaluation

- **Confidence Scoring**: cosine similarity between query & chunks  
- **Precision / Recall / MRR**: benchmark dashboards & notebook  
- **Feedback-driven fine-tuning**: iterative model improvement  

---

## 🧠 Author’s Note

VDoc-RAG demonstrates how retrieval-augmented generation can evolve from plain text retrieval into **visually grounded document reasoning**, enabling future systems that can read, reason, and learn continuously.

---

**Developed as a full multimodal RAG research framework** — suitable for academic reports, enterprise document intelligence, and AI reasoning pipelines.
