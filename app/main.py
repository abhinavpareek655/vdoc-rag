import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.ingest import process_pdf
from app.indexer import ChromaIndexer
from app.embeddings import TextImageEmbedder
from app.reader import LLMReader
from app.visual_highlight import render_highlighted_pages
from app.cache_manager import clear_cache
from app.feedback_manager import record_feedback, get_feedback_summary, _load_feedback
import shutil
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------
app = FastAPI(title="VDoc RAG - Web UI")

# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------
# Get absolute path to this file’s directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define template and static directories relative to BASE_DIR
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure directories exist
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Serve highlighted images
HIGHLIGHTED_DIR = os.path.join(BASE_DIR, "highlighted")
os.makedirs(HIGHLIGHTED_DIR, exist_ok=True)
app.mount("/highlighted", StaticFiles(directory=HIGHLIGHTED_DIR), name="highlighted")

# Load Jinja2 templates safely
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# ---------------------------------------------------------
# Core Components
# ---------------------------------------------------------
embedder = TextImageEmbedder()
# Use a project-local persistent directory for Chroma
STORAGE_DIR = os.path.join(BASE_DIR, "storage", "chroma_db")
indexer = ChromaIndexer(embedding_function=embedder.embed_text, persist_directory=STORAGE_DIR)
reader_provider = os.environ.get("VDOCRAG_READER_PROVIDER", "gemini")
reader = LLMReader(provider=reader_provider)

uploaded_files = []  # track uploaded docs for display

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render main upload + query interface."""
    print(f"✅ Using templates from: {TEMPLATE_DIR}")
    if not os.path.exists(os.path.join(TEMPLATE_DIR, "index.html")):
        print("❌ index.html not found in:", TEMPLATE_DIR)
    else:
        print("✅ index.html found!")

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "uploaded": uploaded_files, "answer": None},
    )


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Handle PDF/image upload and indexing."""
    if not file.filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Save uploaded file temporarily
    temp_dir = os.path.join(BASE_DIR, "uploads")
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, file.filename)

    with open(path, "wb") as f:
        f.write(await file.read())

    # Extract and process text chunks
    docs = process_pdf(path)
    if len(docs) == 0:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "uploaded": uploaded_files,
                "answer": "⚠️ No content extracted from file.",
            },
        )

    # Embed and index chunks
    texts = [d["text"] for d in docs]
    vectors = embedder.embed_text(texts)
    items = [(d["id"], vectors[i].tolist(), d["metadata"], d["text"]) for i, d in enumerate(docs)]
    indexer.upsert(items)

    uploaded_files.append(file.filename)
    print(f"✅ Indexed {len(docs)} chunks from {file.filename}")

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "uploaded": uploaded_files,
            "answer": f"✅ Uploaded and indexed {file.filename} ({len(docs)} chunks).",
        },
    )


@app.post("/ask")
async def ask_question(request: Request, question: str = Form(...)):
    """Handle user query, retrieve relevant chunks, and generate LLM answer."""
    # Step 1 — Embed question
    qvec = embedder.embed_text([question])[0]

    # Step 2 — Retrieve top chunks
    hits = indexer.query(qvec, top_k=10)

    # Debug log
    print("\n🔍 Retrieved Chunks for Query:", question)
    for i, h in enumerate(hits):
        meta = h.get("metadata", {})
        conf = h.get("score", 0)
        print(f"Chunk {i+1}: Page {meta.get('page')} | BBox: {meta.get('bbox')} | Confidence: {conf*100:.1f}%")
        print(f"Text: {h['text'][:500]}...\n")

    # Prioritize chart-type hits for chart-related questions
    chart_keywords = ["chart", "graph", "trend", "plot", "increase", "decrease", "growth"]
    if any(k in question.lower() for k in chart_keywords):
        try:
            hits = sorted(hits, key=lambda h: h.get("metadata", {}).get("type") != "chart")
            print("[INFO] Prioritized chart-type chunks for chart-related question.")
        except Exception as e:
            print("[WARN] Failed to prioritize chart hits:", e)

    # Step 3 — Build context string
    context_blocks = [
        f"[{i+1}] {h['text']} (page: {h['metadata'].get('page')}, bbox: {h['metadata'].get('bbox')})"
        for i, h in enumerate(hits)
    ]
    context = "\n".join(context_blocks)

    # Step 4 — Ask LLM
    answer = reader.answer_question(query=question, context=context, sources=hits)
    sources = answer.get("sources", [])

    # 🖼️ Generate visual highlights
    try:
        first_source_path = hits[0]["metadata"].get("source") if hits else None
        highlight_paths = []
        if first_source_path and os.path.exists(first_source_path):
            highlight_paths = render_highlighted_pages(first_source_path, hits)
            # convert to web URLs for template
            highlight_urls = ["/" + os.path.relpath(p, BASE_DIR).replace("\\", "/") for p in highlight_paths]
        else:
            highlight_urls = []
    except Exception as e:
        print("[WARN] Highlight rendering failed:", e)
        highlight_urls = []

    # Step 5 — Prepare chunk previews for UI
    chunk_previews = [
        {
            "index": i + 1,
            "page": h["metadata"].get("page"),
            "bbox": h["metadata"].get("bbox"),
            "text": h["text"][:300] + ("..." if len(h["text"]) > 300 else ""),
            "confidence": round(h.get("score", 0) * 100, 1),
        }
        for i, h in enumerate(hits)
    ]

    # Average confidence for the retrieved set
    avg_conf = sum(h.get("score", 0) for h in hits) / max(len(hits), 1)

    # Step 6 — Render page
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "uploaded": uploaded_files,
            "answer": answer["text"],
            "question": question,
            "sources": sources,
            "chunks": chunk_previews,
                "highlight_images": highlight_urls,
                "confidence_avg": round(avg_conf * 100, 1),
        },
    )


@app.post("/clear_cache")
async def clear_cache_route(request: Request):
    """Clear all cached chunk data and re-render the index with a message."""
    clear_cache()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "uploaded": uploaded_files,
            "answer": "🧹 Cache cleared successfully!",
        },
    )


@app.post("/clear_index")
async def clear_index(request: Request):
    """Clear the persistent Chroma index by deleting the storage directory."""
    storage_dir = os.path.join(BASE_DIR, "storage", "chroma_db")
    try:
        shutil.rmtree(storage_dir, ignore_errors=True)
        os.makedirs(storage_dir, exist_ok=True)
        # Reinitialize indexer client to the new empty DB
        global indexer
        indexer = ChromaIndexer(embedding_function=embedder.embed_text, persist_directory=storage_dir)
    except Exception as e:
        print("[WARN] clear_index failed:", e)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "uploaded": uploaded_files, "answer": "🧹 Chroma index cleared successfully!"},
    )


@app.post("/feedback")
async def feedback(request: Request, question: str = Form(...), answer: str = Form(...), correctness: str = Form(...)):
    """Record user feedback (correct / incorrect) for RAG answers."""
    try:
        record_feedback(question=question, answer=answer, correctness=correctness)
        summary = get_feedback_summary()
        msg = f"✅ Feedback received! {summary}"
    except Exception as e:
        print("[WARN] Failed to record feedback:", e)
        msg = "⚠️ Failed to record feedback"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "uploaded": uploaded_files, "answer": msg},
    )


@app.get("/feedback_dashboard", response_class=HTMLResponse)
async def feedback_dashboard(request: Request):
    """Display feedback statistics and allow fine-tuning."""
    data = _load_feedback()
    summary = get_feedback_summary()
    total = len(data)
    correct = sum(1 for x in data if x.get("correctness") == "correct")
    incorrect = sum(1 for x in data if x.get("correctness") == "incorrect")

    return templates.TemplateResponse(
        "feedback_dashboard.html",
        {
            "request": request,
            "summary": summary,
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "feedback_data": data[::-1][:50],  # show latest 50
        },
    )


@app.post("/train_feedback_model")
async def train_feedback_model(request: Request):
    """Run fine-tuning script directly from the UI."""
    script_path = os.path.join(BASE_DIR, "..", "train_feedback_embeddings.py")

    try:
        print(f"🚀 Launching fine-tuning process: {script_path}")
        process = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            check=True,
        )
        output = process.stdout[-1000:]
        message = "✅ Fine-tuning complete. Model updated successfully!"
    except subprocess.CalledProcessError as e:
        output = e.stderr or str(e)
        message = "❌ Fine-tuning failed."

    return templates.TemplateResponse(
        "feedback_dashboard.html",
        {
            "request": request,
            "summary": get_feedback_summary(),
            "feedback_data": _load_feedback()[::-1][:50],
            "train_output": output,
            "message": message,
        },
    )


@app.get("/benchmark_dashboard", response_class=HTMLResponse)
async def benchmark_dashboard(request: Request):
    """Render model benchmarking interface."""
    return templates.TemplateResponse(
        "benchmark_dashboard.html",
        {
            "request": request,
            "results": None,
            "plot_precision": None,
            "plot_recall": None,
            "plot_mrr": None,
        },
    )


@app.post("/run_benchmark")
async def run_benchmark(request: Request, models: str = Form(...), chunk_size: int = Form(200), top_k: int = Form(5)):
    """
    Run embedding benchmark across provided models using stored feedback data.
    """
    data = _load_feedback()
    if not data:
        return templates.TemplateResponse(
            "benchmark_dashboard.html",
            {
                "request": request,
                "results": [],
                "message": "⚠️ No feedback data available for benchmarking.",
            },
        )

    queries = [f["question"] for f in data]
    answers = [f["answer"] for f in data]
    MODELS = [m.strip() for m in models.split(",") if m.strip()]

    PDF_PATH = os.path.join(BASE_DIR, "samples", "vdoc_rag_test.pdf")
    try:
        raw_chunks = [d["text"] for d in process_pdf(PDF_PATH)]
    except Exception as e:
        print("[WARN] Could not process sample PDF for benchmark, falling back to small corpus:", e)
        raw_chunks = [
            "Yearly sales have been increasing steadily from 2018 to 2024, with a notable jump in 2021.",
            "Charlie achieved the highest score in the table with 98 points.",
            "The event will be held on November 20, 2025 at the downtown auditorium.",
        ]

    # Split raw_chunks into sub-chunks by character length
    chunks = []
    for ch in raw_chunks:
        for i in range(0, len(ch), chunk_size):
            chunks.append(ch[i : i + chunk_size])

    results = []
    for model_name in MODELS:
        try:
            print(f"🧠 Evaluating {model_name}...")
            model = SentenceTransformer(model_name)
            chunk_embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        except Exception as e:
            print(f"[ERROR] Failed to load model {model_name}:", e)
            continue

        precision_scores, recall_scores, mrr_scores = [], [], []

        for q, ans in zip(queries, answers):
            qvec = model.encode([q], normalize_embeddings=True)
            sims = cosine_similarity(qvec, chunk_embeddings)[0]
            top_idx = np.argsort(sims)[::-1][:top_k]
            retrieved = [chunks[i] for i in top_idx]
            relevant = [1 if ans.lower() in c.lower() else 0 for c in retrieved]
            precision = sum(relevant) / top_k
            recall = sum(relevant) / max(1, len([c for c in chunks if ans.lower() in c.lower()]))
            mrr = 0
            for rank, rel in enumerate(relevant, start=1):
                if rel:
                    mrr = 1 / rank
                    break
            precision_scores.append(precision)
            recall_scores.append(recall)
            mrr_scores.append(mrr)

        results.append({
            "model": model_name,
            "precision": round(np.mean(precision_scores), 3),
            "recall": round(np.mean(recall_scores), 3),
            "mrr": round(np.mean(mrr_scores), 3),
        })

    df = pd.DataFrame(results)
    print(df)

    def make_plot(metric):
        plt.figure(figsize=(6, 4))
        plt.barh(df["model"], df[metric], color="skyblue")
        plt.title(f"{metric.upper()} Comparison")
        plt.xlabel(metric.upper())
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()
        return f"data:image/png;base64,{img_base64}"

    plot_precision = make_plot("precision") if not df.empty else None
    plot_recall = make_plot("recall") if not df.empty else None
    plot_mrr = make_plot("mrr") if not df.empty else None

    return templates.TemplateResponse(
        "benchmark_dashboard.html",
        {
            "request": request,
            "results": results,
            "plot_precision": plot_precision,
            "plot_recall": plot_recall,
            "plot_mrr": plot_mrr,
        },
    )

# ---------------------------------------------------------
# Run app
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
