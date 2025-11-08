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

# Load Jinja2 templates safely
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# ---------------------------------------------------------
# Core Components
# ---------------------------------------------------------
embedder = TextImageEmbedder()
indexer = ChromaIndexer(embedding_function=embedder.embed_text)
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
        print(f"Chunk {i+1}: Page {meta.get('page')} | BBox: {meta.get('bbox')}")
        print(f"Text: {h['text'][:500]}...\n")

    # Step 3 — Build context string
    context_blocks = [
        f"[{i+1}] {h['text']} (page: {h['metadata'].get('page')}, bbox: {h['metadata'].get('bbox')})"
        for i, h in enumerate(hits)
    ]
    context = "\n".join(context_blocks)

    # Step 4 — Ask LLM
    answer = reader.answer_question(query=question, context=context, sources=hits)
    sources = answer.get("sources", [])

    # Step 5 — Prepare chunk previews for UI
    chunk_previews = [
        {
            "index": i + 1,
            "page": h["metadata"].get("page"),
            "bbox": h["metadata"].get("bbox"),
            "text": h["text"][:300] + ("..." if len(h["text"]) > 300 else ""),
        }
        for i, h in enumerate(hits)
    ]

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
        },
    )

# ---------------------------------------------------------
# Run app
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
