# app/ingest.py
"""
PDF → Images → OCR text → Table extraction → Chart detection & reasoning
Generates chunks (text/table/chart) with metadata for embedding and indexing.
"""

import os
import uuid
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from pytesseract import Output
import pdfplumber
import json
from app.tables import extract_tables_from_pdf
from app.chart_detect import detect_charts
from app.chart_reasoner import process_chart_crop


def pdf_to_images(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi)
    paths = []
    tmpdir = "/tmp" if os.name != "nt" else os.path.join(os.path.dirname(__file__), "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    for i, p in enumerate(pages, start=1):
        ppath = os.path.join(tmpdir, f"page_{uuid.uuid4().hex}_{i}.png")
        p.save(ppath, "PNG")
        paths.append(ppath)
    return paths


def ocr_image_to_lines(image_path):
    """
    Return a list of lines with bounding boxes and produce a page_text string.
    Uses pytesseract 'image_to_data' grouped by line_num.
    """
    # Keep for backward compatibility but we will not rely on this for table extraction
    img = Image.open(image_path).convert("RGB")
    data = pytesseract.image_to_data(img, output_type=Output.DICT, config="--psm 6")
    n = len(data['text'])
    lines = {}
    for i in range(n):
        txt = data['text'][i].strip()
        if not txt:
            continue
        line_num = data['line_num'][i]
        left = data['left'][i]; top = data['top'][i]; w = data['width'][i]; h = data['height'][i]
        bbox = (left, top, left + w, top + h)
        if line_num not in lines:
            lines[line_num] = {'text_parts': [], 'bboxes': []}
        lines[line_num]['text_parts'].append(txt)
        lines[line_num]['bboxes'].append(bbox)
    out_lines = []
    page_text_parts = []
    for ln in sorted(lines.keys()):
        text = " ".join(lines[ln]['text_parts'])
        xs = [b[0] for b in lines[ln]['bboxes']]; ys = [b[1] for b in lines[ln]['bboxes']]
        rights = [b[2] for b in lines[ln]['bboxes']]; bottoms = [b[3] for b in lines[ln]['bboxes']]
        merged_bbox = (min(xs), min(ys), max(rights), max(bottoms))
        out_lines.append({'text': text, 'bbox': merged_bbox})
        page_text_parts.append(text)
    page_text = "\n".join(page_text_parts)
    return out_lines, page_text


# ---------- Preprocessing helper ----------
def preprocess_image(image):
    """Enhance image (PIL Image) before OCR for better table and text recognition."""
    from PIL import ImageOps, ImageEnhance
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    img = image.convert("L")  # grayscale
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(1.5)
    return img


def ocr_image_to_blocks(img):
    """Extract words with bounding boxes from a PIL Image using Tesseract."""
    data = pytesseract.image_to_data(img, output_type=Output.DICT, config="--psm 6")
    n = len(data.get('text', []))
    blocks = []
    for i in range(n):
        text = (data.get('text', [''])[i] or '').strip()
        if not text:
            continue
        bbox = (data.get('left', [0])[i], data.get('top', [0])[i],
                data.get('left', [0])[i] + data.get('width', [0])[i],
                data.get('top', [0])[i] + data.get('height', [0])[i])
        blocks.append({'text': text, 'bbox': bbox})
    return blocks


def process_pdf(path):
    """Process a PDF and extract structured chunks using pdfplumber when possible.

    Falls back to OCR if pdfplumber fails to extract selectable text.
    Returns a list of docs: {id, text, metadata}
    """
    docs = []
    try:
        with pdfplumber.open(path) as pdf:
            for pno, page in enumerate(pdf.pages, start=1):
                # Extract selectable text
                text = page.extract_text() or ""
                if text.strip():
                    # split into paragraphs/lines
                    for para in text.split("\n"):
                        if para.strip():
                            docs.append({
                                "id": f"text_{uuid.uuid4().hex}",
                                "text": para.strip(),
                                "metadata": {"source": path, "page": pno, "bbox": None, "type": "text"},
                            })

                # Extract tables (structured)
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []
                for tbl in tables:
                    if not tbl:
                        continue
                    header = tbl[0]
                    rows = [dict(zip(header, r)) for r in tbl[1:] if any(r)]
                    table_text = json.dumps(rows, ensure_ascii=False)
                    docs.append({
                        "id": f"table_{uuid.uuid4().hex}",
                        "text": table_text,
                        "metadata": {"source": path, "page": pno, "bbox": None, "type": "table"},
                    })
    except Exception as e:
        print(f"⚠️ pdfplumber failed, switching to OCR mode: {e}")
        # Fallback to OCR via images
        pages = convert_from_path(path, dpi=300)
        for pno, pil_img in enumerate(pages, start=1):
            proc = preprocess_image(pil_img)
            blocks = ocr_image_to_blocks(proc)
            page_text = " ".join(b['text'] for b in blocks)
            docs.append({
                "id": f"page_{uuid.uuid4().hex}",
                "text": page_text,
                "metadata": {"source": path, "page": pno, "bbox": None, "type": "page"},
            })

    # Save debug file for inspection
    debug_path = os.path.join(os.path.dirname(__file__), "debug_chunks.json")
    try:
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        print(f"✅ Extracted {len(docs)} chunks → saved to {debug_path}")
    except Exception as e:
        print("[WARN] Failed to write debug_chunks.json:", e)

    return docs
