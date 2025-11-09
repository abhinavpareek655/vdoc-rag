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
import pdfplumber
from app.tables import extract_tables_from_pdf
from app.chart_detect import detect_charts
from app.chart_reasoner import process_chart_crop


def pdf_to_images(pdf_path, dpi=200):
    """
    Convert a PDF into page-wise PNG images for OCR and visual analysis.
    """
    pages = convert_from_path(pdf_path, dpi=dpi)
    paths = []
    tmpdir = "/tmp" if os.name != "nt" else os.path.join(os.path.dirname(__file__), "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    for i, p in enumerate(pages, start=1):
        ppath = os.path.join(tmpdir, f"page_{uuid.uuid4().hex}_{i}.png")
        p.save(ppath, "PNG")
        paths.append(ppath)
    return paths


def ocr_image_to_blocks(image_path, min_words_per_line=3):
    """
    Run OCR on an image and merge words into line-level text blocks.
    This preserves full sentences like 'Venue: Delhi Convention Hall'.
    """
    img = Image.open(image_path).convert("RGB")
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--oem 3 --psm 6")
    n = len(data["text"])
    lines = {}
    for i in range(n):
        txt = data["text"][i].strip()
        if not txt:
            continue
        line_no = data["line_num"][i]
        if line_no not in lines:
            lines[line_no] = {"words": [], "lefts": [], "tops": [], "rights": [], "bottoms": []}
        lines[line_no]["words"].append(txt)
        lines[line_no]["lefts"].append(data["left"][i])
        lines[line_no]["tops"].append(data["top"][i])
        lines[line_no]["rights"].append(data["left"][i] + data["width"][i])
        lines[line_no]["bottoms"].append(data["top"][i] + data["height"][i])

    blocks = []
    for ln, d in lines.items():
        if len(d["words"]) < min_words_per_line:
            continue
        text = " ".join(d["words"]).strip()
        bbox = (
            min(d["lefts"]),
            min(d["tops"]),
            max(d["rights"]),
            max(d["bottoms"]),
        )
        blocks.append({"text": text, "bbox": bbox})
    return blocks


def process_pdf(path):
    """
    Process a PDF or image file:
    - Extract text chunks (OCR)
    - Extract tables (pdfplumber)
    - Detect charts (layoutparser or OpenCV)
    - Run chart reasoning model (Donut/Pix2Struct/heuristics)
    Returns: list of document chunks {id, text, metadata}
    """
    items = []

    # 1️⃣ OCR text extraction (page images)
    images = pdf_to_images(path)
    for pno, imgpath in enumerate(images, start=1):
        img = Image.open(imgpath).convert("RGB")
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--oem 3 --psm 6")
        n = len(data["text"])

        current_line = None
        line_words, lefts, tops, rights, bottoms = [], [], [], [], []

        for i in range(n):
            text = data["text"][i].strip()
            if not text:
                continue
            line_num = data["line_num"][i]

            # Start new line if changed
            if current_line is None:
                current_line = line_num

            if line_num != current_line:
                # finalize previous line
                if line_words:
                    doc = {
                        "id": f"{uuid.uuid4().hex}",
                        "text": " ".join(line_words),
                        "metadata": {
                            "source": path,
                            "page": pno,
                            "bbox": (min(lefts), min(tops), max(rights), max(bottoms)),
                            "type": "text",
                        },
                    }
                    items.append(doc)
                # reset
                current_line = line_num
                line_words, lefts, tops, rights, bottoms = [], [], [], [], []

            # collect current word
            line_words.append(text)
            lefts.append(data["left"][i])
            tops.append(data["top"][i])
            rights.append(data["left"][i] + data["width"][i])
            bottoms.append(data["top"][i] + data["height"][i])

        # flush last line
        if line_words:
            doc = {
                "id": f"{uuid.uuid4().hex}",
                "text": " ".join(line_words),
                "metadata": {
                    "source": path,
                    "page": pno,
                    "bbox": (min(lefts), min(tops), max(rights), max(bottoms)),
                    "type": "text",
                },
            }
            items.append(doc)


    # 2️⃣ Table extraction (structured CSVs)
    try:
        tables = extract_tables_from_pdf(path)
        for t in tables:
            doc = {
                "id": f"{uuid.uuid4().hex}",
                "text": t["summary_text"],
                "metadata": {
                    "source": path,
                    "page": t["page"],
                    "type": "table",
                    "csv_path": t["csv_path"],
                    "rows": t["rows"],
                    "bbox": t.get("bbox"),
                },
            }
            items.append(doc)
    except Exception as e:
        print("[WARN] Table extraction failed:", e)

    # 3️⃣ Chart detection + reasoning
    # for pno, imgpath in enumerate(images, start=1):
    #     try:
    #         chart_crops = detect_charts(imgpath, debug=True)
    #         for c in chart_crops:
    #             crop_path = c["image_path"]
    #             bbox = c["bbox"]

    #             # Run reasoning model or OCR heuristic
    #             chart_res = process_chart_crop(crop_path)
    #             summary = chart_res.get("summary_text", "Chart region detected.")
    #             structured = chart_res.get("structured", {})

    #             doc = {
    #                 "id": f"chart_{uuid.uuid4().hex}",
    #                 "text": summary,
    #                 "metadata": {
    #                     "source": path,
    #                     "page": pno,
    #                     "type": "chart",
    #                     "bbox": bbox,
    #                     "image_path": crop_path,
    #                     "structured": structured,
    #                 },
    #             }
    #             items.append(doc)

    #     except Exception as e:
    #         print(f"[WARN] Chart detection/reasoning failed on page {pno}:", e)

    return items
