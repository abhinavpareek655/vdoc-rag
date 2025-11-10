import os
import re
import json
from typing import List, Dict, Any

import pytesseract
from PIL import Image
import numpy as np

# Optional HF/Pix2Struct captioning
USE_PIX2STRUCT = False
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq

    _pix2_processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
    _pix2_model = AutoModelForVision2Seq.from_pretrained("google/pix2struct-textcaps-base")
    USE_PIX2STRUCT = True
    print("[chart_reasoner] Pix2Struct/TextCaps available for chart captioning.")
except Exception:
    USE_PIX2STRUCT = False
    print("[chart_reasoner] Pix2Struct/TextCaps not available — will use OCR fallback.")
import os
import re
import json
from typing import List, Dict, Any, Optional

import pytesseract
from PIL import Image
import numpy as np
import cv2

# Optional Pix2Struct captioning
USE_PIX2STRUCT = False
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq

    _pix2_processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
    _pix2_model = AutoModelForVision2Seq.from_pretrained("google/pix2struct-textcaps-base")
    USE_PIX2STRUCT = True
    print("[chart_reasoner] Pix2Struct/TextCaps available for chart captioning.")
except Exception:
    USE_PIX2STRUCT = False
    print("[chart_reasoner] Pix2Struct/TextCaps not available — will use OCR/geometric fallback.")

# Optional CLIP embeddings via sentence-transformers
USE_CLIP = False
try:
    from sentence_transformers import SentenceTransformer

    _clip_model = SentenceTransformer("clip-ViT-B-32")
    USE_CLIP = True
    print("[chart_reasoner] CLIP (sentence-transformers) available for chart embeddings.")
except Exception:
    USE_CLIP = False


def preprocess_for_ocr(image_path: str) -> Image.Image:
    """Enhance contrast and threshold image to improve OCR inside colored charts."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # adaptive threshold for better text extraction
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    return Image.fromarray(thresh)


def _extract_numbers_from_text(text: str) -> List[float]:
    matches = re.findall(r"\(?-?\d[\d,\.\)\(]*%?", text)
    nums: List[float] = []
    for m in matches:
        s = m.strip()
        negative = False
        if s.startswith("(") and s.endswith(")"):
            negative = True
            s = s[1:-1]
        s = s.replace("%", "").replace(",", "")
        try:
            val = float(s)
            if negative:
                val = -val
            nums.append(val)
        except Exception:
            continue
    return nums


def analyze_bar_chart(image_path: str, debug_save: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Detect vertical bars and compute heights to infer a simple trend.

    Returns None if no bar-like contours are found.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img = img.shape[0]

    bars = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Vertical bar heuristic: taller than wide, reasonable size
        if w < 6 or h < 10:
            continue
        aspect = h / (w + 1e-8)
        if aspect < 1.2:
            continue
        # ignore boxes that almost cover image (likely page border)
        if h > 0.9 * h_img:
            continue
        bars.append((x, y, w, h))

    if not bars:
        return None

    # sort left-to-right
    bars = sorted(bars, key=lambda b: b[0])
    heights = [int(b[3]) for b in bars]
    # normalize heights to 0-1
    max_h = max(heights) if heights else 1
    norm = [h / max_h for h in heights]

    # trend by comparing first vs last
    trend = "increasing" if heights[-1] > heights[0] else ("decreasing" if heights[-1] < heights[0] else "flat")

    res = {
        "bar_count": len(bars),
        "heights": heights,
        "normalized_heights": norm,
        "trend": trend,
        "bars_xywh": bars,
    }

    # debug: save overlay image showing detected bars
    try:
        if debug_save:
            ov = img.copy()
            for (x, y, w, h) in bars:
                cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(debug_save, ov)
    except Exception:
        pass

    return res


def process_chart_crop(image_path: str) -> Dict[str, Any]:
    """Main entry: returns a textual summary and structured analysis for a chart image."""
    if not os.path.exists(image_path):
        return {"summary_text": f"[Error] Chart image not found: {image_path}", "structured": {}}

    pix_caption = None
    if USE_PIX2STRUCT:
        try:
            img = Image.open(image_path).convert("RGB")
            inputs = _pix2_processor(images=img, text="Describe this chart.", return_tensors="pt")
            outputs = _pix2_model.generate(**inputs, max_new_tokens=128)
            try:
                pix_caption = _pix2_processor.decode(outputs[0], skip_special_tokens=True)
            except Exception:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained("google/pix2struct-textcaps-base")
                pix_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print("[chart_reasoner] Pix2Struct failed:", e)
            pix_caption = None

    # Geometric analysis (bars)
    bar_info = None
    try:
        # debug overlay path (optional)
        debug_overlay = None
        # if an environment var set, write overlays to app/charts/debug_*
        charts_dir = os.environ.get("VDOCRAG_CHARTS_DIR", os.path.join(os.path.dirname(__file__), "charts"))
        if os.path.isdir(charts_dir):
            debug_overlay = os.path.join(charts_dir, f"debug_{os.path.basename(image_path)}")
        bar_info = analyze_bar_chart(image_path, debug_save=debug_overlay)
    except Exception as e:
        print("[chart_reasoner] analyze_bar_chart error:", e)
        bar_info = None

    # OCR with preprocessing to capture axis labels / numbers
    ocr_text = ""
    try:
        proc_img = preprocess_for_ocr(image_path)
        ocr_text = pytesseract.image_to_string(proc_img, config="--psm 6")
    except Exception as e:
        try:
            # fallback to raw OCR
            ocr_text = pytesseract.image_to_string(Image.open(image_path))
        except Exception as e2:
            return {"summary_text": f"[Error] OCR failure: {e} / {e2}", "structured": {}}

    nums = _extract_numbers_from_text(ocr_text)
    structured: Dict[str, Any] = {"ocr_text": ocr_text.strip(), "numbers": nums}

    summary_parts = []
    if pix_caption:
        summary_parts.append(pix_caption.strip())

    if ocr_text.strip():
        summary_parts.append("OCR summary: " + " ".join(ocr_text.strip().split())[:300])

    if bar_info:
        structured.update({
            "bar_count": bar_info.get("bar_count"),
            "bar_heights": bar_info.get("heights"),
            "bar_trend": bar_info.get("trend"),
            "bars_xywh": bar_info.get("bars_xywh"),
        })
        summary_parts.append(f"Bar chart trend: {bar_info.get('trend')} (left→right)")

    # Optional CLIP embedding for retrieval
    if USE_CLIP:
        try:
            emb = _clip_model.encode([" ".join(summary_parts) or ocr_text], normalize_embeddings=True)[0]
            structured["clip_vector"] = [float(x) for x in np.asarray(emb).tolist()]
        except Exception as e:
            print("[chart_reasoner] CLIP encode failed:", e)

    final_summary = " | ".join(summary_parts) if summary_parts else (ocr_text.strip() or "No description available.")

    return {"summary_text": final_summary, "structured": structured}


__all__ = ["process_chart_crop"]
