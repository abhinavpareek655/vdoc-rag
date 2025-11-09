import os
import json
import pytesseract
from PIL import Image
import numpy as np

# HuggingFace transformers optional imports
try:
    from transformers import VisionEncoderDecoderModel, AutoProcessor, AutoTokenizer
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# Model IDs (optional, set via env)
DONUT_MODEL_ID = os.environ.get('VDOCRAG_DONUT_MODEL', 'naver-clova-ix/donut-base')
PIX2STRUCT_MODEL_ID = os.environ.get('VDOCRAG_PIX2STRUCT_MODEL', 'google/pix2struct-model')

# Try Pix2Struct/TextCaps style model for captioning (optional)
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    _pix2_processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
    _pix2_model = AutoModelForVision2Seq.from_pretrained("google/pix2struct-textcaps-base")
    USE_PIX2STRUCT = True
    print("[INFO] Using Pix2Struct/TextCaps model for chart reasoning.")
except Exception:
    USE_PIX2STRUCT = False
    print("[INFO] Pix2Struct/TextCaps not available; will use OCR fallback.")


def ocr_crop_summary(image_path):
    """
    Fallback: simple OCR on the crop, then heuristics for numeric extraction.
    Return: dict with summary and structured_data (if found)
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {'summary_text': f"[ERROR] cannot open image {image_path}: {e}", 'structured': {}}

    try:
        text = pytesseract.image_to_string(img)
    except Exception as e:
        return {'summary_text': f"[ERROR] OCR failed for {image_path}: {e}", 'structured': {}}

    import re
    nums = re.findall(r'[-+]?\d[\d,\.]*', text)
    summary = text.strip().replace("\n", " ")
    structured = {}
    if nums:
        structured['numbers'] = nums[:50]
    return {'summary_text': f"Chart OCR summary: {summary[:400]}", 'structured': structured}


def hf_image_to_text_summary(image_path, model_id=PIX2STRUCT_MODEL_ID):
    """
    Generic HF image->text example. Returns None if HF not available or fails.
    """
    if not HF_AVAILABLE:
        return None
    try:
        model = VisionEncoderDecoderModel.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print("[chart_reasoner] HF model load failed:", e)
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        output_ids = model.generate(pixel_values, max_length=256)
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        try:
            parsed = json.loads(decoded)
            summary_text = parsed.get('summary', decoded[:400])
            structured = parsed.get('data', parsed)
        except Exception:
            summary_text = decoded[:400]
            structured = {}
        return {'summary_text': summary_text, 'structured': structured}
    except Exception as e:
        print("[chart_reasoner] HF inference failed:", e)
        return None


def process_chart_crop(image_path):
    """
    Summarize or extract insights from a chart image.
    Returns dict: {"summary_text": "...", "structured": {...}}
    """
    if not os.path.exists(image_path):
        return {"summary_text": f"[Error] Chart image not found: {image_path}", "structured": {}}

    # Try Pix2Struct/TextCaps style captioning first (if available)
    if USE_PIX2STRUCT:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = _pix2_processor(images=image, text="Describe this chart in detail.", return_tensors="pt")
            outputs = _pix2_model.generate(**inputs, max_new_tokens=128)
            # processor.decode may not exist for all processors; try both
            try:
                caption = _pix2_processor.decode(outputs[0], skip_special_tokens=True)
            except Exception:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("google/pix2struct-textcaps-base")
                caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"summary_text": caption.strip(), "structured": {}}
        except Exception as e:
            print("[WARN] Pix2Struct/TextCaps failed:", e)

    # Fallback: OCR-based heuristic summary
    return ocr_crop_summary(image_path)
