# app/chart_reasoner.py
import os
from PIL import Image
import json
import numpy as np
import pytesseract

# HuggingFace transformers imports
try:
    from transformers import VisionEncoderDecoderModel, AutoProcessor, AutoTokenizer
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# Optionally use Donut-like model from HuggingFace (example)
# Model names vary; you should pick a model that outputs JSON or a template.
# Example placeholders:
DONUT_MODEL_ID = os.environ.get('VDOCRAG_DONUT_MODEL', 'naver-clova-ix/donut-base')  # example, may require custom processor
PIX2STRUCT_MODEL_ID = os.environ.get('VDOCRAG_PIX2STRUCT_MODEL', 'google/pix2struct-model')  # placeholder

def ocr_crop_summary(image_path):
    """
    Fallback: simple OCR on the crop, then heuristics for numeric extraction.
    Return: dict with summary and structured_data (if found)
    """
    img = Image.open(image_path).convert("RGB")
    text = pytesseract.image_to_string(img)
    # heuristics: find numbers, axis labels with regex
    import re
    nums = re.findall(r'[-+]?\d[\d,\.]*', text)
    # basic summary
    summary = text.strip().replace("\n", " ")
    structured = {}
    if nums:
        structured['numbers'] = nums[:50]  # keep top 50 matches
    return {'summary_text': f"Chart OCR summary: {summary[:400]}", 'structured': structured}

# Example generic HF pipeline for image->text; you will adapt to Donut/Pix2Struct specifics
def hf_image_to_text_summary(image_path, model_id=PIX2STRUCT_MODEL_ID):
    """
    Simple example using VisionEncoderDecoderModel; many chart-specific models require specialized processors.
    This is a generic approach: encode image and decode text; the decoded text is then parsed.
    """
    if not HF_AVAILABLE:
        return None
    try:
        model = VisionEncoderDecoderModel.from_pretrained(model_id)
        # Processor/tokenizer may differ by model
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print("HF model load failed:", e)
        return None

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=256)
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # attempt to parse JSON inside decoded text
    try:
        parsed = json.loads(decoded)
        summary_text = parsed.get('summary', decoded[:400])
        structured = parsed.get('data', parsed)
    except Exception:
        summary_text = decoded[:400]
        structured = {}
    return {'summary_text': summary_text, 'structured': structured}

def process_chart_crop(image_path):
    """
    High-level function to produce summary + structured data from a chart crop.
    Prioritize specialized HF models; fallback to OCR heuristics.
    """
    # First try HuggingFace model (Pix2Struct or Donut)
    res = hf_image_to_text_summary(image_path)
    if res:
        return res
    # Fallback
    return ocr_crop_summary(image_path)
