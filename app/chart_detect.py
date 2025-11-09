# app/chart_detect.py
import cv2
import os
import uuid
from PIL import Image
import numpy as np

# configurable charts dir (relative to project)
CHARTS_DIR = os.environ.get("VDOCRAG_CHARTS_DIR", os.path.join(os.getcwd(), "charts"))
os.makedirs(CHARTS_DIR, exist_ok=True)

def _ensure_bgr(img_or_path):
    """
    Accept either a file path or a PIL/Image/ndarray and return BGR ndarray or None.
    """
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
        if img is None:
            raise ValueError(f"cv2.imread failed for path: {img_or_path}")
        return img
    # PIL image
    if isinstance(img_or_path, Image.Image):
        return cv2.cvtColor(np.array(img_or_path), cv2.COLOR_RGB2BGR)
    # ndarray assumed BGR or RGB — try to accept it
    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    raise ValueError("Unsupported image type for detect_charts")

def detect_charts(image_or_path, min_area=20000, debug=False):
    """
    Detect likely chart regions and save cropped images to CHARTS_DIR.
    Returns list of dicts: {"bbox": (x0,y0,x1,y1), "image_path": "<abs path>"}
    """
    try:
        img = _ensure_bgr(image_or_path)
    except Exception as e:
        if debug:
            print("[chart_detect] image load error:", e)
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    charts = []
    h_img, w_img = img.shape[:2]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect = w / (h + 1e-8)
        # reject tiny boxes and boxes that cover almost whole page
        if area < min_area or w > 0.95 * w_img or h > 0.95 * h_img:
            continue
        if not (0.25 < aspect < 4.0):
            continue

        # expand bbox slightly to include axis labels
        pad_x = int(min(0.05 * w, 30))
        pad_y = int(min(0.05 * h, 30))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w_img, x + w + pad_x)
        y1 = min(h_img, y + h + pad_y)

        crop = img[y0:y1, x0:x1]
        crop_name = f"chart_{uuid.uuid4().hex}.png"
        crop_path = os.path.join(CHARTS_DIR, crop_name)

        try:
            ok = cv2.imwrite(crop_path, crop)
            if not ok:
                if debug:
                    print(f"[chart_detect] cv2.imwrite failed for {crop_path}")
                continue
        except Exception as e:
            if debug:
                print(f"[chart_detect] Exception saving crop {crop_path}:", e)
            continue

        charts.append({"bbox": (int(x0), int(y0), int(x1), int(y1)), "image_path": crop_path})

    # optional: sort by area desc (largest first)
    charts.sort(key=lambda c: (c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1]), reverse=True)
    if debug:
        print(f"[chart_detect] Found {len(charts)} chart(s). Saved to {CHARTS_DIR}")
    return charts
