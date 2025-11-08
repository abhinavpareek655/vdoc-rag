# app/chart_detect.py
import os
from PIL import Image
import uuid
import numpy as np

# Prefer layoutparser + detectron2 if installed
try:
    import layoutparser as lp
    LP_AVAILABLE = True
except Exception:
    LP_AVAILABLE = False

import cv2

CHARTS_DIR = os.environ.get('VDOCRAG_CHARTS_DIR', '/tmp/vdoc_charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

def detect_charts_with_layoutparser(image_path):
    """
    Use a pretrained layoutparser model (e.g., detectron2 model trained for figure/table detection).
    This code assumes model availability; you should swap the model config if needed.
    """
    try:
        model = lp.Detectron2LayoutModel(
            config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',  # example
            label_map={0: "text", 1: "title", 2: "figure", 3: "table"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]
        )
    except Exception as e:
        print("LayoutParser model load failed:", e)
        return []

    image = Image.open(image_path).convert("RGB")
    layout = model.detect(image)
    # filter for figures/charts
    chart_blocks = [b for b in layout if b.type.lower() in ("figure", "chart", "image")]
    crops = []
    for b in chart_blocks:
        x1, y1, x2, y2 = map(int, [b.x_1, b.y_1, b.x_2, b.y_2])
        crop = image.crop((x1, y1, x2, y2))
        fname = os.path.join(CHARTS_DIR, f"chart_{uuid.uuid4().hex}.png")
        crop.save(fname)
        crops.append({'image_path': fname, 'bbox': (x1, y1, x2, y2)})
    return crops

def detect_charts_opencv(image_path, min_area=5000):
    """
    Simple OpenCV heuristic: find large rectangular contours likely to be charts/figures.
    Works as fallback on many PDFs where chart regions have borders or distinct content.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur + edge detection
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    # Dilate to close gaps
    kernel = np.ones((5,5), np.uint8)
    dil = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    crops = []
    for cnt in contours:
        x,y,ww,hh = cv2.boundingRect(cnt)
        area = ww * hh
        # heuristics: area threshold and aspect ratio
        if area < min_area: 
            continue
        ar = ww / float(hh)
        if ar < 0.3 or ar > 10:  # skip extreme aspect ratios
            continue
        # crop & save
        crop = img[y:y+hh, x:x+ww]
        fname = os.path.join(CHARTS_DIR, f"chart_{uuid.uuid4().hex}.png")
        cv2.imwrite(fname, crop)
        crops.append({'image_path': fname, 'bbox': (x, y, x+ww, y+hh)})
    return crops

def detect_charts(image_path):
    if LP_AVAILABLE:
        try:
            return detect_charts_with_layoutparser(image_path)
        except Exception as e:
            print("layoutparser detection failed:", e)
    # fallback
    return detect_charts_opencv(image_path)
