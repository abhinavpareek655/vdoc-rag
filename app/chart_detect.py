# app/chart_detect.py
import cv2
import os
import uuid
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 🗂️ Ensure charts dir exists inside project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

def _ensure_bgr(img_or_path):
    """
    Accept file path, PIL.Image, or ndarray → return OpenCV BGR ndarray.
    """
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
        if img is None:
            raise ValueError(f"[chart_detect] cv2.imread failed: {img_or_path}")
        return img
    if isinstance(img_or_path, Image.Image):
        return cv2.cvtColor(np.array(img_or_path), cv2.COLOR_RGB2BGR)
    if isinstance(img_or_path, np.ndarray):
        return img
    raise ValueError("[chart_detect] Unsupported image type.")

def detect_charts(image_or_path, min_area=15000, debug=False, visualize=False):
    """
    Detect chart-like rectangular regions in a page image.
    Saves cropped charts into CHARTS_DIR and returns metadata list.
    Each item: {"bbox": (x0,y0,x1,y1), "image_path": "<abs path>"}
    """
    try:
        img = _ensure_bgr(image_or_path)
    except Exception as e:
        print("[chart_detect] load error:", e)
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection — lowered thresholds for faint edges
    edges = cv2.Canny(blur, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = img.shape[:2]
    charts = []

    if debug:
        print(f"[chart_detect] Found {len(contours)} raw contours")

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect = w / (h + 1e-8)

        # 🔧 More forgiving filtering
        if area < min_area * 0.5:
            continue
        if w > 0.98 * w_img or h > 0.98 * h_img:
            continue
        if not (0.1 < aspect < 10.0):
            continue

        # Merge very close bounding boxes
        merged = False
        for prev in charts:
            px0, py0, px1, py1 = prev["bbox"]
            # Overlap or close enough
            if abs(x - px0) < 50 and abs(y - py0) < 50:
                px0, py0 = min(px0, x), min(py0, y)
                px1, py1 = max(px1, x + w), max(py1, y + h)
                prev["bbox"] = (px0, py0, px1, py1)
                merged = True
                break
        if merged:
            continue

        # Slight padding
        pad_x = int(min(0.1 * w, 40))
        pad_y = int(min(0.1 * h, 40))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w_img, x + w + pad_x)
        y1 = min(h_img, y + h + pad_y)

        crop = img[y0:y1, x0:x1]
        crop_name = f"chart_{uuid.uuid4().hex}.png"
        crop_path = os.path.join(CHARTS_DIR, crop_name)

        try:
            cv2.imwrite(crop_path, crop)
            charts.append({"bbox": (x0, y0, x1, y1), "image_path": crop_path})
        except Exception as e:
            print(f"[chart_detect] Failed saving {crop_path}: {e}")

    # Sort by size (largest first)
    charts.sort(key=lambda c: (c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1]), reverse=True)

    if debug:
        print(f"[chart_detect] ✅ Detected {len(charts)} likely chart(s). Saved to {CHARTS_DIR}")

    # 🧠 Optional: Visualize results
    if visualize:
        vis = img.copy()
        for c in charts:
            x0, y0, x1, y1 = c["bbox"]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 3)
        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected {len(charts)} chart(s)")
        plt.axis("off")
        plt.show()

    return charts

# Manual debug run
if __name__ == "__main__":
    test_image = "samples/vdoc_rag_test_page1.png"  # example path
    results = detect_charts(test_image, debug=True, visualize=True)
    for r in results:
        print(r)
