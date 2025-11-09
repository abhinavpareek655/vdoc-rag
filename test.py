import os
import uuid
import json
from pdf2image import convert_from_path
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_calibration(config_path="highlight_calibration.json"):
    """Load calibration values from JSON or fallback to defaults."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            calib = json.load(f)
        print(f"✅ Loaded calibration: {calib}")
        return calib
    else:
        print("⚠️ No calibration file found. Using defaults.")
        return {"x_offset": 0, "x_scale": 1.0, "y_offset": 0, "y_scale": 1.0}


def render_highlighted_pages(pdf_path, hits, output_dir=None, dpi=150):
    """
    Render PDF pages as images and highlight bounding boxes with calibration applied.
    Crops the output image tightly around highlighted area (+20 px padding).
    """
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "highlighted")
    os.makedirs(output_dir, exist_ok=True)

    calib = load_calibration()
    X_OFFSET = calib.get("x_offset", 0)
    X_SCALE  = calib.get("x_scale", 1.0)
    Y_OFFSET = calib.get("y_offset", 0)
    Y_SCALE  = calib.get("y_scale", 1.0)

    # Clean previous outputs
    for old in os.listdir(output_dir):
        try:
            os.remove(os.path.join(output_dir, old))
        except Exception:
            pass

    pages_to_render = sorted({h["metadata"]["page"] for h in hits})
    pdf_images = convert_from_path(pdf_path, dpi=dpi)
    result_paths = []

    for page_num in pages_to_render:
        page_index = page_num - 1
        img = pdf_images[page_index].convert("RGBA")
        w_img, h_img = img.size
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        page_bboxes = []

        for h in hits:
            meta = h["metadata"]
            if meta["page"] != page_num:
                continue
            bbox = meta["bbox"]
            if not bbox or len(bbox) != 4:
                continue

            # Apply calibration
            x0, y0, x1, y1 = [float(v) for v in bbox]
            x0 = x0 * X_SCALE + X_OFFSET
            x1 = x1 * X_SCALE + X_OFFSET
            y0 = y0 * Y_SCALE + Y_OFFSET
            y1 = y1 * Y_SCALE + Y_OFFSET

            left, top = max(0, min(x0, x1)), max(0, min(y0, y1))
            right, bottom = min(w_img, max(x0, x1)), min(h_img, max(y0, y1))

            if right <= left or bottom <= top:
                continue

            page_bboxes.append((left, top, right, bottom))
            draw.rectangle(
                [left, top, right, bottom],
                outline=(255, 0, 0),
                width=4,
                fill=(255, 0, 0, 100)
            )

        # Merge highlights with image
        highlighted = Image.alpha_composite(img, overlay)

        # --- 🧭 Crop around highlighted region (+20px padding) ---
        if page_bboxes:
            min_x = min(b[0] for b in page_bboxes)
            min_y = min(b[1] for b in page_bboxes)
            max_x = max(b[2] for b in page_bboxes)
            max_y = max(b[3] for b in page_bboxes)

            pad = 100
            crop_box = (
                max(0, int(min_x - pad)),
                max(0, int(min_y - pad)),
                int(min(max_x + pad, w_img)),
                int(min(max_y + pad, h_img)),
            )

            cropped = highlighted.crop(crop_box)
        else:
            cropped = highlighted  # fallback if no bbox

        out_path = os.path.join(output_dir, f"highlight_page{page_num}_{uuid.uuid4().hex}.png")
        cropped.convert("RGB").save(out_path)
        result_paths.append(out_path)

        print(f"✅ Highlighted and cropped page {page_num}: {out_path}")

    return result_paths


# Example usage
if __name__ == "__main__":
    hits = [
        {"metadata": {"page": 1, "bbox": [87, 1926, 775, 1957], "type": "text"}},
        {"metadata": {"page": 2, "bbox": [87, 222, 592, 250], "type": "text"}},
    ]

    render_highlighted_pages("samples/vdoc_rag_test.pdf", hits)
