# app/visual_highlight.py
import os
import uuid
from pdf2image import convert_from_path
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def render_highlighted_pages(pdf_path, hits, output_dir=None, dpi=150):
    """
    Render PDF pages as images and highlight bounding boxes accurately.
    Handles both OCR (pixel) and PDF (point-based) coordinates.
    """
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "highlighted")
    os.makedirs(output_dir, exist_ok=True)

    # Clear previous highlights
    for old in os.listdir(output_dir):
        try:
            os.remove(os.path.join(output_dir, old))
        except Exception:
            pass

    # Find which pages to render
    pages_to_render = sorted({h["metadata"].get("page") for h in hits if h.get("metadata")})
    if not pages_to_render:
        return []

    first_page, last_page = min(pages_to_render), max(pages_to_render)

    # Convert the relevant pages to images
    pdf_images = convert_from_path(pdf_path, dpi=dpi, first_page=first_page, last_page=last_page)
    result_paths = []

    for idx, page_num in enumerate(range(first_page, last_page + 1)):
        if page_num not in pages_to_render:
            continue

        img = pdf_images[idx].convert("RGBA")
        w_img, h_img = img.size
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        page_bboxes = []

        for h in hits:
            meta = h.get("metadata", {})
            if meta.get("page") != page_num:
                continue

            bbox = meta.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            try:
                x0, y0, x1, y1 = [float(v) for v in bbox]
            except Exception:
                continue

            # Detect coordinate system: PDF (low values ~ hundreds) vs pixel (high values ~ 1000s)
            if max(x1, y1) < 2000:
                # Convert PDF units (1/72 inch) → pixels (dpi/72)
                scale = dpi / 72
                x0 *= scale
                x1 *= scale
                y0 *= scale
                y1 *= scale
                # Flip y-axis (PDF origin bottom-left → image origin top-left)
                y0, y1 = h_img - y1, h_img - y0

            # Clip to image bounds
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w_img, x1), min(h_img, y1)
            page_bboxes.append((x0, y0, x1, y1))

            # 🟩 Different colors by type
            t = meta.get("type", "text")
            if t == "table":
                color = (0, 0, 255, 100)  # blue
            elif t == "chart":
                color = (0, 255, 0, 100)  # green
            else:
                color = (255, 0, 0, 100)  # red

            draw.rectangle(
                [x0, y0, x1, y1],
                outline=color[:3],
                width=6,
                fill=color
            )

        # Merge overlay
        highlighted = Image.alpha_composite(img, overlay)

        # Crop tightly around highlights
        if page_bboxes:
            min_x = min(b[0] for b in page_bboxes)
            min_y = min(b[1] for b in page_bboxes)
            max_x = max(b[2] for b in page_bboxes)
            max_y = max(b[3] for b in page_bboxes)
            pad = 50
            crop_box = (
                max(0, int(min_x - pad)),
                max(0, int(min_y - pad)),
                int(min(max_x + pad, w_img)),
                int(min(max_y + pad, h_img)),
            )
            highlighted = highlighted.crop(crop_box)

        # Save highlighted image
        out_path = os.path.join(output_dir, f"highlight_{page_num}_{uuid.uuid4().hex}.png")
        highlighted.convert("RGB").save(out_path)
        result_paths.append(out_path)

    return result_paths
