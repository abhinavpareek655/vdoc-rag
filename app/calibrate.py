import fitz
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image
import io
import json
import os

pdf_path = "samples/vdoc_rag_test.pdf"
config_path = "highlight_calibration.json"

# Example hits
hits = [
    {"metadata": {"page": 1, "bbox": [87, 1926, 775, 1957], "type": "text"}},
    {"metadata": {"page": 2, "bbox": [87, 222, 592, 250], "type": "text"}},
]

# Load PDF
doc = fitz.open(pdf_path)

# Render both pages
pix1 = doc[0].get_pixmap(dpi=150)
pix2 = doc[1].get_pixmap(dpi=150)

img1 = Image.open(io.BytesIO(pix1.tobytes("png")))
img2 = Image.open(io.BytesIO(pix2.tobytes("png")))

# Combined figure (2 pages side-by-side)
fig, axes = plt.subplots(1, 2, figsize=(16, 10))
plt.subplots_adjust(bottom=0.25)
axes[0].imshow(img1)
axes[0].set_title("Page 1", fontsize=12)
axes[1].imshow(img2)
axes[1].set_title("Page 2", fontsize=12)
for ax in axes:
    ax.axis("off")

# Keep reference sizes
img1_w, img1_h = img1.size
img2_w, img2_h = img2.size

# Prepare highlight rectangles for both pages
rects_page1, rects_page2 = [], []
for h in hits:
    meta = h["metadata"]
    page_idx = meta["page"] - 1
    x0, y0, x1, y1 = [float(v) for v in meta["bbox"]]
    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                         linewidth=2, edgecolor='r', facecolor='r', alpha=0.4)
    if page_idx == 0:
        rects_page1.append(rect)
        axes[0].add_patch(rect)
    elif page_idx == 1:
        rects_page2.append(rect)
        axes[1].add_patch(rect)

# 🎚️ Shared sliders
axcolor = 'lightgoldenrodyellow'
ax_x_offset = plt.axes([0.25, 0.12, 0.65, 0.03], facecolor=axcolor)
ax_x_scale  = plt.axes([0.25, 0.09, 0.65, 0.03], facecolor=axcolor)
ax_y_offset = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)
ax_y_scale  = plt.axes([0.25, 0.03, 0.65, 0.03], facecolor=axcolor)
ax_save     = plt.axes([0.85, 0.17, 0.10, 0.04])

slider_x_offset = Slider(ax_x_offset, 'X Offset', -500, 500, valinit=0, valstep=0.5)
slider_x_scale  = Slider(ax_x_scale,  'X Scale',  0.3, 2.0, valinit=1.0, valstep=0.002)
slider_y_offset = Slider(ax_y_offset, 'Y Offset', -1500, 1500, valinit=0, valstep=0.5)
slider_y_scale  = Slider(ax_y_scale,  'Y Scale',  0.3, 2.0, valinit=1.0, valstep=0.002)
btn_save        = Button(ax_save, '💾 Save', color=axcolor, hovercolor='0.9')

def update(val):
    xo, xs = slider_x_offset.val, slider_x_scale.val
    yo, ys = slider_y_offset.val, slider_y_scale.val

    # Page 1
    for i, h in enumerate(rects_page1):
        bbox = hits[0]["metadata"]["bbox"]
        x0, y0, x1, y1 = [float(v) for v in bbox]
        x0 = x0 * xs + xo
        x1 = x1 * xs + xo
        y0 = y0 * ys + yo
        y1 = y1 * ys + yo
        h.set_xy((x0, y1))
        h.set_width(x1 - x0)
        h.set_height(y0 - y1)

    # Page 2
    for i, h in enumerate(rects_page2):
        bbox = hits[1]["metadata"]["bbox"]
        x0, y0, x1, y1 = [float(v) for v in bbox]
        x0 = x0 * xs + xo
        x1 = x1 * xs + xo
        y0 = y0 * ys + yo
        y1 = y1 * ys + yo
        h.set_xy((x0, y1))
        h.set_width(x1 - x0)
        h.set_height(y0 - y1)

    fig.suptitle(
        f"Xo={xo:.1f}, Xs={xs:.3f} | Yo={yo:.1f}, Ys={ys:.3f}",
        fontsize=11, color='darkred'
    )
    fig.canvas.draw_idle()

for s in [slider_x_offset, slider_x_scale, slider_y_offset, slider_y_scale]:
    s.on_changed(update)

def save_values(event):
    xo, xs = slider_x_offset.val, slider_x_scale.val
    yo, ys = slider_y_offset.val, slider_y_scale.val
    calib = {
        "x_offset": xo, "x_scale": xs,
        "y_offset": yo, "y_scale": ys
    }
    with open(config_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"✅ Saved combined calibration: {calib}")

btn_save.on_clicked(save_values)

plt.show()
