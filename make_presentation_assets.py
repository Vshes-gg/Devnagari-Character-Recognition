"""Generate presentation assets: DHCD montage, before/after preprocessing,
and a shirorekha segmentation demo figure."""
import os
import glob
import cv2
import numpy as np
from PIL import Image

import devanagari_ocr as engine

OUT = "presentation_assets"
os.makedirs(OUT, exist_ok=True)
ACCENT_BGR = (61, 133, 232)   # E8853D saffron
PRIMARY_BGR = (117, 55, 61)   # 3D3775 indigo


# ---------------------------------------------------------------- 1. montage
def dhcd_montage():
    train_dir = "DevanagariHandwrittenCharacterDataset/Train"
    classes = sorted(d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d)))
    tile, gap, cols = 110, 8, 8
    rows = (len(classes) + cols - 1) // cols
    W = cols * tile + (cols - 1) * gap + 2 * gap
    H = rows * tile + (rows - 1) * gap + 2 * gap
    canvas = np.zeros((H, W), dtype=np.uint8)
    for i, cls in enumerate(classes):
        r, c = divmod(i, cols)
        path = sorted(glob.glob(os.path.join(train_dir, cls, "*.png")))[0]
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, (tile, tile), interpolation=cv2.INTER_CUBIC)
        y = gap + r * (tile + gap)
        x = gap + c * (tile + gap)
        canvas[y:y + tile, x:x + tile] = g
    cv2.imwrite(f"{OUT}/dhcd_montage.png", canvas)
    print("montage", canvas.shape, "classes:", len(classes))


# ------------------------------------------------------- 2. before / after
def before_after(proc_gray):
    before = cv2.imread("my_handwriting.jpeg")
    h = 900
    before = cv2.resize(before, (int(before.shape[1] * h / before.shape[0]), h),
                        interpolation=cv2.INTER_AREA)
    after = cv2.cvtColor(proc_gray, cv2.COLOR_GRAY2BGR)
    after = cv2.resize(after, (int(after.shape[1] * h / after.shape[0]), h),
                       interpolation=cv2.INTER_AREA)
    gap, margin = 28, 2
    W = before.shape[1] + after.shape[1] + gap + 2 * margin
    canvas = np.full((h + 2 * margin, W, 3), 255, dtype=np.uint8)
    x = margin
    canvas[margin:margin + h, x:x + before.shape[1]] = before
    x += before.shape[1] + gap
    canvas[margin:margin + h, x:x + after.shape[1]] = after
    cv2.imwrite(f"{OUT}/before_after.png", canvas)
    print("before_after", canvas.shape)


# ------------------------------------------------- 3. segmentation figure
def segmentation_figure(proc_gray):
    boxes = engine.extract_line_crops(proc_gray)
    # pick the line holding the most character boxes
    best, best_n = None, -1
    for (x1, y1, x2, y2) in boxes:
        n = len(engine.segment_characters_from_line(proc_gray[y1:y2, x1:x2]))
        if n > best_n:
            best, best_n = (x1, y1, x2, y2), n
    x1, y1, x2, y2 = best
    crop = proc_gray[y1:y2, x1:x2]

    binary = engine._ink_binary(crop)
    stripped, _ = engine.remove_shirorekha(binary)
    char_boxes = engine.segment_characters_from_line(crop)

    line_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    for (bx1, by1, bx2, by2) in char_boxes:
        cv2.rectangle(line_bgr, (bx1, by1), (bx2, by2), ACCENT_BGR, 3)

    panels = [line_bgr,
              cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
              cv2.cvtColor(stripped, cv2.COLOR_GRAY2BGR)]
    scale = 1500 / max(p.shape[1] for p in panels)
    panels = [cv2.resize(p, (int(p.shape[1] * scale), int(p.shape[0] * scale)),
                         interpolation=cv2.INTER_CUBIC) for p in panels]
    pad, gap = 14, 26
    W = max(p.shape[1] for p in panels) + 2 * pad
    H = sum(p.shape[0] for p in panels) + 2 * pad + gap * (len(panels) - 1)
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    y = pad
    for p in panels:
        canvas[y:y + p.shape[0], pad:pad + p.shape[1]] = p
        y += p.shape[0] + gap
    cv2.imwrite(f"{OUT}/segmentation_demo.png", canvas)
    print("segmentation", canvas.shape, "chars:", len(char_boxes))


proc = engine.full_preprocessing_pipeline(cv2.imread("my_handwriting.jpeg"))
dhcd_montage()
before_after(proc)
segmentation_figure(proc)
