#!/usr/bin/env python3
"""Generate the layered system-architecture diagram for the demo report.

Mirrors the actual module structure of devanagari_ocr.py + evaluate_accuracy.py:
  Interface layer (CLI entry points)
  Application layer (train / recognize / evaluate pipelines)
  Shared core (CNN, label map, device abstraction)
  Data & artifacts (dataset, checkpoint, outputs)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

W, H = 6.6, 5.75
fig, ax = plt.subplots(figsize=(W, H), dpi=200)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

LX = 0.16            # left edge of content
RW = W - 2 * LX      # content width
COL_W = (RW - 2 * 0.16) / 3
GAPC = 0.16

def col_x(i):
    return LX + i * (COL_W + GAPC)

def rbox(x, y, w, h, fill="#FFFFFF", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.035,rounding_size=0.07",
        linewidth=lw, edgecolor="#1a1a1a", facecolor=fill))

def arrow_v(x, y_from, y_to):
    ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to), arrowstyle="-|>",
        mutation_scale=10, linewidth=1.0, color="#1a1a1a"))

def band_label(y, text):
    ax.text(LX + 0.02, y, text, ha="left", va="bottom",
            fontsize=9.5, fontweight="bold", color="#1a1a1a")

def ctext(x, y_center, lines, weights, sizes, linspacing=1.35, color="#1a1a1a"):
    """centered multi-line text with per-line weight/size"""
    lh = [s * linspacing for s in sizes]           # per-line height in pt
    total = sum(lh)                                 # pt
    y = y_center + (total / 2) / 72.0               # start from top, in inches
    for i, line in enumerate(lines):
        y -= (lh[i] / 2) / 72.0
        ax.text(x, y, line, ha="center", va="center",
                fontsize=sizes[i], fontweight=weights[i], color=color)
        y -= (lh[i] / 2) / 72.0

# ---------------------------------------------------------------- band A
band_label(5.62, "Interface layer \u2014 command-line entry points")
A_h = 0.46
A_y = 5.02
A = [
    ("Train", "devanagari_ocr.py\n--mode train"),
    ("Recognize", "devanagari_ocr.py\n--mode infer"),
    ("Evaluate", "evaluate_accuracy.py"),
]
for i, (t, c) in enumerate(A):
    x = col_x(i)
    rbox(x, A_y, COL_W, A_h, fill="#F2F2F2")
    ctext(x + COL_W / 2, A_y + A_h / 2,
          [t, c], ["bold", "normal"], [9.5, 8.2], 1.3)

# ---------------------------------------------------------------- band B
band_label(4.62, "Application layer \u2014 three pipelines")
B_h = 1.34
B_y = 3.12
B = [
    ("Training pipeline",
     ["DHCD loader (Train/)", "stratified 90/10 split",
      "augmentation (rotate, affine)", "training loop \u2014 AdamW,",
      "ReduceLROnPlateau", "best-val checkpointing"]),
    ("Recognition pipeline",
     ["preprocessing", "line segmentation", "character segmentation",
      "crop conditioning +", "CNN classification", "reading-order assembly"]),
    ("Evaluation pipeline",
     ["Test-split loader", "batch inference (no grad)",
      "Top-1 / Top-3 accuracy", "per-class accuracy",
      "accuracy report"]),
]
for i, (t, lines) in enumerate(B):
    x = col_x(i)
    rbox(x, B_y, COL_W, B_h)
    ctext(x + COL_W / 2, B_y + B_h / 2, [t] + lines,
          ["bold"] + ["normal"] * len(lines),
          [9.2] + [8.2] * len(lines), 1.32)

# ---------------------------------------------------------------- band C
band_label(2.72, "Shared core")
C_h = 0.60
C_y = 1.98
CORE_W = (RW - 3 * 0.12) / 4

def core_x(i):
    return LX + i * (CORE_W + 0.12)

C = [
    ("DevanagariCNN", "823,758 parameters", False),
    ("DHCD label map", "46 classes \u2192 Unicode", False),
    ("DevanagariCRNN", "line-level + CTC\n(future scope)", True),
    ("Device abstraction", "CUDA / MPS / CPU", False),
]
for i, (t, s, dashed) in enumerate(C):
    x = core_x(i)
    box = FancyBboxPatch((x, C_y), CORE_W, C_h,
        boxstyle="round,pad=0.035,rounding_size=0.07",
        linewidth=1.0, edgecolor="#1a1a1a",
        facecolor="#FFFFFF", linestyle="--" if dashed else "-")
    ax.add_patch(box)
    color = "#555555" if dashed else "#1a1a1a"
    lines = s.split("\n")
    ctext(x + CORE_W / 2, C_y + C_h / 2,
          [t] + lines, ["bold"] + ["normal"] * len(lines),
          [9.2] + [8.2] * len(lines), 1.3, color=color)

ax.text(LX + 0.02, 0.20, "dashed = shipped component kept for future scope (inactive in the demonstrated pipeline)",
        ha="left", va="center", fontsize=7.4, style="italic", color="#555555")

# ---------------------------------------------------------------- band D
band_label(1.58, "Data & artifacts (read / write)")
D_h = 0.98
D_y = 0.46
D = [
    ("DHCD dataset", "92,000 glyphs, 46 classes,", "Train / Test splits"),
    ("devanagari_cnn.pth", "weights + class list", "(self-describing checkpoint)"),
    ("Recognition outputs", "UTF-8 text, annotated overlay,", "verification sheet, report"),
]
for i, lines in enumerate(D):
    x = col_x(i)
    rbox(x, D_y, COL_W, D_h, fill="#F2F2F2")
    ctext(x + COL_W / 2, D_y + D_h / 2, list(lines),
          ["bold"] + ["normal"] * (len(lines) - 1),
          [9.2] + [8.2] * (len(lines) - 1), 1.32)

# ---------------------------------------------------------------- arrows
for i in range(3):
    xc = col_x(i) + COL_W / 2
    arrow_v(xc, A_y - 0.035, A_y - 0.30)            # A -> B
    arrow_v(xc, B_y - 0.035, B_y - 0.30)            # B -> C
    arrow_v(xc, C_y - 0.035, C_y - 0.30)            # C -> D

plt.tight_layout(pad=0.25)
plt.savefig("figure_architecture.png", bbox_inches="tight", facecolor="white")
plt.close()
print("architecture figure written")
