#!/usr/bin/env python3
"""Generate the pipeline diagram and Gantt chart for the demo report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

# ---------------------------------------------------------------- pipeline
stages = [
    ("Input", "photographed\npage", "#F2F2F2"),
    ("Stage 1", "Preprocessing\n(shadows,\ndeskew, lines)", "#FFFFFF"),
    ("Stage 2", "Line\nsegmentation\n(projection\nprofiles)", "#FFFFFF"),
    ("Stage 3", "Character\nsegmentation\n(shirorekha\ncut)", "#FFFFFF"),
    ("Stage 4", "CNN classify\n(46 classes)", "#FFFFFF"),
    ("Output", "Devanagari text\n+ annotated\noverlay", "#F2F2F2"),
]

fig, ax = plt.subplots(figsize=(8.6, 2.3), dpi=200)
ax.set_xlim(0, 8.6)
ax.set_ylim(0, 2.3)
ax.axis("off")

n = len(stages)
box_w, box_h = 1.26, 2.02
gap = (8.6 - n * box_w) / (n + 1)
y0 = 0.12

for i, (label, text, fill) in enumerate(stages):
    x = gap + i * (box_w + gap)
    rect = FancyBboxPatch(
        (x, y0), box_w, box_h,
        boxstyle="round,pad=0.05,rounding_size=0.10",
        linewidth=1.1, edgecolor="#1a1a1a", facecolor=fill,
    )
    ax.add_patch(rect)
    ax.text(x + box_w / 2, y0 + box_h - 0.28, label,
            ha="center", va="center", fontsize=13.5,
            fontweight="bold", color="#1a1a1a")
    ax.text(x + box_w / 2, y0 + (box_h - 0.28) / 2 - 0.10, text,
            ha="center", va="center", fontsize=9.5, color="#1a1a1a",
            linespacing=1.4)
    if i < n - 1:
        ax.add_patch(FancyArrowPatch(
            (x + box_w + 0.07, y0 + box_h / 2),
            (x + box_w + gap - 0.07, y0 + box_h / 2),
            arrowstyle="-|>", mutation_scale=12,
            linewidth=1.2, color="#1a1a1a",
        ))

plt.tight_layout(pad=0.3)
plt.savefig("figure_pipeline.png", bbox_inches="tight", facecolor="white")
plt.close()

# ---------------------------------------------------------------- gantt
phases = [
    "Literature survey and proposal",
    "Dataset acquisition and setup",
    "CNN model development and training",
    "Held-out benchmark evaluation",
    "Full-page segmentation engine",
    "Stress testing and robustness tuning",
    "Report writing and demo preparation",
]
# months June-October 2026 (0..5 on axis, May=0)
bars = [
    (0.0, 1.0),   # lit survey: Jun
    (0.5, 1.0),   # dataset: mid Jun - mid Jul
    (1.0, 1.5),   # model dev + training: Jul - mid Aug
    (2.0, 1.0),   # benchmark: Aug - Sep
    (2.0, 1.5),   # segmentation engine: Aug - mid Sep
    (3.0, 1.5),   # stress tests: Sep - mid Oct
    (3.5, 1.5),   # report + demo: mid Sep - Oct
]

months = ["Jun 2026", "Jul 2026", "Aug 2026", "Sep 2026", "Oct 2026"]
fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=200)

for i, (start, dur) in enumerate(bars):
    y = len(phases) - 1 - i
    ax.barh(y, dur, left=start, height=0.52,
            color="#4a4a4a" if i % 2 == 0 else "#8c8c8c",
            edgecolor="#1a1a1a", linewidth=0.7)

ax.set_yticks(range(len(phases)))
ax.set_yticklabels(list(reversed(phases)), fontsize=10.5)
ax.set_xticks([i + 0.5 for i in range(len(months))])
ax.set_xticklabels(months, fontsize=10)
ax.set_xlim(0, 5)
ax.set_ylim(-0.6, len(phases) - 0.4)
ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#bbbbbb", alpha=0.7)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.set_xlabel("Project timeline (Project I, fourth year \u2013 first part)", fontsize=10.5)

plt.tight_layout(pad=0.4)
plt.savefig("figure_gantt.png", bbox_inches="tight", facecolor="white")
plt.close()

print("figures written")
