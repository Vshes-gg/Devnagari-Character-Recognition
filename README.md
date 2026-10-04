# Devanagari Handwritten OCR — PyTorch Engine

Full-page OCR for handwritten Devanagari: photograph a page → the engine
enhances it, segments lines and characters, classifies every glyph with a
CNN trained on the [DHCD](#dataset) (46 classes = 36 characters + 10 digits),
and reconstructs the text in Devanagari Unicode.

Code: [`devanagari_ocr.py`](devanagari_ocr.py) (engine: preprocessing,
segmentation, model, training, inference) ·
[`evaluate_accuracy.py`](evaluate_accuracy.py) (held-out test-set report) ·
[`explanation.md`](explanation.md) (block-by-block walkthrough).

**Runs on Apple Silicon GPU (MPS), CUDA, or CPU** — auto-detected, no
configuration needed.

---

## Measured Results (verified run on Apple Silicon / MPS)

| Metric | Value |
|---|---|
| Validation accuracy (best checkpoint, epoch 13/15) | **99.53%** |
| Test accuracy — Top-1 (13,800 held-out images) | **99.36%** |
| Test accuracy — Top-3 | **99.95%** |
| Classes at 100% test accuracy | 12 of 46 |
| Weakest class | ढ `character_14_dhaa` — 97.00% |
| Training time (15 epochs, batch 128, MPS) | ~12 minutes |
| Handwriting demo (real phone photo, spaced grid) | 10/10 glyphs located & recognized in reading order |

The model was trained on a stratified 90/10 split of the DHCD `Train/`
folder (70,380 / 7,820 images). The 13,800-image `Test/` split was never
touched during training — `evaluate_accuracy.py` reports on it once, at the
end.

### Handwriting demo

Input: [`my_handwriting.jpeg`](my_handwriting.jpeg) (phone photo, shadowed
pink paper, characters written in a spaced grid).

Recognized output (`extracted_devanagari.txt`):

```
कछफ
खबण
गवज
ज्ञ
```

- `ocr_output_visual.png` — the photo with a box around every recognized
  character, color-coded by confidence (green > 0.8, cyan 0.5–0.8, red < 0.5).
- `ocr_glyph_verification.png` — each segmented crop (top row) next to real
  DHCD samples of its predicted class (bottom row), for per-glyph checking.

All 10 glyph predictions came back at 0.99–1.00 confidence. Per-glyph
correctness on free handwriting depends on how closely the writing style
resembles DHCD — use the verification sheet to check each glyph.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dataset layout expected (see [Dataset](#dataset)):

```
DevanagariHandwrittenCharacterDataset/
├── Train/  character_1_ka/ ... digit_9/     (78,200 images)
└── Test/   character_1_ka/ ... digit_9/     (13,800 images)
```

## Usage

```bash
# 1. Train (auto-uses MPS on Apple Silicon; saves devanagari_cnn.pth)
python devanagari_ocr.py --mode train --epochs 15

# 2. Final accuracy on the held-out Test set
python evaluate_accuracy.py

# 3. Read a page of handwriting
python devanagari_ocr.py --mode infer --image my_handwriting.jpeg
```

Outputs of `infer`: `extracted_devanagari.txt` (UTF-8 Devanagari text) and
`ocr_output_visual.png` (annotated photo).

Useful flags: `--data_dir` (dataset root), `--model_path`
(checkpoint to save/load), `--epochs`.

---

## How it works

```
photo ─► preprocessing ─► line segmentation ─► character segmentation
      ─► CNN classify (32×32 crops) ─► reading-order assembly ─► text + overlay
```

1. **Preprocess** — upscale small photos, lift shadows
   (divide-by-background), deskew, CLAHE, bilateral denoise, erase ruled
   lines.
2. **Segment lines** — Otsu binarize, horizontal dilation with a
   page-width-adaptive kernel, contour boxes.
3. **Segment characters** — shirorekha (head-stroke) removal, then vertical
   projection; glyph fragments separated by the strip are re-unioned by
   glyph-scale proximity, and union boxes are measured on the original
   binary so the shirorekha stays in the crop (like DHCD framing).
4. **Classify** — each crop is binarized, padded to DHCD's ~85% glyph
   framing, inverted to the training polarity, and pushed through the CNN.
5. **Assemble** — detections are grouped into lines by y-center and sorted
   left-to-right.

Full WHY/HOW walkthrough with design rationale:
[`explanation.md`](explanation.md).

---

## Dataset

[Devanagari Handwritten Character Dataset (DHCD)](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset)
— 92,000 handwritten 32×32 grayscale glyphs (white ink on black), 46
classes: 36 consonants (`क` … `ज्ञ`) + 10 digits (`०` … `९`), 2,000 writers'
samples split 78,200 train / 13,800 test.

Class folder names map to Unicode in `DHCD_LABEL_MAP`
(`character_1_ka → क`, `character_30_motosaw → श`, `digit_0 → ०`, …).

## Project structure

```
devanagari_ocr.py                  # engine: preprocessing, segmentation, CNN, training, inference
evaluate_accuracy.py               # held-out Test-set report (Top-1/Top-3, per-class)
devanagari_cnn.pth                 # trained checkpoint (best validation epoch)
my_handwriting.jpeg                # demo input
extracted_devanagari.txt           # demo output: recognized text
ocr_output_visual.png              # demo output: annotated photo
ocr_glyph_verification.png         # demo output: per-glyph crop vs DHCD samples
requirements.txt                   # dependencies (pinned versions)
explanation.md                     # block-by-block code walkthrough
DevanagariHandwrittenCharacterDataset/   # DHCD data (Train/ + Test/)
checkpoints/                       # archived model from an earlier project iteration
```
