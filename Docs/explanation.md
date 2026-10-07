# Devanagari OCR Engine — Code Explanation

> Block-by-block walkthrough of [`devanagari_ocr.py`](devanagari_ocr.py), the
> PyTorch OCR engine in this project (training, evaluation, and full-page
> handwriting recognition). Companion files: [`README.md`](README.md) (usage
> guide + measured results) · [`evaluate_accuracy.py`](evaluate_accuracy.py)
> (held-out test-set report).

---

## Section 0: Compute Device Selection

```
get_device()  →  CUDA  →  MPS  →  CPU   (first available wins)
```

The engine auto-detects the fastest backend. On Apple Silicon, **MPS (Metal
Performance Shaders)** runs the model on the GPU — typically 5–20× faster
than CPU for CNN training. `describe_device()` turns the selected
`torch.device` into a readable name for progress prints. Every model and
batch tensor is moved onto this one device with `.to(device)`, so training
and inference never mix backends.

`mps.is_available()` is checked defensively (`getattr(torch.backends, "mps",
None)`) so the file also imports cleanly on older torch builds without MPS.

---

## Section 1: Character Mapping & Constants

`DHCD_LABEL_MAP` is the single source of truth for the dataset's **46
classes**: 36 consonants (`क` … `ज्ञ`) + 10 digits (`०` … `९`). Each entry
maps a class index → `(folder_name, unicode_char)`.

Two reverse views are derived from it:

| View | Purpose |
|---|---|
| `FOLDER_TO_UNICODE` | dataset folder name (`character_1_ka`) → real character (`क`) — used to translate predictions into Devanagari text |
| `INDEX_TO_UNICODE` | class index → character |

The folder names must match the dataset **exactly** (`character_30_motosaw`,
`character_31_petchiryakha`, `character_32_patalosaw` for `श`, `ष`, `स`) —
a mismatch here would print raw folder names instead of characters in the
OCR output.

---

## Section 2: Model Architectures

### `DevanagariCNN` — the workhorse classifier (used by everything)

Input: `1 × 32 × 32` grayscale glyph (DHCD's native crop size).

```
Block1: [Conv 1→32 → BN → LeakyReLU] ×2 → MaxPool   → 32 × 16 × 16
Block2: [Conv 32→64 → BN → LeakyReLU] ×2 → MaxPool  → 64 × 8 × 8
Block3: [Conv 64→128 → BN → LeakyReLU] ×2 → MaxPool → 128 × 4 × 4
Flatten(2048) → Dropout(0.4) → Linear(2048→256) → BN → LeakyReLU
→ Dropout(0.4) → Linear(256→46)
```

Design choices:

- **BatchNorm after every conv** stabilizes activations and lets a single
  LR work; convs use `bias=False` since BN's own offset makes the conv bias
  redundant.
- **LeakyReLU (0.1)** instead of ReLU: keeps a small gradient for negative
  activations, avoiding dead neurons on the sparse ink-dominated inputs.
- **Two dropout layers (0.4)** around the classifier head fight
  co-adaptation on handwriting strokes.
- A wide `2048 → 256` bottleneck gives the head enough capacity to separate
  46 visually similar classes.

### `DevanagariCRNN` — optional sequence model (not used in this pipeline)

A CNN feature extractor + 2× bidirectional LSTM + CTC-style output for
reading a whole line *without* segmenting it first. The CNN collapses the
feature map to height 1, treats columns as time steps, and emits
per-step class logits (46 classes + 1 CTC blank). It ships with the engine
for future line-level recognition; the current pipeline uses the
segment-then-classify approach with `DevanagariCNN`, which trains on the
per-character DHCD data directly.

---

## Section 3: Image Preprocessing (photos → clean grayscale pages)

`full_preprocessing_pipeline()` turns a photographed page into something
close to the training data's clean rendering:

| Step | Function | Why |
|---|---|---|
| Upscale | `h < 1200 → resize` | thin strokes/matras must survive binarization |
| Shadow removal | `remove_shadows()` | phone photos have lighting gradients; see below |
| Deskew | `deskew()` | `minAreaRect` on all ink pixels → rotate page-level tilt flat |
| CLAHE | `apply_clahe()` | tile-wise contrast normalization so faint strokes show up |
| Bilateral filter | `bilateral_denoise()` | smooths paper texture/noise while keeping stroke edges sharp |
| Ruled-line erase | `remove_ruled_lines()` | horizontal rules by morphological opening; **vertical margin lines by column statistics** (faint/broken margin lines evade openings, and inpainting resurrects them from their twin) |
| Text-scale guard | end of pipeline | if the median text row is under ~52px, the page is upscaled (capped ×4) — thin faint strokes vanish at binarization when characters are ~30px tall |

**How binarization works downstream** — `_ink_binary()` chooses the
binarizer by measured ink contrast. Dark-ink pages (pen) use **global Otsu**:
strokes separate cleanly from everything, including crumpled-paper fold
shadows, which adaptive thresholding would wrongly mark as ink (every fold
is locally darker than its neighborhood). Faint-pencil pages use **adaptive
thresholding** (local mean − offset): Otsu's threshold would land between
the paper's texture peaks and the strokes, so texture would binarize as ink
while faint characters drop out. The two regimes are told apart by the
darkness gap between Otsu's ink class and the rest of the page.

**How `remove_shadows()` works** — illumination flattening by
divide-by-background: a large dilation + Gaussian blur estimates the
*background* brightness (paper) at every pixel, then the page is divided by
it (`ink/paper × 255`). A uniformly-lit page comes out unchanged; a shadowed
page comes out with the shadow lifted, keeping only ink darkness.

---

## Section 4: Segmentation (pages → lines → characters)

### `extract_line_crops()` — find text lines

Line detection works on the **dust-cleaned ink binary** (see
`_ink_binary` in Section 3) via **horizontal projection profiles**:

1. **Dust removal**: connected components smaller than page_height/60 are
   erased first — fold-shadow dashes on crumpled paper and paper specks.
   Left in place, the line-merging step would smear a single speck into a
   full-width phantom text line.
2. **Row bands**: contiguous runs of rows with any ink (tiny gaps bridged).
3. **Crease-valley splitting**: within each band, rows far below the band's
   peak ink (a dash contributes 2–10px per row; a row through real glyphs
   carries the sum of all stroke widths it crosses) act as gaps — merged
   rows split apart and crease-only bands drop out.
4. Each line's box spans the full x-extent of its band's ink; dust-height
   bands are dropped. One band per written row remains, however far apart
   the characters are spaced.

### `remove_shirorekha()` — cut the head-stroke

Devanagari characters hang from a horizontal top bar (the *shirorekha*),
which **connects every character of a word into one contour**. The function
openes the binary line with a wide horizontal kernel
(`max(15, 4% of line width) × 1`) — only a stroke as long as the shirorekha
survives — and subtracts that mask from the line. Character boundaries now
appear as column gaps.

### `segment_characters_from_line()` — find individual characters

Connected components on the stripped binary fragment handwriting badly
(subtracting the top bar disconnects upper loops and matras from the body,
and classifying those fragments alone destroys accuracy). Instead,
characters are located by **vertical projection** on the stripped binary:

1. Sum ink per column; contiguous column runs are candidate characters
   (glyph-internal 3–10px gaps left by the strip are bridged).
2. Each character's final box is the **union bounding box of all ink inside
   its column range, measured on the ORIGINAL binary** — so the shirorekha
   and upper-zone matras stay inside the crop, exactly matching how DHCD
   training glyphs are framed.
3. Parts of one glyph separated by the strip (a vertical whose only link to
   the body ran through the head-stroke) sit a few px apart, while distinct
   characters sit far wider — boxes whose horizontal gap is below ~25% of
   the line's median glyph height are unioned back together.
4. Dust filter: boxes far shorter than the median glyph height are
   binarization noise, not characters. Stroke slivers (much narrower than
   both their own height and the line's widest glyph — a vertical whose
   glyph body was separated) are dropped too: classifying them yields junk
   bars and digits.

Boxes are padded 2px and sorted left-to-right.

---

## Section 5: Dataset & Training

### `DevanagariDataset`

Walks a class-folder hierarchy (`character_1_ka/ … digit_9/`), collects
every image path + label, and loads each image as grayscale
(`PIL → 'L'`). Classes are **sorted alphabetically**, so class index `i` is
deterministic given the folder names — the same ordering is baked into the
checkpoint at save time.

### `train_devanagari_model()`

1. **Validation split** — if the dataset has a dedicated `val/` folder it is
   used; otherwise a **stratified 10% hold-out** is carved from the training
   set (`train_test_split(..., stratify=targets)`), keeping every class's
   ratio intact. Two dataset *views* over the same files are built so the
   validation half never receives training augmentations. The dataset's
   `Test/` split is never touched during training — it is reserved for
   `evaluate_accuracy.py`.
2. **Augmentation** (`train_transform`) — `RandomRotation(±10°)`,
   `RandomAffine` translate ±8% / scale 0.92–1.08, then ToTensor +
   `Normalize(0.5, 0.5)` → `[-1, 1]`. `fill=0` matches the dataset's black
   background when rotation exposes corners. (DHCD crops are **white ink on
   black background** — the inference path inverts photo crops to match.)
3. **Optimization** — AdamW (`lr=1e-3`, `weight_decay=1e-4`) +
   `ReduceLROnPlateau` on validation loss (halves the LR after 2 stagnant
   epochs): big corrective steps early, refinement late, without hand-tuned
   decay schedules.
4. **Checkpointing** — after each epoch, if validation accuracy is a new
   best, the model saves to `devanagari_cnn.pth` containing:
   `model_state_dict`, `classes` (name list — the model outputs *indices*,
   this restores the index→character mapping), `class_to_idx`, and
   `accuracy`.

---

## Section 6: Inference Pipeline

### `DevanagariOCRRecognizer.predict_patch()` — classify one glyph

1. **Otsu binarize**: real photos leave medium-gray, textured strokes after
   enhancement, while DHCD crops are clean black/white glyphs — snapping the
   crop to pure binary removes that contrast mismatch (measured on real
   photos: predictions sharpen from ~0.5–0.9 to ~1.00 confidence).
2. **DHCD framing**: training glyphs occupy only ~80–88% of their 32×32
   frame. A tight inference crop (100% occupancy) is out of distribution, so
   the crop is padded by ~10% of its size on each side.
3. **Square-pad (letterbox)**: the crop is centered on a white square
   canvas, so a squashed resize never distorts stroke geometry.
4. **Polarity match**: `bitwise_not` inverts dark-ink-on-paper into
   **white-ink-on-black**, the exact polarity of the DHCD training crops.
   Skipping this would feed the network the photographic negative of its
   training distribution.
5. **Same tensor pipeline as training**: `Resize(32×32)` → ToTensor →
   Normalize(0.5, 0.5), then softmax → `(unicode_char, confidence)`.

### `group_into_lines()` — restore reading order

Character detections arrive segment-by-segment; this groups them into text
lines by comparing each glyph's vertical center to the **running average
center of the current line** (tolerance: 0.6 × glyph height), then sorts
each line left-to-right. Running (not global) averaging keeps slightly
sloped handwritten lines together.

### `run_pipeline()` — end to end

```
image → preprocess → line crops → per-line character boxes
      → CNN classify each crop → group into lines → write outputs
```

Outputs:

- `extracted_devanagari.txt` — recognized text, one line per row, UTF-8.
- `ocr_output_visual.png` — the original photo with a box around every
  recognized character, **color-coded by confidence**: green > 0.8,
  cyan 0.5–0.8, red < 0.5.

`min_confidence=0.10` discards near-noise detections before grouping.

---

## Section 7: CLI

```
python devanagari_ocr.py --mode train --data_dir DevanagariHandwrittenCharacterDataset \
                         --model_path devanagari_cnn.pth --epochs 15
python devanagari_ocr.py --mode infer --image my_handwriting.jpeg --model_path devanagari_cnn.pth
python evaluate_accuracy.py          # held-out Test set: Top-1/Top-3 + per-class report
```

Missing dataset/image paths now fail loudly with `[ERROR]` instead of
silently synthesizing dummy data.
