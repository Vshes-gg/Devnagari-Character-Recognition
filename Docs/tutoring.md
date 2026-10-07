# Tutoring: How Your Devanagari OCR Engine Works

> A teacher's walkthrough of [`devanagari_ocr.py`](devanagari_ocr.py) and
> [`evaluate_accuracy.py`](evaluate_accuracy.py), written for someone starting
> from zero. Every section builds intuition first, then explains the logic,
> then tells you *why* the code does it that way. Read it top to bottom the
> first time; after that, jump to whichever section you're weak on.
>
> Companion files: [`README.md`](README.md) (usage + results) ·
> [`explanation.md`](explanation.md) (design-rationale walkthrough — read this
> tutoring file first, `explanation.md` will then make much more sense).

---

## Part 0 — How to read this document

This document and `explanation.md` cover the same code, but for different readers:

| | `tutoring.md` (this file) | `explanation.md` |
|---|---|---|
| Assumes you know | almost nothing | CNNs, OpenCV basics |
| Teaches | the concepts, then the code | the design decisions |
| Voice | teacher → student | engineer → engineer |

Each technical section ends with a **✅ Check yourself** question. If you can't
answer it, re-read that section before your demo — these are exactly the
questions a judge asks.

---

## Part 1 — The big picture

Your program solves this problem:

> **Input:** a photograph of a page of handwritten Devanagari.
> **Output:** the same text as editable Unicode characters, in a `.txt` file,
> plus an annotated image showing where each character was found.

It cannot just "look at the photo and read it" the way you do. It has to break
one hard problem into five small ones, solved in a chain. Each stage hands its
result to the next:

```
photo ─► ① PREPROCESS      make the photo look like clean training data
      ─► ② FIND LINES       split the page into horizontal text rows
      ─► ③ FIND CHARACTERS  split each row into individual letter crops
      ─► ④ CLASSIFY         ask a CNN: "which of 46 characters is this?"
      ─► ⑤ ASSEMBLE         put the characters back in reading order → text
```

Two things to hold in your head from the start:

1. **The classifier (stage ④) is the only "AI" part.** Everything before it is
   classical image processing (OpenCV), and everything after it is bookkeeping.
   The AI stage only works if stages ①–③ deliver it a clean, well-framed crop
   of exactly one character.
2. **The classifier was trained on a specific look.** The DHCD dataset contains
   clean 32×32 pixels, white ink on a pure black background, one centered
   character per image. A phone photo is the *opposite* of that. So the entire
   job of preprocessing is: **make the photo look like the training data.**
   Almost every strange-looking trick in this codebase exists to close that gap.

---

## Part 2 — Foundations you need first

### 2.1 Images as numbers

A **grayscale image** is a grid of pixels, each a number from 0 (black) to 255
(white). A 32×32 image is 1,024 numbers. **Binarization** means collapsing
those 256 possible values down to two: every pixel becomes *ink* or *paper*,
decided by a threshold.

### 2.2 What a neural network actually is

A neural network is a function with millions of adjustable knobs (called
**weights** or **parameters**). You feed in numbers (pixels), it multiplies and
adds them in learned patterns, and it outputs scores — one score per class.
"Training" means showing it thousands of labeled examples and nudging every
knob a tiny bit each time so the right class gets the highest score. Nothing
more mystical than that.

### 2.3 Your 46-class universe

The classifier knows exactly **46 characters**: the 36 consonant forms
क ख ग … क्ष त्र ज्ञ and the 10 digits ०–९. Anything else on the page — Latin
letters, punctuation, independent vowels like अ — is *not a class it can
output*, so it will be forced into whichever of the 46 looks closest. Keep that
limitation in mind; it explains a lot of real-world errors.

### 2.4 Confidence and softmax

The network's raw outputs are unbounded scores ("logits"). **Softmax** squashes
them into 46 probabilities that sum to 1. The largest one is the prediction;
its value (e.g. 0.997) is the **confidence**. Confidence is what colors the
boxes in your output image: green > 0.8, cyan 0.5–0.8, red < 0.5.

---

## Part 3 — Section 0: Choosing the compute device

```python
get_device()  →  CUDA  →  MPS  →  CPU   (first available wins)
```

The same model code can run on three kinds of hardware:

- **CUDA** — an NVIDIA GPU. Fastest, but needs an NVIDIA card.
- **MPS** — Apple Silicon's GPU (Metal Performance Shaders). What your Mac used;
  roughly 5–20× faster than CPU for CNN training.
- **CPU** — always available, slowest, but fine for inference.

The logic is a priority list: use the fastest thing that exists on the machine,
fall back gracefully. Every model and batch of images is then explicitly moved
onto that one device with `.to(device)` — if the model lived on the GPU but the
images on the CPU, PyTorch would throw an error. One device, everything on it.

The MPS check is written defensively (`getattr(torch.backends, "mps", None)`)
so that on an old PyTorch build without MPS support, the file still imports
instead of crashing — a small robustness habit worth copying.

**✅ Check yourself:** Why must both the model *and* each batch of images be
moved to the same device?

---

## Part 4 — Section 1: The 46-class label map

The dataset stores images in folders named like `character_1_ka`,
`character_30_motosaw`, `digit_0`. Those are *folder names*, not characters.
The network will only ever output a **class index** (a number 0–45). Three
mappings bridge folder-name → index → real Devanagari character:

- `DHCD_LABEL_MAP`: index → (folder name, Unicode character). The single
  source of truth: `0 → ("character_1_ka", "क")`, `45 → ("digit_9", "९")`.
- `FOLDER_TO_UNICODE`: folder name → character. Used at inference: the network
  says `character_30_motosaw`, the pipeline writes `श`.
- `INDEX_TO_UNICODE`: index → character.

The class order is **alphabetical by folder name** (the dataset loader sorts
them — Part 8). That must never change, because the network's output neuron
#14 was trained to mean `character_15_adna` (ण). If the folder order changed,
every prediction would silently map to the wrong character. That's also why
the checkpoint *saves the class list inside it* — the mapping travels with the
weights.

**✅ Check yourself:** The model outputs the number 30. What has to happen
before the character श appears in the output text file?

---

## Part 5 — Section 2: The model architectures

### 5.1 Building blocks, from zero

Before the architecture, the four concepts it's built from:

**Convolution (Conv2d).** A tiny 3×3 grid of weights (a *kernel*) slides across
the image. At each position it multiplies the 9 pixels under it by its 9
weights, sums them into one output pixel. One kernel produces one **feature
map** — a map of "where in the image does this pattern appear." Early kernels
learn to fire on edges and strokes; deeper layers combine those into loops,
matras, full letter shapes. That is why the channel count *grows* as you go
deeper (32 → 64 → 128): the first layer only needs a few edge detectors, but
deeper layers need many more pattern detectors for increasingly complex shapes.

Weight math, one real example: the first conv layer is 3×3 with 1 input channel
and 32 output channels → 3 × 3 × 1 × 32 = **288 weights**. No bias term, for a
reason explained under BatchNorm.

**ReLU / LeakyReLU.** After convolving, the network needs non-linearity or it
could only learn straight-line patterns. ReLU keeps positives, zeroes
negatives. Its failure mode: if a neuron's output goes negative and stays
there, its gradient is exactly 0 and it can never recover ("dead neuron").
**LeakyReLU(0.1)** instead passes 10% of negative values, keeping a trickle of
learning signal. On sparse ink-dominated inputs (most of a glyph is
background), dead neurons are a real risk, hence LeakyReLU.

**BatchNorm (BN).** Normalizes each layer's activations to a stable range
(then rescales them with two learnable numbers per channel). Two effects: (a)
training is stable enough that *one* learning rate works for all layers; (b)
because BN has its own learnable additive offset, the conv layer's bias would
be redundant — that's why every conv here is `bias=False`. Fewer parameters,
same power.

**MaxPool.** A 2×2 window keeps only the largest value in each window. The
feature map shrinks 32×32 → 16×16 → 8×8 → 4×4. It halves the computation at
every stage and makes the network tolerant to small shifts: the pattern fires
somewhere *near* the top-left corner, and after pooling it lands in the same
output cell.

**Dropout.** During training only, a random 40% of the head's inputs are
silenced each step. The head can't over-rely on any single neuron, which
fights **overfitting** (memorizing the training writers instead of learning
general stroke shapes). At inference, dropout is off (`model.eval()`).

**Fully-connected (Linear) layer.** Every input connects to every output with
its own weight — the "vote counting" layer that turns features into class
scores.

### 5.2 `DevanagariCNN` — the actual architecture

```
Input 1×32×32 (grayscale glyph)
Block1: [Conv 1→32 → BN → LeakyReLU] ×2 → MaxPool   →  32 × 16 × 16
Block2: [Conv 32→64 → BN → LeakyReLU] ×2 → MaxPool  →  64 × 8 × 8
Block3: [Conv 64→128 → BN → LeakyReLU] ×2 → MaxPool → 128 × 4 × 4
Flatten 2048 → Dropout(0.4) → Linear 2048→256 → BN → LeakyReLU
→ Dropout(0.4) → Linear 256→46
```

Read the shape story: the image *shrinks* (32→16→8→4) while the number of
*channels grows* (1→32→64→128). Early layers see a big image but detect simple
things; late layers see a tiny 4×4 grid but each cell summarizes a large patch
of the original glyph. After the last pool, the tensor holds 128 × 4 × 4 =
**2,048 numbers** — `Flatten` unrolls the grid into one long vector so the
fully-connected head can vote with all of it. The 2048→256 bottleneck compresses
evidence into a compact summary, and the final 256→46 layer produces one score
per class.

**Where the 823,758 parameters come from** (a table worth memorizing — it's a
classic judge question):

| Layer | Arithmetic | Parameters |
|---|---|---|
| Conv 1→32 (3×3, no bias) | 3·3·1·32 | 288 |
| + BN(32) | 2·32 | 64 |
| Conv 32→32 | 3·3·32·32 | 9,216 |
| + BN(32) | | 64 |
| Conv 32→64 | 3·3·32·64 | 18,432 |
| + BN(64) | | 128 |
| Conv 64→64 | 3·3·64·64 | 36,864 |
| + BN(64) | | 128 |
| Conv 64→128 | 3·3·64·128 | 73,728 |
| + BN(128) | | 256 |
| Conv 128→128 | 3·3·128·128 | 147,456 |
| + BN(128) | | 256 |
| Linear 2048→256 | 2048·256 + 256 | 524,544 |
| + BN(256) | | 512 |
| Linear 256→46 | 256·46 + 46 | 11,822 |
| **Total** | | **823,758** |

Note that the fully-connected head alone (524,544) is ~64% of the whole
network — that's typical for CNNs and exactly why the 4×4 pooling before the
flatten matters: pooling at 8×8 would have made the head 4× bigger.

In round numbers, to say out loud in a demo: **"0.8 million parameters, a
3.2 MB file, runs offline on a laptop."**

### 5.3 `DevanagariCRNN` — the second architecture (unused!)

The file also contains a CRNN: a CNN feature extractor whose output is
collapsed to height 1, so each *column* of the feature map becomes a time step;
two bidirectional LSTMs read that sequence left-to-right *and* right-to-left;
a final layer outputs per-column class scores plus one extra "CTC blank" class
(47 outputs). This is the classic architecture for reading a whole text line
*without ever cutting it into characters* — CTC is the training trick that lets
the network learn the alignment between input columns and output characters by
itself.

**Why isn't it used?** Because DHCD is labeled *per character* (2,000 images
per class). The segment-then-classify approach trains directly on exactly that
data. A CRNN needs line-level transcriptions ("this whole strip says कछफ"),
which DHCD doesn't provide. So the CRNN is shipped for future work, and the
CNN does the real job. If a judge spots the dead code, that's your answer —
it's a deliberate scoping decision, not an accident.

**✅ Check yourself:** Why do the conv layers use `bias=False`, and why does
the channel count grow while the image shrinks?

---

## Part 6 — Section 3: Preprocessing (photo → clean page)

`full_preprocessing_pipeline()` runs a fixed chain of OpenCV steps. For each
one: what it does, the logic inside, and why.

### 6.1 Upscale small photos (h < 1200 → resize to 1200)

Thin strokes and matras are 1–2 px wide in a small photo. Binarization and
morphology destroy features that thin, so everything gets enlarged first.

### 6.2 `remove_shadows()` — divide-by-background

Phone photos have lighting gradients: one corner bright, one in shadow. The
logic:

1. `cv2.dilate` with a 7×7 kernel, then a 21×21 Gaussian blur. Dilation takes,
   for each neighborhood, the *brightest* pixel — so ink (dark) gets swallowed
   by surrounding paper, leaving an estimate of **what the paper brightness is
   at every location**. The blur smooths that estimate.
2. Divide the page by this background estimate, scaled by 255: `gray/bg × 255`.

Worked example. Paper at 200 brightness in sun, 160 in shadow; ink 40 on both:

| pixel | raw value | ÷ background | × 255 |
|---|---|---|---|
| sunlit paper | 200 | 200/200 | 255 |
| shadowed paper | 160 | 160/160 | 255 |
| sunlit ink | 40 | 40/200 | 51 |
| shadowed ink | 40 | 40/160 | 64 |

Every pixel is divided by its *local* paper level, so paper becomes uniformly
white everywhere and the shadow disappears — while ink stays dark *relative to
its own local paper*. The illumination gradient is gone; the ink contrast
survives.

### 6.3 `deskew()`

If the page was photographed at a slight angle, every text line is tilted, and
row-based line detection would blur adjacent lines together. Logic:

1. Binarize (Otsu, ink-white) and collect the coordinates of *all* ink pixels.
2. `cv2.minAreaRect` finds the minimum rotated rectangle around that cloud of
   points — its angle is the page tilt.
3. If the tilt is meaningful (≥ 0.5°), rotate the page back to horizontal.
   (OpenCV's angle convention flips at −45°, hence the `if angle < -45:
   angle += 90` correction — a famous OpenCV quirk.)

### 6.4 `apply_clahe()`

CLAHE = Contrast Limited Adaptive Histogram Equalization. Instead of one global
contrast stretch for the whole page, it splits the image into 8×8 tiles and
stretches contrast *within each tile* (clipping the amplification so noise
doesn't explode). Effect: faint pencil strokes in a low-contrast corner get the
same boost as strong ink elsewhere. It's "local contrast repair."

### 6.5 `bilateral_denoise()`

A Gaussian blur smooths noise but also blurs stroke edges — terrible when the
next step needs crisp strokes. A **bilateral filter** averages only neighboring
pixels whose *values* are similar, so flat paper texture gets smoothed while
the sharp jump at a stroke edge is preserved. `d=9, sigmaColor=75, sigmaSpace=75`
= a fairly strong smoothing that still respects edges.

### 6.6 `remove_ruled_lines()`

Notebook paper has horizontal rules and a vertical margin line, both of which
are long, straight, full-width strokes — perfect bait for line detection.

- **Horizontal rules:** a morphological **opening** with a very wide kernel
  (`max(30, width/4) × 1`). Opening = erode, then dilate. Erosion with a wide
  horizontal kernel destroys anything *thinner than the kernel*, so only strokes
  at least `width/4` long survive; dilation then restores their shape. Text
  strokes are far shorter than a quarter of the page width, so only the rules
  survive the opening — that mask is the rules.
- **Vertical margin line:** openings can't catch it reliably (a faint, broken
  vertical line has segments too short to survive the opening, and inpainting
  the caught segments would resurrect ink from the broken gaps). So the code
  uses **column statistics** on the binarized page instead: for each column, it
  measures `span` (fraction of page height between the first and last ink row)
  and `fill` (fraction of pixels in the column that are ink). A real text column
  is sparse — a few percent fill even on a full page. A margin line has span
  > 33% of the page **and** fill > 15%. Columns meeting both are erased.
- All matched line pixels are painted white and `cv2.inpaint` (TELEA) fills the
  erased strips by blending surrounding paper texture, so characters the line
  crossed get repaired rather than amputated.

### 6.7 Text-scale guard (end of pipeline)

A page can be large while its *text* is small (e.g. a 1280-px photo with 30-px
pencil characters). The pipeline measures the median height of significant text
rows (using the line detector from Part 7 — significant means ≥ page height/40,
so fold shadows and specks don't poison the median) and if it's under **52 px**,
upscales the page (capped at 4×). Why 52? Below roughly that size, thin faint
strokes don't survive binarization — this is a measured safety threshold, and
it's *relative to the text*, not the photo.

### 6.8 `_ink_binary()` — choosing the binarizer

Every downstream step works on a pure black/white "ink mask." Two binarizers
are available, and each fails on the other's regime:

- **Global Otsu** picks one threshold for the whole page by finding the valley
  in the brightness histogram between the paper peak and the ink peak. Perfect
  for dark pen ink. Fails on faint pencil: the threshold lands between the
  paper's texture peaks and the strokes, so *texture binarizes as ink* while
  the actual characters drop out.
- **Adaptive threshold** computes a local mean in a 51×51 window per pixel and
  marks pixels darker than (local mean − 12). It follows the local paper level,
  so faint strokes survive. Fails on crumpled paper: every fold shadow is
  locally darker than its neighborhood, so adaptive marks *the folds* as ink.

The chooser measures which regime you're in: it computes Otsu's threshold, then
the darkness gap between Otsu's ink class and the rest of the page. **Gap ≥ 50
→ strong dark ink → Otsu.** Otherwise → faint pencil → adaptive. One rule,
automatic, no user flag.

**✅ Check yourself:** Why would adaptive thresholding mark a fold shadow as
ink, and why doesn't Otsu have that problem?

---

## Part 7 — Section 4: Segmentation (page → lines → characters)

### 7.1 `extract_line_crops()` — horizontal projection profiles

A **projection profile** is the simplest line-finder there is: count the ink
pixels in each row of the binary image and plot the counts. Rows through a
text line have large counts; rows between lines have zero. Contiguous runs of
inked rows are text lines. The code adds four refinements:

1. **Dust removal first.** Connected-components analysis finds every isolated
   blob (a blob = a group of touching ink pixels); any blob smaller than
   page_height/60 in either dimension is erased *on the page-level binary*.
   Left in place, a single fold-shadow dash would later smear into a
   full-width phantom "line." (Upper matras are safe: character segmentation
   re-binarizes each line crop fresh, without this filter.)
2. **Band formation with bridging.** Rows with any ink form bands; gaps of a
   few rows *within* a word are bridged (bridge ≤ max(3, H/200) rows), so one
   line isn't split by a small between-word gap in its own row band.
3. **Crease-valley splitting.** A crumpled-paper fold crossing a band carries
   only 2–10 ink pixels per row (a dash is narrow), while a row through real
   glyphs carries the *sum* of all stroke widths it crosses. So inside each
   band, rows whose ink count falls below 8% of the band's peak are treated as
   gaps — bands split there, and crease-only bands drop out entirely. This is
   how the engine reads text on crumpled paper without hallucinating lines from
   the creases.
4. **Line boxes.** Each surviving band becomes a box spanning the full
   x-extent of its ink; dust-height bands (< max(10, H/200)) are dropped.

### 7.2 `remove_shirorekha()` — cutting the head-stroke

Devanagari's defining feature: characters hang from a horizontal top bar — the
**shirorekha** — and consecutive characters *share* it, connecting a whole word
into one giant contour. Until it's cut, "find the characters" is impossible.

Logic: morphological opening with a wide horizontal kernel
(`max(15, 4% of line width) × 1`) — exactly the trick from ruled-line removal.
Only a stroke *long* enough survives the opening, and within a text line the
only such stroke is the shirorekha itself. Subtract that mask from the line
binary: now there are vertical gaps between characters.

### 7.3 `segment_characters_from_line()` — vertical projection, done carefully

Why not just take connected components of the stripped line? Because
subtracting the shirorekha *fragments* the writing: upper loops and matras that
connected to the body only through the head-stroke become floating pieces, and
classifying those pieces alone destroys accuracy. So:

1. **Vertical projection** on the stripped binary: count ink per column;
   contiguous inked column runs are candidate characters. Glyph-internal gaps
   of ≤ 5 px (created by the strip cutting through a glyph) are bridged.
2. **Union boxes on the ORIGINAL binary.** For each candidate column range, the
   final box is the bounding box of *all* ink in that range measured on the
   line binary *before* the strip — so the shirorekha and the upper-zone matras
   are back inside the crop. This matches how DHCD training glyphs are framed,
   and it's the single most important trick in the section: the classifier was
   trained on glyphs *with* their head-stroke, so inference crops must have it
   too.
3. **Proximity re-union.** A vertical whose only link to its body ran through
   the head-stroke now sits as a separate box a few px away from its body.
   Distinct characters sit far wider apart. Rule: boxes whose horizontal gap is
   below ~25% of the line's median glyph height are unioned back together.
4. **Dust filter.** Boxes far shorter than the median glyph height are
   binarization noise (paper texture, fold remnants) → dropped.
5. **Stroke-sliver filter.** A lone separated vertical is *narrow*, and
   classifying it yields junk bars and digits. Rule: drop boxes much narrower
   than both their own height and the line's widest glyph — unless the box is
   genuinely a narrow glyph (≥ 30% of the widest glyph's width). That escape
   clause is why the digit १ survives the filter.
6. Boxes are padded 2 px and sorted left-to-right.

**✅ Check yourself:** Why are character boxes measured on the *original*
binary instead of the shirorekha-stripped one? What goes wrong with each
choice?

---

## Part 8 — Section 5: Data & training

### 8.1 `DevanagariDataset`

Walks the class-folder hierarchy, collects every image path with its label, and
loads images as grayscale (`PIL → 'L'`). Classes are `sorted()` alphabetically
— that's what makes index ↔ character deterministic (Part 4). Total: 92,000
images across 46 classes, from 2,000 different writers (78,200 in `Train/`,
13,800 in `Test/`), each 32×32, **white ink on black**.

### 8.2 The split — and the integrity rule

If the dataset has no dedicated `val/` folder, the trainer carves a
**stratified 10% hold-out** from `Train/` (`train_test_split` with
`stratify=targets`, `random_state=42`). *Stratified* means each class keeps its
proportion in both halves — with 46 classes, a naive random split could
under-represent a rare class in validation and give you a lying validation
number.

Two dataset *views* are built over the same files: the training view gets
augmentation, the validation view doesn't — otherwise augmented images would
leak into the validation metrics and flatter them.

**The Test/ folder is never opened during training.** It exists for exactly one
run of `evaluate_accuracy.py` at the very end. If you tune anything using the
test set, it stops being an unbiased estimate — that's the single most common
integrity failure in student ML projects, and your project explicitly avoids
it. Say this in the demo.

### 8.3 Augmentation

Every training image is randomly perturbed before being shown:

- `RandomRotation(±10°, fill=0)` — writers tilt; the network must tolerate it.
  `fill=0` paints exposed corners black, matching the dataset's black
  background (wrong fill would add fake white corners = fake ink).
- `RandomAffine(translate ±8%, scale 0.92–1.08)` — writers don't center their
  characters perfectly, and glyph sizes vary.

The goal is the same as preprocessing, mirrored: make the training distribution
*wider* so the messy reality of real handwriting still falls inside it.

### 8.4 The training loop, line by line

```
optimizer.zero_grad()        # forget last step's gradients
outputs = model(images)      # FORWARD: 128 images → 128×46 scores
loss = criterion(outputs, labels)   # CROSS-ENTROPY: how wrong were we?
loss.backward()              # BACKWARD: compute gradient of loss w.r.t.
                             #   every one of the 823,758 parameters
optimizer.step()             # nudge every parameter a small step downhill
```

- **CrossEntropyLoss** punishes confident wrong answers much harder than
  unconfident ones — it's the standard loss for classification.
- **AdamW** (lr=1e-3, weight_decay=1e-4) is the optimizer: per-parameter
  adaptive step sizes, plus weight decay that penalizes large weights (a mild
  anti-overfitting pressure).
- **ReduceLROnPlateau** watches validation loss; when it stalls for 2 epochs,
  the learning rate is halved. Big corrective steps early, fine adjustment
  late — without hand-crafting a decay schedule.
- One pass over the whole training set = one **epoch**. 15 epochs, batch 128,
  ~12 minutes on Apple Silicon MPS.
- **Checkpointing:** whenever validation accuracy hits a new best, the model
  saves `{model_state_dict, classes, class_to_idx, accuracy}`. Saving `classes`
  alongside the weights means the checkpoint is self-describing — load it on
  another machine and you still know what output neuron 30 means.

**✅ Check yourself:** Why must the validation view not use augmentation, and
why is touching the Test set during training a form of cheating?

---

## Part 9 — Section 6: Inference (the pipeline in reverse gear)

### 9.1 `predict_patch()` — classifying ONE crop

The classifier is picky about its input format. This function reshapes a raw
photo crop into "something DHCD-shaped" in five steps:

1. **Otsu binarize the crop.** After enhancement, photo strokes are medium-gray
   and textured; DHCD glyphs are pure black/white. Snapping to binary removes
   the contrast mismatch (measured effect on real photos: confidences sharpen
   from ~0.5–0.9 to ~1.00).
2. **Pad ~10% margin.** DHCD glyphs occupy only ~80–88% of their 32×32 frame.
   A tight crop at 100% occupancy is *out of distribution* — the network has
   never seen a glyph touching all four walls. So the crop is centered on a
   white canvas with a 10% margin on each side (this also letterboxes a
   non-square crop onto a square, so the resize to 32×32 never squashes stroke
   geometry).
3. **Invert polarity.** Photo crops are dark ink on white paper; DHCD is white
   ink on black. `bitwise_not` flips it. Skip this and you feed the network
   the photographic *negative* of its entire training set.
4. **Same tensor pipeline as training:** Resize(32×32) → ToTensor →
   Normalize(0.5, 0.5) → tensor in [-1, 1].
5. **Forward pass + softmax** → `(unicode_char, confidence)`.

Every step here exists to shrink the gap between the inference distribution and
the training distribution. That sentence *is* the answer to "why so much fuss
over one little crop?"

### 9.2 `group_into_lines()` — reading order

Segmentation already knows line positions, but the *grouping* step is done
separately from the detections: sort all character boxes by vertical center,
then walk them top-to-bottom. A character joins the current line if its
y-center is within 0.6 × glyph-height of the **running average** y-center of
that line. Running average (not a fixed global line position) is what lets
slightly sloped handwritten lines stay together — the "line" drifts as you read
along it. Each finished line is sorted left-to-right, and each line's
characters are concatenated into one string.

### 9.3 `run_pipeline()` — the end-to-end trace

This is what happens when you run `--mode infer`:

```
STEP 1  imread the photo
STEP 2  full_preprocessing_pipeline()          (Part 6)
STEP 3  extract_line_crops()                   (Part 7.1)  → "N candidate lines"
STEP 4  for each line: segment_characters_from_line()       (Part 7.3)
                for each char crop: predict_patch()         (Part 9.1)
                keep detections with confidence ≥ 0.10
STEP 5  group_into_lines()                     (Part 9.2)
STEP 6  write extracted_devanagari.txt (UTF-8, one line per text row)
        draw boxes on the photo: green > 0.8, cyan 0.5–0.8, red < 0.5
        → ocr_output_visual.png
```

`min_confidence=0.10` is deliberately permissive: near-noise detections are
dropped before grouping, but borderline characters are kept and let the
color-coding tell you where to distrust the output. The verification sheet
(`ocr_glyph_verification.png`) shows each segmented crop next to real DHCD
samples of its predicted class — the tool for checking per-glyph correctness by
eye.

**✅ Check yourself:** Name the three distribution-matching tricks in
`predict_patch` and the training-data property each one imitates.

---

## Part 10 — Section 7: The CLI

```
python devanagari_ocr.py --mode train --epochs 15          # train, save devanagari_cnn.pth
python devanagari_ocr.py --mode infer --image page.jpeg    # read a page
python evaluate_accuracy.py                                # final held-out report
```

`argparse` with `--mode`, `--image`, `--data_dir`, `--model_path`, `--epochs`.
Missing dataset or image paths exit loudly with `[ERROR]` — an earlier
iteration silently fell back to synthetic dummy data, which produces a program
that "runs" and proves nothing. Failing loudly is the correct engineering
behavior: a tool should never fake success.

---

## Part 11 — `evaluate_accuracy.py`: where 99.36% comes from

This script is the project's final exam, run **once**, on the 13,800-image
`Test/` split that training never saw:

1. Loads the checkpoint, rebuilds the model, `model.eval()` (dropout off).
2. Same no-augmentation transform as validation: Resize → ToTensor →
   Normalize.
3. For every batch, with `torch.no_grad()` (no gradients needed → faster, less
   memory):
   - **Top-1**: `torch.max(outputs, 1)` — the single highest-scoring class;
     correct if it equals the label. → **99.36%**.
   - **Top-3**: `torch.topk(outputs, 3)` — correct if the true label is among
     the three highest scores. → **99.95%**. Top-3 measures how often the right
     answer was *almost* chosen — a proxy for how useful the model would be
     with even a tiny dictionary-based correction on top.
4. Per-class accuracy: counts correct/total per class. Result: 12 of 46 classes
   at 100%; the weakest is ढ `character_14_dhaa` at 97.00% — plausible, since ढ
   differs from ध and ड by small stroke details.

**What this number is:** accuracy on clean, isolated, 32×32, DHCD-style glyph
crops from unseen *writers*. **What it is not:** page-level accuracy on real
photos. The full-page number is necessarily lower — segmentation errors, unusual
writing styles, pen/paper conditions all tax it — which is exactly why the
project includes the stress tests (faint pencil on ruled paper, crumpled paper)
and the per-glyph verification sheet. Being able to state both sides of that
boundary precisely is the difference between a naive demo and a strong one.

---

## Part 12 — Glossary (one line each)

- **Binarization** — collapsing a grayscale image to pure ink/paper using a threshold.
- **Otsu threshold** — picks the global threshold at the histogram valley between paper and ink peaks.
- **Adaptive threshold** — per-pixel threshold = local mean − offset; follows the local paper level.
- **CLAHE** — contrast stretched independently in 8×8 tiles, limited to avoid noise blow-up.
- **Bilateral filter** — smoothing that averages only similar-valued neighbors, preserving edges.
- **Dilation / erosion** — grow / shrink bright regions by a structuring kernel.
- **Morphological opening** — erode then dilate; deletes anything thinner than the kernel, keeps the rest.
- **Inpainting** — filling an erased region by blending surrounding texture.
- **Connected component** — a maximal group of touching ink pixels (a "blob").
- **Projection profile** — ink count per row (horizontal) or column (vertical); the basis of line and character segmentation.
- **Shirorekha** — the horizontal head-stroke connecting Devanagari characters of a word.
- **Matra** — vowel sign attached above/below/beside a consonant.
- **Convolution / kernel** — small sliding weight grid producing a feature map of "where does this pattern appear."
- **Feature map / channel** — one pattern-detector's response over the image.
- **BatchNorm** — keeps layer activations in a stable range; makes conv bias redundant.
- **ReLU / LeakyReLU** — non-linearities; LeakyReLU keeps 10% of negative signal to avoid dead neurons.
- **MaxPool** — 2×2 max, shrinks the map and adds small-shift tolerance.
- **Flatten** — unroll a tensor grid into one vector for the classifier head.
- **Dropout** — randomly silence 40% of head inputs during training to fight co-adaptation.
- **Softmax** — turns raw scores into 46 probabilities summing to 1.
- **Cross-entropy loss** — classification loss; punishes confident wrong answers hardest.
- **Backpropagation** — computes each parameter's blame (gradient) for the loss.
- **AdamW** — adaptive-step optimizer with decoupled weight decay.
- **ReduceLROnPlateau** — halve the learning rate when validation loss stalls.
- **Epoch / batch** — one full pass over the training set / the 128 images processed per weight update.
- **Overfitting** — memorizing training writers instead of learning general strokes.
- **Stratified split** — hold-out that preserves each class's proportion.
- **Top-1 / Top-3 accuracy** — correct class is rank 1 / within the 3 highest scores.
- **CTC blank** — the extra "no character here" class that lets a CRNN read unsegmented lines.
- **DHCD** — Devanagari Handwritten Character Dataset: 92,000 images, 46 classes, 2,000 writers, 32×32 white-on-black.

---

## Part 13 — Self-quiz (answers at the end)

Try these cold. Q1–Q8 are the demo's basic round; Q9–Q15 are the deep round.

1. What are the five stages of the pipeline, in order, and what does each hand to the next?
2. How many classes does the model know, and what are they? What happens to a Latin word photographed on the page?
3. What is the shirorekha, and what breaks if you don't remove it before character segmentation?
4. Your model has how many parameters? What file size is that, and what does it run on?
5. Why are photo crops inverted before classification?
6. What is the difference between Otsu and adaptive thresholding, and when does the code pick each?
7. What accuracy do you claim, on what data, and how many images?
8. Why is the Test/ folder off-limits during training?
9. Explain divide-by-background shadow removal with the 200/160/40 example.
10. Why is the character bounding box measured on the *original* binary rather than the shirorekha-stripped one?
11. Why does `predict_patch` pad the crop with a 10% margin?
12. Why LeakyReLU instead of ReLU, and why `bias=False` on the convolutions?
13. Why does the crease-valley rule split line bands at 8% of the band's peak ink?
14. The CRNN exists in the code — what would it do differently, and why isn't it used?
15. A judge asks: "99.36%? So if I photograph this page, I get 99.36% of the characters right?" What exactly do you say?

---

### Answers

1. Preprocess (photo → clean page) → line segmentation (page → row boxes) → character segmentation (row → glyph crops) → CNN classification (crop → 1 of 46) → reading-order assembly (detections → text + overlay).
2. 46: 36 consonant forms + 10 digits. A Latin word has no class, so each glyph is forced into the nearest-looking Devanagari class — silent misclassification, which is why the 46-class universe is a stated limitation.
3. The horizontal head-stroke connecting every character of a word into one contour. Without cutting it, vertical projection finds no gaps — the whole word is one "character."
4. 823,758 (~0.8 M); 3.2 MB on disk; runs offline on a laptop (trains in ~12 min on Apple Silicon MPS).
5. DHCD training glyphs are white ink on black; a photo crop is black ink on white. Without `bitwise_not`, the network sees the photographic negative of its training distribution.
6. Otsu = one global threshold from the histogram valley — right for dark ink, fooled by paper texture on faint pencil. Adaptive = local mean − 12 per pixel — right for faint pencil, fooled by fold shadows on crumpled paper. The chooser measures the darkness gap of Otsu's ink class: ≥ 50 → Otsu, else adaptive.
7. 99.36% top-1 (99.95% top-3) on the 13,800-image held-out DHCD Test split — isolated 32×32 glyph crops from writers unseen in training.
8. If any tuning decision used the test set, it stops measuring generalization and starts flattering you — the number would no longer be an unbiased estimate. Held-out means *untouched*, not merely "used less."
9. Paper is 200 in sun, 160 in shadow; ink is 40 on both. Dividing each pixel by its local background estimate (≈ paper level) gives 200/200 = 255 for both papers, and 40/200 ≈ 51 vs 40/160 ≈ 64 for ink — the shadow gradient vanishes while ink stays dark relative to its own local paper.
10. The original binary still contains the shirorekha and upper matras; measuring there keeps them inside the crop, matching DHCD's glyph framing. Measuring on the stripped binary would produce tight body-only crops — out of distribution for the classifier — and would orphan the matras.
11. DHCD glyphs occupy ~80–88% of their 32×32 frame; a 100%-tight crop was never seen in training. The margin also letterboxes non-square crops so the 32×32 resize never distorts stroke geometry.
12. LeakyReLU(0.1) keeps a 10% gradient on negative activations so sparse ink-dominated inputs can't create permanently dead neurons. `bias=False` because BatchNorm's learnable offset makes the conv bias redundant — free parameter savings.
13. A row through real glyphs carries the sum of all stroke widths it crosses; a fold-shadow dash contributes only 2–10 px. 8% of the band's peak separates "rows through text" from "rows through a crease," so creases split bands (or drop out) instead of welding two text lines together.
14. The CRNN reads a whole line as a sequence (CNN features collapsed to a column sequence + BiLSTM + CTC) without character segmentation. It isn't used because DHCD provides per-character labels, which the segment-then-classify CNN trains on directly; the CRNN would need line-level transcriptions. It's scoped future work, not dead weight.
15. "99.36% is the per-character accuracy on clean, isolated 32×32 crops from the benchmark's held-out writers — the standard way this task is measured. Page-level accuracy on real photographs is lower: segmentation, pen type, and paper conditions add errors on top of classification. That's why the repo includes two stress-test pages and a per-glyph verification sheet, and why page capture quality is the dominant factor." — then let the stress-test visuals make the point for you.

---

*End of tutoring document. Read it twice, then go answer Round 1, Q2–Q4.*
