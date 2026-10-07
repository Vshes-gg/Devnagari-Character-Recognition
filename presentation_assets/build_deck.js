/* Build "Devanagari Handwritten OCR — Project Demo" deck (13 slides, 13.33x7.5). */
const pptxgen = require("pptxgenjs");

const p = new pptxgen();
p.layout = "LAYOUT_WIDE";
p.author = "Project-I Devnagari";
p.title = "Devanagari Handwritten OCR — Project Demo";

// ---- palette: ink & saffron ------------------------------------------------
const BG = "FFFFFF";      // light slide background
const DARK = "1E1A38";    // deep indigo ink (cover / closing)
const DARK2 = "2B2653";   // panel on dark
const FAINT = "353061";   // watermark glyph on dark
const PRIMARY = "3D3775"; // indigo — headers, structure
const ACCENT = "E8853D";  // saffron — the one accent
const TEXT = "23213A";
const MUTED = "6F6C8A";
const TINT = "F2F1F8";    // card tint
const LINE = "D9D7E6";    // hairline
const LIGHTTX = "C9C7E0"; // body text on dark
const GREEN = "2E9E4F", CYAN = "1E9BB5", RED = "D64545";

const LAT = "Calibri", DEV = "Nirmala UI", MONO = "Consolas";
const W = 13.33, H = 7.5, M = 0.5;

// image aspect ratios (w/h, from native px)
const AR = {
  handwriting: 1280 / 866,   // my_handwriting.jpeg
  annotated: 1200 / 1600,    // ocr_output_visual.png
  verify: 1700 / 340,        // ocr_glyph_verification.png
  sandesh: 1188 / 2113,      // sandesh_ocr_visual.png
  crumpled: 831 / 1479,      // crumpled_ocr_visual.png
  beforeAfter: 1248 / 904,   // presentation_assets/before_after.png
  segment: 1528 / 887,       // presentation_assets/segmentation_demo.png
  montage: 952 / 716,        // presentation_assets/dhcd_montage.png
};
const A = "presentation_assets/";

const shadow = () => ({ type: "outer", color: "1E1A38", blur: 7, offset: 2, angle: 90, opacity: 0.16 });
const bu = () => ({ code: "2022", indent: 10 });

// ---- helpers ---------------------------------------------------------------
function lightSlide(kicker, title, num) {
  const s = p.addSlide();
  s.background = { color: BG };
  s.addText(kicker, { x: M, y: 0.42, w: 9, h: 0.3, fontFace: LAT, fontSize: 13,
    bold: true, color: ACCENT, charSpacing: 3, margin: 0 });
  s.addText(title, { x: M, y: 0.72, w: 12.33, h: 0.7, fontFace: LAT, fontSize: 30,
    bold: true, color: TEXT, margin: 0 });
  if (num) s.addText(String(num).padStart(2, "0"), { x: W - 0.95, y: H - 0.42, w: 0.45, h: 0.3,
    fontFace: LAT, fontSize: 12, color: MUTED, align: "right", margin: 0 });
  return s;
}

// white rounded card + image inside (photo-frame motif)
function photoCard(s, img, x, y, w, h) {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: x - 0.07, y: y - 0.07, w: w + 0.14, h: h + 0.14,
    fill: { color: "FFFFFF" }, line: { color: LINE, width: 0.75 }, rectRadius: 0.06, shadow: shadow() });
  s.addImage({ path: img, x, y, w, h });
}

function hairline(s, x, y, w) {
  s.addShape(p.shapes.LINE, { x, y, w, h: 0, line: { color: LINE, width: 0.75 } });
}

// bold lead-in + muted description row (container-free grouping)
function row(s, x, y, w, head, desc, opts = {}) {
  s.addText(head, { x, y, w, h: 0.32, fontFace: LAT, fontSize: opts.headSize || 16.5,
    bold: true, color: opts.headColor || TEXT, margin: 0 });
  s.addText(desc, { x, y: y + 0.33, w, h: opts.descH || 0.55, fontFace: LAT,
    fontSize: opts.descSize || 13, color: MUTED, margin: 0 });
}

// ==============================================================================
// S1 — COVER (dark)
// ==============================================================================
{
  const s = p.addSlide();
  s.background = { color: DARK };
  s.addText("क", { x: 8.0, y: 0.4, w: 5.2, h: 6.0, fontFace: DEV, fontSize: 330,
    bold: true, color: FAINT, align: "center", valign: "middle", margin: 0 });

  s.addText("PROJECT DEMO  ·  PYTORCH + OPENCV", { x: M, y: 1.1, w: 9, h: 0.35,
    fontFace: LAT, fontSize: 14, bold: true, color: ACCENT, charSpacing: 3, margin: 0 });
  s.addText([
    { text: "Devanagari", options: { breakLine: true } },
    { text: "Handwritten OCR", options: {} },
  ], { x: M, y: 1.5, w: 9.2, h: 2.15, fontFace: LAT, fontSize: 52, bold: true,
    color: "FFFFFF", margin: 0, lineSpacing: 60 });
  s.addText("Photograph a page of handwriting — get back Devanagari Unicode text.",
    { x: M, y: 3.85, w: 7.4, h: 0.8, fontFace: LAT, fontSize: 20, color: LIGHTTX, margin: 0 });

  // glyph chips — the characters this system reads
  const glyphs = ["क", "ख", "ग", "८", "ज्ञ", "९"];
  let cx = M;
  glyphs.forEach(g => {
    const cw = g.length > 1 ? 0.85 : 0.62;
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: cx, y: 5.05, w: cw, h: 0.62,
      fill: { color: DARK2 }, rectRadius: 0.09 });
    s.addText(g, { x: cx, y: 5.05, w: cw, h: 0.62, fontFace: DEV, fontSize: 20,
      color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
    cx += cw + 0.16;
  });

  s.addText("Trained on DHCD — 92,000 handwritten glyphs, 46 classes", { x: M, y: 6.5, w: 7.5,
    h: 0.35, fontFace: LAT, fontSize: 13, color: MUTED, margin: 0 });
  s.addText("Runs on Apple Silicon (MPS) · CUDA · CPU", { x: 7.6, y: 6.5, w: 5.23, h: 0.35,
    fontFace: LAT, fontSize: 13, color: MUTED, align: "right", margin: 0 });
  s.addNotes("Hook: photograph any page of Devanagari handwriting and the engine returns the text. Everything on this deck was measured on real runs.");
}

// ==============================================================================
// S2 — THE PROBLEM
// ==============================================================================
{
  const s = lightSlide("THE PROBLEM", "Why handwritten Devanagari is hard", 2);
  const rows = [
    ["46 visually similar classes", "36 consonants + 10 digits — many pairs differ by a single stroke or dot."],
    ["The shirorekha fuses words", "The headline bar connects every character into one ink blob; it must be cut before characters can be counted."],
    ["Every writer is a new font", "Stroke width, slant and loop size change from person to person — and pen to pen."],
    ["Photos fight back", "Shadows, ruled lines, margin bars, paper folds and faint pencil all corrupt the ink."],
  ];
  let y = 1.85;
  rows.forEach((r, i) => {
    row(s, M, y, 6.9, r[0], r[1], { descH: 0.6 });
    if (i < rows.length - 1) hairline(s, M, y + 1.06, 6.9);
    y += 1.22;
  });

  const w = 4.93, h = w / AR.handwriting; // 3.34
  photoCard(s, "my_handwriting.jpeg", 7.9, 1.95, w, h);
  s.addText("Real demo input — a phone photo on shadowed pink paper", { x: 7.83, y: 1.95 + h + 0.12,
    w: w + 0.14, h: 0.35, fontFace: LAT, fontSize: 12.5, color: MUTED, align: "center", margin: 0 });
  s.addNotes("Set up the two hard parts the pipeline solves: the shirorekha (segmentation) and 46 similar classes (the CNN).");
}

// ==============================================================================
// S3 — PIPELINE
// ==============================================================================
{
  const s = lightSlide("PIPELINE", "One photograph, five stages", 3);
  const stages = [
    ["01", "Preprocess", "Shadows lifted, page deskewed, contrast normalized, ruled lines erased"],
    ["02", "Find lines", "Horizontal projection profiles on the cleaned ink binary"],
    ["03", "Find characters", "Strip the shirorekha, then vertical projection per line"],
    ["04", "Classify", "CNN softmax over 46 classes on each 32×32 crop"],
    ["05", "Assemble", "Group by row, sort left → right, write text + overlay"],
  ];
  const cw = 2.15, gap = 0.4;
  stages.forEach((st, i) => {
    const x = M + i * (cw + gap);
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y: 2.05, w: cw, h: 3.0,
      fill: { color: TINT }, rectRadius: 0.08 });
    s.addText(st[0], { x: x + 0.18, y: 2.25, w: cw - 0.36, h: 0.5, fontFace: LAT,
      fontSize: 26, bold: true, color: ACCENT, margin: 0 });
    s.addText(st[1], { x: x + 0.18, y: 2.85, w: cw - 0.36, h: 0.65, fontFace: LAT,
      fontSize: 16, bold: true, color: TEXT, margin: 0 });
    s.addText(st[2], { x: x + 0.18, y: 3.55, w: cw - 0.36, h: 1.35, fontFace: LAT,
      fontSize: 12, color: MUTED, margin: 0 });
    if (i < stages.length - 1)
      s.addText("→", { x: x + cw - 0.02, y: 3.35, w: gap + 0.04, h: 0.4, fontFace: LAT,
        fontSize: 18, bold: true, color: PRIMARY, align: "center", margin: 0 });
  });
  s.addText("One engine does it all — device auto-detected: Apple Silicon GPU (MPS), CUDA, or CPU.",
    { x: M, y: 5.65, w: 12.33, h: 0.35, fontFace: LAT, fontSize: 13.5, color: MUTED, margin: 0 });
  s.addNotes("Walk left to right. Stages 2–4 are where the interesting engineering lives; the next slides zoom into each.");
}

// ==============================================================================
// S4 — DATASET
// ==============================================================================
{
  const s = lightSlide("DATASET", "DHCD — 92,000 handwritten glyphs", 4);
  const w = 6.3, h = w / AR.montage; // 4.74
  photoCard(s, A + "dhcd_montage.png", M, 1.75, w, h);
  s.addText("One sample per class — white ink on black, 32×32 px", { x: M - 0.07, y: 1.75 + h + 0.12,
    w: w + 0.14, h: 0.3, fontFace: LAT, fontSize: 12.5, color: MUTED, align: "center", margin: 0 });

  const rx = 7.3, rw = 5.5;
  s.addText("92,000", { x: rx, y: 1.7, w: rw, h: 0.85, fontFace: LAT, fontSize: 54,
    bold: true, color: ACCENT, margin: 0 });
  s.addText("handwritten glyph images from 2,000 writers", { x: rx, y: 2.55, w: rw, h: 0.35,
    fontFace: LAT, fontSize: 14, color: MUTED, margin: 0 });

  const rows = [
    ["46 classes", "36 consonants (क … ज्ञ) + 10 digits (० … ९)", true],
    ["78,200 / 13,800", "train / held-out test — the Test folder is used once, at the very end", false],
    ["32×32 px", "native crop size the CNN consumes; class order baked into the checkpoint", false],
  ];
  let y = 3.25;
  rows.forEach((r, i) => {
    s.addText(r[0], { x: rx, y, w: rw, h: 0.32, fontFace: LAT, fontSize: 16.5, bold: true,
      color: TEXT, margin: 0 });
    s.addText(r[1], { x: rx, y: y + 0.33, w: rw, h: 0.55, fontFace: r[2] ? DEV : LAT,
      fontSize: 13, color: MUTED, margin: 0 });
    if (i < rows.length - 1) hairline(s, rx, y + 1.0, rw);
    y += 1.16;
  });
  s.addText("Source: UCI Machine Learning Repository — Devanagari Handwritten Character Dataset",
    { x: M, y: 6.95, w: 12.33, h: 0.3, fontFace: LAT, fontSize: 12, color: MUTED, margin: 0 });
  s.addNotes("46-way classification is the model's whole world; folder names map to Unicode via DHCD_LABEL_MAP.");
}

// ==============================================================================
// S5 — PREPROCESSING
// ==============================================================================
{
  const s = lightSlide("STAGE 1 · PREPROCESSING", "Make the photo look like training data", 5);
  const w = 6.4, h = w / AR.beforeAfter; // 4.64
  photoCard(s, A + "before_after.png", M, 1.75, w, h);
  s.addText("Phone photo  ·  after the enhancement chain", { x: M - 0.07, y: 1.75 + h + 0.12,
    w: w + 0.14, h: 0.3, fontFace: LAT, fontSize: 12.5, color: MUTED, align: "center", margin: 0 });

  const rx = 7.3, rw = 5.5;
  const rows = [
    ["Shadow removal", "each pixel divided by the estimated paper brightness — shadows lift, ink stays"],
    ["Deskew", "page tilt rotated flat via minAreaRect over all ink pixels"],
    ["CLAHE + bilateral filter", "faint strokes up, paper texture down, stroke edges stay sharp"],
    ["Ruled & margin lines erased", "morphological opening for rules; column statistics for margin bars"],
    ["Text-scale guard", "upscale until the median text row ≈ 52 px — thin strokes vanish below that"],
  ];
  let y = 1.85;
  rows.forEach((r, i) => {
    s.addText(String(i + 1).padStart(2, "0") + "  ", { x: rx, y, w: 0.45, h: 0.32, fontFace: LAT,
      fontSize: 15, bold: true, color: ACCENT, margin: 0 });
    s.addText(r[0], { x: rx + 0.5, y, w: rw - 0.5, h: 0.32, fontFace: LAT, fontSize: 16,
      bold: true, color: TEXT, margin: 0 });
    s.addText(r[1], { x: rx + 0.5, y: y + 0.33, w: rw - 0.5, h: 0.55, fontFace: LAT,
      fontSize: 12.5, color: MUTED, margin: 0 });
    if (i < rows.length - 1) hairline(s, rx, y + 0.98, rw);
    y += 1.12;
  });
  s.addNotes("Goal: make a photographed page resemble the clean DHCD rendering. Highlight divide-by-background shadow removal and the margin-line eraser.");
}

// ==============================================================================
// S6 — SEGMENTATION
// ==============================================================================
{
  const s = lightSlide("STAGE 2 & 3 · SEGMENTATION", "Find lines, then cut characters apart", 6);
  const w = 8.1, h = w / AR.segment; // 4.70
  photoCard(s, A + "segmentation_demo.png", M, 1.72, w, h);
  s.addText("Line crop with found boxes  ·  ink binary  ·  shirorekha stripped",
    { x: M - 0.07, y: 1.72 + h + 0.12, w: w + 0.14, h: 0.3, fontFace: LAT, fontSize: 12.5,
      color: MUTED, align: "center", margin: 0 });

  const rx = 8.95, rw = 3.9;
  const blocks = [
    ["1", "Line detection", "Rows of ink found by horizontal projection profiles; dust and crease valleys drop out."],
    ["2", "Binarize", "Otsu for pen ink — fold shadows stay out; adaptive thresholding for faint pencil."],
    ["3", "Cut characters", "A long horizontal kernel strips only the shirorekha. Column gaps appear → vertical projection finds each glyph; final boxes are re-measured on the original binary so the headline stays in the crop, matching DHCD framing."],
  ];
  let y = 1.78;
  blocks.forEach(b => {
    s.addText(b[0], { x: rx, y, w: 0.35, h: 0.4, fontFace: LAT, fontSize: 20, bold: true,
      color: ACCENT, margin: 0 });
    const bh = b[0] === "3" ? 2.1 : 1.0;
    s.addText(b[1], { x: rx + 0.42, y, w: rw - 0.42, h: 0.35, fontFace: LAT, fontSize: 16,
      bold: true, color: TEXT, margin: 0 });
    s.addText(b[2], { x: rx + 0.42, y: y + 0.36, w: rw - 0.42, h: bh, fontFace: LAT,
      fontSize: 12.5, color: MUTED, margin: 0 });
    y += 0.36 + bh + 0.35;
  });
  s.addNotes("The shirorekha is the reason characters can't simply be contour-counted. Boxes go back on the ORIGINAL binary so crops match how DHCD glyphs are framed.");
}

// ==============================================================================
// S7 — MODEL & TRAINING
// ==============================================================================
{
  const s = lightSlide("STAGE 4 · CLASSIFIER", "A compact CNN for 32×32 glyphs", 7);

  // architecture stack (left)
  const bx = M + 0.15, bw = 4.3, bh = 0.5, bgap = 0.14;
  const layers = [
    ["Input  1 × 32 × 32", "grayscale glyph"],
    ["Conv 32 → BN → LeakyReLU  (×2) → MaxPool", "16 × 16"],
    ["Conv 64 → BN → LeakyReLU  (×2) → MaxPool", "8 × 8"],
    ["Conv 128 → BN → LeakyReLU  (×2) → MaxPool", "4 × 4"],
    ["Flatten 2048 → Dropout 0.4", ""],
    ["FC 256 → BN → LeakyReLU", ""],
    ["FC 46 → softmax", "one class per glyph"],
  ];
  let y = 1.8;
  layers.forEach((l, i) => {
    const isHead = i === 0 || i === layers.length - 1;
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: bx, y, w: bw, h: bh,
      fill: { color: isHead ? PRIMARY : TINT }, rectRadius: 0.05 });
    s.addText(l[0], { x: bx + 0.15, y, w: bw - 0.3, h: bh, fontFace: LAT,
      fontSize: i === 0 ? 12.5 : 12, bold: isHead, color: isHead ? "FFFFFF" : TEXT,
      valign: "middle", margin: 0 });
    if (l[1]) s.addText(l[1], { x: bx + bw + 0.15, y, w: 1.7, h: bh, fontFace: LAT,
      fontSize: 11.5, color: MUTED, valign: "middle", margin: 0 });
    y += bh + bgap;
  });

  // training recipe (right)
  const rx = 7.1, rw = 5.7;
  s.addText("TRAINING RECIPE", { x: rx, y: 1.8, w: rw, h: 0.3, fontFace: LAT, fontSize: 13,
    bold: true, color: ACCENT, charSpacing: 2, margin: 0 });
  const rows = [
    ["AdamW", "lr 1e-3, weight decay 1e-4 — ReduceLROnPlateau halves the lr on plateaus"],
    ["Augmentation", "±10° rotation, ±8% translation, 0.92–1.08 scale"],
    ["Stratified 90/10 split", "validation keeps every class's ratio; Test folder never seen in training"],
    ["Best-checkpoint saving", "highest validation accuracy each epoch wins — 99.53% at epoch 13/15"],
    ["≈ 12 minutes", "15 epochs, batch 128, on Apple Silicon MPS — no data-center hardware"],
  ];
  let ry = 2.25;
  rows.forEach((r, i) => {
    s.addText(r[0], { x: rx, y: ry, w: rw, h: 0.3, fontFace: LAT, fontSize: 15.5, bold: true,
      color: TEXT, margin: 0 });
    s.addText(r[1], { x: rx, y: ry + 0.3, w: rw, h: 0.5, fontFace: LAT, fontSize: 12.5,
      color: MUTED, margin: 0 });
    if (i < rows.length - 1) hairline(s, rx, ry + 0.88, rw);
    ry += 1.02;
  });
  s.addNotes("BatchNorm after every conv, LeakyReLU(0.1) to avoid dead neurons on ink-sparse inputs, dropout 0.4 around the classifier head. A CRNN for line-level reading ships in the repo but is unused.");
}

// ==============================================================================
// S8 — RESULTS
// ==============================================================================
{
  const s = lightSlide("RESULTS", "Held-out test accuracy", 8);
  const cards = [
    ["99.36%", "Top-1 accuracy", "13,800 DHCD test images", true],
    ["99.95%", "Top-3 accuracy", "correct class in the top 3 guesses", false],
    ["99.53%", "Best validation", "epoch 13 of 15", false],
    ["12 / 46", "classes at 100%", "perfect on a quarter of all classes", false],
  ];
  const cw = 2.92, gap = 0.22;
  cards.forEach((c, i) => {
    const x = M + i * (cw + gap);
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y: 1.9, w: cw, h: 2.35,
      fill: { color: TINT }, rectRadius: 0.08 });
    s.addText(c[0], { x: x + 0.22, y: 2.2, w: cw - 0.44, h: 0.85, fontFace: LAT,
      fontSize: 40, bold: true, color: c[3] ? ACCENT : PRIMARY, margin: 0 });
    s.addText(c[1], { x: x + 0.22, y: 3.1, w: cw - 0.44, h: 0.35, fontFace: LAT,
      fontSize: 15, bold: true, color: TEXT, margin: 0 });
    s.addText(c[2], { x: x + 0.22, y: 3.48, w: cw - 0.44, h: 0.6, fontFace: LAT,
      fontSize: 12, color: MUTED, margin: 0 });
  });

  s.addText([{ text: "Hardest class — ढ (dhaa): still 97.0% correct", options: { fontFace: DEV } }],
    { x: M, y: 4.8, w: 12.33, h: 0.35, fontSize: 16.5, bold: true, color: TEXT, margin: 0 });
  s.addText("residual confusion sits in visually similar pairs",
    { x: M, y: 5.16, w: 12.0, h: 0.35, fontFace: LAT, fontSize: 12.5, color: MUTED, margin: 0 });
  hairline(s, M, 5.65, 12.33);
  s.addText("Cheap to retrain", { x: M, y: 5.85, w: 12.33, h: 0.35, fontFace: LAT, fontSize: 16.5,
    bold: true, color: TEXT, margin: 0 });
  s.addText("≈ 12 minutes end-to-end on Apple Silicon MPS — evaluate_accuracy.py reports the held-out split once, at the end",
    { x: M, y: 6.2, w: 12.33, h: 0.35, fontFace: LAT, fontSize: 12.5, color: MUTED, margin: 0 });
  s.addText("Source: evaluate_accuracy.py — single final pass on the DHCD Test split",
    { x: M, y: 6.95, w: 12.33, h: 0.3, fontFace: LAT, fontSize: 12, color: MUTED, margin: 0 });
  s.addNotes("Top-1 99.36% on 13,800 images the model never saw during training. No leakage: Test/ touched exactly once.");
}

// ==============================================================================
// S9 — DEMO
// ==============================================================================
{
  const s = lightSlide("LIVE DEMO", "One phone photo → ten glyphs, one pass", 9);
  const ih = 2.6;
  const iw1 = ih * AR.handwriting;   // 3.84
  const iw2 = ih * AR.annotated;     // 1.95
  photoCard(s, "my_handwriting.jpeg", M, 1.72, iw1, ih);
  s.addText("Input — shadowed pink paper", { x: M - 0.07, y: 1.72 + ih + 0.1, w: iw1 + 0.14, h: 0.3,
    fontFace: LAT, fontSize: 12, color: MUTED, align: "center", margin: 0 });

  // recognized text panel
  const tx = M + iw1 + 0.35, tw = 3.15;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: tx, y: 1.72, w: tw, h: ih,
    fill: { color: TINT }, rectRadius: 0.08 });
  s.addText("RECOGNIZED TEXT", { x: tx + 0.22, y: 1.9, w: tw - 0.44, h: 0.28, fontFace: LAT,
    fontSize: 11.5, bold: true, color: ACCENT, charSpacing: 2, margin: 0 });
  s.addText([
    { text: "कछफ", options: { breakLine: true } },
    { text: "खबण", options: { breakLine: true } },
    { text: "गवज", options: { breakLine: true } },
    { text: "ज्ञ", options: {} },
  ], { x: tx + 0.22, y: 2.2, w: tw - 0.44, h: 1.6, fontFace: DEV, fontSize: 21, color: TEXT,
    margin: 0, lineSpacing: 27 });
  s.addText("10 / 10 glyphs · confidence 0.99–1.00", { x: tx + 0.22, y: 3.92, w: tw - 0.44, h: 0.3,
    fontFace: LAT, fontSize: 11.5, color: MUTED, margin: 0 });

  // annotated output
  const ox = tx + tw + 0.35;
  photoCard(s, "ocr_output_visual.png", ox, 1.72, iw2, ih);
  s.addText("Output — boxes by confidence", { x: ox - 0.07, y: 1.72 + ih + 0.1, w: iw2 + 0.14, h: 0.3,
    fontFace: LAT, fontSize: 12, color: MUTED, align: "center", margin: 0 });

  // confidence legend
  const lx = ox + iw2 + 0.45, lw = W - M - lx;
  s.addText("CONFIDENCE CODE", { x: lx, y: 1.78, w: lw, h: 0.28, fontFace: LAT, fontSize: 11.5,
    bold: true, color: ACCENT, charSpacing: 2, margin: 0 });
  [["green", GREEN, "above 0.8"], ["cyan", CYAN, "0.5 – 0.8"], ["red", RED, "below 0.5"]].forEach((l, i) => {
    const ly = 2.18 + i * 0.42;
    s.addShape(p.shapes.OVAL, { x: lx, y: ly + 0.06, w: 0.16, h: 0.16, fill: { color: l[1] } });
    s.addText(l[2], { x: lx + 0.28, y: ly, w: lw - 0.28, h: 0.3, fontFace: LAT, fontSize: 12.5,
      color: TEXT, margin: 0 });
  });
  s.addText("all ten landed green", { x: lx, y: 3.55, w: lw, h: 0.55, fontFace: LAT, fontSize: 12,
    italic: true, color: MUTED, margin: 0 });

  // verification strip
  const sw = 7.7, sh = sw / AR.verify; // 1.54
  photoCard(s, "ocr_glyph_verification.png", M, 4.85, sw, sh);
  s.addText("Verification sheet", { x: 8.5, y: 4.95, w: 4.3, h: 0.3, fontFace: LAT, fontSize: 15,
    bold: true, color: TEXT, margin: 0 });
  s.addText("Each segmented crop (top row) next to real DHCD samples of its predicted class — per-glyph checking of free handwriting at a glance.",
    { x: 8.5, y: 5.28, w: 4.3, h: 1.1, fontFace: LAT, fontSize: 12.5, color: MUTED, margin: 0 });
  s.addNotes("All ten predictions at 0.99–1.00 confidence. The verification sheet makes per-glyph correctness checkable without re-running anything.");
}

// ==============================================================================
// S10 — STRESS TESTS
// ==============================================================================
{
  const s = lightSlide("ROBUSTNESS", "Stress tests — when the page is not friendly", 10);
  const ih = 3.45, iw = ih * AR.sandesh; // 1.94

  // col 1: pencil on ruled paper
  photoCard(s, "sandesh_ocr_visual.png", M, 1.8, iw, ih);
  s.addText("faint pencil, ruled paper", { x: M - 0.07, y: 1.8 + ih + 0.1, w: iw + 0.14, h: 0.3,
    fontFace: LAT, fontSize: 12, color: MUTED, align: "center", margin: 0 });
  let tx = M + iw + 0.3, tw = 3.7;
  s.addText("Faint pencil on ruled paper", { x: tx, y: 1.85, w: tw, h: 0.35, fontFace: LAT,
    fontSize: 16.5, bold: true, color: TEXT, margin: 0 });
  s.addText([
    { text: "Vertical margin bar erased by column statistics", options: { bullet: bu(), breakLine: true } },
    { text: "Adaptive binarization follows faint strokes", options: { bullet: bu(), breakLine: true } },
    { text: "Text-scale upscale before thresholding", options: { bullet: bu() } },
  ], { x: tx, y: 2.3, w: tw, h: 1.55, fontFace: LAT, fontSize: 12.5, color: MUTED,
    paraSpaceAfter: 6, margin: 0 });
  s.addText("→  All rows detected; per-character accuracy tracks stroke contrast",
    { x: tx, y: 4.15, w: tw, h: 0.75, fontFace: LAT, fontSize: 12.5, bold: true, color: PRIMARY, margin: 0 });

  // col 2: crumpled paper
  const c2x = 6.95;
  photoCard(s, "crumpled_ocr_visual.png", c2x, 1.8, iw, ih);
  s.addText("pen, crumpled paper", { x: c2x - 0.07, y: 1.8 + ih + 0.1, w: iw + 0.14, h: 0.3,
    fontFace: LAT, fontSize: 12, color: MUTED, align: "center", margin: 0 });
  tx = c2x + iw + 0.3; tw = W - M - tx;
  s.addText("Pen on crumpled paper", { x: tx, y: 1.85, w: tw, h: 0.35, fontFace: LAT,
    fontSize: 16.5, bold: true, color: TEXT, margin: 0 });
  s.addText([
    { text: "Otsu keeps fold shadows out of the ink mask", options: { bullet: bu(), breakLine: true } },
    { text: "Dust components removed before profiling", options: { bullet: bu(), breakLine: true } },
    { text: "Line profiles split at crease valleys", options: { bullet: bu() } },
  ], { x: tx, y: 2.3, w: tw, h: 1.55, fontFace: LAT, fontSize: 12.5, color: MUTED,
    paraSpaceAfter: 6, margin: 0 });
  s.addText("→  All glyphs located despite folds and creases",
    { x: tx, y: 4.15, w: tw, h: 0.4, fontFace: LAT, fontSize: 12.5, bold: true, color: PRIMARY, margin: 0 });

  s.addText("Detection stays robust; recognition accuracy follows ink contrast — pen beats pencil.",
    { x: M, y: 5.85, w: 12.33, h: 0.35, fontFace: LAT, fontSize: 13.5, color: MUTED, margin: 0 });
  s.addNotes("Two deliberately hostile pages. The binarizer is chosen per page by measured ink contrast — dark ink gets Otsu, faint pencil gets adaptive.");
}

// ==============================================================================
// S11 — CAPTURE TIPS
// ==============================================================================
{
  const s = lightSlide("FIELD NOTES", "What makes a good capture", 11);
  const tips = [
    ["१", "Pen beats pencil", "Stroke contrast must clear the paper's own texture level."],
    ["२", "Fill the frame", "Shoot close and straight-on so rows stay large and horizontal."],
    ["३", "Even light", "Avoid shadows and paper folds crossing the writing area."],
    ["४", "Leave clear gaps", "Separate characters, or words with normal spacing."],
  ];
  const cw = 2.92, gap = 0.22;
  tips.forEach((t, i) => {
    const x = M + i * (cw + gap);
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y: 2.0, w: cw, h: 2.9,
      fill: { color: TINT }, rectRadius: 0.08 });
    s.addText(t[0], { x: x + 0.22, y: 2.25, w: 1.0, h: 0.6, fontFace: DEV, fontSize: 30,
      bold: true, color: ACCENT, margin: 0 });
    s.addText(t[1], { x: x + 0.22, y: 3.0, w: cw - 0.44, h: 0.6, fontFace: LAT, fontSize: 16.5,
      bold: true, color: TEXT, margin: 0 });
    s.addText(t[2], { x: x + 0.22, y: 3.65, w: cw - 0.44, h: 1.05, fontFace: LAT, fontSize: 12.5,
      color: MUTED, margin: 0 });
  });
  s.addText("The model was trained on dark-ink felt-pen samples (DHCD) — capture quality dominates page-level accuracy.",
    { x: M, y: 5.5, w: 12.33, h: 0.35, fontFace: LAT, fontSize: 13.5, color: MUTED, margin: 0 });
  s.addNotes("Honest limits slide: the model reads what it was trained to see. Good capture habits fix most real-world failures.");
}

// ==============================================================================
// S12 — RUN IT
// ==============================================================================
{
  const s = lightSlide("TRY IT", "Three commands, start to finish", 12);
  const cmds = [
    ["$ python devanagari_ocr.py --mode train --epochs 15",
     "trains on DHCD · saves devanagari_cnn.pth (best validation epoch)"],
    ["$ python evaluate_accuracy.py",
     "Top-1 / Top-3 and per-class report on the held-out Test split"],
    ["$ python devanagari_ocr.py --mode infer --image my_handwriting.jpeg",
     "writes extracted_devanagari.txt + ocr_output_visual.png"],
  ];
  let y = 1.85;
  cmds.forEach((c, i) => {
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y, w: 7.3, h: 1.25, fill: { color: DARK },
      rectRadius: 0.07 });
    s.addText(c[0], { x: M + 0.25, y: y + 0.18, w: 6.8, h: 0.4, fontFace: MONO, fontSize: 13,
      color: "FFFFFF", margin: 0 });
    s.addText(c[1], { x: M + 0.25, y: y + 0.65, w: 6.8, h: 0.4, fontFace: LAT, fontSize: 12,
      color: LIGHTTX, margin: 0 });
    y += 1.45;
  });

  const rx = 8.3, rw = 4.5;
  s.addText("UNDER THE HOOD", { x: rx, y: 1.85, w: rw, h: 0.3, fontFace: LAT, fontSize: 13,
    bold: true, color: ACCENT, charSpacing: 2, margin: 0 });
  const rows = [
    ["PyTorch", "CNN, training loop, MPS / CUDA / CPU auto-device"],
    ["OpenCV", "preprocessing, binarization, segmentation"],
    ["NumPy · Pillow", "projection profiles, dataset loading"],
    ["Outputs", "UTF-8 text · annotated photo · glyph verification sheet"],
  ];
  let ry = 2.3;
  rows.forEach((r, i) => {
    s.addText(r[0], { x: rx, y: ry, w: rw, h: 0.3, fontFace: LAT, fontSize: 15.5, bold: true,
      color: TEXT, margin: 0 });
    s.addText(r[1], { x: rx, y: ry + 0.3, w: rw, h: 0.5, fontFace: LAT, fontSize: 12.5,
      color: MUTED, margin: 0 });
    if (i < rows.length - 1) hairline(s, rx, ry + 0.88, rw);
    ry += 1.02;
  });
  s.addText("Same code path everywhere — the device is detected at startup, never configured.",
    { x: M, y: 6.5, w: 12.33, h: 0.35, fontFace: LAT, fontSize: 13.5, color: MUTED, margin: 0 });
  s.addNotes("Offer to run inference live on any page the audience writes.");
}

// ==============================================================================
// S13 — CLOSING (dark)
// ==============================================================================
{
  const s = p.addSlide();
  s.background = { color: DARK };
  s.addText("क", { x: 8.3, y: 0.6, w: 4.9, h: 5.8, fontFace: DEV, fontSize: 300, bold: true,
    color: FAINT, align: "center", valign: "middle", margin: 0 });

  s.addText([
    { text: "धन्यवाद।", options: { fontFace: DEV, color: ACCENT, breakLine: true } },
    { text: "Thank you — questions?", options: { fontFace: LAT, color: "FFFFFF" } },
  ], { x: M, y: 1.35, w: 9, h: 2.2, fontSize: 44, bold: true, margin: 0, lineSpacing: 62 });

  s.addText("WHAT'S NEXT", { x: M, y: 4.15, w: 6, h: 0.3, fontFace: LAT, fontSize: 13, bold: true,
    color: ACCENT, charSpacing: 3, margin: 0 });
  const next = [
    ["Whole-line CRNN", "already implemented in the repo — reads a line without segmenting it"],
    ["Words & conjuncts", "matras, half-letters and compound glyphs like क्ष त्र"],
    ["Beyond the notebook", "package the engine as an app or API"],
  ];
  const cw = 3.9, gap = 0.25;
  next.forEach((n, i) => {
    const x = M + i * (cw + gap);
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y: 4.55, w: cw, h: 1.45, fill: { color: DARK2 },
      rectRadius: 0.08 });
    s.addText(n[0], { x: x + 0.22, y: 4.75, w: cw - 0.44, h: 0.32, fontFace: LAT, fontSize: 15.5,
      bold: true, color: "FFFFFF", margin: 0 });
    s.addText(n[1], { x: x + 0.22, y: 5.1, w: cw - 0.44, h: 0.75, fontFace: LAT, fontSize: 12,
      color: LIGHTTX, margin: 0 });
  });
  s.addText("Devanagari Handwritten OCR · PyTorch + OpenCV · DHCD",
    { x: M, y: 6.65, w: 12.33, h: 0.3, fontFace: LAT, fontSize: 12, color: MUTED, margin: 0 });
  s.addNotes("Close with the roadmap; invite someone to hand-write a character and run it live.");
}

p.writeFile({ fileName: "Devanagari_OCR_Demo.pptx" }).then(() => console.log("deck written"));
