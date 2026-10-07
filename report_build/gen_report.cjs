/**
 * Devanagari OCR — Project Demo Report (NIET format)
 * Format per Proposal_guidelines.docx: A4, Times New Roman, Heading 14 bold,
 * Sub-heading 12 bold, body 12, line spacing 1.5, margins T/B 1", L 1.2", R 1".
 * Cover replicates the provided "Front Page proposal.docx" sample (template mode).
 */
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, PageBreak, Header, Footer, PageNumber, NumberFormat,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType,
  TabStopType, LevelFormat, TableOfContents, SectionType, VerticalAlign,
} = require("docx");
const fs = require("fs");
const path = require("path");
const { imageSize } = require("image-size");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "Devanagari_OCR_Demo_Report.docx");

// ---------------------------------------------------------------- helpers
const TNR = "Times New Roman";
const DEV_FONT = "Mangal"; // Devanagari-capable; renderers fall back if absent

function safeText(v, ph) {
  if (v === undefined || v === null || v === "" || String(v) === "NaN" || String(v) === "undefined") {
    return ph || "\u3010Please fill in\u3011";
  }
  return String(v);
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    keepNext: true,
    spacing: { before: 400, after: 200, line: 360 },
    children: [new TextRun({ text, bold: true, size: 28, color: "000000", font: TNR })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    keepNext: true,
    spacing: { before: 280, after: 140, line: 360 },
    children: [new TextRun({ text, bold: true, size: 24, color: "000000", font: TNR })],
  });
}

/** body paragraph; segs: string | [{t, b, i}] */
function body(segs, opts = {}) {
  const runs = (typeof segs === "string" ? [{ t: segs }] : segs).map(s =>
    new TextRun({ text: s.t, bold: !!s.b, italics: !!s.i, size: opts.size || 24, color: "000000", font: TNR }));
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: opts.after !== undefined ? opts.after : 140, line: 360 },
    children: runs,
  });
}

/** centered figure with caption below */
function figure(file, displayWidth, caption) {
  const p = path.join(ROOT, file);
  const buf = fs.readFileSync(p);
  const d = imageSize(buf);
  const h = Math.round(displayWidth * d.height / d.width);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      keepNext: true,
      spacing: { before: 160, after: 60, line: 240 },
      children: [new ImageRun({ data: buf, transformation: { width: displayWidth, height: h }, type: "png" })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240, line: 280 },
      children: [new TextRun({ text: caption, size: 21, color: "000000", font: TNR })],
    }),
  ];
}

const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const CELL_MARGINS = { top: 60, bottom: 60, left: 120, right: 120 };

function cellPara(text, bold, alignment) {
  return new Paragraph({
    alignment: alignment || AlignmentType.LEFT,
    spacing: { line: 300 },
    children: [new TextRun({ text: safeText(text, "\u3010Please fill in\u3011"), bold: !!bold, size: 22, color: "000000", font: TNR })],
  });
}

/** academic three-line table with caption above */
function threeLineTable(caption, headers, rows, widths, aligns) {
  const capPara = new Paragraph({
    alignment: AlignmentType.CENTER,
    keepNext: true,
    spacing: { before: 200, after: 100, line: 280 },
    children: [new TextRun({ text: caption, size: 21, color: "000000", font: TNR })],
  });
  const headerRow = new TableRow({
    tableHeader: true,
    cantSplit: true,
    children: headers.map((t, i) => new TableCell({
      width: { size: widths[i], type: WidthType.PERCENTAGE },
      borders: { bottom: { style: BorderStyle.SINGLE, size: 2, color: "000000" }, top: NB, left: NB, right: NB },
      margins: CELL_MARGINS,
      shading: { type: ShadingType.CLEAR, fill: "FFFFFF" },
      children: [cellPara(t, true, AlignmentType.CENTER)],
    })),
  });
  const dataRows = rows.map(r => new TableRow({
    cantSplit: true,
    children: r.map((t, i) => new TableCell({
      width: { size: widths[i], type: WidthType.PERCENTAGE },
      borders: { top: NB, bottom: NB, left: NB, right: NB },
      margins: CELL_MARGINS,
      children: [cellPara(t, false, (aligns && aligns[i]) || AlignmentType.LEFT)],
    })),
  }));
  const table = new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "000000" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "000000" },
      left: NB, right: NB, insideHorizontal: NB, insideVertical: NB,
    },
    rows: [headerRow, ...dataRows],
  });
  return [capPara, table, new Paragraph({ spacing: { after: 160, line: 240 }, children: [] })];
}

function pageFooter() {
  return new Footer({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: 240 },
      children: [
        new TextRun({ text: "- ", size: 21, font: TNR }),
        new TextRun({ children: [PageNumber.CURRENT], size: 21, font: TNR }),
        new TextRun({ text: " -", size: 21, font: TNR }),
      ],
    })],
  });
}

// ---------------------------------------------------------------- cover
function coverLine(text, sizePt, bold, opts = {}) {
  const before = opts.before || 0;
  const line = Math.max(Math.ceil(sizePt * 23), 300);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before, after: opts.after || 0, line, lineRule: "atLeast" },
    children: [new TextRun({ text, bold: !!bold, size: sizePt * 2, color: "000000", font: TNR })],
  });
}

const members = [
  ["Bishes Gurung", "\u3010BCT/2020/XXX\u3011"],
  ["\u3010Member Name 2\u3011", "\u3010BCT/2020/XXX\u3011"],
  ["\u3010Member Name 3\u3011", "\u3010BCT/2020/XXX\u3011"],
  ["\u3010Member Name 4\u3011", "\u3010BCT/2020/XXX\u3011"],
  ["\u3010Member Name 5\u3011", "\u3010BCT/2020/XXX\u3011"],
  ["\u3010Member Name 6\u3011", "\u3010BCT/2020/XXX\u3011"],
];

function buildCover() {
  const out = [];
  out.push(coverLine("Purbanchal University", 16, true, { before: 500, after: 60 }));
  out.push(coverLine("Faculty of Engineering", 16, true, { after: 500 }));
  out.push(coverLine("National Institute of Engineering and Technology", 16, true, { before: 500, after: 60 }));
  out.push(coverLine("Lalitpur, Nepal", 14, true, {}));
  out.push(coverLine("A Project Report On", 14, false, { before: 2100, after: 300 }));
  // Title — 3 semantic lines, 18pt bold (Rule 8: explicit line spacing)
  out.push(coverLine("Devanagari Handwritten Character Recognition", 18, true, { after: 40 }));
  out.push(coverLine("and Full-Page OCR", 18, true, { after: 40 }));
  out.push(coverLine("Using a Convolutional Neural Network", 18, true, { after: 500 }));
  out.push(coverLine("By", 13, false, { after: 120 }));
  for (const [name, id] of members) {
    out.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 60, line: 340, lineRule: "atLeast" },
      children: [
        new TextRun({ text: safeText(name), size: 24, color: "000000", font: TNR }),
        new TextRun({ text: " \u2013 " + safeText(id), size: 24, color: "000000", font: TNR }),
      ],
    }));
  }
  out.push(coverLine("October 2026", 13, false, { before: 1300 }));
  return out;
}

// ---------------------------------------------------------------- abstract
function buildAbstract() {
  const paras = [];
  paras.push(h1("Abstract"));
  paras.push(body("Handwritten Devanagari text is still central to academic, administrative, and personal record keeping in Nepal, yet almost no offline tool can read a complete handwritten page. Existing optical character recognition (OCR) engines perform well on printed Devanagari but degrade sharply on handwriting, and published deep-learning results mostly report accuracy on isolated characters rather than on full pages. This project closes part of that gap: it designs, trains, and demonstrates an end-to-end OCR engine that takes an ordinary photograph of a handwritten Devanagari page and reconstructs the text as editable Devanagari Unicode."));
  paras.push(body("The system couples a robust image-processing front end with a compact convolutional neural network (CNN). Photographs are enhanced by shadow removal, deskewing, contrast-limited adaptive histogram equalization, denoising, and ruled-line removal; text lines and individual characters are then located by projection-profile analysis with shirorekha (head-stroke) removal. Each character crop is classified by a CNN with 823,758 parameters trained on the Devanagari Handwritten Character Dataset (DHCD), which covers 46 classes (36 consonant forms and 10 digits) from 2,000 writers. Training used a stratified 90/10 split with augmentation, and the official 13,800-image Test split was reserved, untouched, for a single final evaluation."));
  paras.push(body("The trained network reached 99.53% validation accuracy (best epoch of 15) and, on the held-out test set, 99.36% Top-1 and 99.95% Top-3 accuracy, with 12 of 46 classes recognized perfectly. In the live demonstration, the engine located and recognized all 10 handwritten characters of a real phone photograph in correct reading order at 0.99\u20131.00 confidence, and stress tests on faint pencil writing on ruled paper and on crumpled paper confirmed that all glyphs are still located under adverse conditions. The project met all of its objectives and delivers a working, offline, full-page handwritten Devanagari OCR demonstrator together with a clear path (sequence modeling and dictionary correction) toward deployment."));
  paras.push(new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 120, after: 140, line: 360 },
    children: [
      new TextRun({ text: "Keywords: ", bold: true, size: 24, color: "000000", font: TNR }),
      new TextRun({ text: "Devanagari OCR; handwritten character recognition; convolutional neural network; image segmentation; deep learning; PyTorch", size: 24, color: "000000", font: TNR }),
      new PageBreak(),
    ],
  }));
  return paras;
}

// ---------------------------------------------------------------- TOC page
function buildTocPage() {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 300, line: 400, lineRule: "atLeast" },
      children: [new TextRun({ text: "Table of Contents", bold: true, size: 28, color: "000000", font: TNR })],
    }),
    new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
    new Paragraph({
      spacing: { before: 200 },
      children: [new TextRun({
        text: "Note: This Table of Contents is generated via field codes. To ensure page number accuracy after editing, please right-click the TOC and select \u201cUpdate Field.\u201d",
        italics: true, size: 18, color: "888888", font: TNR,
      })],
    }),
  ];
}

// ---------------------------------------------------------------- body
function buildBody() {
  const B = [];

  // ============================================================ 1. Introduction
  B.push(h1("1. Introduction"));
  B.push(body("Devanagari is the writing system of Nepali, Hindi, Marathi, and Sanskrit and is used daily by hundreds of millions of people across South Asia. In Nepal, a large share of academic records, land and administrative files, correspondence, and personal notes exists only as handwriting on paper. Converting such pages into editable digital text would make these records searchable, archivable, and translatable, and optical character recognition (OCR) is the bridge technology for that conversion. While OCR for printed Devanagari is well developed, handwriting remains the difficult frontier: every writer shapes characters differently, paper and lighting conditions vary, and the script itself has structural properties that defeat naive recognition."));
  B.push(body([
    { t: "Earlier OCR engines such as Tesseract " },
    { t: "[1]" },
    { t: " were built for printed text and degrade sharply when characters are handwritten. Devanagari handwriting recognition was therefore studied for years with hand-crafted features and classical classifiers, reaching moderate accuracy on small datasets " },
    { t: "[2]" },
    { t: ". The arrival of deep convolutional neural networks (CNNs), pioneered for document reading by LeCun et al. " },
    { t: "[3]" },
    { t: " and popularized by ImageNet-scale vision " },
    { t: "[4]" },
    { t: ", changed the accuracy landscape. For Devanagari specifically, Acharya et al. published the Devanagari Handwritten Character Dataset (DHCD) " },
    { t: "[5]" },
    { t: ", " },
    { t: "[6]" },
    { t: " \u2014 92,000 labeled glyphs from 2,000 writers \u2014 and demonstrated CNN accuracy of roughly 98.5% on isolated characters. However, such results measure one cropped character at a time; reading a whole photographed page additionally requires enhancement, line finding, character separation, and reading-order reconstruction, which is exactly the gap this project addresses." },
  ]));
  B.push(body("This project therefore builds a complete, offline, full-page OCR engine for handwritten Devanagari: a photograph of a page is enhanced, segmented into lines and characters, classified glyph-by-glyph by a CNN trained on DHCD, and reassembled into Devanagari Unicode text together with an annotated, confidence-coded image of the original page. This report documents the completed work for the project demonstration: it states the problem and objectives, reviews the relevant literature, establishes feasibility, explains the methodology in detail, presents the evaluation results obtained, and closes with outcomes, schedule, cost, and references."));

  // ============================================================ 2. Problem statement
  B.push(h1("2. Statement of the Problem"));
  B.push(body("The need addressed by this project is practical: no commonly available, free, offline tool today reads a photographed page of handwritten Devanagari. General-purpose OCR systems handle printed Devanagari, and research systems report high accuracy on isolated handwritten characters, but a user holding a phone photograph of a handwritten page has no tool that returns the page\u2019s text. The reason is structural: Devanagari characters hang from a shared horizontal head-stroke (the shirorekha) that connects every character of a word into a single contour, vowel signs (matras) attach above and below the headline, many of the 46 character forms are visually near-identical at small stroke differences, and real photographs add skew, shadows, ruled lines, and paper texture on top. Recognition therefore fails unless the whole pipeline \u2014 not just the classifier \u2014 is engineered for these conditions."));
  B.push(body("The scope of the present project is a 46-class recognition universe (the 36 consonant forms \u0915\u2026\u091c\u094d\u091e and the 10 digits \u0966\u2013\u096f), offline operation on a laptop, and single-page photographs taken with a smartphone under three deliberately adverse conditions: clean pen writing on plain paper, faint pencil on ruled notebook paper, and pen on crumpled paper. Independent vowels, conjunct consonants beyond the 46-class set, punctuation, and embedded Latin text are outside the current scope and are treated as future work."));

  // ============================================================ 3. Objectives
  B.push(h1("3. Objectives"));
  B.push(h2("3.1 General Objective"));
  B.push(body("The general objective of this project is to develop an end-to-end, offline OCR system that reads full pages of handwritten Devanagari from ordinary smartphone photographs and reconstructs them as editable Devanagari Unicode text, using a convolutional neural network trained on the Devanagari Handwritten Character Dataset. The long-term goal of our study is a reliable digitization path for handwritten Nepali documents; within Project I, the research goal is to determine how accurately a segment-then-classify pipeline can recognize Devanagari handwriting under realistic capture conditions."));
  B.push(h2("3.2 Specific Objectives"));
  B.push(body("The specific objectives are:", { after: 80 }));
  const objs = [
    "To implement an image-preprocessing pipeline that converts phone photographs with uneven lighting, skew, ruled lines, and paper folds into clean binarized pages, verified visually on the three target paper conditions.",
    "To implement robust line and character segmentation based on projection profiles and shirorekha removal, such that handwritten rows are located and characters are cropped with their head-strokes, in correct reading order.",
    "To train a 46-class CNN on DHCD and reach at least 98% Top-1 accuracy on the untouched 13,800-image test split.",
    "To evaluate the system under a strict held-out protocol (Top-1, Top-3, and per-class accuracy) and to demonstrate full-page OCR on real handwriting with a per-glyph verification sheet.",
    "To assemble the recognized characters into reading order and export UTF-8 Devanagari text together with a confidence-annotated overlay of the original photograph.",
  ];
  objs.forEach(t => B.push(new Paragraph({
    numbering: { reference: "obj-list", level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 80, line: 360 },
    children: [new TextRun({ text: t, size: 24, color: "000000", font: TNR })],
  })));

  // ============================================================ 4. Literature review
  B.push(h1("4. Literature Review"));
  B.push(body([
    { t: "Early work on Devanagari handwriting recognition relied on hand-crafted features (moments, stroke geometry, shadow coding) fed to classical classifiers such as k-nearest neighbors and support vector machines. Pal, Wakabayashi, and Kimura " },
    { t: "[2]" },
    { t: " compared such combinations on Devanagari characters and numerals and reported accuracies in the 80\u201390% range on their datasets \u2014 respectable at the time, but fragile across writers because hand-designed features capture only the aspects of glyph shape their authors anticipated." },
  ]));
  B.push(body([
    { t: "Deep learning displaced that paradigm. LeNet established convolutional networks as the tool for character reading " },
    { t: "[3]" },
    { t: ", and AlexNet demonstrated the same architecture family scaling to natural-image recognition " },
    { t: "[4]" },
    { t: ". For Devanagari, Acharya et al. " },
    { t: "[5]" },
    { t: " contributed both a large benchmark \u2014 the DHCD, 92,000 grayscale glyphs, 46 classes, 2,000 writers, distributed through the UCI Machine Learning Repository " },
    { t: "[6]" },
    { t: " \u2014 and a CNN that reached approximately 98.5% accuracy on isolated characters, establishing the standard protocol this project adopts. The theoretical foundations of the deep-learning toolbox used here (training, regularization, evaluation) follow Goodfellow et al. " },
    { t: "[7]" },
    { t: "." },
  ]));
  B.push(body([
    { t: "On the applied side, general-purpose engines such as Tesseract " },
    { t: "[1]" },
    { t: " remain the default OCR choice in most software stacks. Tesseract is print-oriented: its layout analysis and language models assume machine-printed text, and its accuracy on handwritten Devanagari is poor in practice. The literature therefore presents a clear gap: benchmark-grade accuracy exists for isolated Devanagari glyphs, but an offline, end-to-end engine that connects that accuracy to full photographed pages \u2014 with segmentation robust to ruled paper, shadows, and folds \u2014 is not available as a working, documented system. This project was designed to fill exactly that gap." },
  ]));

  // ============================================================ 5. Feasibility
  B.push(h1("5. Feasibility Study"));
  B.push(h2("5.1 Technical Feasibility"));
  B.push(body("The system is built entirely from free, open-source technology: Python, PyTorch, and OpenCV. The chosen classifier is deliberately compact \u2014 823,758 parameters, a 3.2 MB checkpoint \u2014 so it trains in about 12 minutes (15 epochs, batch size 128) on the Apple-Silicon GPU (MPS) of an ordinary laptop and runs inference on CPU if needed, with no cloud dependency. Every component (preprocessing, segmentation, training, inference) was demonstrated to work during development, which removes the main technical uncertainty. Devanagari-specific hazards were identified early and mitigations were built in and tested, as summarized in Table 2."));
  B.push(h2("5.2 Data Feasibility"));
  B.push(body("Training data is freely available: the DHCD is downloadable from the UCI Machine Learning Repository at no cost and requires no licensing. Its size (92,000 images) and writer diversity (2,000 writers) are sufficient for supervised training of the 46-class classifier. Table 1 summarizes the dataset and Figure 1 shows sample glyphs from all 46 classes."));
  B.push(...threeLineTable(
    "Table 1: Summary of the Devanagari Handwritten Character Dataset (DHCD)",
    ["Property", "Value"],
    [
      ["Classes", "46 (36 consonant forms \u0915\u2026\u091c\u094d\u091e + 10 digits \u0966\u2013\u096f)"],
      ["Total images", "92,000 handwritten, grayscale 32\u00d732 (white ink on black)"],
      ["Writers", "2,000 different writers"],
      ["Training split", "78,200 images (engine uses stratified 90/10 \u2192 70,380 train / 7,820 validation)"],
      ["Held-out test split", "13,800 images, never used during training"],
      ["Source", "UCI Machine Learning Repository [6]"],
    ],
    [32, 68]
  ));
  B.push(...figure("presentation_assets/dhcd_montage.png", 440, "Figure 1: Sample glyphs of the 46 DHCD classes used for training"));
  B.push(h2("5.3 Hardware and Software Feasibility"));
  B.push(body("All development ran on a personal Apple-Silicon laptop and page photographs were captured with a regular smartphone camera; no hardware was purchased for the project. The software stack (Python, PyTorch, torchvision, OpenCV, NumPy) is open-source and runs on the three backends the engine supports (CUDA, Apple MPS, and CPU), so the system remains portable to other machines."));
  B.push(h2("5.4 Economic Feasibility"));
  B.push(body("Because the dataset is free, the software is open-source, the model is small enough for local training, and the hardware was already owned, the incremental cost of the project is limited to printing and binding of documentation (Section 10). This makes the project economically feasible and fully reproducible by other student groups."));
  B.push(h2("5.5 Schedule and Risk Feasibility"));
  B.push(body("The work was planned to fit one semester of Project I, with model development and segmentation engine work running largely in parallel (Section 9). The risks that could have derailed the project were identified at proposal time, and each mitigation was implemented and verified \u2014 the stress-test results in Section 7.4 are the direct evidence that the mitigations work."));
  B.push(...threeLineTable(
    "Table 2: Main risks identified at proposal time and their implemented mitigations",
    ["Risk", "Mitigation implemented in the engine"],
    [
      ["Faint pencil strokes vanish at binarization", "CLAHE contrast repair; adaptive binarization selected automatically when measured ink contrast is low; text-scale upscale guard"],
      ["Fold shadows on crumpled paper read as ink", "Global Otsu binarization for dark ink; dust-component removal; crease-valley splitting of projection profiles"],
      ["Shirorekha joins a whole word into one contour", "Morphological head-stroke removal; vertical projection; union boxes measured on the original binary so crops match DHCD framing"],
      ["Skew and shadows in phone photographs", "Divide-by-background shadow removal; minAreaRect deskew"],
      ["Overfitting to training writers", "Rotation/affine augmentation, dropout, stratified split, and a test set kept untouched until the final evaluation"],
    ],
    [42, 58]
  ));

  // ============================================================ 6. Methodology
  B.push(h1("6. Methodology"));
  B.push(h2("6.1 System Overview"));
  B.push(body([
    { t: "The engine solves one hard problem by chaining five small ones, as shown in Figure 2: preprocessing, line segmentation, character segmentation, CNN classification, and reading-order assembly. The single design principle that organizes every stage is " },
    { t: "distribution matching", i: true },
    { t: ": the classifier was trained on clean, isolated, white-on-black 32\u00d732 DHCD glyphs, while a phone photograph is the opposite of that, so each stage exists to make its output look progressively more like the training data. The classifier itself (Stage 4) is the only learned component; Stages 1\u20133 are classical image processing and Stage 5 is geometric bookkeeping." },
  ]));
  B.push(...figure("report_build/figure_pipeline.png", 540, "Figure 2: Five-stage OCR pipeline from photographed page to Devanagari text"));
  B.push(body("Figure 3 shows the same engine at module level, organized in four layers. Three command-line entry points invoke the training, recognition, and evaluation pipelines; all three share the same core \u2014 the DevanagariCNN classifier, the 46-class DHCD label map that translates network outputs into Devanagari Unicode, and a device abstraction that runs the identical code unchanged on CUDA GPUs, Apple-Silicon MPS, or the CPU (the dashed CRNN is shipped but inactive \u2014 Section 6.5). The bottom layer holds the data artifacts the pipelines read and write: the DHCD dataset, the self-describing checkpoint, and the recognition outputs (UTF-8 text, annotated overlay, verification sheet, and accuracy report).", { after: 80 }));
  B.push(...figure("report_build/figure_architecture.png", 500, "Figure 3: Layered system architecture of the Devanagari OCR engine (dashed: CRNN, shipped for future scope)"));
  B.push(h2("6.2 Image Preprocessing"));
  B.push(body([
    { t: "Preprocessing turns a photographed page into a clean grayscale image close to the training rendering. Its steps and their purposes are listed in Table 3. Two steps are worth highlighting. Shadow removal works by divide-by-background: a large dilation followed by Gaussian blur estimates the local paper brightness, and dividing the page by that estimate makes paper uniformly white while ink stays dark relative to its own local paper \u2014 the illumination gradient disappears. Binarization is chosen automatically by measuring the ink contrast: dark-ink pages use the global Otsu threshold " },
    { t: "[8]" },
    { t: " (which correctly ignores fold shadows), while faint-pencil pages use adaptive local thresholding (which correctly ignores paper texture); the two failure modes are complementary, so the engine measures which regime it is in and picks the matching binarizer. Contrast repair uses CLAHE " },
    { t: "[9]" },
    { t: " and a text-scale guard upscales the page until the median text row reaches about 52 pixels, because thin faint strokes do not survive binarization at small sizes." },
  ]));
  B.push(...threeLineTable(
    "Table 3: Preprocessing chain \u2014 steps, techniques, and purposes",
    ["Step", "Technique", "Purpose"],
    [
      ["Upscale small photos", "Resize when height < 1200 px", "Keep thin strokes and matras alive through binarization"],
      ["Shadow removal", "Divide-by-background (dilation + Gaussian)", "Flatten lighting gradients from phone photography"],
      ["Deskew", "minAreaRect on ink pixels, rotate if \u2265 0.5\u00b0", "Keep text rows horizontal for projection profiles"],
      ["Contrast repair", "CLAHE (8\u00d78 tiles)", "Make faint strokes visible without amplifying noise"],
      ["Denoise", "Bilateral filter", "Smooth paper texture while preserving stroke edges"],
      ["Ruled-line erase", "Morphological opening (horizontal) + column statistics (vertical) + inpainting", "Remove notebook rules and margin lines without amputating crossed characters"],
      ["Text-scale guard", "Upscale until median text row \u2248 52 px (cap \u00d74)", "Prevent small faint text from breaking at binarization"],
      ["Binarize", "Otsu or adaptive, chosen by measured ink-contrast gap", "Produce the clean ink mask all later stages need"],
    ],
    [20, 40, 40]
  ));
  B.push(...figure("presentation_assets/before_after.png", 500, "Figure 4: Photographed page before (left) and after (right) preprocessing \u2014 shadow lifted, background cleaned"));
  B.push(h2("6.3 Line and Character Segmentation"));
  B.push(body("Lines are found on the dust-cleaned ink mask by horizontal projection profiles: isolated blobs smaller than a measured dust threshold are erased first (so a fold-shadow dash cannot smear into a phantom line), rows with ink form bands with small gaps bridged, and inside each band rows carrying far less ink than the band\u2019s peak are treated as crease valleys \u2014 merged bands split there and crease-only bands drop out. This is what lets the engine read text on crumpled paper without hallucinating lines from the creases."));
  B.push(body("Character segmentation must first deal with the shirorekha, the horizontal head-stroke that connects every character of a word into one contour. A wide horizontal morphological opening isolates exactly that stroke, and subtracting it restores vertical gaps between characters. Because subtracting the headline also disconnects upper loops and matras from their bodies, characters are located by vertical projection rather than connected components, and each character\u2019s final box is the union bounding box of all ink in its column range measured on the original binary \u2014 so the shirorekha and upper matras stay inside the crop, exactly matching how DHCD training glyphs are framed. Glyph fragments separated by the strip are re-unioned by proximity, and dust and stroke-sliver boxes are filtered out. Figure 5 shows the result: each located character is boxed in reading order."));
  B.push(...figure("presentation_assets/segmentation_demo.png", 540, "Figure 5: Segmentation on a real page \u2014 detected lines (left) and per-character crops (right)"));
  B.push(h2("6.4 CNN Architecture"));
  B.push(body([
    { t: "The classifier is a compact seven-convolution CNN summarized in Table 4. Batch normalization " },
    { t: "[10]" },
    { t: " after every convolution stabilizes training and makes the convolution bias redundant (all convolutions are bias-free); LeakyReLU keeps a small gradient on negative activations, avoiding dead neurons on sparse ink-dominated inputs; two dropout layers " },
    { t: "[11]" },
    { t: " at rate 0.4 around the classifier head fight co-adaptation on handwriting strokes; and the wide 2048\u2192256 bottleneck gives the head capacity to separate 46 visually similar classes. The engine also ships a second, sequence-based architecture \u2014 a CRNN \u2014 described separately in Section 6.5." },
  ]));
  B.push(...threeLineTable(
    "Table 4: DevanagariCNN architecture (input 1\u00d732\u00d732 grayscale glyph; 823,758 parameters total)",
    ["Block", "Layers", "Output shape", "Parameters"],
    [
      ["Block 1", "2 \u00d7 [Conv 3\u00d73 (1\u219232, no bias) \u2192 BN \u2192 LeakyReLU] \u2192 MaxPool 2\u00d72", "32 \u00d7 16 \u00d7 16", "9,632"],
      ["Block 2", "2 \u00d7 [Conv 3\u00d73 (32\u219264) \u2192 BN \u2192 LeakyReLU] \u2192 MaxPool 2\u00d72", "64 \u00d7 8 \u00d7 8", "55,552"],
      ["Block 3", "2 \u00d7 [Conv 3\u00d73 (64\u2192128) \u2192 BN \u2192 LeakyReLU] \u2192 MaxPool 2\u00d72", "128 \u00d7 4 \u00d7 4", "222,080"],
      ["Head", "Flatten(2048) \u2192 Dropout 0.4 \u2192 Linear 2048\u2192256 \u2192 BN \u2192 LeakyReLU \u2192 Dropout 0.4 \u2192 Linear 256\u219246", "46 class scores", "536,494"],
    ],
    [12, 50, 18, 20],
    [AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.CENTER]
  ));
  B.push(h2("6.5 CRNN Architecture (Future Scope)"));
  B.push(body([
    { t: "Alongside the CNN, the engine implements a convolutional recurrent neural network (CRNN) for line-level recognition. A CNN feature extractor processes a whole text line and collapses its feature map to unit height, so each column of the map becomes one time step; two bidirectional LSTMs read that sequence in both directions; and a final layer emits, for every time step, a score over the 46 character classes plus one extra CTC blank symbol. Trained with the connectionist temporal classification (CTC) objective, such a network learns the alignment between input columns and output characters by itself, and can therefore transcribe an entire text line without any character-segmentation stage \u2014 eliminating the class of errors that slicing a line into glyphs introduces." },
  ]));
  B.push(body("The CRNN is shipped in the codebase but deliberately inactive in the demonstrated pipeline, and the reason is the training data. DHCD is labeled per isolated character (2,000 images per class), which the segment-then-classify CNN exploits directly; a CTC-trained CRNN instead requires line-level transcriptions \u2014 whole lines of handwriting paired with their text \u2014 which DHCD does not provide. Training it would therefore require a new labeled dataset, which is why line-level recognition with the CRNN is listed as the first direction of future work (Section 8.3). In the system architecture (Figure 3), the CRNN accordingly appears as a dashed component of the shared core: present and loadable, but not exercised by the demonstrated pipeline."));
  B.push(h2("6.6 Dataset Handling and Training Protocol"));
  B.push(body([
    { t: "Training follows the DHCD protocol with an explicit integrity rule. The 78,200 training images are split stratified 90/10 into 70,380 training and 7,820 validation images, so every class keeps its proportion in both halves; two dataset views are built so augmentation never leaks into validation metrics. The official 13,800-image Test split is never opened during training \u2014 it exists for exactly one run of the evaluation script at the end, which is what makes the reported test accuracy an unbiased estimate. Training augmentation exposes the network to realistic variation: random rotation \u00b110\u00b0 and affine translation/scale perturbations with black corner fill matching the dataset background. Optimization uses AdamW " },
    { t: "[12]" },
    { t: " (learning rate 0.001, weight decay 0.0001) with ReduceLROnPlateau halving the learning rate after two stagnant epochs, cross-entropy loss, batch size 128, and 15 epochs; whenever validation accuracy reaches a new best, a checkpoint is saved that also carries the class list, so the model file is self-describing. Table 5 lists the configuration." },
  ]));
  B.push(...threeLineTable(
    "Table 5: Training configuration",
    ["Item", "Setting"],
    [
      ["Split", "Stratified 90/10 of the DHCD Train folder (70,380 / 7,820); Test split untouched"],
      ["Augmentation", "RandomRotation \u00b110\u00b0 (fill 0); RandomAffine translate \u00b18%, scale 0.92\u20131.08"],
      ["Optimizer", "AdamW, learning rate 0.001, weight decay 0.0001"],
      ["Schedule", "ReduceLROnPlateau on validation loss (halve after 2 stagnant epochs)"],
      ["Loss / batch / epochs", "Cross-entropy; batch 128; 15 epochs (~12 minutes on Apple-Silicon MPS)"],
      ["Checkpointing", "Best validation accuracy; stores weights + class list + accuracy (3.2 MB)"],
    ],
    [30, 70]
  ));
  B.push(h2("6.7 Inference and Reading-Order Assembly"));
  B.push(body("At inference, each segmented crop is conditioned to match the training distribution before classification: it is binarized with Otsu (removing the gray-texture mismatch of real photos), padded with a margin of about 10% on each side because DHCD glyphs occupy only 80\u201388% of their frame, centered on a square letterbox canvas so the resize never distorts stroke geometry, and inverted with a bitwise NOT so dark-ink-on-paper becomes white-ink-on-black, the exact polarity of the training data. The network\u2019s softmax output yields the predicted character and a confidence value; detections below a deliberately permissive 0.10 threshold are discarded. Detections are grouped into lines by comparing each glyph\u2019s vertical center against the running average center of the current line (tolerance 0.6 \u00d7 glyph height), which keeps slightly sloped handwritten rows together, and each line is sorted left-to-right. The engine then writes the UTF-8 text file, an annotated copy of the original photo with every recognized character boxed and color-coded by confidence (green above 0.8, cyan 0.5\u20130.8, red below 0.5), and a per-glyph verification sheet pairing each crop with real DHCD samples of its predicted class."));

  // ============================================================ 7. Evaluation
  B.push(h1("7. Evaluation and Results"));
  B.push(h2("7.1 Evaluation Protocol"));
  B.push(body("Evaluation follows a two-level protocol. At the benchmark level, the trained model is evaluated exactly once on the DHCD Test split \u2014 13,800 images from writers never seen in training \u2014 reporting Top-1 accuracy (the top-scoring class is correct), Top-3 accuracy (the true class is among the three highest scores, a proxy for how recoverable errors are with dictionary correction), and per-class accuracy. At the page level, the complete engine is demonstrated on real photographs: the demo page (pen on shadowed pink paper, characters in a spaced grid), a stress page (faint pencil on ruled notebook paper with a red margin line), and a crumpled-paper page. Because the test split was untouched during training and tuning, the benchmark numbers are unbiased generalization estimates rather than fits."));
  B.push(h2("7.2 Benchmark Results"));
  B.push(body("Table 6 collects the measured results of the verified training run, and Figure 6 shows the training curves. The model crossed its 98% objective with room to spare: 99.36% Top-1 and 99.95% Top-3 on 13,800 held-out images, with 12 of the 46 classes recognized perfectly and the weakest class (\u0922, which differs from \u0927 and \u0921 only in small stroke details) still at 97.00%. The entire training took about 12 minutes, confirming the feasibility claim."));
  B.push(...threeLineTable(
    "Table 6: Measured results of the final model (verified run, Apple-Silicon MPS)",
    ["Metric", "Value"],
    [
      ["Validation accuracy (best checkpoint, epoch 13 of 15)", "99.53%"],
      ["Test accuracy, Top-1 (13,800 held-out images)", "99.36%"],
      ["Test accuracy, Top-3", "99.95%"],
      ["Classes at 100% test accuracy", "12 of 46"],
      ["Weakest class (\u0922, character_14_dhaa)", "97.00%"],
      ["Model size / parameters", "3.2 MB / 823,758"],
      ["Training time (15 epochs, batch 128, MPS)", "~12 minutes"],
    ],
    [62, 38]
  ));
  B.push(...figure("checkpoints/training_curves.png", 540, "Figure 6: Training and validation curves across the 15 epochs"));
  B.push(h2("7.3 Page-Level Demonstration"));
  B.push(body([
    { t: "On the demo page \u2014 a real phone photograph of ten Devanagari characters written on shadowed pink paper \u2014 the engine located and recognized all 10 glyphs in correct reading order, every prediction at 0.99\u20131.00 confidence. Figure 7 shows the annotated output with the recognition boxes; the reconstructed text reads:" },
  ], { after: 80 }));
  B.push(new Paragraph({
    alignment: AlignmentType.LEFT,
    indent: { left: 1440 },
    spacing: { after: 140, line: 400, lineRule: "atLeast" },
    children: [
      new TextRun({ text: "\u0915\u091b\u092b", size: 28, color: "000000", font: DEV_FONT }),
      new TextRun({ break: 1, text: "\u0916\u092c\u0923", size: 28, color: "000000", font: DEV_FONT }),
      new TextRun({ break: 1, text: "\u0917\u0935\u091c", size: 28, color: "000000", font: DEV_FONT }),
      new TextRun({ break: 1, text: "\u091c\u094d\u091e", size: 28, color: "000000", font: DEV_FONT }),
    ],
  }));
  B.push(...figure("ocr_output_visual.png", 320, "Figure 7: Annotated engine output on the demo page \u2014 all 10 glyphs located and recognized (green boxes = confidence above 0.8)"));
  B.push(h2("7.4 Stress Tests under Adverse Conditions"));
  B.push(body("Two harder pages were used to exercise the robustness paths rather than to chase accuracy numbers. The first stress page is faint pencil on ruled notebook paper with a red margin line \u2014 the hardest regime, in which stroke contrast can fall below the paper\u2019s own texture. It drives the adaptive-binarization, margin-line-removal, and text-upscale paths; detection recovers every written row, although per-character accuracy is visibly lower than on the demo page (Figure 8). The second stress page is pen on heavily crumpled paper, driving the Otsu, dust-removal, and crease-valley paths; all glyphs are located despite the folds crossing the text (Figure 9). Together the two tests demonstrate that the pipeline degrades gracefully instead of failing structurally, and they show exactly why capture quality dominates page-level accuracy: pen beats pencil, large characters beat small ones, and even lighting beats shadows."));
  B.push(...figure("report_build/sandesh_crop.png", 440, "Figure 8: Stress test 1 \u2014 faint pencil on ruled paper: all rows detected; per-character accuracy reduced"));
  B.push(...figure("report_build/crumpled_crop.png", 400, "Figure 9: Stress test 2 \u2014 pen on crumpled paper: all glyphs located despite fold shadows"));
  B.push(h2("7.5 Per-Glyph Verification and Interpretation"));
  B.push(body([
    { t: "Because page-level correctness on free handwriting depends on how closely a writer\u2019s style resembles the DHCD writers, the engine emits a verification sheet that pairs every segmented crop with real DHCD samples of its predicted class (Figure 10), so each glyph\u2019s correctness can be checked by eye. The results also have to be interpreted at the right boundary: 99.36% is the per-character accuracy on clean, isolated, 32\u00d732 crops from held-out writers \u2014 the standard way this task is measured \u2014 whereas page-level accuracy on real photographs is necessarily lower, because segmentation errors, writing style, and pen/paper conditions add errors on top of classification. Stating both sides of that boundary precisely, and providing the stress tests and the verification sheet as evidence, is part of the project\u2019s evaluation contribution." },
  ]));
  B.push(...figure("ocr_glyph_verification.png", 540, "Figure 10: Per-glyph verification sheet \u2014 each segmented crop (top) beside real DHCD samples of its predicted class (bottom)"));

  // ============================================================ 8. Outcomes
  B.push(h1("8. Outcomes and Future Work"));
  B.push(h2("8.1 Outcomes Achieved against the Objectives"));
  B.push(body("The project achieved all five specific objectives; Table 7 maps each objective to its measured outcome. The deliverables are the trained self-describing checkpoint, the full OCR engine, the evaluation script, the annotated demonstration outputs, and the verification sheet \u2014 all of which will be shown live in the demonstration."));
  B.push(...threeLineTable(
    "Table 7: Objectives versus achieved outcomes",
    ["Objective", "Outcome"],
    [
      ["Preprocessing pipeline for real photographs", "Implemented; verified on clean pen, faint pencil on ruled paper, and crumpled paper (Figures 4, 8, 9)"],
      ["Robust line/character segmentation in reading order", "Implemented; all rows located on both stress pages; demo page glyphs in reading order (Figures 5, 7)"],
      ["46-class CNN at \u2265 98% on untouched test split", "Achieved 99.36% Top-1 (99.95% Top-3) on 13,800 held-out images (Table 6)"],
      ["Held-out evaluation + per-glyph verification", "Test split used once at the end; per-class report and verification sheet produced (Figure 10)"],
      ["Reading-order export to Devanagari Unicode", "UTF-8 text + confidence-annotated overlay produced for all demo pages (Figure 7)"],
    ],
    [42, 58]
  ));
  B.push(h2("8.2 Limitations"));
  B.push(body("The current system recognizes exactly the 46 DHCD classes: independent vowels (\u0905, \u0907, \u2026), conjunct consonants beyond the three included forms, punctuation, and Latin text have no output class and are forced into the nearest-looking glyph, which is a stated limitation rather than a hidden failure. The segment-then-classify design inherits segmentation errors into page-level accuracy, and adverse capture conditions (faint pencil, textured paper, folds) reduce accuracy below the benchmark number. Finally, because DHCD is labeled per character, the shipped CRNN architecture cannot be trained with this dataset alone, and no language-model correction is applied to the output yet."));
  B.push(h2("8.3 Future Work"));
  B.push(body("Four directions follow directly from the limitations. First, line-level recognition with the shipped CRNN + CTC architecture (Section 6.5), which requires collecting line-level transcriptions and would eliminate per-character segmentation errors. Second, dictionary- and language-model-based post-correction: with Top-3 accuracy at 99.95%, almost every error is recoverable by contextual correction. Third, extending the class inventory to independent vowels, conjuncts, punctuation, and word-level ligature handling, ideally with a Nepali page-level handwriting dataset collected by the group. Fourth, packaging the 3.2 MB model with the segmentation engine into a mobile or desktop application so the demonstration tool becomes a usable product."));

  // ============================================================ 9. Time schedule
  B.push(h1("9. Time Schedule"));
  B.push(body("Figure 11 shows the project schedule as a Gantt chart. Phases deliberately overlapped: dataset acquisition and CNN development began while the literature survey was finishing, and the segmentation engine was built in parallel with benchmark evaluation. The stress-testing phase absorbed most of the mid-project uncertainty, which is where the robustness paths described in Section 6 were hardened; report writing and demonstration preparation started in the second half of September."));
  B.push(...figure("report_build/figure_gantt.png", 520, "Figure 11: Project schedule (Gantt chart), June \u2013 October 2026"));

  // ============================================================ 10. Cost
  B.push(h1("10. Estimated Cost"));
  B.push(body("The project required no purchase: the dataset is free, all software is open-source, training ran locally on an already-owned laptop, and photographs were taken with an already-owned smartphone. Table 8 itemizes the resources and their cost impact for the project budget; the only incremental expense is documentation printing and binding."));
  B.push(...threeLineTable(
    "Table 8: Cost estimation",
    ["Item", "Details", "Cost impact"],
    [
      ["Laptop (Apple Silicon)", "Development, training, inference \u2014 already owned", "NPR 0 (market value \u2248 NPR 160,000)"],
      ["Smartphone camera", "Page capture \u2014 already owned", "NPR 0 (market value \u2248 NPR 30,000)"],
      ["DHCD dataset", "UCI Machine Learning Repository [6]", "Free"],
      ["Software", "Python, PyTorch, OpenCV, NumPy (open-source)", "Free"],
      ["Cloud GPU / storage", "Not required \u2014 local MPS training (3.2 MB model)", "NPR 0"],
      ["Printing and binding", "Report copies and demonstration material", "\u2248 NPR 500"],
      ["Total incremental cost", "\u2014", "\u2248 NPR 500"],
    ],
    [28, 44, 28],
    [AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.CENTER]
  ));

  // ============================================================ References
  B.push(h1("References"));
  const refs = [
    "R. Smith, \u201cAn overview of the Tesseract OCR engine,\u201d in Proc. 9th Int. Conf. Document Analysis and Recognition (ICDAR), Curitiba, Brazil, 2007, pp. 629\u2013633.",
    "U. Pal, T. Wakabayashi, and F. Kimura, \u201cComparative study of Devanagari handwritten character recognition using different features and classifiers,\u201d in Proc. 10th Int. Conf. Document Analysis and Recognition (ICDAR), Barcelona, Spain, 2009, pp. 1111\u20131115.",
    "Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, \u201cGradient-based learning applied to document recognition,\u201d Proceedings of the IEEE, vol. 86, no. 11, pp. 2278\u20132324, Nov. 1998.",
    "A. Krizhevsky, I. Sutskever, and G. E. Hinton, \u201cImageNet classification with deep convolutional neural networks,\u201d in Advances in Neural Information Processing Systems (NeurIPS), 2012, pp. 1097\u20131105.",
    "S. Acharya, A. K. Panda, S. Ghosh, and S. Ghosh, \u201cDeep learning based large scale handwritten Devanagari character recognition,\u201d in AAAI Spring Symposium: Combining Machine Learning with Knowledge Engineering, Stanford, CA, USA, 2019.",
    "S. Acharya, A. K. Panda, S. Ghosh, and S. Ghosh, \u201cDevanagari Handwritten Character Dataset (DHCD),\u201d UCI Machine Learning Repository, Irvine, CA, USA, 2019. [Online]. Available: https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset",
    "I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. Cambridge, MA, USA: MIT Press, 2016.",
    "N. Otsu, \u201cA threshold selection method from gray-level histograms,\u201d IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-9, no. 1, pp. 62\u201366, Jan. 1979.",
    "K. Zuiderveld, \u201cContrast limited adaptive histogram equalization,\u201d in Graphics Gems IV, P. S. Heckbert, Ed. San Diego, CA, USA: Academic Press, 1994, pp. 474\u2013485.",
    "S. Ioffe and C. Szegedy, \u201cBatch normalization: Accelerating deep network training by reducing internal covariate shift,\u201d in Proc. 32nd Int. Conf. Machine Learning (ICML), Lille, France, 2015, pp. 448\u2013456.",
    "N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, \u201cDropout: A simple way to prevent neural networks from overfitting,\u201d Journal of Machine Learning Research, vol. 15, pp. 1929\u20131958, 2014.",
    "I. Loshchilov and F. Hutter, \u201cDecoupled weight decay regularization,\u201d in Proc. 7th Int. Conf. Learning Representations (ICLR), New Orleans, LA, USA, 2019.",
  ];
  refs.forEach((r, i) => B.push(new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { left: 567, hanging: 567 },
    spacing: { after: 100, line: 360 },
    children: [new TextRun({ text: "[" + (i + 1) + "] " + r, size: 24, color: "000000", font: TNR })],
  })));

  return B;
}

// ---------------------------------------------------------------- document
const pgMargin = { top: 1440, bottom: 1440, left: 1728, right: 1440, header: 850, footer: 850 };

const doc = new Document({
  creator: "Project Group",
  title: "Devanagari Handwritten Character Recognition and Full-Page OCR Using a Convolutional Neural Network",
  styles: {
    default: {
      document: {
        run: { font: TNR, size: 24, color: "000000" },
        paragraph: { spacing: { line: 360 } },
      },
      heading1: {
        run: { font: TNR, size: 28, bold: true, color: "000000" },
        paragraph: { spacing: { before: 400, after: 200, line: 360 } },
      },
      heading2: {
        run: { font: TNR, size: 24, bold: true, color: "000000" },
        paragraph: { spacing: { before: 280, after: 140, line: 360 } },
      },
    },
  },
  numbering: {
    config: [
      {
        reference: "obj-list",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      },
    ],
  },
  sections: [
    // ---- Section 1: cover (no page number, no footer)
    {
      properties: {
        page: { size: { width: 11906, height: 16838 }, margin: pgMargin },
      },
      children: buildCover(),
    },
    // ---- Section 2: front matter (abstract + TOC), Roman numerals
    {
      properties: {
        type: SectionType.NEXT_PAGE,
        page: {
          size: { width: 11906, height: 16838 },
          margin: pgMargin,
          pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN },
        },
      },
      footers: { default: pageFooter() },
      children: [...buildAbstract(), ...buildTocPage()],
    },
    // ---- Section 3: body, Arabic from 1
    {
      properties: {
        type: SectionType.NEXT_PAGE,
        page: {
          size: { width: 11906, height: 16838 },
          margin: pgMargin,
          pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL },
        },
      },
      footers: { default: pageFooter() },
      children: buildBody(),
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("written:", OUT, buf.length, "bytes");
});
