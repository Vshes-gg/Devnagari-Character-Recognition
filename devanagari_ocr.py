"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║        PRODUCTION-GRADE PYTORCH DEVANAGARI HANDWRITTEN OCR ENGINE                       ║
║        Supports: Custom PyTorch CNN Model + Segmenter + Training + Pipeline Integrator   ║
║                                                                                          ║
║  Strategy Overview:                                                                      ║
║  ──────────────────────────────────────────────────────────────────────────────────────  ║
║  ① Image Preprocessing: Shadow removal, deskew, CLAHE, bilateral denoising, line erase   ║
║  ② Line & Character Segmentation: Shirorekha removal + horizontal projection clustering  ║
║  ③ Deep Learning Engine: Custom 32x32 Devanagari CNN (DHCD standard 46 classes)           ║
║  ④ High-Level Recognition: CRNN (CNN + BiLSTM + CTC) module for continuous sentences     ║
║  ⑤ Full Training Pipeline: PyTorch Trainer with augmentations, AdamW, & learning rate   ║
║  ⑥ Document Reconstruction: Spatial line grouping & confidence-coded visual exporter     ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import math
import argparse
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split


# ============================================================================
#  SECTION 0: COMPUTE DEVICE SELECTION
# ============================================================================

def get_device():
    """
    Pick the fastest compute backend available, in priority order:
      1. CUDA — NVIDIA GPU.
      2. MPS  — Metal Performance Shaders, Apple Silicon's GPU backend
                (typically 5-20x faster than CPU for CNN training).
      3. CPU  — always-available fallback.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device):
    """Human-readable device name for progress printing."""
    if device.type == "cuda":
        return f"CUDA GPU ({torch.cuda.get_device_name(0)})"
    if device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    return "CPU"


# ============================================================================
#  SECTION 1: DEVANAGARI CHARACTER MAPPING & CONSTANTS
# ============================================================================

# DHCD (Devanagari Handwritten Character Dataset) 46 Class Mapping
# 36 Consonants (क to ज्ञ) + 10 Digits (० to ९)
DHCD_LABEL_MAP = {
    0: ("character_1_ka", "क"), 1: ("character_2_kha", "ख"), 2: ("character_3_ga", "ग"),
    3: ("character_4_gha", "घ"), 4: ("character_5_kna", "ङ"), 5: ("character_6_cha", "च"),
    6: ("character_7_chha", "छ"), 7: ("character_8_ja", "ज"), 8: ("character_9_jha", "झ"),
    9: ("character_10_yna", "ञ"), 10: ("character_11_taamatar", "ट"), 11: ("character_12_thaa", "ठ"),
    12: ("character_13_daa", "ड"), 13: ("character_14_dhaa", "ढ"), 14: ("character_15_adna", "ण"),
    15: ("character_16_tabala", "त"), 16: ("character_17_tha", "थ"), 17: ("character_18_da", "द"),
    18: ("character_19_dha", "ध"), 19: ("character_20_na", "न"), 20: ("character_21_pa", "प"),
    21: ("character_22_pha", "फ"), 22: ("character_23_ba", "ब"), 23: ("character_24_bha", "भ"),
    24: ("character_25_ma", "म"), 25: ("character_26_yaw", "य"), 26: ("character_27_ra", "र"),
    27: ("character_28_la", "ल"), 28: ("character_29_waw", "व"), 29: ("character_30_motosaw", "श"),
    30: ("character_31_petchiryakha", "ष"), 31: ("character_32_patalosaw", "स"), 32: ("character_33_ha", "ह"),
    33: ("character_34_chhya", "क्ष"), 34: ("character_35_tra", "त्र"), 35: ("character_36_gya", "ज्ञ"),
    36: ("digit_0", "०"), 37: ("digit_1", "१"), 38: ("digit_2", "२"), 39: ("digit_3", "३"),
    40: ("digit_4", "४"), 41: ("digit_5", "५"), 42: ("digit_6", "६"), 43: ("digit_7", "७"),
    44: ("digit_8", "८"), 45: ("digit_9", "९")
}

# Reverse mapping from folder/class name to Devanagari character
FOLDER_TO_UNICODE = {v[0]: v[1] for k, v in DHCD_LABEL_MAP.items()}
INDEX_TO_UNICODE = {k: v[1] for k, v in DHCD_LABEL_MAP.items()}
DIGIT_UNICODE = {v[1] for k, v in DHCD_LABEL_MAP.items() if v[0].startswith("digit_")}


# ============================================================================
#  SECTION 2: PYTORCH ARCHITECTURES
# ============================================================================

class DevanagariCNN(nn.Module):
    """
    Deep Convolutional Neural Network custom-designed for 32x32 single-channel
    grayscale Devanagari character recognition.
    
    Architecture Highlights:
      - 3 Residual-style Convolutional Blocks with BatchNorm and LeakyReLU
      - Spatial Dropout to prevent co-adaptation on handwriting strokes
      - Global Average Pooling + Dense layers for high-precision classification
    """
    def __init__(self, num_classes=46):
        super(DevanagariCNN, self).__init__()
        
        # Conv Block 1: Input (1 x 32 x 32) -> Output (32 x 16 x 16)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv Block 2: Input (32 x 16 x 16) -> Output (64 x 8 x 8)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv Block 3: Input (64 x 8 x 8) -> Output (128 x 4 x 4)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully Connected Classifier
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = F.leaky_relu(self.bn_fc(self.fc1(x)), 0.1)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DevanagariCRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) with BiLSTM for continuous line/word
    sequence recognition without explicit character segmentation.
    """
    def __init__(self, num_classes=47, hidden_size=256):
        super(DevanagariCRNN, self).__init__()
        
        # CNN Feature Extractor for variable width input images (1 x 32 x W)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (64, 16, W/2)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (128, 8, W/4)
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # (256, 4, W/4)
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((4, 1), (4, 1))   # (512, 1, W/4)
        )
        
        # Bidirectional LSTM Sequence Encoder
        self.rnn = nn.Sequential(
            nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True),
            nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        )
        
        # Linear projection to class probabilities (+1 for CTC Blank Token)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Input x shape: (Batch, 1, Height=32, Width=W)
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "Height of conv feature map must be collapsed to 1"
        
        conv = conv.squeeze(2)  # (Batch, Channels=512, Width=W')
        conv = conv.permute(0, 2, 1)  # (Batch, TimeSteps=W', FeatureDims=512)
        
        recurrent, _ = self.rnn(conv)
        logits = self.fc(recurrent)  # (Batch, TimeSteps, NumClasses)
        return logits.log_softmax(2)


# ============================================================================
#  SECTION 3: IMAGE PREPROCESSING & GEOMETRY UTILITIES
# ============================================================================

def bbox_to_xyxy(bbox):
    """Convert standard polygon/list points to [x1, y1, x2, y2] integers."""
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def compute_iou(box1, box2):
    """Calculate Intersection-over-Union (IoU) between two bounding boxes."""
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    
    if inter_area == 0:
        return 0.0
        
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def remove_shadows(gray):
    """Lift shadow gradients using background estimation via large-kernel dilation."""
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(gray, kernel)
    bg_estimate = cv2.GaussianBlur(dilated, (21, 21), 0)
    normalized = cv2.divide(gray.astype(np.float32), bg_estimate.astype(np.float32), scale=255.0)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def deskew(gray):
    """Detect tilt using minimum bounding box on text pixels and rotate horizontally."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return gray
        
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45.0:
        angle = 90.0 + angle
    if abs(angle) < 0.5:
        return gray
        
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def apply_clahe(gray):
    """Contrast Limited Adaptive Histogram Equalization for localized contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def bilateral_denoise(gray):
    """Edge-preserving smoothing to eliminate paper texture and camera noise."""
    return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)


def remove_ruled_lines(gray):
    """Isolate and erase notebook ruled lines.

    Horizontal rules are caught by morphological opening with a wide kernel;
    vertical margin lines by opening with a tall kernel - both filtered by
    rule SHAPE afterwards (long and thin at the page's scale). The shape
    filter is what protects the writing: a close-up photo's shirorekha can
    be hundreds of px long but never spans half the page, and stacked letter
    stems never form one continuous vertical run the way a margin line does.

    The opening result is filtered by rule SHAPE before it may touch the
    page: a rule spans most of the page width and is thinner than any text
    stroke at that scale. On close-up photos a long word's shirorekha can
    be hundreds of px long, but it never spans half the page, so the width
    floor - not a longer kernel alone - is what keeps the head-stroke out
    of the erase mask.
    """
    h, w = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 2), 1)), iterations=1)
    # Vertical margin lines by the same principle: a CONTINUOUS vertical ink
    # run. Column statistics cannot do this job - on close-up photos, thick
    # letter stems stacked over several text rows mimic a margin's span and
    # fill, while on a half-page photo a real margin spans too little to
    # pass any sane span floor. Continuity separates them: a margin line is
    # one unbroken run, letter stems are short runs with row gaps between.
    opened |= cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 8))), iterations=1)
    line_mask = np.zeros_like(binary)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    for i in range(1, num):
        cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        horizontal_rule = cw >= 0.45 * w and ch <= max(5, 0.005 * h)
        # Vertical: thin partial-height lines are drawn margin lines; a
        # full-height band can also be the page's dark edge/spine, which is
        # fatter than any drawn line but still not text (no letter stroke
        # forms a continuous run down the whole page).
        vertical_rule = (ch >= 0.3 * h and cw <= max(5, 0.005 * h)) or \
                        (ch >= 0.7 * h and cw <= max(8, 0.02 * h))
        if horizontal_rule or vertical_rule:
            line_mask[labels == i] = 255

    line_mask = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    result = gray.copy()
    result[line_mask == 255] = 255
    return cv2.inpaint(result, line_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)


def full_preprocessing_pipeline(image_bgr):
    """Execute complete enhancement chain on input image."""
    h, w = image_bgr.shape[:2]
    if h < 1200:
        scale = 1200 / h
        image_bgr = cv2.resize(image_bgr, (int(w * scale), 1200), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = remove_shadows(gray)
    gray = deskew(gray)
    gray = apply_clahe(gray)
    gray = bilateral_denoise(gray)
    gray = remove_ruled_lines(gray)

    # Text-scale guard: a page can be large while its text is small (e.g.
    # 1280px tall with ~30px pencil characters). Thin faint strokes vanish
    # at binarization at that size, so upscale until the median text row
    # reaches ~52px - the size at which segmentation stays reliable.
    # The median runs over SIGNIFICANT bands only (>= page height / 40):
    # fold shadows and binarization noise create many small bands that would
    # otherwise dominate the median on crumpled paper and trigger a pointless
    # upscale of an already-readable page.
    line_boxes = extract_line_crops(gray)
    sig = max(12, gray.shape[0] // 40)
    heights = [y2 - y1 for (_, y1, _, y2) in line_boxes if (y2 - y1) >= sig]
    if heights:
        median_h = float(np.median(heights))
        if median_h < 52:
            scale = min(52.0 / median_h, 4.0)
            gray = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)),
                              interpolation=cv2.INTER_CUBIC)
            print(f"  [preprocess] text rows ~{median_h:.0f}px tall - upscaling x{scale:.2f}")

    return gray


# ============================================================================
#  SECTION 4: SEGMENTATION ENGINE (CHARACTER & SHIROREKHA CUTTING)
# ============================================================================

def remove_shirorekha(line_binary):
    """
    Detect and erase the horizontal top bar (Shirorekha) connecting Devanagari 
    characters to allow contour separation into individual letters.
    """
    h, w = line_binary.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * 0.04)), 1))
    shirorekha_mask = cv2.morphologyEx(line_binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # Subtract top bar from original binary line
    char_binary = cv2.subtract(line_binary, shirorekha_mask)
    return char_binary, shirorekha_mask


def _too_wide(box, med_h, ratio=1.4):
    """True when a box is wider than a single DHCD glyph can plausibly be.

    Devanagari glyphs are roughly square to slightly tall; only joined
    cursive words exceed ~1.4x their height (the three wide conjunct
    classes क्ष त्र ज्ञ bottom out around 1.5-1.7, which is why the
    geometric pre-splitter uses the higher ratio=1.7 and leaves anything
    narrower to the classifier-guided refiner). Floating matras
    (ा ी ो ...) are wide-but-short, so the height floor keeps them out.
    Also used to stop the fragment-merge pass from re-fusing split pieces.
    """
    w, h = box[2] - box[0], box[3] - box[1]
    return w > ratio * h and h >= 0.6 * med_h


def _valley_cuts(prof_s, sx1, sx2, med_h):
    """Column positions that cut a cursive-joined run into glyph pieces.

    Joined letters leave no empty columns to cut at, so cut at the lowest
    points of the smoothed ink projection instead - the neck where two
    letter bodies join. Deepest valley first, then recurse into both halves
    while a piece is still wider than a glyph can be. Cuts stay >= 22% of a
    glyph height from the run's ends and >= 30% from each other, so no
    sliver pieces are created here.
    """
    cuts = []
    stack = [(sx1, sx2)]
    while stack and len(cuts) < 24:
        a, b = stack.pop()
        if (b - a + 1) <= 1.4 * med_h:
            continue
        lo, hi = a + int(0.22 * med_h), b - int(0.22 * med_h)
        if hi - lo < 2:
            continue
        cx = lo + int(np.argmin(prof_s[lo:hi + 1]))
        cuts.append(cx)
        stack.append((a, cx - 1))
        stack.append((cx + 1, b))
    cuts.sort()
    filtered, last = [], sx1 - 1
    for c in cuts:
        if c - last >= 0.3 * med_h and sx2 - c >= 0.3 * med_h:
            filtered.append(c)
            last = c
    return filtered


def _projection_debug_image(col_ink, segments, cuts):
    """Render the vertical ink projection for the --debug dump: ink profile
    in gray, segment bounds in green, cursive-join cuts in red."""
    h = 240
    img = np.zeros((h, len(col_ink), 3), dtype=np.uint8)
    peak = float(col_ink.max()) if len(col_ink) else 0.0
    if peak <= 0:
        peak = 1.0
    for x, v in enumerate(col_ink):
        y0 = h - 1 - int((h - 20) * (v / peak))
        cv2.line(img, (x, h - 1), (x, y0), (200, 200, 200), 1)
    for (sx1, sx2) in segments:
        cv2.line(img, (sx1, 0), (sx1, h - 1), (0, 200, 0), 1)
        cv2.line(img, (sx2, 0), (sx2, h - 1), (0, 200, 0), 1)
    for c in cuts:
        cv2.line(img, (c, 0), (c, h - 1), (0, 0, 255), 1)
    return img


def segment_characters_from_line(line_gray, debug_dir=None, line_tag="line"):
    """
    Splits a single text line into individual character crops.

    Characters are located by VERTICAL PROJECTION on the shirorekha-stripped
    binary: contiguous inked column runs are candidate characters (gaps of
    <=5 px are bridged). Each character's final box is the union bounding
    box of ALL ink inside its column range measured on the ORIGINAL binary -
    so the shirorekha and upper-zone matras stay inside the crop, matching
    how DHCD training glyphs are framed.

    Natural cursive writing also joins neighboring letter BODIES below the
    headline, so a whole word can form one column run with no empty column
    to cut at (the run's box ends up wider than any single glyph). Such runs
    are split at the lowest-ink valleys of the projection (_valley_cuts)
    until every piece has a plausible single-glyph width; the recognizer
    refines the exact cut positions later (refine_character_cuts).
    """
    # Binarize line: Text = White, Background = Black
    binary = _ink_binary(line_gray)

    # Strip Shirorekha (used only to find column boundaries)
    char_binary, _ = remove_shirorekha(binary)

    # Morphological clean up of tiny gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    char_binary = cv2.morphologyEx(char_binary, cv2.MORPH_CLOSE, kernel)

    line_h, line_w = line_gray.shape
    col_ink = (char_binary > 0).sum(axis=0)

    # Vertical projection -> contiguous column runs. Handwritten glyphs split
    # by the shirorekha strip (e.g. tra's detached bars) leave 3-10 px gaps
    # between their own parts, while separate characters sit far apart, so a
    # ~5 px bridge reunites glyph parts without fusing distinct characters.
    merge_gap = 5
    segments = []
    run_start, run_gap = None, 0
    for x in range(line_w):
        if col_ink[x] > 0:
            if run_start is None:
                run_start = x
            run_gap = 0
        elif run_start is not None:
            run_gap += 1
            if run_gap > merge_gap:
                segments.append((run_start, x - run_gap))
                run_start, run_gap = None, 0
    if run_start is not None:
        segments.append((run_start, line_w - 1))

    def box_for_range(sx1, sx2):
        # Union bounding box of a column range, measured on the ORIGINAL
        # binary so shirorekha and matras stay inside the crop.
        region = binary[:, sx1:sx2 + 1]
        ys, xs_in = np.where(region > 0)
        if len(ys) == 0:
            return None
        x1, x2 = sx1 + xs_in.min(), sx1 + xs_in.max()
        y1, y2 = ys.min(), ys.max()
        if (x2 - x1) < 4 or (y2 - y1) < 8 or (x2 - x1 + 1) * (y2 - y1 + 1) < 32:
            return None    # speck of noise
        pad = 2
        return (max(0, x1 - pad), max(0, y1 - pad),
                min(line_w, x2 + 1 + pad), min(line_h, y2 + 1 + pad))

    # Glyph-height scale for the splitter, measured BEFORE splitting: word
    # blobs are as tall as single glyphs, so the median is safe to use.
    pre_boxes = [b for b in (box_for_range(s1, s2) for (s1, s2) in segments) if b]
    med_h = float(np.median([b[3] - b[1] for b in pre_boxes])) if pre_boxes else 0.0

    # Cursive-join splitting: re-cut too-wide runs at projection valleys.
    join_cuts = []
    if med_h >= 12:
        smooth_k = max(3, int(med_h * 0.06) | 1)
        prof_s = np.convolve(col_ink.astype(np.float64),
                             np.ones(smooth_k) / smooth_k, mode="same")
        refined_segments = []
        for (sx1, sx2) in segments:
            box = box_for_range(sx1, sx2)
            if box is None:
                continue
            cuts = _valley_cuts(prof_s, sx1, sx2, med_h) if _too_wide(box, med_h, ratio=1.85) else []
            if cuts:
                join_cuts.extend(cuts)
                edges = [sx1] + cuts + [sx2 + 1]
                refined_segments.extend((edges[i], edges[i + 1] - 1)
                                        for i in range(len(edges) - 1))
            else:
                refined_segments.append((sx1, sx2))
        segments = refined_segments

    # Union bounding box per column range, measured on the ORIGINAL binary
    char_boxes = []
    for (sx1, sx2) in segments:
        box = box_for_range(sx1, sx2)
        if box is not None:
            char_boxes.append(box)

    # Sort boxes left-to-right
    char_boxes.sort(key=lambda b: b[0])

    # Rebuild whole glyphs: parts of one character separated by the
    # shirorekha strip (e.g. a vertical whose only link to the body ran
    # through the head-stroke) sit a few px apart, while distinct characters
    # are spaced far wider. Union boxes whose horizontal gap is below ~25%
    # of the line's median glyph height. The merge is refused when the union
    # would be wider than a single glyph - without that guard the merge pass
    # would instantly re-fuse the cursive-join pieces cut above.
    if char_boxes:
        med_h = np.median([b[3] - b[1] for b in char_boxes])
        merged = [char_boxes[0]]
        for box in char_boxes[1:]:
            m = merged[-1]
            if box[0] - m[2] < 0.25 * med_h:
                u = (min(m[0], box[0]), min(m[1], box[1]),
                     max(m[2], box[2]), max(m[3], box[3]))
                if not _too_wide(u, med_h):
                    merged[-1] = u
                    continue
            merged.append(box)
        char_boxes = merged

        # Drop dust: boxes far shorter than the median glyph height are
        # binarization noise (paper texture, fold shadows) or floating
        # matra fragments (ी hooks that lost their word), not characters.
        # Digits, the smallest real glyphs, sit near 0.7x of letter height.
        med_h = np.median([b[3] - b[1] for b in char_boxes])
        char_boxes = [b for b in char_boxes if (b[3] - b[1]) >= 0.35 * med_h]

        # Drop stroke slivers: boxes much narrower than both their own height
        # and the line's widest glyph (e.g. a vertical separated from its
        # glyph body by the shirorekha strip) classify as junk bars/digits.
        # Real narrow glyphs (like १) survive via the aspect-ratio escape.
        max_w = max(b[2] - b[0] for b in char_boxes)
        char_boxes = [b for b in char_boxes
                      if (b[2] - b[0]) >= 0.25 * (b[3] - b[1])
                      or (b[2] - b[0]) >= 0.3 * max_w]

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{line_tag}_a_line.png"), line_gray)
        cv2.imwrite(os.path.join(debug_dir, f"{line_tag}_b_stripped.png"), char_binary)
        cv2.imwrite(os.path.join(debug_dir, f"{line_tag}_c_projection.png"),
                    _projection_debug_image(col_ink, segments, join_cuts))
        vis = cv2.cvtColor(line_gray, cv2.COLOR_GRAY2BGR)
        for b in char_boxes:
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 220, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, f"{line_tag}_d_boxes.png"), vis)

    return char_boxes


def _local_minima(prof_s, lo, hi):
    lo, hi = max(1, int(lo)), min(len(prof_s) - 2, int(hi))
    return [x for x in range(lo, hi + 1)
            if prof_s[x] <= prof_s[x - 1] and prof_s[x] <= prof_s[x + 1]]


def refine_character_cuts(line_gray, char_boxes, recognizer, debug_dir=None, line_tag="line"):
    """
    Recognition-guided refinement of cursive-join cuts.

    The geometric valley cut finds *a* joint; this pass lets the classifier
    choose the best one. Adjacent boxes whose union is wider than a single
    glyph (an unsplit word blob, or the pieces the geometric splitter just
    produced) are re-sliced at candidate valley positions. Every candidate
    slicing is scored by running the recognizer on its pieces:

        score = sum(log(confidence_i) + width_prior_i)

    where the width prior penalizes pieces too wide to be one glyph. Whole
    blobs compete on the same scale, so keeping the blob only ever wins
    when every proposed cut classifies as garbage: the classifier can move
    cuts and pick cut counts, but it cannot un-split a joined word.
    """
    if not char_boxes or recognizer is None:
        return char_boxes

    med_h = float(np.median([b[3] - b[1] for b in char_boxes]))
    if med_h < 12:
        return char_boxes

    binary = _ink_binary(line_gray)
    char_binary, _ = remove_shirorekha(binary)
    col_ink = (char_binary > 0).sum(axis=0).astype(np.float64)
    smooth_k = max(3, int(med_h * 0.06) | 1)
    prof_s = np.convolve(col_ink, np.ones(smooth_k) / smooth_k, mode="same")

    line_h, line_w = line_gray.shape

    def piece_box(a, b):
        region = binary[:, a:b + 1]
        ys, xs_in = np.where(region > 0)
        if len(ys) == 0:
            return None
        x1, x2 = a + xs_in.min(), a + xs_in.max()
        y1, y2 = ys.min(), ys.max()
        if (x2 - x1) < 4 or (y2 - y1) < 8 or (x2 - x1 + 1) * (y2 - y1 + 1) < 32:
            return None
        pad = 2
        return (max(0, x1 - pad), max(0, y1 - pad),
                min(line_w, x2 + 1 + pad), min(line_h, y2 + 1 + pad))

    def slice_score(cuts, a, b, apply_prior=True):
        edges = [a] + list(cuts) + [b + 1]
        total = 0.0
        for i in range(len(edges) - 1):
            pb = piece_box(edges[i], edges[i + 1] - 1)
            if pb is None:
                return -1e9
            crop = line_gray[pb[1]:pb[3], pb[0]:pb[2]]
            if crop.size == 0:
                return -1e9
            _, conf = recognizer.predict_patch(crop)
            total += math.log(max(conf, 1e-3))
            if apply_prior:
                ratio = (pb[2] - pb[0]) / max(pb[3] - pb[1], 1)
                if ratio > 1.1:
                    total -= (ratio - 1.1) * 2.5
        return total

    boxes = sorted(char_boxes, key=lambda b: b[0])

    # Group boxes that sit close together: pieces of one joined word.
    clusters, cur = [], [boxes[0]]
    for b in boxes[1:]:
        if b[0] - cur[-1][2] < 0.12 * med_h:
            cur.append(b)
        else:
            clusters.append(cur)
            cur = [b]
    clusters.append(cur)

    out = []
    cluster_unions = []
    for cluster in clusters:
        cluster_unions.append((min(b[0] for b in cluster), min(b[1] for b in cluster),
                               max(b[2] for b in cluster), max(b[3] for b in cluster)))
    for ci, cluster in enumerate(clusters):
        ux1, uy1, ux2, uy2 = cluster_unions[ci]
        if len(cluster) == 1 and not _too_wide((ux1, uy1, ux2, uy2), med_h):
            out.extend(cluster)
            continue

        lo_lim, hi_lim = ux1 + int(0.3 * med_h), ux2 - int(0.3 * med_h)
        valleys = [v for v in _local_minima(prof_s, lo_lim, hi_lim) if lo_lim <= v <= hi_lim]
        valleys = sorted(sorted(valleys, key=lambda v: prof_s[v])[:12])
        # A cursive join can sit on a smooth slope with no local minimum,
        # so candidates are valleys PLUS a dense grid; the classifier picks
        # the actual joint, on-valley or not.
        step = max(2, int(0.075 * med_h))
        grid = list(range(lo_lim, hi_lim + 1, step))[:28]
        cand_set = sorted(set(valleys) | set(grid))
        if hi_lim - lo_lim < 4 or not cand_set:
            out.extend(cluster)
            continue

        # Evenly spaced letter count: Devanagari letter bodies run ~0.5-0.9x
        # of their height, so divide by 0.7. The classifier arbitrates
        # between this count and one less. Keeping the blob whole is only an
        # option for an ISOLATED too-wide blob (grid-style page, big gaps to
        # both neighbors) that the classifier reads with near-certainty -
        # that is a single wide-written glyph. A blob inside a word flow is
        # a cursive join in disguise (a confident junk label like ख) and
        # must split.
        wchar, wconf = None, 0.0
        wcrop = line_gray[uy1:uy2, ux1:ux2]
        if wcrop.size:
            wchar, wconf = recognizer.predict_patch(wcrop)
        left_gap = ux1 - cluster_unions[ci - 1][2] if ci > 0 else float("inf")
        right_gap = cluster_unions[ci + 1][0] - ux2 if ci + 1 < len(clusters) else float("inf")
        whole_allowed = (wchar is not None and wchar not in DIGIT_UNICODE
                         and wconf >= 0.97
                         and (ux2 - ux1) <= 1.85 * max(uy2 - uy1, 1)
                         and left_gap >= 0.45 * med_h and right_gap >= 0.45 * med_h)
        want = max(2, int(round((ux2 - ux1) / (0.7 * med_h))))
        min_w = 0.3 * med_h
        n_options = {min(want, 6)}
        if whole_allowed:
            n_options.add(1)
        best_cuts, best_score = None, None
        for n_pieces in sorted(n_options):
            if n_pieces == 1:
                # Whole blob: no width prior - a wide single glyph genuinely
                # is wider than tall, that is the whole point of this option.
                cand, score = [], slice_score([], ux1, ux2, apply_prior=False)
            else:
                n_cuts = n_pieces - 1    # too-wide clusters must split
                ideals = [ux1 + (ux2 - ux1) * (i + 1) / float(n_pieces)
                          for i in range(n_cuts)]
                cand, used = [], set()
                for t in ideals:
                    v = min((x for x in cand_set if x not in used),
                            key=lambda x: abs(x - t), default=None)
                    if v is None:
                        continue
                    if cand and v - cand[-1] < min_w:
                        continue
                    used.add(v)
                    cand.append(v)
                if not cand:
                    continue
                score = slice_score(cand, ux1, ux2)
                for _ in range(2):    # coordinate ascent on cut positions
                    improved = False
                    for i in range(len(cand)):
                        low = cand[i - 1] + min_w if i > 0 else lo_lim
                        high = cand[i + 1] - min_w if i < len(cand) - 1 else hi_lim
                        for v in cand_set:
                            if v in cand or not (low <= v <= high):
                                continue
                            if abs(v - cand[i]) > 0.35 * med_h:
                                continue    # keep the search local per cut
                            trial = list(cand)
                            trial[i] = v
                            s = slice_score(trial, ux1, ux2)
                            if s > score + 1e-6:
                                cand, score, improved = trial, s, True
                    if not improved:
                        break
            if best_score is None or score > best_score:
                best_score, best_cuts = score, cand

        final_boxes = []
        if best_cuts:
            edges = [ux1] + best_cuts + [ux2 + 1]
            for i in range(len(edges) - 1):
                pb = piece_box(edges[i], edges[i + 1] - 1)
                if pb is not None:
                    final_boxes.append(pb)
        out.extend(final_boxes if final_boxes else cluster)

    out.sort(key=lambda b: b[0])

    if debug_dir:
        vis = cv2.cvtColor(line_gray, cv2.COLOR_GRAY2BGR)
        for b in out:
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 160, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, f"{line_tag}_e_refined.png"), vis)

    return out


def _ink_binary(gray):
    """Binarize a page or line, choosing the binarizer by ink contrast.

    - Dark-ink pages (pen): global Otsu separates strokes from everything,
      including crumpled-paper fold shadows, which adaptive thresholding
      would wrongly mark as ink (every fold is locally darker than its
      neighborhood).
    - Faint-pencil pages: Otsu's threshold lands between the paper's texture
      peaks and the strokes, so texture binarizes as ink while faint
      characters drop out entirely. Adaptive thresholding follows the local
      paper level instead, keeping light strokes and rejecting texture.

    The two regimes are told apart by the darkness gap between Otsu's ink
    class and the rest of the page: a large gap means strong, reliably
    separable ink (use Otsu); a small gap means faint pencil (use adaptive).
    """
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    t_otsu, otsu_binary = cv2.threshold(blur, 0, 255,
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = blur < t_otsu
    if ink.any() and (~ink).any():
        contrast = blur[~ink].mean() - blur[ink].mean()
        if contrast >= 50:
            return otsu_binary
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 51, 12)


def extract_line_crops(gray_image):
    """
    Extracts full horizontal sentence lines using horizontal projection
    profiles on the dust-cleaned ink binary.

    A line is a run of rows that carry meaningful ink. The per-row floor is
    relative to each band's peak: fold-shadow creases on crumpled paper cross
    a band with only a few ink pixels per row (a dash contributes 2-10px),
    while every row through real glyphs carries the sum of the stroke widths
    it crosses — so crease rows act as gaps and split or drop out, and
    whatever the characters' horizontal spacing, one band per written row
    remains. Each line's box spans the full x-extent of its band's ink.
    """
    binary = _ink_binary(gray_image)
    H, W = binary.shape

    # Remove dust components (fold-shadow dashes on crumpled paper, paper
    # specks) before profiling: real glyphs are far larger than H/60 at this
    # stage; upper-zone matras are unaffected because character segmentation
    # re-binarizes each line crop without this filter.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, num):
        if max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) < H // 60:
            binary[labels == i] = 0

    row_ink = (binary > 0).sum(axis=1)

    # Raw bands: contiguous runs of rows with any ink (small gaps bridged)
    raw = []
    start, gap = None, 0
    bridge0 = max(3, H // 200)
    for y in range(H):
        if row_ink[y] > 0:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap > bridge0:
                raw.append((start, y - gap + 1))
                start = None
    if start is not None:
        raw.append((start, H))

    # Split raw bands at internal crease valleys: rows far below the band's
    # peak ink are crease crossings, not text.
    bands = []
    min_gap = max(8, H // 150)
    for (y1, y2) in raw:
        seg = row_ink[y1:y2]
        floor = max(6, int(0.08 * seg.max()))
        sub_start, gap = None, 0
        for i, v in enumerate(seg):
            if v >= floor:
                if sub_start is None:
                    sub_start = i
                gap = 0
            elif sub_start is not None:
                gap += 1
                if gap > min_gap:
                    bands.append((y1 + sub_start, y1 + i - gap + 1))
                    sub_start = None
        if sub_start is not None:
            bands.append((y1 + sub_start, y1 + len(seg)))

    # Line boxes: x-extent of the band's ink; drop dust-height bands
    line_boxes = []
    min_h = max(10, H // 200)
    for (y1, y2) in bands:
        if y2 - y1 < min_h:
            continue
        strip = binary[y1:y2]
        xs = np.where(strip.any(axis=0))[0]
        if len(xs) == 0:
            continue
        line_boxes.append((int(xs.min()), int(y1), int(xs.max()) + 1, int(y2)))

    line_boxes.sort(key=lambda b: b[1])

    # Drop matra-only bands: a floating upper-zone matra part (ी hook, ं
    # candrabindu) separated from its word by a vertical gap forms its own
    # short band. It has no class of its own among the 46 DHCD classes, so
    # classifying it always yields a junk digit/bar - better to drop the
    # band. Real text lines sit well above 22% of the median line height.
    if line_boxes:
        med_line_h = float(np.median([b[3] - b[1] for b in line_boxes]))
        line_boxes = [b for b in line_boxes if (b[3] - b[1]) >= 0.22 * med_line_h]

    return line_boxes


# ============================================================================
#  SECTION 5: DATASET LOADER & TRAINER
# ============================================================================

class DevanagariDataset(Dataset):
    """
    PyTorch Dataset wrapper for Devanagari character datasets stored in folder hierarchy:
    root_dir/
      ├── character_1_ka/
      ├── character_2_kha/
      └── ...
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append((os.path.join(cls_folder, fname), self.class_to_idx[cls_name]))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('L')  # Convert to Grayscale
        if self.transform:
            image = self.transform(image)
        return image, label


def train_devanagari_model(data_dir, output_model_path="devanagari_cnn.pth", epochs=15, batch_size=128, lr=1e-3):
    """
    Train PyTorch Devanagari CNN model with augmentations and learning rate scheduling.
    """
    device = get_device()
    print(f"\n[TRAIN] Initializing Training on Device: {device} ({describe_device(device)})")
    
    # Image Augmentation Transformations
    train_transform = T.Compose([
        T.Resize((32, 32)),
        T.RandomRotation(degrees=10, fill=0),
        T.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Locate the training folder (accepts both "Train" and "train" layouts)
    train_path = data_dir
    for name in ("Train", "train"):
        if os.path.isdir(os.path.join(data_dir, name)):
            train_path = os.path.join(data_dir, name)
            break

    if not os.path.isdir(train_path):
        print(f"[ERROR] Training path not found: {train_path}")
        return None

    # Two dataset views over the same files: validation must never receive
    # the training augmentations (a shared transform would leak them into
    # the val metrics).
    train_full = DevanagariDataset(train_path, transform=train_transform)
    classes = train_full.classes
    class_to_idx = train_full.class_to_idx
    num_classes = len(classes)

    val_dir = next((os.path.join(data_dir, n) for n in ("Val", "val")
                    if os.path.isdir(os.path.join(data_dir, n))), None)

    if val_dir:
        train_dataset = train_full
        val_dataset = DevanagariDataset(val_dir, transform=val_transform)
    else:
        # No dedicated val folder: carve a stratified 10% hold-out from the
        # training set. The held-out Test/ split stays untouched so that
        # evaluate_accuracy.py reports an unbiased final accuracy.
        eval_view = DevanagariDataset(train_path, transform=val_transform)
        targets = np.array([label for _, label in train_full.samples])
        train_idx, val_idx = train_test_split(
            np.arange(len(targets)), test_size=0.1,
            random_state=42, stratify=targets)
        train_dataset = Subset(train_full, train_idx.tolist())
        val_dataset = Subset(eval_view, val_idx.tolist())
        print(f"[TRAIN] No val folder found - using a stratified 10% hold-out "
              f"({len(val_dataset)} images) carved from the training set.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"[TRAIN] Loaded {len(train_dataset)} training samples across {num_classes} classes.")
    
    model = DevanagariCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% "
              f"|| Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
              
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': classes,
                'class_to_idx': class_to_idx,
                'accuracy': val_acc
            }, output_model_path)
            print(f"  --> Saved new best checkpoint to '{output_model_path}' (Val Acc: {val_acc*100:.2f}%)")
            
    return model


# ============================================================================
#  SECTION 6: INFERENCE PIPELINE & DOCUMENT RECONSTRUCTION
# ============================================================================

class DevanagariOCRRecognizer:
    """
    Inference Engine loading custom PyTorch Devanagari CNN model and performing 
    character-level recognition on segmented document patches.
    """
    def __init__(self, model_path="devanagari_cnn.pth"):
        self.device = get_device()
        
        self.transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        
        if os.path.exists(model_path):
            print(f"[OCR Engine] Loading weights from '{model_path}'...")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.classes = checkpoint['classes']
            self.model = DevanagariCNN(num_classes=len(self.classes)).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"[OCR Engine] Model loaded successfully on {self.device}.")
        else:
            print(f"[WARN] Weight file '{model_path}' not found. Initializing untrained CNN model for testing.")
            self.classes = [v[0] for k, v in DHCD_LABEL_MAP.items()]
            self.model = DevanagariCNN(num_classes=46).to(self.device)
            self.model.eval()

    def predict_patch(self, crop_gray):
        """Perform forward pass on isolated character crop."""
        # Snap the crop to the training domain's pure black/white contrast:
        # real photos leave medium-gray, textured strokes after enhancement,
        # while DHCD crops are clean binarized glyphs.
        _, crop_gray = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        h, w = crop_gray.shape
        max_dim = max(h, w)

        # Match DHCD framing: training glyphs occupy only ~80-88% of their
        # 32x32 frame. A tight inference crop (100% occupancy) is out of
        # distribution, so pad each side by ~10% of the glyph size first.
        margin = int(round(0.10 * max_dim))
        padded = np.full((max_dim + 2 * margin, max_dim + 2 * margin), 255, dtype=np.uint8)
        padded[margin:margin + h, margin:margin + w] = crop_gray

        # Invert for neural network input (Text = White, Background = Black)
        padded_inv = cv2.bitwise_not(padded)

        pil_img = Image.fromarray(padded_inv)
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_img)
            probs = F.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        cls_name = self.classes[pred_idx.item()]
        # Translate folder label name to actual Unicode Devanagari character
        unicode_char = FOLDER_TO_UNICODE.get(cls_name, cls_name)
        return unicode_char, conf.item()


def group_into_lines(detections, line_threshold_ratio=0.6):
    """
    Groups character detections into top-to-bottom document lines and sorts 
    each line left-to-right.
    """
    if not detections:
        return []

    annotated = []
    for (bbox, text, conf) in detections:
        x1, y1, x2, y2 = bbox_to_xyxy(bbox)
        y_center = (y1 + y2) / 2.0
        height = max(y2 - y1, 1)
        annotated.append((y_center, height, bbox, text, conf))

    annotated.sort(key=lambda item: item[0])

    lines = []
    current_line = []
    current_y_sum = 0.0
    current_y_avg = 0.0

    for (y_center, height, bbox, text, conf) in annotated:
        if not current_line:
            current_line.append((bbox, text, conf))
            current_y_sum = y_center
            current_y_avg = y_center
        else:
            tolerance = line_threshold_ratio * height
            if abs(y_center - current_y_avg) <= tolerance:
                current_line.append((bbox, text, conf))
                current_y_sum += y_center
                current_y_avg = current_y_sum / len(current_line)
            else:
                current_line.sort(key=lambda item: bbox_to_xyxy(item[0])[0])
                lines.append(current_line)
                current_line = [(bbox, text, conf)]
                current_y_sum = y_center
                current_y_avg = y_center

    if current_line:
        current_line.sort(key=lambda item: bbox_to_xyxy(item[0])[0])
        lines.append(current_line)

    return lines


def run_pipeline(image_path, model_path="devanagari_cnn.pth", min_confidence=0.10, debug=False):
    """
    Full pipeline entry point for image inference.

    With debug=True, intermediate stage images (preprocessed page, ink
    binary, per-line stripped binary / ink projection / segmentation boxes,
    refined boxes, and every glyph crop with its prediction) are dumped to
    debug_output/ for inspection.
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] Input image not found: {image_path}")
        return

    print("\n" + "═" * 70)
    print("   RUNNING PYTORCH DEVANAGARI OCR PIPELINE")
    print("═" * 70)

    # Step 1: Load image
    orig = cv2.imread(image_path)
    if orig is None:
        print(f"[ERROR] cv2.imread failed on {image_path}")
        return

    print(f"[STEP 1] Image loaded ({orig.shape[1]}x{orig.shape[0]} px)")

    # Step 2: Preprocess Image
    enhanced_gray = full_preprocessing_pipeline(orig)
    display_img = cv2.resize(orig, (enhanced_gray.shape[1], enhanced_gray.shape[0]))
    print("[STEP 2] Applied enhancement chain (Shadow removal, Deskew, CLAHE, Line erase)")

    debug_dir = None
    if debug:
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "page_1_preprocessed.png"), enhanced_gray)
        cv2.imwrite(os.path.join(debug_dir, "page_2_ink_binary.png"), _ink_binary(enhanced_gray))

    # Step 3: Segment Lines & Characters
    line_boxes = extract_line_crops(enhanced_gray)
    print(f"[STEP 3] Detected {len(line_boxes)} candidate document line regions")

    # Step 4: PyTorch Recognition Engine
    recognizer = DevanagariOCRRecognizer(model_path=model_path)
    all_detections = []

    for line_idx, (lx1, ly1, lx2, ly2) in enumerate(line_boxes):
        line_crop = enhanced_gray[ly1:ly2, lx1:lx2]
        if line_crop.size == 0:
            continue

        line_tag = f"line{line_idx:02d}"
        char_boxes = segment_characters_from_line(line_crop, debug_dir=debug_dir, line_tag=line_tag)
        char_boxes = refine_character_cuts(line_crop, char_boxes, recognizer,
                                           debug_dir=debug_dir, line_tag=line_tag)
        med_line = float(np.median([b[3] - b[1] for b in char_boxes])) if char_boxes else 0.0

        for glyph_idx, (cx1, cy1, cx2, cy2) in enumerate(char_boxes):
            char_crop = line_crop[cy1:cy2, cx1:cx2]
            if char_crop.size == 0:
                continue

            pred_char, confidence = recognizer.predict_patch(char_crop)

            # Junk gate: a narrow digit-read fragment is a floating matra
            # tail (the vertical of ा, the hook of ी) that got separated
            # from its word. Matras have no output class among the 46, so
            # such a fragment always classifies as a digit; real written
            # digits are ~0.5+ of a glyph width and survive the filter.
            if med_line > 0 and pred_char in DIGIT_UNICODE and (cx2 - cx1) < 0.4 * med_line:
                continue

            if debug_dir:
                cv2.imwrite(os.path.join(
                    debug_dir, f"glyph_{line_tag}_{glyph_idx:02d}_{pred_char}_{confidence:.2f}.png"),
                    char_crop)

            # Global bounding box coordinates
            gx1 = lx1 + cx1
            gy1 = ly1 + cy1
            gx2 = lx1 + cx2
            gy2 = ly1 + cy2

            if confidence >= min_confidence:
                bbox_format = [[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]]
                all_detections.append((bbox_format, pred_char, confidence))

    print(f"[STEP 4] Recognized {len(all_detections)} character candidates")

    # Step 5: Group Detections into Reading Order
    text_lines = group_into_lines(all_detections, line_threshold_ratio=0.6)

    # Step 6: Export Results & Draw Visual Overlays
    output_txt = "extracted_devanagari.txt"
    output_img = "ocr_output_visual.png"

    with open(output_txt, "w", encoding="utf-8") as f:
        print("\n" + "═" * 70)
        print("  EXTRACTED TEXT OUTPUT")
        print("═" * 70)
        for i, line_items in enumerate(text_lines, 1):
            # Word spacing: a gap wider than ~0.9 glyph heights separates
            # words, not characters - emit a single space there.
            heights = [bbox_to_xyxy(item[0])[3] - bbox_to_xyxy(item[0])[1]
                       for item in line_items]
            med_h = float(np.median(heights)) if heights else 0.0
            parts, prev_x2 = [], None
            for (bbox, text, conf) in line_items:
                x1, y1, x2, y2 = bbox_to_xyxy(bbox)
                if prev_x2 is not None and med_h > 0 and (x1 - prev_x2) > 0.9 * med_h:
                    parts.append(" ")
                parts.append(text)
                prev_x2 = x2
            line_str = "".join(parts)
            print(f"  Line {i:02d}: {line_str}")
            f.write(line_str + "\n")

            for (bbox, text, conf) in line_items:
                x1, y1, x2, y2 = bbox_to_xyxy(bbox)
                color = (0, 220, 0) if conf > 0.8 else ((0, 220, 220) if conf > 0.5 else (0, 0, 220))
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 1)

    cv2.imwrite(output_img, display_img)
    print("═" * 70)
    print(f"[SUCCESS] Results exported to:\n  📄 Text File: {output_txt}\n  🖼 Visual Map: {output_img}\n")


# ============================================================================
#  SECTION 7: MAIN EXECUTION CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Devanagari OCR Engine & Trainer")
    parser.add_argument("--mode", type=str, default="infer", choices=["infer", "train"], help="Operation mode: infer or train")
    parser.add_argument("--image", type=str, default="my_handwriting.jpeg", help="Path to input image for OCR inference")
    parser.add_argument("--data_dir", type=str, default="DevanagariHandwrittenCharacterDataset", help="Directory containing the DHCD dataset for training")
    parser.add_argument("--model_path", type=str, default="devanagari_cnn.pth", help="Path to save/load model checkpoint")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for training mode")
    parser.add_argument("--debug", action="store_true", help="Dump intermediate segmentation/recognition images to debug_output/")

    args = parser.parse_args()

    if args.mode == "train":
        if not os.path.isdir(args.data_dir):
            print(f"[ERROR] Dataset directory not found: {args.data_dir}")
            sys.exit(1)

        train_devanagari_model(data_dir=args.data_dir, output_model_path=args.model_path, epochs=args.epochs)
    else:
        if not os.path.exists(args.image):
            print(f"[ERROR] Input image not found: {args.image}")
            sys.exit(1)

        run_pipeline(image_path=args.image, model_path=args.model_path, debug=args.debug)
