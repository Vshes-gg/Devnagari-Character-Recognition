# Devanagari Character Recognition — CNN (PyTorch) — Final Plan (approved)

Optimizer decision locked: **Adam** (lr 1e-3, weight decay 1e-4); SGD kept as an optional `--optimizer sgd` comparison flag only.

## Execution steps
1. **Recreate venv with Python 3.12** — delete `.venv` (holds nothing but pip), recreate with `/opt/homebrew/bin/python3.12`, upgrade pip.
2. **Write `requirements.txt`** — torch, torchvision, numpy, opencv-python, Pillow, scikit-learn, matplotlib, tqdm + commented easyocr. Install into the new venv (background while docs are written).
3. **Write `cnn_ocr.py`** — self-contained PyTorch system, tutorial-commented:
   - `DevanagariCNN`: 4 conv blocks (1→32→64→128→256, Conv3×3-BN-ReLU ×2 + MaxPool), AdaptiveAvgPool → Dropout(0.5) → Linear → 46 classes; 32×32 grayscale input.
   - Data: ImageFolder over DHCD `Train/`+`Test/`; two ImageFolder instances (train-aug vs eval transforms) Subsett ed by stratified 90/10 split (no transform leak); mild augmentation (±5° rotation, translate/scale, white fill).
   - Training: Adam + cosine annealing, early stopping (patience 7), best checkpoint → `checkpoints/best_model.pt` (state_dict + class names + img_size + val_acc), loss/acc curves PNG, `--limit-batches` smoke flag, macOS-safe DataLoader workers.
   - Evaluation: Test accuracy, per-class report txt, 46×46 confusion matrix PNG.
   - Page-level `predict`: compact preprocessing (upscale/shadow/CLAHE/sharpen), Otsu binarize, horizontal projection → lines, shirorekha strip + vertical projection → glyph columns, glyph bbox union incl. matras, letterbox crops, batched classification, space heuristic, `result.txt` (Devanagari Unicode via class→Unicode map) + `result_output.png` (green/cyan/red legend, ASCII translit labels since cv2.putText can't render Devanagari).
   - CLI: `train` / `evaluate` / `predict`.
4. **Write `explanation.md`** — block-by-block walkthrough (model anatomy, training-step mechanics, Adam vs SGD, segmentation algorithm, inference details).
5. **Write `README.md`** — overview, DHCD dataset description, python3.12 setup, training walkthrough + hyperparameter table, evaluation/prediction usage, full class→Unicode table, tuning tips, domain-gap caveat.
6. **Smoke test** — `--help`, class count = 46 via ImageFolder, forward/backward through model on MPS/CPU, 5-batch mini training run.
7. **Final response** — training walkthrough (commands, what to watch, tuning tips).

No changes to `ocr_pipeline.py`; no full training run.