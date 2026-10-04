import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from devanagari_ocr import DevanagariCNN, DevanagariDataset, FOLDER_TO_UNICODE, get_device, describe_device

def run_evaluation(data_dir="DevanagariHandwrittenCharacterDataset", model_path="devanagari_cnn.pth", batch_size=128):
    device = get_device()
    print(f"\n[EVALUATION] Running on Device: {device} ({describe_device(device)})")
    
    # Auto-detect test or val folder
    test_path = None
    for name in ["test", "Test", "val", "Val"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            test_path = p
            break
            
    if not test_path:
        print(f"[ERROR] Test folder not found inside '{data_dir}'.")
        return

    # Image transformation standard
    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    test_dataset = DevanagariDataset(test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model = DevanagariCNN(num_classes=len(test_dataset.classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    correct = 0
    top3_correct = 0
    total = 0

    class_correct = {i: 0 for i in range(len(test_dataset.classes))}
    class_total = {i: 0 for i in range(len(test_dataset.classes))}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Top-1 Accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

            # Top-3 Accuracy
            _, top3_preds = torch.topk(outputs, 3, dim=1)
            for i in range(labels.size(0)):
                if labels[i] in top3_preds[i]:
                    top3_correct += 1

            total += labels.size(0)

            for l, p in zip(labels, preds):
                if l == p:
                    class_correct[l.item()] += 1
                class_total[l.item()] += 1

    overall_acc = (correct / total) * 100
    top3_acc = (top3_correct / total) * 100

    print("=" * 65)
    print("           DEVANAGARI OCR MODEL EVALUATION REPORT")
    print("=" * 65)
    print(f"Total Test Samples Evaluated : {total}")
    print(f"Top-1 Overall Accuracy      : {overall_acc:.2f}%")
    print(f"Top-3 Overall Accuracy      : {top3_acc:.2f}%\n")

    print("Per-Class Accuracy Breakdown:")
    print("-" * 65)
    for idx, cls_name in enumerate(test_dataset.classes):
        acc = (class_correct[idx] / class_total[idx]) * 100 if class_total[idx] > 0 else 0.0
        unicode_char = FOLDER_TO_UNICODE.get(cls_name, cls_name)
        print(f"Class {idx:02d} | {cls_name:<24} ({unicode_char}) : {acc:.2f}% ({class_correct[idx]}/{class_total[idx]})")

if __name__ == "__main__":
    run_evaluation()