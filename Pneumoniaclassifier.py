
import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
 
# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATA_ROOT = None            
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
CLASSES = ["NORMAL", "PNEUMONIA"]
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_FRACTION = 0.15
USE_PRETRAINED = True       
 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
 
# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
def find_data_root(base="/kaggle/input"):
    for dirpath, dirnames, _ in os.walk(base):
        if {"train", "test"}.issubset(dirnames) and os.path.isdir(os.path.join(dirpath, "train", "NORMAL")):
            return dirpath
    raise FileNotFoundError("Could not find a folder containing train/NORMAL and test/. "
                            "Add the 'Chest X-Ray Images (Pneumonia)' dataset to the notebook, "
                            "or set DATA_ROOT manually.")
 
 
def list_images(folder):
    paths, labels = [], []
    for label_idx, cls in enumerate(CLASSES):
        class_dir = os.path.join(folder, cls)
        if not os.path.isdir(class_dir):
            continue
        for name in sorted(os.listdir(class_dir)):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(class_dir, name))
                labels.append(label_idx)
    return paths, labels
 
 
class PneumoniaDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform
 
    def __len__(self):
        return len(self.paths)
 
    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
 
 
root = DATA_ROOT or find_data_root()
print("Data root:", root)
 
train_p, train_l = list_images(os.path.join(root, "train"))
val_p, val_l = list_images(os.path.join(root, "val"))
test_p, test_l = list_images(os.path.join(root, "test"))
 
# Pool official train + val, then make a bigger stratified validation split
all_p, all_l = train_p + val_p, train_l + val_l
train_p, val_p, train_l, val_l = train_test_split(
    all_p, all_l, test_size=VAL_FRACTION, stratify=all_l, random_state=SEED)
 
counts = np.bincount(train_l, minlength=2)
print(f"Train: {len(train_p)} (NORMAL {counts[0]}, PNEUMONIA {counts[1]}) | "
      f"Val: {len(val_p)} | Test: {len(test_p)} (official test set, never used for training)")
 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    normalize,
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize,
])
 
pin = device.type == "cuda"
train_loader = DataLoader(PneumoniaDataset(train_p, train_l, train_tf), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=pin)
val_loader = DataLoader(PneumoniaDataset(val_p, val_l, eval_tf), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=2, pin_memory=pin)
test_loader = DataLoader(PneumoniaDataset(test_p, test_l, eval_tf), batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2, pin_memory=pin)
 
# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
print("Initializing ResNet18...")
weights = models.ResNet18_Weights.IMAGENET1K_V1 if USE_PRETRAINED else None
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
 
# Class weights computed from the training data (rarer class counts more)
class_weights = torch.tensor(len(train_l) / (2.0 * counts), dtype=torch.float32).to(device)
print("Class weights (NORMAL, PNEUMONIA):", [round(float(w), 3) for w in class_weights])
 
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
 
 
def evaluate(loader):
    model.eval()
    total_loss, y_true, y_pred, y_prob = 0.0, [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            y_true += labels.cpu().tolist()
            y_pred += outputs.argmax(1).cpu().tolist()
            y_prob += probs.cpu().tolist()
    return total_loss / len(loader.dataset), np.array(y_true), np.array(y_pred), np.array(y_prob)
 
 
# ----------------------------------------------------------------------------
# TRAINING
# ----------------------------------------------------------------------------
print("Starting training")
history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_bal_acc": []}
best_bal_acc, best_epoch = -1.0, 0
best_path = os.path.join(OUT_DIR, "pneumonia_classifier.pth")
 
for epoch in range(NUM_EPOCHS):
    model.train()
    running = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
    train_loss = running / len(train_loader.dataset)
 
    val_loss, yt, yp, _ = evaluate(val_loader)
    val_acc = accuracy_score(yt, yp)
    val_bal = balanced_accuracy_score(yt, yp)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_bal_acc"].append(val_bal)
 
    flag = ""
    if val_bal > best_bal_acc:
        best_bal_acc, best_epoch = val_bal, epoch + 1
        torch.save(model.state_dict(), best_path)
        flag = "  <- best so far, saved"
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
          f"val acc {val_acc:.4f} | val balanced acc {val_bal:.4f}{flag}")
 
# ----------------------------------------------------------------------------
# TEST (official test set, best epoch)
# ----------------------------------------------------------------------------
print(f"\nLoading best model (epoch {best_epoch}) and evaluating on the test set")
model.load_state_dict(torch.load(best_path, map_location=device))
_, y_true, y_pred, y_prob = evaluate(test_loader)
 
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, pos_label=1)
rec = recall_score(y_true, y_pred, pos_label=1)                 # sensitivity (PNEUMONIA)
spec = recall_score(y_true, y_pred, pos_label=0)                # specificity (NORMAL)
f1 = f1_score(y_true, y_pred, pos_label=1)
auc = roc_auc_score(y_true, y_prob)
 
print("\n=== TEST RESULTS ===")
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows = true, cols = predicted):\n", cm)
print(f"AUC: {auc:.4f}")
 
# ----------------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"], "b-", label="Training loss")
axes[0].plot(history["val_loss"], "r-", label="Validation loss")
axes[0].set_title("Loss over epochs"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
axes[1].plot(history["val_acc"], "g-", label="Validation accuracy")
axes[1].plot(history["val_bal_acc"], "m--", label="Validation balanced accuracy")
axes[1].set_title("Validation accuracy over epochs"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_history.png"), dpi=300, bbox_inches="tight")
plt.show()
 
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues", values_format="d")
plt.title("Confusion matrix (test set)")
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()
 
print("\n=== SUMMARY ===")
print(f"Images -> train: {len(train_p)} | val: {len(val_p)} | test: {len(test_p)}")
print(f"Best epoch: {best_epoch}/{NUM_EPOCHS}")
print(f"Test accuracy: {acc * 100:.2f}% | Recall (pneumonia): {rec * 100:.2f}% | "
      f"Specificity (normal): {spec * 100:.2f}% | Precision: {prec * 100:.2f}% | "
      f"F1: {f1:.3f} | AUC: {auc:.3f}")
print(f"Saved: {best_path}, training_history.png, confusion_matrix.png")
