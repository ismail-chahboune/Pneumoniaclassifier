
"""
Created on Fri Nov  7 21:31:45 2025

@author: Ismail
"""

import os
from PIL import Image 
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        
        for label in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(root_dir, label)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(0 if label == 'NORMAL' else 1)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

print("Loading datasets...")

train_dataset = PneumoniaDataset(root_dir='chest_xray/train', transform=transform)
test_dataset = PneumoniaDataset(root_dir='chest_xray/test', transform=transform)
val_dataset = PneumoniaDataset(root_dir='chest_xray/val', transform=transform)

print(f"Dataset sizes:")
print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


print("Initializing ResNet18 model...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: NORMAL and PNEUMONIA
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


print("Starting training...")
num_epochs = 10
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    
    model.train() 
    running_loss = 0.0 
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
       
        loss.backward() 
        optimizer.step()
        
        running_loss += loss.item()
    
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    
    model.eval()
    val_labels = []
    val_preds = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
    
    
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_accuracies.append(val_accuracy)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Training Loss: {avg_loss:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print("-" * 50)


print("\nEvaluating on test set...")
model.eval()
test_labels = []
test_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())


test_accuracy = accuracy_score(test_labels, test_preds)
print(f" Final Test Accuracy: {test_accuracy:.4f}")


print("\n Detailed Classification Report:")
print(classification_report(test_labels, test_preds, 
                          target_names=['NORMAL', 'PNEUMONIA']))

torch.save(model.state_dict(), 'pneumonia_classifier.pth')
print(" Model saved as 'pneumonia_classifier.pth'")


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-', label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, 'g-', label='Validation Accuracy')
plt.title('Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n Training completed successfully!")
print(f" Final model achieved {test_accuracy*100:.2f}% accuracy on test set")
