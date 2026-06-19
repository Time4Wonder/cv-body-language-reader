import torch
import torch.nn as nn
from torchvision import models
from dataset import get_dataloaders
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Params:
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

# model for tuning
model = models.resnet18(weights = "DEFAULT")
model.fc = nn.Linear(512, NUM_CLASSES)
model = model.to(device)

print(f"Modell läuft auf: {next(model.parameters()).device}")

# training configs
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# training dataset
train_loader, test_loader = get_dataloaders(BATCH_SIZE)

print(f"Training-Batches pro Epoche: {len(train_loader)}")
print(f"Test-Batches: {len(test_loader)} ")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoche {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# Validation Part:
correct = 0
total = 0 

model.eval()
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    logits = model(images)
    predictions = torch.argmax(logits, dim=1)
    correct += (predictions == labels).sum().item()
    total += labels.size(0)

accuracy = correct/ total * 100
print(f"Accuracy: {accuracy}")


os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/resnet_fer2013.pth")
print("Gespeichert unter models/resnet_fer2013.pth")

