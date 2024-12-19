import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

sns.set_style('darkgrid')

# Location of main dataset
base_dir = "E:\\IPTF\\dataset"

# Define train and test folders paths.
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Number of classes
classes = os.listdir(train_dir)
num_classes = len(classes)

# Plot dataset distribution
counts = []
for class_name in classes:
    class_path = os.path.join(train_dir, class_name)
    count = len(os.listdir(class_path))
    counts.append((class_name, count))
counts = pd.DataFrame(counts, columns=['Class_Names', 'Counts'])
plt.figure(figsize=(15, 15))
ax = sns.barplot(data=counts, y='Class_Names', x='Counts')
ax.set_title('Counts of each class', fontsize=25, fontweight='bold')
plt.show()

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

# Updated data augmentation for training
train_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # Add slight rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Enhance color variations
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Testing transformations
test_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load datasets
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# Load pretrained VGG19 model
pretrained_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

# Freeze layers until block5_conv1
set_trainable = False
for name, param in pretrained_model.named_parameters():
    if "features.27" in name:
        set_trainable = True
    param.requires_grad = set_trainable

# Modify classifier
pretrained_model.classifier = nn.Sequential(
    nn.Linear(25088, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes),
    nn.LogSoftmax(dim=1)
)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model = pretrained_model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pretrained_model.parameters(), lr=0.0001)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    print('Training completed.')

# Train the model for 20 epochs
train_model(pretrained_model, train_loader, criterion, optimizer, num_epochs=20)

# Save the trained model
torch.save(pretrained_model.state_dict(), 'MyModel.pth')

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

# Load and evaluate the saved model
best_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
best_model.classifier = pretrained_model.classifier
best_model.load_state_dict(torch.load('MyModel.pth'))
best_model = best_model.to(device)

evaluate_model(best_model, test_loader)
