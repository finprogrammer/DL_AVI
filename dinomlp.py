import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random

import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, classification_report

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


num_epochs = 20
# Load DINOv2 model
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

# Define your MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Replace the classification head of the DINOv2 model with the MLP layers
dinov2_vits14.fc = MLP(input_size=384, hidden_size=256, num_classes=2)

# Function to load image
transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])

def load_image(img: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.open(img)
    transformed_img = transform_image(img)[:3].unsqueeze(0)
    return transformed_img

def compute_embeddings():
    """
    Create an index that contains all of the images in the specified list of files.
    """
    all_embeddings = []
    json_file_path = "/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Cover/train.json"
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    y = [row["label"] for row in data]
    all_embeddings = {}
    with torch.no_grad():
        for row in data:
            embeddings = dinov2_vits14(load_image('/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Cover/' + row['file_name']))
            all_embeddings[row['file_name']] = np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist()
   
    with open("all_embeddings.json", "w") as f:
        f.write(json.dumps(all_embeddings))
 
    return all_embeddings, y

embeddings, labels = compute_embeddings()

# Load validation data
json_file_path = "/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Cover/val.json"
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Initialize MLP model
mlp_model = MLP(input_size=384, hidden_size=256, num_classes=2).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# Training loop
mlp_model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for row in data:
        inputs = load_image('/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Cover/' + row['file_name']).to(device)
        labels = torch.tensor([row['label']]).to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = mlp_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(data)}")

# Evaluation
mlp_model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for row in data:
        inputs = load_image('/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Cover/' + row['file_name']).to(device)
        outputs = mlp_model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_labels.append(row['label'])
        all_preds.append(predicted.item())

# Calculate F1-score and confusion matrix
f1 = f1_score(all_labels, all_preds, average="weighted")
print(f"F1 Score: {f1}")
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)
tp, fn, fp, tn = cm.ravel()
print("TP = ", tp, "FP = ", fp, "FN = ", fn, "TN = ", tn)
print('\nClassification Report\n')
print(classification_report(all_labels, all_preds, zero_division=0))
