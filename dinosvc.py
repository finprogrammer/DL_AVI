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
 
from sklearn.metrics import confusion_matrix, f1_score, classification_report
#from custom_dataset_dino import CustomSSLDataset
 
from PIL import Image
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
#dinov2_vits14 = torch.hub.load('/home/hpc/iwfa/iwfa048h/.cache/torch/hub/facebookresearch_dinov2_main/dinov2', 'dinov2_vits14', source='local')
dinov2_vits14  = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
#dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')


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
 
from sklearn import svm
 
clf = svm.SVC(gamma='scale')
 
embedding_list = list(embeddings.values())
 
clf.fit(np.array(embedding_list).reshape(-1, 384), labels)
 
json_file_path = "/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Cover/Test.json"
with open(json_file_path, 'r') as f:
    data = json.load(f)
 
all_preds = []
all_labels = []
with torch.no_grad():
    for row in data:
        embedding = dinov2_vits14(load_image('/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Cover/' + row['file_name']))
        prediction = clf.predict(np.array(embedding[0].cpu()).reshape(1, -1))
        all_labels.append(row['label'])
        all_preds.append(prediction[0])
 
f1 = f1_score(all_labels, all_preds, average="weighted")
print(f1)
cm = confusion_matrix(all_labels, all_preds)
print(cm)
tp, fn, fp, tn = cm.ravel()
print("TP = ", tp, "FP = ", fp, "FN = ", fn, "TN = ", tn)
print('\nClassification Report\n')
print(classification_report(all_labels, all_preds, zero_division=0))