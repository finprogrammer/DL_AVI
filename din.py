from sklearn import svm
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torchvision
import random
import torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
from optuna.trial import TrialState
from PIL import Image
import torchvision.models as models
from torchsummary import summary
import torchvision.transforms as transforms
import torchvision.transforms as T
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor    
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import loggers, Trainer, callbacks
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models import densenet121, densenet201
from dataloader import MultiViewDataset, datatest, dataval#,MultiViewDataset2 data, , tar
from loss_function import WeightedCrossEntropyLoss, WeightedBinaryCrossEntropyLoss
from network import  DeepCNN, indices,  targetlist, predlist, name_list, indices_def, predlist_def, indices_val, predlist_val, indices_def_val, predlist_def_val#tb_writer,
from utils.data_splitting import random_split_dataset
from utils.visualisations import visualize_dataloader_for_class_balance
import CONFI
from misclassification import misclassification
from network import DeepCNN
from eval import evaluate
from torch.optim import RMSprop
from torch.optim import Adagrad
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
import cProfile
#from mmpretrain import get_model
from tqdm import tqdm
import json




class Dino:
    def __init__(self, model, files):
        self.model = model
        self.files = files
        #self.labels = labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.transform_image = self._get_transform()

    def compute_embeddings(self):
        all_embeddings = {}

        with torch.no_grad():
            for i, file in enumerate(tqdm(self.files)):
                embeddings = self.model(self.load_image(file).to(self.device))
                all_embeddings[file] = np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist()

        with open("all_embeddings.json", "w") as f:
            f.write(json.dumps(all_embeddings))

        return all_embeddings




def main():
    ROOT_DIR = os.path.join("/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Cover")
    print(ROOT_DIR)
    train_csv_path = os.path.join(ROOT_DIR, "train.csv")
    test_csv_path = os.path.join(ROOT_DIR, "test.csv")
    val_csv_path = os.path.join(ROOT_DIR, "val.csv")

    views = ["file_name"]
    #views = ["1", "2", "3",  "4", "5", "6"]

    # Order matter for labels
    labels = ["label", "~label"]
    batch_size = 32
    num_classes = len(labels)

    train_dataset = MultiViewDataset(train_csv_path, views=views, labels=labels, base_dir=ROOT_DIR, transform=True, normalize=True, Test=False, Val= False, Feature_extraction=False, pil_image_mode=CONFI.IMAGE_MODE)#, pil_image_mode="L"
    print(f"Train Normalize: {train_dataset.normalize}")
    print(f"Train Transform: {train_dataset.transform}")
    print(f"Train Feature_extraction: {train_dataset.Feature_extraction}")
    val_dataset = MultiViewDataset(val_csv_path, views=views, labels=labels, base_dir=ROOT_DIR, transform=False, normalize=True, Test=False, Val= True, Feature_extraction=False, pil_image_mode=CONFI.IMAGE_MODE)#, pil_image_mode="L"
    print(f"Val Normalize: {val_dataset.normalize}")
    print(f"Val Transform: {val_dataset.transform}")
    print(f"Val Feature_extraction: {val_dataset.Feature_extraction}")
    test_dataset = MultiViewDataset(test_csv_path, views=views, labels=labels, base_dir=ROOT_DIR, transform=False, normalize=True, Test=True, Val= False, Feature_extraction=False, pil_image_mode=CONFI.IMAGE_MODE)#, pil_image_mode="L"
    print(f"Test Normalize: {test_dataset.normalize}")
    print(f"Test Transform: {test_dataset.transform}")
    print(f"Test Feature_extraction: {test_dataset.Feature_extraction}")
    

    class_counts = np.ones(num_classes)
    for i, val in enumerate(train_dataset.data):
        label = np.asarray(val["label"])
        # assuming one-hot encoding
        class_ = np.argmax(label)
        class_counts[class_] += 1


    sample_weights = np.zeros(len(train_dataset.data))
    for i, val in enumerate(train_dataset.data):
        label = np.asarray(val["label"])
        # assuming one-hot encoding
        class_ = np.argmax(label)
        sample_weights[i] = 1/class_counts[class_]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    mv_train_loader = DataLoader(train_dataset, sampler=sampler, shuffle=False, batch_size=batch_size, num_workers=4, drop_last=True)
    mv_val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4, drop_last=True)
    mv_test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=4)    
    
    dinov2_vits14  = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

    fusion_model = Dino(model = dinov2_vits14, files = mv_train_loader)
    all_embeddings = fusion_model.compute_embeddings()

    clf = svm.SVC(gamma='scale')
    x1, y = mv_train_loader
    views = []
    target = mv_train_loader["label"]
    #y = [labels[file] for file in files]

    print(len(all_embeddings.values()))

    embedding_list = list(all_embeddings.values())

    clf.fit(np.array(embedding_list).reshape(-1, 384), target)

if __name__ == "__main__":
    main()