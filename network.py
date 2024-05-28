from collections import OrderedDict
from pytorch_lightning import LightningModule
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    fbeta_score,
)
import torch
import torch.nn.functional as F
import torch.optim
import torchvision
import csv
import os
import shared
from torch.utils.tensorboard import SummaryWriter
import math
import cv2
from torchvision.models import resnet18
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import monai
from torchvision.transforms import v2 as T
from pytorch_grad_cam.utils.image import show_cam_on_image
from config_loader import load_config
from extract_gradcam import grad_cam
from config_loader import load_config

mode = "train_2"  # Change this to "debug" when debugging
CONFI = load_config(mode)

if mode == "train":
    DEVICES = 1 if torch.cuda.is_available() else None
else:
    DEVICES = "auto"

Indices = []
Predictions = []
Indices_Def = []
Prediction_Def = []
FP = []
FN = []
iteration_count = 0
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

class DeepCNN(LightningModule):
    def __init__(
        self,
        backbone=None,
        backbone_out=1000,
        num_classes=None,
        beta=CONFI['BETA'],
        optimizer=None,
        lr_schedulers=None,
        output_activation=None,
        loss_func=None,
        views=None,
        labels=None,
        dino=False,
        Val=False,
        num_layers = 1,
        num_neurons_2 = 256,
        dropout = 0.1,
        Gradcam_save_dir =None,
        Reconstructed_image_save_dir =None,
        ROOT_DIR =None
    ):  #
        super().__init__()
        self.backbone = backbone
        self.dropout = dropout
        self.optimizer = optimizer
        self.output_activation = output_activation
        self.loss_func = loss_func
        self.num_classes = num_classes
        self.views = views
        self.beta = beta
        self.labels = labels
        self.num_layers = num_layers
        self.num_neurons_2 = num_neurons_2
        self.Val = Val
        self.Gradcam_save_dir = Gradcam_save_dir
        self.Reconstructed_image_save_dir = Reconstructed_image_save_dir
        self.ROOT_DIR = ROOT_DIR
        self.lr_schedulers = lr_schedulers
        self.dino = dino
        self.total_iterations = 0
        self.image_count = {}
        skip_optuna = CONFI["SKIP_OPTUNA"]
        if not skip_optuna:
            self.dropout = torch.nn.Sequential(torch.nn.Dropout(p=dropout))
        else:    
            self.dropout = torch.nn.Sequential(torch.nn.Dropout(p=CONFI['DROPOUT']))
        self.late_fc_d = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc2",
                        torch.nn.Linear(
                            in_features=len(self.views) * backbone_out,
                            out_features=num_classes,
                        ),
                    ),
                ]
            )
        )
        self.gradcam = monai.visualize.GradCAM(
            nn_module=self.backbone,
            target_layers="features.7.2",  # self.backbone.features[7][2]
        )
        #self.classifier2 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(256, 1))
        if CONFI["Model"] == "Dinov2":
            layers = []              
            for i in range(1, num_layers):
                layers.append(
                torch.nn.Linear(num_neurons_2[i - 1], num_neurons_2[i])
                )  # Connecting layers with different neuron counts
                layers.append(torch.nn.ReLU())
            input_size = num_neurons_2[-1]
            layers.append(torch.nn.Linear(input_size, 1))
            #layers.append(torch.nn.ReLU())
            self.classifier2 = torch.nn.Sequential(*layers)

    def forward(self, view):
        if CONFI['Model'] == "Dinov2":
            x = self.backbone(view)
            #if self.training:
            x = self.dropout(x)
            x = self.classifier2(x)
            return x
        else:
            x = self.backbone(view)
            x = self.late_fc_d(x)
            return x

    def training_step(self, train_batch):
        # for param in self.backbone.classifier.parameters():
        #     param.requires_grad = True

        if CONFI['Model'] == "Dinov2":
            view = self.views[0]
            x = train_batch[view]
            target_ = train_batch["label"]
            pred = self(x)
            pred = pred.squeeze()
            #pred_scalar = pred.item()
            #pred = torch.tensor([pred_scalar])             
            self.total_iterations += 1
            train_loss = self.loss_func(pred, target_)
            self.log("train_loss", train_loss)
            target = target_.cpu().numpy()
            pred_ = (pred > 0.5).float()
            pred = pred_.cpu().detach().numpy()
        else:
            view = self.views[0]
            x = train_batch[view]
            target_ = train_batch["label"]
            pred = self(x)
            pred_ = self.output_activation(pred)
            self.total_iterations += 1
            train_loss = self.loss_func(pred_, target_)
            self.log("train_loss", train_loss)
            target = target_.cpu().argmax(axis=1).numpy()
            pred = pred_.cpu().argmax(axis=1).numpy()

        if self.total_iterations % 500 == 0 and CONFI['Extract_reconstructed_image']:
            target_1 = target_.cpu().argmax(axis=1).numpy()
            pred_1 = pred_.cpu().argmax(axis=1).numpy()
            #target_1 = torch.from_numpy(target_1)
            #pred_1 = torch.from_numpy(pred_1)
            folder = self.Reconstructed_image_save_dir
            os.makedirs(folder, exist_ok=True)
            mismatch_indices = np.where(pred_1 != target_1)[0]
            mismatch_indices = torch.as_tensor(mismatch_indices)
            image_name = train_batch["image_name_string"]
            #global iter
            self.total_iterations += 1
            raw_images = x[mismatch_indices]
            #targets_mis = target_[mismatch_indices]
            raw_images = raw_images
            for i in range(len(mismatch_indices)):
                img_n = raw_images[i].squeeze(0)
                img_np = img_n.cpu().numpy()
                max_value = np.max(img_np)
                img = img_n / max_value
                transform_to_pil = T.Compose([T.ToPILImage()])
                img = transform_to_pil(img)
                filename = image_name[mismatch_indices[i]]
                filename = filename.replace('images/', '')
                filename = filename.replace('.jpg', '')
                log = CONFI["logger_version"]
                #target_img = str(targets_mis[i])
                filename = f"class0__{log}__{filename}.jpg"#str(target_img + filename)
                save_path_img = os.path.join(folder, filename)
                img.save(save_path_img, format="JPEG")


        accuracy = accuracy_score(target, pred)
        f1 = f1_score(target, pred)
        recall = recall_score(target, pred)
        f1_micro = f1_score(target, pred, average="micro")
        self.log("train_loss", train_loss)
        self.log("f1_train", f1)
        self.log("recall_train", recall)
        self.log("f1_micro_train", f1_micro)
        self.log("train_accuracy", accuracy)
        return train_loss

    def validation_step(self, val_batch):

        if CONFI['Model'] == "Dinov2":
            view = self.views[0]
            x = val_batch[view]
            x2 = val_batch[view]
            target_ = val_batch["label"]
            pred = self(x)
            pred = pred.squeeze()            
            val_loss = self.loss_func(pred, target_)
            self.log("val_loss", val_loss)
            pred = (pred > 0.5).float()
            target = target_.cpu().numpy()
            pred = pred.cpu().detach().numpy()
        else:
            view = self.views[0]
            x = val_batch[view]
            x2 = val_batch[view]
            target_ = val_batch["label"]
            pred = self(x)
            pred_ = self.output_activation(pred)
            val_loss = self.loss_func(pred_, target_)
            self.log("val_loss", val_loss)
            target = target_.cpu().argmax(axis=1).numpy()
            pred = pred_.cpu().argmax(axis=1).numpy()

        accuracy = accuracy_score(target, pred)
        f1 = f1_score(target, pred)
        f1_label = f1_score(target, pred, pos_label=0)
        f1_notlabel = f1_score(target, pred, pos_label=1)
        f1_custom = (f1_label * 0.7) + (f1_notlabel * 0.3)
        f1_micro = f1_score(target, pred, average="micro")
        recall = recall_score(target, pred)
        f1_beta_label = fbeta_score(target, pred, pos_label=0, beta=self.beta)
        self.log("vall_acc", accuracy, on_epoch=True)
        self.log("f1", f1, on_epoch=True)
        self.log("f1_micro", f1_micro, on_epoch=True)
        self.log("recall", recall, on_epoch=True)
        self.log("f1_custom", f1_custom, on_epoch=True)
        self.log("f1_label", f1_label, on_epoch=True)
        self.log("f1_notlabel", f1_notlabel, on_epoch=True)
        self.log("f1_beta_label", f1_beta_label, on_epoch=True)

        if CONFI['Extract_Gradcam']:
            target_1 = target_.cpu().argmax(axis=1).numpy()
            pred_1 = pred_.cpu().argmax(axis=1).numpy()
            match_indices = np.where((pred_1 == target_1) & (pred_1 == 0))
            #match_indices = np.where(match_indices == 0)
            match_indices = torch.as_tensor(match_indices)            
            #if pred_1[0] == target_1[0] == 0:
            global iteration_count
            iteration_count += 1
            instance = grad_cam(
            iter = iteration_count,
            backbone = self.backbone,
            logger_version=CONFI["logger_version"], 
            x2=x2,
            match_indices=match_indices, 
            Gradcam_save_dir=self.Gradcam_save_dir, 
            val_batch=val_batch,
            ROOT_DIR = self.ROOT_DIR
            )
            instance.grad_image()
        return val_loss

    def test_step(self, test_batch, batch_idx):

        if CONFI['Model'] == "Dinov2":
            view = self.views[0]
            x = test_batch[view]
            target_ = test_batch["label"]
            pred = self(x)
            pred = (pred > 0.5).float()
            pred = pred.squeeze()
            test_loss = self.loss_func(pred, target_)
            self.log("test_loss", test_loss)
            target = target_.cpu().numpy()
            pred = pred.cpu().detach().numpy()
        else:
            view = self.views[0]
            x = test_batch[view]
            target_ = test_batch["label"]
            pred = self(x)
            pred_ = self.output_activation(pred)
            test_loss = self.loss_func(pred_, target_)
            self.log("test_loss", test_loss)
            target = target_.cpu().argmax(axis=1).numpy()
            pred = pred_.cpu().argmax(axis=1).numpy()
            accuracy = accuracy_score(target, pred)
            self.log("test_acc", accuracy)

        return test_loss

    def predict_step(self, batch):

        if CONFI['Model'] == "Dinov2":
            view = self.views[0]
            x = batch[view]
            target_ = batch["label"]
            pred = self(x)
            pred = pred.squeeze()
            target = target_.cpu().numpy()
            pred_ = (pred > 0.5).float()
            y = pred_.cpu().detach()
        else:
            view = self.views[0]
            x = batch[view]
            y = self(x)
            target = batch["label"]
            pred = self.output_activation(y)
            target_1 = target.cpu().argmax(axis=1).numpy()
            pred_1 = pred.cpu().argmax(axis=1).numpy()
            target = torch.from_numpy(target_1)
            pred = torch.from_numpy(pred_1)
            Indices.append(0) if (target == pred) else Indices.append(1)
            Predictions.append(0) if (target == pred) else Predictions.append(pred)
            FP.append(1) if ((pred_1 != target_1) & (target_1 == 1)) else FP.append(0)
            FN.append(1) if ((pred_1 != target_1) & (target_1 == 0)) else FN.append(0)
            (
                Indices_Def.append(1)
                if ((target == pred) & (target == 0))
                else Indices_Def.append(0)
            )
            (
                Prediction_Def.append(pred)
                if ((target == pred) & (target == 0))
                else Prediction_Def.append(0)
            )
        return y

    def configure_optimizers(self):
        if self.optimizer:
            optimizer = self.optimizer  # (self.parameters(), lr=self.lr)
            lr_schedulers = self.lr_schedulers
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=CONFI['LR'], weight_decay=CONFI['WD']
            )  # weight_decay=0.05)
            lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.1, patience=CONFI['PATIENCE'], verbose=True
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_schedulers,
                "monitor": CONFI['QUANTITY_TO_OPTIMIZE'],  
                "interval": "epoch",
                "frequency": 1,
            },
        }

DeepCNN.example_input_array = torch.randn((32, 3, CONFI['W'], CONFI['H']))


