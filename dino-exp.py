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

import optuna
import logging
import sys

from sklearn.metrics import confusion_matrix, f1_score, classification_report
from tqdm import tqdm
from custom_dataset_dino import CustomSSLDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

tensorcount = 0
def objective(trial):
    lr = trial.suggest_float("lr", 1e-6, 1e-5)
    batch_size = trial.suggest_categorical("bs", [4, 6, 8, 16, 32])
    #backbone = trial.suggest_categorical("backbone", ['dinov2_vits14', 'dinov2_vitb14'])
    dropout = trial.suggest_float("dropout_p", 0.0, 0.5)
    weight_decay = trial.suggest_float("l2_lambda", 0.001, 0.01, log=True)
    # Define a list of your desired values
    patience = trial.suggest_categorical("patience", [5, 6, 7, 8, 9, 10])
    improve_patience = trial.suggest_categorical("improve_patience", [7, 8, 9, 10] )
    def set_seed(no):
        torch.manual_seed(no)
        random.seed(no)
        np.random.seed(no)
        os.environ['PYTHONHASHSEED'] = str()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    set_seed(100)

    output_size = (378, 378)
    data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(output_size),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(degrees=(0,180)),
                #transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1.9)),
                transforms.RandomAffine(degrees=(-45, 45), translate=(0.3, 0.1), scale=(0.8, 1.5)),#trail9_windinghead_optuna_effnetv2s_9/3/24
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # transforms.ColorJitter(
                #     brightness=[0.5, 1.5],
                #     contrast=round(random.uniform(0,1), 1),
                #     saturation=round(random.uniform(0,1), 1),
                #     hue= round(random.uniform(0,0.5), 1)
                # ),
                # transforms.RandomGrayscale(p=round(random.uniform(0.0, 1.0), 1)),
                # transforms.GaussianBlur((random.randrange(1,51,2), random.randrange(1,51,2)+2), sigma=(0.1, 5.)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(output_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
    }


    dataset_train = CustomSSLDataset(json_file_path='/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/WindingHead/train.json', transform=data_transforms["train"])

    dataloader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=8,
    )

    dataset_test = CustomSSLDataset(json_file_path='/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/WindingHead/val.json', transform=data_transforms["test"])

    dataloader_test = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=8,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #dinov2_vits14 = torch.hub.load('/home/hpc/iwfa/iwfa047h/.cache/torch/hub/dinov2', backbone, source='local')
    dinov2_vits14  = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    class DinoVisionTransformerClassifier(nn.Module):
        def __init__(self):
            super(DinoVisionTransformerClassifier, self).__init__()
            self.transformer = dinov2_vits14
            in_features = 384
            # if backbone == "dinov2_vits14":
            #     in_features = 384
            # elif backbone == "dinov2_vitb14":
            #     in_features = 768
            # elif backbone == "dinov2_vitl14":
            #     in_features = 1024
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Dropout(p=dropout),   # Add dr
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            
        def forward(self, x):
            x = self.transformer(x)
            x = self.transformer.norm(x)
            x = self.classifier(x)
            return x

    model = DinoVisionTransformerClassifier()

    criterion = nn.BCEWithLogitsLoss()

        # optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Create SummaryWriter with clear experiment directory
    model = model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, verbose=True)
    global tensorcount
    tensorcount = tensorcount + 1
    tensorString = "dino_optuna_" + str(tensorcount)
    writer = SummaryWriter('runs/' + tensorString)
    min_val_loss = np.Inf  # Minimum validation loss starts at infinity
    f1 = 0
    epochs_no_improve = 0  # No improvement in epochs counter
    report = {}
    for epoch in range(50):
        print('Epoch {}/{}'.format(epoch, 50 - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Initialize metrics for this phase
            running_loss = 0.0  # Accumulate losses over the epoch
            correct = 0  # Count correct predictions
            total = 0  # Count total predictions

            all_preds = []
            all_labels = []

            if phase == 'train':
                dataloader = dataloader_train
            else:
                dataloader = dataloader_test

            # Use tqdm for progress bar
            with tqdm(total=len(dataloader), unit='batch') as p:
                    # Iterate over mini-batches
                for inputs, labels, filenames in dataloader:
                        # Move input and label tensors to the default device (GPU or CPU)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Clear the gradients of all optimized variables
                    optimizer.zero_grad()

                    # Forward pass: compute predicted outputs by passing inputs to the model
                    with torch.set_grad_enabled(phase == 'train'):  # Only calculate gradients in training phase
                        outputs = model(inputs)
                        outputs = outputs.squeeze()
                        labels = labels.float()
                        preds = (outputs > 0.5).float()
                        loss = criterion(outputs, labels)  # Compute the loss

                        # Perform backward pass and optimization only in the training phase
                        if phase == 'train':
                            loss.backward()  # Calculate gradients based on the loss
                            optimizer.step()  # Update model parameters based on the current gradient
                    # Update running loss and correct prediction count
                    running_loss += loss.item() * inputs.size(0)  # Multiply average loss by batch size
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()  # Update correct predictions count
                    all_preds.append(preds)
                    all_labels.append(labels)
                    # Update the progress bar
                    p.set_postfix({'loss': loss.item()})
                    p.update(1)

                # Calculate loss and accuracy for this epoch
                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = 100 * correct / total

                writer.add_scalar(f"{phase}_loss", epoch_loss, epoch)
                if phase == 'val':
                        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                        all_preds = torch.cat([x for x in all_preds])
                        all_labels = torch.cat([x for x in all_labels])
                        f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average="weighted")
                        print(f1)
                        cm = confusion_matrix(all_labels.cpu(), all_preds.cpu())
                        print(cm)
                        tp, fn, fp, tn = cm.ravel()
                        print("TP = ", tp, "FP = ", fp, "FN = ", fn, "TN = ", tn)
                        print('\nClassification Report\n')
                        report = classification_report(all_labels.cpu(), all_preds.cpu(), zero_division=0, output_dict=True)
                        print(classification_report(all_labels.cpu(), all_preds.cpu(), zero_division=0))
                        if epoch_loss < min_val_loss:
                            print(f'Validation Loss Decreased({min_val_loss:.6f}--->{epoch_loss:.6f}) \t Saving The Model')
                            min_val_loss = epoch_loss  # Update minimum validation loss # Save the current model weights
                            f1 = report["1.0"]["f1-score"]
                            epochs_no_improve = 0  # Reset epochs since last improvement
                            print('{} Loss: {:.4f} Acc: {:.2f}%'.format(phase, epoch_loss, epoch_acc))                        
                        else:
                            epochs_no_improve += 1
                            # Implement early stopping
                            if epochs_no_improve == improve_patience:
                                print('Early stopping!')
                                model.eval() # Exit the function early
                                return min_val_loss, f1
            # Adjust the learning rate based on the scheduler
            scheduler.step(epoch_loss)
    return min_val_loss, f1

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "RGBWindingHead"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(directions=['minimize', 'maximize'], study_name=study_name, storage=storage_name)
study.optimize(objective, n_trials=150)


          
# dataset_test = CustomSSLDataset(json_file_path='/home/woody/iwfa/iwfa047h/Cable/Cable_5_20_40/test.json', transform=data_transforms["test"])

# dataloader_test = torch.utils.data.DataLoader(
#         dataset_test,
#         batch_size=6,
#         shuffle=False,
#         drop_last=True,
#         num_workers=8,
# )

# with torch.no_grad():
#     all_labels = []
#     all_preds = []
#     test_loss = 0.0
#     for data in dataloader_test:
#         inputs, labels, filenames = data

#         # Forward pass through the model
#         outputs = model(inputs.to(device))
#         outputs = outputs.squeeze()
#         labels = labels.float()
#         preds = (outputs > 0.5).float()  # Adjust threshold based on your task
#         # Calculate loss on the test data
#         loss = criterion(outputs, labels.to(device))  # Move labels to device
#         test_loss += loss.item()
#         all_labels.append(labels.float())
#         all_preds.append(preds)

#     # Calculate average test loss
#     average_test_loss = test_loss / len(dataloader_test.dataset)

#     # Log test loss to TensorBoard (assuming you have a writer defined)
#     writer.add_scalar("test_loss", average_test_loss, epoch)

# all_preds = torch.cat([x for x in all_preds])
# all_labels = torch.cat([x for x in all_labels])
# f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average="weighted")
# print(f1)
# cm = confusion_matrix(all_labels.cpu(), all_preds.cpu())
# print(cm)
# tp, fn, fp, tn = cm.ravel()
# print("TP = ", tp, "FP = ", fp, "FN = ", fn, "TN = ", tn)
# print('\nClassification Report\n')
# report = classification_report(all_labels.cpu(), all_preds.cpu(), zero_division=0, output_dict=True)
# print(classification_report(all_labels.cpu(), all_preds.cpu(), zero_division=0))