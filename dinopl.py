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
import CONFI
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
from CNN.loss_function import FocalLoss, WeightedCrossEntropyLoss, WeightedBinaryCrossEntropyLoss
from network import  ResNet, DeepCNN, indices,  targetlist, predlist, name_list, indices_def, predlist_def, indices_val, predlist_val, indices_def_val, predlist_def_val, test_FP, test_FN, val_FN, val_FP#tb_writer,
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
from din import Dino
import json
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, f1_score, classification_report
#from custom_dataset_dino import CustomSSLDataset
from PIL import Image
def set_seed(no):
    torch.manual_seed(no)
    random.seed(no)
    np.random.seed(no)
    os.environ['PYTHONHASHSEED'] = str()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(100)






views = ["file_name"]
labels = ["label", "~label"]
num_classes = len(labels)

###get the embeddings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dinov2_vits14  = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])
def load_image(img: str) -> torch.Tensor:
    img = Image.open(img)
    transformed_img = transform_image(img)[:3].unsqueeze(0)
    return transformed_img

def compute_embeddings(data_type):
    all_embeddings = []
    base_path = "/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Cover/"
    json_file_path = os.path.join(base_path, f"{data_type}.json")
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

###callbacks and loggers
early_stop = callbacks.EarlyStopping(monitor=CONFI.QUANTITY_TO_OPTIMIZE, mode=CONFI.DIRECTION_OF_OPTIMIZATION_SHORT, patience=CONFI.PATIENCE, verbose=True, min_delta=CONFI.MINDELTA)#stopping_threshold=0.93,
checkpoint_callback = ModelCheckpoint(dirpath=CONFI.weights_best_trial, monitor=CONFI.QUANTITY_TO_OPTIMIZE, save_top_k=1, mode=CONFI.DIRECTION_OF_OPTIMIZATION_SHORT, save_last=True, verbose=True)
lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum= True, log_weight_decay= True)

non_trial_writter = loggers.TensorBoardLogger(save_dir=CONFI.non_trail_hp_writer,
                             version=CONFI.logger_version,
                             prefix="fu",
                             name='Final_Hyperparameters_Used',
                             log_graph=False
                             )#histogram_interval=1

trainer = Trainer(
    accelerator= CONFI.ACCELERATOR, 
    devices=CONFI.DEVICES,
    max_epochs = CONFI.EPOCHS,
    log_every_n_steps=15,
    overfit_batches =0,
    #auto_insert_metric_name =True,
    #track_grad_norm=2,
    callbacks=[checkpoint_callback, early_stop, lr_monitor],
    logger=[non_trial_writter],
    fast_dev_run=CONFI.fast_dev_run,
    check_val_every_n_epoch=CONFI.validation_interval,
    default_root_dir = CONFI.weights_best_trial,
    enable_checkpointing = True,
    #profiler="simple"
    )


##model
model_used = CONFI.Model
print(model_used)
model = getattr(torchvision.models, model_used)( weights="IMAGENET1K_V1")#dropout=best_dropout,
model.name = model_used

if CONFI.Model == "efficientnet_v2_s":
    backbone_out_features = model.classifier[1].out_features
else:
    backbone_out_features = getattr(model, CONFI.BACKBONE).out_features




##hyperparameters
loss_func = torch.nn.CrossEntropyLoss(weight=None)

output_activation = torch.nn.Softmax(dim=1) #Sigmoid()Sigmoid()#Softmax(dim=1)
output_activation.name = "Softmax"

best_lr = CONFI.LR

batch_size = CONFI.BATCHSIZE

optimizer_idx = 0
optimizer_names = ["AdamW", "RMSprop", "SGD", "Adagrad"]
best_optimizer = optimizer_names[optimizer_idx]
print("Best Optimizer:", best_optimizer)
beta = CONFI.BETA
best_patience = CONFI.PATIENCE
print("Best patience:", best_patience)

H = CONFI.H
print("height:", H)
W = CONFI.W
print("height:", W)  

best_weight_decay = CONFI.WD
print("Best Weight Decay:", best_weight_decay)


fusion_model = DeepCNN(backbone=model,
                                    backbone_out=backbone_out_features,
                                    num_classes=num_classes,
                                    lr=best_lr,
                                    optimizer = best_optimizer,
                                    output_activation=output_activation,
                                    loss_func = loss_func,
                                    views=views,
                                    labels=labels,
                                    Val = False,
                                    beta=beta
                                    #dropout_rate=best_p
                                    )

fusion_model.optimizer = getattr(torch.optim, best_optimizer)(fusion_model.parameters(), lr=best_lr, weight_decay=best_weight_decay)


##train_embeddings
embeddings_train, labels_train = compute_embeddings("train")
embedding_list = list(embeddings_train.values())
img_train = np.array(embedding_list).reshape(-1, 384)
train_dataset = img_train, labels_train


##val_embeddings
embeddings_val, labels_val = compute_embeddings("val")
embedding_list = list(embeddings_val.values())
img_val = np.array(embedding_list).reshape(-1, 384)
val_dataset = img_val, labels_val


mv_train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=4, drop_last=True)    
mv_val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4, drop_last=True)
trainer.fit(fusion_model, mv_train_loader, mv_val_loader, ckpt_path=None)






json_file_path = "/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Cover/Test.json"
with open(json_file_path, 'r') as f:
    data = json.load(f)
 
all_preds = []
all_labels = []
with torch.no_grad():
    for row in data:
        embedding = dinov2_vits14(load_image('/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Cover/' + row['file_name']))
        prediction = trainer.predict(dataloaders=mv_val_loader, ckpt_path=CONFI.CKPT_PATH)
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
















##validation dataloader
json_file_path = "/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Cover/val.json"
with open(json_file_path, 'r') as f:
    data = json.load(f)

with torch.no_grad():
    for row in data:
        embedding = dinov2_vits14(load_image('/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Cover/' + row['file_name']))
        img = np.array(embedding[0].cpu()).reshape(1, -1)
        
val_dataset = img, labels





all_preds = []
all_labels = []
with torch.no_grad():
    for row in data:
        embedding = dinov2_vits14(load_image('/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/Cover/' + row['file_name']))
        prediction = clf.predict(np.array(embedding[0].cpu()).reshape(1, -1))
        trainer.predict(dataloaders=mv_val_loader, ckpt_path=CONFI.CKPT_PATH)
        all_labels.append(row['label'])
        all_preds.append(prediction[0])


