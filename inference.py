import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torchvision
import random
import torch
def set_seed(no):
    torch.manual_seed(no)
    random.seed(no)
    np.random.seed(no)
    os.environ["PYTHONHASHSEED"] = str()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_seed(100)
# seed = 100
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
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
from dataloader import (
    MultiViewDataset,
    datatest,
    dataval,
)  
from loss_function import (
    WeightedCrossEntropyLoss,
    WeightedBinaryCrossEntropyLoss,
)
from network import  DeepCNN
from utils.data_splitting import random_split_dataset
from utils.visualisations import visualize_dataloader_for_class_balance
from pytorch_lightning import LightningModule
from misclassification import misclassification, Data_post_Processor
from network import DeepCNN
from eval import evaluate
from collections import OrderedDict
from torch.optim import RMSprop
from torch.optim import Adagrad
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
import cProfile
import cv2
import monai
import json
from make_folder import make_folder
from extract_gradcam import grad_cam

from config_loader import load_config
mode = "train"  # Change this to "debug" when debugging
CONFI = load_config(mode)
iteration_count = 0

if mode== "train":
    DEVICES = 1 if torch.cuda.is_available() else None 
else:
    DEVICES = "auto" 

folder_maker = make_folder(
    Component=CONFI["Component"],
    Model=CONFI["Model"],
    logger_version=CONFI["logger_version"],
    base_dir=CONFI["base_dir"],
)
(
    non_trail_hp_writer,
    trail_hp_writer,
    weights_best_trial,
    weights_best_trial_inoptuna,
    CKPT_PATH,
    Misclassification_save_dir,
    new_path_result,
    TRAINER_OPTUNA_DIR,
    Reconstructed_image_save_dir, 
    Gradcam_save_dir
) = folder_maker.create_folders(base_dir=CONFI["base_dir"])
ROOT_DIR = os.path.join(
    "/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/",
    CONFI['Component'],
)
print(ROOT_DIR)
test_csv_path = os.path.join(ROOT_DIR, "test.csv")
val_csv_path = os.path.join(ROOT_DIR, "val.csv")
views = ["file_name"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Order matter for labels
labels = ["label", "~label"]
num_classes = len(labels)
validation_interval = CONFI['validation_interval']
output_activation = torch.nn.Softmax(dim=1)  # Sigmoid()Sigmoid()#Softmax(dim=1)
output_activation.name = "Softmax"
val_dataset = MultiViewDataset(
    val_csv_path,
    views=views,
    labels=labels,
    base_dir=ROOT_DIR,
    transform=False,
    normalize=True,
    Test=False,
    Val=True,
    Feature_extraction=False,
    pil_image_mode=CONFI['IMAGE_MODE'],
)  
print(f"Val Normalize: {val_dataset.normalize}")
print(f"Val Transform: {val_dataset.transform}")
print(f"Val Feature_extraction: {val_dataset.Feature_extraction}")

test_dataset = MultiViewDataset(
    test_csv_path,
    views=views,
    labels=labels,
    base_dir=ROOT_DIR,
    transform=False,
    normalize=True,
    Test=True,
    Val=False,
    Feature_extraction=False,
    pil_image_mode=CONFI['IMAGE_MODE'],
)  
print(f"Test Normalize: {test_dataset.normalize}")
print(f"Test Transform: {test_dataset.transform}")
print(f"Test Feature_extraction: {test_dataset.Feature_extraction}")

print(f"Image_mode={CONFI['IMAGE_MODE']}")

non_trial_writter = loggers.TensorBoardLogger(
    save_dir=non_trail_hp_writer,
    version=CONFI['logger_version'],
    prefix="fu",
    name="Final_Hyperparameters_Used",
    log_graph=False,
)

H = CONFI['H']
print("height:", H)
W = CONFI['W']
print("height:", W)
num_neurons_2 = CONFI["NUM_NEURONS"]
print("num_neurons_2:", num_neurons_2)
num_layers = CONFI["NLAYERS"]
print("num_layers:", num_layers)
INFERENCE_CKPT_PATH = CONFI["INFERENCE_CKPT_PATH"]
print("INFERENCE_CKPT_PATH:", INFERENCE_CKPT_PATH)

if CONFI['Model'] == "Dinov2":
    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone_out = 1000
    late_fc_d = torch.nn.Sequential(
        OrderedDict(
            [
                (
                    "fc2",
                    torch.nn.Linear(
                        in_features=len(views) * backbone_out, out_features=num_classes
                    ),
                ),
            ]
        )
    )

    class DinoVisionTransformerClassifier(torch.nn.Module):
        def __init__(self):
            super(DinoVisionTransformerClassifier, self).__init__()
            self.transformer = dinov2_vits14
            # self.late_fc_d = late_fc_d
            # self.dropout = nn.Sequential(nn.Dropout(p=CONFI.DROPOUT))
            self.classifier = torch.nn.Sequential(torch.nn.Linear(384, num_neurons_2[0]))

        def forward(self, x):
            x = self.transformer(x)
            x = self.transformer.norm(x)
            x = self.classifier(x)
            return x

    model = DinoVisionTransformerClassifier()

else:
    model_used = CONFI['Model']
    print(model_used)
    model = getattr(torchvision.models, model_used)(
        weights="IMAGENET1K_V1"
    )  # dropout=best_dropout,
    model.name = model_used

    if CONFI['Model'] == "efficientnet_v2_s":
        backbone_out_features = model.classifier[1].out_features
    else:
        backbone_out_features = getattr(model, CONFI['BACKBONE']).out_features

    backbone_out = backbone_out_features
    late_fc_d = torch.nn.Sequential(
        OrderedDict(
            [
                (
                    "fc2",
                    torch.nn.Linear(
                        in_features=len(views) * backbone_out, out_features=num_classes
                    ),
                ),
            ]
        )
    )

    best_nlayers = CONFI['NLAYERS']
    print("Best nlayers:", best_nlayers)

    best_endlayers = CONFI['ENDLAYERS']
    print("Best endlayers:", best_endlayers)


###freezing of the layers
if CONFI['Model'] == "resnext50_32x4d":

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    total_params = count_parameters(model)
    print(f"Total number of parameters: {total_params}")

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad_(False)

    ###code to check if all the parameters are frozen and get  the names
    # for param in model.parameters():
    #     print(param.requires_grad)
    # for name, param in model..named_parameters():
    #     print(name)

    ##Count parameters and unfreeze specified layers
    for i, (name, layer) in enumerate(model.named_children()):
        layer_params = count_parameters(layer)
        print(f"Parameters in {name}: {layer_params}")

    for i, (name, layer) in enumerate(model.named_children()):
        if best_nlayers <= i < best_endlayers:
            layer_params = count_parameters(layer)
            print(f"Parameters in {name} to be unfreezee {i}: {layer_params}")
            for param in layer.parameters():
                param.requires_grad_(True)

elif CONFI['Model'] == "efficientnet_v2_s":

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    total_params = count_parameters(model)
    print(f"Total number of parameters: {total_params}")

    for i, layer in enumerate(model.features):
        layer_params = count_parameters(layer)
        print(f"Parameters in layer {i}: {layer_params}")

    for param in model.parameters():
        param.requires_grad_(False)

    for i, layer in enumerate(model.features[best_nlayers:best_endlayers]):
        layer_params = count_parameters(layer)
        print(f"Parameters to be unfreezee {i}: {layer_params}")

    for i in range(best_nlayers, best_endlayers):
        for param in model.features[i].parameters():
            param.requires_grad = True

skip_freezing_batch_normalization = CONFI['SKIP_FREEZING_BATCHNORMALIZATION']
if not skip_freezing_batch_normalization:
    m = 0

    def find_batchnorm_layers(module):
        global m
        for name, child in module.named_children():
            if isinstance(child, torch.nn.BatchNorm2d):
                # Do something with the BatchNorm2d layer
                child.requires_grad = False
                # print("Frozen BatchNorm2d layer:", name)
                m += 1
            else:
                # Recursively traverse child modules
                find_batchnorm_layers(child)

    find_batchnorm_layers(model)
    print("Total number of frozen BatchNorm2d layers:", m)


trainer = Trainer(
    accelerator=CONFI['ACCELERATOR'],
    devices=DEVICES,
    max_epochs=CONFI['EPOCHS'],
    log_every_n_steps=15,
    overfit_batches=0,
    fast_dev_run=CONFI['fast_dev_run'],
    check_val_every_n_epoch=validation_interval,
    default_root_dir=weights_best_trial,
    enable_checkpointing=False,
    # profiler="simple  "
)

#CKPT_PATH = "/home/vault/iwfa/iwfa048h/2024-05-20/WindingHead/Dinov2/trial4_best_ipw____WH_NLMLP/checkpoint/epoch=16-step=5202.ckpt"
INFERENCE_CKPT_PATH = CONFI['INFERENCE_CKPT_PATH']
in_features = 384
backbone_out = 1000
late_fc_d = torch.nn.Sequential(
    OrderedDict(
        [
            (
                "fc2",
                torch.nn.Linear(
                    in_features=len(views) * backbone_out, out_features=num_classes
                ),
            ),
        ]
    )
)


class infer(LightningModule):
    def __init__(
        self,
        backbone=model,
        num_classes=num_classes,
        output_activation=output_activation,
        views=views,
        labels=labels,
        num_layers = num_layers,
        num_neurons_2 = num_neurons_2,        
    ):
        super().__init__()
        self.backbone = backbone
        self.output_activation = output_activation
        self.num_classes = num_classes
        self.views = views
        self.labels = labels
        self.num_layers = num_layers
        self.num_neurons_2 = num_neurons_2
        self.gradcam = monai.visualize.GradCAM(
            nn_module=self.backbone,
            target_layers="features.7.2",  # self.backbone.features[7][2]
        )
        self.dropout = torch.nn.Sequential(torch.nn.Dropout(p=CONFI['DROPOUT']))        
        # self.transformer = transformer
        self.late_fc_d = late_fc_d
        if CONFI['Model'] == "Dinov2":
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

        # self.classifier2 = torch.nn.Sequential(
        #         torch.nn.ReLU(),
        #         torch.nn.Linear(74, 1)
        #     )

    def predict_step(self, batch, batch_idx):

        if CONFI['Model'] == "Dinov2":
            target_ = batch["label"]
            target = target_.cpu().numpy()

            view = self.views[0]
            x = batch[view]
            x = self.backbone(x)
            x = self.dropout(x)
            pred = self.classifier2(x)
            #x = self.classifier2(x)
            pred = pred.squeeze()
            pred_ = (pred > 0.5).float()
            y = pred_.cpu().detach()


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

        else:
            view = self.views[0]
            x = batch[view]
            x2 = batch[view]
            x = self.backbone(x)
            y = late_fc_d(x)

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


fusion_model = infer.load_from_checkpoint(INFERENCE_CKPT_PATH, strict=True)

print(
    "Prediction result of the validation dataset__________________________________________________________________________________"
)
Indices = []
Predictions = []
Indices_Def = []
Prediction_Def = []
FP = []
FN = []
Indices_v = Indices
Predictions_v = Predictions
Indices_Def_v = Indices_Def
Prediction_Def_v = Prediction_Def
FP_V = FP
FN_V = FN
mv_val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=4)
preds = trainer.predict(fusion_model, dataloaders=mv_val_loader, ckpt_path=INFERENCE_CKPT_PATH)
result_val = evaluate(val_dataset, preds, labels)

skip_misclassification = CONFI['SKIP_MISCLASSIFY']
if not skip_misclassification:
    validation_processor = Data_post_Processor(Indices_v, Predictions_v, Indices_Def_v, Prediction_Def_v, FP_V, FN_V, datatest, dataval)


print(
    "Prediction result of the test dataset__________________________________________________________________________________"
)
Indices = []
Predictions = []
Indices_Def = []
Prediction_Def = []
FP = []
FN = []
Indices_T = Indices
Predictions_T = Predictions
Indices_Def_T = Indices_Def
Prediction_Def_T = Prediction_Def
FP_T = FP
FN_T = FN
mv_test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=4)
preds = trainer.predict(
    fusion_model, dataloaders=mv_test_loader, ckpt_path=INFERENCE_CKPT_PATH
) 
result = evaluate(test_dataset, preds, labels)



skip_misclassification = CONFI['SKIP_MISCLASSIFY']
if not skip_misclassification:
    pred_val, selected_values_val, pred_def_val, selected_values_def_val = validation_processor.process_samples(Predictions_v, Prediction_Def_v, Indices_v, Indices_Def_v, dataval)
    # Call the plot_validation_samples method
    validation_processor.plot__samples(ROOT_DIR, selected_values_def_val, selected_values_val, pred_val, pred_def_val, Misclassification_save_dir)
    # Call the compute_validation_list method
    validation_processor.compute_list(dataval, FP_V, FN_V)

    print(
    "Prediction result of the test dataset__________________________________________________________________________________"
    )
    test_processor = Data_post_Processor(Indices_T, Predictions_T, Indices_Def_T, Prediction_Def_T, FP_T, FN_T, datatest, dataval)
    pred_val, selected_values_val, pred_def_val, selected_values_def_val = test_processor.process_samples(Predictions_T, Prediction_Def_T, Indices_T, Indices_Def_T, datatest)
    # Call the plot_validation_samples method
    test_processor.plot__samples(ROOT_DIR, selected_values_def_val, selected_values_val, pred_val, pred_def_val, Misclassification_save_dir)
    # Call the compute_validation_list method
    test_processor.compute_list(datatest, FP_T, FN_T)


