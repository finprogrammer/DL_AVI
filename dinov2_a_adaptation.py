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
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torchvision
import random
import torch
from collections import OrderedDict


def set_seed(no):
    torch.manual_seed(no)
    random.seed(no)
    np.random.seed(no)
    os.environ["PYTHONHASHSEED"] = str()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_seed(100)

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
from network import (
    DeepCNN,
)
from utils.data_splitting import random_split_dataset
from utils.visualisations import visualize_dataloader_for_class_balance
from misclassification import misclassification
from network import DeepCNN
from eval import evaluate
from torch.optim import RMSprop
from torch.optim import Adagrad
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
import cProfile
#from din import Dino
import json
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from PIL import Image
from make_folder import make_folder
from config_loader import load_config

mode = "train_2"  # Change this to "debug" when debugging
CONFI = load_config(mode)

# tensorboard --logdir=
# optuna-dashboard sqlite:///RGB__2ndretry__Sheet_metal_packaging_real.db --host 0.0.0.0 --port 7000

###for printing the names of the layers 
# for name, param in self.backbone.named_parameters():
#     print(name) 

# for param in self.classifier2.parameters():
#         print(param.requires_grad)



Sq_lite_file = "sqlite:///{}.db".format(CONFI["logger_version"])
if mode == "train":
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

###paramters
validation_interval = CONFI["validation_interval"]
views = ["file_name"]
labels = ["label", "~label"]
batch_size = CONFI["BATCHSIZE"]
print(f"Batchsize={batch_size}")
num_classes = len(labels)
output_activation = torch.nn.Softmax(dim=1)  # Sigmoid()Sigmoid()#Softmax(dim=1)
output_activation.name = "Softmax"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

###dataloader
ROOT_DIR = os.path.join(
    "/home/woody/iwfa/iwfa048h/Python-Code/database/data_processed/Classification/",
    CONFI["Component"],
)
print(ROOT_DIR)
train_csv_path = os.path.join(ROOT_DIR, "train.csv")
test_csv_path = os.path.join(ROOT_DIR, "test.csv")
val_csv_path = os.path.join(ROOT_DIR, "val.csv")






train_dataset = MultiViewDataset(
    train_csv_path,
    views=views,
    labels=labels,
    base_dir=ROOT_DIR,
    transform=True,
    normalize=True,
    Test=False,
    Val=False,
    Feature_extraction=False,
    pil_image_mode=CONFI["IMAGE_MODE"],
)
print(f"Train Normalize: {train_dataset.normalize}")
print(f"Train Transform: {train_dataset.transform}")
print(f"Train Feature_extraction: {train_dataset.Feature_extraction}")
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
    pil_image_mode=CONFI["IMAGE_MODE"],
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
    pil_image_mode=CONFI["IMAGE_MODE"],
)  
print(f"Test Normalize: {test_dataset.normalize}")
print(f"Test Transform: {test_dataset.transform}")
print(f"Test Feature_extraction: {test_dataset.Feature_extraction}")
print(f"Image_mode={CONFI['IMAGE_MODE']}")

#oversampling
class_counts = np.ones(num_classes)
for i, val in enumerate(train_dataset.data):
    label = np.asarray(val["label"])
    # assuming one-hot encoding
    class_ = np.argmax(label)
    class_counts[class_] += 1

class_weights = [round(1000.0 / count, 3) for count in class_counts]
class_weights = torch.tensor(class_weights).to(device)
print(class_weights)

# loss_func = WeightedCrossEntropyLoss(weights=class_weights)

sample_weights = np.zeros(len(train_dataset.data))
for i, val in enumerate(train_dataset.data):
    label = np.asarray(val["label"])
    # assuming one-hot encoding
    class_ = np.argmax(label)
    sample_weights[i] = 1 / class_counts[class_]
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(train_dataset), replacement=True
)
# class DinoVisionTransformerClassifier(nn.Module):
#     def __init__(self, val=False, Test=False):
#         super(DinoVisionTransformerClassifier, self).__init__()
#         self.transformer = dinov2_vits14
#         self.val = val
#         self.Test = Test
#         # self.dropout = nn.Sequential(nn.Dropout(p=CONFI.DROPOUT))
#         in_features = 384
#         self.classifier1 = nn.Sequential(
#             nn.Linear(in_features, 256),
#             # nn.Linear(291, 431)
#         )

#     def forward(self, x):
#         x = self.transformer(x)
#         x = self.transformer.norm(x)
#         x = self.classifier1(x)
#         return x


# model = DinoVisionTransformerClassifier(val=False, Test=False)


##hyperparameters
loss_func = torch.nn.BCEWithLogitsLoss()

skip_optuna = CONFI["SKIP_OPTUNA"]
if not skip_optuna:

    # pruner = optuna.pruners.MedianPruner(
    #     n_startup_trials=4, n_warmup_steps=5
    # )  

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-7, 1e-3, log=True)
        weight_decay = trial.suggest_uniform(
            "weight_decay",
            0.001,
            0.05,
        )
        num_layers = trial.suggest_int("n_layers", 1, 3)
        num_neurons_2 = [
            trial.suggest_int(f"num_neurons_layer{i}", 64, 512)
            for i in range(num_layers)
        ]
        patience = trial.suggest_int("p_a", 6, 10)
        patience_lr = trial.suggest_int("patience_lr", 2, 4)
        batch_size_choices = [16, 32]
        batch_size = trial.suggest_categorical("btch_size", batch_size_choices)
        dropout = trial.suggest_float("dropout_p", 0.2, 0.7)
        selected_optimizer = "AdamW"
        H = CONFI["H"]
        print("height:", H)
        W = CONFI["W"]
        print("height:", W)
        print(f"Batchsize={batch_size}")
        print(f"model_used={CONFI['Model']}")
        print(f"Component={CONFI['Component']}")
        print(f"PATIENCE={CONFI['PATIENCE']}")

        dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

        class DinoVisionTransformerClassifier(nn.Module):
            def __init__(self):
                super(DinoVisionTransformerClassifier, self).__init__()
                self.transformer = dinov2_vits14
                #layers.append(nn.Dropout(p=dropout))

                self.classifier = nn.Sequential(nn.Linear(384, num_neurons_2[0]))

            def forward(self, x):
                x = self.transformer(x)
                x = self.transformer.norm(x)
                x = self.classifier(x)
                return x
            
        model = DinoVisionTransformerClassifier()    

        mv_train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=4,
            drop_last=True,
        )  

        mv_val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=4,
            drop_last=True,
        )

        early_stop = callbacks.EarlyStopping(
            monitor=CONFI["QUANTITY_TO_OPTIMIZE"],
            mode=CONFI["DIRECTION_OF_OPTIMIZATION_SHORT"],
            patience=patience,
            verbose=True,
            min_delta=CONFI["MINDELTA"],
        )
        # pruning_callback = PyTorchLightningPruningCallback(
        #     trial, monitor=CONFI["QUANTITY_TO_OPTIMIZE"]
        # )
        trial_writer = loggers.TensorBoardLogger(
            save_dir=trail_hp_writer,
            version=str(trial.number),
            prefix="f1",
            name=CONFI["optuna_tensorboard_study_name"],
            log_graph=False,
        )
        fusion_model = DeepCNN(
            backbone=model,
            num_classes=num_classes,
            optimizer=selected_optimizer,
            output_activation=output_activation,
            loss_func=loss_func,
            views=views,
            labels=labels,
            dino=True,
            beta=CONFI["BETA"],
            num_layers = num_layers,
            num_neurons_2 = num_neurons_2,
            dropout = dropout,
        )
        fusion_model.optimizer = getattr(torch.optim, selected_optimizer)(
            fusion_model.parameters(), lr=lr, weight_decay=weight_decay
        )
        lr_monitor = LearningRateMonitor(
            logging_interval="step", log_momentum=True, log_weight_decay=True
        )
        fusion_model.lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(
            fusion_model.optimizer,
            factor=0.1,
            patience=patience_lr,
            verbose=True,
            cooldown=2,
        )

        trainer = Trainer(
            logger=[trial_writer],
            enable_checkpointing=False,
            max_epochs=CONFI["EPOCHS"],
            log_every_n_steps=15,
            accelerator=CONFI["ACCELERATOR"],
            devices=DEVICES,
            callbacks=[
                early_stop,
                #pruning_callback,
                lr_monitor,
            ],
            fast_dev_run=CONFI["fast_dev_run"],
            default_root_dir=TRAINER_OPTUNA_DIR,
            # profiler="advanced"
        )
        hyperparameters = dict(lr=lr, selected_optimizer=selected_optimizer)
        trainer.logger.log_hyperparams(hyperparameters)

        trainer.fit(fusion_model, mv_train_loader, mv_val_loader)
        val_loss = trainer.callback_metrics['val_loss']
        f1_notlabel = trainer.callback_metrics['f1_notlabel']

        return f1_notlabel#f1_notlabel#, val_loss

    study = optuna.create_study(
        directions=[CONFI["DIRECTION_OF_OPTIMIZATION"]],#, 'minimize'
        study_name=CONFI["logger_version"],
        storage=Sq_lite_file,
        load_if_exists=True,    
        #pruner=pruner,
    )

    study.optimize(objective, n_trials=CONFI["n_trials"], callbacks=[])
    print(f"Best value: {study.best_value} (params: {study.best_params})")

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:", {study.best_trial.value})


else:
    ##manual trial rerun
    best_lr = CONFI["LR"]
    print("Best learning rate:", best_lr)

    optimizer_idx = 0
    optimizer_names = ["AdamW", "RMSprop", "SGD", "Adagrad"]
    best_optimizer = optimizer_names[optimizer_idx]
    print("Best Optimizer:", best_optimizer)

    best_dropout = CONFI["DROPOUT"]
    print("Best Dropout:", best_dropout)

    best_patience = CONFI["PATIENCE"]
    print("Best patience:", best_patience)

    best_patience_lr = CONFI["PATIENCE_LR"]
    print("Best patience:", best_patience_lr)

    H = CONFI["H"]
    print("height:", H)

    W = CONFI["W"]
    print("height:", W)

    best_weight_decay = CONFI["WD"]
    print("Best Weight Decay:", best_weight_decay)

    num_layers = CONFI["NLAYERS"]
    print("num_layers:", num_layers)

    num_neurons_2 = CONFI["NUM_NEURONS"]
    print("num_neurons_2:", num_neurons_2)

    print(f"MINDELTA={CONFI['MINDELTA']}")

    ###callbacks and loggers
    early_stop = callbacks.EarlyStopping(
        monitor=CONFI["QUANTITY_TO_OPTIMIZE"],
        mode=CONFI["DIRECTION_OF_OPTIMIZATION_SHORT"],
        patience=best_patience,
        verbose=True,
        min_delta=CONFI["MINDELTA"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=weights_best_trial,
        monitor=CONFI["QUANTITY_TO_OPTIMIZE"],
        save_top_k=1,
        mode=CONFI["DIRECTION_OF_OPTIMIZATION_SHORT"],
        save_last=True,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(
        logging_interval="step", log_momentum=True, log_weight_decay=True
    )

    non_trial_writter = loggers.TensorBoardLogger(
        save_dir=non_trail_hp_writer,
        version=CONFI["logger_version"],
        prefix="fu",
        name="Final_Hyperparameters_Used",
        log_graph=False,
    )

    trainer = Trainer(
        accelerator=CONFI["ACCELERATOR"],
        devices=DEVICES,
        max_epochs=CONFI["EPOCHS"],
        log_every_n_steps=15,
        overfit_batches=0,
        # auto_insert_metric_name =True,
        # track_grad_norm=2,
        callbacks=[checkpoint_callback, early_stop, lr_monitor],
        logger=[non_trial_writter],
        fast_dev_run=CONFI["fast_dev_run"],
        check_val_every_n_epoch=validation_interval,
        default_root_dir=weights_best_trial,
        enable_checkpointing=True,
        # profiler="simple"
    )

    # dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

    # class DinoVisionTransformerClassifier(nn.Module):
    #     def __init__(self):
    #         super(DinoVisionTransformerClassifier, self).__init__()
    #         self.transformer = dinov2_vits14
    #         in_features = 384
    #         layers = []
    #         input_size = in_features
    #         layers.append(nn.Linear(input_size, num_neurons_2[0]))
    #         if self.training:
    #             layers.append(nn.Dropout(p=best_dropout))
    #         for i in range(1, num_layers):
    #             layers.append(
    #                 nn.Linear(num_neurons_2[i - 1], num_neurons_2[i])
    #             )  # Connecting layers with different neuron counts
    #             layers.append(nn.ReLU())
    #         input_size = num_neurons_2[-1]
    #         layers.append(nn.Linear(input_size, 1))
    #         self.classifier = nn.Sequential(*layers)

    #     def forward(self, x):
    #         x = self.transformer(x)
    #         x = self.transformer.norm(x)
    #         x = self.classifier(x)
    #         return x
        
    # model = DinoVisionTransformerClassifier()  
    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

    class DinoVisionTransformerClassifier(nn.Module):
        def __init__(self):
            super(DinoVisionTransformerClassifier, self).__init__()
            self.transformer = dinov2_vits14
            #layers.append(nn.Dropout(p=dropout))

            self.classifier = nn.Sequential(nn.Linear(384, num_neurons_2[0]))

        def forward(self, x):
            x = self.transformer(x)
            x = self.transformer.norm(x)
            x = self.classifier(x)
            return x
        
    model = DinoVisionTransformerClassifier() 



    fusion_model = DeepCNN(
        backbone=model,
        num_classes=num_classes,
        optimizer=best_optimizer,
        output_activation=output_activation,
        loss_func=loss_func,
        views=views,
        labels=labels,
        dino=True,
        beta=CONFI["BETA"],
        num_layers = num_layers,
        num_neurons_2 = num_neurons_2,
        dropout = best_dropout,
    )

    fusion_model.optimizer = getattr(torch.optim, best_optimizer)(
        fusion_model.parameters(), lr=best_lr, weight_decay=best_weight_decay
    )
    fusion_model.lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(
        fusion_model.optimizer, factor=0.1, patience=best_patience_lr, verbose=True
    )
    mv_train_loader = DataLoader(
        train_dataset,
        #sampler=sampler,
        shuffle=True,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
    ) 
    mv_val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size, num_workers=4, drop_last=True
    )
    mv_test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=32, num_workers=4
    )

    trainer.fit(fusion_model, mv_train_loader, mv_val_loader, ckpt_path=None)
    trainer.test(dataloaders=mv_test_loader, ckpt_path=CONFI["CKPT_PATH"], verbose=True)

    print(
        "Prediction result of the validation dataset__________________________________________________________________________________"
    )
    mv_val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=1, num_workers=4
    )
    preds = trainer.predict(dataloaders=mv_val_loader, ckpt_path=CONFI["CKPT_PATH"])
    result_val = evaluate(val_dataset, preds, labels)

    print(
        "Prediction result of the test dataset__________________________________________________________________________________"
    )
    mv_test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=1, num_workers=4
    )
    preds = trainer.predict(dataloaders=mv_test_loader, ckpt_path=CONFI["CKPT_PATH"])
    result = evaluate(test_dataset, preds, labels)
