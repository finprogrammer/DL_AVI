import torch
import os
import glob
from datetime import datetime

logger_version = "trailresent"
# #Cable/Screw/WindingHead/Sheet_Metal_Package/Cover
Component ="Screw"
# #Dinov2/resnext50_32x4d/efficientnet_v2_m/resnet50/swin_v2_b/densenet121/efficientnet_b5/resnet18/resnet34/resnet50/resnet101/efficientnet_v2_s
Model = "efficientnet_v2_s"
PATIENCE = 5
# fc for resnet
#classifier for densenet,
# classifier[1] foe efficientnet,
# head.fc. for swin,
# head for swin pytorchs
# efficientnet mmpretrain, head for deit

BACKBONE = "fc"

SKIP_OPTUNA = True
n_trials = 100

LR = 1e-04
BATCHSIZE = 1
WD = 0.05
DROPOUT = 0.8
#efficientnet_v2_s = 7+1, efficientnet_v2_m = 8+1, resnext50_32x4d = 10
ENDLAYERS = 8
NLAYERS = 5
PATIENCE = 5
MINDELTA = 0.001
validation_interval = 1
BETA = 1
PATIENCE_LR = 5

IMAGE_MODE = "RGB"#"L","RGB"
H = 378
W = 378

QUANTITY_TO_OPTIMIZE = "val_loss" #val_loss,f1_micro,f1,vall_acc,recall
DIRECTION_OF_OPTIMIZATION = "minimize" #maximize,minimize
DIRECTION_OF_OPTIMIZATION_SHORT ="min" #max,min

SKIP_MISCLASSIFY = False
fast_dev_run = True

EPOCHS = 100
SKIP_FREEZING_BATCHNORMALIZATION = False

#efficientnet_v2_m/resnet50/swin_v2_b/densenet121/efficientnet_b5/resnet18/resnet34/resnet50/resnet101/efficientnet_v2_s

ACCELERATOR = "auto"
DEVICES="auto"
SKIP_MISCLASSIFY = False
Sq_lite_file = "sqlite:///{}.db".format(logger_version)
optuna_tensorboard_study_name=logger_version #name under which trials are lodded in tensorboard not the dashboard name
trail = "tr"

today_date = datetime.today().strftime('%Y-%m-%d')
base_dir = "/home/vault/iwfa/iwfa048h/TRIAL"
#base_dir = os.path.join(base_dir, today_date)
base_dir = os.path.join(base_dir, Component)
base_dir = os.path.join(base_dir, trail)
base_dir = os.path.join(base_dir, Component)
base_dir = os.path.join(base_dir, Model)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
new_path = os.path.join(base_dir, logger_version)#path for non trial and trial hp paramter saving
if not os.path.exists(new_path):
    os.makedirs(new_path)

non_trail_hp_writer = new_path #best hp after the trials and graph for train and test run
trail_hp_writer = new_path #per trail hp and tb curves

base_path, last_component = os.path.split(new_path)
new_path = os.path.join(new_path, "checkpoint")
if not os.path.exists(new_path):
    os.makedirs(new_path)
weights_best_trial = new_path #path for saving the model weight with the best hp after the trials

#latest_checkpoint = glob.glob(os.path.join(new_path, "epoch*.ckpt"))
#new_path = os.path.join(new_path, "last.ckpt")
CKPT_PATH = None
checkpoint_files = glob.glob(os.path.join(new_path, "epoch*.ckpt"))
if checkpoint_files:
    latest_checkpoint = checkpoint_files[0]
    CKPT_PATH = latest_checkpoint
else:
    print("No checkpoint files found.")
#CKPT_PATH = "/home/vault/iwfa/iwfa048h/2024-03-12/Screw/efficientnet_v2_s/screw_L_Sheet_Metal_Package_optuna_f1_label/checkpoint/epoch=2-step=318.ckpt"
#CKPT_PATH = "/home/vault/iwfa/iwfa048h/2024-03-15/Cable/swin_v2_b/heavyaug___cable_swin_f1_label_optuna_/checkpoint/epoch=18-step=8569.ckpt"

#dion weights
#CKPT_PATH = "/home/vault/iwfa/iwfa048h/TRIAL/Cable/tr/Cable/resnext50_32x4d/dion_optuna/checkpoint/checkpoint_of the best trial/last.ckpt"
CKPT_PATH = "/home/vault/iwfa/iwfa048h/2024-04-03/Cable/Dinov2/trial11__reducedlr__just__f1_cable_nlayers/checkpoint/last.ckpt"
#resnextweights
#CKPT_PATH = "/home/vault/iwfa/iwfa048h/2024-03-18/Cable/resnext50_32x4d/cable_trial_28_resnext50_cable_optuna/checkpoint/last.ckpt"

base_path, last_component = os.path.split(new_path)
new_path_weigts_best_trial = os.path.join(new_path, "checkpoint_of the best trial")
if not os.path.exists(new_path_weigts_best_trial):
    os.makedirs(new_path_weigts_best_trial)
weights_best_trial_inoptuna = new_path_weigts_best_trial


# if checkpoint_files:
#     # Use the first checkpoint file found (you can modify this logic if needed)
#     latest_checkpoint = checkpoint_files[0]

#     # Update CKPT_PATH to the path of the latest checkpoint
#     CKPT_PATH = latest_checkpoint

#     # Your existing code to call trainer.test with the CKPT_PATH
#     trainer.test(dataloaders=mv_test_loader, ckpt_path=CKPT_PATH)
# else:
#     print("No checkpoint files found.")

 #path for loading the weights of the saved model with best hp after the trials

base_path, last_component = os.path.split(new_path)
new_path_misclassified = os.path.join(new_path, "misclassified")
if not os.path.exists(new_path_misclassified):
    os.makedirs(new_path_misclassified)
Misclassification_save_dir = new_path_misclassified

#Misclassification_save_dir = "/home/vault/iwfa/iwfa048h/CNN/misclassification_TRIAL"
TRAINER_OPTUNA_DIR = "/home/vault/iwfa/iwfa048h/CNN/TRAINER_OPTUNA_DIR_TRIAL"   #checkpoint saving in optun not needed as no need to save model weight after each trail



