import torch
import os
import glob
from datetime import datetime

##train
logger_version = "att2__RGB_WindingHead_trial30_tue_swin_100_trial"

# Cable/Screw/WindingHead/Sheet_Metal_Package/Cover
Component = "Cable"

# Dinov2/resnext50_32x4d/efficientnet_v2_m/resnet50/swin_v2_b/densenet121/efficientnet_b5/resnet18/resnet34/resnet50/resnet101/efficientnet_v2_s
Model = "Dinov2"

# fc for resnet
# classifier for densenet,
# classifier[1] foe efficientnet,
# head.fc. for swin,
# head for swin pytorchs
# efficientnet mmpretrain, head for deit
BACKBONE = "head"

SKIP_OPTUNA = True
n_trials = 150

IMAGE_MODE = "RGB"  # "L","RGB"
H = 378  # 378 for dino
W = 378  # lesss

LR = 0.0000030894010637277747
BATCHSIZE = 32
WD = 0.0017960105252973267
DROPOUT = 0.328659574153416
NLAYERS = 1
# efficientnet_v2_s = 7+1, efficientnet_v2_m = 8+1, resnext50_32x4d = 10
ENDLAYERS = 8
PATIENCE = 8
PATIENCE_LR = 6
BETA = 1
SKIP_FREEZING_BATCHNORMALIZATION = True
CKPT_PATH = "best"  # last, best

QUANTITY_TO_OPTIMIZE = "val_loss"  # val_loss,f1_micro,f1,vall_acc,recall,f1_custom, f1_label, f1_notlabel, f1_beta_label
DIRECTION_OF_OPTIMIZATION = "minimize"  # maximize,minimize
DIRECTION_OF_OPTIMIZATION_SHORT = "min"  # max,min
MINDELTA = 0.001

SKIP_MISCLASSIFY = False
fast_dev_run = False
Extract_Gradcam = False
Extract_asis_image = False 

EPOCHS = 200
validation_interval = 1


ACCELERATOR = "gpu"  # "auto"
DEVICES = 1 if torch.cuda.is_available() else None  # "auto"
Sq_lite_file = "sqlite:///{}.db".format(logger_version)
optuna_tensorboard_study_name = logger_version  # name under which trials are lodded in tensorboard not the dashboard name

today_date = datetime.today().strftime("%Y-%m-%d")
base_dir = "/home/vault/iwfa/iwfa048h"
base_dir = os.path.join(base_dir, today_date)
base_dir = os.path.join(base_dir, Component)
base_dir = os.path.join(base_dir, Model)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
new_path = os.path.join(
    base_dir, logger_version
)  # path for non trial and trial hp paramter saving
if not os.path.exists(new_path):
    os.makedirs(new_path)

non_trail_hp_writer = (
    new_path  # best hp after the trials and graph for train and test run
)
trail_hp_writer = new_path  # per trail hp and tb curves

base_path, last_component = os.path.split(new_path)
new_path_weigts = os.path.join(new_path, "checkpoint")
if not os.path.exists(new_path_weigts):
    os.makedirs(new_path_weigts)
weights_best_trial = new_path_weigts  # path for saving the model weight with the best hp after the trials

base_path, last_component = os.path.split(new_path)
new_path_weigts_best_trial = os.path.join(new_path, "checkpoint_of the best trial")
if not os.path.exists(new_path_weigts_best_trial):
    os.makedirs(new_path_weigts_best_trial)
weights_best_trial_inoptuna = new_path_weigts_best_trial


checkpoint_files = glob.glob(os.path.join(new_path, "epoch*.ckpt"))
if checkpoint_files:
    latest_checkpoint = checkpoint_files[0]
    CKPT_PATH = latest_checkpoint
else:
    print("No checkpoint files found.")
CKPT_PATH = CKPT_PATH

base_path, last_component = os.path.split(new_path)
new_path_misclassified = os.path.join(new_path, "misclassified")
if not os.path.exists(new_path_misclassified):
    os.makedirs(new_path_misclassified)
Misclassification_save_dir = new_path_misclassified

base_path, last_component = os.path.split(new_path)
new_path_result = os.path.join(new_path, "slurm_and_sqlite")
if not os.path.exists(new_path_result):
    os.makedirs(new_path_result)

TRAINER_OPTUNA_DIR = "/home/vault/iwfa/iwfa048h/CNN/TRAINER_OPTUNA_DIR_TRIAL"  # checkpoint saving in optun not needed as no need to save model weight after each trail

"""
  sizes = {
            'b0': (256, 224), 'b1': (256, 240), 'b2': (288, 288), 'b3': (320, 300),
            'b4': (384, 380), 'b5': (489, 456), 'b6': (561, 528), 'b7': (633, 600),
        }

        340 for swin and effnet v2m
   """
