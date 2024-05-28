import os
import glob
from datetime import datetime
from config_loader import load_config

mode = "train_2"  # Change this to "debug" when debugging
CONFI = load_config(mode)

class make_folder(object):
    def __init__(self, Component=None, Model=None, logger_version=None, base_dir=None):
        self.Component = Component
        self.Model = Model
        self.logger_version = logger_version
        self.base_dir = base_dir

    def create_folders(self, base_dir):
        today_date = datetime.today().strftime("%Y-%m-%d")
        base_dir = os.path.join(base_dir, today_date)
        base_dir = os.path.join(base_dir, self.Component)
        base_dir = os.path.join(base_dir, self.Model)
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        new_path = os.path.join(base_dir, self.logger_version)  # Path for non-trial and trial hp parameter saving
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        non_trail_hp_writer = new_path  # Best hp after the trials and graph for train and test run
        trail_hp_writer = new_path  # Per trail hp and tb curves

        new_path_weights = os.path.join(new_path, "checkpoint")
        if not os.path.exists(new_path_weights):
            os.makedirs(new_path_weights)
        weights_best_trial = new_path_weights  # Path for saving the model weight with the best hp after the trials

        new_path_weights_best_trial = os.path.join(new_path, "checkpoint_of the best trial")
        if not os.path.exists(new_path_weights_best_trial):
            os.makedirs(new_path_weights_best_trial)
        weights_best_trial_inoptuna = new_path_weights_best_trial

        checkpoint_files = glob.glob(os.path.join(new_path, "epoch*.ckpt"))
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[0]
            CKPT_PATH = latest_checkpoint
        else:
            print("No checkpoint files found.")
            CKPT_PATH = None

        new_path_misclassified = os.path.join(new_path, "misclassified")
        if not os.path.exists(new_path_misclassified):
            os.makedirs(new_path_misclassified)
        Misclassification_save_dir = new_path_misclassified

        new_path_result = os.path.join(new_path, "slurm_and_sqlite")
        if not os.path.exists(new_path_result):
            os.makedirs(new_path_result)

        if CONFI['Extract_Gradcam'] :
            new_path_gradcam = os.path.join(new_path, "grad_cam")
            if not os.path.exists(new_path_gradcam):
                os.makedirs(new_path_gradcam)
                Gradcam_save_dir = new_path_gradcam
            else:
                Gradcam_save_dir = "/home/vault/iwfa/iwfa048h/grad_cam"
        else:
            Gradcam_save_dir = "/home/vault/iwfa/iwfa048h/grad_cam"

        if CONFI['Extract_reconstructed_image'] :
            new_path_reconstructed_image = os.path.join(new_path, "reconstructed_image")
            if not os.path.exists(new_path_reconstructed_image):
                os.makedirs(new_path_reconstructed_image)                
                Reconstructed_image_save_dir = new_path_reconstructed_image
            else:
                Reconstructed_image_save_dir = "/home/vault/iwfa/iwfa048h/grad_cam"
        else:
            Reconstructed_image_save_dir = "/home/vault/iwfa/iwfa048h/grad_cam"

        TRAINER_OPTUNA_DIR = "/home/vault/iwfa/iwfa048h/CNN/TRAINER_OPTUNA_DIR_TRIAL" 

        return non_trail_hp_writer, trail_hp_writer, weights_best_trial, weights_best_trial_inoptuna, CKPT_PATH, Misclassification_save_dir, new_path_result, TRAINER_OPTUNA_DIR, Reconstructed_image_save_dir, Gradcam_save_dir
