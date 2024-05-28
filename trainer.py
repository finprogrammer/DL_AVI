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
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers, Trainer, callbacks
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from dataloader import (
    MultiViewDataset
)
from network import (
    DeepCNN
)
from dataloader import (
    MultiViewDataset,
    datatest,
    dataval,
) 
from misclassification import Data_post_Processor
from network import DeepCNN
from eval import evaluate
from make_folder import make_folder
from config_loader import load_config

mode = "train"  # Change this to "debug" when debugging
CONFI = load_config(mode)

# sinfo -Nel
# tensorboard --logdir=
# optuna-dashboard sqlite:///cv_trial.db --host 0.0.0.0 --port 7000

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
views = ["file_name"]
labels = ["label", "~label"]
num_classes = len(labels)
#epochs = 
batch_size = CONFI["BATCHSIZE"]
print(f"Batchsize={batch_size}")
validation_interval = CONFI["validation_interval"]
loss_func = torch.nn.CrossEntropyLoss(weight=None)
# loss_func = FocalLoss(gamma=4, alpha=[0.79, 0.21]) #alpha=[0.000726, 0.000191]
output_activation = torch.nn.Softmax(dim=1)  # Sigmoid()#Softmax(dim=1)
output_activation.name = "Softmax"
print(f"Image_mode={CONFI['IMAGE_MODE']}")
Sq_lite_file = "sqlite:///{}.db".format(CONFI["logger_version"])
H = CONFI["H"]
print("height:", H)
W = CONFI["W"]
print("height:", W)

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
)  # , pil_image_mode="L"
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
)  # , pil_image_mode="L"
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
)  # , pil_image_mode="L"
print(f"Test Normalize: {test_dataset.normalize}")
print(f"Test Transform: {test_dataset.transform}")
print(f"Test Feature_extraction: {test_dataset.Feature_extraction}")

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
mv_train_loader = DataLoader(
    train_dataset,
    sampler=sampler,
    shuffle=False,
    batch_size=batch_size,
    num_workers=4,
    drop_last=True,
)
mv_val_loader = DataLoader(
    val_dataset, shuffle=False, batch_size=batch_size, num_workers=4, drop_last=True
)
mv_test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=4)

non_trial_writter = loggers.TensorBoardLogger(
    save_dir=non_trail_hp_writer,
    version=CONFI["logger_version"],
    prefix="fu",
    name="Final_Hyperparameters_Used",
    log_graph=False,
)

skip_optuna = CONFI["SKIP_OPTUNA"]
if not skip_optuna:
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=4, n_warmup_steps=5
    )  #optuna.pruners.NopPruner()
    optimizer = None
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
        weight_decay = trial.suggest_uniform("weight_decay", 1e-5, 1e-1)
        
        if CONFI["Model"] == "Dinov2":
            n_layers = trial.suggest_int("n_layers", 1, 8)
            num_neurons_2 = [
                trial.suggest_int(f"num_neurons_layer{i}", 64, 512)
                for i in range(n_layers)
            ]
        else:
                num_neurons_2 = CONFI["NUM_NEURONS"]
                n_layers = CONFI["NLAYERS"]
        patience = trial.suggest_int("p_a", 7, 20)
        patience_lr = trial.suggest_int("patience_lr", 0, 6)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        selected_optimizer = "AdamW"
        dropout = trial.suggest_float("dropout_p", 0.2, 0.7)
        model_used = CONFI["Model"]
        #model.name = model_used

        if CONFI["Model"] == "Dinov2":
            dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            class DinoVisionTransformerClassifier(torch.nn.Module):
                def __init__(self):
                    super(DinoVisionTransformerClassifier, self).__init__()
                    self.transformer = dinov2_vits14
                    self.classifier = torch.nn.Sequential(torch.nn.Linear(384, num_neurons_2[0]))

                def forward(self, x):
                    x = self.transformer(x)
                    x = self.transformer.norm(x)
                    x = self.classifier(x)
                    return x
            model = DinoVisionTransformerClassifier() 
        else:    
            model = getattr(torchvision.models, model_used)(
                weights="IMAGENET1K_V1"
            )  # dropout=dropout,
            model.name = model_used

            if CONFI["Model"] == "efficientnet_v2_s":
                backbone_out_features = model.classifier[1].out_features
            else:
                backbone_out_features = getattr(model, CONFI["BACKBONE"]).out_features

        mv_train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            shuffle=False,
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

        best_nlayers = n_layers
        print("Best nlayers:", best_nlayers)
        best_endlayers = CONFI["ENDLAYERS"]
        print("Best endlayers:", best_endlayers)

        if CONFI["Model"] == "resnext50_32x4d":
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters())
            total_params = count_parameters(model)
            print(f"Total number of parameters: {total_params}")

            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad_(False)
            ###code to check if all the parameters are frozen
            # for param in model.parameters():
            #     print(param.requires_grad)
            ##Count parameters and unfreeze specified layers
            for i, (name, layer) in enumerate(model.named_children()):
                layer_params = count_parameters(layer)
                print(f"Parameters in {name}: {layer_params}")
            ##print the layers and the parameters frozen
            for i, (name, layer) in enumerate(model.named_children()):
                if best_nlayers <= i < best_endlayers:
                    layer_params = count_parameters(layer)
                    print(f"Parameters in {name} to be unfreezee {i}: {layer_params}")
                    for param in layer.parameters():
                        param.requires_grad_(True)

        elif CONFI["Model"] == "efficientnet_v2_s":
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters())
            total_params = count_parameters(model)
            print(f"Total number of parameters: {total_params}")

            for i, layer in enumerate(model.layer):
                layer_params = count_parameters(layer)
                print(f"Parameters in layer {i}: {layer_params}")

            for param in model.parameters():
                param.requires_grad_(False)

            for i, layer in enumerate(model.layer[n_layers : CONFI["ENDLAYERS"]]):
                layer_params = count_parameters(layer)
                print(f"Parameters to be unfreezee {i}: {layer_params}")

            for i in range(n_layers, CONFI["ENDLAYERS"]):
                for param in model.features[i].parameters():
                    param.requires_grad = True

        
        fusion_model = DeepCNN(
            backbone=model,
            num_classes=num_classes,
            optimizer=selected_optimizer,
            output_activation=output_activation,
            loss_func=loss_func,
            views=views,
            labels=labels,
            backbone_out=backbone_out_features,
            beta=CONFI["BETA"],
            num_layers = n_layers,
            num_neurons_2 = num_neurons_2,
            dropout = dropout,            
        )
        fusion_model.optimizer = getattr(torch.optim, selected_optimizer)(
            fusion_model.parameters(), lr=lr, weight_decay=weight_decay
        )
        fusion_model.lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(
            fusion_model.optimizer, factor=0.1, patience=patience_lr, verbose=True, cooldown=2
        )

        #loggers
        trial_writer = loggers.TensorBoardLogger(
            save_dir=trail_hp_writer,
            version=str(trial.number),
            prefix="ar",
            name=CONFI["optuna_tensorboard_study_name"],
            log_graph=False,
        )        

        #callbacks
        early_stop = callbacks.EarlyStopping(
            monitor=CONFI["QUANTITY_TO_OPTIMIZE"],
            mode=CONFI["DIRECTION_OF_OPTIMIZATION_SHORT"],
            patience=patience,
            verbose=True,
            min_delta=CONFI["MINDELTA"],
        )
        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor=CONFI["QUANTITY_TO_OPTIMIZE"]
        )
        lr_monitor = LearningRateMonitor(
            logging_interval="step", log_momentum=True, log_weight_decay=True
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
                pruning_callback,
                lr_monitor,
            ],
            fast_dev_run=CONFI["fast_dev_run"],
            default_root_dir=TRAINER_OPTUNA_DIR,
            # profiler="advanced"
        )

        hyperparameters = dict(selected_optimizer=selected_optimizer)
        trainer.logger.log_hyperparams(hyperparameters)

        trainer.fit(fusion_model, mv_train_loader, mv_val_loader)
        val_loss = trainer.callback_metrics['val_loss']
        f1_notlabel = trainer.callback_metrics['f1_notlabel']

        return val_loss#f1_notlabel#, val_loss

    study = optuna.create_study(
        directions=[CONFI["DIRECTION_OF_OPTIMIZATION"]],#, 'minimize'
        study_name=CONFI["logger_version"],
        storage=Sq_lite_file,
        load_if_exists=True,
        pruner=pruner,
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
    print(f"MINDELTA={CONFI['MINDELTA']}")
    best_dropout = CONFI["DROPOUT"]
    print("Best Dropout:", best_dropout)
    num_layers = CONFI["NLAYERS"]
    print("num_layers:", num_layers)
    num_neurons_2 = CONFI["NUM_NEURONS"]
    print("num_neurons_2:", num_neurons_2)    
    model_used = CONFI["Model"]
    print(model_used)

    if CONFI["Model"] == "Dinov2":
        dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        class DinoVisionTransformerClassifier(torch.nn.Module):
            def __init__(self):
                super(DinoVisionTransformerClassifier, self).__init__()
                self.transformer = dinov2_vits14
                self.classifier = torch.nn.Sequential(torch.nn.Linear(384, num_neurons_2[0]))

            def forward(self, x):
                x = self.transformer(x)
                x = self.transformer.norm(x)
                x = self.classifier(x)
                return x
        model = DinoVisionTransformerClassifier() 
        backbone_out_features = 1000

    else:    
        model = getattr(torchvision.models, model_used)(
            weights="IMAGENET1K_V1"
        )  # dropout=best_dropout,
        

        if CONFI["Model"] == "efficientnet_v2_s":
            backbone_out_features = model.classifier[1].out_features
        else:
            backbone_out_features = getattr(model, CONFI["BACKBONE"]).out_features

    model.name = model_used
    best_nlayers = CONFI["NLAYERS"]
    print("Best nlayers:", best_nlayers)
    best_endlayers = CONFI["ENDLAYERS"]
    print("Best endlayers:", best_endlayers)

    if CONFI["Model"] == "resnext50_32x4d":

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
            
    elif CONFI["Model"] == "efficientnet_v2_s":

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

    skip_freezing_batch_normalization = CONFI["SKIP_FREEZING_BATCHNORMALIZATION"]
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

    non_trial_writter.log_hyperparams(
        {
            "model": model.name,
            "epochs": CONFI["EPOCHS"],
            "batch_size": batch_size,
            "optimizer": best_optimizer,
            "lr": best_lr,
            "loss_func": loss_func,
            "output_activation": output_activation.name,
        }
    )

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

    trainer = Trainer(
        accelerator=CONFI["ACCELERATOR"],
        devices=DEVICES,
        max_epochs=CONFI["EPOCHS"],
        log_every_n_steps=15,
        overfit_batches=0,
        callbacks=[checkpoint_callback, early_stop, lr_monitor],
        logger=[non_trial_writter],
        fast_dev_run=CONFI["fast_dev_run"],
        check_val_every_n_epoch=validation_interval,
        default_root_dir=weights_best_trial,
        enable_checkpointing=True,
        # profiler="simple"
    )

    fusion_model = DeepCNN(
        backbone=model,
        backbone_out=backbone_out_features,
        num_classes=num_classes,
        optimizer=best_optimizer,
        output_activation=output_activation,
        loss_func=loss_func,
        views=views,
        labels=labels,
        beta=CONFI["BETA"],
        num_layers = num_layers,
        num_neurons_2 = num_neurons_2,        
        dropout = best_dropout,
        Gradcam_save_dir = Gradcam_save_dir,
        Reconstructed_image_save_dir = Reconstructed_image_save_dir,
        ROOT_DIR = ROOT_DIR
    )

    fusion_model.optimizer = getattr(torch.optim, best_optimizer)(
        fusion_model.parameters(), lr=best_lr, weight_decay=best_weight_decay
    )
    fusion_model.lr_schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(
        fusion_model.optimizer, factor=0.1, patience=best_patience_lr, verbose=True
    )
    trainer.fit(fusion_model, mv_train_loader, mv_val_loader, ckpt_path=None)
    trainer.test(dataloaders=mv_test_loader, ckpt_path=CONFI["CKPT_PATH"], verbose=True)

    print(
        "Prediction result of the validation dataset__________________________________________________________________________________"
    )
    mv_val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=1, num_workers=4, drop_last=True
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
    preds = trainer.predict(
        dataloaders=mv_val_loader, ckpt_path=CONFI["CKPT_PATH"]
    ) 
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
    preds = trainer.predict(dataloaders=mv_test_loader, ckpt_path=CONFI["CKPT_PATH"])
    result = evaluate(test_dataset, preds, labels)

    skip_misclassification = CONFI['SKIP_MISCLASSIFY']
    if not skip_misclassification:
        pred_val, selected_values_val, pred_def_val, selected_values_def_val = validation_processor.process_samples(Predictions_v, Prediction_Def_v, Indices_v, Indices_Def_v, dataval)
        # plot_validation_samples method
        validation_processor.plot__samples(ROOT_DIR, selected_values_def_val, selected_values_val, pred_val, pred_def_val, Misclassification_save_dir)
        # compute_validation_list method
        validation_processor.compute_list(dataval, FP_V, FN_V)

        print(
        "Prediction result of the test dataset__________________________________________________________________________________"
        )
        test_processor = Data_post_Processor(Indices_T, Predictions_T, Indices_Def_T, Prediction_Def_T, FP_T, FN_T, datatest, dataval)
        pred_val, selected_values_val, pred_def_val, selected_values_def_val = test_processor.process_samples(Predictions_T, Prediction_Def_T, Indices_T, Indices_Def_T, datatest)
        # plot_validation_samples method
        test_processor.plot__samples(ROOT_DIR, selected_values_def_val, selected_values_val, pred_val, pred_def_val, Misclassification_save_dir)
        # compute_validation_list method
        test_processor.compute_list(datatest, FP_T, FN_T)
