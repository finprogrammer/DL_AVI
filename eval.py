import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import torch
from config_loader import load_config

mode = "train_2"  # Change this to "debug" when debugging
CONFI = load_config(mode)

if mode == "train":
    DEVICES = 1 if torch.cuda.is_available() else None
else:
    DEVICES = "auto"


def evaluate(dataset, preds, classes):
    result = {}
    result_train = {}
    if CONFI["Model"] == "Dinov2":
        gts = np.zeros((len(preds)))
        predsn = np.zeros((len(preds)))
    else:
        gts = np.zeros((len(preds), len(classes)))
        predsn = np.zeros((len(preds), len(classes)))

    for id_ in range(len(preds)):
        gt = dataset[id_]["label"]
        pred = preds[id_]
        # pred = torch.sigmoid(pred)
        gts[id_] = gt.numpy()
        predsn[id_] = pred.numpy()

    print(predsn[:10])
    if CONFI["Model"] == "Dinov2":
        y_true = gts
        y_pred = predsn
        print(y_pred)
    else:
        y_true = gts.argmax(axis=1)
        y_pred = predsn.argmax(axis=1)
        print(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    tp, fn, fp, tn = cm.ravel() 
    print("TP = ", tp, "FP = ", fp, "FN = ", fn, "TN = ", tn)
    result["accuracy"] = accuracy_score(y_true, y_pred)
    result_train["accuracy"] = accuracy_score(y_true, y_pred)
    result["true_positive"] = tp
    result["false_negative"] = fn
    result["micro_precision"] = precision_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    result["micro_recall"] = recall_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    result["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)

    result["macro_precision"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    result["macro_recall"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    result["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    result["f1_label"] = f1_score(y_true, y_pred, pos_label=0)
    result["f1_~label"] = f1_score(y_true, y_pred, pos_label=1)
    print("\nClassification Report\n")
    print(classification_report(y_true, y_pred, zero_division=0, target_names=classes))
    return result
