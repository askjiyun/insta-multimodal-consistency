import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)

LABELS = [0, 1, 2, 3, 4]

FOLD_METRIC_COLUMNS = ["model", "fold", "accuracy", "macro_f1", "weighted_f1", "mae", "qwk", "adj_acc"]


def adjacent_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) <= 1))


def compute_metrics(y_true, y_pred, labels=LABELS):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    mae = mean_absolute_error(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
    adj_acc = adjacent_accuracy(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {
        str(label): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "mae": float(mae),
        "qwk": float(qwk),
        "adj_acc": float(adj_acc),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def fold_metrics_row(model, fold, y_true, y_pred, labels=LABELS):
    m = compute_metrics(y_true, y_pred, labels=labels)
    return {
        "model": model,
        "fold": fold,
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"],
        "mae": m["mae"],
        "qwk": m["qwk"],
        "adj_acc": m["adj_acc"],
    }
