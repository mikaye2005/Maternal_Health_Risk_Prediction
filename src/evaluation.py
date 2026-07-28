import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, precision_score, recall_score,
)


def classification_metrics(y_true, y_pred, probabilities=None):
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "high_risk_recall": recall_score(
            np.asarray(y_true) == "high risk", np.asarray(y_pred) == "high risk",
            zero_division=0,
        ),
    }
    if probabilities is not None:
        result["log_loss"] = log_loss(y_true, probabilities, labels=["high risk", "low risk", "mid risk"])
    return result
