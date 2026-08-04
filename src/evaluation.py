from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

from src.config import LOG_LOSS_CLASS_ORDER, TARGET
from src.feature_engineering import age_group_labels


def classification_metrics(y_true, y_pred, probabilities=None) -> dict:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "high_risk_recall": float(recall_score(
            np.asarray(y_true) == "high risk",
            np.asarray(y_pred) == "high risk",
            zero_division=0,
        )),
    }
    if probabilities is not None:
        result["log_loss"] = float(log_loss(y_true, probabilities, labels=LOG_LOSS_CLASS_ORDER))
    return result


def class_report_table(y_true, y_pred) -> pd.DataFrame:
    rows = []
    for label in ["high risk", "low risk", "mid risk"]:
        mask_true = np.asarray(y_true) == label
        mask_pred = np.asarray(y_pred) == label
        rows.append({
            "class": label,
            "precision": precision_score(mask_true, mask_pred, zero_division=0),
            "recall": recall_score(mask_true, mask_pred, zero_division=0),
            "f1": f1_score(mask_true, mask_pred, zero_division=0),
            "support": int(mask_true.sum()),
        })
    return pd.DataFrame(rows)


def disaggregated_by_age(test_frame: pd.DataFrame, predictions) -> pd.DataFrame:
    work = test_frame.copy()
    work["AgeGroup"] = age_group_labels(work["Age"])
    work["predicted"] = np.asarray(predictions)
    rows = []
    for group in ["<=19", "20-34", "35-49", ">=50"]:
        part = work[work["AgeGroup"] == group]
        if part.empty:
            continue
        metrics = classification_metrics(part[TARGET], part["predicted"])
        high_count = int((part[TARGET] == "high risk").sum())
        rows.append({"AgeGroup": group, "N": len(part), "HighRiskN": high_count, **metrics})
    return pd.DataFrame(rows)


def baseline_predictions(y_train, n_rows: int):
    majority = pd.Series(y_train).mode().iloc[0]
    return np.array([majority] * n_rows), majority


def equal_opportunity_summary(disaggregated: pd.DataFrame) -> dict:
    valid = disaggregated.loc[disaggregated["HighRiskN"] > 0, "high_risk_recall"]
    if valid.empty:
        return {"minimum": None, "maximum": None, "gap": None}
    return {
        "minimum": float(valid.min()),
        "maximum": float(valid.max()),
        "gap": float(valid.max() - valid.min()),
    }
