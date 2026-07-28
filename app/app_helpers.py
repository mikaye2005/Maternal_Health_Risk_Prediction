from pathlib import Path
import json
import joblib
import pandas as pd

from src.config import FEATURES, MODEL_PATH, METADATA_PATH
from src.explanations import local_permutation_explanation
from src.missing_measurements import validate_missing_measurements
from src.uncertainty import is_uncertain


def load_artifacts(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    return joblib.load(Path(model_path)), json.loads(Path(metadata_path).read_text(encoding="utf-8"))


def validate_inputs(values, ranges):
    row = pd.DataFrame([values], columns=FEATURES)
    validate_missing_measurements(row)
    for feature in FEATURES:
        value = row.loc[0, feature]
        if pd.notna(value):
            if value < 0:
                raise ValueError(f"{feature} cannot be negative.")
            low, high = ranges[feature]
            if not low <= value <= high:
                raise ValueError(f"{feature} must be within the observed range {low:g}–{high:g}.")
    return row


def predict_result(model, metadata, values):
    row = validate_inputs(values, metadata["feature_ranges"])
    probabilities = model.predict_proba(row)[0]
    index = int(probabilities.argmax())
    label = str(model.classes_[index])
    explanations = local_permutation_explanation(model, row, metadata["feature_medians"], index)
    return {
        "label": label,
        "probability": float(probabilities[index]),
        "probabilities": {str(c): float(p) for c, p in zip(model.classes_, probabilities)},
        "uncertain": is_uncertain(probabilities, metadata["uncertainty_threshold"]),
        "explanations": explanations,
        "missing": int(row.isna().sum(axis=1).iloc[0]),
    }


def what_if(model, metadata, values, feature, adjusted_value):
    adjusted = dict(values)
    adjusted[feature] = adjusted_value
    return predict_result(model, metadata, adjusted)
