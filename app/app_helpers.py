from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd

from src.config import FEATURES, METADATA_PATH, MODEL_PATH
from src.explanations import local_median_replacement_explanation
from src.missing_measurements import validate_missing_measurements
from src.uncertainty import is_uncertain


DISPLAY_LABELS = {
    "Age": "Age",
    "SystolicBP": "Systolic blood pressure",
    "DiastolicBP": "Diastolic blood pressure",
    "BS": "Blood sugar",
    "BodyTemp": "Body temperature",
    "HeartRate": "Heart rate",
}

UNITS = {
    "Age": "years",
    "SystolicBP": "mmHg",
    "DiastolicBP": "mmHg",
    "BS": "mmol/L",
    "BodyTemp": "degrees Fahrenheit",
    "HeartRate": "beats per minute",
}


def load_artifacts(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    model = joblib.load(Path(model_path))
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return model, metadata


def validate_inputs(values, ranges):
    row = pd.DataFrame([values], columns=FEATURES)
    validate_missing_measurements(row)
    for feature in FEATURES:
        value = row.loc[0, feature]
        if pd.notna(value):
            if value < 0:
                raise ValueError(f"{DISPLAY_LABELS[feature]} cannot be negative.")
            low, high = ranges[feature]
            if not float(low) <= float(value) <= float(high):
                raise ValueError(
                    f"{DISPLAY_LABELS[feature]} must be within the observed dataset range "
                    f"{low:g}-{high:g}."
                )
    if pd.notna(row.loc[0, "SystolicBP"]) and pd.notna(row.loc[0, "DiastolicBP"]):
        if row.loc[0, "SystolicBP"] < row.loc[0, "DiastolicBP"]:
            raise ValueError("Systolic blood pressure cannot be lower than diastolic blood pressure.")
    return row


def predict_result(model, metadata, values):
    row = validate_inputs(values, metadata["feature_ranges"])
    probabilities = model.predict_proba(row)[0]
    class_order = [str(item) for item in model.classes_]
    index = int(probabilities.argmax())
    label = class_order[index]
    explanations = local_median_replacement_explanation(
        model,
        row,
        metadata["feature_medians"],
        index,
    )
    return {
        "label": label,
        "display_label": metadata["label_mapping"][label],
        "probability": float(probabilities[index]),
        "probabilities": {
            class_label: float(probability)
            for class_label, probability in zip(class_order, probabilities)
        },
        "uncertain": is_uncertain(probabilities, metadata["uncertainty_threshold"]),
        "explanations": explanations,
        "missing": int(row.isna().sum(axis=1).iloc[0]),
    }


def what_if(model, metadata, values, feature, adjusted_value):
    adjusted = dict(values)
    adjusted[feature] = adjusted_value
    return predict_result(model, metadata, adjusted)


def build_assessment_summary(metadata, values, result) -> str:
    lines = [
        "MamaCare assessment summary",
        "",
        f"Predicted category: {result['display_label']}",
        f"Model status: {'Review recommended' if result['uncertain'] else 'Higher-confidence model output'}",
        f"Predicted-class model score: {result['probability']:.1%}",
        "",
        "Measurements used:",
    ]
    for feature in FEATURES:
        value = values.get(feature)
        display = "Unavailable" if pd.isna(value) else f"{float(value):g} {UNITS[feature]}"
        lines.append(f"- {DISPLAY_LABELS[feature]}: {display}")
    lines.extend([
        "",
        "Class outputs:",
    ])
    for label, probability in sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {metadata['label_mapping'][label]}: {probability:.1%}")
    lines.extend([
        "",
        "Important: This is an academic model output, not a medical diagnosis. "
        "It must not be used for treatment, triage or emergency decisions.",
    ])
    return "\n".join(lines)
