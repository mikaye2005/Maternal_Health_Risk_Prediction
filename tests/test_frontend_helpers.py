import math

import pytest

from app.app_helpers import build_assessment_summary, load_artifacts, predict_result, what_if
from src.config import FEATURES


def median_values(metadata):
    return {feature: metadata["feature_medians"][feature] for feature in FEATURES}


def test_prediction_helper_complete_and_what_if():
    model, metadata = load_artifacts()
    values = median_values(metadata)
    result = predict_result(model, metadata, values)
    assert result["label"] in metadata["label_mapping"]
    assert len(result["probabilities"]) == 3
    assert len(result["explanations"]) == 3
    adjusted = what_if(model, metadata, values, "Age", metadata["feature_ranges"]["Age"][0])
    assert adjusted["label"] in metadata["label_mapping"]


def test_prediction_helper_accepts_one_missing_and_rejects_two():
    model, metadata = load_artifacts()
    one_missing = median_values(metadata)
    one_missing["Age"] = math.nan
    result = predict_result(model, metadata, one_missing)
    assert result["missing"] == 1

    two_missing = median_values(metadata)
    two_missing["Age"] = math.nan
    two_missing["BS"] = math.nan
    with pytest.raises(ValueError):
        predict_result(model, metadata, two_missing)


def test_prediction_helper_rejects_invalid_pressure_relationship():
    model, metadata = load_artifacts()
    values = median_values(metadata)
    values["SystolicBP"] = 70
    values["DiastolicBP"] = 90
    with pytest.raises(ValueError, match="cannot be lower"):
        predict_result(model, metadata, values)


def test_assessment_summary_contains_measurements_and_disclaimer():
    model, metadata = load_artifacts()
    values = median_values(metadata)
    result = predict_result(model, metadata, values)
    summary = build_assessment_summary(metadata, values, result)
    assert "Predicted category" in summary
    assert "Systolic blood pressure" in summary
    assert "not a medical diagnosis" in summary
