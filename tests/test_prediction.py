import json
import joblib
import numpy as np
import pytest

from app.app_helpers import predict_result, validate_inputs, what_if
from src.config import FEATURES, METADATA_PATH, MODEL_PATH


@pytest.fixture(scope="module")
def artifacts():
    return joblib.load(MODEL_PATH), json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def normal_values(metadata):
    return dict(metadata["feature_medians"])


def test_model_loading_and_output(artifacts):
    model, metadata = artifacts
    values = normal_values(metadata)
    result = predict_result(model, metadata, values)
    assert result["label"] in metadata["label_mapping"]
    assert len(result["probabilities"]) == 3
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert len(result["explanations"]) == 3


def test_prediction_reproducible(artifacts):
    model, metadata = artifacts
    first = predict_result(model, metadata, normal_values(metadata))
    second = predict_result(model, metadata, normal_values(metadata))
    assert first["probabilities"] == second["probabilities"]


def test_label_mapping_and_threshold(artifacts):
    _, metadata = artifacts
    assert set(metadata["label_mapping"]) == {"low risk", "mid risk", "high risk"}
    assert 0 < metadata["uncertainty_threshold"] < 1


def test_negative_and_out_of_range_rejected(artifacts):
    _, metadata = artifacts
    values = normal_values(metadata)
    values["Age"] = -1
    with pytest.raises(ValueError):
        validate_inputs(values, metadata["feature_ranges"])
    values["Age"] = 999
    with pytest.raises(ValueError):
        validate_inputs(values, metadata["feature_ranges"])


def test_what_if(artifacts):
    model, metadata = artifacts
    values = normal_values(metadata)
    changed = what_if(model, metadata, values, "BS", values["BS"] + 1)
    assert changed["label"] in metadata["label_mapping"]
    assert sum(changed["probabilities"].values()) == pytest.approx(1)
