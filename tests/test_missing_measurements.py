import json
import joblib
import numpy as np
import pytest

from app.app_helpers import predict_result
from src.config import METADATA_PATH, MODEL_PATH
from src.uncertainty import is_uncertain


@pytest.fixture(scope="module")
def artifacts():
    return joblib.load(MODEL_PATH), json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def test_one_missing_measurement_works(artifacts):
    model, metadata = artifacts
    values = dict(metadata["feature_medians"])
    values["BS"] = np.nan
    result = predict_result(model, metadata, values)
    assert result["missing"] == 1
    assert result["label"] in metadata["label_mapping"]


def test_two_missing_measurements_rejected(artifacts):
    model, metadata = artifacts
    values = dict(metadata["feature_medians"])
    values["BS"] = np.nan
    values["Age"] = np.nan
    with pytest.raises(ValueError, match="maximum of one"):
        predict_result(model, metadata, values)


def test_uncertainty_boundary():
    assert is_uncertain([0.2, 0.3, 0.5], 0.6)
    assert not is_uncertain([0.1, 0.2, 0.7], 0.6)
