import pandas as pd
import pytest

from src.feature_engineering import add_engineered_features


def test_feature_formulas():
    result = add_engineered_features(pd.DataFrame([{
        "Age": 30, "SystolicBP": 120, "DiastolicBP": 75,
        "BS": 7, "BodyTemp": 98, "HeartRate": 70,
    }]))
    assert result.loc[0, "PulsePressure"] == 45
    assert result.loc[0, "MeanArterialPressure"] == pytest.approx(90)
    assert result.loc[0, "AgeBand"] == 1
