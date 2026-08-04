from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import FEATURES


class MaternalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X, columns=getattr(X, "columns", FEATURES)).copy()
        frame["PulsePressure"] = frame["SystolicBP"] - frame["DiastolicBP"]
        frame["MeanArterialPressure"] = (frame["SystolicBP"] + 2 * frame["DiastolicBP"]) / 3
        frame["AgeBand"] = pd.cut(
            frame["Age"],
            bins=[-np.inf, 19, 34, 49, np.inf],
            labels=False,
            include_lowest=True,
        ).astype(float)
        return frame

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURES + ["PulsePressure", "MeanArterialPressure", "AgeBand"])


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    return MaternalFeatureEngineer().transform(frame)


def age_group_labels(series: pd.Series) -> pd.Series:
    return pd.cut(
        series,
        bins=[-np.inf, 19, 34, 49, np.inf],
        labels=["<=19", "20-34", "35-49", ">=50"],
        include_lowest=True,
    ).astype(str)
