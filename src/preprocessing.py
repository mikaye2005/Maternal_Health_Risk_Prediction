from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, ENGINEERED
from src.feature_engineering import MaternalFeatureEngineer


def build_preprocessor(engineered: bool = True):
    columns = FEATURES + ENGINEERED if engineered else FEATURES
    steps = []
    if engineered:
        steps.append(("features", MaternalFeatureEngineer()))
    steps.append(
        (
            "prepare",
            ColumnTransformer(
                [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                       ("scaler", StandardScaler())]), columns)],
                remainder="drop",
                verbose_feature_names_out=False,
            ),
        )
    )
    return Pipeline(steps)
