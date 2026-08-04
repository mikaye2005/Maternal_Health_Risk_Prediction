from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import FEATURES, TRAINING_GROUP_COLUMN
from src.preprocessing import build_preprocessor


class XGBoostLabelClassifier(ClassifierMixin, BaseEstimator):
    """Allow XGBoost to train on the dataset's string class labels."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y):
        from xgboost import XGBClassifier

        self.encoder_ = LabelEncoder()
        y_encoded = self.encoder_.fit_transform(y)
        self.classes_ = self.encoder_.classes_
        self.model_ = XGBClassifier(
            n_estimators=260,
            learning_rate=0.04,
            max_depth=3,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X, y_encoded)
        return self

    def predict(self, X):
        encoded = self.model_.predict(X).astype(int)
        return self.encoder_.inverse_transform(encoded)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class MLPLabelClassifier(ClassifierMixin, BaseEstimator):
    """Feed-forward network with group-aware stopping and a full-data refit."""

    def __init__(
        self,
        hidden_layer_sizes=(32,),
        alpha=0.001,
        learning_rate_init=0.001,
        batch_size=32,
        max_iter=900,
        validation_fraction=0.20,
        n_iter_no_change=35,
        random_state=42,
        engineered=True,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.engineered = engineered

    def fit(self, X, y):
        frame = pd.DataFrame(X).copy()
        missing_features = [feature for feature in FEATURES if feature not in frame.columns]
        if missing_features:
            raise ValueError(f"MLP training data is missing required features: {missing_features}")
        feature_frame = frame[FEATURES].copy()
        if TRAINING_GROUP_COLUMN in frame.columns:
            groups = frame[TRAINING_GROUP_COLUMN].astype(str).to_numpy()
        else:
            groups = pd.util.hash_pandas_object(feature_frame, index=False).astype(str).to_numpy()

        self.encoder_ = LabelEncoder()
        target = pd.Series(y).reset_index(drop=True)
        encoded = self.encoder_.fit_transform(target)
        self.classes_ = self.encoder_.classes_

        splitter = StratifiedGroupKFold(
            n_splits=max(2, int(round(1 / self.validation_fraction))),
            shuffle=True,
            random_state=self.random_state,
        )
        fit_idx, validation_idx = next(splitter.split(feature_frame, target, groups))
        fit_groups = set(groups[fit_idx])
        validation_groups = set(groups[validation_idx])
        self.internal_fit_records_ = int(len(fit_idx))
        self.internal_validation_records_ = int(len(validation_idx))
        self.internal_validation_signature_overlap_ = len(fit_groups & validation_groups)
        if self.internal_validation_signature_overlap_:
            raise RuntimeError("Measurement-signature overlap detected in MLP early-stopping split.")

        selection_preprocessor = build_preprocessor(self.engineered)
        X_fit = selection_preprocessor.fit_transform(feature_frame.iloc[fit_idx], target.iloc[fit_idx])
        X_validation = selection_preprocessor.transform(feature_frame.iloc[validation_idx])
        y_fit = encoded[fit_idx]
        y_validation = encoded[validation_idx]

        selection_model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            batch_size=self.batch_size,
            max_iter=1,
            early_stopping=False,
            warm_start=True,
            random_state=self.random_state,
        )

        self.loss_curve_ = []
        self.validation_scores_ = []
        best_score = -np.inf
        best_epoch = 1
        stale_epochs = 0
        encoded_classes = np.arange(len(self.classes_))
        for epoch in range(1, self.max_iter + 1):
            selection_model.partial_fit(X_fit, y_fit, classes=encoded_classes)
            validation_score = float(accuracy_score(y_validation, selection_model.predict(X_validation)))
            self.loss_curve_.append(float(selection_model.loss_))
            self.validation_scores_.append(validation_score)
            if validation_score > best_score + 1e-4:
                best_score = validation_score
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.n_iter_no_change:
                break

        self.best_epoch_ = int(best_epoch)
        self.preprocessor_ = build_preprocessor(self.engineered)
        transformed = self.preprocessor_.fit_transform(feature_frame, target)
        self.model_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            batch_size=self.batch_size,
            max_iter=self.best_epoch_,
            early_stopping=False,
            tol=0.0,
            n_iter_no_change=self.best_epoch_ + 1,
            random_state=self.random_state,
        )
        self.model_.fit(transformed, encoded)
        self.refit_loss_curve_ = list(self.model_.loss_curve_)
        self.n_iter_ = self.model_.n_iter_
        if self.n_iter_ != self.best_epoch_:
            raise RuntimeError(
                "The full-data MLP refit did not complete the validation-selected epoch count."
            )
        return self

    def predict(self, X):
        frame = pd.DataFrame(X)[FEATURES]
        encoded = self.model_.predict(self.preprocessor_.transform(frame)).astype(int)
        return self.encoder_.inverse_transform(encoded)

    def predict_proba(self, X):
        frame = pd.DataFrame(X)[FEATURES]
        return self.model_.predict_proba(self.preprocessor_.transform(frame))


def _pipeline(model, engineered: bool = True) -> Pipeline:
    if isinstance(model, MLPLabelClassifier):
        model.engineered = engineered
        return Pipeline([("model", model)])
    return Pipeline([
        ("preprocess", build_preprocessor(engineered)),
        ("model", model),
    ])


def model_candidates(seed: int = 42, engineered: bool = True) -> dict[str, Pipeline]:
    """Return the full capstone comparison set using one consistent pipeline."""

    estimators = {
        "MajorityBaseline": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=seed,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=7,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "SVC": SVC(
            C=3,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=2,
            random_state=seed,
        ),
        "XGBoost": XGBoostLabelClassifier(random_state=seed),
        "ShallowMLP": MLPLabelClassifier(
            hidden_layer_sizes=(32,),
            alpha=0.001,
            learning_rate_init=0.001,
            batch_size=32,
            max_iter=900,
            validation_fraction=0.20,
            n_iter_no_change=35,
            random_state=seed,
        ),
        "DeepMLP": MLPLabelClassifier(
            hidden_layer_sizes=(64, 32, 16),
            alpha=0.003,
            learning_rate_init=0.0007,
            batch_size=32,
            max_iter=1100,
            validation_fraction=0.20,
            n_iter_no_change=45,
            random_state=seed,
        ),
    }
    return {name: _pipeline(model, engineered) for name, model in estimators.items()}


def voting_candidate(seed: int = 42, engineered: bool = True) -> Pipeline:
    estimators = [
        ("lr", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed)),
        ("rf", RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )),
        ("svc", SVC(C=3, probability=True, class_weight="balanced", random_state=seed)),
    ]
    return _pipeline(VotingClassifier(estimators=estimators, voting="soft"), engineered)


def make_candidate(name: str, seed: int = 42, engineered: bool = True) -> Pipeline:
    if name == "SoftVoting":
        return voting_candidate(seed=seed, engineered=engineered)
    candidates = model_candidates(seed=seed, engineered=engineered)
    if name not in candidates:
        raise KeyError(f"Unknown model candidate: {name}")
    return candidates[name]
