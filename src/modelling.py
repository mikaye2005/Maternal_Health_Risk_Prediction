from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor


def model_candidates(seed=42, engineered=True):
    prep = lambda: build_preprocessor(engineered)
    estimators = {
        "Dummy": DummyClassifier(strategy="prior"),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "DecisionTree": DecisionTreeClassifier(max_depth=7, min_samples_leaf=3, class_weight="balanced", random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
        "SVC": SVC(C=3, probability=True, class_weight="balanced", random_state=seed),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=150, learning_rate=.05, max_depth=2, random_state=seed),
        "ShallowMLP": MLPClassifier(hidden_layer_sizes=(32,), max_iter=1200, early_stopping=True, random_state=seed),
        "DeepMLP": MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=1500, early_stopping=True, random_state=seed),
    }
    return {name: Pipeline([("preprocess", prep()), ("model", model)])
            for name, model in estimators.items()}


def voting_candidate(seed=42):
    estimators = [
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ("svc", SVC(C=3, probability=True, class_weight="balanced", random_state=seed)),
    ]
    return Pipeline([("preprocess", build_preprocessor(True)),
                     ("model", VotingClassifier(estimators=estimators, voting="soft"))])
