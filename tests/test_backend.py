import json
import hashlib
from importlib.metadata import version as installed_version
import platform

import joblib
import pandas as pd
import pytest

from src.config import FEATURES, METADATA_PATH, MODEL_PATH, RAW_DATA, TABLES
from src.data_loader import feature_signature, validate_dataset
from src.feature_engineering import add_engineered_features
from src.missing_measurements import validate_missing_measurements
from src.evaluation import classification_metrics


def test_dataset_schema_shape_and_labels():
    frame = validate_dataset(RAW_DATA)
    assert frame.shape == (1014, 7)
    assert frame.columns.tolist() == FEATURES + ["RiskLevel"]
    assert set(frame["RiskLevel"].str.lower()) == {"low risk", "mid risk", "high risk"}


def test_feature_engineering_formulas_and_age_band():
    row = pd.DataFrame([{
        "Age": 30,
        "SystolicBP": 120,
        "DiastolicBP": 80,
        "BS": 7.0,
        "BodyTemp": 98.0,
        "HeartRate": 76,
    }])
    output = add_engineered_features(row)
    assert output.loc[0, "PulsePressure"] == 40
    assert output.loc[0, "MeanArterialPressure"] == pytest.approx(93.333333, rel=1e-4)
    assert output.loc[0, "AgeBand"] == 1


def test_age_band_boundaries():
    rows = pd.DataFrame([
        {"Age": age, "SystolicBP": 120, "DiastolicBP": 80, "BS": 7, "BodyTemp": 98, "HeartRate": 76}
        for age in [19, 20, 35, 50]
    ])
    assert add_engineered_features(rows)["AgeBand"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_one_missing_allowed_and_two_rejected():
    one = pd.DataFrame([[None, 120, 80, 7, 98, 76]], columns=FEATURES)
    validate_missing_measurements(one)
    two = pd.DataFrame([[None, 120, None, 7, 98, 76]], columns=FEATURES)
    with pytest.raises(ValueError):
        validate_missing_measurements(two)


def test_saved_splits_have_no_measurement_signature_overlap():
    train = pd.read_csv(TABLES / "train_split.csv")
    validation = pd.read_csv(TABLES / "validation_split.csv")
    test = pd.read_csv(TABLES / "test_split.csv")
    train_signatures = set(feature_signature(train))
    validation_signatures = set(feature_signature(validation))
    test_signatures = set(feature_signature(test))
    assert not train_signatures & validation_signatures
    assert not train_signatures & test_signatures
    assert not validation_signatures & test_signatures


def test_model_artifact_and_metadata_are_consistent():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    row = pd.DataFrame([{feature: metadata["feature_medians"][feature] for feature in FEATURES}])
    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    assert prediction in {"low risk", "mid risk", "high risk"}
    assert len(probabilities) == 3
    assert probabilities.sum() == pytest.approx(1.0)
    assert metadata["primary_metric"] == "weighted_f1"
    assert metadata["features"] == FEATURES
    assert metadata["class_order"] == [str(label) for label in model.classes_]
    assert isinstance(metadata["calibrated"], bool)
    assert metadata["final_calibration_method"]
    assert metadata["random_seed"] == 42
    assert metadata["model_artifact_sha256"] == hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    assert metadata["selected_validation_metrics"]["weighted_f1"] >= 0
    assert metadata["split_sizes"] == {"train": 608, "validation": 203, "test": 203}
    assert 0.40 <= metadata["uncertainty_threshold"] <= 0.85
    assert metadata["signature_overlap_checks"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert metadata["python_version"] == platform.python_version()
    for package_name in ["pandas", "numpy", "scikit-learn", "xgboost", "streamlit", "joblib"]:
        assert metadata["package_versions"][package_name] == installed_version(package_name)


def test_raw_dataset_hash_matches_metadata():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(RAW_DATA.read_bytes()).hexdigest()
    assert digest == "a1f7025719f84715096e0d1f95ae2e56b57809b9b15449e1836c96a7d976ae9b"
    assert metadata["dataset_sha256"] == digest


def test_saved_model_reproduces_recorded_test_metrics():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    test = pd.read_csv(TABLES / "test_split.csv")
    predictions = model.predict(test[FEATURES])
    probabilities = model.predict_proba(test[FEATURES])
    recomputed = classification_metrics(test["RiskLevel"], predictions, probabilities)
    for metric, expected in metadata["test_metrics"].items():
        assert recomputed[metric] == pytest.approx(expected, abs=1e-12)


def test_required_evaluation_tables_exist_and_have_content():
    required = [
        "baseline_vs_model.csv",
        "model_comparison_validation.csv",
        "class_report_test.csv",
        "feature_ablation.csv",
        "clustering_metrics.csv",
        "disaggregated_age_evaluation.csv",
        "equal_opportunity_summary.csv",
        "error_analysis_records.csv",
        "error_type_summary.csv",
        "error_rate_by_age.csv",
        "top_confident_errors.csv",
        "contradictory_measurement_signatures.csv",
        "conflicting_signature_performance.csv",
        "cluster_profiles.csv",
        "cluster_risk_distribution.csv",
        "neural_network_history.csv",
        "uncertainty_thresholds.csv",
        "split_distribution.csv",
        "test_confusion_matrix.csv",
        "train_split.csv",
        "validation_split.csv",
        "test_split.csv",
    ]
    for file_name in required:
        path = TABLES / file_name
        assert path.exists(), file_name
        assert path.stat().st_size > 20, file_name


def test_validation_comparison_is_ranked_and_complete():
    comparison = pd.read_csv(TABLES / "model_comparison_validation.csv")
    expected = {
        "MajorityBaseline", "LogisticRegression", "DecisionTree", "RandomForest", "SVC",
        "GradientBoosting", "XGBoost", "ShallowMLP", "DeepMLP", "SoftVoting",
    }
    assert set(comparison["model"]) == expected
    assert comparison[["weighted_f1", "macro_f1", "high_risk_recall"]].notna().all().all()
    successful = comparison.dropna(subset=["weighted_f1"])
    ranked = successful.sort_values(
        ["weighted_f1", "macro_f1", "high_risk_recall"],
        ascending=False,
    )
    assert successful["model"].tolist() == ranked["model"].tolist()


def test_clustering_and_neural_validation_evidence():
    clustering = pd.read_csv(TABLES / "clustering_metrics.csv")
    assert {"method", "clusters", "silhouette", "davies_bouldin"} <= set(clustering.columns)
    assert {"KMeans", "Hierarchical"} <= set(clustering["method"])
    history = pd.read_csv(TABLES / "neural_network_history.csv")
    assert set(history["model"]) == {"ShallowMLP", "DeepMLP"}
    assert history["signature_overlap"].max() == 0
