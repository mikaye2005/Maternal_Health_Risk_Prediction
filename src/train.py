from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.clustering import clustering_analysis
from src.config import (
    CLASS_ORDER, CLEAN_DATA, FEATURES, METADATA_PATH, MODEL_PATH, RAW_DATA, ROOT, SEED,
    TARGET, UCI_URL,
)
from src.data_loader import clean_dataset, validate_dataset
from src.evaluation import classification_metrics
from src.missing_measurements import simulate_one_missing
from src.modelling import model_candidates, voting_candidate
from src.preprocessing import build_preprocessor
from src.uncertainty import select_threshold


def grouped_three_way_split(frame):
    X, y = frame[FEATURES], frame[TARGET]
    groups = pd.util.hash_pandas_object(X, index=False).astype(str)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_val_idx, test_idx = next(outer.split(X, y, groups))
    tv = frame.iloc[train_val_idx].reset_index(drop=True)
    tv_groups = pd.util.hash_pandas_object(tv[FEATURES], index=False).astype(str)
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    train_idx, val_idx = next(inner.split(tv[FEATURES], tv[TARGET], tv_groups))
    return (
        tv.iloc[train_idx].reset_index(drop=True),
        tv.iloc[val_idx].reset_index(drop=True),
        frame.iloc[test_idx].reset_index(drop=True),
    )


def save_eda(frame):
    out = ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    order = CLASS_ORDER
    ax = sns.countplot(data=frame, x=TARGET, order=order)
    ax.set(title="Maternal Health Risk Class Distribution", xlabel="Risk level", ylabel="Records")
    plt.tight_layout(); plt.savefig(out / "class_distribution.png", dpi=160); plt.close()
    frame[FEATURES].hist(figsize=(12, 8), bins=18, color="#2a9d8f")
    plt.suptitle("Distributions of Original Measurements")
    plt.tight_layout(); plt.savefig(out / "feature_histograms.png", dpi=160); plt.close()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for feature, ax in zip(FEATURES, axes.flat):
        sns.boxplot(data=frame, x=TARGET, y=feature, order=order, ax=ax)
        ax.set_title(f"{feature} by risk level")
    plt.tight_layout(); plt.savefig(out / "feature_boxplots_by_risk.png", dpi=160); plt.close()
    plt.figure(figsize=(8, 6))
    sns.heatmap(frame[FEATURES].corr(), annot=True, cmap="vlag", center=0, fmt=".2f")
    plt.title("Measurement Correlation Heatmap")
    plt.tight_layout(); plt.savefig(out / "correlation_heatmap.png", dpi=160); plt.close()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for feature, ax in zip(FEATURES, axes.flat):
        sns.violinplot(data=frame, x=TARGET, y=feature, order=order, inner="quart", ax=ax)
        ax.set_title(f"{feature} and risk category")
    plt.tight_layout(); plt.savefig(out / "measurement_risk_associations.png", dpi=160); plt.close()
    sns.scatterplot(data=frame, x="SystolicBP", y="DiastolicBP", hue=TARGET, hue_order=order)
    plt.title("Blood Pressure Measurements by Risk Category")
    plt.tight_layout(); plt.savefig(out / "blood_pressure_scatter.png", dpi=160); plt.close()


def evaluate_missing_by_feature(model, test):
    rows = []
    X, y = test[FEATURES], test[TARGET]
    complete_pred = model.predict(X)
    rows.append({"missing_feature": "none", **classification_metrics(y, complete_pred, model.predict_proba(X))})
    for feature in FEATURES:
        masked = X.copy()
        masked[feature] = np.nan
        pred = model.predict(masked)
        rows.append({"missing_feature": feature, **classification_metrics(y, pred, model.predict_proba(masked))})
    return pd.DataFrame(rows)


def main():
    np.random.seed(SEED)
    for path in [CLEAN_DATA.parent, MODEL_PATH.parent, ROOT / "reports" / "tables",
                 ROOT / "reports" / "figures"]:
        path.mkdir(parents=True, exist_ok=True)
    raw = validate_dataset(RAW_DATA)
    data = clean_dataset(raw)
    data.to_csv(CLEAN_DATA, index=False)
    save_eda(data)
    train, validation, test = grouped_three_way_split(data)
    split_rows = []
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        for label, count in part[TARGET].value_counts().items():
            split_rows.append({"split": name, "class": label, "count": int(count)})
    pd.DataFrame(split_rows).to_csv(ROOT / "reports" / "tables" / "split_distribution.csv", index=False)

    # Group-aware three-fold CV provides leakage-resistant model comparison.
    X_train, y_train = train[FEATURES], train[TARGET]
    groups = pd.util.hash_pandas_object(X_train, index=False).astype(str)
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    scoring = {"macro_f1": "f1_macro", "accuracy": "accuracy", "macro_recall": "recall_macro"}
    candidates = model_candidates(SEED, engineered=True)
    candidates["SoftVoting"] = voting_candidate(SEED)
    result_rows = []
    fitted = {}
    for name, model in candidates.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, groups=groups, scoring=scoring, n_jobs=1)
        model.fit(X_train, y_train)
        pred = model.predict(validation[FEATURES])
        prob = model.predict_proba(validation[FEATURES])
        metrics = classification_metrics(validation[TARGET], pred, prob)
        result_rows.append({
            "model": name,
            "cv_macro_f1_mean": scores["test_macro_f1"].mean(),
            "cv_macro_f1_std": scores["test_macro_f1"].std(),
            "cv_accuracy_mean": scores["test_accuracy"].mean(),
            **{f"validation_{k}": v for k, v in metrics.items()},
        })
        fitted[name] = model
    results = pd.DataFrame(result_rows).sort_values(
        ["validation_macro_f1", "validation_high_risk_recall"], ascending=False
    )
    results.to_csv(ROOT / "reports" / "model_results.csv", index=False)
    best_name = str(results.iloc[0]["model"])
    selected = fitted[best_name]

    # Controlled masking uses copies only; the raw and cleaned complete datasets stay complete.
    masked_copy = simulate_one_missing(train[FEATURES], fraction=.40, random_state=SEED)
    augmented_X = pd.concat([train[FEATURES], masked_copy], ignore_index=True)
    augmented_y = pd.concat([train[TARGET], train[TARGET]], ignore_index=True)
    robust = candidates[best_name]
    robust.fit(augmented_X, augmented_y)
    robust_pred = robust.predict(validation[FEATURES])
    robust_score = f1_score(validation[TARGET], robust_pred, average="macro")
    if robust_score + .01 >= float(results.iloc[0]["validation_macro_f1"]):
        selected = robust

    # Compare median, KNN, and model-native missing handling on validation masks.
    missing_val = simulate_one_missing(validation[FEATURES], fraction=1.0, random_state=SEED + 2)
    missing_rows = []
    for name, estimator in [
        ("median", Pipeline([("preprocess", build_preprocessor(True)),
                             ("model", RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=SEED, n_jobs=-1))])),
        ("knn", Pipeline([("imputer", KNNImputer(n_neighbors=5)),
                          ("model", RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=SEED, n_jobs=-1))])),
        ("native_hist_gradient", HistGradientBoostingClassifier(random_state=SEED)),
    ]:
        fit_X = augmented_X if name == "median" else train[FEATURES]
        fit_y = augmented_y if name == "median" else train[TARGET]
        estimator.fit(fit_X, fit_y)
        pred = estimator.predict(missing_val)
        missing_rows.append({"approach": name, **classification_metrics(validation[TARGET], pred)})
    pd.DataFrame(missing_rows).to_csv(ROOT / "reports" / "tables" / "imputation_comparison.csv", index=False)

    # Calibrate the frozen selected model on the held-out validation partition.
    uncal_prob = selected.predict_proba(validation[FEATURES])
    uncal_loss = classification_metrics(validation[TARGET], selected.predict(validation[FEATURES]), uncal_prob)["log_loss"]
    calibrated = CalibratedClassifierCV(FrozenEstimator(selected), method="sigmoid")
    calibrated.fit(validation[FEATURES], validation[TARGET])
    cal_prob = calibrated.predict_proba(validation[FEATURES])
    cal_loss = classification_metrics(validation[TARGET], calibrated.predict(validation[FEATURES]), cal_prob)["log_loss"]
    final_model = calibrated if cal_loss <= uncal_loss else selected
    val_prob = final_model.predict_proba(validation[FEATURES])
    val_pred = final_model.predict(validation[FEATURES])
    threshold, threshold_rows = select_threshold(val_prob, validation[TARGET], val_pred)
    pd.DataFrame(threshold_rows).to_csv(ROOT / "reports" / "tables" / "uncertainty_thresholds.csv", index=False)

    test_pred = final_model.predict(test[FEATURES])
    test_prob = final_model.predict_proba(test[FEATURES])
    test_metrics = classification_metrics(test[TARGET], test_pred, test_prob)
    report = classification_report(test[TARGET], test_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(test[TARGET], test_pred, labels=CLASS_ORDER)
    pd.DataFrame(matrix, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        ROOT / "reports" / "tables" / "test_confusion_matrix.csv"
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
    plt.title(f"Final Test Confusion Matrix — {best_name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
    plt.savefig(ROOT / "reports" / "figures" / "test_confusion_matrix.png", dpi=160); plt.close()
    missing_results = evaluate_missing_by_feature(final_model, test)
    missing_results.to_csv(ROOT / "reports" / "tables" / "missing_measurement_results.csv", index=False)

    cluster_metrics, projection, best_k = clustering_analysis(data, FEATURES, SEED)
    cluster_metrics.to_csv(ROOT / "reports" / "tables" / "clustering_metrics.csv", index=False)
    projection.to_csv(ROOT / "reports" / "tables" / "cluster_projection.csv", index=False)
    plt.plot(cluster_metrics["clusters"], cluster_metrics["inertia"], marker="o")
    plt.title("K-Means Elbow Analysis"); plt.xlabel("Clusters"); plt.ylabel("Inertia")
    plt.tight_layout(); plt.savefig(ROOT / "reports" / "figures" / "kmeans_elbow.png", dpi=160); plt.close()
    sns.scatterplot(data=projection, x="PC1", y="PC2", hue="KMeansCluster", palette="tab10")
    plt.title(f"PCA Projection of Exploratory K-Means Clusters (k={best_k})")
    plt.tight_layout(); plt.savefig(ROOT / "reports" / "figures" / "cluster_pca.png", dpi=160); plt.close()

    ranges = {feature: [float(data[feature].min()), float(data[feature].max())] for feature in FEATURES}
    medians = {feature: float(train[feature].median()) for feature in FEATURES}
    class_idx = {label: i for i, label in enumerate(CLASS_ORDER)}
    errors = {
        "actual_high_predicted_low": int(matrix[class_idx["high risk"], class_idx["low risk"]]),
        "actual_high_predicted_mid": int(matrix[class_idx["high risk"], class_idx["mid risk"]]),
        "actual_mid_predicted_low": int(matrix[class_idx["mid risk"], class_idx["low risk"]]),
    }
    metadata = {
        "model_name": best_name,
        "model_version": "1.0.0",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "package_versions": {"pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "features": FEATURES,
        "class_order": [str(c) for c in final_model.classes_],
        "label_mapping": {"low risk": "Low Risk", "mid risk": "Mid Risk", "high risk": "High Risk"},
        "feature_ranges": ranges,
        "feature_medians": medians,
        "uncertainty_threshold": threshold,
        "calibrated": final_model is calibrated,
        "validation_log_loss_uncalibrated": uncal_loss,
        "validation_log_loss_calibrated": cal_loss,
        "test_metrics": test_metrics,
        "classification_report": report,
        "error_analysis": errors,
        "dataset_source": UCI_URL,
        "dataset_sha256": hashlib.sha256(RAW_DATA.read_bytes()).hexdigest(),
        "random_seed": SEED,
        "duplicate_count_reported": int(raw.duplicated().sum()),
        "split_strategy": "StratifiedGroupKFold using signatures of all six input measurements",
        "explanation_method": "local median-replacement probability sensitivity",
    }
    joblib.dump(final_model, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary = [
        "# Final Results", "",
        f"- Selected model: **{best_name}**",
        f"- Test Macro F1: **{test_metrics['macro_f1']:.3f}**",
        f"- Test High Risk recall: **{test_metrics['high_risk_recall']:.3f}**",
        f"- Test accuracy: **{test_metrics['accuracy']:.3f}**",
        f"- Test log loss: **{test_metrics['log_loss']:.3f}**",
        f"- Uncertainty threshold: **{threshold:.2f}**",
        f"- Duplicate rows reported before splitting: **{int(raw.duplicated().sum())}**",
        f"- High Risk → Low Risk: **{errors['actual_high_predicted_low']}**",
        f"- High Risk → Mid Risk: **{errors['actual_high_predicted_mid']}**",
        f"- Mid Risk → Low Risk: **{errors['actual_mid_predicted_low']}**", "",
        "Metrics are from the untouched group-separated test set. Cluster labels are exploratory profiles, not diagnoses.",
    ]
    (ROOT / "reports" / "final_results.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps({"best_model": best_name, "test_metrics": test_metrics,
                      "threshold": threshold, "errors": errors}, indent=2))


if __name__ == "__main__":
    main()
