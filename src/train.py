from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from src.clustering import clustering_analysis
from src.config import (
    CLASS_ORDER,
    CLEAN_DATA,
    DATASET_DOI,
    DATASET_LICENSE,
    DATASET_PAGE,
    FEATURES,
    FIGURES,
    LABEL_MAPPING,
    METADATA_PATH,
    MODEL_PATH,
    RAW_DATA,
    REPORTS,
    SEED,
    TABLES,
    TARGET,
    TRAINING_GROUP_COLUMN,
    UCI_URL,
)
from src.data_loader import clean_dataset, download_dataset, feature_signature, validate_dataset
from src.evaluation import (
    baseline_predictions,
    class_report_table,
    classification_metrics,
    disaggregated_by_age,
    equal_opportunity_summary,
)
from src.missing_measurements import simulate_one_missing
from src.modelling import make_candidate, model_candidates, voting_candidate
from src.uncertainty import select_threshold


RISK_PALETTE = {"low risk": "#10b981", "mid risk": "#f59e0b", "high risk": "#dc2626"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_version(distribution: str) -> str | None:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None


def serializable_parameters(model) -> dict:
    result = {}
    for key, value in model.get_params(deep=True).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            result[key] = value
        elif isinstance(value, (tuple, list)) and all(
            item is None or isinstance(item, (str, int, float, bool)) for item in value
        ):
            result[key] = list(value)
    return result


def grouped_three_way_split(frame: pd.DataFrame):
    """Create train, validation and test partitions without signature leakage."""

    X, y = frame[FEATURES], frame[TARGET]
    groups = feature_signature(frame)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_val_idx, test_idx = next(outer.split(X, y, groups))
    train_val = frame.iloc[train_val_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)

    inner_groups = feature_signature(train_val)
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    train_idx, val_idx = next(inner.split(train_val[FEATURES], train_val[TARGET], inner_groups))
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    return train, val, test


def verify_no_signature_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    signatures = {
        "train": set(feature_signature(train)),
        "validation": set(feature_signature(val)),
        "test": set(feature_signature(test)),
    }
    overlaps = {
        "train_validation": len(signatures["train"] & signatures["validation"]),
        "train_test": len(signatures["train"] & signatures["test"]),
        "validation_test": len(signatures["validation"] & signatures["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Measurement-signature leakage detected: {overlaps}")
    return overlaps


def save_eda(frame: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    counts = frame[TARGET].value_counts().reindex(CLASS_ORDER)
    percentages = counts / len(frame) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [LABEL_MAPPING[label] for label in CLASS_ORDER],
        counts.values,
        color=[RISK_PALETTE[label] for label in CLASS_ORDER],
    )
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8, f"{count}\n({pct:.1f}%)", ha="center", fontweight="bold")
    ax.set_title("Distribution of maternal health risk classes", fontweight="bold")
    ax.set_ylabel("Records")
    ax.set_ylim(0, counts.max() * 1.22)
    fig.tight_layout()
    fig.savefig(FIGURES / "class_distribution_annotated.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(frame.isna(), cbar=False, yticklabels=False, cmap=["#ecfdf5", "#dc2626"], ax=ax)
    ax.set_title("Missing-data map: no source values are missing", fontweight="bold")
    ax.set_xlabel("Dataset fields")
    fig.tight_layout()
    fig.savefig(FIGURES / "missing_data_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    axes = frame[FEATURES].hist(figsize=(13, 8), bins=18, color="#0f766e", edgecolor="white")
    for ax in np.asarray(axes).flat:
        ax.set_ylabel("Records")
    plt.suptitle("Distributions of the six source measurements", y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_histograms.png", dpi=180, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for feature, ax in zip(FEATURES, axes.flat):
        sns.boxplot(
            data=frame,
            x=TARGET,
            y=feature,
            order=CLASS_ORDER,
            hue=TARGET,
            palette=RISK_PALETTE,
            legend=False,
            ax=ax,
        )
        ax.set_title(f"{feature} by risk level")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Measurement distributions by target class", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_boxplots_by_risk.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    corr = frame[FEATURES].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="vlag", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlation among source measurements", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=frame,
        x="SystolicBP",
        y="DiastolicBP",
        hue=TARGET,
        hue_order=CLASS_ORDER,
        palette=RISK_PALETTE,
        alpha=0.7,
        ax=ax,
    )
    ax.set_title("Blood-pressure measurements by risk class", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "blood_pressure_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def contradictory_label_analysis(frame: pd.DataFrame) -> dict:
    grouped = (
        frame.groupby(FEATURES, dropna=False)[TARGET]
        .agg(label_count="nunique", record_count="size", labels=lambda s: " | ".join(sorted(set(s))))
        .reset_index()
    )
    conflicts = grouped[grouped["label_count"] > 1].sort_values(["record_count", "label_count"], ascending=False)
    conflicts.to_csv(TABLES / "contradictory_measurement_signatures.csv", index=False)
    rows_in_conflicts = frame.merge(conflicts[FEATURES], on=FEATURES, how="inner").shape[0]
    return {
        "unique_signatures": int(len(grouped)),
        "conflicting_signatures": int(len(conflicts)),
        "rows_in_conflicting_signatures": int(rows_in_conflicts),
    }


def prepare_augmented_train(frame: pd.DataFrame, random_state: int = SEED) -> pd.DataFrame:
    groups = feature_signature(frame).to_numpy()
    original = frame.copy()
    original[TRAINING_GROUP_COLUMN] = groups
    masked = simulate_one_missing(frame, fraction=0.35, random_state=random_state)
    masked[TRAINING_GROUP_COLUMN] = groups
    return pd.concat([original, masked], ignore_index=True)


def evaluate_model(name: str, model, train_aug: pd.DataFrame, val: pd.DataFrame):
    training_columns = FEATURES + ([TRAINING_GROUP_COLUMN] if TRAINING_GROUP_COLUMN in train_aug else [])
    model.fit(train_aug[training_columns], train_aug[TARGET])
    prediction = model.predict(val[FEATURES])
    probabilities = model.predict_proba(val[FEATURES]) if hasattr(model, "predict_proba") else None
    metrics = classification_metrics(val[TARGET], prediction, probabilities)
    return {"model": name, **metrics}, model


def save_model_comparison(model_results: pd.DataFrame) -> None:
    plot = model_results.dropna(subset=["weighted_f1"]).sort_values("weighted_f1")
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(plot["model"], plot["weighted_f1"], color="#0f766e")
    ax.set_xlim(0, max(0.65, plot["weighted_f1"].max() + 0.08))
    ax.set_xlabel("Validation Weighted F1")
    ax.set_title("Model comparison on the group-separated validation set", fontweight="bold")
    for bar, value in zip(bars, plot["weighted_f1"]):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison_weighted_f1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def feature_ablation(model_name: str, train_aug: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for engineered in [False, True]:
        model = make_candidate(model_name, seed=SEED, engineered=engineered)
        row, _ = evaluate_model(
            "Original + engineered" if engineered else "Original six features",
            model,
            train_aug,
            val,
        )
        rows.append({"feature_set": row.pop("model"), **row})
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "feature_ablation.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(result["feature_set"], result["weighted_f1"], color=["#94a3b8", "#0f766e"])
    ax.set_ylim(0, max(0.65, result["weighted_f1"].max() + 0.08))
    ax.set_ylabel("Validation Weighted F1")
    ax.set_title(f"Effect of engineered features using {model_name}", fontweight="bold")
    for bar, value in zip(bars, result["weighted_f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontweight="bold")
    ax.tick_params(axis="x", rotation=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return result


def save_neural_network_curves(fitted: dict) -> pd.DataFrame:
    history_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    plotted = 0
    for name in ["ShallowMLP", "DeepMLP"]:
        pipeline = fitted.get(name)
        if pipeline is None:
            continue
        estimator = pipeline.named_steps["model"]
        losses = list(getattr(estimator, "loss_curve_", []))
        validation = list(getattr(estimator, "validation_scores_", []))
        for epoch, loss in enumerate(losses, start=1):
            history_rows.append({
                "model": name,
                "epoch": epoch,
                "training_loss": float(loss),
                "internal_validation_accuracy": float(validation[epoch - 1]) if epoch <= len(validation) else np.nan,
                "selected_epoch": int(getattr(estimator, "best_epoch_", len(losses))),
                "internal_validation_records": int(getattr(estimator, "internal_validation_records_", 0)),
                "signature_overlap": int(getattr(estimator, "internal_validation_signature_overlap_", 0)),
            })
        axes[0].plot(range(1, len(losses) + 1), losses, label=name)
        if validation:
            axes[1].plot(range(1, len(validation) + 1), validation, label=name)
        plotted += 1
    axes[0].set_title("Neural-network training loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Internal validation accuracy")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.grid(alpha=0.25)
        if plotted:
            ax.legend()
    fig.suptitle("Shallow and deep feed-forward neural-network training", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "neural_network_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    history = pd.DataFrame(history_rows)
    history.to_csv(TABLES / "neural_network_history.csv", index=False)
    return history


def save_baseline_figure(baseline_table: pd.DataFrame) -> None:
    chart = baseline_table[baseline_table["metric"].isin(["accuracy", "weighted_f1", "macro_f1", "high_risk_recall"])].copy()
    labels = [label.replace("_", " ").title() for label in chart["metric"]]
    x = np.arange(len(chart))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, chart["majority_baseline"], width, label="Majority baseline", color="#cbd5e1")
    b2 = ax.bar(x + width / 2, chart["selected_model"], width, label="Selected model", color="#047857")
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Selected model compared with the no-learning baseline", fontweight="bold")
    ax.legend()
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"{bar.get_height():.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "baseline_vs_model.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_age_fairness_figure(disaggregated: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(disaggregated["AgeGroup"], disaggregated["high_risk_recall"], color="#0f766e")
    ax.axhline(disaggregated["high_risk_recall"].mean(), color="#f97316", linestyle="--", label="Subgroup mean")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("High Risk recall")
    ax.set_xlabel("Age group")
    ax.set_title("Equal Opportunity check: High Risk recall by age group", fontweight="bold")
    for bar, value, n in zip(bars, disaggregated["high_risk_recall"], disaggregated["HighRiskN"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.3f}\n(n={int(n)})", ha="center", fontsize=9, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "age_group_high_risk_recall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def calibration_check(selected_name: str, train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """Calibrate on a group-separated holdout and evaluate on the untouched validation split."""

    groups = feature_signature(train)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 9)
    base_idx, calibration_idx = next(splitter.split(train[FEATURES], train[TARGET], groups))
    base_train = train.iloc[base_idx].reset_index(drop=True)
    calibration = train.iloc[calibration_idx].reset_index(drop=True)
    base_aug = prepare_augmented_train(base_train, random_state=SEED + 9)

    base_model = make_candidate(selected_name, seed=SEED, engineered=True)
    base_model.fit(base_aug[FEATURES + [TRAINING_GROUP_COLUMN]], base_aug[TARGET])
    uncalibrated_pred = base_model.predict(val[FEATURES])
    uncalibrated_prob = base_model.predict_proba(val[FEATURES])
    uncalibrated_loss = classification_metrics(val[TARGET], uncalibrated_pred, uncalibrated_prob)["log_loss"]

    try:
        calibrator = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
        calibrator.fit(calibration[FEATURES], calibration[TARGET])
        calibrated_pred = calibrator.predict(val[FEATURES])
        calibrated_prob = calibrator.predict_proba(val[FEATURES])
        calibrated_loss = classification_metrics(val[TARGET], calibrated_pred, calibrated_prob)["log_loss"]
        retained = bool(calibrated_loss < uncalibrated_loss)
    except Exception as exc:
        calibrated_loss = None
        retained = False
        return {
            "uncalibrated_log_loss": uncalibrated_loss,
            "calibrated_log_loss": None,
            "retained": False,
            "note": f"Calibration check could not be completed: {exc}",
        }

    return {
        "uncalibrated_log_loss": float(uncalibrated_loss),
        "calibrated_log_loss": float(calibrated_loss),
        "retained": retained,
        "note": "Calibration was trained on a group-separated calibration subset and evaluated on the external validation split.",
    }


def save_cluster_figures(metrics: pd.DataFrame, projection: pd.DataFrame, profiles: pd.DataFrame) -> None:
    kmeans_metrics = metrics.loc[metrics["method"] == "KMeans"].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(kmeans_metrics["clusters"], kmeans_metrics["silhouette"], marker="o", linewidth=2, color="#0f766e")
    ax.set_xticks(kmeans_metrics["clusters"])
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette score")
    ax.set_title("K-Means cluster selection", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "kmeans_silhouette.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=projection, x="PC1", y="PC2", hue="KMeansCluster", palette="viridis", alpha=0.75, ax=ax)
    ax.set_title("K-Means profiles projected to two principal components", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "cluster_pca.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    numeric_profiles = profiles[FEATURES]
    normalized = (numeric_profiles - numeric_profiles.mean()) / numeric_profiles.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.heatmap(normalized, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Cluster profile medians relative to the other clusters", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "cluster_profile_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for path in [TABLES, FIGURES, REPORTS, MODEL_PATH.parent, CLEAN_DATA.parent]:
        path.mkdir(parents=True, exist_ok=True)

    download_dataset(RAW_DATA)
    raw = validate_dataset(RAW_DATA)
    frame = clean_dataset(raw)
    frame.to_csv(CLEAN_DATA, index=False)
    save_eda(frame)
    conflict_info = contradictory_label_analysis(frame)

    train, val, test = grouped_three_way_split(frame)
    overlaps = verify_no_signature_leakage(train, val, test)
    train.to_csv(TABLES / "train_split.csv", index=False)
    val.to_csv(TABLES / "validation_split.csv", index=False)
    test.to_csv(TABLES / "test_split.csv", index=False)

    split_rows = []
    for split_name, split_frame in [("train", train), ("validation", val), ("test", test)]:
        counts = split_frame[TARGET].value_counts().reindex(CLASS_ORDER, fill_value=0)
        split_rows.append({
            "split": split_name,
            "records": len(split_frame),
            **{f"{label.replace(' ', '_')}_count": int(counts[label]) for label in CLASS_ORDER},
        })
    pd.DataFrame(split_rows).to_csv(TABLES / "split_distribution.csv", index=False)

    train_aug = prepare_augmented_train(train)
    rows, fitted = [], {}
    for name, model in model_candidates(seed=SEED, engineered=True).items():
        try:
            row, fitted_model = evaluate_model(name, model, train_aug, val)
            rows.append(row)
            fitted[name] = fitted_model
        except Exception as exc:
            rows.append({"model": name, "error": str(exc)})

    try:
        row, fitted_model = evaluate_model("SoftVoting", voting_candidate(SEED), train_aug, val)
        rows.append(row)
        fitted["SoftVoting"] = fitted_model
    except Exception as exc:
        rows.append({"model": "SoftVoting", "error": str(exc)})

    model_results = pd.DataFrame(rows).sort_values(
        ["weighted_f1", "macro_f1", "high_risk_recall"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    model_results.to_csv(TABLES / "model_comparison_validation.csv", index=False)
    save_model_comparison(model_results)
    save_neural_network_curves(fitted)
    valid_results = model_results.dropna(subset=["weighted_f1"]).copy()
    selected_name = str(valid_results.iloc[0]["model"])
    feature_ablation_result = feature_ablation(selected_name, train_aug, val)

    # External validation probabilities are used to select the uncertainty threshold.
    validation_model = fitted[selected_name]
    val_pred = validation_model.predict(val[FEATURES])
    val_proba = validation_model.predict_proba(val[FEATURES])
    uncertainty_threshold, threshold_rows = select_threshold(val_proba, val[TARGET], val_pred)
    pd.DataFrame(threshold_rows).to_csv(TABLES / "uncertainty_thresholds.csv", index=False)
    calibration = calibration_check(selected_name, train, val)

    combined = pd.concat([train, val], ignore_index=True)
    final_calibrated = False
    final_calibration_method = "Not retained because external validation log loss did not improve."
    if calibration["retained"]:
        try:
            combined_groups = feature_signature(combined)
            calibration_cv = list(
                StratifiedGroupKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=SEED + 10,
                ).split(combined[FEATURES], combined[TARGET], combined_groups)
            )
            final_model = CalibratedClassifierCV(
                make_candidate(selected_name, seed=SEED, engineered=True),
                method="sigmoid",
                cv=calibration_cv,
            )
            final_model.fit(combined[FEATURES], combined[TARGET])
            final_calibrated = True
            final_calibration_method = (
                "Sigmoid calibration retained after external validation improvement; "
                "the deployed calibrator uses five measurement-signature-separated folds "
                "over the combined train and validation partitions."
            )
        except Exception as exc:
            combined_aug = prepare_augmented_train(combined, random_state=SEED + 2)
            final_model = make_candidate(selected_name, seed=SEED, engineered=True)
            final_model.fit(combined_aug[FEATURES + [TRAINING_GROUP_COLUMN]], combined_aug[TARGET])
            final_calibration_method = f"Calibration retention failed during final refit; uncalibrated fallback used: {exc}"
    else:
        combined_aug = prepare_augmented_train(combined, random_state=SEED + 2)
        final_model = make_candidate(selected_name, seed=SEED, engineered=True)
        final_model.fit(combined_aug[FEATURES + [TRAINING_GROUP_COLUMN]], combined_aug[TARGET])
    test_pred = final_model.predict(test[FEATURES])
    test_proba = final_model.predict_proba(test[FEATURES])
    test_metrics = classification_metrics(test[TARGET], test_pred, test_proba)

    base_pred, majority_class = baseline_predictions(combined[TARGET], len(test))
    baseline_metrics = classification_metrics(test[TARGET], base_pred)
    baseline_table = pd.DataFrame([
        {
            "metric": key,
            "majority_baseline": baseline_metrics.get(key),
            "selected_model": test_metrics.get(key),
            "absolute_improvement": test_metrics.get(key, 0) - baseline_metrics.get(key, 0),
        }
        for key in ["accuracy", "weighted_f1", "macro_f1", "high_risk_recall"]
    ])
    baseline_table.to_csv(TABLES / "baseline_vs_model.csv", index=False)
    save_baseline_figure(baseline_table)

    class_table = class_report_table(test[TARGET], test_pred)
    class_table.to_csv(TABLES / "class_report_test.csv", index=False)

    disaggregated = disaggregated_by_age(test, test_pred)
    disaggregated.to_csv(TABLES / "disaggregated_age_evaluation.csv", index=False)
    fairness_summary = equal_opportunity_summary(disaggregated)
    pd.DataFrame([fairness_summary]).to_csv(TABLES / "equal_opportunity_summary.csv", index=False)
    save_age_fairness_figure(disaggregated)

    matrix_labels = CLASS_ORDER
    conf = confusion_matrix(test[TARGET], test_pred, labels=matrix_labels)
    confusion_df = pd.DataFrame(
        conf,
        index=[f"actual_{label.replace(' ', '_')}" for label in matrix_labels],
        columns=[f"pred_{label.replace(' ', '_')}" for label in matrix_labels],
    )
    confusion_df.to_csv(TABLES / "test_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        conf,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=[LABEL_MAPPING[label] for label in matrix_labels],
        yticklabels=[LABEL_MAPPING[label] for label in matrix_labels],
        ax=ax,
    )
    ax.set_title("Untouched test-set confusion matrix", fontweight="bold")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    fig.tight_layout()
    fig.savefig(FIGURES / "test_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Structured error analysis.
    error_df = test.copy()
    error_df["AgeGroup"] = pd.cut(
        error_df["Age"],
        bins=[-np.inf, 19, 34, 49, np.inf],
        labels=["<=19", "20-34", "35-49", ">=50"],
        include_lowest=True,
    ).astype(str)
    error_df["predicted"] = test_pred
    error_df["correct"] = error_df[TARGET] == error_df["predicted"]
    error_df["confidence"] = np.max(test_proba, axis=1)
    for index, label in enumerate(final_model.classes_):
        error_df[f"prob_{str(label).replace(' ', '_')}"] = test_proba[:, index]
    signature_conflicts = pd.read_csv(TABLES / "contradictory_measurement_signatures.csv")
    conflict_keys = signature_conflicts[FEATURES].assign(conflicting_signature=True)
    error_df = error_df.merge(conflict_keys, on=FEATURES, how="left")
    error_df["conflicting_signature"] = error_df["conflicting_signature"].eq(True)
    error_df.sort_values(["correct", "confidence"], ascending=[True, False]).to_csv(
        TABLES / "error_analysis_records.csv", index=False
    )
    confident_errors = (
        error_df.loc[~error_df["correct"]]
        .sort_values("confidence", ascending=False)
        .drop_duplicates(subset=FEATURES + [TARGET, "predicted"])
        .head(15)
    )
    confident_errors.to_csv(TABLES / "top_confident_errors.csv", index=False)
    error_summary = (
        error_df.loc[~error_df["correct"]]
        .groupby([TARGET, "predicted"], observed=True)
        .size()
        .rename("records")
        .reset_index()
        .sort_values("records", ascending=False)
    )
    error_summary.to_csv(TABLES / "error_type_summary.csv", index=False)
    age_error = (
        error_df.groupby("AgeGroup", observed=True)["correct"]
        .agg(records="count", accuracy="mean")
        .reset_index()
    )
    age_error["error_rate"] = 1 - age_error["accuracy"]
    age_error.to_csv(TABLES / "error_rate_by_age.csv", index=False)
    conflict_rows = []
    for is_conflicting, part in error_df.groupby("conflicting_signature", observed=True):
        metrics = classification_metrics(part[TARGET], part["predicted"])
        conflict_rows.append({
            "conflicting_signature": bool(is_conflicting),
            "records": len(part),
            "error_rate": float((~part["correct"]).mean()),
            **metrics,
        })
    conflict_performance = pd.DataFrame(conflict_rows)
    conflict_performance.to_csv(TABLES / "conflicting_signature_performance.csv", index=False)

    cluster_metrics, cluster_projection, cluster_profiles, cluster_risk, best_k, cluster_agreement = clustering_analysis(
        frame,
        FEATURES,
        TARGET,
        SEED,
    )
    cluster_metrics.to_csv(TABLES / "clustering_metrics.csv", index=False)
    cluster_projection.to_csv(TABLES / "cluster_projection.csv", index=False)
    cluster_profiles.to_csv(TABLES / "cluster_profiles.csv")
    cluster_risk.to_csv(TABLES / "cluster_risk_distribution.csv")
    save_cluster_figures(cluster_metrics, cluster_projection, cluster_profiles)

    feature_ranges = {feature: [float(frame[feature].min()), float(frame[feature].max())] for feature in FEATURES}
    feature_medians = {feature: float(combined[feature].median()) for feature in FEATURES}
    mid_row = class_table.loc[class_table["class"] == "mid risk"].iloc[0]
    selected_validation_metrics = {
        metric: float(valid_results.iloc[0][metric])
        for metric in ["accuracy", "weighted_f1", "macro_f1", "high_risk_recall", "log_loss"]
    }

    metadata = {
        "model_name": selected_name,
        "model_version": "3.0.0",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "package_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": installed_version("xgboost"),
            "streamlit": installed_version("streamlit"),
            "joblib": installed_version("joblib"),
        },
        "random_seed": SEED,
        "features": FEATURES,
        "engineered_features": ["PulsePressure", "MeanArterialPressure", "AgeBand"],
        "class_order": [str(c) for c in final_model.classes_],
        "label_mapping": LABEL_MAPPING,
        "feature_ranges": feature_ranges,
        "feature_range_basis": "Full source-dataset bounds used only for interface input validation; not clinical ranges.",
        "feature_medians": feature_medians,
        "feature_median_basis": "Combined train and validation partitions; the untouched test partition is excluded.",
        "uncertainty_threshold": float(uncertainty_threshold),
        "calibrated": final_calibrated,
        "calibration_check": calibration,
        "final_calibration_method": final_calibration_method,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "majority_class_baseline": majority_class,
        "classification_report": class_table.to_dict(orient="records"),
        "mid_risk_recall": float(mid_row["recall"]),
        "dataset_source": UCI_URL,
        "dataset_page": DATASET_PAGE,
        "dataset_doi": DATASET_DOI,
        "dataset_license": DATASET_LICENSE,
        "dataset_population": "Hospitals, community clinics and maternal-health facilities in rural Bangladesh, as documented by UCI.",
        "dataset_sha256": sha256(RAW_DATA),
        "downloaded_csv_rows": int(len(frame)),
        "uci_metadata_rows_note": "UCI metadata reports 1,013 instances; the downloaded CSV used here contains 1,014 rows.",
        "duplicate_count_reported": int(frame.duplicated().sum()),
        "unique_measurement_signatures": conflict_info["unique_signatures"],
        "conflicting_signature_count": conflict_info["conflicting_signatures"],
        "rows_in_conflicting_signatures": conflict_info["rows_in_conflicting_signatures"],
        "split_strategy": "StratifiedGroupKFold using signatures of all six input measurements",
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "signature_overlap_checks": overlaps,
        "primary_metric": "weighted_f1",
        "model_ranking_order": ["weighted_f1", "macro_f1", "high_risk_recall"],
        "selected_validation_metrics": selected_validation_metrics,
        "selected_model_parameters": serializable_parameters(final_model),
        "secondary_metrics": ["macro_f1", "high_risk_recall", "accuracy", "log_loss"],
        "fairness_criterion": "Equal Opportunity for High Risk identification across age groups",
        "equal_opportunity_summary": fairness_summary,
        "best_kmeans_k": int(best_k),
        "cluster_method_agreement_adjusted_rand": cluster_agreement,
        "feature_ablation": feature_ablation_result.to_dict(orient="records"),
        "missing_measurement_policy": "Median imputation inside the fitted pipeline; at most one unavailable source measurement is accepted.",
        "local_explanation_method": "Replace each source measurement with its combined train-validation median and report the predicted-class score change.",
        "neural_network_scope": "Scikit-learn shallow/deep MLP comparison with group-aware early stopping; no Keras dropout or batch normalization.",
    }
    joblib.dump(final_model, MODEL_PATH)
    metadata["model_artifact_sha256"] = sha256(MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    final_lines = [
        "# Final Results",
        "",
        f"- Selected model: **{selected_name}**",
        "- Primary metric: **Weighted F1**",
        f"- Test Weighted F1: **{test_metrics['weighted_f1']:.3f}**",
        f"- Test Macro F1: **{test_metrics['macro_f1']:.3f}**",
        f"- Test High Risk recall: **{test_metrics['high_risk_recall']:.3f}**",
        f"- Test Mid Risk recall: **{mid_row['recall']:.3f}**",
        f"- Test accuracy: **{test_metrics['accuracy']:.3f}**",
        f"- Test log loss: **{test_metrics['log_loss']:.3f}**",
        f"- Majority-class baseline Weighted F1: **{baseline_metrics['weighted_f1']:.3f}**",
        f"- Uncertainty threshold: **{uncertainty_threshold:.2f}**",
        f"- Duplicate rows reported before splitting: **{int(frame.duplicated().sum())}**",
        f"- Conflicting measurement signatures: **{conflict_info['conflicting_signatures']}**",
        f"- Equal Opportunity gap across age groups: **{fairness_summary['gap']:.3f}**",
        "",
        "Metrics are from the untouched group-separated test set. The system is an academic screening-support demonstration only.",
    ]
    (REPORTS / "final_results.md").write_text("\n".join(final_lines), encoding="utf-8")

    print(json.dumps({"selected_model": selected_name, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
