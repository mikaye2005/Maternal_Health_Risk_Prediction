from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version as installed_version
import json
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import tempfile
import zipfile

import nbformat
import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
RAW_HASH = "a1f7025719f84715096e0d1f95ae2e56b57809b9b15449e1836c96a7d976ae9b"
PROPOSAL_HASH = "a44aa67e4a6ad85f417d2e38fa3fd38f4f98c84e358aeac0aa7ce668e5053be5"
ARCHIVE_ROOT = PurePosixPath("MamaCare_Final_Submission")
BLOCKED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "dist"}
BLOCKED_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp"}

REQUIRED = [
    "README.md", "LICENSE", "DATA_LICENSE.md", "CITATION.cff", "requirements.txt",
    "requirements-lock.txt",
    ".streamlit/config.toml", "run_app.ps1", "run_app.sh",
    "data/raw/Maternal Health Risk Data Set.csv",
    "data/processed/maternal_health_risk_cleaned.csv",
    "notebooks/MamaCare_End_to_End.ipynb",
    "reports/notebook/MamaCare_End_to_End.html",
    "reports/MamaCare_Capstone_Report.docx", "reports/MamaCare_Capstone_Report.pdf",
    "reports/responsible_ai_statement.md", "reports/responsible_ai_statement.pdf",
    "presentation/MamaCare_Capstone_Presentation.pptx",
    "presentation/MamaCare_Capstone_Presentation.pdf", "presentation/speaker_notes.md",
    "models/maternal_risk_pipeline.joblib", "models/model_metadata.json",
    "docs/MODEL_CARD.md", "docs/DATA_DICTIONARY.md", "docs/DATASET_SELECTION.md",
    "docs/SCOPE_DECISIONS.md", "docs/EXECUTION_GUIDE.md", "docs/TESTING_REPORT.md",
    "docs/CAPSTONE_COMPLIANCE_CHECKLIST.md", "docs/SUBMISSION_CHECKLIST.md",
    "docs/FINAL_SUBMISSION_MANIFEST.md", "docs/BACKUP_MANIFEST_PRE_UPDATE.md",
    "docs/references/Original_MamaCare_Project_Proposal.pdf",
]

REQUIRED_FIGURES = [
    "class_distribution_annotated.png", "missing_data_heatmap.png", "feature_histograms.png",
    "feature_boxplots_by_risk.png", "correlation_heatmap.png", "blood_pressure_scatter.png",
    "kmeans_silhouette.png", "cluster_pca.png", "cluster_profile_heatmap.png",
    "feature_ablation.png", "model_comparison_weighted_f1.png",
    "neural_network_training_curves.png", "baseline_vs_model.png",
    "test_confusion_matrix.png", "age_group_high_risk_recall.png",
]

REQUIRED_TABLES = [
    "model_comparison_validation.csv", "baseline_vs_model.csv", "class_report_test.csv",
    "feature_ablation.csv", "clustering_metrics.csv", "cluster_profiles.csv",
    "cluster_risk_distribution.csv", "contradictory_measurement_signatures.csv",
    "conflicting_signature_performance.csv", "disaggregated_age_evaluation.csv",
    "equal_opportunity_summary.csv", "error_analysis_records.csv", "top_confident_errors.csv",
    "error_type_summary.csv", "error_rate_by_age.csv", "uncertainty_thresholds.csv",
    "train_split.csv", "validation_split.csv", "test_split.csv",
]

REQUIRED_SCREENSHOTS = [
    "mamacare_frontend_landing.png", "mamacare_frontend_preview.png",
    "mamacare_frontend_uncertainty.png", "mamacare_frontend_missing_measurement.png",
    "mamacare_frontend_what_if.png", "mamacare_frontend_model_evidence.png",
    "mamacare_frontend_responsible_use.png", "mamacare_frontend_mobile.png",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def required_project_paths() -> list[str]:
    relative_paths = REQUIRED[:]
    relative_paths += [f"reports/figures/{name}" for name in REQUIRED_FIGURES]
    relative_paths += [f"reports/tables/{name}" for name in REQUIRED_TABLES]
    relative_paths += [f"assets/screenshots/{name}" for name in REQUIRED_SCREENSHOTS]
    return relative_paths


def require_files() -> None:
    relative_paths = required_project_paths()
    missing = [path for path in relative_paths if not (ROOT / path).is_file()]
    empty = [path for path in relative_paths if (ROOT / path).is_file() and (ROOT / path).stat().st_size == 0]
    if missing or empty:
        details = "\n".join([*(f"missing: {item}" for item in missing), *(f"empty: {item}" for item in empty)])
        raise RuntimeError(f"Submission file check failed:\n{details}")
    if (ROOT / "CODEX_PROMPT.md").exists():
        raise RuntimeError("Internal CODEX_PROMPT.md must not be included in the public project.")


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def verify_tracked_cleanliness() -> None:
    bad = []
    for text_path in tracked_files():
        path = Path(text_path)
        if BLOCKED_PARTS.intersection(path.parts) or path.suffix.lower() in BLOCKED_SUFFIXES:
            bad.append(text_path)
    if bad:
        raise RuntimeError(f"Blocked runtime files are tracked: {bad}")


def verify_text_controls() -> None:
    text_suffixes = {".py", ".md", ".txt", ".toml", ".json", ".cff", ".sh", ".ps1", ".csv"}
    failures = []
    for relative in tracked_files():
        path = ROOT / relative
        if path.suffix.lower() not in text_suffixes or not path.is_file():
            continue
        data = path.read_bytes()
        controls = sorted({byte for byte in data if byte < 32 and byte not in {9, 10, 13}})
        if controls:
            failures.append(f"{relative}: {controls}")
    if failures:
        raise RuntimeError("Embedded control characters found:\n" + "\n".join(failures))


def verify_data_and_metadata() -> None:
    raw_path = ROOT / "data" / "raw" / "Maternal Health Risk Data Set.csv"
    metadata_path = ROOT / "models" / "model_metadata.json"
    digest = sha256(raw_path.read_bytes())
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if digest != RAW_HASH or metadata.get("dataset_sha256") != RAW_HASH:
        raise RuntimeError("Raw dataset hash or metadata hash does not match the preserved source.")
    proposal_path = ROOT / "docs" / "references" / "Original_MamaCare_Project_Proposal.pdf"
    if sha256(proposal_path.read_bytes()) != PROPOSAL_HASH:
        raise RuntimeError("The preserved original proposal no longer matches the mentor copy.")
    model_digest = sha256((ROOT / "models" / "maternal_risk_pipeline.joblib").read_bytes())
    if metadata.get("model_artifact_sha256") != model_digest:
        raise RuntimeError("Model artifact checksum does not match metadata.")
    if metadata.get("python_version") != platform.python_version():
        raise RuntimeError("Runtime Python version does not match the training metadata.")
    for package_name in ["pandas", "numpy", "scikit-learn", "xgboost", "streamlit", "joblib"]:
        if metadata.get("package_versions", {}).get(package_name) != installed_version(package_name):
            raise RuntimeError(f"Runtime {package_name} version does not match training metadata.")


def verify_analysis_outputs() -> None:
    comparison = pd.read_csv(ROOT / "reports" / "tables" / "model_comparison_validation.csv")
    required_models = {
        "MajorityBaseline", "LogisticRegression", "DecisionTree", "RandomForest", "SVC",
        "GradientBoosting", "XGBoost", "ShallowMLP", "DeepMLP", "SoftVoting",
    }
    if set(comparison["model"]) != required_models:
        raise RuntimeError("Model comparison does not contain all ten required candidates.")
    if comparison[["weighted_f1", "macro_f1", "high_risk_recall"]].isna().any().any():
        raise RuntimeError("At least one required model failed to produce validation metrics.")
    ranked = comparison.sort_values(
        ["weighted_f1", "macro_f1", "high_risk_recall"], ascending=False
    )["model"].tolist()
    if comparison["model"].tolist() != ranked:
        raise RuntimeError("Model comparison is not saved in the required ranking order.")

    clustering = pd.read_csv(ROOT / "reports" / "tables" / "clustering_metrics.csv")
    if not {"method", "silhouette", "davies_bouldin"} <= set(clustering.columns):
        raise RuntimeError("Clustering metrics are incomplete.")
    if not {"KMeans", "Hierarchical"} <= set(clustering["method"]):
        raise RuntimeError("K-Means and hierarchical clustering evidence are both required.")

    history = pd.read_csv(ROOT / "reports" / "tables" / "neural_network_history.csv")
    if history["signature_overlap"].max() != 0:
        raise RuntimeError("Neural-network internal validation contains signature overlap.")


def verify_documents() -> None:
    statement = (ROOT / "reports" / "responsible_ai_statement.md").read_text(encoding="utf-8")
    word_count = len(re.findall(r"\b[\w'-]+\b", re.sub(r"^#+\s*", "", statement, flags=re.MULTILINE)))
    if not 500 <= word_count <= 700:
        raise RuntimeError(f"Responsible AI statement must contain 500-700 words; found {word_count}.")

    notebook = nbformat.read(ROOT / "notebooks" / "MamaCare_End_to_End.ipynb", as_version=4)
    errors = [
        output
        for cell in notebook.cells if cell.cell_type == "code"
        for output in cell.get("outputs", []) if output.get("output_type") == "error"
    ]
    if errors:
        raise RuntimeError("Executed notebook contains error outputs.")

    pptx = ROOT / "presentation" / "MamaCare_Capstone_Presentation.pptx"
    with zipfile.ZipFile(pptx) as archive:
        slide_count = len([
            name for name in archive.namelist()
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        ])
    if slide_count != 5:
        raise RuntimeError(f"Presentation must contain exactly five slides; found {slide_count}.")
    pdf_slide_count = len(PdfReader(ROOT / "presentation" / "MamaCare_Capstone_Presentation.pdf").pages)
    if pdf_slide_count != 5:
        raise RuntimeError(f"Presentation PDF must contain five pages; found {pdf_slide_count}.")
    if len(PdfReader(ROOT / "reports" / "responsible_ai_statement.pdf").pages) < 1:
        raise RuntimeError("Responsible AI PDF has no pages.")
    if len(PdfReader(ROOT / "reports" / "MamaCare_Capstone_Report.pdf").pages) < 5:
        raise RuntimeError("Capstone report PDF is unexpectedly short.")


def verify_archive(archive_path: Path) -> None:
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise RuntimeError(f"Archive CRC check failed for {corrupt}")
        names = set(archive.namelist())
        wrapper_required = {
            (ARCHIVE_ROOT / "README_FIRST.txt").as_posix(),
            (ARCHIVE_ROOT / "GITHUB_URL.txt").as_posix(),
            (ARCHIVE_ROOT / "SUBMISSION_CHECKLIST.md").as_posix(),
            (ARCHIVE_ROOT / "SUBMISSION_MANIFEST.txt").as_posix(),
            (ARCHIVE_ROOT / "SHA256SUMS.txt").as_posix(),
        }
        if not wrapper_required <= names:
            raise RuntimeError(f"Archive wrapper files are missing: {sorted(wrapper_required - names)}")
        project_paths = required_project_paths()
        for required in project_paths:
            source_name = (ARCHIVE_ROOT / "source_project" / PurePosixPath(required)).as_posix()
            if source_name not in names:
                raise RuntimeError(f"Archive is missing {source_name}")
            path = PurePosixPath(required)
            convenience_name = None
            if path.parts and path.parts[0] in {"reports", "presentation"}:
                convenience_name = (ARCHIVE_ROOT / path).as_posix()
            elif path.parts[:2] == ("assets", "screenshots"):
                convenience_name = (
                    ARCHIVE_ROOT / "screenshots" / PurePosixPath(*path.parts[2:])
                ).as_posix()
            if convenience_name and convenience_name not in names:
                raise RuntimeError(f"Archive is missing convenience copy {convenience_name}")
        for name in names:
            path = PurePosixPath(name)
            if BLOCKED_PARTS.intersection(path.parts) or path.suffix.lower() in BLOCKED_SUFFIXES:
                raise RuntimeError(f"Archive contains blocked runtime path: {name}")
            if path.name == "CODEX_PROMPT.md" or "PreExecution_Audit" in name:
                raise RuntimeError(f"Archive contains an internal audit/input file: {name}")

        manifest_name = (ARCHIVE_ROOT / "SUBMISSION_MANIFEST.txt").as_posix()
        manifest_text = archive.read(manifest_name).decode("utf-8")
        if "Archive paths:\n" not in manifest_text:
            raise RuntimeError("Submission manifest does not contain an archive-path inventory.")
        manifest_paths = {
            line for line in manifest_text.split("Archive paths:\n", 1)[1].splitlines() if line
        }
        if manifest_paths != names:
            raise RuntimeError("Submission manifest does not enumerate the complete archive payload.")

        checksum_name = (ARCHIVE_ROOT / "SHA256SUMS.txt").as_posix()
        checksum_lines = archive.read(checksum_name).decode("utf-8").splitlines()
        checksum_entries: dict[str, str] = {}
        for line in checksum_lines:
            expected, name = line.split("  ", 1)
            if name in checksum_entries:
                raise RuntimeError(f"Duplicate checksum entry: {name}")
            checksum_entries[name] = expected
            if name not in names:
                raise RuntimeError(f"Checksum entry is missing from archive: {name}")
            actual = sha256(archive.read(name))
            if actual != expected:
                raise RuntimeError(f"Checksum mismatch for {name}")
        expected_checksum_names = names - {checksum_name}
        if set(checksum_entries) != expected_checksum_names:
            raise RuntimeError("Checksum inventory does not cover every archive payload file.")

        with tempfile.TemporaryDirectory(prefix="mamacare-archive-check-") as temp_dir:
            archive.extractall(temp_dir)
            if not (Path(temp_dir) / ARCHIVE_ROOT.as_posix() / "source_project" / "README.md").is_file():
                raise RuntimeError("Archive extraction did not produce the source-project README.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the complete MamaCare capstone submission.")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()

    require_files()
    verify_tracked_cleanliness()
    verify_text_controls()
    verify_data_and_metadata()
    verify_analysis_outputs()
    verify_documents()
    if args.archive:
        verify_archive(args.archive)
    print("All MamaCare submission checks passed.")


if __name__ == "__main__":
    main()
