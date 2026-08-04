from pathlib import Path
import hashlib
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_mandatory_capstone_deliverables_exist():
    required = [
        "README.md",
        "LICENSE",
        "DATA_LICENSE.md",
        "CITATION.cff",
        "requirements.txt",
        "requirements-lock.txt",
        ".streamlit/config.toml",
        "run_app.ps1",
        "run_app.sh",
        "notebooks/MamaCare_End_to_End.ipynb",
        "reports/notebook/MamaCare_End_to_End.html",
        "reports/responsible_ai_statement.md",
        "reports/responsible_ai_statement.pdf",
        "reports/MamaCare_Capstone_Report.pdf",
        "reports/MamaCare_Capstone_Report.docx",
        "presentation/MamaCare_Capstone_Presentation.pptx",
        "presentation/MamaCare_Capstone_Presentation.pdf",
        "presentation/speaker_notes.md",
        "docs/CAPSTONE_COMPLIANCE_CHECKLIST.md",
        "docs/SUBMISSION_CHECKLIST.md",
        "docs/TESTING_REPORT.md",
        "docs/SCOPE_DECISIONS.md",
        "docs/DATASET_SELECTION.md",
        "docs/DATA_DICTIONARY.md",
        "docs/MODEL_CARD.md",
        "docs/EXECUTION_GUIDE.md",
        "docs/FINAL_SUBMISSION_MANIFEST.md",
        "docs/BACKUP_MANIFEST_PRE_UPDATE.md",
        "docs/references/Original_MamaCare_Project_Proposal.pdf",
        "assets/screenshots/mamacare_frontend_preview.png",
        "assets/screenshots/mamacare_frontend_mobile.png",
        "models/maternal_risk_pipeline.joblib",
        "models/model_metadata.json",
    ]
    for relative in required:
        path = ROOT / relative
        assert path.exists(), relative
        assert path.stat().st_size > 100, relative


def test_complete_frontend_screenshot_evidence_exists():
    required = [
        "mamacare_frontend_landing.png",
        "mamacare_frontend_preview.png",
        "mamacare_frontend_uncertainty.png",
        "mamacare_frontend_missing_measurement.png",
        "mamacare_frontend_what_if.png",
        "mamacare_frontend_model_evidence.png",
        "mamacare_frontend_responsible_use.png",
        "mamacare_frontend_mobile.png",
    ]
    for file_name in required:
        path = ROOT / "assets" / "screenshots" / file_name
        assert path.exists(), file_name
        assert path.stat().st_size > 1_000, file_name


def test_preserved_proposal_is_byte_identical_to_mentor_copy():
    path = ROOT / "docs" / "references" / "Original_MamaCare_Project_Proposal.pdf"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "a44aa67e4a6ad85f417d2e38fa3fd38f4f98c84e358aeac0aa7ce668e5053be5"
    )


def test_gitignore_blocks_runtime_cache_directories():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in [".venv/", "__pycache__/", ".pytest_cache/", ".ipynb_checkpoints/"]:
        assert entry in text


def test_no_forbidden_runtime_files_are_tracked():
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    blocked_parts = {".venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
    assert not [path for path in tracked if blocked_parts.intersection(Path(path).parts)]
    assert not [path for path in tracked if Path(path).suffix.lower() in {".pyc", ".pyo", ".log", ".tmp"}]


def test_frontend_source_contains_required_safety_and_actions():
    text = (ROOT / "app" / "app.py").read_text(encoding="utf-8")
    required_phrases = [
        "Assess maternal risk",
        "Reset measurements",
        "Start a new assessment",
        "Download assessment summary",
        "not a medical diagnosis",
        "Dataset-bound ranges",
        "Responsible use",
        "Model evidence",
    ]
    for phrase in required_phrases:
        assert phrase in text
