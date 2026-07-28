from pathlib import Path
import requests
import pandas as pd

from src.config import FEATURES, TARGET, UCI_URL


def download_dataset(destination: Path, timeout: int = 30) -> Path:
    """Download the canonical UCI CSV without overwriting an existing raw file."""
    destination = Path(destination)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(UCI_URL, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"UCI dataset download failed: {exc}") from exc
    destination.write_bytes(response.content)
    validate_dataset(destination)
    return destination


def validate_dataset(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    expected = FEATURES + [TARGET]
    if frame.columns.tolist() != expected:
        raise ValueError(f"Unexpected schema. Expected {expected}, got {frame.columns.tolist()}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    labels = set(frame[TARGET].astype(str).str.strip().str.lower())
    if labels != {"low risk", "mid risk", "high risk"}:
        raise ValueError(f"Unexpected target labels: {sorted(labels)}")
    return frame


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned[TARGET] = cleaned[TARGET].astype(str).str.strip().str.lower()
    return cleaned
