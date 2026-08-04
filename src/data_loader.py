from __future__ import annotations

from pathlib import Path
import requests
import pandas as pd

from src.config import FEATURES, TARGET, UCI_URL


def download_dataset(destination: Path, timeout: int = 30) -> Path:
    """Download the UCI CSV only when the raw dataset is not already present."""
    destination = Path(destination)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(UCI_URL, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Dataset download failed. Add the CSV to {destination} and rerun.") from exc
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
        raise ValueError(f"Unexpected dataset schema. Expected {expected}; got {frame.columns.tolist()}")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    labels = set(frame[TARGET].astype(str).str.strip().str.lower())
    if labels != {"low risk", "mid risk", "high risk"}:
        raise ValueError(f"Unexpected target labels: {sorted(labels)}")
    return frame


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned[TARGET] = cleaned[TARGET].astype(str).str.strip().str.lower()
    for column in FEATURES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
    return cleaned


def feature_signature(frame: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(frame[FEATURES], index=False).astype(str)
