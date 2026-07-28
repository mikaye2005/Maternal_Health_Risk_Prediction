import numpy as np
import pandas as pd

from src.config import FEATURES


def count_missing_measurements(values) -> int:
    frame = pd.DataFrame([values], columns=FEATURES) if not isinstance(values, pd.DataFrame) else values
    return int(frame[FEATURES].isna().sum(axis=1).max())


def validate_missing_measurements(values) -> None:
    count = count_missing_measurements(values)
    if count > 1:
        raise ValueError("A maximum of one unavailable measurement is permitted.")


def simulate_one_missing(
    frame: pd.DataFrame, fraction: float = 0.35, random_state: int = 42
) -> pd.DataFrame:
    masked = frame.copy()
    rng = np.random.default_rng(random_state)
    count = int(round(len(masked) * fraction))
    rows = rng.choice(len(masked), size=count, replace=False)
    columns = rng.integers(0, len(FEATURES), size=count)
    for row, column in zip(rows, columns):
        masked.iloc[row, masked.columns.get_loc(FEATURES[column])] = np.nan
    return masked
