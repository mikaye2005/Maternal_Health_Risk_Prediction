import pandas as pd

from src.config import FEATURES, RAW_DATA, TARGET
from src.data_loader import validate_dataset


def test_dataset_schema_and_shape():
    data = validate_dataset(RAW_DATA)
    assert data.columns.tolist() == FEATURES + [TARGET]
    assert data.shape == (1014, 7)
    assert data[FEATURES].notna().all().all()


def test_target_labels():
    assert set(pd.read_csv(RAW_DATA)[TARGET]) == {"low risk", "mid risk", "high risk"}
