from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "Maternal Health Risk Data Set.csv"
CLEAN_DATA = ROOT / "data" / "processed" / "maternal_health_risk_cleaned.csv"
MODEL_PATH = ROOT / "models" / "maternal_risk_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
DOCS = ROOT / "docs"

FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
ENGINEERED = ["PulsePressure", "MeanArterialPressure", "AgeBand"]
TARGET = "RiskLevel"
TRAINING_GROUP_COLUMN = "_MeasurementSignatureGroup"
CLASS_ORDER = ["low risk", "mid risk", "high risk"]
LOG_LOSS_CLASS_ORDER = ["high risk", "low risk", "mid risk"]
SEED = 42
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00639/"
    "Maternal%20Health%20Risk%20Data%20Set.csv"
)
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/863/maternal+health+risk"
DATASET_DOI = "10.24432/C5DP5D"
DATASET_LICENSE = "CC BY 4.0"
LABEL_MAPPING = {"low risk": "Low Risk", "mid risk": "Mid Risk", "high risk": "High Risk"}
