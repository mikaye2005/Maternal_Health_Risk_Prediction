from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "Maternal Health Risk Data Set.csv"
CLEAN_DATA = ROOT / "data" / "processed" / "maternal_health_risk_cleaned.csv"
MODEL_PATH = ROOT / "models" / "maternal_risk_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
ENGINEERED = ["PulsePressure", "MeanArterialPressure", "AgeBand"]
TARGET = "RiskLevel"
CLASS_ORDER = ["low risk", "mid risk", "high risk"]
SEED = 42
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00639/"
    "Maternal%20Health%20Risk%20Data%20Set.csv"
)
