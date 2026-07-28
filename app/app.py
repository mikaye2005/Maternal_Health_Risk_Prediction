from pathlib import Path
import sys

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from app.app_helpers import load_artifacts, predict_result, what_if
from src.config import FEATURES


st.set_page_config(page_title="MamaCare Risk Screening Demo", page_icon="🤱", layout="wide")
st.title("MamaCare — Maternal Health Risk Screening")
st.write(
    "A capstone demonstration that classifies six measurements as Low, Mid, or High Risk. "
    "It supports one unavailable measurement and reports uncertainty."
)

try:
    model, metadata = load_artifacts()
except Exception as exc:
    st.error(f"Model artifacts could not be loaded: {exc}")
    st.stop()

labels = {
    "Age": "Age (years)", "SystolicBP": "Systolic blood pressure (mmHg)",
    "DiastolicBP": "Diastolic blood pressure (mmHg)", "BS": "Blood sugar (mmol/L)",
    "BodyTemp": "Body temperature (°F)", "HeartRate": "Heart rate (beats/min)",
}
unavailable = st.selectbox(
    "Unavailable measurement (optional)", ["None"] + FEATURES,
    help="At most one input may be unavailable. The model will use training-data imputation.",
)
values = {}
cols = st.columns(2)
for i, feature in enumerate(FEATURES):
    low, high = metadata["feature_ranges"][feature]
    disabled = unavailable == feature
    value = cols[i % 2].number_input(
        labels[feature], min_value=float(low), max_value=float(high),
        value=float(metadata["feature_medians"][feature]),
        step=0.1 if feature in {"BS", "BodyTemp"} else 1.0,
        disabled=disabled,
    )
    values[feature] = np.nan if disabled else value

if unavailable != "None":
    st.warning("One measurement is unavailable. Prediction reliability may be reduced.")

if st.button("Predict risk", type="primary"):
    try:
        result = predict_result(model, metadata, values)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.session_state["base_values"] = values.copy()
        st.session_state["base_result"] = result

if "base_result" in st.session_state:
    result = st.session_state["base_result"]
    display = metadata["label_mapping"][result["label"]]
    left, middle, right = st.columns(3)
    left.metric("Predicted risk", display)
    middle.metric("Estimated probability", f"{result['probability']:.0%}")
    right.metric("Status", "Uncertain result" if result["uncertain"] else "Above uncertainty threshold")
    if result["uncertain"]:
        st.warning(
            "The measurements produce an uncertain prediction. Further professional assessment "
            "is recommended."
        )
    st.subheader("Top measurement influences")
    st.caption(
        "Influence is estimated by replacing one measurement at a time with its training median. "
        "It shows model association, not causation."
    )
    for item in result["explanations"]:
        st.write(f"- **{labels[item['feature']]}** {item['direction']} this prediction "
                 f"(sensitivity {item['importance']:.3f}).")

    st.subheader("Limited what-if analysis")
    st.caption("This is a model-sensitivity demonstration, not treatment advice.")
    feature = st.selectbox("Measurement to adjust", [f for f in FEATURES if f != unavailable])
    low, high = metadata["feature_ranges"][feature]
    adjusted = st.slider(
        f"Adjusted {labels[feature]}", float(low), float(high),
        float(st.session_state["base_values"][feature]),
        0.1 if feature in {"BS", "BodyTemp"} else 1.0,
    )
    comparison = what_if(model, metadata, st.session_state["base_values"], feature, adjusted)
    st.write(
        f"Original: **{metadata['label_mapping'][result['label']]}** ({result['probability']:.0%})  \n"
        f"Adjusted: **{metadata['label_mapping'][comparison['label']]}** "
        f"({comparison['probability']:.0%})"
    )

st.divider()
st.info(
    "Academic screening-support prediction only — not a medical diagnosis. Do not use this "
    "demonstration for treatment decisions or emergency assessment."
)
with st.expander("Dataset and project limitations"):
    st.write(
        "The model uses a small public dataset of 1,014 records with many duplicate observations, "
        "six measurements, no demographic or care-context variables, and uncertain external "
        "generalizability. Inputs and predictions are not stored. Local clinical validation would "
        "be required before any real-world use."
    )
