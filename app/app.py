from __future__ import annotations

import base64
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from app.app_helpers import (
    DISPLAY_LABELS,
    UNITS,
    build_assessment_summary,
    load_artifacts,
    predict_result,
    what_if,
)
from src.config import FEATURES, FIGURES, REPORTS, TABLES


st.set_page_config(
    page_title="MamaCare Maternal Health Risk Capstone",
    page_icon="🤱",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def svg_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


LOGO_URI = svg_data_uri(ROOT / "assets" / "brand" / "mamacare_mark.svg")

css = """
<style>
:root {
  --brand-teal-dark:#064e3b; --brand-teal:#0f766e; --brand-coral:#ec4899;
  --orange-500:#f97316; --slate-900:#0f172a; --slate-600:#475569;
  --slate-200:#e2e8f0; --mint:#ecfdf5; --peach:#fff0f4;
}
html, body, [class*="css"] {font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}
.block-container {padding-top:1.2rem; padding-bottom:2.5rem; max-width:1220px;}
.hero {
  display:grid; grid-template-columns:110px 1fr; gap:1.4rem; align-items:center;
  background:linear-gradient(135deg,#fff0f4 0%,#f5f8f7 55%,#ecfdf5 100%);
  border:1px solid rgba(15,118,110,.12); border-radius:30px; padding:1.5rem 1.8rem;
  box-shadow:0 22px 55px rgba(15,118,110,.10); margin-bottom:1rem;
}
.hero img {width:100px; height:100px;}
.brand-badge {display:inline-flex; padding:.35rem .75rem; border-radius:999px; background:#e0f7f3; color:var(--brand-teal-dark); font-weight:800; font-size:.78rem; letter-spacing:.05em;}
.hero h1 {font-size:3.15rem; line-height:1; margin:.4rem 0 .3rem; color:var(--brand-teal-dark);}
.hero p {font-size:1.08rem; color:#334155; margin:0; max-width:860px;}
.panel {background:#fff; border:1px solid var(--slate-200); border-radius:24px; padding:1.25rem 1.35rem; box-shadow:0 14px 34px rgba(15,23,42,.05);}
.section-title {font-size:1.2rem; font-weight:900; color:var(--brand-teal-dark); margin-bottom:.4rem;}
.notice, .safe, .danger-note {border-radius:16px; padding:.9rem 1rem;}
.notice {background:#fff0f4; border-left:5px solid var(--brand-coral); color:#9f1239;}
.safe {background:#ecfdf5; border-left:5px solid var(--brand-teal); color:var(--brand-teal-dark);}
.danger-note {background:#fef2f2; border-left:5px solid #dc2626; color:#7f1d1d;}
.kpi-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.85rem;}
.kpi {background:white; border:1px solid var(--slate-200); border-radius:20px; padding:1rem;}
.kpi .label {font-size:.76rem; text-transform:uppercase; letter-spacing:.08em; color:#64748b; font-weight:800;}
.kpi .value {font-size:1.65rem; font-weight:900; color:var(--slate-900);}
.result-card {border-radius:26px; padding:1.45rem; border:1px solid #e2e8f0; background:white; box-shadow:0 20px 45px rgba(15,23,42,.07);}
.result-low {border-left:10px solid var(--brand-teal);}
.result-mid {border-left:10px solid var(--brand-coral);}
.result-high {border-left:10px solid #dc2626;}
.result-title {font-size:.78rem; text-transform:uppercase; letter-spacing:.09em; color:#64748b; font-weight:900;}
.result-value {font-size:2rem; font-weight:950; color:#0f172a; margin:.2rem 0;}
.prob-track {height:12px; background:#e2e8f0; border-radius:999px; overflow:hidden;}
.prob-fill {height:100%; border-radius:999px;}
.prob-line {display:grid; grid-template-columns:120px 1fr 58px; align-items:center; gap:.8rem; margin:.58rem 0;}
.prob-label {font-weight:750; color:#334155;}
.prob-value {font-weight:850; text-align:right; color:#0f172a;}
.workflow {display:grid; grid-template-columns:repeat(5,1fr); gap:.55rem; margin-top:.75rem;}
.workflow div {background:#f8fafc; border:1px solid #e2e8f0; border-radius:16px; padding:.75rem; text-align:center; font-weight:750; color:#334155;}
.small-muted {font-size:.88rem; color:#64748b;}
.hero-subtitle {max-width:780px; color:#334155; margin-top:.35rem; line-height:1.6;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:.35rem;}
[data-testid="stTabs"] button {border-radius:999px; padding:.5rem .9rem;}
footer {visibility:hidden;}
@media (max-width:760px) {
  .hero {grid-template-columns:72px 1fr; padding:1.1rem; border-radius:22px;}
  .hero img {width:66px; height:66px;}
  .hero h1 {font-size:2.2rem;}
  .hero-subtitle {font-size:.95rem;}
  .kpi-grid, .workflow {grid-template-columns:1fr;}
  .prob-line {grid-template-columns:95px 1fr 48px; gap:.45rem;}
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def cached_artifacts():
    return load_artifacts()


@st.cache_data(show_spinner=False)
def read_table(name: str) -> pd.DataFrame:
    path = TABLES / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def clear_assessment_result() -> None:
    st.session_state["base_result"] = None
    st.session_state["base_values"] = None
    for key in list(st.session_state):
        if key == "what_if_feature" or key.startswith("what_if_value_"):
            del st.session_state[key]


def reset_assessment(metadata) -> None:
    clear_assessment_result()
    st.session_state["input_unavailable"] = "All measurements are available"
    st.session_state["assessment_acknowledgement"] = False
    for feature in FEATURES:
        st.session_state[f"input_{feature}"] = float(metadata["feature_medians"][feature])


def load_demo_case(case_name: str) -> None:
    """Load a reproducible demonstration record without producing a prediction."""
    cases = {
        "routine": {
            "Age": 24.0, "SystolicBP": 120.0, "DiastolicBP": 80.0,
            "BS": 7.0, "BodyTemp": 98.0, "HeartRate": 76.0,
        },
        "elevated": {
            "Age": 48.0, "SystolicBP": 140.0, "DiastolicBP": 90.0,
            "BS": 15.0, "BodyTemp": 98.0, "HeartRate": 90.0,
        },
        "young": {
            "Age": 16.0, "SystolicBP": 90.0, "DiastolicBP": 65.0,
            "BS": 6.9, "BodyTemp": 98.0, "HeartRate": 76.0,
        },
    }
    selected = cases[case_name]
    clear_assessment_result()
    st.session_state["input_unavailable"] = "All measurements are available"
    st.session_state["assessment_acknowledgement"] = False
    for feature, value in selected.items():
        st.session_state[f"input_{feature}"] = value


def probability_html(label: str, probability: float, color: str) -> str:
    return f"""
<div class="prob-line">
  <div class="prob-label">{label}</div>
  <div class="prob-track"><div class="prob-fill" style="width:{probability*100:.1f}%;background:{color};"></div></div>
  <div class="prob-value">{probability:.0%}</div>
</div>
"""


try:
    model, metadata = cached_artifacts()
except Exception as exc:
    st.error(f"Model artifacts could not be loaded: {exc}")
    st.stop()

st.session_state.setdefault("base_result", None)
st.session_state.setdefault("base_values", None)
st.session_state.setdefault("input_unavailable", "All measurements are available")
st.session_state.setdefault("assessment_acknowledgement", False)
for feature in FEATURES:
    st.session_state.setdefault(f"input_{feature}", float(metadata["feature_medians"][feature]))

st.markdown(
    f"""
<div class="hero">
  <img src="{LOGO_URI}" alt="MamaCare logo" />
  <div>
    <span class="brand-badge">MATERNAL HEALTH CLASSIFICATION CAPSTONE</span>
    <h1>MamaCare</h1>
    <p class="hero-subtitle">A maternal-health risk classifier aligned to the new MamaCare branding, using six routine measurements.</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

about_tab, assess_tab, model_tab, ethics_tab = st.tabs(
    ["Overview", "Assessment", "Model evidence", "Responsible use"]
)

with about_tab:
    left, right = st.columns([1.25, .75], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Project purpose</div>', unsafe_allow_html=True)
        st.write(
            "MamaCare uses age, systolic blood pressure, diastolic blood pressure, "
            "blood sugar, body temperature and heart rate to classify a record into "
            "one of three maternal-health risk categories."
        )
        st.write(
            "The source is the UCI Maternal Health Risk dataset collected through "
            "hospitals, community clinics and maternal-health facilities in rural Bangladesh. "
            "The project has not been clinically validated for Kenya."
        )
        st.markdown(
            '<div class="workflow"><div>Inspect</div><div>Clean</div><div>Model</div><div>Evaluate</div><div>Explain</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown(
            '<div class="notice"><b>Academic use only.</b><br>This is not a diagnosis, treatment tool, triage system or emergency assessment.</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            '<div class="safe"><b>Privacy by design.</b><br>The assessment does not request names, phone numbers, addresses or national identification details.</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        f"""
<div class="kpi-grid">
  <div class="kpi"><div class="label">Downloaded records</div><div class="value">{metadata['downloaded_csv_rows']:,}</div></div>
  <div class="kpi"><div class="label">Input measurements</div><div class="value">6</div></div>
  <div class="kpi"><div class="label">Target categories</div><div class="value">3</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

with assess_tab:
    st.markdown(
        '<div class="notice"><b>Dataset-bound ranges:</b> the permitted values reflect the full source dataset and are used only for input validation. They are not clinical reference ranges or definitions of normal and abnormal health.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    with st.expander("Load a demonstration case", expanded=False):
        st.caption(
            "These values are included only to make the project demonstration repeatable. "
            "They are not clinical examples or treatment guidance."
        )
        demo1, demo2, demo3 = st.columns(3)
        demo1.button(
            "Routine-value example",
            width="stretch",
            on_click=load_demo_case,
            args=("routine",),
        )
        demo2.button(
            "Elevated-value example",
            width="stretch",
            on_click=load_demo_case,
            args=("elevated",),
        )
        demo3.button(
            "Younger-age example",
            width="stretch",
            on_click=load_demo_case,
            args=("young",),
        )

    unavailable_options = {
        "All measurements are available": "None",
        **{DISPLAY_LABELS[feature]: feature for feature in FEATURES},
    }
    selected_display = st.selectbox(
        "Unavailable measurement",
        options=list(unavailable_options.keys()),
        key="input_unavailable",
        help="At most one measurement may be unavailable. The trained pipeline uses median imputation for that value.",
        on_change=clear_assessment_result,
    )
    unavailable = unavailable_options[selected_display]

    st.markdown("### Maternal measurements")
    info_col, guidance_col = st.columns([1.5, .5], gap="large")
    with info_col:
        st.markdown("**Patient information**")
        low, high = metadata["feature_ranges"]["Age"]
        st.number_input(
            f"{DISPLAY_LABELS['Age']} ({UNITS['Age']})",
            min_value=float(low),
            max_value=float(high),
            step=1.0,
            key="input_Age",
            disabled=unavailable == "Age",
            on_change=clear_assessment_result,
        )

        st.markdown("**Blood pressure**")
        bp1, bp2 = st.columns(2)
        for feature, column in [("SystolicBP", bp1), ("DiastolicBP", bp2)]:
            low, high = metadata["feature_ranges"][feature]
            column.number_input(
                f"{DISPLAY_LABELS[feature]} ({UNITS[feature]})",
                min_value=float(low),
                max_value=float(high),
                step=1.0,
                key=f"input_{feature}",
                disabled=unavailable == feature,
                on_change=clear_assessment_result,
            )

        st.markdown("**Other observations**")
        other_columns = st.columns(3)
        for feature, column in zip(["BS", "BodyTemp", "HeartRate"], other_columns):
            low, high = metadata["feature_ranges"][feature]
            column.number_input(
                f"{DISPLAY_LABELS[feature]} ({UNITS[feature]})",
                min_value=float(low),
                max_value=float(high),
                step=0.1 if feature in {"BS", "BodyTemp"} else 1.0,
                key=f"input_{feature}",
                disabled=unavailable == feature,
                on_change=clear_assessment_result,
            )
    with guidance_col:
        st.markdown(
            '<div class="panel"><div class="section-title">Before assessing</div><div class="small-muted">Check every unit carefully. The model accepts one unavailable measurement, but more complete inputs are preferable. Do not enter personal identifiers.</div></div>',
            unsafe_allow_html=True,
        )

    values = {
        feature: np.nan if unavailable == feature else st.session_state[f"input_{feature}"]
        for feature in FEATURES
    }
    if unavailable != "None":
        st.warning("One measurement is unavailable. Prediction reliability may be reduced.")

    st.checkbox(
        "I understand that this is an academic model output and not a medical diagnosis.",
        key="assessment_acknowledgement",
    )
    assess_col, reset_col = st.columns([1, 1])
    submitted = assess_col.button(
        "Assess maternal risk",
        type="primary",
        width="stretch",
        disabled=not st.session_state["assessment_acknowledgement"],
    )
    reset_col.button(
        "Reset measurements",
        width="stretch",
        on_click=reset_assessment,
        args=(metadata,),
    )

    if submitted:
        try:
            st.session_state["base_values"] = dict(values)
            st.session_state["base_result"] = predict_result(model, metadata, values)
        except ValueError as exc:
            st.error(str(exc))
            clear_assessment_result()

    result = st.session_state.get("base_result")
    base_values = st.session_state.get("base_values")
    if result is not None and base_values is not None:
        risk = result["label"]
        card_class = {
            "low risk": "result-low",
            "mid risk": "result-mid",
            "high risk": "result-high",
        }[risk]
        status = "Review recommended" if result["uncertain"] else "Higher-confidence model output"
        st.write("")
        st.markdown(f'<div class="result-card {card_class}">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">Screening result</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-value">{result["display_label"]}</div>', unsafe_allow_html=True)
        st.write(f"**Model status:** {status}")
        st.caption("This result reflects the measurements shown in the submitted-values table below.")
        st.caption(
            "The class scores describe the model output. They are not medically validated probabilities."
        )
        colors = {"low risk": "#0f766e", "mid risk": "#ec4899", "high risk": "#dc2626"}
        for label, probability in sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True):
            st.markdown(
                probability_html(metadata["label_mapping"][label], probability, colors[label]),
                unsafe_allow_html=True,
            )
        if result["uncertain"]:
            st.warning(
                f"The highest class score is below the {metadata['uncertainty_threshold']:.0%} review threshold. Professional assessment is recommended."
            )
        else:
            st.info(
                "The result is above the model's review threshold, but professional assessment remains necessary."
            )
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Measurements used for this result"):
            rows = []
            for feature in FEATURES:
                value = base_values[feature]
                rows.append({
                    "Measurement": DISPLAY_LABELS[feature],
                    "Value": "Unavailable" if pd.isna(value) else f"{float(value):g}",
                    "Unit": "" if pd.isna(value) else UNITS[feature],
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.subheader("Measurements influencing this model output")
        for rank, item in enumerate(result["explanations"], start=1):
            change = abs(item["delta"]) * 100
            st.write(
                f"{rank}. **{DISPLAY_LABELS[item['feature']]}** {item['direction']} "
                f"by about **{change:.1f} percentage points** when compared with the training median."
            )
        st.caption("This is local model sensitivity, not medical causation or treatment effect.")

        st.subheader("Explore model behaviour")
        st.caption(
            "This changes one measurement mathematically while keeping the other values fixed. "
            "It does not simulate treatment, recovery or future outcomes."
        )
        adjustable = [feature for feature in FEATURES if not pd.isna(base_values.get(feature))]
        feature_to_adjust = st.selectbox(
            "Measurement to adjust",
            adjustable,
            format_func=lambda feature: DISPLAY_LABELS[feature],
            key="what_if_feature",
        )
        low, high = metadata["feature_ranges"][feature_to_adjust]
        if feature_to_adjust == "SystolicBP" and pd.notna(base_values.get("DiastolicBP")):
            low = max(float(low), float(base_values["DiastolicBP"]))
        if feature_to_adjust == "DiastolicBP" and pd.notna(base_values.get("SystolicBP")):
            high = min(float(high), float(base_values["SystolicBP"]))
        adjusted = st.slider(
            f"Adjusted {DISPLAY_LABELS[feature_to_adjust]} ({UNITS[feature_to_adjust]})",
            min_value=float(low),
            max_value=float(high),
            value=float(base_values[feature_to_adjust]),
            step=0.1 if feature_to_adjust in {"BS", "BodyTemp"} else 1.0,
            key=f"what_if_value_{feature_to_adjust}",
        )
        comparison = what_if(model, metadata, base_values, feature_to_adjust, adjusted)
        c1, c2 = st.columns(2)
        c1.metric("Original model output", result["display_label"], f"{result['probability']:.0%} score")
        c2.metric("Adjusted model output", comparison["display_label"], f"{comparison['probability']:.0%} score")
        with st.expander("Compare all original and adjusted class scores"):
            comparison_rows = []
            for label in result["probabilities"]:
                comparison_rows.append({
                    "Risk class": metadata["label_mapping"][label],
                    "Original": result["probabilities"][label],
                    "Adjusted": comparison["probabilities"][label],
                })
            st.dataframe(
                pd.DataFrame(comparison_rows).style.format({"Original": "{:.1%}", "Adjusted": "{:.1%}"}),
                hide_index=True,
                width="stretch",
            )

        summary_text = build_assessment_summary(metadata, base_values, result)
        action1, action2 = st.columns(2)
        action1.download_button(
            "Download assessment summary",
            data=summary_text,
            file_name="mamacare_assessment_summary.txt",
            mime="text/plain",
            width="stretch",
        )
        action2.button(
            "Start a new assessment",
            width="stretch",
            on_click=reset_assessment,
            args=(metadata,),
        )

with model_tab:
    metrics = metadata["test_metrics"]
    st.markdown(
        f"""
<div class="kpi-grid">
  <div class="kpi"><div class="label">Weighted F1</div><div class="value">{metrics['weighted_f1']:.3f}</div></div>
  <div class="kpi"><div class="label">Macro F1</div><div class="value">{metrics['macro_f1']:.3f}</div></div>
  <div class="kpi"><div class="label">High Risk recall</div><div class="value">{metrics['high_risk_recall']:.3f}</div></div>
  <div class="kpi"><div class="label">Accuracy</div><div class="value">{metrics['accuracy']:.3f}</div></div>
  <div class="kpi"><div class="label">Log loss</div><div class="value">{metrics['log_loss']:.3f}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption("All final metrics come from the untouched, measurement-signature-separated test set.")
    st.markdown(
        f'<div class="danger-note"><b>Known weakness:</b> Mid Risk recall is {metadata["mid_risk_recall"]:.1%}. The model is not suitable for clinical deployment.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    chart1, chart2 = st.columns(2)
    if (FIGURES / "baseline_vs_model.png").exists():
        chart1.image(FIGURES / "baseline_vs_model.png", caption="Baseline comparison", width="stretch")
    if (FIGURES / "test_confusion_matrix.png").exists():
        chart2.image(FIGURES / "test_confusion_matrix.png", caption="Test-set errors by class", width="stretch")

    st.subheader("Age-group evaluation")
    if (FIGURES / "age_group_high_risk_recall.png").exists():
        st.image(FIGURES / "age_group_high_risk_recall.png", width="stretch")
    disaggregated = read_table("disaggregated_age_evaluation.csv")
    if not disaggregated.empty:
        show = disaggregated[["AgeGroup", "N", "HighRiskN", "weighted_f1", "macro_f1", "high_risk_recall"]].rename(columns={
            "AgeGroup": "Age group",
            "N": "Records",
            "HighRiskN": "Actual High Risk records",
            "weighted_f1": "Weighted F1",
            "macro_f1": "Macro F1",
            "high_risk_recall": "High Risk recall",
        })
        st.dataframe(
            show.style.format({
                "Weighted F1": "{:.3f}",
                "Macro F1": "{:.3f}",
                "High Risk recall": "{:.3f}",
            }),
            hide_index=True,
            width="stretch",
        )

    with st.expander("Dataset and model limitations"):
        st.write(
            "The dataset is small, duplicated and narrow. It has no Kenyan clinical validation, "
            "and some values appear implausible, including a heart-rate value of 7 beats per minute."
        )
        st.write(f"Exact duplicate rows: **{metadata['duplicate_count_reported']}**")
        st.write(f"Conflicting measurement signatures: **{metadata['conflicting_signature_count']}**")
        st.write(
            "Records with identical measurements can have different target labels. This places a hard limit on what a deterministic model can learn from the available variables."
        )

with ethics_tab:
    fairness_gap = metadata["equal_opportunity_summary"]["gap"]
    age_evidence = read_table("disaggregated_age_evaluation.csv")
    youngest_recall = None
    if not age_evidence.empty:
        youngest = age_evidence.loc[age_evidence["AgeGroup"] == "<=19", "high_risk_recall"]
        if not youngest.empty:
            youngest_recall = float(youngest.iloc[0])
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Responsible use summary</div>', unsafe_allow_html=True)
        st.write(
            "Affected groups include pregnant people whose records are classified, learners or health workers interpreting the output, "
            "families influenced by its wording and populations not represented in the rural-Bangladesh source data."
        )
        st.write(
            "The most serious plausible harm is false reassurance when an actual High Risk record is assigned a lower category."
        )
        st.write(
            "The fairness criterion is **Equal Opportunity for High Risk identification across age groups**. "
            f"The current High Risk recall gap is **{fairness_gap:.1%}**. "
            + (f"The `<=19` group recall is **{youngest_recall:.1%}** in this test split." if youngest_recall is not None else "")
        )
        st.write(
            "A research deployment would require local clinical validation and human review. "
            "Correction requests should be acknowledged within one working day and reassessed within two; this route must never replace urgent care."
        )
        st.write(
            "Monthly monitoring would withdraw the system if High Risk recall falls below 0.75 in a review with at least 50 actual High Risk records, "
            "or if a sufficiently sized age group trails overall recall by more than 0.10."
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown(
            '<div class="danger-note"><b>Do not use MamaCare for:</b><br>diagnosis, treatment, emergency triage, patient management or automatic referral decisions.</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        responsible_pdf = REPORTS / "responsible_ai_statement.pdf"
        if responsible_pdf.exists():
            st.download_button(
                "Download Responsible AI statement",
                data=responsible_pdf.read_bytes(),
                file_name="MamaCare_Responsible_AI_Statement.pdf",
                mime="application/pdf",
                width="stretch",
            )

st.divider()
st.caption(
    "MamaCare is an academic capstone prototype. It must not be used for diagnosis, treatment, "
    "triage or emergency decisions. Project author: Collins Mikaye."
)
