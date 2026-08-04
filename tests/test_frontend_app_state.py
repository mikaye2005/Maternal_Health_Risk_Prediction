import json
import math
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import app.app_helpers as app_helpers


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "app.py"
ACKNOWLEDGEMENT = (
    "I understand that this is an academic model output and not a medical diagnosis."
)


class DeterministicModel:
    """Small model double that keeps AppTest focused on frontend state."""

    classes_ = np.array(["high risk", "low risk", "mid risk"])

    def predict_proba(self, frame):
        probabilities = np.array([0.20, 0.62, 0.18])
        return np.tile(probabilities, (len(frame), 1))


def widget_by_label(elements, label):
    matches = [element for element in elements if element.label == label]
    assert len(matches) == 1, f"Expected one widget labelled {label!r}, found {len(matches)}"
    return matches[0]


def assert_no_app_exception(app):
    assert len(app.exception) == 0, [str(exception.value) for exception in app.exception]


@pytest.fixture
def app_factory(monkeypatch):
    metadata = json.loads(
        (ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8")
    )
    model = DeterministicModel()
    monkeypatch.setattr(
        app_helpers,
        "load_artifacts",
        lambda: (model, metadata),
    )
    st.cache_resource.clear()

    def create_app():
        app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        assert_no_app_exception(app)
        return app

    yield create_app
    st.cache_resource.clear()


def submit_assessment(app):
    widget_by_label(app.checkbox, ACKNOWLEDGEMENT).check().run()
    widget_by_label(app.button, "Assess maternal risk").click().run()
    assert_no_app_exception(app)


def test_demo_case_does_not_auto_acknowledge_or_predict(app_factory):
    app = app_factory()

    widget_by_label(app.checkbox, ACKNOWLEDGEMENT).check().run()
    assert widget_by_label(app.checkbox, ACKNOWLEDGEMENT).value is True

    widget_by_label(app.button, "Elevated-value example").click().run()

    assert_no_app_exception(app)
    assert widget_by_label(app.checkbox, ACKNOWLEDGEMENT).value is False
    assert widget_by_label(app.button, "Assess maternal risk").disabled is True
    assert widget_by_label(app.number_input, "Age (years)").value == 48.0
    assert app.session_state["base_result"] is None
    assert not any(
        selectbox.label == "Measurement to adjust" for selectbox in app.selectbox
    )


def test_changing_an_input_clears_the_submitted_prediction(app_factory):
    app = app_factory()
    submit_assessment(app)

    assert app.session_state["base_result"] is not None
    assert any(
        selectbox.label == "Measurement to adjust" for selectbox in app.selectbox
    )
    assert len(app.slider) == 1

    widget_by_label(app.number_input, "Age (years)").set_value(27.0).run()

    assert_no_app_exception(app)
    assert app.session_state["base_result"] is None
    assert app.session_state["base_values"] is None
    assert not any(
        selectbox.label == "Measurement to adjust" for selectbox in app.selectbox
    )
    assert len(app.slider) == 0


def test_blood_pressure_what_if_cannot_submit_impossible_combinations(app_factory):
    systolic_app = app_factory()
    widget_by_label(systolic_app.button, "Elevated-value example").click().run()
    submit_assessment(systolic_app)
    widget_by_label(
        systolic_app.selectbox, "Measurement to adjust"
    ).select("SystolicBP").run()

    systolic = widget_by_label(
        systolic_app.slider, "Adjusted Systolic blood pressure (mmHg)"
    )
    assert (systolic.min, systolic.max, systolic.value) == (90.0, 160.0, 140.0)
    systolic.set_value(90.0).run()
    assert_no_app_exception(systolic_app)

    # AppTest simulates an invalid client value; Streamlit rejects it and restores
    # a value inside the declared widget bounds before application code sees it.
    widget_by_label(
        systolic_app.slider, "Adjusted Systolic blood pressure (mmHg)"
    ).set_value(80.0).run()
    assert_no_app_exception(systolic_app)
    assert (
        widget_by_label(
            systolic_app.slider, "Adjusted Systolic blood pressure (mmHg)"
        ).value
        >= 90.0
    )

    diastolic_app = app_factory()
    widget_by_label(diastolic_app.button, "Younger-age example").click().run()
    submit_assessment(diastolic_app)
    widget_by_label(
        diastolic_app.selectbox, "Measurement to adjust"
    ).select("DiastolicBP").run()

    diastolic = widget_by_label(
        diastolic_app.slider, "Adjusted Diastolic blood pressure (mmHg)"
    )
    assert (diastolic.min, diastolic.max, diastolic.value) == (49.0, 90.0, 65.0)
    diastolic.set_value(90.0).run()
    assert_no_app_exception(diastolic_app)

    widget_by_label(
        diastolic_app.slider, "Adjusted Diastolic blood pressure (mmHg)"
    ).set_value(100.0).run()
    assert_no_app_exception(diastolic_app)
    assert (
        widget_by_label(
            diastolic_app.slider, "Adjusted Diastolic blood pressure (mmHg)"
        ).value
        <= 90.0
    )


def test_one_missing_measurement_and_reset_flow(app_factory):
    app = app_factory()

    widget_by_label(app.selectbox, "Unavailable measurement").select(
        "Blood sugar"
    ).run()

    assert_no_app_exception(app)
    assert widget_by_label(
        app.number_input, "Blood sugar (mmol/L)"
    ).disabled is True
    assert any(
        "One measurement is unavailable" in str(warning.value)
        for warning in app.warning
    )

    submit_assessment(app)

    assert app.session_state["base_result"]["missing"] == 1
    assert math.isnan(app.session_state["base_values"]["BS"])
    measurement_to_adjust = widget_by_label(
        app.selectbox, "Measurement to adjust"
    )
    assert "Blood sugar" not in list(measurement_to_adjust.options)

    widget_by_label(app.button, "Reset measurements").click().run()

    assert_no_app_exception(app)
    assert (
        widget_by_label(app.selectbox, "Unavailable measurement").value
        == "All measurements are available"
    )
    assert widget_by_label(app.checkbox, ACKNOWLEDGEMENT).value is False
    assert widget_by_label(app.button, "Assess maternal risk").disabled is True
    assert widget_by_label(
        app.number_input, "Blood sugar (mmol/L)"
    ).disabled is False
    assert widget_by_label(app.number_input, "Blood sugar (mmol/L)").value == 7.5
    assert app.session_state["base_result"] is None
    assert app.session_state["base_values"] is None
    assert len(app.slider) == 0


def test_evidence_and_responsible_use_tabs_render(app_factory):
    app = app_factory()

    assert [tab.label for tab in app.tabs] == [
        "Overview",
        "Assessment",
        "Model evidence",
        "Responsible use",
    ]

    rendered_markdown = "\n".join(str(element.value) for element in app.markdown)
    for expected in [
        "Weighted F1",
        "Macro F1",
        "High Risk recall",
        "Log loss",
        "Mid Risk recall",
        "Affected groups include",
        "Equal Opportunity for High Risk identification across age groups",
        "acknowledged within one working day",
        "Monthly monitoring would withdraw the system",
    ]:
        assert expected in rendered_markdown

    assert any(
        button.label == "Download Responsible AI statement"
        for button in app.get("download_button")
    )
    assert_no_app_exception(app)
