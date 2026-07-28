# MamaCare

MamaCare is a reproducible data-science capstone that predicts **Low Risk**, **Mid
Risk**, or **High Risk** from six measurements in the UCI Maternal Health Risk
dataset. It is a lightweight academic screening-support demonstration—not a
diagnostic, treatment, patient-management, or hospital system.

## Dataset

The project uses the [UCI Maternal Health Risk dataset](https://archive.ics.uci.edu/dataset/863/maternal+health+risk):
1,014 rows, six numerical inputs, and one target. The raw CSV is preserved at
`data/raw/Maternal Health Risk Data Set.csv` (SHA-256
`a1f7025719f84715096e0d1f95ae2e56b57809b9b15449e1836c96a7d976ae9b`).
There are 562 exact duplicate rows. Splits group identical six-measurement
signatures to prevent them crossing train, validation, and test partitions.

## Structure

- `src/`: validation, features, preprocessing, modelling, evaluation, clustering,
  uncertainty, explanations, and training
- `app/`: Streamlit interface and tested helpers
- `data/raw/`, `data/processed/`: immutable source and cleaned copy
- `models/`: fitted pipeline and metadata
- `notebooks/`: executable end-to-end analysis
- `reports/`: actual tables, figures, HTML notebook, and final results
- `docs/`: data dictionary, model card, execution guide, and testing report
- `tests/`: pytest suite

## Setup

The run completed with Python 3.14.3 because the user authorized use of the
installed version after Python 3.12 was unavailable. For Streamlit Community
Cloud, select Python 3.12 and install the direct dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Local verified commands:

```powershell
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m nbconvert --execute --to notebook --inplace notebooks\MamaCare_End_to_End.ipynb
.\run_app.ps1
```

## Contributions

1. **One unavailable measurement:** controlled masking affects training copies
   only; the app allows at most one missing input and rejects two.
2. **Evidence-based uncertainty:** candidate probability thresholds are compared
   on validation coverage and error rate. The selected threshold is 0.65.
3. **Individual explanation and what-if:** median-replacement probability
   sensitivity reports three influential measurements. One measurement can be
   adjusted to show model sensitivity, not treatment effect.

## Actual results

SVC was selected on validation Macro F1, High Risk recall, probability quality,
robustness, and deployment simplicity. On the untouched group-separated test
set: Macro F1 **0.520**, High Risk recall **0.727**, accuracy **0.522**, and log
loss **0.904**. Calibration was tested but not retained because validation log
loss worsened from 0.718 to 0.729. See `reports/final_results.md` and the model
card for complete results and limitations.

## Deployment preparation

Commit the repository, push it to a GitHub repository after reviewing artifacts,
then in Streamlit Community Cloud choose the repository, branch, and
`app/app.py`; select Python 3.12 and deploy. No secrets or database are required.
Test complete input, one unavailable measurement, uncertainty, explanation, and
what-if behavior at the public URL.

## Limitations and disclaimer

The small public dataset has extensive duplication, limited features, no
external validation, and uncertain representativeness. Group separation exposes
weaker generalization than a random row split. The interface does not store
submitted measurements.

**This is an academic screening-support prediction, not a medical diagnosis.
It must not guide treatment or emergency decisions. Local clinical validation
would be required before any real-world use.**
