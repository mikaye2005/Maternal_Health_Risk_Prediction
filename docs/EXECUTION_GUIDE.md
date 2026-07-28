# Execution Guide

The requested Python 3.12 was unavailable. The user explicitly authorized the
installed Python 3.14.3. The existing environment was configured to inherit the
verified global scientific packages because the local package index was
unreachable.

```powershell
py -0p
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -c "import pandas,numpy,scipy,sklearn,streamlit,pytest"
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe scripts\build_notebook.py
.\.venv\Scripts\python.exe -m nbconvert --execute --to notebook --inplace notebooks\MamaCare_End_to_End.ipynb
py -3.14 -m nbconvert --to html --output MamaCare_End_to_End.html --output-dir reports notebooks\MamaCare_End_to_End.ipynb
.\run_app.ps1
```

The canonical acquisition function is `src.data_loader.download_dataset`, using
the official UCI direct CSV with a timeout and explicit errors. During this run,
local HTTPS access to UCI stalled, so the identically named file was obtained
from a public GitHub mirror and verified against the expected schema, 1,014-row
shape, class counts, ranges, and SHA-256 recorded in model metadata.

For a fresh deployment environment, use Python 3.12, create `.venv`, and install
`requirements.txt`. Run training only when rebuilding artifacts; the Streamlit
app loads the committed joblib artifact without retraining.
