# Execution Guide

## 1. Open the project root

The terminal must be in the folder containing `README.md`, `app/`, `src/`, `models/` and `requirements.txt`.

## 2. Create a clean environment

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The pinned dependency set is tested with Python 3.14. Using an older interpreter can make binary packages resolve differently.

## 3. Train and regenerate model evidence

```powershell
.\.venv\Scripts\python.exe -m src.train
```

Training validates the raw data, recreates measurement-signature-separated splits, compares the requested models, selects by validation Weighted F1, writes all tables and figures, and serializes the final pipeline and metadata.

## 4. Rebuild, execute and export the notebook

```powershell
.\.venv\Scripts\python.exe scripts\build_notebook.py
.\.venv\Scripts\python.exe -m nbconvert --execute --to notebook --inplace notebooks\MamaCare_End_to_End.ipynb
.\.venv\Scripts\python.exe -m nbconvert --to html --output MamaCare_End_to_End.html --output-dir reports\notebook notebooks\MamaCare_End_to_End.ipynb
```

## 5. Run automated and submission checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\verify_submission.py
```

## 6. Start and test the frontend

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\app.py
```

Manually test the demonstration cases, acknowledgement, manual inputs, one unavailable measurement, Assess, Reset, Start a new assessment, summary download, uncertainty state, model-behaviour control, Model evidence and Responsible use tabs. Repeat at desktop and phone widths.

## 7. Clean runtime files

```powershell
.\.venv\Scripts\python.exe scripts\clean_project.py
```

The cleanup script removes runtime caches and temporary files. The ignored environment may be recreated from `requirements.txt` at any time.

## 8. Commit and publish the correction branch

```powershell
git switch mamacare-capstone-final
git status
git add --all
git commit -m "Complete MamaCare capstone submission"
git push -u origin mamacare-capstone-final
```

Open `https://github.com/mikaye2005/Maternal_Health_Risk_Prediction`, verify the branch contents and README images, then merge through the repository's normal review workflow if `main` should become the public submission branch.

## 9. Build the final archive

```powershell
.\.venv\Scripts\python.exe scripts\package_submission.py
.\.venv\Scripts\python.exe scripts\verify_submission.py --archive dist\MamaCare_Final_Submission.zip
```

The archive contains a manifest, repository URL and SHA-256 checksums and excludes `.git`, virtual environments, caches, logs and notebook checkpoints.
