# Final Submission Checklist

## Rebuild and test

- [ ] Install `requirements.txt` in a clean environment.
- [ ] Run `.\.venv\Scripts\python.exe -m src.train` and confirm all ten candidates produce metrics.
- [ ] Rebuild, execute and export the notebook with no failed cells.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`.
- [ ] Run `.\.venv\Scripts\python.exe scripts/verify_submission.py`.
- [ ] Confirm the raw CSV SHA-256 remains `a1f7025719f84715096e0d1f95ae2e56b57809b9b15449e1836c96a7d976ae9b`.

## Interface

- [ ] Start Streamlit without errors.
- [ ] Test all three demonstration cases and manual assessment.
- [ ] Test one unavailable measurement and invalid blood-pressure handling.
- [ ] Confirm input changes clear the old result.
- [ ] Test Assess, Reset, Start a new assessment and summary download.
- [ ] Test the model-behaviour control without invalid pressure combinations.
- [ ] Verify Model evidence and Responsible use content.
- [ ] Check desktop and phone layouts and refresh all eight live screenshot states.

## Documents and presentation

- [ ] Confirm the final metrics match across metadata, README, notebook, report and slides.
- [ ] Confirm the Responsible AI statement remains 500-700 words.
- [ ] Render and inspect every DOCX/PDF report page.
- [ ] Render and inspect all five PPTX/PDF slides.
- [ ] Rehearse the slide notes and demonstration within five to eight minutes.

## GitHub

- [ ] Commit on `mamacare-capstone-final`.
- [ ] Push to `https://github.com/mikaye2005/Maternal_Health_Risk_Prediction`.
- [ ] Verify that the public branch shows the rebuilt README, images and links.
- [ ] Merge through the repository's normal review workflow if `main` is the required public submission branch.

## Final archive

- [ ] Run `.\.venv\Scripts\python.exe scripts/package_submission.py`.
- [ ] Run `.\.venv\Scripts\python.exe scripts/verify_submission.py --archive dist/MamaCare_Final_Submission.zip`.
- [ ] Confirm the ZIP extracts, checksums pass and no environment/cache files are present.
- [ ] Submit the required repository link, ZIP, presentation PDF and Responsible AI PDF before the deadline.
