# Final Submission Manifest

The packaging script creates `dist/MamaCare_Final_Submission.zip` with this layout:

```text
MamaCare_Final_Submission/
|-- README_FIRST.txt
|-- GITHUB_URL.txt
|-- SUBMISSION_CHECKLIST.md
|-- SUBMISSION_MANIFEST.txt
|-- SHA256SUMS.txt
|-- source_project/        # complete runnable, tracked repository
|   `-- docs/references/Original_MamaCare_Project_Proposal.pdf
|-- reports/               # convenient copy of final reports/evidence
|-- presentation/          # five-slide PPTX/PDF and speaker notes
`-- screenshots/           # tested desktop/mobile interface states
```

`SUBMISSION_MANIFEST.txt` enumerates every archive path. `SHA256SUMS.txt` covers every payload file other than the checksum file itself. The archive excludes Git metadata, virtual environments, caches, checkpoints, logs, temporary files, the mentor pre-execution dossier and internal execution prompts.

The preserved original proposal has SHA-256 `a44aa67e4a6ad85f417d2e38fa3fd38f4f98c84e358aeac0aa7ce668e5053be5`.

The public repository is `https://github.com/mikaye2005/Maternal_Health_Risk_Prediction`.

Build and verify from the project environment:

```powershell
.\.venv\Scripts\python.exe scripts\package_submission.py
.\.venv\Scripts\python.exe scripts\verify_submission.py --archive dist\MamaCare_Final_Submission.zip
```
