# Capstone Compliance Checklist

| Mentor requirement | Status | Evidence |
|---|---|---|
| Specific problem statement with target, population, data and metric | Complete | README and notebook Section 1 |
| One target variable | Complete | `RiskLevel` |
| Dataset size above 500 rows | Complete | 1014 rows |
| Dataset source, DOI, licence and collection context | Complete | README, DATASET_SELECTION.md and DATA_LICENSE.md |
| Original proposal preserved unchanged | Complete | `docs/references/Original_MamaCare_Project_Proposal.pdf`, SHA-256 recorded in backup manifest |
| Target distribution and annotated proportions | Complete | `class_distribution_annotated.png` |
| Missing-data analysis | Complete | notebook and `missing_data_heatmap.png` |
| Numerical distributions and outlier discussion | Complete | histograms, boxplots and report |
| Duplicate and contradictory-label audit | Complete | `contradictory_measurement_signatures.csv` |
| Baseline comparison | Complete | `baseline_vs_model.csv` and chart |
| Random Forest and XGBoost | Complete | model-comparison table |
| Two neural-network architectures and training curves | Complete with documented scope revision | ShallowMLP, DeepMLP, training-curves figure and `docs/SCOPE_DECISIONS.md` |
| Primary metric Weighted F1 | Complete | model selection and README |
| Secondary metrics | Complete | Macro F1, High Risk recall, accuracy and log loss |
| Disaggregated evaluation | Complete | age-group table and Equal Opportunity chart |
| Structured error analysis | Complete | top errors, error types, age errors and conflicting-signature analysis |
| Clustering with interpretation | Complete | K-Means, hierarchical clustering, silhouette, Davies-Bouldin, method agreement and profiles |
| Responsible AI statement, 500-700 words | Complete | `reports/responsible_ai_statement.pdf` |
| Professional README | Complete | README.md |
| Version-pinned requirements | Complete | requirements.txt |
| Working demonstration interface | Complete | app/app.py and helper tests |
| Presentation | Complete | PPTX, PDF and speaker notes |
| Automated tests | Complete | tests/ |
| Public GitHub repository | Verification pending final push | Existing public remote: `https://github.com/mikaye2005/Maternal_Health_Risk_Prediction` |
| Final ZIP manifest and checksums | Verification pending final build | `scripts/package_submission.py` and `scripts/verify_submission.py` |
| Final Moodle submission | User action required | Submit ZIP/repository link before the deadline |
