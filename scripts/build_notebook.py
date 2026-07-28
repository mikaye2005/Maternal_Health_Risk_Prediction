from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []


def md(title, text):
    cells.append(nbf.v4.new_markdown_cell(f"## {title}\n\n{text}"))


def code(source):
    cells.append(nbf.v4.new_code_cell(source))


cells.append(nbf.v4.new_markdown_cell(
    "# MamaCare: End-to-End Maternal Health Risk Prediction\n\n"
    "A reproducible capstone using the UCI Maternal Health Risk dataset. Results are "
    "academic screening support only and are not medical diagnoses."
))
md("1. Project overview", "The task is three-class prediction with group-separated evaluation, one-missing-measurement support, uncertainty warnings, and local model-sensitivity explanations.")
md("2. Dataset source and citation", "Source: [UCI Maternal Health Risk](https://archive.ics.uci.edu/dataset/863/maternal+health+risk). The raw source is preserved unchanged.")
md("3. Imports and reproducibility", "A fixed seed is used throughout.")
code("from pathlib import Path\nimport sys, json, joblib\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import display, Image\nSEED=42\nnp.random.seed(SEED)\nROOT=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()\nif str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))\nprint(ROOT)")
md("4. Data loading", "The validated raw file is loaded from a repository-relative path.")
code("raw=pd.read_csv(ROOT/'data/raw/Maternal Health Risk Data Set.csv')\nprint(raw.shape)\ndisplay(raw.head(), raw.tail())")
md("5. Data-quality checks", "Checks cover schema, types, missing values, duplicates, labels, ranges, and summary statistics.")
code("display(raw.dtypes.to_frame('dtype'))\nprint('Missing:', raw.isna().sum().to_dict())\nprint('Duplicates:', raw.duplicated().sum())\nprint('Classes:', raw.RiskLevel.value_counts().to_dict())\ndisplay(raw.describe())")
md("6. Cleaning", "Labels are stripped and lower-cased. No rows are removed. The HeartRate minimum of 7 appears implausible but is retained because provenance is insufficient for correction.")
code("clean=pd.read_csv(ROOT/'data/processed/maternal_health_risk_cleaned.csv')\nprint('Cleaned shape:', clean.shape, 'Effect on rows:', len(clean)-len(raw))")
md("7. Exploratory analysis", "Plots describe associations only; they do not establish causation.")
code("display(Image(filename=str(ROOT/'reports/figures/class_distribution.png')))\ndisplay(Image(filename=str(ROOT/'reports/figures/feature_histograms.png')))")
code("display(Image(filename=str(ROOT/'reports/figures/feature_boxplots_by_risk.png')))\ndisplay(Image(filename=str(ROOT/'reports/figures/correlation_heatmap.png')))")
md("8. Feature engineering", "Pulse pressure, mean arterial pressure, and documented general age bands are computed without target information.")
code("from src.feature_engineering import add_engineered_features\nengineered=add_engineered_features(clean.drop(columns='RiskLevel'))\ndisplay(engineered.head())")
md("9. Clustering", "K-Means, silhouette comparison, hierarchical clustering, and PCA are exploratory profiles—not diagnoses or replacements for RiskLevel.")
code("display(pd.read_csv(ROOT/'reports/tables/clustering_metrics.csv'))\ndisplay(Image(filename=str(ROOT/'reports/figures/cluster_pca.png')))")
md("10. Data splitting", "StratifiedGroupKFold groups the six-measurement signature. This prevents identical inputs crossing train, validation, and test sets.")
code("display(pd.read_csv(ROOT/'reports/tables/split_distribution.csv'))")
md("11. Baseline models", "DummyClassifier provides a prior-frequency baseline.")
md("12. Traditional models", "Logistic regression, decision tree, random forest, SVC, and gradient boosting are compared.")
md("13. Neural-network models", "Shallow and deeper MLPClassifier models are used because TensorFlow is not reliable for the authorized Python 3.14 environment.")
md("14. Ensemble experiment", "A soft-voting logistic/random-forest/SVC ensemble is retained only as a comparison candidate.")
md("15. Hyperparameter choices", "Search scope is computationally modest: regularized SVC, bounded trees, 250–350 forest trees, and early-stopped MLPs.")
md("16. Model comparison", "Selection prioritizes validation Macro F1, then High Risk recall and probability quality—not accuracy alone.")
code("results=pd.read_csv(ROOT/'reports/model_results.csv')\ndisplay(results)")
md("17. Missing-measurement experiment", "Reproducible training copies have at most one masked feature. Complete raw records remain unchanged. Median, KNN, and native missing handling are compared.")
code("display(pd.read_csv(ROOT/'reports/tables/imputation_comparison.csv'))\ndisplay(pd.read_csv(ROOT/'reports/tables/missing_measurement_results.csv'))")
md("18. Probability calibration", "Sigmoid calibration was tested and rejected because validation log loss worsened.")
code("meta=json.loads((ROOT/'models/model_metadata.json').read_text())\nprint('Uncalibrated:', meta['validation_log_loss_uncalibrated'])\nprint('Calibrated:', meta['validation_log_loss_calibrated'])\nprint('Retained calibration:', meta['calibrated'])")
md("19. Uncertainty threshold", "Validation maximum-probability thresholds compare prediction coverage with error rate. The predicted class remains visible when flagged.")
code("display(pd.read_csv(ROOT/'reports/tables/uncertainty_thresholds.csv'))\nprint('Selected:', meta['uncertainty_threshold'])")
md("20. Explainability", "Median-replacement probability sensitivity is transparent and tested. Direction describes model association, not causation.")
code("from app.app_helpers import predict_result\nmodel=joblib.load(ROOT/'models/maternal_risk_pipeline.joblib')\nexample=dict(meta['feature_medians'])\ndisplay(predict_result(model, meta, example))")
md("21. What-if demonstration", "One selected measurement is adjusted. This demonstrates model sensitivity, not treatment advice.")
code("from app.app_helpers import what_if\ndisplay(what_if(model, meta, example, 'BS', example['BS']+1))")
md("22. Final untouched test evaluation", "The test set was not used for model selection.")
code("display(pd.Series(meta['test_metrics']).to_frame('value'))\ndisplay(pd.DataFrame(meta['classification_report']).T)\nprint(meta['error_analysis'])")
md("23. Model export", "The complete fitted estimator, metadata, label mapping, feature ranges, medians, threshold, package versions, and explanation configuration are saved in `models/`.")
code("print((ROOT/'models/maternal_risk_pipeline.joblib').stat().st_size, (ROOT/'models/model_metadata.json').stat().st_size)")
md("24. Conclusions", "SVC led validation Macro F1. Group-separated test Macro F1 is lower, showing that duplicate-safe evaluation materially changes the generalization estimate.")
md("25. Limitations", "The dataset is small, heavily duplicated, narrow, potentially includes implausible measurements, and lacks external validation and care context.")
md("26. Ethical and practical considerations", "False reassurance is possible. The app stores no submitted measurements. Independent local clinical validation, governance, fairness review, and monitoring are prerequisites for any real-world consideration.")
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.14"}
destination = ROOT / "notebooks" / "MamaCare_End_to_End.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, destination)
print(destination)
