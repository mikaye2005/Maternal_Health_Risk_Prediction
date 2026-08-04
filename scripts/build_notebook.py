from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "MamaCare_End_to_End.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}

cells = []
md = lambda text: cells.append(nbf.v4.new_markdown_cell(text.strip()))
code = lambda text: cells.append(nbf.v4.new_code_cell(text.strip()))

md("""
# MamaCare: End-to-End Maternal Health Risk Classification

This notebook documents the complete capstone workflow: problem definition, dataset selection, data-quality checks, exploratory analysis, feature engineering, clustering, model comparison, leakage-resistant evaluation, error analysis, fairness review and responsible-use conclusions.

**Scope:** academic screening-support demonstration only. The system is not a diagnosis, treatment tool, triage system or emergency assessment.
""")

md("""
## 1. Problem Statement and Evaluation Plan

MamaCare classifies records represented in the UCI Maternal Health Risk dataset into **Low Risk**, **Mid Risk** or **High Risk** using age, systolic blood pressure, diastolic blood pressure, blood sugar, body temperature and heart rate.

- **Target variable:** `RiskLevel`
- **Population represented:** maternal-health records documented by UCI as collected through hospitals, community clinics and maternal-health facilities in rural Bangladesh
- **Primary metric:** Weighted F1 on an untouched group-separated test set
- **Secondary metrics:** Macro F1, High Risk recall, accuracy and log loss
- **Fairness criterion:** Equal Opportunity for High Risk identification across age groups

Dataset citation: Ahmed, M. (2020). *Maternal Health Risk*. UCI Machine Learning Repository. DOI: 10.24432/C5DP5D.
""")

code("""
from pathlib import Path
import base64
import html
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML, display

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

DATA = ROOT / "data" / "raw" / "Maternal Health Risk Data Set.csv"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda value: f"{value:.4f}")
sns.set_theme(style="whitegrid")

def show_figure(path, alt_text):
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    safe_alt = html.escape(alt_text, quote=True)
    display(HTML(
        f'<img src="data:image/png;base64,{encoded}" alt="{safe_alt}" '
        'style="max-width:100%;height:auto;" />'
    ))
""")

md("""
## 2. Load and Inspect the Dataset

The raw source file is preserved in `data/raw`. The training script validates the column order, converts the six inputs to numeric values and normalizes the three target labels without changing the source measurements.
""")

code("""
df = pd.read_csv(DATA)
print("Shape:", df.shape)
print()
print("Columns:", df.columns.tolist())
print()
print("Data types:")
display(df.dtypes.to_frame("dtype"))
display(df.head())
""")

md("""
### Dataset structure

The downloaded CSV contains 1,014 rows and seven columns. UCI's metadata page reports 1,013 instances, so this project records the discrepancy and preserves the exact file checksum in `models/model_metadata.json`.
""")

code("""
summary = df.describe(include="all").T
summary
""")

md("""
## 3. Data Quality Audit

The source has no missing fields, but duplicates and contradictory labels are substantial. These issues are not removed silently because there is insufficient provenance to decide which copy or label is correct. Instead, the project prevents identical measurement signatures from crossing data partitions and reports the resulting limitations.
""")

code("""
missing = df.isna().sum().to_frame("missing_values")
missing["missing_percent"] = missing["missing_values"] / len(df) * 100
print("Exact duplicate rows:", int(df.duplicated().sum()))
display(missing)
""")

code("""
features = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
conflicts = pd.read_csv(TABLES / "contradictory_measurement_signatures.csv")
print("Unique measurement signatures:", df[features].drop_duplicates().shape[0])
print("Signatures with more than one target label:", len(conflicts))
print("Rows covered by conflicting signatures:", int(conflicts["record_count"].sum()))
display(conflicts.head(10))
""")

md("""
### Interpretation of the contradictory labels

A deterministic classifier receives the same six values for records that sometimes have different target labels. It therefore cannot perfectly separate those records. This is one reason the group-separated evaluation is much lower and more credible than a random row split that allows duplicate copies to appear in both training and testing data.
""")

code("""
show_figure(FIGURES / "missing_data_heatmap.png", "Heatmap of missing values in the six model inputs and target")
""")

md("""
## 4. Exploratory Data Analysis

The target distribution is moderately imbalanced rather than dominated by a single class. Weighted F1 is therefore used as the capstone primary metric, while Macro F1 and class-specific recall remain important checks.
""")

code("""
target_counts = df["RiskLevel"].value_counts().rename_axis("RiskLevel").to_frame("records")
target_counts["percentage"] = target_counts["records"] / len(df) * 100
display(target_counts)
show_figure(FIGURES / "class_distribution_annotated.png", "Annotated counts for Low Risk, Mid Risk and High Risk records")
""")

md("""
### Numerical distributions and outliers

The plots show repeated values and narrow measurement ranges. Some values appear implausible, especially a minimum heart rate of 7 beats per minute. The value is retained because the project has no source documentation that justifies correcting it. Application ranges are therefore dataset bounds, not clinical definitions of normal health.
""")

code("""
show_figure(FIGURES / "feature_histograms.png", "Histograms of the six maternal-health measurements")
""")

code("""
show_figure(FIGURES / "feature_boxplots_by_risk.png", "Feature boxplots grouped by recorded risk class")
""")

md("""
### Relationships among measurements

Blood-pressure variables are positively related, while the remaining correlations are weaker. Correlation alone does not prove that a variable is clinically causal or sufficient for risk prediction.
""")

code("""
show_figure(FIGURES / "correlation_heatmap.png", "Correlation heatmap for the numeric source measurements")
""")

code("""
show_figure(FIGURES / "blood_pressure_scatter.png", "Systolic versus diastolic blood pressure coloured by recorded risk class")
""")

md("""
## 5. Feature Engineering

Three derived variables are created inside the model pipeline so that the same transformation is applied during training and prediction:

- `PulsePressure = SystolicBP - DiastolicBP`
- `MeanArterialPressure = (SystolicBP + 2 x DiastolicBP) / 3`
- `AgeBand` groups ages as `<=19`, `20-34`, `35-49` and `>=50`

The ablation experiment compares the selected model with and without these derived features.

The original proposal also mentioned blood-pressure flags and glucose-risk bands. Those threshold features were excluded because the project has no clinically reviewed rule set that can be assumed to transfer safely from the represented rural-Bangladesh records to a Kenyan audience. The application therefore treats observed ranges as dataset bounds, not clinical thresholds.
""")

code("""
ablation = pd.read_csv(TABLES / "feature_ablation.csv")
display(ablation)
show_figure(FIGURES / "feature_ablation.png", "Validation Weighted F1 with and without engineered features")
""")

md("""
### Feature-engineering finding

For the selected shallow neural network, the engineered feature set improved validation Weighted F1, Macro F1 and High Risk recall. This supports retaining the derived variables, while still treating them as modelling features rather than medical rules.
""")

md("""
## 6. Exploratory Clustering

K-Means is evaluated for two to six clusters using silhouette score and Davies-Bouldin index. Hierarchical clustering is fitted with the selected number of clusters and evaluated with the same separation measures. Cluster labels describe mathematical profiles; they are not diagnoses or patient categories for clinical use.
""")

code("""
cluster_metrics = pd.read_csv(TABLES / "clustering_metrics.csv")
cluster_profiles = pd.read_csv(TABLES / "cluster_profiles.csv")
cluster_risk = pd.read_csv(TABLES / "cluster_risk_distribution.csv")
display(cluster_metrics)
display(cluster_profiles)
display(cluster_risk)
""")

code("""
show_figure(FIGURES / "kmeans_silhouette.png", "K-Means silhouette score by cluster count")
show_figure(FIGURES / "cluster_pca.png", "Two-dimensional PCA projection of K-Means and hierarchical cluster assignments")
show_figure(FIGURES / "cluster_profile_heatmap.png", "Standardized feature profiles for the selected descriptive clusters")
""")

md("""
### Cluster interpretation

The four-cluster solution has the strongest silhouette score, but separation is only moderate. One profile has high median blood sugar and a large proportion of High Risk labels, while another has lower blood-pressure measurements and mostly Low Risk labels. These associations are exploratory and should not be treated as clinical subtypes.
""")

md("""
## 7. Leakage-Resistant Train, Validation and Test Splits

The dataset contains many duplicates. A normal random row split could place one copy in training and another identical copy in testing, inflating performance. The project hashes the six input measurements and uses `StratifiedGroupKFold` so each measurement signature stays in only one partition.
""")

code("""
train = pd.read_csv(TABLES / "train_split.csv")
validation = pd.read_csv(TABLES / "validation_split.csv")
test = pd.read_csv(TABLES / "test_split.csv")
split_distribution = pd.read_csv(TABLES / "split_distribution.csv")
display(split_distribution)

signature = lambda frame: set(pd.util.hash_pandas_object(frame[features], index=False).astype(str))
print("Train-validation overlap:", len(signature(train) & signature(validation)))
print("Train-test overlap:", len(signature(train) & signature(test)))
print("Validation-test overlap:", len(signature(validation) & signature(test)))
""")

md("""
## 8. Model Development and Selection

The comparison includes a no-learning majority baseline, Logistic Regression, Decision Tree, Random Forest, SVC, Gradient Boosting, XGBoost, shallow and deep feed-forward neural networks, and a soft-voting ensemble. Models are ranked by validation Weighted F1 before the untouched test set is used.
""")

code("""
model_comparison = pd.read_csv(TABLES / "model_comparison_validation.csv")
model_comparison = model_comparison.sort_values("weighted_f1", ascending=False, na_position="last")
display(model_comparison)
show_figure(FIGURES / "model_comparison_weighted_f1.png", "Validation Weighted F1 for all ten candidate models")
""")

md("""
### Neural-network training

Two scikit-learn feed-forward architectures were tested. Both use ReLU activation, L2 regularization and early stopping. The shallow network performed better than the deeper architecture, showing that additional depth did not improve this small dataset.

These MLPs are a transparent approximation of the proposal's Keras experiment. They do not implement dropout or batch normalization. The final scope uses the lighter comparison because the dataset is small and the deeper network did not improve validation Weighted F1; this limitation is recorded in `docs/SCOPE_DECISIONS.md`.
""")

code("""
network_history = pd.read_csv(TABLES / "neural_network_history.csv")
print(network_history.groupby("model")["epoch"].max())
show_figure(FIGURES / "neural_network_training_curves.png", "Training loss and group-separated internal validation accuracy for shallow and deep MLPs")
""")

md("""
## 9. Baseline Comparison and Final Test Results

The selected shallow neural network is retrained on the combined train and validation partitions, then evaluated once on the untouched test set. The majority-class baseline provides the no-learning comparison required by the capstone rubric.
""")

code("""
baseline = pd.read_csv(TABLES / "baseline_vs_model.csv")
class_report = pd.read_csv(TABLES / "class_report_test.csv")
metadata = json.loads((MODELS / "model_metadata.json").read_text(encoding="utf-8"))
display(baseline)
display(class_report)
print("Selected model:", metadata["model_name"])
print("Test metrics:", metadata["test_metrics"])
show_figure(FIGURES / "baseline_vs_model.png", "Untouched-test comparison of the majority baseline and selected model")
""")

md("""
### Final interpretation

The selected model clearly beats the majority baseline. High Risk recall is materially stronger than Mid Risk recall. The model therefore demonstrates useful signal but remains unsuitable for clinical deployment, especially because Mid Risk records are frequently assigned to another class.
""")

code("""
show_figure(FIGURES / "test_confusion_matrix.png", "Confusion matrix for the selected model on the untouched test split")
""")

md("""
## 10. Disaggregated Evaluation and Fairness

The chosen fairness criterion is **Equal Opportunity for High Risk identification across age groups**. This asks whether actual High Risk records are identified at similar rates for different age groups.
""")

code("""
disaggregated = pd.read_csv(TABLES / "disaggregated_age_evaluation.csv")
equal_opportunity = pd.read_csv(TABLES / "equal_opportunity_summary.csv")
display(disaggregated)
display(equal_opportunity)
show_figure(FIGURES / "age_group_high_risk_recall.png", "High Risk recall and record count by age group")
""")

md("""
### Fairness finding

High Risk recall differs substantially by age group. The `<=19` subgroup performs markedly worse than the older groups in this split. The subgroup sample counts are limited, so the estimates are uncertain, but the gap is too large to claim equal performance. A real deployment would require more representative local data and prospective review.
""")

md("""
## 11. Structured Error Analysis

The analysis below examines the most confidently wrong predictions, errors by age group and performance on records whose measurement signatures have conflicting labels.
""")

code("""
top_errors = pd.read_csv(TABLES / "top_confident_errors.csv")
error_types = pd.read_csv(TABLES / "error_type_summary.csv")
age_errors = pd.read_csv(TABLES / "error_rate_by_age.csv")
conflict_performance = pd.read_csv(TABLES / "conflicting_signature_performance.csv")

display(error_types)
display(age_errors)
display(conflict_performance)
display(top_errors.head(10))
""")

md("""
### Three required error-analysis questions

**Are the errors random?**  
No. They concentrate in Mid Risk records, younger records and signatures with contradictory target labels.

**Are the errors concentrated?**  
Yes. Mid Risk recall is low, and records belonging to contradictory signatures have a much higher error rate than records with consistent labels. The age-group table also shows weaker performance for the youngest group.

**Are the errors consequential?**  
Yes. The most serious error is an actual High Risk record predicted as Low Risk because it could create false reassurance if the demonstration were misused. False positive High Risk outputs can also create anxiety, but the project gives priority to identifying actual High Risk cases.
""")

md("""
## 12. Probability Calibration and Uncertainty

Probability calibration is assessed with group-separated data and is retained only when it improves external-validation log loss. The metadata output below records the actual decision from the current run. The interface uses a validation-selected review threshold and describes displayed values as model scores rather than medically validated probabilities.
""")

code("""
print(json.dumps(metadata["calibration_check"], indent=2))
thresholds = pd.read_csv(TABLES / "uncertainty_thresholds.csv")
display(thresholds)
print("Selected review threshold:", metadata["uncertainty_threshold"])
""")

md("""
## 13. Responsible Use

The model affects people whose records are classified, people interpreting the output and groups that were not represented in the design process. The worst plausible outcome is false reassurance after a missed High Risk case. The person whose record is misclassified bears the main cost.

There is no formal redress mechanism because this is an academic prototype. Any real deployment would require human review, a correction or appeal route, monitoring, local clinical validation and clear documentation of overrides.

The full 500-700 word statement is stored in `reports/responsible_ai_statement.pdf`.
""")

md("""
## 14. Conclusion

MamaCare meets the capstone objective of delivering a complete, reproducible classification project with a baseline, multiple models, XGBoost, two neural-network architectures, feature engineering, clustering, error analysis, disaggregated evaluation, a Responsible AI statement, tests and a working demonstration interface.

The strongest technical contribution is the measurement-signature group split, which prevents duplicate leakage and exposes the dataset's real limitations. The selected shallow neural network beats the majority baseline and identifies most High Risk records, but Mid Risk performance and age-group gaps prevent any clinical claim.

**Next phase:** collect richer, locally validated maternal-health data with facility context, pregnancy stage and clinically reviewed outcomes, then repeat the fairness, calibration and prospective validation process.
""")

nb["cells"] = cells
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
