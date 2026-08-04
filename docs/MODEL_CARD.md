# MamaCare Model Card

## Model summary
- **Selected model:** ShallowMLP
- **Task:** three-class maternal-health risk classification
- **Primary metric:** Weighted F1
- **Model version:** 3.0.0
- **Training data:** 1014 rows from the downloaded UCI CSV

## Intended use
MamaCare is an educational demonstration of multiclass machine learning and maternal-health screening-support. It is suitable for classroom explanation, portfolio review and controlled capstone demonstration.

## Excluded uses
The system must not be used for diagnosis, treatment, emergency triage, clinical decision-making, patient management, automatic referral, denial of care or replacement of professional assessment.

## Inputs and preprocessing
The model accepts Age, SystolicBP, DiastolicBP, BS, BodyTemp and HeartRate. The pipeline performs median imputation, feature engineering and standardization before classification. Engineered features are PulsePressure, MeanArterialPressure and AgeBand.

Blood-pressure risk flags and glucose-risk bands proposed in the early project plan were not retained because no clinically reviewed threshold specification was available for this dataset. Dataset bounds must not be interpreted as medical thresholds.

## Neural-network scope
The final comparison uses scikit-learn `MLPClassifier` models with shallow `(32,)` and deeper `(64, 32, 16)` hidden layers, ReLU activation, L2 regularisation and early stopping. They provide the required shallow-versus-deep experiment but are not the proposal's Keras architectures: dropout and batch normalisation are not implemented. The decision is documented in `docs/SCOPE_DECISIONS.md`.

## Data quality
The downloaded file contains 1014 rows. UCI metadata reports 1,013 instances. The project reports 562 exact duplicate rows and 35 identical measurement signatures with conflicting labels, covering 215 rows. These contradictions place a hard limit on learnable performance.

## Leakage prevention
Identical six-measurement signatures are grouped using `StratifiedGroupKFold`. The saved split audit reports zero overlap between train, validation and test signatures.

## Evaluation
Untouched test results:
- Weighted F1: 0.558
- Macro F1: 0.571
- Accuracy: 0.596
- High Risk recall: 0.818
- Mid Risk recall: 0.149
- Log loss: 0.773

The model beats the majority baseline but remains weak for Mid Risk discrimination. Performance is not equivalent across age groups.

## Missing measurements
The interface accepts at most one unavailable measurement. Training copies are reproducibly masked, while the original source data remains unchanged. Two or more missing measurements are rejected.

## Probability and uncertainty
The interface reports model scores, not medically validated probabilities. A validation-selected threshold of 0.80 flags outputs for review. Calibration was tested on a separate group-partitioned subset and not retained because validation log loss did not improve.

## Explanation
Local explanations replace one measurement at a time with its training median and measure the change in the predicted-class score. This is a sensitivity analysis, not causation or treatment effect.

## Fairness
The project applies Equal Opportunity for High Risk identification across age groups. The observed gap is 0.580. The youngest age group has materially weaker High Risk recall in the current test split.

## Monitoring requirements for any future study
A future clinical study would need local validation, prospective monitoring, human review, data-protection governance, override documentation and a redress mechanism. High Risk recall and subgroup recall should be monitored before overall accuracy.
