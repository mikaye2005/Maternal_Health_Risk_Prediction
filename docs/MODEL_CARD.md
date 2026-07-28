# Model Card

## Intended use

Educational demonstration of multiclass machine learning and screening-support
prediction from the UCI Maternal Health Risk dataset.

## Excluded uses

Diagnosis, treatment, emergency triage, clinical decision-making, patient
management, or replacement of professional assessment.

## Training data and features

The data has 1,014 rows (406 low, 336 mid, 272 high), six numerical measurements,
no missing inputs, and 562 exact duplicates. Identical measurement signatures
are group-separated. The selected SVC pipeline uses median imputation,
standardization, PulsePressure, MeanArterialPressure, and AgeBand.

## Evaluation

Untouched test results: Macro F1 0.520, High Risk recall 0.727, accuracy 0.522,
Macro precision 0.541, Macro recall 0.523, and log loss 0.904. Errors included
4 High Risk→Low Risk, 11 High Risk→Mid Risk, and 55 Mid Risk→Low Risk cases.
These results show material limitations, especially Mid Risk discrimination.

## Missing-measurement behavior

Training copies were reproducibly masked with at most one missing feature; the
complete source stayed unchanged. Test Macro F1 by missing feature ranged from
0.473 (BS) to 0.557 (Age), versus 0.520 complete. Median, KNN, and native
histogram-gradient approaches were compared. The deployed pipeline uses median
imputation for a consistent serialized workflow. More than one missing input is
rejected and the interface warns that reliability may be reduced.

## Probability and uncertainty

Sigmoid calibration was tested. It worsened validation log loss from 0.718 to
0.729, so the uncalibrated SVC probability output was retained. Validation
coverage/error comparisons selected a maximum-probability threshold of 0.65.
Below it, the predicted class remains visible but is flagged as uncertain.
Estimated probability is not presented as dependable clinical confidence.

## Explanation

For the predicted class, each measurement is replaced with its training median
and the probability change is measured. The three largest absolute changes are
shown with direction. This is local model sensitivity and association, not
causation.

## Limitations and ethics

The dataset is small, duplicated, narrow, may include implausible values, lacks
care context and key population descriptors, and has no external validation.
Performance may vary across populations and measurement practices. False
reassurance is possible. Submitted measurements are not stored.

Any real-world consideration requires independent local data evaluation,
fairness review, clinical governance, prospective validation, monitoring, and
regulatory assessment.
