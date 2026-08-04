# Data Dictionary

| Field | Type | Observed range / values | Description |
|---|---:|---|---|
| Age | numeric | 10-70 | Age in years |
| SystolicBP | numeric | 70-160 | Systolic blood pressure in mmHg |
| DiastolicBP | numeric | 49-100 | Diastolic blood pressure in mmHg |
| BS | numeric | 6-19 | Blood sugar in mmol/L as documented by UCI |
| BodyTemp | numeric | 98-103 | Body temperature in degrees Fahrenheit |
| HeartRate | numeric | 7-90 | Heart rate in beats per minute |
| RiskLevel | category | low risk, mid risk, high risk | Target label |
| PulsePressure | derived | SystolicBP - DiastolicBP | Engineered blood-pressure difference |
| MeanArterialPressure | derived | (SystolicBP + 2 x DiastolicBP) / 3 | Engineered blood-pressure summary |
| AgeBand | derived | <=19, 20-34, 35-49, >=50 | Age subgroup used for evaluation and feature engineering |

The source file contains no missing values, 562 exact duplicate rows and 35 measurement signatures with conflicting labels. Some values, especially HeartRate = 7, appear implausible and are retained because there is insufficient provenance to correct them safely. Application input ranges are dataset bounds, not clinical reference ranges.
