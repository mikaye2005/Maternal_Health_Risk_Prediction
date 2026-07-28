# Data Dictionary

| Field | Type | Observed range / values | Description |
|---|---:|---|---|
| Age | numeric | 10–70 | Age in years |
| SystolicBP | numeric | 70–160 | Systolic blood pressure in mmHg |
| DiastolicBP | numeric | 49–100 | Diastolic blood pressure in mmHg |
| BS | numeric | 6–19 | Blood sugar in mmol/L, as documented by UCI |
| BodyTemp | numeric | 98–103 | Body temperature in degrees Fahrenheit |
| HeartRate | numeric | 7–90 | Heart rate in beats per minute |
| RiskLevel | category | low risk, mid risk, high risk | Dataset target label |
| PulsePressure | derived | SystolicBP − DiastolicBP | Engineered pressure difference |
| MeanArterialPressure | derived | (SystolicBP + 2×DiastolicBP) / 3 | Engineered pressure summary |
| AgeBand | derived | ≤19, 20–34, 35–49, ≥50 | General age grouping used by the model |

The source contains no missing values and 562 exact duplicate rows. Some values,
especially HeartRate = 7, appear implausible and are retained because there is
insufficient provenance to correct them safely. The app accepts only observed
ranges; those ranges are dataset bounds, not clinical thresholds.
