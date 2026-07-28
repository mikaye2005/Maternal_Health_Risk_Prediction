# Testing Report

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result: **11 passed in 3.82 seconds**.

Covered behavior:

- source schema, shape, completeness, and target labels;
- PulsePressure, MeanArterialPressure, and AgeBand formulas;
- serialized model loading, class output, three probabilities, and probability sum;
- label mapping and uncertainty-threshold bounds;
- complete input and deterministic repeat prediction;
- one missing input accepted and two rejected;
- negative and out-of-range input rejection;
- three local influences and what-if recalculation;
- uncertain/above-threshold boundary behavior.

There were 32 joblib/NumPy deprecation warnings while loading artifacts. They do
not affect the current results but should be rechecked when dependency versions
are updated.
