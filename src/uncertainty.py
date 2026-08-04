from __future__ import annotations

import numpy as np


def is_uncertain(probabilities, threshold: float) -> bool:
    return float(np.max(probabilities)) < float(threshold)


def select_threshold(probabilities, truth, predictions, candidates=None):
    """Select a practical confidence threshold using validation coverage and error."""

    candidates = candidates if candidates is not None else np.arange(0.40, 0.86, 0.05)
    confidence = np.max(probabilities, axis=1)
    correct = np.asarray(predictions) == np.asarray(truth)
    overall_error = float((~correct).mean())
    rows = []
    for threshold in candidates:
        covered = confidence >= threshold
        coverage = float(covered.mean())
        error_rate = float((~correct[covered]).mean()) if covered.any() else np.nan
        rows.append({
            "threshold": round(float(threshold), 2),
            "coverage": coverage,
            "error_rate": error_rate,
            "overall_error_rate": overall_error,
        })

    # Prefer the lowest threshold that retains at least half the cases and
    # reduces validation error by at least three percentage points.
    eligible = [
        row for row in rows
        if row["coverage"] >= 0.50
        and np.isfinite(row["error_rate"])
        and row["error_rate"] <= overall_error - 0.03
    ]
    if eligible:
        selected = min(eligible, key=lambda row: row["threshold"])
    else:
        selected = min(
            rows,
            key=lambda row: (
                np.nan_to_num(row["error_rate"], nan=1.0) + 0.15 * (1 - row["coverage"]),
                row["threshold"],
            ),
        )
    return selected["threshold"], rows
