import numpy as np


def is_uncertain(probabilities, threshold: float) -> bool:
    return float(np.max(probabilities)) < float(threshold)


def select_threshold(probabilities, truth, predictions, candidates=None):
    candidates = candidates if candidates is not None else np.arange(0.40, 0.86, 0.05)
    rows = []
    confidence = np.max(probabilities, axis=1)
    correct = np.asarray(predictions) == np.asarray(truth)
    for threshold in candidates:
        covered = confidence >= threshold
        coverage = float(covered.mean())
        error_rate = float((~correct[covered]).mean()) if covered.any() else np.nan
        rows.append({"threshold": round(float(threshold), 2), "coverage": coverage,
                     "error_rate": error_rate})
    eligible = [r for r in rows if r["coverage"] >= 0.55 and r["error_rate"] <= 0.20]
    selected = min(eligible, key=lambda r: r["threshold"]) if eligible else min(
        rows, key=lambda r: (np.nan_to_num(r["error_rate"], nan=1.0), -r["coverage"])
    )
    return selected["threshold"], rows
