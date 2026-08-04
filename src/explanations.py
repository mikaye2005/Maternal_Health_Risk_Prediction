from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FEATURES


def local_median_replacement_explanation(model, row: pd.DataFrame, reference: dict, class_index=None):
    base = model.predict_proba(row)[0]
    class_index = int(np.argmax(base)) if class_index is None else int(class_index)
    effects = []
    for feature in FEATURES:
        altered = row.copy()
        altered.loc[altered.index[0], feature] = reference[feature]
        changed = model.predict_proba(altered)[0, class_index]
        delta = float(base[class_index] - changed)
        effects.append({
            "feature": feature,
            "importance": abs(delta),
            "delta": delta,
            "direction": "increased the model score" if delta >= 0 else "reduced the model score",
        })
    return sorted(effects, key=lambda item: item["importance"], reverse=True)[:3]
