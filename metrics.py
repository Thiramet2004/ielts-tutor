"""Band-score metrics shared by training and evaluation scripts."""
import numpy as np
from sklearn.metrics import cohen_kappa_score


def snap(x, lo=1.0, hi=9.0):
    """Round to the nearest half band and clip to the IELTS scale."""
    return np.clip(np.round(np.asarray(x, dtype=float) * 2) / 2, lo, hi)


def band_metrics(y_true, y_pred):
    """MAE / agreement / QWK on half-band-snapped scores, plus raw Pearson r."""
    raw_t = np.asarray(y_true, dtype=float)
    raw_p = np.asarray(y_pred, dtype=float)
    t, p = snap(raw_t), snap(raw_p)
    diff = np.abs(t - p)
    qwk = cohen_kappa_score((t * 2).astype(int), (p * 2).astype(int), weights="quadratic")
    r = np.corrcoef(raw_t, raw_p)[0, 1] if len(raw_t) > 1 else float("nan")
    return {
        "mae": float(diff.mean()),
        "within_0.5": float((diff <= 0.5).mean()),
        "exact": float((diff == 0).mean()),
        "qwk": float(qwk),
        "pearson": float(r),
    }
