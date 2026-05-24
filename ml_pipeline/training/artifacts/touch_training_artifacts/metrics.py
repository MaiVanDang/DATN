"""metrics.py — AUC, EER, FAR, FRR."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auc(y_true, scores) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return 0.5


def _clean_roc(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    mask = np.isfinite(thresholds)
    return fpr[mask], tpr[mask], thresholds[mask]


def compute_eer(y_true, scores) -> float:
    fpr, tpr, thresholds = _clean_roc(y_true, scores)
    if len(fpr) < 2:
        return 0.5
    fnr = 1.0 - tpr
    diffs = fpr - fnr
    for i in range(len(diffs) - 1):
        if diffs[i] * diffs[i + 1] <= 0:
            d0, d1 = diffs[i], diffs[i + 1]
            if d0 == d1:
                eer = (fpr[i] + fnr[i]) / 2
            else:
                t = d0 / (d0 - d1)
                eer = fpr[i] + t * (fpr[i + 1] - fpr[i])
            return float(eer)
    idx = np.argmin(np.abs(diffs))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_far_frr_at_threshold(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    far = fp / (fp + tn + 1e-10)
    frr = fn / (fn + tp + 1e-10)
    return float(far), float(frr)


def find_eer_threshold(y_true, scores) -> float:
    fpr, tpr, thresholds = _clean_roc(y_true, scores)
    if len(thresholds) < 2:
        return float(np.median(scores)) if len(scores) > 0 else 0.5
    fnr = 1.0 - tpr
    diffs = fpr - fnr
    for i in range(len(diffs) - 1):
        if diffs[i] * diffs[i + 1] <= 0:
            d0, d1 = diffs[i], diffs[i + 1]
            t = 0.5 if d0 == d1 else d0 / (d0 - d1)
            val = thresholds[i] + t * (thresholds[i + 1] - thresholds[i])
            if np.isfinite(val):
                return float(val)
            return float(thresholds[i])
    idx = np.argmin(np.abs(diffs))
    res = float(thresholds[idx])
    return res if np.isfinite(res) else float(np.median(scores))
