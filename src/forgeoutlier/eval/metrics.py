"""Open-set and Out-of-Distribution evaluation metrics.

Provides pure-NumPy implementations for AUROC, AUPRC, FPR@95% TPR,
and Tri-State Confusion Matrix for open-set triage evaluation.
"""

from typing import Dict, List, Tuple, Union
import numpy as np


def compute_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true: 1D binary array of ground truth (0 = Negative/ID, 1 = Positive/OOD).
        scores: 1D array of anomaly or classification scores (higher = more positive).

    Returns:
        (fpr, tpr, thresholds) sorted by ascending FPR.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(y_true)):
        raise ValueError("Non-finite values (NaN or Inf) detected in ROC inputs.")

    if y_true.ndim != 1 or scores.ndim != 1:
        raise ValueError("Inputs y_true and scores must be 1D arrays.")
    if len(y_true) != len(scores):
        raise ValueError("Length of y_true and scores must match.")

    pos_count = int(np.sum(y_true == 1))
    neg_count = int(np.sum(y_true == 0))
    if pos_count == 0 or neg_count == 0:
        raise ValueError("y_true must contain both positive (1) and negative (0) samples.")

    # Sort descending by score
    desc_indices = np.argsort(-scores)
    y_sorted = y_true[desc_indices]
    scores_sorted = scores[desc_indices]

    # Find distinct threshold boundaries
    distinct_mask = np.diff(scores_sorted) != 0
    distinct_indices = np.where(distinct_mask)[0]
    threshold_indices = np.r_[distinct_indices, len(y_sorted) - 1]

    tps = np.cumsum(y_sorted == 1)[threshold_indices]
    fps = np.cumsum(y_sorted == 0)[threshold_indices]

    tpr = np.r_[0.0, tps / pos_count]
    fpr = np.r_[0.0, fps / neg_count]
    thresholds = np.r_[scores_sorted[0] + 1.0, scores_sorted[threshold_indices]]

    return fpr, tpr, thresholds


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under Receiver Operating Characteristic Curve (AUROC).

    Uses the trapezoidal rule over (fpr, tpr).
    """
    fpr, tpr, _ = compute_roc_curve(y_true, scores)
    return float(np.trapezoid(tpr, fpr))


def compute_pr_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Precision-Recall curve."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(y_true)):
        raise ValueError("Non-finite values (NaN or Inf) detected in PR inputs.")

    pos_count = int(np.sum(y_true == 1))
    if pos_count == 0:
        raise ValueError("y_true must contain at least one positive (1) sample.")

    desc_indices = np.argsort(-scores)
    y_sorted = y_true[desc_indices]
    scores_sorted = scores[desc_indices]

    distinct_mask = np.diff(scores_sorted) != 0
    threshold_indices = np.r_[np.where(distinct_mask)[0], len(y_sorted) - 1]

    tps = np.cumsum(y_sorted == 1)[threshold_indices]
    fps = np.cumsum(y_sorted == 0)[threshold_indices]

    precision = tps / (tps + fps)
    recall = tps / pos_count

    # Anchor at recall 0, precision 1
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    thresholds = scores_sorted[threshold_indices]

    return precision, recall, thresholds


def compute_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under Precision-Recall Curve (Average Precision) using step-wise integration.

    Matches sklearn.metrics.average_precision_score calculation standard.
    Formula: AP = sum_n (R_n - R_{n-1}) * P_n
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(y_true)):
        raise ValueError("Non-finite values (NaN or Inf) detected in AUPRC inputs.")

    pos_count = int(np.sum(y_true == 1))
    if pos_count == 0:
        raise ValueError("y_true must contain at least one positive (1) sample.")

    # Sort scores descending
    desc_indices = np.argsort(-scores)
    y_sorted = y_true[desc_indices]
    scores_sorted = scores[desc_indices]

    distinct_mask = np.diff(scores_sorted) != 0
    threshold_indices = np.r_[np.where(distinct_mask)[0], len(y_sorted) - 1]

    tps = np.cumsum(y_sorted == 1)[threshold_indices]
    fps = np.cumsum(y_sorted == 0)[threshold_indices]

    precision = tps / (tps + fps)
    recall = tps / pos_count

    # Prepend recall 0.0 to compute step differences (R_n - R_{n-1})
    recall_diffs = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_diffs * precision))


def compute_risk_coverage(
    risk_errors: np.ndarray,
    confidence_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute Selective Classification Risk-Coverage Curve and AURC.

    Args:
        risk_errors: 1D binary array where 1 = error/misclassification, 0 = correct.
        confidence_scores: 1D array of confidence/certainty scores (higher = more certain to keep).

    Returns:
        (coverages, risks, aurc):
            coverages: fractions of accepted samples in [0, 1].
            risks: empirical error rate on accepted samples.
            aurc: Area Under Risk-Coverage curve.
    """
    risk_errors = np.asarray(risk_errors, dtype=int)
    confidence_scores = np.asarray(confidence_scores, dtype=float)

    if len(risk_errors) != len(confidence_scores):
        raise ValueError("risk_errors and confidence_scores must have identical length.")
    if len(risk_errors) == 0:
        return np.array([0.0]), np.array([0.0]), 0.0

    # Sort descending by confidence (keep most confident first)
    desc_idx = np.argsort(-confidence_scores)
    err_sorted = risk_errors[desc_idx]

    n = len(err_sorted)
    coverages = np.arange(1, n + 1) / n
    cumulative_errors = np.cumsum(err_sorted)
    risks = cumulative_errors / np.arange(1, n + 1)

    # Area under risk-coverage curve (trapezoidal integration over coverage)
    aurc = float(np.trapezoid(risks, coverages))
    return coverages, risks, aurc


def compute_fpr_at_tpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """Compute False Positive Rate at target True Positive Rate (e.g. FPR@95% TPR).

    In OOD detection literature:
    When OOD is positive (1) and ID is negative (0), this measures the fraction of
    ID samples misclassified as OOD when 95% of OOD samples are successfully caught.
    """
    fpr, tpr, _ = compute_roc_curve(y_true, scores)
    # Find the smallest FPR where TPR >= target_tpr
    valid_indices = np.where(tpr >= target_tpr)[0]
    if len(valid_indices) == 0:
        return 1.0
    return float(fpr[valid_indices[0]])


def compute_ece(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) pure NumPy.

    ECE = sum_{b=1}^B (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(probs, dtype=float).ravel()
    n = len(y_t)
    if n == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        b_low, b_high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= b_low) & (p <= b_high)
        else:
            mask = (p >= b_low) & (p < b_high)

        count = np.sum(mask)
        if count > 0:
            bin_acc = np.mean(y_t[mask])
            bin_conf = np.mean(p[mask])
            ece += (count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_tristate_confusion_matrix(
    y_true_labels: List[str],
    decisions: List[str],
) -> Dict[str, Dict[str, int]]:
    """Compute 2x3 Tri-State Confusion Matrix.

    Rows: Ground Truth ('REAL', 'FAKE', or 'OOD' / 'UNKNOWN')
    Columns: Predicted Decision ('REAL', 'FAKE', 'UNKNOWN')

    Returns:
        dict with counts, e.g.:
        {
            "REAL": {"REAL": c1, "FAKE": c2, "UNKNOWN": c3},
            "FAKE": {"REAL": c4, "FAKE": c5, "UNKNOWN": c6},
            "OOD":  {"REAL": c7, "FAKE": c8, "UNKNOWN": c9},
        }
    """
    matrix: Dict[str, Dict[str, int]] = {
        "REAL": {"REAL": 0, "FAKE": 0, "UNKNOWN": 0},
        "FAKE": {"REAL": 0, "FAKE": 0, "UNKNOWN": 0},
        "OOD": {"REAL": 0, "FAKE": 0, "UNKNOWN": 0},
    }

    for gt, pred in zip(y_true_labels, decisions):
        gt_norm = str(gt).upper()
        pred_norm = str(pred).upper()
        if gt_norm not in matrix:
            # Map UNKNOWN ground truth to OOD
            if gt_norm == "UNKNOWN":
                gt_norm = "OOD"
            else:
                continue
        if pred_norm not in matrix[gt_norm]:
            continue
        matrix[gt_norm][pred_norm] += 1

    return matrix
