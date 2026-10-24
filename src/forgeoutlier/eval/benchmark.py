"""Evaluation benchmark runner for ID and OOD evaluation."""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import numpy as np

from forgeoutlier.pipeline.inference import InferencePipeline
from forgeoutlier.eval.metrics import (
    compute_auprc,
    compute_auroc,
    compute_fpr_at_tpr,
    compute_tristate_confusion_matrix,
    compute_risk_coverage,
    compute_ece,
)


@dataclass
class BenchmarkSummary:
    """Summary of benchmark run results."""

    total_samples: int
    id_count: int
    ood_count: int
    auroc: float
    auprc: float
    fpr_at_95_tpr: float
    confusion_matrix: Dict[str, Dict[str, int]]
    fatal_error_rate_ood: float  # Fraction of OOD samples misclassified as REAL
    abstained_rate_ood: float    # Fraction of OOD samples routed to UNKNOWN
    aurc: float = 0.0            # Area under risk-coverage curve
    ece: float = 0.0             # Expected Calibration Error

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_dataset(
    pipeline: InferencePipeline,
    image_paths: List[str],
    labels: List[str],
) -> BenchmarkSummary:
    """Evaluate pipeline across labeled dataset of images.

    Args:
        pipeline: Initialized InferencePipeline.
        image_paths: List of absolute or relative image paths.
        labels: List of ground-truth strings ('REAL', 'FAKE', or 'OOD').

    Returns:
        BenchmarkSummary containing performance metrics.
    """
    if len(image_paths) != len(labels):
        raise ValueError("image_paths and labels must have identical length.")

    scores_ood = []
    probs_fake = []
    y_true_binary_ood = []  # 1 for OOD, 0 for ID (REAL/FAKE)
    decisions = []

    for path, gt in zip(image_paths, labels):
        res = pipeline.predict_image(path)
        gt_norm = gt.upper()
        decisions.append(res.decision.value)
        scores_ood.append(res.scores.ood_score)
        probs_fake.append(res.scores.prob_fake)

        if gt_norm in ("OOD", "UNKNOWN"):
            y_true_binary_ood.append(1)
        else:
            y_true_binary_ood.append(0)

    y_arr = np.array(y_true_binary_ood, dtype=int)
    s_arr = np.array(scores_ood, dtype=float)

    id_count = int(np.sum(y_arr == 0))
    ood_count = int(np.sum(y_arr == 1))

    # If dataset has both ID and OOD, compute curve metrics; otherwise NaN
    if id_count > 0 and ood_count > 0:
        auroc = compute_auroc(y_arr, s_arr)
        auprc = compute_auprc(y_arr, s_arr)
        fpr95 = compute_fpr_at_tpr(y_arr, s_arr, target_tpr=0.95)
    else:
        auroc = float("nan")
        auprc = float("nan")
        fpr95 = float("nan")

    conf_mat = compute_tristate_confusion_matrix(labels, decisions)

    # Compute open-set triage metrics
    total_ood = sum(conf_mat["OOD"].values())
    if total_ood > 0:
        fatal_rate = conf_mat["OOD"]["REAL"] / total_ood
        abstained_rate = conf_mat["OOD"]["UNKNOWN"] / total_ood
    else:
        fatal_rate = 0.0
        abstained_rate = 0.0

    # Compute selective classification risk-coverage curve
    # Sample is considered an error if decision is incorrect on ID, or if OOD is accepted as REAL/FAKE
    errors = []
    confidences = []
    for gt, dec, s_ood in zip(labels, decisions, scores_ood):
        gt_norm = str(gt).upper()
        dec_norm = str(dec).upper()
        if dec_norm == "UNKNOWN":
            continue
        confidences.append(-s_ood)  # lower OOD distance = higher confidence
        if gt_norm in ("OOD", "UNKNOWN"):
            errors.append(1)  # OOD accepted as ID is fatal error
        elif gt_norm != dec_norm:
            errors.append(1)  # Misclassification within ID
        else:
            errors.append(0)

    if len(errors) > 0:
        _, _, aurc_val = compute_risk_coverage(np.array(errors), np.array(confidences))
    else:
        aurc_val = 0.0

    # Compute ECE on ID samples
    id_gt_binary = []
    id_probs_fake = []
    for gt, pf in zip(labels, probs_fake):
        gt_norm = str(gt).upper()
        if gt_norm in ("REAL", "FAKE"):
            id_gt_binary.append(1 if gt_norm == "FAKE" else 0)
            id_probs_fake.append(pf)

    if len(id_gt_binary) > 0:
        ece_val = compute_ece(np.array(id_gt_binary), np.array(id_probs_fake))
    else:
        ece_val = 0.0

    return BenchmarkSummary(
        total_samples=len(image_paths),
        id_count=id_count,
        ood_count=ood_count,
        auroc=round(auroc, 4) if not np.isnan(auroc) else float("nan"),
        auprc=round(auprc, 4) if not np.isnan(auprc) else float("nan"),
        fpr_at_95_tpr=round(fpr95, 4) if not np.isnan(fpr95) else float("nan"),
        confusion_matrix=conf_mat,
        fatal_error_rate_ood=round(fatal_rate, 4),
        abstained_rate_ood=round(abstained_rate, 4),
        aurc=round(aurc_val, 4),
        ece=round(ece_val, 4),
    )
