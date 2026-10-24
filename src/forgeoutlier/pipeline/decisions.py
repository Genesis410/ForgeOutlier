"""Tri-state decision triage policies."""

from typing import Tuple
from forgeoutlier.core.types import Decision, Policy


def evaluate_decision(
    prob_fake: float,
    ood_score: float,
    threshold: float,
    policy: Policy = Policy.OUTLIER_FIRST,
    clf_threshold: float = 0.50,
) -> Tuple[Decision, str]:
    """Evaluate tri-state decision (REAL, FAKE, UNKNOWN) based on policy.

    Args:
        prob_fake: Probability of manipulation [0.0, 1.0].
        ood_score: Distance or dissimilarity score from reference ID distribution.
        threshold: Calibrated outlier threshold tau.
        policy: Policy.OUTLIER_FIRST or Policy.DUAL_SCORE.
        clf_threshold: Binary classification threshold (default 0.50).

    Returns:
        Tuple of (Decision, reason_string).
    """
    is_outlier = ood_score > threshold

    if policy == Policy.OUTLIER_FIRST:
        # Policy 1: Outlier-First (Default)
        # If is_outlier == True -> UNKNOWN ("OOD_THRESHOLD_EXCEEDED")
        # Else if prob_fake >= clf_threshold -> FAKE ("CONFIDENT_CLASSIFICATION")
        # Else -> REAL ("CONFIDENT_CLASSIFICATION")
        if is_outlier:
            return Decision.UNKNOWN, "OOD_THRESHOLD_EXCEEDED"
        if prob_fake >= clf_threshold:
            return Decision.FAKE, "CONFIDENT_CLASSIFICATION"
        return Decision.REAL, "CONFIDENT_CLASSIFICATION"

    elif policy == Policy.DUAL_SCORE:
        # Policy 2: Dual-Score (Alternative)
        # Allows flagging high-confidence fake even if mildly OOD
        # when prob_fake >= 0.95 and ood_score <= 1.25 * threshold
        if prob_fake >= 0.95 and ood_score <= (1.25 * threshold):
            return Decision.FAKE, "CONFIDENT_CLASSIFICATION"
        if is_outlier:
            return Decision.UNKNOWN, "OOD_THRESHOLD_EXCEEDED"
        if prob_fake >= clf_threshold:
            return Decision.FAKE, "CONFIDENT_CLASSIFICATION"
        return Decision.REAL, "CONFIDENT_CLASSIFICATION"

    elif policy == Policy.BASELINE_NAIVE:
        # Baseline: Naive binary classifier without OOD gating (ignores ood_score)
        if prob_fake >= clf_threshold:
            return Decision.FAKE, "CONFIDENT_CLASSIFICATION"
        return Decision.REAL, "CONFIDENT_CLASSIFICATION"

    else:
        raise ValueError(f"Unknown decision policy: {policy}")
