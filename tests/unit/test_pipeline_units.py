"""Unit tests for Phase 4: Tri-state triage policies and video aggregation."""

import pytest
from forgeoutlier.core.types import Decision, Policy, VideoSummary
from forgeoutlier.core.exceptions import InsufficientDataError
from forgeoutlier.pipeline.decisions import evaluate_decision
from forgeoutlier.pipeline.aggregation import aggregate_video_frames


def test_decision_outlier_first() -> None:
    # 1. Outlier detected -> UNKNOWN regardless of fake prob
    dec, reason = evaluate_decision(
        prob_fake=0.99,
        ood_score=1.5,
        threshold=1.0,
        policy=Policy.OUTLIER_FIRST,
    )
    assert dec == Decision.UNKNOWN
    assert reason == "OOD_THRESHOLD_EXCEEDED"

    # 2. In-distribution, fake prob >= 0.50 -> FAKE
    dec, reason = evaluate_decision(
        prob_fake=0.85,
        ood_score=0.8,
        threshold=1.0,
        policy=Policy.OUTLIER_FIRST,
    )
    assert dec == Decision.FAKE
    assert reason == "CONFIDENT_CLASSIFICATION"

    # 3. In-distribution, fake prob < 0.50 -> REAL
    dec, reason = evaluate_decision(
        prob_fake=0.15,
        ood_score=0.8,
        threshold=1.0,
        policy=Policy.OUTLIER_FIRST,
    )
    assert dec == Decision.REAL
    assert reason == "CONFIDENT_CLASSIFICATION"


def test_decision_dual_score() -> None:
    # 1. High confidence fake (>= 0.95) with mild OOD (score <= 1.25 * tau) -> FAKE
    dec, reason = evaluate_decision(
        prob_fake=0.96,
        ood_score=1.20,
        threshold=1.0,
        policy=Policy.DUAL_SCORE,
    )
    assert dec == Decision.FAKE
    assert reason == "CONFIDENT_CLASSIFICATION"

    # 2. High confidence fake but severe OOD (> 1.25 * tau) -> UNKNOWN
    dec, reason = evaluate_decision(
        prob_fake=0.96,
        ood_score=1.30,
        threshold=1.0,
        policy=Policy.DUAL_SCORE,
    )
    assert dec == Decision.UNKNOWN
    assert reason == "OOD_THRESHOLD_EXCEEDED"

    # 3. Moderate confidence fake (0.80) with mild OOD (> 1.0) -> UNKNOWN
    dec, reason = evaluate_decision(
        prob_fake=0.80,
        ood_score=1.10,
        threshold=1.0,
        policy=Policy.DUAL_SCORE,
    )
    assert dec == Decision.UNKNOWN
    assert reason == "OOD_THRESHOLD_EXCEEDED"

    # 4. In-distribution (<= 1.0), normal classification
    dec, reason = evaluate_decision(
        prob_fake=0.30,
        ood_score=0.70,
        threshold=1.0,
        policy=Policy.DUAL_SCORE,
    )
    assert dec == Decision.REAL
    assert reason == "CONFIDENT_CLASSIFICATION"


def test_video_aggregation_rules() -> None:
    # Less than 4 frames -> InsufficientDataError
    with pytest.raises(InsufficientDataError):
        aggregate_video_frames(
            frame_predictions=[{"decision": Decision.REAL, "is_outlier": False}],
            min_frames=4,
        )

    # 1. Majority OOD (> 40% OOD) -> UNKNOWN
    # 5 frames: 3 OOD, 2 FAKE
    frames_ood = [
        {"decision": Decision.UNKNOWN, "is_outlier": True, "prob_fake": 0.9, "ood_score": 1.5},
        {"decision": Decision.UNKNOWN, "is_outlier": True, "prob_fake": 0.9, "ood_score": 1.6},
        {"decision": Decision.UNKNOWN, "is_outlier": True, "prob_fake": 0.1, "ood_score": 1.4},
        {"decision": Decision.FAKE, "is_outlier": False, "prob_fake": 0.9, "ood_score": 0.5},
        {"decision": Decision.FAKE, "is_outlier": False, "prob_fake": 0.85, "ood_score": 0.6},
    ]
    dec, reason, summary = aggregate_video_frames(frames_ood)
    assert dec == Decision.UNKNOWN
    assert reason == "VIDEO_FRAMES_MOSTLY_OOD"
    assert summary.ood_frame_ratio == 0.6
    assert summary.total_frames_analyzed == 5

    # 2. Not mostly OOD, but >= 25% FAKE frames -> FAKE
    # 4 frames: 0 OOD, 1 FAKE, 3 REAL (1/4 = 25% fake)
    frames_fake = [
        {"decision": Decision.FAKE, "is_outlier": False, "prob_fake": 0.8, "ood_score": 0.3},
        {"decision": Decision.REAL, "is_outlier": False, "prob_fake": 0.1, "ood_score": 0.3},
        {"decision": Decision.REAL, "is_outlier": False, "prob_fake": 0.1, "ood_score": 0.2},
        {"decision": Decision.REAL, "is_outlier": False, "prob_fake": 0.2, "ood_score": 0.4},
    ]
    dec, reason, summary = aggregate_video_frames(frames_fake)
    assert dec == Decision.FAKE
    assert reason == "PERSISTENT_FAKE_FRAMES"
    assert summary.fake_frame_ratio == 0.25

    # 3. Clean real frames (<25% fake, <=40% OOD) -> REAL
    frames_real = [
        {"decision": Decision.REAL, "is_outlier": False, "prob_fake": 0.1, "ood_score": 0.3},
        {"decision": Decision.REAL, "is_outlier": False, "prob_fake": 0.15, "ood_score": 0.2},
        {"decision": Decision.REAL, "is_outlier": False, "prob_fake": 0.05, "ood_score": 0.4},
        {"decision": Decision.REAL, "is_outlier": False, "prob_fake": 0.2, "ood_score": 0.3},
    ]
    dec, reason, summary = aggregate_video_frames(frames_real)
    assert dec == Decision.REAL
    assert reason == "HOMOGENEOUS_REAL_FRAMES"
    assert summary.real_frame_ratio == 1.0
