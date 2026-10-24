"""Smoke test for Phase 0 setup and basic data contracts."""

import forgeoutlier
from forgeoutlier.core.types import Decision, MediaMeta, Policy, PredictionResult, ScoreResult


def test_package_metadata() -> None:
    assert forgeoutlier.__version__ == "1.0.0"


def test_data_contract_serialization() -> None:
    media = MediaMeta(path="test.jpg", media_type="image", dimensions=[224, 224])
    scores = ScoreResult(
        prob_fake=0.85,
        prob_real=0.15,
        ood_score=10.5,
        is_outlier=True,
        threshold=8.0,
    )
    result = PredictionResult(
        media=media,
        decision=Decision.UNKNOWN,
        policy=Policy.OUTLIER_FIRST,
        reason="OOD_THRESHOLD_EXCEEDED",
        scores=scores,
    )

    data = result.to_dict()
    assert data["status"] == "SUCCESS"
    assert data["result"]["decision"] == "UNKNOWN"
    assert data["scores"]["is_outlier"] is True
    assert data["scores"]["prob_fake"] == 0.85
