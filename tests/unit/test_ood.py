"""Tests for Layer 2A: OOD engines, mathematical stability, fitting, and serialization."""

from pathlib import Path
import numpy as np
import pytest

from forgeoutlier.core.exceptions import (
    InsufficientDataError,
    NumericalInstabilityError,
    SecurityIntegrityError,
)
from forgeoutlier.core.protocols import OODScorerProtocol
from forgeoutlier.ood import (
    CosineCentroidScorer,
    MahalanobisScorer,
    calibrate_quantile_threshold,
    fit_centroid_and_covariance,
    load_ood_artifact,
    normalize_l2,
    save_ood_artifact,
)


def test_normalize_l2() -> None:
    # 2D array
    data = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    normed = normalize_l2(data)
    norms = np.linalg.norm(normed, ord=2, axis=-1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

    # Rejection of NaN / Inf
    bad_data = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    with pytest.raises(NumericalInstabilityError):
        normalize_l2(bad_data)


def test_fit_centroid_and_covariance() -> None:
    # Deterministic cluster
    rng = np.random.RandomState(42)
    embeddings = rng.randn(50, 16).astype(np.float32) + 2.0
    mu, cov, shrinkage = fit_centroid_and_covariance(embeddings)

    assert mu.shape == (16,)
    assert cov.shape == (16, 16)
    assert 0.0 <= shrinkage <= 1.0
    # Covariance should be symmetric and finite
    np.testing.assert_allclose(cov, cov.T, atol=1e-5)

    # Insufficient data rejection
    with pytest.raises(InsufficientDataError):
        fit_centroid_and_covariance(embeddings[:1])


def test_calibrate_quantile_threshold() -> None:
    scores = np.arange(1, 101, dtype=np.float32)
    th95 = calibrate_quantile_threshold(scores, percentile=95.0)
    assert pytest.approx(th95, 0.1) == 95.05

    # Empty scores
    with pytest.raises(InsufficientDataError):
        calibrate_quantile_threshold(np.array([]))


def test_mahalanobis_scorer() -> None:
    # Build distribution with D=4
    mu = np.zeros(4, dtype=np.float32)
    cov = np.eye(4, dtype=np.float32)
    scorer = MahalanobisScorer(mu=mu, cov=cov, threshold=3.0)

    assert isinstance(scorer, OODScorerProtocol)
    assert scorer.threshold == 3.0
    assert scorer.feature_dim == 4

    # Point at origin: Mahalanobis distance ≈ 0
    origin_point = np.zeros(4, dtype=np.float32)
    score = scorer.score(origin_point)
    assert pytest.approx(float(score[0]), abs=1e-3) == 0.0

    # Point far away: distance ≈ 5
    far_point = np.array([5.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scores, outliers = scorer.is_outlier(far_point)
    assert float(scores[0]) > scorer.threshold
    assert bool(outliers[0]) is True


def test_cosine_scorer() -> None:
    mu = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    scorer = CosineCentroidScorer(mu=mu, threshold=0.5)

    assert isinstance(scorer, OODScorerProtocol)

    # Identical direction: distance ≈ 0
    same_dir = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
    score_same = scorer.score(same_dir)
    assert pytest.approx(float(score_same[0]), abs=1e-4) == 0.0

    # Orthogonal direction: distance ≈ 1
    ortho_dir = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    scores, outliers = scorer.is_outlier(ortho_dir)
    assert pytest.approx(float(scores[0]), abs=1e-4) == 1.0
    assert bool(outliers[0]) is True


def test_safe_serialization(tmp_path: Path) -> None:
    mu = np.array([1.0, 2.0], dtype=np.float32)
    cov = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float32)
    target_file = tmp_path / "model_ood.npz"

    # Save
    sha = save_ood_artifact(
        str(target_file),
        method="mahalanobis",
        mu=mu,
        threshold=2.5,
        cov=cov,
    )
    assert len(sha) == 64

    # Load with hash check
    loaded_scorer = load_ood_artifact(str(target_file), expected_sha256=sha)
    assert isinstance(loaded_scorer, MahalanobisScorer)
    assert loaded_scorer.threshold == 2.5

    # Integrity violation on mismatched hash
    with pytest.raises(SecurityIntegrityError, match="SHA-256 mismatch"):
        load_ood_artifact(str(target_file), expected_sha256="wronghash" * 8)
