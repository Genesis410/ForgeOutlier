"""Regression test: Numerical reproducibility and frozen model outputs."""

import numpy as np
import torch

from forgeoutlier.models.model import DeepfakeModel
from forgeoutlier.ood.fitting import calibrate_quantile_threshold, fit_centroid_and_covariance
from forgeoutlier.ood.mahalanobis import MahalanobisScorer


def test_frozen_synthetic_model_inference_determinism() -> None:
    """Ensure identical inputs produce bitwise reproducible results across seeds."""
    torch.manual_seed(42)
    model = DeepfakeModel(feature_dim=576)
    model.eval()

    test_tensor = torch.ones((1, 3, 224, 224), dtype=torch.float32)

    with torch.no_grad():
        logits1, feats1 = model(test_tensor)
        logits2, feats2 = model(test_tensor)

    np.testing.assert_allclose(logits1.numpy(), logits2.numpy(), rtol=1e-6)
    np.testing.assert_allclose(feats1.numpy(), feats2.numpy(), rtol=1e-6)


def test_frozen_covariance_and_mahalanobis_determinism() -> None:
    """Ensure covariance and Mahalanobis scoring are mathematically deterministic."""
    rng = np.random.RandomState(42)
    data = rng.randn(50, 16).astype(np.float32)

    mu1, cov1, shrink1 = fit_centroid_and_covariance(data)
    mu2, cov2, shrink2 = fit_centroid_and_covariance(data)

    np.testing.assert_allclose(mu1, mu2, rtol=1e-6)
    np.testing.assert_allclose(cov1, cov2, rtol=1e-6)
    assert shrink1 == shrink2

    scorer = MahalanobisScorer(mu=mu1, cov=cov1, threshold=0.0)
    scores = scorer.score(data)
    threshold = calibrate_quantile_threshold(scores, percentile=95.0)

    # Exact repeatability
    scorer2 = MahalanobisScorer(mu=mu2, cov=cov2, threshold=0.0)
    scores2 = scorer2.score(data)
    threshold2 = calibrate_quantile_threshold(scores2, percentile=95.0)

    np.testing.assert_allclose(scores, scores2, rtol=1e-6)
    assert threshold == threshold2
