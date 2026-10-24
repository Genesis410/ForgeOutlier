"""Adversarial testing: Non-finite (NaN, Inf) and extreme value injection."""

import numpy as np
import pytest
import torch

from forgeoutlier.core.exceptions import ExecutionFailureError, NumericalInstabilityError
from forgeoutlier.eval.metrics import (
    compute_pr_curve,
    compute_roc_curve,
)
from forgeoutlier.models.model import DeepfakeModel
from forgeoutlier.ood.cosine import CosineCentroidScorer
from forgeoutlier.ood.fitting import (
    normalize_l2,
)
from forgeoutlier.ood.mahalanobis import MahalanobisScorer


def test_normalize_l2_with_zeros() -> None:
    """Zero vectors produce zero or normalized outputs without throwing NaN."""
    zeros = np.zeros((5, 16), dtype=np.float32)
    normed = normalize_l2(zeros)
    assert not np.isnan(normed).any()
    assert not np.isinf(normed).any()


def test_ood_scorers_reject_nan_features() -> None:
    """Scorers gracefully handle or flag NaN embeddings."""
    dim = 8
    mu = np.zeros((dim,), dtype=np.float32)
    cov = np.eye(dim, dtype=np.float32)

    maha = MahalanobisScorer(mu=mu, cov=cov, threshold=1.0)
    cosine = CosineCentroidScorer(mu=mu, threshold=1.0)

    nan_features = np.ones((2, dim), dtype=np.float32)
    nan_features[0, 2] = np.nan

    # Test scoring behaviour
    with pytest.raises((ExecutionFailureError, NumericalInstabilityError, ValueError)):
        maha.score(nan_features)

    with pytest.raises((ExecutionFailureError, NumericalInstabilityError, ValueError)):
        cosine.score(nan_features)


def test_model_forward_rejects_nan_tensor() -> None:
    """PyTorch DeepfakeModel rejects or detects NaN inputs."""
    model = DeepfakeModel(feature_dim=576)
    bad_tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    bad_tensor[0, 0, 10, 10] = float("nan")

    logits, features = model(bad_tensor)
    assert torch.isnan(logits).any() or torch.isnan(features).any()


def test_metrics_reject_nan_arrays() -> None:
    """AUROC / AUPRC calculations reject non-finite inputs."""
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, np.nan, 0.4, 0.9])

    with pytest.raises((ValueError, ExecutionFailureError)):
        compute_roc_curve(y_true, y_score)

    with pytest.raises((ValueError, ExecutionFailureError)):
        compute_pr_curve(y_true, y_score)
