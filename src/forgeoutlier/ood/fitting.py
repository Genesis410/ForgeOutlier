"""Statistical fitting, L2-normalization, Ledoit-Wolf covariance estimation, and quantile calibration."""

from typing import Tuple
import numpy as np

from forgeoutlier.core.config import NUMERICAL_CONSTANTS
from forgeoutlier.core.exceptions import (
    InsufficientDataError,
    NumericalInstabilityError,
)


def normalize_l2(x: np.ndarray, eps: float = NUMERICAL_CONSTANTS.eps) -> np.ndarray:
    """Normalize vectors along the last axis to unit L2 norm.

    Formula: x_norm = x / (||x||_2 + eps)

    Args:
        x: Input numpy array of shape (..., D).
        eps: Small non-zero constant for numerical stability.

    Returns:
        L2-normalized array of identical shape and float32 dtype.

    Raises:
        NumericalInstabilityError: If input contains NaN or Inf values.
    """
    if not np.all(np.isfinite(x)):
        raise NumericalInstabilityError("Input contains non-finite values (NaN or Inf).")

    norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    return (x / (norm + eps)).astype(np.float32)


def fit_centroid_and_covariance(
    embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Calculate mean centroid and well-conditioned covariance matrix via Ledoit-Wolf shrinkage.

    Args:
        embeddings: Training feature array of shape (N, D).

    Returns:
        Tuple of (mu [D], cov [D, D], shrinkage [float]).

    Raises:
        InsufficientDataError: If sample count N < 2.
        NumericalInstabilityError: If Ledoit-Wolf estimation fails or returns non-finite values.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array of embeddings (N, D), got {embeddings.ndim}D.")

    n_samples, n_features = embeddings.shape
    if n_samples < 2:
        raise InsufficientDataError(
            f"At least 2 samples required to compute covariance, got {n_samples}."
        )

    # Validate finite values
    if not np.all(np.isfinite(embeddings)):
        raise NumericalInstabilityError("Embeddings contain NaN or Inf.")

    # Centroid
    mu = np.mean(embeddings, axis=0, dtype=np.float64).astype(np.float32)

    # Pure NumPy Ledoit-Wolf shrinkage (eliminating scikit-learn dependency)
    try:
        X = embeddings.astype(np.float64)
        X_centered = X - np.mean(X, axis=0)
        emp_cov = (X_centered.T @ X_centered) / n_samples
        mu_trace = float(np.trace(emp_cov) / n_features)

        X2 = X_centered ** 2
        xt_x = X_centered.T @ X_centered
        delta_ = float(np.sum(xt_x ** 2) / (n_samples ** 2))
        beta_ = float(np.sum(X2.T @ X2))

        beta = (1.0 / (n_features * n_samples)) * ((beta_ / n_samples) - delta_)
        emp_cov_trace_sum = float(np.sum(np.diag(emp_cov)))
        delta = delta_ - 2.0 * mu_trace * emp_cov_trace_sum + n_features * (mu_trace ** 2)
        delta /= n_features

        beta = min(beta, delta)
        shrinkage = 0.0 if (delta == 0.0 or beta == 0.0) else max(0.0, min(1.0, float(beta / delta)))
        shrunk_cov = (1.0 - shrinkage) * emp_cov + shrinkage * mu_trace * np.eye(n_features, dtype=np.float64)
        cov = shrunk_cov.astype(np.float32)
    except Exception as exc:
        raise NumericalInstabilityError(f"Ledoit-Wolf covariance estimation failed: {exc}") from exc

    if not np.all(np.isfinite(cov)):
        raise NumericalInstabilityError("Calculated covariance matrix contains non-finite values.")

    return mu, cov, shrinkage


def calibrate_quantile_threshold(
    scores: np.ndarray,
    percentile: float = NUMERICAL_CONSTANTS.default_threshold_percentile,
) -> float:
    """Compute empirical quantile threshold for OOD scores.

    Args:
        scores: Array of 1D distance or anomaly scores from validation ID set.
        percentile: Quantile percentile (e.g. 95.0 for 95th percentile).

    Returns:
        Scalar threshold as float.

    Raises:
        InsufficientDataError: If scores array is empty.
        NumericalInstabilityError: If scores contain NaN or Inf.
    """
    if scores.size == 0:
        raise InsufficientDataError("Cannot calibrate threshold on empty scores.")

    if not np.all(np.isfinite(scores)):
        raise NumericalInstabilityError("Scores array contains non-finite values.")

    threshold_val = float(np.percentile(scores, percentile))
    return threshold_val
