"""Numerically stable Mahalanobis distance scoring via Cholesky decomposition."""

from typing import Tuple
import numpy as np
import scipy.linalg

from forgeoutlier.core.config import NUMERICAL_CONSTANTS
from forgeoutlier.core.exceptions import NumericalInstabilityError
from forgeoutlier.core.protocols import OODScorerProtocol


class MahalanobisScorer(OODScorerProtocol):
    """Computes Mahalanobis distance using Cholesky factor L: Sigma = L L^T.

    Instead of inverting Sigma directly (O(D^3) and numerically brittle),
    we solve the lower triangular system L y = (z - mu) via forward substitution.
    Mahalanobis distance d = ||y||_2 = sqrt(y^T y).
    """

    def __init__(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        threshold: float,
        regularization: float = 1e-6,
    ) -> None:
        """Initialize scorer with centroid, covariance matrix, and threshold.

        Args:
            mu: Mean vector of shape (D,).
            cov: Covariance matrix of shape (D, D).
            threshold: Calibrated decision boundary.
            regularization: Diagonal jitter to guarantee positive definiteness.
        """
        self._mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        self._dim = self._mu.shape[0]
        self._threshold = float(threshold)

        cov_matrix = np.asarray(cov, dtype=np.float64)
        if cov_matrix.shape != (self._dim, self._dim):
            raise ValueError(
                f"Covariance shape {cov_matrix.shape} does not match centroid dimension {self._dim}."
            )

        # Apply diagonal regularization jitter for unconditional positive-definiteness
        cov_reg = cov_matrix + np.eye(self._dim, dtype=np.float64) * regularization

        try:
            # Lower triangular factor L: cov = L @ L.T
            self._L = scipy.linalg.cholesky(cov_reg, lower=True)
        except scipy.linalg.LinAlgError as exc:
            raise NumericalInstabilityError(
                f"Cholesky decomposition failed; covariance matrix is not positive-definite: {exc}"
            ) from exc

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def feature_dim(self) -> int:
        return self._dim

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate Mahalanobis distance for each embedding vector.

        Args:
            embeddings: Array of shape (N, D) or (D,).

        Returns:
            1D array of distances of shape (N,) or scalar array (1,).
        """
        z = np.asarray(embeddings, dtype=np.float64)
        single_sample = False
        if z.ndim == 1:
            z = z.reshape(1, -1)
            single_sample = True
        elif z.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array, got {z.ndim}D.")

        if z.shape[1] != self._dim:
            raise ValueError(
                f"Embedding dimension {z.shape[1]} does not match expected {self._dim}."
            )

        if not np.all(np.isfinite(z)):
            raise NumericalInstabilityError("Input embeddings contain non-finite values.")

        # Center embeddings: diff = (z - mu)^T shape (D, N)
        diff = (z - self._mu).T

        # Solve L @ y = diff -> y has shape (D, N)
        # Using scipy.linalg.solve_triangular (O(D^2) per sample)
        try:
            y = scipy.linalg.solve_triangular(self._L, diff, lower=True, check_finite=False)
        except Exception as exc:
            raise NumericalInstabilityError(f"Triangular solve failed: {exc}") from exc

        # Distances: sqrt(sum(y^2, axis=0))
        distances = np.linalg.norm(y, ord=2, axis=0).astype(np.float32)

        return distances[0:1] if single_sample else distances

    def is_outlier(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Determine outlier status based on calibrated threshold."""
        scores = self.score(embeddings)
        outlier_flags = scores > self._threshold
        return scores, outlier_flags
