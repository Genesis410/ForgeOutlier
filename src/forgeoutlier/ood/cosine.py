"""Cosine distance to centroid OOD scoring engine."""

import numpy as np

from forgeoutlier.core.protocols import OODScorerProtocol
from forgeoutlier.ood.fitting import normalize_l2


class CosineCentroidScorer(OODScorerProtocol):
    """Computes Cosine distance d = 1.0 - (z_norm . mu_norm) as OOD anomaly score."""

    def __init__(self, mu: np.ndarray, threshold: float) -> None:
        """Initialize scorer with centroid and threshold.

        Args:
            mu: Mean vector of shape (D,).
            threshold: Calibrated decision boundary for cosine distance.
        """
        self._raw_mu = np.asarray(mu, dtype=np.float32).reshape(-1)
        self._dim = self._raw_mu.shape[0]
        self._threshold = float(threshold)

        # Pre-normalize centroid
        self._mu_norm = normalize_l2(self._raw_mu.reshape(1, -1)).reshape(-1)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def feature_dim(self) -> int:
        return self._dim

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine distance: d = 1.0 - (z_norm @ mu_norm).

        Args:
            embeddings: Array of shape (N, D) or (D,).

        Returns:
            Cosine distances of shape (N,). Range: [0.0, 2.0].
        """
        z = np.asarray(embeddings, dtype=np.float32)
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

        # Normalize input vectors
        z_norm = normalize_l2(z)

        # Dot product with normalized centroid: cosine similarity
        sim = np.dot(z_norm, self._mu_norm)

        # Distance: 1.0 - similarity, clipped to [0.0, 2.0] for safety
        dist = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float32)

        return dist[0:1] if single_sample else dist

    def is_outlier(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Determine outlier status based on calibrated threshold."""
        scores = self.score(embeddings)
        outlier_flags = scores > self._threshold
        return scores, outlier_flags
