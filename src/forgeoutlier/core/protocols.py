"""Architectural protocols defining structural contracts between layers."""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EncoderProtocol(Protocol):
    """Protocol for feature extractors and classifiers."""

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the feature embedding vector."""
        ...

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        """Extract L2-normalized feature embeddings from input images.

        Args:
            x: Input array of shape (N, C, H, W) or (C, H, W).

        Returns:
            Embeddings of shape (N, feature_dim) or (feature_dim,).
        """
        ...

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities [P(REAL), P(FAKE)] from embeddings.

        Args:
            features: Embeddings of shape (N, feature_dim).

        Returns:
            Probabilities of shape (N, 2).
        """
        ...


@runtime_checkable
class OODScorerProtocol(Protocol):
    """Protocol for Out-of-Distribution scoring engines."""

    @property
    def threshold(self) -> float:
        """Calibrated distance/score threshold for OOD decision."""
        ...

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate OOD distance/anomaly score for given embeddings.

        Args:
            embeddings: Normalized embeddings of shape (N, D).

        Returns:
            OOD scores of shape (N,).
        """
        ...

    def is_outlier(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Determine if embeddings are OOD against calibrated threshold.

        Args:
            embeddings: Normalized embeddings of shape (N, D).

        Returns:
            Tuple of (scores [N], is_outlier_flags [N, bool]).
        """
        ...
