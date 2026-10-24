"""Tests for core module: types, exceptions, protocols, and security."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from forgeoutlier.core.config import MEDIA_LIMITS, NUMERICAL_CONSTANTS
from forgeoutlier.core.exceptions import (
    CorruptedMediaError,
    ForgeOutlierException,
    InsufficientDataError,
    NumericalInstabilityError,
    ResourceLimitExceededError,
    SecurityIntegrityError,
)
from forgeoutlier.core.protocols import EncoderProtocol, OODScorerProtocol
from forgeoutlier.core.security import sanitize_input_path
from forgeoutlier.core.types import (
    VideoSummary,
)


def test_exception_hierarchy() -> None:
    exc = ResourceLimitExceededError("Dimension too large")
    assert isinstance(exc, CorruptedMediaError)
    assert isinstance(exc, ForgeOutlierException)
    assert exc.code == "RESOURCE_LIMIT_EXCEEDED"
    assert "[RESOURCE_LIMIT_EXCEEDED] Dimension too large" in str(exc)

    num_exc = NumericalInstabilityError("Matrix singular")
    assert num_exc.code == "NUMERICAL_INSTABILITY_ERROR"

    sec_exc = SecurityIntegrityError("Hash mismatch")
    assert sec_exc.code == "SECURITY_INTEGRITY_VIOLATION"

    data_exc = InsufficientDataError("Samples < 10")
    assert data_exc.code == "INSUFFICIENT_DATA_ERROR"


def test_config_immutability() -> None:
    assert MEDIA_LIMITS.max_image_dim == 4096
    assert MEDIA_LIMITS.max_image_bytes == 50 * 1024 * 1024
    assert NUMERICAL_CONSTANTS.eps == 1e-12
    with pytest.raises(FrozenInstanceError):
        MEDIA_LIMITS.max_image_dim = 2048  # type: ignore


def test_security_sanitize_path(tmp_path: Path) -> None:
    # Safe path inside base_dir
    safe_file = tmp_path / "test.jpg"
    safe_file.touch()
    resolved = sanitize_input_path(str(safe_file), base_dir=str(tmp_path))
    assert resolved == safe_file.resolve()

    # Empty path rejected
    with pytest.raises(SecurityIntegrityError, match="cannot be empty"):
        sanitize_input_path("")

    # Null byte rejected
    with pytest.raises(SecurityIntegrityError, match="Null byte detected"):
        sanitize_input_path("test\0bad.jpg")

    # Path traversal rejected when base_dir given
    outside_file = tmp_path.parent / "escape.jpg"
    with pytest.raises(SecurityIntegrityError, match="Path traversal detected"):
        sanitize_input_path(str(outside_file), base_dir=str(tmp_path))


class DummyEncoder:
    """Compliant mock of EncoderProtocol."""

    @property
    def feature_dim(self) -> int:
        return 128

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        return np.ones((x.shape[0], self.feature_dim), dtype=np.float32)

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        return np.full((features.shape[0], 2), 0.5, dtype=np.float32)


class DummyScorer:
    """Compliant mock of OODScorerProtocol."""

    @property
    def threshold(self) -> float:
        return 5.0

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        return np.ones(embeddings.shape[0], dtype=np.float32)

    def is_outlier(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = self.score(embeddings)
        return scores, scores > self.threshold


def test_protocol_runtime_checkable() -> None:
    encoder = DummyEncoder()
    assert isinstance(encoder, EncoderProtocol)

    scorer = DummyScorer()
    assert isinstance(scorer, OODScorerProtocol)


def test_video_summary_serialization() -> None:
    summary = VideoSummary(
        total_frames_analyzed=16,
        ood_frame_ratio=0.125,
        fake_frame_ratio=0.875,
        real_frame_ratio=0.0,
        frame_details=[{"frame_idx": 0, "decision": "FAKE"}],
    )
    d = summary.to_dict()
    assert d["total_frames_analyzed"] == 16
    assert d["ood_frame_ratio"] == 0.125
    assert len(d["frame_details"]) == 1
