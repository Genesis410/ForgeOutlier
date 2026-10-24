"""Core primitives, types, protocols, and data contracts."""

from forgeoutlier.core.config import MEDIA_LIMITS, NUMERICAL_CONSTANTS
from forgeoutlier.core.exceptions import (
    CorruptedMediaError,
    ExecutionFailureError,
    ForgeOutlierException,
    InsufficientDataError,
    NumericalInstabilityError,
    ResourceLimitExceededError,
    SecurityIntegrityError,
)
from forgeoutlier.core.protocols import EncoderProtocol, OODScorerProtocol
from forgeoutlier.core.security import sanitize_input_path
from forgeoutlier.core.types import (
    Decision,
    MediaMeta,
    Policy,
    PredictionResult,
    ScoreResult,
    VideoSummary,
)

__all__ = [
    "Decision",
    "Policy",
    "MediaMeta",
    "ScoreResult",
    "VideoSummary",
    "PredictionResult",
    "ForgeOutlierException",
    "SecurityIntegrityError",
    "CorruptedMediaError",
    "ResourceLimitExceededError",
    "NumericalInstabilityError",
    "InsufficientDataError",
    "ExecutionFailureError",
    "EncoderProtocol",
    "OODScorerProtocol",
    "MEDIA_LIMITS",
    "NUMERICAL_CONSTANTS",
    "sanitize_input_path",
]

