"""Data contracts, enums, and output schemas for ForgeOutlier Lite."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class Decision(str, Enum):
    """Tri-state detection decision."""

    REAL = "REAL"
    FAKE = "FAKE"
    UNKNOWN = "UNKNOWN"


class Policy(str, Enum):
    """Decision triage policy."""

    OUTLIER_FIRST = "outlier_first"
    DUAL_SCORE = "dual_score"
    BASELINE_NAIVE = "baseline_naive"


@dataclass(frozen=True)
class MediaMeta:
    """Metadata of processed media."""

    path: str
    media_type: str  # "image" or "video"
    dimensions: list[int]  # [height, width]
    duration_seconds: float | None = None
    total_frames_sampled: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoreResult:
    """Quantitative scoring output."""

    prob_fake: float
    prob_real: float
    ood_score: float
    is_outlier: bool
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoSummary:
    """Summary of video frame analysis."""

    total_frames_analyzed: int
    ood_frame_ratio: float
    fake_frame_ratio: float
    real_frame_ratio: float
    frame_details: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionResult:
    """Hierarchical output data contract matching JSON schema."""

    media: MediaMeta
    decision: Decision
    policy: Policy
    reason: str
    scores: ScoreResult
    status: str = "SUCCESS"
    schema_version: str = "1.0.0"
    video_summary: VideoSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "media": self.media.to_dict(),
            "result": {
                "decision": self.decision.value,
                "policy": self.policy.value,
                "reason": self.reason,
            },
            "scores": self.scores.to_dict(),
            "video_summary": self.video_summary.to_dict() if self.video_summary else None,
        }
