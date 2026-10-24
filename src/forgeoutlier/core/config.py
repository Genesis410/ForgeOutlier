"""System limits, numeric constants, and runtime budgets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MediaLimits:
    """Resource boundaries for image and video processing."""

    max_image_dim: int = 4096
    max_image_pixels: int = 4096 * 4096
    max_image_bytes: int = 50 * 1024 * 1024  # 50 MB
    max_video_bytes: int = 500 * 1024 * 1024  # 500 MB
    max_video_frames: int = 1000
    default_video_samples: int = 16


@dataclass(frozen=True)
class NumericalConstants:
    """Epsilon constants and calibration parameters."""

    eps: float = 1e-12
    default_threshold_percentile: float = 95.0
    float32_rtol: float = 1e-5
    float32_atol: float = 1e-6
    float64_rtol: float = 1e-7
    float64_atol: float = 1e-9


# Global frozen instances
MEDIA_LIMITS = MediaLimits()
NUMERICAL_CONSTANTS = NumericalConstants()
