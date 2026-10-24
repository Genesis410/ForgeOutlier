"""Video ingestion, uniform frame sampling, and bounded streaming."""

from pathlib import Path
from typing import Generator, List, Optional, Tuple
import cv2
import numpy as np

from forgeoutlier.core.config import MEDIA_LIMITS
from forgeoutlier.core.exceptions import (
    CorruptedMediaError,
    ExecutionFailureError,
    ResourceLimitExceededError,
)
from forgeoutlier.core.security import sanitize_input_path
from forgeoutlier.core.types import MediaMeta


class VideoStream:
    """Context manager for bounded streaming frames from a video file."""

    def __init__(
        self,
        video_path: str,
        target_samples: int = MEDIA_LIMITS.default_video_samples,
        target_size: Tuple[int, int] = (224, 224),
    ) -> None:
        self.video_path = video_path
        self.target_samples = target_samples
        self.target_size = target_size
        self._cap: Optional[cv2.VideoCapture] = None
        self._meta: Optional[MediaMeta] = None
        self._sample_indices: List[int] = []

    def __enter__(self) -> "VideoStream":
        resolved_path = sanitize_input_path(self.video_path)
        if not resolved_path.is_file():
            raise CorruptedMediaError(f"Video file does not exist: {resolved_path}")

        file_bytes = resolved_path.stat().st_size
        if file_bytes > MEDIA_LIMITS.max_video_bytes:
            raise ResourceLimitExceededError(
                f"Video file size {file_bytes} bytes exceeds maximum {MEDIA_LIMITS.max_video_bytes} bytes."
            )

        self._cap = cv2.VideoCapture(str(resolved_path))
        if not self._cap.isOpened():
            raise CorruptedMediaError(f"Could not open video file: {resolved_path.name}")

        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))

        if total_frames <= 0:
            raise CorruptedMediaError("Video reported 0 or negative total frames.")

        duration_sec = total_frames / fps if fps > 0 else 0.0
        n_samples = min(self.target_samples, total_frames, MEDIA_LIMITS.max_video_frames)
        self._sample_indices = np.linspace(0, total_frames - 1, num=n_samples, dtype=int).tolist()

        self._meta = MediaMeta(
            path=str(resolved_path),
            media_type="video",
            dimensions=[orig_h, orig_w],
            duration_seconds=round(duration_sec, 2),
            total_frames_sampled=len(self._sample_indices),
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def meta(self) -> MediaMeta:
        if self._meta is None:
            raise ExecutionFailureError("VideoStream context has not been entered.")
        return self._meta

    @property
    def sample_indices(self) -> List[int]:
        return list(self._sample_indices)

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield (frame_index, frame_array_chw) one frame at a time."""
        if self._cap is None or not self._cap.isOpened():
            raise ExecutionFailureError("VideoStream is not open.")

        target_h, target_w = self.target_size
        for idx in self._sample_indices:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = self._cap.read()
            if not ret or frame_bgr is None:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if (frame_rgb.shape[1], frame_rgb.shape[0]) != (target_w, target_h):
                frame_rgb = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            arr = frame_rgb.astype(np.float32) / 255.0
            chw = np.transpose(arr, (2, 0, 1))
            yield idx, chw


def extract_video_frames(
    video_path: str,
    target_samples: int = MEDIA_LIMITS.default_video_samples,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, List[int], MediaMeta]:
    """Uniformly sample frames from video file within memory bounds.

    Args:
        video_path: Path to MP4 or container video.
        target_samples: Number of equidistant frames to sample.
        target_size: Target (H, W) per frame.

    Returns:
        Tuple of:
            - Array of sampled frames: (N, 3, H, W) normalized to [0.0, 1.0], float32.
            - List of selected 0-indexed frame indices.
            - MediaMeta dataclass with video properties.

    Raises:
        ResourceLimitExceededError: If video file size or frame count exceeds budget.
        CorruptedMediaError: If video cannot be opened or parsed.
    """
    resolved_path = sanitize_input_path(video_path)
    if not resolved_path.is_file():
        raise CorruptedMediaError(f"Video file does not exist: {resolved_path}")

    file_bytes = resolved_path.stat().st_size
    if file_bytes > MEDIA_LIMITS.max_video_bytes:
        raise ResourceLimitExceededError(
            f"Video file size {file_bytes} bytes exceeds maximum {MEDIA_LIMITS.max_video_bytes} bytes."
        )

    cap = cv2.VideoCapture(str(resolved_path))
    if not cap.isOpened():
        raise CorruptedMediaError(f"Could not open video file: {resolved_path.name}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        if total_frames <= 0:
            raise CorruptedMediaError("Video reported 0 or negative total frames.")

        duration_sec = total_frames / fps if fps > 0 else 0.0

        # Uniform sampling indices
        n_samples = min(target_samples, total_frames)
        # Linspace generates equidistant indices across the video duration
        sample_indices = np.linspace(0, total_frames - 1, num=n_samples, dtype=int).tolist()

        frames_list: List[np.ndarray] = []
        target_h, target_w = target_size

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                # Fallback: if seek fails, grab next available frame
                continue

            # Convert BGR (OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Resize if needed
            if (frame_rgb.shape[1], frame_rgb.shape[0]) != (target_w, target_h):
                frame_rgb = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Convert to float32 (3, H, W) in [0.0, 1.0]
            arr = frame_rgb.astype(np.float32) / 255.0
            chw = np.transpose(arr, (2, 0, 1))
            frames_list.append(chw)

        if not frames_list:
            raise CorruptedMediaError("No readable frames could be extracted from video.")

        frames_batch = np.stack(frames_list, axis=0)  # Shape: (N, 3, H, W)

        meta = MediaMeta(
            path=str(resolved_path),
            media_type="video",
            dimensions=[orig_h, orig_w],
            duration_seconds=round(duration_sec, 2),
            total_frames_sampled=len(frames_list),
        )

        return frames_batch, sample_indices[: len(frames_list)], meta

    finally:
        cap.release()
