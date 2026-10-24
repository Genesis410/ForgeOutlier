"""Video frame voting and aggregation rules."""

from typing import List, Tuple, Dict, Any, Optional
from forgeoutlier.core.types import Decision, VideoSummary
from forgeoutlier.core.exceptions import InsufficientDataError


def aggregate_video_frames(
    frame_predictions: List[Dict[str, Any]],
    min_frames: int = 4,
    ood_ratio_threshold: float = 0.40,
    fake_ratio_threshold: float = 0.25,
) -> Tuple[Decision, str, VideoSummary]:
    """Aggregate individual frame predictions into video-level tri-state decision (D007).

    Rules:
    - If total sampled frames M < min_frames (4) -> raise InsufficientDataError("INSUFFICIENT_DATA").
    - R_OOD = (frames where is_outlier == True) / M
    - R_FAKE = (frames where decision == FAKE) / M
    - If R_OOD > 0.40 -> Decision.UNKNOWN, reason: "VIDEO_FRAMES_MOSTLY_OOD"
    - Else if R_FAKE >= 0.25 -> Decision.FAKE, reason: "PERSISTENT_FAKE_FRAMES"
    - Else -> Decision.REAL, reason: "HOMOGENEOUS_REAL_FRAMES"

    Args:
        frame_predictions: List of dicts per frame containing:
                           {"decision": Decision, "is_outlier": bool, "prob_fake": float, "ood_score": float}
        min_frames: Minimum required successfully sampled frames (default 4).
        ood_ratio_threshold: OOD frame ratio triggering UNKNOWN (default 0.40).
        fake_ratio_threshold: Fake frame ratio triggering FAKE (default 0.25).

    Returns:
        Tuple of (video_decision, reason, VideoSummary).
    """
    m = len(frame_predictions)
    if m < min_frames:
        raise InsufficientDataError(
            f"INSUFFICIENT_DATA: Video has only {m} frames successfully sampled (minimum {min_frames} required)."
        )

    ood_count = sum(1 for f in frame_predictions if f.get("is_outlier", False))
    fake_count = sum(1 for f in frame_predictions if f.get("decision") == Decision.FAKE or f.get("decision") == "FAKE")
    real_count = sum(1 for f in frame_predictions if f.get("decision") == Decision.REAL or f.get("decision") == "REAL")

    r_ood = float(ood_count / m)
    r_fake = float(fake_count / m)
    r_real = float(real_count / m)

    if r_ood > ood_ratio_threshold:
        decision = Decision.UNKNOWN
        reason = "VIDEO_FRAMES_MOSTLY_OOD"
    elif r_fake >= fake_ratio_threshold:
        decision = Decision.FAKE
        reason = "PERSISTENT_FAKE_FRAMES"
    else:
        decision = Decision.REAL
        reason = "HOMOGENEOUS_REAL_FRAMES"

    summary = VideoSummary(
        total_frames_analyzed=m,
        ood_frame_ratio=round(r_ood, 4),
        fake_frame_ratio=round(r_fake, 4),
        real_frame_ratio=round(r_real, 4),
        frame_details=frame_predictions,
    )

    return decision, reason, summary
