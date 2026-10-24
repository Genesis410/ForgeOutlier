"""End-to-end inference pipeline for image and video deepfake triage."""

from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import numpy as np
from PIL import Image

from forgeoutlier.core.types import (
    Decision,
    Policy,
    MediaMeta,
    ScoreResult,
    PredictionResult,
    VideoSummary,
)
from forgeoutlier.core.protocols import EncoderProtocol, OODScorerProtocol
from forgeoutlier.core.exceptions import CorruptedMediaError, ExecutionFailureError
from forgeoutlier.media.image import load_image
from forgeoutlier.media.video import VideoStream
from forgeoutlier.ood.fitting import normalize_l2
from forgeoutlier.pipeline.decisions import evaluate_decision
from forgeoutlier.pipeline.aggregation import aggregate_video_frames

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def normalize_imagenet_tensor(chw_array: np.ndarray) -> np.ndarray:
    """Normalize (3, H, W) float32 array in [0.0, 1.0] to ImageNet distribution."""
    if chw_array.ndim != 3 or chw_array.shape[0] != 3:
        raise ExecutionFailureError(f"Expected array of shape (3, H, W), got {chw_array.shape}")
    return (chw_array - IMAGENET_MEAN) / IMAGENET_STD


class InferencePipeline:
    """End-to-end pipeline uniting media loader, model encoder, and OOD scorer."""

    def __init__(
        self,
        model: EncoderProtocol,
        ood_scorer: OODScorerProtocol,
        policy: Policy = Policy.OUTLIER_FIRST,
        clf_threshold: float = 0.50,
    ) -> None:
        self.model = model
        self.ood_scorer = ood_scorer
        self.policy = policy
        self.clf_threshold = clf_threshold

    def _process_single_frame(
        self, chw_array: np.ndarray
    ) -> Dict[str, Any]:
        """Run single (3, H, W) frame through model and scorer.

        Returns dict with:
        prob_fake, prob_real, ood_score, threshold, is_outlier, decision, reason.
        """
        # Normalize to ImageNet stats
        normed_input = normalize_imagenet_tensor(chw_array)
        batch_input = np.expand_dims(normed_input, axis=0)  # (1, 3, H, W)

        # 1. Penultimate features
        raw_feats = self.model.extract_features(batch_input)  # (1, D)

        # 2. L2 normalize embedding for OOD scorer
        z_norm = normalize_l2(raw_feats)  # (1, D)

        # 3. Softmax probabilities [prob_real, prob_fake]
        probs = self.model.predict_probabilities(batch_input)  # (1, 2)
        prob_real = float(probs[0, 0])
        prob_fake = float(probs[0, 1])

        # 4. Outlier score
        ood_score_arr, outlier_flags = self.ood_scorer.is_outlier(z_norm)
        ood_score = float(ood_score_arr[0])
        is_outlier = bool(outlier_flags[0])

        # 5. Tri-state decision
        decision, reason = evaluate_decision(
            prob_fake=prob_fake,
            ood_score=ood_score,
            threshold=self.ood_scorer.threshold,
            policy=self.policy,
            clf_threshold=self.clf_threshold,
        )

        return {
            "prob_real": prob_real,
            "prob_fake": prob_fake,
            "ood_score": ood_score,
            "threshold": self.ood_scorer.threshold,
            "is_outlier": is_outlier,
            "decision": decision,
            "reason": reason,
        }

    def predict_image(
        self,
        source: Union[str, Path, Image.Image, np.ndarray],
    ) -> PredictionResult:
        """Run triage on a single image.

        Args:
            source: Path, PIL Image, or NumPy array.

        Returns:
            PredictionResult dataclass complying with standard schema.
        """
        chw_tensor, meta = load_image(source, target_size=(224, 224))
        frame_res = self._process_single_frame(chw_tensor)

        score_res = ScoreResult(
            prob_fake=round(frame_res["prob_fake"], 4),
            prob_real=round(frame_res["prob_real"], 4),
            ood_score=round(frame_res["ood_score"], 4),
            is_outlier=frame_res["is_outlier"],
            threshold=round(frame_res["threshold"], 4),
        )

        return PredictionResult(
            media=meta,
            decision=frame_res["decision"],
            policy=self.policy,
            reason=frame_res["reason"],
            scores=score_res,
            status="SUCCESS",
            schema_version="1.0.0",
            video_summary=None,
        )

    def predict_video(
        self,
        video_path: Union[str, Path],
        target_samples: int = 16,
    ) -> PredictionResult:
        """Run triage on a video using bounded frame streaming and voting.

        Args:
            video_path: Path to video file.
            target_samples: Number of uniform frames to sample (default 16).

        Returns:
            PredictionResult dataclass with video_summary.
        """
        frame_preds: List[Dict[str, Any]] = []
        meta: Optional[MediaMeta] = None

        with VideoStream(str(video_path), target_samples=target_samples) as stream:
            meta = stream.meta
            for frame_idx, chw_tensor in stream.frames():
                f_res = self._process_single_frame(chw_tensor)
                frame_preds.append({
                    "frame_index": frame_idx,
                    "prob_fake": f_res["prob_fake"],
                    "prob_real": f_res["prob_real"],
                    "ood_score": f_res["ood_score"],
                    "is_outlier": f_res["is_outlier"],
                    "decision": f_res["decision"],
                    "reason": f_res["reason"],
                })

        if meta is None or not frame_preds:
            raise CorruptedMediaError("No valid frames could be decoded from video stream.")

        video_decision, video_reason, video_summary = aggregate_video_frames(
            frame_predictions=frame_preds,
            min_frames=4,
            ood_ratio_threshold=0.40,
            fake_ratio_threshold=0.25,
        )

        # Average classifier score & OOD score over frames for top-level summary
        avg_fake = float(np.mean([f["prob_fake"] for f in frame_preds]))
        avg_real = float(np.mean([f["prob_real"] for f in frame_preds]))
        avg_ood = float(np.mean([f["ood_score"] for f in frame_preds]))
        any_outlier = any(f["is_outlier"] for f in frame_preds)

        scores = ScoreResult(
            prob_fake=round(avg_fake, 4),
            prob_real=round(avg_real, 4),
            ood_score=round(avg_ood, 4),
            is_outlier=any_outlier,
            threshold=round(self.ood_scorer.threshold, 4),
        )

        return PredictionResult(
            media=meta,
            decision=video_decision,
            policy=self.policy,
            reason=video_reason,
            scores=scores,
            status="SUCCESS",
            schema_version="1.0.0",
            video_summary=video_summary,
        )
