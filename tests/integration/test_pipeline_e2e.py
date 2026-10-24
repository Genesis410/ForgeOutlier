"""Integration tests for end-to-end image and video triage pipeline."""

import json
from pathlib import Path
import numpy as np
import cv2
import pytest
from PIL import Image

from forgeoutlier import (
    load_pipeline,
    Decision,
    Policy,
    PredictionResult,
)
from forgeoutlier.core.exceptions import CorruptedMediaError, SecurityIntegrityError
from forgeoutlier.models.model import DeepfakeModel
from forgeoutlier.models.loader import compute_file_sha256
from forgeoutlier.ood.serialization import save_ood_artifact
import torch


def create_synthetic_assets(
    tmp_path: Path,
    feature_dim: int = 576,
):
    """Helper to generate tiny mock model checkpoint, manifest, and OOD artifact."""
    # 1. Model checkpoint
    model = DeepfakeModel(feature_dim=feature_dim)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    model_sha256 = compute_file_sha256(checkpoint_path)

    # 2. Manifest
    manifest_path = tmp_path / "trusted_checkpoints.json"
    manifest_data = {
        "manifest_version": "1.0.0",
        "checkpoints": [
            {
                "filename": "model.pt",
                "sha256": model_sha256,
                "feature_dim": feature_dim,
                "architecture": "MobileNetV3SmallBackbone",
            }
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # 3. OOD Profile .npz
    mu = np.zeros((feature_dim,), dtype=np.float32)
    threshold = 0.50
    ood_path = tmp_path / "ood_profile.npz"
    ood_sha256 = save_ood_artifact(
        file_path=str(ood_path),
        method="cosine",
        mu=mu,
        threshold=threshold,
    )

    return checkpoint_path, manifest_path, ood_path, ood_sha256


def test_pipeline_e2e_image_inference(tmp_path: Path) -> None:
    ckpt, manifest, ood_path, ood_hash = create_synthetic_assets(tmp_path, feature_dim=576)

    pipeline = load_pipeline(
        checkpoint_path=ckpt,
        ood_profile_path=ood_path,
        manifest_path=manifest,
        ood_expected_sha256=ood_hash,
        policy=Policy.OUTLIER_FIRST,
    )

    # Generate synthetic 224x224 RGB image
    img_path = tmp_path / "test_sample.png"
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    img.save(img_path)

    # Predict image
    result = pipeline.predict_image(str(img_path))
    assert isinstance(result, PredictionResult)
    assert result.status == "SUCCESS"
    assert result.decision in (Decision.REAL, Decision.FAKE, Decision.UNKNOWN)
    assert result.media.media_type == "image"
    assert result.media.dimensions == [224, 224]
    assert 0.0 <= result.scores.prob_fake <= 1.0
    assert result.scores.ood_score >= 0.0

    # Test dictionary export matching schema
    d = result.to_dict()
    assert d["schema_version"] == "1.0.0"
    assert d["status"] == "SUCCESS"
    assert "media" in d
    assert "result" in d
    assert "scores" in d
    assert d["video_summary"] is None


def test_pipeline_e2e_video_inference(tmp_path: Path) -> None:
    ckpt, manifest, ood_path, ood_hash = create_synthetic_assets(tmp_path, feature_dim=576)

    pipeline = load_pipeline(
        checkpoint_path=ckpt,
        ood_profile_path=ood_path,
        manifest_path=manifest,
        ood_expected_sha256=ood_hash,
        policy=Policy.OUTLIER_FIRST,
    )

    # Generate synthetic 10-frame MP4 video
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (128, 128))
    for i in range(10):
        frame = np.full((128, 128, 3), i * 20, dtype=np.uint8)
        out.write(frame)
    out.release()

    # Predict video
    result = pipeline.predict_video(str(video_path), target_samples=6)
    assert isinstance(result, PredictionResult)
    assert result.status == "SUCCESS"
    assert result.decision in (Decision.REAL, Decision.FAKE, Decision.UNKNOWN)
    assert result.media.media_type == "video"
    assert result.video_summary is not None
    assert result.video_summary.total_frames_analyzed >= 4
    assert len(result.video_summary.frame_details) == result.video_summary.total_frames_analyzed

    # Test to_dict() serialization
    res_dict = result.to_dict()
    assert res_dict["video_summary"] is not None
    assert res_dict["video_summary"]["total_frames_analyzed"] >= 4
