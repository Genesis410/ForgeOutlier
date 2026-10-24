"""Unit tests for models, backbone, and secure checkpoint loader."""

from pathlib import Path
import json
import pytest
import numpy as np
import torch

from forgeoutlier.core.protocols import EncoderProtocol
from forgeoutlier.core.exceptions import SecurityIntegrityError, ExecutionFailureError
from forgeoutlier.models.backbone import MobileNetV3SmallBackbone
from forgeoutlier.models.classifier import BinaryClassifierHead
from forgeoutlier.models.model import DeepfakeModel
from forgeoutlier.models.loader import load_checkpoint, compute_file_sha256


def test_backbone_feature_dim_and_forward():
    """Verify MobileNetV3SmallBackbone output shape and feature_dim."""
    backbone = MobileNetV3SmallBackbone(feature_dim=576)
    backbone.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        z = backbone(x)
    assert z.shape == (2, 576)
    assert backbone.feature_dim == 576


def test_backbone_1024_dim():
    """Verify MobileNetV3SmallBackbone with 1024 embedding dimension."""
    backbone = MobileNetV3SmallBackbone(feature_dim=1024)
    backbone.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        z = backbone(x)
    assert z.shape == (1, 1024)
    assert backbone.feature_dim == 1024


def test_classifier_head():
    """Verify BinaryClassifierHead forward shape."""
    head = BinaryClassifierHead(in_features=576, num_classes=2)
    head.eval()
    z = torch.randn(4, 576)
    logits = head(z)
    assert logits.shape == (4, 2)


def test_deepfake_model_protocol_and_extraction():
    """Verify DeepfakeModel satisfies EncoderProtocol and extracts features."""
    model = DeepfakeModel(feature_dim=576)
    assert isinstance(model, EncoderProtocol)

    # Test extract_features with (N, C, H, W)
    batch_img = np.random.randn(2, 3, 224, 224).astype(np.float32)
    feats = model.extract_features(batch_img)
    assert feats.shape == (2, 576)
    assert feats.dtype == np.float32

    # Test single 3D image (C, H, W)
    single_img = np.random.randn(3, 224, 224).astype(np.float32)
    feats_single = model.extract_features(single_img)
    assert feats_single.shape == (1, 576)

    # Test predict_probabilities
    probs = model.predict_probabilities(batch_img)
    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_deepfake_model_invalid_inputs():
    """Verify DeepfakeModel rejects non-image dimensions or channel mismatches."""
    model = DeepfakeModel(feature_dim=576)
    with pytest.raises(ExecutionFailureError):
        model.extract_features(np.zeros((10, 10), dtype=np.float32))

    with pytest.raises(ExecutionFailureError):
        model.extract_features(np.zeros((1, 1, 224, 224), dtype=np.float32))


def test_secure_loader_and_manifest_verification(tmp_path):
    """Verify SHA-256 validation against trusted manifest and weights_only=True."""
    # 1. Save dummy valid checkpoint
    dummy_model = DeepfakeModel(feature_dim=576)
    ckpt_path = tmp_path / "model_weights.pt"
    torch.save(dummy_model.state_dict(), ckpt_path)

    # Compute genuine sha256
    file_sha256 = compute_file_sha256(ckpt_path)

    # 2. Create trusted manifest
    manifest_path = tmp_path / "trusted_checkpoints.json"
    manifest_data = {
        "mobilenetv3_small_v1": {
            "sha256": file_sha256,
            "architecture": "MobileNetV3Small",
            "feature_dim": 576,
        }
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    # 3. Successful load
    loaded_model = load_checkpoint(ckpt_path, manifest_path=manifest_path, verify_hash=True)
    assert isinstance(loaded_model, DeepfakeModel)
    assert loaded_model.feature_dim == 576

    # 4. Tampered manifest / mismatched hash should raise SecurityIntegrityError
    tampered_manifest = tmp_path / "tampered_manifest.json"
    tampered_manifest.write_text(
        json.dumps({"model": {"sha256": "0" * 64}}), encoding="utf-8"
    )
    with pytest.raises(SecurityIntegrityError, match="SHA-256 verification failed"):
        load_checkpoint(ckpt_path, manifest_path=tampered_manifest, verify_hash=True)

    # 5. Non-existent checkpoint
    with pytest.raises(SecurityIntegrityError):
        load_checkpoint(tmp_path / "nonexistent.pt", manifest_path=manifest_path)

    # 6. Corrupted file
    corrupt_path = tmp_path / "corrupt.pt"
    corrupt_path.write_bytes(b"not a valid pytorch checkpoint")
    corrupt_manifest = tmp_path / "corrupt_manifest.json"
    corrupt_manifest.write_text(
        json.dumps({"corrupt": {"sha256": compute_file_sha256(corrupt_path)}}),
        encoding="utf-8",
    )
    with pytest.raises(SecurityIntegrityError, match="Checkpoint deserialization failed"):
        load_checkpoint(corrupt_path, manifest_path=corrupt_manifest, verify_hash=True)
