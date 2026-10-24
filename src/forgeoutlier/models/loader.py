"""Secure checkpoint loading and integrity verification."""

import hashlib
import json
from pathlib import Path

import torch

from forgeoutlier.core.exceptions import ExecutionFailureError, SecurityIntegrityError
from forgeoutlier.core.security import sanitize_input_path
from forgeoutlier.models.model import DeepfakeModel


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def load_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path | None = None,
    verify_hash: bool = True,
    device: str = "cpu",
    feature_dim: int = 576,
) -> DeepfakeModel:
    """Load model checkpoint securely with SHA-256 verification and weights_only=True.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        manifest_path: Optional path to trusted_checkpoints.json manifest.
                       If None, searches in manifests/trusted_checkpoints.json.
        verify_hash: Whether to strictly enforce SHA-256 validation against manifest.
        device: Torch device (defaults to 'cpu').
        feature_dim: Embedding dimension (576 or 1024).

    Returns:
        DeepfakeModel instance initialized and loaded with state_dict.

    Raises:
        SecurityIntegrityError: If hash mismatch, untrusted file, or corrupted format.
        ExecutionFailureError: If architecture or tensor shapes do not match.
    """
    clean_path = sanitize_input_path(str(checkpoint_path))
    if not clean_path.is_file():
        raise SecurityIntegrityError(f"Checkpoint file does not exist: {clean_path}")

    # Compute hash
    actual_hash = compute_file_sha256(clean_path)

    if verify_hash:
        # Resolve manifest
        if manifest_path is None:
            # Look up standard manifest location
            manifest_candidates = [
                clean_path.parent / "trusted_checkpoints.json",
                clean_path.parent / "manifests" / "trusted_checkpoints.json",
                Path(__file__).resolve().parent.parent.parent
                / "manifests"
                / "trusted_checkpoints.json",
                Path.cwd() / "manifests" / "trusted_checkpoints.json",
            ]
            manifest_file = None
            for cand in manifest_candidates:
                if cand.is_file():
                    manifest_file = cand
                    break

            if manifest_file is None:
                raise SecurityIntegrityError(
                    f"Checkpoint manifest 'trusted_checkpoints.json' not found for verification: {clean_path.name}"
                )
        else:
            manifest_file = sanitize_input_path(str(manifest_path))
            if not manifest_file.is_file():
                raise SecurityIntegrityError(f"Manifest file not found: {manifest_file}")

        try:
            with open(manifest_file, encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception as e:
            raise SecurityIntegrityError(f"Failed to read trusted manifest: {e}") from e

        # Check hash against manifest entries
        # Manifest schema supports:
        # 1. {"manifest_version": "...", "checkpoints": [{"sha256": "..."}, ...]}
        # 2. {"filename_or_model_name": {"sha256": "...", ...}}
        # 3. List of dicts [{"name": "...", "sha256": "..."}, ...]
        # 4. Dict mapping sha256 -> metadata
        trusted_hashes = set()
        if isinstance(manifest_data, dict):
            # If standard envelope with 'checkpoints' key
            if "checkpoints" in manifest_data and isinstance(manifest_data["checkpoints"], list):
                for item in manifest_data["checkpoints"]:
                    if isinstance(item, dict) and "sha256" in item:
                        trusted_hashes.add(str(item["sha256"]).lower())
            for k, v in manifest_data.items():
                if isinstance(v, dict) and "sha256" in v:
                    trusted_hashes.add(str(v["sha256"]).lower())
                elif isinstance(v, str) and len(v) == 64:
                    trusted_hashes.add(v.lower())
                if len(k) == 64:
                    trusted_hashes.add(k.lower())
        elif isinstance(manifest_data, list):
            for item in manifest_data:
                if isinstance(item, dict) and "sha256" in item:
                    trusted_hashes.add(str(item["sha256"]).lower())

        if actual_hash not in trusted_hashes:
            raise SecurityIntegrityError(
                f"SHA-256 verification failed for {clean_path.name}: "
                f"hash {actual_hash} is not present in trusted manifest."
            )

    # Strictly load using weights_only=True and map_location='cpu'
    try:
        state_dict = torch.load(clean_path, map_location=device, weights_only=True)
    except Exception as e:
        raise SecurityIntegrityError(f"Checkpoint deserialization failed: {e}") from e

    if not isinstance(state_dict, dict):
        raise SecurityIntegrityError(
            f"Expected state_dict dictionary, got {type(state_dict).__name__}"
        )

    # Build model and load state_dict
    model = DeepfakeModel(feature_dim=feature_dim)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        raise ExecutionFailureError(f"Failed to apply state_dict to model architecture: {e}") from e

    model.to(device)
    model.eval()
    return model
