"""Adversarial testing: corrupted, truncated, and random binary inputs."""

import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from forgeoutlier.core.exceptions import CorruptedMediaError, ExecutionFailureError, SecurityIntegrityError
from forgeoutlier.media.image import load_image
from forgeoutlier.media.video import extract_video_frames
from forgeoutlier.models.loader import load_checkpoint
from forgeoutlier.ood.serialization import load_ood_artifact


def test_corrupted_image_header(tmp_path: Path) -> None:
    """Corrupted image header raises CorruptedMediaError or ExecutionFailureError."""
    corrupted_file = tmp_path / "corrupted.jpg"
    corrupted_file.write_bytes(b"\xFF\xD8\xFF\xE0" + b"\x00" * 20)

    with pytest.raises((CorruptedMediaError, ExecutionFailureError)):
        load_image(str(corrupted_file))


def test_random_binary_garbage_as_image(tmp_path: Path) -> None:
    """Random noise binary input is rejected gracefully."""
    garbage_file = tmp_path / "garbage.png"
    garbage_file.write_bytes(np.random.bytes(1024))

    with pytest.raises((CorruptedMediaError, ExecutionFailureError)):
        load_image(str(garbage_file))


def test_empty_zero_byte_media(tmp_path: Path) -> None:
    """Zero-byte media files fail immediately."""
    empty_img = tmp_path / "empty.jpg"
    empty_img.write_bytes(b"")

    with pytest.raises((CorruptedMediaError, ExecutionFailureError)):
        load_image(str(empty_img))

    empty_video = tmp_path / "empty.mp4"
    empty_video.write_bytes(b"")

    with pytest.raises((CorruptedMediaError, ExecutionFailureError)):
        extract_video_frames(str(empty_video))


def test_corrupted_ood_npz_archive(tmp_path: Path) -> None:
    """Corrupted or non-ZIP OOD artifact raises ExecutionFailureError."""
    bad_npz = tmp_path / "invalid.npz"
    bad_npz.write_bytes(b"PK\x03\x04" + b"\xFF" * 50)

    with pytest.raises(ExecutionFailureError):
        load_ood_artifact(str(bad_npz))
