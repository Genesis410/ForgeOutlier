"""Unit and security tests for Layer 2B: media ingestion, decompression bomb defense, and frame extraction."""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytest

from forgeoutlier.core.exceptions import (
    CorruptedMediaError,
    ExecutionFailureError,
    ResourceLimitExceededError,
    SecurityIntegrityError,
)
from forgeoutlier.media import VideoStream, crop_face, extract_video_frames, load_image


def test_load_image_numpy_and_pil() -> None:
    # 1. NumPy (H, W, 3) in uint8
    raw_arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    chw, meta = load_image(raw_arr, target_size=(224, 224))
    assert chw.shape == (3, 224, 224)
    assert chw.dtype == np.float32
    assert 0.0 <= chw.min() and chw.max() <= 1.0
    assert meta.media_type == "image"
    assert meta.dimensions == [64, 64]

    # 2. PIL Image
    pil_img = Image.new("RGB", (100, 50), color="blue")
    chw_pil, meta_pil = load_image(pil_img, target_size=(112, 112))
    assert chw_pil.shape == (3, 112, 112)
    assert meta_pil.dimensions == [50, 100]


def test_load_image_file_and_decompression_bomb(tmp_path: Path) -> None:
    # Valid image on disk
    valid_file = tmp_path / "sample.jpg"
    img = Image.new("RGB", (80, 80), color="red")
    img.save(valid_file)

    chw, meta = load_image(str(valid_file))
    assert chw.shape == (3, 224, 224)
    assert meta.dimensions == [80, 80]
    assert meta.path == str(valid_file)

    # Null byte detection
    with pytest.raises(SecurityIntegrityError):
        load_image("test\0bad.jpg", target_size=(224, 224))

    # Corrupted / missing file rejection
    with pytest.raises(CorruptedMediaError):
        load_image(str(tmp_path / "non_existent.jpg"))

    # Decompression bomb / dimension limit rejection
    oversized = Image.new("RGB", (5000, 5000))
    with pytest.raises(ResourceLimitExceededError, match="exceed maximum allowed"):
        load_image(oversized)


def test_extract_video_frames(tmp_path: Path) -> None:
    # Generate synthetic MP4 video with OpenCV
    video_path = tmp_path / "synthetic_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 64))

    for i in range(20):
        # Frame with changing color gradient
        frame = np.full((64, 64, 3), i * 10, dtype=np.uint8)
        out.write(frame)
    out.release()

    # Extract 5 uniform frames
    frames, indices, meta = extract_video_frames(str(video_path), target_samples=5, target_size=(224, 224))

    assert frames.shape == (5, 3, 224, 224)
    assert len(indices) == 5
    assert indices == [0, 4, 9, 14, 19]
    assert meta.media_type == "video"
    assert meta.dimensions == [64, 64]
    assert meta.total_frames_sampled == 5
    assert meta.duration_seconds == pytest.approx(2.0, abs=0.1)

    # Missing video rejection
    with pytest.raises(CorruptedMediaError):
        extract_video_frames(str(tmp_path / "non_existent.mp4"))


def test_face_crop_strict() -> None:
    # 1. Blank image without face -> must raise ExecutionFailureError('NO_FACE_DETECTED')
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ExecutionFailureError, match="NO_FACE_DETECTED"):
        crop_face(blank)

    # 2. Custom detector injection succeeding
    def mock_detector(img: np.ndarray):
        return (10, 10, 40, 40)

    cropped, bbox = crop_face(blank, target_size=(224, 224), detector_fn=mock_detector)
    assert cropped.shape == (3, 224, 224)
    assert bbox == (10, 10, 40, 40)

    # 3. Malformed input array
    with pytest.raises(CorruptedMediaError):
        crop_face(np.zeros((10, 10), dtype=np.uint8))


def test_video_stream_context_manager(tmp_path: Path) -> None:
    video_path = tmp_path / "stream_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 64))

    for i in range(15):
        frame = np.full((64, 64, 3), i * 15, dtype=np.uint8)
        out.write(frame)
    out.release()

    with VideoStream(str(video_path), target_samples=4, target_size=(224, 224)) as stream:
        assert stream.meta.media_type == "video"
        assert stream.meta.dimensions == [64, 64]
        assert len(stream.sample_indices) == 4

        yielded = list(stream.frames())
        assert len(yielded) == 4
        for idx, frame_arr in yielded:
            assert isinstance(idx, int)
            assert frame_arr.shape == (3, 224, 224)
            assert frame_arr.dtype == np.float32
