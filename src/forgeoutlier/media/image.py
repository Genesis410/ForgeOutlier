"""Image ingestion, Pillow decompression-bomb defense, EXIF orientation handling, and tensor conversion."""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image, ImageOps

from forgeoutlier.core.config import MEDIA_LIMITS
from forgeoutlier.core.exceptions import (
    CorruptedMediaError,
    ResourceLimitExceededError,
)
from forgeoutlier.core.security import sanitize_input_path
from forgeoutlier.core.types import MediaMeta

# Enforce strict pixel limit on Pillow to thwart decompression bomb attacks
Image.MAX_IMAGE_PIXELS = MEDIA_LIMITS.max_image_pixels


def load_image(
    source: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, MediaMeta]:
    """Load, sanitize, auto-orient, resize, and convert image to RGB float32 tensor array.

    Args:
        source: File path, Path object, PIL Image, or NumPy array.
        target_size: (H, W) for bilinear resizing (default: 224x224).

    Returns:
        Tuple of:
            - Preprocessed array of shape (3, H, W) normalized to [0.0, 1.0], float32.
            - MediaMeta dataclass with original metadata.

    Raises:
        ResourceLimitExceededError: If file size, dimensions, or pixel count exceed safety limits.
        CorruptedMediaError: If image file is invalid, truncated, or unreadable.
    """
    img: Image.Image
    file_bytes: int = 0

    if isinstance(source, (str, Path)):
        resolved_path = sanitize_input_path(str(source))
        if not resolved_path.is_file():
            raise CorruptedMediaError(f"Image file does not exist: {resolved_path}")

        file_bytes = resolved_path.stat().st_size
        if file_bytes > MEDIA_LIMITS.max_image_bytes:
            raise ResourceLimitExceededError(
                f"File size {file_bytes} bytes exceeds limit of {MEDIA_LIMITS.max_image_bytes} bytes."
            )

        try:
            raw_img = Image.open(resolved_path)
            raw_img.verify()  # Check for header truncation / corruption
            # Reopen because verify() mutates file pointer
            img = Image.open(resolved_path)
            # Enforce decoding of pixels into memory immediately
            img.load()
        except Image.DecompressionBombError as exc:
            raise ResourceLimitExceededError(f"Decompression bomb detected: {exc}") from exc
        except Exception as exc:
            raise CorruptedMediaError(f"Failed to decode image {resolved_path.name}: {exc}") from exc

    elif isinstance(source, Image.Image):
        img = source.copy()
    elif isinstance(source, np.ndarray):
        if source.ndim == 2:
            img = Image.fromarray(source)
        elif source.ndim == 3 and source.shape[2] in (1, 3, 4):
            img = Image.fromarray(source)
        elif source.ndim == 3 and source.shape[0] in (1, 3, 4):
            # Transpose (C, H, W) -> (H, W, C)
            img = Image.fromarray(np.transpose(source, (1, 2, 0)))
        else:
            raise CorruptedMediaError(f"Unsupported NumPy array shape for image: {source.shape}")
    else:
        raise CorruptedMediaError(f"Unsupported image input type: {type(source)}")

    orig_width, orig_height = img.size

    # Validate dimension limits
    if (
        orig_width > MEDIA_LIMITS.max_image_dim
        or orig_height > MEDIA_LIMITS.max_image_dim
        or (orig_width * orig_height) > MEDIA_LIMITS.max_image_pixels
    ):
        raise ResourceLimitExceededError(
            f"Image dimensions ({orig_width}x{orig_height}) exceed maximum allowed ({MEDIA_LIMITS.max_image_dim}px)."
        )

    # Correct EXIF orientation (e.g. photos taken from smartphones)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass  # If EXIF reading fails, continue without transpose

    # Ensure 3-channel RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize to target (W, H)
    target_h, target_w = target_size
    if (img.width, img.height) != (target_w, target_h):
        img_resized = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    else:
        img_resized = img

    # Convert to NumPy float32 (H, W, 3) in [0.0, 1.0]
    arr = np.asarray(img_resized, dtype=np.float32) / 255.0

    # Permute to CHW format: (3, H, W)
    chw_tensor = np.transpose(arr, (2, 0, 1))

    meta = MediaMeta(
        path=str(source) if isinstance(source, (str, Path)) else "<in-memory>",
        media_type="image",
        dimensions=[orig_height, orig_width],
        duration_seconds=None,
        total_frames_sampled=1,
    )

    return chw_tensor, meta
