"""Optional face detection and bounding box cropping hook with Haar cascades / OpenCV."""

from typing import Callable, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from forgeoutlier.core.exceptions import CorruptedMediaError, ExecutionFailureError


def crop_face(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    margin_ratio: float = 0.2,
    detector_fn: Optional[Callable[[np.ndarray], Optional[Tuple[int, int, int, int]]]] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Detect primary face and crop with bounding margin.

    Args:
        image: RGB NumPy array of shape (H, W, 3) in uint8 or float32.
        target_size: Output (H, W) for bilinear resize.
        margin_ratio: Extra padding around bounding box as ratio of face box size.
        detector_fn: Optional custom detector callable returning (x, y, w, h) or None.

    Returns:
        Tuple of:
            - Cropped and resized face tensor of shape (3, H, W) in float32 [0.0, 1.0].
            - Bounding box tuple (x, y, w, h) in original pixel coordinates.

    Raises:
        ExecutionFailureError: With reason 'NO_FACE_DETECTED' if no face is found and crop was strictly required.
        CorruptedMediaError: If input array is malformed.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise CorruptedMediaError("Input to crop_face must be 3-dimensional NumPy array.")

    # Convert to uint8 (H, W, 3) for OpenCV cascade if passed as float
    if image.dtype in (np.float32, np.float64):
        if image.max() <= 1.0:
            uint8_img = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            uint8_img = image.clip(0, 255).astype(np.uint8)
    else:
        uint8_img = image.astype(np.uint8)

    bbox: Optional[Tuple[int, int, int, int]] = None

    if detector_fn is not None:
        bbox = detector_fn(uint8_img)
    else:
        # Fallback to Haar cascade if cv2 exposes it
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                face_cascade = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30),
                )
                if len(faces) > 0:
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    bbox = (int(x), int(y), int(w), int(h))
            except Exception:
                pass

    if bbox is None:
        raise ExecutionFailureError("NO_FACE_DETECTED: No frontal face detected in media.")

    x, y, w, h = bbox

    # Apply margin padding
    h_img, w_img = uint8_img.shape[:2]
    dx = int(w * margin_ratio)
    dy = int(h * margin_ratio)

    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(w_img, x + w + dx)
    y2 = min(h_img, y + h + dy)

    face_crop = uint8_img[y1:y2, x1:x2]

    # Resize to target
    target_h, target_w = target_size
    pil_crop = Image.fromarray(face_crop).resize((target_w, target_h), Image.Resampling.BILINEAR)

    # Float32 normalized (3, H, W)
    arr = np.asarray(pil_crop, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))

    return chw, (int(x), int(y), int(w), int(h))
