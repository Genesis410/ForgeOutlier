"""Dataset synthesis generator for empirical benchmarking of ForgeOutlier.

Generates controlled sets of ID (Real / Face-like), Manipulated (Fake / Artifacted),
and Hard OOD (Anime/Cartoon, Medical/Microscopy, Deep Underwater, Scenery/Abstract).
Saves images directly as valid JPEG/PNG and creates ground-truth manifest JSON.
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image


def generate_benchmark_dataset(
    base_dir: Path,
    n_per_category: Optional[int] = None,
    n_real: int = 100,
    n_fake: int = 100,
    n_ood_total: int = 50,
    seed: int = 42,
) -> Path:
    """Generate benchmark dataset and return path to manifest JSON.

    If n_per_category is provided, overrides category allocations (legacy compatibility).
    Default allocation matches PRD-EVAL-SMALLDATA-001 (250 images: 100 REAL, 100 FAKE, 50 OOD).
    """
    import hashlib

    rng = np.random.default_rng(seed)
    base_dir.mkdir(parents=True, exist_ok=True)
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if n_per_category is not None:
        c_real = n_per_category
        c_fake = n_per_category
        ood_share = n_per_category // 4
        c_anime = ood_share
        c_medical = ood_share
        c_underwater = ood_share
        c_noise = n_per_category - (ood_share * 3)
    else:
        c_real = n_real
        c_fake = n_fake
        ood_share = n_ood_total // 4
        c_anime = ood_share
        c_medical = ood_share
        c_underwater = ood_share
        c_noise = n_ood_total - (ood_share * 3)

    manifest_entries = []

    def _save_and_record(img_arr: np.ndarray, filename: str, label: str, domain: str, fmt: str = "JPEG", quality: int = 95):
        img_p = images_dir / filename
        if fmt == "PNG":
            Image.fromarray(img_arr).save(img_p, format="PNG")
        else:
            Image.fromarray(img_arr).save(img_p, format="JPEG", quality=quality)
        
        sha256 = hashlib.sha256(img_p.read_bytes()).hexdigest()
        manifest_entries.append({
            "path": str(img_p),
            "label": label,
            "domain": domain,
            "sha256": sha256,
            "size_bytes": img_p.stat().st_size,
        })

    # Category 1: ID - Real Faces
    for i in range(c_real):
        r = rng.normal(180, 15, (224, 224))
        g = rng.normal(140, 15, (224, 224))
        b = rng.normal(120, 15, (224, 224))
        img_arr = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
        _save_and_record(img_arr, f"id_real_{i:03d}.jpg", "REAL", "id_natural", fmt="JPEG", quality=95)

    # Category 2: ID - Fake / Manipulated
    for i in range(c_fake):
        r = rng.normal(180, 15, (224, 224))
        g = rng.normal(140, 15, (224, 224))
        b = rng.normal(120, 15, (224, 224))
        r[80:144, 80:144] += rng.uniform(20, 50, (64, 64))
        img_arr = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
        _save_and_record(img_arr, f"id_fake_{i:03d}.jpg", "FAKE", "id_manipulated", fmt="JPEG", quality=95)

    # Category 3: Hard OOD - Anime / Cartoon
    for i in range(c_anime):
        img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
        img_arr[:, :] = [255, 105, 180]
        img_arr[50:180, 50:180] = [0, 255, 255]
        _save_and_record(img_arr, f"ood_anime_{i:03d}.png", "OOD", "ood_anime", fmt="PNG")

    # Category 4: Hard OOD - Medical Microscopy
    for i in range(c_medical):
        gray = rng.gamma(shape=2.0, scale=30.0, size=(224, 224)).clip(0, 255).astype(np.uint8)
        img_arr = np.stack([gray, gray, gray], axis=-1)
        _save_and_record(img_arr, f"ood_medical_{i:03d}.png", "OOD", "ood_medical", fmt="PNG")

    # Category 5: Hard OOD - Deep Underwater
    for i in range(c_underwater):
        r = rng.uniform(5, 25, (224, 224))
        g = rng.uniform(40, 90, (224, 224))
        b = rng.uniform(90, 160, (224, 224))
        img_arr = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
        _save_and_record(img_arr, f"ood_underwater_{i:03d}.jpg", "OOD", "ood_underwater", fmt="JPEG", quality=85)

    # Category 6: Hard OOD - Geometric / High Frequency Noise
    for i in range(c_noise):
        img_arr = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        _save_and_record(img_arr, f"ood_noise_{i:03d}.png", "OOD", "ood_noise", fmt="PNG")

    manifest_p = base_dir / "manifest.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    return manifest_p
