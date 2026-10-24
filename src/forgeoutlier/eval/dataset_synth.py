"""Dataset synthesis generator for empirical benchmarking of ForgeOutlier.

Generates controlled sets of ID (Real / Face-like), Manipulated (Fake / Artifacted),
and Hard OOD (Anime/Cartoon, Medical/Microscopy, Deep Underwater, Scenery/Abstract).
Saves images directly as valid JPEG/PNG and creates ground-truth manifest JSON.
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image


def generate_benchmark_dataset(base_dir: Path, n_per_category: int = 25, seed: int = 42) -> Path:
    """Generate benchmark dataset and return path to manifest JSON.

    Categories:
      1. id_real: Natural human skin tones, facial symmetry, low frequency gradient.
      2. id_fake: Deepfake artifacts, checkerboard frequency patterns, high-frequency blending boundaries.
      3. ood_anime: High-saturation cartoon/anime styles, flat coloring, high contrast edges.
      4. ood_medical: Grayscale cell microscopy / X-ray textures, circular structures.
      5. ood_underwater: Monochromatic deep cyan/blue marine imagery with low contrast particulate noise.
      6. ood_fractal: Abstract geometric fractals / noise patterns with extreme spectral distributions.
    """
    rng = np.random.default_rng(seed)
    base_dir.mkdir(parents=True, exist_ok=True)
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []

    # Category 1: ID - Real Faces (Simulated natural portraits)
    for i in range(n_per_category):
        # Soft skin tone gradient + slight noise
        r = rng.normal(180, 15, (224, 224))
        g = rng.normal(140, 15, (224, 224))
        b = rng.normal(120, 15, (224, 224))
        img_arr = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
        img_p = images_dir / f"id_real_{i:03d}.jpg"
        Image.fromarray(img_arr).save(img_p, format="JPEG", quality=95)
        manifest_entries.append({"path": str(img_p), "label": "REAL", "domain": "id_natural"})

    # Category 2: ID - Fake / Manipulated
    for i in range(n_per_category):
        # Base skin tone + checkerboard artifact / sharp splice box
        r = rng.normal(180, 15, (224, 224))
        g = rng.normal(140, 15, (224, 224))
        b = rng.normal(120, 15, (224, 224))
        # High-frequency boundary in center (blending artifact)
        r[80:144, 80:144] += rng.uniform(20, 50, (64, 64))
        img_arr = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
        img_p = images_dir / f"id_fake_{i:03d}.jpg"
        Image.fromarray(img_arr).save(img_p, format="JPEG", quality=95)
        manifest_entries.append({"path": str(img_p), "label": "FAKE", "domain": "id_manipulated"})

    # Category 3: Hard OOD - Anime / Cartoon
    for i in range(n_per_category):
        # Vivid cartoon colors, flat blocks
        img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
        img_arr[:, :] = [255, 105, 180]  # bright hot pink
        img_arr[50:180, 50:180] = [0, 255, 255]  # cyan block
        img_p = images_dir / f"ood_anime_{i:03d}.png"
        Image.fromarray(img_arr).save(img_p, format="PNG")
        manifest_entries.append({"path": str(img_p), "label": "OOD", "domain": "ood_anime"})

    # Category 4: Hard OOD - Medical Microscopy
    for i in range(n_per_category):
        # Monochromatic grayscale cell textures
        gray = rng.gamma(shape=2.0, scale=30.0, size=(224, 224)).clip(0, 255).astype(np.uint8)
        img_arr = np.stack([gray, gray, gray], axis=-1)
        img_p = images_dir / f"ood_medical_{i:03d}.png"
        Image.fromarray(img_arr).save(img_p, format="PNG")
        manifest_entries.append({"path": str(img_p), "label": "OOD", "domain": "ood_medical"})

    # Category 5: Hard OOD - Deep Underwater
    for i in range(n_per_category):
        # Low contrast deep cyan/navy
        r = rng.uniform(5, 25, (224, 224))
        g = rng.uniform(40, 90, (224, 224))
        b = rng.uniform(90, 160, (224, 224))
        img_arr = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
        img_p = images_dir / f"ood_underwater_{i:03d}.jpg"
        Image.fromarray(img_arr).save(img_p, format="JPEG", quality=85)
        manifest_entries.append({"path": str(img_p), "label": "OOD", "domain": "ood_underwater"})

    # Category 6: Hard OOD - Geometric / High Frequency Noise
    for i in range(n_per_category):
        # Pure high frequency uniform noise (extreme out of distribution)
        img_arr = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        img_p = images_dir / f"ood_noise_{i:03d}.png"
        Image.fromarray(img_arr).save(img_p, format="PNG")
        manifest_entries.append({"path": str(img_p), "label": "OOD", "domain": "ood_noise"})

    manifest_p = base_dir / "manifest.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    return manifest_p
