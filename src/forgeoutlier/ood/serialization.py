"""Safe serialization and deserialization of OOD artifacts via NPZ without pickle."""

import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

from forgeoutlier.core.exceptions import ExecutionFailureError, SecurityIntegrityError
from forgeoutlier.ood.cosine import CosineCentroidScorer
from forgeoutlier.ood.mahalanobis import MahalanobisScorer


def calculate_sha256(file_path: Path) -> str:
    """Compute hex-digest SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_ood_artifact(
    file_path: str,
    method: str,
    mu: np.ndarray,
    threshold: float,
    cov: np.ndarray = None,
    extra_meta: Dict[str, Any] = None,
) -> str:
    """Save OOD parameters to a compressed .npz archive without pickle.

    Args:
        file_path: Output target path (.npz).
        method: 'mahalanobis' or 'cosine'.
        mu: Centroid array.
        threshold: Quantile threshold.
        cov: Covariance matrix (required for mahalanobis).
        extra_meta: Optional dictionary for tracking metadata.

    Returns:
        Hex-digest SHA-256 hash of the generated artifact.
    """
    path = Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        "method": np.array(method),
        "mu": np.asarray(mu, dtype=np.float32),
        "threshold": np.array(threshold, dtype=np.float64),
    }

    if cov is not None:
        data["cov"] = np.asarray(cov, dtype=np.float32)

    # Save using strictly allow_pickle=False
    np.savez_compressed(path, **data)

    return calculate_sha256(path)


def load_ood_artifact(
    file_path: str,
    expected_sha256: str = None,
) -> Any:
    """Load and verify an OOD scorer from a .npz archive using allow_pickle=False.

    Args:
        file_path: Path to .npz artifact.
        expected_sha256: Optional expected SHA-256 hash for integrity validation.

    Returns:
        Instance of MahalanobisScorer or CosineCentroidScorer.

    Raises:
        SecurityIntegrityError: If SHA-256 mismatches or unsupported method.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}")

    # Verify integrity if expected hash provided
    if expected_sha256 is not None:
        actual_sha256 = calculate_sha256(path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise SecurityIntegrityError(
                f"Artifact SHA-256 mismatch for {path.name}: expected {expected_sha256}, got {actual_sha256}"
            )

    # Safe loading strictly with allow_pickle=False
    try:
        with np.load(path, allow_pickle=False) as npz:
            method = str(npz["method"])
            mu = npz["mu"]
            threshold = float(npz["threshold"])

            if method == "mahalanobis":
                if "cov" not in npz:
                    raise SecurityIntegrityError("Corrupted Mahalanobis artifact: missing 'cov' array.")
                cov = npz["cov"]
                return MahalanobisScorer(mu=mu, cov=cov, threshold=threshold)
            elif method == "cosine":
                return CosineCentroidScorer(mu=mu, threshold=threshold)
            else:
                raise SecurityIntegrityError(f"Unsupported OOD method: '{method}'")
    except Exception as e:
        if isinstance(e, (SecurityIntegrityError, FileNotFoundError)):
            raise e
        raise ExecutionFailureError(f"Failed to load OOD artifact: {e}") from e
