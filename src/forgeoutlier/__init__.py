"""ForgeOutlier Lite: Open-Set Deepfake Detection Library."""

from pathlib import Path
from typing import Union, Optional

from forgeoutlier.core.types import Decision, Policy, PredictionResult
from forgeoutlier.models.loader import load_checkpoint
from forgeoutlier.ood.serialization import load_ood_artifact
from forgeoutlier.pipeline.inference import InferencePipeline

__version__ = "1.0.0"


def load_pipeline(
    checkpoint_path: Union[str, Path],
    ood_profile_path: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    ood_expected_sha256: Optional[str] = None,
    policy: Policy = Policy.OUTLIER_FIRST,
    clf_threshold: float = 0.50,
    device: str = "cpu",
    feature_dim: int = 576,
) -> InferencePipeline:
    """Load model and calibrated OOD profile securely to construct InferencePipeline.

    Args:
        checkpoint_path: Path to PyTorch .pt model checkpoint.
        ood_profile_path: Path to .npz OOD profile archive.
        manifest_path: Path to trusted_checkpoints.json manifest (optional).
        ood_expected_sha256: Optional expected SHA-256 for the .npz archive.
        policy: Tri-state decision policy (OUTLIER_FIRST or DUAL_SCORE).
        clf_threshold: Classifier threshold.
        device: Device to place model on ('cpu').
        feature_dim: Feature embedding dimension (576 or 1024).

    Returns:
        Configured and initialized InferencePipeline instance.
    """
    model = load_checkpoint(
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        verify_hash=True,
        device=device,
        feature_dim=feature_dim,
    )
    ood_scorer = load_ood_artifact(
        file_path=str(ood_profile_path),
        expected_sha256=ood_expected_sha256,
    )

    return InferencePipeline(
        model=model,
        ood_scorer=ood_scorer,
        policy=policy,
        clf_threshold=clf_threshold,
    )


__all__ = [
    "Decision",
    "Policy",
    "PredictionResult",
    "InferencePipeline",
    "load_pipeline",
    "__version__",
]
