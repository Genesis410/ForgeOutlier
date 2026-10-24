"""Out-of-Distribution (OOD) detection engines and statistical profiles."""

from forgeoutlier.ood.cosine import CosineCentroidScorer
from forgeoutlier.ood.fitting import (
    calibrate_quantile_threshold,
    fit_centroid_and_covariance,
    normalize_l2,
)
from forgeoutlier.ood.mahalanobis import MahalanobisScorer
from forgeoutlier.ood.serialization import (
    calculate_sha256,
    load_ood_artifact,
    save_ood_artifact,
)

__all__ = [
    "normalize_l2",
    "fit_centroid_and_covariance",
    "calibrate_quantile_threshold",
    "MahalanobisScorer",
    "CosineCentroidScorer",
    "calculate_sha256",
    "save_ood_artifact",
    "load_ood_artifact",
]

