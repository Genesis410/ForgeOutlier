"""Evaluation, benchmarking, and reporting suite for ForgeOutlier."""

from forgeoutlier.eval.metrics import (
    compute_roc_curve,
    compute_pr_curve,
    compute_auroc,
    compute_auprc,
    compute_fpr_at_tpr,
    compute_tristate_confusion_matrix,
)
from forgeoutlier.eval.benchmark import BenchmarkSummary, evaluate_dataset
from forgeoutlier.eval.export import (
    export_summary_json,
    export_confusion_matrix_csv,
    export_predictions_csv,
)

__all__ = [
    "compute_roc_curve",
    "compute_pr_curve",
    "compute_auroc",
    "compute_auprc",
    "compute_fpr_at_tpr",
    "compute_tristate_confusion_matrix",
    "BenchmarkSummary",
    "evaluate_dataset",
    "export_summary_json",
    "export_confusion_matrix_csv",
    "export_predictions_csv",
]
