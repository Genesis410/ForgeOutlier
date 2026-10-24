"""Unit and integration tests for Evaluation & Benchmarking Suite (Phase 5/6)."""

import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from forgeoutlier.eval import (
    compute_roc_curve,
    compute_pr_curve,
    compute_auroc,
    compute_auprc,
    compute_fpr_at_tpr,
    compute_tristate_confusion_matrix,
    BenchmarkSummary,
    evaluate_dataset,
    export_summary_json,
    export_confusion_matrix_csv,
    export_predictions_csv,
)
from forgeoutlier import (
    load_pipeline,
    Policy,
)
from forgeoutlier.models.model import DeepfakeModel
from forgeoutlier.models.loader import compute_file_sha256
from forgeoutlier.ood.serialization import save_ood_artifact
import torch


def test_metrics_curves_and_scalars() -> None:
    # Ground truth: 5 ID (0), 5 OOD (1)
    # Perfectly separated scores: ID in [0.1..0.5], OOD in [0.6..1.0]
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    fpr, tpr, thresh = compute_roc_curve(y_true, scores)
    auroc = compute_auroc(y_true, scores)
    assert auroc == 1.0

    prec, rec, _ = compute_pr_curve(y_true, scores)
    auprc = compute_auprc(y_true, scores)
    assert auprc >= 0.99

    fpr95 = compute_fpr_at_tpr(y_true, scores, target_tpr=0.95)
    assert fpr95 == 0.0  # Perfect separation means 0 false positives at 95% TPR


def test_metrics_random_separation() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    # Inverted scores: bad model (worse than random)
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
    auroc = compute_auroc(y_true, scores)
    assert auroc == 0.0

    fpr95 = compute_fpr_at_tpr(y_true, scores, target_tpr=0.95)
    assert fpr95 == 1.0


def test_metrics_validation_errors() -> None:
    with pytest.raises(ValueError, match="1D arrays"):
        compute_roc_curve(np.zeros((2, 2)), np.zeros((2, 2)))

    with pytest.raises(ValueError, match="Length of y_true and scores must match"):
        compute_roc_curve(np.array([0, 1]), np.array([0.5]))

    with pytest.raises(ValueError, match="both positive"):
        compute_roc_curve(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))


def test_tristate_confusion_matrix() -> None:
    gt_labels = ["REAL", "REAL", "FAKE", "FAKE", "OOD", "OOD", "OOD"]
    decisions = ["REAL", "UNKNOWN", "FAKE", "REAL", "UNKNOWN", "UNKNOWN", "REAL"]

    mat = compute_tristate_confusion_matrix(gt_labels, decisions)
    assert mat["REAL"]["REAL"] == 1
    assert mat["REAL"]["UNKNOWN"] == 1
    assert mat["FAKE"]["FAKE"] == 1
    assert mat["FAKE"]["REAL"] == 1
    assert mat["OOD"]["UNKNOWN"] == 2
    assert mat["OOD"]["REAL"] == 1
    assert mat["OOD"]["FAKE"] == 0


def test_compute_risk_coverage() -> None:
    from forgeoutlier.eval.metrics import compute_risk_coverage
    errors = np.array([0, 0, 1, 0, 1])
    confs = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    cov, risk, aurc = compute_risk_coverage(errors, confs)
    assert len(cov) == 5
    assert cov[-1] == 1.0
    assert risk[0] == 0.0
    assert aurc > 0.0


def test_benchmark_evaluate_and_exports(tmp_path: Path) -> None:
    # 1. Setup mock pipeline
    feature_dim = 576
    model = DeepfakeModel(feature_dim=feature_dim)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    model_sha256 = compute_file_sha256(checkpoint_path)

    manifest_path = tmp_path / "trusted_checkpoints.json"
    manifest_data = {
        "manifest_version": "1.0.0",
        "checkpoints": [{"filename": "model.pt", "sha256": model_sha256}],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    mu = np.zeros((feature_dim,), dtype=np.float32)
    threshold = 0.50
    ood_path = tmp_path / "ood_profile.npz"
    ood_sha256 = save_ood_artifact(
        file_path=str(ood_path),
        method="cosine",
        mu=mu,
        threshold=threshold,
    )

    pipeline = load_pipeline(
        checkpoint_path=checkpoint_path,
        ood_profile_path=ood_path,
        manifest_path=manifest_path,
        ood_expected_sha256=ood_sha256,
        policy=Policy.OUTLIER_FIRST,
    )

    # 2. Create small dataset of images
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    Image.new("RGB", (224, 224), color=(50, 50, 50)).save(img1)
    Image.new("RGB", (224, 224), color=(200, 200, 200)).save(img2)

    summary = evaluate_dataset(
        pipeline=pipeline,
        image_paths=[str(img1), str(img2)],
        labels=["REAL", "OOD"],
    )

    assert isinstance(summary, BenchmarkSummary)
    assert summary.total_samples == 2
    assert summary.id_count == 1
    assert summary.ood_count == 1
    assert 0.0 <= summary.auroc <= 1.0

    # 3. Test exports
    json_out = tmp_path / "summary.json"
    export_summary_json(summary, str(json_out))
    assert json_out.exists()
    with open(json_out, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
        assert loaded_json["total_samples"] == 2

    csv_matrix_out = tmp_path / "confusion.csv"
    export_confusion_matrix_csv(summary.confusion_matrix, str(csv_matrix_out))
    assert csv_matrix_out.exists()

    csv_pred_out = tmp_path / "predictions.csv"
    preds = [
        {"path": str(img1), "decision": "REAL", "reason": "CONFIDENT", "prob_fake": 0.1, "prob_real": 0.9, "ood_score": 0.2, "is_outlier": False, "threshold": 0.5},
        {"path": str(img2), "decision": "UNKNOWN", "reason": "OOD", "prob_fake": 0.1, "prob_real": 0.9, "ood_score": 0.8, "is_outlier": True, "threshold": 0.5},
    ]
    export_predictions_csv(preds, str(csv_pred_out))
    assert csv_pred_out.exists()
