"""Script to execute full empirical benchmarking across policies on realistic and hard OOD data.
Generates datasets, fits reference ID OOD profile, executes benchmarks, and logs tables.
"""

import json
from pathlib import Path

import numpy as np
import torch

from forgeoutlier.core.types import Policy
from forgeoutlier.eval.benchmark import evaluate_dataset, evaluate_robustness
from forgeoutlier.eval.dataset_synth import generate_benchmark_dataset
from forgeoutlier.eval.export import (
    export_confusion_matrix_csv,
    export_summary_json,
)
from forgeoutlier.media.image import load_image
from forgeoutlier.models.model import DeepfakeModel
from forgeoutlier.ood.mahalanobis import MahalanobisScorer
from forgeoutlier.pipeline.inference import InferencePipeline, normalize_imagenet_tensor


def run_empirical_benchmark():
    base_dir = Path("benchmarks/empirical_run")
    base_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Generating benchmark dataset (250 images: 100 REAL, 100 FAKE, 50 OOD)...")
    manifest_p = generate_benchmark_dataset(base_dir, seed=1337)
    with open(manifest_p, encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Filter ID Real images for OOD profiling
    id_real_paths = [item["path"] for item in manifest_data if item["domain"] == "id_natural"]

    print(
        f"[2] Extracting embeddings from {len(id_real_paths)} reference ID-real images for OOD calibration..."
    )
    torch.manual_seed(42)
    model = DeepfakeModel(feature_dim=576)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for p in id_real_paths:
            img, _ = load_image(p, target_size=(224, 224))
            norm_t = normalize_imagenet_tensor(img)
            batch = np.expand_dims(norm_t, axis=0)
            feat = model.extract_features(batch)
            feat_norm = feat / np.maximum(np.linalg.norm(feat, axis=-1, keepdims=True), 1e-12)
            embeddings.append(feat_norm.squeeze(0))

    embeddings_arr = np.array(embeddings)
    from forgeoutlier.ood.fitting import calibrate_quantile_threshold, fit_centroid_and_covariance
    from forgeoutlier.ood.serialization import save_ood_artifact

    mu, cov, shrinkage = fit_centroid_and_covariance(embeddings_arr)
    # Raw scorer to compute calibration distances
    temp_scorer = MahalanobisScorer(mu=mu, cov=cov, threshold=100.0)
    train_scores = temp_scorer.score(embeddings_arr)
    tau = calibrate_quantile_threshold(train_scores, percentile=95.0)
    scorer = MahalanobisScorer(mu=mu, cov=cov, threshold=float(tau))
    print(
        f"    Fitted Mahalanobis profile: dim={scorer.feature_dim}, shrinkage={shrinkage:.4f}, tau threshold={scorer.threshold:.4f}"
    )

    # Save fitted profile
    profile_p = base_dir / "id_profile.npz"
    save_ood_artifact(
        file_path=str(profile_p),
        method="mahalanobis",
        mu=mu,
        threshold=float(tau),
        cov=cov,
    )

    all_paths = [item["path"] for item in manifest_data]
    all_labels = [item["label"] for item in manifest_data]

    results = {}
    policies = [
        (Policy.OUTLIER_FIRST, "outlier_first"),
        (Policy.DUAL_SCORE, "dual_score"),
        (Policy.BASELINE_NAIVE, "baseline_naive"),
    ]

    print("\n[3] Executing benchmark across 3 triage policies...")
    for pol, pol_name in policies:
        print(f"    Evaluating policy: {pol_name}...")
        pipeline = InferencePipeline(
            model=model,
            ood_scorer=scorer,
            policy=pol,
        )
        summary = evaluate_dataset(pipeline, all_paths, all_labels)
        results[pol_name] = summary.to_dict()

        # Save individual policy artifacts
        export_summary_json(summary, str(base_dir / f"summary_{pol_name}.json"))
        export_confusion_matrix_csv(
            summary.confusion_matrix, str(base_dir / f"confusion_{pol_name}.csv")
        )

    # Save consolidated results
    results_p = base_dir / "consolidated_benchmark_results.json"
    with open(results_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Phase 4: Robustness evaluation on champion policy
    print("\n[4] Executing Phase 4 robustness evaluation (JPEG Q50, Q30, downscaling 112x112)...")
    champion_pipeline = InferencePipeline(
        model=model, ood_scorer=scorer, policy=Policy.OUTLIER_FIRST
    )
    robustness_metrics = evaluate_robustness(
        champion_pipeline, all_paths, all_labels, output_dir=base_dir
    )
    with open(base_dir / "robustness_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(robustness_metrics, f, indent=2)

    print("\n[5] Empirical Benchmark Completed. Results summary:")
    for pol_name, res in results.items():
        print(f"--- Policy: {pol_name} ---")
        print(f"    AUROC: {res['auroc']}")
        print(f"    AUPRC: {res['auprc']}")
        print(f"    AURC: {res['aurc']}")
        print(f"    FPR@95%TPR: {res['fpr_at_95_tpr']}")
        print(f"    Fatal Error Rate (OOD -> REAL): {res['fatal_error_rate_ood'] * 100:.2f}%")
        print(f"    Abstained Rate (OOD -> UNKNOWN): {res['abstained_rate_ood'] * 100:.2f}%")
        print(f"    Confusion Matrix: {res['confusion_matrix']}")


if __name__ == "__main__":
    run_empirical_benchmark()
