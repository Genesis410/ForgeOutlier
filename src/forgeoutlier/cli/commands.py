"""CLI command implementations and dispatching."""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from forgeoutlier import load_pipeline
from forgeoutlier.cli.formatters import format_box, format_json, format_tsv, format_tsv_header
from forgeoutlier.cli.parser import build_parser
from forgeoutlier.cli.progress import StderrProgressBar
from forgeoutlier.core.exceptions import ForgeOutlierException
from forgeoutlier.core.types import Decision, Policy
from forgeoutlier.eval.benchmark import evaluate_dataset
from forgeoutlier.media.image import load_image
from forgeoutlier.media.video import VideoStream
from forgeoutlier.models.loader import compute_file_sha256, load_checkpoint
from forgeoutlier.ood.cosine import CosineCentroidScorer
from forgeoutlier.ood.fitting import (
    calibrate_quantile_threshold,
    fit_centroid_and_covariance,
)
from forgeoutlier.ood.mahalanobis import MahalanobisScorer
from forgeoutlier.ood.serialization import save_ood_artifact


def handle_predict(args: Any) -> int:
    """Execute single prediction command."""
    if args.policy == "dual_score":
        pol = Policy.DUAL_SCORE
    elif args.policy == "baseline_naive":
        pol = Policy.BASELINE_NAIVE
    else:
        pol = Policy.OUTLIER_FIRST

    pipeline = load_pipeline(
        checkpoint_path=args.model_path,
        ood_profile_path=args.profile_path,
        manifest_path=args.manifest_path,
        ood_expected_sha256=args.ood_sha256,
        policy=pol,
    )

    if args.threshold is not None:
        pipeline.ood_scorer.threshold = float(args.threshold)

    input_p = Path(args.input_path)
    if not input_p.exists():
        sys.stderr.write(f"Error: input file does not exist: {input_p}\n")
        return 1

    suffix = input_p.suffix.lower()
    if suffix in (".mp4", ".avi", ".mov", ".mkv"):
        res = pipeline.predict_video(str(input_p))
    else:
        res = pipeline.predict_image(str(input_p))

    # Determine output format
    if args.json:
        sys.stdout.write(format_json(res) + "\n")
    elif args.plain or not sys.stdout.isatty():
        sys.stdout.write(format_tsv_header() + "\n")
        sys.stdout.write(format_tsv(res) + "\n")
    else:
        sys.stdout.write(format_box(res, use_color=not args.no_color) + "\n")

    if args.exit_code_decision:
        if res.decision == Decision.REAL:
            return 0
        elif res.decision == Decision.FAKE:
            return 2
        elif res.decision == Decision.UNKNOWN:
            return 3

    return 0


def handle_batch_predict(args: Any) -> int:
    """Execute batch prediction across multiple files."""
    if args.policy == "dual_score":
        pol = Policy.DUAL_SCORE
    elif args.policy == "baseline_naive":
        pol = Policy.BASELINE_NAIVE
    else:
        pol = Policy.OUTLIER_FIRST

    pipeline = load_pipeline(
        checkpoint_path=args.model_path,
        ood_profile_path=args.profile_path,
        manifest_path=args.manifest_path,
        ood_expected_sha256=args.ood_sha256,
        policy=pol,
    )

    all_files: list[Path] = []
    for item in args.inputs:
        p = Path(item)
        if p.is_dir():
            for f in p.glob("*"):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".mp4"):
                    all_files.append(f)
        elif p.exists():
            all_files.append(p)

    if not all_files:
        sys.stderr.write("No valid media files found to process.\n")
        return 1

    progress = StderrProgressBar(total=len(all_files), prefix="Evaluating Media")
    results = []

    if not args.json and (args.plain or not sys.stdout.isatty()):
        sys.stdout.write(format_tsv_header() + "\n")

    for f in all_files:
        suffix = f.suffix.lower()
        if suffix in (".mp4", ".avi", ".mov", ".mkv"):
            res = pipeline.predict_video(str(f))
        else:
            res = pipeline.predict_image(str(f))
        results.append(res)
        progress.update(1)

        if not args.json:
            if args.plain or not sys.stdout.isatty():
                sys.stdout.write(format_tsv(res) + "\n")
            else:
                sys.stdout.write(format_box(res, use_color=True) + "\n")

    progress.finish()

    if args.json:
        dict_list = [r.to_dict() for r in results]
        sys.stdout.write(json.dumps(dict_list, indent=2) + "\n")

    return 0


def handle_fit_ood(args: Any) -> int:
    """Fit OOD statistical baseline from saved embeddings (.npy)."""
    emb_path = Path(args.embeddings_path)
    if not emb_path.exists():
        sys.stderr.write(f"Error: embeddings file not found: {emb_path}\n")
        return 1

    embeddings = np.load(str(emb_path), allow_pickle=False)
    if embeddings.ndim != 2:
        sys.stderr.write(f"Error: expected 2D embeddings (N, D), got {embeddings.shape}\n")
        return 1

    if args.method == "mahalanobis":
        mu, cov, _ = fit_centroid_and_covariance(embeddings)
        scorer = MahalanobisScorer(mu=mu, cov=cov, threshold=0.0)
    else:
        mu = np.mean(embeddings, axis=0).astype(np.float32)
        scorer = CosineCentroidScorer(mu=mu, threshold=0.0)

    scores = scorer.score(embeddings)
    threshold = calibrate_quantile_threshold(
        scores, percentile=args.quantile * 100.0 if args.quantile <= 1.0 else args.quantile
    )

    out_p = Path(args.output_path)
    sha256 = save_ood_artifact(
        file_path=str(out_p),
        method=args.method,
        mu=mu,
        threshold=threshold,
        cov=cov if args.method == "mahalanobis" else None,
    )

    sys.stdout.write(
        f"OOD profile fitted successfully.\nMethod: {args.method}\nSamples: {len(embeddings)}\nThreshold ({args.quantile}): {threshold:.4f}\nSHA256: {sha256}\nOutput: {out_p}\n"
    )
    return 0


def handle_inspect_model(args: Any) -> int:
    """Inspect model checkpoint SHA-256."""
    p = Path(args.checkpoint_path)
    if not p.exists():
        sys.stderr.write(f"Error: checkpoint path does not exist: {p}\n")
        return 1

    sha256 = compute_file_sha256(p)
    size_mb = p.stat().st_size / (1024 * 1024)

    sys.stdout.write(f"Checkpoint: {p.name}\nSize: {size_mb:.2f} MB\nSHA256: {sha256}\n")
    return 0


def handle_extract_embeddings(args: Any) -> int:
    """Extract feature embeddings from images or media directory into .npy or .npz."""
    input_p = Path(args.input_path)
    if not input_p.exists():
        sys.stderr.write(f"Error: input path does not exist: {input_p}\n")
        return 1

    if args.model_path is None:
        from forgeoutlier.models.model import DeepfakeModel

        model = DeepfakeModel()
    else:
        model = load_checkpoint(
            checkpoint_path=args.model_path,
            manifest_path=args.manifest_path,
            verify_hash=True,
            device="cpu",
        )
    model.eval()

    embeddings_list: list[np.ndarray] = []
    if input_p.is_file():
        suffix = input_p.suffix.lower()
        if suffix in (".mp4", ".avi", ".mov", ".mkv"):
            with VideoStream(str(input_p), max_frames=100) as vs:
                for _, frame_bgr in vs.frames():
                    rgb = frame_bgr[..., ::-1].copy()
                    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                    norm_chw = (chw - mean) / std
                    feat = model.extract_features(np.expand_dims(norm_chw, axis=0))
                    embeddings_list.append(feat[0])
        else:
            chw, _ = load_image(str(input_p), target_size=(224, 224))
            feat = model.extract_features(np.expand_dims(chw, axis=0))
            embeddings_list.append(feat[0])
    elif input_p.is_dir():
        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = [p for p in input_p.iterdir() if p.suffix.lower() in image_exts]
        if not files:
            sys.stderr.write(f"Error: no valid images found in directory: {input_p}\n")
            return 1
        for f in files:
            chw, _ = load_image(str(f), target_size=(224, 224))
            feat = model.extract_features(np.expand_dims(chw, axis=0))
            embeddings_list.append(feat[0])

    if not embeddings_list:
        sys.stderr.write("Error: no embeddings could be extracted.\n")
        return 1

    embs_array = np.vstack(embeddings_list).astype(np.float32)
    out_p = Path(args.output_path)
    if out_p.suffix.lower() == ".npz":
        np.savez_compressed(out_p, embeddings=embs_array)
    else:
        np.save(out_p, embs_array)

    sys.stdout.write(
        f"Extracted {len(embs_array)} embeddings with shape {embs_array.shape} to {out_p}\n"
    )
    return 0


def handle_evaluate(args: Any) -> int:
    """Evaluate pipeline on labeled dataset with ground truth."""
    manifest_p = Path(args.manifest_path)
    if not manifest_p.exists():
        sys.stderr.write(f"Error: evaluation manifest does not exist: {manifest_p}\n")
        return 1

    with open(manifest_p, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        sys.stderr.write("Error: manifest must be a JSON array of items {path, label}.\n")
        return 1

    if args.policy == "dual_score":
        pol = Policy.DUAL_SCORE
    elif args.policy == "baseline_naive":
        pol = Policy.BASELINE_NAIVE
    else:
        pol = Policy.OUTLIER_FIRST

    if args.model_path is None:
        from forgeoutlier.models.model import DeepfakeModel
        from forgeoutlier.ood.mahalanobis import MahalanobisScorer
        from forgeoutlier.pipeline.inference import InferencePipeline

        model = DeepfakeModel()
        ood_scorer = MahalanobisScorer(
            mu=np.zeros(576, dtype=np.float32),
            cov=np.eye(576, dtype=np.float32),
            threshold=100.0,
        )
        pipeline = InferencePipeline(
            model=model,
            ood_scorer=ood_scorer,
            policy=pol,
        )
    else:
        pipeline = load_pipeline(
            checkpoint_path=args.model_path,
            ood_profile_path=args.profile_path,
            policy=pol,
        )

    image_paths = [item["path"] for item in data]
    labels = [item["label"] for item in data]

    summary = evaluate_dataset(pipeline, image_paths, labels)
    summary_dict = summary.to_dict()

    if args.output_json:
        out_f = Path(args.output_json)
        with open(out_f, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

    if args.json:
        sys.stdout.write(json.dumps(summary_dict, indent=2) + "\n")
    else:
        sys.stdout.write(
            f"Evaluation Complete (N={summary.total_samples}):\n"
            f"AUROC: {summary.auroc:.4f}\n"
            f"AUPRC: {summary.auprc:.4f}\n"
            f"FPR@95% TPR: {summary.fpr_at_95_tpr:.4f}\n"
            f"Abstained (OOD->UNKNOWN): {summary.abstained_rate_ood * 100:.1f}%\n"
            f"Fatal OOD Error (OOD->REAL): {summary.fatal_error_rate_ood * 100:.1f}%\n"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.subcommand == "predict":
            return handle_predict(args)
        elif args.subcommand == "batch-predict":
            return handle_batch_predict(args)
        elif args.subcommand == "fit-ood":
            return handle_fit_ood(args)
        elif args.subcommand == "inspect-model":
            return handle_inspect_model(args)
        elif args.subcommand == "extract-embeddings":
            return handle_extract_embeddings(args)
        elif args.subcommand == "evaluate":
            return handle_evaluate(args)
        else:
            parser.print_help(sys.stderr)
            return 1
    except ForgeOutlierException as e:
        sys.stderr.write(f"ForgeOutlier Error [{type(e).__name__}]: {e}\n")
        if args.debug:
            import traceback

            traceback.print_exc(file=sys.stderr)
        return 1
    except Exception as e:
        sys.stderr.write(f"System Error: {e}\n")
        if args.debug:
            import traceback

            traceback.print_exc(file=sys.stderr)
        return 1
