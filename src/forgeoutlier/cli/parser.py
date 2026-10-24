"""Command-line argument parsing for ForgeOutlier CLI."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build root CLI parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="forgeoutlier",
        description="ForgeOutlier-Lite: Open-Set Deepfake Detection & OOD Triage CLI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
        help="Display version string and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Display full traceback diagnostics to stderr on failure.",
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # 1. predict
    p_predict = subparsers.add_parser("predict", help="Predict single image or video file.")
    p_predict.add_argument(
        "input_path", type=str, help="Path to image (.png, .jpg) or video (.mp4) file."
    )
    p_predict.add_argument(
        "--model-path", type=str, default=None, help="Path to PyTorch checkpoint (.pt)."
    )
    p_predict.add_argument(
        "--profile-path", type=str, default=None, help="Path to OOD profile (.npz)."
    )
    p_predict.add_argument(
        "--manifest-path", type=str, default=None, help="Path to trusted checkpoints manifest JSON."
    )
    p_predict.add_argument(
        "--ood-sha256", type=str, default=None, help="Expected SHA-256 for OOD profile."
    )
    p_predict.add_argument(
        "--policy",
        type=str,
        choices=["outlier_first", "dual_score", "baseline_naive"],
        default="outlier_first",
        help="Tri-state triage policy.",
    )
    p_predict.add_argument(
        "--threshold", type=float, default=None, help="Override calibrated OOD threshold."
    )
    p_predict.add_argument(
        "--crop-face", action="store_true", help="Crop face before running inference."
    )
    p_predict.add_argument("--json", action="store_true", help="Output pure JSON to stdout.")
    p_predict.add_argument(
        "--plain", action="store_true", help="Force TSV output without ANSI styling."
    )
    p_predict.add_argument("--no-color", action="store_true", help="Disable ANSI color codes.")
    p_predict.add_argument(
        "--exit-code-decision",
        action="store_true",
        help="Semantic exit codes: 0=REAL, 2=FAKE, 3=UNKNOWN.",
    )

    # 2. batch-predict
    p_batch = subparsers.add_parser(
        "batch-predict", help="Batch predict directory or manifest of media files."
    )
    p_batch.add_argument("inputs", nargs="+", help="Files or directory containing media.")
    p_batch.add_argument(
        "--model-path", type=str, default=None, help="Path to PyTorch checkpoint (.pt)."
    )
    p_batch.add_argument(
        "--profile-path", type=str, default=None, help="Path to OOD profile (.npz)."
    )
    p_batch.add_argument(
        "--manifest-path", type=str, default=None, help="Path to trusted checkpoints manifest JSON."
    )
    p_batch.add_argument(
        "--ood-sha256", type=str, default=None, help="Expected SHA-256 for OOD profile."
    )
    p_batch.add_argument(
        "--policy",
        type=str,
        choices=["outlier_first", "dual_score", "baseline_naive"],
        default="outlier_first",
        help="Tri-state triage policy.",
    )
    p_batch.add_argument("--json", action="store_true", help="Output JSON array to stdout.")
    p_batch.add_argument(
        "--plain", action="store_true", help="Force TSV output without ANSI styling."
    )

    # 3. fit-ood
    p_fit = subparsers.add_parser(
        "fit-ood", help="Fit OOD statistical baseline on In-Distribution embeddings."
    )
    p_fit.add_argument(
        "--embeddings-path", type=str, required=True, help="Input .npy embeddings file."
    )
    p_fit.add_argument("--output-path", type=str, required=True, help="Output .npz profile path.")
    p_fit.add_argument(
        "--method",
        type=str,
        choices=["mahalanobis", "cosine"],
        default="mahalanobis",
        help="Scorer method.",
    )
    p_fit.add_argument(
        "--quantile", type=float, default=0.95, help="Quantile threshold (default 0.95)."
    )

    # 4. inspect-model
    p_inspect = subparsers.add_parser(
        "inspect-model", help="Inspect PyTorch checkpoint metadata and SHA-256."
    )
    p_inspect.add_argument("checkpoint_path", type=str, help="Path to checkpoint file.")

    # 5. extract-embeddings
    p_extract = subparsers.add_parser(
        "extract-embeddings", help="Extract feature embeddings from images or video."
    )
    p_extract.add_argument("input_path", type=str, help="Input directory of images or media file.")
    p_extract.add_argument(
        "--output-path", type=str, required=True, help="Output .npy or .npz file path."
    )
    p_extract.add_argument(
        "--model-path", type=str, default=None, help="Path to PyTorch checkpoint (.pt)."
    )
    p_extract.add_argument(
        "--manifest-path", type=str, default=None, help="Path to trusted checkpoints manifest JSON."
    )

    # 6. evaluate
    p_eval = subparsers.add_parser(
        "evaluate", help="Evaluate pipeline on labeled dataset with ground truth."
    )
    p_eval.add_argument(
        "--manifest-path", type=str, required=True, help="Evaluation manifest JSON or CSV."
    )
    p_eval.add_argument(
        "--model-path", type=str, default=None, help="Path to PyTorch checkpoint (.pt)."
    )
    p_eval.add_argument(
        "--profile-path", type=str, default=None, help="Path to OOD profile (.npz)."
    )
    p_eval.add_argument(
        "--output-json", type=str, default=None, help="Optional output JSON report path."
    )
    p_eval.add_argument(
        "--policy",
        type=str,
        choices=["outlier_first", "dual_score", "baseline_naive"],
        default="outlier_first",
        help="Tri-state triage policy.",
    )
    p_eval.add_argument(
        "--json", action="store_true", help="Output JSON evaluation summary to stdout."
    )

    return parser
