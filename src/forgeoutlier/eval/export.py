"""Evaluation results export to JSON, CSV, and summary reports."""

import csv
import json
from pathlib import Path
from typing import Any

from forgeoutlier.eval.benchmark import BenchmarkSummary


def export_summary_json(summary: BenchmarkSummary | dict[str, Any], output_path: str) -> None:
    """Export benchmark summary metrics to formatted JSON."""
    data = summary.to_dict() if isinstance(summary, BenchmarkSummary) else summary
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_confusion_matrix_csv(
    confusion_matrix: dict[str, dict[str, int]],
    output_path: str,
) -> None:
    """Export Tri-State Confusion Matrix to standard tabular CSV."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    columns = ["GROUND_TRUTH", "PRED_REAL", "PRED_FAKE", "PRED_UNKNOWN"]
    rows = []
    for gt in ["REAL", "FAKE", "OOD"]:
        counts = confusion_matrix.get(gt, {"REAL": 0, "FAKE": 0, "UNKNOWN": 0})
        rows.append(
            [
                gt,
                counts.get("REAL", 0),
                counts.get("FAKE", 0),
                counts.get("UNKNOWN", 0),
            ]
        )

    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def export_predictions_csv(
    predictions: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Export individual prediction rows to CSV."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "path",
        "decision",
        "reason",
        "prob_fake",
        "prob_real",
        "ood_score",
        "is_outlier",
        "threshold",
    ]

    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)
