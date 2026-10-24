"""Formatters for CLI output (JSON, TSV, ANSI Box)."""

import json

from forgeoutlier.core.types import PredictionResult


def format_json(result: PredictionResult) -> str:
    """Format PredictionResult as strict indented JSON."""
    return json.dumps(result.to_dict(), indent=2)


def format_tsv_header() -> str:
    """Format TSV header line according to D029."""
    return "media_path\tmedia_type\tdecision\tprob_fake\tood_score\tthreshold\treason"


def format_tsv(result: PredictionResult) -> str:
    """Format PredictionResult as single TSV line according to D029."""
    cols = [
        str(result.media.path),
        str(result.media.media_type),
        str(result.decision.value),
        f"{result.scores.prob_fake:.4f}",
        f"{result.scores.ood_score:.4f}",
        f"{result.scores.threshold:.4f}",
        str(result.reason),
    ]
    return "\t".join(cols)


def format_box(result: PredictionResult, use_color: bool = True) -> str:
    """Format PredictionResult in interactive ANSI Box format."""
    decision_str = result.decision.value
    if use_color:
        c_green = "\033[92m"
        c_red = "\033[91m"
        c_yellow = "\033[93m"
        c_bold = "\033[1m"
        c_reset = "\033[0m"
        if decision_str == "REAL":
            display_decision = f"{c_green}{c_bold}REAL{c_reset}"
        elif decision_str == "FAKE":
            display_decision = f"{c_red}{c_bold}FAKE{c_reset}"
        else:
            display_decision = f"{c_yellow}{c_bold}UNKNOWN (OOD){c_reset}"
    else:
        display_decision = decision_str

    lines = [
        "┌────────────────────────────────────────────────────────────┐",
        f"│ MEDIA: {str(result.media.path):<51} │",
        f"│ TYPE:  {str(result.media.media_type):<51} │",
        "├────────────────────────────────────────────────────────────┤",
        f"│ DECISION:   {display_decision:<46} │",
        f"│ REASON:     {str(result.reason):<46} │",
        f"│ PROB_FAKE:  {result.scores.prob_fake:<46.4f} │",
        f"│ PROB_REAL:  {result.scores.prob_real:<46.4f} │",
        f"│ OOD_SCORE:  {result.scores.ood_score:<46.4f} │",
        f"│ THRESHOLD:  {result.scores.threshold:<46.4f} │",
        f"│ IS_OUTLIER: {str(result.scores.is_outlier):<46} │",
        "└────────────────────────────────────────────────────────────┘",
    ]
    return "\n".join(lines)
