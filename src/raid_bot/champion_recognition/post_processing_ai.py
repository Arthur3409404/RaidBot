"""Post-training analysis and threshold recommendation for the recognizer."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import RecognitionConfig
from .dataset import ChampionIndex
from .evaluation import evaluate_recognizer
from .prototypes import PrototypeBank, build_prototypes
from .runtime import load_trained_model


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PostProcessingReport:
    """Summary of post-training evaluation and threshold recommendations."""

    report_path: Path
    threshold_report_path: Path
    evaluation_path: Path
    current_top1_accuracy: float
    current_accepted_accuracy: float | None
    current_none_rate: float
    recommended_similarity_threshold: float
    recommended_margin_threshold: float
    recommended_similarity_metrics: dict[str, Any]
    recommended_margin_metrics: dict[str, Any]
    margin_statistics: dict[str, Any]


def analyze_post_training(config: RecognitionConfig) -> PostProcessingReport:
    """Evaluate saved weights and recommend threshold values for the final algorithm."""
    model, _, device = load_trained_model(config)
    champion_index = ChampionIndex.from_labels_csv(config.labels_csv_path, config.reference_icon_folder)
    prototype_bank = _load_or_build_prototypes(config, model, champion_index, device)

    evaluation = evaluate_recognizer(
        config,
        model=model,
        prototype_bank=prototype_bank,
        champion_index=champion_index,
    )
    predictions = evaluation["predictions"]
    margin_candidates = _candidate_thresholds(row["margin"] for row in predictions)
    similarity_candidates = _candidate_thresholds(row["similarity"] for row in predictions)

    margin_sweep = _sweep_thresholds(
        predictions,
        axis="margin",
        candidates=margin_candidates,
        fixed_similarity=config.min_similarity_threshold,
    )
    similarity_sweep = _sweep_thresholds(
        predictions,
        axis="similarity",
        candidates=similarity_candidates,
        fixed_margin=config.min_margin_threshold,
    )

    margin_statistics = _margin_statistics(predictions)
    threshold_report_path = _write_threshold_report(config, margin_sweep, similarity_sweep)
    report = {
        "config": config.to_json_dict(),
        "current_metrics": {
            "top1_accuracy": evaluation["top1_accuracy"],
            "accepted_accuracy": evaluation["accepted_accuracy"],
            "none_rate": evaluation["none_rate"],
            "wrong_accepted_predictions": evaluation["wrong_accepted_predictions"],
        },
        "recommended_thresholds": {
            "similarity_threshold": similarity_sweep["recommended_threshold"],
            "similarity_metrics": similarity_sweep["recommended_metrics"],
            "margin_threshold": margin_sweep["recommended_threshold"],
            "margin_metrics": margin_sweep["recommended_metrics"],
        },
        "margin_statistics": margin_statistics,
        "current_thresholds": {
            "min_similarity_threshold": config.min_similarity_threshold,
            "min_margin_threshold": config.min_margin_threshold,
        },
    }
    report_path = config.log_dir / "post_processing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return PostProcessingReport(
        report_path=report_path,
        threshold_report_path=threshold_report_path,
        evaluation_path=config.log_dir / "evaluation.json",
        current_top1_accuracy=float(evaluation["top1_accuracy"]),
        current_accepted_accuracy=None if evaluation["accepted_accuracy"] is None else float(evaluation["accepted_accuracy"]),
        current_none_rate=float(evaluation["none_rate"]),
        recommended_similarity_threshold=float(similarity_sweep["recommended_threshold"]),
        recommended_margin_threshold=float(margin_sweep["recommended_threshold"]),
        recommended_similarity_metrics=similarity_sweep["recommended_metrics"],
        recommended_margin_metrics=margin_sweep["recommended_metrics"],
        margin_statistics=margin_statistics,
    )


def _load_or_build_prototypes(
    config: RecognitionConfig,
    model,
    champion_index: ChampionIndex,
    device: torch.device,
) -> PrototypeBank:
    """Load saved prototypes when available, otherwise build them from clean icons."""
    if config.prototype_path.exists():
        return PrototypeBank.load(config.prototype_path, device)
    bank = build_prototypes(model, config, champion_index, device)
    bank.save(config.prototype_path, {"config": config.to_json_dict()})
    return bank


def _candidate_thresholds(values) -> list[float]:
    """Build a threshold sweep grid from observed scores."""
    numbers = sorted({float(value) for value in values})
    if not numbers:
        return [0.0]
    span = numbers[-1] - numbers[0]
    epsilon = max(1e-6, span * 1e-6)
    return [numbers[0] - epsilon, *numbers, numbers[-1] + epsilon]


def _sweep_thresholds(
    predictions: list[dict[str, Any]],
    *,
    axis: str,
    candidates: list[float],
    fixed_similarity: float | None = None,
    fixed_margin: float | None = None,
) -> dict[str, Any]:
    """Sweep a single threshold axis and choose the best operating point."""
    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        if axis == "margin":
            metrics = _score_predictions(predictions, fixed_similarity=fixed_similarity, margin_threshold=threshold)
        else:
            metrics = _score_predictions(predictions, similarity_threshold=threshold, fixed_margin=fixed_margin)
        row = {"axis": axis, "threshold": threshold, **metrics}
        rows.append(row)

    best = min(
        rows,
        key=lambda row: (
            row["wrong_accepted_predictions"],
            -row["accepted_count"],
            -(row["accepted_accuracy"] if row["accepted_accuracy"] is not None else -1.0),
            row["threshold"],
        ),
    )
    return {
        "rows": rows,
        "recommended_threshold": best["threshold"],
        "recommended_metrics": {key: best[key] for key in _sweep_fieldnames()},
    }


def _score_predictions(
    predictions: list[dict[str, Any]],
    *,
    similarity_threshold: float | None = None,
    margin_threshold: float | None = None,
    fixed_similarity: float | None = None,
    fixed_margin: float | None = None,
) -> dict[str, Any]:
    """Score predictions for a given acceptance rule."""
    if similarity_threshold is None:
        similarity_threshold = fixed_similarity
    if margin_threshold is None:
        margin_threshold = fixed_margin

    accepted = []
    for row in predictions:
        if similarity_threshold is not None and row["similarity"] < similarity_threshold:
            continue
        if margin_threshold is not None and row["margin"] < margin_threshold:
            continue
        accepted.append(row)

    wrong_accepted = [row for row in accepted if not row["correct"]]
    accepted_accuracy = _mean(row["correct"] for row in accepted)
    return {
        "accepted_count": len(accepted),
        "accepted_accuracy": accepted_accuracy,
        "none_rate": 1.0 - (len(accepted) / len(predictions) if predictions else 0.0),
        "wrong_accepted_predictions": len(wrong_accepted),
    }


def _margin_statistics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize margin distributions for correct and incorrect predictions."""
    correct_margins = [float(row["margin"]) for row in predictions if row["correct"]]
    wrong_margins = [float(row["margin"]) for row in predictions if not row["correct"]]
    return {
        "correct": _distribution(correct_margins),
        "wrong": _distribution(wrong_margins),
        "all": _distribution([float(row["margin"]) for row in predictions]),
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    """Compute compact distribution statistics."""
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": _percentile(ordered, 10),
        "median": _percentile(ordered, 50),
        "p90": _percentile(ordered, 90),
        "max": ordered[-1],
    }


def _percentile(values: list[float], percentile: float) -> float:
    """Compute a simple percentile for sorted values."""
    if not values:
        raise ValueError("percentile requires non-empty values")
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * (percentile / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def _mean(values) -> float | None:
    """Return the arithmetic mean or None for empty inputs."""
    values = list(values)
    if not values:
        return None
    return float(sum(values) / len(values))


def _sweep_fieldnames() -> list[str]:
    """Return the stable metric keys included in threshold recommendations."""
    return ["accepted_count", "accepted_accuracy", "none_rate", "wrong_accepted_predictions"]


def _write_threshold_report(
    config: RecognitionConfig,
    margin_sweep: dict[str, Any],
    similarity_sweep: dict[str, Any],
) -> Path:
    """Write a combined threshold sweep report for later inspection."""
    path = config.log_dir / "post_processing_thresholds.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["axis", "threshold", *_sweep_fieldnames()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in margin_sweep["rows"]:
            writer.writerow(row)
        for row in similarity_sweep["rows"]:
            writer.writerow(row)
    return path
