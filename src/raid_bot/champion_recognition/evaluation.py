"""Evaluation and threshold calibration for champion icon recognition."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .confidence import is_confident_match
from .config import RecognitionConfig
from .dataset import ChampionIndex, ChampionRecord, IconDataset, load_records, load_rgb_image
from .model import IconTransform
from .prototypes import PrototypeBank, nearest_prototype
from .runtime import load_trained_model


def evaluate_recognizer(
    config: RecognitionConfig,
    *,
    model,
    prototype_bank: PrototypeBank,
    champion_index: ChampionIndex,
) -> dict[str, Any]:
    """Evaluate nearest-prototype predictions on clean reference icons."""
    records = load_records(config.reference_icon_folder, champion_index)
    return evaluate_records(config, model=model, prototype_bank=prototype_bank, records=records)


def evaluate_records(
    config: RecognitionConfig,
    *,
    model,
    prototype_bank: PrototypeBank,
    records,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Evaluate nearest-prototype predictions on an explicit record list."""
    device = next(model.parameters()).device
    transform = IconTransform(config.image_size)
    predictions: list[dict[str, Any]] = []
    correct_similarities: list[float] = []
    wrong_similarities: list[float] = []
    wrong_pairs: Counter[tuple[str, str]] = Counter()

    model.eval()
    with torch.no_grad():
        for record in records:
            image = load_rgb_image(record.path)
            tensor = transform(image).unsqueeze(0).to(device)
            embedding = model(tensor, return_embedding=True)
            best_index, similarity, margin = nearest_prototype(embedding, prototype_bank)
            predicted_label = prototype_bank.labels[best_index]
            accepted = is_confident_match(similarity, margin, config)
            correct = predicted_label == record.label
            row = {
                "path": record.path.as_posix(),
                "label": record.label,
                "champion_name": record.champion_name,
                "predicted_label": predicted_label,
                "predicted_champion_name": prototype_bank.champion_names[best_index],
                "similarity": similarity,
                "margin": margin,
                "accepted": accepted,
                "correct": correct,
            }
            predictions.append(row)
            if correct:
                correct_similarities.append(similarity)
            else:
                wrong_similarities.append(similarity)
                wrong_pairs[(record.label, predicted_label)] += 1

    accepted_predictions = [row for row in predictions if row["accepted"]]
    wrong_accepted = [row for row in accepted_predictions if not row["correct"]]
    summary = {
        "top1_accuracy": mean([row["correct"] for row in predictions]),
        "accepted_accuracy": mean([row["correct"] for row in accepted_predictions]),
        "none_rate": 1.0 - (len(accepted_predictions) / len(predictions) if predictions else 0.0),
        "average_correct_similarity": mean(correct_similarities),
        "average_wrong_similarity": mean(wrong_similarities),
        "wrong_accepted_predictions": len(wrong_accepted),
        "most_common_wrong_predictions": [
            {"label": label, "predicted_label": predicted, "count": count}
            for (label, predicted), count in wrong_pairs.most_common(20)
        ],
        "low_confidence_predictions": sorted(
            predictions,
            key=lambda row: (row["accepted"], row["similarity"], row["margin"]),
        )[:50],
        "predictions": predictions,
    }
    if write_outputs:
        write_evaluation_outputs(config, summary)
    return summary


def evaluate_classifier_records(
    config: RecognitionConfig,
    *,
    model,
    criterion,
    records: list[ChampionRecord],
    label_to_index: dict[str, int],
) -> dict[str, Any]:
    """Evaluate classifier loss and accuracy on an explicit record list."""
    device = next(model.parameters()).device
    transform = IconTransform(config.image_size)
    dataset = IconDataset(records, label_to_index, transform)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    total_loss = 0.0
    total_correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            total += batch_size

    return {
        "loss": total_loss / total if total else None,
        "accuracy": total_correct / total if total else None,
        "num_items": total,
    }


def evaluate_saved_recognizer(config: RecognitionConfig) -> dict[str, Any]:
    """Evaluate a saved checkpoint and prototype bank without starting training."""
    from .prototypes import PrototypeBank, build_prototypes

    model, _, device = load_trained_model(config)
    champion_index = ChampionIndex.from_labels_csv(config.labels_csv_path, config.reference_icon_folder)
    if config.prototype_path.exists():
        prototype_bank = PrototypeBank.load(config.prototype_path, device)
    else:
        prototype_bank = build_prototypes(model, config, champion_index, device)
        prototype_bank.save(config.prototype_path, {"config": config.to_json_dict()})
    summary = evaluate_recognizer(
        config,
        model=model,
        prototype_bank=prototype_bank,
        champion_index=champion_index,
    )
    write_threshold_report(config, summary["predictions"])
    return summary


def write_evaluation_outputs(config: RecognitionConfig, summary: dict[str, Any]) -> None:
    """Write logs and review samples for manual inspection."""
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.review_samples_dir.mkdir(parents=True, exist_ok=True)
    (config.log_dir / "evaluation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (config.log_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "path",
            "label",
            "champion_name",
            "predicted_label",
            "predicted_champion_name",
            "similarity",
            "margin",
            "accepted",
            "correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["predictions"]:
            writer.writerow({key: row[key] for key in fieldnames})

    for row in summary["low_confidence_predictions"][:50]:
        source = Path(row["path"])
        suffix = "none" if not row["accepted"] else "wrong"
        target = config.review_samples_dir / f"{suffix}_{source.name}"
        if source.exists() and not target.exists():
            shutil.copy2(source, target)


def write_threshold_report(config: RecognitionConfig, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate several similarity thresholds and save a compact report."""
    thresholds = [round(value / 100, 2) for value in range(50, 96, 5)]
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = [
            row
            for row in predictions
            if is_confident_match(row["similarity"], row["margin"], config, min_similarity_threshold=threshold)
        ]
        wrong_accepted = [row for row in accepted if not row["correct"]]
        rows.append(
            {
                "threshold": threshold,
                "accepted_accuracy": mean([row["correct"] for row in accepted]),
                "none_rate": 1.0 - (len(accepted) / len(predictions) if predictions else 0.0),
                "wrong_accepted_predictions": len(wrong_accepted),
            }
        )

    path = config.log_dir / "threshold_report.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["threshold", "accepted_accuracy", "none_rate", "wrong_accepted_predictions"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def mean(values) -> float | None:
    """Return a float mean for numeric or boolean values."""
    values = list(values)
    if not values:
        return None
    return float(sum(values) / len(values))
