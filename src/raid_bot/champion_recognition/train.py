"""Training entry point for the prototype-based champion recognizer."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any
import logging
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import RecognitionConfig
from .dataset import ChampionIndex, ChampionRecognitionError, ChampionRecord, IconDataset, load_records
from .evaluation import evaluate_classifier_records, evaluate_recognizer, write_threshold_report
from .model import EmbeddingClassifier, IconTransform, set_seed
from .prototypes import build_prototypes, build_prototypes_from_records
from .runtime import get_device


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingSummary:
    """Paths and metrics from a training run."""

    checkpoint_path: Path
    prototype_path: Path
    log_path: Path
    num_train_images: int
    num_reference_images: int
    num_classes: int
    final_loss: float
    evaluation: dict[str, Any]
    best_val_loss_checkpoint_path: Path | None = None
    best_val_loss_prototype_path: Path | None = None
    best_val_accuracy_checkpoint_path: Path | None = None
    best_val_accuracy_prototype_path: Path | None = None


def train_recognizer(config: RecognitionConfig, *, force: bool = False) -> TrainingSummary:
    """Train classifier head first, optionally fine-tune the last backbone blocks, and save artifacts."""
    set_seed(config.random_seed)
    ensure_outputs(config)
    if config.checkpoint_path.exists() and not force:
        raise ChampionRecognitionError(f"Refusing to overwrite checkpoint without --force: {config.checkpoint_path}")

    LOGGER.info("Loading dataset records")
    champion_index = ChampionIndex.from_labels_csv(config.labels_csv_path, config.reference_icon_folder)
    train_records = load_records(config.train_image_folder, champion_index)
    reference_records = load_records(config.reference_icon_folder, champion_index)
    champion_index, train_records, reference_records = _apply_benchmark_subset(
        config,
        champion_index,
        train_records,
        reference_records,
    )
    label_to_index = {label: index for index, label in enumerate(sorted(champion_index.labels))}
    LOGGER.info("Training set: %d images across %d classes", len(train_records), len(label_to_index))
    LOGGER.info("Reference set: %d images", len(reference_records))

    transform = IconTransform(config.image_size)
    dataset = IconDataset(train_records, label_to_index, transform)
    loader = build_train_loader(dataset, train_records, config)

    device = get_device(config)
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Loading backbone: %s", config.backbone)
    model = EmbeddingClassifier(config.backbone, len(label_to_index), pretrained=config.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    best_snapshots: dict[str, dict[str, Any]] = {"loss": {}, "accuracy": {}}

    model.freeze_backbone()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=config.learning_rate)
    LOGGER.info("Training classifier head for %d epoch(s)", config.epochs)
    history = run_epochs(
        model,
        loader,
        criterion,
        optimizer,
        device,
        config.epochs,
        stage="head",
        config=config,
        champion_index=champion_index,
        validation_records=reference_records,
        label_to_index=label_to_index,
        best_snapshots=best_snapshots,
    )

    if config.fine_tune_epochs > 0:
        LOGGER.info(
            "Fine-tuning last %d backbone block(s) for %d epoch(s)",
            config.unfreeze_last_blocks,
            config.fine_tune_epochs,
        )
        model.unfreeze_last_blocks(config.unfreeze_last_blocks)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.fine_tune_learning_rate,
        )
        history.extend(
            run_epochs(
                model,
                loader,
                criterion,
                optimizer,
                device,
                config.fine_tune_epochs,
                stage="fine_tune",
                config=config,
            )
        )

    LOGGER.info("Saving checkpoint")
    save_checkpoint(config, model, label_to_index, history)
    LOGGER.info("Building prototypes")
    prototype_bank = build_prototypes(model, config, champion_index, device)
    prototype_bank.save(config.prototype_path, {"config": config.to_json_dict(), "label_to_index": label_to_index})
    LOGGER.info("Running evaluation")
    evaluation = evaluate_recognizer(config, model=model, prototype_bank=prototype_bank, champion_index=champion_index)
    write_threshold_report(config, evaluation["predictions"])

    log_path = config.log_dir / "training_summary.json"
    log_path.write_text(
        json.dumps(
            {
                "config": config.to_json_dict(),
                "history": history,
                "evaluation": evaluation,
                "best_snapshots": best_snapshots,
                "num_train_images": len(train_records),
                "num_reference_images": len(reference_records),
                "num_classes": len(label_to_index),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return TrainingSummary(
        checkpoint_path=config.checkpoint_path,
        prototype_path=config.prototype_path,
        log_path=log_path,
        num_train_images=len(train_records),
        num_reference_images=len(reference_records),
        num_classes=len(label_to_index),
        final_loss=float(history[-1]["loss"]),
        evaluation=evaluation,
        best_val_loss_checkpoint_path=Path(best_snapshots["loss"]["checkpoint_path"])
        if "checkpoint_path" in best_snapshots["loss"]
        else None,
        best_val_loss_prototype_path=Path(best_snapshots["loss"]["prototype_path"])
        if "prototype_path" in best_snapshots["loss"]
        else None,
        best_val_accuracy_checkpoint_path=Path(best_snapshots["accuracy"]["checkpoint_path"])
        if "checkpoint_path" in best_snapshots["accuracy"]
        else None,
        best_val_accuracy_prototype_path=Path(best_snapshots["accuracy"]["prototype_path"])
        if "prototype_path" in best_snapshots["accuracy"]
        else None,
    )


def _apply_benchmark_subset(
    config: RecognitionConfig,
    champion_index: ChampionIndex,
    train_records: list,
    reference_records: list,
) -> tuple[ChampionIndex, list, list]:
    """Optionally restrict training to a small reproducible set of champions."""
    class_count = int(getattr(config, "benchmark_class_count", 0) or 0)
    if class_count <= 0:
        return champion_index, train_records, reference_records

    unique_train_labels = sorted({record.label for record in train_records})
    if class_count > len(unique_train_labels):
        raise ChampionRecognitionError(
            f"benchmark_class_count={class_count} exceeds the available training classes ({len(unique_train_labels)})"
        )

    rng = random.Random(int(getattr(config, "benchmark_seed", config.random_seed)))
    selected_labels = sorted(rng.sample(unique_train_labels, class_count))
    selected_label_set = set(selected_labels)
    filtered_train_records = [record for record in train_records if record.label in selected_label_set]
    filtered_reference_records = [record for record in reference_records if record.label in selected_label_set]
    filtered_index = ChampionIndex({label: champion_index.champion_name(label) for label in selected_labels})

    LOGGER.info(
        "Benchmark mode enabled: %d classes selected with seed %d",
        class_count,
        int(getattr(config, "benchmark_seed", config.random_seed)),
    )
    LOGGER.info("Benchmark classes: %s", ", ".join(selected_labels))
    if not filtered_train_records or not filtered_reference_records:
        raise ChampionRecognitionError("Benchmark subset produced no training or reference records")
    return filtered_index, filtered_train_records, filtered_reference_records


def run_epochs(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epochs: int,
    *,
    stage: str,
    config: RecognitionConfig,
    champion_index: ChampionIndex | None = None,
    validation_records: list[ChampionRecord] | None = None,
    label_to_index: dict[str, int] | None = None,
    best_snapshots: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run training epochs and return compact history."""
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        progress = _ProgressBar(len(loader), f"{stage} epoch {epoch}/{epochs}", enabled=sys.stderr.isatty())
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            total += batch_size
            progress.update(batch_index, total_correct=total_correct, total_seen=total, batch_loss=float(loss.item()))
        progress.close()
        history.append(
            {
                "stage": stage,
                "epoch": epoch,
                "loss": total_loss / total,
                "accuracy": total_correct / total,
            }
        )
        if stage == "head" and validation_records is not None:
            if champion_index is None:
                raise ChampionRecognitionError("Validation requires champion_index")
            if label_to_index is None:
                raise ChampionRecognitionError("Validation requires label_to_index")
            validation_summary = validate_epoch(
                model,
                config,
                champion_index,
                validation_records,
                criterion,
                label_to_index,
            )
            history[-1].update({f"val_{key}": value for key, value in validation_summary.items()})
            if best_snapshots is not None:
                _maybe_save_best_snapshots(
                    model,
                    config,
                    history,
                    champion_index,
                    validation_records,
                    label_to_index,
                    epoch,
                    validation_summary,
                    best_snapshots,
                )
    return history


def build_train_loader(dataset: IconDataset, records, config: RecognitionConfig):
    """Create a train loader, optionally balancing classes by inverse frequency."""
    sampler = None
    shuffle = True
    if config.use_balanced_sampling:
        counts = Counter(record.label for record in records)
        weights = [1.0 / counts[record.label] for record in records]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
    )


def save_checkpoint(config: RecognitionConfig, model, label_to_index: dict[str, int], history: list[dict[str, Any]]) -> None:
    """Save enough state for prediction and future prototype rebuilding."""
    save_checkpoint_to_path(config, model, label_to_index, history, config.checkpoint_path)


def save_checkpoint_to_path(
    config: RecognitionConfig,
    model,
    label_to_index: dict[str, int],
    history: list[dict[str, Any]],
    checkpoint_path: Path,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Save a checkpoint to an explicit path."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_index": label_to_index,
            "index_to_label": {str(index): label for label, index in label_to_index.items()},
            "backbone": config.backbone,
            "image_size": config.image_size,
            "history": history,
            "config": config.to_json_dict(),
            "extra_metadata": extra_metadata or {},
        },
        checkpoint_path,
    )


def ensure_outputs(config: RecognitionConfig) -> None:
    """Create output directories."""
    for path in (
        config.checkpoint_path.parent,
        config.prototype_path.parent,
        config.log_dir,
        config.plot_dir,
        config.review_samples_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def validate_epoch(
    model,
    config: RecognitionConfig,
    champion_index: ChampionIndex,
    validation_records: list[ChampionRecord],
    criterion,
    label_to_index: dict[str, int],
) -> dict[str, Any]:
    """Score validation loss and accuracy on the current validation set."""
    classifier_summary = evaluate_classifier_records(
        config,
        model=model,
        criterion=criterion,
        records=validation_records,
        label_to_index=label_to_index,
    )
    LOGGER.info(
        "Validation: loss=%.4f acc=%.3f",
        classifier_summary["loss"] if classifier_summary["loss"] is not None else float("nan"),
        classifier_summary["accuracy"] if classifier_summary["accuracy"] is not None else float("nan"),
    )
    return {
        "loss": classifier_summary["loss"],
        "accuracy": classifier_summary["accuracy"],
    }


def _maybe_save_best_snapshots(
    model,
    config: RecognitionConfig,
    history: list[dict[str, Any]],
    champion_index: ChampionIndex,
    validation_records: list[ChampionRecord],
    label_to_index: dict[str, int],
    epoch: int,
    validation_summary: dict[str, Any],
    best_snapshots: dict[str, dict[str, Any]],
) -> None:
    """Persist best-loss and best-accuracy snapshots during head training."""
    device = next(model.parameters()).device
    snapshot_root = config.checkpoint_path.parent

    def save_snapshot(kind: str, metric_value: float, checkpoint_name: str, prototype_name: str) -> None:
        checkpoint_path = snapshot_root / checkpoint_name
        prototype_path = snapshot_root / prototype_name
        save_checkpoint_to_path(
            config,
            model,
            label_to_index,
            history,
            checkpoint_path,
            extra_metadata={
                "kind": kind,
                "epoch": epoch,
                "validation_summary": validation_summary,
            },
        )
        prototype_bank = build_prototypes_from_records(model, config, champion_index, device, validation_records)
        prototype_bank.save(
            prototype_path,
            {
                "config": config.to_json_dict(),
                "label_to_index": label_to_index,
                "kind": kind,
                "epoch": epoch,
                "validation_summary": validation_summary,
            },
        )
        best_snapshots[kind] = {
            "epoch": epoch,
            "metric_value": metric_value,
            "checkpoint_path": checkpoint_path.as_posix(),
            "prototype_path": prototype_path.as_posix(),
            "validation_summary": validation_summary,
        }

    loss_value = validation_summary["loss"]
    if loss_value is not None:
        current_best = best_snapshots["loss"].get("metric_value")
        if current_best is None or loss_value < current_best:
            LOGGER.info("New best validation loss %.4f at head epoch %d", loss_value, epoch)
            save_snapshot(
                "loss",
                float(loss_value),
                f"{config.checkpoint_path.stem}_best_val_loss.pt",
                f"{config.prototype_path.stem}_best_val_loss.pt",
            )

    accuracy_value = validation_summary["accuracy"]
    if accuracy_value is not None:
        current_best = best_snapshots["accuracy"].get("metric_value")
        if current_best is None or accuracy_value > current_best:
            LOGGER.info("New best validation accuracy %.4f at head epoch %d", accuracy_value, epoch)
            save_snapshot(
                "accuracy",
                float(accuracy_value),
                f"{config.checkpoint_path.stem}_best_val_accuracy.pt",
                f"{config.prototype_path.stem}_best_val_accuracy.pt",
            )


class _ProgressBar:
    """Render compact batch progress to stderr, with logging fallback when not interactive."""

    def __init__(self, total: int, label: str, *, enabled: bool) -> None:
        self.total = max(1, total)
        self.label = label
        self.enabled = enabled
        self._last_percent = -1
        self._finished = False

    def update(self, batch_index: int, *, total_correct: int, total_seen: int, batch_loss: float) -> None:
        if not self.enabled:
            if batch_index == 1 or batch_index == self.total or batch_index % max(1, self.total // 5) == 0:
                LOGGER.info(
                    "%s: %d/%d batches, loss=%.4f, acc=%.3f",
                    self.label,
                    batch_index,
                    self.total,
                    batch_loss,
                    total_correct / max(1, total_seen),
                )
            return

        percent = int((batch_index * 100) / self.total)
        if percent == self._last_percent and batch_index != self.total:
            return
        self._last_percent = percent
        width = 28
        filled = int(width * batch_index / self.total)
        bar = "#" * filled + "-" * (width - filled)
        accuracy = total_correct / max(1, total_seen)
        sys.stderr.write(
            f"\r{self.label}: [{bar}] {percent:3d}% ({batch_index}/{self.total}) "
            f"loss={batch_loss:.4f} acc={accuracy:.3f}"
        )
        sys.stderr.flush()
        if batch_index == self.total:
            self._finished = True
            sys.stderr.write("\n")
            sys.stderr.flush()

    def close(self) -> None:
        if self.enabled and not self._finished:
            sys.stderr.write("\n")
            sys.stderr.flush()
