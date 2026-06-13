from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.tagteam_arena_model.utils.data_loading import (
    load_prepared_arrays,
    normalize_powers,
    print_npz_inspection,
    save_json,
)
from training.tagteam_arena_model.utils.metrics import binary_metrics

torch = None
nn = None
DataLoader = None
Subset = None
TagTeamArenaDataset = None
TagTeamArenaModel = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a standalone Tag Team Arena best-of-3 PyTorch model.")
    parser.add_argument("--data_path", default="data/tagteam arena.npz")
    parser.add_argument("--output_dir", default="training/tagteam_arena_model/outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker processes.")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Batches prefetched per worker when num_workers > 0.")
    parser.add_argument("--pin_memory", action="store_true", help="Use pinned host memory for faster CUDA transfers.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience. Defaults to --epochs.")
    parser.add_argument("--val_fraction", type=float, default=0.20, help="Fraction of samples used for validation.")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--split_axis", choices=["width", "height", "auto"], default="width")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_key")
    parser.add_argument("--power_key")
    parser.add_argument("--label_key")
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--grayscale_to_rgb", action="store_true")
    parser.add_argument("--inspect_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--single_file", action="store_true", help="Use only --data_path and ignore numbered shard files.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.patience is None:
        args.patience = args.epochs
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val_fraction must be greater than 0 and less than 1.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def versioned_path(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(2, 10_000):
        candidate = path.with_name(f"{path.stem}_v{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free versioned path for {path}")


def split_indices(n: int, seed: int, val_fraction: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("Need at least 2 samples for a train/validation split.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    train_fraction = 1.0 - val_fraction
    train_end = int(n * train_fraction)
    train_end = min(max(1, train_end), n - 1)
    return indices[:train_end], indices[train_end:]


def make_dataloaders(args: argparse.Namespace):
    prepared = load_prepared_arrays(
        args.data_path,
        image_key=args.image_key,
        power_key=args.power_key,
        label_key=args.label_key,
        split_axis=args.split_axis,
        grayscale_to_rgb=args.grayscale_to_rgb,
        include_shards=not args.single_file,
    )
    print("\nPrepared tensors:")
    print(f"  crops:  {prepared.crops.shape} float32 [N, 3, C, H, W]")
    print(f"  powers: {prepared.powers.shape} float32 [N, 3]")
    print(f"  labels: {prepared.labels.shape} float32 [N]")
    print(f"  keys: image={prepared.image_key!r}, powers={prepared.power_key!r}, labels={prepared.label_key!r}")
    print(f"  source files: {len(prepared.source_files)}")

    train_idx, val_idx = split_indices(len(prepared.labels), args.seed, val_fraction=args.val_fraction)
    train_powers, other_powers, stats = normalize_powers(
        prepared.powers[train_idx],
        prepared.powers[val_idx],
    )
    powers = prepared.powers.copy()
    powers[train_idx] = train_powers
    powers[val_idx] = other_powers[0]

    base_dataset = TagTeamArenaDataset(prepared.crops, powers, prepared.labels, augment=False)
    train_dataset = TagTeamArenaDataset(
        prepared.crops[train_idx],
        powers[train_idx],
        prepared.labels[train_idx],
        augment=not args.no_augmentation,
    )
    val_dataset = Subset(base_dataset, val_idx.tolist())

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    print(f"  split: train={len(train_idx)} validation={len(val_idx)}")
    print(
        f"  dataloader: batch_size={args.batch_size} num_workers={args.num_workers} "
        f"pin_memory={args.pin_memory}"
    )
    return train_loader, val_loader, stats, prepared.crops.shape[2]


def run_epoch(model, loader, criterion, optimizer, device, grad_clip: float | None = None) -> tuple[float, float]:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for crops, powers, labels in loader:
        non_blocking = device.type == "cuda"
        crops = crops.to(device, non_blocking=non_blocking)
        powers = powers.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking).unsqueeze(1)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            final_logits, _ = model(crops, powers)
            loss = criterion(final_logits, labels)
            if optimizer is not None:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        final_p = torch.sigmoid(final_logits.detach())
        total_loss += float(loss.item()) * labels.shape[0]
        total_correct += int(((final_p >= 0.5).float() == labels).sum().item())
        total_count += labels.shape[0]
    return total_loss / total_count, total_correct / total_count


def collect_predictions(model, loader, device):
    model.eval()
    labels_all, final_all, hidden_all = [], [], []
    with torch.no_grad():
        for crops, powers, labels in loader:
            non_blocking = device.type == "cuda"
            final_logits, hidden = model(
                crops.to(device, non_blocking=non_blocking),
                powers.to(device, non_blocking=non_blocking),
            )
            final_p = torch.sigmoid(final_logits)
            labels_all.append(labels.numpy())
            final_all.append(final_p.cpu().numpy().reshape(-1))
            hidden_all.append(hidden.cpu().numpy())
    return np.concatenate(labels_all), np.concatenate(final_all), np.concatenate(hidden_all)


def save_history(history: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_examples(labels: np.ndarray, final_p: np.ndarray, hidden: np.ndarray, path: Path, limit: int = 25) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["true_label", "final_probability", "hidden_p1", "hidden_p2", "hidden_p3"])
        writer.writeheader()
        for i in range(min(limit, len(labels))):
            writer.writerow(
                {
                    "true_label": int(labels[i]),
                    "final_probability": float(final_p[i]),
                    "hidden_p1": float(hidden[i, 0]),
                    "hidden_p2": float(hidden[i, 1]),
                    "hidden_p3": float(hidden[i, 2]),
                }
            )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    try:
        print_npz_inspection(args.data_path, include_shards=not args.single_file)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Data inspection failed: {exc}")
        return 1
    if args.inspect_only:
        return 0

    global torch, nn, DataLoader, Subset, TagTeamArenaDataset, TagTeamArenaModel
    try:
        import torch as torch_module
        from torch import nn as nn_module
        from torch.utils.data import DataLoader as DataLoader_module
        from torch.utils.data import Subset as Subset_module

        from training.tagteam_arena_model.models.tagteam_model import TagTeamArenaModel as TagTeamArenaModel_module
        from training.tagteam_arena_model.utils.data_loading import TagTeamArenaDataset as TagTeamArenaDataset_module
    except ModuleNotFoundError as exc:
        print(f"Training requires PyTorch and its dependencies. Missing module: {exc.name}")
        return 1

    torch = torch_module
    nn = nn_module
    DataLoader = DataLoader_module
    Subset = Subset_module
    TagTeamArenaDataset = TagTeamArenaDataset_module
    TagTeamArenaModel = TagTeamArenaModel_module
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        train_loader, val_loader, norm_stats, in_channels = make_dataloaders(args)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"Data preparation failed: {exc}")
        return 1
    device = torch.device(args.device)
    model = TagTeamArenaModel(in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    first_crops, first_powers, first_labels = next(iter(train_loader))
    with torch.no_grad():
        final_logits, hidden = model(first_crops.to(device), first_powers.to(device))
    print(f"\nForward pass OK: final_logits={tuple(final_logits.shape)}, hidden={tuple(hidden.shape)}")

    if args.dry_run:
        train_loss, train_acc = run_epoch(
            model,
            [next(iter(train_loader))],
            criterion,
            optimizer,
            device,
            args.grad_clip,
        )
        print(f"Dry run training batch OK: loss={train_loss:.4f}, accuracy={train_acc:.4f}")
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = versioned_path(output_dir / "best_tagteam_model.pt")
    last_checkpoint_path = versioned_path(output_dir / "last_tagteam_model.pt")
    history_path = versioned_path(output_dir / "training_history.csv")
    stats_path = versioned_path(output_dir / "normalization_stats.json")
    metrics_path = versioned_path(output_dir / "validation_metrics.json")
    examples_path = versioned_path(output_dir / "validation_example_predictions.csv")

    save_json(norm_stats, stats_path)
    best_val_loss = float("inf")
    best_epoch = 0
    history = []
    epochs_without_improvement = 0
    last_epoch = 0

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "normalization_stats": norm_stats,
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs. Best epoch: {best_epoch}")
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": in_channels,
            "normalization_stats": norm_stats,
            "args": vars(args),
            "last_epoch": last_epoch,
            "last_train_loss": history[-1]["train_loss"],
            "last_train_accuracy": history[-1]["train_accuracy"],
            "last_val_loss": history[-1]["val_loss"],
            "last_val_accuracy": history[-1]["val_accuracy"],
        },
        last_checkpoint_path,
    )

    save_history(history, history_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    labels, final_p, hidden = collect_predictions(model, val_loader, device)
    metrics = binary_metrics(labels, final_p)
    metrics["best_epoch"] = best_epoch
    metrics["split"] = "validation"
    metrics["note"] = "hidden p1/p2/p3 are latent estimates and are not directly supervised."
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    save_examples(labels, final_p, hidden, examples_path)

    print("\nValidation metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nExample predictions (hidden probabilities are latent inspection values only):")
    for i in range(min(5, len(labels))):
        print(
            f"  true={int(labels[i])} final_p={final_p[i]:.4f} "
            f"p1={hidden[i, 0]:.4f} p2={hidden[i, 1]:.4f} p3={hidden[i, 2]:.4f}"
        )
    print("\nSaved artifacts:")
    for path in (checkpoint_path, last_checkpoint_path, history_path, stats_path, metrics_path, examples_path):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
