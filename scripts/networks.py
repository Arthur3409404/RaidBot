# -*- coding: utf-8 -*-
"""Train name-based Classic Arena and Tag Team Arena evaluation models."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from raid_bot.handlers.ai_networks_handler import (
    ClassicCompositionEvaluationNetwork,
    DEFAULT_CHAMPION_LABELS_CSV,
    TagTeamCompositionEvaluationNetwork,
)


DEFAULT_DATASETS = {
    "classic": Path("data/database_champions/datasets/enemy_dataset_classic_arena.npz"),
    "tagteam": Path("data/database_champions/datasets/enemy_dataset_tagteam_arena.npz"),
}
DEFAULT_OUTPUTS = {
    "classic": Path("data/models/neural_networks/enemy_eval_classic_arena/composition_model"),
    "tagteam": Path("data/models/neural_networks/enemy_eval_tagteam_arena/composition_model"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train name-based arena evaluation networks")
    parser.add_argument("--mode", choices=("classic", "tagteam"), default="classic")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--labels-csv", default=str(DEFAULT_CHAMPION_LABELS_CSV))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_path = Path(args.dataset_path) if args.dataset_path else DEFAULT_DATASETS[args.mode]
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else DEFAULT_OUTPUTS[args.mode]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model_cls = ClassicCompositionEvaluationNetwork if args.mode == "classic" else TagTeamCompositionEvaluationNetwork
    model = model_cls(labels_csv_path=args.labels_csv)
    model.train_network(
        dataset_path=str(dataset_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=str(checkpoint_path),
        val_split=args.val_split,
        device=args.device,
    )
    torch.save(model.state_dict(), f"{checkpoint_path}.pt")
    print(f"Saved final {args.mode} composition model: {checkpoint_path}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
