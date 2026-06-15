"""Runtime helpers shared by recognizer training, evaluation, and prediction."""

from __future__ import annotations

from pathlib import Path

import torch

from .config import RecognitionConfig
from .dataset import ChampionRecognitionError
from .model import EmbeddingClassifier


def get_device(config: RecognitionConfig) -> torch.device:
    """Resolve the configured torch device."""
    return torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load a recognizer checkpoint with a clear error when it is missing."""
    if not path.exists():
        raise ChampionRecognitionError(
            f"Checkpoint not found: {path}. Run `uv run python scripts/champion_icons.py train-recognizer` first."
        )
    return torch.load(path, map_location=device)


def load_trained_model(config: RecognitionConfig) -> tuple[EmbeddingClassifier, dict, torch.device]:
    """Create the configured backbone and load trained recognizer weights."""
    device = get_device(config)
    checkpoint = load_checkpoint(config.checkpoint_path, device)
    label_to_index = checkpoint["label_to_index"]
    model = EmbeddingClassifier(
        str(checkpoint.get("backbone", config.backbone)),
        len(label_to_index),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, device
