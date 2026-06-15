"""Configuration for prototype-based champion icon recognition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecognitionConfig:
    """End-to-end settings for training, evaluation, and prediction."""

    train_image_folder: Path = Path("data/processed/icons_noised")
    reference_icon_folder: Path = Path("data/processed/icons")
    labels_csv_path: Path = Path("data/processed/labels.csv")
    backbone: str = "convnext_tiny"
    image_size: int = 224
    batch_size: int = 32
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    epochs: int = 10
    fine_tune_epochs: int = 0
    unfreeze_last_blocks: int = 1
    checkpoint_path: Path = Path("data/models/champion_recognition/checkpoints/champion_icon_recognizer.pt")
    prototype_path: Path = Path("data/models/champion_recognition/checkpoints/champion_prototypes.pt")
    log_dir: Path = Path("data/models/champion_recognition/logs")
    plot_dir: Path = Path("data/models/champion_recognition/plots")
    review_samples_dir: Path = Path("data/models/champion_recognition/review_samples")
    random_seed: int = 42
    benchmark_class_count: int = 0
    benchmark_seed: int = 42
    min_similarity_threshold: float = 0.65
    min_margin_threshold: float = 0.02
    pretrained: bool = True
    use_balanced_sampling: bool = True
    num_workers: int = 0
    device: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = value.as_posix()
        return data

    @classmethod
    def from_data_dir(cls, data_dir: Path, **overrides: Any) -> "RecognitionConfig":
        """Build a config rooted at a dataset directory."""
        processed = data_dir / "processed"
        defaults: dict[str, Any] = {
            "train_image_folder": processed / "icons_noised",
            "reference_icon_folder": processed / "icons",
            "labels_csv_path": processed / "labels.csv",
        }
        defaults.update(overrides)
        return cls(**defaults)
