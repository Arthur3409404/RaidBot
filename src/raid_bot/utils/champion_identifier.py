from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
from raid_bot.champion_recognition.config import RecognitionConfig
from raid_bot.champion_recognition.dataset import ChampionIndex, prettify_label
from raid_bot.champion_recognition.model import EmbeddingClassifier, IconTransform


CHECKPOINTS_DIR = REPO_ROOT / "data" / "models" / "champion_recognition" / "checkpoints"
DEFAULT_CHECKPOINT_PATH = CHECKPOINTS_DIR / "champion_icon_recognizer_best_val_loss.pt"
FALLBACK_CHECKPOINT_PATH = CHECKPOINTS_DIR / "champion_icon_recognizer.pt"
DEFAULT_PROTOTYPE_PATH = CHECKPOINTS_DIR / "champion_prototypes_best_val_loss.pt"
FALLBACK_PROTOTYPE_PATH = CHECKPOINTS_DIR / "champion_prototypes.pt"
DEFAULT_LABELS_CSV = REPO_ROOT / "data" / "processed" / "labels.csv"
DEFAULT_REFERENCE_ICON_FOLDER = REPO_ROOT / "data" / "processed" / "icons"


class UnavailableChampionIdentifier:
    """Null-object recognizer used when model artifacts are unavailable."""

    is_available = False

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.checkpoint_path: Path | None = None

    def predict_portrait(self, portrait: np.ndarray | Image.Image) -> None:
        return None

    def predict_portraits(
        self,
        portraits: Iterable[np.ndarray | Image.Image] | np.ndarray,
    ) -> list[None]:
        return [None] * len(_coerce_portrait_batch(portraits))

    def predict_portraits_or_one(
        self,
        portraits: np.ndarray | Image.Image | Iterable[np.ndarray | Image.Image],
    ) -> None | list[None]:
        batch = _coerce_portrait_batch(portraits)
        if len(batch) == 1:
            return None
        return [None] * len(batch)


def _coerce_portrait_batch(
    portraits: np.ndarray | Image.Image | Iterable[np.ndarray | Image.Image],
) -> list[np.ndarray | Image.Image]:
    if isinstance(portraits, Image.Image):
        return [portraits]

    if isinstance(portraits, np.ndarray):
        if portraits.ndim in {2, 3}:
            return [portraits]
        if portraits.ndim == 4:
            return [portraits[index] for index in range(portraits.shape[0])]
        raise ValueError(f"Expected a portrait image or portrait batch, got shape {portraits.shape}")

    return list(portraits)


class ChampionIdentifier:
    """Load the best-validation champion recognizer and predict champion names from portrait crops."""

    is_available = True

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        *,
        labels_csv_path: Path | None = None,
        reference_icon_folder: Path | None = None,
        prototype_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        checkpoint_config = checkpoint.get("config") or {}

        self.backbone = str(checkpoint.get("backbone", "convnext_tiny"))
        self.image_size = int(checkpoint.get("image_size", 224))
        self.index_to_label = {
            int(str(index)): str(label)
            for index, label in dict(checkpoint["index_to_label"]).items()
        }
        self.prototype_path = self._resolve_prototype_path(self.checkpoint_path, prototype_path)
        self.labels_csv_path = self._resolve_optional_path(
            labels_csv_path,
            checkpoint_config.get("labels_csv_path"),
            DEFAULT_LABELS_CSV,
        )
        self.reference_icon_folder = self._resolve_optional_path(
            reference_icon_folder,
            checkpoint_config.get("reference_icon_folder"),
            DEFAULT_REFERENCE_ICON_FOLDER,
        )

        config = RecognitionConfig(
            backbone=self.backbone,
            image_size=self.image_size,
            checkpoint_path=self.checkpoint_path,
            labels_csv_path=self.labels_csv_path,
            reference_icon_folder=self.reference_icon_folder,
            min_similarity_threshold=0.0,
            min_margin_threshold=0.0,
            device=str(self.device),
        )
        self.config = config
        self.champion_index = self._load_champion_index()
        self.model = EmbeddingClassifier(self.backbone, len(self.index_to_label), pretrained=False).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = IconTransform(self.image_size)

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: Path | None) -> Path:
        if checkpoint_path is not None:
            resolved = Path(checkpoint_path)
            if not resolved.exists():
                raise FileNotFoundError(f"Champion recognizer checkpoint not found: {resolved}")
            return resolved
        if DEFAULT_CHECKPOINT_PATH.exists():
            return DEFAULT_CHECKPOINT_PATH
        best_val_checkpoints = sorted(
            (
                path
                for path in CHECKPOINTS_DIR.glob("*_best_val_loss.pt")
                if "prototype" not in path.stem
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if best_val_checkpoints:
            return best_val_checkpoints[0]
        if FALLBACK_CHECKPOINT_PATH.exists():
            return FALLBACK_CHECKPOINT_PATH
        raise FileNotFoundError(
            "Could not find a champion recognizer checkpoint. Expected one of: "
            f"{DEFAULT_CHECKPOINT_PATH} or {FALLBACK_CHECKPOINT_PATH}"
        )

    @staticmethod
    def _resolve_optional_path(
        explicit_path: Path | None,
        checkpoint_config_path: str | None,
        default_path: Path,
    ) -> Path:
        if explicit_path is not None:
            return Path(explicit_path)
        if checkpoint_config_path:
            config_path = Path(checkpoint_config_path)
            if not config_path.is_absolute():
                config_path = REPO_ROOT / config_path
            if config_path.exists():
                return config_path
        return default_path

    @staticmethod
    def _resolve_prototype_path(checkpoint_path: Path, prototype_path: Path | None) -> Path | None:
        if prototype_path is not None:
            resolved = Path(prototype_path)
            return resolved if resolved.exists() else None

        checkpoint_name = checkpoint_path.name
        candidates = [
            DEFAULT_PROTOTYPE_PATH,
            checkpoint_path.with_name(checkpoint_name.replace("icon_recognizer", "prototypes")),
            checkpoint_path.with_name(checkpoint_name.replace("recognizer", "prototypes")),
            checkpoint_path.with_name(checkpoint_name.replace("recognizer", "prototype")),
            FALLBACK_PROTOTYPE_PATH,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        prototype_matches = sorted(
            {candidate for candidate in CHECKPOINTS_DIR.glob("*prototype*.pt")} | {candidate for candidate in CHECKPOINTS_DIR.glob("*prototypes*.pt")},
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in prototype_matches:
            if candidate.exists():
                return candidate
        return None

    def _load_champion_index(self) -> ChampionIndex:
        labels_csv = Path(self.config.labels_csv_path)
        reference_folder = Path(self.config.reference_icon_folder)
        if labels_csv.exists():
            return ChampionIndex.from_labels_csv(labels_csv, reference_folder)
        if self.prototype_path is not None and self.prototype_path.exists():
            prototype_data = torch.load(self.prototype_path, map_location="cpu")
            labels = [str(label) for label in prototype_data.get("labels", [])]
            champion_names = [str(name) for name in prototype_data.get("champion_names", [])]
            if labels and len(labels) == len(champion_names):
                return ChampionIndex(dict(zip(labels, champion_names)))
        return ChampionIndex({label: prettify_label(label) for label in self.index_to_label.values()})

    @staticmethod
    def _to_pil(image: np.ndarray | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        elif array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
            array = np.transpose(array, (1, 2, 0))

        if array.ndim != 3:
            raise ValueError(f"Expected a portrait image, got shape {array.shape}")

        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.shape[-1] != 3:
            raise ValueError(f"Expected an RGB portrait image, got shape {array.shape}")

        if array.dtype != np.uint8:
            arr = np.asarray(array, dtype=np.float32)
            if arr.max(initial=0.0) <= 1.5:
                arr = arr * 255.0
            array = np.clip(arr, 0.0, 255.0).astype(np.uint8)

        return Image.fromarray(array, mode="RGB")

    def _label_to_name(self, label: str) -> str:
        try:
            return self.champion_index.champion_name(label)
        except Exception:
            return prettify_label(label)

    def predict_portrait(self, portrait: np.ndarray | Image.Image) -> str | None:
        """Predict one champion name from one portrait crop."""
        return self.predict_portraits([portrait])[0]

    def predict_portraits(
        self,
        portraits: Iterable[np.ndarray | Image.Image] | np.ndarray,
    ) -> list[str | None]:
        """Predict champion names for multiple portrait crops in a single batch."""
        portraits = _coerce_portrait_batch(portraits)
        if not portraits:
            return []

        tensors = [self.transform(self._to_pil(p)) for p in portraits]
        batch = torch.stack(tensors, dim=0).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            predicted_indices = torch.argmax(logits, dim=1).tolist()

        names: list[str | None] = []
        for predicted_index in predicted_indices:
            label = self.index_to_label.get(int(predicted_index))
            if label is None:
                names.append(None)
                continue
            names.append(self._label_to_name(label))
        return names

    def predict_portraits_or_one(
        self,
        portraits: np.ndarray | Image.Image | Iterable[np.ndarray | Image.Image],
    ) -> str | None | list[str | None]:
        """Convenience wrapper that preserves single-input vs batch behavior."""
        batch = _coerce_portrait_batch(portraits)
        predictions = self.predict_portraits(batch)
        if len(batch) == 1:
            return predictions[0]
        return predictions

    def predict(
        self,
        portraits: np.ndarray | Image.Image | Iterable[np.ndarray | Image.Image],
    ) -> str | None | list[str | None]:
        """Predict one portrait or a batch while preserving input cardinality."""
        return self.predict_portraits_or_one(portraits)


@lru_cache(maxsize=1)
def load_default_champion_identifier() -> ChampionIdentifier:
    """Load and cache the default champion identifier once at startup."""
    return ChampionIdentifier()
