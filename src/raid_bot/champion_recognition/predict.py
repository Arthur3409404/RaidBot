"""User-facing champion prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .confidence import is_confident_match
from .config import RecognitionConfig
from .dataset import ChampionIndex, load_rgb_image
from .model import IconTransform
from .prototypes import PrototypeBank, build_prototypes, nearest_prototype
from .runtime import load_trained_model


@dataclass(frozen=True)
class PredictionDebug:
    """Internal prediction details for evaluation and debugging."""

    champion_name: str | None
    label: str | None
    similarity: float
    margin: float
    accepted: bool


class ChampionPredictor:
    """Load a trained backbone and clean prototypes for inference."""

    def __init__(self, config: RecognitionConfig) -> None:
        self.config = config
        self.model, _, self.device = load_trained_model(config)
        self.champion_index = ChampionIndex.from_labels_csv(config.labels_csv_path, config.reference_icon_folder)
        self.prototype_bank = self._load_or_build_prototypes()
        self.transform = IconTransform(config.image_size)

    def predict_champion(self, image_path: str | Path) -> str | None:
        """Return exactly one champion name or None."""
        return self.predict_debug(image_path).champion_name

    def predict_debug(self, image_path: str | Path) -> PredictionDebug:
        """Return prediction plus confidence details for evaluation."""
        image = load_rgb_image(Path(image_path))
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            embedding = self.model(tensor, return_embedding=True)
        best_index, similarity, margin = nearest_prototype(embedding, self.prototype_bank)
        accepted = is_confident_match(similarity, margin, self.config)
        if not accepted:
            return PredictionDebug(None, None, similarity, margin, False)
        return PredictionDebug(
            champion_name=self.prototype_bank.champion_names[best_index],
            label=self.prototype_bank.labels[best_index],
            similarity=similarity,
            margin=margin,
            accepted=True,
        )

    def _load_or_build_prototypes(self) -> PrototypeBank:
        if self.config.prototype_path.exists():
            return PrototypeBank.load(self.config.prototype_path, self.device)
        bank = build_prototypes(self.model, self.config, self.champion_index, self.device)
        bank.save(self.config.prototype_path, {"config": self.config.to_json_dict()})
        return bank


def predict_champion(image_path: str | Path, config: RecognitionConfig | None = None) -> str | None:
    """Convenience function returning one champion name or None."""
    return ChampionPredictor(config or RecognitionConfig()).predict_champion(image_path)
