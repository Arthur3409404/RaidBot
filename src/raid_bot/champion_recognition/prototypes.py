"""Clean-icon prototype embedding creation and nearest-neighbor search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import RecognitionConfig
from .dataset import ChampionIndex, ChampionRecord, IconDataset, load_records
from .model import IconTransform
from .runtime import load_trained_model


@dataclass(frozen=True)
class PrototypeBank:
    """Clean champion reference embeddings."""

    labels: list[str]
    champion_names: list[str]
    embeddings: torch.Tensor

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        """Persist the prototype bank."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "labels": self.labels,
                "champion_names": self.champion_names,
                "embeddings": self.embeddings.cpu(),
                "metadata": metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "PrototypeBank":
        """Load a prototype bank."""
        data = torch.load(path, map_location=device)
        return cls(
            labels=list(data["labels"]),
            champion_names=list(data["champion_names"]),
            embeddings=data["embeddings"].to(device),
        )


def build_prototypes(model, config: RecognitionConfig, champion_index: ChampionIndex, device: torch.device) -> PrototypeBank:
    """Build one normalized embedding prototype per clean champion icon."""
    records = load_records(config.reference_icon_folder, champion_index)
    return build_prototypes_from_records(model, config, champion_index, device, records)


def build_prototypes_from_records(
    model,
    config: RecognitionConfig,
    champion_index: ChampionIndex,
    device: torch.device,
    records: list[ChampionRecord],
) -> PrototypeBank:
    """Build normalized prototypes from an explicit record list."""
    label_to_records: dict[str, list[ChampionRecord]] = {}
    for record in records:
        label_to_records.setdefault(record.label, []).append(record)

    transform = IconTransform(config.image_size)
    ordered_labels = sorted(label_to_records)
    prototype_vectors: list[torch.Tensor] = []
    champion_names: list[str] = []
    model.eval()
    with torch.no_grad():
        for label in ordered_labels:
            dataset = IconDataset(label_to_records[label], {label: 0}, transform)
            loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
            embeddings: list[torch.Tensor] = []
            for images, _ in loader:
                embeddings.append(model(images.to(device), return_embedding=True))
            prototype = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)
            prototype = torch.nn.functional.normalize(prototype, p=2, dim=1).squeeze(0)
            prototype_vectors.append(prototype)
            champion_names.append(champion_index.champion_name(label))

    return PrototypeBank(
        labels=ordered_labels,
        champion_names=champion_names,
        embeddings=torch.stack(prototype_vectors).to(device),
    )


def build_and_save_prototypes(config: RecognitionConfig) -> PrototypeBank:
    """Build clean champion prototypes from a trained checkpoint and save them."""
    model, _, device = load_trained_model(config)
    champion_index = ChampionIndex.from_labels_csv(config.labels_csv_path, config.reference_icon_folder)
    bank = build_prototypes(model, config, champion_index, device)
    bank.save(config.prototype_path, {"config": config.to_json_dict()})
    return bank


def nearest_prototype(query_embedding: torch.Tensor, bank: PrototypeBank) -> tuple[int, float, float]:
    """Return best index, best cosine similarity, and best-vs-second margin."""
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.unsqueeze(0)
    similarities = torch.matmul(query_embedding, bank.embeddings.T).squeeze(0)
    topk = torch.topk(similarities, k=min(2, similarities.numel()))
    best_similarity = float(topk.values[0].item())
    second_similarity = float(topk.values[1].item()) if topk.values.numel() > 1 else -1.0
    return int(topk.indices[0].item()), best_similarity, best_similarity - second_similarity
