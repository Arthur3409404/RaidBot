"""Backbone and preprocessing utilities."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from .dataset import ChampionRecognitionError


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class EmbeddingClassifier(nn.Module):
    """A timm image backbone with a classifier head for fine-tuning."""

    def __init__(self, backbone_name: str, num_classes: int, *, pretrained: bool = True) -> None:
        super().__init__()
        try:
            import timm
        except ModuleNotFoundError as exc:
            raise ChampionRecognitionError(
                "The prototype recognizer requires `timm`. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        try:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
        except Exception as exc:
            raise ChampionRecognitionError(
                f"Could not create timm backbone {backbone_name!r}. "
                "If pretrained weights are unavailable offline, rerun with --no-pretrained."
            ) from exc

        self.backbone_name = backbone_name
        self.embedding_dim = int(self.backbone.num_features)
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, images: torch.Tensor, *, return_embedding: bool = False) -> torch.Tensor:
        """Return logits or L2-normalized embeddings."""
        embedding = self.extract_embedding(images)
        if return_embedding:
            return embedding
        return self.classifier(embedding)

    def extract_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized embeddings from the backbone."""
        embedding = self.backbone(images)
        return nn.functional.normalize(embedding, p=2, dim=1)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_last_blocks(self, blocks: int) -> None:
        """Unfreeze the last N top-level backbone children."""
        self.freeze_backbone()
        children = list(self.backbone.children())
        for child in children[-max(1, blocks):]:
            for parameter in child.parameters():
                parameter.requires_grad = True


class IconTransform:
    """Resize a PIL icon and convert it to an ImageNet-normalized tensor."""

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, image):
        image = image.resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array.copy())


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
