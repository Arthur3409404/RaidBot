"""Datasets and label recovery for champion icon recognition."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        """Minimal import-time fallback when torch is unavailable."""

        def __class_getitem__(cls, item):
            return cls


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ChampionRecognitionError(RuntimeError):
    """Raised when the champion recognition pipeline cannot continue."""


@dataclass(frozen=True)
class ChampionRecord:
    """One labeled champion image."""

    path: Path
    label: str
    champion_name: str


class ChampionIndex:
    """Maps normalized labels to readable champion names."""

    def __init__(self, label_to_name: dict[str, str]) -> None:
        if not label_to_name:
            raise ChampionRecognitionError("No champion labels found")
        self.label_to_name = dict(sorted(label_to_name.items()))
        self.name_to_label = {name: label for label, name in self.label_to_name.items()}
        self.labels = tuple(self.label_to_name)

    @classmethod
    def from_labels_csv(cls, labels_csv_path: Path, reference_folder: Path | None = None) -> "ChampionIndex":
        """Load labels from the collector CSV or infer them from reference files."""
        if labels_csv_path.exists():
            table = pd.read_csv(labels_csv_path)
            required = {"label", "champion_name"}
            missing = required.difference(table.columns)
            if missing:
                raise ChampionRecognitionError(
                    f"Labels CSV is missing required columns: {sorted(missing)}"
                )
            pairs = {
                str(row["label"]): str(row["champion_name"])
                for row in table.to_dict(orient="records")
            }
            return cls(pairs)

        if reference_folder is None:
            raise ChampionRecognitionError(f"Labels CSV not found: {labels_csv_path}")

        files = list_image_files(reference_folder)
        return cls({path.stem: prettify_label(path.stem) for path in files})

    def recover_label(self, image_path: Path, dataset_root: Path) -> str:
        """Recover the champion label for a clean or augmented icon path."""
        parent_label = image_path.parent.name
        if image_path.parent != dataset_root and parent_label in self.label_to_name:
            return parent_label

        stem = normalize_label_like(image_path.stem)
        if stem in self.label_to_name:
            return stem

        stripped = strip_augmentation_suffix(stem)
        if stripped in self.label_to_name:
            return stripped

        for label in sorted(self.labels, key=len, reverse=True):
            if stem == label or stem.startswith(f"{label}_") or stem.startswith(f"{label}-"):
                return label

        raise ChampionRecognitionError(
            f"Could not recover champion label for {image_path}. "
            "Use filenames starting with the clean label, or place variants in label-named folders."
        )

    def champion_name(self, label: str) -> str:
        """Return the readable champion name for a normalized label."""
        return self.label_to_name[label]


class IconDataset(Dataset):
    """Torch dataset of champion icon images."""

    def __init__(
        self,
        records: list[ChampionRecord],
        label_to_index: dict[str, int],
        transform,
    ) -> None:
        if not records:
            raise ChampionRecognitionError("No image records found")
        self.records = records
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image = load_rgb_image(record.path)
        return self.transform(image), self.label_to_index[record.label]


def load_records(folder: Path, champion_index: ChampionIndex) -> list[ChampionRecord]:
    """Load all image records in a folder, recovering labels from path names."""
    if not folder.exists():
        raise ChampionRecognitionError(f"Image folder not found: {folder}")
    records: list[ChampionRecord] = []
    for path in list_image_files(folder):
        label = champion_index.recover_label(path, folder)
        records.append(ChampionRecord(path=path, label=label, champion_name=champion_index.champion_name(label)))
    return records


def list_image_files(folder: Path) -> list[Path]:
    """Return all image files below a folder in stable order."""
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_rgb_image(path: Path) -> Image.Image:
    """Load an image with EXIF correction and RGB conversion."""
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ChampionRecognitionError(f"Could not read image {path}: {exc}") from exc


def normalize_label_like(value: str) -> str:
    """Normalize a file stem into collector-style label text."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def strip_augmentation_suffix(stem: str) -> str:
    """Remove common augmentation suffixes while preserving champion labels."""
    patterns: Iterable[str] = (
        r"(_aug(?:mented)?|_noise(?:d)?|_variant|_copy)?_\d{1,4}$",
        r"[-_](?:aug|noise|noised|variant|blur|crop|jpeg|overlay)[-_]?\d{0,4}$",
    )
    stripped = stem
    for pattern in patterns:
        stripped = re.sub(pattern, "", stripped)
    return stripped


def prettify_label(label: str) -> str:
    """Create a readable name when no labels CSV exists."""
    return " ".join(part.capitalize() for part in normalize_label_like(label).split("_"))
