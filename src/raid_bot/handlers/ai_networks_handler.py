# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 15:36:23 2025

@author: Arthur
"""


from __future__ import annotations

import os
import re
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import lru_cache
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path

from raid_bot.utils.tagteam_portraits import crop_tagteam_portraits
from raid_bot.utils.classic_arena_portraits import crop_classic_arena_portraits


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHAMPION_LABELS_CSV = (
    REPO_ROOT / "data" / "processed" / "labels.csv"
)


def _as_object_vector(values) -> np.ndarray:
    sequence = list(values)
    array = np.empty(len(sequence), dtype=object)
    for index, value in enumerate(sequence):
        array[index] = value
    return array


def _normalize_screenshot_array(screenshot) -> np.ndarray | None:
    if screenshot is None:
        return None

    array = np.asarray(screenshot)
    if array.size == 0:
        return None

    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"Enemy screenshot must be HxW or HxWxC, got shape {array.shape}")
    if array.shape[-1] == 4:
        array = array[:, :, :3]

    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Enemy screenshot must be numeric, got dtype {array.dtype}")

    if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.5:
        array = array * 255.0
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def _empty_screenshot_vector(length: int) -> np.ndarray:
    screenshots = np.empty(int(length), dtype=object)
    screenshots[:] = None
    return screenshots


def _infer_screenshot_available(screenshots: np.ndarray, sample_count: int) -> np.ndarray:
    if screenshots.ndim >= 4:
        return np.ones((sample_count,), dtype=bool)

    available = np.zeros((sample_count,), dtype=bool)
    for index, value in enumerate(screenshots[:sample_count]):
        available[index] = value is not None
    return available


def _normalize_name_lookup(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


class ChampionRowEncoder:
    """
    Encode champion names to stable integer ids based on their row position in labels.csv.

    Id 0 is reserved for unknown / padding.
    The first data row in labels.csv maps to id 1, the second to id 2, and so on.
    """

    def __init__(self, labels_csv_path: str | os.PathLike | None = None):
        self.labels_csv_path = Path(labels_csv_path or DEFAULT_CHAMPION_LABELS_CSV)
        self.name_to_id: dict[str, int] = {}
        self.id_to_name: dict[int, str] = {0: "<UNK>"}
        self._load_labels_csv()

    def _load_labels_csv(self) -> None:
        if not self.labels_csv_path.exists():
            raise FileNotFoundError(f"Champion labels CSV not found: {self.labels_csv_path}")

        with self.labels_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader, start=1):
                champion_name = str(row.get("champion_name", "")).strip()
                champion_label = str(row.get("label", "")).strip()
                if not champion_name and not champion_label:
                    continue

                self.id_to_name[row_index] = champion_name or champion_label or f"Champion{row_index}"
                candidate_keys = {
                    _normalize_name_lookup(champion_name),
                    _normalize_name_lookup(champion_label),
                }
                for candidate in candidate_keys:
                    if candidate:
                        self.name_to_id.setdefault(candidate, row_index)

    @property
    def vocab_size(self) -> int:
        return max(self.id_to_name) + 1

    def encode_name(self, champion_name: object) -> int:
        return self.name_to_id.get(_normalize_name_lookup(champion_name), 0)

    def encode_teamcomposition(
        self,
        teamcomposition,
        *,
        team_size: int | None = None,
    ) -> np.ndarray:
        if teamcomposition is None:
            names = []
        elif isinstance(teamcomposition, np.ndarray):
            names = teamcomposition.tolist()
        else:
            names = list(teamcomposition)

        encoded = [self.encode_name(name) for name in names]
        if team_size is not None:
            if len(encoded) < team_size:
                encoded.extend([0] * (team_size - len(encoded)))
            else:
                encoded = encoded[:team_size]
        return np.asarray(encoded, dtype=np.int64)

    def decode_id(self, champion_id: int) -> str:
        return self.id_to_name.get(int(champion_id), "<UNK>")


@lru_cache(maxsize=4)
def load_champion_row_encoder(labels_csv_path: str | None = None) -> ChampionRowEncoder:
    resolved = str(Path(labels_csv_path or DEFAULT_CHAMPION_LABELS_CSV).resolve())
    return ChampionRowEncoder(resolved)


# -----------------------------
# Dataset Class
# -----------------------------
class EnemyDataset(Dataset):
    """
    Dataset storing enemy team compositions, power values, and labels in .npz files.
    Creates the dataset if it does not exist and appends new entries.
    Optionally supports sharded saving with max_entries_per_file.
    Power can be a single value or a numpy array.
    """

    def __init__(
        self,
        dataset_path,
        use_power=True,
        transform=None,
        max_power=350000.0,
        max_entries_per_file=None,
        labels_csv_path=None,
        use_name_encoding=False,
        team_size=None,
    ):
        self.base_dataset_path = dataset_path
        self.dataset_path = dataset_path
        self.transform = transform
        self.max_power = max_power
        self.use_power = use_power
        self.max_entries_per_file = max_entries_per_file
        self.use_name_encoding = bool(use_name_encoding)
        self.team_size = team_size
        self.name_encoder = None
        if self.use_name_encoding:
            self.name_encoder = load_champion_row_encoder(labels_csv_path)

        if self.max_entries_per_file is not None and self.max_entries_per_file <= 0:
            raise ValueError("max_entries_per_file must be > 0 when provided")

        (
            self._dataset_dir,
            self._dataset_stem,
            self._dataset_ext,
        ) = self._split_dataset_path(dataset_path)

        # Ensure the folder exists
        os.makedirs(self._dataset_dir, exist_ok=True)

        if self.max_entries_per_file is not None:
            self.dataset_path = self._resolve_active_dataset_path()

        self._ensure_dataset_file(self.dataset_path)
        self._load_dataset(self.dataset_path)
        self.team_size = self._resolve_team_size(self.team_size)

    def _split_dataset_path(self, dataset_path):
        dataset_dir = os.path.dirname(dataset_path) or "."
        dataset_filename = os.path.basename(dataset_path)
        dataset_stem, dataset_ext = os.path.splitext(dataset_filename)

        if dataset_ext.lower() != ".npz":
            raise ValueError("dataset_path must end with .npz")

        return dataset_dir, dataset_stem, dataset_ext

    def _dataset_path_for_index(self, index):
        if index == 0:
            filename = f"{self._dataset_stem}{self._dataset_ext}"
        else:
            filename = f"{self._dataset_stem}_{index}{self._dataset_ext}"
        return os.path.join(self._dataset_dir, filename)

    def _dataset_index_from_path(self, path):
        filename = os.path.basename(path)
        pattern = rf"^{re.escape(self._dataset_stem)}(?:_(\d+))?{re.escape(self._dataset_ext)}$"
        match = re.match(pattern, filename)
        if not match:
            return 0
        return int(match.group(1)) if match.group(1) is not None else 0

    def _existing_dataset_indices(self):
        pattern = re.compile(
            rf"^{re.escape(self._dataset_stem)}(?:_(\d+))?{re.escape(self._dataset_ext)}$"
        )
        indices = []
        for filename in os.listdir(self._dataset_dir):
            match = pattern.match(filename)
            if not match:
                continue
            if match.group(1) is None:
                indices.append(0)
            else:
                indices.append(int(match.group(1)))
        return sorted(set(indices))

    def _create_empty_dataset_file(self, path):
        np.savez_compressed(
            path,
            teamcomposition=np.empty((0,), dtype=object),
            powers=np.zeros((0,), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.float32),
            screenshots=_empty_screenshot_vector(0),
            screenshot_available=np.zeros((0,), dtype=bool),
            schema=np.array("teamcomposition_v1"),
        )

    def _ensure_dataset_file(self, path):
        if not os.path.exists(path):
            self._create_empty_dataset_file(path)

    def _read_dataset_length(self, path):
        try:
            with np.load(path, allow_pickle=True) as data:
                return len(data["labels"])
        except Exception as exc:
            print(f"[EnemyDataset] Could not read dataset shard '{path}': {exc}")
            return None

    def _resolve_active_dataset_path(self):
        existing_indices = self._existing_dataset_indices()
        if not existing_indices:
            return self._dataset_path_for_index(0)

        highest_index = max(existing_indices)
        highest_path = self._dataset_path_for_index(highest_index)
        highest_len = self._read_dataset_length(highest_path)

        if highest_len is None:
            # If the latest shard is unreadable, continue with a new shard.
            return self._dataset_path_for_index(highest_index + 1)

        if highest_len >= self.max_entries_per_file:
            return self._dataset_path_for_index(highest_index + 1)

        return highest_path

    def _load_dataset(self, path):
        with np.load(path, allow_pickle=True) as data:
            # Copy arrays so file handles can be released immediately.
            if "teamcomposition" in data:
                teamcomposition = data["teamcomposition"]
            elif "teamcompositions" in data:
                teamcomposition = data["teamcompositions"]
            elif "images" in data:
                # Legacy fallback: preserve the sample count even when the old
                # screenshot-based schema is encountered.
                teamcomposition = [[] for _ in range(len(data["labels"]))]
            else:
                raise KeyError(f"Unsupported dataset schema in {path}: {list(data.keys())}")

            self.teamcomposition = _as_object_vector(teamcomposition)
            self.labels = np.array(data["labels"], copy=True)
            self.powers = np.array(data["powers"], copy=True) if "powers" in data else np.zeros((0,), dtype=np.float32)
            if "screenshots" in data:
                self.screenshots = np.array(data["screenshots"], copy=True)
            elif "images" in data:
                self.screenshots = np.array(data["images"], copy=True)
            else:
                self.screenshots = _empty_screenshot_vector(len(self.labels))

            if "screenshot_available" in data:
                self.screenshot_available = np.array(data["screenshot_available"], dtype=bool, copy=True)
            else:
                self.screenshot_available = _infer_screenshot_available(self.screenshots, len(self.labels))

    def _save_dataset(self):
        np.savez_compressed(
            self.dataset_path,
            teamcomposition=self.teamcomposition,
            powers=self.powers if self.use_power else np.zeros((len(self.labels),), dtype=np.float32),
            labels=self.labels,
            screenshots=self.screenshots,
            screenshot_available=self.screenshot_available,
            schema=np.array("teamcomposition_v1"),
        )

    def _resolve_team_size(self, configured_team_size):
        if configured_team_size is not None:
            return int(configured_team_size)

        observed_team_size = 0
        for entry in self.teamcomposition:
            if isinstance(entry, np.ndarray):
                entry = entry.tolist()
            if isinstance(entry, (list, tuple)):
                observed_team_size = max(observed_team_size, len(entry))

        return observed_team_size or 12

    def _rotate_to_next_dataset_file(self):
        current_index = self._dataset_index_from_path(self.dataset_path)
        next_index = current_index + 1

        while True:
            next_path = self._dataset_path_for_index(next_index)

            if not os.path.exists(next_path):
                self.dataset_path = next_path
                self._ensure_dataset_file(self.dataset_path)
                self._load_dataset(self.dataset_path)
                return

            existing_len = self._read_dataset_length(next_path)
            if existing_len is None:
                next_index += 1
                continue

            if existing_len < self.max_entries_per_file:
                self.dataset_path = next_path
                self._load_dataset(self.dataset_path)
                return

            next_index += 1

    def _append_screenshot_to_current_dataset(self, screenshot_to_add):
        screenshot_to_add = _normalize_screenshot_array(screenshot_to_add)
        sample_count = len(self.labels)

        if screenshot_to_add is None:
            if self.screenshots.ndim >= 4:
                empty = np.zeros((1, *self.screenshots.shape[1:]), dtype=self.screenshots.dtype)
                self.screenshots = np.concatenate([self.screenshots, empty], axis=0)
            else:
                self.screenshots = np.concatenate([self.screenshots, _empty_screenshot_vector(1)], axis=0)
            self.screenshot_available = np.concatenate(
                [self.screenshot_available, np.array([False], dtype=bool)],
                axis=0,
            )
            return

        screenshot_to_add = screenshot_to_add.reshape((1, *screenshot_to_add.shape))
        if self.screenshots.ndim >= 4 and self.screenshots.shape[1:] == screenshot_to_add.shape[1:]:
            self.screenshots = np.concatenate([self.screenshots, screenshot_to_add], axis=0)
        elif self.screenshots.ndim == 1 and not self.screenshot_available.any():
            previous = np.zeros((sample_count, *screenshot_to_add.shape[1:]), dtype=screenshot_to_add.dtype)
            self.screenshots = np.concatenate([previous, screenshot_to_add], axis=0)
        else:
            next_screenshots = _as_object_vector(self.screenshots.tolist())
            next_screenshots = np.concatenate(
                [next_screenshots, _as_object_vector([screenshot_to_add[0]])],
                axis=0,
            )
            self.screenshots = next_screenshots

        self.screenshot_available = np.concatenate(
            [self.screenshot_available, np.array([True], dtype=bool)],
            axis=0,
        )

    def _append_to_current_dataset(self, teamcomposition_to_add, power_to_add, labels_to_add, screenshot_to_add=None):
        next_teamcomposition = np.concatenate([self.teamcomposition, teamcomposition_to_add], axis=0)
        next_powers = None
        if self.use_power:
            power_to_add = np.asarray(power_to_add, dtype=np.float32)

            if self.powers.size == 0:
                next_powers = np.array(power_to_add, copy=True)
            else:
                same_rank = self.powers.ndim == power_to_add.ndim
                same_feature_shape = (
                    self.powers.shape[1:] == power_to_add.shape[1:]
                    if self.powers.ndim > 1 and power_to_add.ndim > 1
                    else True
                )

                if not same_rank or not same_feature_shape:
                    if self.max_entries_per_file is not None:
                        self._rotate_to_next_dataset_file()
                        self._append_to_current_dataset(
                            teamcomposition_to_add,
                            power_to_add,
                            labels_to_add,
                            screenshot_to_add=screenshot_to_add,
                        )
                        return

                    raise ValueError(
                        "Incompatible power shape for existing dataset. "
                        f"Stored shape rank={self.powers.ndim}, new rank={power_to_add.ndim}."
                    )

                next_powers = np.concatenate([self.powers, power_to_add], axis=0)

        self.teamcomposition = next_teamcomposition
        if self.use_power:
            self.powers = next_powers
        self._append_screenshot_to_current_dataset(screenshot_to_add)
        self.labels = np.concatenate([self.labels, labels_to_add], axis=0)
        self._save_dataset()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        teamcomposition = self.teamcomposition[idx]
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)

        if self.use_name_encoding and self.name_encoder is not None:
            encoded_teamcomposition = self.name_encoder.encode_teamcomposition(
                teamcomposition,
                team_size=self.team_size,
            )
            teamcomposition = torch.tensor(encoded_teamcomposition, dtype=torch.long)

        if self.use_power:
            power = torch.tensor(self.powers[idx], dtype=torch.float32) / float(self.max_power)
            if power.ndim == 0:
                power = power.unsqueeze(0)
            return teamcomposition, power, label

        return teamcomposition, label

    # -----------------------------
    # Append new entry / entries
    # -----------------------------
    def append_entry(self, enemy_record, battle_result, enemy_screenshot=None):
        """
        Append one or multiple entries and save dataset.

        enemy_record: dict with keys:
          - teamcomposition: list[str]
          - powervalue: scalar or power vector
          - screenshot: optional HxWxC enemy-team screenshot
        enemy_screenshot: optional HxWxC enemy-team screenshot. This is kept
          separate from enemy_record so avoid-list profile entries stay small.
        battle_result: int, 0=Loss, 1=Win
        """

        # Validate battle result
        if battle_result not in [0, 1]:
            raise ValueError("battle_result must be 0 (Loss) or 1 (Win)")

        if isinstance(enemy_record, dict):
            teamcomposition = enemy_record.get("teamcomposition", [])
            power_val = enemy_record.get("powervalue", 0.0)
            if enemy_screenshot is None:
                enemy_screenshot = enemy_record.get("screenshot", enemy_record.get("image"))
        else:
            teamcomposition = enemy_record
            power_val = 0.0

        teamcomposition_to_add = _as_object_vector([teamcomposition])
        power_val = np.asarray(power_val, dtype=np.float32)
        powers_to_add = power_val.reshape(1, -1) if power_val.ndim > 0 else power_val.reshape(1)
        labels_to_add = np.array([battle_result], dtype=np.float32)

        if self.max_entries_per_file is None:
            self._append_to_current_dataset(
                teamcomposition_to_add,
                powers_to_add,
                labels_to_add,
                screenshot_to_add=enemy_screenshot,
            )
            print(f"Appended 1 new entry. Dataset now has {len(self.labels)} samples.")
            return

        current_len = len(self.labels)
        if current_len >= self.max_entries_per_file:
            self._rotate_to_next_dataset_file()

        self._append_to_current_dataset(
            teamcomposition_to_add,
            powers_to_add,
            labels_to_add,
            screenshot_to_add=enemy_screenshot,
        )

        print(
            f"Appended 1 new entry. Active shard '{self.dataset_path}' now has "
            f"{len(self.labels)} samples."
        )


class CompositionEvaluationNetwork(nn.Module):
    """
    Win/loss predictor for champion-name team compositions plus power values.

    Public APIs may still pass champion names. The model converts those names to
    integer ids using the row order of data/processed/labels.csv
    before the data reaches the embedding layers.
    """

    def __init__(
        self,
        team_size: int,
        power_dim: int,
        *,
        labels_csv_path: str | os.PathLike | None = None,
        embedding_dim: int = 32,
        slot_embedding_dim: int = 8,
        hidden_dim: int = 128,
        max_power: float = 350000.0,
        weights_path: str | None = None,
    ):
        super().__init__()
        self.team_size = int(team_size)
        self.power_dim = int(power_dim)
        self.max_power = float(max_power)
        self.labels_csv_path = Path(labels_csv_path or DEFAULT_CHAMPION_LABELS_CSV)
        self.name_encoder = load_champion_row_encoder(str(self.labels_csv_path))

        self.champion_embedding = nn.Embedding(
            self.name_encoder.vocab_size,
            embedding_dim,
            padding_idx=0,
        )
        self.slot_embedding = nn.Embedding(self.team_size, slot_embedding_dim)
        self.team_encoder = nn.Sequential(
            nn.Linear(self.team_size * (embedding_dim + slot_embedding_dim), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.power_encoder = nn.Sequential(
            nn.Linear(self.power_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear((hidden_dim // 2) + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path, map_location="cpu"))

    def _encode_batch_from_names(self, teamcomposition_batch) -> torch.Tensor:
        if isinstance(teamcomposition_batch, np.ndarray) and np.issubdtype(teamcomposition_batch.dtype, np.integer):
            encoded = np.asarray(teamcomposition_batch, dtype=np.int64)
            if encoded.ndim == 1:
                encoded = encoded.reshape(1, -1)
            return torch.tensor(encoded, dtype=torch.long)

        if isinstance(teamcomposition_batch, torch.Tensor):
            tensor = teamcomposition_batch.long()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            return tensor

        if isinstance(teamcomposition_batch, np.ndarray):
            teamcomposition_batch = teamcomposition_batch.tolist()

        if not isinstance(teamcomposition_batch, (list, tuple)):
            teamcomposition_batch = [teamcomposition_batch]

        if teamcomposition_batch and not isinstance(teamcomposition_batch[0], (list, tuple, np.ndarray, torch.Tensor)):
            teamcomposition_batch = [teamcomposition_batch]

        encoded_rows = [
            self.name_encoder.encode_teamcomposition(row, team_size=self.team_size)
            for row in teamcomposition_batch
        ]
        return torch.tensor(np.asarray(encoded_rows, dtype=np.int64), dtype=torch.long)

    def _coerce_power_tensor(self, power) -> torch.Tensor:
        if isinstance(power, torch.Tensor):
            tensor = power.float()
        else:
            tensor = torch.tensor(power, dtype=torch.float32)

        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 1:
            if tensor.numel() == self.power_dim:
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.reshape(-1, self.power_dim)
        return tensor / self.max_power

    def forward(self, teamcomposition, power):
        team_ids = self._encode_batch_from_names(teamcomposition)
        power_tensor = self._coerce_power_tensor(power)

        device = next(self.parameters()).device
        team_ids = team_ids.to(device)
        power_tensor = power_tensor.to(device)

        if team_ids.shape[1] != self.team_size:
            raise ValueError(
                f"Expected teamcomposition width {self.team_size}, got {team_ids.shape[1]}"
            )
        if power_tensor.shape[1] != self.power_dim:
            raise ValueError(
                f"Expected power width {self.power_dim}, got {power_tensor.shape[1]}"
            )

        batch_size = team_ids.shape[0]
        slot_ids = torch.arange(self.team_size, device=device).unsqueeze(0).expand(batch_size, -1)
        champion_features = self.champion_embedding(team_ids)
        slot_features = self.slot_embedding(slot_ids)
        team_features = torch.cat([champion_features, slot_features], dim=2).reshape(batch_size, -1)
        team_features = self.team_encoder(team_features)
        power_features = self.power_encoder(power_tensor)
        return self.head(torch.cat([team_features, power_features], dim=1))

    def predict(self, teamcomposition, power_val, threshold=0.5):
        self.eval()
        with torch.no_grad():
            logits = self(teamcomposition, power_val)
            prob = torch.sigmoid(logits).item()
            label = int(prob >= threshold)
        return prob, label

    def train_network(
        self,
        dataset_path: str,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        checkpoint_interval: int = 10,
        checkpoint_path: str = "checkpoint.pt",
        val_split: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        dataset = EnemyDataset(
            dataset_path,
            max_power=self.max_power,
            labels_csv_path=self.labels_csv_path,
            use_name_encoding=True,
            team_size=self.team_size,
        )

        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        self.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for teamcomposition, powers, labels in train_loader:
                teamcomposition = teamcomposition.to(device)
                powers = powers.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = self(teamcomposition, powers)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * teamcomposition.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss /= total
            epoch_acc = correct / total

            self.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for teamcomposition, powers, labels in val_loader:
                    teamcomposition = teamcomposition.to(device)
                    powers = powers.to(device)
                    labels = labels.to(device)

                    logits = self(teamcomposition, powers)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * teamcomposition.size(0)

                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total

            if epoch % checkpoint_interval == 0:
                torch.save(self.state_dict(), f"{checkpoint_path}_epoch{epoch}.pt")
                print(f"Checkpoint saved at epoch {epoch}")
                print(
                    f"Epoch [{epoch}/{epochs}] | Train Loss: {epoch_loss:.4f} | "
                    f"Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )


class ClassicCompositionEvaluationNetwork(CompositionEvaluationNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(team_size=4, power_dim=1, *args, **kwargs)


class TagTeamCompositionEvaluationNetwork(CompositionEvaluationNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(team_size=12, power_dim=4, *args, **kwargs)


# -----------------------------
# Depthwise Block
# -----------------------------
class DepthwiseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3,
            stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pointwise(self.depthwise(x))))
    
    
# -----------------------------
# Network Class with Training
# -----------------------------
class EvaluationNetwork(nn.Module):
    """
    Lightweight Win/Loss predictor for enemy lineup images + power
    """
    def __init__(self, weights_path: str | None = None):
        super().__init__()

        # Image Encoder
        self.image_encoder = nn.Sequential(
            DepthwiseBlock(3, 32, stride=2),    # 440x130 -> 220x65
            DepthwiseBlock(32, 64, stride=2),   # -> 110x33
            DepthwiseBlock(64, 128, stride=2),  # -> 55x17
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.image_fc = nn.Linear(128, 128)

        # Power Encoder / Gate
        self.power_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 128),
            nn.Sigmoid()
        )

        # Decision Head
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        # Optional weight loading
        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path, map_location="cpu"))

    def forward(self, image, power):
        # Image branch
        x = self.image_encoder(image)
        x = x.view(x.size(0), -1)
        x = F.relu(self.image_fc(x))

        # Power branch as gate
        gate = self.power_gate(power)
        x = x * gate

        # Decision
        return torch.sigmoid(self.head(x))

    # -----------------------------
    # Training Method
    # -----------------------------
    def train_network(self,
                    dataset_path: str,
                    epochs: int = 50,
                    batch_size: int = 32,
                    lr: float = 1e-3,
                    checkpoint_interval: int = 10,
                    checkpoint_path: str = "checkpoint.pt",
                    val_split: float = 0.2,
                    device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        dataset = EnemyDataset(dataset_path)

        # -----------------------------
        # Train/Validation split
        # -----------------------------
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # -----------------------------
        # Device, Loss, Optimizer
        # -----------------------------
        self.to(device)
        criterion = nn.BCEWithLogitsLoss()  # more stable than BCE + sigmoid
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # -----------------------------
        # Track metrics
        # -----------------------------
        train_losses = []
        train_accuracies = []

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for images, powers, labels in train_loader:
                images = images.to(device)
                powers = powers.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = self(images, powers)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                # Metrics
                epoch_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss /= total
            epoch_acc = correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            # Validation
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, powers, labels in val_loader:
                    images = images.to(device)
                    powers = powers.to(device)
                    labels = labels.to(device)

                    logits = self(images, powers)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * images.size(0)

                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total

            # Checkpoint & print metrics
            if epoch % checkpoint_interval == 0:
                torch.save(self.state_dict(), f"{checkpoint_path}_epoch{epoch}.pt")
                print(f"Checkpoint saved at epoch {epoch}")
                print(f"Epoch [{epoch}/{epochs}] | Train Loss: {epoch_loss:.4f} | "
                    f"Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Final save
        torch.save(self.state_dict(), f"{checkpoint_path}_final.pt")
        print("Training finished, final weights saved.")
        
    

class EvaluationNetworkANN(nn.Module):
    """
    Fully connected ANN Win/Loss predictor for enemy lineup images + power.
    """
    def __init__(self, weights_path: str | None = None):
        super().__init__()

        # -----------------------------
        # Image Encoder: simple FC network
        # -----------------------------
        self.image_fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 130 * 440, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        # -----------------------------
        # Power Encoder / Gate
        # -----------------------------
        self.power_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 128),
            nn.Sigmoid()
        )

        # -----------------------------
        # Decision Head
        # -----------------------------
        self.head = nn.Linear(128, 1)  # raw logits

        # Optional weight loading
        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path, map_location="cpu"))

    def forward(self, image, power):
        # Image branch
        x = self.image_fc_layers(image)

        # Power branch as gate
        gate = self.power_gate(power)
        x = x * gate

        # Decision (raw logits)
        return self.head(x)

    # -----------------------------
    # Training Method
    # -----------------------------
    def train_network(self,
                      dataset_path: str,
                      epochs: int = 50,
                      batch_size: int = 32,
                      lr: float = 1e-3,
                      checkpoint_interval: int = 10,
                      checkpoint_path: str = "checkpoint.pt",
                      val_split: float = 0.02,
                      device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        dataset = EnemyDataset(dataset_path)

        # -----------------------------
        # Train/Validation split
        # -----------------------------
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # -----------------------------
        # Device, Loss, Optimizer
        # -----------------------------
        self.to(device)
        criterion = nn.BCEWithLogitsLoss()  # more stable than BCE + sigmoid
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # -----------------------------
        # Track metrics
        # -----------------------------
        train_losses = []
        train_accuracies = []

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for images, powers, labels in train_loader:
                images = images.to(device)
                powers = powers.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = self(images, powers)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                # Metrics
                epoch_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss /= total
            epoch_acc = correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            # Validation
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, powers, labels in val_loader:
                    images = images.to(device)
                    powers = powers.to(device)
                    labels = labels.to(device)

                    logits = self(images, powers)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * images.size(0)

                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total


            # Checkpoint
            if epoch % checkpoint_interval == 0:
                torch.save(self.state_dict(), f"{checkpoint_path}_epoch{epoch}.pt")
                print(f"Checkpoint saved at epoch {epoch}")
                print(f"Epoch [{epoch}/{epochs}] | Train Loss: {epoch_loss:.4f} | "
                  f"Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Final save
        torch.save(self.state_dict(), f"{checkpoint_path}_final.pt")
        print("Training finished, final weights saved.")



class EvaluationNetworkCNN(nn.Module):
    """
    CNN-based Win/Loss predictor for enemy lineup images + power.
    The CNN reduces the image to a single scalar, then combines with power input.
    """
    def __init__(self, weights_path: str | None = None):
        super().__init__()

        # -----------------------------
        # Image Encoder: CNN -> 1 scalar
        # -----------------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # 3x130x440 -> 32x65x220
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # 64x33x110
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# 128x17x55
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 256x9x28
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # 256x1x1
        )
        self.image_fc = nn.Linear(256, 8)  # reduce to 1 scalar

        # -----------------------------
        # Power Encoder: simple FC
        # -----------------------------
        self.power_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        # -----------------------------
        # Decision Head
        # -----------------------------
        self.head = nn.Linear(16, 1)  # combine image scalar + power scalar

        # Optional weight loading
        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path, map_location="cpu"))

    def forward(self, image, power):
        # CNN image branch
        x = self.cnn(image)
        x = x.view(x.size(0), -1)  # flatten 256x1x1 -> 256
        x = self.image_fc(x)       # reduce to scalar [batch,1]

        # Power branch
        p = self.power_fc(power)   # [batch,1]

        # Combine
        combined = torch.cat([x, p], dim=1)  # [batch,2]

        # Decision (raw logits)
        return self.head(combined)

    def predict(self, image_np, power_val, threshold=0.5):
        self.eval()
        with torch.no_grad():
            image = (
                torch.from_numpy(image_np.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            power = torch.tensor([[power_val / 350000.0]], dtype=torch.float32)

            logits = self(image, power)
            prob = torch.sigmoid(logits).item()
            label = int(prob >= threshold)

        return prob, label
    # -----------------------------
    # Training method same as ANN
    # -----------------------------
    def train_network(self,
                      dataset_path: str,
                      epochs: int = 50,
                      batch_size: int = 32,
                      lr: float = 1e-3,
                      checkpoint_interval: int = 10,
                      checkpoint_path: str = "checkpoint.pt",
                      val_split: float = 0.2,
                      device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        dataset = EnemyDataset(dataset_path)

        # Train/Validation split
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Device, Loss, Optimizer
        self.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Track metrics
        train_losses, train_accuracies = [], []

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for images, powers, labels in train_loader:
                images = images.to(device)
                powers = powers.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = self(images, powers)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss /= total
            epoch_acc = correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            # Validation
            self.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, powers, labels in val_loader:
                    images = images.to(device)
                    powers = powers.to(device)
                    labels = labels.to(device)

                    logits = self(images, powers)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * images.size(0)

                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total

            if epoch % checkpoint_interval == 0:
                torch.save(self.state_dict(), f"{checkpoint_path}_epoch{epoch}.pt")
                print(f"Checkpoint saved at epoch {epoch}")
                print(f"Epoch [{epoch}/{epochs}] | Train Loss: {epoch_loss:.4f} | "
                      f"Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


class EvaluationNetworkCNN_ImageOnly(nn.Module):
    """
    CNN-based Win/Loss predictor using only enemy lineup images.
    """
    def __init__(self, weights_path: str | None = None):
        super().__init__()

        # -----------------------------
        # Image Encoder: CNN -> 1 scalar
        # -----------------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # 3x130x440 -> 32x65x220
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # 64x33x110
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# 128x17x55
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 256x9x28
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # 256x1x1
        )
        self.image_fc = nn.Linear(256, 1)  # directly to scalar output

        # Optional weight loading
        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path, map_location="cpu"))

    def forward(self, image):
        # CNN image branch
        x = self.cnn(image)
        x = x.view(x.size(0), -1)  # flatten 256x1x1 -> 256
        logits = self.image_fc(x)  # [batch, 1]
        return logits

    def predict(self, image_np, threshold=0.5):
        self.eval()
        with torch.no_grad():
            image = (
                torch.from_numpy(image_np.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )

            logits = self(image)
            prob = torch.sigmoid(logits).item()
            label = int(prob >= threshold)

        return prob, label

    # -----------------------------
    # Training method
    # -----------------------------
    def train_network(self,
                      dataset_path: str,
                      epochs: int = 50,
                      batch_size: int = 32,
                      lr: float = 1e-3,
                      checkpoint_interval: int = 10,
                      checkpoint_path: str = "checkpoint.pt",
                      val_split: float = 0.2,
                      device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        dataset = EnemyDataset(dataset_path, use_power = False)  # ensure dataset returns only images

        # Train/Validation split
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Device, Loss, Optimizer
        self.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Track metrics
        train_losses, train_accuracies = [], []

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = self(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss /= total
            epoch_acc = correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            # Validation
            self.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = self(images)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * images.size(0)

                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total

            if epoch % checkpoint_interval == 0:
                torch.save(self.state_dict(), f"{checkpoint_path}_epoch{epoch}.pt")
                print(f"Checkpoint saved at epoch {epoch}")
                print(f"Epoch [{epoch}/{epochs}] | Train Loss: {epoch_loss:.4f} | "
                      f"Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                


class TagTeamEvaluationNetworkCNN(nn.Module):
    """
    CNN-based Win/Loss predictor for enemy lineup images + power vector (4 values).
    """

    def __init__(self, weights_path: str | None = None):
        super().__init__()

        # -----------------------------
        # Image Encoder
        # -----------------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), 
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # -> [B, 256, 1, 1]
        )

        self.image_fc = nn.Linear(256, 12)

        # -----------------------------
        # Power Encoder (4 inputs)
        # -----------------------------
        self.power_fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

        # -----------------------------
        # Decision Head
        # -----------------------------
        self.head = nn.Linear(16, 1)  # 8 image + 8 power

        # Optional weight loading
        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path, map_location="cpu"))

    # --------------------------------------------------
    # Forward Pass
    # --------------------------------------------------
    def forward(self, image, power):
        # Image branch
        x = self.cnn(image)
        x = x.view(x.size(0), -1)   # [B, 256]
        x = self.image_fc(x)       # [B, 8]

        # Power branch
        power = power.view(power.size(0), -1)  # ensure [B, 4]
        p = self.power_fc(power)               # [B, 8]

        # Combine
        combined = torch.cat([x, p], dim=1)    # [B, 16]

        return self.head(combined)

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    def predict(self, image_np, power_vals, threshold=0.5):
        """
        image_np   : numpy image (H, W, 3)
        power_vals : array-like of shape (4,)
        """
        self.eval()
        with torch.no_grad():
            image = (
                torch.from_numpy(image_np.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )

            power = torch.tensor([power_vals], dtype=torch.float32)

            logits = self(image, power)
            prob = torch.sigmoid(logits).item()
            label = int(prob >= threshold)

        return prob, label

    def crop_classic_arena_portraits(self, image_np):
        """Return the 4 classic arena portrait crops in left-to-right order."""
        return crop_classic_arena_portraits(image_np)

    def crop_tagteam_portraits(self, image_np):
        """Return the 12 portrait crops in slot order: slot 1, slot 2, slot 3."""
        return crop_tagteam_portraits(image_np)

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    def train_network(self,
                      dataset_path: str,
                      epochs: int = 50,
                      batch_size: int = 32,
                      lr: float = 1e-3,
                      checkpoint_interval: int = 10,
                      checkpoint_path: str = "checkpoint.pt",
                      val_split: float = 0.2,
                      device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        dataset = EnemyDataset(dataset_path)

        # Train/Validation split
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Device, Loss, Optimizer
        self.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for images, powers, labels in train_loader:
                images = images.to(device)
                powers = powers.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = self(images, powers)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss /= total
            epoch_acc = correct / total

            # Validation
            self.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, powers, labels in val_loader:
                    images = images.to(device)
                    powers = powers.to(device)
                    labels = labels.to(device)

                    logits = self(images, powers)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * images.size(0)

                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total

            if epoch % checkpoint_interval == 0:
                torch.save(self.state_dict(), f"{checkpoint_path}_epoch{epoch}.pt")
                print(f"Checkpoint saved at epoch {epoch}")
                print(
                    f"Epoch [{epoch}/{epochs}] | "
                    f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )
