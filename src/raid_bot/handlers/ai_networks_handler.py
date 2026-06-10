# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 15:36:23 2025

@author: Arthur
"""


import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image


# -----------------------------
# Dataset Class
# -----------------------------
class EnemyDataset(Dataset):
    """
    Dataset storing images, power (normalized), and labels (0/1) in .npz files.
    Creates the dataset if it does not exist and appends new entries.
    Optionally supports sharded saving with max_entries_per_file.
    Images are normalized to [0,1], power is normalized by max_power.
    Power can be a single value or a numpy array.
    """

    def __init__(
        self,
        dataset_path,
        use_power=True,
        transform=None,
        max_power=350000.0,
        max_entries_per_file=None
    ):
        self.base_dataset_path = dataset_path
        self.dataset_path = dataset_path
        self.transform = transform
        self.max_power = max_power
        self.use_power = use_power
        self.max_entries_per_file = max_entries_per_file

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
            images=np.zeros((0, 130, 440, 3), dtype=np.float32),
            powers=np.zeros((0,), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.float32),
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
            self.images = np.array(data["images"], copy=True)
            self.labels = np.array(data["labels"], copy=True)

            if self.use_power:
                self.powers = np.array(data["powers"], copy=True)

    def _save_dataset(self):
        np.savez_compressed(
            self.dataset_path,
            images=self.images,
            powers=self.powers if self.use_power else np.zeros((len(self.labels),), dtype=np.float32),
            labels=self.labels,
        )

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

    def _append_to_current_dataset(self, images_to_add, power_to_add, labels_to_add):
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
                    # Compatibility fallback for legacy shards:
                    # rotate to a fresh shard so new schema can be stored safely.
                    if self.max_entries_per_file is not None:
                        self._rotate_to_next_dataset_file()
                        self._append_to_current_dataset(images_to_add, power_to_add, labels_to_add)
                        return

                    raise ValueError(
                        "Incompatible power shape for existing dataset. "
                        f"Stored shape rank={self.powers.ndim}, new rank={power_to_add.ndim}."
                    )

                next_powers = np.concatenate([self.powers, power_to_add], axis=0)

        self.images = np.concatenate([self.images, images_to_add], axis=0)
        if self.use_power:
            self.powers = next_powers
        self.labels = np.concatenate([self.labels, labels_to_add], axis=0)
        self._save_dataset()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get image, power, label
        image = self.images[idx]  # already normalized [H,W,C]
        label = self.labels[idx]  # 0 or 1

        # Convert image to tensor [C,H,W]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        if self.use_power:
            power = self.powers[idx]
            power = torch.tensor(power, dtype=torch.float32)
            if power.ndim == 0:
                power = power.unsqueeze(0)

        label = torch.tensor([label], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        if self.use_power:
            return image, power, label
        else:
            return image, label

    # -----------------------------
    # Append new entry / entries
    # -----------------------------
    def append_entry(self, image_np, power_val, battle_result):
        """
        Append one or multiple entries and save dataset.

        image_np: np.ndarray (H,W,C) uint8
        power_val:
          - scalar -> one entry with one power value
          - 1D array -> one entry with a power vector (e.g. Tag Team [total, t1, t2, t3])
          - 2D array -> multiple entries, one row per entry
        battle_result: int, 0=Loss, 1=Win
        """

        # Validate battle result
        if battle_result not in [0, 1]:
            raise ValueError("battle_result must be 0 (Loss) or 1 (Win)")

        # Resize image if needed
        if image_np.shape[:2] != (130, 440):
            image_np = np.array(Image.fromarray(image_np).resize((440, 130)))

        # Normalize image to [0,1]
        image_np = image_np.astype(np.float32) / 255.0

        power_val = np.asarray(power_val, dtype=np.float32) / self.max_power

        if power_val.ndim == 0:
            powers_to_add = power_val.reshape(1)
            num_entries = 1
        elif power_val.ndim == 1:
            # Treat a 1D vector as a single battle record (not N separate records).
            powers_to_add = power_val.reshape(1, -1) if power_val.size > 1 else power_val.reshape(1)
            num_entries = 1
        elif power_val.ndim == 2:
            powers_to_add = power_val
            num_entries = power_val.shape[0]
        else:
            raise ValueError("power_val must be a scalar, 1D array, or 2D array.")

        images_to_add = np.repeat(image_np[np.newaxis, ...], num_entries, axis=0)
        labels_to_add = np.full((num_entries,), battle_result, dtype=np.float32)

        if self.max_entries_per_file is None:
            self._append_to_current_dataset(images_to_add, powers_to_add, labels_to_add)
            print(f"Appended {num_entries} new entries. Dataset now has {len(self.labels)} samples.")
            return

        added_entries = 0
        total_entries = num_entries

        while added_entries < total_entries:
            current_len = len(self.labels)
            remaining_capacity = self.max_entries_per_file - current_len

            if remaining_capacity <= 0:
                self._rotate_to_next_dataset_file()
                continue

            chunk_size = min(remaining_capacity, total_entries - added_entries)
            chunk_start = added_entries
            chunk_end = added_entries + chunk_size

            self._append_to_current_dataset(
                images_to_add[chunk_start:chunk_end],
                powers_to_add[chunk_start:chunk_end],
                labels_to_add[chunk_start:chunk_end],
            )
            added_entries += chunk_size

        print(
            f"Appended {num_entries} new entries. Active shard '{self.dataset_path}' now has "
            f"{len(self.labels)} samples."
        )


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
