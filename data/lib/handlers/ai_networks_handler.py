# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 15:36:23 2025

@author: Arthur
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
import pandas as pd



# -----------------------------
# Dataset Class
# -----------------------------
class EnemyDataset(Dataset):
    """
    Dataset storing all images, power (normalized), and labels (0/1) in a single .npz file.
    Creates the dataset if it does not exist and appends new entries.
    Images are normalized to [0,1], power is normalized by 350000.
    """
    def __init__(self, dataset_path, transform=None, max_power=350000.0):
        self.dataset_path = dataset_path
        self.transform = transform
        self.max_power = max_power

        # Ensure the folder exists
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

        # Create empty dataset if file does not exist
        if not os.path.exists(dataset_path):
            np.savez_compressed(
                dataset_path,
                images=np.zeros((0, 130, 440, 3), dtype=np.float32),
                powers=np.zeros((0,), dtype=np.float32),
                labels=np.zeros((0,), dtype=np.float32)
            )

        # Load existing data
        data = np.load(dataset_path, allow_pickle=True)
        self.images = data["images"]
        self.powers = data["powers"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get image, power, label
        image = self.images[idx]  # already normalized
        power = self.powers[idx]  # already normalized
        label = self.labels[idx]  # 0 or 1

        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # [C,H,W]
        power = torch.tensor([power], dtype=torch.float32)
        label = torch.tensor([label], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, power, label

    # -----------------------------
    # Append a new entry
    # -----------------------------
    def append_entry(self, image_np, power_val, battle_result):
        """
        Append a new entry and save dataset.

        image_np: np.ndarray (H,W,C) uint8
        power_val: float, between 0 and 350000
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

        # Normalize power
        power_val = power_val / self.max_power

        # Append to arrays
        self.images = np.concatenate([self.images, image_np[np.newaxis, ...]], axis=0)
        self.powers = np.concatenate([self.powers, np.array([power_val], dtype=np.float32)], axis=0)
        self.labels = np.concatenate([self.labels, np.array([battle_result], dtype=np.float32)], axis=0)

        # Save to npz (overwrite old dataset)
        np.savez_compressed(
            self.dataset_path,
            images=self.images,
            powers=self.powers,
            labels=self.labels
        )
        print(f"Appended new entry. Dataset now has {len(self.labels)} samples.")


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
                      device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    
        # Use your custom dataset (no transform needed, data already normalized)
        dataset = EnemyDataset(dataset_path)
        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
    
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
        # Move model to device
        self.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
    
        for epoch in range(1, epochs + 1):
            self.train()
            for images, powers, labels in train_loader:
                images = images.to(device)
                powers = powers.to(device)
                labels = labels.to(device)
    
                optimizer.zero_grad()
                outputs = self(images, powers)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
            # Validation
            self.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, powers, labels in val_loader:
                    images = images.to(device)
                    powers = powers.to(device)
                    labels = labels.to(device)
    
                    outputs = self(images, powers)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
    
                    preds = (outputs >= 0.5).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
    
            val_loss /= total
            val_acc = correct / total
            print(f"Epoch [{epoch}/{epochs}] | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
            # Checkpoint
            if epoch % checkpoint_interval == 0:
                torch.save(self.state_dict(), f"{checkpoint_path}_epoch{epoch}.pt")
                print(f"Checkpoint saved at epoch {epoch}")
    
        # Final save
        torch.save(self.state_dict(), f"{checkpoint_path}_final.pt")
        print("Training finished, final weights saved.")
        
    