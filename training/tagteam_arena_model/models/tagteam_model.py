from __future__ import annotations

import torch
from torch import nn


class SlotEvaluator(nn.Module):
    """Encode one enemy team crop plus its power into slot features."""

    def __init__(self, in_channels: int, feature_dim: int = 64):
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.power_branch = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
        )
        self.slot_embedding = nn.Embedding(3, 8)
        self.feature_head = nn.Sequential(
            nn.Linear(48 + 16 + 8, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )
        self.hidden_head = nn.Sequential(
            nn.Linear(feature_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
            nn.Sigmoid(),
        )

    def forward(self, crop: torch.Tensor, power: torch.Tensor, slot_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if power.ndim == 1:
            power = power.unsqueeze(1)
        image_features = self.image_branch(crop)
        power_features = self.power_branch(power)
        slot_features = self.slot_embedding(slot_id.long())
        features = self.feature_head(torch.cat([image_features, power_features, slot_features], dim=1))
        hidden_prob = self.hidden_head(features)
        return features, hidden_prob


class TagTeamArenaModel(nn.Module):
    """Tag Team Arena model with shared slot CNNs and a learned 3-layer interface net."""

    def __init__(self, in_channels: int, feature_dim: int = 64):
        super().__init__()
        self.slot_evaluator = SlotEvaluator(in_channels=in_channels, feature_dim=feature_dim)
        self.interface_net = nn.Sequential(
            nn.Linear(feature_dim * 3 + 3 + 1 + 4, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )

    def forward(self, crops: torch.Tensor, powers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if crops.ndim != 5:
            raise ValueError(f"Expected crops with shape [B, 3, C, H, W], got {tuple(crops.shape)}")
        if powers.ndim != 2 or powers.shape[1] != 3:
            raise ValueError(f"Expected powers with shape [B, 3], got {tuple(powers.shape)}")

        batch_size = crops.shape[0]
        slot_features = []
        hidden_probs = []
        for slot in range(3):
            slot_ids = torch.full((batch_size,), slot, device=crops.device, dtype=torch.long)
            features, hidden_prob = self.slot_evaluator(crops[:, slot], powers[:, slot], slot_ids)
            slot_features.append(features)
            hidden_probs.append(hidden_prob)

        hidden = torch.cat(hidden_probs, dim=1)
        p1 = hidden[:, 0:1]
        p2 = hidden[:, 1:2]
        p3 = hidden[:, 2:3]
        best_of_three_prior = p1 * p2 + p1 * p3 + p2 * p3 - 2.0 * p1 * p2 * p3

        power_summary = torch.stack(
            [
                powers.mean(dim=1),
                powers.amin(dim=1),
                powers.amax(dim=1),
                powers.amax(dim=1) - powers.amin(dim=1),
            ],
            dim=1,
        )
        interface_features = torch.cat(
            [
                torch.cat(slot_features, dim=1),
                hidden,
                best_of_three_prior,
                power_summary,
            ],
            dim=1,
        )
        final_logits = self.interface_net(interface_features)
        return final_logits, hidden
