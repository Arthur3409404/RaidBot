# -*- coding: utf-8 -*-
"""
Train EvaluationNetwork on classic arena dataset
and analyze correlation between enemy power and win/loss.

Assumes classes are imported from neuralnetworks.py:
from neuralnetworks import *
"""

import numpy as np
import torch
from scipy.stats import pointbiserialr
from data.lib.handlers.ai_networks_handler import *
import matplotlib.pyplot as plt

# =============================
# Paths
# =============================
DATASET_PATH = "data/database_champions/datasets/enemy_dataset_tagteam_arena.npz"
CHECKPOINT_PATH = "neural_networks/enemy_eval_tagteam_arena/"

# =============================
# Training Routine
# =============================
def train_enemy_evaluation_model(image_only: bool = True):
    print("Starting training...")

    # -----------------------------
    # Load original dataset
    # -----------------------------
    data = np.load(DATASET_PATH)
    images = data["images"]
    labels = data["labels"]
    powers = data["powers"] if not image_only else None

    total = len(labels)
    count_1 = np.sum(labels == 1)
    count_0 = np.sum(labels == 0)
    print(f"Original Dataset with: {total} samples ({count_1}=1 / {count_0}=0)")

    # -----------------------------
    # Balance dataset (50/50 labels)
    # -----------------------------
    win_idx = np.where(labels == 1)[0]
    loss_idx = np.where(labels == 0)[0]
    min_count = min(len(win_idx), len(loss_idx))

    # Randomly sample from each class
    np.random.seed(42)  # reproducible
    win_sample = np.random.choice(win_idx, min_count, replace=False)
    loss_sample = np.random.choice(loss_idx, min_count, replace=False)
    selected_idx = np.concatenate([win_sample, loss_sample])
    np.random.shuffle(selected_idx)

    # Create balanced dataset
    balanced_dataset_path = DATASET_PATH.replace(".npz", "_balanced.npz")
    if image_only:
        np.savez_compressed(
            balanced_dataset_path,
            images=images[selected_idx],
            labels=labels[selected_idx]
        )
    else:
        np.savez_compressed(
            balanced_dataset_path,
            images=images[selected_idx],
            powers=powers[selected_idx],
            labels=labels[selected_idx]
        )
    print(f"Balanced dataset created: {len(selected_idx)} samples (50/50)")

    # -----------------------------
    # Initialize network
    # -----------------------------
    if image_only:
        model = EvaluationNetworkCNN_ImageOnly()
    else:
        model = EvaluationNetworkCNN()

    # -----------------------------
    # Train the network
    # -----------------------------
    model.train_network(
        dataset_path=balanced_dataset_path,
        epochs=1000,
        batch_size=16,
        lr=1e-4,
        val_split=0.02,
        checkpoint_interval=50,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Training complete.")
# =============================
# Power vs Label Correlation Analysis
# =============================
def analyze_power_label_correlation():
    print("\nAnalyzing power vs label correlation...")
    data = np.load(DATASET_PATH)
    images = data["images"]
    powers = data["powers"]       # normalized [0,1]
    labels = data["labels"]       # 0/1

    # Restore original scale
    powers_raw = powers * 350000.0

    # Pearson correlation
    pearson_corr = np.corrcoef(powers_raw, labels)[0, 1]

    # Point-biserial correlation
    from scipy.stats import pointbiserialr
    pb_corr, pb_p = pointbiserialr(labels, powers_raw)

    # Mean / median
    win_powers = powers_raw[labels == 1]
    loss_powers = powers_raw[labels == 0]

    print(f"Number of samples: {len(labels)}")
    print(f"Win rate: {labels.mean():.3f}")
    print(f"Pearson correlation: {pearson_corr:.3f}")
    print(f"Point-biserial correlation: {pb_corr:.3f} (p-value: {pb_p:.3e})")
    print(f"Mean power (win): {win_powers.mean():.1f}")
    print(f"Mean power (loss): {loss_powers.mean():.1f}")
    print(f"Median power (win): {np.median(win_powers):.1f}")
    print(f"Median power (loss): {np.median(loss_powers):.1f}")

    # -----------------------------
    # Plot win/loss over power
    # -----------------------------
    plot = True
    if plot:
        plt.figure(figsize=(10, 6))

        # Scatter plot with jitter on y
        jitter = np.random.uniform(-0.02, 0.02, size=len(labels))
        plt.scatter(powers_raw, labels + jitter, alpha=0.3, color='blue', label='Samples')

        plt.xlabel("Enemy Power")
        plt.ylabel("Win (1) / Loss (0)")
        plt.ylim(1000, 1e7)
        plt.title("Win/Loss vs Enemy Power")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


# =============================
# Optional: Power-only baseline
# =============================
def power_only_baseline_accuracy():
    data = np.load(DATASET_PATH)
    powers = data["powers"]
    labels = data["labels"]

    threshold = powers.mean()
    preds = (powers >= threshold).astype(np.float32)
    acc = (preds == labels).mean()

    # Number of correctly predicted wins
    correct_wins = np.sum((preds == 1) & (labels == 1))
    total_wins = np.sum(labels == 1)

    print(f"Power-only baseline accuracy: {acc:.3f}")
    print(f"Correctly predicted wins: {correct_wins}/{total_wins} ({100*correct_wins/total_wins:.1f}%)")



def plot_images_with_labels(dataset_path=DATASET_PATH, grid_size=(15, 15)):
    """
    Plots all images from dataset in a grid with their labels as titles.
    
    Args:
        dataset_path: Path to the .npz dataset
        grid_size: Tuple (rows, cols) for the grid
    """
    data = np.load(dataset_path)
    images = data["images"]
    labels = data["labels"]
    print(len(labels))

    total_images = len(labels)
    rows, cols = grid_size
    if total_images > rows * cols:
        print(f"Warning: More images ({total_images}) than grid spaces ({rows*cols}), only plotting first {rows*cols}")
        total_images = rows * cols

    plt.figure(figsize=(cols * 1.5, rows * 1.5))  # adjust figure size

    for i in range(total_images):
        img = images[i]
        if img.max() <= 1.0:
            img = img * 255  # scale to 0-255

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img.astype(np.uint8))
        plt.title(f"{labels[i]}", fontsize=6)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# =============================
# Main
# =============================
if __name__ == "__main__":
    #plot_images_with_labels()
    # analyze_power_label_correlation()
    train_enemy_evaluation_model()
    # power_only_baseline_accuracy()