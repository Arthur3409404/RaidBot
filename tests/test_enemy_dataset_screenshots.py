from __future__ import annotations

import numpy as np

from raid_bot.handlers.ai_networks_handler import EnemyDataset


def _object_vector(values):
    array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return array


def test_enemy_dataset_saves_enemy_screenshot(tmp_path):
    dataset_path = tmp_path / "enemy_dataset_classic_arena.npz"
    dataset = EnemyDataset(str(dataset_path))
    screenshot = np.full((2, 3, 3), 7, dtype=np.uint8)

    dataset.append_entry(
        {"teamcomposition": ["Athel", "Kael"], "powervalue": 1234.0},
        1,
        enemy_screenshot=screenshot,
    )

    with np.load(dataset_path, allow_pickle=True) as data:
        assert "screenshots" in data
        assert "screenshot_available" in data
        assert data["screenshots"].shape == (1, 2, 3, 3)
        assert data["screenshots"].dtype == np.uint8
        assert data["screenshots"][0].tolist() == screenshot.tolist()
        assert data["screenshot_available"].tolist() == [True]


def test_enemy_dataset_backfills_legacy_entries_without_screenshots(tmp_path):
    dataset_path = tmp_path / "enemy_dataset_tagteam_arena.npz"
    np.savez_compressed(
        dataset_path,
        teamcomposition=_object_vector([["Athel"]]),
        powers=np.array([[1000.0, 300.0, 300.0, 400.0]], dtype=np.float32),
        labels=np.array([0.0], dtype=np.float32),
        schema=np.array("teamcomposition_v1"),
    )
    dataset = EnemyDataset(str(dataset_path))
    screenshot = np.full((2, 3, 3), 11, dtype=np.float32)

    dataset.append_entry(
        {
            "teamcomposition": ["Athel", "Kael"],
            "powervalue": [1200.0, 400.0, 400.0, 400.0],
        },
        1,
        enemy_screenshot=screenshot,
    )

    with np.load(dataset_path, allow_pickle=True) as data:
        assert data["screenshots"].shape == (2, 2, 3, 3)
        assert data["screenshots"].dtype == np.uint8
        assert data["screenshots"][0].sum() == 0
        assert data["screenshots"][1].tolist() == screenshot.astype(np.uint8).tolist()
        assert data["screenshot_available"].tolist() == [False, True]
