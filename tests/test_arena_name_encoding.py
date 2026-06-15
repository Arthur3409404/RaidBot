from __future__ import annotations

import csv

import numpy as np
import torch

from raid_bot.handlers.ai_networks_handler import (
    ChampionRowEncoder,
    ClassicCompositionEvaluationNetwork,
    TagTeamCompositionEvaluationNetwork,
)


def _write_labels_csv(path):
    rows = [
        {"label": "athel", "champion_name": "Athel"},
        {"label": "kael", "champion_name": "Kael"},
        {"label": "arbiter", "champion_name": "Arbiter"},
        {"label": "sun_wukong", "champion_name": "Sun Wukong"},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "champion_name"])
        writer.writeheader()
        writer.writerows(rows)


def test_champion_row_encoder_uses_csv_row_order(tmp_path):
    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv)

    encoder = ChampionRowEncoder(labels_csv)

    assert encoder.encode_name("Athel") == 1
    assert encoder.encode_name("Kael") == 2
    assert encoder.encode_name("arbiter") == 3
    assert encoder.encode_name("sun wukong") == 4


def test_champion_row_encoder_unknown_falls_back_to_zero(tmp_path):
    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv)

    encoder = ChampionRowEncoder(labels_csv)

    assert encoder.encode_name("Not A Champion") == 0
    assert encoder.encode_name(None) == 0


def test_classic_team_encoding_is_four_slots(tmp_path):
    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv)

    encoder = ChampionRowEncoder(labels_csv)

    encoded = encoder.encode_teamcomposition(["Athel", "Kael"], team_size=4)

    assert encoded.tolist() == [1, 2, 0, 0]
    assert encoded.dtype == np.int64


def test_tagteam_team_encoding_is_twelve_slots(tmp_path):
    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv)

    encoder = ChampionRowEncoder(labels_csv)

    encoded = encoder.encode_teamcomposition(
        ["Athel", "Kael", "Arbiter", "Sun Wukong", "Unknown"] * 3,
        team_size=12,
    )

    assert encoded.tolist() == [1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2]


def test_classic_evaluation_model_accepts_names_and_converts_to_ids(tmp_path):
    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv)
    model = ClassicCompositionEvaluationNetwork(labels_csv_path=labels_csv)

    encoded = model._encode_batch_from_names(["Athel", "Kael", "Missing", "Arbiter"])
    logits = model(["Athel", "Kael", "Missing", "Arbiter"], 123456)

    assert encoded.tolist() == [[1, 2, 0, 3]]
    assert logits.shape == torch.Size([1, 1])


def test_tagteam_evaluation_model_accepts_names_and_converts_to_ids(tmp_path):
    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv)
    model = TagTeamCompositionEvaluationNetwork(labels_csv_path=labels_csv)
    names = ["Athel", "Kael", "Arbiter", "Sun Wukong"] * 3

    encoded = model._encode_batch_from_names(names)
    logits = model(names, [300000, 100000, 100000, 100000])

    assert encoded.tolist() == [[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]
    assert logits.shape == torch.Size([1, 1])
