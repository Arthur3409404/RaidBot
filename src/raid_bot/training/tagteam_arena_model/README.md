# Standalone Tag Team Arena Training

This folder contains a standalone PyTorch training pipeline for a Raid Shadow Legends Tag Team Arena best-of-3 prediction model. It does not modify or integrate with the existing runtime Tag Team Arena module.

The model trains on final match results only:

- `1` means the best-of-3 match was won.
- `0` means the best-of-3 match was lost.

The hidden battle probabilities `p1`, `p2`, and `p3` are latent internal estimates. They are printed for inspection, but they are not directly supervised and should not be treated as verified per-battle truth.

## Model

`SlotEvaluator` processes one enemy team crop, one enemy power value, and one slot ID. The slot ID represents which fixed friendly team is facing that enemy team.

`TagTeamArenaModel` applies the same CNN+power evaluator to all three slots, concatenates the three slot feature vectors, and sends them through a compact 3-layer interface network. The interface network sees slot features, latent `p1/p2/p3`, power summary features, and the deterministic best-of-3 probability as a prior:

```python
best_of_three_prior = p1*p2 + p1*p3 + p2*p3 - 2*p1*p2*p3
final_logits = interface_net(slot_features, p1_p2_p3, best_of_three_prior, power_summary)
final_p = sigmoid(final_logits)
```

The final prediction is no longer forced to equal the deterministic formula. The formula is available as an input feature, while the interface network can learn cross-slot effects from final match labels.

Training uses `BCEWithLogitsLoss` on `final_logits` for numerical stability. Validation metrics and example prediction files still store sigmoid probabilities.

## Expected Data

The loader starts by inspecting the `.npz` file and printing keys, shapes, dtypes, ranges, sample labels, and sample powers when available.

After loading, tensors are converted to:

```python
crops.shape == [N, 3, C, H, W]
powers.shape == [N, 3]
labels.shape == [N]
```

If the image array is a full enemy composition image, it can be split into three crops with `--split_axis width`, `--split_axis height`, or `--split_axis auto`.

Power values must be present as a separate array. If powers are only visually embedded in screenshots, OCR or a separate powers array is required.

## Commands

Inspect only. When the path has numbered siblings like `enemy_dataset_tagteam_arena_1.npz`, they are included automatically:

```bash
uv run python scripts/train_tagteam_arena_model.py --data_path "data/database_champions/datasets/enemy_dataset_tagteam_arena.npz" --inspect_only
```

Dry run:

```bash
uv run python scripts/train_tagteam_arena_model.py --data_path "data/database_champions/datasets/enemy_dataset_tagteam_arena.npz" --dry_run
```

Train:

```bash
uv run python scripts/train_tagteam_arena_model.py \
  --data_path "data/database_champions/datasets/enemy_dataset_tagteam_arena.npz" \
  --output_dir "data/models/tagteam_arena_model/outputs" \
  --batch_size 32 \
  --num_workers 8 \
  --pin_memory \
  --epochs 200 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --patience 25 \
  --split_axis width \
  --seed 42
```

If key inference is ambiguous, pass explicit keys:

```bash
uv run python scripts/train_tagteam_arena_model.py \
  --data_path "data/tagteam arena.npz" \
  --image_key images \
  --power_key powers \
  --label_key labels \
  --dry_run
```

Use `--single_file` if you want to ignore numbered sibling shards and load only the exact `--data_path`.

`--num_workers` enables multiprocessing in PyTorch's `DataLoader`. The default is `8`; compare epoch time against `0`, `2`, or `4` if training feels slower than expected. This dataset is preloaded in memory, so more workers are not always faster. `--pin_memory` is most useful when training on CUDA.

## Outputs

Artifacts are saved under `data/models/tagteam_arena_model/outputs/` by default:

- `best_tagteam_model.pt`
- `training_history.csv`
- `normalization_stats.json`
- `validation_metrics.json`
- `validation_example_predictions.csv`

Existing artifact filenames are not overwritten. If a target file already exists, a versioned filename such as `_v2` is used.
