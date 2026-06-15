# Raid Champion Icon Recognizer

This project collects Raid Shadow Legends champion portrait icons and trains a
pretrained, prototype-based recognizer. Data collection and recognition are kept
separate: collector commands create/crop icons, while recognizer commands train,
build prototypes, evaluate, and predict from existing image folders.

## Collect Data

Collect one AyumiLove champion page:

```bash
uv run python scripts/champion_icons.py collect --url "https://ayumilove.net/raid-shadow-legends-relickeeper-skill-mastery-equip-guide/"
```

The collector caches pages and raw images, writes processed icons to `data/processed/icons/`, and records the per-champion record in `data/processed/labels.csv`. Each row saves:

- normalized `label` such as `relickeeper`
- readable `champion_name` such as `Relickeeper`
- `source_url`
- `raw_image_path`
- `processed_image_path`
- `crop_box`
- original `image_width` and `image_height`
- processed icon size, fixed to `48x64`

The collector also refreshes `data/processed/dataset_manifest.csv` automatically as a convenience dataset export. The manifest keeps the core fields above plus ranking metadata and source-page details for future tooling. Existing files are reused unless `--force` is passed.

## Configure The Shared Crop

The champion portrait is expected to live at the same relative position in future source images. To select that crop manually from a downloaded source image:

```bash
uv run python scripts/champion_icons.py configure-crop --url "https://ayumilove.net/raid-shadow-legends-relickeeper-skill-mastery-equip-guide/"
```

Draw a rectangle around the icon in the image window, then press Enter. The selected relative coordinates are saved to `data/processed/crop_config.json`, and a visual check image is saved in `data/processed/debug/`. Existing crop configs and processed outputs are not overwritten unless `--force` is passed.

## Recognizer Pipeline

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

The recognizer is independent from data collection. It expects the training and
reference folders below:

- clean reference icons: `data/processed/icons/`
- noisy augmented training icons: `data/processed/icons_noised/`
- checkpoints: `data/models/champion_recognition/checkpoints/`
- logs: `outputs/logs/`
- plots: `outputs/plots/`
- review samples: `outputs/review_samples/`

Create offline augmented training icons from the clean processed icons when needed:

```bash
uv run python scripts/champion_icons.py noise-data --variants-per-icon 25 --seed 42
```

This reads clean/source icons from `data/processed/icons/` by default, falls back to
the legacy collector folder `data/processed/champion_icons/` when needed, and writes
augmented training images to `data/processed/icons_noised/` without overwriting the
clean originals. It also writes `data/processed/augmentation_metadata.csv`.

To inspect examples:

```bash
uv run python scripts/champion_icons.py noise-data --preview-grid data/models/champion_recognition/plots/augmentation_preview.png --overwrite
```

### Train

Train the pretrained embedding pipeline. By default it trains on noisy variants
in `data/processed/icons_noised/` and builds clean prototypes from
`data/processed/icons/`:

```bash
uv run python scripts/champion_icons.py train-recognizer --force
```

Outputs are written to:

- `data/models/champion_recognition/checkpoints/champion_icon_recognizer.pt`
- `data/models/champion_recognition/checkpoints/champion_prototypes.pt`
- `outputs/logs/training_summary.json`
- `outputs/logs/evaluation.json`
- `outputs/logs/threshold_report.csv`
- `outputs/review_samples/`

The default backbone is `convnext_tiny` from `timm`. You can also try backbones such as
`efficientnet_b0`, `mobilenetv3_large_100`, or `resnet50`:

```bash
uv run python scripts/champion_icons.py train-recognizer \
  --train-image-folder data/processed/icons_noised \
  --reference-icon-folder data/processed/icons \
  --backbone convnext_tiny \
  --min-similarity-threshold 0.65 \
  --min-margin-threshold 0.02 \
  --force
```

This trains a classifier head with the backbone frozen, optionally fine-tunes the last
backbone blocks with `--fine-tune-epochs`, then saves one clean prototype embedding per
champion. The final prediction still uses nearest-prototype cosine similarity.

Validation on `data/processed/icons/` is a clean-reference sanity check. A stronger
real-world test set should later be created from actual noisy in-game screenshots.

### Build Champion Prototypes

Training creates prototypes automatically, but you can rebuild them from a saved
checkpoint without retraining:

```bash
uv run python scripts/champion_icons.py build-prototypes
```

This reads clean champion icons from `data/processed/icons/` and writes:

- `data/models/champion_recognition/checkpoints/champion_prototypes.pt`

### Evaluate

```bash
uv run python scripts/champion_icons.py evaluate-recognizer
```

Evaluation writes:

- `outputs/logs/evaluation.json`
- `outputs/logs/predictions.csv`
- `outputs/logs/threshold_report.csv`
- uncertain or wrong review images under `outputs/review_samples/`

### Predict

```bash
uv run python scripts/champion_icons.py predict --image some_icon.png
```

This prints exactly one result: the predicted champion name, or `None` when the best
similarity or best-vs-second margin is below the configured thresholds.

## Add More Champions Later

Run the collector for more AyumiLove champion page URLs:

```bash
uv run python scripts/champion_icons.py collect --url "https://ayumilove.net/raid-shadow-legends-another-champion-guide/"
```

Inspect the generated crop debug image in `data/processed/debug/`. If a page layout needs crop tuning, adjust the relative crop box through the CLI options such as `--crop-left`, `--crop-top`, `--crop-right`, and `--crop-bottom`, then rerun with `--force` for that page.

To build a larger local dataset from the AyumiLove ranking page:

```bash
uv run python scripts/champion_icons.py collect-ranking --url "https://ayumilove.net/raid-shadow-legends-list-of-champions-by-ranking/"
```

For a quick smoke test, collect only the first few discovered links:

```bash
uv run python scripts/champion_icons.py collect-ranking --url "https://ayumilove.net/raid-shadow-legends-list-of-champions-by-ranking/" --limit 5
```

The batch collector reuses cached pages/images, waits between network requests, skips overwriting processed files unless `--force` is passed, and continues past individual page failures by default.
When you use `collect-ranking`, the collected `labels.csv` also includes `ranking_tier` and `ranking_code` columns. `ranking_tier` stores the champion's tier from the index page, such as `SS`, `S`, `A`, `B`, `C`, or `F`, and `ranking_code` stores the abbreviation code from the link text, such as `KR-MSF` for Embrys.

To check whether AyumiLove has new champion links compared with your current dataset:

```bash
uv run python scripts/champion_icons.py check4updates
```

If you adjust `data/processed/crop_config.json` after collecting some icons, reprocess cached raw images without redownloading them:

```bash
uv run python scripts/champion_icons.py reprocess-icons --force
```

After adding more champions, retrain with:

```bash
uv run python scripts/champion_icons.py train-recognizer --force
```
