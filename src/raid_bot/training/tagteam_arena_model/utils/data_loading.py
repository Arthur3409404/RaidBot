from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # Allows .npz inspection before PyTorch is installed.
    torch = None
    Dataset = object


IMAGE_KEY_HINTS = ("image", "images", "img", "screenshot", "screenshots", "crop", "crops")
POWER_KEY_HINTS = ("power", "powers", "team_power", "enemy_power", "enemy_powers")
LABEL_KEY_HINTS = ("label", "labels", "result", "results", "outcome", "outcomes", "won", "win")


@dataclass(frozen=True)
class PreparedArrays:
    crops: np.ndarray
    powers: np.ndarray
    labels: np.ndarray
    image_key: str
    power_key: str
    label_key: str
    source_files: tuple[str, ...]


def describe_array(name: str, array: np.ndarray, sample_count: int = 10) -> dict[str, Any]:
    info: dict[str, Any] = {
        "key": name,
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size and np.issubdtype(array.dtype, np.number):
        numeric = array.astype(np.float64, copy=False)
        finite = numeric[np.isfinite(numeric)]
        if finite.size:
            info["min"] = float(np.min(finite))
            info["max"] = float(np.max(finite))
    if array.size <= 30 or array.ndim <= 1:
        info["sample"] = array.reshape(-1)[:sample_count].tolist()
    return info


def shard_sort_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    prefix, sep, suffix = stem.rpartition("_")
    if sep and suffix.isdigit():
        return prefix, int(suffix)
    return stem, 0


def resolve_npz_paths(path: str | Path, include_shards: bool = True) -> list[Path]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if not include_shards:
        return [path]

    base_stem = path.stem
    pattern = f"{base_stem}*.npz"
    paths = []
    for candidate in path.parent.glob(pattern):
        if candidate.stem == base_stem:
            paths.append(candidate)
            continue
        suffix = candidate.stem.removeprefix(base_stem)
        if suffix.startswith("_") and suffix[1:].isdigit():
            paths.append(candidate)
    return sorted(set(paths), key=shard_sort_key)


def print_npz_inspection(path: str | Path, include_shards: bool = True) -> list[dict[str, Any]]:
    paths = resolve_npz_paths(path, include_shards=include_shards)
    print(f"Source files ({len(paths)}):")
    for source in paths:
        print(f"  {source}")

    summaries = []
    total_samples = 0
    merged_shapes: dict[str, list[tuple[int, ...]]] = {}
    first_data = np.load(paths[0], allow_pickle=True)
    try:
        keys = list(first_data.keys())
    finally:
        first_data.close()

    for source in paths:
        with np.load(source, allow_pickle=True) as data:
            shard_keys = list(data.keys())
            print(f"\nLoaded: {source}")
            print(f"Available keys: {shard_keys}")
            for key in shard_keys:
                array = data[key]
                summary = describe_array(key, array)
                summaries.append({"source": str(source), **summary})
                merged_shapes.setdefault(key, []).append(tuple(array.shape))
                print(f"\n[{key}]")
                print(f"  shape: {summary['shape']}")
                print(f"  dtype: {summary['dtype']}")
                if "min" in summary:
                    print(f"  range: {summary['min']} .. {summary['max']}")
                if "sample" in summary:
                    print(f"  sample: {summary['sample']}")
            if shard_keys and data[shard_keys[0]].shape:
                total_samples += int(data[shard_keys[0]].shape[0])

    print("\nRaw shard shape summary:")
    for key, shapes in merged_shapes.items():
        unique_shapes = sorted(set(shapes))
        print(f"  {key}: {unique_shapes}")
    print(f"Raw samples across shards, inferred from first dimensions: {total_samples}")

    try:
        arrays, image_key, power_key, label_key, _ = load_raw_arrays(paths)
    except ValueError as exc:
        print("\nStructured dataset detected; legacy image preparation is skipped.")
        print(f"  {exc}")
        return summaries

    print("\nAfter shard merge and legacy cleanup:")
    for key, array in arrays.items():
        summary = describe_array(key, array)
        print(f"  {key}: shape={summary['shape']} dtype={summary['dtype']}", end="")
        if "min" in summary:
            print(f" range={summary['min']}..{summary['max']}")
        else:
            print()
    print(f"Number of usable samples: {len(arrays[label_key])}")
    print(f"Example labels from '{label_key}': {arrays[label_key].reshape(-1)[:10].tolist()}")
    print(f"Example power values from '{power_key}': {arrays[power_key].reshape(-1)[:12].tolist()}")
    return summaries


def load_raw_arrays(
    paths: list[Path],
    image_key: str | None = None,
    power_key: str | None = None,
    label_key: str | None = None,
) -> tuple[dict[str, np.ndarray], str, str, str, tuple[str, ...]]:
    with np.load(paths[0], allow_pickle=True) as data:
        keys = list(data.keys())
        image_key = image_key or infer_key(keys, data, "image")
        power_key = power_key or infer_key(keys, data, "power")
        label_key = label_key or infer_key(keys, data, "label")
    if not image_key or not power_key or not label_key:
        raise ValueError(_manual_mapping_message(keys, image_key, power_key, label_key))

    images_list = []
    powers_list = []
    labels_list = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            missing = [key for key in (image_key, power_key, label_key) if key not in data.keys()]
            if missing:
                raise KeyError(f"Requested key(s) not found in {path}: {missing}. Available keys: {list(data.keys())}")
            images = np.asarray(data[image_key])
            powers = np.asarray(data[power_key])
            labels = np.asarray(data[label_key])
            images, powers, labels = cleanup_legacy_tagteam_shard(images, powers, labels, path)
            images_list.append(images)
            powers_list.append(powers)
            labels_list.append(labels)

    arrays = {
        image_key: np.concatenate(images_list, axis=0),
        power_key: np.concatenate(powers_list, axis=0),
        label_key: np.concatenate(labels_list, axis=0),
    }
    return arrays, image_key, power_key, label_key, tuple(str(path) for path in paths)


def cleanup_legacy_tagteam_shard(
    images: np.ndarray,
    powers: np.ndarray,
    labels: np.ndarray,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        powers.ndim == 1
        and labels.ndim == 1
        and images.shape[0] == powers.shape[0] == labels.shape[0]
        and powers.shape[0] % 4 == 0
    ):
        print(
            f"Legacy flat tag-team shard detected in {path.name}; "
            "using every 4th image/label and reshaping powers to [N, 4]."
        )
        return images[::4], powers.reshape(-1, 4), labels[::4]
    return images, powers, labels


def infer_key(keys: list[str], data: Any, kind: str) -> str | None:
    hints = {"image": IMAGE_KEY_HINTS, "power": POWER_KEY_HINTS, "label": LABEL_KEY_HINTS}[kind]
    scored: list[tuple[int, str]] = []
    for key in keys:
        lower = key.lower()
        array = data[key]
        score = 0
        if any(hint in lower for hint in hints):
            score += 10
        if kind == "image" and array.ndim in {3, 4, 5}:
            score += 4
            if np.issubdtype(array.dtype, np.number):
                score += 1
        elif kind == "power" and array.ndim in {1, 2} and np.issubdtype(array.dtype, np.number):
            score += 4
            if array.ndim == 2 and array.shape[-1] in {3, 4}:
                score += 3
        elif kind == "label" and array.ndim <= 2:
            unique_count = len(np.unique(array.reshape(-1))) if array.size else 0
            if unique_count <= 10:
                score += 4
            if array.ndim == 1 or (array.ndim == 2 and 1 in array.shape):
                score += 2
        if score > 0:
            scored.append((score, key))
    if not scored:
        return None
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def _manual_mapping_message(keys: list[str], image_key: str | None, power_key: str | None, label_key: str | None) -> str:
    return (
        "Could not infer dataset keys safely.\n"
        f"Found keys: {keys}\n"
        f"Likely image key: {image_key}\n"
        f"Likely power key: {power_key}\n"
        f"Likely label key: {label_key}\n"
        "Please rerun with explicit mapping, for example:\n"
        "  --image_key images --power_key powers --label_key labels\n"
        "If powers are only visible in the screenshots, OCR or a separate powers array is required."
    )


def load_prepared_arrays(
    data_path: str | Path,
    image_key: str | None = None,
    power_key: str | None = None,
    label_key: str | None = None,
    split_axis: str = "width",
    grayscale_to_rgb: bool = False,
    include_shards: bool = True,
) -> PreparedArrays:
    paths = resolve_npz_paths(data_path, include_shards=include_shards)
    arrays, image_key, power_key, label_key, source_files = load_raw_arrays(
        paths,
        image_key=image_key,
        power_key=power_key,
        label_key=label_key,
    )
    images = arrays[image_key]
    powers = arrays[power_key]
    labels = arrays[label_key]

    crops = images_to_crops_n3chw(images, split_axis=split_axis, grayscale_to_rgb=grayscale_to_rgb)
    powers = powers_to_n3(powers)
    labels = labels_to_float(labels)

    n = len(labels)
    if crops.shape[0] != n or powers.shape[0] != n:
        raise ValueError(
            f"Sample count mismatch: crops={crops.shape[0]}, powers={powers.shape[0]}, labels={labels.shape[0]}"
        )
    return PreparedArrays(
        crops=crops,
        powers=powers,
        labels=labels,
        image_key=image_key,
        power_key=power_key,
        label_key=label_key,
        source_files=source_files,
    )


def images_to_crops_n3chw(images: np.ndarray, split_axis: str, grayscale_to_rgb: bool) -> np.ndarray:
    if images.ndim < 3:
        raise ValueError(f"Image array must have at least 3 dimensions, got {images.shape}")
    array = np.asarray(images)

    if array.ndim == 3:
        array = array[..., None]  # [N, H, W, 1]

    if array.ndim == 4:
        nhwc = to_nhwc(array)
        crops = split_full_images(nhwc, split_axis=split_axis)  # [N, 3, H, W, C]
    elif array.ndim == 5:
        crops = to_n3hwc(array)
    else:
        raise ValueError(f"Unsupported image array shape: {array.shape}")

    crops = crops.astype(np.float32)
    if crops.max(initial=0) > 1.5:
        crops = crops / 255.0
    crops = np.clip(crops, 0.0, 1.0)
    if crops.shape[-1] == 1 and grayscale_to_rgb:
        crops = np.repeat(crops, 3, axis=-1)
    return np.transpose(crops, (0, 1, 4, 2, 3)).astype(np.float32)


def to_nhwc(array: np.ndarray) -> np.ndarray:
    if array.shape[-1] in {1, 3, 4}:
        return array
    if array.shape[1] in {1, 3, 4}:
        return np.transpose(array, (0, 2, 3, 1))
    return array[..., None] if array.ndim == 3 else array


def to_n3hwc(array: np.ndarray) -> np.ndarray:
    if array.shape[1] == 3:
        slots_first = array
    elif array.shape[-4] == 3:
        slots_first = np.moveaxis(array, -4, 1)
    else:
        raise ValueError(f"Could not find a 3-slot dimension in image shape {array.shape}")

    if slots_first.shape[-1] in {1, 3, 4}:
        return slots_first
    if slots_first.shape[2] in {1, 3, 4}:
        return np.transpose(slots_first, (0, 1, 3, 4, 2))
    if slots_first.ndim == 5:
        return slots_first[..., None]
    raise ValueError(f"Could not convert 5D image shape {array.shape} to [N, 3, H, W, C]")


def split_full_images(images_nhwc: np.ndarray, split_axis: str) -> np.ndarray:
    if split_axis not in {"width", "height", "auto"}:
        raise ValueError("--split_axis must be width, height, or auto")
    height, width = images_nhwc.shape[1], images_nhwc.shape[2]
    axis = split_axis
    if axis == "auto":
        axis = "width" if width >= height else "height"
    if axis == "width":
        pieces = np.array_split(images_nhwc, 3, axis=2)
    else:
        pieces = np.array_split(images_nhwc, 3, axis=1)
    min_h = min(piece.shape[1] for piece in pieces)
    min_w = min(piece.shape[2] for piece in pieces)
    pieces = [piece[:, :min_h, :min_w, :] for piece in pieces]
    return np.stack(pieces, axis=1)


def powers_to_n3(powers: np.ndarray) -> np.ndarray:
    array = np.asarray(powers, dtype=np.float32)
    if array.ndim == 1:
        if array.size % 3 == 0:
            array = array.reshape(-1, 3)
        else:
            raise ValueError(f"Power array is 1D with length {array.size}; cannot reshape safely to [N, 3].")
    if array.ndim != 2:
        raise ValueError(f"Power array must be [N, 3] or [N, 4], got {array.shape}")
    if array.shape[1] == 3:
        return array.astype(np.float32)
    if array.shape[1] == 4:
        first = array[:, 0]
        rest_sum = array[:, 1:].sum(axis=1)
        if np.nanmedian(np.abs(first - rest_sum) / np.maximum(rest_sum, 1.0)) < 0.25:
            print("Power array has 4 columns; using columns 1..3 as team powers and treating column 0 as total power.")
            return array[:, 1:].astype(np.float32)
        raise ValueError(
            f"Power array has shape {array.shape}. It may include total+3 team powers, but this was not clear."
        )
    raise ValueError(f"Power array must have 3 team-power columns, got {array.shape}")


def labels_to_float(labels: np.ndarray) -> np.ndarray:
    flat = np.asarray(labels).reshape(-1)
    if np.issubdtype(flat.dtype, np.number):
        values = flat.astype(np.float32)
        unique = set(np.unique(values).tolist())
        if not unique.issubset({0.0, 1.0}):
            print(f"Warning: numeric labels are not strictly binary: {sorted(unique)[:10]}")
        return values

    converted = []
    for value in flat:
        text = str(value).strip().lower()
        if text in {"1", "win", "won", "true", "yes", "w"}:
            converted.append(1.0)
        elif text in {"0", "loss", "lost", "lose", "false", "no", "l"}:
            converted.append(0.0)
        else:
            raise ValueError(f"Could not convert label value {value!r} to 0/1")
    return np.asarray(converted, dtype=np.float32)


class TagTeamArenaDataset(Dataset):
    def __init__(self, crops: np.ndarray, powers: np.ndarray, labels: np.ndarray, augment: bool = False):
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required to create TagTeamArenaDataset.")
        self.crops = torch.from_numpy(crops.astype(np.float32))
        self.powers = torch.from_numpy(powers.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.augment = augment

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        crops = self.crops[index].clone()
        if self.augment:
            crops = augment_crops(crops)
        crops = (crops - 0.5) / 0.5
        return crops, self.powers[index], self.labels[index]


def augment_crops(crops: torch.Tensor) -> torch.Tensor:
    brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.12
    contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.12
    mean = crops.mean(dim=(-2, -1), keepdim=True)
    crops = (crops - mean) * contrast + mean
    crops = crops * brightness
    if torch.rand(1).item() < 0.5:
        shift_y = int(torch.randint(-1, 2, (1,)).item())
        shift_x = int(torch.randint(-1, 2, (1,)).item())
        crops = torch.roll(crops, shifts=(shift_y, shift_x), dims=(-2, -1))
    crops = crops + torch.randn_like(crops) * 0.01
    return crops.clamp(0.0, 1.0)


def normalize_powers(
    train_powers: np.ndarray,
    *other_powers: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, float]]:
    train_log = np.log1p(train_powers.astype(np.float32))
    mean = float(train_log.mean())
    std = float(train_log.std())
    if std < 1e-6:
        std = 1.0
    normalized_train = (train_log - mean) / std
    normalized_other = [(np.log1p(arr.astype(np.float32)) - mean) / std for arr in other_powers]
    return normalized_train.astype(np.float32), [arr.astype(np.float32) for arr in normalized_other], {
        "power_transform": "log1p_then_standardize",
        "log1p_mean": mean,
        "log1p_std": std,
    }


def save_json(data: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
