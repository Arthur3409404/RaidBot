"""Offline augmentation for champion icon source images."""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from .dataset import ChampionIndex, ChampionRecognitionError, list_image_files, load_rgb_image, normalize_label_like


LOGGER = logging.getLogger(__name__)
DEFAULT_SOURCE_FOLDER = Path("data/processed/icons")
LEGACY_SOURCE_FOLDER = Path("data/processed/champion_icons")
DEFAULT_OUTPUT_FOLDER = Path("data/processed/icons_noised")
DEFAULT_METADATA_PATH = Path("data/processed/augmentation_metadata.csv")
IMAGE_RESAMPLING = Image.Resampling.BICUBIC


@dataclass(frozen=True)
class AugmentationSummary:
    """Metadata about a completed augmentation run."""

    source_folder: Path
    output_folder: Path
    metadata_path: Path
    preview_path: Path | None
    original_icons_found: int
    labels_found: int
    variants_per_icon: int
    total_augmented_images_created: int


def create_noise_data(
    source_folder: Path | str = DEFAULT_SOURCE_FOLDER,
    output_folder: Path | str = DEFAULT_OUTPUT_FOLDER,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
    *,
    variants_per_icon: int = 25,
    random_seed: int = 42,
    labels_csv_path: Path | str = Path("data/processed/labels.csv"),
    preview_grid_path: Path | str | None = None,
    preview_examples: int = 8,
    overwrite: bool = False,
    target_size: tuple[int, int] | None = None,
) -> AugmentationSummary:
    """Create realistic noisy training variants from processed champion icons.

    The function is intentionally standalone: it only reads clean/source icons,
    writes augmented PNG files plus metadata, and never starts collection or
    training.
    """
    if variants_per_icon < 1:
        raise ValueError("variants_per_icon must be at least 1")

    source_path = _resolve_source_folder(Path(source_folder))
    output_path = Path(output_folder)
    metadata_output_path = Path(metadata_path)
    labels_path = Path(labels_csv_path)
    preview_path = Path(preview_grid_path) if preview_grid_path else None

    image_paths = list_image_files(source_path)
    if not image_paths:
        raise ChampionRecognitionError(f"No source icons found in {source_path}")

    champion_index = ChampionIndex.from_labels_csv(labels_path, source_path)
    records = []
    for image_path in image_paths:
        label = champion_index.recover_label(image_path, source_path)
        records.append((image_path, label))

    if target_size is None:
        target_size = _infer_target_size(image_paths)

    output_path.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_seed)
    metadata_rows: list[dict[str, str | int]] = []
    preview_pairs: list[tuple[Image.Image, Image.Image, str]] = []
    created = 0
    total_work = len(records) * variants_per_icon
    progress = _ProgressBar(total_work, "Generating augmented icons", enabled=sys.stderr.isatty())
    completed = 0

    try:
        for image_path, label in records:
            original = load_rgb_image(image_path).resize(target_size, IMAGE_RESAMPLING)
            for augmentation_id in range(1, variants_per_icon + 1):
                completed += 1
                augmented, applied = _augment_icon(original, rng)
                augmented = _ensure_size(augmented, target_size)
                output_file = output_path / _augmented_filename(label, image_path.stem, augmentation_id)

                if output_file.exists() and not overwrite:
                    progress.update(completed, created)
                    continue

                augmented.save(output_file, format="PNG", optimize=True)
                created += 1
                metadata_rows.append(
                    {
                        "original_image_path": image_path.as_posix(),
                        "augmented_image_path": output_file.as_posix(),
                        "champion_label": label,
                        "original_filename": image_path.name,
                        "augmentation_id": augmentation_id,
                        "random_seed": random_seed,
                        "applied_augmentations": "|".join(applied),
                    }
                )
                if preview_path and len(preview_pairs) < preview_examples:
                    preview_pairs.append((original.copy(), augmented.copy(), label))
                progress.update(completed, created)
    finally:
        progress.close()

    _write_metadata(metadata_output_path, metadata_rows, overwrite=overwrite)
    if preview_path and preview_pairs:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        _save_preview_grid(preview_pairs, preview_path)

    labels_found = len({label for _, label in records})
    summary = AugmentationSummary(
        source_folder=source_path,
        output_folder=output_path,
        metadata_path=metadata_output_path,
        preview_path=preview_path,
        original_icons_found=len(records),
        labels_found=labels_found,
        variants_per_icon=variants_per_icon,
        total_augmented_images_created=created,
    )
    _log_summary(summary)
    return summary


def _resolve_source_folder(source_folder: Path) -> Path:
    """Return the requested icon folder, with a compatibility fallback."""
    if source_folder.exists():
        return source_folder
    if source_folder == DEFAULT_SOURCE_FOLDER and LEGACY_SOURCE_FOLDER.exists():
        LOGGER.info("Source folder %s not found; using %s", source_folder, LEGACY_SOURCE_FOLDER)
        return LEGACY_SOURCE_FOLDER
    return source_folder


class _ProgressBar:
    """Render a compact single-line progress indicator to stderr."""

    def __init__(self, total: int, label: str, *, enabled: bool) -> None:
        self.total = max(1, total)
        self.label = label
        self.enabled = enabled
        self._last_percent = -1
        self._finished = False

    def update(self, completed: int, created: int) -> None:
        if not self.enabled:
            if completed == 1 or completed == self.total or created % 250 == 0:
                LOGGER.info(
                    "%s: %d/%d created=%d",
                    self.label,
                    completed,
                    self.total,
                    created,
                )
            return

        percent = int((completed * 100) / self.total)
        if percent == self._last_percent and completed != self.total:
            return
        self._last_percent = percent
        width = 24
        filled = int(width * completed / self.total)
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(f"\r{self.label}: [{bar}] {percent:3d}% ({completed}/{self.total}) created={created}")
        sys.stderr.flush()
        if completed == self.total:
            self._finished = True
            sys.stderr.write("\n")
            sys.stderr.flush()

    def close(self) -> None:
        if self.enabled and not self._finished:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _infer_target_size(image_paths: list[Path]) -> tuple[int, int]:
    """Use the first readable source icon size as the saved training size."""
    first = load_rgb_image(image_paths[0])
    return first.size


def _augment_icon(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, list[str]]:
    """Apply a randomized but mild sequence of screenshot-like augmentations."""
    augmented = image.copy()
    applied: list[str] = []

    augmented, applied = _maybe_apply(augmented, applied, 0.85, rng, _small_crop_shift)
    augmented, applied = _maybe_apply(augmented, applied, 0.80, rng, _small_scale_and_aspect)
    augmented, applied = _maybe_apply(augmented, applied, 0.80, rng, _tiny_translation)
    augmented, applied = _maybe_apply(augmented, applied, 0.55, rng, _very_small_rotation)
    augmented, applied = _maybe_apply(augmented, applied, 0.95, rng, _color_jitter)
    augmented, applied = _maybe_apply(augmented, applied, 0.65, rng, _blur_or_sharpen)
    augmented, applied = _maybe_apply(augmented, applied, 0.68, rng, _jpeg_artifacts)
    augmented, applied = _maybe_apply(augmented, applied, 0.82, rng, _gaussian_screenshot_noise)
    augmented, applied = _maybe_apply(augmented, applied, 0.40, rng, _small_text_overlay)
    augmented, applied = _maybe_apply(augmented, applied, 0.35, rng, _transparent_ui_patch)
    augmented, applied = _maybe_apply(augmented, applied, 0.40, rng, _small_border_overlay)
    augmented, applied = _maybe_apply(augmented, applied, 0.35, rng, _edge_occlusion)

    if not applied:
        augmented, description = _gaussian_screenshot_noise(augmented, rng)
        applied.append(description)
    return augmented, applied


def _maybe_apply(
    image: Image.Image,
    applied: list[str],
    probability: float,
    rng: np.random.Generator,
    transform: Callable[[Image.Image, np.random.Generator], tuple[Image.Image, str]],
) -> tuple[Image.Image, list[str]]:
    """Apply one transform with a probability and record its description."""
    if rng.random() >= probability:
        return image, applied
    image, description = transform(image, rng)
    applied.append(description)
    return image, applied


def _small_crop_shift(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, str]:
    """Crop a tiny shifted region and resize back to the original icon size."""
    width, height = image.size
    crop_fraction = float(rng.uniform(0.01, 0.045))
    inset_x = max(1, int(round(width * crop_fraction)))
    inset_y = max(1, int(round(height * crop_fraction)))
    max_shift_x = inset_x
    max_shift_y = inset_y
    shift_x = int(rng.integers(-max_shift_x, max_shift_x + 1))
    shift_y = int(rng.integers(-max_shift_y, max_shift_y + 1))
    left = max(0, min(2 * inset_x, inset_x + shift_x))
    top = max(0, min(2 * inset_y, inset_y + shift_y))
    right = min(width, width - (2 * inset_x - left))
    bottom = min(height, height - (2 * inset_y - top))
    cropped = image.crop((left, top, right, bottom)).resize((width, height), IMAGE_RESAMPLING)
    return cropped, f"crop_shift:{left},{top},{right},{bottom}"


def _small_scale_and_aspect(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, str]:
    """Resize with small scale and aspect-ratio variation, then fit back."""
    width, height = image.size
    scale = float(rng.uniform(0.965, 1.045))
    aspect = float(rng.uniform(0.975, 1.025))
    new_width = max(1, int(round(width * scale * aspect)))
    new_height = max(1, int(round(height * scale / aspect)))
    resized = image.resize((new_width, new_height), IMAGE_RESAMPLING)
    return _fit_to_canvas(resized, (width, height), rng), f"scale_aspect:{scale:.3f},{aspect:.3f}"


def _tiny_translation(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, str]:
    """Move the icon by only a few pixels without wrapping edge content."""
    width, height = image.size
    dx = int(rng.integers(-3, 4))
    dy = int(rng.integers(-4, 5))
    canvas = Image.new("RGB", (width, height), _edge_fill_color(image))
    canvas.paste(image, (dx, dy))
    return canvas, f"translation:{dx},{dy}"


def _very_small_rotation(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, str]:
    """Rotate by a very small angle while preserving the canvas size."""
    angle = float(rng.uniform(-2.2, 2.2))
    rotated = image.rotate(angle, resample=IMAGE_RESAMPLING, fillcolor=_edge_fill_color(image))
    return rotated, f"rotation:{angle:.2f}"


def _color_jitter(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, str]:
    """Apply screenshot-like brightness, contrast, and saturation changes."""
    brightness = float(rng.uniform(0.88, 1.14))
    contrast = float(rng.uniform(0.88, 1.16))
    saturation = float(rng.uniform(0.88, 1.14))
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(saturation)
    return image, f"color:{brightness:.2f},{contrast:.2f},{saturation:.2f}"


def _blur_or_sharpen(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, str]:
    """Apply mild blur or mild sharpening."""
    if rng.random() < 0.58:
        radius = float(rng.uniform(0.25, 0.85))
        return image.filter(ImageFilter.GaussianBlur(radius=radius)), f"blur:{radius:.2f}"
    factor = float(rng.uniform(1.12, 1.55))
    return ImageEnhance.Sharpness(image).enhance(factor), f"sharpen:{factor:.2f}"


def _jpeg_artifacts(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, str]:
    """Round-trip through JPEG to mimic compressed screenshots."""
    quality = int(rng.integers(42, 83))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB"), f"jpeg:{quality}"


def _gaussian_screenshot_noise(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, str]:
    """Add low-amplitude Gaussian pixel noise."""
    sigma = float(rng.uniform(2.5, 8.5))
    array = np.asarray(image, dtype=np.float32)
    noise = rng.normal(0.0, sigma, size=array.shape)
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB"), f"gaussian_noise:{sigma:.1f}"


def _small_text_overlay(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, str]:
    """Draw a tiny UI-like number or short text near an edge or corner."""
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    text = str(int(rng.integers(1, 120))) if rng.random() < 0.85 else rng.choice(["I", "II", "III", "IV", "V", "+", "x", "?"])
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x, y = _random_edge_position(width, height, text_width, text_height, rng)
    shadow = (0, 0, 0, int(rng.integers(90, 150)))
    color = (235, 230, 190, int(rng.integers(135, 205)))
    draw.text((x + 1, y + 1), text, font=font, fill=shadow)
    draw.text((x, y), text, font=font, fill=color)
    return overlay.convert("RGB"), f"text_overlay:{text}"


def _transparent_ui_patch(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, str]:
    """Add a translucent rectangular UI-like patch."""
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = image.size
    patch_width = int(rng.integers(max(4, width // 8), max(5, width // 3)))
    patch_height = int(rng.integers(max(4, height // 10), max(5, height // 4)))
    x, y = _random_edge_position(width, height, patch_width, patch_height, rng)
    alpha = int(rng.integers(35, 95))
    color = tuple(int(value) for value in rng.integers(15, 70, size=3)) + (alpha,)
    draw.rounded_rectangle((x, y, x + patch_width, y + patch_height), radius=2, fill=color)
    return overlay.convert("RGB"), f"ui_patch:{x},{y},{patch_width},{patch_height}"


def _small_border_overlay(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, str]:
    """Overlay a thin border, as capture crops often include UI frame edges."""
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = image.size
    thickness = int(rng.integers(1, 3))
    alpha = int(rng.integers(50, 130))
    color = tuple(int(value) for value in rng.integers(120, 230, size=3)) + (alpha,)
    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    for side in sides[: int(rng.integers(1, 3))]:
        if side == "top":
            draw.rectangle((0, 0, width, thickness), fill=color)
        elif side == "bottom":
            draw.rectangle((0, height - thickness, width, height), fill=color)
        elif side == "left":
            draw.rectangle((0, 0, thickness, height), fill=color)
        else:
            draw.rectangle((width - thickness, 0, width, height), fill=color)
    return overlay.convert("RGB"), f"border:{thickness}"


def _edge_occlusion(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, str]:
    """Cover a small edge/corner region while keeping the face recognizable."""
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = image.size
    occ_width = int(rng.integers(max(3, width // 10), max(4, width // 4)))
    occ_height = int(rng.integers(max(3, height // 12), max(4, height // 5)))
    x, y = _random_edge_position(width, height, occ_width, occ_height, rng)
    color = tuple(int(value) for value in rng.integers(0, 45, size=3)) + (int(rng.integers(90, 170)),)
    draw.rectangle((x, y, x + occ_width, y + occ_height), fill=color)
    return overlay.convert("RGB"), f"edge_occlusion:{x},{y},{occ_width},{occ_height}"


def _fit_to_canvas(image: Image.Image, size: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    """Crop or pad an image to exactly the requested canvas size."""
    target_width, target_height = size
    width, height = image.size

    if width > target_width:
        max_left = width - target_width
        left = int(rng.integers(0, max_left + 1))
        image = image.crop((left, 0, left + target_width, height))
        width = target_width
    if height > target_height:
        max_top = height - target_height
        top = int(rng.integers(0, max_top + 1))
        image = image.crop((0, top, width, top + target_height))
        height = target_height

    canvas = Image.new("RGB", size, _edge_fill_color(image))
    x = int(rng.integers(0, target_width - width + 1)) if width < target_width else 0
    y = int(rng.integers(0, target_height - height + 1)) if height < target_height else 0
    canvas.paste(image, (x, y))
    return canvas


def _ensure_size(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Guarantee final saved image dimensions."""
    if image.size == size:
        return image.convert("RGB")
    return ImageOps.fit(image.convert("RGB"), size, method=IMAGE_RESAMPLING, centering=(0.5, 0.5))


def _edge_fill_color(image: Image.Image) -> tuple[int, int, int]:
    """Estimate a neutral fill color from image edges."""
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    edges = np.concatenate((array[0, :, :], array[-1, :, :], array[:, 0, :], array[:, -1, :]), axis=0)
    color = np.median(edges, axis=0)
    return tuple(int(value) for value in color)


def _random_edge_position(
    width: int,
    height: int,
    object_width: int,
    object_height: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Choose a position close to an edge or corner."""
    margin = 2
    max_x = max(0, width - object_width - margin)
    max_y = max(0, height - object_height - margin)
    side = str(rng.choice(["top", "bottom", "left", "right", "corner"]))
    if side == "top":
        return int(rng.integers(margin, max_x + 1)), margin
    if side == "bottom":
        return int(rng.integers(margin, max_x + 1)), max_y
    if side == "left":
        return margin, int(rng.integers(margin, max_y + 1))
    if side == "right":
        return max_x, int(rng.integers(margin, max_y + 1))
    return int(rng.choice([margin, max_x])), int(rng.choice([margin, max_y]))


def _augmented_filename(label: str, original_stem: str, augmentation_id: int) -> str:
    """Build a stable filename that preserves label and source identity."""
    safe_label = normalize_label_like(label)
    safe_stem = normalize_label_like(original_stem)
    prefix = safe_stem if safe_stem.startswith(safe_label) else f"{safe_label}_{safe_stem}"
    return f"{prefix}_aug_{augmentation_id:04d}.png"


def _write_metadata(metadata_path: Path, rows: list[dict[str, str | int]], *, overwrite: bool) -> None:
    """Write or append augmentation metadata rows."""
    fieldnames = [
        "original_image_path",
        "augmented_image_path",
        "champion_label",
        "original_filename",
        "augmentation_id",
        "random_seed",
        "applied_augmentations",
    ]
    mode = "w" if overwrite or not metadata_path.exists() else "a"
    write_header = mode == "w" or metadata_path.stat().st_size == 0
    with metadata_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _save_preview_grid(pairs: list[tuple[Image.Image, Image.Image, str]], preview_path: Path) -> None:
    """Save a compact grid showing original and augmented examples."""
    cell_width = max(image.width for pair in pairs for image in pair[:2])
    cell_height = max(image.height for pair in pairs for image in pair[:2]) + 12
    columns = 2
    rows = len(pairs)
    grid = Image.new("RGB", (columns * cell_width, rows * cell_height), (24, 24, 24))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for row, (original, augmented, label) in enumerate(pairs):
        y = row * cell_height
        grid.paste(original, (0, y))
        grid.paste(augmented, (cell_width, y))
        draw.text((1, y + original.height + 1), f"orig {label[:12]}", fill=(230, 230, 230), font=font)
        draw.text((cell_width + 1, y + augmented.height + 1), "aug", fill=(230, 230, 230), font=font)

    grid.save(preview_path)


def _log_summary(summary: AugmentationSummary) -> None:
    """Log the requested run summary fields."""
    LOGGER.info("Original icons found: %s", summary.original_icons_found)
    LOGGER.info("Labels/champions found: %s", summary.labels_found)
    LOGGER.info("Variants per icon: %s", summary.variants_per_icon)
    LOGGER.info("Total augmented images created: %s", summary.total_augmented_images_created)
    LOGGER.info("Output folder path: %s", summary.output_folder)
