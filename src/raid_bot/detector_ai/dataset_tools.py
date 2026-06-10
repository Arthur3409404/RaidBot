from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_NAME = "pov_cyan_brighter_binary.png"
DEFAULT_CLASS_NAMES = ["label"]


@dataclass(frozen=True)
class AnnotationBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def normalized(self, width: int, height: int) -> tuple[float, float, float, float]:
        left = max(0, min(int(self.x1), int(self.x2)))
        top = max(0, min(int(self.y1), int(self.y2)))
        right = min(max(int(self.x1), int(self.x2)), int(width))
        bottom = min(max(int(self.y1), int(self.y2)), int(height))
        if right <= left or bottom <= top:
            return 0.0, 0.0, 0.0, 0.0
        x_center = ((left + right) / 2.0) / float(width)
        y_center = ((top + bottom) / 2.0) / float(height)
        box_w = (right - left) / float(width)
        box_h = (bottom - top) / float(height)
        return x_center, y_center, box_w, box_h


@dataclass(frozen=True)
class DatasetItem:
    source_image: Path
    annotation_path: Path
    boxes: list[AnnotationBox]


def _safe_slug(name: str) -> str:
    cleaned = [ch.lower() if ch.isalnum() else "_" for ch in str(name)]
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "dataset"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_single_annotation(annotation_path: Path) -> DatasetItem | None:
    if not annotation_path.exists():
        return None

    payload = _read_json(annotation_path)
    image_path = payload.get("image_path")
    if not image_path:
        return None

    source_image = Path(str(image_path)).expanduser()
    if not source_image.exists():
        return None

    boxes_raw = payload.get("boxes") or []
    boxes: list[AnnotationBox] = []
    for item in boxes_raw:
        try:
            boxes.append(
                AnnotationBox(
                    x1=int(item["x1"]),
                    y1=int(item["y1"]),
                    x2=int(item["x2"]),
                    y2=int(item["y2"]),
                )
            )
        except Exception:
            continue

    return DatasetItem(source_image=source_image, annotation_path=annotation_path, boxes=boxes)


def load_annotation_items(annotation_root: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    if not annotation_root.exists():
        return items

    for annotation_path in sorted(annotation_root.rglob("*.json")):
        item = _load_single_annotation(annotation_path)
        if item is not None:
            items.append(item)
    return items


def _split_items(items: list[DatasetItem], *, seed: int, val_ratio: float) -> tuple[list[DatasetItem], list[DatasetItem]]:
    if not items:
        return [], []
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, []
    val_count = max(1, int(round(len(shuffled) * float(val_ratio))))
    val_count = min(val_count, len(shuffled) - 1)
    val_items = shuffled[:val_count]
    train_items = shuffled[val_count:]
    return train_items, val_items


def _copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))


def _write_yolo_label(path: Path, boxes: Iterable[AnnotationBox], width: int, height: int) -> None:
    lines: list[str] = []
    for box in boxes:
        x_center, y_center, box_w, box_h = box.normalized(width, height)
        if box_w <= 0.0 or box_h <= 0.0:
            continue
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_yolo_dataset(
    annotation_root: Path,
    output_root: Path | None = None,
    *,
    dataset_name: str = "detector_ai_yolo",
    val_ratio: float = 0.2,
    seed: int = 42,
    class_names: list[str] | None = None,
    image_copy_prefix: str | None = None,
) -> dict:
    items = load_annotation_items(annotation_root)
    output_root = Path(output_root or (DATA_ROOT / "detector_ai_dataset"))
    dataset_root = output_root / _safe_slug(dataset_name)
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    dataset_root.mkdir(parents=True, exist_ok=True)

    train_items, val_items = _split_items(items, seed=seed, val_ratio=val_ratio)
    all_splits = {"train": train_items, "val": val_items}

    # Clear out only the generated dataset tree.
    for child in [images_root, labels_root]:
        if child.exists():
            shutil.rmtree(child)

    summary = {
        "dataset_root": str(dataset_root),
        "annotation_root": str(annotation_root),
        "dataset_name": dataset_name,
        "class_names": list(class_names or DEFAULT_CLASS_NAMES),
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "splits": {},
        "total_images": len(items),
        "total_boxes": sum(len(item.boxes) for item in items),
    }

    for split_name, split_items in all_splits.items():
        split_image_dir = images_root / split_name
        split_label_dir = labels_root / split_name
        split_image_dir.mkdir(parents=True, exist_ok=True)
        split_label_dir.mkdir(parents=True, exist_ok=True)

        used_names: set[str] = set()
        split_boxes = 0
        split_images = 0

        for index, item in enumerate(split_items, start=1):
            base_name = item.source_image.stem
            if image_copy_prefix:
                base_name = f"{_safe_slug(image_copy_prefix)}_{base_name}"
            candidate_name = f"{base_name}_{index:04d}"
            while candidate_name in used_names:
                candidate_name = f"{base_name}_{index:04d}_{len(used_names):02d}"
            used_names.add(candidate_name)

            destination_image = split_image_dir / f"{candidate_name}{item.source_image.suffix.lower()}"
            destination_label = split_label_dir / f"{candidate_name}.txt"

            _copy_image(item.source_image, destination_image)

            import cv2

            image = cv2.imread(str(item.source_image), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(item.source_image.as_posix())
            height, width = image.shape[:2]
            _write_yolo_label(destination_label, item.boxes, width=width, height=height)

            split_images += 1
            split_boxes += len(item.boxes)

        summary["splits"][split_name] = {
            "images": split_images,
            "boxes": split_boxes,
        }

    yaml_lines = [
        f"path: {dataset_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, class_name in enumerate(list(class_names or DEFAULT_CLASS_NAMES)):
        yaml_lines.append(f"  {idx}: {class_name}")
    (dataset_root / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    _write_json(dataset_root / "dataset_summary.json", summary)
    return summary


def save_annotation(annotation_path: Path, image_path: Path, boxes: list[AnnotationBox], *, image_size: tuple[int, int]) -> None:
    payload = {
        "image_path": image_path.resolve().as_posix(),
        "image_name": image_path.name,
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "class_names": DEFAULT_CLASS_NAMES,
        "boxes": [asdict(box) for box in boxes],
    }
    _write_json(annotation_path, payload)
