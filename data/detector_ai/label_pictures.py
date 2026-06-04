from __future__ import annotations

import argparse
import sys
import ctypes
from dataclasses import dataclass
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.detector_ai.dataset_tools import AnnotationBox, DEFAULT_IMAGE_NAME, export_yolo_dataset, save_annotation


DEBUG_ROOT = Path("debug")
ANNOTATION_ROOT = Path("data") / "detector_ai" / "annotations"
YOLO_OUTPUT_ROOT = Path("data") / "detector_ai"
WINDOW_NAME = "detector_ai_labeler"
DISPLAY_MAX_WIDTH = 1600
DISPLAY_MAX_HEIGHT = 900
MIN_BOX_SIZE = 4


@dataclass
class LabelState:
    image: object
    image_path: Path
    annotation_path: Path
    boxes: list[AnnotationBox]
    display_image: object
    scale: float
    draft_start: tuple[int, int] | None = None
    draft_end: tuple[int, int] | None = None
    drawing: bool = False


def _list_debug_collections(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _list_run_dirs(collection_dir: Path, image_name: str) -> list[Path]:
    runs: list[Path] = []
    for path in sorted(collection_dir.iterdir()):
        if not path.is_dir() or not path.name.startswith("run_"):
            continue
        if (path / image_name).exists():
            runs.append(path)
    return runs


def _pick_collection(collections: list[Path]) -> Path:
    print("Select debug collection:")
    for i, collection in enumerate(collections, start=1):
        print(f"  {i}. {collection.as_posix()}")

    while True:
        choice = input("Enter number: ").strip()
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue
        idx = int(choice) - 1
        if 0 <= idx < len(collections):
            return collections[idx]
        print("Choice out of range.")


def _load_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path.as_posix())
    return image


def _fit_scale(width: int, height: int) -> float:
    return 1.0


def _resize_for_display(image, scale: float):
    return image.copy()


def _overlay_help(canvas):
    lines = [
        "Left-drag: draw box   Enter: next image   u: undo   c: clear   q/esc: quit",
        "Multiple boxes are allowed. Empty images are allowed.",
    ]
    y = 24
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 22


def _draw_boxes(canvas, boxes: list[AnnotationBox], scale: float, draft: AnnotationBox | None = None):
    for box in boxes:
        x1 = int(round(box.x1 * scale))
        y1 = int(round(box.y1 * scale))
        x2 = int(round(box.x2 * scale))
        y2 = int(round(box.y2 * scale))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if draft is not None:
        x1 = int(round(draft.x1 * scale))
        y1 = int(round(draft.y1 * scale))
        x2 = int(round(draft.x2 * scale))
        y2 = int(round(draft.y2 * scale))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)


def _make_canvas(state: LabelState):
    canvas = cv2.cvtColor(state.display_image if state.scale < 1.0 else state.image, cv2.COLOR_GRAY2BGR)

    def _to_display(box: AnnotationBox) -> AnnotationBox:
        if state.scale >= 1.0:
            return box
        return AnnotationBox(
            x1=int(round(box.x1 * state.scale)),
            y1=int(round(box.y1 * state.scale)),
            x2=int(round(box.x2 * state.scale)),
            y2=int(round(box.y2 * state.scale)),
        )

    for box in state.boxes:
        disp = _to_display(box)
        cv2.rectangle(canvas, (disp.x1, disp.y1), (disp.x2, disp.y2), (0, 255, 0), 2)

    draft = None
    if state.draft_start and state.draft_end:
        draft = AnnotationBox(
            x1=min(state.draft_start[0], state.draft_end[0]),
            y1=min(state.draft_start[1], state.draft_end[1]),
            x2=max(state.draft_start[0], state.draft_end[0]),
            y2=max(state.draft_start[1], state.draft_end[1]),
        )
        draft = _to_display(draft)
        cv2.rectangle(canvas, (draft.x1, draft.y1), (draft.x2, draft.y2), (0, 0, 255), 2)

    _overlay_help(canvas)
    info = f"{state.image_path.parent.name} | boxes={len(state.boxes)} | {state.image_path.name}"
    cv2.putText(canvas, info, (16, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, info, (16, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def _clamp_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    return max(0, min(int(x), width - 1)), max(0, min(int(y), height - 1))


def _normalize_box(box: AnnotationBox, width: int, height: int) -> AnnotationBox | None:
    x1 = max(0, min(int(box.x1), width - 1))
    y1 = max(0, min(int(box.y1), height - 1))
    x2 = max(0, min(int(box.x2), width - 1))
    y2 = max(0, min(int(box.y2), height - 1))
    left, right = sorted([x1, x2])
    top, bottom = sorted([y1, y2])
    if (right - left) < MIN_BOX_SIZE or (bottom - top) < MIN_BOX_SIZE:
        return None
    return AnnotationBox(x1=left, y1=top, x2=right, y2=bottom)


def _existing_boxes(annotation_path: Path) -> list[AnnotationBox]:
    if not annotation_path.exists():
        return []
    try:
        payload = annotation_path.read_text(encoding="utf-8")
        import json

        data = json.loads(payload)
        boxes = []
        for item in data.get("boxes", []):
            boxes.append(
                AnnotationBox(
                    x1=int(item["x1"]),
                    y1=int(item["y1"]),
                    x2=int(item["x2"]),
                    y2=int(item["y2"]),
                )
            )
        return boxes
    except Exception:
        return []


def _annotate_run(run_dir: Path, image_name: str, annotation_root: Path) -> list[AnnotationBox] | None:
    image_path = run_dir / image_name
    image = _load_image(image_path)
    h, w = image.shape[:2]
    scale = _fit_scale(w, h)
    display_image = _resize_for_display(image, scale)
    annotation_path = annotation_root / run_dir.parent.name / f"{run_dir.name}.json"
    boxes = _existing_boxes(annotation_path)

    state = LabelState(
        image=image,
        image_path=image_path,
        annotation_path=annotation_path,
        boxes=boxes,
        display_image=display_image,
        scale=scale,
    )

    start_original: tuple[int, int] | None = None
    current_original: tuple[int, int] | None = None
    finished = False

    def on_mouse(event, x, y, _flags, _param):
        nonlocal start_original, current_original
        if scale < 1.0:
            ox = int(round(x / scale))
            oy = int(round(y / scale))
        else:
            ox, oy = x, y
        ox, oy = _clamp_point(ox, oy, w, h)
        if event == cv2.EVENT_LBUTTONDOWN:
            start_original = (ox, oy)
            current_original = (ox, oy)
        elif event == cv2.EVENT_MOUSEMOVE and start_original is not None:
            current_original = (ox, oy)
        elif event == cv2.EVENT_LBUTTONUP and start_original is not None:
            current_original = (ox, oy)
            candidate = AnnotationBox(
                x1=start_original[0],
                y1=start_original[1],
                x2=current_original[0],
                y2=current_original[1],
            )
            normalized = _normalize_box(candidate, w, h)
            if normalized is not None:
                state.boxes.append(normalized)
            start_original = None
            current_original = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        hwnd = ctypes.windll.user32.FindWindowW(None, WINDOW_NAME)
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 5)
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            ctypes.windll.user32.BringWindowToTop(hwnd)
    except Exception:
        pass

    while True:
        state.draft_start = start_original
        state.draft_end = current_original
        canvas = _make_canvas(state)
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):
            save_annotation(annotation_path, image_path, state.boxes, image_size=(w, h))
            finished = True
            break
        if key in (27, ord("q")):
            save_annotation(annotation_path, image_path, state.boxes, image_size=(w, h))
            cv2.destroyWindow(WINDOW_NAME)
            return None
        if key == ord("u") and state.boxes:
            state.boxes.pop()
        if key == ord("c"):
            state.boxes.clear()

    cv2.destroyWindow(WINDOW_NAME)
    if finished:
        return state.boxes
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Label debug pictures with drawn rectangles and export a YOLO dataset.")
    parser.add_argument("--debug-root", default=str(DEBUG_ROOT), help="Root debug directory.")
    parser.add_argument("--image-name", default=DEFAULT_IMAGE_NAME, help="Image file name inside each run directory.")
    parser.add_argument("--annotation-root", default=str(ANNOTATION_ROOT), help="Where per-image annotations are stored.")
    parser.add_argument("--output-root", default=str(YOLO_OUTPUT_ROOT), help="Where the exported YOLO dataset is written.")
    parser.add_argument("--dataset-name", default="detector_ai_yolo", help="Name of the exported YOLO dataset folder.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for dataset export.")
    args = parser.parse_args()

    debug_root = Path(args.debug_root)
    annotation_root = Path(args.annotation_root)
    output_root = Path(args.output_root)

    collections = _list_debug_collections(debug_root)
    if not collections:
        print(f"No debug collections found under: {debug_root.as_posix()}")
        return 1

    selected_collection = _pick_collection(collections)
    run_dirs = _list_run_dirs(selected_collection, args.image_name)
    if not run_dirs:
        print(f"No runs with {args.image_name} found in: {selected_collection.as_posix()}")
        return 1

    print(f"Loaded {len(run_dirs)} runs from: {selected_collection.as_posix()}")
    print("Press Enter after each picture to confirm and move on.")

    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[{index}/{len(run_dirs)}] {run_dir.name}")
        result = _annotate_run(run_dir, args.image_name, annotation_root)
        if result is None:
            print("Stopped by user.")
            break

    summary = export_yolo_dataset(
        annotation_root=annotation_root,
        output_root=output_root,
        dataset_name=args.dataset_name,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        class_names=["label"],
        image_copy_prefix=selected_collection.name,
    )
    print(f"YOLO dataset exported to: {summary['dataset_root']}")
    print(f"Images: {summary['total_images']} | Boxes: {summary['total_boxes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
