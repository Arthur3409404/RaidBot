from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2

import _bootstrap  # noqa: F401


WINDOW_NAME = "tagteam_portrait_annotator"
DISPLAY_MAX_WIDTH = 1600
DISPLAY_MAX_HEIGHT = 900
MIN_BOX_SIZE = 4
DEFAULT_IMAGE = Path("temp") / "tagteam_cropped_samples" / "sample_00_combined.png"
DEFAULT_OUTPUT = Path("temp") / "tagteam_cropped_samples" / "sample_00_combined_portraits.json"


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def normalized(self, width: int, height: int) -> dict[str, float]:
        left = max(0, min(int(self.x1), int(self.x2)))
        top = max(0, min(int(self.y1), int(self.y2)))
        right = min(max(int(self.x1), int(self.x2)), int(width))
        bottom = min(max(int(self.y1), int(self.y2)), int(height))
        return {
            "x1": left,
            "y1": top,
            "x2": right,
            "y2": bottom,
            "x_center": ((left + right) / 2.0) / float(width),
            "y_center": ((top + bottom) / 2.0) / float(height),
            "width": (right - left) / float(width),
            "height": (bottom - top) / float(height),
        }


@dataclass
class AnnotatorState:
    image: object
    display_image: object
    scale: float
    image_path: Path
    boxes: list[Box]
    draft_start: tuple[int, int] | None = None
    draft_end: tuple[int, int] | None = None


def _load_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path.as_posix())
    return image


def _fit_scale(width: int, height: int) -> float:
    scale_w = DISPLAY_MAX_WIDTH / float(width) if width > DISPLAY_MAX_WIDTH else 1.0
    scale_h = DISPLAY_MAX_HEIGHT / float(height) if height > DISPLAY_MAX_HEIGHT else 1.0
    return min(1.0, scale_w, scale_h)


def _resize_for_display(image, scale: float):
    if scale >= 1.0:
        return image.copy()
    height, width = image.shape[:2]
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _clamp_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    return max(0, min(int(x), width - 1)), max(0, min(int(y), height - 1))


def _normalize_box(box: Box, width: int, height: int) -> Box | None:
    x1 = max(0, min(int(box.x1), width - 1))
    y1 = max(0, min(int(box.y1), height - 1))
    x2 = max(0, min(int(box.x2), width - 1))
    y2 = max(0, min(int(box.y2), height - 1))
    left, right = sorted([x1, x2])
    top, bottom = sorted([y1, y2])
    if (right - left) < MIN_BOX_SIZE or (bottom - top) < MIN_BOX_SIZE:
        return None
    return Box(x1=left, y1=top, x2=right, y2=bottom)


def _overlay_help(canvas, count: int, total: int, boxes_per_slot: int, slots: int):
    if count >= total:
        current = "All boxes drawn. Press Enter to save."
    else:
        slot_index = min(slots - 1, count // boxes_per_slot)
        slot_box = count % boxes_per_slot + 1
        current = f"Draw {total} portrait boxes in order. Current target: slot {slot_index + 1}, portrait {slot_box}/{boxes_per_slot}."
    lines = [
        f"Left-drag: draw box   Enter: save and exit   u: undo   c: clear   q/esc: quit",
        current,
        f"Draw only portrait rectangles. Suggested order: slots 1 through {slots} in order.",
    ]
    y = 24
    for line in lines:
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
        y += 22


def _draw_boxes(canvas, state: AnnotatorState, boxes_per_slot: int):
    def to_display(box: Box) -> Box:
        if state.scale >= 1.0:
            return box
        return Box(
            x1=int(round(box.x1 * state.scale)),
            y1=int(round(box.y1 * state.scale)),
            x2=int(round(box.x2 * state.scale)),
            y2=int(round(box.y2 * state.scale)),
        )

    for index, box in enumerate(state.boxes, start=1):
        disp = to_display(box)
        color = (0, 255, 0)
        cv2.rectangle(canvas, (disp.x1, disp.y1), (disp.x2, disp.y2), color, 2)
        slot = ((index - 1) // boxes_per_slot) + 1
        portrait = ((index - 1) % boxes_per_slot) + 1
        label = f"{slot}:{portrait}"
        cv2.putText(canvas, label, (disp.x1 + 3, max(16, disp.y1 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (disp.x1 + 3, max(16, disp.y1 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if state.draft_start and state.draft_end:
        draft = Box(
            x1=min(state.draft_start[0], state.draft_end[0]),
            y1=min(state.draft_start[1], state.draft_end[1]),
            x2=max(state.draft_start[0], state.draft_end[0]),
            y2=max(state.draft_start[1], state.draft_end[1]),
        )
        if state.scale < 1.0:
            draft = Box(
                x1=int(round(draft.x1 * state.scale)),
                y1=int(round(draft.y1 * state.scale)),
                x2=int(round(draft.x2 * state.scale)),
                y2=int(round(draft.y2 * state.scale)),
            )
        cv2.rectangle(canvas, (draft.x1, draft.y1), (draft.x2, draft.y2), (0, 0, 255), 2)


def _make_canvas(state: AnnotatorState, boxes_per_slot: int, total_boxes: int, slots: int):
    canvas = state.display_image.copy()
    _draw_boxes(canvas, state, boxes_per_slot)
    _overlay_help(canvas, len(state.boxes), total_boxes, boxes_per_slot, slots)
    info = f"{state.image_path.name} | boxes={len(state.boxes)}/{total_boxes}"
    cv2.putText(canvas, info, (16, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, info, (16, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def _save_payload(
    output_path: Path,
    image_path: Path,
    boxes: list[Box],
    width: int,
    height: int,
    boxes_per_slot: int,
    slots: int,
):
    grouped: list[dict[str, object]] = []
    for slot_index in range((len(boxes) + boxes_per_slot - 1) // boxes_per_slot):
        start = slot_index * boxes_per_slot
        end = start + boxes_per_slot
        slot_boxes = boxes[start:end]
        grouped.append(
            {
                "slot_index": slot_index + 1,
                "boxes": [asdict(box) for box in slot_boxes],
                "boxes_normalized": [box.normalized(width, height) for box in slot_boxes],
            }
        )

    payload = {
        "image_path": image_path.resolve().as_posix(),
        "image_name": image_path.name,
        "image_size": {"width": int(width), "height": int(height)},
        "boxes_per_slot": int(boxes_per_slot),
        "slots": int(slots),
        "expected_boxes": int(boxes_per_slot * slots),
        "boxes": [asdict(box) for box in boxes],
        "boxes_normalized": [box.normalized(width, height) for box in boxes],
        "slots": grouped,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactively annotate the 12 champion portrait rectangles in a Tag Team sample.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Combined Tag Team sample image to annotate.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to write the annotation JSON.")
    parser.add_argument("--boxes-per-slot", type=int, default=4, help="How many portrait rectangles belong to each slot.")
    parser.add_argument("--slots", type=int, default=3, help="How many slots the image contains.")
    args = parser.parse_args()

    image_path = Path(args.image)
    output_path = Path(args.output)
    boxes_per_slot = int(args.boxes_per_slot)
    total_boxes = int(args.slots * args.boxes_per_slot)

    image = _load_image(image_path)
    height, width = image.shape[:2]
    scale = _fit_scale(width, height)
    display_image = _resize_for_display(image, scale)

    state = AnnotatorState(
        image=image,
        display_image=display_image,
        scale=scale,
        image_path=image_path,
        boxes=[],
    )

    start_original: tuple[int, int] | None = None
    current_original: tuple[int, int] | None = None

    def on_mouse(event, x, y, _flags, _param):
        nonlocal start_original, current_original
        if state.scale < 1.0:
            ox = int(round(x / state.scale))
            oy = int(round(y / state.scale))
        else:
            ox, oy = x, y
        ox, oy = _clamp_point(ox, oy, width, height)

        if event == cv2.EVENT_LBUTTONDOWN:
            start_original = (ox, oy)
            current_original = (ox, oy)
        elif event == cv2.EVENT_MOUSEMOVE and start_original is not None:
            current_original = (ox, oy)
        elif event == cv2.EVENT_LBUTTONUP and start_original is not None:
            current_original = (ox, oy)
            candidate = Box(
                x1=start_original[0],
                y1=start_original[1],
                x2=current_original[0],
                y2=current_original[1],
            )
            normalized = _normalize_box(candidate, width, height)
            if normalized is not None:
                state.boxes.append(normalized)
            start_original = None
            current_original = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            state.draft_start = start_original
            state.draft_end = current_original
            canvas = _make_canvas(state, boxes_per_slot, total_boxes, int(args.slots))
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):
                payload = _save_payload(output_path, image_path, state.boxes, width, height, boxes_per_slot, int(args.slots))
                print(json.dumps(payload, indent=2))
                break
            if key in (27, ord("q")):
                print("Annotation cancelled. Nothing was saved.")
                return 1
            if key == ord("u") and state.boxes:
                state.boxes.pop()
            if key == ord("c"):
                state.boxes.clear()
    finally:
        cv2.destroyAllWindows()

    print(f"Saved annotation to: {output_path.resolve().as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
