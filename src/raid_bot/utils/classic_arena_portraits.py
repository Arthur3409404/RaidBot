from __future__ import annotations

import numpy as np


CLASSIC_ARENA_PORTRAIT_TEMPLATE_SIZE = (440, 130)

# Ordered left-to-right from the captured classic arena sample.
CLASSIC_ARENA_PORTRAIT_BOXES_440X130: tuple[tuple[int, int, int, int], ...] = (
    (124, 12, 192, 101),
    (192, 12, 262, 103),
    (263, 14, 332, 104),
    (333, 11, 402, 103),
)


def _normalize_image_np(image_np: np.ndarray) -> np.ndarray:
    image = np.asarray(image_np)
    if image.ndim != 3:
        raise ValueError(f"Expected a classic arena card image [H, W, C], got {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3:
        raise ValueError(f"Could not interpret classic arena card shape {image.shape}")
    return image


def scale_classic_arena_portrait_boxes(
    image_shape: tuple[int, int, int] | tuple[int, int],
    *,
    template_size: tuple[int, int] = CLASSIC_ARENA_PORTRAIT_TEMPLATE_SIZE,
) -> list[tuple[int, int, int, int]]:
    height, width = int(image_shape[0]), int(image_shape[1])
    template_width, template_height = template_size
    scale_x = width / float(template_width)
    scale_y = height / float(template_height)

    boxes: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in CLASSIC_ARENA_PORTRAIT_BOXES_440X130:
        left = max(0, min(width, int(round(x1 * scale_x))))
        top = max(0, min(height, int(round(y1 * scale_y))))
        right = max(0, min(width, int(round(x2 * scale_x))))
        bottom = max(0, min(height, int(round(y2 * scale_y))))
        if right <= left:
            right = min(width, left + 1)
        if bottom <= top:
            bottom = min(height, top + 1)
        boxes.append((left, top, right, bottom))
    return boxes


def crop_classic_arena_portraits(image_np: np.ndarray) -> list[np.ndarray]:
    image = _normalize_image_np(image_np)
    boxes = scale_classic_arena_portrait_boxes(image.shape)
    crops: list[np.ndarray] = []
    for left, top, right, bottom in boxes:
        crops.append(image[top:bottom, left:right].copy())
    return crops
