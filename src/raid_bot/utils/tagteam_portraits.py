from __future__ import annotations

from typing import Iterable

import numpy as np


TAGTEAM_PORTRAIT_TEMPLATE_SIZE = (454, 152)

# Ordered exactly as annotated from sample_00_combined.png:
# slot 1 portraits, then slot 2, then slot 3.
TAGTEAM_PORTRAIT_BOXES_454X152: tuple[tuple[int, int, int, int], ...] = (
    (38, 35, 92, 87),
    (37, 86, 92, 135),
    (91, 34, 144, 86),
    (90, 86, 145, 134),
    (191, 34, 243, 86),
    (188, 87, 243, 134),
    (243, 35, 294, 87),
    (245, 84, 297, 138),
    (340, 35, 394, 87),
    (342, 86, 395, 138),
    (393, 88, 446, 139),
    (396, 31, 450, 86),
)


def _normalize_image_np(image_np: np.ndarray) -> np.ndarray:
    image = np.asarray(image_np)
    if image.ndim != 3:
        raise ValueError(f"Expected a portrait source image [H, W, C], got {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3:
        raise ValueError(f"Could not interpret portrait source image shape {image.shape}")
    return image


def scale_tagteam_portrait_boxes(
    image_shape: tuple[int, int, int] | tuple[int, int],
    *,
    template_size: tuple[int, int] = TAGTEAM_PORTRAIT_TEMPLATE_SIZE,
) -> list[tuple[int, int, int, int]]:
    height, width = int(image_shape[0]), int(image_shape[1])
    template_width, template_height = template_size
    scale_x = width / float(template_width)
    scale_y = height / float(template_height)

    boxes: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in TAGTEAM_PORTRAIT_BOXES_454X152:
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


def crop_tagteam_portraits(image_np: np.ndarray) -> list[np.ndarray]:
    image = _normalize_image_np(image_np)
    boxes = scale_tagteam_portrait_boxes(image.shape)
    crops: list[np.ndarray] = []
    for left, top, right, bottom in boxes:
        crops.append(image[top:bottom, left:right].copy())
    return crops

