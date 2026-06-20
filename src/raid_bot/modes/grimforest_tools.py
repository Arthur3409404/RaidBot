# -*- coding: utf-8 -*-
"""Grim Forest RaidBot mode integration."""

from __future__ import annotations

import difflib
import json
import logging
import random
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import raid_bot.utils.auto_battle_tools as auto_battle_tools
import raid_bot.utils.image_tools as image_tools
import raid_bot.utils.map_tools as map_tools
import raid_bot.utils.window_tools as window_tools


MENU_TITLE = "Bosque Lugubre"
KEY_DENOMINATOR = 30
EXPECTED_STRUCTURE_COUNT = 5
MAX_RUN_DURATION_SECONDS = int(3.5 * 60 * 60)
MIN_ACCEPTED_DETECTOR_SCORE = 0.50
DEFAULT_TEMPLATE_MATCH_THRESHOLD = 0.38
DEFAULT_AVOID_MATCH_THRESHOLD = 0.72
DEFAULT_TOPK_PER_TEMPLATE_SCALE = 24
DEFAULT_PRE_SCORE_CANDIDATE_LIMIT = 320
AVOID_TEMPLATE_STEM = "avoid"
GROUP_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
DEFAULT_YOLO_MODEL_CANDIDATES = [
    Path("data") / "models" / "grimforest_detector" / "best.pt",
    Path("data") / "models" / "grimforest_detector" / "last.pt",
    Path("data") / "output" / "runs" / "detect" / "data" / "detector_ai" / "runs" / "grimforest_detector-3" / "weights" / "best.pt",
    Path("data") / "output" / "runs" / "detect" / "data" / "detector_ai" / "runs" / "grimforest_detector-3" / "weights" / "last.pt",
    Path("data") / "output" / "runs" / "detect" / "data" / "detector_ai" / "runs" / "grimforest_detector" / "weights" / "best.pt",
    Path("data") / "output" / "runs" / "detect" / "data" / "detector_ai" / "runs" / "grimforest_detector" / "weights" / "last.pt",
]
LATEST_YOLO_TRAINING_METRICS = {
    "precision": 0.96132,
    "recall": 0.90476,
    "mAP50": 0.92908,
    "mAP50_95": 0.44352,
    "run_name": "grimforest_detector-3",
}
DEFAULT_GROUP_THRESHOLDS = {
    "threshold_T1": 0.38292209881325756,
    "threshold_T2": 0.3845197692841958,
    "threshold_T3": 0.375449596703175,
    "threshold_T4": 0.38160037653257334,
    "threshold_T5": 0.38600473941339264,
    "threshold_T6": 0.38305471264914087,
    "threshold_avoid": 0.7144083658862909,
}


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    score: float


def _center_in_exclusion_zone(center_x_rel: float, center_y_rel: float) -> bool:
    zones = (
        (0.0, 0.95, 1.0, 1.0),
        (0.85, 0.5, 1.0, 1.0),
        (0.0, 0.0, 1.0, 0.05),
    )
    for x1, y1, x2, y2 in zones:
        if x1 <= center_x_rel <= x2 and y1 <= center_y_rel <= y2:
            return True
    return False


def _filter_boxes_by_center_exclusion(boxes: list[BoundingBox], image_shape: tuple[int, ...]) -> list[BoundingBox]:
    if not boxes:
        return []
    height = int(image_shape[0]) if len(image_shape) >= 1 else 0
    width = int(image_shape[1]) if len(image_shape) >= 2 else 0
    if width <= 0 or height <= 0:
        return list(boxes)

    kept: list[BoundingBox] = []
    for box in boxes:
        center_x = (float(box.x) + float(box.width) / 2.0) / float(width)
        center_y = (float(box.y) + float(box.height) / 2.0) / float(height)
        if _center_in_exclusion_zone(center_x, center_y):
            continue
        kept.append(box)
    return kept


def _normalize_grayscale_to_bgr(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return arr.astype(np.uint8)


def _load_yolo_detector(model_path: str | Path | None):
    try:
        from raid_bot.detector_ai.yolo_detector import YoloDetector
    except Exception:
        return None

    candidates: list[Path] = []
    if model_path:
        candidates.append(Path(model_path))
    candidates.extend(DEFAULT_YOLO_MODEL_CANDIDATES)

    for candidate in candidates:
        try:
            if candidate.exists():
                return YoloDetector(candidate)
        except Exception:
            continue
    return None


def _detect_with_template_fallback(
    binary: np.ndarray,
    reference_dir: Path,
    *,
    threshold_T1: float,
    threshold_T2: float,
    threshold_T3: float,
    threshold_T4: float,
    threshold_T5: float,
    threshold_T6: float,
    threshold_avoid: float,
    topk_per_template_scale: int,
) -> list[BoundingBox]:
    grouped_templates = _load_grouped_templates(Path(reference_dir))
    thresholds = {
        "T1": _clamp01(threshold_T1),
        "T2": _clamp01(threshold_T2),
        "T3": _clamp01(threshold_T3),
        "T4": _clamp01(threshold_T4),
        "T5": _clamp01(threshold_T5),
        "T6": _clamp01(threshold_T6),
        "avoid": _clamp01(threshold_avoid),
    }

    per_group_accepted: dict[str, list[dict]] = {}
    for group in GROUP_ORDER:
        per_group_accepted[group] = _template_match_group_candidates(
            binary,
            grouped_templates[group],
            threshold=float(thresholds[group]),
            topk_per_template_scale=int(topk_per_template_scale),
            group_name=group,
        )

    avoid_candidates = _template_match_group_candidates(
        binary,
        grouped_templates[AVOID_TEMPLATE_STEM],
        threshold=float(thresholds["avoid"]),
        topk_per_template_scale=int(topk_per_template_scale),
        group_name=AVOID_TEMPLATE_STEM,
    )

    selected = []
    for group in GROUP_ORDER:
        for candidate in per_group_accepted[group]:
            if any(_boxes_intersect(candidate, avoid) for avoid in avoid_candidates):
                continue
            selected.append(candidate)

    return [
        BoundingBox(
            x=int(candidate["x"]),
            y=int(candidate["y"]),
            width=int(candidate["w"]),
            height=int(candidate["h"]),
            score=round(float(candidate["score"]), 6),
        )
        for candidate in selected
    ]


def _largest_component_bbox(binary_img: np.ndarray) -> tuple[int, int, int, int] | None:
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    if count <= 1:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (
        int(stats[index, cv2.CC_STAT_LEFT]),
        int(stats[index, cv2.CC_STAT_TOP]),
        int(stats[index, cv2.CC_STAT_WIDTH]),
        int(stats[index, cv2.CC_STAT_HEIGHT]),
    )


def _crop_to_bbox(binary_img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = bbox
    return binary_img[y : y + height, x : x + width]


def _dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    size = int(np.count_nonzero(a)) + int(np.count_nonzero(b))
    if size <= 0:
        return 0.0
    return float((2.0 * np.count_nonzero(a & b)) / size)


def _load_reference_stats(reference_dir: Path, target_size: int = 64) -> dict | None:
    templates: list[np.ndarray] = []
    raw_templates_gray: list[np.ndarray] = []
    aspect_values: list[float] = []
    fill_values: list[float] = []
    paths = sorted(reference_dir.glob("*.png"))
    avoid_path = next((path for path in paths if path.stem.strip().lower() == AVOID_TEMPLATE_STEM), None)

    for path in paths:
        if path == avoid_path:
            continue
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        bbox = _largest_component_bbox(binary)
        if bbox is None:
            continue
        binary_roi = _crop_to_bbox(binary, bbox)
        gray_roi = _crop_to_bbox(gray, bbox)
        height, width = binary_roi.shape
        if width <= 0 or height <= 0:
            continue
        templates.append(cv2.resize(binary_roi, (target_size, target_size), interpolation=cv2.INTER_NEAREST))
        raw_templates_gray.append(gray_roi)
        aspect_values.append(float(width) / float(height))
        fill_values.append(float(np.count_nonzero(binary_roi)) / float(max(1, width * height)))

    if not templates:
        return None

    avoid_template = None
    if avoid_path is not None:
        avoid_gray = cv2.imread(str(avoid_path), cv2.IMREAD_GRAYSCALE)
        if avoid_gray is not None:
            _, avoid_binary = cv2.threshold(avoid_gray, 127, 255, cv2.THRESH_BINARY)
            avoid_bbox = _largest_component_bbox(avoid_binary)
            if avoid_bbox is not None:
                avoid_template = cv2.resize(
                    _crop_to_bbox(avoid_binary, avoid_bbox),
                    (target_size, target_size),
                    interpolation=cv2.INTER_NEAREST,
                )

    return {
        "templates": templates,
        "raw_templates_gray": raw_templates_gray,
        "aspect_mean": float(np.mean(aspect_values)),
        "aspect_std": float(np.std(aspect_values) + 1e-6),
        "fill_mean": float(np.mean(fill_values)),
        "fill_std": float(np.std(fill_values) + 1e-6),
        "avoid_template": avoid_template,
        "target_size": int(target_size),
    }


def _classify_template_group(path: Path) -> str | None:
    stem = path.stem.strip().lower()
    if stem == AVOID_TEMPLATE_STEM or stem.startswith(f"{AVOID_TEMPLATE_STEM}_"):
        return AVOID_TEMPLATE_STEM
    for group in GROUP_ORDER:
        lower = group.lower()
        if stem == lower or stem.startswith(f"{lower}_"):
            return group
    return None


def _load_grouped_templates(reference_dir: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {group: [] for group in GROUP_ORDER}
    grouped[AVOID_TEMPLATE_STEM] = []
    for path in sorted(reference_dir.glob("*.png")):
        group = _classify_template_group(path)
        if group is None:
            continue
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        grouped[group].append({"name": path.name, "image": binary})

    missing = [group for group in GROUP_ORDER if len(grouped[group]) == 0]
    if missing:
        raise RuntimeError(f"Missing grouped templates in {reference_dir.as_posix()}: {missing}")
    if len(grouped[AVOID_TEMPLATE_STEM]) == 0:
        raise RuntimeError(
            f"Missing avoid template group in {reference_dir.as_posix()} "
            f"(expected {AVOID_TEMPLATE_STEM}.png or {AVOID_TEMPLATE_STEM}_*.png)."
        )
    return grouped


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


def _select_peak_points(result: np.ndarray, min_score: float, topk: int) -> tuple[np.ndarray, np.ndarray]:
    mask = result >= float(min_score)
    if not bool(np.any(mask)):
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    local_max = cv2.dilate(result, np.ones((3, 3), dtype=np.float32))
    ys, xs = np.where(mask & (result >= local_max))
    if len(ys) == 0:
        ys, xs = np.where(mask)
    if int(topk) > 0 and len(ys) > int(topk):
        indices = np.argpartition(result[ys, xs], -int(topk))[-int(topk) :]
        ys = ys[indices]
        xs = xs[indices]
    return ys.astype(np.int32), xs.astype(np.int32)


def _template_match_candidates(
    binary_img: np.ndarray,
    ref_stats: dict,
    *,
    match_threshold: float,
    topk_per_template_scale: int,
) -> list[dict]:
    candidates = []
    binary_float = binary_img.astype(np.float32)
    for template_id, reference in enumerate(ref_stats.get("raw_templates_gray", [])):
        for scale in np.linspace(1.0, 3.2, 12):
            width = max(8, int(reference.shape[1] * float(scale)))
            height = max(8, int(reference.shape[0] * float(scale)))
            if width >= binary_img.shape[1] or height >= binary_img.shape[0]:
                continue
            template = cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
            if float(np.std(template)) < 1e-6:
                continue
            result = cv2.matchTemplate(binary_float, template, cv2.TM_CCOEFF_NORMED)
            ys, xs = _select_peak_points(result, min_score=match_threshold, topk=topk_per_template_scale)
            for y, x in zip(ys, xs):
                roi = binary_img[int(y) : int(y) + height, int(x) : int(x) + width]
                if roi.shape[:2] != (height, width):
                    continue
                foreground = float(np.count_nonzero(roi))
                candidates.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "w": int(width),
                        "h": int(height),
                        "area": foreground,
                        "fill": foreground / float(max(1, width * height)),
                        "aspect": float(width) / float(height),
                        "roi": roi,
                        "template_score_raw": float(result[int(y), int(x)]),
                        "template_id": int(template_id),
                    }
                )
    return candidates


def _template_match_group_candidates(
    binary_img: np.ndarray,
    templates: list[dict],
    *,
    threshold: float,
    topk_per_template_scale: int,
    group_name: str,
) -> list[dict]:
    candidates = []
    binary_float = binary_img.astype(np.float32)
    for template in templates:
        template_img = np.asarray(template["image"], dtype=np.uint8)
        height, width = template_img.shape[:2]
        if width >= binary_img.shape[1] or height >= binary_img.shape[0]:
            continue
        template_float = template_img.astype(np.float32)
        if float(np.std(template_float)) < 1e-6:
            continue
        result = cv2.matchTemplate(binary_float, template_float, cv2.TM_CCOEFF_NORMED)
        ys, xs = _select_peak_points(result, min_score=float(threshold), topk=int(topk_per_template_scale))
        for y, x in zip(ys, xs):
            candidates.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(width),
                    "h": int(height),
                    "score": float(result[int(y), int(x)]),
                    "template_score_raw": float(result[int(y), int(x)]),
                    "group": str(group_name),
                    "template_name": str(template["name"]),
                }
            )
    candidates.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            int(item.get("y", 0)),
            int(item.get("x", 0)),
            str(item.get("template_name", "")),
        )
    )
    return _nms(candidates, iou_threshold=0.25)


def _score_candidate(candidate: dict, ref_stats: dict, target_size: int = 64) -> float:
    resized = cv2.resize(candidate["roi"], (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    template_score = max(_dice_score(resized, template) for template in ref_stats["templates"])
    aspect_z = abs(candidate["aspect"] - ref_stats["aspect_mean"]) / max(ref_stats["aspect_std"], 0.02)
    fill_z = abs(candidate["fill"] - ref_stats["fill_mean"]) / max(ref_stats["fill_std"], 0.03)
    aspect_score = float(np.exp(-0.5 * aspect_z))
    fill_score = float(np.exp(-0.5 * fill_z))
    rectangularity_score = float(np.clip(candidate["fill"] / max(ref_stats["fill_mean"], 1e-6), 0.0, 1.0))
    return float(
        (0.62 * template_score)
        + (0.18 * aspect_score)
        + (0.14 * fill_score)
        + (0.06 * rectangularity_score)
    )


def _boxes_intersect(a: dict, b: dict) -> bool:
    x1 = max(int(a["x"]), int(b["x"]))
    y1 = max(int(a["y"]), int(b["y"]))
    x2 = min(int(a["x"] + a["w"]), int(b["x"] + b["w"]))
    y2 = min(int(a["y"] + a["h"]), int(b["y"] + b["h"]))
    return x2 > x1 and y2 > y1


def _bbox_iou(a: dict, b: dict) -> float:
    x1 = max(int(a["x"]), int(b["x"]))
    y1 = max(int(a["y"]), int(b["y"]))
    x2 = min(int(a["x"] + a["w"]), int(b["x"] + b["w"]))
    y2 = min(int(a["y"] + a["h"]), int(b["y"] + b["h"]))
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0.0
    union = (a["w"] * a["h"]) + (b["w"] * b["h"]) - intersection
    return float(intersection / union) if union > 0 else 0.0


def _nms(candidates: list[dict], iou_threshold: float = 0.25) -> list[dict]:
    kept = []
    for candidate in candidates:
        if all(_bbox_iou(candidate, previous) <= iou_threshold for previous in kept):
            kept.append(candidate)
    return kept


def _avoid_match_score(binary_img: np.ndarray, candidate: dict, ref_stats: dict) -> float:
    avoid_template = ref_stats.get("avoid_template")
    if avoid_template is None:
        return 0.0
    roi = binary_img[
        int(candidate["y"]) : int(candidate["y"] + candidate["h"]),
        int(candidate["x"]) : int(candidate["x"] + candidate["w"]),
    ]
    if roi.size == 0:
        return 0.0
    resized = cv2.resize(
        roi,
        (int(ref_stats["target_size"]), int(ref_stats["target_size"])),
        interpolation=cv2.INTER_NEAREST,
    )
    return _dice_score(resized, avoid_template)


def detect_grimforest_like_structures(
    binary_img: np.ndarray,
    reference_dir: Path,
    *,
    detector_method: str = "yolo",
    detector_model_path: str | Path | None = None,
    detector_confidence: float = 0.25,
    detector_imgsz: int = 640,
    expected_count: int = EXPECTED_STRUCTURE_COUNT,
    max_objects: int = 0,
    threshold_T1: float = DEFAULT_GROUP_THRESHOLDS["threshold_T1"],
    threshold_T2: float = DEFAULT_GROUP_THRESHOLDS["threshold_T2"],
    threshold_T3: float = DEFAULT_GROUP_THRESHOLDS["threshold_T3"],
    threshold_T4: float = DEFAULT_GROUP_THRESHOLDS["threshold_T4"],
    threshold_T5: float = DEFAULT_GROUP_THRESHOLDS["threshold_T5"],
    threshold_T6: float = DEFAULT_GROUP_THRESHOLDS["threshold_T6"],
    threshold_avoid: float = DEFAULT_GROUP_THRESHOLDS["threshold_avoid"],
    topk_per_template_scale: int = DEFAULT_TOPK_PER_TEMPLATE_SCALE,
) -> list[BoundingBox]:
    """Detect selectable Grim Forest structures.

    YOLO is the default path. Template matching is only used when explicitly requested.
    """
    if binary_img is None or np.asarray(binary_img).size == 0:
        return []
    binary = np.asarray(binary_img, dtype=np.uint8)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    yolo_detector = None
    if str(detector_method or "").strip().lower() == "yolo":
        yolo_detector = _load_yolo_detector(detector_model_path)

    boxes: list[BoundingBox] = []
    if yolo_detector is not None:
        try:
            detections = yolo_detector.predict(
                _normalize_grayscale_to_bgr(binary),
                conf=float(detector_confidence),
                imgsz=int(detector_imgsz),
            )
            for det in detections:
                x1 = int(round(min(det.x1, det.x2)))
                y1 = int(round(min(det.y1, det.y2)))
                x2 = int(round(max(det.x1, det.x2)))
                y2 = int(round(max(det.y1, det.y2)))
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                if width <= 0 or height <= 0:
                    continue
                score = float(det.confidence)
                if score < float(MIN_ACCEPTED_DETECTOR_SCORE):
                    continue
                boxes.append(BoundingBox(x=x1, y=y1, width=width, height=height, score=round(score, 6)))
        except Exception:
            boxes = []

    if not boxes:
        if str(detector_method or "").strip().lower() != "yolo":
            boxes = _detect_with_template_fallback(
                binary,
                reference_dir,
                threshold_T1=threshold_T1,
                threshold_T2=threshold_T2,
                threshold_T3=threshold_T3,
                threshold_T4=threshold_T4,
                threshold_T5=threshold_T5,
                threshold_T6=threshold_T6,
                threshold_avoid=threshold_avoid,
                topk_per_template_scale=topk_per_template_scale,
            )
        else:
            # An empty YOLO result is a valid outcome when no selectable structures
            # are visible in the current view. The caller already handles this by
            # moving the camera and trying again.
            boxes = []

    boxes = _filter_boxes_by_center_exclusion(boxes, binary.shape)
    if int(max_objects) > 0 and len(boxes) > int(max_objects):
        boxes = boxes[: int(max_objects)]
    return boxes


def _start_run_deadline(bot, max_run_duration_seconds=None):
    limit = bot.max_run_duration_seconds if max_run_duration_seconds is None else float(max_run_duration_seconds)
    bot._run_deadline = time.time() + limit


def _ensure_within_run_deadline(bot, context: str):
    if bot._run_deadline and time.time() > bot._run_deadline:
        hours = bot.max_run_duration_seconds / 3600.0
        raise TimeoutError(f"{bot.__class__.__name__} exceeded max runtime of {hours:.1f}h while {context}.")


def _spiral_direction_for_step(step_index: int, start_direction_index: int = 0) -> str:
    directions = ("right", "down", "left", "up")
    remaining = max(0, int(step_index))
    segment_length = 1
    direction_index = int(start_direction_index) % len(directions)
    while remaining >= segment_length:
        remaining -= segment_length
        direction_index = (direction_index + 1) % len(directions)
        if direction_index % 2 == 0:
            segment_length += 1
    return directions[direction_index]


class RSL_Bot_GrimForest:
    def __init__(self, title_substring="Raid: Shadow Legends", reader=None, window=None, verbose=True, setup=None):
        self.title_substring = title_substring
        self.reader = reader
        self.window = window
        self.verbose = verbose
        self.log = logging.getLogger(self.__class__.__name__)
        self.search_areas = {
            "menu_name": [0.0, 0.02, 0.36, 0.07],
            "mode_keys_row": [0.18, 0.02, 0.80, 0.07],
            "mode_difficulty_current": [0.03, 0.917, 0.079, 0.043],
            "mode_difficulty_switch_normal": [0.092, 0.798, 0.08, 0.036],
            "mode_difficulty_switch_hard": [0.096, 0.865, 0.066, 0.038],
            "pov": [0.0, 0.0, 1.0, 1.0],
            "go_to_higher_menu": [0.928, 0.031, 0.046, 0.039],
            "stage_lower_half_text_scan": [0.0, 0.5, 1.0, 0.5],
            "stage_confirm_button_champion_selection": [0.762, 0.876, 0.213, 0.104],
            "stage_robar_box_1": [0.3948, 0.8299, 0.2143, 0.0775],
            "stage_robar_box_2": [0.51, 0.8781, 0.2089, 0.0813],
            "stage_auto_battle_button": [0.026, 0.899, 0.058, 0.07],
            "stage_battle_result": [0.389, 0.148, 0.204, 0.071],
            "stage_battle_result_2": [0.38, 0.085, 0.224, 0.059],
            "post_battle_level_prompt": [0.3955, 0.6881, 0.1912, 0.0643],
            "post_battle_stat_options": [0.1782, 0.5794, 0.6306, 0.0851],
            "post_battle_stat_confirm": [0.4386, 0.7750, 0.1098, 0.0435],
        }
        self.setup = {
            "difficulty": "hard",
            "alternate_difficulty": True,
            "difficulty_switch_retries": 3,
            "difficulty_dropdown_open_delay_seconds": 0.8,
            "difficulty_switch_confirm_delay_seconds": 2.5,
            "post_entry_wait_seconds": 5.0,
            "startup_check_timeout_seconds": 45.0,
            "startup_check_poll_interval_seconds": 1.0,
            "initial_candidate_zoom_out_steps": 3,
            "initial_candidate_zoom_out_amount_per_step": -600,
            "initial_candidate_zoom_out_delay_seconds": 0.75,
            "candidate_detection_retries_per_view": 1,
            "max_random_repositions_when_no_candidates": 20,
            "max_spiral_repositions_when_no_candidates": 20,
            "target_hex": "CEC329",
            "dark_tolerance": 40,
            "reference_dir": str(Path("data") / "assets" / "images" / "grimforest"),
            "expected_structure_count": EXPECTED_STRUCTURE_COUNT,
            "detector_max_objects": EXPECTED_STRUCTURE_COUNT,
            "detector_threshold_T1": DEFAULT_GROUP_THRESHOLDS["threshold_T1"],
            "detector_threshold_T2": DEFAULT_GROUP_THRESHOLDS["threshold_T2"],
            "detector_threshold_T3": DEFAULT_GROUP_THRESHOLDS["threshold_T3"],
            "detector_threshold_T4": DEFAULT_GROUP_THRESHOLDS["threshold_T4"],
            "detector_threshold_T5": DEFAULT_GROUP_THRESHOLDS["threshold_T5"],
            "detector_threshold_T6": DEFAULT_GROUP_THRESHOLDS["threshold_T6"],
            "detector_threshold_avoid": DEFAULT_GROUP_THRESHOLDS["threshold_avoid"],
            "detector_topk_per_template_scale": DEFAULT_TOPK_PER_TEMPLATE_SCALE,
            "detector_method": "yolo",
            "detector_model_path": str(
                Path("data")
                / "output"
                / "runs"
                / "detect"
                / "data"
                / "detector_ai"
                / "runs"
                / "grimforest_detector-3"
                / "weights"
                / "best.pt"
            ),
            "detector_confidence": 0.25,
            "detector_imgsz": 640,
            "detector_training_metrics": dict(LATEST_YOLO_TRAINING_METRICS),
            "stage_select_delay_seconds": 3.0,
            "stage_start_retries": 3,
            "stage_battle_timeout_seconds": 420.0,
            "stage_battle_poll_interval_seconds": 2.0,
            "stage_battle_outcome_confirm_delay_seconds": 10.0,
            "pan_strength": 1.0,
            "run_state_path": str(Path("data") / "tmp" / "grim_forest_run_state.json"),
            "last_defeat_path": str(Path("data") / "tmp" / "grim_forest_last_defeat.json"),
        }
        provided_setup = dict(setup or {})
        if setup:
            self.setup.update(provided_setup)
        if (
            "max_random_repositions_when_no_candidates" in provided_setup
            and "max_spiral_repositions_when_no_candidates" not in provided_setup
        ):
            self.setup["max_spiral_repositions_when_no_candidates"] = self.setup[
                "max_random_repositions_when_no_candidates"
            ]

        self.reference_dir = Path(str(self.setup["reference_dir"]))
        self.target_bgr_as_rgb = self._hex_to_bgr_as_rgb(str(self.setup["target_hex"]))
        self.run_state_path = Path(str(self.setup["run_state_path"]))
        self.last_defeat_path = Path(str(self.setup["last_defeat_path"]))
        self.max_run_duration_seconds = MAX_RUN_DURATION_SECONDS
        self._run_deadline = None
        self.reset_run_state()

    @staticmethod
    def _hex_to_bgr_as_rgb(hex_color: str) -> np.ndarray:
        value = str(hex_color or "").lstrip("#")
        if len(value) != 6:
            value = "CEC329"
        red = int(value[0:2], 16)
        green = int(value[2:4], 16)
        blue = int(value[4:6], 16)
        return np.array([blue, green, red], dtype=np.uint8)

    @staticmethod
    def resembles(text, target, threshold=0.8):
        return difflib.SequenceMatcher(None, (text or "").lower(), (target or "").lower()).ratio() >= threshold

    def reset_run_state(self):
        self.running = True
        self.main_loop_running = True
        self.current_difficulty = None
        self.current_run_difficulty = None
        self.available_keys = 0
        self.key_counter = None
        self.mode_transitioned_out = False
        self.detected_candidates = []
        self.selected_candidate = None
        self.selection_succeeded = False
        self.stage_start_status = None
        self.battle_outcome = None
        self.post_battle_menu_status = None
        self.post_battle_stat_choice = None
        self.initial_candidate_zoom_out_done = False
        self.random_reposition_count = 0
        self.spiral_reposition_index = 0
        self.spiral_start_direction_index = random.randrange(4)
        self.exit_reason = None
        self.completed_battles = 0
        self.last_defeat_by_difficulty = self._load_last_defeat_state()
        self.no_candidate_failures_by_difficulty = getattr(
            self,
            "no_candidate_failures_by_difficulty",
            {"hard": 0, "normal": 0},
        )
        self.last_no_candidate_start_direction_by_difficulty = getattr(
            self,
            "last_no_candidate_start_direction_by_difficulty",
            {"hard": None, "normal": None},
        )

    @staticmethod
    def _read_json_file(path: Path, fallback: dict) -> dict:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else dict(fallback)
        except (OSError, ValueError):
            return dict(fallback)

    @staticmethod
    def _write_json_file(path: Path, payload: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        temp_path.replace(path)

    def _load_last_defeat_state(self) -> dict:
        stored = self._read_json_file(self.last_defeat_path, {"hard": None, "normal": None})
        return {
            "hard": stored.get("hard") if isinstance(stored.get("hard"), dict) else None,
            "normal": stored.get("normal") if isinstance(stored.get("normal"), dict) else None,
        }

    def _record_last_defeat_candidate(self, difficulty: str, candidate: dict | None):
        key = str(difficulty or "").strip().lower()
        if key not in {"hard", "normal"} or not isinstance(candidate, dict):
            return
        record = dict(candidate)
        record["difficulty"] = key
        record["outcome"] = "Derrota"
        record["recorded_at"] = datetime.now().isoformat(timespec="seconds")
        self.last_defeat_by_difficulty[key] = record
        self._write_json_file(self.last_defeat_path, self.last_defeat_by_difficulty)
        self.log.info("[Grim Forest] Stored last defeat location for '%s'.", key)

    def _plan_and_commit_run_difficulty(self) -> str:
        configured = self._normalize_difficulty_value(self.setup.get("difficulty", "hard")) or "hard"
        state = self._read_json_file(self.run_state_path, {"run_counter": 0})
        run_counter = max(0, int(state.get("run_counter", 0) or 0))
        planned = configured
        if bool(self.setup.get("alternate_difficulty", False)):
            planned = "hard" if run_counter % 2 == 0 else "normal"
        self._write_json_file(
            self.run_state_path,
            {
                "run_counter": run_counter + 1,
                "last_used_difficulty": planned,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        return planned

    def _read_menu_name(self) -> str | None:
        try:
            objects = image_tools.get_text_in_relative_area(
                self.reader, self.window, self.search_areas["menu_name"], power_detection=False
            )
            return objects[0].text.strip() if objects else None
        except Exception:
            self.log.debug("[Grim Forest] Menu OCR failed.", exc_info=True)
            return None

    def _read_text_objects(self, area_key: str, power_detection=False):
        try:
            return image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                search_area=self.search_areas[area_key],
                power_detection=power_detection,
            )
        except Exception:
            self.log.debug("[Grim Forest] OCR failed for '%s'.", area_key, exc_info=True)
            return []

    @staticmethod
    def _normalize_difficulty_value(value) -> str | None:
        normalized = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
        compact = " ".join(normalized.strip().lower().split())
        if compact in {"normal", "modo normal"}:
            return "normal"
        if compact in {"hard", "dificil", "difficult", "hard mode", "modo dificil"}:
            return "hard"
        return None

    def detect_current_difficulty(self):
        objects = self._read_text_objects("mode_difficulty_current")
        values = [getattr(obj, "text", "") for obj in objects if getattr(obj, "text", "")]
        for value in values + [" ".join(values), "".join(values)]:
            normalized = self._normalize_difficulty_value(value)
            if normalized:
                self.current_difficulty = normalized
                return normalized
        return self.current_difficulty

    def set_difficulty(self, set_level=None):
        requested = self._normalize_difficulty_value(set_level)
        if requested not in {"normal", "hard"}:
            self.log.warning("[Grim Forest] Ignoring invalid difficulty request: %s", set_level)
            return self.detect_current_difficulty()
        current = self.detect_current_difficulty()
        if current == requested:
            return current

        retries = max(1, int(self.setup.get("difficulty_switch_retries", 3)))
        for attempt in range(1, retries + 1):
            self.log.info("[Grim Forest] Switching difficulty to '%s' (%s/%s).", requested, attempt, retries)
            window_tools.click_center(
                self.window,
                self.search_areas["mode_difficulty_current"],
                delay=float(self.setup.get("difficulty_dropdown_open_delay_seconds", 0.8)),
            )
            window_tools.click_center(
                self.window,
                self.search_areas[f"mode_difficulty_switch_{requested}"],
                delay=float(self.setup.get("difficulty_switch_confirm_delay_seconds", 2.5)),
            )
            current = self.detect_current_difficulty()
            if current == requested:
                return current
        self.log.warning(
            "[Grim Forest] Could not confirm difficulty switch to '%s'; continuing with detected='%s'.",
            requested,
            current,
        )
        return current

    def _is_in_game_modes_menu(self, menu_text: str | None) -> bool:
        return bool(
            menu_text
            and (
                self.resembles(menu_text, "Modos de juego", threshold=0.55)
                or self.resembles(menu_text, "Modo de juego", threshold=0.55)
            )
        )

    def is_in_grim_forest_mode(self, menu_text: str | None = None) -> bool:
        if menu_text is None:
            menu_text = self._read_menu_name()
        return bool(menu_text and self.resembles(menu_text, MENU_TITLE, threshold=0.35))

    def _normalize_menu_fragment(self, value: str | None) -> str:
        normalized = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
        return "".join(ch for ch in normalized.lower() if ch.isalnum())

    def _is_forbidden_hard_encounter_name(self, menu_text: str | None) -> bool:
        if self._normalize_difficulty_value(self.current_run_difficulty) != "hard":
            return False
        return "mimeto" in self._normalize_menu_fragment(menu_text)

    def _perform_startup_check(self) -> bool:
        wait_seconds = float(self.setup.get("post_entry_wait_seconds", 5.0) or 0.0)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

        timeout_seconds = max(0.0, float(self.setup.get("startup_check_timeout_seconds", 45.0) or 0.0))
        poll_interval = max(0.0, float(self.setup.get("startup_check_poll_interval_seconds", 1.0) or 0.0))
        deadline = time.time() + timeout_seconds
        last_menu_text = None

        while self.main_loop_running:
            last_menu_text = self._read_menu_name()
            if last_menu_text and self.resembles(last_menu_text, MENU_TITLE, threshold=0.55):
                return True

            selected_text = self.select_post_battle_stat_reward()
            if selected_text is not None:
                self.log.info("[Grim Forest] Cleared startup level prompt: %s.", selected_text)

            if time.time() >= deadline:
                break
            time.sleep(poll_interval)

        self.log.warning("[Grim Forest] Startup check failed. Expected '%s', got '%s'.", MENU_TITLE, last_menu_text)
        return False

    def update_available_keys(self) -> int:
        counter = image_tools.read_fraction_counter_in_relative_area(
            self.reader,
            self.window,
            search_area=self.search_areas["mode_keys_row"],
            expected_denominator=KEY_DENOMINATOR,
        )
        self.key_counter = counter
        self.available_keys = int(counter.get("current") or 0)
        return self.available_keys

    def has_grim_forest_keys_remaining(self, retries: int = 3) -> bool:
        for attempt in range(1, max(1, int(retries)) + 1):
            try:
                keys = self.update_available_keys()
            except Exception:
                keys = 0
                self.log.debug("[Grim Forest] Key OCR attempt failed.", exc_info=True)
            self.log.info("[Grim Forest] Key check (%s/%s): %s/%s", attempt, retries, keys, KEY_DENOMINATOR)
            if keys > 0:
                return True
            time.sleep(0.6)
        return False

    def _capture_grim_forest_mask(self):
        _, image_np, region = map_tools.capture_relative_area(self.window, self.search_areas["pov"])
        base = self.target_bgr_as_rgb.astype(np.int16)
        difference = image_np.astype(np.int16) - base
        tolerance = int(self.setup.get("dark_tolerance", 40))
        mask = (
            (difference[:, :, 0] >= -tolerance)
            & (difference[:, :, 1] >= -tolerance)
            & (difference[:, :, 2] >= -tolerance)
        )
        return np.where(mask, 255, 0).astype(np.uint8), region

    @staticmethod
    def _detected_boxes_to_candidates(boxes: list[BoundingBox], region) -> list[dict]:
        width = max(1.0, float(region[2] or 1.0))
        height = max(1.0, float(region[3] or 1.0))
        candidates = []
        for index, box in enumerate(boxes, start=1):
            center_x = int(box.x + (box.width / 2.0))
            center_y = int(box.y + (box.height / 2.0))
            candidates.append(
                {
                    "index": index,
                    "score": float(box.score or 0.0),
                    "center_rel_x": float(center_x / width),
                    "center_rel_y": float(center_y / height),
                    "bbox_rel": {
                        "x": float(max(0, box.x) / width),
                        "y": float(max(0, box.y) / height),
                        "width": float(max(0, box.width) / width),
                        "height": float(max(0, box.height) / height),
                    },
                    "center_abs_x": int(region[0] + center_x),
                    "center_abs_y": int(region[1] + center_y),
                }
            )
        return sorted(candidates, key=lambda item: item["score"], reverse=True)

    def detect_grim_forest_candidates(self, retries: int | None = None) -> list[dict]:
        retries = max(1, int(retries or self.setup.get("candidate_detection_retries_per_view", 1)))
        for attempt in range(1, retries + 1):
            mask, region = self._capture_grim_forest_mask()
            boxes = detect_grimforest_like_structures(
                mask,
                self.reference_dir,
                detector_method=self.setup.get("detector_method", "yolo"),
                detector_model_path=self.setup.get("detector_model_path"),
                detector_confidence=float(self.setup.get("detector_confidence", 0.25)),
                detector_imgsz=int(self.setup.get("detector_imgsz", 640)),
                expected_count=int(self.setup.get("expected_structure_count", EXPECTED_STRUCTURE_COUNT)),
                max_objects=int(self.setup.get("detector_max_objects", 0)),
                threshold_T1=float(self.setup.get("detector_threshold_T1", DEFAULT_GROUP_THRESHOLDS["threshold_T1"])),
                threshold_T2=float(self.setup.get("detector_threshold_T2", DEFAULT_GROUP_THRESHOLDS["threshold_T2"])),
                threshold_T3=float(self.setup.get("detector_threshold_T3", DEFAULT_GROUP_THRESHOLDS["threshold_T3"])),
                threshold_T4=float(self.setup.get("detector_threshold_T4", DEFAULT_GROUP_THRESHOLDS["threshold_T4"])),
                threshold_T5=float(self.setup.get("detector_threshold_T5", DEFAULT_GROUP_THRESHOLDS["threshold_T5"])),
                threshold_T6=float(self.setup.get("detector_threshold_T6", DEFAULT_GROUP_THRESHOLDS["threshold_T6"])),
                threshold_avoid=float(
                    self.setup.get("detector_threshold_avoid", DEFAULT_GROUP_THRESHOLDS["threshold_avoid"])
                ),
                topk_per_template_scale=int(
                    self.setup.get("detector_topk_per_template_scale", DEFAULT_TOPK_PER_TEMPLATE_SCALE)
                ),
            )
            candidates = self._detected_boxes_to_candidates(boxes, region)
            self.detected_candidates = candidates
            self.log.info("[Grim Forest] Candidate detection (%s/%s): %s candidates.", attempt, retries, len(candidates))
            if candidates:
                return candidates
            time.sleep(0.8)
        return []

    @staticmethod
    def _bbox_overlap_ratio(a: dict, b: dict) -> float:
        ax1, ay1 = float(a.get("x", 0.0)), float(a.get("y", 0.0))
        ax2, ay2 = ax1 + float(a.get("width", 0.0)), ay1 + float(a.get("height", 0.0))
        bx1, by1 = float(b.get("x", 0.0)), float(b.get("y", 0.0))
        bx2, by2 = bx1 + float(b.get("width", 0.0)), by1 + float(b.get("height", 0.0))
        intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
        denominator = min(max(0.0, (ax2 - ax1) * (ay2 - ay1)), max(0.0, (bx2 - bx1) * (by2 - by1)))
        return float(intersection / denominator) if denominator > 0 else 0.0

    def _filter_candidates_against_last_defeat(self, candidates: list[dict], difficulty: str):
        previous = self.last_defeat_by_difficulty.get(str(difficulty or "").lower())
        previous_bbox = previous.get("bbox_rel") if isinstance(previous, dict) else None
        if not isinstance(previous_bbox, dict):
            return list(candidates)
        filtered = [
            candidate
            for candidate in candidates
            if not isinstance(candidate.get("bbox_rel"), dict)
            or self._bbox_overlap_ratio(previous_bbox, candidate["bbox_rel"]) < 0.50
        ]
        if len(filtered) != len(candidates):
            self.log.info("[Grim Forest] Skipped %s candidate(s) matching last defeat location.", len(candidates) - len(filtered))
        return filtered

    def _max_spiral_repositions_when_no_candidates(self) -> int:
        return max(0, int(self.setup.get("max_spiral_repositions_when_no_candidates", 20)))

    def _difficulty_state_key(self, difficulty: str | None) -> str | None:
        key = self._normalize_difficulty_value(difficulty)
        return key if key in {"hard", "normal"} else None

    def _spiral_stride_for_difficulty(self, difficulty: str | None) -> int:
        key = self._difficulty_state_key(difficulty)
        failures = int(self.no_candidate_failures_by_difficulty.get(key, 0) or 0) if key else 0
        if failures >= 3:
            return 3
        if failures >= 1:
            return 2
        return 1

    def _record_candidate_scan_result(self, difficulty: str | None, found: bool):
        key = self._difficulty_state_key(difficulty)
        if not key:
            return
        if found:
            self.no_candidate_failures_by_difficulty[key] = 0
            self.last_no_candidate_start_direction_by_difficulty[key] = None
            return
        self.no_candidate_failures_by_difficulty[key] = int(
            self.no_candidate_failures_by_difficulty.get(key, 0) or 0
        ) + 1
        self.last_no_candidate_start_direction_by_difficulty[key] = self.spiral_start_direction_index

    def _prepare_spiral_start_for_difficulty(self, difficulty: str | None, stride: int):
        self.spiral_reposition_index = 0
        self.spiral_start_direction_index = random.randrange(4)

        key = self._difficulty_state_key(difficulty)
        if stride < 3 or not key:
            return

        previous_start = self.last_no_candidate_start_direction_by_difficulty.get(key)
        if previous_start is None:
            return
        self.spiral_start_direction_index = (int(previous_start) + 2) % 4

    def _move_spiral_direction_once(self):
        direction = _spiral_direction_for_step(
            self.spiral_reposition_index,
            self.spiral_start_direction_index,
        )
        self.spiral_reposition_index += 1
        move = {
            "up": window_tools.move_up,
            "down": window_tools.move_down,
            "left": window_tools.move_left,
            "right": window_tools.move_right,
        }[direction]
        self.random_reposition_count += 1
        self.log.info("[Grim Forest] No candidates. Moving in spiral: %s.", direction)
        move(self.window, strength=float(self.setup.get("pan_strength", 1.0)))

    def _move_random_direction_once(self):
        self._move_spiral_direction_once()

    def _zoom_out_before_initial_candidate_detection(self):
        if self.initial_candidate_zoom_out_done:
            return
        self.initial_candidate_zoom_out_done = True

        steps = max(0, int(self.setup.get("initial_candidate_zoom_out_steps", 3)))
        if steps <= 0:
            return

        self.log.info("[Grim Forest] Zooming out before initial candidate detection (%s steps).", steps)
        try:
            window_tools.zoom_out(
                self.window,
                steps=steps,
                amount_per_step=int(self.setup.get("initial_candidate_zoom_out_amount_per_step", -600)),
                delay=float(self.setup.get("initial_candidate_zoom_out_delay_seconds", 0.75)),
            )
        except Exception:
            self.log.debug("[Grim Forest] Initial zoom-out failed; continuing with candidate detection.", exc_info=True)

    def detect_candidates_with_random_reposition(self, difficulty: str | None = None) -> list[dict]:
        max_moves = self._max_spiral_repositions_when_no_candidates()
        stride = self._spiral_stride_for_difficulty(difficulty)
        self._prepare_spiral_start_for_difficulty(difficulty, stride)
        self._zoom_out_before_initial_candidate_detection()
        moves_done = 0
        while True:
            _ensure_within_run_deadline(self, "detecting Grim Forest candidates")
            candidates = self._filter_candidates_against_last_defeat(
                self.detect_grim_forest_candidates(), difficulty or ""
            )
            if candidates:
                self._record_candidate_scan_result(difficulty, found=True)
                return candidates
            if moves_done >= max_moves or not self.main_loop_running:
                self._record_candidate_scan_result(difficulty, found=False)
                break
            moves_to_make = min(stride, max_moves - moves_done)
            for _ in range(moves_to_make):
                self._move_random_direction_once()
                moves_done += 1
        return []

    def select_grim_forest_candidate(self, candidate: dict) -> bool:
        time.sleep(float(self.setup.get("stage_select_delay_seconds", 3.0)))
        window_tools.click_at(
            int(candidate["center_abs_x"]),
            int(candidate["center_abs_y"]),
            delay=2.5,
            window=self.window,
        )
        menu_text = self._read_menu_name()
        if self.is_in_grim_forest_mode(menu_text):
            return False
        if self._is_forbidden_hard_encounter_name(menu_text):
            self.log.info("[Grim Forest] Skipping forbidden hard encounter: %s.", menu_text)
            window_tools.sendkey("esc", delay=1.0, window=self.window)
            return False
        return True

    def _find_start_button_in_lower_half(self):
        for obj in self._read_text_objects("stage_lower_half_text_scan"):
            text = (getattr(obj, "text", "") or "").strip()
            if self.resembles(text, "Empezar", threshold=0.6) or self.resembles(text, "Iniciar", threshold=0.6):
                return obj
        return None

    def _press_pre_start_click_sequence(self):
        for relative_square in (
            [0.1045, 0.2809, 0.0760, 0.1235],
            [0.1982, 0.2846, 0.0707, 0.1169],
            [0.1091, 0.4260, 0.0707, 0.1169],
            [0.1982, 0.4251, 0.0737, 0.1206],
            [0.2880, 0.3563, 0.0760, 0.1178],
            [0.0238, 0.7578, 0.0445, 0.0735],
            [0.0238, 0.8360, 0.0469, 0.0735],
            [0.0238, 0.9161, 0.0445, 0.0735],
            [0.0776, 0.7568, 0.0453, 0.0716],
            [0.0783, 0.8351, 0.0422, 0.0697],
        ):
            window_tools.click_center(self.window, relative_square, delay=0.30)

    def click_grim_forest_start_button(self, retries: int | None = None) -> str:
        retries = max(1, int(retries or self.setup.get("stage_start_retries", 3)))
        for attempt in range(1, retries + 1):
            _ensure_within_run_deadline(self, "starting Grim Forest battle")
            self.log.info("[Grim Forest] Start button attempt (%s/%s).", attempt, retries)
            if any(
                self.resembles((getattr(obj, "text", "") or "").strip(), "Robar Campeones", threshold=0.6)
                for obj in self._read_text_objects("stage_robar_box_1")
            ):
                window_tools.click_center(self.window, self.search_areas["stage_robar_box_1"], delay=2.0)
                window_tools.click_center(self.window, self.search_areas["stage_robar_box_2"], delay=2.0)
            self._press_pre_start_click_sequence()
            menu_before = (self._read_menu_name() or "").strip()
            start_object = self._find_start_button_in_lower_half()
            if start_object is not None:
                adjusted_y = int(start_object.mean_pos_y) + int(0.10 * self.window.height)
                adjusted_y = max(int(self.window.top), min(adjusted_y, int(self.window.top + self.window.height - 1)))
                window_tools.click_at(int(start_object.mean_pos_x), adjusted_y, delay=1.0, window=self.window)
            else:
                window_tools.click_center(self.window, self.search_areas["stage_confirm_button_champion_selection"], delay=1.0)
            time.sleep(5.0)
            if menu_before and menu_before == (self._read_menu_name() or "").strip():
                window_tools.sendkey("esc", delay=1.0, window=self.window)
                return "battle_not_started_same_menu"
            try:
                if image_tools.check_startup(self):
                    return "battle_started"
            except Exception:
                self.log.debug("[Grim Forest] Startup validation failed; treating battle as started.", exc_info=True)
                return "battle_started"
            time.sleep(0.8)
        return "start_button_not_found_or_not_started"

    def _battle_result_text(self):
        for area_key in ("stage_battle_result", "stage_battle_result_2"):
            for obj in self._read_text_objects(area_key):
                text = (getattr(obj, "text", "") or "").strip()
                if self.resembles(text, "VICTORIA", threshold=0.68):
                    return "Victoria"
                if self.resembles(text, "DERROTA", threshold=0.68):
                    return "Derrota"
        return None

    def _is_auto_battle_visible(self) -> bool:
        objects = self._read_text_objects("stage_auto_battle_button")
        return bool(objects and self.resembles((getattr(objects[0], "text", "") or "").strip(), "Auto", threshold=0.7))

    def get_battle_outcome(self, timeout_seconds: float | None = None, poll_interval_seconds: float | None = None):
        timeout = float(timeout_seconds or self.setup.get("stage_battle_timeout_seconds", 420.0))
        interval = float(poll_interval_seconds or self.setup.get("stage_battle_poll_interval_seconds", 2.0))
        confirmation_delay = float(self.setup.get("stage_battle_outcome_confirm_delay_seconds", 10.0))
        started_at = time.time()
        auto_seen = False
        auto_battle_tools.reset_auto_battle_watchdog(self)
        while self.main_loop_running and (time.time() - started_at) < timeout:
            _ensure_within_run_deadline(self, "waiting for Grim Forest battle result")
            result = self._battle_result_text()
            if result:
                time.sleep(confirmation_delay)
                if self._battle_result_text() == result:
                    return result
            auto_seen = auto_seen or self._is_auto_battle_visible()
            menu_text = self._read_menu_name()
            if auto_seen and menu_text and self.resembles(menu_text, MENU_TITLE, threshold=0.55):
                return None
            if self._is_in_game_modes_menu(menu_text):
                return None
            auto_battle_tools.ensure_auto_battle_running(
                self,
                auto_button_area=self.search_areas["stage_auto_battle_button"],
            )
            time.sleep(max(0.6, interval))
        return None

    def return_to_mode_root_after_battle(self, max_attempts: int = 4) -> str:
        for _ in range(max(1, int(max_attempts))):
            menu_text = self._read_menu_name()
            if menu_text and self.resembles(menu_text, MENU_TITLE, threshold=0.55):
                return "mode"
            if self._is_in_game_modes_menu(menu_text):
                self.mode_transitioned_out = True
                return "game_modes"
            window_tools.sendkey("esc", delay=4.0, window=self.window)
        menu_text = self._read_menu_name()
        if menu_text and self.resembles(menu_text, MENU_TITLE, threshold=0.55):
            return "mode"
        if self._is_in_game_modes_menu(menu_text):
            self.mode_transitioned_out = True
            return "game_modes"
        return "unknown"

    def select_post_battle_stat_reward(self) -> str | None:
        prompts = self._read_text_objects("post_battle_level_prompt")
        prompt_text = " ".join((getattr(obj, "text", "") or "").lower() for obj in prompts)
        has_level_prompt = "nivel" in prompt_text
        choices = self._read_text_objects("post_battle_stat_options")
        choice_text = " ".join((getattr(obj, "text", "") or "").upper() for obj in choices)
        confirm_texts = self._read_text_objects("post_battle_stat_confirm")
        confirm_text = " ".join((getattr(obj, "text", "") or "").upper() for obj in confirm_texts)
        has_trait_card_prompt = "ELEGIR" in confirm_text and any(
            preferred_text in choice_text for preferred_text in ("RES", "VEL", "DEF", "HP", "ATK", "PUNT")
        )
        if not has_level_prompt and not has_trait_card_prompt:
            return None
        if has_level_prompt:
            window_tools.click_center(self.window, self.search_areas["post_battle_level_prompt"], delay=2.0)
            choices = self._read_text_objects("post_battle_stat_options")
        selected_text = None
        for preferred_text in ("RES", "VEL", "DEF", "HP"):
            match = next(
                (obj for obj in choices if preferred_text in (getattr(obj, "text", "") or "").upper()),
                None,
            )
            if match is not None:
                selected_text = (getattr(match, "text", "") or "").strip()
                window_tools.click_at(int(match.mean_pos_x), int(match.mean_pos_y), delay=2.0, window=self.window)
                break
        if selected_text is None:
            window_tools.click_center(self.window, self.search_areas["pov"], delay=2.0)
            selected_text = "pov_fallback"
        window_tools.click_center(self.window, self.search_areas["post_battle_stat_confirm"], delay=2.0)
        self.log.info("[Grim Forest] Post-battle stat selection: %s.", selected_text)
        return selected_text

    def exit_grim_forest_to_main_menu(self, reason: str) -> bool:
        self.exit_reason = reason
        self.log.info("[Grim Forest] Exiting to game modes. Reason: %s.", reason)
        for _ in range(3):
            if self._is_in_game_modes_menu(self._read_menu_name()):
                self.mode_transitioned_out = True
                return True
            window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"], delay=1.8)
        self.mode_transitioned_out = self._is_in_game_modes_menu(self._read_menu_name())
        return self.mode_transitioned_out

    def run_grimforest(self, main_loop_running=True, max_run_duration_seconds=MAX_RUN_DURATION_SECONDS):
        _start_run_deadline(self, max_run_duration_seconds)
        self.reset_run_state()
        self.main_loop_running = main_loop_running
        if not self.reader or not self.window:
            self.log.warning("[Grim Forest] Reader/window unavailable; skipping mode.")
            return False
        if not self._perform_startup_check():
            return False

        self.post_battle_stat_choice = self.select_post_battle_stat_reward()

        planned_difficulty = self._plan_and_commit_run_difficulty()
        confirmed_difficulty = self.set_difficulty(planned_difficulty)
        self.current_run_difficulty = confirmed_difficulty if confirmed_difficulty in {"normal", "hard"} else planned_difficulty
        self.log.info("[Grim Forest] Running with difficulty '%s' (requested '%s').", self.current_run_difficulty, planned_difficulty)

        while self.main_loop_running and self.running:
            _ensure_within_run_deadline(self, "running Grim Forest loop")
            if not self.has_grim_forest_keys_remaining(retries=3):
                self.exit_grim_forest_to_main_menu(reason="no_keys_remaining")
                break

            candidates = self.detect_candidates_with_random_reposition(difficulty=self.current_run_difficulty)
            if not candidates:
                self.exit_grim_forest_to_main_menu(reason="no_candidates_detected_after_random_repositions")
                break

            selected = next(
                (candidate for candidate in candidates if self.main_loop_running and self.select_grim_forest_candidate(candidate)),
                None,
            )
            self.selected_candidate = selected
            self.selection_succeeded = selected is not None
            if selected is None:
                self.exit_grim_forest_to_main_menu(reason="no_candidate_selected")
                break

            self.stage_start_status = self.click_grim_forest_start_button()
            if self.stage_start_status != "battle_started":
                self.exit_grim_forest_to_main_menu(reason=self.stage_start_status or "stage_start_failed")
                break

            self.battle_outcome = self.get_battle_outcome()
            self.post_battle_menu_status = self.return_to_mode_root_after_battle()
            if self.post_battle_menu_status == "mode":
                time.sleep(5.0)
                self.post_battle_stat_choice = self.select_post_battle_stat_reward()
            self.completed_battles += 1

            if self.battle_outcome == "Derrota":
                self._record_last_defeat_candidate(self.current_run_difficulty, selected)
                self.exit_grim_forest_to_main_menu(reason="battle_lost")
                break
            if self.post_battle_menu_status == "game_modes":
                break
            if self.post_battle_menu_status == "unknown":
                self.exit_grim_forest_to_main_menu(reason="unknown_menu_after_battle")
                break
            if self.battle_outcome != "Victoria":
                self.exit_grim_forest_to_main_menu(reason="battle_outcome_unknown_or_timeout")
                break

        return True

    def test(self):
        return self.run_grimforest()
