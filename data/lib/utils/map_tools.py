# -*- coding: utf-8 -*-
"""Map capture, stage detection, and persistence helpers for map modes."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyautogui


def _json_default(value):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def sanitize_name(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unknown"


def normalize_grid_position(grid_position: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(grid_position, dict):
        return {"x": 0, "y": 0}
    return {
        "x": int(grid_position.get("x", 0) or 0),
        "y": int(grid_position.get("y", 0) or 0),
    }


def build_grid_navigation_plan(
    current_grid: dict[str, Any] | None,
    target_grid: dict[str, Any] | None,
    *,
    inward_action: str,
    outward_action: str,
    horizontal_first: bool = True,
) -> dict[str, Any]:
    current = normalize_grid_position(current_grid)
    target = normalize_grid_position(target_grid)
    working = dict(current)
    steps: list[dict[str, Any]] = []

    def append_horizontal_moves():
        delta_x = int(target["x"] - working["x"])
        if delta_x == 0:
            return

        action = inward_action if delta_x > 0 else outward_action
        step_delta = 1 if delta_x > 0 else -1
        for _ in range(abs(delta_x)):
            previous = dict(working)
            working["x"] += step_delta
            steps.append(
                {
                    "action": action,
                    "from_grid": previous,
                    "to_grid": dict(working),
                }
            )

    def append_vertical_moves():
        delta_y = int(target["y"] - working["y"])
        if delta_y == 0:
            return

        action = "down" if delta_y > 0 else "up"
        step_delta = 1 if delta_y > 0 else -1
        for _ in range(abs(delta_y)):
            previous = dict(working)
            working["y"] += step_delta
            steps.append(
                {
                    "action": action,
                    "from_grid": previous,
                    "to_grid": dict(working),
                }
            )

    if horizontal_first:
        append_horizontal_moves()
        append_vertical_moves()
    else:
        append_vertical_moves()
        append_horizontal_moves()

    return {
        "start_grid": current,
        "target_grid": target,
        "horizontal_first": bool(horizontal_first),
        "step_count": len(steps),
        "steps": steps,
    }


def relative_area_to_absolute(window, search_area) -> tuple[int, int, int, int]:
    rel_left, rel_top, rel_width, rel_height = search_area
    abs_left = int(window.left + rel_left * window.width)
    abs_top = int(window.top + rel_top * window.height)
    abs_width = int(rel_width * window.width)
    abs_height = int(rel_height * window.height)
    return abs_left, abs_top, abs_width, abs_height


def capture_relative_area(window, search_area):
    region = relative_area_to_absolute(window, search_area)
    screenshot = pyautogui.screenshot(region=region)
    image_np = np.array(screenshot)
    return screenshot, image_np, region


def save_image(path: str | Path, image_np: np.ndarray) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_array = np.asarray(image_np)
    if image_array.ndim == 2:
        cv2.imwrite(str(output_path), image_array)
    else:
        cv2.imwrite(str(output_path), cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    return output_path


def compute_view_signature(image_np: np.ndarray) -> dict[str, Any]:
    if image_np.size == 0:
        return {
            "avg_hash": "",
            "mean_rgb": [0.0, 0.0, 0.0],
            "std_rgb": [0.0, 0.0, 0.0],
            "brightness": 0.0,
            "blue_ratio": 0.0,
        }

    rgb = image_np if image_np.ndim == 3 else cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    downscaled = cv2.resize(rgb, (16, 16), interpolation=cv2.INTER_AREA)
    grayscale = cv2.cvtColor(downscaled, cv2.COLOR_RGB2GRAY)
    grayscale_mean = float(grayscale.mean())
    avg_hash = "".join("1" if value >= grayscale_mean else "0" for value in grayscale.flatten())

    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    blue_mask = (blue > 100) & (blue > green + 10) & (blue > red + 20)

    return {
        "avg_hash": avg_hash,
        "mean_rgb": [round(float(channel), 3) for channel in rgb.mean(axis=(0, 1))],
        "std_rgb": [round(float(channel), 3) for channel in rgb.std(axis=(0, 1))],
        "brightness": round(float(grayscale_mean), 3),
        "blue_ratio": round(float(np.mean(blue_mask)), 6),
    }


def compute_fragment_signatures(
    image_np: np.ndarray,
    rows: int = 4,
    cols: int = 4,
    fragment_width_ratio: float = 0.45,
    fragment_height_ratio: float = 0.45,
) -> list[dict[str, Any]]:
    if image_np.size == 0:
        return []

    rgb = image_np if image_np.ndim == 3 else cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    height, width = rgb.shape[:2]
    fragment_width = max(24, min(width, int(round(width * float(fragment_width_ratio)))))
    fragment_height = max(24, min(height, int(round(height * float(fragment_height_ratio)))))
    max_x = max(0, width - fragment_width)
    max_y = max(0, height - fragment_height)
    x_positions = np.linspace(0, max_x, max(1, int(cols))).round().astype(int)
    y_positions = np.linspace(0, max_y, max(1, int(rows))).round().astype(int)

    fragments: list[dict[str, Any]] = []
    for row_index, y in enumerate(y_positions):
        for col_index, x in enumerate(x_positions):
            fragment = rgb[y : y + fragment_height, x : x + fragment_width]
            fragments.append(
                {
                    "fragment_id": f"r{row_index}_c{col_index}",
                    "origin_rel": {
                        "x": round(float(x) / max(1.0, float(width)), 6),
                        "y": round(float(y) / max(1.0, float(height)), 6),
                    },
                    "size_rel": {
                        "width": round(float(fragment_width) / max(1.0, float(width)), 6),
                        "height": round(float(fragment_height) / max(1.0, float(height)), 6),
                    },
                    "signature": compute_view_signature(fragment),
                }
            )
    return fragments


def compare_fragment_signature_sets(
    query_fragments: list[dict[str, Any]] | None,
    candidate_fragments: list[dict[str, Any]] | None,
    top_k: int = 3,
) -> dict[str, Any]:
    query_fragments = list(query_fragments or [])
    candidate_fragments = list(candidate_fragments or [])
    if not query_fragments or not candidate_fragments:
        return {
            "score": 0.0,
            "selected_scores": [],
            "strong_match_count": 0,
            "top_matches": [],
        }

    best_matches: list[dict[str, Any]] = []
    for query_fragment in query_fragments:
        best_match = None
        for candidate_fragment in candidate_fragments:
            score = compare_view_signatures(
                query_fragment.get("signature", {}),
                candidate_fragment.get("signature", {}),
            )
            if best_match is None or score > best_match["score"]:
                best_match = {
                    "query_fragment_id": query_fragment.get("fragment_id"),
                    "candidate_fragment_id": candidate_fragment.get("fragment_id"),
                    "score": score,
                }
        if best_match is not None:
            best_matches.append(best_match)

    best_matches.sort(key=lambda item: item["score"], reverse=True)
    selected = best_matches[: max(1, int(top_k))]
    selected_scores = [float(item["score"]) for item in selected]
    strong_match_count = sum(1 for score in selected_scores if score >= 0.84)
    selected_average = sum(selected_scores) / len(selected_scores)
    selected_max = max(selected_scores) if selected_scores else 0.0
    score = round(selected_average * 0.75 + selected_max * 0.25, 6)
    return {
        "score": score,
        "selected_scores": [round(score, 6) for score in selected_scores],
        "strong_match_count": int(strong_match_count),
        "top_matches": selected,
    }


def compute_stage_layout_signature(candidates: list[dict[str, Any]] | None, bins: int = 4) -> dict[str, Any]:
    normalized_candidates = list(candidates or [])
    if not normalized_candidates:
        return {
            "count": 0,
            "centers": [],
            "grid_histogram": [0 for _ in range(max(1, int(bins)) ** 2)],
            "mean_blue_fill": 0.0,
            "mean_mask_area": 0.0,
        }

    histogram = [0 for _ in range(max(1, int(bins)) ** 2)]
    centers = []
    blue_fill_values = []
    mask_area_values = []
    for candidate in normalized_candidates:
        center = candidate.get("center_rel") or {}
        center_x = min(1.0, max(0.0, float(center.get("x", 0.0) or 0.0)))
        center_y = min(1.0, max(0.0, float(center.get("y", 0.0) or 0.0)))
        bin_x = min(max(0, int(center_x * bins)), bins - 1)
        bin_y = min(max(0, int(center_y * bins)), bins - 1)
        histogram[bin_y * bins + bin_x] += 1
        centers.append({"x": round(center_x, 6), "y": round(center_y, 6)})
        blue_fill_values.append(float(candidate.get("blue_fill", 0.0) or 0.0))
        mask_area_values.append(float(candidate.get("mask_area", 0.0) or 0.0))

    centers.sort(key=lambda item: (item["y"], item["x"]))
    return {
        "count": len(centers),
        "centers": centers,
        "grid_histogram": histogram,
        "mean_blue_fill": round(sum(blue_fill_values) / max(1, len(blue_fill_values)), 6),
        "mean_mask_area": round(sum(mask_area_values) / max(1, len(mask_area_values)), 3),
    }


def compare_stage_layout_signatures(
    query_layout: dict[str, Any] | None,
    candidate_layout: dict[str, Any] | None,
    center_tolerance: float = 0.11,
) -> dict[str, Any]:
    query_layout = query_layout or {}
    candidate_layout = candidate_layout or {}
    query_centers = list(query_layout.get("centers", []))
    candidate_centers = list(candidate_layout.get("centers", []))
    if not query_centers or not candidate_centers:
        return {
            "score": 0.0,
            "matched_center_count": 0,
            "histogram_similarity": 0.0,
        }

    used_candidate_indexes: set[int] = set()
    matched_distances: list[float] = []
    matched_centers = 0

    for query_center in query_centers:
        best_index = None
        best_distance = None
        for index, candidate_center in enumerate(candidate_centers):
            if index in used_candidate_indexes:
                continue
            distance = float(
                np.hypot(
                    float(query_center.get("x", 0.0)) - float(candidate_center.get("x", 0.0)),
                    float(query_center.get("y", 0.0)) - float(candidate_center.get("y", 0.0)),
                )
            )
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is not None and best_distance is not None and best_distance <= float(center_tolerance):
            used_candidate_indexes.add(best_index)
            matched_distances.append(best_distance)
            matched_centers += 1

    histogram_similarity = 0.0
    query_histogram = list(query_layout.get("grid_histogram", []))
    candidate_histogram = list(candidate_layout.get("grid_histogram", []))
    if query_histogram and candidate_histogram and len(query_histogram) == len(candidate_histogram):
        histogram_similarity = float(
            sum(min(int(query_value), int(candidate_value)) for query_value, candidate_value in zip(query_histogram, candidate_histogram))
        ) / max(1.0, float(min(sum(query_histogram), sum(candidate_histogram))))

    overlap_ratio = float(matched_centers) / max(1.0, float(min(len(query_centers), len(candidate_centers))))
    distance_score = 0.0
    if matched_distances:
        distance_score = max(
            0.0,
            1.0 - (sum(matched_distances) / len(matched_distances)) / max(float(center_tolerance), 1e-6),
        )

    score = round(overlap_ratio * 0.7 + histogram_similarity * 0.2 + distance_score * 0.1, 6)
    return {
        "score": score,
        "matched_center_count": int(matched_centers),
        "histogram_similarity": round(histogram_similarity, 6),
    }


def compare_view_signatures(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_hash = first.get("avg_hash", "")
    second_hash = second.get("avg_hash", "")
    if first_hash and second_hash and len(first_hash) == len(second_hash):
        differences = sum(a != b for a, b in zip(first_hash, second_hash))
        hash_similarity = 1.0 - differences / len(first_hash)
    else:
        hash_similarity = 0.0

    blue_similarity = max(
        0.0,
        1.0 - abs(float(first.get("blue_ratio", 0.0)) - float(second.get("blue_ratio", 0.0))) * 4.0,
    )
    brightness_similarity = max(
        0.0,
        1.0 - abs(float(first.get("brightness", 0.0)) - float(second.get("brightness", 0.0))) / 255.0,
    )
    return round(hash_similarity * 0.7 + blue_similarity * 0.2 + brightness_similarity * 0.1, 6)


def detect_blue_stage_candidates(
    image_np: np.ndarray,
    min_area: int = 120,
    max_area_ratio: float = 0.08,
    min_blue_fill: float = 0.08,
    lower_hsv: list[int] | tuple[int, int, int] | None = None,
    upper_hsv: list[int] | tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Detect blue-highlighted stage candidates in a viewport image."""
    if image_np.size == 0:
        return {"mask": np.zeros((1, 1), dtype=np.uint8), "candidates": []}

    rgb = image_np if image_np.ndim == 3 else cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    lower_blue = np.array(lower_hsv if lower_hsv is not None else [85, 50, 50], dtype=np.uint8)
    upper_blue = np.array(upper_hsv if upper_hsv is not None else [140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = mask.shape[0] * mask.shape[1]
    max_area = max(float(min_area), image_area * float(max_area_ratio))

    candidates: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(min_area) or area > max_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if width < 10 or height < 10:
            continue

        aspect_ratio = width / float(height)
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            continue

        roi_mask = mask[y : y + height, x : x + width]
        blue_fill = float(np.count_nonzero(roi_mask)) / max(1.0, float(width * height))
        if blue_fill < float(min_blue_fill):
            continue

        center_x = x + width / 2.0
        center_y = y + height / 2.0
        candidates.append(
            {
                "bbox_local": {"x": int(x), "y": int(y), "width": int(width), "height": int(height)},
                "center_local": {"x": round(float(center_x), 3), "y": round(float(center_y), 3)},
                "center_rel": {
                    "x": round(float(center_x) / max(1.0, float(mask.shape[1])), 6),
                    "y": round(float(center_y) / max(1.0, float(mask.shape[0])), 6),
                },
                "mask_area": round(area, 3),
                "blue_fill": round(blue_fill, 6),
            }
        )

    candidates.sort(key=lambda item: (item["center_local"]["y"], item["center_local"]["x"]))
    return {"mask": mask, "candidates": candidates}


def render_candidate_overlay(image_np: np.ndarray, candidates: list[dict[str, Any]]) -> np.ndarray:
    """Draw candidate rectangles and centers over the viewport image."""
    rgb = image_np.copy() if image_np.ndim == 3 else cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    for idx, candidate in enumerate(candidates, start=1):
        bbox = candidate["bbox_local"]
        center = candidate["center_local"]
        x1 = int(bbox["x"])
        y1 = int(bbox["y"])
        x2 = int(bbox["x"] + bbox["width"])
        y2 = int(bbox["y"] + bbox["height"])
        center_x = int(round(center["x"]))
        center_y = int(round(center["y"]))

        predicted_class = str(candidate.get("predicted_class") or "unknown")
        confidence = float(candidate.get("confidence", candidate.get("score", 0.0)) or 0.0)
        if bool(candidate.get("accepted")):
            color = (0, 255, 0)
        elif predicted_class.endswith("_avoid"):
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.circle(overlay, (center_x, center_y), 4, color, -1)
        cv2.putText(
            overlay,
            f"{idx}:{predicted_class}:{confidence:.2f}",
            (x1, max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


class MapMemoryStore:
    """Persist map viewports so later sessions can compare and localize against them."""

    def __init__(self, mode_name: str, root_dir: str | Path = Path("data") / "map_data"):
        self.mode_name = sanitize_name(mode_name)
        self.root_dir = Path(root_dir)
        self.mode_dir = self.root_dir / self.mode_name
        self.last_session_dir = self.mode_dir / "last_session"
        self.latest_file = self.mode_dir / "latest.json"
        self.mode_dir.mkdir(parents=True, exist_ok=True)

        self.session_dir: Path | None = None
        self.session_data: dict[str, Any] | None = None
        self.loaded_map = self.load_latest()
        self._viewport_image_cache: dict[str, np.ndarray | None] = {}

    def load_latest(self) -> dict[str, Any] | None:
        if not self.latest_file.exists():
            return None
        try:
            return json.loads(self.latest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def start_session(self, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        self.session_dir = self.last_session_dir
        session_dir_resolved = self.session_dir.resolve()
        mode_dir_resolved = self.mode_dir.resolve()
        if mode_dir_resolved not in session_dir_resolved.parents:
            raise ValueError(f"Refusing to reset unexpected session directory: {self.session_dir}")
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir)
        (self.session_dir / "viewports").mkdir(parents=True, exist_ok=True)
        self.session_data = {
            "mode_name": self.mode_name,
            "map_scope": "shared_by_mode",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "meta": meta or {},
            "viewports": [],
            "localization_candidates": [],
            "scan_events": [],
            "executed_stages": [],
        }
        return self.session_data

    def append_scan_event(self, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if self.session_data is None:
            return None

        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event_type,
            "payload": payload or {},
        }
        self.session_data.setdefault("scan_events", []).append(event)
        return event

    def record_executed_stage(self, stage_record: dict[str, Any]) -> None:
        if self.session_data is None:
            return
        self.session_data.setdefault("executed_stages", []).append(stage_record)

    def get_loaded_viewport(self, view_id: str | None) -> dict[str, Any] | None:
        if not self.loaded_map or not view_id:
            return None

        for viewport in self.loaded_map.get("viewports", []):
            if viewport.get("view_id") == view_id:
                return viewport
        return None

    def _viewport_extra(self, viewport: dict[str, Any]) -> dict[str, Any]:
        extra = viewport.get("extra")
        return extra if isinstance(extra, dict) else {}

    def _viewport_image_path(self, viewport: dict[str, Any]) -> Path | None:
        image_path = viewport.get("image_path")
        if not image_path:
            return None
        return Path(image_path)

    def _load_viewport_image(self, viewport: dict[str, Any]) -> np.ndarray | None:
        image_path = self._viewport_image_path(viewport)
        if image_path is None:
            return None

        cache_key = image_path.as_posix()
        if cache_key in self._viewport_image_cache:
            return self._viewport_image_cache[cache_key]

        if not image_path.exists():
            self._viewport_image_cache[cache_key] = None
            return None

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            self._viewport_image_cache[cache_key] = None
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._viewport_image_cache[cache_key] = image_rgb
        return image_rgb

    def _ensure_viewport_match_data(self, viewport: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(viewport, dict):
            return {}

        extra = viewport.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            viewport["extra"] = extra

        if not viewport.get("fragment_signatures"):
            image_np = self._load_viewport_image(viewport)
            if image_np is not None:
                viewport["fragment_signatures"] = compute_fragment_signatures(image_np)

        if not extra.get("stage_layout_signature"):
            stage_candidates = extra.get("stage_candidates", [])
            if stage_candidates:
                extra["stage_layout_signature"] = compute_stage_layout_signature(stage_candidates)

        return viewport

    def find_best_matches(
        self,
        signature: dict[str, Any],
        limit: int = 3,
        difficulty: str | None = None,
        query_fragments: list[dict[str, Any]] | None = None,
        query_layout: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.loaded_map:
            return []

        matches: list[dict[str, Any]] = []
        for viewport in self.loaded_map.get("viewports", []):
            self._ensure_viewport_match_data(viewport)
            existing_signature = viewport.get("signature")
            if not existing_signature:
                continue

            viewport_extra = self._viewport_extra(viewport)
            viewport_difficulty = viewport.get("difficulty") or viewport_extra.get("difficulty")
            if difficulty and viewport_difficulty and viewport_difficulty != difficulty:
                continue

            global_score = compare_view_signatures(signature, existing_signature)
            fragment_comparison = compare_fragment_signature_sets(
                query_fragments,
                viewport.get("fragment_signatures", []),
            )
            layout_comparison = compare_stage_layout_signatures(
                query_layout,
                viewport_extra.get("stage_layout_signature"),
            )
            fragment_score = float(fragment_comparison.get("score", 0.0))
            layout_score = float(layout_comparison.get("score", 0.0))
            score = round(
                max(
                    global_score * 0.5 + fragment_score * 0.35 + layout_score * 0.15,
                    fragment_score * 0.75 + layout_score * 0.25,
                    global_score,
                ),
                6,
            )
            matches.append(
                {
                    "view_id": viewport.get("view_id"),
                    "grid_position": normalize_grid_position(viewport.get("grid_position")),
                    "score": score,
                    "score_components": {
                        "global": round(global_score, 6),
                        "fragment": round(fragment_score, 6),
                        "layout": round(layout_score, 6),
                        "strong_fragment_matches": int(fragment_comparison.get("strong_match_count", 0)),
                        "matched_layout_centers": int(layout_comparison.get("matched_center_count", 0)),
                    },
                    "image_path": viewport.get("image_path"),
                    "difficulty": viewport_difficulty,
                    "label": viewport.get("label"),
                    "move_action": viewport.get("move_action"),
                }
            )

        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[:limit]

    def compare_current_view(
        self,
        window,
        search_area,
        limit: int = 3,
        difficulty: str | None = None,
        detector: Any | None = None,
        lower_hsv: list[int] | tuple[int, int, int] | None = None,
        upper_hsv: list[int] | tuple[int, int, int] | None = None,
        min_area: int = 120,
        max_area_ratio: float = 0.08,
        min_blue_fill: float = 0.08,
    ) -> dict[str, Any]:
        _, image_np, _ = capture_relative_area(window, search_area)
        signature = compute_view_signature(image_np)
        fragment_signatures = compute_fragment_signatures(image_np)
        if detector is not None and hasattr(detector, "detect"):
            detection = detector.detect(image_np)
        else:
            detection = detect_blue_stage_candidates(
                image_np,
                min_area=min_area,
                max_area_ratio=max_area_ratio,
                min_blue_fill=min_blue_fill,
                lower_hsv=lower_hsv,
                upper_hsv=upper_hsv,
            )
        stage_layout_signature = compute_stage_layout_signature(detection.get("candidates", []))
        matches = self.find_best_matches(
            signature,
            limit=limit,
            difficulty=difficulty,
            query_fragments=fragment_signatures,
            query_layout=stage_layout_signature,
        )
        return {
            "signature": signature,
            "fragment_signatures": fragment_signatures,
            "stage_layout_signature": stage_layout_signature,
            "stage_candidate_count": len(detection.get("candidates", [])),
            "matches": matches,
            "best_match": matches[0] if matches else None,
        }

    def add_viewport(
        self,
        screenshot,
        image_np: np.ndarray,
        relative_area,
        grid_position: dict[str, int],
        sequence_index: int,
        move_action: str | None = None,
        label: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.session_data is None or self.session_dir is None:
            self.start_session()

        view_id = f"{sequence_index:03d}_{sanitize_name(label or 'viewport')}"
        image_path = self.session_dir / "viewports" / f"{view_id}.png"
        screenshot.save(image_path)

        signature = compute_view_signature(image_np)
        fragment_signatures = compute_fragment_signatures(image_np)
        matches = self.find_best_matches(signature, limit=3, query_fragments=fragment_signatures)
        entry = {
            "view_id": view_id,
            "image_path": image_path.as_posix(),
            "grid_position": grid_position,
            "move_action": move_action,
            "label": label,
            "relative_area": list(relative_area),
            "signature": signature,
            "fragment_signatures": fragment_signatures,
            "best_saved_matches": matches,
            "captured_at": datetime.now().isoformat(timespec="seconds"),
        }
        if extra:
            entry["extra"] = extra

        self.session_data["viewports"].append(entry)
        return entry

    def _viewport_merge_key(self, viewport: dict[str, Any]) -> tuple[str, int, int]:
        difficulty = str(viewport.get("difficulty") or self._viewport_extra(viewport).get("difficulty") or "")
        grid_position = normalize_grid_position(viewport.get("grid_position"))
        return (difficulty, int(grid_position["x"]), int(grid_position["y"]))

    def _stage_candidate_count(self, viewport: dict[str, Any]) -> int:
        extra = viewport.get("extra", {})
        if not isinstance(extra, dict):
            return 0
        if "stage_candidate_count" in extra:
            return int(extra.get("stage_candidate_count") or 0)
        return len(extra.get("stage_candidates", []))

    def _select_preferred_viewport(
        self,
        existing_viewport: dict[str, Any] | None,
        candidate_viewport: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if existing_viewport is None:
            return candidate_viewport
        if candidate_viewport is None:
            return existing_viewport

        existing_stage_count = self._stage_candidate_count(existing_viewport)
        candidate_stage_count = self._stage_candidate_count(candidate_viewport)
        if candidate_stage_count != existing_stage_count:
            return candidate_viewport if candidate_stage_count > existing_stage_count else existing_viewport

        existing_saved_matches = len(existing_viewport.get("best_saved_matches", []))
        candidate_saved_matches = len(candidate_viewport.get("best_saved_matches", []))
        if candidate_saved_matches != existing_saved_matches:
            return candidate_viewport if candidate_saved_matches >= existing_saved_matches else existing_viewport

        return candidate_viewport

    def _build_latest_payload(self) -> dict[str, Any] | None:
        if self.session_data is None:
            return self.loaded_map

        merged_viewports: dict[tuple[str, int, int], dict[str, Any]] = {}
        for payload in [self.loaded_map, self.session_data]:
            if not payload:
                continue
            for viewport in payload.get("viewports", []):
                self._ensure_viewport_match_data(viewport)
                key = self._viewport_merge_key(viewport)
                merged_viewports[key] = self._select_preferred_viewport(merged_viewports.get(key), viewport)

        current_meta = dict(self.session_data.get("meta", {}))
        previous_meta = dict((self.loaded_map or {}).get("meta", {}))
        merged_meta = {**previous_meta, **current_meta}
        merged_meta["merged_from_previous_map"] = bool(self.loaded_map)
        merged_meta["merged_viewport_count"] = len(merged_viewports)
        if self.session_dir is not None:
            merged_meta["last_session_dir"] = self.session_dir.as_posix()

        sorted_viewports = sorted(
            merged_viewports.values(),
            key=lambda viewport: (
                str(viewport.get("difficulty") or self._viewport_extra(viewport).get("difficulty") or ""),
                int(normalize_grid_position(viewport.get("grid_position"))["y"]),
                int(normalize_grid_position(viewport.get("grid_position"))["x"]),
                str(viewport.get("view_id") or ""),
            ),
        )
        return {
            "mode_name": self.mode_name,
            "map_scope": self.session_data.get("map_scope", "shared_by_mode"),
            "created_at": (self.loaded_map or {}).get("created_at") or self.session_data.get("created_at"),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "meta": merged_meta,
            "viewports": sorted_viewports,
            "localization_candidates": list(self.session_data.get("localization_candidates", [])),
            "scan_events": list(self.session_data.get("scan_events", [])),
            "executed_stages": list(self.session_data.get("executed_stages", [])),
        }

    def save_session(self) -> Path | None:
        if self.session_data is None or self.session_dir is None:
            return None

        self.session_data["saved_at"] = datetime.now().isoformat(timespec="seconds")
        session_file = self.session_dir / "map_state.json"
        session_payload = json.dumps(
            self.session_data,
            indent=2,
            ensure_ascii=True,
            default=_json_default,
        )
        session_file.write_text(session_payload + "\n", encoding="utf-8")

        latest_payload = self._build_latest_payload() or self.session_data
        latest_text = json.dumps(
            latest_payload,
            indent=2,
            ensure_ascii=True,
            default=_json_default,
        )
        self.latest_file.write_text(latest_text + "\n", encoding="utf-8")
        self.loaded_map = latest_payload
        return session_file
