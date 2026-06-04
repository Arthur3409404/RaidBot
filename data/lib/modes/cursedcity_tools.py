# -*- coding: utf-8 -*-
"""Cursed City RaidBot mode integration."""

from __future__ import annotations

import difflib
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import data.lib.utils.auto_battle_tools as auto_battle_tools
import data.lib.utils.image_tools as image_tools
import data.lib.utils.map_tools as map_tools
import data.lib.utils.window_tools as window_tools


MENU_TITLE = "Ciudad Maldita"
KEY_DENOMINATOR = 8
MAX_RUN_DURATION_SECONDS = int(3.5 * 60 * 60)


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    score: float


def _largest_component_bbox(binary_img: np.ndarray) -> tuple[int, int, int, int] | None:
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    if n_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    label_idx = 1 + int(np.argmax(areas))
    return (
        int(stats[label_idx, cv2.CC_STAT_LEFT]),
        int(stats[label_idx, cv2.CC_STAT_TOP]),
        int(stats[label_idx, cv2.CC_STAT_WIDTH]),
        int(stats[label_idx, cv2.CC_STAT_HEIGHT]),
    )


def _crop_to_bbox(binary_img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return binary_img[y : y + h, x : x + w]


def _dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    size = int(np.count_nonzero(a)) + int(np.count_nonzero(b))
    if size <= 0:
        return 0.0
    return float((2.0 * np.count_nonzero(a & b)) / size)


def _load_reference_stats(reference_dir: Path, target_size: int = 64) -> dict | None:
    templates = []
    raw_templates = []
    aspect_values = []
    fill_values = []

    for path in sorted(reference_dir.glob("*.png")):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        bbox = _largest_component_bbox(bw)
        if bbox is None:
            continue
        roi = _crop_to_bbox(bw, bbox)
        height, width = roi.shape
        if height <= 0 or width <= 0:
            continue
        templates.append(cv2.resize(roi, (target_size, target_size), interpolation=cv2.INTER_NEAREST))
        raw_templates.append(roi)
        aspect_values.append(float(width) / float(height))
        fill_values.append(float(np.count_nonzero(roi)) / float(max(1, width * height)))

    if not templates:
        return None

    return {
        "templates": templates,
        "raw_templates": raw_templates,
        "aspect_mean": float(np.mean(aspect_values)),
        "aspect_std": float(np.std(aspect_values) + 1e-6),
        "fill_mean": float(np.mean(fill_values)),
        "fill_std": float(np.std(fill_values) + 1e-6),
    }


def _component_candidates(binary_img: np.ndarray) -> list[dict]:
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue
        roi = binary_img[y : y + height, x : x + width]
        area = float(cv2.contourArea(contour))
        bbox_area = float(width * height)
        fill = float(np.count_nonzero(roi)) / float(max(1, width * height))
        perimeter = float(cv2.arcLength(contour, True))
        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter)) if perimeter > 0 else 0.0
        candidates.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(width),
                "h": int(height),
                "area": area,
                "bbox_area": bbox_area,
                "fill": fill,
                "aspect": float(width) / float(height),
                "circularity": circularity,
                "roi": roi,
            }
        )
    return candidates


def _template_match_candidates(binary_img: np.ndarray, ref_stats: dict) -> list[dict]:
    candidates = []
    for template_id, ref in enumerate(ref_stats.get("raw_templates", [])):
        for scale in np.linspace(1.0, 3.2, 12):
            width = max(8, int(ref.shape[1] * float(scale)))
            height = max(8, int(ref.shape[0] * float(scale)))
            if width >= binary_img.shape[1] or height >= binary_img.shape[0]:
                continue
            template = cv2.resize(ref, (width, height), interpolation=cv2.INTER_NEAREST)
            result = cv2.matchTemplate(binary_img, template, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(result >= 0.38)
            for y, x in zip(ys, xs):
                roi = binary_img[int(y) : int(y) + height, int(x) : int(x) + width]
                if roi.shape[:2] != (height, width):
                    continue
                candidates.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "w": int(width),
                        "h": int(height),
                        "area": float(np.count_nonzero(roi)),
                        "bbox_area": float(width * height),
                        "fill": float(np.count_nonzero(roi)) / float(max(1, width * height)),
                        "aspect": float(width) / float(height),
                        "circularity": 0.0,
                        "roi": roi,
                        "template_score_raw": float(result[int(y), int(x)]),
                        "template_id": int(template_id),
                    }
                )
    return candidates


def _score_candidate(candidate: dict, ref_stats: dict, target_size: int = 64) -> float:
    roi = candidate["roi"]
    resized = cv2.resize(roi, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    template_score = max(_dice_score(resized, template) for template in ref_stats["templates"])
    aspect_z = abs(candidate["aspect"] - ref_stats["aspect_mean"]) / max(ref_stats["aspect_std"], 0.02)
    fill_z = abs(candidate["fill"] - ref_stats["fill_mean"]) / max(ref_stats["fill_std"], 0.03)
    aspect_score = float(np.exp(-0.5 * aspect_z))
    fill_score = float(np.exp(-0.5 * fill_z))
    rectangularity_score = float(np.clip(candidate["fill"] / max(ref_stats["fill_mean"], 1e-6), 0.0, 1.0))
    return float((0.62 * template_score) + (0.18 * aspect_score) + (0.14 * fill_score) + (0.06 * rectangularity_score))


def _bbox_iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (a["w"] * a["h"]) + (b["w"] * b["h"]) - inter
    return float(inter / union) if union > 0 else 0.0


def _nms(sorted_candidates: list[dict], iou_thresh: float = 0.35) -> list[dict]:
    kept = []
    for candidate in sorted_candidates:
        if all(_bbox_iou(candidate, previous) <= iou_thresh for previous in kept):
            kept.append(candidate)
    return kept


def detect_cursedcity_like_structures(
    binary_img: np.ndarray,
    reference_dir: Path,
    *,
    expected_count: int = 5,
    max_objects: int = 12,
    min_score: float = 0.52,
    min_template_score_raw: float = 0.0,
) -> list[BoundingBox]:
    """Detect Cursed City stage structures from an in-memory binary mask."""
    if binary_img is None or np.asarray(binary_img).size == 0:
        return []

    bw = np.asarray(binary_img, dtype=np.uint8)
    _, bw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY)
    ref_stats = _load_reference_stats(reference_dir)
    if not ref_stats:
        return []

    img_area = float(max(1, bw.shape[0] * bw.shape[1]))
    best: dict | None = None
    param_grid = [
        {"open": 0, "close": 0, "dilate": 0},
        {"open": 3, "close": 3, "dilate": 0},
        {"open": 3, "close": 5, "dilate": 0},
        {"open": 3, "close": 5, "dilate": 1},
        {"open": 5, "close": 5, "dilate": 0},
        {"open": 5, "close": 7, "dilate": 1},
    ]
    size_profiles = [
        {"min_area_ratio": 0.00005, "max_area_ratio": 0.015, "min_fill": 0.05, "max_fill": 0.70, "aspect_lo": 0.45, "aspect_hi": 1.8},
        {"min_area_ratio": 0.00008, "max_area_ratio": 0.020, "min_fill": 0.04, "max_fill": 0.75, "aspect_lo": 0.40, "aspect_hi": 2.0},
        {"min_area_ratio": 0.00012, "max_area_ratio": 0.028, "min_fill": 0.03, "max_fill": 0.82, "aspect_lo": 0.35, "aspect_hi": 2.2},
    ]

    for params in param_grid:
        proc = bw.copy()
        if params["open"] > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (params["open"], params["open"]))
            proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, kernel)
        if params["close"] > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (params["close"], params["close"]))
            proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel)
        if params["dilate"] > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            proc = cv2.dilate(proc, kernel, iterations=int(params["dilate"]))

        components = _component_candidates(proc)
        for profile in size_profiles:
            min_area = profile["min_area_ratio"] * img_area
            max_area = profile["max_area_ratio"] * img_area
            filtered = []
            for candidate in components:
                if not (min_area <= candidate["area"] <= max_area):
                    continue
                if not (profile["min_fill"] <= candidate["fill"] <= profile["max_fill"]):
                    continue
                if not (profile["aspect_lo"] <= candidate["aspect"] <= profile["aspect_hi"]):
                    continue
                candidate = dict(candidate)
                candidate["score"] = _score_candidate(candidate, ref_stats)
                filtered.append(candidate)

            filtered.sort(key=lambda item: item["score"], reverse=True)
            filtered = _nms(filtered)
            top = filtered[:expected_count]
            mean_score = float(np.mean([item["score"] for item in top])) if top else 0.0
            quality = mean_score - (abs(expected_count - len(top)) * 0.18)
            if best is None or quality > best["quality"]:
                best = {"quality": quality, "candidates": filtered}

    template_candidates = _template_match_candidates(bw, ref_stats)
    for candidate in template_candidates:
        candidate["score"] = (
            0.55 * float(candidate.get("template_score_raw", 0.0))
            + 0.45 * _score_candidate(candidate, ref_stats)
        )
    template_candidates.sort(key=lambda item: item["score"], reverse=True)
    template_candidates = _nms(template_candidates, iou_thresh=0.25)
    template_top = template_candidates[:expected_count]
    template_mean = float(np.mean([item["score"] for item in template_top])) if template_top else 0.0
    template_quality = template_mean - (abs(expected_count - len(template_top)) * 0.18)
    if best is None or template_quality > best["quality"]:
        best = {"quality": template_quality, "candidates": template_candidates}

    selected = [
        candidate
        for candidate in list((best or {}).get("candidates", []))
        if float(candidate.get("score", 0.0)) >= min_score
        and float(candidate.get("template_score_raw", 1.0)) >= min_template_score_raw
    ][: max(0, int(max_objects))]

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


def _start_run_deadline(bot, max_run_duration_seconds=None):
    limit = bot.max_run_duration_seconds if max_run_duration_seconds is None else float(max_run_duration_seconds)
    bot._run_deadline = time.time() + limit


def _ensure_within_run_deadline(bot, context: str):
    deadline = getattr(bot, "_run_deadline", None)
    if deadline and time.time() > deadline:
        hours = getattr(bot, "max_run_duration_seconds", MAX_RUN_DURATION_SECONDS) / 3600.0
        raise TimeoutError(f"{bot.__class__.__name__} exceeded max runtime of {hours:.1f}h while {context}.")


class RSL_Bot_CursedCity:
    def __init__(self, title_substring="Raid: Shadow Legends", reader=None, window=None, verbose=True, setup=None):
        self.reader = reader
        self.window = window
        self.verbose = verbose
        self.log = logging.getLogger(self.__class__.__name__)
        self.title_substring = title_substring

        self.search_areas = {
            "menu_name": [0.008, 0.034, 0.23, 0.037],
            "mode_keys_row": [0.18, 0.02, 0.80, 0.07],
            "mode_difficulty_current": [0.032, 0.903, 0.166, 0.051],
            "mode_difficulty_switch_normal": [0.046, 0.841, 0.139, 0.056],
            "mode_difficulty_switch_hard": [0.046, 0.891, 0.139, 0.056],
            "pov": [0.0, 0.0, 1.0, 1.0],
            "go_to_higher_menu": [0.928, 0.031, 0.046, 0.039],
            "stage_lower_half_text_scan": [0.0, 0.5, 1.0, 0.5],
            "stage_confirm_button_champion_selection": [0.762, 0.876, 0.213, 0.104],
            "stage_auto_battle_button": [0.026, 0.899, 0.058, 0.07],
            "stage_battle_result": [0.389, 0.148, 0.204, 0.071],
            "stage_battle_result_2": [0.38, 0.085, 0.224, 0.059],
        }
        self.translation_mapping = {
            "Normal": "normal",
            "Dificil": "hard",
            "Hard": "hard",
            "Difícil": "hard",
            "Difficult": "hard",
            "Hard Mode": "hard",
            "Modo Normal": "normal",
            "Modo Dificil": "hard",
            "Modo Difícil": "hard",
        }
        self.setup = {
            "execution_mode": "normal",
            "difficulty": "hard",
            "alternate_difficulty": True,
            "difficulty_switch_retries": 3,
            "difficulty_dropdown_open_delay_seconds": 0.8,
            "difficulty_switch_confirm_delay_seconds": 2.5,
            "post_entry_wait_seconds": 5.0,
            "candidate_detection_retries_per_view": 2,
            "max_random_repositions_when_no_candidates": 3,
            "max_failed_selection_repositions": 3,
            "target_hex": "CEC329",
            "reference_dir": str(Path("pic") / "cursedcity"),
            "expected_structure_count": 5,
            "detector_max_objects": 12,
            "detector_min_score": 0.52,
            "detector_min_template_score_raw": 0.0,
            "stage_select_delay_seconds": 3.0,
            "stage_start_retries": 3,
            "stage_battle_timeout_seconds": 420.0,
            "stage_battle_poll_interval_seconds": 2.0,
            "stage_battle_outcome_confirm_delay_seconds": 10.0,
        }
        if setup:
            self.setup.update(setup)

        self.reference_dir = Path(str(self.setup.get("reference_dir") or Path("pic") / "cursedcity"))
        self.target_bgr_as_rgb = self._hex_to_bgr_as_rgb(str(self.setup.get("target_hex", "CEC329")))
        self.max_run_duration_seconds = MAX_RUN_DURATION_SECONDS
        self._run_deadline = None
        self._difficulty_toggle = 0
        self.reset_run_state()

    @staticmethod
    def _hex_to_bgr_as_rgb(hex_color: str) -> np.ndarray:
        value = str(hex_color or "").lstrip("#")
        if len(value) != 6:
            value = "CEC329"
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        return np.array([b, g, r], dtype=np.uint8)

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
        self.last_defeat_by_difficulty = getattr(self, "last_defeat_by_difficulty", {"hard": None, "normal": None})

    def _read_menu_name(self) -> str | None:
        try:
            texts = image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                self.search_areas["menu_name"],
                power_detection=False,
            )
            return texts[0].text.strip() if texts else None
        except Exception:
            self.log.debug("[Cursed City] Menu OCR failed.", exc_info=True)
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
            self.log.debug("[Cursed City] OCR failed for area '%s'.", area_key, exc_info=True)
            return []

    def _normalize_difficulty_value(self, value) -> str | None:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return None

        normalized = (
            normalized.replace("í", "i")
            .replace("ì", "i")
            .replace("ï", "i")
            .replace("á", "a")
            .replace("é", "e")
            .replace("ó", "o")
            .replace("ú", "u")
        )
        compact = " ".join(normalized.split())

        if compact in {"normal", "modo normal"}:
            return "normal"
        if compact in {"hard", "dificil", "difficult", "hard mode", "modo dificil"}:
            return "hard"

        for text_value, internal_value in self.translation_mapping.items():
            if self.resembles(compact, text_value, threshold=0.70):
                return internal_value
        return None

    def _is_in_game_modes_menu(self, menu_text: str | None) -> bool:
        return bool(
            menu_text
            and (
                self.resembles(menu_text, "Modos de juego", threshold=0.55)
                or self.resembles(menu_text, "Modo de juego", threshold=0.55)
            )
        )

    def is_in_cursed_city_mode(self) -> bool:
        menu_text = self._read_menu_name()
        return bool(menu_text and self.resembles(menu_text, MENU_TITLE, threshold=0.55))

    def _perform_startup_check(self) -> bool:
        wait_seconds = float(self.setup.get("post_entry_wait_seconds", 5.0) or 0.0)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        menu_text = self._read_menu_name()
        ok = bool(menu_text and self.resembles(menu_text, MENU_TITLE, threshold=0.55))
        if not ok:
            self.log.warning("[Cursed City] Startup check failed. Expected '%s', got '%s'.", MENU_TITLE, menu_text)
        return ok

    def detect_current_difficulty(self):
        try:
            texts = image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                search_area=self.search_areas["mode_difficulty_current"],
            )
            detected_texts = [getattr(text, "text", "") for text in texts if getattr(text, "text", "")]
        except Exception:
            detected_texts = []

        candidates = list(detected_texts)
        if detected_texts:
            candidates.append(" ".join(detected_texts))
            candidates.append("".join(detected_texts))

        for detected_text in candidates:
            normalized = self._normalize_difficulty_value(detected_text)
            if normalized:
                self.current_difficulty = normalized
                return self.current_difficulty
        return self.current_difficulty

    def set_difficulty(self, set_level=None):
        requested = self._normalize_difficulty_value(set_level)
        if requested not in {"normal", "hard"}:
            self.log.warning("[Cursed City] Ignoring invalid difficulty request: %s", set_level)
            return self.detect_current_difficulty()

        current = self.detect_current_difficulty()
        if current == requested:
            self.log.info("[Cursed City] Difficulty already set to '%s'.", requested)
            return current

        switch_key = f"mode_difficulty_switch_{requested}"
        retries = max(1, int(self.setup.get("difficulty_switch_retries", 3)))
        open_delay = float(self.setup.get("difficulty_dropdown_open_delay_seconds", 0.8))
        confirm_delay = float(self.setup.get("difficulty_switch_confirm_delay_seconds", 2.5))

        for attempt in range(1, retries + 1):
            self.log.info(
                "[Cursed City] Switching difficulty to '%s' (%s/%s).",
                requested,
                attempt,
                retries,
            )
            window_tools.click_center(
                self.window,
                self.search_areas["mode_difficulty_current"],
                delay=open_delay,
            )
            window_tools.click_center(
                self.window,
                self.search_areas[switch_key],
                delay=confirm_delay,
            )

            current = self.detect_current_difficulty()
            if current == requested:
                self.log.info("[Cursed City] Difficulty switch confirmed: '%s'.", requested)
                return current

            self.log.warning(
                "[Cursed City] Difficulty switch not confirmed yet. Requested='%s', detected='%s'.",
                requested,
                current,
            )

        self.log.warning(
            "[Cursed City] Could not confirm difficulty switch to '%s'; continuing with detected='%s'.",
            requested,
            current,
        )
        return current

    def _planned_difficulty(self) -> str:
        configured = self._normalize_difficulty_value(self.setup.get("difficulty", "hard"))
        if bool(self.setup.get("alternate_difficulty", False)):
            self._difficulty_toggle += 1
            return "hard" if self._difficulty_toggle % 2 == 1 else "normal"
        if configured not in {"normal", "hard"}:
            self.log.warning("[Cursed City] Invalid difficulty '%s'. Defaulting to hard.", self.setup.get("difficulty"))
            return "hard"
        return configured

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

    def has_cursed_city_keys_remaining(self, retries: int = 3) -> bool:
        for attempt in range(1, max(1, int(retries)) + 1):
            try:
                keys = self.update_available_keys()
            except Exception:
                keys = 0
                self.log.debug("[Cursed City] Key OCR attempt failed.", exc_info=True)
            self.log.info("[Cursed City] Key check (%s/%s): %s/%s", attempt, retries, keys, KEY_DENOMINATOR)
            if keys > 0:
                return True
            time.sleep(0.6)
        return False

    def _capture_cursed_city_mask(self):
        _, pov_np, pov_region = map_tools.capture_relative_area(self.window, self.search_areas["pov"])
        base = self.target_bgr_as_rgb.astype(np.int16)
        diff = pov_np.astype(np.int16) - base
        brighter_tones_mask = (
            (diff[:, :, 0] >= 0)
            & (diff[:, :, 1] >= 0)
            & (diff[:, :, 2] >= 0)
        )
        return np.where(brighter_tones_mask, 255, 0).astype(np.uint8), pov_region

    def detect_cursed_city_candidates(self, retries: int | None = None) -> list[dict]:
        retries = max(1, int(retries or self.setup.get("candidate_detection_retries_per_view", 2)))
        for attempt in range(1, retries + 1):
            mask, pov_region = self._capture_cursed_city_mask()
            boxes = detect_cursedcity_like_structures(
                mask,
                self.reference_dir,
                expected_count=int(self.setup.get("expected_structure_count", 5)),
                max_objects=int(self.setup.get("detector_max_objects", 12)),
                min_score=float(self.setup.get("detector_min_score", 0.52)),
                min_template_score_raw=float(self.setup.get("detector_min_template_score_raw", 0.0)),
            )
            width = max(1.0, float(pov_region[2] or 1.0))
            height = max(1.0, float(pov_region[3] or 1.0))
            candidates = []
            for idx, box in enumerate(boxes, start=1):
                center_local_x = int(box.x + (box.width / 2.0))
                center_local_y = int(box.y + (box.height / 2.0))
                candidates.append(
                    {
                        "index": idx,
                        "score": float(box.score or 0.0),
                        "center_rel_x": float(center_local_x / width),
                        "center_rel_y": float(center_local_y / height),
                        "bbox_rel": {
                            "x": float(max(0, box.x) / width),
                            "y": float(max(0, box.y) / height),
                            "width": float(max(0, box.width) / width),
                            "height": float(max(0, box.height) / height),
                        },
                        "center_abs_x": int(pov_region[0] + center_local_x),
                        "center_abs_y": int(pov_region[1] + center_local_y),
                    }
                )
            candidates.sort(key=lambda item: item["score"], reverse=True)
            self.log.info("[Cursed City] Candidate detection (%s/%s): %s candidates.", attempt, retries, len(candidates))
            if candidates:
                return candidates
            time.sleep(0.8)
        return []

    @staticmethod
    def _bbox_overlap_ratio(a: dict, b: dict) -> float:
        ax1 = float(a.get("x", 0.0) or 0.0)
        ay1 = float(a.get("y", 0.0) or 0.0)
        ax2 = ax1 + float(a.get("width", 0.0) or 0.0)
        ay2 = ay1 + float(a.get("height", 0.0) or 0.0)
        bx1 = float(b.get("x", 0.0) or 0.0)
        by1 = float(b.get("y", 0.0) or 0.0)
        bx2 = bx1 + float(b.get("width", 0.0) or 0.0)
        by2 = by1 + float(b.get("height", 0.0) or 0.0)
        inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
        denom = min(max(0.0, (ax2 - ax1) * (ay2 - ay1)), max(0.0, (bx2 - bx1) * (by2 - by1)))
        return float(inter / denom) if denom > 0 else 0.0

    def _filter_candidates_against_last_defeat(self, candidates: list[dict], difficulty: str):
        defeat_entry = self.last_defeat_by_difficulty.get(str(difficulty or "").lower())
        if not isinstance(defeat_entry, dict):
            return list(candidates)
        defeat_bbox = defeat_entry.get("bbox_rel")
        if not isinstance(defeat_bbox, dict):
            return list(candidates)
        filtered = [
            candidate
            for candidate in candidates
            if not isinstance(candidate.get("bbox_rel"), dict)
            or self._bbox_overlap_ratio(defeat_bbox, candidate["bbox_rel"]) < 0.50
        ]
        skipped = len(candidates) - len(filtered)
        if skipped:
            self.log.info("[Cursed City] Skipped %s candidate(s) matching last defeat location.", skipped)
        return filtered

    def _move_random_direction_once(self):
        moves = [
            ("up", window_tools.move_up),
            ("down", window_tools.move_down),
            ("left", window_tools.move_left),
            ("right", window_tools.move_right),
        ]
        direction_name, move_fn = random.choice(moves)
        self.log.info("[Cursed City] No candidates. Moving randomly: %s", direction_name)
        move_fn(self.window, strength=float(self.setup.get("pan_strength", 1.0)))

    def detect_candidates_with_random_reposition(self, difficulty: str | None = None):
        max_moves = max(0, int(self.setup.get("max_random_repositions_when_no_candidates", 3)))
        for wave_index in range(max_moves + 1):
            _ensure_within_run_deadline(self, "detecting Cursed City candidates")
            candidates = self.detect_cursed_city_candidates()
            candidates = self._filter_candidates_against_last_defeat(candidates, difficulty or "")
            if candidates:
                return candidates
            if wave_index < max_moves and self.main_loop_running:
                self._move_random_direction_once()
        return []

    def select_cursed_city_candidate(self, candidate: dict) -> bool:
        click_x = int(candidate["center_abs_x"])
        click_y = int(candidate["center_abs_y"])
        time.sleep(float(self.setup.get("stage_select_delay_seconds", 3.0)))
        self.log.info(
            "[Cursed City] Candidate click #%s at (%s, %s), score=%.4f",
            candidate.get("index"),
            click_x,
            click_y,
            float(candidate.get("score", 0.0) or 0.0),
        )
        window_tools.click_at(click_x, click_y, delay=2.5, window=self.window)
        return not self.is_in_cursed_city_mode()

    def _find_empezar_button_in_lower_half(self):
        for obj in self._read_text_objects("stage_lower_half_text_scan"):
            text = (getattr(obj, "text", "") or "").strip()
            if text and (self.resembles(text, "Empezar", threshold=0.6) or self.resembles(text, "Iniciar", threshold=0.6)):
                return obj
        return None

    def _press_pre_start_click_sequence(self):
        for rel_square in (
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
            window_tools.click_center(self.window, rel_square, delay=0.30)

    def click_cursed_city_start_button(self, retries: int | None = None) -> str:
        retries = max(1, int(retries or self.setup.get("stage_start_retries", 3)))
        for attempt in range(1, retries + 1):
            _ensure_within_run_deadline(self, "starting Cursed City battle")
            self.log.info("[Cursed City] Start button attempt (%s/%s).", attempt, retries)
            self._press_pre_start_click_sequence()
            menu_before = (self._read_menu_name() or "").strip()
            empezar_obj = self._find_empezar_button_in_lower_half()
            if empezar_obj is not None:
                adjusted_y = int(empezar_obj.mean_pos_y) + int(0.10 * self.window.height)
                adjusted_y = max(int(self.window.top), min(adjusted_y, int(self.window.top + self.window.height - 1)))
                window_tools.click_at(int(empezar_obj.mean_pos_x), adjusted_y, delay=1.0, window=self.window)
            else:
                window_tools.click_center(self.window, self.search_areas["stage_confirm_button_champion_selection"], delay=1.0)

            time.sleep(5.0)
            menu_after = (self._read_menu_name() or "").strip()
            if menu_before and menu_before == menu_after:
                window_tools.sendkey("esc", delay=1.0, window=self.window)
                return "battle_not_started_same_menu"

            try:
                if image_tools.check_startup(self):
                    return "battle_started"
            except Exception:
                self.log.debug("[Cursed City] Startup validation failed; treating click as started.", exc_info=True)
                return "battle_started"
            time.sleep(0.8)
        return "start_button_not_found_or_not_started"

    def _battle_result_text(self):
        for area_key in ("stage_battle_result", "stage_battle_result_2"):
            for text_object in self._read_text_objects(area_key):
                text = (getattr(text_object, "text", "") or "").strip()
                if self.resembles(text, "VICTORIA", threshold=0.68):
                    return "Victoria"
                if self.resembles(text, "DERROTA", threshold=0.68):
                    return "Derrota"
        for text_object in self._read_text_objects("pov"):
            text = (getattr(text_object, "text", "") or "").strip()
            if text and self.resembles(text, "Pausa", threshold=0.68):
                return "Pausa"
        return None

    def _is_auto_battle_visible(self) -> bool:
        objects = self._read_text_objects("stage_auto_battle_button")
        text = (getattr(objects[0], "text", "") or "").strip() if objects else ""
        return bool(text and self.resembles(text, "Auto", threshold=0.7))

    def get_battle_outcome(self, timeout_seconds: float | None = None, poll_interval_seconds: float | None = None):
        timeout_seconds = float(timeout_seconds or self.setup.get("stage_battle_timeout_seconds", 420.0))
        poll_interval_seconds = float(poll_interval_seconds or self.setup.get("stage_battle_poll_interval_seconds", 2.0))
        confirm_delay = float(self.setup.get("stage_battle_outcome_confirm_delay_seconds", 10.0))
        started_at = time.time()
        auto_seen = False
        auto_battle_tools.reset_auto_battle_watchdog(self)
        while self.main_loop_running and (time.time() - started_at) < timeout_seconds:
            _ensure_within_run_deadline(self, "waiting for Cursed City battle result")
            result = self._battle_result_text()
            if result:
                if result == "Pausa":
                    window_tools.sendkey("esc", delay=0.2, window=self.window)
                    time.sleep(max(0.6, poll_interval_seconds))
                    continue
                time.sleep(confirm_delay)
                if self._battle_result_text() == result:
                    return result
            if self._is_auto_battle_visible():
                auto_seen = True
            menu_text = self._read_menu_name()
            if auto_seen and menu_text and self.resembles(menu_text, MENU_TITLE, threshold=0.55):
                return None
            if self._is_in_game_modes_menu(menu_text):
                return None
            auto_battle_tools.ensure_auto_battle_running(
                self,
                auto_button_area=self.search_areas["stage_auto_battle_button"],
            )
            time.sleep(max(0.6, poll_interval_seconds))
        return None

    def return_to_mode_root_after_battle(self, max_attempts: int = 4) -> str:
        for _ in range(max(1, int(max_attempts))):
            menu_text = self._read_menu_name()
            if menu_text and self.resembles(menu_text, MENU_TITLE, threshold=0.55):
                return "mode"
            if self._is_in_game_modes_menu(menu_text):
                self.mode_transitioned_out = True
                return "game_modes"
            window_tools.sendkey("esc", delay=1.2, window=self.window)
        menu_text = self._read_menu_name()
        if menu_text and self.resembles(menu_text, MENU_TITLE, threshold=0.55):
            return "mode"
        if self._is_in_game_modes_menu(menu_text):
            self.mode_transitioned_out = True
            return "game_modes"
        return "unknown"

    def exit_cursed_city_to_main_menu(self, reason: str):
        self.log.info("[Cursed City] Exiting to game modes. Reason: %s", reason)
        for _ in range(3):
            menu_text = self._read_menu_name()
            if self._is_in_game_modes_menu(menu_text):
                self.mode_transitioned_out = True
                return True
            window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"], delay=1.8)
        self.mode_transitioned_out = self._is_in_game_modes_menu(self._read_menu_name())
        return self.mode_transitioned_out

    def _record_last_defeat_candidate(self, difficulty: str, candidate: dict):
        key = str(difficulty or "").strip().lower()
        if key in {"hard", "normal"}:
            self.last_defeat_by_difficulty[key] = dict(candidate)
            self.log.info("[Cursed City] Remembered defeat location for '%s' in runtime state.", key)

    def run_cursedcity(self, main_loop_running=True, max_run_duration_seconds=MAX_RUN_DURATION_SECONDS):
        _start_run_deadline(self, max_run_duration_seconds)
        self.reset_run_state()
        self.main_loop_running = main_loop_running
        if not self.reader or not self.window:
            self.log.warning("[Cursed City] Reader/window unavailable; skipping mode.")
            return False
        if not self._perform_startup_check():
            return False

        planned_difficulty = self._planned_difficulty()
        confirmed_difficulty = self.set_difficulty(planned_difficulty)
        self.current_run_difficulty = confirmed_difficulty if confirmed_difficulty in {"normal", "hard"} else planned_difficulty
        self.log.info(
            "[Cursed City] Running with difficulty '%s' (requested '%s').",
            self.current_run_difficulty,
            planned_difficulty,
        )

        failed_selection_cycles = 0
        max_failed_selection_repositions = max(1, int(self.setup.get("max_failed_selection_repositions", 3)))

        while self.main_loop_running and self.running:
            _ensure_within_run_deadline(self, "running Cursed City loop")
            if not self.has_cursed_city_keys_remaining(retries=3):
                self.exit_cursed_city_to_main_menu(reason="no_keys_remaining")
                break

            effective_difficulty = self.current_run_difficulty or self.current_difficulty or ""
            candidates = self.detect_candidates_with_random_reposition(difficulty=effective_difficulty)
            if not candidates:
                self.exit_cursed_city_to_main_menu(reason="no_candidates_detected_after_random_repositions")
                break

            selected = None
            for candidate in candidates:
                if not self.main_loop_running:
                    break
                if self.select_cursed_city_candidate(candidate):
                    selected = candidate
                    break

            if selected is None:
                if self.is_in_cursed_city_mode():
                    failed_selection_cycles += 1
                    if failed_selection_cycles <= max_failed_selection_repositions:
                        self._move_random_direction_once()
                        continue
                    self.exit_cursed_city_to_main_menu(reason="no_valid_candidate_selected_after_random_repositions")
                    break
                self.exit_cursed_city_to_main_menu(reason="left_cursed_city_after_candidate_click")
                break

            failed_selection_cycles = 0
            start_status = self.click_cursed_city_start_button()
            if start_status != "battle_started":
                if start_status == "battle_not_started_same_menu" and self.is_in_cursed_city_mode():
                    failed_selection_cycles += 1
                    if failed_selection_cycles <= max_failed_selection_repositions:
                        self._move_random_direction_once()
                        continue
                    self.exit_cursed_city_to_main_menu(reason="battle_not_started_after_random_repositions")
                    break
                self.exit_cursed_city_to_main_menu(reason="start_button_not_found")
                break

            outcome = self.get_battle_outcome()
            self.log.info("[Cursed City] Battle outcome: %s", outcome if outcome else "unknown")
            menu_status = self.return_to_mode_root_after_battle(max_attempts=4)
            if menu_status == "game_modes":
                break
            if menu_status == "unknown":
                self.exit_cursed_city_to_main_menu(reason="unknown_menu_after_battle")
                break
            if outcome == "Derrota":
                self._record_last_defeat_candidate(effective_difficulty, selected)
                self.exit_cursed_city_to_main_menu(reason="battle_lost")
                break
            if outcome != "Victoria":
                self.exit_cursed_city_to_main_menu(reason="battle_outcome_unknown_or_timeout")
                break

        return True

    def test(self):
        return self.run_cursedcity()
