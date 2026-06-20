# -*- coding: utf-8 -*-
"""Cursed City RaidBot mode integration."""

from __future__ import annotations

import difflib
import logging
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

import raid_bot.utils.auto_battle_tools as auto_battle_tools
import raid_bot.utils.image_tools as image_tools
import raid_bot.utils.map_tools as map_tools
import raid_bot.utils.window_tools as window_tools


MENU_TITLE = "Ciudad Maldita"
KEY_DENOMINATOR = 8
MAX_RUN_DURATION_SECONDS = int(3.5 * 60 * 60)
DEFAULT_DETECTOR_MODEL_PATH = Path("data") / "models" / "grimforest_detector" / "best.pt"
DEFAULT_DETECTOR_CONFIDENCE = 0.25
DEFAULT_DETECTOR_IMGSZ = 640
FORBIDDEN_ENCOUNTER_NAMES = ("borgoth", "siroth")


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    score: float


@lru_cache(maxsize=4)
def _load_yolo_detector(model_path: str):
    from raid_bot.detector_ai.yolo_detector import YoloDetector

    return YoloDetector(model_path)


def detect_cursedcity_like_structures(
    binary_img: np.ndarray,
    reference_dir: Path | None = None,
    *,
    detector_model_path: str | Path | None = None,
    detector_confidence: float = DEFAULT_DETECTOR_CONFIDENCE,
    detector_imgsz: int = DEFAULT_DETECTOR_IMGSZ,
    **_ignored_detector_params,
) -> list[BoundingBox]:
    """Detect Cursed City stage structures from an in-memory cyan/brighter mask.

    This path intentionally trusts YOLO's confidence threshold. It does not apply
    the old template score cutoff, center exclusion, or max-object truncation.
    """
    if binary_img is None or np.asarray(binary_img).size == 0:
        return []

    model_path = Path(detector_model_path or DEFAULT_DETECTOR_MODEL_PATH)
    if not model_path.exists():
        return []

    binary = np.asarray(binary_img, dtype=np.uint8)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    try:
        detections = _load_yolo_detector(str(model_path)).predict(
            binary,
            conf=float(detector_confidence),
            imgsz=int(detector_imgsz),
        )
    except Exception:
        return []
    if not detections:
        return []

    boxes: list[BoundingBox] = []
    for detection in sorted(detections, key=lambda item: float(item.confidence), reverse=True):
        x1 = int(round(min(detection.x1, detection.x2)))
        y1 = int(round(min(detection.y1, detection.y2)))
        x2 = int(round(max(detection.x1, detection.x2)))
        y2 = int(round(max(detection.y1, detection.y2)))
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        if width <= 0 or height <= 0:
            continue
        boxes.append(
            BoundingBox(
                x=x1,
                y=y1,
                width=width,
                height=height,
                score=round(float(detection.confidence), 6),
            )
        )
    return boxes


def _start_run_deadline(bot, max_run_duration_seconds=None):
    limit = bot.max_run_duration_seconds if max_run_duration_seconds is None else float(max_run_duration_seconds)
    bot._run_deadline = time.time() + limit


def _ensure_within_run_deadline(bot, context: str):
    deadline = getattr(bot, "_run_deadline", None)
    if deadline and time.time() > deadline:
        hours = getattr(bot, "max_run_duration_seconds", MAX_RUN_DURATION_SECONDS) / 3600.0
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


class RSL_Bot_CursedCity:
    def __init__(self, title_substring="Raid: Shadow Legends", reader=None, window=None, verbose=True, setup=None):
        self.reader = reader
        self.window = window
        self.verbose = verbose
        self.log = logging.getLogger(self.__class__.__name__)
        self.title_substring = title_substring

        self.search_areas = {
            "menu_name": [0.0, 0.02, 0.36, 0.07],
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
            "DifÃ­cil": "hard",
            "Difficult": "hard",
            "Hard Mode": "hard",
            "Modo Normal": "normal",
            "Modo Dificil": "hard",
            "Modo DifÃ­cil": "hard",
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
            "max_random_repositions_when_no_candidates": 6,
            "max_spiral_repositions_when_no_candidates": 6,
            "max_failed_selection_repositions": 3,
            "target_hex": "CEC329",
            "expected_structure_count": 5,
            "detector_model_path": str(DEFAULT_DETECTOR_MODEL_PATH),
            "detector_confidence": DEFAULT_DETECTOR_CONFIDENCE,
            "detector_imgsz": DEFAULT_DETECTOR_IMGSZ,
            "stage_select_delay_seconds": 3.0,
            "stage_start_retries": 3,
            "stage_battle_timeout_seconds": 420.0,
            "stage_battle_poll_interval_seconds": 2.0,
            "stage_battle_outcome_confirm_delay_seconds": 10.0,
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
        self.no_candidate_failures_by_difficulty = getattr(
            self,
            "no_candidate_failures_by_difficulty",
            {"hard": 0, "normal": 0},
        )
        self.spiral_reposition_index = 0
        self.spiral_start_direction_index = random.randrange(4)

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
            normalized.replace("Ã­", "i")
            .replace("Ã¬", "i")
            .replace("Ã¯", "i")
            .replace("Ã¡", "a")
            .replace("Ã©", "e")
            .replace("Ã³", "o")
            .replace("Ãº", "u")
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
                detector_model_path=self.setup.get("detector_model_path", DEFAULT_DETECTOR_MODEL_PATH),
                detector_confidence=float(self.setup.get("detector_confidence", DEFAULT_DETECTOR_CONFIDENCE)),
                detector_imgsz=int(self.setup.get("detector_imgsz", DEFAULT_DETECTOR_IMGSZ)),
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

    def _is_forbidden_encounter_name(self, text: str | None) -> bool:
        normalized = str(text or "").strip().lower()
        return any(name in normalized for name in FORBIDDEN_ENCOUNTER_NAMES)

    def _read_visible_encounter_text(self) -> str | None:
        texts: list[str] = []
        for obj in self._read_text_objects("stage_lower_half_text_scan"):
            text = (getattr(obj, "text", "") or "").strip()
            if text:
                texts.append(text)

        menu_text = (self._read_menu_name() or "").strip()
        if menu_text:
            texts.append(menu_text)

        if not texts:
            return None
        return " ".join(texts).strip() or None

    def _max_spiral_repositions_when_no_candidates(self) -> int:
        return max(0, int(self.setup.get("max_spiral_repositions_when_no_candidates", 6)))

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
            return
        self.no_candidate_failures_by_difficulty[key] = int(
            self.no_candidate_failures_by_difficulty.get(key, 0) or 0
        ) + 1

    def _move_spiral_direction_once(self):
        direction_name = _spiral_direction_for_step(
            self.spiral_reposition_index,
            self.spiral_start_direction_index,
        )
        self.spiral_reposition_index += 1
        move_fn = {
            "up": window_tools.move_up,
            "down": window_tools.move_down,
            "left": window_tools.move_left,
            "right": window_tools.move_right,
        }[direction_name]
        self.log.info("[Cursed City] No candidates. Moving in spiral: %s", direction_name)
        move_fn(self.window, strength=float(self.setup.get("pan_strength", 1.0)))

    def _move_random_direction_once(self):
        self._move_spiral_direction_once()

    def detect_candidates_with_random_reposition(self, difficulty: str | None = None):
        max_moves = self._max_spiral_repositions_when_no_candidates()
        stride = self._spiral_stride_for_difficulty(difficulty)
        moves_done = 0
        while True:
            _ensure_within_run_deadline(self, "detecting Cursed City candidates")
            candidates = self.detect_cursed_city_candidates()
            candidates = self._filter_candidates_against_last_defeat(candidates, difficulty or "")
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
        encounter_text = self._read_visible_encounter_text()
        if self._is_forbidden_encounter_name(encounter_text):
            self.log.info("[Cursed City] Skipping forbidden encounter: %s.", encounter_text)
            window_tools.sendkey("esc", delay=1.0, window=self.window)
            return False
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
