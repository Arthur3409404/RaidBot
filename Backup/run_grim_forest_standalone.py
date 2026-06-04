from __future__ import annotations

import difflib
import json
import logging
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

RAID_WINDOW_TITLE = "Raid: Shadow Legends"
MENU_TITLE_EXPECTED = "Bosque Lugubre"
KEY_DENOMINATOR = 30
EXPECTED_STRUCTURE_COUNT = 5
MIN_DETECTOR_TEMPLATE_COUNT = 1
AVOID_TEMPLATE_STEM = "avoid"
DEFAULT_AVOID_MATCH_THRESHOLD = 0.72
ENABLE_AVOID_FILTER = True
DEFAULT_TEMPLATE_MATCH_THRESHOLD = 0.38
DEFAULT_TOPK_PER_TEMPLATE_SCALE = 24
DEFAULT_PRE_SCORE_CANDIDATE_LIMIT = 320
MIN_ACCEPTED_DETECTOR_SCORE = 0.54
DEFAULT_STAGE_SELECT_DELAY_SECONDS = 3.0
DEFAULT_STAGE_START_RETRIES = 3
DEFAULT_STAGE_BATTLE_TIMEOUT_SECONDS = 420.0
DEFAULT_STAGE_BATTLE_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_STAGE_BATTLE_OUTCOME_CONFIRM_DELAY_SECONDS = 10.0
DEFAULT_MAX_RANDOM_REPOSITIONS_WHEN_NO_CANDIDATES = 3
DEFAULT_PAN_STRENGTH = 1.0
DEFAULT_DIFFICULTY = "hard"
DEBUG_DIR = Path("debug") / "grim_forest_standalone"
GROUP_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6"]
BEST_THRESHOLD_DEFAULTS = {
    "threshold_T1": 0.38292209881325756,
    "threshold_T2": 0.3845197692841958,
    "threshold_T3": 0.375449596703175,
    "threshold_T4": 0.38160037653257334,
    "threshold_T5": 0.38600473941339264,
    "threshold_T6": 0.38305471264914087,
    "threshold_avoid": 0.7144083658862909,
}


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    score: float


class GrimForestStandaloneRunner:
    def __init__(self, title_substring: str = RAID_WINDOW_TITLE, run_detector: bool = True):
        from data.lib.utils import image_tools, map_tools, window_tools

        self.image_tools = image_tools
        self.map_tools = map_tools
        self.window_tools = window_tools
        self.log = logging.getLogger(self.__class__.__name__)
        self.search_areas = {
            "menu_name": [0.008, 0.034, 0.23, 0.037],
            "mode_keys_row": [0.18, 0.02, 0.80, 0.07],
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
        self.target_hex = "CEC329"
        target_rgb = self._hex_to_rgb(f"#{self.target_hex}")
        self.target_bgr_as_rgb = np.array([target_rgb[2], target_rgb[1], target_rgb[0]], dtype=np.uint8)
        self.reference_dir = Path("pic") / "grimforest"
        self.run_detector = bool(run_detector)

        self.reader = self._build_reader()
        self.window = self._resolve_window(title_substring)
        self.debug_dir = self._prepare_debug_dir()
        self.key_counter = None
        self.available_keys = 0
        self.detected_candidates: list[dict] = []
        self.selected_candidate: dict | None = None
        self.selection_succeeded = False
        self.stage_start_status: str | None = None
        self.battle_outcome: str | None = None
        self.post_battle_menu_status: str | None = None
        self.post_battle_stat_choice: str | None = None
        self.current_difficulty = DEFAULT_DIFFICULTY
        self.last_defeat_by_difficulty: dict[str, dict | None] = {"hard": None, "normal": None}
        self.random_reposition_count = 0
        self.exit_reason: str | None = None
        self.completed_battles = 0

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> np.ndarray:
        hex_value = hex_color.lstrip("#")
        r = int(hex_value[0:2], 16)
        g = int(hex_value[2:4], 16)
        b = int(hex_value[4:6], 16)
        return np.array([r, g, b], dtype=np.uint8)

    @staticmethod
    def resembles(text: str | None, target: str, threshold: float = 0.8) -> bool:
        ratio = difflib.SequenceMatcher(None, (text or "").lower(), (target or "").lower()).ratio()
        return ratio >= threshold

    def _build_reader(self):
        import easyocr

        self.log.info("Initializing OCR reader (easyocr, lang='en').")
        return easyocr.Reader(["en"])

    def _resolve_window(self, title_substring: str):
        detected = self.window_tools.find_window(title_substring)
        if not detected:
            raise RuntimeError(f"Raid window not found. Expected title containing: '{title_substring}'.")
        return self.window_tools.WindowObject(detected, title_substring=title_substring)

    def _prepare_debug_dir(self) -> Path:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        session_dir = DEBUG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _read_menu_name(self) -> str | None:
        texts = self.image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            self.search_areas["menu_name"],
            power_detection=False,
        )
        return texts[0].text if texts else None

    def _update_available_keys(self) -> int:
        counter = self.image_tools.read_fraction_counter_in_relative_area(
            self.reader,
            self.window,
            search_area=self.search_areas["mode_keys_row"],
            expected_denominator=KEY_DENOMINATOR,
        )
        self.key_counter = counter
        self.available_keys = int(counter["current"] or 0)
        return self.available_keys

    def _read_text_objects(self, area_key: str):
        return self.image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            search_area=self.search_areas[area_key],
            power_detection=False,
        )

    def _detected_boxes_to_candidates(self, boxes: list[BoundingBox], pov_region) -> list[dict]:
        width = max(1.0, float(pov_region[2] or 1.0))
        height = max(1.0, float(pov_region[3] or 1.0))
        candidates = []
        for index, box in enumerate(boxes, start=1):
            center_local_x = int(box.x + (box.width / 2.0))
            center_local_y = int(box.y + (box.height / 2.0))
            candidates.append(
                {
                    "index": index,
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
        return candidates

    def select_grim_forest_candidate(self, candidate: dict) -> bool:
        click_x = int(candidate["center_abs_x"])
        click_y = int(candidate["center_abs_y"])
        time.sleep(DEFAULT_STAGE_SELECT_DELAY_SECONDS)
        self.log.info(
            "Candidate click #%s at (%s, %s), score=%.4f",
            candidate.get("index"),
            click_x,
            click_y,
            float(candidate.get("score", 0.0) or 0.0),
        )
        self.window_tools.click_at(click_x, click_y, delay=2.5, window=self.window)
        menu_text = self._read_menu_name()
        return not self.resembles(menu_text, MENU_TITLE_EXPECTED, threshold=0.55)

    def _select_detected_candidate(self, detected_boxes: list[BoundingBox], pov_region) -> dict | None:
        return self._select_ranked_candidates(self._detected_boxes_to_candidates(detected_boxes, pov_region))

    def _select_ranked_candidates(self, candidates: list[dict]) -> dict | None:
        self.detected_candidates = list(candidates)
        self.log.info("Ranked %d detected candidate(s) for selection.", len(self.detected_candidates))
        for candidate in self.detected_candidates:
            if self.select_grim_forest_candidate(candidate):
                self.selection_succeeded = True
                self.selected_candidate = candidate
                return candidate
        self.selection_succeeded = False
        self.selected_candidate = None
        return None

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

    def _filter_candidates_against_last_defeat(self, candidates: list[dict], difficulty: str) -> list[dict]:
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
            self.log.info("Skipped %s candidate(s) matching last defeat location.", skipped)
        return filtered

    def _record_last_defeat_candidate(self, difficulty: str, candidate: dict | None):
        key = str(difficulty or "").strip().lower()
        if key in self.last_defeat_by_difficulty and isinstance(candidate, dict):
            self.last_defeat_by_difficulty[key] = dict(candidate)
            self.log.info("Remembered defeat location for '%s' in runtime state.", key)

    def _move_random_direction_once(self):
        moves = [
            ("up", self.window_tools.move_up),
            ("down", self.window_tools.move_down),
            ("left", self.window_tools.move_left),
            ("right", self.window_tools.move_right),
        ]
        direction_name, move_fn = random.choice(moves)
        self.random_reposition_count += 1
        self.log.info("No selectable candidates. Moving randomly: %s.", direction_name)
        move_fn(self.window, strength=DEFAULT_PAN_STRENGTH)

    def _find_start_button_in_lower_half(self):
        for obj in self._read_text_objects("stage_lower_half_text_scan"):
            text = (getattr(obj, "text", "") or "").strip()
            if text and (
                self.resembles(text, "Empezar", threshold=0.6)
                or self.resembles(text, "Iniciar", threshold=0.6)
            ):
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
            self.window_tools.click_center(self.window, rel_square, delay=0.30)

    def click_grim_forest_start_button(self, retries: int = DEFAULT_STAGE_START_RETRIES) -> str:
        for attempt in range(1, max(1, int(retries)) + 1):
            self.log.info("Start button attempt (%s/%s).", attempt, retries)
            box_1_objects = self._read_text_objects("stage_robar_box_1")
            if any(
                self.resembles((getattr(obj, "text", "") or "").strip(), "Robar Campeones", threshold=0.6)
                for obj in box_1_objects
            ):
                self.window_tools.click_center(self.window, self.search_areas["stage_robar_box_1"], delay=2)
                self.window_tools.click_center(self.window, self.search_areas["stage_robar_box_2"], delay=2)

            self._press_pre_start_click_sequence()
            menu_before = (self._read_menu_name() or "").strip()
            start_obj = self._find_start_button_in_lower_half()
            if start_obj is not None:
                adjusted_y = int(start_obj.mean_pos_y) + int(0.10 * self.window.height)
                adjusted_y = max(int(self.window.top), min(adjusted_y, int(self.window.top + self.window.height - 1)))
                self.window_tools.click_at(int(start_obj.mean_pos_x), adjusted_y, delay=1.0, window=self.window)
            else:
                self.window_tools.click_center(
                    self.window,
                    self.search_areas["stage_confirm_button_champion_selection"],
                    delay=1.0,
                )

            time.sleep(5.0)
            menu_after = (self._read_menu_name() or "").strip()
            if menu_before and menu_before == menu_after:
                self.window_tools.sendkey("esc", delay=1.0, window=self.window)
                return "battle_not_started_same_menu"

            try:
                if self.image_tools.check_startup(self):
                    return "battle_started"
            except Exception:
                self.log.debug("Startup validation failed; treating click as started.", exc_info=True)
                return "battle_started"
            time.sleep(0.8)
        return "start_button_not_found_or_not_started"

    def _is_in_game_modes_menu(self, menu_text: str | None) -> bool:
        return bool(
            menu_text
            and (
                self.resembles(menu_text, "Modos de juego", threshold=0.55)
                or self.resembles(menu_text, "Modo de juego", threshold=0.55)
            )
        )

    def _battle_result_text(self) -> str | None:
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

    def get_battle_outcome(
        self,
        timeout_seconds: float = DEFAULT_STAGE_BATTLE_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_STAGE_BATTLE_POLL_INTERVAL_SECONDS,
    ) -> str | None:
        started_at = time.time()
        auto_seen = False
        while (time.time() - started_at) < float(timeout_seconds):
            result = self._battle_result_text()
            if result:
                if result == "Pausa":
                    self.window_tools.sendkey("esc", delay=0.2, window=self.window)
                    time.sleep(max(0.6, float(poll_interval_seconds)))
                    continue
                time.sleep(DEFAULT_STAGE_BATTLE_OUTCOME_CONFIRM_DELAY_SECONDS)
                if self._battle_result_text() == result:
                    return result
            if self._is_auto_battle_visible():
                auto_seen = True
            menu_text = self._read_menu_name()
            if auto_seen and menu_text and self.resembles(menu_text, MENU_TITLE_EXPECTED, threshold=0.55):
                return None
            if self._is_in_game_modes_menu(menu_text):
                return None
            time.sleep(max(0.6, float(poll_interval_seconds)))
        return None

    def return_to_mode_root_after_battle(self, max_attempts: int = 4) -> str:
        for _ in range(max(1, int(max_attempts))):
            menu_text = self._read_menu_name()
            if menu_text and self.resembles(menu_text, MENU_TITLE_EXPECTED, threshold=0.55):
                return "mode"
            if self._is_in_game_modes_menu(menu_text):
                return "game_modes"
            self.window_tools.sendkey("esc", delay=1.2, window=self.window)
        menu_text = self._read_menu_name()
        if menu_text and self.resembles(menu_text, MENU_TITLE_EXPECTED, threshold=0.55):
            return "mode"
        if self._is_in_game_modes_menu(menu_text):
            return "game_modes"
        return "unknown"

    def select_post_battle_stat_reward(self) -> str | None:
        prompt_objects = self._read_text_objects("post_battle_level_prompt")
        if not any("nivel" in (getattr(obj, "text", "") or "").lower() for obj in prompt_objects):
            return None

        self.window_tools.click_center(self.window, self.search_areas["post_battle_level_prompt"], delay=2.0)
        option_objects = self._read_text_objects("post_battle_stat_options")
        selected_text = None
        for preferred_text in ("RES", "VEL", "DEF", "HP"):
            match = next(
                (
                    obj
                    for obj in option_objects
                    if preferred_text in (getattr(obj, "text", "") or "").upper()
                ),
                None,
            )
            if match is not None:
                selected_text = (getattr(match, "text", "") or "").strip()
                self.window_tools.click_at(
                    int(match.mean_pos_x),
                    int(match.mean_pos_y),
                    delay=2.0,
                    window=self.window,
                )
                break

        if selected_text is None:
            self.window_tools.click_center(self.window, self.search_areas["pov"], delay=2.0)
            selected_text = "pov_fallback"

        self.window_tools.click_center(self.window, self.search_areas["post_battle_stat_confirm"], delay=2.0)
        self.log.info("Post-battle stat selection: %s.", selected_text)
        return selected_text

    def _capture_detection_candidates(self) -> list[dict]:
        _, pov_np, pov_region = self.map_tools.capture_relative_area(self.window, self.search_areas["pov"])
        match_bgr_interpretation = np.all(pov_np == self.target_bgr_as_rgb, axis=2)
        cyan_mask = np.where(match_bgr_interpretation, 255, 0).astype(np.uint8)

        base = self.target_bgr_as_rgb.astype(np.int16)
        diff = pov_np.astype(np.int16) - base
        # Allow brighter tones and moderately darker variants of the same base color.
        dark_tolerance = 40
        brighter_tones_mask = (
            (diff[:, :, 0] >= -dark_tolerance)
            & (diff[:, :, 1] >= -dark_tolerance)
            & (diff[:, :, 2] >= -dark_tolerance)
        )
        brighter_tones_mask = np.where(brighter_tones_mask, 255, 0).astype(np.uint8)

        raw_path = self.debug_dir / "pov_raw.png"
        bin_path = self.debug_dir / "pov_cyan_binary.png"
        bright_bin_path = self.debug_dir / "pov_cyan_brighter_binary.png"
        self.map_tools.save_image(raw_path, pov_np)
        self.map_tools.save_image(bin_path, cyan_mask)
        self.map_tools.save_image(bright_bin_path, brighter_tones_mask)

        if not self.run_detector:
            self.log.info("Object detection disabled for this run (dataset collection mode).")
            return []
        if not self.reference_dir.exists():
            self.log.warning("Reference directory missing for grim forest detector: %s", self.reference_dir.as_posix())
            return []

        # IMPORTANT: Keep detector input bound to the brighter-tone binary mask only.
        detected_boxes = detect_grimforest_like_structures(
            binary_image_path=str(bright_bin_path),
            reference_dir=str(self.reference_dir),
            expected_count=EXPECTED_STRUCTURE_COUNT,
            debug=True,
        )
        self.log.info("Detected %d grim-forest-like structures.", len(detected_boxes))
        return self._detected_boxes_to_candidates(detected_boxes, pov_region)

    def detect_candidates_with_random_reposition(self, difficulty: str | None = None) -> list[dict]:
        for attempt in range(DEFAULT_MAX_RANDOM_REPOSITIONS_WHEN_NO_CANDIDATES + 1):
            candidates = self._capture_detection_candidates()
            candidates = self._filter_candidates_against_last_defeat(candidates, difficulty or "")
            if candidates:
                return candidates
            if attempt < DEFAULT_MAX_RANDOM_REPOSITIONS_WHEN_NO_CANDIDATES:
                self._move_random_direction_once()
        return []

    def exit_grim_forest_to_main_menu(self, reason: str) -> bool:
        self.exit_reason = reason
        self.log.info("Exiting to game modes. Reason: %s.", reason)
        for _ in range(3):
            menu_text = self._read_menu_name()
            if self._is_in_game_modes_menu(menu_text):
                return True
            self.window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"], delay=1.8)
        return self._is_in_game_modes_menu(self._read_menu_name())

    def _capture_pov_and_binarize(self) -> str:
        if not self.run_detector:
            self._capture_detection_candidates()
            return "detector_disabled"

        candidates = self.detect_candidates_with_random_reposition(difficulty=self.current_difficulty)
        if not candidates:
            self.exit_grim_forest_to_main_menu(reason="no_candidates_detected_after_random_repositions")
            return "no_candidates"

        selected = self._select_ranked_candidates(candidates)
        if selected is None:
            self.log.warning("No detected Grim Forest candidate could be selected.")
            return "no_candidate_selected"

        self.log.info("Selected candidate #%s.", selected.get("index"))
        self.stage_start_status = self.click_grim_forest_start_button()
        self.log.info("Stage start status: %s.", self.stage_start_status)
        if self.stage_start_status != "battle_started":
            self.exit_grim_forest_to_main_menu(reason=self.stage_start_status or "stage_start_failed")
            return "stage_start_failed"

        self.battle_outcome = self.get_battle_outcome()
        self.log.info("Battle outcome: %s.", self.battle_outcome or "unknown")
        self.post_battle_menu_status = self.return_to_mode_root_after_battle()
        self.log.info("Post-battle menu status: %s.", self.post_battle_menu_status)
        self.post_battle_stat_choice = self.select_post_battle_stat_reward()
        self.completed_battles += 1

        if self.battle_outcome == "Derrota":
            self._record_last_defeat_candidate(self.current_difficulty, selected)
            self.exit_grim_forest_to_main_menu(reason="battle_lost")
            return "defeat"
        if self.battle_outcome != "Victoria":
            self.exit_grim_forest_to_main_menu(reason="battle_outcome_unknown_or_timeout")
            return "unknown_outcome"
        if self.post_battle_menu_status != "mode":
            self.exit_grim_forest_to_main_menu(reason="not_in_grim_forest_after_victory")
            return "left_mode"
        return "victory"

    def _write_run_metadata(self, menu_text: str | None):
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_detector": bool(self.run_detector),
            "menu_expected": MENU_TITLE_EXPECTED,
            "menu_detected": menu_text,
            "menu_check_passed": bool(self.resembles(menu_text, MENU_TITLE_EXPECTED)),
            "keys_expected_denominator": KEY_DENOMINATOR,
            "keys_current": int(self.available_keys or 0),
            "key_counter_raw": self.key_counter,
            "detected_candidate_count": len(self.detected_candidates),
            "selected_candidate": self.selected_candidate,
            "selection_succeeded": bool(self.selection_succeeded),
            "stage_start_status": self.stage_start_status,
            "battle_outcome": self.battle_outcome,
            "post_battle_menu_status": self.post_battle_menu_status,
            "post_battle_stat_choice": self.post_battle_stat_choice,
            "current_difficulty": self.current_difficulty,
            "last_defeat_by_difficulty": self.last_defeat_by_difficulty,
            "random_reposition_count": int(self.random_reposition_count),
            "exit_reason": self.exit_reason,
            "completed_battles": int(self.completed_battles),
            "label": {"true_object_count": None, "notes": ""},
        }
        meta_path = self.debug_dir / "run_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.log.info("Saved run metadata: %s", meta_path.as_posix())

    def run(self):
        menu_text = self._read_menu_name()
        self.log.info("Detected menu name: %s", menu_text if menu_text else "<none>")
        if not self.resembles(menu_text, MENU_TITLE_EXPECTED, threshold=0.55):
            self.log.warning(
                "Menu OCR mismatch. Expected '%s', got '%s'. Continuing run anyway.",
                MENU_TITLE_EXPECTED,
                menu_text,
            )

        while True:
            self._update_available_keys()
            self.log.info("Detected keys: %s/%s", self.available_keys, KEY_DENOMINATOR)
            self.log.info("Key OCR payload: %s", self.key_counter)
            if self.available_keys <= 0:
                self.exit_grim_forest_to_main_menu(reason="no_keys_remaining")
                break
            cycle_status = self._capture_pov_and_binarize()
            if cycle_status != "victory":
                break
            self.log.info("Victory completed; starting another Grim Forest detection cycle.")
        self._write_run_metadata(menu_text=menu_text)


def _load_binary_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path.as_posix()}")
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return bw


def _load_grayscale_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path.as_posix()}")
    return img


def _largest_component_bbox(binary_img: np.ndarray) -> tuple[int, int, int, int] | None:
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
    return x, y, w, h


def _crop_to_bbox(binary_img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return binary_img[y : y + h, x : x + w]


def _dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = (mask_a > 0).astype(np.uint8)
    b = (mask_b > 0).astype(np.uint8)
    inter = int(np.count_nonzero((a == 1) & (b == 1)))
    size = int(np.count_nonzero(a)) + int(np.count_nonzero(b))
    if size == 0:
        return 0.0
    return float((2.0 * inter) / size)


def _extract_largest_component_rois(gray_img: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    _, bw = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    bbox = _largest_component_bbox(bw)
    if bbox is None:
        return None
    roi_bw = _crop_to_bbox(bw, bbox)
    roi_gray = _crop_to_bbox(gray_img, bbox)
    h, w = roi_bw.shape
    if h <= 0 or w <= 0:
        return None
    return roi_bw, roi_gray


def _prepare_reference_stats(reference_dir: Path, target_size: int = 64) -> dict:
    templates = []
    raw_templates = []
    raw_templates_gray = []
    aspect_values = []
    fill_values = []
    paths = sorted(reference_dir.glob("*.png"))
    if not paths:
        raise RuntimeError(f"No reference templates found in: {reference_dir.as_posix()}")

    avoid_path = next((p for p in paths if p.stem.strip().lower() == AVOID_TEMPLATE_STEM), None)
    detector_paths = [p for p in paths if p != avoid_path]
    if len(detector_paths) < MIN_DETECTOR_TEMPLATE_COUNT:
        raise RuntimeError(
            f"Expected at least {MIN_DETECTOR_TEMPLATE_COUNT} detector template(s) in: {reference_dir.as_posix()} "
            f"(excluding '{(avoid_path.name if avoid_path else f'{AVOID_TEMPLATE_STEM}.png')}'). Found: {len(detector_paths)}"
        )

    for path in detector_paths:
        ref_gray = _load_grayscale_image(path)
        rois = _extract_largest_component_rois(ref_gray)
        if rois is None:
            continue
        ref_roi, ref_roi_gray = rois
        h, w = ref_roi.shape
        resized = cv2.resize(ref_roi, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        templates.append(resized)
        raw_templates.append(ref_roi)
        raw_templates_gray.append(ref_roi_gray)
        aspect_values.append(float(w) / float(h))
        fill_values.append(float(np.count_nonzero(ref_roi)) / float(max(1, w * h)))

    if len(templates) < MIN_DETECTOR_TEMPLATE_COUNT:
        raise RuntimeError(
            f"Loaded {len(templates)} detector templates from: {reference_dir.as_posix()}, "
            f"but expected at least {MIN_DETECTOR_TEMPLATE_COUNT}."
        )

    avoid_resized = None
    if ENABLE_AVOID_FILTER:
        if avoid_path is None:
            raise RuntimeError(
                f"Avoid filtering is enabled, but '{AVOID_TEMPLATE_STEM}.png' was not found in: {reference_dir.as_posix()}"
            )
        avoid_gray = _load_grayscale_image(avoid_path)
        avoid_rois = _extract_largest_component_rois(avoid_gray)
        if avoid_rois is None:
            raise RuntimeError(f"Avoid template is empty or invalid: {avoid_path.as_posix()}")
        avoid_roi_bw, _ = avoid_rois
        avoid_resized = cv2.resize(avoid_roi_bw, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

    return {
        "templates": templates,
        "raw_templates": raw_templates,
        "raw_templates_gray": raw_templates_gray,
        "aspect_mean": float(np.mean(aspect_values)),
        "aspect_std": float(np.std(aspect_values) + 1e-6),
        "fill_mean": float(np.mean(fill_values)),
        "fill_std": float(np.std(fill_values) + 1e-6),
        "template_count": len(templates),
        "avoid_template": avoid_resized,
        "avoid_template_path": avoid_path.as_posix() if avoid_path is not None else None,
        "target_size": int(target_size),
    }


def _classify_template_group(path: Path) -> str | None:
    stem = path.stem.strip().lower()
    if stem == AVOID_TEMPLATE_STEM or stem.startswith(f"{AVOID_TEMPLATE_STEM}_"):
        return AVOID_TEMPLATE_STEM
    for group in GROUP_ORDER:
        group_lower = group.lower()
        if stem == group_lower or stem.startswith(f"{group_lower}_"):
            return group
    return None


def _has_grouped_templates(reference_dir: Path) -> bool:
    grouped = {group: 0 for group in GROUP_ORDER}
    grouped[AVOID_TEMPLATE_STEM] = 0
    for path in sorted(reference_dir.glob("*.png")):
        group = _classify_template_group(path)
        if group is None:
            continue
        grouped[group] += 1
    return all(grouped[group] > 0 for group in GROUP_ORDER) and grouped[AVOID_TEMPLATE_STEM] > 0


def _resolve_grouped_reference_dir(reference_dir: Path) -> Path:
    if _has_grouped_templates(reference_dir):
        return reference_dir
    fallback = reference_dir.parent / "grimforest_test"
    if fallback != reference_dir and _has_grouped_templates(fallback):
        return fallback
    raise RuntimeError(
        "Grouped Grim Forest templates were not found. Expected T1..T6 and avoid templates in "
        f"'{reference_dir.as_posix()}' (or fallback '{fallback.as_posix()}')."
    )


def _load_grouped_templates(reference_dir: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {group: [] for group in GROUP_ORDER}
    grouped[AVOID_TEMPLATE_STEM] = []
    for path in sorted(reference_dir.glob("*.png")):
        group = _classify_template_group(path)
        if group is None:
            continue
        img = _load_grayscale_image(path)
        _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        grouped[group].append({"name": path.name, "image": bw})

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


def _select_peak_points(
    result: np.ndarray,
    min_score: float,
    topk: int,
    *,
    peak_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    mask = result >= float(min_score)
    if not bool(np.any(mask)):
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    if bool(peak_only):
        # Keep only local maxima to reduce near-duplicate hits before NMS.
        dilated = cv2.dilate(result, np.ones((3, 3), dtype=np.float32))
        peak_mask = mask & (result >= dilated)
        ys, xs = np.where(peak_mask)
        if len(ys) == 0:
            ys, xs = np.where(mask)
    else:
        ys, xs = np.where(mask)

    topk = int(topk)
    if topk > 0 and len(ys) > topk:
        values = result[ys, xs]
        idx = np.argpartition(values, -topk)[-topk:]
        ys = ys[idx]
        xs = xs[idx]
    return ys.astype(np.int32), xs.astype(np.int32)


def _template_match_candidates(
    binary_img: np.ndarray,
    ref_stats: dict,
    *,
    match_threshold: float = DEFAULT_TEMPLATE_MATCH_THRESHOLD,
    topk_per_template_scale: int = DEFAULT_TOPK_PER_TEMPLATE_SCALE,
    peak_only_per_template_scale: bool = True,
) -> list[dict]:
    candidates = []
    binary_float = binary_img.astype(np.float32)
    scales = np.linspace(1.0, 3.2, 12)
    raw_templates_gray = ref_stats.get("raw_templates_gray", [])
    raw_templates = ref_stats.get("raw_templates", [])
    if len(raw_templates_gray) != len(raw_templates):
        raw_templates_gray = raw_templates

    for template_id, ref_gray in enumerate(raw_templates_gray):
        for scale in scales:
            tw = max(8, int(ref_gray.shape[1] * float(scale)))
            th = max(8, int(ref_gray.shape[0] * float(scale)))
            if tw >= binary_img.shape[1] or th >= binary_img.shape[0]:
                continue
            template_gray = cv2.resize(ref_gray, (tw, th), interpolation=cv2.INTER_AREA)
            template_float = template_gray.astype(np.float32)
            if float(np.std(template_float)) < 1e-6:
                continue
            result = cv2.matchTemplate(binary_float, template_float, cv2.TM_CCOEFF_NORMED)
            ys, xs = _select_peak_points(
                result,
                min_score=match_threshold,
                topk=topk_per_template_scale,
                peak_only=peak_only_per_template_scale,
            )
            for y, x in zip(ys, xs):
                x = int(x)
                y = int(y)
                roi = binary_img[y : y + th, x : x + tw]
                if roi.shape[0] != th or roi.shape[1] != tw:
                    continue
                fg_count = float(np.count_nonzero(roi))
                candidates.append(
                    {
                        "x": x,
                        "y": y,
                        "w": tw,
                        "h": th,
                        "area": fg_count,
                        "bbox_area": float(tw * th),
                        "fill": fg_count / float(max(1, tw * th)),
                        "aspect": float(tw) / float(th),
                        "circularity": 0.0,
                        "roi": roi,
                        "template_score_raw": float(result[y, x]),
                        "template_id": int(template_id),
                        "scale": float(scale),
                    }
                )
    return candidates


def _score_candidate(candidate: dict, ref_stats: dict, target_size: int = 64) -> float:
    roi = candidate["roi"]
    resized = cv2.resize(roi, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    template_score = max(_dice_score(resized, tmpl) for tmpl in ref_stats["templates"])

    aspect_z = abs(candidate["aspect"] - ref_stats["aspect_mean"]) / max(ref_stats["aspect_std"], 0.02)
    fill_z = abs(candidate["fill"] - ref_stats["fill_mean"]) / max(ref_stats["fill_std"], 0.03)
    aspect_score = float(np.exp(-0.5 * aspect_z))
    fill_score = float(np.exp(-0.5 * fill_z))

    rectangularity_score = float(np.clip(candidate["fill"] / max(ref_stats["fill_mean"], 1e-6), 0.0, 1.0))
    total = (0.62 * template_score) + (0.18 * aspect_score) + (0.14 * fill_score) + (0.06 * rectangularity_score)
    return float(total)


def _bbox_iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (a["w"] * a["h"]) + (b["w"] * b["h"]) - inter
    return float(inter / union) if union > 0 else 0.0


def _nms(sorted_candidates: list[dict], iou_thresh: float = 0.35) -> list[dict]:
    kept = []
    for cand in sorted_candidates:
        if all(_bbox_iou(cand, prev) <= iou_thresh for prev in kept):
            kept.append(cand)
    return kept


def _boxes_intersect(a: dict, b: dict) -> bool:
    ax1, ay1, ax2, ay2 = int(a["x"]), int(a["y"]), int(a["x"] + a["w"]), int(a["y"] + a["h"])
    bx1, by1, bx2, by2 = int(b["x"]), int(b["y"]), int(b["x"] + b["w"]), int(b["y"] + b["h"])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return (ix2 - ix1) > 0 and (iy2 - iy1) > 0


def _avoid_match_score(binary_img: np.ndarray, candidate: dict, ref_stats: dict) -> float:
    avoid_template = ref_stats.get("avoid_template")
    if avoid_template is None:
        return 0.0
    x, y = int(candidate["x"]), int(candidate["y"])
    w, h = int(candidate["w"]), int(candidate["h"])
    if w <= 0 or h <= 0:
        return 0.0
    roi = binary_img[y : y + h, x : x + w]
    if roi.shape[0] != h or roi.shape[1] != w:
        return 0.0
    target_size = int(ref_stats.get("target_size", 64))
    resized = cv2.resize(roi, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return float(_dice_score(resized, avoid_template))


def detect_grimforest_like_structures(
    binary_image_path: str,
    reference_dir: str,
    expected_count: int = 4,
    debug: bool = True,
    detector_params: dict | None = None,
) -> list[BoundingBox]:
    # IMPORTANT: Keep detector input bound to the brighter-tone binary mask only.
    # Future tuning/adaptation should use `pov_cyan_brighter_binary.png` as the
    # source image for this function, not `pov_raw.png` or `pov_cyan_binary.png`.
    image_path = Path(binary_image_path)
    if not image_path.exists():
        alt = image_path.with_name(image_path.name.replace("poiv_", "pov_"))
        if alt.exists():
            image_path = alt
    bw = _load_binary_image(image_path)
    detector_params = dict(detector_params or {})

    debug_rows = []

    match_threshold = float(detector_params.get("template_match_threshold", DEFAULT_TEMPLATE_MATCH_THRESHOLD))
    topk_per_template_scale = int(detector_params.get("topk_per_template_scale", DEFAULT_TOPK_PER_TEMPLATE_SCALE))
    pre_score_candidate_limit = int(detector_params.get("pre_score_candidate_limit", DEFAULT_PRE_SCORE_CANDIDATE_LIMIT))
    peak_only_per_template_scale = bool(detector_params.get("peak_only_per_template_scale", True))
    template_candidates = _template_match_candidates(
        bw,
        ref_stats,
        match_threshold=match_threshold,
        topk_per_template_scale=topk_per_template_scale,
        peak_only_per_template_scale=peak_only_per_template_scale,
    )
    if pre_score_candidate_limit > 0 and len(template_candidates) > pre_score_candidate_limit:
        template_candidates.sort(key=lambda c: float(c.get("template_score_raw", 0.0)), reverse=True)
        template_candidates = template_candidates[:pre_score_candidate_limit]
    for cand in template_candidates:
        cand["score"] = 0.55 * float(cand.get("template_score_raw", 0.0)) + 0.45 * _score_candidate(cand, ref_stats)

    template_candidates.sort(key=lambda c: c["score"], reverse=True)
    template_candidates = _nms(template_candidates, iou_thresh=0.25)
    template_top = template_candidates[:expected_count]
    template_count = len(template_top)
    template_mean = float(np.mean([c["score"] for c in template_top])) if template_top else 0.0
    template_quality = template_mean - (abs(expected_count - template_count) * 0.18)
    template_row = {
        "method": "template_matching_fast_exclusive",
        "component_count_raw": len(template_candidates),
        "top_count": template_count,
        "top_mean_score": round(template_mean, 5),
        "quality": round(template_quality, 5),
    }
    debug_rows.append(template_row)
    best = {
        "quality": template_quality,
        "method": "template_matching_fast_exclusive",
        "params": {
            "threshold": round(match_threshold, 4),
            "peak_only_per_template_scale": bool(peak_only_per_template_scale),
            "topk_per_template_scale": int(topk_per_template_scale),
            "pre_score_candidate_limit": int(pre_score_candidate_limit),
            "iou_thresh": 0.25,
            "scales": "1.0..3.2x12",
        },
        "profile": {},
        "processed": bw,
        "candidates": template_candidates,
        "top": template_top,
        "debug_row": template_row,
    }

    max_objects = int(detector_params.get("max_objects", expected_count))
    min_score = max(
        MIN_ACCEPTED_DETECTOR_SCORE,
        float(detector_params.get("min_score", MIN_ACCEPTED_DETECTOR_SCORE)),
    )
    min_template_score_raw = float(detector_params.get("min_template_score_raw", -1.0))
    avoid_match_threshold = float(detector_params.get("avoid_match_threshold", DEFAULT_AVOID_MATCH_THRESHOLD))
    # Hard constraint: detected objects must not overlap.
    enforce_no_overlap = True
    candidate_pool = list(best["candidates"])
    eligible = [
        c
        for c in candidate_pool
        if float(c.get("score", 0.0)) >= min_score
        and float(c.get("template_score_raw", 1.0)) >= min_template_score_raw
    ]
    eligible.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
    selected = []
    avoid_rejected = []
    for group in GROUP_ORDER:
        accepted = []
        for cand in group_candidates_before_avoid[group]:
            if ENABLE_AVOID_FILTER and any(_boxes_intersect(cand, bad) for bad in avoid_candidates):
                avoid_rejected.append(cand)
                continue
            accepted.append(cand)
        group_candidates_after_avoid[group] = accepted

    selected: list[dict] = []
    for group in GROUP_ORDER:
        selected.extend(group_candidates_after_avoid[group])
    if max_objects > 0 and len(selected) > max_objects:
        selected = selected[: max(0, max_objects)]

    boxes = [
        BoundingBox(
            x=int(c["x"]),
            y=int(c["y"]),
            width=int(c["w"]),
            height=int(c["h"]),
            score=round(float(c["score"]), 6),
        )
        for c in selected
    ]

    output_img = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    for box in boxes:
        cv2.rectangle(
            output_img,
            (box.x, box.y),
            (box.x + box.width, box.y + box.height),
            color=(0, 0, 255),
            thickness=2,
        )

    detected_path = image_path.with_name(f"{image_path.stem}_detected_structures.png")
    cv2.imwrite(str(detected_path), output_img)

    results = {
        "input_image": image_path.as_posix(),
        "reference_dir": Path(reference_dir).as_posix(),
        "resolved_reference_dir": resolved_reference_dir.as_posix(),
        "expected_count": int(expected_count),
        "selected_method": "grouped_template_matching_T1_T6",
        "detector_params": {
            "max_objects": max_objects,
            "thresholds": thresholds,
            "topk_per_template_scale": int(topk_per_template),
            "nms_iou_thresh": float(nms_iou_thresh),
            "avoid_filter_enabled": ENABLE_AVOID_FILTER,
            "group_order": GROUP_ORDER,
        },
        "detected_count": len(boxes),
        "avoid_rejected_count": len(avoid_rejected),
        "boxes": [asdict(box) for box in boxes],
    }
    results_path = image_path.with_name(f"{image_path.stem}_detected_structures.json")
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if debug:
        top_candidates = sorted(
            selected,
            key=lambda c: (
                -float(c.get("score", 0.0)),
                int(c.get("y", 0)),
                int(c.get("x", 0)),
                str(c.get("template_name", "")),
            ),
        )
        debug_payload = {
            "selected": {
                "method": "grouped_template_matching_T1_T6",
                "detected_count": len(selected),
                "expected_count": int(expected_count),
            },
            "group_counts_before_avoid": {
                group: len(group_candidates_before_avoid[group]) for group in GROUP_ORDER
            },
            "group_counts_after_avoid": {
                group: len(group_candidates_after_avoid[group]) for group in GROUP_ORDER
            },
            "avoid_count": len(avoid_candidates),
            "top_candidates": [
                {
                    "rank": int(i + 1),
                    "x": int(c["x"]),
                    "y": int(c["y"]),
                    "width": int(c["w"]),
                    "height": int(c["h"]),
                    "score": round(float(c["score"]), 6),
                    "template_score_raw": round(float(c.get("template_score_raw", 0.0)), 6),
                    "group": str(c.get("group", "")),
                    "template_name": str(c.get("template_name", "")),
                }
                for i, c in enumerate(top_candidates[: max(20, expected_count + 5)])
            ],
            "avoid_rejected_candidates": [
                {
                    "x": int(c["x"]),
                    "y": int(c["y"]),
                    "width": int(c["w"]),
                    "height": int(c["h"]),
                    "score": round(float(c.get("score", 0.0)), 6),
                    "group": str(c.get("group", "")),
                    "template_name": str(c.get("template_name", "")),
                }
                for c in avoid_rejected
            ],
            "avoid_candidates": [
                {
                    "x": int(c["x"]),
                    "y": int(c["y"]),
                    "width": int(c["w"]),
                    "height": int(c["h"]),
                    "score": round(float(c.get("score", 0.0)), 6),
                    "template_name": str(c.get("template_name", "")),
                }
                for c in avoid_candidates[: max(20, expected_count + 5)]
            ],
        }
        debug_json_path = image_path.with_name(f"{image_path.stem}_detector_debug.json")
        debug_json_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")

        debug_vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        selected_keys = {(int(c["x"]), int(c["y"]), int(c["w"]), int(c["h"])) for c in selected}
        all_candidates_for_vis = []
        for group in GROUP_ORDER:
            all_candidates_for_vis.extend(group_candidates_before_avoid[group])
        for cand in all_candidates_for_vis[: max(20, expected_count + 5)]:
            x, y, ww, hh = int(cand["x"]), int(cand["y"]), int(cand["w"]), int(cand["h"])
            key = (x, y, ww, hh)
            color = (0, 0, 255) if key in selected_keys else (255, 0, 0)
            thickness = 2 if key in selected_keys else 1
            cv2.rectangle(debug_vis, (x, y), (x + ww, y + hh), color, thickness)
        debug_vis_path = image_path.with_name(f"{image_path.stem}_detector_debug_candidates.png")
        cv2.imwrite(str(debug_vis_path), debug_vis)

    return boxes


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        GrimForestStandaloneRunner().run()
        return 0
    except Exception as exc:
        logging.error("Grim Forest standalone runner failed: %s", exc)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
