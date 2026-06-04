from __future__ import annotations

import difflib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

EXPECTED_CONDA_ENV = "RaidEnv"
_ENV_RELAUNCH_MARKER = "RAID_CC_STANDALONE_ENV_RELAUNCH_ATTEMPTED"
RAID_WINDOW_TITLE = "Raid: Shadow Legends"
MENU_TITLE_EXPECTED = "Ciudad Maldita"
KEY_DENOMINATOR = 8
DEBUG_DIR = Path("debug") / "cursed_city_standalone"
RUN_STATE_FILE = Path("data") / "tmp" / "cursed_city_standalone_state.json"
LAST_DEFEAT_FILE = Path("data") / "tmp" / "cursed_city_last_defeat.json"
EXPECTED_STRUCTURE_COUNT = 5
CURSED_CITY_DETECTOR_PARAMS = {"max_objects": 12, "min_score": 0.52, "min_template_score_raw": 0.0}


def _is_running_in_expected_env(expected_env_name: str = EXPECTED_CONDA_ENV) -> bool:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env == expected_env_name:
        return True
    return os.path.basename(sys.prefix).lower() == expected_env_name.lower()


def _resolve_conda_executable() -> str | None:
    candidates = []
    conda_exe_env = os.environ.get("CONDA_EXE")
    if conda_exe_env:
        candidates.append(conda_exe_env)

    conda_from_path = shutil.which("conda")
    if conda_from_path:
        candidates.append(conda_from_path)

    user_profile = os.environ.get("USERPROFILE", "")
    if user_profile:
        candidates.extend(
            [
                os.path.join(user_profile, "anaconda3", "Scripts", "conda.exe"),
                os.path.join(user_profile, "anaconda3", "condabin", "conda.bat"),
                os.path.join(user_profile, "miniconda3", "Scripts", "conda.exe"),
                os.path.join(user_profile, "miniconda3", "condabin", "conda.bat"),
            ]
        )

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _ensure_expected_conda_env() -> None:
    if _is_running_in_expected_env():
        return

    if os.environ.get(_ENV_RELAUNCH_MARKER) == "1":
        raise RuntimeError(
            f"Unable to start inside '{EXPECTED_CONDA_ENV}'. "
            "Please activate the environment manually and rerun."
        )

    conda_exe = _resolve_conda_executable()
    if not conda_exe:
        raise RuntimeError(
            "Conda was not found. Add Conda to PATH, set CONDA_EXE, or run this script "
            f"from the '{EXPECTED_CONDA_ENV}' environment."
        )

    relaunch_env = os.environ.copy()
    relaunch_env[_ENV_RELAUNCH_MARKER] = "1"

    launch_target = os.path.abspath(__file__)
    cmd = [
        conda_exe,
        "run",
        "--no-capture-output",
        "-n",
        EXPECTED_CONDA_ENV,
        "python",
        launch_target,
        *sys.argv[1:],
    ]

    use_shell = conda_exe.lower().endswith(".bat")
    if use_shell:
        cmd = subprocess.list2cmdline(cmd)
    result = subprocess.run(cmd, env=relaunch_env, check=False, shell=use_shell)
    sys.exit(result.returncode)


_ensure_expected_conda_env()

import cv2
import numpy as np


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    score: float


class CursedCityStandaloneRunner:
    def __init__(
        self,
        title_substring: str = RAID_WINDOW_TITLE,
        run_detector: bool = True,
        debug_enabled: bool = False,
    ):
        from data.lib.utils import image_tools, map_tools, window_tools

        self.image_tools = image_tools
        self.map_tools = map_tools
        self.window_tools = window_tools
        self.log = logging.getLogger(self.__class__.__name__)
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
        }
        # Exact color matching for AutoHotkey PixelGetColor value CEC329 (BGR interpretation).
        self.target_hex = "CEC329"
        target_rgb = self._hex_to_rgb(f"#{self.target_hex}")
        self.target_bgr_as_rgb = np.array(
            [target_rgb[2], target_rgb[1], target_rgb[0]],
            dtype=np.uint8,
        )
        self.reference_dir = Path("pic") / "cursedcity"
        self.run_detector = bool(run_detector)
        self.debug_enabled = bool(debug_enabled)

        self.reader = self._build_reader()
        self.window = self._resolve_window(title_substring)
        self.debug_dir = self._prepare_debug_dir() if self.debug_enabled else None
        self.key_counter = None
        self.available_keys = 0
        self.running = True
        self.candidate_detection_retries_per_view = 2
        self.max_random_repositions_when_no_candidates = 3
        self.current_difficulty = None
        self.current_run_difficulty = None
        self.pre_start_click_sequence = [
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
        ]

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> np.ndarray:
        hex_value = hex_color.lstrip("#")
        if len(hex_value) != 6:
            raise ValueError(f"Invalid HEX color '{hex_color}'. Expected format '#RRGGBB'.")
        r = int(hex_value[0:2], 16)
        g = int(hex_value[2:4], 16)
        b = int(hex_value[4:6], 16)
        return np.array([r, g, b], dtype=np.uint8)

    def _build_reader(self):
        import easyocr

        self.log.info("Initializing OCR reader (easyocr, lang='en').")
        return easyocr.Reader(["en"])

    def _resolve_window(self, title_substring: str):
        detected = self.window_tools.find_window(title_substring)
        if not detected:
            raise RuntimeError(
                f"Raid window not found. Expected title containing: '{title_substring}'."
            )
        return self.window_tools.WindowObject(detected, title_substring=title_substring)

    def _prepare_debug_dir(self) -> Path:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        session_dir = DEBUG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    @staticmethod
    def resembles(text: str | None, target: str, threshold: float = 0.8) -> bool:
        ratio = difflib.SequenceMatcher(
            None, (text or "").lower(), (target or "").lower()
        ).ratio()
        return ratio >= threshold

    def _read_menu_name(self) -> str | None:
        texts = self.image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            self.search_areas["menu_name"],
            power_detection=False,
        )
        return texts[0].text if texts else None

    def _load_last_defeat_state(self) -> dict:
        default = {"hard": None, "normal": None}
        try:
            if not LAST_DEFEAT_FILE.exists():
                return dict(default)
            payload = json.loads(LAST_DEFEAT_FILE.read_text(encoding="utf-8"))
            state = dict(default)
            for key in ("hard", "normal"):
                value = payload.get(key)
                state[key] = value if isinstance(value, dict) else None
            return state
        except Exception:
            return dict(default)

    def _save_last_defeat_state(self, state: dict):
        try:
            LAST_DEFEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
            LAST_DEFEAT_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
        except Exception as exc:
            self.log.warning("[Cursed City] Failed to persist last defeat state: %s", exc)

    def _record_last_defeat_candidate(self, difficulty: str, candidate: dict):
        difficulty_key = str(difficulty or "").strip().lower()
        if difficulty_key not in {"hard", "normal"}:
            return
        state = self._load_last_defeat_state()
        state[difficulty_key] = {
            "center_rel_x": float(candidate.get("center_rel_x", 0.0) or 0.0),
            "center_rel_y": float(candidate.get("center_rel_y", 0.0) or 0.0),
            "bbox_rel": {
                "x": float((candidate.get("bbox_rel") or {}).get("x", 0.0) or 0.0),
                "y": float((candidate.get("bbox_rel") or {}).get("y", 0.0) or 0.0),
                "width": float((candidate.get("bbox_rel") or {}).get("width", 0.0) or 0.0),
                "height": float((candidate.get("bbox_rel") or {}).get("height", 0.0) or 0.0),
            },
            "score": float(candidate.get("score", 0.0) or 0.0),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._save_last_defeat_state(state)
        self.log.info(
            "[Cursed City] Stored last defeat location for '%s': (%.4f, %.4f).",
            difficulty_key,
            float(state[difficulty_key]["center_rel_x"]),
            float(state[difficulty_key]["center_rel_y"]),
        )

    @staticmethod
    def _bbox_overlap_ratio(a: dict, b: dict) -> float:
        try:
            ax1 = float(a.get("x", 0.0) or 0.0)
            ay1 = float(a.get("y", 0.0) or 0.0)
            ax2 = ax1 + float(a.get("width", 0.0) or 0.0)
            ay2 = ay1 + float(a.get("height", 0.0) or 0.0)

            bx1 = float(b.get("x", 0.0) or 0.0)
            by1 = float(b.get("y", 0.0) or 0.0)
            bx2 = bx1 + float(b.get("width", 0.0) or 0.0)
            by2 = by1 + float(b.get("height", 0.0) or 0.0)
        except Exception:
            return 0.0

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0

        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
        denom = min(area_a, area_b)
        if denom <= 0.0:
            return 0.0

        # Overlap normalized by the smaller area so near-same locations
        # still match even if sizes vary slightly.
        return float(inter / denom)

    def _filter_candidates_against_last_defeat(self, candidates: list[dict], difficulty: str):
        difficulty_key = str(difficulty or "").strip().lower()
        if difficulty_key not in {"hard", "normal"}:
            return list(candidates)

        state = self._load_last_defeat_state()
        defeat_entry = state.get(difficulty_key)
        if not isinstance(defeat_entry, dict):
            return list(candidates)

        defeat_bbox = defeat_entry.get("bbox_rel")
        if not isinstance(defeat_bbox, dict):
            return list(candidates)

        filtered = []
        skipped = 0
        for candidate in candidates:
            candidate_bbox = candidate.get("bbox_rel")
            if not isinstance(candidate_bbox, dict):
                filtered.append(candidate)
                continue

            overlap_ratio = self._bbox_overlap_ratio(defeat_bbox, candidate_bbox)
            if overlap_ratio >= 0.50:
                skipped += 1
                continue
            filtered.append(candidate)

        if skipped > 0:
            self.log.info(
                "[Cursed City] Skipped %s candidate(s) matching last defeat location for '%s'.",
                skipped,
                difficulty_key,
            )
        return filtered

    def _load_run_state(self) -> dict:
        default = {"run_counter": 0}
        try:
            if not RUN_STATE_FILE.exists():
                return dict(default)
            payload = json.loads(RUN_STATE_FILE.read_text(encoding="utf-8"))
            run_counter = int(payload.get("run_counter", 0) or 0)
            return {"run_counter": run_counter}
        except Exception:
            return dict(default)

    def _save_run_state(self, state: dict):
        try:
            RUN_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            RUN_STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
        except Exception as exc:
            self.log.warning("[Cursed City] Failed to persist run state: %s", exc)

    def _plan_and_commit_run_difficulty(self) -> str:
        state = self._load_run_state()
        run_counter = int(state.get("run_counter", 0) or 0) + 1
        planned = "hard" if (run_counter % 2 == 1) else "normal"
        self._save_run_state(
            {
                "run_counter": run_counter,
                "last_used_difficulty": planned,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self.log.info(
            "[Cursed City] Run #%s planned difficulty: %s.",
            run_counter,
            planned,
        )
        return planned

    def set_difficulty(self, set_level=None):
        try:
            difficulty = self.image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                search_area=self.search_areas["mode_difficulty_current"],
            )[0]
        except Exception:
            difficulty = None

        if difficulty and getattr(difficulty, "text", None) in self.translation_mapping:
            self.current_difficulty = self.translation_mapping[difficulty.text]

        if set_level and self.current_difficulty != set_level:
            switch_key = f"mode_difficulty_switch_{set_level}"
            self.window_tools.click_center(self.window, self.search_areas["mode_difficulty_current"])
            self.window_tools.click_center(self.window, self.search_areas[switch_key], delay=5)
            self.current_difficulty = set_level

    def _capture_pov_and_binarize_cyan(self):
        _, pov_np, pov_region = self.map_tools.capture_relative_area(self.window, self.search_areas["pov"])
        match_bgr_interpretation = np.all(pov_np == self.target_bgr_as_rgb, axis=2)
        cyan_mask = np.where(match_bgr_interpretation, 255, 0).astype(np.uint8)

        # Separate mask for brighter tones of the same base color (up to very bright).
        base = self.target_bgr_as_rgb.astype(np.int16)
        diff = pov_np.astype(np.int16) - base
        brighter_tones_mask = (
            (diff[:, :, 0] >= 0)
            & (diff[:, :, 1] >= 0)
            & (diff[:, :, 2] >= 0)
        )
        brighter_tones_mask = np.where(brighter_tones_mask, 255, 0).astype(np.uint8)

        raw_path = None
        bin_path = None
        bright_bin_path = None
        if self.debug_enabled and self.debug_dir:
            raw_path = self.debug_dir / "pov_raw.png"
            bin_path = self.debug_dir / "pov_cyan_binary.png"
            bright_bin_path = self.debug_dir / "pov_cyan_brighter_binary.png"
            self.map_tools.save_image(raw_path, pov_np)
            self.map_tools.save_image(bin_path, cyan_mask)
            self.map_tools.save_image(bright_bin_path, brighter_tones_mask)

        white_pixels = int(np.count_nonzero(cyan_mask))
        total_pixels = int(cyan_mask.size)
        fill_ratio = (white_pixels / total_pixels) if total_pixels else 0.0
        bright_white_pixels = int(np.count_nonzero(brighter_tones_mask))
        bright_fill_ratio = (bright_white_pixels / total_pixels) if total_pixels else 0.0

        if self.debug_enabled and raw_path and bin_path and bright_bin_path:
            self.log.info("Saved raw POV image: %s", raw_path.as_posix())
            self.log.info("Saved cyan binary image: %s", bin_path.as_posix())
            self.log.info("Saved brighter-tone binary image: %s", bright_bin_path.as_posix())
        self.log.info(
            "Exact color match enabled for HEX %s (BGR interpretation). BGR as RGB=%s",
            self.target_hex,
            self.target_bgr_as_rgb.tolist(),
        )
        self.log.info(
            "Cyan coverage: %d / %d pixels (%.4f)",
            white_pixels,
            total_pixels,
            fill_ratio,
        )
        self.log.info(
            "Brighter-tone coverage: %d / %d pixels (%.4f)",
            bright_white_pixels,
            total_pixels,
            bright_fill_ratio,
        )

        if self.run_detector:
            temp_path = None
            detector_input_path = bright_bin_path
            if detector_input_path is None:
                fd, tmp_name = tempfile.mkstemp(prefix="cursed_city_mask_", suffix=".png")
                os.close(fd)
                temp_path = Path(tmp_name)
                cv2.imwrite(str(temp_path), brighter_tones_mask)
                detector_input_path = temp_path
            try:
                detected_boxes = detect_cursedcity_like_structures(
                    binary_image_path=str(detector_input_path),
                    reference_dir=str(self.reference_dir),
                    expected_count=EXPECTED_STRUCTURE_COUNT,
                    debug=self.debug_enabled,
                    detector_params=CURSED_CITY_DETECTOR_PARAMS,
                    save_artifacts=self.debug_enabled,
                )
            finally:
                if temp_path and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
            self.log.info("Detected %d cursed-city-like structures.", len(detected_boxes))
            return detected_boxes, pov_region
        else:
            self.log.info("Object detection disabled for this run (dataset collection mode).")
            return [], pov_region

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

    def _read_text_objects(self, area_key: str, power_detection=False):
        try:
            return self.image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                search_area=self.search_areas[area_key],
                power_detection=power_detection,
            )
        except Exception:
            return []

    def _is_in_game_modes_menu(self, menu_text: str | None) -> bool:
        if not menu_text:
            return False
        return self.resembles(menu_text, "Modos de juego", threshold=0.55) or self.resembles(
            menu_text, "Modo de juego", threshold=0.55
        )

    def is_in_cursed_city_mode(self) -> bool:
        menu_text = self._read_menu_name()
        return bool(menu_text and self.resembles(menu_text, MENU_TITLE_EXPECTED, threshold=0.55))

    def has_cursed_city_keys_remaining(self, retries: int = 3) -> bool:
        for attempt in range(1, max(1, int(retries)) + 1):
            try:
                keys = int(self._update_available_keys())
            except Exception:
                keys = 0
            self.log.info("[Cursed City] Key check (%s/%s): %s/%s", attempt, retries, keys, KEY_DENOMINATOR)
            if keys > 0:
                return True
            time.sleep(0.6)
        return False

    def detect_cursed_city_candidates(self, retries: int = 2) -> list[dict]:
        retries = max(1, int(retries))
        for attempt in range(1, retries + 1):
            boxes, pov_region = self._capture_pov_and_binarize_cyan()
            pov_width = max(1.0, float(pov_region[2] or 1.0))
            pov_height = max(1.0, float(pov_region[3] or 1.0))
            candidates = []
            for idx, box in enumerate(boxes, start=1):
                center_local_x = int(box.x + (box.width / 2.0))
                center_local_y = int(box.y + (box.height / 2.0))
                candidates.append(
                    {
                        "index": idx,
                        "score": float(getattr(box, "score", 0.0) or 0.0),
                        "center_local_x": center_local_x,
                        "center_local_y": center_local_y,
                        "center_rel_x": float(center_local_x / pov_width),
                        "center_rel_y": float(center_local_y / pov_height),
                        "bbox_rel": {
                            "x": float(max(0.0, box.x) / pov_width),
                            "y": float(max(0.0, box.y) / pov_height),
                            "width": float(max(0.0, box.width) / pov_width),
                            "height": float(max(0.0, box.height) / pov_height),
                        },
                        # `pov_region` is already absolute screen coordinates.
                        "center_abs_x": int(pov_region[0] + center_local_x),
                        "center_abs_y": int(pov_region[1] + center_local_y),
                    }
                )
            candidates.sort(key=lambda c: c["score"], reverse=True)
            self.log.info(
                "[Cursed City] Candidate detection (%s/%s): %s candidates.",
                attempt,
                retries,
                len(candidates),
            )
            if candidates:
                return candidates
            time.sleep(0.8)
        return []

    def _move_random_direction_once(self):
        directions = [
            ("up", self.window_tools.move_up),
            ("down", self.window_tools.move_down),
            ("left", self.window_tools.move_left),
            ("right", self.window_tools.move_right),
        ]
        direction_name, direction_fn = random.choice(directions)
        self.log.info("[Cursed City] No candidates. Moving randomly: %s", direction_name)
        direction_fn(self.window, strength=1.0)

    def detect_candidates_with_random_reposition(self, difficulty: str | None = None):
        max_moves = max(0, int(self.max_random_repositions_when_no_candidates))
        retries_per_view = max(1, int(self.candidate_detection_retries_per_view))

        for wave_index in range(max_moves + 1):
            candidates = self.detect_cursed_city_candidates(retries=retries_per_view)
            candidates = self._filter_candidates_against_last_defeat(candidates, difficulty or "")
            if candidates:
                return candidates
            if wave_index < max_moves:
                self._move_random_direction_once()

        return []

    def select_cursed_city_candidate(self, candidate: dict) -> bool:
        click_x = int(candidate["center_abs_x"])
        click_y = int(candidate["center_abs_y"])
        self.log.info("[Cursed City] Waiting 3.0s before clicking detected candidate.")
        time.sleep(3.0)
        self.log.info(
            "[Cursed City] Candidate click #%s at (%s, %s), score=%.4f",
            candidate.get("index"),
            click_x,
            click_y,
            float(candidate.get("score", 0.0) or 0.0),
        )
        self.window_tools.click_at(click_x, click_y, delay=2.5, window=self.window)

        still_in_mode = self.is_in_cursed_city_mode()
        self.log.info("[Cursed City] 'Ciudad Maldita' still visible after click: %s", still_in_mode)
        return not still_in_mode

    def _find_empezar_button_in_lower_half(self):
        for obj in self._read_text_objects("stage_lower_half_text_scan"):
            text = (getattr(obj, "text", "") or "").strip()
            if not text:
                continue
            if self.resembles(text, "Empezar", threshold=0.6) or self.resembles(text, "Iniciar", threshold=0.6):
                return obj
        return None

    def _press_pre_start_click_sequence(self):
        for index, rel_square in enumerate(self.pre_start_click_sequence, start=1):
            self.log.info("[Cursed City] Pre-start click %s/%s", index, len(self.pre_start_click_sequence))
            self.window_tools.click_center(
                self.window,
                rel_square,
                delay=0.30,
            )

    def click_cursed_city_start_button(self, retries: int = 3) -> str:
        retries = max(1, int(retries))
        for attempt in range(1, retries + 1):
            self.log.info("[Cursed City] Start button attempt (%s/%s).", attempt, retries)
            self._press_pre_start_click_sequence()
            menu_name_before_start = (self._read_menu_name() or "").strip()
            empezar_obj = self._find_empezar_button_in_lower_half()
            if empezar_obj is not None:
                y_offset = int(0.10 * self.window.height)
                adjusted_y = int(empezar_obj.mean_pos_y) + y_offset
                min_y = int(self.window.top)
                max_y = int(self.window.top + self.window.height - 1)
                adjusted_y = max(min_y, min(adjusted_y, max_y))
                self.window_tools.click_at(
                    int(empezar_obj.mean_pos_x),
                    adjusted_y,
                    delay=1.0,
                    window=self.window,
                )
                self.log.info(
                    "[Cursed City] Empezar click adjusted by +10%% screen height (y: %s -> %s).",
                    int(empezar_obj.mean_pos_y),
                    adjusted_y,
                )
            else:
                self.window_tools.click_center(
                    self.window,
                    self.search_areas["stage_confirm_button_champion_selection"],
                    delay=1.0,
                )

            # Explicit start validation requested by user:
            # if menu name is unchanged 5s after start click, battle never started.
            time.sleep(5.0)
            menu_name_after_start = (self._read_menu_name() or "").strip()
            if menu_name_before_start and (menu_name_before_start == menu_name_after_start):
                self.log.info(
                    "[Cursed City] Start validation failed: menu unchanged after 5s ('%s').",
                    menu_name_after_start,
                )
                self.window_tools.sendkey("esc", delay=1.0, window=self.window)
                return "battle_not_started_same_menu"

            # Reuse existing startup verification style from other modules.
            try:
                started = bool(self.image_tools.check_startup(self))
            except Exception:
                started = True
            if started:
                self.log.info("[Cursed City] Start button click successful.")
                return "battle_started"
            time.sleep(0.8)
        return "start_button_not_found_or_not_started"

    def _battle_result_text(self):
        for area_key in ("stage_battle_result", "stage_battle_result_2"):
            for text_object in self._read_text_objects(area_key):
                text = (getattr(text_object, "text", "") or "").strip()
                if not text:
                    continue
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
        auto_objects = self._read_text_objects("stage_auto_battle_button")
        if not auto_objects:
            return False
        text = (getattr(auto_objects[0], "text", "") or "").strip()
        return bool(text and self.resembles(text, "Auto", threshold=0.7))

    def get_battle_outcome(self, timeout_seconds: float = 420.0, poll_interval_seconds: float = 2.0):
        started_at = time.time()
        auto_seen = False
        while (time.time() - started_at) < float(timeout_seconds):
            result = self._battle_result_text()
            if result:
                if result == "Pausa":
                    self.window_tools.sendkey("esc", delay=0.2, window=self.window)
                    time.sleep(max(0.6, poll_interval_seconds))
                    continue
                # Confirm outcome twice, same style as other modules.
                time.sleep(10.0)
                second = self._battle_result_text()
                if second == result:
                    return result
            if self._is_auto_battle_visible():
                auto_seen = True

            menu_text = self._read_menu_name()
            if auto_seen and menu_text and self.resembles(menu_text, MENU_TITLE_EXPECTED, threshold=0.55):
                return None
            if self._is_in_game_modes_menu(menu_text):
                return None
            time.sleep(max(0.6, poll_interval_seconds))
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

    def exit_cursed_city_to_main_menu(self, reason: str):
        self.log.info("[Cursed City] Exiting to main menu. Reason: %s", reason)
        for _ in range(3):
            menu_text = self._read_menu_name()
            if self._is_in_game_modes_menu(menu_text):
                return True
            self.window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"], delay=1.8)
        return self._is_in_game_modes_menu(self._read_menu_name())

    def run(self):
        menu_text = self._read_menu_name()
        self.log.info("Detected menu name: %s", menu_text if menu_text else "<none>")
        if not self.resembles(menu_text, MENU_TITLE_EXPECTED, threshold=0.55):
            raise RuntimeError(
                f"Not in Cursed City mode. Expected '{MENU_TITLE_EXPECTED}', got '{menu_text}'."
            )
        planned_difficulty = self._plan_and_commit_run_difficulty()
        self.set_difficulty(planned_difficulty)
        self.current_run_difficulty = planned_difficulty
        if self.current_difficulty == planned_difficulty:
            self.log.info("[Cursed City] Difficulty set to '%s' for this run.", planned_difficulty)
        else:
            self.log.warning(
                "[Cursed City] Could not confirm difficulty switch to '%s'. Continuing anyway.",
                planned_difficulty,
            )

        failed_candidate_selection_cycles = 0
        max_failed_selection_repositions = max(1, int(self.max_random_repositions_when_no_candidates))

        while self.running:
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
                if self.select_cursed_city_candidate(candidate):
                    selected = candidate
                    break
            if selected is None:
                # Candidate(s) existed but none opened a valid stage.
                # If we are still in Cursed City, recover with random movement instead of exiting immediately.
                if self.is_in_cursed_city_mode():
                    failed_candidate_selection_cycles += 1
                    self.log.info(
                        "[Cursed City] No valid candidate selected on this view. Random reposition retry %s/%s.",
                        failed_candidate_selection_cycles,
                        max_failed_selection_repositions,
                    )
                    if failed_candidate_selection_cycles <= max_failed_selection_repositions:
                        self._move_random_direction_once()
                        continue
                    self.exit_cursed_city_to_main_menu(
                        reason="no_valid_candidate_selected_after_random_repositions"
                    )
                    break

                # If we are no longer in Cursed City here, exit cleanly.
                self.exit_cursed_city_to_main_menu(reason="left_cursed_city_after_candidate_click")
                break

            # Reset failed candidate cycle count once a valid stage is opened.
            failed_candidate_selection_cycles = 0

            start_status = self.click_cursed_city_start_button(retries=3)
            if start_status != "battle_started":
                if start_status == "battle_not_started_same_menu" and self.is_in_cursed_city_mode():
                    failed_candidate_selection_cycles += 1
                    self.log.info(
                        "[Cursed City] Battle did not start. Random reposition retry %s/%s.",
                        failed_candidate_selection_cycles,
                        max_failed_selection_repositions,
                    )
                    if failed_candidate_selection_cycles <= max_failed_selection_repositions:
                        self._move_random_direction_once()
                        continue
                    self.exit_cursed_city_to_main_menu(
                        reason="battle_not_started_after_random_repositions"
                    )
                    break

                self.exit_cursed_city_to_main_menu(reason="start_button_not_found")
                break

            outcome = self.get_battle_outcome()
            self.log.info("[Cursed City] Battle outcome: %s", outcome if outcome else "unknown")

            menu_status = self.return_to_mode_root_after_battle(max_attempts=4)
            if menu_status == "game_modes":
                self.log.info("[Cursed City] Returned to game modes after battle.")
                break
            if menu_status == "unknown":
                self.exit_cursed_city_to_main_menu(reason="unknown_menu_after_battle")
                break

            if outcome == "Derrota":
                self._record_last_defeat_candidate(
                    difficulty=(self.current_run_difficulty or self.current_difficulty or ""),
                    candidate=selected,
                )
                self.exit_cursed_city_to_main_menu(reason="battle_lost")
                break
            if outcome != "Victoria":
                self.exit_cursed_city_to_main_menu(reason="battle_outcome_unknown_or_timeout")
                break

        if self.debug_enabled:
            self._write_run_metadata(menu_text=self._read_menu_name())

    def _write_run_metadata(self, menu_text: str | None):
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_detector": bool(self.run_detector),
            "current_run_difficulty": self.current_run_difficulty,
            "menu_expected": MENU_TITLE_EXPECTED,
            "menu_detected": menu_text,
            "menu_check_passed": bool(self.resembles(menu_text, MENU_TITLE_EXPECTED)),
            "keys_expected_denominator": KEY_DENOMINATOR,
            "keys_current": int(self.available_keys or 0),
            "key_counter_raw": self.key_counter,
            "label": {
                "true_object_count": None,
                "notes": "",
            },
        }
        meta_path = self.debug_dir / "run_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.log.info("Saved run metadata: %s", meta_path.as_posix())


def _load_binary_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path.as_posix()}")
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return bw


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


def _prepare_reference_stats(reference_dir: Path, target_size: int = 64) -> dict:
    templates = []
    raw_templates = []
    aspect_values = []
    fill_values = []
    paths = sorted(reference_dir.glob("*.png"))
    if not paths:
        raise RuntimeError(f"No reference templates found in: {reference_dir.as_posix()}")

    for path in paths:
        ref_bw = _load_binary_image(path)
        bbox = _largest_component_bbox(ref_bw)
        if bbox is None:
            continue
        ref_roi = _crop_to_bbox(ref_bw, bbox)
        h, w = ref_roi.shape
        if h <= 0 or w <= 0:
            continue
        resized = cv2.resize(ref_roi, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        templates.append(resized)
        raw_templates.append(ref_roi)
        aspect_values.append(float(w) / float(h))
        fill_values.append(float(np.count_nonzero(ref_roi)) / float(max(1, w * h)))

    if not templates:
        raise RuntimeError(f"Reference templates are empty in: {reference_dir.as_posix()}")

    return {
        "templates": templates,
        "raw_templates": raw_templates,
        "aspect_mean": float(np.mean(aspect_values)),
        "aspect_std": float(np.std(aspect_values) + 1e-6),
        "fill_mean": float(np.mean(fill_values)),
        "fill_std": float(np.std(fill_values) + 1e-6),
        "template_count": len(templates),
    }


def _template_match_candidates(binary_img: np.ndarray, ref_stats: dict) -> list[dict]:
    candidates = []
    scales = np.linspace(1.0, 3.2, 12)
    for template_id, ref in enumerate(ref_stats.get("raw_templates", [])):
        for scale in scales:
            tw = max(8, int(ref.shape[1] * float(scale)))
            th = max(8, int(ref.shape[0] * float(scale)))
            if tw >= binary_img.shape[1] or th >= binary_img.shape[0]:
                continue
            template = cv2.resize(ref, (tw, th), interpolation=cv2.INTER_NEAREST)
            result = cv2.matchTemplate(binary_img, template, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(result >= 0.38)
            for y, x in zip(ys, xs):
                x = int(x)
                y = int(y)
                roi = binary_img[y : y + th, x : x + tw]
                if roi.shape[0] != th or roi.shape[1] != tw:
                    continue
                candidates.append(
                    {
                        "x": x,
                        "y": y,
                        "w": tw,
                        "h": th,
                        "area": float(np.count_nonzero(roi)),
                        "bbox_area": float(tw * th),
                        "fill": float(np.count_nonzero(roi)) / float(max(1, tw * th)),
                        "aspect": float(tw) / float(th),
                        "circularity": 0.0,
                        "roi": roi,
                        "template_score_raw": float(result[y, x]),
                        "template_id": int(template_id),
                        "scale": float(scale),
                    }
                )
    return candidates


def _component_candidates(binary_img: np.ndarray) -> list[dict]:
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        area = float(cv2.contourArea(contour))
        bbox_area = float(w * h)
        if bbox_area <= 0:
            continue
        roi = binary_img[y : y + h, x : x + w]
        fg_area = float(np.count_nonzero(roi))
        fill = fg_area / bbox_area
        aspect = float(w) / float(h)
        perimeter = float(cv2.arcLength(contour, True))
        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter)) if perimeter > 0 else 0.0
        candidates.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": area,
                "bbox_area": bbox_area,
                "fill": fill,
                "aspect": aspect,
                "circularity": circularity,
                "roi": roi,
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


def detect_cursedcity_like_structures(
    binary_image_path: str,
    reference_dir: str,
    expected_count: int = 4,
    debug: bool = True,
    detector_params: dict | None = None,
    save_artifacts: bool = True,
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
    ref_stats = _prepare_reference_stats(Path(reference_dir))
    detector_params = dict(detector_params or {})

    h, w = bw.shape
    img_area = float(h * w)
    debug_rows = []
    best = None

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
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (params["open"], params["open"]))
            proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, k)
        if params["close"] > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (params["close"], params["close"]))
            proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, k)
        if params["dilate"] > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            proc = cv2.dilate(proc, k, iterations=int(params["dilate"]))

        all_components = _component_candidates(proc)
        for profile in size_profiles:
            min_area = profile["min_area_ratio"] * img_area
            max_area = profile["max_area_ratio"] * img_area
            filtered = []
            rejected = {"size": 0, "fill": 0, "aspect": 0}

            for cand in all_components:
                if not (min_area <= cand["area"] <= max_area):
                    rejected["size"] += 1
                    continue
                if not (profile["min_fill"] <= cand["fill"] <= profile["max_fill"]):
                    rejected["fill"] += 1
                    continue
                if not (profile["aspect_lo"] <= cand["aspect"] <= profile["aspect_hi"]):
                    rejected["aspect"] += 1
                    continue
                cand = dict(cand)
                cand["score"] = _score_candidate(cand, ref_stats)
                filtered.append(cand)

            filtered.sort(key=lambda c: c["score"], reverse=True)
            filtered = _nms(filtered, iou_thresh=0.35)

            top = filtered[:expected_count]
            count = len(top)
            mean_score = float(np.mean([c["score"] for c in top])) if top else 0.0
            count_penalty = abs(expected_count - count) * 0.18
            quality = mean_score - count_penalty

            row = {
                "method": "components",
                "params": params,
                "profile": profile,
                "component_count_raw": len(all_components),
                "component_count_filtered": len(filtered),
                "rejected": rejected,
                "top_count": count,
                "top_mean_score": round(mean_score, 5),
                "quality": round(quality, 5),
            }
            debug_rows.append(row)

            if best is None or quality > best["quality"]:
                best = {
                    "quality": quality,
                    "method": "components",
                    "params": params,
                    "profile": profile,
                    "processed": proc,
                    "candidates": filtered,
                    "top": top,
                    "debug_row": row,
                }

    template_candidates = _template_match_candidates(bw, ref_stats)
    for cand in template_candidates:
        cand["score"] = 0.55 * float(cand.get("template_score_raw", 0.0)) + 0.45 * _score_candidate(cand, ref_stats)

    template_candidates.sort(key=lambda c: c["score"], reverse=True)
    template_candidates = _nms(template_candidates, iou_thresh=0.25)
    template_top = template_candidates[:expected_count]
    template_count = len(template_top)
    template_mean = float(np.mean([c["score"] for c in template_top])) if template_top else 0.0
    template_quality = template_mean - (abs(expected_count - template_count) * 0.18)
    template_row = {
        "method": "template_matching",
        "component_count_raw": len(template_candidates),
        "top_count": template_count,
        "top_mean_score": round(template_mean, 5),
        "quality": round(template_quality, 5),
    }
    debug_rows.append(template_row)
    if best is None or template_quality > best["quality"]:
        best = {
            "quality": template_quality,
            "method": "template_matching",
            "params": {"threshold": 0.38, "iou_thresh": 0.25, "scales": "1.0..3.2x12"},
            "profile": {},
            "processed": bw,
            "candidates": template_candidates,
            "top": template_top,
            "debug_row": template_row,
        }

    if best is None:
        return []

    max_objects = int(detector_params.get("max_objects", expected_count))
    min_score = float(detector_params.get("min_score", 0.0))
    min_template_score_raw = float(detector_params.get("min_template_score_raw", -1.0))
    candidate_pool = list(best["candidates"])
    selected = [
        c
        for c in candidate_pool
        if float(c.get("score", 0.0)) >= min_score
        and float(c.get("template_score_raw", 1.0)) >= min_template_score_raw
    ]
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

    # Red boxes need a 3-channel canvas.
    output_img = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    for box in boxes:
        cv2.rectangle(
            output_img,
            (box.x, box.y),
            (box.x + box.width, box.y + box.height),
            color=(0, 0, 255),
            thickness=2,
        )

    if save_artifacts:
        detected_path = image_path.with_name(f"{image_path.stem}_detected_structures.png")
        cv2.imwrite(str(detected_path), output_img)

        results = {
            "input_image": image_path.as_posix(),
            "reference_dir": Path(reference_dir).as_posix(),
            "expected_count": int(expected_count),
            "selected_method": best.get("method"),
            "detector_params": {
                "max_objects": max_objects,
                "min_score": min_score,
                "min_template_score_raw": min_template_score_raw,
            },
            "selected_pipeline": best["debug_row"],
            "detected_count": len(boxes),
            "boxes": [asdict(box) for box in boxes],
        }
        results_path = image_path.with_name(f"{image_path.stem}_detected_structures.json")
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if debug and save_artifacts:
        debug_payload = {
            "selected": best["debug_row"],
            "all_attempts": debug_rows,
            "top_candidates": [
                {
                    "rank": int(i + 1),
                    "x": int(c["x"]),
                    "y": int(c["y"]),
                    "width": int(c["w"]),
                    "height": int(c["h"]),
                    "score": round(float(c["score"]), 6),
                    "template_score_raw": round(float(c.get("template_score_raw", 0.0)), 6),
                    "area": round(float(c.get("area", 0.0)), 3),
                    "fill": round(float(c.get("fill", 0.0)), 6),
                    "aspect": round(float(c.get("aspect", 0.0)), 6),
                }
                for i, c in enumerate(best["candidates"][: max(20, expected_count + 5)])
            ],
        }
        debug_json_path = image_path.with_name(f"{image_path.stem}_detector_debug.json")
        debug_json_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")

        debug_vis = cv2.cvtColor(best["processed"], cv2.COLOR_GRAY2BGR)
        # Show selected candidates in red, nearby extras in blue for missing-object inspection.
        selected_keys = {(int(c["x"]), int(c["y"]), int(c["w"]), int(c["h"])) for c in selected}
        for cand in best["candidates"][: max(20, expected_count + 5)]:
            x, y, ww, hh = int(cand["x"]), int(cand["y"]), int(cand["w"]), int(cand["h"])
            key = (x, y, ww, hh)
            color = (0, 0, 255) if key in selected_keys else (255, 0, 0)
            thickness = 2 if key in selected_keys else 1
            cv2.rectangle(debug_vis, (x, y), (x + ww, y + hh), color, thickness)
        debug_vis_path = image_path.with_name(f"{image_path.stem}_detector_debug_candidates.png")
        cv2.imwrite(str(debug_vis_path), debug_vis)

    return boxes


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        CursedCityStandaloneRunner().run()
        return 0
    except Exception as exc:
        logging.error("Cursed City standalone runner failed: %s", exc)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
