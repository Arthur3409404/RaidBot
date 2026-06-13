# -*- coding: utf-8 -*-
"""Hydra mode module for integrated RaidBot execution."""

from __future__ import annotations

import difflib
import logging
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import raid_bot.utils.auto_battle_tools as auto_battle_tools
import raid_bot.utils.image_tools as image_tools
import raid_bot.utils.window_tools as window_tools

MAX_RUN_DURATION_SECONDS = int(3.5 * 60 * 60)

HYDRA_SEARCH_AREAS = {
    "menu_name": [0.008, 0.034, 0.23, 0.06],
    "go_to_higher_menu": [0.928, 0.031, 0.046, 0.039],
    "main_menu_labels": [0.007, 0.27, 0.984, 0.044],
    "clanboss_Hydra": [0.445, 0.307, 0.11, 0.196],
    "Hydra_Keys": [0.72, 0.038, 0.048, 0.035],
    "hydra_entry_search": [0.2, 0.18, 0.6, 0.5],
    "hydra_difficulty_list": [0.55, 0.12, 0.42, 0.74],
    "Hydra_Hard": [0.599, 0.380, 0.384, 0.105],
    "Hydra_Brutal": [0.599, 0.496, 0.384, 0.105],
    "Hydra_Nightmare": [0.599, 0.618, 0.384, 0.105],
    "Hydra_NameList": [0.103, 0.136, 0.173, 0.846],
    "hydra_start_battle": [0.78, 0.85, 0.13, 0.1],
    "hydra_finished": [0.37, 0.13, 0.25, 0.08],
    "hydra_score": [0.2, 0.27, 0.6, 0.08],
    "hydra_retry_battle": [0.65, 0.92, 0.12, 0.05],
    "hydra_save_battle": [0.8, 0.92, 0.12, 0.05],
}


def resembles(text: str, target: str, threshold: float = 0.8) -> bool:
    ratio = difflib.SequenceMatcher(None, (text or "").lower(), target.lower()).ratio()
    return ratio >= threshold


def _start_run_deadline(bot, max_run_duration_seconds=None):
    limit = (
        bot.max_run_duration_seconds
        if max_run_duration_seconds is None
        else float(max_run_duration_seconds)
    )
    bot._run_deadline = time.time() + limit


def _ensure_within_run_deadline(bot, context: str):
    deadline = getattr(bot, "_run_deadline", None)
    if deadline and time.time() > deadline:
        hours = getattr(bot, "max_run_duration_seconds", MAX_RUN_DURATION_SECONDS) / 3600.0
        raise TimeoutError(
            f"{bot.__class__.__name__} exceeded max runtime of {hours:.1f}h while {context}."
        )


class RSL_Bot_Hydra:
    DIFFICULTY_ALIASES = {
        "Nightmare": ["Nightmare", "NM", "Pesadilla"],
        "Brutal": ["Brutal"],
        "Hard": ["Hard", "Dificil"],
    }

    def __init__(
        self,
        title_substring="Raid: Shadow Legends",
        reader=None,
        window=None,
        verbose=True,
        player_names=None,
        difficulty_order=None,
        thresholds=None,
        manual_play_enabled=False,
        manual_play_difficulties=None,
        manual_profile_name="Hydra_InfinityTeam",
        search_areas=None,
    ):
        self.reader = reader
        self.window = window
        self.running = True
        self.main_loop_running = True
        self.verbose = bool(verbose)
        self.log = logging.getLogger(self.__class__.__name__)

        self.search_areas = dict(HYDRA_SEARCH_AREAS)
        if search_areas:
            self.search_areas.update(dict(search_areas))

        self.player_names = list(player_names or [])
        self.max_run_duration_seconds = MAX_RUN_DURATION_SECONDS
        self._run_deadline = None

        self.difficulty_order = [self.normalize_difficulty(item) for item in (difficulty_order or [])]
        if not self.difficulty_order:
            self.difficulty_order = ["Nightmare", "Brutal", "Hard"]

        raw_thresholds = thresholds or {}
        self.thresholds = {}
        for key, value in raw_thresholds.items():
            self.thresholds[self.normalize_difficulty(key)] = float(value)
        for difficulty in self.difficulty_order:
            self.thresholds.setdefault(difficulty, 0.0)

        self.num_of_keys = 0
        self.hydra_encounters_cleared = []
        self.hydra_encounter_difficulty = None
        self.battle_status = None
        self.lost_encounter = False

        self.manual_play_enabled = bool(manual_play_enabled)
        self.manual_player = None
        self.manual_profile_name = str(manual_profile_name or "Hydra_InfinityTeam")
        if manual_play_difficulties is None:
            self.manual_play_difficulties = ["Hard"]
        elif isinstance(manual_play_difficulties, str):
            self.manual_play_difficulties = [manual_play_difficulties]
        else:
            self.manual_play_difficulties = list(manual_play_difficulties)
        self.auto_button_clicked = False
        self.search_areas.setdefault("auto_battle_button", [0.026, 0.899, 0.058, 0.07])

        if self.manual_play_enabled:
            try:
                from raid_bot.utils.manual_run_tools import ManualRunPlayer

                self.manual_player = ManualRunPlayer(
                    profile_name=self.manual_profile_name,
                    reader=self.reader,
                    window=self.window,
                    image_tools=image_tools,
                    window_tools=window_tools,
                    search_areas=self.search_areas,
                    logger=self.log,
                )
                self.log.info("Manual run initialized with profile '%s'.", self.manual_profile_name)
            except Exception as exc:
                self.log.warning("Manual run initialization skipped: %s", exc)
                self.manual_player = None
        else:
            self.log.info("Manual run disabled by configuration.")

    def check_if_wednesday_berlin(self):
        return datetime.now(ZoneInfo("Europe/Berlin")).weekday() == 2

    def _normalize_text(self, text):
        return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

    def normalize_difficulty(self, value):
        normalized = self._normalize_text(str(value))
        for canonical, aliases in self.DIFFICULTY_ALIASES.items():
            for alias in aliases:
                alias_normalized = self._normalize_text(alias)
                if (
                    normalized == alias_normalized
                    or normalized in alias_normalized
                    or alias_normalized in normalized
                ):
                    return canonical
        return str(value)

    def _matches_player_name(self, text, threshold=0.75):
        normalized = self._normalize_text(text)
        for player_name in self.player_names:
            target = self._normalize_text(player_name)
            if not normalized or not target:
                continue
            if normalized == target or normalized in target or target in normalized:
                return True
            if difflib.SequenceMatcher(None, normalized, target).ratio() >= threshold:
                return True
        return False

    def _threshold_for_difficulty(self, difficulty):
        canonical = self.normalize_difficulty(difficulty)
        return float(self.thresholds.get(canonical, 0.0)) * 1_000_000.0

    def update_available_keys(self):
        try:
            keys_text = image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                search_area=self.search_areas["Hydra_Keys"],
            )[0].text
            self.num_of_keys = int(re.findall(r"\d+", keys_text)[0])
        except Exception:
            self.num_of_keys = 0
        self.log.info("Hydra keys detected: %s", self.num_of_keys)

    def detect_cleared_difficulties(self, max_attempts=3):
        self.hydra_encounters_cleared = []
        for difficulty in self.difficulty_order:
            if not self._click_hydra_difficulty(difficulty):
                continue

            window_tools.move_up(self.window, strength=3, relative_x=0.25)
            found = False

            for _ in range(max_attempts):
                name_strings = image_tools.get_text_in_relative_area(
                    self.reader,
                    self.window,
                    search_area=self.search_areas["Hydra_NameList"],
                )
                if any(self._matches_player_name(name.text) for name in name_strings):
                    self.hydra_encounters_cleared.append(difficulty)
                    found = True
                    break
                window_tools.move_down(self.window, strength=0.5, relative_x=0.25)

            self.log.info("Hydra '%s' cleared_by_name_scan=%s", difficulty, found)

    def select_next_difficulty(self):
        for difficulty in self.difficulty_order:
            if difficulty not in self.hydra_encounters_cleared:
                self.hydra_encounter_difficulty = difficulty
                return difficulty
        self.hydra_encounter_difficulty = None
        return None

    def _click_hydra_difficulty(self, difficulty, repetitions=3, delay_seconds=2.0):
        difficulty = self.normalize_difficulty(difficulty)
        key = f"Hydra_{difficulty}"
        if key not in self.search_areas:
            self.log.warning("Missing search area for difficulty '%s'.", key)
            return False

        for _ in range(max(1, int(repetitions))):
            window_tools.click_center(
                self.window,
                self.search_areas[key],
                delay=float(delay_seconds),
            )
        return True

    def _parse_score_value(self, text):
        matches = re.findall(r"(\d[\d.,]*)([a-zA-Z]*)", text or "")
        if not matches:
            return 0.0
        number_part, suffix = matches[-1]
        number_part = number_part.replace(".", "").replace(",", ".").replace(" ", "")
        try:
            value = float(number_part)
        except ValueError:
            return 0.0
        suffix = suffix.lower()
        if suffix.startswith("k"):
            return value * 1_000.0
        if suffix.startswith("m"):
            return value * 1_000_000.0
        if suffix.startswith("b"):
            return value * 1_000_000_000.0
        return value

    def _read_score_value(self):
        text_objects = image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            self.search_areas["hydra_score"],
        )
        candidates = [obj.text for obj in text_objects if obj.text]
        if candidates:
            candidates.extend([" ".join(candidates), "".join(candidates)])

        for candidate in candidates:
            parsed = self._parse_score_value(candidate)
            if parsed > 0:
                return parsed
        raise ValueError("No numeric value found in Hydra score text.")

    def _is_result_text(self, text):
        return resembles(text, "RESULTADO", threshold=0.72) or resembles(
            text, "RESULTADOS", threshold=0.72
        )

    def update_battle_status(self):
        try:
            result = image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                search_area=self.search_areas["hydra_finished"],
            )[0]
            time.sleep(5)
            result2 = image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                search_area=self.search_areas["hydra_finished"],
            )[0]
            if self._is_result_text(result.text) and self._is_result_text(result2.text):
                self.battle_status = "Done"
        except Exception:
            pass

    def _is_setup_already_selected(self, setup_text, selected_markers, y_offset=70.0, tolerance=60.0) -> bool:
        if not setup_text or not selected_markers:
            return False

        target_y = float(setup_text.mean_pos_y) + float(y_offset)
        return any(
            abs(float(marker.mean_pos_y) - target_y) <= float(tolerance)
            for marker in selected_markers
        )

    def _click_hydra_setup(self, setup_text, y_offset=70.0):
        window_tools.click_at(
            setup_text.mean_pos_x - 320.0,
            setup_text.mean_pos_y + y_offset,
            delay=2,
            window=self.window,
        )

    def _select_hydra_setup(self, difficulty):
        setup_name = "NM" if self.normalize_difficulty(difficulty) == "Nightmare" else difficulty
        setup_variants = [setup_name]
        if setup_name == "NM":
            setup_variants.append("Nightmare")

        setup_section_area = self.search_areas.get("hydra_setup_section_groups", [0.026, 0.435, 0.059, 0.072])
        setup_names_area = self.search_areas.get("hydra_setup_names", [0.2, 0.0, 0.4, 0.9])
        setup_check_area = self.search_areas.get("hydra_setup_check", [0.028, 0.087, 0.049, 0.898])

        current_setup = None
        try:
            window_tools.click_center(self.window, setup_section_area, delay=1.5)
            window_tools.move_up(self.window, strength=3, relative_x=0.15)

            for _ in range(3):
                if not self.main_loop_running:
                    break

                setups = image_tools.get_text_in_relative_area(
                    self.reader,
                    self.window,
                    search_area=setup_names_area,
                )

                for name in setups:
                    if any(resembles(name.text, variant, threshold=0.7) for variant in setup_variants):
                        current_setup = name
                        break

                if current_setup:
                    break

                window_tools.move_down(self.window, strength=0.5, relative_x=0.15)

            if current_setup:
                selected_markers = image_tools.get_similarities_in_relative_area(
                    self.window,
                    setup_check_area,
                    "data\\assets\\images\\doom_tower_completed_stage.png",
                )
                if not self._is_setup_already_selected(current_setup, selected_markers):
                    self._click_hydra_setup(current_setup)
                    self.log.info(
                        "Hydra build selection: clicked setup '%s'.",
                        setup_name,
                    )
                else:
                    self.log.info("Hydra build selection: setup '%s' already selected.", setup_name)
                self.log.info("Hydra build selection: using setup '%s'.", setup_name)
                return True

            self.log.warning("Hydra build selection: setup '%s' not found.", setup_name)
        except Exception as exc:
            self.log.warning("Hydra build selection failed for '%s': %s", setup_name, exc)
        return False

    def execute_hydra_encounter(self):
        difficulty = self.hydra_encounter_difficulty
        if not difficulty:
            raise RuntimeError("No Hydra difficulty selected.")

        difficulty_key = f"Hydra_{difficulty}"
        if difficulty_key not in self.search_areas:
            raise RuntimeError(f"Missing search area for selected difficulty: {difficulty_key}")

        self._click_hydra_difficulty(difficulty)

        reclaim_status = image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            search_area=self.search_areas["hydra_start_battle"],
        )
        if reclaim_status and (
            resembles(reclaim_status[0].text, "Reclamar", threshold=0.7)
            or resembles(reclaim_status[0].text, "Claim", threshold=0.7)
        ):
            window_tools.click_center(self.window, self.search_areas["hydra_start_battle"])
            window_tools.click_center(self.window, self.search_areas["Hydra_NameList"])
            window_tools.click_center(self.window, self.search_areas["Hydra_NameList"])

        window_tools.click_center(self.window, self.search_areas["hydra_start_battle"])

        self._select_hydra_setup(difficulty)
        window_tools.click_center(self.window, self.search_areas["hydra_start_battle"])

        self.auto_button_clicked = False
        manual_run_enabled = (
            self.manual_play_enabled
            and
            self.manual_player is not None
            and self.hydra_encounter_difficulty in self.manual_play_difficulties
        )
        if manual_run_enabled:
            try:
                self.manual_player.load_profile(self.manual_profile_name)
                self.log.info(
                    "Manual run active for Hydra '%s' with profile '%s'.",
                    self.hydra_encounter_difficulty,
                    self.manual_profile_name,
                )
            except Exception as exc:
                self.log.error("Manual profile load failed: %s", exc)

        window_tools.click_center(self.window, self.search_areas["hydra_start_battle"])

        if not image_tools.check_startup(self):
            window_tools.click_center(self.window, self.search_areas["hydra_start_battle"])

        self.battle_status = "Starting"
        auto_battle_tools.reset_auto_battle_watchdog(self)
        while self.main_loop_running and self.battle_status != "Done":
            _ensure_within_run_deadline(self, "waiting for hydra encounter result")
            auto_battle_tools.ensure_auto_battle_running(self)

            if manual_run_enabled:
                if self.battle_status == "Starting":
                    try:
                        auto_battle_area = self.search_areas["auto_battle_button"]
                        auto_button = image_tools.get_text_in_relative_area(
                            self.reader,
                            self.window,
                            search_area=auto_battle_area,
                        )
                        if auto_button and resembles(auto_button[0].text, "Auto", threshold=0.65):
                            self.battle_status = "Battle active"
                            time.sleep(1.5)
                            if not self.auto_button_clicked:
                                window_tools.click_center(self.window, auto_battle_area, delay=0.8)
                                self.auto_button_clicked = True
                    except Exception:
                        pass
                    if self.battle_status == "Starting":
                        continue
                try:
                    self.manual_player.take_turn()
                    time.sleep(4)
                except Exception as exc:
                    self.log.error("Manual run turn execution failed: %s", exc)

            self.update_battle_status()

        score = self._read_score_value()
        threshold = self._threshold_for_difficulty(difficulty)
        self.log.info(
            "Hydra %s score: %.1fM (threshold %.1fM).",
            difficulty,
            score / 1_000_000.0,
            threshold / 1_000_000.0,
        )

        if score > threshold:
            if difficulty not in self.hydra_encounters_cleared:
                self.hydra_encounters_cleared.append(difficulty)
            window_tools.click_center(self.window, self.search_areas["hydra_save_battle"])
            window_tools.click_center(self.window, self.search_areas["hydra_save_battle"])
        else:
            window_tools.click_center(self.window, self.search_areas["hydra_retry_battle"])
            window_tools.sendkey("esc", window=self.window)
            self.lost_encounter = True

    def run_hydra(self, main_loop_running=True, max_run_duration_seconds=MAX_RUN_DURATION_SECONDS):
        _start_run_deadline(self, max_run_duration_seconds)
        self.main_loop_running = main_loop_running
        self.lost_encounter = False
        self.update_available_keys()
        if self.num_of_keys == 0:
            self.log.info("No Hydra keys available; skipping run.")
            return

        window_tools.move_down(self.window, strength=0.5, relative_x=0.8)
        self.detect_cleared_difficulties()

        while self.main_loop_running and len(self.hydra_encounters_cleared) < len(self.difficulty_order):
            _ensure_within_run_deadline(self, "running hydra loop")
            window_tools.move_down(self.window, strength=0.5, relative_x=0.8)
            self.update_available_keys()
            if self.num_of_keys == 0 or self.lost_encounter:
                break

            if not self.select_next_difficulty():
                break
            self.execute_hydra_encounter()

        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
