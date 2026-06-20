# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 14:00:37 2025

@author: Arthur
"""

import time
import re
import math
import json
import unicodedata
from datetime import datetime
from pathlib import Path
import pyautogui
import raid_bot.utils.auto_battle_tools as auto_battle_tools
import raid_bot.utils.image_tools as image_tools
import raid_bot.utils.window_tools as window_tools
import difflib

MAX_RUN_DURATION_SECONDS = int(3.5 * 60 * 60)
MAX_DOOMTOWER_MODE_SECONDS = int(15 * 60)
DEBUG_ROOT_DIR = Path("data") / "output" / "debug" / "doomtower"

DEBUG_STAGE_OVERVIEW = [
    ("enter_mode", "Doom Tower mode is entered and debug session starts."),
    ("rotation_missing_exit", "Rotation was not detected, mode exits immediately."),
    ("post_entry_wait", "10 second stabilization wait after entering Doom Tower."),
    ("keys_updated", "Silver/Gold key OCR update snapshot."),
    ("loop_start", "Start of one Doom Tower boss loop iteration."),
    ("mode_timeout_exit", "15-minute limit reached before selecting/starting next battle."),
    ("rotation_boss_selected", "Boss selected from current detected rotation."),
    ("tile_probe_attempt", "Known tile probe click attempt."),
    ("tile_probe_miss", "Known tile did not open a valid boss stage."),
    ("tile_probe_match", "Known tile opened a valid boss stage."),
    ("scan_start", "Boss template scan starts in current viewport."),
    ("scan_candidate_click", "A possible boss icon was clicked for validation."),
    ("scan_match_confirmed", "Boss stage validated and accepted."),
    ("no_visible_boss_exit", "No valid boss stage found in current viewport."),
    ("stage_opened", "Selected stage panel is open."),
    ("stage_validation_failed_exit", "Opened stage is not a valid boss for current rotation."),
    ("battle_start", "Encounter start click sequence begins."),
    ("battle_complete", "Encounter result reached and closed."),
    ("no_silver_keys_exit", "Silver keys depleted, mode exits."),
    ("mode_finished", "Doom Tower mode finished."),
]


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


class RSL_Bot_DoomTower():
    def __init__(self, title_substring="Raid: Shadow Legends", reader = None, window = None, verbose = True, setup_build_names = None, setup = None):
        self.reader = reader
        
        self.running = True
        self.main_loop_running = True
        
        self.verbose = verbose
        self.setup = setup

        self.window = window
            
        self.search_areas = {
            "menu_name": [0.008, 0.034, 0.43, 0.037],   # [left, top, width, height]
            "go_to_higher_menu":   [0.928, 0.031, 0.046, 0.039],
            "pov":   [0.0, 0.1, 1, 0.9],
            "pov_full":   [0.0, 0.0, 1.0, 1.0],
            "detect_doomtower_rotation": [0.121, 0.696, 0.189, 0.035],

            "doom_tower_keys":   [0.682, 0.033, 0.212, 0.04],
            "doom_tower_menu_name":   [0.009, 0.033, 0.4, 0.041],
            "doom_tower_difficulty_current":   [0.03, 0.917, 0.079, 0.043],
            "doom_tower_difficulty_switch_normal":   [0.092, 0.798, 0.08, 0.036],
            "doom_tower_difficulty_switch_hard":   [0.096, 0.865, 0.066, 0.038],

            "doom_tower_setup_section_groups":   [0.026, 0.435, 0.059, 0.072],
            "doom_tower_setup_names":   [0.162, 0.087, 0.202, 0.75],
            "doom_tower_setup_check":   [0.028, 0.087, 0.049, 0.898],

            "doom_tower_check_boss_stage_complete":   [0.679, 0.698, 0.04, 0.051],


            "doom_tower_check_boss":   [0.905, 0.54, 0.068, 0.09],
            "doom_tower_check_boss_name":   [0.012, 0.032, 0.458, 0.046],
            "doom_tower_start_encounter":   [0.763, 0.877, 0.211, 0.106],
            "doom_tower_restart_encounter":   [0.423, 0.877, 0.211, 0.106],
            "doom_tower_automatic_climb":    [0.373, 0.936, 0.194, 0.036],

            "doom_tower_battle_result_automatic_climb":    [0.369, 0.339, 0.263, 0.04],
            "doom_tower_battle_result_automatic_climb_2":    [0.355, 0.381, 0.283, 0.044],
            "doom_tower_battle_result":    [0.396, 0.161, 0.195, 0.056],
            # "doom_tower_battle_result_2":    [0.38, 0.085, 0.224, 0.059],
            "doom_tower_close_encounter":   [0.13, 0.898, 0.064, 0.076],

            "doom_tower_auto_battle_button": [0.026, 0.899, 0.058, 0.07],

            "doom_tower_farm_encounter": [0.763, 0.764, 0.211, 0.105],
            "doom_tower_start_multibattles": [0.254, 0.634, 0.23, 0.075],
            "doomtower_multibattles_setup_1": [0.222, 0.458, 0.032, 0.04],
            "doomtower_multibattles_setup_2": [0.221, 0.502, 0.034, 0.045],
            "doom_tower_farming_status": [0.366, 0.609, 0.269, 0.102],
            
            


        }

        self.translation_mapping = {
            'Normal' : 'normal',
            'Dificil' : 'hard',
            'Dragon de Magma': 'magma_dragon',
            'Arana de Escarcha': 'frost_spider',
            'Arana Abisal': 'nether_spider',
            'Arana del Vacio': 'nether_spider',
            'Escarabajo Rey': 'scarab_king',
            'Dragon Eterno': 'eternal_dragon',
            'Hada Oscura': 'dark_fae',
            'Grifo Celestial': 'gryphon',
            'Cuernoterror': 'bommal',
        }


        

        self.doomtower_rotations = {
            '1': {
                '10': 'Magma Dragon',
                '20': 'Nether Spider',
                '30': 'Scarab King',
                '40': 'Frost Spider',
                '50': 'Scarab King',
                '60': 'Nether Spider',
                '70': 'Frost Spider',
                '80': 'Magma Dragon',
                '90': 'Nether Spider',
                '100': 'Scarab King',
                '110': 'Magma Dragon',
                '120': 'Frost Spider'
            },
            '2': {
                '10': 'Gryphon',
                '20': 'Magma Dragon',
                '30': 'Nether Spider',
                '40': 'Eternal Dragon',
                '50': 'Gryphon',
                '60': 'Magma Dragon',
                '70': 'Nether Spider',
                '80': 'Eternal Dragon',
                '90': 'Gryphon',
                '100': 'Magma Dragon',
                '110': 'Nether Spider',
                '120': 'Eternal Dragon'
            },
            '3': {
                '10': 'Bommal',
                '20': 'Scarab King',
                '30': 'Gryphon',
                '40': 'Dark Fae',
                '50': 'Bommal',
                '60': 'Scarab King',
                '70': 'Gryphon',
                '80': 'Dark Fae',
                '90': 'Bommal',
                '100': 'Scarab King',
                '110': 'Gryphon',
                '120': 'Dark Fae'
            }
        }
    
        self.current_rotation = None
        self.current_difficulty = None
        self.battles_done = 0
        self.battles_won = 0
        self.battle_status = 'Starting'
        self.doomtower_completed = False
        self.doomtower_climb_status_hard = False
        self.doomtower_climb_status_normal = False
        self.highest_stage_available = {'normal': 1, 'hard': 1}
        self.max_run_duration_seconds = MAX_RUN_DURATION_SECONDS
        self.max_mode_duration_seconds = MAX_DOOMTOWER_MODE_SECONDS
        self._run_deadline = None
        self._mode_started_at = None
        self.debug_enabled = True
        self.debug_session_dir = None
        self.debug_trace_file = None
        self.debug_stage_counter = 0

    # ------------------------- Reset -------------------------
    def reset_run_state(self):
        self.battle_status = 'Starting'

    # ------------------------- Debug -------------------------
    def _window_region(self):
        if not self.window:
            return None
        return (
            int(self.window.left),
            int(self.window.top),
            int(self.window.width),
            int(self.window.height),
        )

    def _safe_stage_name(self, stage_name):
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", stage_name).strip("_")
        return safe or "stage"

    def _ensure_debug_stage_reference(self):
        if not self.debug_enabled:
            return
        try:
            DEBUG_ROOT_DIR.mkdir(parents=True, exist_ok=True)
            reference_file = DEBUG_ROOT_DIR / "execution_stages.md"
            lines = [
                "# Doom Tower Execution Stages",
                "",
                "This file documents the checkpoints captured by the Doom Tower debug pipeline.",
                "",
            ]
            for idx, (name, desc) in enumerate(DEBUG_STAGE_OVERVIEW, start=1):
                lines.append(f"{idx}. `{name}`: {desc}")
            reference_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"Debug reference write failed: {exc}")

    def _start_debug_session(self):
        if not self.debug_enabled:
            return
        try:
            self._ensure_debug_stage_reference()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_session_dir = DEBUG_ROOT_DIR / f"run_{timestamp}"
            self.debug_session_dir.mkdir(parents=True, exist_ok=True)
            self.debug_trace_file = self.debug_session_dir / "trace.log"
            self.debug_stage_counter = 0
            self._debug_snapshot(
                "enter_mode",
                capture=True,
                rotation=self.current_rotation,
                setup=self.setup,
            )
        except Exception as exc:
            print(f"Debug session start failed: {exc}")

    def _debug_snapshot(self, stage_name, capture=True, **meta):
        if not self.debug_enabled or not self.debug_session_dir:
            return
        try:
            self.debug_stage_counter += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            payload = json.dumps(meta, ensure_ascii=True, default=str)
            line = (
                f"{timestamp} | {self.debug_stage_counter:03d} | "
                f"{stage_name} | {payload}\n"
            )
            with open(self.debug_trace_file, "a", encoding="utf-8") as handle:
                handle.write(line)

            if capture:
                region = self._window_region()
                if region:
                    filename = (
                        f"{self.debug_stage_counter:03d}_"
                        f"{self._safe_stage_name(stage_name)}.png"
                    )
                    screenshot = pyautogui.screenshot(region=region)
                    screenshot.save(self.debug_session_dir / filename)
        except Exception as exc:
            print(f"Debug snapshot failed for '{stage_name}': {exc}")

    # ------------------------- Stage Field -------------------------
    def scan_visible_stages(self):
        text_objects = image_tools.get_text_in_relative_area(
            self.reader, self.window, search_area=self.search_areas['pov']
        )

        list_of_stages = []
        for s in text_objects:
            match = re.search(r'^P(\d+)$', s.text.strip())
            if match:
                s.text = match.group(1)
                list_of_stages.append(s)
        return list_of_stages

    # ------------------------- Utilities -------------------------
    def _get_highest_key_for_value(self, inner_dict, target_value):
        matching_keys = [
            int(k) for k, v in inner_dict.items()
            if v == target_value and int(k) <= self.highest_stage_available['hard']
        ]
        if not matching_keys:
            print('Error getting Farming Stage value')
            return None
        return max(matching_keys)

    def objects_within_radius(self, obj1, obj2, radius):
        return math.hypot(
            obj1.mean_pos_x - obj2.mean_pos_x,
            obj1.mean_pos_y - obj2.mean_pos_y
        ) <= radius

    def _mode_time_exceeded(self):
        if self._mode_started_at is None:
            return False
        return (time.time() - self._mode_started_at) >= self.max_mode_duration_seconds

    def _get_menu_name_text(self):
        try:
            menu = image_tools.get_text_in_relative_area(
                self.reader, self.window,
                search_area=self.search_areas["doom_tower_menu_name"]
            )[0]
            return menu.text.strip()
        except:
            return ""

    def _extract_stage_number(self, menu_text):
        if not menu_text:
            return None
        try:
            return int(re.findall(r"\d+", menu_text)[0])
        except:
            return None

    def _normalize_text(self, text):
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKD", str(text))
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = re.sub(r"[^a-zA-Z0-9\s]+", " ", normalized).lower()
        return re.sub(r"\s+", " ", normalized).strip()

    def _translate_menu_boss_name(self, menu_text):
        normalized_menu = self._normalize_text(menu_text)
        if not normalized_menu:
            return None

        rotation_bosses = set(self._get_rotation_bosses())
        if not rotation_bosses:
            return None
        rotation_boss_by_slug = {
            self.translation_mapping[boss]: boss
            for boss in rotation_bosses
            if boss in self.translation_mapping
        }

        # Direct canonical-name match (works if OCR returns English or identical names).
        for boss in rotation_bosses:
            if self._normalize_text(boss) in normalized_menu:
                return boss

        # Translation-based match (e.g. Spanish OCR key -> slug -> canonical boss).
        for source_name, mapped_slug in self.translation_mapping.items():
            if mapped_slug not in rotation_boss_by_slug:
                continue
            if self._normalize_text(source_name) in normalized_menu:
                return rotation_boss_by_slug[mapped_slug]

        return None

    def _is_selected_stage_rotation_boss(self):
        if not self.current_rotation or self.current_rotation not in self.doomtower_rotations:
            return False
        # print('1')
        menu_text = self._get_menu_name_text()
        if not menu_text or self.resembles(menu_text, "Torre del Destino"):
            return False
        # print('11')
        if "Jefe Final" in menu_text:
            expected_boss = self.doomtower_rotations[self.current_rotation].get("120")
        else:
            stage_number = self._extract_stage_number(menu_text)
            if not stage_number or stage_number % 10 != 0:
                return False
            expected_boss = self.doomtower_rotations[self.current_rotation].get(str(stage_number))
            return expected_boss
        if not expected_boss:
            return False
        # print('111')
        translated_boss = self._translate_menu_boss_name(menu_text)
        # print(translated_boss)
        if translated_boss:
            return self.resembles(translated_boss, expected_boss, threshold=0.85)
        # print('1111')
        if "Jefe Final" in menu_text:
            return True
        # print('11111')
        # print(menu_text)
        # print(expected_boss)
        return self.resembles(menu_text, expected_boss, threshold=0.55)

    def _close_stage_menu_if_open(self):
        menu_text = self._get_menu_name_text()
        if menu_text and not self.resembles(menu_text, "Torre del Destino"):
            window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"], delay=2)

    def _dismiss_stage_overlay(self):
        # Prefer close button, then ESC fallback in case a different overlay is open.
        self._close_stage_menu_if_open()
        window_tools.sendkey("esc", delay=1, window=self.window)

    def _probe_known_boss_tiles(self, loop_index):
        # Requested fixed probe order:
        # 1) right edge tile: x=[0.9,1.0], y=[0.5,0.6]
        # 2) center tile:     x=[0.4,0.6], y=[0.4,0.6]
        known_tiles = [
            ("right_edge_tile", [0.9, 0.5, 0.1, 0.1]),
            ("right_edge_tile_2", [0.8727, 0.4939, 0.1134, 0.207]),
            ("center_tile", [0.4051, 0.2879, 0.1628, 0.2277]),
        ]

        for idx, (tile_name, tile_area) in enumerate(known_tiles, start=1):
            self._debug_snapshot(
                "tile_probe_attempt",
                capture=True,
                loop_index=loop_index,
                tile_index=idx,
                tile_name=tile_name,
                tile_area=tile_area,
            )

            window_tools.click_center(self.window, tile_area, delay=2)
            menu_text = self._get_menu_name_text()
            is_boss = self._is_selected_stage_rotation_boss()

            self._debug_snapshot(
                "stage_opened",
                capture=True,
                loop_index=loop_index,
                tile_index=idx,
                tile_name=tile_name,
                menu_text=menu_text,
                is_boss=is_boss,
            )

            if is_boss:
                self._debug_snapshot(
                    "tile_probe_match",
                    capture=True,
                    loop_index=loop_index,
                    tile_index=idx,
                    tile_name=tile_name,
                    menu_text=menu_text,
                )
                return True, tile_name

            

            self._debug_snapshot(
                "tile_probe_miss",
                capture=True,
                loop_index=loop_index,
                tile_index=idx,
                tile_name=tile_name,
                menu_text=menu_text,
            )
            menu_text2 = self._get_menu_name_text()
            if self.resembles(menu_text2, 'Torre del Destino'):
                continue
            self._dismiss_stage_overlay()

        return False, None

    # ------------------------- Keys -------------------------
    def update_available_keys(self):
        """Check Doom Tower keys."""
        try:
            doom_tower_keys = image_tools.get_text_in_relative_area(
                self.reader, self.window,
                search_area=self.search_areas['doom_tower_keys']
            )

            self.num_of_gold_keys = int(re.findall(r"\d+", doom_tower_keys[0].text)[0])
            self.num_of_silver_keys = int(re.findall(r"\d+", doom_tower_keys[1].text)[0])

            print(doom_tower_keys[0].text)
            print(doom_tower_keys[1].text)

        except:
            self.num_of_gold_keys = 0
            self.num_of_silver_keys = 0

    # ------------------------- Builds -------------------------
    def check_list_of_builds(self, max_attempts=3):
        window_tools.move_down(self.window, strength=0.5, relative_x=0.25)

    # ------------------------- Stage Reconstruction -------------------------
    def _reconstruct_stage_numbers(self, stages_numbers):
        if not stages_numbers:
            return stages_numbers

        original_nums = []
        for s in stages_numbers:
            try:
                original_nums.append(int(s.text))
            except ValueError:
                original_nums.append(None)

        max_original = max(num for num in original_nums if num is not None)

        best_sequence = None
        best_score = -1

        for start in range(max_original, max_original - 21, -1):
            seq = []
            current = start
            for _ in range(len(stages_numbers)):
                while self.main_loop_running and (current % 10 == 0):
                    current -= 1
                seq.append(current)
                current -= 1

            score = sum(
                1 for s_num, o_num in zip(seq, original_nums)
                if o_num is not None and s_num == o_num
            )

            if score > best_score:
                best_score = score
                best_sequence = seq

        for obj, num in zip(stages_numbers, best_sequence):
            obj.text = str(num)

        return stages_numbers

    # ------------------------- Difficulty -------------------------
    def set_difficulty(self, set_level=None):
        difficulty = image_tools.get_text_in_relative_area(
            self.reader, self.window,
            search_area=self.search_areas['doom_tower_difficulty_current']
        )[0]

        if difficulty.text in self.translation_mapping:
            self.current_difficulty = self.translation_mapping[difficulty.text]

        if set_level and self.current_difficulty != set_level:
            switch_key = f'doom_tower_difficulty_switch_{set_level}'
            window_tools.click_center(self.window, self.search_areas['doom_tower_difficulty_current'])
            window_tools.click_center(self.window, self.search_areas[switch_key], delay=5)
            self.current_difficulty = set_level

    # ------------------------- Battle Outcome -------------------------
    def _read_battle_status_once(self):
        try:
            result_objects = image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                search_area=self.search_areas["doom_tower_automatic_climb"],
            )
            for result in result_objects:
                text = (getattr(result, "text", "") or "").strip()
                if text and "Batallas completadas" in text:
                    return "AUTOCLIMB"
        except Exception:
            pass

        for key in (
            "doom_tower_battle_result_automatic_climb",
            "doom_tower_battle_result_automatic_climb_2",
        ):
            try:
                result_objects = image_tools.get_text_in_relative_area(
                    self.reader,
                    self.window,
                    search_area=self.search_areas[key],
                )
                for result in result_objects:
                    text = (getattr(result, "text", "") or "").strip()
                    if text and self.resembles(text, "Autoescalada completada"):
                        return "AUTOCLIMB_DONE"
            except Exception:
                pass

        try:
            auto_button = image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                search_area=self.search_areas["doom_tower_auto_battle_button"],
            )
            if auto_button:
                return None
        except Exception:
            pass

        try:
            result_objects = image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                search_area=self.search_areas["doom_tower_battle_result"],
            )
            for result in result_objects:
                text = (getattr(result, "text", "") or "").strip()
                if not text:
                    continue
                if self.resembles(text, "VICTORIA"):
                    return "VICTORIA"
                if self.resembles(text, "DERROTA"):
                    return "DERROTA"
        except Exception:
            pass

        return None

    def update_battle_status(self):
        first_result = self._read_battle_status_once()
        if first_result is None:
            return

        time.sleep(10)
        second_result = self._read_battle_status_once()
        if second_result is None or first_result != second_result:
            if self.verbose:
                print(
                    "Doom Tower battle result mismatch between checks. "
                    f"First='{first_result}', second='{second_result}'."
                )
            return

        if first_result == "AUTOCLIMB":
            self.battle_status = 'autoclimb'
            return

        if first_result == "AUTOCLIMB_DONE":
            self.battle_status = 'autoclimb_Done'
            window_tools.click_center(
                self.window,
                self.search_areas["doom_tower_close_encounter"],
                delay=5,
            )
            return

        if first_result in ("VICTORIA", "DERROTA") and self.battle_status != 'autoclimb':
            self.battle_status = 'Done'
            self.battles_done += 1
            if first_result == "VICTORIA":
                self.battles_won += 1
            else:
                self.no_run_failed = False
        return

    # ------------------------- Encounter Setup -------------------------
    def prepare_encounter(self):
        doom_tower_menu_name = image_tools.get_text_in_relative_area(
            self.reader, self.window,
            search_area=self.search_areas["doom_tower_menu_name"]
        )[0]

        number_match = re.findall(r'\d+', doom_tower_menu_name.text)
        number = number_match[0] if number_match else None

        if 'Jefe Final' in doom_tower_menu_name.text or (number and int(number) % 10 == 0):
            stage = '120' if 'Jefe Final' in doom_tower_menu_name.text else number
            current_opponent = self.doomtower_rotations.get(self.current_rotation, {}).get(stage, 'Waves')
            print(current_opponent)
        else:
            current_opponent = 'Waves'

        self.select_encounter_build(current_opponent)

    # ------------------------- Build Selection -------------------------
    def _is_setup_already_selected(self, setup_text, selected_markers, y_offset=70.0, tolerance=60.0) -> bool:
        if not setup_text or not selected_markers:
            return False

        target_y = float(setup_text.mean_pos_y) + float(y_offset)
        return any(
            abs(float(marker.mean_pos_y) - target_y) <= float(tolerance)
            for marker in selected_markers
        )

    def _click_doom_tower_setup(self, setup_text, y_offset=70.0):
        window_tools.click_at(
            setup_text.mean_pos_x - 268.0,
            setup_text.mean_pos_y + y_offset
        )

    def select_encounter_build(self, setup):
        self.current_setup = False

        window_tools.click_center(self.window, self.search_areas["doom_tower_setup_section_groups"])
        window_tools.move_up(self.window, strength=3, relative_x=0.15)

        for _ in range(3):
            if not self.main_loop_running:
                break            
            setups = image_tools.get_text_in_relative_area(
                self.reader, self.window,
                search_area=self.search_areas["doom_tower_setup_names"]
            )

            for name in setups:
                if self.resembles(name.text, setup):
                    self.current_setup = name
                    break

            if self.current_setup:
                break

            window_tools.move_down(self.window, strength=0.5, relative_x=0.15)

        if self.current_setup:
            completed = image_tools.get_similarities_in_relative_area(
                self.window,
                self.search_areas["doom_tower_setup_check"],
                'data\\assets\\images\\doom_tower_completed_stage.png'
            )

            if not self._is_setup_already_selected(self.current_setup, completed):
                self._click_doom_tower_setup(self.current_setup)
            window_tools.click_center(self.window, self.search_areas["doom_tower_start_encounter"])

    # ------------------------- Run Encounter -------------------------
    def farm_encounter(self):
        self.prepare_encounter()
        self.battle_status = 'Starting'
        window_tools.click_center(self.window, self.search_areas["doom_tower_farm_encounter"])
        doomtower_multibattles_setup_1 = image_tools.get_similarities_in_relative_area(
                self.window,
                self.search_areas["doomtower_multibattles_setup_1"],
                'data\\assets\\images\\doom_tower_multibattles_setup.png'
            )
        doomtower_multibattles_setup_2 = image_tools.get_similarities_in_relative_area(
                self.window,
                self.search_areas["doomtower_multibattles_setup_2"],
                'data\\assets\\images\\doom_tower_multibattles_setup.png'
            )
        if not doomtower_multibattles_setup_1:
            window_tools.click_center(self.window, self.search_areas["doomtower_multibattles_setup_1"])

        if not doomtower_multibattles_setup_2:
            window_tools.click_center(self.window, self.search_areas["doomtower_multibattles_setup_2"])

        window_tools.click_center(self.window, self.search_areas["doom_tower_start_multibattles"], delay = 5)
        self.battle_status = 'Running'
        auto_battle_tools.reset_auto_battle_watchdog(self)
        window_tools.click_at(self.stage_found.mean_pos_x, self.stage_found.mean_pos_y)

        while self.battle_status == "Running":
            _ensure_within_run_deadline(self, "waiting for doom tower farming result")
            auto_battle_tools.ensure_auto_battle_running(
                self,
                auto_button_area=self.search_areas["doom_tower_auto_battle_button"],
            )
            farming_status = image_tools.get_text_in_relative_area(
                self.reader, self.window,
                search_area=self.search_areas["doom_tower_farming_status"]
            )

            time.sleep(5)
            farming_status_2 = image_tools.get_text_in_relative_area(
                self.reader, self.window,
                search_area=self.search_areas["doom_tower_farming_status"]
            )

            time.sleep(2)
            if getattr(farming_status[0],'text', False) and getattr(farming_status_2[0],'text', False):
                if self.resembles(farming_status[0].text, "Resultados") and self.resembles(farming_status_2[0].text, "Resultados"):
                    self.battle_status = 'Finished'
                    window_tools.click_center(self.window, self.search_areas["doom_tower_farming_status"])
                    window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])





    def execute_encounter(self, farming=False, max_attempts=40):
        self._debug_snapshot("battle_start", capture=True, farming=farming)
        self.prepare_encounter()
        self.battle_status = 'Starting'
        check_correct_start = image_tools.get_text_in_relative_area(
                self.reader, self.window,
                search_area=self.search_areas["doom_tower_menu_name"]
            )
            
        window_tools.click_center(self.window, self.search_areas["doom_tower_start_encounter"])
        window_tools.click_center(self.window, self.search_areas["doom_tower_start_encounter"])
        window_tools.click_center(self.window, self.search_areas["doom_tower_start_encounter"])
        time.sleep(10)
        check_correct_execution = image_tools.get_text_in_relative_area(
                self.reader, self.window,
                search_area=self.search_areas["doom_tower_menu_name"]
            )
        
        if getattr(check_correct_start,'text', False) and getattr(check_correct_execution,'text', False):
            if self.resembles(check_correct_execution.text, check_correct_start.text):
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])


        attempt = 0
        auto_battle_tools.reset_auto_battle_watchdog(self)
        while self.main_loop_running and (True):
            _ensure_within_run_deadline(self, "waiting for doom tower encounter result")
            self.update_battle_status()
            auto_battle_tools.ensure_auto_battle_running(
                self,
                auto_button_area=self.search_areas["doom_tower_auto_battle_button"],
            )
            time.sleep(2)

            if self.battle_status == 'Done' and not farming:
                break

            # if self.battle_status == 'Done' and farming:
            #     attempt += 1
            #     if attempt >= max_attempts:
            #         break

            #     window_tools.click_center(
            #         self.window,
            #         self.search_areas["doom_tower_restart_encounter"]
            #     )
            #     time.sleep(10)
            #     self.battle_status = 'Starting'

            #     try:
            #         battle_result = image_tools.get_text_in_relative_area(
            #             self.reader, self.window,
            #             search_area=self.search_areas["doom_tower_battle_result"]
            #         )[0]
            #         if battle_result.text in ("VICTORIA", "DERROTA"):
            #             self.battle_status = 'Done'
            #     except:
            #         pass

        print('Battle_Done')
        window_tools.click_center(
            self.window,
            self.search_areas["doom_tower_close_encounter"],
            delay=5
        )
        self._debug_snapshot(
            "battle_complete",
            capture=True,
            battles_done=self.battles_done,
            battles_won=self.battles_won,
            final_status=self.battle_status,
        )

    # ------------------------- Boss Stage Check -------------------------
    def _check_boss_stage(self):
        if self.highest_stage_available == 119:
            x_pos = int(self.window.left + self.window.width / 2)
            y_pos = int(self.highest_stage.mean_pos_y - self.window.height * 0.2)
        else:
            x_pos = int(self.window.left + self.window.width * 0.98)
            y_pos = int(self.highest_stage.mean_pos_y - self.window.height * 0.1)

        window_tools.click_at(x_pos, y_pos)

        stage_completed = image_tools.get_similarities_in_relative_area(
            self.window,
            self.search_areas["doom_tower_check_boss_stage_complete"],
            'data\\assets\\images\\doom_tower_locked_stage.png'
        )

        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
        return not len(stage_completed) == 1

    # ------------------------- Highest Stage Detection -------------------------
    def detect_highest_unlocked_stage(self, max_attempts=10):
        self.highest_stage_available = 1
        end_reached = False
        attempts = 0

        while self.main_loop_running and (self.highest_stage_available != 120 and attempts != max_attempts and not end_reached):
            _ensure_within_run_deadline(self, "detecting highest unlocked doom tower stage")
            attempts += 1

            stages_completed = image_tools.get_similarities_in_relative_area(
                self.window,
                self.search_areas["pov"],
                'data\\assets\\images\\doom_tower_completed_stage.png'
            )

            if not stages_completed:
                self.highest_stage_available = None
                return None

            stages_completed.sort(key=lambda o: o.mean_pos_y)

            stages_numbers = self.scan_visible_stages()
            self._reconstruct_stage_numbers(stages_numbers)

            backtrack = 0
            for completed in stages_completed:
                for number in stages_numbers:
                    if number.text and self.objects_within_radius(completed, number, 100):
                        self.highest_stage_available = int(number.text) + backtrack
                        self.highest_stage = completed
                        break
                else:
                    backtrack += 1
                    continue
                break

            rel_y = (self.highest_stage.mean_pos_y - self.window.top) / self.window.height
            if rel_y < 0.3:
                window_tools.move_up(self.window, strength=1, relative_x=0.1)
                continue
            else:
                end_reached = True

            if self.highest_stage_available % 10 == 9:
                increment = 2 if self._check_boss_stage() else 1
                self.highest_stage_available += increment

            self.highest_stage_available = max(1, min(120, self.highest_stage_available))

    # ------------------------- Stage Scan -------------------------


    def resembles(self, text, target, threshold=0.8):
        ratio = difflib.SequenceMatcher(None, text.lower(), target.lower()).ratio()
        return ratio >= threshold


    def scan_for_boss_or_current_stage(self, farming=False):

        FIRST_PATH = 'data\\assets\\images\\doom_tower_current_stage.png'
        list_of_paths = [] if farming else [FIRST_PATH]
        expected_menu_names = {} if farming else {FIRST_PATH: 'Planta'}
        scan_area = self.search_areas.get("pov_full", self.search_areas["pov"])

        self.stage_found = False
        self.doomtower_climb_status = False
        candidate_debug_count = 0
        self._debug_snapshot(
            "scan_start",
            capture=True,
            farming=farming,
            rotation=self.current_rotation,
            scan_area=scan_area,
        )

        def add_boss_paths():
            for value in self.doomtower_rotations[self.current_rotation].values():
                if value in self.translation_mapping:
                    path = f"data\\assets\\images\\doom_tower_{self.translation_mapping[value]}.png"
                    if path not in list_of_paths:
                        list_of_paths.append(path)
                        expected_menu_names[path] = value

        add_boss_paths()

        for path in list_of_paths:
            threshold = 0.8
            possible = []
            if not self.main_loop_running:
                break

            while self.main_loop_running and (not possible and threshold > 0.4):
                _ensure_within_run_deadline(self, "scanning doom tower stage templates")
                possible = image_tools.get_similarities_in_relative_area(
                    self.window,
                    scan_area,
                    path,
                    threshold=threshold,
                    scales=[0.7, 0.8, 0.9, 1.0]
                )
                threshold -= 0.03

            for stage in possible:
                window_tools.click_at(stage.mean_pos_x, stage.mean_pos_y, delay=4)
                if candidate_debug_count < 5:
                    self._debug_snapshot(
                        "scan_candidate_click",
                        capture=True,
                        path=path,
                        x=round(stage.mean_pos_x, 1),
                        y=round(stage.mean_pos_y, 1),
                    )
                    candidate_debug_count += 1

                if not self.main_loop_running:
                    break

                try:
                    menu = image_tools.get_text_in_relative_area(
                        self.reader, self.window,
                        search_area=self.search_areas["doom_tower_menu_name"]
                    )[0]

                    number = re.findall(r'\d+', menu.text)
                    number = number[0] if number else '10'

                    print(menu.text)
                    if self.resembles(menu.text, 'Torre del Destino'):
                        continue

                    expected = expected_menu_names[path]

                    if expected == 'Planta' and expected in menu.text:
                        self.stage_found = stage
                        self.highest_stage_available[self.current_difficulty] = number
                        self._debug_snapshot(
                            "scan_match_confirmed",
                            capture=True,
                            matched_type="current_stage",
                            menu_text=menu.text,
                            stage_number=number,
                        )
                        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
                        return

                    if 'Jefe Final' not in menu.text and (
                        expected != self.doomtower_rotations[self.current_rotation].get(str(number))
                    ):
                        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
                        continue

                    if 'Jefe Final' in menu.text or int(number) % 10 == 0:
                        locked = image_tools.get_similarities_in_relative_area(
                            self.window,
                            self.search_areas["doom_tower_check_boss_stage_complete"],
                            'data\\assets\\images\\doom_tower_locked_boss.png'
                        )

                        if 'Jefe Final' in menu.text and not locked:
                            self.doomtower_climb_status = 'completed'

                        self.highest_stage_available[self.current_difficulty] = (
                            120 if 'Jefe Final' in menu.text else number
                        )

                        self.stage_found = stage
                        self._debug_snapshot(
                            "scan_match_confirmed",
                            capture=True,
                            matched_type="boss_stage",
                            menu_text=menu.text,
                            stage_number=number,
                        )
                        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
                        return
                except:
                    pass

        if not self.stage_found:
            self._debug_snapshot(
                "scan_no_match",
                capture=True,
                farming=farming,
            )

    # ------------------------- Simple Scan -------------------------
    def locate_highest_stage_simple(self, farming=False):

        window_tools.move_up(self.window, strength=15, relative_x=0.1)

        for _ in range(25):
            if not self.main_loop_running:
                break
            self.scan_for_boss_or_current_stage(farming=farming)
            if self.stage_found:
                break
            window_tools.move_down(self.window, strength=0.6, relative_x=0.1)

    # ------------------------- Climb -------------------------
    def progress_doom_tower(self):
        self.set_difficulty('hard')

        if self.doomtower_climb_status_hard != 'completed':
            self.locate_highest_stage_simple()

        if self.doomtower_climb_status in ('completed',) or \
           self.doomtower_climb_status_hard == 'completed':

            self.doomtower_climb_status_hard = 'completed'
            self.set_difficulty('normal')

            if not self.doomtower_completed:
                self.locate_highest_stage_simple()

            if self.doomtower_climb_status == 'completed':
                self.doomtower_climb_status_normal = 'completed'
                self.doomtower_completed = True

        if not self.doomtower_completed and self.stage_found:
            window_tools.click_at(self.stage_found.mean_pos_x, self.stage_found.mean_pos_y)
            self.execute_encounter()

    # ------------------------- Farming -------------------------
    def _get_rotation_bosses(self):
        if not self.current_rotation or self.current_rotation not in self.doomtower_rotations:
            return []
        return list(dict.fromkeys(self.doomtower_rotations[self.current_rotation].values()))

    def _get_farming_opponent_candidates(self):
        return self._get_rotation_bosses()

    def farm_doom_tower_bosses(self):
        self.set_difficulty(self.setup['difficulty'])
        if not self._get_farming_opponent_candidates():
            return

        self.locate_highest_stage_simple(farming=True)

        if self.stage_found:
            window_tools.click_at(self.stage_found.mean_pos_x, self.stage_found.mean_pos_y, delay=2)
            self._debug_snapshot(
                "stage_opened",
                capture=True,
                menu_text=self._get_menu_name_text(),
            )
            if not self._is_selected_stage_rotation_boss():
                self.stage_found = False
                self._debug_snapshot(
                    "stage_validation_failed_exit",
                    capture=True,
                    menu_text=self._get_menu_name_text(),
                )
                self._close_stage_menu_if_open()
                return
            self.farm_encounter()


    def farm_doom_tower_bosses_no_search(self):
        self.set_difficulty(self.setup['difficulty'])
        if not self._get_farming_opponent_candidates():
            return

        self.scan_for_boss_or_current_stage(farming=True)

        if not self.stage_found:
            return

        window_tools.click_at(self.stage_found.mean_pos_x, self.stage_found.mean_pos_y, delay=2)
        self._debug_snapshot(
            "stage_opened",
            capture=True,
            menu_text=self._get_menu_name_text(),
        )
        if not self._is_selected_stage_rotation_boss():
            self.stage_found = False
            self._debug_snapshot(
                "stage_validation_failed_exit",
                capture=True,
                menu_text=self._get_menu_name_text(),
            )
            self._close_stage_menu_if_open()
            return
        self.execute_encounter()

    # ------------------------- Runner -------------------------
    def run_doomtower(
        self,
        main_loop_running=True,
        max_run_duration_seconds=MAX_RUN_DURATION_SECONDS,
    ):
        _start_run_deadline(self, max_run_duration_seconds)
        self.reset_run_state()
        self.no_run_failed = True
        self.main_loop_running = main_loop_running
        self._mode_started_at = time.time()

        if not self.setup:
            self.setup = {
                "difficulty": "hard",
                "only_farming": False,
            }

        self.setup.setdefault("difficulty", "hard")
        time.sleep(5)
        self.setup.setdefault("only_farming", False)
        self._start_debug_session()

        if not self.current_rotation or self.current_rotation not in self.doomtower_rotations:
            print("Doom Tower rotation not detected. Exiting mode.")
            self._debug_snapshot("rotation_missing_exit", capture=True)
            self._debug_snapshot("mode_finished", capture=False, reason="rotation_missing")
            return

        # Let the tower screen settle before first scan.
        time.sleep(10)
        self._debug_snapshot("post_entry_wait", capture=True)
        self.update_available_keys()
        self._debug_snapshot(
            "keys_updated",
            capture=True,
            silver_keys=getattr(self, "num_of_silver_keys", None),
            gold_keys=getattr(self, "num_of_gold_keys", None),
        )

        loop_index = 0
        while self.main_loop_running:
            _ensure_within_run_deadline(self, "running doom tower boss loop")
            self.update_available_keys()
            loop_index += 1
            self._debug_snapshot(
                "loop_start",
                capture=True,
                loop_index=loop_index,
                silver_keys=getattr(self, "num_of_silver_keys", None),
                gold_keys=getattr(self, "num_of_gold_keys", None),
            )

            if self.num_of_silver_keys <= 0:
                self._debug_snapshot(
                    "no_silver_keys_exit",
                    capture=True,
                    loop_index=loop_index,
                )
                break

            # Per request: check mode timeout only before selecting/starting a battle.
            if self._mode_time_exceeded():
                print("Doom Tower mode reached 15 minutes. Exiting mode.")
                self._debug_snapshot(
                    "mode_timeout_exit",
                    capture=True,
                    checkpoint="before_scan",
                    loop_index=loop_index,
                )
                break

            self.set_difficulty(self.setup['difficulty'])
            matched, matched_tile = self._probe_known_boss_tiles(loop_index=loop_index)
            if not matched:
                self._debug_snapshot(
                    "no_visible_boss_exit",
                    capture=True,
                    loop_index=loop_index,
                    reason="known_tile_probe_no_match",
                )
                break

            # Timeout check before starting battle.
            if self._mode_time_exceeded():
                print("Doom Tower mode reached 15 minutes. Exiting mode.")
                self._debug_snapshot(
                    "mode_timeout_exit",
                    capture=True,
                    checkpoint="before_battle_start",
                    matched_tile=matched_tile,
                    loop_index=loop_index,
                )
                self._dismiss_stage_overlay()
                break

            self.execute_encounter()

        self._debug_snapshot(
            "mode_finished",
            capture=True,
            silver_keys=getattr(self, "num_of_silver_keys", None),
            gold_keys=getattr(self, "num_of_gold_keys", None),
            loops=loop_index,
        )

    
    # LEGACY CODE
    # def run_doomtower(self, main_loop_running = True):
    #     self.reset_run_state()
    #     self.update_available_keys()
    #     self.no_run_failed = True
    #     self.main_loop_running = main_loop_running

    #     if self.num_of_gold_keys == 0 and self.num_of_silver_keys < 2:
    #         return

    #     if self.setup['only_farming']:
    #         while self.main_loop_running and self.no_run_failed and self.num_of_silver_keys > 1:
    #             self.farm_doom_tower_bosses()
    #             if not self.stage_found:
    #                 break
    #             self.update_available_keys()
                
    #     else:
    #         while self.main_loop_running and (self.no_run_failed or (
    #             (self.doomtower_completed or self.num_of_gold_keys == 0)
    #             and self.num_of_silver_keys > 1
    #         )):
    #             self.stage_found = False
    #             if self.num_of_gold_keys > 0 and not self.doomtower_completed:
    #                 self.progress_doom_tower()
    #             if self.num_of_silver_keys>1:
    #                 self.farm_doom_tower_bosses()

    #             if not self.stage_found:
    #                 break

    # ------------------------- Test -------------------------
    def test(self):
        self.run_doomtower()
