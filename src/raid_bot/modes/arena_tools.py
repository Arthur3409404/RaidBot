# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:45:00 2025

@author: Arthur
"""


import pyautogui
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time
import re
from datetime import  timedelta
import os
from pathlib import Path
import raid_bot.utils.auto_battle_tools as auto_battle_tools
import raid_bot.utils.image_tools as image_tools
import raid_bot.utils.file_tools as file_tools
from raid_bot.utils.classic_arena_portraits import crop_classic_arena_portraits
from raid_bot.utils.tagteam_portraits import crop_tagteam_portraits
from raid_bot.utils.champion_identifier import load_default_champion_identifier
import raid_bot.utils.window_tools as window_tools
from raid_bot.handlers.ai_networks_handler import (
    ClassicCompositionEvaluationNetwork,
    EnemyDataset,
    TagTeamCompositionEvaluationNetwork,
)
import difflib

MAX_RUN_DURATION_SECONDS = int(3.5 * 60 * 60)
_AUTO_LOAD_CHAMPION_IDENTIFIER = object()
_ARENA_POWER_MATCH_ABS_TOLERANCE = 1000.0
DEFAULT_CLASSIC_EVALUATION_MODEL = Path("data/models/neural_networks/enemy_eval_classic_arena/composition_model.pt")
DEFAULT_TAGTEAM_EVALUATION_MODEL = Path("data/models/neural_networks/enemy_eval_tagteam_arena/composition_model.pt")


def _normalize_champion_name(name, slot_index):
    text = "" if name is None else str(name).strip()
    return text if text else f"UnknownChampion{slot_index}"


def _normalize_teamcomposition(teamcomposition):
    composition = []
    for index, name in enumerate(list(teamcomposition), start=1):
        composition.append(_normalize_champion_name(name, index))
    return composition


def _coerce_powervalue(powervalue):
    if isinstance(powervalue, np.ndarray):
        powervalue = powervalue.tolist()
    if isinstance(powervalue, (list, tuple)):
        return [float(value) for value in powervalue]
    return float(powervalue)


def _power_values_match(left, right, abs_tolerance=_ARENA_POWER_MATCH_ABS_TOLERANCE):
    if isinstance(left, np.ndarray):
        left = left.tolist()
    if isinstance(right, np.ndarray):
        right = right.tolist()

    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        if len(left) != len(right):
            return False
        return all(_power_values_match(a, b, abs_tolerance=abs_tolerance) for a, b in zip(left, right))

    try:
        return abs(float(left) - float(right)) <= abs_tolerance
    except (TypeError, ValueError):
        return False


def _normalize_saved_enemy_entry(entry):
    if isinstance(entry, dict):
        composition = entry.get("teamcomposition")
        if composition is None:
            composition = []
        if not isinstance(composition, (list, tuple)):
            composition = [composition]
        normalized = {
            "teamcomposition": _normalize_teamcomposition(composition),
            "powervalue": _coerce_powervalue(entry.get("powervalue", [] if composition else 0.0)),
            "label": str(entry.get("label", "loss")),
        }
        return normalized

    if isinstance(entry, np.ndarray):
        entry = entry.tolist()

    if isinstance(entry, (list, tuple)):
        try:
            return {
                "teamcomposition": [],
                "powervalue": [float(value) for value in entry],
                "label": "loss",
            }
        except (TypeError, ValueError):
            return None

    try:
        return {
            "teamcomposition": [],
            "powervalue": float(entry),
            "label": "loss",
        }
    except (TypeError, ValueError):
        return None


def _enemy_entries_match(existing_entry, candidate_entry):
    existing = _normalize_saved_enemy_entry(existing_entry)
    candidate = _normalize_saved_enemy_entry(candidate_entry)
    if existing is None or candidate is None:
        return False

    existing_composition = existing["teamcomposition"]
    candidate_composition = candidate["teamcomposition"]
    if existing_composition and candidate_composition and existing_composition != candidate_composition:
        return False

    return _power_values_match(existing["powervalue"], candidate["powervalue"])


def _load_composition_evaluation_ai(model_cls, model_path, *, verbose=True):
    if model_path in {None, "", False}:
        return None

    path = Path(model_path)
    if not path.exists():
        if verbose:
            print(f"[Arena AI] Name-based evaluation model not found: {path}")
        return None

    try:
        model = model_cls(weights_path=str(path))
        model.eval()
        if verbose:
            print(f"[Arena AI] Loaded name-based evaluation model: {path}")
        return model
    except Exception as exc:
        if verbose:
            print(f"[Arena AI] Could not load name-based evaluation model '{path}': {exc}")
        return None


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



# =============================================================================
#   CLASSIC ARENA BOT     
# =============================================================================

class RSL_Bot_ClassicArena:
    
    def __init__(
        self,
        title_substring="Raid: Shadow Legends",
        reader=None,
        window=None,
        param_file=None,
        champion_identifier=_AUTO_LOAD_CHAMPION_IDENTIFIER,
        verbose=True,
        update_dataset=True,
        num_multi_refresh=0,
        multi_refresh=False,
        power_threshold=70000,
        use_gems=True,
        enemies_lost=[0],
        evaluation_model_path=DEFAULT_CLASSIC_EVALUATION_MODEL,
    ):
        """
        Initialize the Classic Arena bot.
        """
        if reader is None:
            print("Error When Loading Reader")
        self.reader = reader
        if champion_identifier is _AUTO_LOAD_CHAMPION_IDENTIFIER:
            champion_identifier = load_default_champion_identifier()
        self.champion_identifier = champion_identifier

        self.running = True
        self.update_dataset = bool(update_dataset)
        self.dataset = None
        if self.update_dataset:
            self.dataset = EnemyDataset(
                "data/database_champions/datasets/enemy_dataset_classic_arena.npz",
                max_entries_per_file=100,
            )
        self.battles_done = 0
        self.classic_arena_multi_refresh = multi_refresh
        self.classic_arena_num_multi_refresh = num_multi_refresh
        self.verbose = verbose
        self.classic_arena_enemies_lost = enemies_lost
        self.classic_arena_use_gems = use_gems
        self.offset_wins = len(self.classic_arena_enemies_lost)
        self.window = window
        self.param_file = param_file or os.path.join("data", "params_mainframe.txt")
        self.init_time = time.time()
        self.classic_arena_power_threshold = power_threshold
        self.refresh_minutes = 15.2
        self.max_battle_time = 90
        self.no_coin_status = False
        self.max_run_duration_seconds = MAX_RUN_DURATION_SECONDS
        self._run_deadline = None
        self.recently_skipped_luchar_slots = {}
        self.skip_luchar_cooldown_seconds = 45.0
        self._pausa_esc_sent = False

        if self.window:
            self.coords = (self.window.left, self.window.top, self.window.width, self.window.height)
            print(f"Window Coordinates: {self.coords}")
        else:
            self.coords = None

        self.evaluation_ai = _load_composition_evaluation_ai(
            ClassicCompositionEvaluationNetwork,
            evaluation_model_path,
            verbose=self.verbose,
        )

        # Search Areas
        self.search_areas = {
            "go_to_higher_menu":   [0.928, 0.031, 0.046, 0.039],
            'pov' : [0, 0, 1, 1],
            "main_menu_labels":      [0.007, 0.27, 0.984, 0.044],

            "bronce_medals": [0.235, 0.038, 0.066, 0.03],
            "silver_medals": [0.342, 0.038, 0.068, 0.028],
            "gold_medals": [0.449, 0.04, 0.067, 0.026],
            "refresh_timer": [0.789, 0.15, 0.177, 0.059],
            "arena_coins": [0.65, 0.039, 0.3, 0.026],
            "add_arena_coins": [0.701, 0.039, 0.024, 0.028],
            "confirm_add_arena_coins": [0.394, 0.615, 0.208, 0.083],
            "confirm_use_gems": [0.394, 0.615, 0.208, 0.083],
            "gem_amount": [0.497, 0.629, 0.031, 0.03],
            "list_enemies": [0.665, 0.225, 0.314, 0.761],
            "start_battle": [0.762, 0.876, 0.213, 0.104],
            "battle_finished": [0.362, 0.897, 0.269, 0.081],
            "battle_result": [0.389, 0.148, 0.204, 0.071],
            "test": [0.05, 0.30, 0.15, 0.08],
            'luchar_area' : [0.5, 0, 0.5, 1],

            "enemy_total_power_value": [0.684, 0.708, 0.189, 0.037]
        }

        # Enemy positions
        self.corresponding_enemy_positions = {
            "Pos1": [[0.583, 0.31, 0.173, 0.023], [0.787, 0.237, 0.181, 0.082]],
            "Pos2": [[0.586, 0.43, 0.16, 0.019], [0.789, 0.357, 0.181, 0.078]],
            "Pos3": [[0.586, 0.548, 0.164, 0.019], [0.787, 0.473, 0.184, 0.084]],
            "Pos4": [[0.587, 0.664, 0.162, 0.021], [0.787, 0.592, 0.183, 0.08]],
            "Pos5": [[0.582, 0.782, 0.166, 0.022], [0.787, 0.709, 0.182, 0.082]],
            "Pos6": [[0.586, 0.901, 0.16, 0.018], [0.787, 0.827, 0.183, 0.082]],
            "Pos7": [[0.586, 0.592, 0.16, 0.02], [0.788, 0.519, 0.181, 0.079]],
            "Pos8": [[0.586, 0.709, 0.163, 0.021], [0.787, 0.636, 0.181, 0.08]],
            "Pos9": [[0.586, 0.829, 0.164, 0.019], [0.787, 0.754, 0.183, 0.081]],
            "Pos10": [[0.583, 0.946, 0.166, 0.02], [0.787, 0.874, 0.181, 0.078]],
        }

    def identify_portrait(self, portrait):
        if self.champion_identifier is None:
            return None
        return self.champion_identifier.predict_portrait(portrait)

    def identify_portraits(self, portraits):
        if self.champion_identifier is None:
            return [None] * len(list(portraits))
        return self.champion_identifier.predict_portraits(portraits)

    def crop_classic_arena_portraits(self, image_np):
        """Return the four classic arena portraits in left-to-right order."""
        return crop_classic_arena_portraits(image_np)

    def _build_enemy_composition_record(self, image_np, enemy_power, label="loss"):
        portraits = self.crop_classic_arena_portraits(image_np)
        champion_names = _normalize_teamcomposition(self.identify_portraits(portraits))
        return {
            "teamcomposition": champion_names,
            "powervalue": float(enemy_power),
            "label": str(label),
        }

    def _has_saved_enemy_match(self, enemy_record):
        return any(
            _enemy_entries_match(existing_entry, enemy_record)
            for existing_entry in self.classic_arena_enemies_lost
        )

    def _save_enemy_record(self, enemy_record):
        if self._has_saved_enemy_match(enemy_record):
            return False
        self.classic_arena_enemies_lost.append(enemy_record)
        self.persist_enemy_avoid_list()
        return True

    def _should_attack_enemy(self, enemy_record, enemy_power):
        if enemy_power < 500 or self._has_saved_enemy_match(enemy_record):
            return False

        if self.evaluation_ai is None:
            return enemy_power < self.classic_arena_power_threshold

        try:
            probability, label = self.evaluation_ai.predict(
                enemy_record["teamcomposition"],
                enemy_record["powervalue"],
            )
            if self.verbose:
                print(f"[Classic Arena AI] win_probability={probability:.3f} label={label}")
            return bool(label)
        except Exception as exc:
            if self.verbose:
                print(f"[Classic Arena AI] Evaluation failed, using power threshold fallback: {exc}")
            return enemy_power < self.classic_arena_power_threshold

    def _read_battle_result_once(self):
        try:
            battle_results = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas["battle_result"]
            )
            for battle_result in battle_results:
                text = (getattr(battle_result, "text", "") or "").strip()
                if not text:
                    continue
                if self.resembles(text, "VICTORIA"):
                    return "VICTORIA"
                if self.resembles(text, "DERROTA"):
                    return "DERROTA"
        except Exception:
            pass

        try:
            pov_results = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas["pov"]
            )
            for pov_result in pov_results:
                text = (getattr(pov_result, "text", "") or "").strip()
                if text and self.resembles(text, "Pausa"):
                    return "PAUSA"
        except Exception:
            pass

        return None

    def update_battle_outcome(self, enemy_record):
        """
        Determine outcome of a battle and update enemy memory if lost.
        """
        first_result = self._read_battle_result_once()
        if first_result is None:
            return False

        time.sleep(10)
        second_result = self._read_battle_result_once()
        if second_result is None or first_result != second_result:
            if self.verbose:
                print(
                    "Classic Arena battle result mismatch between checks. "
                    f"First='{first_result}', second='{second_result}'."
                )
            return False

        if first_result == "PAUSA":
            if not self._pausa_esc_sent:
                window_tools.sendkey("esc", delay=0.2, window=self.window)
                self._pausa_esc_sent = True
            return False

        self._pausa_esc_sent = False
        if first_result == "VICTORIA":
            print("Victory")
            self.recent_battle_outcome = 1
            return True

        if self._save_enemy_record(enemy_record):
            print("Updated Enemy Avoid List")
        self.recent_battle_outcome = 0
        return True

    def persist_enemy_avoid_list(self):
        """
        Persist lost enemies to the active params profile file.
        """
        file_tools.update_param_file_value(
            self.param_file,
            "classic_arena_enemies_lost",
            self.classic_arena_enemies_lost,
            create_if_missing=True,
        )

    def execute_arena_battle(self, enemy_record, start_button=None):
        """
        Engage an enemy and handle the battle loop.
        """
        battle_running = True
        if start_button is not None:
            window_tools.click_at(start_button.mean_pos_x, start_button.mean_pos_y)
        else:
            try:
                detected = image_tools.get_text_in_relative_area(
                    self.reader, self.window, self.search_areas["start_battle"]
                )[0]
                window_tools.click_at(detected.mean_pos_x, detected.mean_pos_y)
                start_button = detected
            except Exception:
                window_tools.click_center(self.window, self.search_areas["start_battle"])

        if not image_tools.check_startup(self):
            if start_button is not None:
                window_tools.click_at(start_button.mean_pos_x, start_button.mean_pos_y)
            else:
                window_tools.click_center(self.window, self.search_areas["start_battle"])

        auto_battle_tools.reset_auto_battle_watchdog(self)
        while self.main_loop_running and (battle_running):
            _ensure_within_run_deadline(self, "waiting for classic arena battle result")
            auto_battle_tools.ensure_auto_battle_running(self)
            try:
                battle_finished = image_tools.get_text_in_relative_area(
                    self.reader, self.window, search_area=self.search_areas["battle_finished"]
                )[0]
                if self.resembles(battle_finished.text ,"PULSA PARA CONTINUAR"):
                    time.sleep(3)
                    if not self.update_battle_outcome(enemy_record):
                        continue
                    battle_running = False
                    time.sleep(3)
                    # Multiple clicks to ensure continuation
                    for _ in range(4):
                        window_tools.click_at(battle_finished.mean_pos_x, battle_finished.mean_pos_y)
                        time.sleep(0.2)
                    
            except:
                pass
            time.sleep(3)

    def _parse_enemy_power_value(self, text):
        text = text.replace(".", "").replace(",", ".").replace(" ", "")
        matches = re.findall(r"(\d[\d.,]*)([a-zA-Z]*)", text)
        if not matches:
            raise ValueError("No numeric value found")

        number, suffix = matches[-1]
        value = float(number)

        suffix = suffix.lower()
        if suffix.startswith("k"):
            value *= 1_000
        elif suffix.startswith("m"):
            value *= 1_000_000

        return value

    def _capture_area_np(self, area_key):
        rel_left, rel_top, rel_width, rel_height = self.search_areas[area_key]
        abs_left = self.window.left + int(rel_left * self.window.width)
        abs_top = self.window.top + int(rel_top * self.window.height)
        abs_width = int(rel_width * self.window.width)
        abs_height = int(rel_height * self.window.height)
        screenshot = pyautogui.screenshot(region=(abs_left, abs_top, abs_width, abs_height))
        return np.array(screenshot)

    def _is_wrong_luchar_click(self, before_np, after_np, ssim_threshold=0.965, mad_threshold=0.014):
        if before_np is None or after_np is None:
            return False
        if before_np.shape != after_np.shape:
            return False

        if before_np.ndim == 3:
            before_gray = before_np[:, :, :3].mean(axis=2).astype(np.uint8)
            after_gray = after_np[:, :, :3].mean(axis=2).astype(np.uint8)
        else:
            before_gray = before_np.astype(np.uint8)
            after_gray = after_np.astype(np.uint8)

        similarity = ssim(before_gray, after_gray, data_range=255)
        mad = np.mean(np.abs(before_gray.astype(np.float32) - after_gray.astype(np.float32))) / 255.0
        return similarity >= float(ssim_threshold) and mad <= float(mad_threshold)

    def _luchar_slot_key(self, obj):
        return (int(round(float(obj.mean_pos_x) / 14.0)), int(round(float(obj.mean_pos_y) / 14.0)))

    def _prune_recently_skipped_luchar_slots(self):
        now = time.time()
        cooldown = float(self.skip_luchar_cooldown_seconds)
        stale = [key for key, ts in self.recently_skipped_luchar_slots.items() if (now - float(ts)) > cooldown]
        for key in stale:
            self.recently_skipped_luchar_slots.pop(key, None)

    def _mark_luchar_slot_skipped(self, slot_key):
        self.recently_skipped_luchar_slots[slot_key] = time.time()

    def _slot_recently_skipped(self, slot_key):
        self._prune_recently_skipped_luchar_slots()
        ts = self.recently_skipped_luchar_slots.get(slot_key)
        if ts is None:
            return False
        return (time.time() - float(ts)) <= float(self.skip_luchar_cooldown_seconds)

    def evaluate_arena_enemies(self):
        text_objects = image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            search_area=self.search_areas["luchar_area"],
            power_detection=False,
        )

        filtered = image_tools.filter_text_objects(text_objects)
        scanned_slots = set()

        for obj in filtered:
            if not self.main_loop_running:
                break

            if obj.text.strip() != "Luchar":
                continue

            slot_key = self._luchar_slot_key(obj)
            if slot_key in scanned_slots:
                continue
            scanned_slots.add(slot_key)
            if self._slot_recently_skipped(slot_key):
                continue
            list_card_screenshot = pyautogui.screenshot(
                region=(int(obj.mean_pos_x - 500), int(obj.mean_pos_y - 65), 440, 130)
            )
            list_card_np = np.array(list_card_screenshot)

            before_click = self._capture_area_np("pov")
            window_tools.click_at(obj.mean_pos_x, obj.mean_pos_y)
            time.sleep(1.1)
            after_click = self._capture_area_np("pov")

            if self._is_wrong_luchar_click(before_click, after_click):
                if self.verbose:
                    print("[Classic Arena] Ignored false 'Luchar' detection (screen unchanged).")
                self._mark_luchar_slot_skipped(slot_key)
                continue

            window_tools.click_center(self.window, self.search_areas["pov"])
            start_button = None
            try:
                start_button = image_tools.get_text_in_relative_area(
                    self.reader, self.window, search_area=self.search_areas["start_battle"], power_detection=False
                )[0]
            except Exception:
                start_button = None

            power_obj = image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                search_area=self.search_areas["enemy_total_power_value"],
                power_detection=False,
            )

            try:
                enemy_power = self._parse_enemy_power_value(power_obj[0].text)
            except Exception as e:
                print(f"[!] Error parsing Power Object: {e}")
                self._mark_luchar_slot_skipped(slot_key)
                window_tools.sendkey("esc", delay=5, window=self.window)
                continue

            image_np = list_card_np
            enemy_record = self._build_enemy_composition_record(image_np, enemy_power, label="loss")
            if self._should_attack_enemy(enemy_record, enemy_power):
                print("start execute")
                self.execute_arena_battle(enemy_record, start_button=start_button)
                self.battles_done += 1
                self.battle_occured = True
                if self.update_dataset and self.dataset is not None:
                    self.dataset.append_entry(enemy_record, self.recent_battle_outcome)

                outcome = "Win" if self.recent_battle_outcome else "Loss"
                print(f"Battle outcome: {outcome}")
                return True
            else:
                self._mark_luchar_slot_skipped(slot_key)
                window_tools.sendkey("esc", delay=5, window=self.window)

        return False


    def exit_battle_screen(self):
        window_tools.click_center(self.window, self.search_areas["battle_finished"])

    def refresh_enemy_list(self):
        if not self.coords or "refresh_timer" not in self.search_areas:
            return
        window_tools.click_center(self.window, self.search_areas["refresh_timer"])
        self.recently_skipped_luchar_slots.clear()

    def ensure_arena_coins(self):
        """
        Checks if arena coins are available; if not, attempt to use gems if allowed.
        """
        window_tools.click_center(self.window, self.search_areas["pov"])
        time.sleep(1)
        self.no_coin_status = False
        for _ in range(3):  # retry 3 times
            try:
                keys = image_tools.get_text_in_relative_area(
                    self.reader, self.window, self.search_areas["arena_coins"]
                )

                # Filter only entries that contain "/"
                coins_text = [key for key in keys if "/" in key.text][0]
                break

            except (IndexError, AttributeError):
                time.sleep(0.5)

        if "0/" in coins_text.text and coins_text.text != "10/10":
            rel_left, rel_top, rel_width, rel_height = self.search_areas["add_arena_coins"]#[0.701, 0.039, 0.024, 0.028] 962/1280 [0.681, 0.044, 0.028, 0.029]
            rel_left = coins_text.mean_pos_x/1280 - 0.06
            abs_left = self.window.left + int(rel_left * self.window.width)
            abs_top = self.window.top + int(rel_top * self.window.height)
            abs_width = int(rel_width * self.window.width)
            abs_height = int(rel_height * self.window.height)
            center_x = abs_left + abs_width // 2
            center_y = abs_top + abs_height // 2

            pyautogui.click(center_x, center_y)
            time.sleep(3)
            confirm_text = image_tools.get_text_in_relative_area(
                self.reader, self.window, self.search_areas["confirm_add_arena_coins"]
            )[0]
            confirm_gems_text = image_tools.get_text_in_relative_area(
                self.reader, self.window, self.search_areas["confirm_use_gems"]
            )[0]
            try:
                gem_amount_text = image_tools.get_text_in_relative_area(
                    self.reader, self.window, self.search_areas["gem_amount"]
                )[0].text
                numbers = re.findall(r"\d+", gem_amount_text)
                gem_amount = int("".join(numbers)) if numbers else 0
            except:
                gem_amount = 0

            if not self.classic_arena_use_gems and gem_amount > 0:
                pyautogui.click(center_x, center_y)
                time.sleep(3)
                self.no_coin_status = True
                return

            window_tools.click_at(confirm_text.mean_pos_x, confirm_text.mean_pos_y)
            time.sleep(3)

    def report_run_status(self):
        elapsed = time.time() - self.init_time
        formatted_elapsed = str(timedelta(seconds=int(elapsed)))
        medals = (self.battles_done - len(self.classic_arena_enemies_lost) + self.offset_wins) * 4

        print("\n" + "=" * 40)
        print("ðŸ›¡ï¸  RAID Classic Arena Bot Status")
        print("-" * 40)
        print(f"ðŸ” Mode: Multi Refresh ({self.classic_arena_num_multi_refresh})")
        print(f"â±ï¸  Time Since Start: {formatted_elapsed}")
        print(f"âš”ï¸  Battles Won: {self.battles_done - len(self.classic_arena_enemies_lost) + self.offset_wins}")
        print(f"âš”ï¸  Battles Lost: {len(self.classic_arena_enemies_lost) - + self.offset_wins}")
        print(f"ðŸŽ–ï¸  Estimated Medals: {medals}")
        print("-" * 40)
        print("ðŸ›‘ To stop the bot, press 'v'")
        print("=" * 40 + "\n")
        
    def resembles(self, text, target, threshold=0.8):
        ratio = difflib.SequenceMatcher(None, text.lower(), target.lower()).ratio()
        return ratio >= threshold


    def run_classic_arena_continuous(self, max_run_duration_seconds=MAX_RUN_DURATION_SECONDS):
        """
        Run the Classic Arena bot indefinitely without a time limit.
        """
        _start_run_deadline(self, max_run_duration_seconds)
        self.recently_skipped_luchar_slots.clear()
        time.sleep(5)

        time_start = time.time()
        last_refresh_time = time_start
        self.start_time = time_start
        counter_multi_refresh = 0

        while self.main_loop_running and (self.running):
            _ensure_within_run_deadline(self, "running classic arena continuous loop")
            self.report_run_status()
            self.battle_occured = False

            self.ensure_arena_coins()

            self.battle_occured = self.evaluate_arena_enemies()
            if self.battle_occured:
                continue

            window_tools.move_down(self.window)

            self.battle_occured = self.evaluate_arena_enemies()
            if self.battle_occured:
                continue

            window_tools.move_up(self.window)

            if self.classic_arena_multi_refresh:
                if counter_multi_refresh < self.classic_arena_num_multi_refresh:
                    self.refresh_enemy_list()
                    counter_multi_refresh += 1
                    continue
                else:
                    counter_multi_refresh = 0

            print("Waiting for free Refresh")
            time_start_loop = time.time()
            while self.main_loop_running and ((time.time() - time_start_loop) < 62):
                _ensure_within_run_deadline(self, "waiting for classic arena refresh")
                time.sleep(1)

            elapsed = time.time() - last_refresh_time
            if elapsed >= self.refresh_minutes * 60:
                if self.running:
                    self.refresh_enemy_list()
                    last_refresh_time = time.time()
                        
                        
    def run_classic_arena_until_empty(
        self,
        main_loop_running=True,
        max_run_duration_seconds=MAX_RUN_DURATION_SECONDS,
    ):
        """
        Run the Classic Arena bot once until no arena coins remain.
        """
        _start_run_deadline(self, max_run_duration_seconds)
        self.recently_skipped_luchar_slots.clear()
        time.sleep(5)
        self.main_loop_running = main_loop_running

        time_start = time.time()
        last_refresh_time = time_start
        self.start_time = time_start
        counter_multi_refresh = 0
        self.running = True

        time.sleep(5)

        while self.main_loop_running and (self.running):
            _ensure_within_run_deadline(self, "running classic arena single cycle")
            self.report_run_status()
            self.battle_occured = False

            self.ensure_arena_coins()
            if self.no_coin_status:
                self.running = False
                print("Waiting for coins")
                continue

            self.battle_occured = self.evaluate_arena_enemies()
            if self.battle_occured:
                continue

            window_tools.move_down(self.window)

            self.battle_occured = self.evaluate_arena_enemies()
            if self.battle_occured:
                continue

            window_tools.move_up(self.window)

            if self.classic_arena_multi_refresh:
                if counter_multi_refresh < self.classic_arena_num_multi_refresh:
                    self.refresh_enemy_list()
                    counter_multi_refresh += 1
                    continue
                else:
                    counter_multi_refresh = 0

            print("Waiting for free Refresh")
            self.running = False

        return
    
    
# =============================================================================
#   TAG TEAM BOT     
# =============================================================================

class RSL_Bot_TagTeamArena:
    """
    Automates Tag Team Arena battles in Raid: Shadow Legends.
    """

    def __init__(
        self,
        title_substring="Raid: Shadow Legends",
        reader=None,
        window=None,
        param_file=None,
        champion_identifier=_AUTO_LOAD_CHAMPION_IDENTIFIER,
        verbose=True,
        update_dataset=True,
        num_multi_refresh=0,
        multi_refresh=False,
        power_threshold=70000,
        use_gems=True,
        use_gems_max_amount=0,
        enemies_lost=[0],
        evaluation_model_path=DEFAULT_TAGTEAM_EVALUATION_MODEL,
    ):
        if reader is None:
            print("Error When Loading Reader")

        # Core state
        self.reader = reader
        if champion_identifier is _AUTO_LOAD_CHAMPION_IDENTIFIER:
            champion_identifier = load_default_champion_identifier()
        self.champion_identifier = champion_identifier
        self.window = window
        self.param_file = param_file or os.path.join("data", "params_mainframe.txt")
        self.running = True
        self.verbose = verbose
        self.update_dataset = bool(update_dataset)

        # Battle tracking
        self.battles_done = 0
        self.recent_battle_outcome = 0
        self.battle_occured = False

        # Enemy memory
        self.tagteam_arena_enemies_lost = enemies_lost
        self.offset_wins = len(self.tagteam_arena_enemies_lost)

        # Arena configuration
        self.tagteam_arena_power_threshold = power_threshold
        self.tagteam_arena_use_gems = use_gems
        self.tagteam_arena_use_gems_max_amount = use_gems_max_amount
        self.tagteam_arena_multi_refresh = multi_refresh
        self.tagteam_arena_num_multi_refresh = num_multi_refresh

        # Timing
        self.init_time = time.time()
        self.refresh_minutes = 15.2
        self.max_battle_time = 200
        self.max_run_duration_seconds = MAX_RUN_DURATION_SECONDS
        self._run_deadline = None
        self.recently_skipped_luchar_slots = {}
        self.skip_luchar_cooldown_seconds = 45.0
        self._pausa_esc_sent = False

        # Dataset
        self.dataset = None
        if self.update_dataset:
            self.dataset = EnemyDataset(
                "data/database_champions/datasets/enemy_dataset_tagteam_arena.npz",
                max_entries_per_file=100,
            )

        self.evaluation_ai = _load_composition_evaluation_ai(
            TagTeamCompositionEvaluationNetwork,
            evaluation_model_path,
            verbose=self.verbose,
        )

        # Window coordinates
        if self.window:
            self.coords = (
                self.window.left,
                self.window.top,
                self.window.width,
                self.window.height,
            )
            print(f"Window Coordinates: {self.coords}")
        else:
            self.coords = None

        # UI search areas (relative)
        self.search_areas = {
            "go_to_higher_menu":   [0.928, 0.031, 0.046, 0.039],
            'pov' : [0, 0, 1, 1],
            "main_menu_labels":      [0.007, 0.27, 0.984, 0.044],

            "bronce_medals": [0.235, 0.038, 0.066, 0.03],
            "silver_medals": [0.342, 0.038, 0.068, 0.028],
            "gold_medals": [0.449, 0.04, 0.067, 0.026],
            "refresh_timer": [0.789, 0.15, 0.177, 0.059],
            "arena_coins": [0.65, 0.039, 0.3, 0.026],
            "add_arena_coins": [0.701, 0.039, 0.024, 0.028],
            "confirm_add_arena_coins": [0.394, 0.615, 0.208, 0.083],
            "confirm_use_gems": [0.394, 0.615, 0.208, 0.083],
            "gem_amount": [0.498, 0.66, 0.03, 0.023],
            "list_enemies": [0.665, 0.225, 0.314, 0.761],
            "start_battle": [0.762, 0.876, 0.213, 0.104],
            "battle_finished": [0.362, 0.897, 0.269, 0.081],
            "battle_result": [0.389, 0.148, 0.204, 0.071],
            "close_encounter": [0.376, 0.639, 0.231, 0.071],
            'luchar_area' : [0.5, 0, 0.5, 1],

            "enemy_total_power_value": [0.707, 0.566, 0.17, 0.034],
            "enemy_team1_power_value": [0.763, 0.24, 0.155, 0.025],
            "enemy_team2_power_value": [0.758, 0.351, 0.156, 0.023],
            "enemy_team3_power_value": [0.763, 0.459, 0.154, 0.023],
        }

        # Enemy slot â†’ power + button areas
        self.corresponding_enemy_positions = {
            "Pos1": [[0.535, 0.324, 0.136, 0.017], [0.786, 0.236, 0.182, 0.081]],
            "Pos2": [[0.538, 0.453, 0.137, 0.018], [0.786, 0.362, 0.183, 0.086]],
            "Pos3": [[0.535, 0.582, 0.137, 0.019], [0.786, 0.496, 0.183, 0.082]],
            "Pos4": [[0.534, 0.709, 0.137, 0.019], [0.785, 0.623, 0.182, 0.082]],
            "Pos5": [[0.537, 0.84, 0.136, 0.021], [0.787, 0.754, 0.182, 0.08]],
            "Pos6": [[0.532, 0.416, 0.148, 0.021], [0.787, 0.33, 0.184, 0.08]],
            "Pos7": [[0.536, 0.544, 0.137, 0.019], [0.787, 0.457, 0.183, 0.083]],
            "Pos8": [[0.535, 0.674, 0.14, 0.02], [0.788, 0.59, 0.18, 0.077]],
            "Pos9": [[0.535, 0.803, 0.14, 0.022], [0.788, 0.72, 0.181, 0.076]],
            "Pos10": [[0.533, 0.93, 0.142, 0.026], [0.788, 0.848, 0.181, 0.079]],
        }

    def identify_portrait(self, portrait):
        if self.champion_identifier is None:
            return None
        return self.champion_identifier.predict_portrait(portrait)

    def identify_portraits(self, portraits):
        if self.champion_identifier is None:
            return [None] * len(list(portraits))
        return self.champion_identifier.predict_portraits(portraits)

    def crop_tagteam_portraits(self, image_np):
        """Return the 12 tag-team portraits in slot order: slot 1, then 2, then 3."""
        return crop_tagteam_portraits(image_np)

    def _build_enemy_composition_record(self, image_np, enemy_power_collection, label="loss"):
        portraits = self.crop_tagteam_portraits(image_np)
        champion_names = _normalize_teamcomposition(self.identify_portraits(portraits))
        return {
            "teamcomposition": champion_names,
            "powervalue": [float(value) for value in enemy_power_collection],
            "label": str(label),
        }

    def _has_saved_enemy_match(self, enemy_record):
        return any(
            _enemy_entries_match(existing_entry, enemy_record)
            for existing_entry in self.tagteam_arena_enemies_lost
        )

    def _save_enemy_record(self, enemy_record):
        if self._has_saved_enemy_match(enemy_record):
            return False
        self.tagteam_arena_enemies_lost.append(enemy_record)
        self.persist_enemy_avoid_list()
        return True

    def _should_attack_enemy(self, enemy_record, enemy_power):
        if enemy_power < 500 or self._has_saved_enemy_match(enemy_record):
            return False

        if self.evaluation_ai is None:
            return enemy_power < self.tagteam_arena_power_threshold

        try:
            probability, label = self.evaluation_ai.predict(
                enemy_record["teamcomposition"],
                enemy_record["powervalue"],
            )
            if self.verbose:
                print(f"[Tag Team Arena AI] win_probability={probability:.3f} label={label}")
            return bool(label)
        except Exception as exc:
            if self.verbose:
                print(f"[Tag Team Arena AI] Evaluation failed, using power threshold fallback: {exc}")
            return enemy_power < self.tagteam_arena_power_threshold

    # ------------------------------------------------------------------
    # Battle outcome & memory
    # ------------------------------------------------------------------

    def _read_battle_result_once(self):
        try:
            result_objects = image_tools.get_text_in_relative_area(
                self.reader, self.window, self.search_areas["battle_result"]
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

        try:
            pov_results = image_tools.get_text_in_relative_area(
                self.reader, self.window, self.search_areas["pov"]
            )
            for pov_result in pov_results:
                text = (getattr(pov_result, "text", "") or "").strip()
                if text and self.resembles(text, "Pausa"):
                    return "PAUSA"
        except Exception:
            pass

        return None

    def update_battle_outcome(self, enemy_record):
        first_result = self._read_battle_result_once()
        if first_result is None:
            return False

        time.sleep(10)
        second_result = self._read_battle_result_once()
        if second_result is None or first_result != second_result:
            if self.verbose:
                print(
                    "Tag Team battle result mismatch between checks. "
                    f"First='{first_result}', second='{second_result}'."
                )
            return False

        if first_result == "PAUSA":
            if not self._pausa_esc_sent:
                window_tools.sendkey("esc", delay=0.2, window=self.window)
                self._pausa_esc_sent = True
            return False

        self._pausa_esc_sent = False
        if first_result == "VICTORIA":
            print("Victory")
            self.recent_battle_outcome = 1
            return True

        print("Defeat - updating enemy avoid list")
        self.recent_battle_outcome = 0
        self._save_enemy_record(enemy_record)
        return True

    def resembles(self, text, target, threshold=0.8):
        ratio = difflib.SequenceMatcher(None, text.lower(), target.lower()).ratio()
        return ratio >= threshold
    

    def persist_enemy_avoid_list(self):
        file_tools.update_param_file_value(
            self.param_file,
            "tagteam_arena_enemies_lost",
            self.tagteam_arena_enemies_lost,
            create_if_missing=True,
        )

    # ------------------------------------------------------------------
    # Battle execution
    # ------------------------------------------------------------------

    def execute_tagteam_battle(self, enemy_record, start_button=None):
        if start_button is not None:
            window_tools.click_at(start_button.mean_pos_x, start_button.mean_pos_y)
            start_btn = start_button
        else:
            start_btn = image_tools.get_text_in_relative_area(
                self.reader, self.window, self.search_areas["start_battle"]
            )[0]
            window_tools.click_at(start_btn.mean_pos_x, start_btn.mean_pos_y)

        if not image_tools.check_startup(self):
            window_tools.click_at(start_btn.mean_pos_x, start_btn.mean_pos_y)

        auto_battle_tools.reset_auto_battle_watchdog(self)
        while self.main_loop_running and (True):
            _ensure_within_run_deadline(self, "waiting for tag team battle result")
            auto_battle_tools.ensure_auto_battle_running(self)
            try:
                finished = image_tools.get_text_in_relative_area(
                    self.reader, self.window, self.search_areas["battle_finished"]
                )[0]
                finished_2 = image_tools.get_text_in_relative_area(
                    self.reader, self.window, self.search_areas["battle_finished"]
                )[0]

                if self.resembles(finished.text, "PULSA PARA CONTINUAR") and self.resembles(finished_2.text, "PULSA PARA CONTINUAR"):
                    time.sleep(3)
                    if not self.update_battle_outcome(enemy_record):
                        continue

                    for _ in range(2):
                        window_tools.click_at(
                            finished.mean_pos_x, finished.mean_pos_y
                        )
                        time.sleep(1)

                    window_tools.click_center(
                        self.window, self.search_areas["close_encounter"]
                    )
                    return
            except Exception:
                pass

            time.sleep(3)

    # ------------------------------------------------------------------
    # Enemy evaluation
    # ------------------------------------------------------------------

    def _parse_enemy_power_value(self, text):
        text = text.replace(".", "").replace(",", ".").replace(" ", "")
        matches = re.findall(r"(\d[\d.,]*)([a-zA-Z]*)", text)
        if not matches:
            raise ValueError("No numeric value found")

        number, suffix = matches[-1]
        value = float(number)

        suffix = suffix.lower()
        if suffix.startswith("k"):
            value *= 1_000
        elif suffix.startswith("m"):
            value *= 1_000_000

        return value

    def _luchar_slot_key(self, obj):
        return (int(round(float(obj.mean_pos_x) / 14.0)), int(round(float(obj.mean_pos_y) / 14.0)))

    def _prune_recently_skipped_luchar_slots(self):
        now = time.time()
        cooldown = float(self.skip_luchar_cooldown_seconds)
        stale = [key for key, ts in self.recently_skipped_luchar_slots.items() if (now - float(ts)) > cooldown]
        for key in stale:
            self.recently_skipped_luchar_slots.pop(key, None)

    def _mark_luchar_slot_skipped(self, slot_key):
        self.recently_skipped_luchar_slots[slot_key] = time.time()

    def _slot_recently_skipped(self, slot_key):
        self._prune_recently_skipped_luchar_slots()
        ts = self.recently_skipped_luchar_slots.get(slot_key)
        if ts is None:
            return False
        return (time.time() - float(ts)) <= float(self.skip_luchar_cooldown_seconds)

    def evaluate_tagteam_enemies(self):
        text_objects = image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            search_area=self.search_areas["luchar_area"],
            power_detection=False,
        )

        filtered = image_tools.filter_text_objects(text_objects)
        scanned_slots = set()

        for idx, obj in enumerate(filtered):

            self.ensure_arena_coins()
            if self.no_coin_status:
                break

            if obj.text.strip() != "Luchar":
                continue
            slot_key = self._luchar_slot_key(obj)
            if slot_key in scanned_slots:
                continue
            scanned_slots.add(slot_key)
            if self._slot_recently_skipped(slot_key):
                continue
            list_card_screenshot = pyautogui.screenshot(
                region=(
                    int(obj.mean_pos_x - 540),
                    int(obj.mean_pos_y - 65),
                    440,
                    130,
                )
            )
            image_np = np.array(list_card_screenshot).astype(np.float32)
            window_tools.click_at(obj.mean_pos_x, obj.mean_pos_y)
            window_tools.click_center(self.window, self.search_areas["pov"])
            start_button = None
            try:
                start_button = image_tools.get_text_in_relative_area(
                    self.reader,
                    self.window,
                    search_area=self.search_areas["start_battle"],
                    power_detection=False,
                )[0]
            except Exception:
                start_button = None

            power_obj = image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            search_area=self.search_areas["enemy_total_power_value"],
            power_detection=False,
        )
            power_team1 = image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            search_area=self.search_areas["enemy_team1_power_value"],
            power_detection=False,
        )
            power_team2 = image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            search_area=self.search_areas["enemy_team2_power_value"],
            power_detection=False,
        )
            power_team3 = image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            search_area=self.search_areas["enemy_team3_power_value"],
            power_detection=False,
        )
            try:
                try:
                    enemy_power = self._parse_enemy_power_value(power_obj[0].text)
                    enemy_power_team1 = self._parse_enemy_power_value(power_team1[0].text)
                    enemy_power_team2 = self._parse_enemy_power_value(power_team2[0].text)
                    enemy_power_team3 = self._parse_enemy_power_value(power_team3[0].text)
                except:
                    enemy_power = 10
                    enemy_power_team1 = 10
                    enemy_power_team2 = 10
                    enemy_power_team3 = 10

                enemy_power_collection = [enemy_power, enemy_power_team1, enemy_power_team2, enemy_power_team3]
                enemy_record = self._build_enemy_composition_record(
                    image_np,
                    enemy_power_collection,
                    label="loss",
                )
                #print(f"Team1: {enemy_power_team1} Team2:{enemy_power_team2} Team3: {enemy_power_team3}  Total:{enemy_power}")
                if self._should_attack_enemy(enemy_record, enemy_power):
                    self.execute_tagteam_battle(enemy_record, start_button=start_button)
                    self.battles_done += 1
                    self.battle_occured = True
                    if self.update_dataset and self.dataset is not None:
                        self.dataset.append_entry(enemy_record, self.recent_battle_outcome)

                    outcome = "Win" if self.recent_battle_outcome else "Loss"
                    print(f"Battle outcome: {outcome}")

                    return True
                else:
                    self._mark_luchar_slot_skipped(slot_key)
                    window_tools.sendkey("esc", delay=5, window=self.window)

            except Exception as e:
                print(f"[!] Error parsing Power Object")
                self._mark_luchar_slot_skipped(slot_key)
                window_tools.sendkey("esc", delay=5, window=self.window)

        return False

    # ------------------------------------------------------------------
    # Arena utility
    # ------------------------------------------------------------------

    def refresh_enemy_list(self):
        if self.coords:
            window_tools.click_center(self.window, self.search_areas["refresh_timer"])
            self.recently_skipped_luchar_slots.clear()

    def ensure_arena_coins(self):
        window_tools.click_center(self.window, self.search_areas["pov"])
        self.no_coin_status = False
        time.sleep(1)

        for _ in range(3):  # number of retries
            try:
                keys = image_tools.get_text_in_relative_area(
                    self.reader, self.window, self.search_areas["arena_coins"]
                )

                # Filter only entries that contain "/"
                coins = [key for key in keys if "/" in key.text][0]
                break

            except (IndexError, AttributeError):
                time.sleep(0.5)

        if "0/" not in coins.text or coins.text == "10/10":
            return
        rel_left, rel_top, rel_width, rel_height = self.search_areas["add_arena_coins"]
        rel_left = coins.mean_pos_x / 1280 - 0.06
        abs_left = self.window.left + int(rel_left * self.window.width)
        abs_top = self.window.top + int(rel_top * self.window.height)
        abs_width = int(rel_width * self.window.width)
        abs_height = int(rel_height * self.window.height)
        center_x = abs_left + abs_width // 2
        center_y = abs_top + abs_height // 2

        pyautogui.click(center_x, center_y)
        time.sleep(3)

        confirm = image_tools.get_text_in_relative_area(
            self.reader, self.window, self.search_areas["confirm_add_arena_coins"]
        )[0]

        gems = image_tools.get_text_in_relative_area(
            self.reader, self.window, self.search_areas["confirm_use_gems"]
        )[0]

        try:
            gem_amount_text = image_tools.get_text_in_relative_area(
                self.reader, self.window, self.search_areas["gem_amount"]
            )[0].text
            numbers = re.findall(r"\d+", gem_amount_text)
            gem_cost = int("".join(numbers)) if numbers else 0
        except Exception:
            numbers = re.findall(r"\d+", gems.text)
            gem_cost = int("".join(numbers)) if numbers else 0

        if (
            (self.tagteam_arena_use_gems and gem_cost > self.tagteam_arena_use_gems_max_amount)
            or (not self.tagteam_arena_use_gems and gem_cost > 0)
        ):
            pyautogui.click(center_x, center_y)
            time.sleep(3)
            self.no_coin_status = True
            return

        window_tools.click_at(confirm.mean_pos_x, confirm.mean_pos_y)

    # ------------------------------------------------------------------
    # Status & main loops
    # ------------------------------------------------------------------

    def report_run_status(self):
        elapsed = str(timedelta(seconds=int(time.time() - self.init_time)))
        wins = self.battles_done - len(self.tagteam_arena_enemies_lost) + self.offset_wins
        medals = wins * 4

        print("\n" + "=" * 40)
        print("ðŸ›¡ï¸ RAID TagTeam Arena Bot Status")
        print("-" * 40)
        print(f"â±ï¸ Runtime: {elapsed}")
        print(f"âš”ï¸ Wins: {wins}")
        print(f"âŒ Losses: {len(self.tagteam_arena_enemies_lost) - self.offset_wins}")
        print(f"ðŸŽ–ï¸ Estimated Medals: {medals}")
        print("=" * 40)

    def run_tagteam_arena_continuous(self, max_run_duration_seconds=MAX_RUN_DURATION_SECONDS):
        _start_run_deadline(self, max_run_duration_seconds)
        self.recently_skipped_luchar_slots.clear()
        time.sleep(5)
        last_refresh = time.time()
        refresh_count = 0

        while self.main_loop_running and (self.running):
            _ensure_within_run_deadline(self, "running tag team arena continuous loop")
            self.report_run_status()
            self.ensure_arena_coins()

            if self.no_coin_status:
                print("Waiting for coins")
                break

            if self.evaluate_tagteam_enemies():
                continue

            window_tools.move_down(self.window)
            if self.evaluate_tagteam_enemies():
                continue
            window_tools.move_up(self.window)

            if self.tagteam_arena_multi_refresh:
                if refresh_count < self.tagteam_arena_num_multi_refresh:
                    self.refresh_enemy_list()
                    refresh_count += 1
                    continue
                refresh_count = 0

            print("Waiting for free refresh")
            time.sleep(62)

            if time.time() - last_refresh >= self.refresh_minutes * 60:
                self.refresh_enemy_list()
                last_refresh = time.time()

    def run_tagteam_arena_single_cycle(
        self,
        main_loop_running=True,
        max_run_duration_seconds=MAX_RUN_DURATION_SECONDS,
    ):
        _start_run_deadline(self, max_run_duration_seconds)
        self.recently_skipped_luchar_slots.clear()
        self.main_loop_running = main_loop_running
        time.sleep(5)
        self.running = True

        while self.main_loop_running and (self.running):
            _ensure_within_run_deadline(self, "running tag team arena single cycle")
            self.report_run_status()
            self.ensure_arena_coins()

            if self.no_coin_status:
                print("Waiting for coins")
                break

            if self.evaluate_tagteam_enemies():
                continue

            window_tools.move_down(self.window)
            if self.evaluate_tagteam_enemies():
                continue
            window_tools.move_up(self.window)

            print("Finished one cycle")
            break
    
    
    
# =============================================================================
#   LIVE ARENA BOT 
# =============================================================================

class RSL_Bot_LiveArena:
    
    def __init__(self, title_substring="Raid: Shadow Legends", reader = None, window =None, verbose = True, use_gems = True, use_gems_max_amount = 0, memory = dict()):

        if reader is None:
            print('Error When Loading Reader')
            
        self.reader = reader
        
        self.running = True
        
        self.battles_done = 0
        self.battles_won = 0
        self.no_coin_status = False
        
        self.verbose = verbose
        self.live_arena_memory = memory
        self.live_arena_use_gems = use_gems
        self.live_arena_use_gems_max_amount = use_gems_max_amount
        self.window = window
        self.init_time = time.time()
        
        self.battle_status = 'menu'
        self.auto_button_clicked = False
        self._pausa_esc_sent = False
        self.max_run_duration_seconds = MAX_RUN_DURATION_SECONDS
        self._run_deadline = None
        
        
        if self.window:
            self.coords = (self.window.left, self.window.top, self.window.width, self.window.height)
            print(f"Window Coordinates: {self.coords}")
        else:
            self.coords = None
            
        # Search Areas
        self.search_areas = {
            
            "live_arena_status":   [0.203, 0.834, 0.536, 0.04], #many red or green points in one spot
            'pov' : [0, 0, 1, 1],
            
            "live_arena_coins":   [0.6, 0.036, 0.4, 0.03],
            "live_add_arena_coins":   [0.702, 0.039, 0.026, 0.029],
            "live_confirm_add_arena_coins":   [0.394, 0.615, 0.208, 0.083],
            "live_amount_gems":   [0.498, 0.658, 0.032, 0.026],
            "live_confirm_use_gems":   [0.395, 0.642, 0.209, 0.089],
            "gem_amount":   [0.494, 0.659, 0.039, 0.024],
            
            "live_arena_reward_1":   [0.924, 0.137, 0.042, 0.056],
            "live_arena_reward_2":   [0.924, 0.201, 0.041, 0.054],
            "live_arena_reward_3":   [0.792, 0.293, 0.044, 0.056],
            "live_arena_reward_4":   [0.858, 0.293, 0.042, 0.053],
            "live_arena_reward_5":   [0.922, 0.292, 0.044, 0.053],
    
            "start_encounter":     [0.34, 0.884, 0.258, 0.086],
            "encounter_status":  [0.363, 0.076, 0.268, 0.038],
            "pick_status":    [0.412, 0.476, 0.173, 0.084],
            "champion_roster_complete": [0.104, 0.758, 0.611, 0.231],
            "turn_counter_roster": [0.423, 0.076, 0.084, 0.03],
            
            "team_roster_1": [0.242, 0.333, 0.076, 0.122],
            "team_roster_2": [0.195, 0.47, 0.079, 0.123],
            "team_roster_3": [0.152, 0.332, 0.076, 0.125],
            "team_roster_4": [0.106, 0.469, 0.078, 0.124],
            "team_roster_5": [0.062, 0.332, 0.079, 0.124],
            
            "enemy_roster_1": [0.674, 0.33, 0.086, 0.133],
            "enemy_roster_2": [0.718, 0.464, 0.086, 0.136],
            "enemy_roster_3": [0.765, 0.328, 0.084, 0.135],
            "enemy_roster_4": [0.815, 0.471, 0.079, 0.124],
            "enemy_roster_5": [0.86, 0.333, 0.079, 0.124],
            
            "preset_champion_1": [0.107, 0.762, 0.046, 0.074],
            "preset_champion_2": [0.106, 0.839, 0.049, 0.074],
            "preset_champion_3": [0.106, 0.914, 0.047, 0.075],
            "preset_champion_4": [0.157, 0.763, 0.046, 0.071],
            "preset_champion_5": [0.158, 0.84, 0.046, 0.071],
            "preset_champion_6": [0.159, 0.918, 0.046, 0.071],
            "preset_champion_7": [0.21, 0.764, 0.044, 0.071],
            "preset_champion_8": [0.21, 0.841, 0.046, 0.071],
            
            "confirm_button_champion_selection": [0.762, 0.876, 0.213, 0.104],
            
            "auto_battle_button": [0.026, 0.899, 0.058, 0.07], # its existance means battle started
            "battle_status_finished":  [0.362, 0.897, 0.269, 0.081], # check regularly for if enemy quits
            "battle_result":    [0.389, 0.148, 0.204, 0.071],
            "battle_result_2":    [0.38, 0.085, 0.224, 0.059],
    
            
            "test":   [0.05, 0.30, 0.15, 0.08],
            
        }
        
        
    # ------------------------- Reset Methods -------------------------
    def reset_battle_state(self):
        self.battle_status = 'menu'
        self.auto_button_clicked = False
        self.no_coin_status = False
        self._pausa_esc_sent = False
        auto_battle_tools.reset_auto_battle_watchdog(self)

    # ------------------------- Battle Outcome -------------------------
    def _read_battle_result_once(self):
        for result_area in ("battle_result", "battle_result_2"):
            try:
                battle_results = image_tools.get_text_in_relative_area(
                    self.reader, self.window, search_area=self.search_areas[result_area]
                )
            except Exception:
                continue

            for battle_result in battle_results:
                text = (getattr(battle_result, "text", "") or "").strip()
                if not text:
                    continue
                if self.resembles(text, "VICTORIA"):
                    return "VICTORIA"
                if self.resembles(text, "DERROTA"):
                    return "DERROTA"

        try:
            pov_results = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas["pov"]
            )
            for pov_result in pov_results:
                text = (getattr(pov_result, "text", "") or "").strip()
                if text and self.resembles(text, "Pausa"):
                    return "PAUSA"
        except Exception:
            pass

        return None

    def update_battle_outcome(self):
        first_result = self._read_battle_result_once()
        if first_result is None:
            return

        time.sleep(10)
        second_result = self._read_battle_result_once()
        if second_result is None or first_result != second_result:
            if self.verbose:
                print(
                    "Live Arena battle result mismatch between checks. "
                    f"First='{first_result}', second='{second_result}'."
                )
            return

        if first_result == "PAUSA":
            if not self._pausa_esc_sent:
                window_tools.sendkey("esc", delay=0.2, window=self.window)
                self._pausa_esc_sent = True
            return

        self._pausa_esc_sent = False
        self.battle_status = 'Done'
        self.battles_done += 1
        if first_result == "VICTORIA":
            self.battles_won += 1
        return

    # ------------------------- Enemy Memory -------------------------
    def persist_enemy_avoid_list(self):
        # Not used currently
        pass

    def resembles(self, text, target, threshold=0.8):
        ratio = difflib.SequenceMatcher(None, text.lower(), target.lower()).ratio()
        return ratio >= threshold

    # ------------------------- Battle Status -------------------------
    def update_battle_activity_status(self):
        try:
            auto_button = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas["auto_battle_button"]
            )[0]
            if self.resembles(auto_button.text, 'Auto'):
                self.battle_status = 'Battle active'
                time.sleep(3)
                if not self.auto_button_clicked:
                    window_tools.click_center(self.window, self.search_areas["auto_battle_button"])
                    self.auto_button_clicked = True
                    window_tools.click_center(self.window, self.search_areas["live_arena_status"])
            else:
                self.battle_status = 'Battle inactive'
        except:
            pass

    # ------------------------- Arena Coins -------------------------
    def ensure_arena_coins(self):
        window_tools.click_center(self.window, self.search_areas["pov"])
        time.sleep(1)
        self.no_coin_status = False

        for _ in range(3):  # number of retries
            try:
                keys = image_tools.get_text_in_relative_area(
                    self.reader, self.window, self.search_areas["live_arena_coins"]
                )
                
                        # Fix OCR cases where 3-digit key means the 2nd digit should be "/"
                for key in keys:
                    if len(key.text) == 3 and key.text.isdigit():
                        key.text = f"{key.text[0]}/{key.text[2]}"

                # Filter only entries that contain "/"
                coins_text = [key for key in keys if "/" in key.text][0]
                break

            except (IndexError, AttributeError):
                time.sleep(0.5)

        if "0/" in coins_text.text and coins_text.text != '5/5':
            # Calculate absolute center coordinates
            rel_left, rel_top, rel_width, rel_height = self.search_areas["live_add_arena_coins"]
            rel_left = coins_text.mean_pos_x/1280 - 0.06 
            abs_left = self.window.left + int(rel_left * self.window.width)
            abs_top = self.window.top + int(rel_top * self.window.height)
            abs_width = int(rel_width * self.window.width)
            abs_height = int(rel_height * self.window.height)
            center_x = abs_left + abs_width // 2
            center_y = abs_top + abs_height // 2

            pyautogui.click(center_x, center_y)
            time.sleep(10)

            #confirm_text = image_tools.get_text_in_relative_area(
            #    self.reader, self.window, self.search_areas["live_confirm_add_arena_coins"]
            #)[0]

            # Check gem usage
            try:
                gem_amount = image_tools.get_text_in_relative_area(
                    self.reader, self.window, self.search_areas["gem_amount"]
                )[0].text
                gem_amount = int("".join(re.findall(r"\d+", gem_amount))) if gem_amount else 0
            except:
                gem_amount = 0

            if (self.live_arena_use_gems and gem_amount > self.live_arena_use_gems_max_amount) or (not self.live_arena_use_gems and gem_amount > 0):
                pyautogui.click(center_x, center_y)
                time.sleep(3)
                self.no_coin_status = True
                return

            time.sleep(3)
            window_tools.click_center(self.window, self.search_areas["live_confirm_add_arena_coins"])
            #window_tools.click_at(confirm_text.mean_pos_x, confirm_text.mean_pos_y)

    # ------------------------- Status Print -------------------------
    def report_run_status(self):
        elapsed = int(time.time() - self.init_time)
        medals = self.battles_won * 70
        formatted_elapsed = str(timedelta(seconds=elapsed))

        print("\n" + "=" * 40)
        print("ðŸ›¡ï¸  RAID live Arena Bot Status")
        print("-" * 40)
        print(f"ðŸ” Mode: Simple Pick")
        print(f"â±ï¸  Time Since Start: {formatted_elapsed}")
        print(f"âš”ï¸  Battles Won: {self.battles_won}")
        print(f"âš”ï¸  Battles Lost: {self.battles_done - self.battles_won}")
        print(f"ðŸŽ–ï¸  Estimated Medals: {medals}")
        print("-" * 40)
        print("ðŸ›‘ To stop the bot, press 'v'")
        print("=" * 40 + "\n")

    # ------------------------- Live Arena -------------------------
    def is_live_arena_active(self):
        rel_left, rel_top, rel_width, rel_height = self.search_areas["live_arena_status"]
        x = int(self.window.left + rel_left * self.window.width)
        y = int(self.window.top + rel_top * self.window.height)
        w = int(rel_width * self.window.width)
        h = int(rel_height * self.window.height)

        result = image_tools.detect_red_or_green_circle_stable(
            region_coords=(x, y, w, h),
            samples=50,
            required_ratio=0.8,
            min_pixels=10,
            tolerance=50
        )
        return result != "red" if result else False

    def claim_live_arena_rewards(self):
        for reward in ["live_arena_reward_1", "live_arena_reward_2", "live_arena_reward_3", "live_arena_reward_4", "live_arena_reward_5"]:
            # Click twice per reward
            for _ in range(2):
                window_tools.click_center(self.window, self.search_areas[reward], delay=1)

    # ------------------------- Pick Phase -------------------------
    def execute_simple_pick_phase(self):
        try:
            confirm_button = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas["confirm_button_champion_selection"]
            )[0]
            turn_counter = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas["turn_counter_roster"]
            )[0]

            if self.resembles(confirm_button.text, 'Confirmar') and self.resembles(turn_counter.text, 'Tu turno'):
                for i in range(1, 9):
                    window_tools.click_center(self.window, self.search_areas[f"preset_champion_{i}"], delay=1)
                window_tools.click_center(self.window, self.search_areas["confirm_button_champion_selection"], delay=1)
        except:
            pass

    def execute_complex_pick_phase(self):
        pass

    # ------------------------- Encounter -------------------------
    def execute_live_arena_encounter(self):
        self.reset_battle_state()
        window_tools.click_center(self.window, self.search_areas["start_encounter"])
        if not image_tools.check_startup(self):
            window_tools.click_center(self.window, self.search_areas["start_encounter"])

        while self.main_loop_running and (self.battle_status != 'Done'):
            _ensure_within_run_deadline(self, "running live arena encounter")
            self.update_battle_outcome()
            self.execute_simple_pick_phase()
            self.update_battle_activity_status()
            if self.battle_status in ("Battle active", "Battle inactive"):
                auto_battle_tools.ensure_auto_battle_running(self)

        while self.main_loop_running and (self.battle_status == 'Done'):
            _ensure_within_run_deadline(self, "closing live arena result screen")
            battle_finished = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas['battle_status_finished']
            )[0]
            if self.resembles(battle_finished.text, "VOLVER A LA ARENA"):
                window_tools.click_center(self.window, self.search_areas["battle_status_finished"])
                self.battle_status = 'menu'

    # ------------------------- Main Live Arena Loop -------------------------
    def run_live_arena_loop(
        self,
        main_loop_running=True,
        max_run_duration_seconds=MAX_RUN_DURATION_SECONDS,
    ):
        _start_run_deadline(self, max_run_duration_seconds)
        self.main_loop_running = main_loop_running
        time.sleep(5)
        self.start_time = time.time()
        self.running = True
        time.sleep(5)

        while self.main_loop_running and (self.running):
            _ensure_within_run_deadline(self, "running live arena loop")
            if not self.is_live_arena_active():
                self.running = False
                print("Live arena not active")
                continue

            self.claim_live_arena_rewards()
            self.ensure_arena_coins()
            self.battle_occured = False

            if self.no_coin_status:
                self.running = False
            else:
                self.execute_live_arena_encounter()

            self.report_run_status()
