import time
import re
from datetime import  timedelta
from statistics import median
import raid_bot.utils.auto_battle_tools as auto_battle_tools
import raid_bot.utils.image_tools as image_tools
import raid_bot.utils.window_tools as window_tools
import difflib

MAX_RUN_DURATION_SECONDS = int(3.5 * 60 * 60)
FW_STAGE_SCROLL_STEP = 0.2
FW_STAGE_EDGE_THRESHOLD = 0.88
FW_STAGE_DEFAULT_SPACING_RATIO = 0.1
DEFAULT_FW_FACTIONS = [
    "Banner Lords",
    "Barbarians",
    "Dark Elves",
    "Demonspawn",
    "Dwarves",
    "High Elves",
    "Knight Revenant",
    "Lizardmen",
    "Ogryn Tribes",
    "Orcs",
    "Sacred Order",
    "Undead Hordes",
    "Shadowkin",
    "Skinwalkers",
    "Sylvan Watchers",
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


class RSL_Bot_FactionWars:
    
    def __init__(self, title_substring="Raid: Shadow Legends", reader = None, window =None, verbose = True, farm_stages = {"Banner Lords":[17,"normal"],"Barbarians":[17,"normal"],"Dark Elves":[17,"normal"],"Demonspawn":[17,"normal"],"Dwarves":[17,"normal"],"High Elves":[17,"normal"],"Knight Revenant":[17,"normal"],"Lizardmen":[17,"normal"],"Ogryn Tribes":[17,"normal"],"Orcs":[17,"normal"],"Sacred Order":[17,"normal"],"Undead Hordes":[17,"normal"],"Shadowkin":[17,"normal"],"Skinwalkers":[17,"normal"],"Sylvan Watchers":[17,"normal"]}, farm_superraid = True, progress_mode_factions = None, progress_mode = None):

        if reader is None:
            print('Error When Loading Reader')
            
        self.reader = reader
        
        self.running = True
        
        self.battles_done = 0
        self.battles_won = 0
        self.no_coin_status = False
        
        self.verbose = verbose
        self.farm_stages = farm_stages
        self.farm_superraid = farm_superraid
        self.progress_mode_factions = progress_mode_factions
        self.progress_mode = progress_mode
        self.multiplier = 1
        if self.farm_superraid:
            self.multiplier+=1
        self.progress_failed_factions_runtime = set()
        self.current_faction_key = None
        self.current_base_stage = None
        self.progress_attempt_active = False
        self.last_battle_result = None
        self.persist_stage_update_callback = None

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
            
            "faction_wars_keys":   [0.50, 0.036, 0.4, 0.04],
            "faction_name":   [0.01, 0.033, 0.448, 0.046],
            'pov' : [0, 0, 1, 1],
            "go_to_higher_menu":   [0.928, 0.031, 0.046, 0.039],
            
            "confirm_button_champion_selection": [0.762, 0.876, 0.213, 0.104],
            
            "auto_battle_button": [0.026, 0.899, 0.058, 0.07], # its existance means battle started
            "battle_status_finished":  [0.362, 0.897, 0.269, 0.081], # check regularly for if enemy quits
            "battle_result":    [0.389, 0.148, 0.204, 0.071],
            "battle_result_2":    [0.38, 0.085, 0.224, 0.059],
            "restart_encounter":   [0.423, 0.877, 0.211, 0.106],

            'get_difficulty':[0.032, 0.903, 0.166, 0.051],
            'change_difficulty_normal':[0.062, 0.778, 0.117, 0.047],
            'change_difficulty_hard':[0.046, 0.841, 0.139, 0.056],

            "faction_wars_farm_encounter": [0.763, 0.764, 0.211, 0.105],
            "faction_wars_start_multibattles": [0.254, 0.634, 0.23, 0.075],
            "faction_wars_multibattles_setup_1": [0.222, 0.458, 0.032, 0.04],
            "faction_wars_multibattles_setup_2": [0.221, 0.502, 0.034, 0.045],
            "faction_wars_farming_status": [0.366, 0.609, 0.269, 0.102],

            "faction_wars_etapa_window": [0.3, 0, 0.2, 1],


            'go_to_map': [0.134, 0.905, 0.059, 0.071],

            
            "test":   [0.05, 0.30, 0.15, 0.08],
            
        }

        self.faction_menu_names = {
            'Banner Lords': 'Hidalgos',
            'Barbarians': 'Barbaros',
            'Dark Elves': 'Elfos Oscuros',
            'Demonspawn': 'Engendros',
            'Dwarves': 'Enanos',
            'High Elves': 'Altos Elfos',
            'Knight Revenant': 'Aparecidos',
            'Lizardmen': "H. Lagarto",
            'Ogryn Tribes': 'Ogretes',
            'Orcs': 'Orcos',
            'Sacred Order': 'Orden Sagrada',
            'Undead Hordes': 'No Muertos',
            'Shadowkin': 'Cripta de Sombrios',
            'Skinwalkers': 'Cambiapieles',
            'Sylvan Watchers': 'Cripta de Vigias Silvanos',
            'hard': "Dificil",
            "normal": "Normal"
        }
        self._refresh_progress_mode_faction_set()
        
        self.current_difficulty = 'normal'

        self.stages_buttons = [[0.804, 0.114, 0.163, 0.083],
                               [0.785, 0.192, 0.176, 0.078],
                               [0.785, 0.311, 0.176, 0.074],
                               [0.785, 0.429, 0.179, 0.079],
                               [0.782, 0.546, 0.182, 0.081],
                               [0.784, 0.663, 0.181, 0.081],
                               [0.784, 0.784, 0.18, 0.078],
                               [0.783, 0.898, 0.181, 0.082],
                               ]
        self.stages_buttons_hard = [[0.787, 0.083, 0.177, 0.071],
                    [0.804, 0.232, 0.159, 0.082],
                    [0.815, 0.359, 0.145, 0.072],
                    [0.815, 0.5, 0.145, 0.072],
                    ]

    def _max_stage_for_difficulty(self, difficulty):
        return 21 if difficulty == "hard" else 21

    def _clear_progress_context(self):
        self.current_faction_key = None
        self.current_base_stage = None
        self.progress_attempt_active = False

    def _refresh_progress_mode_faction_set(self):
        configured = self.progress_mode_factions

        if configured is None:
            configured = list(DEFAULT_FW_FACTIONS)
        elif isinstance(configured, dict):
            configured = [name for name, enabled in configured.items() if bool(enabled)]
        elif not isinstance(configured, (list, tuple, set)):
            configured = []

        normalized = []
        for faction_name in configured:
            if not isinstance(faction_name, str):
                continue
            cleaned = faction_name.strip()
            if cleaned:
                normalized.append(cleaned)

        if self.progress_mode is False:
            normalized = []
        elif self.progress_mode is True and self.progress_mode_factions is None:
            normalized = list(DEFAULT_FW_FACTIONS)

        self.progress_mode_factions = normalized
        self.progress_mode_faction_set = set(normalized)

    def _should_try_progress_stage(self, faction_key, configured_stage, configured_difficulty):
        if faction_key not in self.progress_mode_faction_set:
            return False
        if configured_difficulty != "hard":
            return False
        if faction_key in self.progress_failed_factions_runtime:
            return False
        return configured_stage < self._max_stage_for_difficulty(configured_difficulty)

    def _build_relative_area_around_abs_point(self, abs_x, abs_y, rel_width=0.17, rel_height=0.08):
        rel_center_x = (abs_x - self.window.left) / self.window.width
        rel_center_y = (abs_y - self.window.top) / self.window.height

        rel_left = max(0.0, min(1.0 - rel_width, rel_center_x - rel_width / 2))
        rel_top = max(0.0, min(1.0 - rel_height, rel_center_y - rel_height / 2))
        return [rel_left, rel_top, rel_width, rel_height]

    def _extract_stage_number(self, raw_text):
        if not raw_text:
            return None
        normalized = str(raw_text).replace("O", "0").replace("o", "0")
        matches = re.findall(r"\d+", normalized)
        if not matches:
            return None
        try:
            return int(matches[0])
        except ValueError:
            return None

    def _collect_visible_stage_candidates(self):
        try:
            stage_objects = image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                self.search_areas["faction_wars_etapa_window"],
            )
        except Exception:
            return []

        stage_candidates = []
        for obj in stage_objects:
            stage_number = self._extract_stage_number(getattr(obj, "text", ""))
            mean_pos_x = getattr(obj, "mean_pos_x", None)
            mean_pos_y = getattr(obj, "mean_pos_y", None)
            if stage_number is None or mean_pos_x is None or mean_pos_y is None:
                continue
            stage_candidates.append(
                {
                    "stage": int(stage_number),
                    "obj": obj,
                    "x": float(mean_pos_x),
                    "y": float(mean_pos_y),
                }
            )

        stage_candidates.sort(key=lambda item: item["y"])
        return stage_candidates

    def _estimate_stage_spacing(self, stage_candidates):
        if len(stage_candidates) < 2:
            return None

        spacing_samples = []
        for previous, current in zip(stage_candidates, stage_candidates[1:]):
            stage_gap = current["stage"] - previous["stage"]
            y_gap = current["y"] - previous["y"]
            if stage_gap <= 0 or y_gap <= 0:
                continue
            spacing_samples.append(y_gap / stage_gap)

        if spacing_samples:
            return median(spacing_samples)

        return None

    def _infer_visible_stage_point(self, stage_candidates, target_stage):
        if not stage_candidates:
            return None

        target_stage = int(target_stage)
        direct_match = next((item for item in stage_candidates if item["stage"] == target_stage), None)
        if direct_match:
            return {
                "x": direct_match["x"],
                "y": direct_match["y"],
                "kind": "direct",
                "inferred": False,
            }

        spacing = self._estimate_stage_spacing(stage_candidates)
        if spacing is None:
            spacing = float(self.window.height) * FW_STAGE_DEFAULT_SPACING_RATIO

        if spacing <= 0:
            return None

        stages = [item["stage"] for item in stage_candidates]
        for previous, current in zip(stage_candidates, stage_candidates[1:]):
            lower_stage = previous["stage"]
            upper_stage = current["stage"]
            if lower_stage < target_stage < upper_stage:
                stage_gap = upper_stage - lower_stage
                if stage_gap <= 0:
                    continue
                local_spacing = (current["y"] - previous["y"]) / stage_gap if stage_gap else spacing
                inferred_x = (previous["x"] + current["x"]) / 2.0
                inferred_y = previous["y"] + local_spacing * (target_stage - lower_stage)
                return {
                    "x": inferred_x,
                    "y": inferred_y,
                    "kind": "between",
                    "inferred": True,
                }

        if target_stage < min(stages):
            anchor = stage_candidates[0]
            inferred_y = anchor["y"] - spacing * (anchor["stage"] - target_stage)
            return {
                "x": anchor["x"],
                "y": inferred_y,
                "kind": "outside",
                "inferred": True,
            }

        if target_stage > max(stages):
            anchor = stage_candidates[-1]
            inferred_y = anchor["y"] + spacing * (target_stage - anchor["stage"])
            return {
                "x": anchor["x"],
                "y": inferred_y,
                "kind": "outside",
                "inferred": True,
            }

        return None

    def _select_stage_button_area_dynamic(self, target_stage, max_scroll_attempts=8):
        target_stage = int(target_stage)
        for _ in range(max_scroll_attempts):
            stage_candidates = self._collect_visible_stage_candidates()
            if not stage_candidates:
                window_tools.move_down(self.window, strength=FW_STAGE_SCROLL_STEP)
                continue

            for stage_candidate in stage_candidates:
                if stage_candidate["stage"] == target_stage:
                    return self._build_relative_area_around_abs_point(
                        stage_candidate["x"],
                        stage_candidate["y"],
                    )

            stage_values = [candidate["stage"] for candidate in stage_candidates]
            visible_target = self._infer_visible_stage_point(stage_candidates, target_stage)
            if visible_target is not None and visible_target.get("kind") in {"direct", "between"}:
                rel_y = (visible_target["y"] - self.window.top) / float(self.window.height)
                if 0.08 <= rel_y <= 0.92:
                    return self._build_relative_area_around_abs_point(
                        visible_target["x"],
                        visible_target["y"],
                    )

            highest_stage = max(stage_values)
            lowest_stage = min(stage_values)
            highest_y_rel = (stage_candidates[0]["y"] - self.window.top) / float(self.window.height)

            if target_stage > highest_stage:
                window_tools.move_down(self.window, strength=FW_STAGE_SCROLL_STEP)
                continue

            if target_stage < lowest_stage:
                window_tools.move_up(self.window, strength=FW_STAGE_SCROLL_STEP)
                continue

            if visible_target is not None and visible_target["y"] > self.window.top + self.window.height * FW_STAGE_EDGE_THRESHOLD:
                window_tools.move_down(self.window, strength=FW_STAGE_SCROLL_STEP)
                continue

            if visible_target is not None and visible_target["y"] < self.window.top + self.window.height * (1.0 - FW_STAGE_EDGE_THRESHOLD):
                window_tools.move_up(self.window, strength=FW_STAGE_SCROLL_STEP)
                continue

            if highest_y_rel > FW_STAGE_EDGE_THRESHOLD and target_stage >= highest_stage:
                window_tools.move_down(self.window, strength=FW_STAGE_SCROLL_STEP)
                continue

        return None

    def _select_stage_button_area_legacy_fallback(self):
        if self.current_difficulty == "hard":
            if self.current_stage >= 15:
                hard_index = self.current_stage - 14
            else:
                hard_index = self.current_stage
            hard_index = max(1, min(int(hard_index), len(self.stages_buttons_hard))) - 1
            return self.stages_buttons_hard[hard_index]

        normal_index = max(1, min(int(self.current_stage), len(self.stages_buttons))) - 1
        return self.stages_buttons[normal_index]

    def _persist_won_progress_stage(self, faction_key, stage, difficulty):
        farm_data = self.farm_stages.get(faction_key)
        if isinstance(farm_data, (list, tuple)) and len(farm_data) >= 2:
            self.farm_stages[faction_key][0] = int(stage)
            self.farm_stages[faction_key][1] = difficulty
        else:
            self.farm_stages[faction_key] = [int(stage), difficulty]

        callback = getattr(self, "persist_stage_update_callback", None)
        if callable(callback):
            try:
                callback(faction_key, int(stage), difficulty)
            except Exception as exc:
                print(f"Failed to persist faction wars stage update for {faction_key}: {exc}")

    def _resolve_progress_result(self):
        if not self.progress_attempt_active or not self.current_faction_key:
            self._clear_progress_context()
            return

        attempted_stage = self.current_stage
        if self.last_battle_result == "victory":
            self._persist_won_progress_stage(
                self.current_faction_key,
                attempted_stage,
                self.current_difficulty,
            )
            print(
                f"Progress mode: {self.current_faction_key} advanced to stage {attempted_stage} ({self.current_difficulty})."
            )
        elif self.last_battle_result == "defeat":
            self.progress_failed_factions_runtime.add(self.current_faction_key)
            self.current_stage = self.current_base_stage
            print(
                f"Progress mode: {self.current_faction_key} failed stage {attempted_stage}. "
                f"Falling back to stage {self.current_base_stage} for this runtime."
            )

        self._clear_progress_context()
        
        
    # ------------------------- Reset Methods -------------------------
    def reset_battle_state(self):
        self.battle_status = 'menu'
        self.last_battle_result = None
        self._pausa_esc_sent = False
        auto_battle_tools.reset_auto_battle_watchdog(self)

    def _read_battle_result_once(self):
        for result_area in ("battle_result", "battle_result_2"):
            try:
                text_objects = image_tools.get_text_in_relative_area(
                    self.reader,
                    self.window,
                    search_area=self.search_areas[result_area],
                )
            except Exception:
                continue

            for text_object in text_objects:
                result_text = (getattr(text_object, "text", "") or "").strip()
                if not result_text:
                    continue
                if self.resembles(result_text, "VICTORIA"):
                    return "VICTORIA"
                if self.resembles(result_text, "DERROTA"):
                    return "DERROTA"

        try:
            pov_objects = image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                search_area=self.search_areas["pov"],
            )
            for text_object in pov_objects:
                result_text = (getattr(text_object, "text", "") or "").strip()
                if result_text and self.resembles(result_text, "Pausa"):
                    return "PAUSA"
        except Exception:
            pass

        return None

    def resembles(self, text, target, threshold=0.8):
        ratio = difflib.SequenceMatcher(None, text.lower(), target.lower()).ratio()
        return ratio >= threshold

    # ------------------------- Difficulty -------------------------
    def ensure_correct_difficulty(self):
        try:
            difc_txt = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas["get_difficulty"]
            )[0]
            if difc_txt.text != self.faction_menu_names[self.current_difficulty]:
                window_tools.click_center(self.window, self.search_areas["get_difficulty"])
                string = f'change_difficulty_{self.current_difficulty}'
                window_tools.click_center(self.window, self.search_areas[string])
        except:
            print('Error changing Difficulties')

    # ------------------------- Battle Outcome -------------------------
    def update_battle_outcome(self):
        first_result = self._read_battle_result_once()
        if first_result is None:
            return

        time.sleep(10)
        second_result = self._read_battle_result_once()
        if second_result is None or first_result != second_result:
            if self.verbose:
                print(
                    "Faction Wars battle result mismatch between checks. "
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
            self.last_battle_result = "victory"
        else:
            self.last_battle_result = "defeat"
        return

    # ------------------------- Battle Status -------------------------
    def update_battle_activity_status(self):
        try:
            auto_button = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas["auto_battle_button"]
            )[0]
            self.battle_status = 'Battle active' if self.resembles(auto_button.text, 'Auto') else 'Battle inactive'
        except:
            pass

    # ------------------------- Status Print -------------------------
    def report_run_status(self):
        elapsed = int(time.time() - self.init_time)
        formatted_elapsed = str(timedelta(seconds=elapsed))
        medals = self.battles_won * 70

        print("\n" + "=" * 40)
        print("ðŸ›¡ï¸  RAID Faction Wars Bot Status")
        print("-" * 40)
        print(f"ðŸ” Mode: Simple Pick")
        print(f"â±ï¸  Time Since Start: {formatted_elapsed}")
        print(f"âš”ï¸  Battles Won: {self.battles_won}")
        print(f"âš”ï¸  Battles Lost: {self.battles_done - self.battles_won}")
        print("-" * 40)
        print("ðŸ›‘ To stop the bot, press 'v'")
        print("=" * 40 + "\n")

    # ------------------------- FW Keys -------------------------
    def get_available_fw_keys(self):
        try:
            keys = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas['faction_wars_keys']
            )
            fw_keys = [key for key in keys if "/" in key.text][0]
            fw_keys = re.findall(r"\d+", fw_keys.text)[0]
        except:
            fw_keys = 0
        return fw_keys

    # ------------------------- Fuzzy Matching -------------------------
    def match_faction_name_fuzzy(self, name, flat_values, cutoff=0.75):
        """Return closest match from flat_values or None if below cutoff."""
        matches = difflib.get_close_matches(name, flat_values, n=1, cutoff=cutoff)
        return matches[0] if matches else None

    # ------------------------- Encounter Selection -------------------------
    def locate_faction_encounter(self, max_attempts=6):
        obj_found = False
        attempts = 0
        flat_values = self.faction_menu_names.values()

        while self.main_loop_running and (attempts < max_attempts and not obj_found):
            _ensure_within_run_deadline(self, "searching faction wars encounter")
            attempts += 1
            time.sleep(2)

            objects = image_tools.get_text_in_relative_area(self.reader, self.window, self.search_areas['pov'])

            for obj in objects:
                if not self.main_loop_running:
                    break
                try:
                    if 'Cripta' not in obj.text:
                        continue

                    window_tools.click_at(
                        obj.mean_pos_x,
                        obj.mean_pos_y - int(0.05 * self.window.height),
                        delay=4
                    )

                    raw_faction = image_tools.get_text_in_relative_area(
                        self.reader, self.window, self.search_areas['faction_name'], power_detection=False
                    )[0]

                    faction_name = raw_faction.text.replace("Cripta: ", "")
                    faction_name_alt = raw_faction.text.replace("Cripta de ", "") if raw_faction.text else '____________'

                    if faction_name == 'Guerras de Facciones':
                        continue
                    else:
                        print(f"Detected faction: {faction_name}")

                    # Fuzzy match if not exact
                    if faction_name not in flat_values:
                        faction_name = self.match_faction_name_fuzzy(faction_name, flat_values)
                    if not faction_name and faction_name_alt not in flat_values:
                        faction_name = self.match_faction_name_fuzzy(faction_name_alt, flat_values)
                    if not faction_name:
                        print("Could not match faction_name, skipping this object.")
                        continue

                    # Find key in faction_menu_names
                    key = [
                        k for k, v in self.faction_menu_names.items()
                        if v == faction_name or (isinstance(v, list) and faction_name in v)
                    ]
                    if not key:
                        print("Matched faction_name but could not find corresponding key, skipping.")
                        continue

                    configured_stage = int(self.farm_stages[key[0]][0])
                    configured_difficulty = self.farm_stages[key[0]][1]
                    self.current_stage = configured_stage
                    self.current_difficulty = configured_difficulty
                    self.current_faction_key = key[0]
                    self.current_base_stage = configured_stage
                    self.progress_attempt_active = False

                    if self._should_try_progress_stage(key[0], configured_stage, configured_difficulty):
                        self.current_stage = configured_stage + 1
                        self.progress_attempt_active = True

                    current_fw_keys = self.get_available_fw_keys()
                    print(current_fw_keys)
                    if (int(current_fw_keys) < int(3 * self.multiplier) and self.current_difficulty == 'normal') or \
                       (int(current_fw_keys) < int(5 * self.multiplier) and self.current_difficulty == 'hard'):
                        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
                        continue

                    obj_found = True
                    break

                except Exception as e:
                    print(f"Error processing object: {e}")
                    pass

            if not obj_found:
                if attempts < 3:
                    window_tools.move_right(self.window, strength=1.2)
                else:
                    window_tools.move_left(self.window, strength=1.2)

        if obj_found:
            self.ensure_correct_difficulty()

            # This is the location where we want our update to take place for dynamic selection mode

            self.current_stage_button_farming_area = self._select_stage_button_area_dynamic(self.current_stage)
            if self.current_stage_button_farming_area is None:
                if self.progress_attempt_active:
                    self.current_stage = self.current_base_stage
                    self.progress_attempt_active = False
                    self.current_stage_button_farming_area = self._select_stage_button_area_dynamic(self.current_stage)
                if self.current_stage_button_farming_area is None:
                    self.current_stage_button_farming_area = self._select_stage_button_area_legacy_fallback()
            offset = [0.4946, 0.0047, 0, 0]
            self.current_stage_button_farming_area = [
                base + add for base, add in zip(self.current_stage_button_farming_area, offset)
            ]

            window_tools.click_center(self.window, self.current_stage_button_farming_area, delay=2)
            # This is the end of that update location.
        else:
            self._clear_progress_context()

        return obj_found

    # ------------------------- Run Encounter -------------------------
    def farm_encounter(self):
        self.battle_status = 'Starting'
        window_tools.click_center(self.window, self.search_areas["faction_wars_farm_encounter"])
        faction_wars_multibattles_setup_1 = image_tools.get_similarities_in_relative_area(
                self.window,
                self.search_areas["faction_wars_multibattles_setup_1"],
                'data\\assets\\images\\doom_tower_multibattles_setup.png'
            )
        faction_wars_multibattles_setup_2 = image_tools.get_similarities_in_relative_area(
                self.window,
                self.search_areas["faction_wars_multibattles_setup_2"],
                'data\\assets\\images\\doom_tower_multibattles_setup.png'
            )
        if not faction_wars_multibattles_setup_1:
            window_tools.click_center(self.window, self.search_areas["faction_wars_multibattles_setup_1"])

        if faction_wars_multibattles_setup_2:
            window_tools.click_center(self.window, self.search_areas["faction_wars_multibattles_setup_2"])

        window_tools.click_center(self.window, self.search_areas["faction_wars_start_multibattles"], delay = 5)
        self.battle_status = 'Running'
        auto_battle_tools.reset_auto_battle_watchdog(self)

        window_tools.move_down(self.window)

        while self.battle_status == "Running":
            _ensure_within_run_deadline(self, "waiting for faction wars farming result")
            auto_battle_tools.ensure_auto_battle_running(self)
            farming_status = image_tools.get_text_in_relative_area(
                self.reader, self.window,
                search_area=self.current_stage_button_farming_area
            )

            time.sleep(2)
            try:
                if getattr(farming_status[0],'text', False):
                    if self.resembles(farming_status[0].text, "Resultados"):
                        self.battle_status = 'Finished'
                        window_tools.click_center(self.window, self.search_areas["faction_wars_farming_status"])
                        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
            except:
                pass


    def execute_faction_encounter(self):
        window_tools.click_center(self.window, self.search_areas["confirm_button_champion_selection"])
        if not image_tools.check_startup(self):
            window_tools.click_center(self.window, self.search_areas["confirm_button_champion_selection"])

        
        self.reset_battle_state()

        while self.main_loop_running and (self.battle_status != 'Done'):
            _ensure_within_run_deadline(self, "waiting for faction wars encounter result")
            self.update_battle_outcome()
            auto_battle_tools.ensure_auto_battle_running(self)
            self.update_battle_activity_status()

        window_tools.click_center(self.window, self.search_areas["go_to_map"])

    # ------------------------- Main Loop -------------------------
    def run_factionwars(
        self,
        main_loop_running=True,
        max_run_duration_seconds=MAX_RUN_DURATION_SECONDS,
    ):
        _start_run_deadline(self, max_run_duration_seconds)
        time.sleep(5)
        self.start_time = time.time()
        self.running = True
        self.main_loop_running = main_loop_running

        while self.main_loop_running and (self.running):
            _ensure_within_run_deadline(self, "running faction wars loop")
            encounter_found = self.locate_faction_encounter()
            if encounter_found:


                self.execute_faction_encounter()
                self._resolve_progress_result()
                #self.farm_encounter()
                self.report_run_status()
            else:
                print('Could not find encounter')
                self.running = False
