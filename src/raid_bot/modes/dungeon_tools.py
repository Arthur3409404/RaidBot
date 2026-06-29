
import numpy as np
import time
import re
import difflib
import unicodedata
from datetime import timedelta


import raid_bot.utils.auto_battle_tools as auto_battle_tools
import raid_bot.utils.image_tools as image_tools
import raid_bot.utils.window_tools as window_tools

MAX_RUN_DURATION_SECONDS = int(3.5 * 60 * 60)
ENCOUNTER_SWEEP_SPAN = 8
DUNGEON_STAGE_BUTTON_Y_OFFSET = 0.02


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


class RSL_Bot_Dungeons:
    
    def __init__(
        self,
        title_substring="Raid: Shadow Legends",
        reader=None,
        window=None,
        verbose=True,
        iron_twins_priority=True,
        essence_priority='shogun',
        defaults_available=[
            "fire_knight",
            "dragon",
            "spider",
            "ice_golem",
            "sand_devil",
            "shogun",
            "minotaur",
            "event_dungeon",
        ],
        difficulty='hard',
        fusion_difficulty='normal',
        level=None,
        build_name=None,
        dungeon='fire_knight',
        eventdungeon_level=29,
        disable_fusion_override=False,
        fusion_active=False,
    ):

        if reader is None:
            print('Error When Loading Reader')
            
        self.reader = reader
        
        self.running = True
        
        self.battles_done = 0
        self.battles_won = 0
        self.no_coin_status = False
        
        self.verbose = verbose
        self.iron_twins_priority = iron_twins_priority
        self.essence_priority = essence_priority
        self.defaults_available = defaults_available
        self.valid_difficulties = {"normal", "hard"}
        self.difficulty = self.normalize_difficulty(difficulty, default="normal", context="initialization")
        self.fusion_difficulty = self.normalize_difficulty(
            fusion_difficulty,
            default="normal",
            context="fusion initialization",
        )
        self.level = self.normalize_level(level)
        self.eventdungeon_level = self.normalize_level(eventdungeon_level) or 29
        self.build_name = str(build_name).strip() if build_name is not None else None
        self.disable_fusion_override = bool(disable_fusion_override)
        self.dungeon = dungeon
        self.fusion_active = bool(fusion_active)

        self.window = window
        self.init_time = time.time()
        
        self.battle_status = 'menu'
        self.auto_button_clicked = False
        self.max_run_duration_seconds = MAX_RUN_DURATION_SECONDS
        self._run_deadline = None
        self.energy = 0
        self.iron_twins_keys = 0
        self.minimum_expected_energy = None
        self.last_run_energy_cost = None
        self.last_energy_encounter = None
        self.encounter_search_direction = "right"
        self.encounter_search_moves_in_direction = 0
        
        
        if self.window:
            self.coords = (self.window.left, self.window.top, self.window.width, self.window.height)
            print(f"Window Coordinates: {self.coords}")
        else:
            self.coords = None
            
        # Search Areas
        self.search_areas = {
            
            "energy":   [0.599, 0.038, 0.069, 0.027],
            "iron_twins_keys":  [0.510, 0.039, 0.045, 0.04],
            "iron_twins_keys_and_energy":  [0.4, 0.039, 0.4, 0.04],
            'pov' : [0, 0, 1, 1],
            #"fire_knight_hard_8":   [0.238, 0.659, 0.222, 0.025],
            #"iron_twins_15":   [0.238, 0.659, 0.222, 0.025],
            #"shogun_25":   [0.238, 0.659, 0.222, 0.025],
            
            "confirm_button_champion_selection": [0.762, 0.876, 0.213, 0.104],
            "dungeon_setup_section_groups": [0.026, 0.435, 0.059, 0.072],
            "dungeon_setup_names": [0.2, 0.0, 0.4, 0.9],
            "dungeon_setup_check": [0.028, 0.087, 0.049, 0.898],
            
            "auto_battle_button": [0.026, 0.899, 0.058, 0.07], # its existance means battle started
            "battle_status_finished":  [0.362, 0.897, 0.269, 0.081], # check regularly for if enemy quits
            "battle_result":    [0.389, 0.148, 0.204, 0.071],
            "battle_result_2":    [0.38, 0.085, 0.224, 0.059],

            'get_dungeon_difficulty':[0.029, 0.924, 0.084, 0.035],
            'change_dungeon_difficulty_normal':[0.097, 0.803, 0.065, 0.031],
            'change_dungeon_difficulty_hard':[0.103, 0.873, 0.061, 0.034],

            'go_to_map': [0.134, 0.905, 0.059, 0.071],

            
            "test":   [0.05, 0.30, 0.15, 0.08],
            "dungeons_etapa_window": [0.3364, 0.0716, 0.159, 0.93],
            
        }

        self.dungeon_menu_names = {
            "iron_twins": "Fortaleza de los Gemelos",
            "dragon": "Guarida del Dragon",
            "fire_knight": "Castillo del Caballero de Fuego",
            "sand_devil": "Necropolis de la Arena",
            "shogun": "Arboleda del Shoc",
            'minotaur': 'Laberinto del Minotauro',
            "ice_golem":"Pico del Golem de Hielo",
            "spider":"Nido de Aranas",
            "event_dungeon": "Mazmorra de Evento",
            "hard": "Dificil",
            "normal": "Normal"
        }
        self.hardmode_available = ["fire_knight","dragon","ice_golem",'spider']
        self.fixed_last_stage_dungeons = {
            "iron_twins": 15,
            "minotaur": 15,
            "shogun": 25,
            "sand_devil": 25,
        }
        self.fixed_last_stage_button_index = 7
        self.hard_only_dungeons = {"minotaur", "shogun", "sand_devil"}
        self.always_skip_build_dungeons = {"iron_twins", "shogun"}
        self.always_build_dungeons = {"sand_devil", "minotaur"}
        self.hard_build_dungeons = {"dragon", "ice_golem", "spider", "fire_knight"}
        self.fusion_normal_build_dungeons = {"dragon", "ice_golem", "spider"}
        self.dungeon_build_names = {
            "dragon": "Dragon",
            "ice_golem": "Ice Golem",
            "spider": "Spider",
            "fire_knight": "Fire Knight",
            "sand_devil": "Sand Devil",
            "minotaur": "Minotaur",
            "event_dungeon": "Event Dungeon",
        }

        self.stages_buttons = [[0.787, 0.083, 0.177, 0.071],
                               [0.785, 0.192, 0.176, 0.078],
                               [0.785, 0.311, 0.176, 0.074],
                               [0.785, 0.429, 0.179, 0.079],
                               [0.782, 0.546, 0.182, 0.081],
                               [0.784, 0.663, 0.181, 0.081],
                               [0.784, 0.784, 0.18, 0.078],
                               [0.783, 0.898, 0.181, 0.082],
                               ]
        
        
    def reset_battle_parameters(self):
        self.battle_status = 'menu'

    def _read_battle_result_once(self):
        for area_name in ("battle_result", "battle_result_2"):
            try:
                text_objects = image_tools.get_text_in_relative_area(
                    self.reader,
                    self.window,
                    search_area=self.search_areas[area_name],
                )
            except Exception:
                continue

            for text_object in text_objects:
                text = (text_object.text or "").strip()
                if not text:
                    continue
                if self.resembles(text, "VICTORIA"):
                    return "VICTORIA"
                if self.resembles(text, "DERROTA"):
                    return "DERROTA"

        return None

    def resembles(self, text, target, threshold=0.8):
        ratio = difflib.SequenceMatcher(
            None,
            self._normalize_text(text),
            self._normalize_text(target),
        ).ratio()
        return ratio >= threshold

    def _normalize_text(self, text):
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKD", str(text))
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = re.sub(r"[^a-zA-Z0-9]+", " ", normalized).lower()
        return re.sub(r"\s+", " ", normalized).strip()

    def _contains_gemelos(self, text):
        return bool(re.search(r"\bgemelos\b", self._normalize_text(text)))

    def normalize_difficulty(self, value, default="normal", context="dungeon"):
        normalized = str(value or "").strip().lower()
        if normalized in self.valid_difficulties:
            return normalized
        if normalized and self.verbose:
            print(
                f"Invalid dungeon difficulty '{value}' during {context}. "
                f"Defaulting to '{default}'."
            )
        return default

    def normalize_level(self, value):
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            if self.verbose:
                print(f"Invalid dungeon level '{value}'. Ignoring explicit stage override.")
            return None
        if parsed <= 0:
            if self.verbose:
                print(f"Invalid dungeon level '{value}'. Level must be positive.")
            return None
        return parsed

    def normalize_encounter_name(self, encounter_name):
        return str(encounter_name or "").strip().lower()

    def resolve_dungeon_difficulty(self, encounter_name=None, input_difficulty=None):
        dungeon_name = self.normalize_encounter_name(
            self.dungeon if encounter_name is None else encounter_name
        )
        if dungeon_name == "event_dungeon":
            return "normal"
        if dungeon_name in self.hard_only_dungeons:
            return "hard"

        requested = self.difficulty if input_difficulty is None else input_difficulty
        requested = self.normalize_difficulty(
            requested,
            default="normal",
            context=f"difficulty resolve ({dungeon_name or 'unknown'})",
        )
        if requested == "hard":
            return "hard"
        return "normal"

    def resolve_dungeon_level(self, encounter_name=None, input_difficulty=None):
        dungeon_name = self.normalize_encounter_name(
            self.dungeon if encounter_name is None else encounter_name
        )
        if dungeon_name == "event_dungeon":
            return self.eventdungeon_level
        explicit_level = self.normalize_level(self.level)
        if explicit_level is not None and dungeon_name in self.hardmode_available:
            return explicit_level

        difficulty = self.resolve_dungeon_difficulty(dungeon_name, input_difficulty)

        if dungeon_name in self.hardmode_available:
            return 10 if difficulty == "hard" else 20
        return None

    def _build_relative_area_around_abs_point(self, abs_x, abs_y, rel_width=0.17, rel_height=0.08):
        rel_center_x = (abs_x - self.window.left) / self.window.width
        rel_center_y = (abs_y - self.window.top) / self.window.height

        rel_left = max(0.0, min(1.0 - rel_width, rel_center_x - rel_width / 2))
        rel_top = max(0.0, min(1.0 - rel_height, rel_center_y - rel_height / 2))
        return [rel_left, rel_top, rel_width, rel_height]

    def _build_stage_button_area_for_abs_row(self, abs_y):
        rel_center_y = (abs_y - self.window.top) / self.window.height
        rel_center_y = max(0.0, min(1.0, rel_center_y + DUNGEON_STAGE_BUTTON_Y_OFFSET))
        template = min(
            self.stages_buttons,
            key=lambda area: abs((area[1] + area[3] / 2.0) - rel_center_y),
        )
        rel_left, _, rel_width, rel_height = template
        rel_top = max(0.0, min(1.0 - rel_height, rel_center_y - rel_height / 2.0))
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

    def _select_stage_button_area_dynamic(self, target_stage, max_scroll_attempts=8):
        for _ in range(max_scroll_attempts):
            stage_objects = image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                self.search_areas["dungeons_etapa_window"],
            )

            stage_candidates = []
            for obj in stage_objects:
                stage_number = self._extract_stage_number(getattr(obj, "text", ""))
                if stage_number is None:
                    continue
                stage_candidates.append((stage_number, obj))

            for stage_number, obj in stage_candidates:
                if stage_number == int(target_stage):
                    return self._build_stage_button_area_for_abs_row(obj.mean_pos_y)

            if not stage_candidates:
                window_tools.move_down(self.window, strength=0.6)
                continue

            stage_values = [value for value, _ in stage_candidates]
            if int(target_stage) > max(stage_values):
                window_tools.move_down(self.window, strength=0.8)
            elif int(target_stage) < min(stage_values):
                window_tools.move_up(self.window, strength=0.8)
            else:
                break

        return None

    def _legacy_stage_boxes_support_target_level(self, target_level, difficulty):
        try:
            level = int(target_level)
        except (TypeError, ValueError):
            return False
        normalized_difficulty = str(difficulty or "").strip().lower()
        if normalized_difficulty == "hard":
            return 3 <= level <= 10
        return 18 <= level <= 25

    def get_required_build_name(self, encounter_name):
        dungeon_name = self.normalize_encounter_name(encounter_name)
        if not dungeon_name:
            return None

        if dungeon_name == "event_dungeon":
            return None

        if self.build_name:
            return self.build_name

        if dungeon_name in self.always_skip_build_dungeons:
            return None

        build_name = self.dungeon_build_names.get(dungeon_name)
        if not build_name:
            if self.verbose:
                print(f"Unknown dungeon '{encounter_name}'. Skipping build selection.")
            return None

        if dungeon_name in self.always_build_dungeons:
            return build_name

        if dungeon_name in self.hard_build_dungeons:
            difficulty = self.resolve_dungeon_difficulty(
                encounter_name=dungeon_name,
                input_difficulty=self.difficulty,
            )
            if difficulty == "hard":
                return build_name
            if bool(self.fusion_active) and dungeon_name in self.fusion_normal_build_dungeons:
                return "Fusion"
            return None

        return None

    def _is_setup_already_selected(self, setup_text, selected_markers, y_offset=70.0, tolerance=60.0) -> bool:
        if not setup_text or not selected_markers:
            return False

        target_y = float(setup_text.mean_pos_y) + float(y_offset)
        return any(
            abs(float(marker.mean_pos_y) - target_y) <= float(tolerance)
            for marker in selected_markers
        )

    def _click_dungeon_setup(self, setup_text, y_offset=70.0):
        window_tools.click_at(
            setup_text.mean_pos_x - 268.0,
            setup_text.mean_pos_y + y_offset,
            delay=2,
            window=self.window,
        )

    def select_build_if_needed(self, encounter_name):
        build_name = self.get_required_build_name(encounter_name)
        if not build_name:
            return True

        current_setup = None
        try:
            window_tools.click_center(self.window, self.search_areas["dungeon_setup_section_groups"], delay=1.5)
            window_tools.move_up(self.window, strength=3, relative_x=0.15)

            for _ in range(3):
                if not self.main_loop_running:
                    break

                setups = image_tools.get_text_in_relative_area(
                    self.reader,
                    self.window,
                    search_area=self.search_areas["dungeon_setup_names"],
                )

                for name in setups:
                    if self.resembles(name.text, build_name, threshold=0.7):
                        current_setup = name
                        break

                if current_setup:
                    break

                window_tools.move_down(self.window, strength=0.5, relative_x=0.15)

            if current_setup:
                completed = image_tools.get_similarities_in_relative_area(
                    self.window,
                    self.search_areas["dungeon_setup_check"],
                    "data\\assets\\images\\doom_tower_completed_stage.png",
                )
                if not self._is_setup_already_selected(current_setup, completed):
                    self._click_dungeon_setup(current_setup)
                if self.verbose:
                    print(f"Dungeon build selection: using setup '{build_name}'.")
            elif self.verbose:
                print(f"Dungeon build selection: setup '{build_name}' not found.")
        except Exception as exc:
            if self.verbose:
                print(f"Dungeon build selection failed for '{build_name}': {exc}")
        window_tools.click_center(self.window, self.search_areas["confirm_button_champion_selection"])
        return True

    def check_difficulty(self, encounter_name=None):
        try:
            resolved_dungeon = self.normalize_encounter_name(
                self.dungeon if encounter_name is None else encounter_name
            )
            target_difficulty = self.resolve_dungeon_difficulty(
                encounter_name=resolved_dungeon,
                input_difficulty=self.difficulty,
            )
            if resolved_dungeon in self.hard_only_dungeons:
                self.difficulty = target_difficulty
                return

            self.difficulty = target_difficulty
            difc_txt = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas["get_dungeon_difficulty"])[0]
            if self.resembles(difc_txt.text, self.dungeon_menu_names[target_difficulty]):
                pass
            else:
                window_tools.click_center(self.window, self.search_areas["get_dungeon_difficulty"])
                string = 'change_dungeon_difficulty_' + target_difficulty
                window_tools.click_center(self.window, self.search_areas[string])

        except:
            print('Error changing Difficulties')
        
    def get_battle_outcome(self):
        first_result = self._read_battle_result_once()
        if first_result is None:
            return

        time.sleep(10)
        second_result = self._read_battle_result_once()
        if second_result is None or first_result != second_result:
            if self.verbose:
                print(
                    "Battle result mismatch between checks. "
                    f"First='{first_result}', second='{second_result}'."
                )
            return

        self.battle_status = 'Done'
        self.battles_done += 1
        if first_result == "VICTORIA":
            self.battles_won += 1
        return
            
    
    
    def get_battle_status(self):
        try:
            auto_button = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas["auto_battle_button"])[0]
            if self.resembles(auto_button.text, 'Auto'):
                self.battle_status = 'Battle active'
                battle_running = True

            else: 
                self.battle_status = 'Battle inactive'
        except:
            pass      
        return




    def print_status(self):
        elapsed = time.time() - self.init_time
        formatted_elapsed = str(timedelta(seconds=int(elapsed)))
        medals = (self.battles_won) * 70
    
        print("\n" + "=" * 40)
        print("ðŸ›¡ï¸  RAID Dungeon Bot Status")
        print("-" * 40)
        print(f"ðŸ” Mode: Simple Pick)")
        print(f"â±ï¸  Time Since Start: {formatted_elapsed}")
        print(f"âš”ï¸  Battles Won: {self.battles_won}")
        print(f"âš”ï¸  Battles Lost: {self.battles_done - self.battles_won}")
        print("-" * 40)
        print("ðŸ›‘ To stop the bot, press 'v'")
        print("=" * 40 + "\n")

    def get_minimum_energy_cost(self, encounter_name):
        if encounter_name == "iron_twins":
            return 40

        difficulty = self.resolve_dungeon_difficulty(
            encounter_name=encounter_name,
            input_difficulty=self.difficulty,
        )
        level = self.resolve_dungeon_level(
            encounter_name=encounter_name,
            input_difficulty=difficulty,
        )

        if difficulty == "hard":
            return 40
        if difficulty == "normal" and level == 20:
            return 32
        return None

    def remember_minimum_expected_energy(self, encounter_name):
        minimum_cost = self.get_minimum_energy_cost(encounter_name)
        self.last_energy_encounter = encounter_name
        self.last_run_energy_cost = minimum_cost

        if minimum_cost is None:
            self.minimum_expected_energy = None
            return

        self.minimum_expected_energy = max(0, int(self.energy) - minimum_cost)

    def apply_energy_estimator_failsafe(self):
        if self.minimum_expected_energy is None:
            return

        try:
            current_energy = int(self.energy)
        except (TypeError, ValueError):
            current_energy = 0

        if current_energy >= self.minimum_expected_energy:
            return

        if self.verbose:
            print(
                "Energy OCR below estimated minimum after "
                f"{self.last_energy_encounter}: read {current_energy}, "
                f"using {self.minimum_expected_energy} instead."
            )

        self.energy = self.minimum_expected_energy

                    
    def select_encounter(self, encounter_name, max_attempts = 4):
        encounter_name = self.normalize_encounter_name(encounter_name)
        name_string = self.dungeon_menu_names.get(encounter_name)
        if not name_string:
            if self.verbose:
                print(f"Unknown dungeon encounter '{encounter_name}'.")
            return False

        obj_found = False
        attempts = 0
        while self.main_loop_running and (attempts<max_attempts and not obj_found):
            _ensure_within_run_deadline(self, "searching dungeon encounter")
            attempts+=1

            time.sleep(2)
            objects = image_tools.get_text_in_relative_area(self.reader, self.window, self.search_areas['pov'], power_detection=False)

            try:
                for obj in objects:
                    print(obj.text)
                    if encounter_name == "iron_twins":
                        if self._contains_gemelos(obj.text):
                            window_tools.click_at(obj.mean_pos_x, obj.mean_pos_y, delay = 4)
                            obj_found = True
                            break
                    elif self.resembles(obj.text ,name_string):
                        window_tools.click_at(obj.mean_pos_x, obj.mean_pos_y, delay = 4)
                        obj_found = True
                        break
            except:
                pass
            if not obj_found:
                if self.encounter_search_direction == "right":
                    window_tools.move_right(self.window, strength = 1.2)
                else:
                    window_tools.move_left(self.window, strength = 1.2)

                self.encounter_search_moves_in_direction += 1
                if self.encounter_search_moves_in_direction >= ENCOUNTER_SWEEP_SPAN:
                    self.encounter_search_moves_in_direction = 0
                    self.encounter_search_direction = (
                        "left" if self.encounter_search_direction == "right" else "right"
                    )


        if obj_found:
            self.encounter_search_direction = "right"
            self.encounter_search_moves_in_direction = 0
            if encounter_name == "event_dungeon":
                event_stage_level = self.resolve_dungeon_level(encounter_name=encounter_name)
                event_stage_area = self._select_stage_button_area_dynamic(event_stage_level)
                if event_stage_area is None:
                    if self.verbose:
                        print(
                            f"Event Dungeon stage {event_stage_level} was not found via OCR. "
                            "No static fallback will be used."
                        )
                    return False
                window_tools.click_center(self.window, event_stage_area, delay=2)
                return obj_found
            if encounter_name in self.fixed_last_stage_dungeons:
                level = self.fixed_last_stage_dungeons[encounter_name]
                difficulty = "hard"
                stage = self.fixed_last_stage_button_index

            elif encounter_name in self.hardmode_available:
                self.check_difficulty(encounter_name=encounter_name)

                window_tools.move_down(self.window, strength = 1)

                difficulty = self.resolve_dungeon_difficulty(
                    encounter_name=encounter_name,
                    input_difficulty=self.difficulty,
                )
                self.difficulty = difficulty
                level = self.resolve_dungeon_level(
                    encounter_name=encounter_name,
                    input_difficulty=difficulty,
                )
                if level is None:
                    return False

                if difficulty == 'hard':
                    stage = np.clip(level - 3, 0, 7)
                else:
                    stage = np.clip(level - 18, 2, 7)
            
            else:
                stage = self.fixed_last_stage_button_index
                window_tools.click_center(self.window, self.stages_buttons[stage], delay=2)
                return obj_found

            use_dynamic_stage_selection = not self._legacy_stage_boxes_support_target_level(
                level,
                difficulty,
            )
            if use_dynamic_stage_selection:
                stage_area_dynamic = self._select_stage_button_area_dynamic(level)
                if stage_area_dynamic is not None:
                    window_tools.click_center(self.window, stage_area_dynamic, delay=2)
                else:
                    window_tools.click_center(self.window, self.stages_buttons[stage], delay=2)
            else:
                window_tools.click_center(self.window, self.stages_buttons[stage], delay = 2)

        return obj_found



    def run_encounter(self, encounter_name):
        self.reset_battle_parameters()
        if not self.select_build_if_needed(encounter_name):
            return False
        time.sleep(5)
        window_tools.click_center(self.window, self.search_areas["confirm_button_champion_selection"])
        window_tools.click_center(self.window, self.search_areas["confirm_button_champion_selection"])

        if not image_tools.check_startup(self):
            window_tools.click_center(self.window, self.search_areas["confirm_button_champion_selection"])

        auto_battle_tools.reset_auto_battle_watchdog(self)
        while self.main_loop_running and (self.battle_status != 'Done'):
            _ensure_within_run_deadline(self, "waiting for dungeon encounter result")

            auto_battle_tools.handle_pausa_popup(self)

            self.get_battle_outcome()
            auto_battle_tools.ensure_auto_battle_running(self)

            self.get_battle_status()
        
        time.sleep(2)
        window_tools.click_center(self.window, self.search_areas["go_to_map"])

        return True

    # def check_energy(self):
    #     try:
    #         energy = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas['energy'])[0]
    #         energy = re.findall(r"\d+", energy.text)[0]
    #     except:
    #         energy = 0
    #     return energy

    def check_iron_twins_keys_and_energy(self):
        try:
            keys = image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                search_area=self.search_areas['iron_twins_keys_and_energy']
            )

            # Filter only entries that contain "/"
            keys_filtered = [key for key in keys if "/" in key.text]

            # Ensure we found at least two values (keys and energy)
            if len(keys_filtered) < 2:
                raise ValueError("Not enough OCR results found.")

            # Extract iron twins keys (first number before /)
            iron_twins_keys_match = re.findall(r"\d+", keys_filtered[0].text)
            if not iron_twins_keys_match:
                raise ValueError("Iron Twins keys not found.")

            self.iron_twins_keys = int(iron_twins_keys_match[0])
            print(self.iron_twins_keys)

            # Extract energy (first number before /)
            energy_match = re.findall(r"\d+", keys_filtered[1].text)
            if not energy_match:
                raise ValueError("Energy not found.")

            self.energy = int(energy_match[0].replace('.', ''))
            print(self.energy)
            self.apply_energy_estimator_failsafe()

        except Exception as e:
            print(f"Error checking iron twins keys and energy: {e}")
            self.energy = 0
            self.iron_twins_keys = 0
            self.apply_energy_estimator_failsafe()

                    
    def run_dungeons(
        self,
        main_loop_running=True,
        max_run_duration_seconds=MAX_RUN_DURATION_SECONDS,
        forced_encounter=None,
        energy_target=None,
        ignore_iron_twins_priority=False,
    ):
        _start_run_deadline(self, max_run_duration_seconds)
        time.sleep(5)
        self.start_time = time.time()
        self.running = True
        self.main_loop_running = main_loop_running
        self.minimum_expected_energy = None
        self.last_run_energy_cost = None
        self.last_energy_encounter = None
        self.energy_spent_this_run = 0
        self.starting_iron_twins_keys = None
        self.iron_twins_keys_used_this_run = 0

        while self.main_loop_running and (self.running):
            _ensure_within_run_deadline(self, "running dungeon loop")
            if energy_target is not None and self.energy_spent_this_run >= int(energy_target):
                self.running = False
                break

            # Stop if not enough energy
            self.check_iron_twins_keys_and_energy()
            if self.starting_iron_twins_keys is None:
                self.starting_iron_twins_keys = int(self.iron_twins_keys or 0)
            encounter = None

            # Decide which encounter to run
            if forced_encounter:
                encounter = self.normalize_encounter_name(forced_encounter)
            elif self.iron_twins_priority and not ignore_iron_twins_priority and self.iron_twins_keys != 0:
                encounter = "iron_twins"
            else:
                encounter = self.dungeon

            minimum_required_energy = self.get_minimum_energy_cost(encounter) or 40
            if self.energy < minimum_required_energy:
                self.running = False
                break

            # Try to select and run encounter
            if self.select_encounter(encounter):
                if not self.run_encounter(encounter):
                    self.running = False
                    break
                self.remember_minimum_expected_energy(encounter)
                self.energy_spent_this_run += int(self.last_run_energy_cost or 0)
            else:
                print("Could not find encounter")
                self.running = False

            self.print_status()

        self.iron_twins_keys_used_this_run = max(
            0,
            int(self.starting_iron_twins_keys or 0) - int(self.iron_twins_keys or 0),
        )
        return self.energy_spent_this_run
