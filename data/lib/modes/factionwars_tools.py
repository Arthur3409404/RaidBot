import numpy as np
import time
import re
from datetime import  timedelta
import data.lib.utils.image_tools as image_tools
import data.lib.utils.window_tools as window_tools
import difflib

class RSL_Bot_FactionWars:
    
    def __init__(self, title_substring="Raid: Shadow Legends", reader = None, window =None, verbose = True, farm_stages = {"Banner Lords":[17,"normal"],"Barbarians":[17,"normal"],"Dark Elves":[17,"normal"],"Demonspawn":[17,"normal"],"Dwarves":[17,"normal"],"High Elves":[17,"normal"],"Knight Revenant":[17,"normal"],"Lizardmen":[17,"normal"],"Ogryn Tribes":[17,"normal"],"Orcs":[17,"normal"],"Sacred Order":[17,"normal"],"Undead Hordes":[17,"normal"],"Shadowkin":[17,"normal"],"Skinwalkers":[17,"normal"],"Sylvan Watchers":[17,"normal"]}, farm_superraid = True):

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
        self.multiplier = 1
        if self.farm_superraid:
            self.multiplier+=1

        self.window = window
        self.init_time = time.time()
        
        self.battle_status = 'menu'
        self.auto_button_clicked = False
        
        
        if self.window:
            self.coords = (self.window.left, self.window.top, self.window.width, self.window.height)
            print(f"Window Coordinates: {self.coords}")
        else:
            self.coords = None
            
        # Search Areas
        self.search_areas = {
            
            "faction_wars_keys":   [0.611, 0.041, 0.072, 0.036],
            "faction_name":   [0.01, 0.033, 0.448, 0.046],
            'pov' : [0, 0, 1, 1],
            "go_to_higher_menu":   [0.928, 0.031, 0.046, 0.039],
            
            "confirm_button_champion_selection": [0.762, 0.876, 0.213, 0.104],
            
            "auto_battle_button": [0.026, 0.899, 0.058, 0.07], # its existance means battle started
            "battle_status_finished":  [0.362, 0.897, 0.269, 0.081], # check regularly for if enemy quits
            "battle_result":    [0.389, 0.148, 0.204, 0.071],
            "battle_result_2":    [0.38, 0.085, 0.224, 0.059],
            "restart_encounter":   [0.423, 0.877, 0.211, 0.106],

            'get_difficulty':[0.029, 0.924, 0.084, 0.035],
            'change_difficulty_normal':[0.097, 0.803, 0.065, 0.031],
            'change_difficulty_hard':[0.103, 0.873, 0.061, 0.034],

            "faction_wars_farm_encounter": [0.763, 0.764, 0.211, 0.105],
            "faction_wars_start_multibattles": [0.254, 0.634, 0.23, 0.075],
            "faction_wars_multibattles_setup_1": [0.222, 0.458, 0.032, 0.04],
            "faction_wars_multibattles_setup_2": [0.221, 0.502, 0.034, 0.045],
            "faction_wars_farming_status": [0.366, 0.609, 0.269, 0.102],

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
        
        self.current_difficulty = 'normal'

        self.stages_buttons = [[0.787, 0.083, 0.177, 0.071],
                               [0.785, 0.192, 0.176, 0.078],
                               [0.785, 0.311, 0.176, 0.074],
                               [0.785, 0.429, 0.179, 0.079],
                               [0.782, 0.546, 0.182, 0.081],
                               [0.784, 0.663, 0.181, 0.081],
                               [0.784, 0.784, 0.18, 0.078],
                               [0.783, 0.898, 0.181, 0.082],
                               ]
        
        
    # ------------------------- Reset Methods -------------------------
    def reset_battle_state(self):
        self.battle_status = 'menu'

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
        for result_area in ["battle_result", "battle_result_2"]:
            try:
                battle_result = image_tools.get_text_in_relative_area(
                    self.reader, self.window, search_area=self.search_areas[result_area]
                )[0]
                if battle_result.text in ["VICTORIA", "DERROTA"]:
                    self.battle_status = 'Done'
                    self.battles_done += 1
                    if self.resembles(battle_result.text, "VICTORIA"):
                        self.battles_won += 1
                    return
            except:
                continue

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
        print("🛡️  RAID Faction Wars Bot Status")
        print("-" * 40)
        print(f"🔁 Mode: Simple Pick")
        print(f"⏱️  Time Since Start: {formatted_elapsed}")
        print(f"⚔️  Battles Won: {self.battles_won}")
        print(f"⚔️  Battles Lost: {self.battles_done - self.battles_won}")
        print("-" * 40)
        print("🛑 To stop the bot, press 'v'")
        print("=" * 40 + "\n")

    # ------------------------- FW Keys -------------------------
    def get_available_fw_keys(self):
        try:
            fw_keys = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas['faction_wars_keys']
            )[0]
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

                    self.current_stage = self.farm_stages[key[0]][0]
                    self.current_difficulty = self.farm_stages[key[0]][1]

                    current_fw_keys = self.get_available_fw_keys()
                    if (int(current_fw_keys) < 4 * self.multiplier and self.current_difficulty == 'normal') or \
                       (int(current_fw_keys) < 6 * self.multiplier and self.current_difficulty == 'hard'):
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
            stage = np.clip(self.current_stage - 14, 0, 7) if self.current_difficulty == 'hard' else np.clip(self.current_stage - 14, 3, 7)
            self.current_stage_button_farming_area = self.stages_buttons[stage]
            window_tools.click_center(self.window, self.stages_buttons[stage], delay=2)

        return obj_found

    # ------------------------- Run Encounter -------------------------
    def farm_encounter(self):
        self.battle_status = 'Starting'
        window_tools.click_center(self.window, self.search_areas["faction_wars_farm_encounter"])
        faction_wars_multibattles_setup_1 = image_tools.get_similarities_in_relative_area(
                self.window,
                self.search_areas["faction_wars_multibattles_setup_1"],
                'pic\\doom_tower_multibattles_setup.png'
            )
        faction_wars_multibattles_setup_2 = image_tools.get_similarities_in_relative_area(
                self.window,
                self.search_areas["faction_wars_multibattles_setup_2"],
                'pic\\doom_tower_multibattles_setup.png'
            )
        if not faction_wars_multibattles_setup_1:
            window_tools.click_center(self.window, self.search_areas["faction_wars_multibattles_setup_1"])

        if faction_wars_multibattles_setup_2:
            window_tools.click_center(self.window, self.search_areas["faction_wars_multibattles_setup_2"])

        window_tools.click_center(self.window, self.search_areas["faction_wars_start_multibattles"], delay = 5)
        self.battle_status = 'Running'

        window_tools.move_down(self.window)

        while self.battle_status == "Running":
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
        self.reset_battle_state()

        while self.main_loop_running and (self.battle_status != 'Done'):
            self.update_battle_outcome()
            self.update_battle_activity_status()

        window_tools.click_center(self.window, self.search_areas["go_to_map"])

    # ------------------------- Main Loop -------------------------
    def run_factionwars(self, main_loop_running = True):
        time.sleep(5)
        self.start_time = time.time()
        self.running = True
        self.main_loop_running = main_loop_running

        while self.main_loop_running and (self.running):
            encounter_found = self.locate_faction_encounter()
            if encounter_found:
                self.execute_faction_encounter()
                #self.farm_encounter()
                self.report_run_status()
            else:
                print('Could not find encounter')
                self.running = False