# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 14:00:37 2025

@author: Arthur
"""

import numpy as np
import time
import re
import math
import data.lib.utils.image_tools as image_tools
import data.lib.utils.window_tools as window_tools


class RSL_Bot_DoomTower():
    def __init__(self, title_substring="Raid: Shadow Legends", reader = None, window = None, verbose = True, setup_build_names = None, setup = None):
        self.reader = reader
        
        self.running = True
        
        self.verbose = verbose
        self.setup = setup

        self.window = window
            
        self.search_areas = {
            "menu_name": [0.008, 0.034, 0.23, 0.037],   # [left, top, width, height]
            "go_to_higher_menu":   [0.928, 0.031, 0.046, 0.039],
            "pov":   [0.0, 0.07, 1, 0.93],
            "detect_doomtower_rotation": [0.121, 0.696, 0.189, 0.035],

            "doom_tower_keys":   [0.682, 0.033, 0.212, 0.04],
            "doom_tower_menu_name":   [0.009, 0.033, 0.221, 0.041],
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
            
            


        }

        self.translation_mapping = {
            'Normal' : 'normal',
            'Dificil' : 'hard',
            'Magma Dragon': 'magma_dragon',
            'Frost Spider': 'frost_spider',
            'Nether Spider': 'nether_spider',
            'Scarab King': 'scarab_king',
            'Eternal Dragon': 'eternal_dragon',
            'Bommal': 'bommal',
            'Dark Fae': 'dark_fae',
            'Gryphon': 'gryphon'
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

    # ------------------------- Reset -------------------------
    def reset(self):
        self.battle_status = 'Starting'

    # ------------------------- Stage Field -------------------------
    def get_current_stagefield(self):
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

    def is_within_radius(self, obj1, obj2, radius):
        return math.hypot(
            obj1.mean_pos_x - obj2.mean_pos_x,
            obj1.mean_pos_y - obj2.mean_pos_y
        ) <= radius

    # ------------------------- Keys -------------------------
    def check_doomtower_keys(self):
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
        window_tools.move_down(self.window, strength=0.5, relative_x_pos=0.25)

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
                while current % 10 == 0:
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
    def check_battle_outcome(self):
        try:
            result = image_tools.get_text_in_relative_area(
                self.reader, self.window,
                search_area=self.search_areas["doom_tower_automatic_climb"]
            )[0]
            if 'Batallas completadas' in result.text:
                self.battle_status = 'autoclimb'
        except:
            pass

        for key in [
            "doom_tower_battle_result_automatic_climb",
            "doom_tower_battle_result_automatic_climb_2"
        ]:
            try:
                result = image_tools.get_text_in_relative_area(
                    self.reader, self.window,
                    search_area=self.search_areas[key]
                )[0]
                if result.text == "Autoescalada completada":
                    self.battle_status = 'autoclimb_Done'
                    window_tools.click_center(
                        self.window,
                        self.search_areas["doom_tower_close_encounter"],
                        delay=5
                    )
            except:
                pass

        try:
            auto_button = image_tools.get_text_in_relative_area(
                self.reader, self.window,
                search_area=self.search_areas["doom_tower_auto_battle_button"]
            )
            if auto_button:
                return

            battle_result = image_tools.get_text_in_relative_area(
                self.reader, self.window,
                search_area=self.search_areas["doom_tower_battle_result"]
            )[0]

            if battle_result.text in ("VICTORIA", "DERROTA") and self.battle_status != 'autoclimb':
                self.battle_status = 'Done'
                self.battles_done += 1
                if battle_result.text == "VICTORIA":
                    self.battles_won += 1
                else:
                    self.no_run_failed = False
        except:
            pass

    # ------------------------- Encounter Setup -------------------------
    def setup_encounter(self):
        doom_tower_menu_name = image_tools.get_text_in_relative_area(
            self.reader, self.window,
            search_area=self.search_areas["doom_tower_menu_name"]
        )[0]

        number = re.findall(r'\d+', doom_tower_menu_name.text)[0]

        if 'Jefe Final' in doom_tower_menu_name.text or int(number) % 10 == 0:
            stage = '120' if 'Jefe Final' in doom_tower_menu_name.text else number
            current_opponent = self.doomtower_rotations[self.current_rotation][stage]
            print(current_opponent)
        else:
            current_opponent = 'Waves'

        self.select_build(current_opponent)

    # ------------------------- Build Selection -------------------------
    def select_build(self, setup):
        self.current_setup = False

        window_tools.click_center(self.window, self.search_areas["doom_tower_setup_section_groups"])
        window_tools.move_up(self.window, strength=3, relative_x_pos=0.15)

        for _ in range(3):
            setups = image_tools.get_text_in_relative_area(
                self.reader, self.window,
                search_area=self.search_areas["doom_tower_setup_names"]
            )

            for name in setups:
                if name.text == setup:
                    self.current_setup = name
                    break

            if self.current_setup:
                break

            window_tools.move_down(self.window, strength=0.5, relative_x_pos=0.15)

        if self.current_setup:
            completed = image_tools.get_simliarities_in_relative_area(
                self.window,
                self.search_areas["doom_tower_setup_check"],
                'pic\\doom_tower_completed_stage.png'
            )

            if completed:
                window_tools.click_center(self.window, self.search_areas["doom_tower_start_encounter"])
            else:
                window_tools.click_at(
                    self.current_setup.mean_pos_x - 268.0,
                    self.current_setup.mean_pos_y + 70
                )
                window_tools.click_center(self.window, self.search_areas["doom_tower_start_encounter"])

    # ------------------------- Run Encounter -------------------------
    def run_encounter(self, farming=False, max_attempts=50):
        self.setup_encounter()
        self.battle_status = 'Starting'

        window_tools.click_center(self.window, self.search_areas["doom_tower_start_encounter"])
        time.sleep(10)

        attempt = 0
        while True:
            self.check_battle_outcome()
            time.sleep(2)

            if self.battle_status == 'Done' and not farming:
                break

            if self.battle_status == 'Done' and farming:
                attempt += 1
                if attempt >= max_attempts:
                    break

                window_tools.click_center(
                    self.window,
                    self.search_areas["doom_tower_restart_encounter"]
                )
                time.sleep(10)
                self.battle_status = 'Starting'

                try:
                    battle_result = image_tools.get_text_in_relative_area(
                        self.reader, self.window,
                        search_area=self.search_areas["doom_tower_battle_result"]
                    )[0]
                    if battle_result.text in ("VICTORIA", "DERROTA"):
                        self.battle_status = 'Done'
                except:
                    pass

        print('Battle_Done')
        window_tools.click_center(
            self.window,
            self.search_areas["doom_tower_close_encounter"],
            delay=5
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

        stage_completed = image_tools.get_simliarities_in_relative_area(
            self.window,
            self.search_areas["doom_tower_check_boss_stage_complete"],
            'pic\\doom_tower_locked_stage.png'
        )

        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
        return not len(stage_completed) == 1

    # ------------------------- Highest Stage Detection -------------------------
    def get_highest_stage_available(self, max_attempts=10):
        self.highest_stage_available = 1
        end_reached = False
        attempts = 0

        while self.highest_stage_available != 120 and attempts != max_attempts and not end_reached:
            attempts += 1

            stages_completed = image_tools.get_simliarities_in_relative_area(
                self.window,
                self.search_areas["pov"],
                'pic\\doom_tower_completed_stage.png'
            )

            if not stages_completed:
                self.highest_stage_available = None
                return None

            stages_completed.sort(key=lambda o: o.mean_pos_y)

            stages_numbers = self.get_current_stagefield()
            self._reconstruct_stage_numbers(stages_numbers)

            backtrack = 0
            for completed in stages_completed:
                for number in stages_numbers:
                    if number.text and self.is_within_radius(completed, number, 100):
                        self.highest_stage_available = int(number.text) + backtrack
                        self.highest_stage = completed
                        break
                else:
                    backtrack += 1
                    continue
                break

            rel_y = (self.highest_stage.mean_pos_y - self.window.top) / self.window.height
            if rel_y < 0.3:
                window_tools.move_up(self.window, strength=1, relative_x_pos=0.1)
                continue
            else:
                end_reached = True

            if self.highest_stage_available % 10 == 9:
                increment = 2 if self._check_boss_stage() else 1
                self.highest_stage_available += increment

            self.highest_stage_available = max(1, min(120, self.highest_stage_available))

    # ------------------------- Stage Scan -------------------------
    def check_for_boss_and_current_stage(self, target=None):
        FIRST_PATH = 'pic\\doom_tower_current_stage.png'
        list_of_paths = [FIRST_PATH]
        expected_menu_names = {FIRST_PATH: 'Planta'}

        self.stage_found = False
        self.doomtower_climb_status = False

        def add_boss_paths():
            for value in self.doomtower_rotations[self.current_rotation].values():
                if value in self.translation_mapping:
                    path = f"pic\\doom_tower_{self.translation_mapping[value]}.png"
                    if path not in list_of_paths:
                        list_of_paths.append(path)
                        expected_menu_names[path] = value

        if target:
            add_boss_paths()
            list_of_paths = [
                p for p in list_of_paths
                if expected_menu_names.get(p) == target or p == FIRST_PATH
            ]
        else:
            add_boss_paths()

        for path in list_of_paths:
            threshold = 0.8
            possible = []

            while not possible and threshold > 0.25:
                possible = image_tools.get_simliarities_in_relative_area(
                    self.window,
                    self.search_areas["pov"],
                    path,
                    threshold=threshold,
                    scales=[0.7, 0.8, 0.9, 1.0]
                )
                threshold -= 0.03

            for stage in possible:
                window_tools.click_at(stage.mean_pos_x, stage.mean_pos_y, delay=4)

                try:
                    menu = image_tools.get_text_in_relative_area(
                        self.reader, self.window,
                        search_area=self.search_areas["doom_tower_menu_name"]
                    )[0]

                    number = re.findall(r'\d+', menu.text)
                    number = number[0] if number else '10'

                    if menu.text == 'Torre del Destino':
                        continue

                    expected = expected_menu_names[path]

                    if expected == 'Planta' and expected in menu.text:
                        self.stage_found = stage
                        self.highest_stage_available[self.current_difficulty] = number
                        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
                        return

                    if expected not in self.doomtower_rotations[self.current_rotation][str(number)] \
                       and 'Jefe Final' not in menu.text:
                        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
                        continue

                    if 'Jefe Final' in menu.text or int(number) % 10 == 0:
                        locked = image_tools.get_simliarities_in_relative_area(
                            self.window,
                            self.search_areas["doom_tower_check_boss_stage_complete"],
                            'pic\\doom_tower_locked_boss.png'
                        )

                        if 'Jefe Final' in menu.text and not locked:
                            self.doomtower_climb_status = 'completed'

                        self.highest_stage_available[self.current_difficulty] = (
                            120 if 'Jefe Final' in menu.text else number
                        )

                        self.stage_found = stage
                        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
                        return
                except:
                    pass

    # ------------------------- Simple Scan -------------------------
    def simple_find_highest_stage(self, target=None):
        for _ in range(7):
            window_tools.move_up(self.window, strength=3, relative_x_pos=0.1)

        for _ in range(25):
            self.check_for_boss_and_current_stage(target=target)
            if self.stage_found:
                break
            window_tools.move_down(self.window, strength=0.6, relative_x_pos=0.1)

    # ------------------------- Climb -------------------------
    def climb_doomtower(self):
        self.set_difficulty('hard')

        if self.doomtower_climb_status_hard != 'completed':
            self.simple_find_highest_stage()

        if self.doomtower_climb_status in ('completed',) or \
           self.doomtower_climb_status_hard == 'completed':

            self.doomtower_climb_status_hard = 'completed'
            self.set_difficulty('normal')

            if not self.doomtower_completed:
                self.simple_find_highest_stage()

            if self.doomtower_climb_status == 'completed':
                self.doomtower_climb_status_normal = 'completed'
                self.doomtower_completed = True

        if not self.doomtower_completed and self.stage_found:
            window_tools.click_at(self.stage_found.mean_pos_x, self.stage_found.mean_pos_y)
            self.run_encounter()

    # ------------------------- Farming -------------------------
    def farm_doomtower(self):
        self.set_difficulty(self.setup['difficulty'])

        for opponent in self.setup['priority_bosses']:
            if opponent in self.doomtower_rotations[self.current_rotation].values():
                self.farming_opponent = opponent
                break

        self.simple_find_highest_stage(target=self.farming_opponent)

        if self.stage_found:
            window_tools.click_at(self.stage_found.mean_pos_x, self.stage_found.mean_pos_y)
            self.run_encounter(farming=True)

    # ------------------------- Runner -------------------------
    def run_doomtower(self):
        self.reset()
        self.check_doomtower_keys()
        self.no_run_failed = True

        if self.num_of_gold_keys == 0 and self.num_of_silver_keys < 2:
            return

        while self.no_run_failed or (
            (self.doomtower_completed or self.num_of_gold_keys == 0)
            and self.num_of_silver_keys < 2
        ):
            self.stage_found = False
            if self.num_of_gold_keys > 0 and not self.doomtower_completed:
                self.climb_doomtower()
            if self.num_of_silver_keys>1:
                self.farm_doomtower()

            if not self.stage_found:
                break

    # ------------------------- Test -------------------------
    def test(self):
        self.run_doomtower()