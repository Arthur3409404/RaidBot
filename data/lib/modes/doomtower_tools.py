# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 14:00:37 2025

@author: Arthur
"""

import pygetwindow as gw
import pyautogui
import matplotlib.pyplot as plt
import cv2
import numpy as np
import easyocr
from skimage.metrics import structural_similarity as ssim
import time
import keyboard 
import re
from datetime import datetime, timedelta
import os
import math
import ast


import data.lib.utils.image_tools as image_tools
import data.lib.utils.window_tools as window_tools
import data.lib.utils.file_tools as file_tools


import data.lib.modes.arena_tools as arena_tools
import Backup.hydra_tools_old as hydra_tools_old

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
            "doom_tower_restart_encounter":   [0.463, 0.877, 0.211, 0.106],
            "doom_tower_automatic_climb":    [0.373, 0.936, 0.194, 0.036],

            "doom_tower_battle_result_automatic_climb":    [0.369, 0.339, 0.263, 0.04],
            "doom_tower_battle_result_automatic_climb_2":    [0.355, 0.381, 0.283, 0.044],
            "doom_tower_battle_result":    [0.396, 0.161, 0.195, 0.056],
            # "doom_tower_battle_result_2":    [0.38, 0.085, 0.224, 0.059],
            "doom_tower_close_encounter":   [0.13, 0.898, 0.064, 0.076],
            
            


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

    def reset(self):
        self.battle_status = 'Starting'


    def get_current_stagefield(self):
        text_objects = image_tools.get_text_in_relative_area(self.reader, self.window, search_area=self.search_areas['pov'])
        list_of_stages = []
        for s in text_objects:
            match = re.search(r'^P(\d+)$', s.text.strip())  # stricter: must start with P and digits only
            if match:
                s.text = match.group(1)  # replace with just the digits
                list_of_stages.append(s)
        return list_of_stages

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

    def check_doomtower_keys(self):
        """Check if demon lord keys are available."""
        try:
            doom_tower_keys = image_tools.get_text_in_relative_area(self.reader, self.window, search_area=self.search_areas['doom_tower_keys'])
            # Gold Keys
            num_of_gold_keys = re.findall(r"\d+", doom_tower_keys[0].text)[0]
            self.num_of_gold_keys = int(num_of_gold_keys)
            print(doom_tower_keys[0].text)
            # Silver Keys
            num_of_silver_keys = re.findall(r"\d+", doom_tower_keys[1].text)[0]
            self.num_of_silver_keys = int(num_of_silver_keys)
            print(doom_tower_keys[1].text)

        except:
            self.num_of_gold_keys = 0
            self.num_of_silver_keys = 0
        

    def check_list_of_builds(self, max_attempts = 3):
        """Validate or refresh the list of demon lord names."""
        window_tools.move_down(self.window, strength = 0.5, relative_x_pos= 0.25)

    def _reconstruct_stage_numbers(self, stages_numbers):
        if not stages_numbers:
            return stages_numbers

        # Extract current numbers (None for invalid text)
        original_nums = []
        for s in stages_numbers:
            try:
                original_nums.append(int(s.text))
            except ValueError:
                original_nums.append(None)

        n = len(stages_numbers)

        # Determine candidate starting numbers from the largest valid number
        max_original = max(num for num in original_nums if num is not None)

        best_sequence = None
        best_score = -1

        # Try candidate starts in a reasonable range
        for start in range(max_original, max_original - 21, -1):
            seq = []
            current = start
            for _ in range(n):
                while current % 10 == 0:
                    current -= 1
                seq.append(current)
                current -= 1

            # Score sequence: how many numbers match the original
            score = sum(1 for s_num, o_num in zip(seq, original_nums) if o_num is not None and s_num == o_num)

            if score > best_score:
                best_score = score
                best_sequence = seq

        # Update .text in-place
        for obj, num in zip(stages_numbers, best_sequence):
            obj.text = str(num)

        return stages_numbers


    def set_difficulty(self, set_level = None):
        """Set the next demon lord difficulty."""
        difficulty = image_tools.get_text_in_relative_area(self.reader, self.window, search_area=self.search_areas['doom_tower_difficulty_current'])[0]
        if difficulty.text in self.translation_mapping.keys():
            self.current_difficulty = self.translation_mapping[difficulty.text]
        
        if set_level is not None and self.current_difficulty != set_level:
            string = 'doom_tower_difficulty_switch_' + set_level
            window_tools.click_center(self.window, self.search_areas['doom_tower_difficulty_current'])
            window_tools.click_center(self.window, self.search_areas[string], delay = 5)
            self.current_difficulty = set_level

    def check_battle_outcome(self):
        try:
            battle_result = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas["doom_tower_automatic_climb"])[0]
            if 'Batallas completadas' in battle_result.text:
                self.battle_status = 'autoclimb'

        except:
            pass

        try:
            battle_result = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas["doom_tower_battle_result_automatic_climb"])[0]
            if battle_result.text == "Autoescalada completada":
                self.battle_status = 'autoclimb_Done'
                window_tools.click_center(self.window, self.search_areas["doom_tower_close_encounter"], delay=5)
        except:
            pass
        
        try:
            battle_result = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas["doom_tower_battle_result_automatic_climb_2"])[0]
            if battle_result.text == "Autoescalada completada":
                self.battle_status = 'autoclimb_Done'
                window_tools.click_center(self.window, self.search_areas["doom_tower_close_encounter"], delay=5)
        except:
            pass


        try:
            battle_result = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas["doom_tower_battle_result"])[0]
            if (battle_result.text == "VICTORIA" or battle_result.text == "DERROTA") and self.battle_status != 'autoclimb':
                self.battle_status = 'Done'
                self.battles_done +=1
                if battle_result.text =="VICTORIA":
                    self.battles_won +=1
                else:
                    self.no_run_failed = False
        except:
            pass

    def setup_encounter(self):
        doom_tower_menu_name = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas["doom_tower_menu_name"])[0]
        number = re.findall(r'\d+', doom_tower_menu_name.text)[0]

        if 'Jefe Final' in doom_tower_menu_name.text or int(number)%10 == 0:
            if 'Jefe Final' in doom_tower_menu_name.text:
                stage = '120'
            else:
                stage = number
            current_opponent = self.doomtower_rotations[self.current_rotation][stage]
            print(current_opponent)
        else:
            stage = number
            current_opponent = 'Waves'
        self.select_build(current_opponent)


    def select_build(self, setup):
        self.current_setup = False
        window_tools.click_center(self.window, self.search_areas["doom_tower_setup_section_groups"])
        window_tools.move_up(self.window, strength = 3, relative_x_pos= 0.15)
        for i in range(3):
            doom_tower_setup_names = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas["doom_tower_setup_names"])
            for name in doom_tower_setup_names:
                if name.text == setup:
                    self.current_setup = name
                    break
            if self.current_setup != False:
                break
            window_tools.move_down(self.window, strength = 0.5, relative_x_pos= 0.15)
        if self.current_setup != False:
            object =  image_tools.get_simliarities_in_relative_area(self.window, self.search_areas["doom_tower_setup_check"], 'pic\doom_tower_completed_stage.png')
        if len(object) > 0:
            window_tools.click_center(self.window, self.search_areas["doom_tower_start_encounter"])
        else:
            window_tools.click_at(self.current_setup.mean_pos_x - 268.0, self.current_setup.mean_pos_y + 70)
            window_tools.click_center(self.window, self.search_areas["doom_tower_start_encounter"])


    def run_encounter(self, farming = False, max_attempts = 15):
        self.setup_encounter()
        run = True
        window_tools.click_center(self.window, self.search_areas["doom_tower_start_encounter"])
        time.sleep(10)
        attempt = 0
        while run:
            attempt +=1
            if max_attempts>attempt:
                break

            self.check_battle_outcome()
            time.sleep(2)
            if self.battle_status == 'Done' and not farming:
                break
            if self.battle_status == 'Done' and farming:
                window_tools.click_center(self.window, self.search_areas["doom_tower_restart_encounter"])
                time.sleep(10)
                self.battle_status == 'Starting'
                
                try:
                    battle_result = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas["doom_tower_battle_result"])[0]
                    if (battle_result.text == "VICTORIA" or battle_result.text == "DERROTA"):
                        self.battle_status = 'Done'
                except:
                    pass

                if self.battle_status == 'Done' and not farming:
                    break

        print('Battle_Done')
        window_tools.click_center(self.window, self.search_areas["doom_tower_close_encounter"], delay=5)

    def _check_boss_stage(self):
        if self.highest_stage_available== 119:
            x_pos = int( self.window.left + self.window.width/2 ) 
            y_pos = int( self.highest_stage.mean_pos_y - self.window.height * 0.2 )
            window_tools.click_at(x_pos, y_pos)
        else:
            x_pos = int( self.window.left + self.window.width * 0.98 ) 
            y_pos = int( self.highest_stage.mean_pos_y - self.window.height * 0.1 )
            window_tools.click_at(x_pos, y_pos)

        stage_completed = image_tools.get_simliarities_in_relative_area(
            self.window,
            self.search_areas["doom_tower_check_boss_stage_complete"],
            'pic\\doom_tower_locked_stage.png'
        )
        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
        if len(stage_completed)==1:
            return False
        else:
            return True
        


    def get_highest_stage_available(self, max_attempts = 10):
        # 1. Detect completed stages (icons)
        self.highest_stage_available = 1
        end_reached = False
        attempts = 0
        while self.highest_stage_available != 120 and attempts != max_attempts and end_reached == False:
            attempts += 0
            stages_completed = image_tools.get_simliarities_in_relative_area(
                self.window,
                self.search_areas["pov"],
                'pic\\doom_tower_completed_stage.png'
            )

            if not stages_completed:
                self.highest_stage_available = None
                return None

            # 2. Highest stage is visually highest (smallest Y)
            stages_completed.sort(key=lambda o: o.mean_pos_y)

            # 3. Detect stage number text objects
            stages_numbers = self.get_current_stagefield()
            self._reconstruct_stage_numbers(stages_numbers)

            radius = 100
            backtrack_count = 0

            # 4. Walk from highest completed stage downward
            stage_match = False
            for completed_stage in stages_completed:
                for stage_number in stages_numbers:
                    if stage_number.text is None:
                        continue

                    if self.is_within_radius(completed_stage, stage_number, radius):
                        base_stage = int(stage_number.text)
                        self.highest_stage_available = base_stage + backtrack_count
                        self.highest_stage = completed_stage
                        stage_match = True
                        break
                if stage_match:
                    break
                # No number found → move down one completed stage
                backtrack_count += 1

            if (self.highest_stage.mean_pos_y - self.window.top)/self.window.height <0.3:
                end_reached = False
                window_tools.move_up(self.window, strength = 1, relative_x_pos= 0.1)
                continue
            if (self.highest_stage.mean_pos_y - self.window.top)/self.window.height >=0.3:
                end_reached = True

            if self.highest_stage_available%10 == 9:
                stage_done = self._check_boss_stage()
                if stage_done :
                    increment = 2
                else:
                    increment = 1
                self.highest_stage_available += increment 
            self.highest_stage_available = max(1, min(120, self.highest_stage_available))
            

    def check_for_boss_and_current_stage(self, target = None):
        FIRST_PATH = 'pic\\doom_tower_current_stage.png'
        list_of_paths = [FIRST_PATH]
        expected_menu_names = {FIRST_PATH: 'Planta'}

        self.stage_found = False
        self.doomtower_climb_status = False

        if target:
            list_of_paths = []
            expected_menu_names = {}    
            for value in self.doomtower_rotations[self.current_rotation].values():
                if value in self.translation_mapping and value == target:
                    path = f"pic\\doom_tower_{self.translation_mapping[value]}.png"
                    if path not in list_of_paths:      # prevents duplicates
                        list_of_paths.append(path)
                        expected_menu_names[path] = value

        else:
            for value in self.doomtower_rotations[self.current_rotation].values():
                if value in self.translation_mapping:
                    path = f"pic\\doom_tower_{self.translation_mapping[value]}.png"
                    if path not in list_of_paths:      # prevents duplicates
                        list_of_paths.append(path)
                        expected_menu_names[path] = value

        for path in list_of_paths:
            set_threshold_comparison = 0.8
            possible_stages = []
            threshold_min = 0.25

            while len(possible_stages) <1 and set_threshold_comparison>threshold_min:
                possible_stages = image_tools.get_simliarities_in_relative_area(
                    self.window,
                    self.search_areas["pov"],
                    path,
                    threshold = set_threshold_comparison,
                    scales =  [0.7, 0.8, 0.9, 1.0]
                )
                set_threshold_comparison -=0.03


            for i in range(len(possible_stages)):
                window_tools.click_at(possible_stages[i].mean_pos_x, possible_stages[i].mean_pos_y)
                
                try:
                    doom_tower_menu_name = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas["doom_tower_menu_name"])[0]
                    excpected_menu_name = expected_menu_names[path]
                    try:
                        number = re.findall(r'\d+', doom_tower_menu_name.text)[0]
                    except:
                        number = 1

                    if doom_tower_menu_name.text == 'Torre del Destino':
                        continue 

                    if excpected_menu_name == 'Planta' and excpected_menu_name in doom_tower_menu_name.text:
                        self.stage_found = possible_stages[i]
                        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
                        self.highest_stage_available[self.current_difficulty] = number
                        break

                    if excpected_menu_name not in self.doomtower_rotations[self.current_rotation][str(number)] and 'Jefe Final' not in doom_tower_menu_name.text:
                        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
                        continue
                    

                    if 'Jefe Final' in doom_tower_menu_name.text or int(number)%10 == 0:
                        stage_completed = image_tools.get_simliarities_in_relative_area(
                        self.window,
                        self.search_areas["doom_tower_check_boss_stage_complete"],
                        'pic\\doom_tower_locked_stage.png'
                        )
                        if not (stage_completed)==1 and 'Jefe Final' in doom_tower_menu_name.text:
                            self.doomtower_climb_status = 'completed'
                        if 'Jefe Final' in doom_tower_menu_name.text:
                            self.highest_stage_available[self.current_difficulty] = 120
                        else:
                            self.highest_stage_available[self.current_difficulty] = number
                        
                        self.stage_found = possible_stages[i]
                        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
                        break
                except:
                    pass
            if self.stage_found:
                break



    def simple_find_highest_stage(self, target = None):
        stage_found = False
        for i in range(7):
            window_tools.move_up(self.window, strength = 3, relative_x_pos= 0.1)
        for i in range(25):
            self.check_for_boss_and_current_stage(target = target)
            if self.stage_found:
                break
            window_tools.move_down(self.window, strength = 0.6, relative_x_pos= 0.1)
        

            
 
    def goto_stage(self):
        pass



    def climb_doomtower(self):
        self.set_difficulty('hard')
        if not self.doomtower_climb_status_hard == 'completed':
            self.simple_find_highest_stage()
        if self.doomtower_climb_status == 'completed' or self.doomtower_climb_status_hard == 'completed':
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


    def farm_doomtower(self):
        self.set_difficulty(self.setup['difficulty'])
        for opponent in self.setup['priority_bosses']:
            if opponent in self.doomtower_rotations[self.current_rotation].keys():
                self.farming_opponent = opponent
                break
        self.simple_find_highest_stage(target = opponent)

        if self.stage_found:
            window_tools.click_at(self.stage_found.mean_pos_x, self.stage_found.mean_pos_y)
            self.run_encounter(farming = True)
        

        



    def run_doomtower(self):
        """Run Demon Lord Encounter"""    
        self.reset()  
        self.check_doomtower_keys()
        self.no_run_failed = True
        if self.num_of_gold_keys == 0 and self.num_of_silver_keys<2:
            return
        while self.no_run_failed or ((self.doomtower_completed or self.num_of_gold_keys==0) and self.num_of_silver_keys<2):
            if self.num_of_gold_keys>0 and self.doomtower_completed == False:
                self.climb_doomtower()
            else:
                self.farm_doomtower()
            if not self.stage_found:
                break


        
            
                
            


    def test(self):
        # self.check_for_boss_and_current_stage()
        # if not self.doomtower_completed:
        #     window_tools.click_at(self.stage_found.mean_pos_x, self.stage_found.mean_pos_y)
        #     self.run_encounter()

        self.run_doomtower()

