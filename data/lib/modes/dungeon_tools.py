
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
import ast


import data.lib.utils.image_tools as image_tools
import data.lib.utils.window_tools as window_tools

from data.lib.handlers.ai_networks_handler import EnemyDataset, EvaluationNetwork

class RSL_Bot_Dungeons:
    
    def __init__(self, title_substring="Raid: Shadow Legends", reader = None,window =None, verbose = True, iron_twins_priority = True, essence_priority = 'shogun', defaults_available = ["fire_knight","dragon",'sand_devil', 'shogun'], default_difficulty = 'hard', default_level = 8 ,default_dungeon = 'fire_knight'):

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
        self.default_difficulty = default_difficulty
        self.default_level = default_level
        self.default_dungeon = default_dungeon

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
            
            "energy":   [0.599, 0.038, 0.069, 0.027],
            "iron_twins_keys":   [0.51, 0.038, 0.035, 0.025],
            'pov' : [0, 0, 1, 1],
            #"fire_knight_hard_8":   [0.238, 0.659, 0.222, 0.025],
            #"iron_twins_15":   [0.238, 0.659, 0.222, 0.025],
            #"shogun_25":   [0.238, 0.659, 0.222, 0.025],
            
            "confirm_button_champion_selection": [0.762, 0.876, 0.213, 0.104],
            
            "auto_battle_button": [0.026, 0.899, 0.058, 0.07], # its existance means battle started
            "battle_status_finished":  [0.362, 0.897, 0.269, 0.081], # check regularly for if enemy quits
            "battle_result":    [0.389, 0.148, 0.204, 0.071],
            "battle_result_2":    [0.38, 0.085, 0.224, 0.059],

            'get_dungeon_difficulty':[0.029, 0.924, 0.084, 0.035],
            'change_dungeon_difficulty_normal':[0.097, 0.803, 0.065, 0.031],
            'change_dungeon_difficulty_hard':[0.103, 0.873, 0.061, 0.034],

            'go_to_map': [0.134, 0.905, 0.059, 0.071],

            
            "test":   [0.05, 0.30, 0.15, 0.08],
            
        }

        self.dungeon_menu_names = {
            "iron_twins": "Fortaleza de los Gemelos",
            "dragon": "Guarida del Dragon",
            "fire_knight": "Castillo del Caballero de Fuego",
            "sand_devil": "Necropolis de la Arena",
            "shogun": "Arboleda del Shoc",
            "hard": "Dificil",
            "normal": "Normal"
        }
        self.hardmode_available = ["fire_knight","dragon","ice_golem",'spider']

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

    def check_difficulty(self):
        try:
            difc_txt = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas["get_dungeon_difficulty"])[0]
            if difc_txt.text == self.dungeon_menu_names[self.default_difficulty]:
                pass
            else:
                window_tools.click_center(self.window, self.search_areas["get_dungeon_difficulty"])
                string = 'change_dungeon_difficulty_' + self.default_difficulty
                window_tools.click_center(self.window, self.search_areas[string])

        except:
            print('Error changing Difficulties')
        
    def get_battle_outcome(self):
        try:
            battle_result = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas["battle_result"])[0]
            if battle_result.text == "VICTORIA" or battle_result.text == "DERROTA":
                self.battle_status = 'Done'
                self.battles_done +=1
                if battle_result.text =="VICTORIA":
                    self.battles_won +=1
                return
        except:
            pass
        
        try:
            battle_result = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas["battle_result_2"])[0]
            if battle_result.text == "VICTORIA" or battle_result.text == "DERROTA":
                self.battle_status = 'Done'
                self.battles_done +=1
                if battle_result.text =="VICTORIA":
                    self.battles_won +=1
                return
        except:
            pass
            
    
    
    def get_battle_status(self):
        try:
            auto_button = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas["auto_battle_button"])[0]
            if auto_button.text == 'Auto':
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
        print("🛡️  RAID Dungeon Bot Status")
        print("-" * 40)
        print(f"🔁 Mode: Simple Pick)")
        print(f"⏱️  Time Since Start: {formatted_elapsed}")
        print(f"⚔️  Battles Won: {self.battles_won}")
        print(f"⚔️  Battles Lost: {self.battles_done - self.battles_won}")
        print("-" * 40)
        print("🛑 To stop the bot, press 'v'")
        print("=" * 40 + "\n")

                    
    def select_encounter(self, encounter_name, max_attempts = 4):
        obj_found = False
        attempts = 0
        while attempts<max_attempts and not obj_found:
            attempts+=1

            time.sleep(2)
            objects = image_tools.get_text_in_relative_area(self.reader, self.window, self.search_areas['pov'], powerdetection=False)
            
            name_string = self.dungeon_menu_names[encounter_name]

            try:
                for obj in objects:
                    if obj.text == name_string:
                        window_tools.click_at(obj.mean_pos_x, obj.mean_pos_y, delay = 4)
                        obj_found = True
                        break
            except:
                pass
            if not obj_found:
                window_tools.move_right(self.window, strength = 1.2)


        if obj_found:
            if encounter_name in self.hardmode_available:
                self.check_difficulty()

                window_tools.move_down(self.window, strength = 1)

                if self.default_difficulty == 'hard':
                    stage = np.clip(self.default_level-3,0,7)
                else:
                    stage = np.clip(self.default_level-18,2,7)
            
            else:
                stage = 7
            

            window_tools.click_center(self.window, self.stages_buttons[stage], delay = 2)

        return obj_found



    def run_encounter(self):
        self.reset_battle_parameters()
        window_tools.click_center(self.window, self.search_areas["confirm_button_champion_selection"])
        
        while self.battle_status != 'Done':
            
            self.get_battle_outcome()

            self.get_battle_status()
        
        time.sleep(2)
        window_tools.click_center(self.window, self.search_areas["go_to_map"])

        return

    def check_energy(self):
        try:
            energy = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas['energy'])[0]
            energy = re.findall(r"\d+", energy.text)[0]
        except:
            energy = 0
        return energy

    def check_iron_twins_keys(self):
        try:
            keys = image_tools.get_text_in_relative_area(self.reader, self.window,search_area=self.search_areas['iron_twins_keys'])[0]
            keys = re.findall(r"\d+", keys.text)[0]
        except:
            keys = 0
        return keys
                    
    def run_dungeons(self):
        time.sleep(5)
        self.start_time = time.time()
        self.running = True
        time.sleep(5)

        while self.running:
            # Stop if not enough energy
            if int(self.check_energy()) < 60:
                self.running = False
                break
            encounter = None

            # Decide which encounter to run
            if self.iron_twins_priority and int(self.check_iron_twins_keys()) > 0 :
                encounter = "iron_twins"
            else:
                encounter = self.default_dungeon

            # Try to select and run encounter
            if self.select_encounter(encounter):
                self.run_encounter()
            else:
                print("Could not find encounter")

            self.print_status()