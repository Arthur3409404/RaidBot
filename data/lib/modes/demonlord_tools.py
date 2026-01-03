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
import ast


import data.lib.utils.image_tools as image_tools
import data.lib.utils.window_tools as window_tools
import data.lib.utils.file_tools as file_tools


import data.lib.modes.arena_tools as arena_tools
import data.lib.modes.hydra_tools as hydra_tools

class RSL_Bot_DemonLord():
    def __init__(self, title_substring="Raid: Shadow Legends", reader = None, window = None, verbose = True, player_names = None, difficulty_order = None):
        self.reader = reader
        
        self.running = True
        
        self.verbose = verbose
        self.player_names = player_names
        self.difficulty_order = difficulty_order

        self.window = window
            
        self.search_areas = {
            "menu_name": [0.008, 0.034, 0.23, 0.037],   # [left, top, width, height]
            "go_to_higher_menu":   [0.928, 0.031, 0.046, 0.039],
            "clanboss_DemonLord":   [0.007, 0.307, 0.072, 0.196],
            "clanboss_DemonLord_Keys":   [0.554, 0.036, 0.039, 0.027],
            
            "DemonLord_Keys":   [0.732, 0.036, 0.042, 0.029],

            "DemonLord_Easy":   [0.599, 0.140, 0.384, 0.105],
            "DemonLord_Normal":   [0.599, 0.260, 0.384, 0.105],
            "DemonLord_Hard":   [0.599, 0.380, 0.384, 0.105],
            "DemonLord_Brutal":   [0.599, 0.496, 0.384, 0.105],
            "DemonLord_NM":   [0.599, 0.618, 0.384, 0.105],
            "DemonLord_UNM":   [0.596, 0.738, 0.39, 0.109],

            "DemonLord_NameList":   [0.103, 0.136, 0.173, 0.846],
            "DemonLord_EnterEncounter":   [0.756, 0.885, 0.212, 0.084],
            "DemonLord_StartEncounter":   [0.763, 0.877, 0.211, 0.101],
            "DemonLord_Result":   [0.385, 0.154, 0.217, 0.06],
            "DemonLord_Result_Message":   [0.208, 0.255, 0.54, 0.083],
            "DemonLord_EndEncounter":   [0.371, 0.886, 0.217, 0.07],
            


        }

        self.demonlord_encounter_difficulty = None 

    def check_demonlord_keys(self):
        """Check if demon lord keys are available."""
        try:
            DemonLord_Keys = image_tools.get_text_in_relative_area(self.reader, self.window, search_area=self.search_areas['DemonLord_Keys'])
            num_of_keys = re.findall(r"\d+", DemonLord_Keys[0].text)[0]
            self.num_of_keys = int(num_of_keys)

        except:
            self.num_of_keys = 0

    def check_list_of_names(self, max_attempts = 3):
        """Validate or refresh the list of demon lord names."""
        self.demonlord_encounters_cleared = []
        for difficulty in self.difficulty_order:
            string = 'DemonLord_'+difficulty
            window_tools.click_center(self.window, self.search_areas[string])
            name_found = False

            window_tools.move_up(self.window, strength = 3, relative_x_pos= 0.25)
            for attempt in range(max_attempts): 
                name_strings = image_tools.get_text_in_relative_area(self.reader, self.window, search_area=self.search_areas['DemonLord_NameList'])
                for name_string in name_strings:
                    if name_string.text in self.player_names:
                        name_found = True
                        self.demonlord_encounters_cleared.append(difficulty)
                        break
                if name_found:
                    break
                window_tools.move_down(self.window, strength = 0.5, relative_x_pos= 0.25)


    def set_difficulty(self):
        """Set the next demon lord difficulty."""
        self.demonlord_encounter_difficulty = self.difficulty_order[len(self.demonlord_encounters_cleared)]

    def check_battle_outcome(self):
        result = image_tools.get_text_in_relative_area(self.reader, self.window, search_area=self.search_areas['DemonLord_Result'])
        try:
            if result[0].text == 'RESULTADO':
                #result_message = image_tools.get_text_in_relative_area(self.reader, self.window, search_area=self.search_areas['DemonLord_Result_Message'])
                self.battle_status = 'Done'
                self.demonlord_encounters_cleared.append(self.demonlord_encounter_difficulty)
        except:
            pass



    def handle_demonlord_encounter(self):
        """Execute the demon lord fight logic."""
        string = 'DemonLord_'+ self.demonlord_encounter_difficulty
        window_tools.click_center(self.window, self.search_areas[string])
        reclaim_status = image_tools.get_text_in_relative_area(self.reader, self.window, search_area=self.search_areas['DemonLord_EnterEncounter'])
        if reclaim_status[0].text == 'Reclamar':
            window_tools.click_center(self.window, self.search_areas["DemonLord_EnterEncounter"])
            window_tools.click_center(self.window, self.search_areas["DemonLord_NameList"])
            window_tools.click_center(self.window, self.search_areas["DemonLord_NameList"])

        window_tools.click_center(self.window, self.search_areas["DemonLord_EnterEncounter"])
        window_tools.click_center(self.window, self.search_areas["DemonLord_StartEncounter"])

        self.battle_status = 'Starting'
        while self.battle_status != 'Done':
                    self.check_battle_outcome()

        window_tools.click_center(self.window, self.search_areas["DemonLord_EndEncounter"])
        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])


    def run_demonlord(self):
        """Run Demon Lord Encounter"""      
        self.check_demonlord_keys()
        if self.num_of_keys == 0:
            return

        self.check_list_of_names()
        while not len(self.demonlord_encounters_cleared) == len(self.difficulty_order):
            self.check_demonlord_keys()
            if self.num_of_keys == 0:
                break
            
            if self.demonlord_encounters_cleared != self.difficulty_order:
                self.set_difficulty()
                self.handle_demonlord_encounter()
        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
            
                
            


    def test(self):
        window_tools.click_center(self.window, self.search_areas["clanboss_DemonLord"])
        self.run_demonlord()

