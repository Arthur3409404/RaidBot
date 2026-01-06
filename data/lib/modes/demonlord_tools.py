# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 14:00:37 2025

@author: Arthur
"""

import numpy as np
import re
import data.lib.utils.image_tools as image_tools
import data.lib.utils.window_tools as window_tools

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
            
            "DemonLord_Keys":   [0.72, 0.038, 0.048, 0.035],

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

    # ------------------------- Keys -------------------------
    def update_available_keys(self):
        """Check if Demon Lord keys are available."""
        try:
            keys_text = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas['DemonLord_Keys']
            )[0].text
            self.num_of_keys = int(re.findall(r"\d+", keys_text)[0])
            print(self.num_of_keys)
        except:
            self.num_of_keys = 0

    # ------------------------- Name Scan -------------------------
    def detect_cleared_difficulties(self, max_attempts=3):
        """Check which Demon Lord difficulties are already cleared."""
        self.demonlord_encounters_cleared = []

        for difficulty in self.difficulty_order:
            window_tools.click_center(
                self.window, self.search_areas[f'DemonLord_{difficulty}']
            )

            window_tools.move_up(self.window, strength=3, relative_x=0.25)
            name_found = False

            for _ in range(max_attempts):
                name_strings = image_tools.get_text_in_relative_area(
                    self.reader, self.window, search_area=self.search_areas['DemonLord_NameList']
                )

                if any(name.text in self.player_names for name in name_strings):
                    self.demonlord_encounters_cleared.append(difficulty)
                    name_found = True
                    break

                window_tools.move_down(self.window, strength=0.5, relative_x=0.25)

            if name_found:
                continue

    # ------------------------- Difficulty -------------------------
    def select_next_difficulty(self):
        """Set next Demon Lord difficulty."""
        self.demonlord_encounter_difficulty = self.difficulty_order[
            len(self.demonlord_encounters_cleared)
        ]

    # ------------------------- Battle Result -------------------------
    def update_battle_status(self):
        try:
            result = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas['DemonLord_Result']
            )[0]
            if result.text == 'RESULTADO':
                self.battle_status = 'Done'
                self.demonlord_encounters_cleared.append(
                    self.demonlord_encounter_difficulty
                )
        except:
            pass

    # ------------------------- Encounter -------------------------
    def execute_demonlord_encounter(self):
        """Execute Demon Lord fight."""
        difficulty_key = f'DemonLord_{self.demonlord_encounter_difficulty}'
        window_tools.click_center(self.window, self.search_areas[difficulty_key])

        reclaim_status = image_tools.get_text_in_relative_area(
            self.reader, self.window, search_area=self.search_areas['DemonLord_EnterEncounter']
        )

        if reclaim_status and reclaim_status[0].text == 'Reclamar':
            window_tools.click_center(self.window, self.search_areas["DemonLord_EnterEncounter"])
            window_tools.click_center(self.window, self.search_areas["DemonLord_NameList"])
            window_tools.click_center(self.window, self.search_areas["DemonLord_NameList"])

        window_tools.click_center(self.window, self.search_areas["DemonLord_EnterEncounter"])
        window_tools.click_center(self.window, self.search_areas["DemonLord_StartEncounter"])

        self.battle_status = 'Starting'
        while self.battle_status != 'Done':
            self.update_battle_status()

        window_tools.click_center(self.window, self.search_areas["DemonLord_EndEncounter"])
        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

    # ------------------------- Main Runner -------------------------
    def run_demonlord(self):
        """Run Demon Lord encounters."""
        self.update_available_keys()
        if self.num_of_keys == 0:
            return

        self.detect_cleared_difficulties()

        while len(self.demonlord_encounters_cleared) < len(self.difficulty_order):
            self.update_available_keys()
            if self.num_of_keys == 0:
                break

            if self.demonlord_encounters_cleared != self.difficulty_order:
                self.select_next_difficulty()
                self.execute_demonlord_encounter()

        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

    # ------------------------- Test -------------------------
    def test(self):
        window_tools.click_center(self.window, self.search_areas["clanboss_DemonLord"])
        self.run_demonlord()
