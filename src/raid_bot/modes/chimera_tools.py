# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 14:00:37 2025

@author: Arthur
"""

import re
import raid_bot.utils.auto_battle_tools as auto_battle_tools
import raid_bot.utils.image_tools as image_tools
import raid_bot.utils.window_tools as window_tools
import difflib
import time
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

MAX_RUN_DURATION_SECONDS = int(3.5 * 60 * 60)


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


class RSL_Bot_Chimera():
    def __init__(self, title_substring="Raid: Shadow Legends", reader = None, window = None, verbose = True, player_names = None, difficulty_order = None, threshold = 60):
        self.reader = reader
        
        self.running = True
        
        self.verbose = verbose
        self.player_names = player_names
        self.difficulty_order = difficulty_order
        self.threshold = threshold *1e6

        self.window = window
            
        self.search_areas = {
            "menu_name": [0.0, 0.02, 0.36, 0.07],   # [left, top, width, height]
            "go_to_higher_menu":   [0.928, 0.031, 0.046, 0.039],
            "clanboss_Chimera":   [0.007, 0.307, 0.072, 0.196],
            "clanboss_Chimera_Keys":   [0.554, 0.036, 0.039, 0.027],
            
            "Chimera_Keys":   [0.72, 0.038, 0.048, 0.035],

            "Chimera_Brutal":    [0.685, 0.361, 0.144, 0.152],
            "Chimera_NM":    [0.688, 0.535, 0.089, 0.142],
            "Chimera_UNM":   [0.688, 0.708, 0.089, 0.142],

            "Chimera_NameList":   [0.103, 0.136, 0.173, 0.846],
            "Chimera_EnterEncounter":   [0.756, 0.885, 0.212, 0.084],
            "Chimera_StartEncounter":   [0.763, 0.877, 0.211, 0.101],
            "Chimera_Result":   [0.385, 0.154, 0.217, 0.06],
            "Chimera_Result_Message":   [0.208, 0.255, 0.54, 0.083],
            "Chimera_FreeEncounter":   [0.701, 0.893, 0.177, 0.093],
            "Chimera_EndEncounter":   [0.801, 0.893, 0.177, 0.093],
            "Chimera_Score":     [0.2, 0.27, 0.6, 0.08],
            


        }

        self.chimera_encounter_difficulty = None 
        self.max_run_duration_seconds = MAX_RUN_DURATION_SECONDS
        self._run_deadline = None

    # ------------------------- Keys -------------------------

    def resembles(self, text, target, threshold=0.8):
        ratio = difflib.SequenceMatcher(None, text.lower(), target.lower()).ratio()
        return ratio >= threshold
    
    def update_available_keys(self):
        """Check if Demon Lord keys are available."""
        try:
            keys_text = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas['Chimera_Keys']
            )[0].text
            self.num_of_keys = int(re.findall(r"\d+", keys_text)[0])
            print(self.num_of_keys)
        except:
            self.num_of_keys = 0

    # ------------------------- Name Scan -------------------------
    def detect_cleared_difficulties(self, max_attempts=3):
        """Check which Demon Lord difficulties are already cleared."""
        self.chimera_encounters_cleared = []

        for difficulty in self.difficulty_order:
            window_tools.click_center(
                self.window, self.search_areas[f'Chimera_{difficulty}']
            )

            window_tools.move_up(self.window, strength=3, relative_x=0.25)
            name_found = False

            for _ in range(max_attempts):
                name_strings = image_tools.get_text_in_relative_area(
                    self.reader, self.window, search_area=self.search_areas['Chimera_NameList']
                )

                if any(name.text in self.player_names for name in name_strings):
                    self.chimera_encounters_cleared.append(difficulty)
                    name_found = True
                    break

                window_tools.move_down(self.window, strength=0.5, relative_x=0.25)

            if name_found:
                continue

    def check_if_friday_cet(self):
        cet_now = datetime.now(ZoneInfo("Europe/Paris"))  # CET/CEST with DST handling
        return cet_now.weekday() == 4  # Monday=0 ... Friday=4

    # ------------------------- Difficulty -------------------------
    def select_next_difficulty(self):
        """Set next Demon Lord difficulty."""
        self.chimera_encounter_difficulty = self.difficulty_order[
            len(self.chimera_encounters_cleared)
        ]

    # ------------------------- Battle Result -------------------------
    def update_battle_status(self):
        result_confirmed = False
        try:
            result = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas['Chimera_Result']
            )[0]
            time.sleep(5)
            result2 = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas['Chimera_Result']
            )[0]
            if self.resembles(result.text, 'RESULTADO') and self.resembles(result2.text, 'RESULTADO'):
                self.battle_status = 'Done'
                result_confirmed = True
        except:
            pass

    # ------------------------- Encounter -------------------------
    def execute_chimera_encounter(self):
        """Execute Demon Lord fight."""
        difficulty_key = f'Chimera_{self.chimera_encounter_difficulty}'
        window_tools.click_center(self.window, self.search_areas[difficulty_key])

        reclaim_status = image_tools.get_text_in_relative_area(
            self.reader, self.window, search_area=self.search_areas['Chimera_EnterEncounter']
        )

        if reclaim_status and self.resembles(reclaim_status[0].text ,'Reclamar'):
            window_tools.click_center(self.window, self.search_areas["Chimera_EnterEncounter"])
            window_tools.click_center(self.window, self.search_areas["Chimera_NameList"])
            window_tools.click_center(self.window, self.search_areas["Chimera_NameList"])

        window_tools.click_center(self.window, self.search_areas["Chimera_EnterEncounter"])

        window_tools.click_center(self.window, self.search_areas["Chimera_StartEncounter"])
        if not image_tools.check_startup(self):
            window_tools.click_center(self.window, self.search_areas["Chimera_StartEncounter"])


        self.battle_status = 'Starting'
        auto_battle_tools.reset_auto_battle_watchdog(self)
        while self.main_loop_running and (self.battle_status != 'Done'):
            _ensure_within_run_deadline(self, "waiting for chimera encounter result")
            
            auto_battle_tools.handle_pausa_popup(self)


            self.update_battle_status()
            auto_battle_tools.ensure_auto_battle_running(self)

        score_text = image_tools.get_text_in_relative_area(
            self.reader, self.window,self.search_areas["Chimera_Score"])[0]
        
        # Match patterns like '66.65k', '5432abc', capturing number and optional trailing letters
        matches = re.findall(r"(\d[\d.,]*)([a-zA-Z]*)", score_text.text)
        if not matches:
            raise ValueError("No numeric value found in text")

        number_part, suffix = matches[-1]  # Take the last match
        number_part = number_part.replace('.', '').replace(',', '.').replace(' ', '')

        # Try converting number part

        num = float(str(number_part))

        # Handle known suffixes
        suffix = suffix.lower()
        if suffix.startswith('k'):
            score = num * 1000
        elif suffix.startswith('m'):
            score = num * 1_000_000
        else:
            score = num

        if score > self.threshold:
            self.chimera_encounters_cleared.append(
                    self.chimera_encounter_difficulty
                )
            window_tools.click_center(self.window, self.search_areas["Chimera_EndEncounter"])
            time.sleep(5)
            window_tools.click_center(self.window, self.search_areas["Chimera_EndEncounter"])
        else:
            window_tools.click_center(self.window, self.search_areas["Chimera_FreeEncounter"])
            window_tools.sendkey("esc", window=self.window)
            self.lost_encounter = True
    # ------------------------- Main Runner -------------------------
    def run_chimera(
        self,
        main_loop_running=True,
        max_run_duration_seconds=MAX_RUN_DURATION_SECONDS,
    ):
        """Run Demon Lord encounters."""
        _start_run_deadline(self, max_run_duration_seconds)
        self.update_available_keys()
        self.main_loop_running = main_loop_running
        self.lost_encounter = False
        if self.num_of_keys == 0:
            return
        
        if self.check_if_friday_cet():
            pass
            #return

        window_tools.move_down(self.window, strength=0.5, relative_x=0.8)
        self.detect_cleared_difficulties()

        while self.main_loop_running and (len(self.chimera_encounters_cleared) < len(self.difficulty_order)):
            _ensure_within_run_deadline(self, "running chimera loop")
            window_tools.move_down(self.window, strength=0.5, relative_x=0.8)
            self.update_available_keys()
            if self.num_of_keys == 0 or self.lost_encounter:
                break

            if self.chimera_encounters_cleared != self.difficulty_order:
                self.select_next_difficulty()
                self.execute_chimera_encounter()

        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

    # ------------------------- Test -------------------------
    def test(self):
        window_tools.click_center(self.window, self.search_areas["clanboss_Chimera"])
        self.run_chimera()
