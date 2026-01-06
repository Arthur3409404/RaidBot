# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 20:04:52 2025

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
from collections import defaultdict
import threading
import subprocess
import sys
import os

import data.lib.gui.gui_tools as gui_tools

from data.lib.utils import *

import data.lib.handlers.error_handler as error_handler 

from data.lib.modes import *

class RSL_Bot_Mainframe():
    def __init__(self, title_substring="Raid: Shadow Legends"):
        self.reader = easyocr.Reader(['en'])  # You can add 'de' if you expect German text
        
        param_file = os.path.join("data", "params_mainframe.txt")   
        params = file_tools.read_params(param_file)
        self.params = self.group_params(params)
        
        self.running = True
        
        self.verbose = self.params['mainframe']['verbose']

        self.window = window_tools.find_window(title_substring)
        
        if self.window:
            self.window = window_tools.WindowObject(self.window)
            self.coords = (self.window.left, self.window.top, self.window.width, self.window.height)
            drift = self.params['mainframe']['screen_drift']
            #self.window.left += drift[0] * self.window.width
            #self.window.top += drift[1] * self.window.height
            #self.window.left = int(self.window.left)
            #self.window.top = int(self.window.top)

            # Drift is different for each screen. Drift for main station is 0
            # Drift for Omen Laptop is 0.07,0.07,0,0

            self
            print(f"Window Coordinates: {self.coords}")
        else:
            self.coords = None


        self.search_areas = {
            "menu_name": [0.008, 0.034, 0.23, 0.037],   # [left, top, width, height]
            "go_to_higher_menu":   [0.928, 0.031, 0.046, 0.039],
            "go_to_bastion":      [0.903, 0.9, 0.064, 0.059],
            "bastion_to_main_menu":      [0.808, 0.904, 0.168, 0.074],
            "detect_doomtower_rotation": [0.121, 0.696, 0.189, 0.035],
            
            "quest_menu":      [0.259, 0.911, 0.066, 0.068],
            "quest_menu_name":      [0.013, 0.032, 0.125, 0.038],
            "daily_quest_menu":      [0.033, 0.105, 0.151, 0.074],
            "weekly_quest_menu":      [0.039, 0.214, 0.143, 0.059],
            "monthly_quest_menu":      [0.036, 0.313, 0.147, 0.067],
            "advanced_quest_menu":      [0.033, 0.417, 0.15, 0.073],
            "claim_quest_rewards":      [0.455, 0.924, 0.3, 0.054],

            "time_gated_reward_menu" : [0.894, 0.817, 0.051, 0.067], 
            "time_gated_reward_menu_name" : [0.707, 0.733, 0.142, 0.04], 

            "guardian_faction_menu_name" : [0.01, 0.033, 0.24, 0.039],
            "guardian_faction_character_1" : [0.194, 0.797, 0.147, 0.073],
            "guardian_faction_character_2" : [0.353, 0.797, 0.15, 0.071],
            "guardian_faction_character_3" : [0.513, 0.796, 0.148, 0.073],
            "guardian_faction_character_4" : [0.673, 0.797, 0.147, 0.071],
            "guardian_faction_character_5" : [0.833, 0.798, 0.146, 0.071],

            'pov' : [0, 0, 1, 1],
            
            
            "main_menu_labels":      [0.007, 0.27, 0.984, 0.044],
            

            "campaign":   [0.789, 0.15, 0.177, 0.059],
            
            
            "dungeon":   [0.722, 0.039, 0.042, 0.026],
            "energy":   [0.599, 0.038, 0.069, 0.027],
            "iron_twins":   [0.137, 0.52, 0.184, 0.024],
            "iron_twins_keys":   [0.51, 0.038, 0.035, 0.025],
            "dragon":   [0.115, 0.391, 0.14, 0.024],
            "fire_knight":   [0.238, 0.659, 0.222, 0.025],
            "sand_devil":   [0.549, 0.393, 0.165, 0.029],
            "shogun":   [0.771, 0.712, 0.154, 0.028],
            
            #"fire_knight_hard_8":   [0.238, 0.659, 0.222, 0.025],
            #"iron_twins_15":   [0.238, 0.659, 0.222, 0.025],
            #"shogun_25":   [0.238, 0.659, 0.222, 0.025],

            
            
            "faction_wars":   [0.722, 0.039, 0.042, 0.026],
            
            
            "classic_arena":   [0.055, 0.407, 0.111, 0.173],
            "live_arena":   [0.452, 0.351, 0.123, 0.2],
            "tagteam_arena":   [0.829, 0.409, 0.105, 0.177],
            
            
            "clanboss_DemonLord":   [0.007, 0.307, 0.072, 0.196],
            "clanboss_DemonLord_Keys":   [0.394, 0.615, 0.208, 0.083],
            
            "clanboss_Hydra":   [0.394, 0.615, 0.208, 0.083],
            "clanboss_Hydra_Keys":   [0.394, 0.615, 0.208, 0.083],
            
            "clanboss_Chimera":   [0.394, 0.615, 0.208, 0.083],
            "clanboss_Chimera_Keys":   [0.394, 0.615, 0.208, 0.083],
            
            
            "Doom_Tower":     [0.55, 0.225, 0.429, 0.762],
            
            "Cursed_City":     [0.55, 0.225, 0.429, 0.762],
            
            "Siege":     [0.55, 0.225, 0.429, 0.762],
            
            "Grim_Forest":     [0.55, 0.225, 0.429, 0.762],



            "internet_connectivity_error_name":      [0.408, 0.335, 0.179, 0.036],
            "internet_connectivity_error_retry_connection":      [0.506, 0.53, 0.211, 0.084],

            "remote_override_error_name":      [0.428, 0.36, 0.121, 0.049],
            "remote_override_error_retry_connection":      [0.279, 0.539, 0.215, 0.087],
        }

             
        self.main_menu_names = {
            "Campaign": "Campana",
            "Dungeons": "Mazmorras",
            "FactionWars": "Guerras de Facciones",
            "Arena": "Arena",
            "ClanBoss1": "Jefes",
            "ClanBoss2": "e Clan",
            "DoomTower": "Torre del Destino",
            "CursedCity": "Ciudad Maldita",
            "Siege": "Asedio",
            "GrimmForest": "Bosque Lugubre"
        }
        
        self.dungeon_menu_names = {
            "iron_twins": "Fortaleza de los Gemelos",
            "dragon": "Guarida del Dragon",
            "fire_knight": "Castillo del Caballero de Fuego",
            "sand_devil": "Necropolis de la Arena",
            "shogun": "Arboleda del Shogun",
            
        }
        
        
        # initialize handlers
        params_classic_arena = self.params['classic_arena']
        params_tagteam_arena = self.params['tagteam_arena']
        params_live_arena = self.params['live_arena']
        params_dungeons = self.params['dungeons']
        params_factionwars = self.params['faction_wars']
        params_demonlord = self.params['demon_lord']
        params_doomtower = self.params['doom_tower']
        
        self.classic_arena_bot = arena_tools.RSL_Bot_ClassicArena(reader = self.reader, window = self.window, **params_classic_arena)
        self.tagteam_arena_bot = arena_tools.RSL_Bot_TagTeamArena(reader = self.reader, window = self.window, **params_tagteam_arena)
        self.live_arena_bot = arena_tools.RSL_Bot_LiveArena(reader = self.reader, window = self.window, **params_live_arena)
        self.dungeon_bot = dungeon_tools.RSL_Bot_Dungeons(reader = self.reader, window = self.window, **params_dungeons)
        self.factionwars_bot = factionwars_tools.RSL_Bot_FactionWars(reader = self.reader, window = self.window, **params_factionwars)
        self.demonlord_bot = demonlord_tools.RSL_Bot_DemonLord(reader = self.reader, window = self.window, **params_demonlord)
        self.doomtower_bot = doomtower_tools.RSL_Bot_DoomTower(reader = self.reader, window = self.window, **params_doomtower)
        

        self.error_handler = error_handler.RSL_Bot_ErrorHandler(reader = self.reader, window = self.window)
        self.main_loop_running = False
        self.main_loop_thread = None
        self.main_loop_stopped = False

        self.handler_init_time = time.time()
        self._start_error_checker()
        self.remote_override_time_minutes = 0.5
        # ...to be continued

        
    # =========================
    # Error Handling
    # =========================
    def _start_error_checker(self):
        """
        Start a background thread to monitor errors continuously.
        If a remote override is detected, it stops the main loop, waits,
        clicks retry, navigates back to menu, and restarts the main loop.
        """

        def run_loop():
            while self.running:
                try:
                    # Run the error handler once
                    self.error_handler.run_once()

                    # Check for remote override
                    if getattr(self.error_handler, 'remote_override_detected', True):
                        print("[ErrorHandler] Remote override detected. Handling...")

                        # Stop main loop safely
                        print("[ErrorHandler] Stopping current run_main_loop thread...")
                        self.main_loop_running = False  # signal main loop to exit
                        while not self.main_loop_stopped:
                            time.sleep(1)

                        time.sleep((self.remote_override_time_minutes+1)*60)
                        print("[ErrorHandler] Restarted Errorhandler")
                        self.main_loop_running = True
                        self.main_loop_stopped = False
                        self.error_handler.remote_override_detected = False


                except Exception as e:
                    print(f"[ErrorHandler] Exception in error checker: {e}")

                # Check every second
                time.sleep(1)

        # Start the error checker in a background daemon thread
        threading.Thread(target=run_loop, daemon=True).start()
    # =========================
    # Params Grouping
    # =========================
    def group_params(self, params: dict, min_shared_keys: int = 3):
        prefix_counts = defaultdict(int)

        for key in params:
            parts = key.split("_")
            for i in range(1, len(parts)):
                prefix_counts["_".join(parts[:i]) + "_"] += 1

        valid_prefixes = sorted(
            (p for p, c in prefix_counts.items() if c >= min_shared_keys),
            key=len,
            reverse=True
        )

        grouped = {"mainframe": {}}

        for key, value in params.items():
            for prefix in valid_prefixes:
                if key.startswith(prefix):
                    grouped.setdefault(prefix.rstrip("_"), {})[key[len(prefix):]] = value
                    break
            else:
                grouped["mainframe"][key] = value

        return grouped

    # =========================
    # Navigation Helpers
    # =========================
    def navigate_bastion_menu(self, button_area, confirm_area, confirm_string, max_attempts=10):
        for attempt in range(max_attempts):
            if not self.main_loop_running:
                break
            try:
                if attempt % 2 == 0:
                    window_tools.sendkey("esc")

                try:
                    advert = image_tools.get_text_in_relative_area(
                        self.reader, self.window,
                        self.search_areas["advert"],
                        power_detection=False
                    )[0]
                    if advert.text == "Cancelar":
                        window_tools.click_center(self.window, self.search_areas["advert"])
                except Exception:
                    pass

                window_tools.click_center(self.window, button_area)

                menu_name = image_tools.get_text_in_relative_area(
                    self.reader, self.window,
                    confirm_area,
                    power_detection=False
                )[0]

                if menu_name.text == confirm_string:
                    return True
            except Exception:
                pass

        return False

    def navigate_to_menu(self, menu_name, max_attempts=5, detect_doomtower_rotation=False):
        for _ in range(max_attempts):
            if not self.main_loop_running:
                break
            try:
                texts = image_tools.get_text_in_relative_area(
                    self.reader, self.window,
                    self.search_areas["menu_name"],
                    power_detection=False
                )

                if texts and 'Modos de' in texts[0].text:
                    break

                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
            except Exception:
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

        for direction in ("left", "right"):
            if not self.main_loop_running:
                break
            move = window_tools.move_left if direction == "left" else window_tools.move_right
            move(self.window, strength=2)
            time.sleep(2)

            labels = image_tools.get_text_in_relative_area(
                self.reader, self.window,
                self.search_areas["main_menu_labels"],
                power_detection=False
            )

            for label in labels:
                if label.text == menu_name:
                    if detect_doomtower_rotation:
                        self.detect_doomtower_rotation()
                    window_tools.click_at(label.mean_pos_x, label.mean_pos_y)
                    return

        print(f"[WARN] Menu '{menu_name}' not found")

    def navigate_to_bastion(self):
        self.navigate_to_menu(self.main_menu_names['Dungeons'])
        window_tools.click_center(self.window, self.search_areas["go_to_bastion"])

    # =========================
    # Doom Tower
    # =========================
    def detect_doomtower_rotation(self):
        rotation_text = image_tools.get_text_in_relative_area(
            self.reader, self.window,
            self.search_areas["detect_doomtower_rotation"]
        )[0].text

        if 'Arana' in rotation_text:
            self.doomtower_bot.current_rotation = '1'
        elif 'Dragon' in rotation_text:
            self.doomtower_bot.current_rotation = '2'
        elif 'Hada' in rotation_text:
            self.doomtower_bot.current_rotation = '3'

    # =========================
    # Quest Rewards
    # =========================
    def collect_quest_rewards(self, delay=2):
        time.sleep(delay)
        self.navigate_to_bastion()

        if not self.navigate_bastion_menu(
            self.search_areas["quest_menu"],
            self.search_areas["quest_menu_name"],
            'Misiones'
        ):
            print("[WARN] Missions menu not found")
            window_tools.click_center(self.window, self.search_areas["bastion_to_main_menu"])
            return

        for menu in (
            "daily_quest_menu",
            "weekly_quest_menu",
            "monthly_quest_menu",
            "advanced_quest_menu"
        ):
            window_tools.click_center(self.window, self.search_areas[menu])
            window_tools.click_center(self.window, self.search_areas["claim_quest_rewards"])

        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

        self.navigate_bastion_menu(
            self.search_areas["time_gated_reward_menu"],
            self.search_areas["time_gated_reward_menu_name"],
            'Reclamar todo'
        )

        window_tools.click_center(self.window, self.search_areas["time_gated_reward_menu"])

        for _ in range(2):
            objs = image_tools.get_text_in_relative_area(
                self.reader, self.window,
                self.search_areas['pov'],
                power_detection=False
            )
            for obj in objs:
                if obj.text == 'Ring de Guardianes':
                    window_tools.click_at(obj.mean_pos_x, obj.mean_pos_y, delay=2)
                    for i in range(1, 6):
                        window_tools.click_center(
                            self.window,
                            self.search_areas[f"guardian_faction_character_{i}"]
                        )
                    window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
                    break
            window_tools.move_left(self.window, strength=1.2)

        self.navigate_bastion_menu(
            self.search_areas["bastion_to_main_menu"],
            self.search_areas["menu_name"],
            'Modos de juego'
        )

    # =========================
    # MAIN LOOP
    # =========================
    def run_main_loop(self):
        print('Starting Main Loop...')
        self.main_loop_stopped = False
        timers = {
            "classic": None,
            "tagteam": None,
            "live": None
        }

        REFRESH_INTERVAL = 15.1 * 60
        self.main_loop_running = True
        while self.main_loop_running:
            print(self.main_loop_running)

            if self.params['run']['classic_arena'] and self.main_loop_running:
                self.navigate_to_menu(self.main_menu_names['Arena'])
                window_tools.click_center(self.window, self.search_areas['classic_arena'])
                self._handle_refresh(self.classic_arena_bot, timers, "classic", REFRESH_INTERVAL)
                self.classic_arena_bot.run_classic_arena_until_empty(main_loop_running = self.main_loop_running)
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

            if self.params['run']['tagteam_arena'] and self.main_loop_running:
                self.navigate_to_menu(self.main_menu_names['Arena'])
                window_tools.click_center(self.window, self.search_areas['tagteam_arena'])
                self._handle_refresh(self.tagteam_arena_bot, timers, "tagteam", REFRESH_INTERVAL)
                self.tagteam_arena_bot.run_tagteam_arena_single_cycle(main_loop_running = self.main_loop_running)
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

            if self.params['run']['live_arena'] and self.main_loop_running:
                self.navigate_to_menu(self.main_menu_names['Arena'])
                window_tools.click_center(self.window, self.search_areas['live_arena'])
                self.live_arena_bot.run_live_arena_loop(main_loop_running = self.main_loop_running)
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

            if self.params['run']['dungeons'] and not self.params['run']['effective_unit_leveling'] and self.main_loop_running:
                self.navigate_to_menu(self.main_menu_names['Dungeons'])
                self.dungeon_bot.run_dungeons(main_loop_running = self.main_loop_running)
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

            if self.params['run']['factionwars'] and self.main_loop_running:
                self.navigate_to_menu(self.main_menu_names['FactionWars'])
                self.factionwars_bot.run_factionwars(main_loop_running = self.main_loop_running)
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

            if self.params['run']['demonlord'] and self.main_loop_running:
                self.navigate_to_menu(self.main_menu_names['ClanBoss1'])
                window_tools.click_center(self.window, self.search_areas["clanboss_DemonLord"])
                self.demonlord_bot.run_demonlord(main_loop_running = self.main_loop_running)
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

            if self.params['run']['doomtower'] and self.main_loop_running:
                self.navigate_to_menu(self.main_menu_names['DoomTower'], detect_doomtower_rotation=True)
                self.doomtower_bot.run_doomtower(main_loop_running = self.main_loop_running)
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

            if self.main_loop_running:
                self.collect_quest_rewards()

        self.main_loop_stopped = True
    # =========================
    # Refresh Helper
    # =========================
    def _handle_refresh(self, bot, timers, key, interval):
        now = time.time()
        if timers[key] is None:
            timers[key] = now
            if now - self.handler_init_time > interval:
                bot.refresh()
        elif now - timers[key] > interval:
            timers[key] = now
            bot.refresh()

    def start_main_loop(self):
        remote_overide_possible = True
        while remote_overide_possible:
            self.run_main_loop()
            if self.main_loop_stopped:
                # Wait before retrying
                time.sleep(self.remote_override_time_minutes*60)

                # Retry internet connectivity (thread-safe for GUI)
                window_tools.click_center(
                    self.window,
                    self.search_areas["remote_override_error_retry_connection"],
                    delay=60
                )

                # Navigate back to main menu (thread-safe)
                self.main_loop_running = True
                self.navigate_bastion_menu(
                    self.search_areas["bastion_to_main_menu"],
                    self.search_areas["menu_name"],
                    'Modos de juego'
                )

                # Restart main loop in a new thread
                print("[ErrorHandler] Restarting run_main_loop after override.")
                self.start_main_loop() 







# Run test if script is executed directly
if __name__ == "__main__":
    print('ALWAYS RUN THE PROGRAM IN 1280 x 1024')

    gui = gui_tools.BotGUI()
    gui.run()


    # bot = RSL_Bot_Mainframe()
    # bot.doomtower_bot.current_rotation = '1'
    # bot.doomtower_bot.run_doomtower()
    #bot.run_main_loop()
    #bot.classic_arena_bot.run_classic_arena_until_empty()
    # bot.factionwars_bot.run_encounter()
    #bot.live_arena_bot.check_arena_coins()
    #bot.tagteam_arena_bot.run_tagteam_arena_single_cycle()

    
    # num = 1
    # for i in range(num):
    #     ul_click, lr_click = window_tools.get_two_clicks()
    #     search_area = window_tools.compile_search_area_from_clicks(
    #         ul_click,
    #         lr_click,
    #         bot
    #     )
    #     print("search_area =", search_area)
    #     image_tools.visualize_search_area(bot.coords, search_area=search_area)
    #     text_objects = image_tools.get_text_in_relative_area(bot.reader, bot.window, search_area=search_area)
    #     print(text_objects)
    

    # Tests
    
    # bot.find_clusters(text_objects)
    
    # window_tools.test_window(window_tools.find_window(bot.title_substring))

    # print("Text found in area:")
    # print(image_tools.get_text_in_full_area())
    # image_tools.visualize_text_detection(bot.coords, bot.reader)