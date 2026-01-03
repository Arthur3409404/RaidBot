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

import data.lib.utils.image_tools as image_tools
import data.lib.utils.window_tools as window_tools
import data.lib.utils.file_tools as file_tools

import data.lib.handlers.error_handler as error_handler 

import data.lib.modes.arena_tools as arena_tools
import data.lib.modes.dungeon_tools as dungeon_tools
import data.lib.modes.factionwars_tools as factionwars_tools
import data.lib.modes.demonlord_tools as demonlord_tools

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
        # param_file_classic_arena = os.path.join("data", "params_classicarena.txt")
        
        # params_classic_arena = file_tools.read_params(param_file_classic_arena)
        
        # param_file_tagteam_arena = os.path.join("data", "params_tagteamarena.txt")
        # params_tagteam_arena = file_tools.read_params(param_file_tagteam_arena)
        
        # param_file_live_arena = os.path.join("data", "params_livearena.txt")
        # params_live_arena = file_tools.read_params(param_file_live_arena)
        
        # param_file_classic_arena = os.path.join("data", "params_classicarena.txt")
        
        # params_classic_arena = file_tools.read_params(param_file_classic_arena)
        params_classic_arena = self.params['classic_arena']
        params_tagteam_arena = self.params['tagteam_arena']
        params_live_arena = self.params['live_arena']
        params_dungeons = self.params['dungeons']
        params_factionwars = self.params['faction_wars']
        params_demonlord = self.params['demon_lord']
        
        self.classic_arena_bot = arena_tools.RSL_Bot_ClassicArena(reader = self.reader, window = self.window, **params_classic_arena)
        self.tagteam_arena_bot = arena_tools.RSL_Bot_TagTeamArena(reader = self.reader, window = self.window, **params_tagteam_arena)
        self.live_arena_bot = arena_tools.RSL_Bot_LiveArena(reader = self.reader, window = self.window, **params_live_arena)
        self.dungeon_bot = dungeon_tools.RSL_Bot_Dungeons(reader = self.reader, window = self.window, **params_dungeons)
        self.factionwars_bot = factionwars_tools.RSL_Bot_FactionWars(reader = self.reader, window = self.window, **params_factionwars)
        self.demonlord_bot = demonlord_tools.RSL_Bot_DemonLord(reader = self.reader, window = self.window, **params_demonlord)

        self.error_handler = error_handler.RSL_Bot_ErrorHandler(reader = self.reader, window = self.window)
        
        self.handler_init_time = time.time()
        # ...to be continued

        
     
    def _start_error_checker(self):
        """Start a background thread to call self.check_for_errors() every 15 seconds."""
        def run_loop():
            while self.running:  # stops when self.running = False
                try:
                    self.error_handler.run_once()
                except Exception as e:
                    print(f"Error in check_for_errors: {e}")
                
                if not self.running:
                    break
                time.sleep(1)

        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()

    def group_params(self, params: dict, min_shared_keys: int = 3):
        """
        Groups params by common prefixes.
        Everything not belonging to a detected group goes into 'mainframe'.
        """
    
        # Step 1: find all possible prefixes
        prefix_counts = defaultdict(int)
    
        for key in params.keys():
            parts = key.split("_")
            for i in range(1, len(parts)):
                prefix = "_".join(parts[:i]) + "_"
                prefix_counts[prefix] += 1
    
        # Step 2: keep only prefixes with enough shared keys
        valid_prefixes = {
            p for p, count in prefix_counts.items()
            if count >= min_shared_keys
        }
    
        # Step 3: prefer the longest matching prefix (avoid overlaps)
        valid_prefixes = sorted(valid_prefixes, key=len, reverse=True)
    
        grouped = {"mainframe": {}}
    
        for key, value in params.items():
            matched = False
    
            for prefix in valid_prefixes:
                if key.startswith(prefix):
                    group_name = prefix.rstrip("_")
                    stripped_key = key[len(prefix):]
    
                    grouped.setdefault(group_name, {})
                    grouped[group_name][stripped_key] = value
                    matched = True
                    break
    
            if not matched:
                grouped["mainframe"][key] = value
    
        return grouped
        
    def manouver_bastion(self, button_area, confirm_area, confirm_string, max_attempts=10):
        attempts = 0
        manouver_success = False
    
        while attempts < max_attempts:
            attempts += 1
            try:
                # Press ESC to clear any open menus (default delay = 2s)
                if attempts%2==0:
                    window_tools.sendkey("esc")
    
                # Try to close advert popup if present
                try:
                    advert = image_tools.get_text_in_relative_area(
                        self.reader,
                        self.window,
                        self.search_areas["advert"],
                        powerdetection=False
                    )[0]
                    if advert and advert.text == "Cancelar":
                        window_tools.click_center(
                            self.window,
                            self.search_areas["advert"]
                        )
                except Exception:
                    pass  # Advert not present or unreadable
    
                # Click the bastion manoeuvre button
                window_tools.click_center(self.window, button_area)
    
                # Confirm manoeuvre menu opened correctly
                quest_menu_name = image_tools.get_text_in_relative_area(
                    self.reader,
                    self.window,
                    confirm_area,
                    powerdetection=False
                )[0]
    
                if quest_menu_name.text == confirm_string:
                    manouver_success = True
                    break
    
            except Exception:
                pass  # Any unexpected failure → retry
    
        return manouver_success
     
    def check_quest_rewards(self, delay = 2):
        time.sleep(delay)
        self.go_to_bastion_from_menu()
        attempts = 0
        quest_menu_found = self.manouver_bastion(self.search_areas["quest_menu"], self.search_areas["quest_menu_name"], 'Misiones')

        if quest_menu_found:
            window_tools.click_center(self.window, self.search_areas["daily_quest_menu"])
            window_tools.click_center(self.window, self.search_areas["claim_quest_rewards"])
            window_tools.click_center(self.window, self.search_areas["weekly_quest_menu"])
            window_tools.click_center(self.window, self.search_areas["claim_quest_rewards"])
            window_tools.click_center(self.window, self.search_areas["monthly_quest_menu"])
            window_tools.click_center(self.window, self.search_areas["claim_quest_rewards"])
            window_tools.click_center(self.window, self.search_areas["advanced_quest_menu"])
            window_tools.click_center(self.window, self.search_areas["claim_quest_rewards"])
            
            window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
            
            # Time Gated Rewards
            quest_menu_found = self.manouver_bastion(self.search_areas["time_gated_reward_menu"], self.search_areas["time_gated_reward_menu_name"], 'Reclamar todo')
            window_tools.click_center(self.window, self.search_areas["time_gated_reward_menu_name"])
            window_tools.click_center(self.window, self.search_areas["time_gated_reward_menu"])

            # Guardian Faction Ring
            obj_found = False
            for i in range(2):
                objects = image_tools.get_text_in_relative_area(self.reader, self.window, self.search_areas['pov'], powerdetection=False)
                for obj in objects:
                    if obj.text == 'Ring de Guardianes':
                        obj_found = True
                        break
                if obj_found:
                    window_tools.click_at(obj.mean_pos_x, obj.mean_pos_y, delay = 2)

                    # menu_name = image_tools.get_text_in_relative_area(self.reader, self.window, self.search_areas["guardian_faction_menu_name"], powerdetection=False)[0]
                    # if menu_name == 'Ring de Guardianes'
                    #     in_menu = True
                    
                    for i in range(5):
                        string = "guardian_faction_character_" + str(i+1)
                        window_tools.click_center(self.window, self.search_areas[string])
                    window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

                    break
                window_tools.move_left(self.window, strength = 1.2)

            



            main_menu_found = self.manouver_bastion(self.search_areas["bastion_to_main_menu"], self.search_areas["menu_name"], 'Modos de juego')
            if main_menu_found:
                return
            else:
                print('Main Menu Manouver Failed')
        else:
            print('Missions Menu could not be found')
            window_tools.click_center(self.window, self.search_areas["bastion_to_main_menu"])
        
        
    def go_to_bastion_from_menu(self):
        self.go_to_menu(self.main_menu_names['Dungeons'])
        window_tools.click_center(self.window, self.search_areas["go_to_bastion"])
        
        
    def go_to_menu(self, menu_name, max_attempts=5):
        attempts = 0
    
        while attempts < max_attempts:
            try:
                # Get current text objects in the menu area
                objects = image_tools.get_text_in_relative_area(
                    self.reader,
                    self.window,
                    self.search_areas["menu_name"],
                    powerdetection=False
                )
    
                if not objects:
                    print("No text objects found. Clicking to higher menu...")
                    window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
                    attempts += 1

                    continue
    
                first_text = objects[0].text.strip()
    
                if 'Modos de' in first_text:
                    # Found the main menu
                    break
                else:
                    print(f"Current text: '{first_text}'. Not main menu. Going up one level...")
                    window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
                    attempts += 1

    
            except Exception as e:
                print(f"Exception occurred: {e}. Trying higher menu...")
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
                attempts += 1

        window_tools.move_left(self.window, strength = 2)
        time.sleep(2)
        objects = image_tools.get_text_in_relative_area(self.reader, self.window, self.search_areas["main_menu_labels"], powerdetection=False)
        obj_found = False
        
        for obj in objects:
            #print(obj.text)
            if obj.text == menu_name:
                window_tools.click_at(obj.mean_pos_x, obj.mean_pos_y)
                obj_found = True
                return
        window_tools.move_right(self.window, strength = 2)
        time.sleep(2)
        objects = image_tools.get_text_in_relative_area(self.reader, self.window, self.search_areas["main_menu_labels"], powerdetection=False)
        for obj in objects:
            #print(obj.text)
            if obj.text == menu_name:
                window_tools.click_at(obj.mean_pos_x, obj.mean_pos_y)
                obj_found = True
                return
        window_tools.move_left(self.window, strength = 2)
        
        if obj_found == False:
            print(f'Menu {menu_name} was not found')
            
    def test_logic(self):
        run = True
        self._start_error_checker()
        # Track first-time entry + timers
        start_time_classic_arena = None
        start_time_tagteam_arena = None
        start_time_live_arena = None
    
        REFRESH_INTERVAL = 15.1 * 60  # 15.1 minutes in seconds
    
        while run:

            # =========================
            # 1. CLASSIC ARENA
            # =========================
            if self.params['run']['classic_arena']:
                self.go_to_menu(self.main_menu_names['Arena'])
                
                window_tools.click_center(self.window, self.search_areas['classic_arena'])
        
                if start_time_classic_arena is None:
                    start_time_classic_arena = time.time()
                    # If handler was initialized earlier, refresh once
                    if start_time_classic_arena - self.handler_init_time > REFRESH_INTERVAL:
                        self.classic_arena_bot.refresh()
        
                elif time.time() - start_time_classic_arena > REFRESH_INTERVAL:
                    start_time_classic_arena = time.time()
                    self.classic_arena_bot.refresh()
        
                self.classic_arena_bot.run_classic_arena_once()
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
            
            
            # =========================
            # 2. TAG TEAM ARENA
            # =========================
            if self.params['run']['tagteam_arena']:
                self.go_to_menu(self.main_menu_names['Arena'])
                window_tools.click_center(self.window, self.search_areas['tagteam_arena'])
        
                if start_time_tagteam_arena is None:
                    start_time_tagteam_arena = time.time()
                    if start_time_tagteam_arena - self.handler_init_time > REFRESH_INTERVAL:
                        self.tagteam_arena_bot.refresh()

        
                elif time.time() - start_time_tagteam_arena > REFRESH_INTERVAL:
                    start_time_tagteam_arena = time.time()
                    self.tagteam_arena_bot.refresh()

        
                self.tagteam_arena_bot.run_tagteam_arena_once()
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
            
    
            # =========================
            # 3. LIVE ARENA
            # =========================
            if self.params['run']['live_arena']:
                self.go_to_menu(self.main_menu_names['Arena'])
                window_tools.click_center(self.window, self.search_areas['live_arena'])
        
                if start_time_live_arena is None:
                    start_time_live_arena = time.time()

                self.live_arena_bot.run_live_arena()
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
            
            
            # =========================
            # 4. Dungeons
            # =========================   
            if self.params['run']['dungeons'] and not self.params['run']['effective_unit_leveling']:  
                self.go_to_menu(self.main_menu_names['Dungeons'])     
                self.dungeon_bot.run_dungeons()
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
                
            # =========================
            # 5. Faction Wars
            # =========================
            if self.params['run']['factionwars']:
                self.go_to_menu(self.main_menu_names['FactionWars'])     
                self.factionwars_bot.run_factionwars()
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
        
            
            # =========================
            # 6. DemonLord Clanboss
            # =========================
            if self.params['run']['demonlord']:
                self.go_to_menu(self.main_menu_names['ClanBoss1'])    
                window_tools.click_center(self.window, self.search_areas["clanboss_DemonLord"]) 
                self.demonlord_bot.run_demonlord()
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
            # Check demonlord_keys
            #     if not enough demonlord_keys -> skip
            # self.go_to_menu(self.main_menu_names['clanboss'])
            # self.demonlord_bot.run_demonlord()
            # window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
            

            # run_classic_arena = True
            # run_tagteam_arena = True
            # run_live_arena = True
            # run_dungeons = True
            # run_factionwars = True
            # run_demonlord = True
            # run_doomtower = True
            # run_cursedcity = True
            # run_grimforest = True
            # run_effective_unit_leveling = False


            # =========================
            # 99. QuestRewards
            # =========================
            self.check_quest_rewards()






# Run test if script is executed directly
if __name__ == "__main__":
    print('ALWAYS RUN THE PROGRAM IN 1280 x 1024')
    bot = RSL_Bot_Mainframe()
    bot.demonlord_bot.test()
    gui_tools.BotGUI(bot).run()

    #bot.test_logic()

    #bot.live_arena_bot.check_arena_coins()
    # bot.tagteam_arena_bot.run_tagteam_arena_once()

    
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