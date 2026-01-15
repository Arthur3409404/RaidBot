# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:45:00 2025

@author: Arthur
"""


import pyautogui
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time
import re
from datetime import  timedelta
import os
import data.lib.utils.image_tools as image_tools
import data.lib.utils.window_tools as window_tools
from data.lib.handlers.ai_networks_handler import EnemyDataset, TagTeamEvaluationNetworkCNN, EvaluationNetworkCNN_ImageOnly
import difflib



# =============================================================================
#   CLASSIC ARENA BOT     
# =============================================================================

class RSL_Bot_ClassicArena:
    
    def __init__(
        self,
        title_substring="Raid: Shadow Legends",
        reader=None,
        window=None,
        verbose=True,
        num_multi_refresh=0,
        multi_refresh=False,
        power_threshold=70000,
        use_gems=True,
        enemies_lost=[0]
    ):
        """
        Initialize the Classic Arena bot.
        """
        if reader is None:
            print("Error When Loading Reader")
        self.reader = reader  

        self.running = True
        self.dataset = EnemyDataset("data/database_champions/datasets/enemy_dataset_classic_arena.npz")
        self.battles_done = 0
        self.classic_arena_multi_refresh = multi_refresh
        self.classic_arena_num_multi_refresh = num_multi_refresh
        self.verbose = verbose
        self.classic_arena_enemies_lost = enemies_lost
        self.classic_arena_use_gems = use_gems
        self.offset_wins = len(self.classic_arena_enemies_lost)
        self.window = window
        self.init_time = time.time()
        self.classic_arena_power_threshold = power_threshold
        self.refresh_minutes = 15.2
        self.max_battle_time = 90
        self.no_coin_status = False

        if self.window:
            self.coords = (self.window.left, self.window.top, self.window.width, self.window.height)
            print(f"Window Coordinates: {self.coords}")
        else:
            self.coords = None

        # Search Areas
        self.search_areas = {
            "bronce_medals": [0.235, 0.038, 0.066, 0.03],
            "silver_medals": [0.342, 0.038, 0.068, 0.028],
            "gold_medals": [0.449, 0.04, 0.067, 0.026],
            "refresh_timer": [0.789, 0.15, 0.177, 0.059],
            "arena_coins": [0.722, 0.039, 0.042, 0.026],
            "add_arena_coins": [0.701, 0.039, 0.024, 0.028],
            "confirm_add_arena_coins": [0.394, 0.615, 0.208, 0.083],
            "confirm_use_gems": [0.394, 0.615, 0.208, 0.083],
            "gem_amount": [0.497, 0.629, 0.031, 0.03],
            "list_enemies": [0.665, 0.225, 0.314, 0.761],
            "start_battle": [0.762, 0.876, 0.213, 0.104],
            "battle_finished": [0.362, 0.897, 0.269, 0.081],
            "battle_result": [0.389, 0.148, 0.204, 0.071],
            "test": [0.05, 0.30, 0.15, 0.08],
        }

        # Enemy positions
        self.corresponding_enemy_positions = {
            "Pos1": [[0.583, 0.31, 0.173, 0.023], [0.787, 0.237, 0.181, 0.082]],
            "Pos2": [[0.586, 0.43, 0.16, 0.019], [0.789, 0.357, 0.181, 0.078]],
            "Pos3": [[0.586, 0.548, 0.164, 0.019], [0.787, 0.473, 0.184, 0.084]],
            "Pos4": [[0.587, 0.664, 0.162, 0.021], [0.787, 0.592, 0.183, 0.08]],
            "Pos5": [[0.582, 0.782, 0.166, 0.022], [0.787, 0.709, 0.182, 0.082]],
            "Pos6": [[0.586, 0.901, 0.16, 0.018], [0.787, 0.827, 0.183, 0.082]],
            "Pos7": [[0.586, 0.592, 0.16, 0.02], [0.788, 0.519, 0.181, 0.079]],
            "Pos8": [[0.586, 0.709, 0.163, 0.021], [0.787, 0.636, 0.181, 0.08]],
            "Pos9": [[0.586, 0.829, 0.164, 0.019], [0.787, 0.754, 0.183, 0.081]],
            "Pos10": [[0.583, 0.946, 0.166, 0.02], [0.787, 0.874, 0.181, 0.078]],
        }

    def update_battle_outcome(self, power_level):
        """
        Determine outcome of a battle and update enemy memory if lost.
        """
        battle_result = image_tools.get_text_in_relative_area(
            self.reader, self.window, search_area=self.search_areas["battle_result"]
        )[0]

        if self.resembles(battle_result.text, "VICTORIA"):
            print("Victory")
            self.recent_battle_outcome = 1
        else:
            self.classic_arena_enemies_lost.append(power_level)
            self.persist_enemy_avoid_list()
            print("Updated Enemy Avoid List")
            self.recent_battle_outcome = 0

    def persist_enemy_avoid_list(self):
        """
        Persist lost enemies to params_mainframe.txt
        """
        param_file = os.path.join("data", "params_mainframe.txt")
        updated_line = f"classic_arena_enemies_lost = {self.classic_arena_enemies_lost}\n"

        with open(param_file, "r") as f:
            lines = f.readlines()
        with open(param_file, "w") as f:
            for line in lines:
                if line.strip().startswith("classic_arena_enemies_lost ="):
                    f.write(updated_line)
                else:
                    f.write(line)

    def execute_arena_battle(self, obj, next_obj, power_level):
        """
        Engage an enemy and handle the battle loop.
        """
        battle_running = True
        window_tools.click_at(obj.mean_pos_x, obj.mean_pos_y)
        time.sleep(3)

        start_button = image_tools.get_text_in_relative_area(
            self.reader, self.window, self.search_areas["start_battle"]
        )[0]
        window_tools.click_at(start_button.mean_pos_x, start_button.mean_pos_y)

        while self.main_loop_running and (battle_running):
            try:
                battle_finished = image_tools.get_text_in_relative_area(
                    self.reader, self.window, search_area=self.search_areas["battle_finished"]
                )[0]
                if self.resembles(battle_finished.text ,"PULSA PARA CONTINUAR"):
                    battle_running = False
                    time.sleep(3)
                    self.update_battle_outcome(power_level)
                    time.sleep(3)
                    # Multiple clicks to ensure continuation
                    for _ in range(4):
                        window_tools.click_at(battle_finished.mean_pos_x, battle_finished.mean_pos_y)
                        time.sleep(0.2)
                    
            except:
                pass
            time.sleep(3)

    def evaluate_arena_enemies(self):
        """
        Scan and evaluate all enemies in the arena.
        """
        text_objects = image_tools.get_text_from_cluster_area(
            self.reader, self.window, search_areas=self.corresponding_enemy_positions, power_detection=False
        )
        filtered_objects = image_tools.filter_text_objects(text_objects)

        idx = 0
        while self.main_loop_running and (idx < len(filtered_objects)):
            obj = filtered_objects[idx]
            if obj.text.strip() == "Luchar":
                if idx - 1 < len(filtered_objects):
                    next_obj = filtered_objects[idx - 1]
                    raw_text = next_obj.text.strip()
                    try:
                        matches = re.findall(r"(\d[\d.,]*)([a-zA-Z]*)", raw_text)
                        if not matches:
                            raise ValueError("No numeric value found in text")
                        number_part, suffix = matches[-1]
                        number_part = number_part.replace('.', '').replace(',', '.').replace(' ', '')
                        num = float(str(number_part))
                        suffix = suffix.lower()
                        if suffix.startswith("k"):
                            power_val = num * 1000
                        elif suffix.startswith("m"):
                            power_val = num * 1_000_000
                        else:
                            power_val = num

                        if power_val < self.classic_arena_power_threshold and power_val >= 500 and power_val not in self.classic_arena_enemies_lost:
                            screenshot_enemy = pyautogui.screenshot(
                                region=(int(obj.mean_pos_x - 500), int(obj.mean_pos_y - 65), 440, 130)
                            )
                            image_np_enemy = np.array(screenshot_enemy)
                            print(f"Power {power_val} < threshold {self.classic_arena_power_threshold}")
                            self.execute_arena_battle(obj, next_obj, power_val)
                            time.sleep(3)
                            self.battle_occured = True
                            self.battles_done += 1
                            self.dataset.append_entry(image_np_enemy, power_val, self.recent_battle_outcome)
                            return self.battle_occured
                    except Exception as e:
                        print(f"[!] Error parsing '{next_obj.text}': {e}")
            idx += 1
        return getattr(self, "battle_occured", False)

    def exit_battle_screen(self):
        window_tools.click_center(self.window, self.search_areas["battle_finished"])

    def refresh_enemy_list(self):
        if not self.coords or "refresh_timer" not in self.search_areas:
            return
        window_tools.click_center(self.window, self.search_areas["refresh_timer"])

    def ensure_arena_coins(self):
        """
        Checks if arena coins are available; if not, attempt to use gems if allowed.
        """
        time.sleep(1)
        self.no_coin_status = False
        coins_text = image_tools.get_text_in_relative_area(
            self.reader, self.window, self.search_areas["arena_coins"]
        )[0]

        if "0/" in coins_text.text and coins_text.text != "10/10":
            rel_left, rel_top, rel_width, rel_height = self.search_areas["add_arena_coins"]
            abs_left = self.window.left + int(rel_left * self.window.width)
            abs_top = self.window.top + int(rel_top * self.window.height)
            abs_width = int(rel_width * self.window.width)
            abs_height = int(rel_height * self.window.height)
            center_x = abs_left + abs_width // 2
            center_y = abs_top + abs_height // 2

            pyautogui.click(center_x, center_y)
            time.sleep(3)
            confirm_text = image_tools.get_text_in_relative_area(
                self.reader, self.window, self.search_areas["confirm_add_arena_coins"]
            )[0]
            confirm_gems_text = image_tools.get_text_in_relative_area(
                self.reader, self.window, self.search_areas["confirm_use_gems"]
            )[0]
            try:
                gem_amount_text = image_tools.get_text_in_relative_area(
                    self.reader, self.window, self.search_areas["gem_amount"]
                )[0].text
                numbers = re.findall(r"\d+", gem_amount_text)
                gem_amount = int("".join(numbers)) if numbers else 0
            except:
                gem_amount = 0

            if not self.classic_arena_use_gems and gem_amount > 0:
                pyautogui.click(center_x, center_y)
                time.sleep(3)
                self.no_coin_status = True
                return

            window_tools.click_at(confirm_text.mean_pos_x, confirm_text.mean_pos_y)
            time.sleep(3)

    def report_run_status(self):
        elapsed = time.time() - self.init_time
        formatted_elapsed = str(timedelta(seconds=int(elapsed)))
        medals = (self.battles_done - len(self.classic_arena_enemies_lost) + self.offset_wins) * 4

        print("\n" + "=" * 40)
        print("🛡️  RAID Classic Arena Bot Status")
        print("-" * 40)
        print(f"🔁 Mode: Multi Refresh ({self.classic_arena_num_multi_refresh})")
        print(f"⏱️  Time Since Start: {formatted_elapsed}")
        print(f"⚔️  Battles Won: {self.battles_done - len(self.classic_arena_enemies_lost) + self.offset_wins}")
        print(f"⚔️  Battles Lost: {len(self.classic_arena_enemies_lost) - + self.offset_wins}")
        print(f"🎖️  Estimated Medals: {medals}")
        print("-" * 40)
        print("🛑 To stop the bot, press 'v'")
        print("=" * 40 + "\n")
        
    def resembles(self, text, target, threshold=0.8):
        ratio = difflib.SequenceMatcher(None, text.lower(), target.lower()).ratio()
        return ratio >= threshold


    def run_classic_arena_continuous(self):
        """
        Run the Classic Arena bot indefinitely without a time limit.
        """
        time.sleep(5)

        time_start = time.time()
        last_refresh_time = time_start
        self.start_time = time_start
        counter_multi_refresh = 0

        while self.main_loop_running and (self.running):
            self.report_run_status()
            self.battle_occured = False

            self.ensure_arena_coins()

            self.battle_occured = self.evaluate_arena_enemies()
            if self.battle_occured:
                continue

            window_tools.move_down(self.window)

            self.battle_occured = self.evaluate_arena_enemies()
            if self.battle_occured:
                continue

            window_tools.move_up(self.window)

            if self.classic_arena_multi_refresh:
                if counter_multi_refresh < self.classic_arena_num_multi_refresh:
                    self.refresh_enemy_list()
                    counter_multi_refresh += 1
                    continue
                else:
                    counter_multi_refresh = 0

            print("Waiting for free Refresh")
            time_start_loop = time.time()
            while self.main_loop_running and ((time.time() - time_start_loop) < 62):
                time.sleep(1)

            elapsed = time.time() - last_refresh_time
            if elapsed >= self.refresh_minutes * 60:
                if self.running:
                    self.refresh_enemy_list()
                    last_refresh_time = time.time()
                        
                        
    def run_classic_arena_until_empty(self, main_loop_running = True):
        """
        Run the Classic Arena bot once until no arena coins remain.
        """
        time.sleep(5)
        self.main_loop_running = main_loop_running

        time_start = time.time()
        last_refresh_time = time_start
        self.start_time = time_start
        counter_multi_refresh = 0
        self.running = True

        time.sleep(5)

        while self.main_loop_running and (self.running):
            self.report_run_status()
            self.battle_occured = False

            self.ensure_arena_coins()
            if self.no_coin_status:
                self.running = False
                print("Waiting for coins")
                continue

            self.battle_occured = self.evaluate_arena_enemies()
            if self.battle_occured:
                continue

            window_tools.move_down(self.window)

            self.battle_occured = self.evaluate_arena_enemies()
            if self.battle_occured:
                continue

            window_tools.move_up(self.window)

            if self.classic_arena_multi_refresh:
                if counter_multi_refresh < self.classic_arena_num_multi_refresh:
                    self.refresh_enemy_list()
                    counter_multi_refresh += 1
                    continue
                else:
                    counter_multi_refresh = 0

            print("Waiting for free Refresh")
            self.running = False

        return
    
    
# =============================================================================
#   TAG TEAM BOT     
# =============================================================================

class RSL_Bot_TagTeamArena:
    """
    Automates Tag Team Arena battles in Raid: Shadow Legends.
    """

    def __init__(
        self,
        title_substring="Raid: Shadow Legends",
        reader=None,
        window=None,
        verbose=True,
        num_multi_refresh=0,
        multi_refresh=False,
        power_threshold=70000,
        use_gems=True,
        enemies_lost=[0],
    ):
        if reader is None:
            print("Error When Loading Reader")

        # Core state
        self.reader = reader
        self.window = window
        self.running = True
        self.verbose = verbose

        # Battle tracking
        self.battles_done = 0
        self.recent_battle_outcome = 0
        self.battle_occured = False

        # Enemy memory
        self.tagteam_arena_enemies_lost = enemies_lost
        self.offset_wins = len(self.tagteam_arena_enemies_lost)

        # Arena configuration
        self.tagteam_arena_power_threshold = power_threshold
        self.tagteam_arena_use_gems = use_gems
        self.tagteam_arena_multi_refresh = multi_refresh
        self.tagteam_arena_num_multi_refresh = num_multi_refresh

        # Timing
        self.init_time = time.time()
        self.refresh_minutes = 15.2
        self.max_battle_time = 200

        # Dataset
        self.dataset = EnemyDataset(
            "data/database_champions/datasets/enemy_dataset_tagteam_arena.npz"
        )

        # AI evaluation network
        weights_path = r"neural_networks\enemy_eval_tagteam_arena\_epoch350.pt"
        self.evaluation_ai = TagTeamEvaluationNetworkCNN(weights_path=weights_path)
        self.evaluation_ai.eval()

        # Window coordinates
        if self.window:
            self.coords = (
                self.window.left,
                self.window.top,
                self.window.width,
                self.window.height,
            )
            print(f"Window Coordinates: {self.coords}")
        else:
            self.coords = None

        # UI search areas (relative)
        self.search_areas = {
            "go_to_higher_menu":   [0.928, 0.031, 0.046, 0.039],
            'pov' : [0, 0, 1, 1],
            "main_menu_labels":      [0.007, 0.27, 0.984, 0.044],

            "bronce_medals": [0.235, 0.038, 0.066, 0.03],
            "silver_medals": [0.342, 0.038, 0.068, 0.028],
            "gold_medals": [0.449, 0.04, 0.067, 0.026],
            "refresh_timer": [0.789, 0.15, 0.177, 0.059],
            "arena_coins": [0.722, 0.039, 0.042, 0.026],
            "add_arena_coins": [0.701, 0.039, 0.024, 0.028],
            "confirm_add_arena_coins": [0.394, 0.615, 0.208, 0.083],
            "confirm_use_gems": [0.394, 0.615, 0.208, 0.083],
            "gem_amount": [0.498, 0.66, 0.03, 0.023],
            "list_enemies": [0.665, 0.225, 0.314, 0.761],
            "start_battle": [0.762, 0.876, 0.213, 0.104],
            "battle_finished": [0.362, 0.897, 0.269, 0.081],
            "battle_result": [0.389, 0.148, 0.204, 0.071],
            "close_encounter": [0.376, 0.639, 0.231, 0.071],
            'luchar_area' : [0.5, 0, 0.5, 1],

            "enemy_total_power_value": [0.707, 0.566, 0.17, 0.034],
            "enemy_team1_power_value": [0.763, 0.24, 0.155, 0.025],
            "enemy_team2_power_value": [0.758, 0.351, 0.156, 0.023],
            "enemy_team3_power_value": [0.763, 0.459, 0.154, 0.023],
        }

        # Enemy slot → power + button areas
        self.corresponding_enemy_positions = {
            "Pos1": [[0.535, 0.324, 0.136, 0.017], [0.786, 0.236, 0.182, 0.081]],
            "Pos2": [[0.538, 0.453, 0.137, 0.018], [0.786, 0.362, 0.183, 0.086]],
            "Pos3": [[0.535, 0.582, 0.137, 0.019], [0.786, 0.496, 0.183, 0.082]],
            "Pos4": [[0.534, 0.709, 0.137, 0.019], [0.785, 0.623, 0.182, 0.082]],
            "Pos5": [[0.537, 0.84, 0.136, 0.021], [0.787, 0.754, 0.182, 0.08]],
            "Pos6": [[0.532, 0.416, 0.148, 0.021], [0.787, 0.33, 0.184, 0.08]],
            "Pos7": [[0.536, 0.544, 0.137, 0.019], [0.787, 0.457, 0.183, 0.083]],
            "Pos8": [[0.535, 0.674, 0.14, 0.02], [0.788, 0.59, 0.18, 0.077]],
            "Pos9": [[0.535, 0.803, 0.14, 0.022], [0.788, 0.72, 0.181, 0.076]],
            "Pos10": [[0.533, 0.93, 0.142, 0.026], [0.788, 0.848, 0.181, 0.079]],
        }

    # ------------------------------------------------------------------
    # Battle outcome & memory
    # ------------------------------------------------------------------

    def update_battle_outcome(self, enemy_power):
        result = image_tools.get_text_in_relative_area(
            self.reader, self.window, self.search_areas["battle_result"]
        )[0]

        if self.resembles(result.text , "VICTORIA"):
            print("Victory")
            self.recent_battle_outcome = 1
        else:
            print("Defeat – updating enemy avoid list")
            self.recent_battle_outcome = 0
            self.tagteam_arena_enemies_lost.append(enemy_power)
            self.persist_enemy_avoid_list()

    def resembles(self, text, target, threshold=0.8):
        ratio = difflib.SequenceMatcher(None, text.lower(), target.lower()).ratio()
        return ratio >= threshold
    

    def persist_enemy_avoid_list(self):
        param_file = os.path.join("data", "params_mainframe.txt")
        updated_line = (
            f"tagteam_arena_enemies_lost = {self.tagteam_arena_enemies_lost}\n"
        )

        with open(param_file, "r") as f:
            lines = f.readlines()

        with open(param_file, "w") as f:
            for line in lines:
                if line.strip().startswith("tagteam_arena_enemies_lost ="):
                    f.write(updated_line)
                else:
                    f.write(line)

    # ------------------------------------------------------------------
    # Battle execution
    # ------------------------------------------------------------------

    def execute_tagteam_battle(self, fight_button, power_text_obj, enemy_power):
        window_tools.click_at(fight_button.mean_pos_x, fight_button.mean_pos_y)
        time.sleep(3)

        start_btn = image_tools.get_text_in_relative_area(
            self.reader, self.window, self.search_areas["start_battle"]
        )[0]
        window_tools.click_at(start_btn.mean_pos_x, start_btn.mean_pos_y)

        while self.main_loop_running and (True):
            try:
                finished = image_tools.get_text_in_relative_area(
                    self.reader, self.window, self.search_areas["battle_finished"]
                )[0]

                if self.resembles(finished.text, "PULSA PARA CONTINUAR"):
                    time.sleep(3)
                    self.update_battle_outcome(enemy_power)

                    for _ in range(2):
                        window_tools.click_at(
                            finished.mean_pos_x, finished.mean_pos_y
                        )
                        time.sleep(1)

                    window_tools.click_center(
                        self.window, self.search_areas["close_encounter"]
                    )
                    return
            except Exception:
                pass

            time.sleep(3)

    # ------------------------------------------------------------------
    # Enemy evaluation
    # ------------------------------------------------------------------

    def _parse_enemy_power_value(self, text):
        text = text.replace(".", "").replace(",", ".").replace(" ", "")
        matches = re.findall(r"(\d[\d.,]*)([a-zA-Z]*)", text)
        if not matches:
            raise ValueError("No numeric value found")

        number, suffix = matches[-1]
        value = float(number)

        suffix = suffix.lower()
        if suffix.startswith("k"):
            value *= 1_000
        elif suffix.startswith("m"):
            value *= 1_000_000

        return value

    def evaluate_tagteam_enemies(self):
        text_objects = image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            search_area=self.search_areas["luchar_area"],
            power_detection=False,
        )

        filtered = image_tools.filter_text_objects(text_objects)

        for idx, obj in enumerate(filtered):

            self.ensure_arena_coins()
            if self.no_coin_status:
                break

            if obj.text.strip() != "Luchar":
                continue
            window_tools.click_at(obj.mean_pos_x, obj.mean_pos_y)
            window_tools.click_center(self.window, self.search_areas["pov"])

            power_obj = image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            search_area=self.search_areas["enemy_total_power_value"],
            power_detection=False,
        )
            power_team1 = image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            search_area=self.search_areas["enemy_team1_power_value"],
            power_detection=False,
        )
            power_team2 = image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            search_area=self.search_areas["enemy_team2_power_value"],
            power_detection=False,
        )
            power_team3 = image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            search_area=self.search_areas["enemy_team3_power_value"],
            power_detection=False,
        )

            window_tools.sendkey("esc", delay= 5)
            try:
                try:
                    enemy_power = self._parse_enemy_power_value(power_obj[0].text)
                    enemy_power_team1 = self._parse_enemy_power_value(power_team1[0].text)
                    enemy_power_team2 = self._parse_enemy_power_value(power_team2[0].text)
                    enemy_power_team3 = self._parse_enemy_power_value(power_team3[0].text)
                except:
                    enemy_power = 10

                screenshot = pyautogui.screenshot(
                    region=(
                        int(obj.mean_pos_x - 540),
                        int(obj.mean_pos_y - 65),
                        440,
                        130,
                    )
                )

                image_np = np.array(screenshot).astype(np.float32)
                powers = np.array([enemy_power, enemy_power_team1, enemy_power_team2, enemy_power_team3])/350000
                prob, label = self.evaluation_ai.predict(image_np, powers)
                print(prob)
                #print(f"Team1: {enemy_power_team1} Team2:{enemy_power_team2} Team3: {enemy_power_team3}  Total:{enemy_power}")
                if label == 1 and enemy_power not in self.tagteam_arena_enemies_lost and enemy_power>500:
                #if enemy_power not in self.tagteam_arena_enemies_lost and enemy_power<1300000:
                    self.execute_tagteam_battle(obj, power_obj, enemy_power)
                    self.battles_done += 1
                    self.battle_occured = True
                    enemy_power_collection = np.array([enemy_power,enemy_power_team1,enemy_power_team2,enemy_power_team3])
                    self.dataset.append_entry(
                        image_np, enemy_power_collection, self.recent_battle_outcome
                    )

                    outcome = "Win" if self.recent_battle_outcome else "Loss"
                    print(f"Battle outcome: {outcome} (Win prob: {prob:.2f})")

                    return True
                else:
                    pass

            except Exception as e:
                print(f"[!] Error parsing Power Object")

        return False

    # ------------------------------------------------------------------
    # Arena utility
    # ------------------------------------------------------------------

    def refresh_enemy_list(self):
        if self.coords:
            window_tools.click_center(self.window, self.search_areas["refresh_timer"])

    def ensure_arena_coins(self):
        self.no_coin_status = False
        time.sleep(1)

        coins = image_tools.get_text_in_relative_area(
            self.reader, self.window, self.search_areas["arena_coins"]
        )[0]

        if "0/" not in coins.text or coins.text == "10/10":
            return

        window_tools.click_center(self.window, self.search_areas["add_arena_coins"])
        time.sleep(3)

        confirm = image_tools.get_text_in_relative_area(
            self.reader, self.window, self.search_areas["confirm_add_arena_coins"]
        )[0]

        gems = image_tools.get_text_in_relative_area(
            self.reader, self.window, self.search_areas["confirm_use_gems"]
        )[0]

        numbers = re.findall(r"\d+", gems.text)
        gem_cost = int("".join(numbers)) if numbers else 0

        if not self.tagteam_arena_use_gems and gem_cost > 0:
            self.no_coin_status = True
            return

        window_tools.click_at(confirm.mean_pos_x, confirm.mean_pos_y)

    # ------------------------------------------------------------------
    # Status & main loops
    # ------------------------------------------------------------------

    def report_run_status(self):
        elapsed = str(timedelta(seconds=int(time.time() - self.init_time)))
        wins = self.battles_done - len(self.tagteam_arena_enemies_lost) + self.offset_wins
        medals = wins * 4

        print("\n" + "=" * 40)
        print("🛡️ RAID TagTeam Arena Bot Status")
        print("-" * 40)
        print(f"⏱️ Runtime: {elapsed}")
        print(f"⚔️ Wins: {wins}")
        print(f"❌ Losses: {len(self.tagteam_arena_enemies_lost) - self.offset_wins}")
        print(f"🎖️ Estimated Medals: {medals}")
        print("=" * 40)

    def run_tagteam_arena_continuous(self):
        time.sleep(5)
        last_refresh = time.time()
        refresh_count = 0

        while self.main_loop_running and (self.running):
            self.report_run_status()
            self.ensure_arena_coins()

            if self.no_coin_status:
                print("Waiting for coins")
                break

            if self.evaluate_tagteam_enemies():
                continue

            window_tools.move_down(self.window)
            if self.evaluate_tagteam_enemies():
                continue
            window_tools.move_up(self.window)

            if self.tagteam_arena_multi_refresh:
                if refresh_count < self.tagteam_arena_num_multi_refresh:
                    self.refresh_enemy_list()
                    refresh_count += 1
                    continue
                refresh_count = 0

            print("Waiting for free refresh")
            time.sleep(62)

            if time.time() - last_refresh >= self.refresh_minutes * 60:
                self.refresh_enemy_list()
                last_refresh = time.time()

    def run_tagteam_arena_single_cycle(self, main_loop_running = True):
        self.main_loop_running = main_loop_running
        time.sleep(5)
        self.running = True

        while self.main_loop_running and (self.running):
            self.report_run_status()
            self.ensure_arena_coins()

            if self.no_coin_status:
                print("Waiting for coins")
                break

            if self.evaluate_tagteam_enemies():
                continue

            window_tools.move_down(self.window)
            if self.evaluate_tagteam_enemies():
                continue
            window_tools.move_up(self.window)

            print("Finished one cycle")
            break
    
    
    
# =============================================================================
#   LIVE ARENA BOT 
# =============================================================================

class RSL_Bot_LiveArena:
    
    def __init__(self, title_substring="Raid: Shadow Legends", reader = None, window =None, verbose = True, use_gems = True, use_gems_max_amount = 0, memory = dict()):

        if reader is None:
            print('Error When Loading Reader')
            
        self.reader = reader
        
        self.running = True
        
        self.battles_done = 0
        self.battles_won = 0
        self.no_coin_status = False
        
        self.verbose = verbose
        self.live_arena_memory = memory
        self.live_arena_use_gems = use_gems
        self.live_arena_use_gems_max_amount = use_gems_max_amount
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
            
            "live_arena_status":   [0.203, 0.834, 0.536, 0.04], #many red or green points in one spot
            
            "live_arena_coins":   [0.725, 0.036, 0.039, 0.03],
            "live_add_arena_coins":   [0.702, 0.039, 0.026, 0.029],
            "live_confirm_add_arena_coins":   [0.394, 0.615, 0.208, 0.083],
            "live_amount_gems":   [0.498, 0.658, 0.032, 0.026],
            "live_confirm_use_gems":   [0.395, 0.642, 0.209, 0.089],
            "gem_amount":   [0.494, 0.659, 0.039, 0.024],
            
            "live_arena_reward_1":   [0.924, 0.137, 0.042, 0.056],
            "live_arena_reward_2":   [0.924, 0.201, 0.041, 0.054],
            "live_arena_reward_3":   [0.792, 0.293, 0.044, 0.056],
            "live_arena_reward_4":   [0.858, 0.293, 0.042, 0.053],
            "live_arena_reward_5":   [0.922, 0.292, 0.044, 0.053],
    
            "start_encounter":     [0.34, 0.884, 0.258, 0.086],
            "encounter_status":  [0.363, 0.076, 0.268, 0.038],
            "pick_status":    [0.412, 0.476, 0.173, 0.084],
            "champion_roster_complete": [0.104, 0.758, 0.611, 0.231],
            "turn_counter_roster": [0.423, 0.076, 0.084, 0.03],
            
            "team_roster_1": [0.242, 0.333, 0.076, 0.122],
            "team_roster_2": [0.195, 0.47, 0.079, 0.123],
            "team_roster_3": [0.152, 0.332, 0.076, 0.125],
            "team_roster_4": [0.106, 0.469, 0.078, 0.124],
            "team_roster_5": [0.062, 0.332, 0.079, 0.124],
            
            "enemy_roster_1": [0.674, 0.33, 0.086, 0.133],
            "enemy_roster_2": [0.718, 0.464, 0.086, 0.136],
            "enemy_roster_3": [0.765, 0.328, 0.084, 0.135],
            "enemy_roster_4": [0.815, 0.471, 0.079, 0.124],
            "enemy_roster_5": [0.86, 0.333, 0.079, 0.124],
            
            "preset_champion_1": [0.107, 0.762, 0.046, 0.074],
            "preset_champion_2": [0.106, 0.839, 0.049, 0.074],
            "preset_champion_3": [0.106, 0.914, 0.047, 0.075],
            "preset_champion_4": [0.157, 0.763, 0.046, 0.071],
            "preset_champion_5": [0.158, 0.84, 0.046, 0.071],
            "preset_champion_6": [0.159, 0.918, 0.046, 0.071],
            "preset_champion_7": [0.21, 0.764, 0.044, 0.071],
            "preset_champion_8": [0.21, 0.841, 0.046, 0.071],
            
            "confirm_button_champion_selection": [0.762, 0.876, 0.213, 0.104],
            
            "auto_battle_button": [0.026, 0.899, 0.058, 0.07], # its existance means battle started
            "battle_status_finished":  [0.362, 0.897, 0.269, 0.081], # check regularly for if enemy quits
            "battle_result":    [0.389, 0.148, 0.204, 0.071],
            "battle_result_2":    [0.38, 0.085, 0.224, 0.059],
    
            
            "test":   [0.05, 0.30, 0.15, 0.08],
            
        }
        
        
    # ------------------------- Reset Methods -------------------------
    def reset_battle_state(self):
        self.battle_status = 'menu'
        self.auto_button_clicked = False
        self.no_coin_status = False

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
                    if self.resembles(battle_result.text ,"VICTORIA"):
                        self.battles_won += 1
                    return
            except:
                continue

    # ------------------------- Enemy Memory -------------------------
    def persist_enemy_avoid_list(self):
        # Not used currently
        pass

    def resembles(self, text, target, threshold=0.8):
        ratio = difflib.SequenceMatcher(None, text.lower(), target.lower()).ratio()
        return ratio >= threshold

    # ------------------------- Battle Status -------------------------
    def update_battle_activity_status(self):
        try:
            auto_button = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas["auto_battle_button"]
            )[0]
            if self.resembles(auto_button.text, 'Auto'):
                self.battle_status = 'Battle active'
                time.sleep(3)
                if not self.auto_button_clicked:
                    window_tools.click_center(self.window, self.search_areas["auto_battle_button"])
                    self.auto_button_clicked = True
                    window_tools.click_center(self.window, self.search_areas["live_arena_status"])
            else:
                self.battle_status = 'Battle inactive'
        except:
            pass

    # ------------------------- Arena Coins -------------------------
    def ensure_arena_coins(self):
        time.sleep(1)
        self.no_coin_status = False

        coins_text = image_tools.get_text_in_relative_area(
            self.reader, self.window, self.search_areas["live_arena_coins"]
        )[0]

        if "0/" in coins_text.text and coins_text.text != '5/5':
            # Calculate absolute center coordinates
            rel_left, rel_top, rel_width, rel_height = self.search_areas["live_add_arena_coins"]
            abs_left = self.window.left + int(rel_left * self.window.width)
            abs_top = self.window.top + int(rel_top * self.window.height)
            abs_width = int(rel_width * self.window.width)
            abs_height = int(rel_height * self.window.height)
            center_x = abs_left + abs_width // 2
            center_y = abs_top + abs_height // 2

            pyautogui.click(center_x, center_y)
            time.sleep(3)

            confirm_text = image_tools.get_text_in_relative_area(
                self.reader, self.window, self.search_areas["live_confirm_add_arena_coins"]
            )[0]

            # Check gem usage
            try:
                gem_amount = image_tools.get_text_in_relative_area(
                    self.reader, self.window, self.search_areas["gem_amount"]
                )[0].text
                gem_amount = int("".join(re.findall(r"\d+", gem_amount))) if gem_amount else 0
            except:
                gem_amount = 0

            if (self.live_arena_use_gems and gem_amount > self.live_arena_use_gems_max_amount) or (not self.live_arena_use_gems and gem_amount > 0):
                pyautogui.click(center_x, center_y)
                time.sleep(3)
                self.no_coin_status = True
                return

            time.sleep(3)
            window_tools.click_at(confirm_text.mean_pos_x, confirm_text.mean_pos_y)

    # ------------------------- Status Print -------------------------
    def report_run_status(self):
        elapsed = int(time.time() - self.init_time)
        medals = self.battles_won * 70
        formatted_elapsed = str(timedelta(seconds=elapsed))

        print("\n" + "=" * 40)
        print("🛡️  RAID live Arena Bot Status")
        print("-" * 40)
        print(f"🔁 Mode: Simple Pick")
        print(f"⏱️  Time Since Start: {formatted_elapsed}")
        print(f"⚔️  Battles Won: {self.battles_won}")
        print(f"⚔️  Battles Lost: {self.battles_done - self.battles_won}")
        print(f"🎖️  Estimated Medals: {medals}")
        print("-" * 40)
        print("🛑 To stop the bot, press 'v'")
        print("=" * 40 + "\n")

    # ------------------------- Live Arena -------------------------
    def is_live_arena_active(self):
        rel_left, rel_top, rel_width, rel_height = self.search_areas["live_arena_status"]
        x = int(self.window.left + rel_left * self.window.width)
        y = int(self.window.top + rel_top * self.window.height)
        w = int(rel_width * self.window.width)
        h = int(rel_height * self.window.height)

        result = image_tools.detect_red_or_green_circle_stable(
            region_coords=(x, y, w, h),
            samples=50,
            required_ratio=0.8,
            min_pixels=10,
            tolerance=50
        )
        return result != "red" if result else False

    def claim_live_arena_rewards(self):
        for reward in ["live_arena_reward_1", "live_arena_reward_2", "live_arena_reward_3", "live_arena_reward_4", "live_arena_reward_5"]:
            # Click twice per reward
            for _ in range(2):
                window_tools.click_center(self.window, self.search_areas[reward], delay=1)

    # ------------------------- Pick Phase -------------------------
    def execute_simple_pick_phase(self):
        try:
            confirm_button = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas["confirm_button_champion_selection"]
            )[0]
            turn_counter = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas["turn_counter_roster"]
            )[0]

            if self.resembles(confirm_button.text, 'Confirmar') and self.resembles(turn_counter.text, 'Tu turno'):
                for i in range(1, 9):
                    window_tools.click_center(self.window, self.search_areas[f"preset_champion_{i}"], delay=1)
                window_tools.click_center(self.window, self.search_areas["confirm_button_champion_selection"], delay=1)
        except:
            pass

    def execute_complex_pick_phase(self):
        pass

    # ------------------------- Encounter -------------------------
    def execute_live_arena_encounter(self):
        self.reset_battle_state()
        window_tools.click_center(self.window, self.search_areas["start_encounter"])

        while self.main_loop_running and (self.battle_status != 'Done'):
            self.update_battle_outcome()
            self.execute_simple_pick_phase()
            self.update_battle_activity_status()

        while self.main_loop_running and (self.battle_status == 'Done'):
            battle_finished = image_tools.get_text_in_relative_area(
                self.reader, self.window, search_area=self.search_areas['battle_status_finished']
            )[0]
            if self.resembles(battle_finished.text, "VOLVER A LA ARENA"):
                window_tools.click_center(self.window, self.search_areas["battle_status_finished"])
                self.battle_status = 'menu'

    # ------------------------- Main Live Arena Loop -------------------------
    def run_live_arena_loop(self, main_loop_running = True):
        self.main_loop_running = main_loop_running
        time.sleep(5)
        self.start_time = time.time()
        self.running = True
        time.sleep(5)

        while self.main_loop_running and (self.running):
            if not self.is_live_arena_active():
                self.running = False
                print("Live arena not active")
                continue

            self.claim_live_arena_rewards()
            self.ensure_arena_coins()
            self.battle_occured = False

            if self.no_coin_status:
                self.running = False
            else:
                self.execute_live_arena_encounter()

            self.report_run_status()