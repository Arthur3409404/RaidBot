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



def read_params(param_file):
    params = {}
    with open(param_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                try:
                    params[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    params[key] = value
    return params


import tkinter as tk
from tkinter import messagebox
import threading

class BotGUI:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.root = tk.Tk()
        self.root.title("Raid Bot Control Panel")
        self.build_layout()

    def build_layout(self):
        tk.Label(self.root, text="Select a bot action:", font=("Arial", 14)).pack(pady=10)

        run_hydra_btn = tk.Button(self.root, text="Run Hydra", font=("Arial", 12), width=20, command=self.threaded_run_hydra)
        run_hydra_btn.pack(pady=5)

        run_arena_btn = tk.Button(self.root, text="Run Classic Arena", font=("Arial", 12), width=20, command=self.threaded_run_classic_arena)
        run_arena_btn.pack(pady=5)

        exit_btn = tk.Button(self.root, text="Exit", font=("Arial", 12), width=20, command=self.root.destroy)
        exit_btn.pack(pady=20)

    def run(self):
        self.root.mainloop()

    def threaded_run_hydra(self):
        threading.Thread(target=self.run_hydra, daemon=True).start()

    def threaded_run_classic_arena(self):
        threading.Thread(target=self.run_classic_arena, daemon=True).start()

    def run_hydra(self):
        try:
            print("[INFO] Starting Hydra run...")
            self.bot.run_hydra()
            messagebox.showinfo("Hydra", "Hydra run finished.")
        except Exception as e:
            messagebox.showerror("Error", f"Hydra failed:\n{e}")

    def run_classic_arena(self):
        try:
            print("[INFO] Starting Classic Arena...")
            self.bot.run_classic_arena()
            messagebox.showinfo("Classic Arena", "Classic Arena run finished.")
        except Exception as e:
            messagebox.showerror("Error", f"Arena failed:\n{e}")
            

class TextObject:
    def __init__(self, text, mean_pos_x, mean_pos_y):
        self.text = text
        self.mean_pos_x = mean_pos_x
        self.mean_pos_y = mean_pos_y
    
    def __repr__(self):
        return f"TextObject(text={self.text!r}, mean_pos_x={self.mean_pos_x:.1f}, mean_pos_y={self.mean_pos_y:.1f})"


class RSL_Bot:
    
    def __init__(self, title_substring="Raid: Shadow Legends", verbose = True, num_multi_refresh = 0, multi_refresh = False, power_threshold = 70000, use_gems = False, enemies_lost= [0], hydra_thresholds = [80, 14, 1.7], hydra_team = 2):

        self.reader = easyocr.Reader(['en'])  # You can add 'de' if you expect German text
        
        self.running = True
        
        self.battles_done = 0
        self.multi_refresh = multi_refresh
        self.num_multi_refresh = num_multi_refresh
        self.verbose = verbose
        self.enemies_lost = enemies_lost
        self.use_gems = use_gems
        self.offset_wins = len(self.enemies_lost)
        self.window = self.find_window(title_substring)
        
        self.hydra_thresholds = hydra_thresholds
        self.hydra_team = hydra_team
        
        if self.window:
            self.left = self.window.left
            self.top = self.window.top
            self.width = self.window.width
            self.height = self.window.height
            self.coords = (self.left, self.top, self.width, self.height)
            print(f"Window Coordinates: {self.coords}")
        else:
            self.coords = None
            
        self.power_threshold = power_threshold
        self.refresh_minutes = 15.2
        # Search Areas
        self.search_areas = {
            "bronce_medals": [0.05, 0.10, 0.20, 0.15],   # [left, top, width, height]
            "silver_medals":   [0.70, 0.10, 0.25, 0.20],
            "gold_medals":      [0.40, 0.80, 0.20, 0.10],
            
            "refresh_timer":   [0.8, 0.18, 0.15, 0.08],
            "arena_coins":   [0.72, 0.05, 0.04, 0.03],
            "add_arena_coins":   [0.70, 0.05, 0.025, 0.035],
            "confirm_add_arena_coins":   [0.40, 0.65, 0.2, 0.1],
            "confirm_use_gems":   [0.4, 0.66, 0.2, 0.08],
            
            
            "list_enemies":     [0.69, 0.28, 0.21, 0.7],
            "start_battle":     [0.8, 0.8, 0.2, 0.2],
            "battle_finished":  [0.35, 0.88, 0.3, 0.08],
            "battle_result":    [0.4, 0.15, 0.2, 0.05],
            
            "test":   [0.05, 0.30, 0.15, 0.08],
            
            "hydra_finished":     [0.37, 0.13, 0.25, 0.08],
            "hydra_score":     [0.2, 0.27, 0.6, 0.08],
            "hydra_retry_battle":  [0.65, 0.92, 0.12, 0.05],
            "hydra_save_battle":  [0.8, 0.92, 0.12, 0.05],
            "hydra_start_battle":    [0.78, 0.85, 0.13, 0.1],
            "hydra_dygest":    [0.02, 0.15, 0.96, 0.06],
        }

    @staticmethod
    def find_window(title_substring="Raid: Shadow Legends"):
        windows = gw.getWindowsWithTitle(title_substring)
        for win in windows:
            if title_substring.lower() in win.title.lower():
                print(f"Found window: {win.title}")
                return win
        print("Game window not found.")
        return None

    @staticmethod
    def test_window():
        """
        Captures a screenshot of the game window and plots it.
        """
        win = RSL_Bot.find_window()
        if win is None:
            return

        # Get window region
        left, top, width, height = win.left, win.top, win.width, win.height
        #print(f"Window Coordinates: left={left}, top={top}, width={width}, height={height}")

        # Take screenshot of the window region
        screenshot = pyautogui.screenshot(region=(left, top, width, height))

        # Plot the image
        plt.figure(figsize=(10, 6))
        plt.imshow(screenshot)
        plt.title("Captured Game Window")
        plt.axis("off")
        plt.show()

    @staticmethod
    def compare_pngs(path1, path2):
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        if img1.shape != img2.shape:
            #print("Images have different sizes.")
            return None
        score, _ = ssim(img1, img2, full=True)
        #print(f"SSIM similarity score: {score:.4f}")
        return score

    def get_text_from_image(self, image):
            """Uses EasyOCR to extract text from an image."""
            result = self.reader.readtext(np.array(image))
            return "\n".join([text for (_, text, conf) in result if conf > 0.6])
    
    def get_text_in_full_area(self):
        if not self.coords:
            #print("No window coordinates found.")
            return None
        screenshot = pyautogui.screenshot(region=self.coords)
        text = self.get_text_from_image(screenshot)
        return text

    def get_text_in_relative_area(self, search_area=[0.7, 0.3, 0.3, 0.3], powerdetection = False):
        """
        Takes a screenshot of a relative area within the game window and extracts text using OCR.
        Returns a list of TextObject instances, each with text and absolute mean position (x, y).
        
        search_area is [rel_left, rel_top, rel_width, rel_height]
        """
        if not self.coords:
            #print("No window coordinates found.")
            return []
    
        rel_left, rel_top, rel_width, rel_height = search_area
    
        # Convert relative to absolute window coordinates
        abs_left = self.left + int(rel_left * self.width)
        abs_top = self.top + int(rel_top * self.height)
        abs_width = int(rel_width * self.width)
        abs_height = int(rel_height * self.height)
    
        # Screenshot of the relative area
        screenshot = pyautogui.screenshot(region=(abs_left, abs_top, abs_width, abs_height))
        image_np = np.array(screenshot)
    
        # OCR using easyocr
        if powerdetection:
            results = self.reader.readtext(image_np, allowlist='0123456789.,KkMmLUCHARluchar ')
            
        else:
            results = self.reader.readtext(image_np)

            
        text_objects = []
        for bbox, text, confidence in results:
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
    
            mean_x = sum(xs) / 4
            mean_y = sum(ys) / 4
    
            abs_x = abs_left + mean_x
            abs_y = abs_top + mean_y
    
            text_obj = TextObject(text=text, mean_pos_x=abs_x, mean_pos_y=abs_y)
            text_objects.append(text_obj)
            
        #print(text_objects)
        return text_objects
    
    

    def visualize_text_detection(self):
        if not self.coords:
            #print("No window to capture.")
            return
        screenshot = pyautogui.screenshot(region=self.coords)
        image_cv = np.array(screenshot)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        result = self.reader.readtext(image_cv)

        for (bbox, text, conf) in result:
            if conf > 0.6:
                # bbox is a list of 4 points
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(image_cv, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                x, y = pts[0]
                cv2.putText(image_cv, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        plt.title("Detected Text")
        plt.axis("off")
        plt.show()
        
    def visualize_search_area(self, search_area=[0.57, 0.28, 0.22, 0.27]):
        """
        Visualizes a rectangle based on relative percentages of the window's dimensions.
        search_area should be a list or tuple: [rel_left, rel_top, rel_width, rel_height]
        """
        if not self.coords:
            #print("No window to visualize.")
            return

        rel_left, rel_top, rel_width, rel_height = search_area

        # Screenshot of the full window
        screenshot = pyautogui.screenshot(region=self.coords)
        image_cv = np.array(screenshot)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Convert relative to absolute
        x = int(rel_left * self.width)
        y = int(rel_top * self.height)
        w = int(rel_width * self.width)
        h = int(rel_height * self.height)

        # Draw rectangle
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image_cv, "Search Area", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        plt.title("Relative Search Area")
        plt.axis("off")
        plt.show()


    def find_clusters(self, text_objects, num_majorities=2, tolerance=8):
        """
        Groups text_objects by proximity of mean_pos_x within tolerance,
        finds top `num_majorities` clusters with most objects,
        returns a cleaned list of text_objects in original order
        including only those belonging to the top clusters.
        """
    
        # Step 1: Sort objects by mean_pos_x to form clusters
        sorted_objs = sorted(text_objects, key=lambda o: o.mean_pos_x)
    
        clusters = []
        current_cluster = []
        current_x = None
    
        for obj in sorted_objs:
            if current_x is None:
                current_x = obj.mean_pos_x
                current_cluster = [obj]
            else:
                if abs(obj.mean_pos_x - current_x) <= tolerance:
                    current_cluster.append(obj)
                    # Update cluster center to mean of members' mean_pos_x
                    current_x = sum(o.mean_pos_x for o in current_cluster) / len(current_cluster)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [obj]
                    current_x = obj.mean_pos_x
        if current_cluster:
            clusters.append(current_cluster)
    
        # Step 2: Sort clusters by size descending, take top N
        clusters.sort(key=len, reverse=True)
        top_clusters = clusters[:num_majorities]
    
        # Flatten the top clusters into a set for quick membership checking
        top_objs_set = set()
        for cluster in top_clusters:
            for obj in cluster:
                top_objs_set.add(obj)
    
        # Step 3: Filter original list preserving order
        cleaned_list = [obj for obj in text_objects if obj in top_objs_set]
    
        # Optional: print cluster info
        #print()
        #print(cleaned_list)
        
        return cleaned_list
    
    def get_battle_outcome(self,power_level):
        battle_result = self.get_text_in_relative_area(search_area=self.search_areas["battle_result"])[0]
        if battle_result.text == "VICTORIA":
            print('Victory')
        else: 
            self.enemies_lost.append(power_level)
            self.update_enemy_memory()
            print('Updated Enemy Avoid List')
            #save to txt file self.enemies_lost
            
    def update_enemy_memory(self):
        param_file = os.path.join("data", "params.txt")

        # Prepare the string representation of the list
        updated_line = f"enemies_lost = {self.enemies_lost}\n"
        
        # Read and update lines
        with open(param_file, "r") as f:
            lines = f.readlines()
        
        with open(param_file, "w") as f:
            for line in lines:
                if line.strip().startswith("enemies_lost ="):
                    f.write(updated_line)
                else:
                    f.write(line)
    
    def battle_enemy(self, obj, next_obj, power_level):
        battle_running = True
        
        self.click_at(obj.mean_pos_x,obj.mean_pos_y)
        time.sleep(3)
        start_button = self.get_text_in_relative_area(self.search_areas["start_battle"])[0]
        self.click_at(start_button.mean_pos_x,start_button.mean_pos_y)
        time.sleep(90)
        
        while battle_running:
            battle_finished = self.get_text_in_relative_area(search_area=self.search_areas['battle_finished'])[0]
            if battle_finished.text == "PULSA PARA CONTINUAR":
                battle_running = False
                time.sleep(3)
                #print('Continue but not working')
                #self.end_battle()
                self.get_battle_outcome(power_level)
                time.sleep(3)
                self.click_at(battle_finished.mean_pos_x,battle_finished.mean_pos_y)
                time.sleep(3)
                #self.end_battle()
                self.click_at(battle_finished.mean_pos_x,battle_finished.mean_pos_y)
            time.sleep(3)
        return

    def evaluate_enemies(self):
        text_objects = self.get_text_in_relative_area(search_area=self.search_areas["list_enemies"], powerdetection = True)
        filtered_objects = self.find_clusters(text_objects)
        
        idx = 0
        while idx < len(filtered_objects):
                obj = filtered_objects[idx]
        
                if obj in filtered_objects and obj.text.strip() == "Luchar":
                    if idx + 1 < len(filtered_objects):
                        next_obj = filtered_objects[idx + 1]
                        raw_text = next_obj.text.strip()
        
                        try:
                            # Match patterns like '66.65k', '5432abc', capturing number and optional trailing letters
                            matches = re.findall(r"(\d[\d.,]*)([a-zA-Z]*)", raw_text)
                            if not matches:
                                raise ValueError("No numeric value found in text")
                
                            number_part, suffix = matches[-1]  # Take the last match
                            number_part = number_part.replace('.', '').replace(',', '.').replace(' ', '')
                
                            # Try converting number part

                            num = float(str(number_part))

                            # Handle known suffixes
                            suffix = suffix.lower()
                            if suffix.startswith('k'):
                                power_val = num * 1000
                            elif suffix.startswith('m'):
                                power_val = num * 1_000_000
                            else:
                                power_val = num 
                        
                            if power_val >= self.power_threshold or power_val<500 or power_val in self.enemies_lost:
                                #print(f"Power {power_val} > threshold {self.power_threshold}")
                                pass  # Meets requirement
                            else:
                                print(f"Power {power_val} < threshold {self.power_threshold}")
                                self.battle_enemy(obj, next_obj, power_val)
                                time.sleep(3)
                                self.battle_occured = True
                                self.battles_done += 1
                                return self.battle_occured
                                
                        except Exception as e:
                            print(f"[!] Error parsing '{next_obj.text}': {e}")
        
                    idx += 2  # skip "Luchar" and its associated number
                else:
                    idx += 1
                    
        return self.battle_occured

    def move_up(self):
        """Click and drag from center downward."""
        if not self.coords:
            #print("No window coordinates available.")
            return

        start_x = self.left + self.width // 2
        start_y = self.top + self.height // 2
        end_y = start_y + int(self.height * 0.49)  # Drag 30% of height downward

        pyautogui.moveTo(start_x, start_y)
        pyautogui.mouseDown()
        time.sleep(0.1)
        pyautogui.moveTo(start_x, end_y, duration=0.2)
        pyautogui.mouseUp()
        time.sleep(5)
        return

    def move_down(self):
        """Click and drag from center upward."""
        if not self.coords:
            #print("No window coordinates available.")
            return

        start_x = self.left + self.width // 2
        start_y = self.top + self.height // 2
        end_y = start_y - int(self.height * 0.49)  # Drag 30% of height upward

        pyautogui.moveTo(start_x, start_y)
        pyautogui.mouseDown()
        time.sleep(0.1)
        pyautogui.moveTo(start_x, end_y, duration=0.2)
        pyautogui.mouseUp()
        time.sleep(5)
        return
    
    def click_at(self, x, y):
        """
        Click at the given absolute screen coordinates (x, y).
        """
        pyautogui.click(x, y)
        return

    def end_battle(self):
    
        rel_left, rel_top, rel_width, rel_height = self.search_areas["battle_finished"]
    
        abs_left = self.left + int(rel_left * self.width)
        abs_top = self.top + int(rel_top * self.height)
        abs_width = int(rel_width * self.width)
        abs_height = int(rel_height * self.height)
    
        center_x = abs_left + abs_width // 2
        center_y = abs_top + abs_height // 2
    
        pyautogui.click(center_x, center_y)
        #print(f"Clicked at ({center_x}, {center_y}) to refresh.")

    def refresh(self):
        if not self.coords or "refresh_timer" not in self.search_areas:
            #print("Missing window coordinates or refresh area definition.")
            return
    
        rel_left, rel_top, rel_width, rel_height = self.search_areas["refresh_timer"]
    
        abs_left = self.left + int(rel_left * self.width)
        abs_top = self.top + int(rel_top * self.height)
        abs_width = int(rel_width * self.width)
        abs_height = int(rel_height * self.height)
    
        center_x = abs_left + abs_width // 2
        center_y = abs_top + abs_height // 2
    
        pyautogui.click(center_x, center_y)
        #print(f"Clicked at ({center_x}, {center_y}) to refresh.")

    def check_arena_coins(self):
        time.sleep(1)
        coins_text = self.get_text_in_relative_area(self.search_areas["arena_coins"])[0]
        if coins_text.text == "0/10":
            rel_left, rel_top, rel_width, rel_height = self.search_areas["add_arena_coins"]
        
            abs_left = self.left + int(rel_left * self.width)
            abs_top = self.top + int(rel_top * self.height)
            abs_width = int(rel_width * self.width)
            abs_height = int(rel_height * self.height)
        
            center_x = abs_left + abs_width // 2
            center_y = abs_top + abs_height // 2
            

            pyautogui.click(center_x, center_y)
            time.sleep(3)
            confirm_text = self.get_text_in_relative_area(self.search_areas["confirm_add_arena_coins"])[0]
            
            confirm_gems_text = self.get_text_in_relative_area(self.search_areas["confirm_use_gems"])[0]
            if confirm_gems_text.text == "40" or confirm_gems_text.text == "Obtener y usar":
                if not self.use_gems:
                    pyautogui.click(center_x, center_y)
                    time.sleep(3)
                    return
            
            self.click_at(confirm_text.mean_pos_x,confirm_text.mean_pos_y)
            time.sleep(3)
            
        pass

    def print_status(self):
        elapsed = time.time() - self.start_time
        formatted_elapsed = str(timedelta(seconds=int(elapsed)))
        medals = (self.battles_done - len(self.enemies_lost) + self.offset_wins) * 4
    
        print("\n" + "=" * 40)
        print("🛡️  RAID Arena Bot Status")
        print("-" * 40)
        print(f"🔁 Mode: Multi Refresh ({self.num_multi_refresh})")
        print(f"⏱️  Time Since Start: {formatted_elapsed}")
        print(f"⚔️  Battles Won: {self.battles_done - len(self.enemies_lost) + self.offset_wins}")
        print(f"⚔️  Battles Lost: {len(self.enemies_lost) - + self.offset_wins}")
        print(f"🎖️  Estimated Medals: {medals}")
        print("-" * 40)
        print("🛑 To stop the bot, press 'v'")
        print("=" * 40 + "\n")
        

    def run_classic_arena(self):
        
        time.sleep(5)
        time_start = time.time()
        last_refresh_time = time_start
        self.start_time = time_start
        counter_multi_refresh = 0
        while self.running:
            
            self.print_status()
            
            self.battle_occured = False
            
            
            self.check_arena_coins()
            
            self.battle_occured = self.evaluate_enemies()
            if self.battle_occured:
                continue  # Restart loop if battle occurred
    
            self.move_down()
    
            self.battle_occured = self.evaluate_enemies()
            if self.battle_occured:
                continue  # Restart loop if battle occurred
            self.move_up()
            
            if self.multi_refresh == True:
                if counter_multi_refresh<self.num_multi_refresh:
                    time_start = time.time()
                    self.refresh()
                    counter_multi_refresh += 1
                    continue
                else:
                    counter_multi_refresh = 0
                    
            # Wait until `refresh_minutes` have passed
            print("Waiting for free Refresh")
            warned_2min = False
            warned_1min = False
            warned_30s = False
            time_start_loop = time.time()
            while (time.time() - time_start_loop) < (62):
                # if keyboard.is_pressed('v'):
                #     print("Stopping because 'v' was pressed.")
                #     self.running = False
                #     break
                # remaining = (self.refresh_minutes * 60) - (time.time() - time_start)

                # if remaining <= 120 and not warned_2min:
                #     print("⚠️ 2 minutes remaining until action.")
                #     warned_2min = True
                # elif remaining <= 60 and not warned_1min:
                #     print("⚠️ 1 minute remaining until action.")
                #     warned_1min = True
                # elif remaining <= 30 and not warned_30s:
                #     print("⚠️ 30 seconds remaining until action.")
                #     warned_30s = True
                
                time.sleep(1)


            elapsed = time.time() - last_refresh_time
            if elapsed >= self.refresh_minutes * 60:
                if self.running:
                    self.refresh()
                    last_refresh_time = time.time()
            
    def click_center(self, search_area):
        rel_left, rel_top, rel_width, rel_height = self.search_areas[search_area]
    
        abs_left = self.left + int(rel_left * self.width)
        abs_top = self.top + int(rel_top * self.height)
        abs_width = int(rel_width * self.width)
        abs_height = int(rel_height * self.height)
    
        center_x = abs_left + abs_width // 2
        center_y = abs_top + abs_height // 2
        
    
        pyautogui.click(center_x, center_y)
            
        # def evaluate_hydra(self):
        #     self.hydra_statistics - create 10 bins from 0 10 20 30 ...100
            
        #     for 20 runs:
        #         self.run_hydra
         
            
    def run_hydra(self):
        self.hydra_threshold = self.hydra_thresholds[self.hydra_team - 1]
        self.current_score = 0
    
        while True:
            start_text = self.get_text_in_relative_area(self.search_areas["hydra_start_battle"])[0]

            if start_text.text == 'Empezar':
                print("Status: playing hydra")
                self.click_center("hydra_start_battle")
    
                self.play_hydra()
    
                if self.current_score == 0:
                    print("Error: Hydra has not started!")
                    # You may want to break or continue depending on your logic
                    continue
    
                if self.current_score <= self.hydra_threshold * 1e6:
                    print(f"Score of {self.current_score} was achieved — below threshold. Continuing Hydra...")
                    self.click_center("hydra_retry_battle")
                    time.sleep(2)
                    continue
    
                else:  # current_score > threshold
                    print(f"Score of {self.current_score} was achieved — reached threshold. Hydra run finished for team {self.hydra_team}")
                    #self.click_center("hydra_save_battle")
                    break  # Exit loop as run is finished
            
        
    def check_dygest_status_hydra(self, 
                                    samples=50, 
                                    required_ratio=0.8, 
                                    min_pixels=10,
                                    color_hex="349E1E", 
                                    tolerance=40):
        """
        Rapidly samples the dygest area 'samples' times with no delay and
        detects if any column consistently contains green pixels in at least
        `required_ratio` of cases.

        Returns:
            (dygest_status: bool, dygest_target: int|None)
        """
        green_history = []

        for _ in range(samples):
            # Prepare BGR target color
            hex_str = color_hex.lstrip("#")
            r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
            target_bgr = np.array([b, g, r], dtype=np.uint8)

            # Get window coordinates and region
            abs_left, abs_top, win_width, win_height = self.coords
            rel_left, rel_top, rel_w, rel_h = self.search_areas["hydra_dygest"]
            region_left = abs_left + int(rel_left * win_width)
            region_top = abs_top + int(rel_top * win_height)
            region_width = int(rel_w * win_width)
            region_height = int(rel_h * win_height)

            # Capture screenshot and convert
            screenshot = pyautogui.screenshot(region=(region_left, region_top, region_width, region_height))
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            # Calculate color difference mask
            diff = np.linalg.norm(frame.astype(np.int16) - target_bgr.astype(np.int16), axis=2)
            mask = (diff <= tolerance).astype(np.uint8)

            # Divide into 4 columns and count green pixels
            h, w = mask.shape
            col_width = w // 4
            green_counts = [
                cv2.countNonZero(mask[:, i * col_width:(i + 1) * col_width])
                for i in range(4)
            ]

            # Convert counts to 0/1 (presence)
            green_presence = [int(count >= min_pixels) for count in green_counts]
            green_history.append(green_presence)

        # Analyze results
        history_arr = np.array(green_history)  # shape: (samples, 4)
        presence_sums = np.sum(history_arr, axis=0)

        threshold = int(samples * required_ratio)
        #print(f"[DEBUG] Green presence per column over {samples} samples: {presence_sums} (threshold: {threshold})")

        for i, count in enumerate(presence_sums):
            if count >= threshold:
                return True, i

        return False, None

        

    def set_priority_target_hydra(self, dygest_target):
        """
        Clicks the center of the column specified by dygest_target (0–3)
        """
        if dygest_target is None or dygest_target not in [0, 1, 2, 3]:
            return

        if "hydra_dygest" not in self.search_areas:
            raise ValueError("search_areas must contain 'hydra_dygest'")

        abs_left, abs_top, win_width, win_height = self.coords
        rel_left, rel_top, rel_w, rel_h = self.search_areas["hydra_dygest"]

        region_left = abs_left + int(rel_left * win_width)
        region_top = abs_top + int(rel_top * win_height)
        region_width = int(rel_w * win_width)
        region_height = int(rel_h * win_height)

        col_width = region_width // 4
        center_x = region_left + dygest_target * col_width + col_width // 2
        center_y = region_top + 7*region_height // 2

        pyautogui.click(center_x, center_y)
            
            
    def play_hydra(self):
        
        self.priority_target = False
        target = 0
        playing_hydra = True  # You need to define this variable
        
        while playing_hydra:
            dygest_status, dygest_target = self.check_dygest_status_hydra()
    
            if dygest_status and not self.priority_target:
                self.set_priority_target_hydra(dygest_target)
                target = dygest_target
                self.priority_target = True
                #print('Targeted')
    
            elif not dygest_status and self.priority_target:
                # When dygest is not present but priority_target was set,
                # clear priority target (assuming click on None or some reset)
                self.set_priority_target_hydra(target)
                self.priority_target = False
                #print('Un-Targeted')
                
            try:
                finish_text = self.get_text_in_relative_area(self.search_areas["hydra_finished"])[0]
                
                if finish_text.text == 'RESULTADO':
                    print('finished')
                    score_text = self.get_text_in_relative_area(self.search_areas["hydra_score"])[0]
                    
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
                        power_val = num * 1000
                    elif suffix.startswith('m'):
                        power_val = num * 1_000_000
                    else:
                        power_val = num
        
                    playing_hydra = False
                    
            except:
                finish_text = None
                
        self.current_score = power_val


# Run test if script is executed directly
if __name__ == "__main__":
    param_file = os.path.join("data", "params.txt")
    params = read_params(param_file)
    bot = RSL_Bot(**params)
    
    gui = BotGUI(bot)
    gui.run()


    
    # search_area=[0.02, 0.15, 0.96, 0.06]
    # bot.visualize_search_area(search_area=search_area)
    # text_objects = bot.get_text_in_relative_area(search_area=search_area)
    # print(text_objects)
    

    
    # bot.find_clusters(text_objects)
    

    # print("Text found in area:")
    # print(bot.get_text_in_full_area())
    # bot.visualize_text_detection()