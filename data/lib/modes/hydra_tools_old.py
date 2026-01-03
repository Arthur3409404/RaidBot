# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:44:43 2025

@author: Arthur
"""

         
# =============================================================================
# FUNCTIONS ARE OUTDATED, AND NEED TO BE IMPLEMENTED IN A NEW CLASS - LOOK CLASSIC_ARENA Bot
# =============================================================================


# def run_hydra(self):
#     self.hydra_threshold = self.hydra_thresholds[self.hydra_team - 1]
#     self.current_score = 0

#     while True:
#         start_text = self.get_text_in_relative_area(self.search_areas["hydra_start_battle"])[0]

#         if start_text.text == 'Empezar':
#             print("Status: playing hydra")
#             self.click_center("hydra_start_battle")

#             self.play_hydra()

#             if self.current_score == 0:
#                 print("Error: Hydra has not started!")
#                 # You may want to break or continue depending on your logic
#                 continue

#             if self.current_score <= self.hydra_threshold * 1e6:
#                 print(f"Score of {self.current_score} was achieved — below threshold. Continuing Hydra...")
#                 self.click_center("hydra_retry_battle")
#                 time.sleep(2)
#                 continue

#             else:  # current_score > threshold
#                 print(f"Score of {self.current_score} was achieved — reached threshold. Hydra run finished for team {self.hydra_team}")
#                 #self.click_center("hydra_save_battle")
#                 break  # Exit loop as run is finished
        
    
# def check_dygest_status_hydra(self, 
#                                 samples=50, 
#                                 required_ratio=0.8, 
#                                 min_pixels=10,
#                                 color_hex="349E1E", 
#                                 tolerance=40):
#     """
#     Rapidly samples the dygest area 'samples' times with no delay and
#     detects if any column consistently contains green pixels in at least
#     `required_ratio` of cases.

#     Returns:
#         (dygest_status: bool, dygest_target: int|None)
#     """
#     green_history = []

#     for _ in range(samples):
#         # Prepare BGR target color
#         hex_str = color_hex.lstrip("#")
#         r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
#         target_bgr = np.array([b, g, r], dtype=np.uint8)

#         # Get window coordinates and region
#         abs_left, abs_top, win_width, win_height = self.coords
#         rel_left, rel_top, rel_w, rel_h = self.search_areas["hydra_dygest"]
#         region_left = abs_left + int(rel_left * win_width)
#         region_top = abs_top + int(rel_top * win_height)
#         region_width = int(rel_w * win_width)
#         region_height = int(rel_h * win_height)

#         # Capture screenshot and convert
#         screenshot = pyautogui.screenshot(region=(region_left, region_top, region_width, region_height))
#         frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

#         # Calculate color difference mask
#         diff = np.linalg.norm(frame.astype(np.int16) - target_bgr.astype(np.int16), axis=2)
#         mask = (diff <= tolerance).astype(np.uint8)

#         # Divide into 4 columns and count green pixels
#         h, w = mask.shape
#         col_width = w // 4
#         green_counts = [
#             cv2.countNonZero(mask[:, i * col_width:(i + 1) * col_width])
#             for i in range(4)
#         ]

#         # Convert counts to 0/1 (presence)
#         green_presence = [int(count >= min_pixels) for count in green_counts]
#         green_history.append(green_presence)

#     # Analyze results
#     history_arr = np.array(green_history)  # shape: (samples, 4)
#     presence_sums = np.sum(history_arr, axis=0)

#     threshold = int(samples * required_ratio)
#     #print(f"[DEBUG] Green presence per column over {samples} samples: {presence_sums} (threshold: {threshold})")

#     for i, count in enumerate(presence_sums):
#         if count >= threshold:
#             return True, i

#     return False, None

    

# def set_priority_target_hydra(self, dygest_target):
#     """
#     Clicks the center of the column specified by dygest_target (0–3)
#     """
#     if dygest_target is None or dygest_target not in [0, 1, 2, 3]:
#         return

#     if "hydra_dygest" not in self.search_areas:
#         raise ValueError("search_areas must contain 'hydra_dygest'")

#     abs_left, abs_top, win_width, win_height = self.coords
#     rel_left, rel_top, rel_w, rel_h = self.search_areas["hydra_dygest"]

#     region_left = abs_left + int(rel_left * win_width)
#     region_top = abs_top + int(rel_top * win_height)
#     region_width = int(rel_w * win_width)
#     region_height = int(rel_h * win_height)

#     col_width = region_width // 4
#     center_x = region_left + dygest_target * col_width + col_width // 2
#     center_y = region_top + 7*region_height // 2

#     pyautogui.click(center_x, center_y)
        
        
# def play_hydra(self):
    
#     self.priority_target = False
#     target = 0
#     playing_hydra = True  # You need to define this variable
    
#     while playing_hydra:
#         dygest_status, dygest_target = self.check_dygest_status_hydra()

#         if dygest_status and not self.priority_target:
#             self.set_priority_target_hydra(dygest_target)
#             target = dygest_target
#             self.priority_target = True
#             #print('Targeted')

#         elif not dygest_status and self.priority_target:
#             # When dygest is not present but priority_target was set,
#             # clear priority target (assuming click on None or some reset)
#             self.set_priority_target_hydra(target)
#             self.priority_target = False
#             #print('Un-Targeted')
            
#         try:
#             finish_text = self.get_text_in_relative_area(self.search_areas["hydra_finished"])[0]
            
#             if finish_text.text == 'RESULTADO':
#                 print('finished')
#                 score_text = self.get_text_in_relative_area(self.search_areas["hydra_score"])[0]
                
#                 # Match patterns like '66.65k', '5432abc', capturing number and optional trailing letters
#                 matches = re.findall(r"(\d[\d.,]*)([a-zA-Z]*)", score_text.text)
#                 if not matches:
#                     raise ValueError("No numeric value found in text")

#                 number_part, suffix = matches[-1]  # Take the last match
#                 number_part = number_part.replace('.', '').replace(',', '.').replace(' ', '')

#                 # Try converting number part

#                 num = float(str(number_part))

#                 # Handle known suffixes
#                 suffix = suffix.lower()
#                 if suffix.startswith('k'):
#                     power_val = num * 1000
#                 elif suffix.startswith('m'):
#                     power_val = num * 1_000_000
#                 else:
#                     power_val = num
    
#                 playing_hydra = False
                
#         except:
#             finish_text = None
            
#     self.current_score = power_val