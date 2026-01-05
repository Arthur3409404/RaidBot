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


import data.lib.utils.image_tools as image_tools
import data.lib.utils.window_tools as window_tools
import data.lib.utils.file_tools as file_tools


import data.lib.modes.arena_tools as arena_tools




# Run test if script is executed directly
if __name__ == "__main__":
    print('ALWAYS RUN THE PROGRAM IN 1280 x 1024')
    
    param_file = os.path.join("data", "params.txt")
    
    params = file_tools.read_params(param_file)
    bot = arena_tools.RSL_Bot_ClassicArena(**params)
    
    #bot.evaluate_enemies()
    
    #bot.run_classic_arena_noTimeLimit()
    
    bot.run_classic_arena_once()
    
    # gui = BotGUI(bot)
    # gui.run()

    # self.corresponding_enemy_positions = {
    #     "Pos1": [[0.583, 0.31, 0.173, 0.023], [0.787, 0.237, 0.181, 0.082]],
    #     "Pos2": [[0.586, 0.43, 0.16, 0.019], [0.789, 0.357, 0.181, 0.078]],
    
    #     "Pos3": [[0.586, 0.548, 0.164, 0.019], [0.787, 0.473, 0.184, 0.084]],
    #     "Pos4": [[0.587, 0.664, 0.162, 0.021], [0.787, 0.592, 0.183, 0.08]],
    #     "Pos5": [[0.582, 0.782, 0.166, 0.022], [0.787, 0.709, 0.182, 0.082]],
    #     "Pos6": [[0.586, 0.901, 0.16, 0.018], [0.787, 0.827, 0.183, 0.082]],
    
    #     "Pos7": [[0.586, 0.592, 0.16, 0.02], [0.788, 0.519, 0.181, 0.079]],
    #     "Pos8": [[0.586, 0.709, 0.163, 0.021], [0.787, 0.636, 0.181, 0.08]],
    #     "Pos9": [[0.586, 0.829, 0.164, 0.019], [0.787, 0.754, 0.183, 0.081]],
    #     "Pos10": [[0.583, 0.946, 0.166, 0.02], [0.787, 0.874, 0.181, 0.078]],
    # }

    # for i in range(20):
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