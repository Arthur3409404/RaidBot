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


import data.lib.modes.arena_tools as arena_tools
import data.lib.modes.hydra_tools as hydra_tools

class RSL_Bot_DemonLord():
    def __init__(self, title_substring="Raid: Shadow Legends", verbose = True):
        self.reader = easyocr.Reader(['en'])  # You can add 'de' if you expect German text
        
        self.running = True
        
        self.verbose = verbose

        self.window = window_tools.find_window(title_substring)
        
        if self.window:
            self.window = window_tools.WindowObject(self.window)
            self.coords = (self.window.left, self.window.top, self.window.width, self.window.height)
            print(f"Window Coordinates: {self.coords}")
        else:
            self.coords = None
            
        self.search_areas = {
            "menu_name": [0.008, 0.034, 0.23, 0.037],   # [left, top, width, height]
            "go_to_higher_menu":   [0.928, 0.031, 0.046, 0.039],
            
            "DemonLord_Keys":   [0.728, 0.035, 0.035, 0.03],
            "DemonLord_Brutal":   [0.599, 0.496, 0.384, 0.105],
            "DemonLord_NM":   [0.599, 0.618, 0.384, 0.105],
            "DemonLord_UNM":   [0.596, 0.738, 0.39, 0.109],
            "DemonLord_NameList":   [0.103, 0.136, 0.173, 0.846],
            "DemonLord_EnterEncounter":   [0.756, 0.885, 0.212, 0.084],
            "DemonLord_StartEncounter":   [0.763, 0.877, 0.211, 0.101],
            "DemonLord_Result":   [0.385, 0.154, 0.217, 0.06],
            "DemonLord_EndEncounter":   [0.371, 0.886, 0.217, 0.07],
            


        }