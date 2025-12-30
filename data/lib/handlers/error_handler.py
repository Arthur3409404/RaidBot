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


import data.lib.utils.image_tools as image_tools
import data.lib.utils.window_tools as window_tools
import data.lib.utils.file_tools as file_tools

class RSL_Bot_ErrorHandler():
    def __init__(self, reader = None, window =None, title_substring="Raid: Shadow Legends"):
        self.reader = reader  # You can add 'de' if you expect German text
        
        self.running = True

        self.window = window
        
        if self.window:
            self.coords = (self.window.left, self.window.top, self.window.width, self.window.height)
            print(f"Window Coordinates: {self.coords}")
        else:
            self.coords = None


        self.search_areas = {
            "menu_name": [0.008, 0.034, 0.23, 0.037],   # [left, top, width, height]
            "go_to_higher_menu":   [0.928, 0.031, 0.046, 0.039],
            "go_to_bastion":      [0.903, 0.9, 0.064, 0.059],
            "bastion_to_main_menu":      [0.808, 0.904, 0.168, 0.074],

            "internet_connectivity_error_name":      [0.408, 0.335, 0.179, 0.036],
            "internet_connectivity_error_continue":      [0.279, 0.529, 0.215, 0.086],
            "internet_connectivity_error_retry_connection":      [0.506, 0.53, 0.211, 0.084],


            'pov' : [0, 0, 1, 1],

        }
        
        
    def check_for_internet_connectivity_error(self):
        try:
            error_name = image_tools.get_text_in_relative_area(self.reader, self.window, search_area=self.search_areas["internet_connectivity_error_name"])[0]
            error_retry_connection = image_tools.get_text_in_relative_area(self.reader, self.window, search_area=self.search_areas["internet_connectivity_error_retry_connection"])[0]
            if error_name.text == 'ERROR DE CONEXION' or error_retry_connection.text == 'Reintentar':
                window_tools.click_center(self.window, self.search_areas["internet_connectivity_error_retry_connection"], delay = 5)
        except:
            pass
            
    def run_once(self):
        # =========================
        # 1. Internet Connectivity Error
        # =========================
        self.check_for_internet_connectivity_error()


    def run_permanently(self):
        run = True
    
        while run:

            # =========================
            # 1. Internet Connectivity Error
            # =========================
            self.check_for_internet_connectivity_error()
