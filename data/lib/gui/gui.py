# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 23:43:28 2025

@author: Arthur
"""

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
            
