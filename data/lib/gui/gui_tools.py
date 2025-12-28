# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import messagebox
import threading
import subprocess
import sys
import os
from collections import defaultdict

import data.lib.utils.file_tools as file_tools


class BotGUI:
    def __init__(self, bot_instance):
        param_file = os.path.join("data", "params_mainframe.txt")
        params = file_tools.read_params(param_file)
        self.params = self.group_params(params)

        self.bot = bot_instance

        self.bot_process = None
        self.start_btn = None
        self.stop_btn = None

        self.root = tk.Tk()
        self.root.title("Raid Bot Control Panel")
        self.root.resizable(False, False)

        # -----------------------------
        # Set window icon (.ico file)
        # -----------------------------
        try:
            self.root.iconbitmap("pic\icon.ico")  
        except Exception as e:
            print(f"[WARNING] Could not set icon: {e}")

        self.build_layout()

    def group_params(self, params: dict, min_shared_keys: int = 3):
        """
        Groups params by common prefixes.
        Everything not belonging to a detected group goes into 'mainframe'.
        """
        prefix_counts = defaultdict(int)

        for key in params.keys():
            parts = key.split("_")
            for i in range(1, len(parts)):
                prefix = "_".join(parts[:i]) + "_"
                prefix_counts[prefix] += 1

        valid_prefixes = {
            p for p, count in prefix_counts.items()
            if count >= min_shared_keys
        }

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

    # -------------------------------------------------
    # UI LAYOUT
    # -------------------------------------------------
    def build_layout(self):
        self.build_run_display()
        self.build_log_display()
        self.build_controls()

    def build_run_display(self):
        frame = tk.LabelFrame(self.root, text="Modules To Run", padx=10, pady=10)
        frame.pack(padx=10, pady=10, fill="both")

        run_flags = [
            ("Classic Arena", "classic_arena"),
            ("Tag Team Arena", "tagteam_arena"),
            ("Live Arena", "live_arena"),
            ("Dungeons", "dungeons"),
            ("Faction Wars", "factionwars"),
            ("Demon Lord", "demonlord"),
            ("Doom Tower", "doomtower"),
            ("Cursed City", "cursedcity"),
            ("Grim Forest", "grimforest"),
            ("Effective Unit Leveling", "effective_unit_leveling"),
        ]

        for row, (label, key) in enumerate(run_flags):
            value = self.params.get("run", {}).get(key, False)
            status = "✔ ENABLED" if value else "✖ DISABLED"

            tk.Label(frame, text=f"{label}:",
                     anchor="w", width=25).grid(row=row, column=0, sticky="w")

            tk.Label(frame, text=status,
                     fg="green" if value else "red",
                     width=12).grid(row=row, column=1, sticky="w")

    def build_log_display(self):
        log_frame = tk.LabelFrame(self.root, text="Bot Feedback", padx=5, pady=5)
        log_frame.pack(padx=10, pady=(0,10), fill="both", expand=True)

        # Text widget
        self.log_text = tk.Text(log_frame, height=30, state="disabled", wrap="word")
        self.log_text.pack(side="left", fill="both", expand=True)

        # Scrollbar attached to frame, not to Text itself
        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

    def log_message(self, message, is_error=False):
        """Insert a message and always scroll to the bottom."""
        self.log_text.configure(state="normal")
        if is_error:
            self.log_text.insert(tk.END, f"[ERROR] {message}\n")
        else:
            self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)  # Scroll reliably
        self.log_text.update_idletasks()  # Force update immediately
        self.log_text.configure(state="disabled")

    def build_controls(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        self.start_btn = tk.Button(
            frame,
            text="START BOT",
            width=20,
            height=2,
            command=self.start_bot
        )
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = tk.Button(
            frame,
            text="STOP BOT",
            width=20,
            height=2,
            command=self.stop_bot,
            state="disabled"
        )
        self.stop_btn.grid(row=0, column=1, padx=5)

        update_btn = tk.Button(
            frame,
            text="UPDATE BOT",
            width=20,
            height=2,
            command=self.run_updater
        )
        update_btn.grid(row=0, column=2, padx=5)

    # -------------------------------------------------
    # BOT CONTROL
    # -------------------------------------------------
    def start_bot(self):
        if self.bot_process:
            return

        try:
            self.log_message("[INFO] Starting bot...")

            # Start the bot process WITHOUT capturing stdout/stderr
            self.bot_process = subprocess.Popen(
                [sys.executable, "run_bot.py"]  # adjust your bot script path
            )

            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")

            # Start a thread to wait for bot to finish
            threading.Thread(target=self.wait_for_bot, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Bot Error", str(e))


    def wait_for_bot(self):
        """Wait for the bot process to finish without reading stdout/stderr."""
        self.bot_process.wait()
        self.bot_process = None
        self.root.after(0, lambda: self.start_btn.config(state="normal"))
        self.root.after(0, lambda: self.stop_btn.config(state="disabled"))
        self.root.after(0, lambda: self.log_message("[INFO] Bot stopped."))


    def stop_bot(self):
        if self.bot_process:
            self.log_message("[INFO] Stopping bot...")
            self.bot_process.terminate()
            self.bot_process = None

            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")

    # -------------------------------------------------
    # UPDATE LOGIC
    # -------------------------------------------------
    def run_updater(self):
        if not messagebox.askyesno(
            "Update Bot",
            "The bot will close and update itself.\nContinue?"
        ):
            return

        try:
            updater_path = os.path.join(os.getcwd(), "updater.py")

            subprocess.Popen([sys.executable, updater_path])
            self.root.destroy()

        except Exception as e:
            messagebox.showerror("Update Failed", str(e))

    # -------------------------------------------------
    def run(self):
        self.root.mainloop()