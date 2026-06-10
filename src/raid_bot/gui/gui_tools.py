# -*- coding: utf-8 -*-
"""Desktop control panel for starting/stopping/updating the bot runtime."""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox

from raid_bot.utils import file_tools


class BotGUI:
    PID_FILE = os.path.join("data", "tmp", "run_bot.pid")

    def __init__(self):
        requested_account = os.environ.get(
            "RAID_ACCOUNT_NAME",
            file_tools.DEFAULT_MAIN_ACCOUNT_NAME,
        )
        self.profile_resolution = file_tools.resolve_profile_params_file(
            account_name=requested_account,
            allow_main_profile_fallback_for_missing_account=False,
        )
        self.param_store = file_tools.ParameterStore(self.profile_resolution.selected_param_file)
        self.params = self.param_store.get_grouped_copy()
        self._log_profile_resolution()

        self.bot_process = None
        self.bot_pid = None
        self.start_btn = None
        self.stop_btn = None

        self.root = tk.Tk()
        self.root.title("Raid Bot Control Panel")
        self.root.resizable(False, False)

        try:
            self.root.iconbitmap(r"data\\assets\\images\\icon.ico")
        except Exception as exc:
            print(f"[WARNING] Could not set icon: {exc}")

        self.build_layout()
        self._schedule_process_poll()

    def _log_profile_resolution(self):
        print(
            "[INFO] Loaded GUI profile params: "
            f"account='{self.profile_resolution.account_name}', "
            f"profile_account='{self.profile_resolution.selected_profile_account_name}', "
            f"file='{self.profile_resolution.selected_param_file}'."
        )
        if self.profile_resolution.migrated_legacy:
            print(
                "[INFO] Migrated legacy params file "
                f"'{self.profile_resolution.legacy_param_file}' -> "
                f"'{self.profile_resolution.main_profile_file}'."
            )
        elif self.profile_resolution.used_legacy_fallback:
            print(
                "[WARNING] Using legacy params fallback file: "
                f"{self.profile_resolution.legacy_param_file}"
            )

        if self.profile_resolution.used_main_profile_fallback:
            print(
                "[WARNING] Missing requested account profile. "
                f"Using main profile: {self.profile_resolution.main_profile_file}"
            )

        for missing_path in self.profile_resolution.missing_profile_files:
            print(f"[WARNING] Missing profile params file: {missing_path}")

        for generated_path in self.profile_resolution.generated_secondary_profiles:
            generated_name = generated_path.stem.split("_params_mainframe")[0]
            print(
                "[INFO] Generated secondary profile params for account "
                f"'{generated_name}': {generated_path}"
            )

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

        run_flags = self.params.get("run", {})
        if not run_flags:
            tk.Label(frame, text="No run flags detected in params file.").grid(row=0, column=0, sticky="w")
            return

        for row, key in enumerate(sorted(run_flags.keys())):
            value = bool(run_flags[key])
            label = key.replace("_", " ").title()
            status = "ENABLED" if value else "DISABLED"

            tk.Label(frame, text=f"{label}:", anchor="w", width=30).grid(row=row, column=0, sticky="w")
            tk.Label(
                frame,
                text=status,
                fg="green" if value else "red",
                width=12,
            ).grid(row=row, column=1, sticky="w")

    def build_log_display(self):
        log_frame = tk.LabelFrame(self.root, text="Bot Feedback", padx=5, pady=5)
        log_frame.pack(padx=10, pady=(0, 10), fill="both", expand=True)

        self.log_text = tk.Text(log_frame, height=4, state="disabled", wrap="word")
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

    def log_message(self, message, is_error=False):
        self.log_text.configure(state="normal")
        prefix = "[ERROR]" if is_error else "[INFO]"
        self.log_text.insert(tk.END, f"{prefix} {message}\n")
        self.log_text.see(tk.END)
        self.log_text.update_idletasks()
        self.log_text.configure(state="disabled")

    def build_controls(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        self.start_btn = tk.Button(
            frame,
            text="START BOT",
            width=20,
            height=2,
            command=self.start_bot,
        )
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = tk.Button(
            frame,
            text="STOP BOT",
            width=20,
            height=2,
            command=self.stop_bot,
            state="disabled",
        )
        self.stop_btn.grid(row=0, column=1, padx=5)

        update_btn = tk.Button(
            frame,
            text="UPDATE BOT",
            width=20,
            height=2,
            command=self.run_updater,
        )
        update_btn.grid(row=0, column=2, padx=5)

    def _is_pid_running(self, pid: int | None) -> bool:
        if not pid or pid <= 0:
            return False

        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        output = (result.stdout or "").strip()
        return bool(output and not output.startswith("INFO:"))

    def _read_pid_file(self) -> int | None:
        if not os.path.exists(self.PID_FILE):
            return None

        try:
            with open(self.PID_FILE, "r", encoding="utf-8") as handle:
                raw_pid = handle.read().strip()
        except OSError:
            return None

        if not raw_pid:
            return None

        try:
            pid = int(raw_pid)
        except ValueError:
            return None

        if self._is_pid_running(pid):
            return pid

        try:
            os.remove(self.PID_FILE)
        except OSError:
            pass
        return None

    def _sync_button_state(self):
        tracked_pid = self._read_pid_file()
        if self.bot_process and self.bot_process.poll() is not None:
            self.bot_process = None

        self.bot_pid = tracked_pid
        is_running = bool(
            self.bot_pid
            or (self.bot_process and self.bot_process.poll() is None)
        )

        self.start_btn.config(state="disabled" if is_running else "normal")
        self.stop_btn.config(state="normal" if is_running else "disabled")

    def _schedule_process_poll(self):
        self._sync_button_state()
        self.root.after(1000, self._schedule_process_poll)

    # -------------------------------------------------
    # BOT CONTROL
    # -------------------------------------------------
    def start_bot(self):
        if (self.bot_process and self.bot_process.poll() is None) or self._read_pid_file():
            self.log_message("Bot is already running.")
            return
        self.bot_process = None

        try:
            self.log_message("Starting bot process...")
            self.bot_process = subprocess.Popen([sys.executable, "run_bot.py"], cwd=os.getcwd())
            self.bot_pid = self.bot_process.pid
            self._sync_button_state()
        except Exception as exc:
            messagebox.showerror("Bot Error", str(exc))
            self.log_message(str(exc), is_error=True)

    def stop_bot(self):
        tracked_pid = self._read_pid_file()

        if not self.bot_process and not tracked_pid:
            self.log_message("Bot is not running.")
            return

        self.log_message("Stopping bot process...")
        if self.bot_process and self.bot_process.poll() is None:
            self.bot_process.terminate()
            try:
                self.bot_process.wait(timeout=5)
            except Exception:
                self.bot_process.kill()
        elif tracked_pid:
            subprocess.run(
                ["taskkill", "/F", "/PID", str(tracked_pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

        self.bot_process = None
        self.bot_pid = None
        self._sync_button_state()

    # -------------------------------------------------
    # UPDATE LOGIC
    # -------------------------------------------------
    def run_updater(self):
        if not messagebox.askyesno(
            "Update Bot",
            "The bot will close and update itself.\nContinue?",
        ):
            return

        try:
            updater_path = os.path.join(os.getcwd(), "scripts", "updater.py")
            subprocess.Popen([sys.executable, updater_path])
            self.root.destroy()
        except Exception as exc:
            messagebox.showerror("Update Failed", str(exc))
            self.log_message(str(exc), is_error=True)

    def run(self):
        self.root.mainloop()
