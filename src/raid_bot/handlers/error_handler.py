# -*- coding: utf-8 -*-
"""Runtime error handlers and Discord remote-control integration."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import threading
import time
from concurrent.futures import Future
import difflib

try:
    import discord
except Exception:  # pragma: no cover - optional runtime dependency
    discord = None

import raid_bot.utils.image_tools as image_tools
import raid_bot.utils.window_tools as window_tools


logger = logging.getLogger(__name__)


class DiscordRemoteOverride:
    """Discord client wrapper for receiving control commands and sending status."""

    _KNOWN_COMMANDS = {
        "start",
        "stop",
        "restart",
        "help",
        "status",
        "show_stats",
        "stats",
        "modes",
        "params",
        "get",
        "set",
        "toggle",
        "reload",
        "reload_config",
        "ping",
        "resume",
        "pause",
        "mode",
        "commands",
        "?",
    }

    def __init__(self, token: str, guild_name: str, channel_name: str):
        if discord is None:
            raise RuntimeError("discord.py is not installed.")

        self.token = token
        self.guild_name = guild_name
        self.channel_name = channel_name

        self._last_command: str | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self.loop: asyncio.AbstractEventLoop | None = None

        intents = discord.Intents.default()
        intents.messages = True
        intents.guilds = True
        intents.message_content = True
        self.client = discord.Client(intents=intents)

        @self.client.event
        async def on_ready():
            logger.info("[Discord] Connected as %s", self.client.user)
            self._ready_event.set()

        @self.client.event
        async def on_disconnect():
            logger.warning("[Discord] Gateway disconnected.")
            self._ready_event.clear()

        @self.client.event
        async def on_resumed():
            logger.info("[Discord] Gateway session resumed.")
            self._ready_event.set()

        @self.client.event
        async def on_message(message: discord.Message):
            if message.author == self.client.user:
                return
            if message.channel.name != self.channel_name:
                return

            command = self._extract_command(message.content)
            if not command:
                return

            logger.info("[Discord] Command received: %s", command)
            with self._lock:
                self._last_command = command

    def _extract_command(self, message_content: str) -> str | None:
        content = (message_content or "").strip()
        if not content:
            return None

        try:
            tokens = shlex.split(content, posix=False)
        except ValueError:
            tokens = content.split()

        if not tokens:
            return None

        if tokens[0].lower() not in self._KNOWN_COMMANDS:
            return None

        return content

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        self._ready_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="discord-override")
        self._thread.start()

    def _run(self):
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.client.start(self.token))
        except Exception:
            logger.exception("[Discord] Failed to start remote override client.")

    def stop(self):
        if not self.loop or not self.loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self.client.close(), self.loop)

    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    def is_running(self) -> bool:
        return bool(
            self._thread
            and self._thread.is_alive()
            and self.loop
            and self.loop.is_running()
            and not self.client.is_closed()
        )

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        return self._ready_event.wait(timeout)

    def get_last_command(self) -> str | None:
        with self._lock:
            return self._last_command

    def pop_last_command(self) -> str | None:
        with self._lock:
            command = self._last_command
            self._last_command = None
        return command

    def clear_last_command(self):
        with self._lock:
            self._last_command = None

    async def _resolve_channel(self):
        guild = discord.utils.get(self.client.guilds, name=self.guild_name)
        if not guild:
            return None
        return discord.utils.get(guild.text_channels, name=self.channel_name)

    @staticmethod
    def _chunk_message(text: str, limit: int = 1800) -> list[str]:
        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        pending = text
        while len(pending) > limit:
            split_at = pending.rfind("\n", 0, limit)
            if split_at <= 0:
                split_at = limit
            chunks.append(pending[:split_at].rstrip())
            pending = pending[split_at:].lstrip()
        if pending:
            chunks.append(pending)
        return chunks

    async def _send_message_async(self, text: str):
        channel = await self._resolve_channel()
        if not channel:
            return

        for chunk in self._chunk_message(text):
            await channel.send(chunk)

    async def _send_image_async(self, image_path: str, caption: str | None = None):
        channel = await self._resolve_channel()
        if not channel:
            return

        file = discord.File(image_path)
        if caption:
            await channel.send(content=caption, file=file)
        else:
            await channel.send(file=file)

    def _dispatch_coroutine(self, coro) -> Future | None:
        if not self.loop or not self.loop.is_running():
            return None
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)

        def _log_failure(done_future: Future):
            try:
                done_future.result()
            except Exception:
                logger.exception("[Discord] Async dispatch failed.")

        future.add_done_callback(_log_failure)
        return future

    def send_message(self, text: str):
        future = self._dispatch_coroutine(self._send_message_async(text))
        if future is None:
            logger.debug("[Discord] send_message skipped (client not ready).")

    def send_message_blocking(self, text: str, timeout: float = 15.0) -> bool:
        self.wait_until_ready(timeout=timeout)
        future = self._dispatch_coroutine(self._send_message_async(text))
        if future is None:
            logger.debug("[Discord] send_message_blocking skipped (client not ready).")
            return False

        try:
            future.result(timeout=timeout)
            return True
        except Exception:
            logger.exception("[Discord] Blocking message dispatch failed.")
            return False

    def send_image(self, image_path: str, caption: str | None = None):
        if not os.path.exists(image_path):
            logger.warning("[Discord] Image not found: %s", image_path)
            return
        future = self._dispatch_coroutine(self._send_image_async(image_path, caption))
        if future is None:
            logger.debug("[Discord] send_image skipped (client not ready).")


class RSL_Bot_ErrorHandler:
    """OCR-driven error detector for recoverable in-game failure popups."""

    def __init__(self, reader=None, window=None, title_substring="Raid: Shadow Legends"):
        self.reader = reader
        self.running = True
        self.window = window
        self.manual_play_enabled = False

        if self.window:
            self.coords = (self.window.left, self.window.top, self.window.width, self.window.height)
            logger.info("Window Coordinates: %s", self.coords)
        else:
            self.coords = None

        self.search_areas = {
            "internet_connectivity_error_name": [0.408, 0.335, 0.179, 0.036],
            "internet_connectivity_error_retry_connection": [0.506, 0.53, 0.211, 0.084],
        }

    @staticmethod
    def resembles(text: str, target: str, threshold: float = 0.75) -> bool:
        ratio = difflib.SequenceMatcher(None, (text or "").lower(), target.lower()).ratio()
        return ratio >= threshold

    def _read_first_text(self, area_key: str) -> str:
        objects = image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            search_area=self.search_areas[area_key],
            power_detection=False,
        )
        if not objects:
            return ""
        return (objects[0].text or "").strip()

    def check_for_internet_connectivity_error(self):
        try:
            error_name = self._read_first_text("internet_connectivity_error_name")
            retry_text = self._read_first_text("internet_connectivity_error_retry_connection")

            if self.resembles(error_name, "ERROR DE CONEXION") or self.resembles(retry_text, "Reintentar"):
                window_tools.click_center(
                    self.window,
                    self.search_areas["internet_connectivity_error_retry_connection"],
                    delay=5,
                )
        except Exception:
            logger.debug("Connectivity error check failed.", exc_info=True)

    def run_once(self):
        self.check_for_internet_connectivity_error()

    def run_permanently(self, poll_interval: float = 1.0):
        while self.running:
            self.run_once()
            time.sleep(poll_interval)

