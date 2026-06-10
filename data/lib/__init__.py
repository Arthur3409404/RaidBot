"""Compatibility aliases for the former ``data.lib`` package."""

from __future__ import annotations

from importlib import import_module
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_ALIASES = {
    "core": "raid_bot.core",
    "gui": "raid_bot.gui",
    "handlers": "raid_bot.handlers",
    "modes": "raid_bot.modes",
    "utils": "raid_bot.utils",
}

for legacy_name, target_name in _ALIASES.items():
    sys.modules[f"{__name__}.{legacy_name}"] = import_module(target_name)


def __getattr__(name: str):
    if name in _ALIASES:
        return sys.modules[f"{__name__}.{name}"]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
