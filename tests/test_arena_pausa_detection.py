from __future__ import annotations

import sys
import types
from pathlib import Path


def _install_stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


REPO_ROOT = Path(__file__).resolve().parents[1]

raid_bot_pkg = sys.modules.setdefault("raid_bot", types.ModuleType("raid_bot"))
raid_bot_pkg.__path__ = [str(REPO_ROOT / "src" / "raid_bot")]
utils_pkg = sys.modules.setdefault("raid_bot.utils", types.ModuleType("raid_bot.utils"))
utils_pkg.__path__ = [str(REPO_ROOT / "src" / "raid_bot" / "utils")]
handlers_pkg = sys.modules.setdefault("raid_bot.handlers", types.ModuleType("raid_bot.handlers"))
handlers_pkg.__path__ = [str(REPO_ROOT / "src" / "raid_bot" / "handlers")]
modes_pkg = sys.modules.setdefault("raid_bot.modes", types.ModuleType("raid_bot.modes"))
modes_pkg.__path__ = [str(REPO_ROOT / "src" / "raid_bot" / "modes")]

_install_stub_module("pyautogui")
_install_stub_module("raid_bot.utils.image_tools", get_text_in_relative_area=lambda *args, **kwargs: [])
_install_stub_module("raid_bot.utils.window_tools", sendkey=lambda *args, **kwargs: None)


class DummyText:
    def __init__(self, text: str, mean_pos_x: float, mean_pos_y: float):
        self.text = text
        self.mean_pos_x = mean_pos_x
        self.mean_pos_y = mean_pos_y


def _bot():
    return types.SimpleNamespace(
        reader=object(),
        window=object(),
        verbose=False,
        _pausa_esc_sent=False,
        resembles=lambda text, target: text.lower() == target.lower(),
    )


def test_stable_pausa_presses_esc_once(monkeypatch):
    sent_keys = []
    detections = [
        [DummyText("Pausa", 500, 250)],
        [DummyText("Pausa", 507, 244)],
    ]

    from raid_bot.utils import auto_battle_tools

    monkeypatch.setattr(
        auto_battle_tools.image_tools,
        "get_text_in_relative_area",
        lambda *args, **kwargs: detections.pop(0),
    )
    monkeypatch.setattr(auto_battle_tools.window_tools, "sendkey", lambda key, **kwargs: sent_keys.append(key))
    monkeypatch.setattr(auto_battle_tools.time, "sleep", lambda *_args, **_kwargs: None)

    bot = _bot()

    assert auto_battle_tools.handle_stable_pausa(bot) is True
    assert sent_keys == ["esc"]
    assert bot._pausa_esc_sent is True


def test_moved_pausa_does_not_press_esc(monkeypatch):
    sent_keys = []
    detections = [
        [DummyText("Pausa", 500, 250)],
        [DummyText("Pausa", 520, 250)],
    ]

    from raid_bot.utils import auto_battle_tools

    monkeypatch.setattr(
        auto_battle_tools.image_tools,
        "get_text_in_relative_area",
        lambda *args, **kwargs: detections.pop(0),
    )
    monkeypatch.setattr(auto_battle_tools.window_tools, "sendkey", lambda key, **kwargs: sent_keys.append(key))
    monkeypatch.setattr(auto_battle_tools.time, "sleep", lambda *_args, **_kwargs: None)

    bot = _bot()

    assert auto_battle_tools.handle_stable_pausa(bot) is False
    assert sent_keys == []
    assert bot._pausa_esc_sent is False
