from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from raid_bot.modes.session_encounter_state import (
    clear_session_lost_encounter_persistence_hook,
    reset_session_lost_encounters,
)


@pytest.fixture(autouse=True)
def _reset_session_lost_encounters():
    clear_session_lost_encounter_persistence_hook()
    reset_session_lost_encounters()
    yield
    clear_session_lost_encounter_persistence_hook()
