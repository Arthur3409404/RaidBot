"""Compatibility launcher for the RaidBot direct runner."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from raid_bot.run_bot import main


if __name__ == "__main__":
    raise SystemExit(main())
