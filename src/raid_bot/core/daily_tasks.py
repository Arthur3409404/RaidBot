"""Daily task configuration and state helpers."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DungeonDailyTask:
    dungeon: str
    energy: int
    level: int | None = None
    difficulty: str | None = None

    @property
    def signature(self) -> str:
        difficulty = self.difficulty or "highest"
        level = "auto" if self.level is None else str(self.level)
        return f"{self.dungeon}:{difficulty}:{level}:{self.energy}"


def utc_date_string(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _parse_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return ""

    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        pass

    if stripped.isdigit():
        return int(stripped)

    return stripped.strip("'\"")


def _parse_assignment_blob(raw: str) -> dict[str, Any]:
    stripped = raw.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1].strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        stripped = stripped[1:-1].strip()

    parsed: dict[str, Any] = {}
    for part in stripped.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        if not key:
            continue
        parsed[key] = _parse_scalar(value)
    return parsed


def _coerce_task(raw_task: Any) -> DungeonDailyTask | None:
    if raw_task in (None, False, ""):
        return None

    if isinstance(raw_task, str):
        stripped = raw_task.strip()
        if not stripped:
            return None
        try:
            raw_task = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            raw_task = _parse_assignment_blob(stripped)

    if isinstance(raw_task, DungeonDailyTask):
        return raw_task

    if not isinstance(raw_task, dict):
        return None

    dungeon = str(raw_task.get("dungeon", "")).strip().lower()
    if not dungeon:
        return None

    energy = raw_task.get("energy")
    try:
        energy_int = int(energy)
    except (TypeError, ValueError):
        return None
    if energy_int <= 0:
        return None

    level = raw_task.get("level")
    try:
        level_int = int(level) if level is not None else None
    except (TypeError, ValueError):
        level_int = None

    difficulty = raw_task.get("difficulty")
    difficulty_text = str(difficulty).strip().lower() if difficulty is not None else None
    if difficulty_text not in {None, "", "normal", "hard"}:
        difficulty_text = None

    return DungeonDailyTask(
        dungeon=dungeon,
        energy=energy_int,
        level=level_int,
        difficulty=difficulty_text or None,
    )


def parse_dungeon_daily_tasks(raw_tasks: Any) -> list[DungeonDailyTask]:
    """Parse profile-configured dungeon daily tasks.

    Accepts Python literals and the loose profile syntax:
    ``[dungeon= shogun, level = 25, energy = 60]``.
    """
    if raw_tasks in (None, False, ""):
        return []

    if isinstance(raw_tasks, str):
        stripped = raw_tasks.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            parsed = _parse_assignment_blob(stripped)
        raw_tasks = parsed

    if isinstance(raw_tasks, dict):
        raw_items = [raw_tasks]
    elif isinstance(raw_tasks, (list, tuple)):
        if raw_tasks and all(not isinstance(item, (dict, str, DungeonDailyTask)) for item in raw_tasks):
            raw_items = []
        elif raw_tasks and all(isinstance(item, str) and "=" in item for item in raw_tasks):
            raw_items = [_parse_assignment_blob(",".join(raw_tasks))]
        else:
            raw_items = list(raw_tasks)
    else:
        raw_items = []

    tasks = []
    for raw_item in raw_items:
        task = _coerce_task(raw_item)
        if task is not None:
            tasks.append(task)
    return tasks


def load_daily_task_state(path: str | Path, utc_date: str) -> dict[str, Any]:
    state_path = Path(path)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {}

    if not isinstance(state, dict) or state.get("utc_date") != utc_date:
        return {"utc_date": utc_date, "dungeons": {}}

    dungeons = state.get("dungeons")
    if not isinstance(dungeons, dict):
        state["dungeons"] = {}
    return state


def save_daily_task_state(path: str | Path, state: dict[str, Any]) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
