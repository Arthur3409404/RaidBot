"""Pure message builders for mainframe status and configuration reporting."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any


def format_value(value: Any, max_len: int = 120) -> str:
    raw = repr(value)
    if len(raw) <= max_len:
        return raw
    return f"{raw[: max_len - 3]}..."


def build_help_lines(snapshot: Mapping[str, Any], editable_params_lines: Iterable[str]) -> list[str]:
    lines = [
        "[Bot Help] Available commands:",
        "- start / resume",
        "- stop / pause",
        "- restart",
        "- status",
        "- modes",
        "- params [filter]",
        "- get <parameter_name>",
        "- set <parameter_name> <value>",
        "- toggle <mode_name> [on|off]",
        "- reload / reload_config",
        "- ping",
        "",
        "[Bot Help] Runtime snapshot:",
        f"- current_mode: {snapshot['current_mode']}",
        f"- manual_mode: {snapshot['manual_mode']}",
        f"- main_loop_running: {snapshot['main_loop_running']}",
        (
            "- enabled_modes: "
            f"{', '.join(snapshot['enabled_modes']) if snapshot['enabled_modes'] else 'none'}"
        ),
    ]
    lines.extend(editable_params_lines)
    return lines


def build_editable_params_overview_lines(
    keys: list[str],
    max_line_length: int = 190,
) -> list[str]:
    if not keys:
        return ["", "[Bot Help] Editable parameters: none loaded."]

    lines = ["", f"[Bot Help] Editable parameters ({len(keys)} total):"]
    current_line = "- "
    for key in keys:
        token = f"`{key}`"
        candidate = token if current_line == "- " else f", {token}"
        if len(current_line) + len(candidate) > max_line_length:
            lines.append(current_line)
            current_line = f"- {token}"
        else:
            current_line += candidate

    if current_line != "- ":
        lines.append(current_line)

    lines.append("- tip: use `params <text>` to filter, then `get <key>` / `set <key> <value>`.")
    lines.append("- tip: dotted keys are supported too (example: `set run.hydra false`).")
    return lines


def build_status_lines(snapshot: Mapping[str, Any]) -> list[str]:
    lines = [
        "[Bot Status]",
        f"- timestamp_cest: {snapshot['timestamp_cest']}",
        f"- uptime_seconds: {snapshot['uptime_seconds']}",
        f"- current_mode: {snapshot['current_mode']}",
        f"- manual_mode: {snapshot['manual_mode']}",
        f"- main_loop_running: {snapshot['main_loop_running']}",
        (
            "- enabled_modes: "
            f"{', '.join(snapshot['enabled_modes']) if snapshot['enabled_modes'] else 'none'}"
        ),
    ]
    if snapshot["last_error"]:
        lines.append(f"- last_error: {snapshot['last_error']}")
    return lines


def build_modes_lines(run_params: Mapping[str, Any]) -> list[str]:
    lines = ["[Bot Modes]"]
    for key in sorted(run_params.keys()):
        state = "ENABLED" if bool(run_params[key]) else "DISABLED"
        lines.append(f"- {key}: {state}")
    return lines


def build_params_lines(
    keys: list[str],
    get_value: Callable[[str], Any],
    *,
    search_term: str | None = None,
    max_items: int = 80,
    value_formatter: Callable[[Any], str] = format_value,
) -> list[str]:
    if search_term:
        needle = search_term.lower()
        keys = [key for key in keys if needle in key.lower()]

    if not keys:
        return [f"[Bot Config] No parameters match filter: `{search_term}`"]

    truncated = len(keys) > max_items
    selected_keys = keys[:max_items]

    lines = ["[Bot Config] Parameters:"]
    for key in selected_keys:
        lines.append(f"- {key} = {value_formatter(get_value(key))}")

    if truncated:
        lines.append(f"- ... {len(keys) - max_items} more (refine filter with `params <text>`)")

    return lines
