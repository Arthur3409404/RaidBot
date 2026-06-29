"""Pure message builders for mainframe status and configuration reporting."""

from __future__ import annotations

import base64
import json
from math import ceil
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from raid_bot.utils import file_tools


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
        "- status (current daily entry)",
        "- show_stats / stats",
        "- cursedcity_avoid_current / cc_avoid_current",
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


def build_daily_entry_messages(
    day_key: str,
    daily_entry: Mapping[str, Any] | None,
    *,
    max_message_chars: int = 1800,
) -> list[str]:
    payload = daily_entry if isinstance(daily_entry, Mapping) else {}
    pretty = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    lines = pretty.splitlines() or ["{}"]

    chunks: list[str] = []
    current_lines: list[str] = []
    current_chars = 0
    overhead = len("```json\n\n```")

    for line in lines:
        line_len = len(line) + 1
        if current_lines and (current_chars + line_len + overhead) > max_message_chars:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_chars = line_len
        else:
            current_lines.append(line)
            current_chars += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))

    messages = [f"[Bot Status] Current daily entry for `{day_key}`"]
    messages.extend(f"```json\n{chunk}\n```" for chunk in chunks)
    return messages


def _nested_lookup(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _coerce_number(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _sum_metric_from_mapping(mapping: Mapping[str, Any], metric: str) -> int:
    total = 0
    for value in mapping.values():
        if isinstance(value, Mapping):
            total += _coerce_number(value.get(metric))
    return total


def _build_cumulative_series(day_entries: list[Mapping[str, Any]], extractor: Callable[[Mapping[str, Any]], int]) -> list[int]:
    cumulative = 0
    series = [0]
    for entry in day_entries:
        cumulative += max(0, int(extractor(entry)))
        series.append(cumulative)
    return series


def _daily_series_spec(
    title: str,
    series: list[tuple[str, Callable[[Mapping[str, Any]], int]]],
) -> dict[str, Any]:
    return {"title": title, "series": series}


_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3Z8RkAAAAASUVORK5CYII="
)


def _write_placeholder_png(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_PLACEHOLDER_PNG)
    return output_path


def build_daily_stats_figure(
    log_path: str | Path,
    output_path: str | Path | None = None,
    *,
    max_plots: int = 16,
) -> Path:
    """Build a cumulative multi-panel figure from the shared daily log."""
    document = file_tools.load_daily_log_document(log_path)
    days = document.get("days", {}) if isinstance(document, Mapping) else {}
    if not isinstance(days, Mapping):
        days = {}

    ordered_day_keys = [
        key for key in sorted(days.keys()) if isinstance(days.get(key), Mapping)
    ]
    day_entries = [days[key] for key in ordered_day_keys]

    plot_specs = [
        _daily_series_spec(
            "Classic Arena",
            [
                (
                    "wins",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "pvp", "classic_arena", "wins"))
                    ),
                ),
                (
                    "losses",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "pvp", "classic_arena", "losses"))
                    ),
                ),
            ],
        ),
        _daily_series_spec(
            "Tag Team Arena",
            [
                (
                    "wins",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "pvp", "tagteam_arena", "wins"))
                    ),
                ),
                (
                    "losses",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "pvp", "tagteam_arena", "losses"))
                    ),
                ),
            ],
        ),
        _daily_series_spec(
            "Live Arena",
            [
                (
                    "wins",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "pvp", "live_arena", "wins"))
                    ),
                ),
                (
                    "losses",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "pvp", "live_arena", "losses"))
                    ),
                ),
            ],
        ),
        _daily_series_spec(
            "Faction Wars",
            [
                (
                    "wins",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "faction_wars", "wins"))
                    ),
                ),
                (
                    "losses",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "faction_wars", "losses"))
                    ),
                ),
                (
                    "progress factions",
                    lambda entry: (
                        len(progress)
                        if isinstance(
                            progress := _nested_lookup(
                                entry, ("summary", "faction_wars", "progress_mode_factions")
                            ),
                            list,
                        )
                        else 0
                    ),
                ),
            ],
        ),
        _daily_series_spec(
            "Iron Twins",
            [
                (
                    "wins",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "dungeons", "iron_twins", "wins"))
                    ),
                ),
                (
                    "losses",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "dungeons", "iron_twins", "losses"))
                    ),
                ),
                (
                    "energy",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "dungeons", "iron_twins", "energy_spent"))
                    ),
                ),
                (
                    "keys used",
                    lambda entry: _coerce_number(
                        _nested_lookup(
                            entry, ("summary", "dungeons", "iron_twins", "iron_twins_keys_used")
                        )
                    ),
                ),
            ],
        ),
        _daily_series_spec(
            "Dungeons",
            [
                (
                    "wins",
                    lambda entry: _sum_metric_from_mapping(
                        _nested_lookup(entry, ("summary", "dungeons")) or {},
                        "wins",
                    ),
                ),
                (
                    "losses",
                    lambda entry: _sum_metric_from_mapping(
                        _nested_lookup(entry, ("summary", "dungeons")) or {},
                        "losses",
                    ),
                ),
                (
                    "energy",
                    lambda entry: _sum_metric_from_mapping(
                        _nested_lookup(entry, ("summary", "dungeons")) or {},
                        "energy_spent",
                    ),
                ),
            ],
        ),
        _daily_series_spec(
            "Daily Shogun",
            [
                (
                    "wins",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "dungeons", "daily_shogun", "wins"))
                    ),
                ),
                (
                    "losses",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "dungeons", "daily_shogun", "losses"))
                    ),
                ),
                (
                    "energy",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "dungeons", "daily_shogun", "energy_spent"))
                    ),
                ),
            ],
        ),
        _daily_series_spec(
            "Cursed City",
            [
                (
                    "keys used",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "keys", "cursed_city", "used_keys"))
                    ),
                ),
            ],
        ),
        _daily_series_spec(
            "Grim Forest",
            [
                (
                    "keys used",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "keys", "grim_forest", "used_keys"))
                    ),
                ),
            ],
        ),
        _daily_series_spec(
            "Doom Tower",
            [
                (
                    "silver keys",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "keys", "doom_tower", "silver_keys_used"))
                    ),
                ),
                (
                    "gold keys",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "keys", "doom_tower", "gold_keys_used"))
                    ),
                ),
            ],
        ),
        _daily_series_spec(
            "Market",
            [
                (
                    "mystery shards",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "market", "mystery_shards_bought"))
                    ),
                ),
            ],
        ),
        _daily_series_spec(
            "Guardian Ring",
            [
                (
                    "successful entries",
                    lambda entry: _coerce_number(
                        _nested_lookup(entry, ("summary", "guardian_ring", "successful_entries"))
                    ),
                ),
            ],
        ),
    ][:max_plots]

    if not output_path:
        output_path = Path(log_path).with_name("raidbot_daily_stats.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        if hasattr(matplotlib, "use"):
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        if not all(hasattr(plt, attr) for attr in ("subplots", "close")):
            return _write_placeholder_png(output_path)
    except Exception:
        return _write_placeholder_png(output_path)

    plot_count = len(plot_specs)
    cols = min(4, max(1, plot_count))
    rows = ceil(plot_count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.6 * rows), squeeze=False)
    fig.suptitle("RaidBot Daily Statistics (UTC)", fontsize=16, fontweight="bold")

    for index, spec in enumerate(plot_specs):
        ax = axes[index // cols][index % cols]
        ax.set_title(spec["title"])
        for label, extractor in spec["series"]:
            series = _build_cumulative_series(day_entries, extractor)
            x_values = list(range(len(series)))
            ax.plot(x_values, series, marker="o", linewidth=2, label=label)
        ax.set_xlabel("Day order")
        ax.set_ylabel("Cumulative")
        ax.grid(True, alpha=0.25)
        if len(spec["series"]) > 1:
            ax.legend(fontsize=8, loc="best")

        if ordered_day_keys:
            tick_positions = [0, len(ordered_day_keys)]
            tick_labels = ["origin", ordered_day_keys[-1].replace("_", "-")]
            if len(ordered_day_keys) > 1:
                middle_index = len(ordered_day_keys) // 2
                tick_positions.insert(1, middle_index)
                tick_labels.insert(1, ordered_day_keys[middle_index].replace("_", "-"))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=25, ha="right")
            ax.set_xlim(0, len(ordered_day_keys))
        else:
            ax.set_xticks([0])
            ax.set_xticklabels(["origin"])
            ax.set_xlim(0, 1)

    for index in range(plot_count, rows * cols):
        axes[index // cols][index % cols].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
