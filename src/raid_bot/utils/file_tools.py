# -*- coding: utf-8 -*-
"""File and configuration helpers used across the bot runtime."""

from __future__ import annotations

import ast
import logging
import shutil
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PARAM_GROUP_PREFIXES = (
    "run",
    "classic_arena",
    "tagteam_arena",
    "live_arena",
    "cursed_city",
    "dungeons",
    "daily_tasks",
    "faction_wars",
    "demon_lord",
    "hydra",
    "grim_forest",
    "chimera",
    "doom_tower",
)

DEFAULT_MAIN_ACCOUNT_NAME = "artus"
DEFAULT_SECONDARY_ACCOUNT_NAMES = (
    "artus2",
    "artus3",
    "artus4",
    "artus5",
    "artus6",
    "artus7",
)
PARAMS_FILE_STEM = "params_mainframe"
LEGACY_PARAM_PATH = Path("data") / "params_mainframe.txt"
PROFILES_DIR = Path("data") / "profiles"
DAILY_LOGS_DIR = Path("data") / "logs" / "daily"
DEFAULT_PARAMS_EXTENSION = ".txt"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParameterUpdate:
    key: str
    old_value: Any
    new_value: Any
    changed: bool
    persisted: bool


@dataclass(frozen=True)
class ProfileParamsResolution:
    account_name: str
    selected_profile_account_name: str
    selected_param_file: Path
    main_profile_file: Path
    legacy_param_file: Path
    migrated_legacy: bool
    used_legacy_fallback: bool
    used_main_profile_fallback: bool
    created_profiles_directory: bool
    generated_secondary_profiles: tuple[Path, ...]
    missing_profile_files: tuple[Path, ...]


def apply_global_drift(search_areas: dict, drift: list):
    """Add a global drift ``[dx, dy, dw, dh]`` to all search areas."""
    for key, values in search_areas.items():
        search_areas[key] = [values[i] + drift[i] for i in range(4)]


def _parse_param_line(line: str) -> tuple[str, str] | None:
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    return key, value


def _deserialize_value(raw_value: str) -> Any:
    try:
        return ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return raw_value


def serialize_param_value(value: Any) -> str:
    """Convert a Python value to a stable params-file representation."""
    return repr(value)


def normalize_account_name(
    account_name: str | None,
    default_account_name: str = DEFAULT_MAIN_ACCOUNT_NAME,
) -> str:
    if account_name is None:
        return default_account_name
    normalized = str(account_name).strip().lower()
    return normalized or default_account_name


def _detect_params_extension() -> str:
    if LEGACY_PARAM_PATH.exists():
        return LEGACY_PARAM_PATH.suffix

    if PROFILES_DIR.exists():
        for path in sorted(PROFILES_DIR.glob(f"*_{PARAMS_FILE_STEM}.*")):
            return path.suffix
        for path in sorted(PROFILES_DIR.glob(f"*_{PARAMS_FILE_STEM}")):
            return path.suffix

    return DEFAULT_PARAMS_EXTENSION


def _build_profile_param_path(
    account_name: str,
    extension: str,
) -> Path:
    return PROFILES_DIR / f"{account_name}_{PARAMS_FILE_STEM}{extension}"


def get_daily_log_path(log_date: datetime | None = None, log_dir: str | Path = DAILY_LOGS_DIR) -> Path:
    """Return the shared daily log file for the given date."""
    current_date = log_date or datetime.now()
    daily_dir = Path(log_dir)
    daily_dir.mkdir(parents=True, exist_ok=True)
    return daily_dir / f"{current_date:%Y_%m_%d}.log"


def ensure_daily_log_header(log_path: str | Path, header_lines: list[str]) -> bool:
    """Create the daily log with an initial header if it does not yet exist."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    path.write_text("\n".join(header_lines).rstrip() + "\n", encoding="utf-8")
    return True


def append_daily_log_lines(log_path: str | Path, lines: list[str] | tuple[str, ...] | str) -> None:
    """Append one or more lines to the shared daily log."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(lines, str):
        payload = [lines]
    else:
        payload = list(lines)
    with path.open("a", encoding="utf-8") as handle:
        for line in payload:
            handle.write(f"{str(line).rstrip()}\n")


def _update_params_file_with_overrides(
    param_file: str | Path,
    overrides: dict[str, Any],
    create_missing: bool = True,
) -> bool:
    param_path = Path(param_file)
    if not param_path.exists():
        return False

    lines = param_path.read_text(encoding="utf-8").splitlines(keepends=True)
    remaining_overrides = dict(overrides)
    updated_lines: list[str] = []
    changed = False

    for line in lines:
        stripped = line.strip()
        parsed = _parse_param_line(stripped) if stripped and not stripped.startswith("#") else None
        if parsed and parsed[0] in remaining_overrides:
            key = parsed[0]
            replacement = f"{key} = {serialize_param_value(remaining_overrides.pop(key))}\n"
            if line != replacement:
                changed = True
            updated_lines.append(replacement)
            continue
        updated_lines.append(line)

    if create_missing and remaining_overrides:
        if updated_lines and not updated_lines[-1].endswith("\n"):
            updated_lines[-1] = updated_lines[-1] + "\n"
        if updated_lines and updated_lines[-1].strip():
            updated_lines.append("\n")
        for key, value in remaining_overrides.items():
            updated_lines.append(f"{key} = {serialize_param_value(value)}\n")
        changed = True

    if changed:
        param_path.write_text("".join(updated_lines), encoding="utf-8")

    return changed


def _build_secondary_profile_overrides() -> dict[str, Any]:
    return {
        "run_classic_arena": True,
        "run_tagteam_arena": True,
        "run_live_arena": False,
        "run_dungeons": True,
        "run_factionwars": False,
        "run_demonlord": False,
        "run_hydra": False,
        "run_chimera": False,
        "run_doomtower": False,
        "run_cursedcity": False,
        "run_grimforest": False,
        "run_effective_unit_leveling": False,
        "classic_arena_update_dataset": False,
        "classic_arena_pushrank": False,
        "tagteam_arena_update_dataset": False,
        "tagteam_arena_pushrank": False,
        "dungeons_iron_twins_priority": False,
        "dungeons_defaults_available": ["dragon"],
        "dungeons_difficulty": "normal",
        "dungeons_fusion_difficulty": "normal",
        "dungeons_dungeon": "dragon",
        "dungeons_level": 11,
        "dungeons_eventdungeon_level": 29,
        "dungeons_build_name": "Fusion",
        "dungeons_disable_fusion_override": True,
    }


def resolve_profile_params_file(
    account_name: str | None = None,
    main_account_name: str = DEFAULT_MAIN_ACCOUNT_NAME,
    secondary_account_names: tuple[str, ...] = DEFAULT_SECONDARY_ACCOUNT_NAMES,
    generate_secondary_profiles: bool = True,
    allow_main_profile_fallback_for_missing_account: bool = True,
) -> ProfileParamsResolution:
    normalized_main_account = normalize_account_name(main_account_name, DEFAULT_MAIN_ACCOUNT_NAME)
    normalized_account = normalize_account_name(account_name, normalized_main_account)
    normalized_secondaries = tuple(
        normalize_account_name(name, normalized_main_account)
        for name in secondary_account_names
    )
    extension = _detect_params_extension()

    created_profiles_directory = False
    if not PROFILES_DIR.exists():
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        created_profiles_directory = True

    main_profile_file = _build_profile_param_path(normalized_main_account, extension)
    migrated_legacy = False
    if not main_profile_file.exists() and LEGACY_PARAM_PATH.exists():
        try:
            shutil.move(str(LEGACY_PARAM_PATH), str(main_profile_file))
            migrated_legacy = True
        except OSError as exc:
            logger.warning(
                "Failed migrating legacy params file from '%s' to '%s': %s",
                LEGACY_PARAM_PATH,
                main_profile_file,
                exc,
            )

    generated_secondary_profiles: list[Path] = []
    if generate_secondary_profiles and main_profile_file.exists():
        overrides = _build_secondary_profile_overrides()
        for secondary_account in normalized_secondaries:
            secondary_path = _build_profile_param_path(secondary_account, extension)
            if not secondary_path.exists():
                shutil.copy2(main_profile_file, secondary_path)
                _update_params_file_with_overrides(
                    secondary_path,
                    overrides,
                    create_missing=True,
                )
                generated_secondary_profiles.append(secondary_path)

    selected_profile_account_name = normalized_account
    selected_param_file = _build_profile_param_path(selected_profile_account_name, extension)
    missing_profile_files: list[Path] = []
    used_legacy_fallback = False
    used_main_profile_fallback = False

    if not selected_param_file.exists():
        missing_profile_files.append(selected_param_file)
        if selected_profile_account_name == normalized_main_account and LEGACY_PARAM_PATH.exists():
            selected_param_file = LEGACY_PARAM_PATH
            used_legacy_fallback = True
        elif main_profile_file.exists() and allow_main_profile_fallback_for_missing_account:
            selected_profile_account_name = normalized_main_account
            selected_param_file = main_profile_file
            used_main_profile_fallback = True

    if not selected_param_file.exists():
        missing_profile_files.append(main_profile_file)
        raise FileNotFoundError(
            "No usable params file found. "
            f"Tried profile '{_build_profile_param_path(normalized_account, extension)}', "
            f"main profile '{main_profile_file}', and legacy '{LEGACY_PARAM_PATH}'."
        )

    return ProfileParamsResolution(
        account_name=normalized_account,
        selected_profile_account_name=selected_profile_account_name,
        selected_param_file=selected_param_file,
        main_profile_file=main_profile_file,
        legacy_param_file=LEGACY_PARAM_PATH,
        migrated_legacy=migrated_legacy,
        used_legacy_fallback=used_legacy_fallback,
        used_main_profile_fallback=used_main_profile_fallback,
        created_profiles_directory=created_profiles_directory,
        generated_secondary_profiles=tuple(generated_secondary_profiles),
        missing_profile_files=tuple(missing_profile_files),
    )


def read_params(param_file: str | Path) -> dict[str, Any]:
    """Read params from ``key = value`` formatted text file."""
    params: dict[str, Any] = {}
    with open(param_file, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parsed = _parse_param_line(stripped)
            if not parsed:
                continue
            key, raw_value = parsed
            params[key] = _deserialize_value(raw_value)
    return params


def group_params(
    params: dict[str, Any],
    min_shared_keys: int = 3,
    preferred_prefixes: tuple[str, ...] = DEFAULT_PARAM_GROUP_PREFIXES,
) -> dict[str, dict[str, Any]]:
    """
    Group flat params by inferred common prefix.

    Unknown/non-grouped keys are placed under ``mainframe``.
    """
    prefix_counts: dict[str, int] = defaultdict(int)

    for key in params:
        parts = key.split("_")
        for idx in range(1, len(parts)):
            prefix_counts["_".join(parts[:idx]) + "_"] += 1

    valid_prefixes = {
        prefix
        for prefix, count in prefix_counts.items()
        if count >= min_shared_keys
    }

    for prefix in preferred_prefixes:
        valid_prefixes.add(f"{prefix}_")

    ordered_prefixes = sorted(valid_prefixes, key=len, reverse=True)
    grouped: dict[str, dict[str, Any]] = {"mainframe": {}}

    for key, value in params.items():
        for prefix in ordered_prefixes:
            if key.startswith(prefix):
                group_name = prefix.rstrip("_")
                grouped.setdefault(group_name, {})[key[len(prefix):]] = value
                break
        else:
            grouped["mainframe"][key] = value

    return grouped


def normalize_param_key(key: str) -> str:
    """Accept dotted keys in commands (``group.key`` -> ``group_key``)."""
    return key.strip().replace(".", "_")


def coerce_value(raw_value: str, reference: Any) -> Any:
    """Parse string input into a value compatible with ``reference`` type."""
    if isinstance(reference, bool):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError("Expected boolean (true/false, on/off, 1/0).")

    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(raw_value)

    if isinstance(reference, float):
        return float(raw_value)

    if isinstance(reference, str):
        trimmed = raw_value.strip()
        if (trimmed.startswith("'") and trimmed.endswith("'")) or (
            trimmed.startswith('"') and trimmed.endswith('"')
        ):
            return _deserialize_value(trimmed)
        return raw_value

    parsed = _deserialize_value(raw_value)
    if reference is None:
        return parsed

    if isinstance(reference, (list, tuple, dict)) and not isinstance(parsed, type(reference)):
        raise ValueError(f"Expected value of type {type(reference).__name__}.")

    return parsed


def update_param_file_value(
    param_file: str | Path,
    key: str,
    value: Any,
    create_if_missing: bool = False,
) -> bool:
    """
    Update a single key in the params file while preserving the rest of the file.

    Returns True when the key was written/updated.
    """
    param_path = Path(param_file)
    lines = param_path.read_text(encoding="utf-8").splitlines(keepends=True)
    serialized_value = serialize_param_value(value)
    target_line = f"{key} = {serialized_value}\n"

    written = False
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        parsed = _parse_param_line(stripped) if stripped and not stripped.startswith("#") else None
        if parsed and parsed[0] == key:
            new_lines.append(target_line)
            written = True
            continue
        new_lines.append(line)

    if not written and create_if_missing:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] = new_lines[-1] + "\n"
        new_lines.append(target_line)
        written = True

    if written:
        param_path.write_text("".join(new_lines), encoding="utf-8")

    return written


class ParameterStore:
    """In-memory + file-backed parameter store."""

    def __init__(self, param_file: str | Path):
        self.param_file = Path(param_file)
        self.params: dict[str, Any] = {}
        self.grouped_params: dict[str, dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        self.params = read_params(self.param_file)
        self.grouped_params = group_params(self.params)

    def get_flat_copy(self) -> dict[str, Any]:
        return deepcopy(self.params)

    def get_grouped_copy(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self.grouped_params)

    def keys(self) -> list[str]:
        return sorted(self.params.keys())

    def has_key(self, key: str) -> bool:
        return key in self.params

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def set(
        self,
        key: str,
        value: Any,
        persist: bool = True,
        create_if_missing: bool = False,
    ) -> ParameterUpdate:
        old_value = self.params.get(key)
        changed = (old_value != value) or (key not in self.params)
        persisted = False

        if persist and changed:
            try:
                persisted = update_param_file_value(
                    self.param_file,
                    key,
                    value,
                    create_if_missing=create_if_missing,
                )
            except OSError as exc:
                raise RuntimeError(f"Failed writing params file: {exc}") from exc

            if not persisted and key in self.params and not create_if_missing:
                raise RuntimeError(f"Parameter '{key}' could not be persisted.")

        if changed:
            self.params[key] = value
            self.grouped_params = group_params(self.params)

        return ParameterUpdate(
            key=key,
            old_value=old_value,
            new_value=value,
            changed=changed,
            persisted=persisted,
        )
