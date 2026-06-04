# RSL Automation Bot (Mainframe)

[![Python](https://img.shields.io/badge/python-conda%20env-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Personal-use-lightgrey)](https://choosealicense.com/)

Automated multi-mode controller for Raid: Shadow Legends with OCR navigation, runtime mode switching, Discord remote commands, crash recovery, and update tooling.

## What This Project Is

This project is a mainframe-style automation runtime that orchestrates multiple game-mode bots in a loop.

Core responsibilities:
- Load and apply all runtime config from `data/profiles/artus_params_mainframe.txt`
- Navigate game menus with OCR and mouse/keyboard automation
- Run enabled modes in sequence
- Collect quest and timed rewards between cycles
- Handle recoverable errors automatically
- Accept live remote commands (Discord) for status, toggles, and config updates

## Quick Start

1. Set Raid to windowed mode, language Spanish, resolution `1280 x 1024`.
2. Start Raid and log into the game account.
3. Run `Raid_Bot.bat`.
4. On first run, let Conda create environment `RaidEnv` (can take up to ~5 minutes).
5. The mainframe loop starts after the Raid window is available.

## Requirements

- Windows desktop environment
- Anaconda/Conda installed
- Raid running in windowed mode
- Resolution exactly `1280 x 1024` (other resolutions are not supported)
- Game language set to Spanish
- Window title containing `Raid: Shadow Legends`

## Installation

Installation behavior is handled by `Raid_Bot.bat`:

1. Detect Conda and environment `RaidEnv`.
2. If missing, create `RaidEnv` from `data/env.yml`.
3. Activate environment and start `Raid_Bot.py`.

Notes:
- First setup is slower because dependencies are installed.
- Later launches skip setup and start immediately.

## Launch Flows

### Flow A: Normal Batch Flow (Recommended)

`Raid_Bot.bat` -> `Raid_Bot.py` -> mainframe loop

What happens:
1. Startup checks for the Raid window and can launch Plarium Play automatically if needed.
2. `Raid_Bot.py` creates `RSL_Bot_Mainframe` and starts the continuous automation loop.

### Flow B: Alternate Direct Runner

`run_bot.py` creates `RSL_Bot_Mainframe` directly and also supports the restart-helper and PID-file flow:

```bash
conda activate RaidEnv
python run_bot.py
```

### Flow C: Self-Update

Run `python updater.py`.

Updater flow:
1. Finds latest remote branch matching `vX.Y.Z`
2. Clones to temp folder
3. Replaces local project files (excluding protected paths)
4. Restarts bot process

### Flow D: Stop Automation

- If running direct/main terminal: close the process (`Ctrl+C`).
- If using Discord control: send `stop` or `pause` to enter manual mode.

## Main Automation Cycle

Each cycle follows this order:

1. Process pending remote command (if any)
2. Run each enabled mode in configured sequence
3. Send mode heartbeat/status update
4. After all enabled modes: collect quest rewards and timed rewards
5. Repeat

Mode sequence in mainframe:
1. Classic Arena
2. Tag Team Arena
3. Live Arena
4. Dungeons
5. Faction Wars
6. Demon Lord
7. Hydra
8. Chimera
9. Cursed City
10. Grim Forest
11. Doom Tower

## Mode Reference

| Mode | Toggle Key | Runtime Behavior | Typical Stop Condition |
|---|---|---|---|
| Classic Arena | `run_classic_arena` | Evaluates enemies, fights acceptable targets, refreshes list based on timer/settings | No usable coins / cycle done |
| Tag Team Arena | `run_tagteam_arena` | Evaluates team power sets and runs one cycle per mainframe pass | No usable coins / cycle done |
| Live Arena | `run_live_arena` | Checks active window state, claims rewards, auto-picks from preset slots, runs encounter loop | Live Arena inactive or no coins |
| Dungeons | `run_dungeons` | Chooses encounter (Iron Twins priority optional) and farms selected dungeon/stage | Energy too low or selection fails |
| Faction Wars | `run_factionwars` | Detects open crypts, selects configured stage/difficulty per faction | No valid encounter found |
| Demon Lord | `run_demonlord` | Reads keys, detects already-cleared difficulties by player names, runs remaining order | No keys left |
| Hydra | `run_hydra` | Selects difficulty and saved team, fights until score threshold, retries when below threshold | All configured difficulties cleared or retry fail |
| Chimera | `run_chimera` | Similar to Demon Lord flow with score threshold gate | Keys depleted or threshold not reached |
| Cursed City | `run_cursedcity` | Current wrapper runs scaffold demo probe (keys + blue-stage candidate check) | Demo probe completes |
| Grim Forest | `run_grimforest` | Selects difficulty, detects selectable structures, fights stages, handles level rewards, and avoids the most recent defeat location | Keys depleted, candidate/battle failure, or defeat |
| Doom Tower | `run_doomtower` | Rotation-aware boss loop, key tracking, setup selection, debug snapshots | Silver keys depleted, no valid boss tile, or 15-minute mode cap |

Important behavior:
- Quest rewards are collected after mode pass, not as separate run toggle.
- If `run_effective_unit_leveling = True`, dungeon mode is conditionally suppressed in the main loop.

## Configuration Guide

All runtime settings are in:

`data/profiles/artus_params_mainframe.txt`

Per-account profiles follow:
- `data/profiles/{account_name}_params_mainframe.txt`
- runtime account selection env var: `RAID_ACCOUNT_NAME` (defaults to `artus`)

File format:
- `key = value`
- Values are parsed as Python literals where possible (`True`, `123`, `[1,2]`, `{"a":1}`)

Main groups (prefix-based):
- `run_...` mode on/off switches
- `classic_arena_...`
- `tagteam_arena_...`
- `live_arena_...`
- `dungeons_...`
- `faction_wars_...`
- `demon_lord_...`
- `hydra_...`
- `chimera_...`
- `doom_tower_...`
- `cursed_city_...`
- `grim_forest_...`
- mainframe-level keys such as `verbose`, `screen_drift`, `override_timer_minutes`

Do not manually edit:
- `classic_arena_enemies_lost`
- `tagteam_arena_enemies_lost`

These avoid-lists are updated automatically by the arena bots.

## Dungeon Tournament Override Flow

Before dungeon runs, mainframe checks Fastidious calendar for active dungeon tournaments.

If active:
- Effective dungeon is forced to `normal`, level `20`, matching active tournament dungeon.

If not active:
- Uses your configured `dungeons_difficulty`, `dungeons_level`, `dungeons_dungeon`.

## Discord Remote Control

### Setup

Create `.ssh` in project root with token:

```text
DISCORD_TOKEN=your_discord_bot_token
```

Default runtime target is hardcoded to:
- Guild: `Discord_Sandbox`
- Channel: `raid_sandbox`

### Supported Commands

- `help`
- `status`
- `modes`
- `params [filter]`
- `get <parameter_name>`
- `set <parameter_name> <value>`
- `toggle <mode_name> [on|off]`
- `reload` or `reload_config`
- `start` or `resume`
- `stop` or `pause`
- `restart`
- `ping`

### Live Config Flow Example

1. `stop`
2. `params dungeons`
3. `set dungeons_level 25`
4. `toggle hydra off`
5. `reload`
6. `start`

## Error Handling and Recovery Flows

### Connectivity Popup

- Background checker detects internet error popups and clicks retry.

### Remote Override Popup

- Main loop pauses all mode loops.
- Waits `override_timer_minutes`.
- Clicks reconnect area.
- Returns to mode menu and resumes automation.

### Fatal Exception

1. Error is logged
2. Screenshot captured to `data/tmp/raid_error_latest.png`
3. Error + screenshot sent to Discord (if configured)
4. Bot waits for `restart` command
5. Current bot process exits completely
6. Restart helper closes Raid/Plarium, launches Raid again, starts a fresh bot process, and posts a Discord success message

## Map-Scaffold Artifacts and Debug Output

Generated runtime artifacts:
- Doom Tower debug: `debug/doomtower/run_*`
- Map mode debug: `debug/map_modes/<mode>/run_*`
- Persisted map memory: `data/map_data/<mode>/latest.json`
- Last map session (overwritten each run): `data/map_data/<mode>/last_session/map_state.json`

## Optional Utility Scripts

These are not required for normal bot operation:

- `python updater.py`
  - Self-update runtime from latest version branch.

- `python run_error_handler.py`
  - Runs standalone OCR error popup watcher.

- `python networks.py`
  - Trains or analyzes arena evaluation models (developer workflow).

- `web_page_handler.HellhadesScraper`
  - Scrapes champion tier list into `data/database_champions/hellhades_tier_list.csv`.
  - Example run:
    ```bash
    python -c "from data.lib.handlers.web_page_handler import HellhadesScraper; s=HellhadesScraper(headless=True); s.run()"
    ```

## Known Constraints

- OCR is probabilistic; occasional misreads are expected.
- Manual keyboard/mouse usage can interfere during automation.
- Resolution/language assumptions are strict.
- Project is for personal experimental use; no stability or account-safety guarantees.

## Disclaimer

This project is a personal automation experiment.
Use at your own risk.
