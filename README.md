# RSL Automation Bot

Windows automation runtime for Raid: Shadow Legends. The bot orchestrates multiple game-mode routines, uses OCR and image matching for navigation, supports Discord remote commands, and includes recovery helpers for common runtime failures.

This is a personal automation project. Use at your own risk.

## Main Features

- Mainframe loop that runs enabled game modes in sequence.
- OCR-driven navigation and runtime status reporting.
- Discord command handling for status, parameter updates, pausing, resuming, and restart requests.
- Crash handling with screenshot capture and restart handoff.
- Mode modules for Arena, Dungeons, Faction Wars, Demon Lord, Hydra, Chimera, Cursed City, Grim Forest, and Doom Tower.
- Utility scripts for model training, detector tuning, map labeling, manual run capture, benchmarks, and updates.

## Repository Structure

```text
.
|-- src/raid_bot/          # Application package and runtime modules
|-- tests/                 # Unit tests
|-- scripts/               # Operational, developer, and benchmark scripts
|-- data/                  # Profiles, config, assets, models, datasets, and local output
|-- Raid_Bot.py            # Main runtime entry point
|-- run_bot.py             # Compatibility launcher for the direct runner
|-- Raid_Bot.bat           # Windows Conda launcher
|-- pyproject.toml         # Test/tool configuration
`-- README.md
```

Generated runtime output such as `data/output/`, `data/tmp/`, and logs is ignored by Git.

## Requirements

- Windows desktop environment.
- Anaconda or Miniconda.
- Raid: Shadow Legends running in windowed mode.
- Game resolution exactly `1280 x 1024`.
- Game language set to Spanish.
- Raid window title containing `Raid: Shadow Legends`.

## Installation

The normal setup path is handled by `Raid_Bot.bat`.

1. Install Anaconda or Miniconda.
2. Start Raid and log into the game account.
3. Run `Raid_Bot.bat`.
4. If Conda environment `RaidEnv` is missing, the launcher creates it from `data/config/env.yml`.
5. Later launches reuse the existing environment.

## Running

Normal Windows launcher:

```bat
Raid_Bot.bat
```

Direct mainframe launcher and canonical Python entry point:

```bash
python Raid_Bot.py
```

Use `Raid_Bot.py` for normal runs and restart handoffs. It owns the mainframe
startup flow, Raid window launch/wait behavior, duplicate-process/PID
protection, and restart replacement logic.

If `Raid_Bot.py` is started outside the `RaidEnv` Conda environment, it
relaunches itself through Conda and continues from the same entry point.

Legacy direct runner:

```bash
python run_bot.py
```

Package runner, useful during development:

```bash
python -m raid_bot
```

## Configuration

Runtime profiles are stored in `data/profiles`.

- Default account: `data/profiles/artus_params_mainframe.txt`
- Additional accounts: `data/profiles/{account_name}_params_mainframe.txt`
- Runtime account selector: `RAID_ACCOUNT_NAME`

Profile files use `key = value` lines. Values are parsed as Python literals where possible, so booleans, numbers, lists, and dictionaries can be written naturally.

Sensitive local values are not committed. The Discord token is read from `.ssh` in the project root:

```text
DISCORD_TOKEN=your_discord_bot_token
```

## Mode Sequence

The mainframe checks enabled modes in this order:

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

After a mode pass, the bot collects configured quest and timed rewards before starting the next cycle.

## Useful Scripts

```bash
python scripts/run_error_handler.py
python scripts/configure_manual_player.py
python scripts/networks.py
python scripts/tune_grimforest_adam.py
python scripts/tune_map_mode_detector.py
python scripts/updater.py
python scripts/benchmarks/multi_account_startup.py
```

Detector AI helpers live in `src/raid_bot/detector_ai`. Required detector datasets remain in `data/detector_ai`. Benchmark helpers live in `scripts/benchmarks`.

## Development

Run tests:

```bash
python -m pytest
```

Compile-check source and scripts:

```bash
python -m compileall src scripts Raid_Bot.py run_bot.py
```

The project uses a `src` layout. `pyproject.toml` configures pytest to include `src` on `PYTHONPATH`.

## Runtime Artifacts

Ignored local/generated paths include:

- `data/output/`
- `data/tmp/`
- `data/logs/`
- Python caches and tool caches

Some generated artifacts were already tracked before this cleanup. They should be removed from Git tracking with `git rm --cached` commands after review, while leaving local files on disk if you still need them.

## Known Constraints

- OCR and image matching are probabilistic.
- Manual mouse or keyboard use can interfere while automation is running.
- Resolution and language assumptions are strict.
- This project is for personal experimental use and does not provide account-safety guarantees.
