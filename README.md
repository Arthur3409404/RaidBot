# RSL Automation Bot (Mainframe)

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-Personal-use-lightgrey)](https://choosealicense.com/)

> Automated multi-mode bot for *Raid: Shadow Legends*.  
> Personal project for automating Classic Arena, Tag Team Arena, Live Arena, Dungeons, Faction Wars, Clan Boss, Doom Tower, and quest rewards.

---

## Overview

This bot acts as a **central mainframe controller** for *Raid: Shadow Legends*. It performs **unattended automation** across multiple game modes:

- Classic Arena  
- Tag Team Arena  
- Live Arena  
- Dungeons  
- Faction Wars  
- Clan Boss (Demon Lord)  
- Doom Tower  
- Quest & time-gated reward collection  

It uses **OCR-based detection** and **screen automation** to navigate the UI, manage enemies, refresh targets, and handle errors such as remote overrides automatically.

---

## Requirements

- **Anaconda** (Python environment)  
- *Raid: Shadow Legends* running in **windowed mode**  
- **Screen resolution:** 1280 × 1024  
  ⚠️ Other resolutions are **not supported**  
- Game **language must be set to Spanish**  
- Game window title must contain: `Raid: Shadow Legends`  
- Start the bot **while logged in** and on the correct screen

---

## Installation

1. Run the provided batch file to install all dependencies.  
2. Initial setup may take approximately **3 minutes**.  
3. Subsequent runs skip dependency installation and start immediately.

---

## Configuration

All bot settings are stored in:

data/params_mainframe.txt

### Mainframe settings

- Logging verbosity  
- Screen drift adjustments  
- Remote override cooldown timer  

### Run toggles

Enable or disable specific modes:

- `classic_arena`  
- `tagteam_arena`  
- `live_arena`  
- `dungeons`  
- `factionwars`  
- `demonlord`  
- `doomtower`  

### Mode-specific settings

- Power thresholds for arena attacks  
- Refresh intervals for enemies  
- Dungeon targeting rules  
- Doom Tower rotation detection  
- Avoidance lists for failed encounters  

> ⚠️ **Do not manually edit avoidance lists**; the bot manages them automatically.

---

## Features

- **Automated multi-mode gameplay:** runs arenas, dungeons, faction wars, clan bosses, doom tower, and quests in a loop  
- **OCR-based menu navigation** with error handling  
- **Automatic enemy refresh** based on configurable intervals  
- **Quest reward collection** including daily, weekly, monthly, and advanced quests  
- **Guardian Ring handling** with faction character selection  
- **Remote override recovery:** detects connection or override issues, waits, retries, and continues automation  

---

## Usage

Run the bot using the GUI by executing RaidBot.bat

- Recommended when PC is idle for best performance  
- Can run alongside user activity, but manual input may interfere  
- To stop the bot, simply close the command prompt window  

---

## Notes

- OCR detection is not perfect; occasional misreads are expected  
- Losses in arena or other modes may occur during initial runs  
- The bot is intended for personal use only  
- No guarantees regarding accuracy, stability, or account safety  

---

## Disclaimer

This bot is a personal, experimental project.  
Use at your own risk. The developer is not responsible for any game account issues.