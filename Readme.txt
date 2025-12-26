README - Arena Bot Project

This is a homemade project, not a professionally developed tool. Please follow the instructions carefully for proper setup and usage.

REQUIREMENTS:
--------------
- Anaconda must be installed for this project to function correctly.
- The Plarium Raid application must be running at a resolution of 1157x792. Other Resolutions might fail or Impact the object detection.
  Note: This resolution can be set by Trail and error as the width and height of the current found window are displayed in the commandwindow. Then the window can be rearranged and the program started again to find a sweet spot.
- Make sure you are on the "Classic Arena" screen where enemy teams are visible.
- The game's language must be set to Spanish. This is essential for accurate object detection.

NOTE:
-----
- Object detection is not perfect. The reader may give false predictions over time.
- Best performance is achieved by avoiding user input while the bot is running.

SETUP:
------
1. Confirm that all requirements above are met.
2. Open the `params` file located in the `data` folder:
   - Set the **threshold** value. The bot will attack teams with power below this value.
   - Configure the **multi-refresh** setting:
     - 0 = Only use the free daily refresh.
     - 1 = One paid refresh in addition to the free refresh.
     - N = N paid refreshes plus the free one.
   - After all allowed refreshes are used, the bot will wait for the next free refresh and repeat.

USAGE:
------
- Ideal for running when the PC is not being actively used.
- Can run during active use, but manual input may interfere with emulated inputs.
- The bot provides an update after each refresh or battle.
- To stop the bot, simply close the command prompt window.

INSTALLATION:
-------------
- Run the provided batch file to install all required dependencies.
- The initial setup may take approximately 3 minutes.
- After the first run, the bot will skip dependency installation and start immediately.

BATTLE STRATEGY:
----------------
- The bot attacks all opponents with team power below the configured threshold.
- If it loses a battle, it records the enemy team’s power and avoids attacking them again.
- Losses are expected in the beginning while the bot builds its avoidance list.
- The list of avoided enemies is stored in the `params` file. Do **not** edit this manually.
- You may clear the list once a month if too many teams have been recorded.
  (Keeping the list under 50–100 entries is recommended.)

DISCLAIMER:
-----------
This bot is intended for personal use only.
No guarantees are made regarding its accuracy, stability, or impact on your game account.
Use at your own risk.