"""Profile-driven configuration for manual Hydra turns.

The old TXT file has been migrated to this Python structure to keep manual-run
profiles easy to read, edit, and import from code.
"""

MANUAL_PLAY_DATABASE = {
    "defaults": {
        "manual_difficulties": ["Hard"],
        "profile_name_for_manual_difficulty": "Hydra_InfinityTeam",
        "pixel_tolerance": 2,
        "identify_neighbor_rings": 2,
        "click_delay_seconds": 1,
        "auto_battle_button": [0.026, 0.899, 0.058, 0.070],
        "action_buttons": {
            "A1": [0.697, 0.92, 0.01, 0.01],
            "A2": [0.81, 0.92, 0.01, 0.01],
            "A3": [0.926, 0.92, 0.01, 0.01],
        },
        "targets": {
            "Head_1": [0.125, 0.315, 0.130, 0.220],
            "Head_2": [0.668, 0.768, 0.01, 0.01],
            "Head_3": [0.505, 0.315, 0.130, 0.220],
            "Head_4": [0.695, 0.315, 0.130, 0.220],
            "Head_5": [0.220, 0.560, 0.130, 0.220],
            "Head_6": [0.600, 0.560, 0.130, 0.220],
        },
    },
    "profiles": {
        "Hydra_InfinityTeam": {
            "description": (
                "Manual Hydra profile converted from the TXT template. "
                "Replace placeholder pixel colors/coords with calibrated values."
            ),
            "manual_difficulties": ["Normal"],
            "turn_order": ["Lamasu", "Name2", "Name3", "Name4", "Name5", "Name6"],
            "champions": {
                "Lamasu": {
                    "identify_pixel": {"x": 0.586, "y": 0.897, "rgb": [203, 255, 255]},
                    "priority": [ "A2", "A3", "A1"],
                    "priority_switch_once": {
                        "after_base_turns": 3,
                        "swapped_priority": ["A3", "A2", "A1"],
                    },
                    "targets": {"A1": "Auto", "A2": "Head_2", "A3": "Auto"},
                },
                "Name2": {
                    "identify_pixel": {"x": 0.250, "y": 0.925, "rgb": [255, 255, 255]},
                    "priority": ["A3", "A2", "A1"],
                    "targets": {"A1": "Auto", "A2": "Auto", "A3": "Head_2"},
                },
                "Name3": {
                    "identify_pixel": {"x": 0.375, "y": 0.925, "rgb": [255, 255, 255]},
                    "priority": ["A3", "A2", "A1"],
                    "targets": {"A1": "Auto", "A2": "Auto", "A3": "Head_3"},
                },
                "Name4": {
                    "identify_pixel": {"x": 0.500, "y": 0.925, "rgb": [255, 255, 255]},
                    "priority": ["A3", "A2", "A1"],
                    "targets": {"A1": "Auto", "A2": "Auto", "A3": "Head_4"},
                },
                "Name5": {
                    "identify_pixel": {"x": 0.625, "y": 0.925, "rgb": [255, 255, 255]},
                    "priority": ["A3", "A2", "A1"],
                    "targets": {"A1": "Auto", "A2": "Auto", "A3": "Head_5"},
                },
                "Name6": {
                    "identify_pixel": {"x": 0.750, "y": 0.925, "rgb": [255, 255, 255]},
                    "priority": ["A3", "A2", "A1"],
                    "targets": {"A1": "Auto", "A2": "Auto", "A3": "Head_6"},
                },
            },
        }
    },
}
