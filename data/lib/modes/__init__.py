"""Mode package exports with lazy module loading."""

from importlib import import_module

__all__ = [
    "arena_tools",
    "chimera_tools",
    "cursedcity_tools",
    "demonlord_tools",
    "doomtower_tools",
    "dungeon_tools",
    "factionwars_tools",
    "grimforest_tools",
    "hydra_tools",
]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
