"""Utility package exports with lazy module loading."""

from importlib import import_module

__all__ = ["blue_stage_detector", "cyan_stage_detector", "file_tools", "image_tools", "map_tools", "window_tools"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
