import os
import importlib

# Get the folder path of this __init__.py
folder = os.path.dirname(__file__)

# Automatically import all Python files in this folder (except __init__.py)
for filename in os.listdir(folder):
    if filename.endswith(".py") and filename != "__init__.py":
        modulename = filename[:-3]  # strip '.py'
        globals()[modulename] = importlib.import_module(f".{modulename}", package=__name__)

# Optional: define __all__ to control what import * does
__all__ = [name for name in globals() if not name.startswith("_")]