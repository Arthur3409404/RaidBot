from __future__ import annotations

import json
import time
import ctypes
from pathlib import Path

import matplotlib.pyplot as plt
import cv2
import tkinter as tk


DEBUG_ROOT = Path("debug")
IMAGE_NAME = "pov_cyan_brighter_binary.png"
META_NAME = "run_meta.json"


def _list_debug_collections(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _list_run_dirs(collection_dir: Path) -> list[Path]:
    runs = []
    for path in sorted(collection_dir.iterdir()):
        if not path.is_dir():
            continue
        if not path.name.startswith("run_"):
            continue
        if (path / IMAGE_NAME).exists():
            runs.append(path)
    return runs


def _prompt_collection(collections: list[Path]) -> Path:
    print("Select debug collection:")
    for i, collection in enumerate(collections, start=1):
        print(f"  {i}. {collection.as_posix()}")

    while True:
        choice = input("Enter number: ").strip()
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue
        idx = int(choice) - 1
        if 0 <= idx < len(collections):
            return collections[idx]
        print("Choice out of range.")


def _ensure_meta(run_dir: Path) -> dict:
    meta_path = run_dir / META_NAME
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "label": {
            "true_object_count": None,
            "notes": "",
        }
    }


def _write_meta(run_dir: Path, meta: dict) -> None:
    meta_path = run_dir / META_NAME
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _show_image(image_path: Path, title: str) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path.as_posix()}")

    plt.figure(figsize=(10, 6))
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show(block=False)
    time.sleep(3.0)
    _focus_console_window()


def _focus_console_window() -> None:
    try:
        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            user32.ShowWindow(hwnd, 5)  # SW_SHOW
            user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


def _prompt_count(existing_value) -> int | None:
    result = {"value": None}

    root = tk.Tk()
    root.title("Label Run")
    root.attributes("-topmost", True)
    root.geometry("+0+0")
    root.resizable(False, False)

    prompt = f"Enter true object count [0-12] (current={existing_value})\nUse 's' to skip or 'q' to quit."
    tk.Label(root, text=prompt, justify="left", padx=10, pady=8).pack()

    entry = tk.Entry(root, width=24)
    entry.pack(padx=10, pady=(0, 8))
    root.lift()
    root.focus_force()
    root.grab_set()
    entry.focus_force()

    def _refocus():
        try:
            root.lift()
            root.focus_force()
            root.grab_set()
            entry.focus_force()
        except Exception:
            pass

    root.after(60, _refocus)
    root.after(180, _refocus)

    error_label = tk.Label(root, text="", fg="red")
    error_label.pack(padx=10, pady=(0, 8))

    def submit():
        value = entry.get().strip().lower()
        if value == "q":
            result["value"] = "quit"
            root.destroy()
            return
        if value == "s":
            result["value"] = None
            root.destroy()
            return
        if value.isdigit() and 0 <= int(value) <= 12:
            result["value"] = int(value)
            root.destroy()
            return
        error_label.config(text="Invalid input. Enter 0..12, 's', or 'q'.")

    def close_window():
        result["value"] = "quit"
        root.destroy()

    tk.Button(root, text="OK", command=submit).pack(pady=(0, 10))
    root.protocol("WM_DELETE_WINDOW", close_window)
    root.bind("<Return>", lambda _event: submit())
    root.mainloop()
    return result["value"]


def main() -> int:
    collections = _list_debug_collections(DEBUG_ROOT)
    if not collections:
        print(f"No debug collections found under: {DEBUG_ROOT.as_posix()}")
        return 1

    selected_collection = _prompt_collection(collections)
    run_dirs = _list_run_dirs(selected_collection)
    if not run_dirs:
        print(f"No runs with {IMAGE_NAME} found in: {selected_collection.as_posix()}")
        return 1

    print(f"Loaded {len(run_dirs)} runs from: {selected_collection.as_posix()}")

    for index, run_dir in enumerate(run_dirs, start=1):
        image_path = run_dir / IMAGE_NAME
        meta = _ensure_meta(run_dir)
        label = meta.setdefault("label", {})
        existing_value = label.get("true_object_count")

        title = f"{run_dir.name} ({index}/{len(run_dirs)})"
        _show_image(image_path, title)

        result = _prompt_count(existing_value)
        plt.close("all")

        if result == "quit":
            print("Stopped by user.")
            return 0
        if result is None:
            print(f"Skipped {run_dir.name}")
            continue

        label["true_object_count"] = int(result)
        _write_meta(run_dir, meta)
        print(f"Saved {run_dir.name}: true_object_count={result}")

    print("Labeling finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
