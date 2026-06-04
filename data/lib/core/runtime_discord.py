"""Discord runtime configuration helpers."""

from __future__ import annotations

import os


def load_discord_token(secret_file: str = ".ssh") -> str:
    """Load a Discord token from the supported local secret-file formats."""
    if not os.path.exists(secret_file):
        raise FileNotFoundError(f"Discord token file not found: {secret_file}")

    with open(secret_file, "r", encoding="utf-8") as handle:
        lines = [
            line.strip()
            for line in handle
            if line.strip() and not line.strip().startswith("#")
        ]

    if not lines:
        raise ValueError(f"Discord token file is empty: {secret_file}")

    for line in lines:
        if line.startswith("DISCORD_TOKEN="):
            token = line.split("=", 1)[1].strip()
            if token:
                return token

    return lines[0]
