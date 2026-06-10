# -*- coding: utf-8 -*-
"""Fastidious raid calendar helpers for dungeon tournament overrides."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
import re

import requests
from bs4 import BeautifulSoup


FASTIDIOUS_CALENDAR_URL = "https://fastidious.gg/raid/calendar"

_DUNGEON_TOURNAMENTS = {
    "fire knight tournament": "fire_knight",
    "ice golem tournament": "ice_golem",
    "dragon tournament": "dragon",
    "spider tournament": "spider",
}

_DATE_RANGE_PATTERN = re.compile(
    (
        r"(?P<start_day>\d{1,2})\s*/\s*(?P<start_month>\d{1,2})"
        r".*?"
        r"(?P<end_day>\d{1,2})\s*/\s*(?P<end_month>\d{1,2})"
    ),
    re.DOTALL,
)
_YEAR_PATTERN = re.compile(r"\b(20\d{2})-\d{2}-\d{2}\b")


@dataclass(frozen=True)
class DungeonTournamentEvent:
    name: str
    dungeon: str
    start_date: date
    end_date: date


def _extract_reference_years(html: str) -> list[int]:
    years = [int(match.group(1)) for match in _YEAR_PATTERN.finditer(html)]
    if not years:
        return []
    counts = Counter(years)
    return [year for year, _ in counts.most_common()]


def _resolve_date_range(
    date_text: str,
    *,
    today: date,
    reference_years: list[int],
) -> tuple[date, date]:
    match = _DATE_RANGE_PATTERN.search(date_text)
    if not match:
        raise ValueError(f"Unrecognized tournament date range: {date_text!r}")

    start_day = int(match.group("start_day"))
    start_month = int(match.group("start_month"))
    end_day = int(match.group("end_day"))
    end_month = int(match.group("end_month"))

    seed_years = reference_years or [today.year]
    candidate_start_years: list[int] = []
    seen_years: set[int] = set()

    for seed in seed_years:
        for candidate in (seed - 1, seed, seed + 1):
            if candidate in seen_years:
                continue
            seen_years.add(candidate)
            candidate_start_years.append(candidate)

    best_pair: tuple[date, date] | None = None
    best_distance: int | None = None

    for start_year in candidate_start_years:
        try:
            start_value = date(start_year, start_month, start_day)
            end_year = start_year + 1 if (end_month, end_day) < (start_month, start_day) else start_year
            end_value = date(end_year, end_month, end_day)
        except ValueError:
            continue

        if start_value <= today <= end_value:
            distance = 0
        elif today < start_value:
            distance = (start_value - today).days
        else:
            distance = (today - end_value).days

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_pair = (start_value, end_value)

    if best_pair is None:
        raise ValueError(f"Could not resolve date range from text: {date_text!r}")

    return best_pair


def fetch_dungeon_tournament_events(
    *,
    today: date | None = None,
    url: str = FASTIDIOUS_CALENDAR_URL,
    timeout_seconds: float = 8.0,
) -> list[DungeonTournamentEvent]:
    if today is None:
        today = date.today()

    response = requests.get(
        url,
        timeout=timeout_seconds,
        headers={"User-Agent": "RaidBot/1.0"},
    )
    response.raise_for_status()

    html = response.text
    reference_years = _extract_reference_years(html)
    soup = BeautifulSoup(html, "html.parser")

    events: list[DungeonTournamentEvent] = []
    for card in soup.select('div[wire\\:click^="changeEventStatus("]'):
        name_node = card.select_one("div.text-sm.font-medium")
        date_node = card.select_one("div.text-xs.opacity-75")
        if not name_node or not date_node:
            continue

        event_name = " ".join(name_node.get_text(" ", strip=True).split())
        dungeon_name = _DUNGEON_TOURNAMENTS.get(event_name.lower())
        if not dungeon_name:
            continue

        date_text = date_node.get_text(" ", strip=True)
        try:
            start_date, end_date = _resolve_date_range(
                date_text,
                today=today,
                reference_years=reference_years,
            )
        except ValueError:
            continue

        events.append(
            DungeonTournamentEvent(
                name=event_name,
                dungeon=dungeon_name,
                start_date=start_date,
                end_date=end_date,
            )
        )

    return events


def get_active_dungeon_tournament(
    *,
    today: date | None = None,
    url: str = FASTIDIOUS_CALENDAR_URL,
    timeout_seconds: float = 8.0,
) -> DungeonTournamentEvent | None:
    if today is None:
        today = date.today()

    events = fetch_dungeon_tournament_events(
        today=today,
        url=url,
        timeout_seconds=timeout_seconds,
    )
    active = [event for event in events if event.start_date <= today <= event.end_date]
    if not active:
        return None

    active.sort(key=lambda event: (event.start_date, event.end_date), reverse=True)
    return active[0]


__all__ = [
    "DungeonTournamentEvent",
    "FASTIDIOUS_CALENDAR_URL",
    "fetch_dungeon_tournament_events",
    "get_active_dungeon_tournament",
]
