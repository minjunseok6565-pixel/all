from __future__ import annotations

from typing import Any, Dict

from state import (
    get_cached_playoff_news_snapshot,
    get_cached_weekly_news_snapshot,
    set_cached_playoff_news_snapshot,
    set_cached_weekly_news_snapshot,
)


def get_weekly_cache() -> Dict[str, Any]:
    cache = get_cached_weekly_news_snapshot() or {}
    if not isinstance(cache, dict):
        cache = {}
    # Respect state_schema: must have exact keys
    cache.setdefault("last_generated_week_start", None)
    cache.setdefault("items", [])
    # Do not add extra top-level keys here.
    return cache


def set_weekly_cache(cache: Dict[str, Any]) -> None:
    out = {
        "last_generated_week_start": cache.get("last_generated_week_start"),
        "items": cache.get("items") or [],
    }
    set_cached_weekly_news_snapshot(out)


def get_playoff_cache() -> Dict[str, Any]:
    cache = get_cached_playoff_news_snapshot() or {}
    if not isinstance(cache, dict):
        cache = {}
    cache.setdefault("series_game_counts", {})
    cache.setdefault("items", [])
    return cache


def set_playoff_cache(cache: Dict[str, Any]) -> None:
    out = {
        "series_game_counts": cache.get("series_game_counts") or {},
        "items": cache.get("items") or [],
    }
    set_cached_playoff_news_snapshot(out)
