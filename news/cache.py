from __future__ import annotations

from typing import Any, Dict

from state import (
    get_cached_playoff_news_snapshot,
    get_cached_weekly_news_snapshot,
    set_cached_playoff_news_snapshot,
    set_cached_weekly_news_snapshot,
)

# Cache generator versions (used for invalidation when logic changes)
WEEKLY_GENERATOR_VERSION = "news.weekly.v2"
PLAYOFF_GENERATOR_VERSION = "news.playoffs.v2"


def _normalize_weekly_cache(value: Any) -> Dict[str, Any]:
    """Normalize weekly_news cache for state_schema 4.1.

    We accept legacy (4.0) shapes too and expand them into the 4.1 structure.

    Expected 4.1 keys:
      - last_generated_week_start: str|None
      - last_generated_as_of_date: str|None
      - built_from_turn: int
      - season_id: str|None
      - generator_version: str
      - llm: {used: bool, model: str|None, error: str|None}
      - items: list
    """
    cache: Dict[str, Any] = value if isinstance(value, dict) else {}

    llm_in = cache.get("llm")
    llm: Dict[str, Any] = llm_in if isinstance(llm_in, dict) else {}
    llm_norm = {
        "used": llm.get("used") if isinstance(llm.get("used"), bool) else False,
        "model": llm.get("model") if isinstance(llm.get("model"), str) else None,
        "error": llm.get("error") if isinstance(llm.get("error"), str) else None,
    }

    built_from_turn = cache.get("built_from_turn")
    if not isinstance(built_from_turn, int):
        built_from_turn = -1

    season_id = cache.get("season_id")
    if season_id is not None and not isinstance(season_id, str):
        season_id = None

    generator_version = cache.get("generator_version")
    if not isinstance(generator_version, str) or not generator_version:
        generator_version = WEEKLY_GENERATOR_VERSION

    items = cache.get("items")
    if not isinstance(items, list):
        items = []

    return {
        "last_generated_week_start": cache.get("last_generated_week_start"),
        "last_generated_as_of_date": cache.get("last_generated_as_of_date"),
        "built_from_turn": built_from_turn,
        "season_id": season_id,
        "generator_version": generator_version,
        "llm": llm_norm,
        "items": items,
    }


def _normalize_playoff_cache(value: Any) -> Dict[str, Any]:
    """Normalize playoff_news cache for state_schema 4.1.

    Expected 4.1 keys:
      - processed_game_ids: list[str]
      - built_from_turn: int
      - season_id: str|None
      - generator_version: str
      - items: list
    """
    cache: Dict[str, Any] = value if isinstance(value, dict) else {}


    processed = cache.get("processed_game_ids")
    if not isinstance(processed, list):
        processed = []
    processed = [str(x) for x in processed if x]

    built_from_turn = cache.get("built_from_turn")
    if not isinstance(built_from_turn, int):
        built_from_turn = -1

    season_id = cache.get("season_id")
    if season_id is not None and not isinstance(season_id, str):
        season_id = None

    generator_version = cache.get("generator_version")
    if not isinstance(generator_version, str) or not generator_version:
        generator_version = PLAYOFF_GENERATOR_VERSION

    items = cache.get("items")
    if not isinstance(items, list):
        items = []

    return {
        "processed_game_ids": processed,
        "built_from_turn": built_from_turn,
        "season_id": season_id,
        "generator_version": generator_version,
        "items": items,
    }


def get_weekly_cache() -> Dict[str, Any]:
    """Read and normalize cached weekly news."""
    return _normalize_weekly_cache(get_cached_weekly_news_snapshot() or {})


def set_weekly_cache(cache: Dict[str, Any]) -> None:
    """Persist weekly cache in strict 4.1 shape."""
    out = _normalize_weekly_cache(cache)
    set_cached_weekly_news_snapshot(out)


def get_playoff_cache() -> Dict[str, Any]:
    """Read and normalize cached playoff news."""
    return _normalize_playoff_cache(get_cached_playoff_news_snapshot() or {})


def set_playoff_cache(cache: Dict[str, Any]) -> None:
    """Persist playoff cache in strict 4.1 shape."""
    out = _normalize_playoff_cache(cache)
    set_cached_playoff_news_snapshot(out)


def weekly_cache_is_fresh(
    cache: Dict[str, Any],
    *,
    week_start: str,
    as_of_date: str,
    season_id: str | None,
    generator_version: str = WEEKLY_GENERATOR_VERSION,
) -> bool:
    """Return True if the weekly cache is safe to serve without regenerating.

    Policy (v2):
      - same week window (week_start)
      - same in-game 'as_of_date'
      - same active season
      - same generator_version
      - non-empty items
    """
    if not cache.get("items"):
        return False
    if cache.get("generator_version") != generator_version:
        return False
    if cache.get("season_id") != season_id:
        return False
    if cache.get("last_generated_week_start") != week_start:
        return False
    if cache.get("last_generated_as_of_date") != as_of_date:
        return False
    return True
