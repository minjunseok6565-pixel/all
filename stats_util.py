"""Compatibility facade for statistical leaderboards.

The original project contained a small `stats_util.py` module used by:
- /api/stats/leaders
- /api/stats/playoffs/leaders
- season_report_ai

This file keeps that import path stable while delegating to the new analytics package.

Backwards compatibility goals:
- Existing calls like `compute_league_leaders(player_stats)` keep working
- Returned shape remains a flat mapping with uppercase metric keys (PTS/AST/REB/3PM)
- Rows still include `per_game` and the metric key (e.g., row["PTS"])
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from analytics.stats.cache import get_or_build_cached_leaderboards as _get_or_build_cached_bundle
from analytics.stats.leaders import compute_flat_legacy_leaders, compute_leaderboards
from analytics.stats.types import LeaderboardConfig, LeaderboardsBundle

__all__ = [
    "compute_league_leaders",
    "compute_playoff_league_leaders",
    "compute_leaderboards_bundle",
    "get_or_build_cached_leaderboards",
]


def compute_league_leaders(player_stats: Mapping[str, Any], team_stats: Optional[Mapping[str, Any]] = None, *, top_n: int = 5) -> dict:
    """Compute per-game leaders from regular-season player_stats (legacy flat mapping).

    Returns:
        { "PTS": [...], "AST": [...], "REB": [...], "3PM": [...] }
    """
    return compute_flat_legacy_leaders(player_stats, team_stats, top_n=top_n, include_ties=False, phase="regular")


def compute_playoff_league_leaders(player_stats: Mapping[str, Any], team_stats: Optional[Mapping[str, Any]] = None, *, top_n: int = 5) -> dict:
    """Compute per-game leaders from playoff player_stats (legacy flat mapping)."""
    return compute_flat_legacy_leaders(player_stats, team_stats, top_n=top_n, include_ties=False, phase="playoffs")


def compute_leaderboards_bundle(
    player_stats: Mapping[str, Any],
    team_stats: Optional[Mapping[str, Any]] = None,
    *,
    phase: str = "regular",
    config: LeaderboardConfig | None = None,
) -> LeaderboardsBundle:
    """Compute a rich leaderboards bundle (recommended for new UI/features)."""
    return compute_leaderboards(player_stats, team_stats, phase=phase, config=config)


def get_or_build_cached_leaderboards(*, phase: str = "regular", config: LeaderboardConfig | None = None) -> LeaderboardsBundle:
    """Return cached leaderboards bundle for the given phase, rebuilding if needed."""
    return _get_or_build_cached_bundle(phase=phase, config=config)
