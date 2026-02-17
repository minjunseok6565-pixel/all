"""Statistical analytics: leaderboards, advanced metrics, and awards.

This package is designed to be:
- Deterministic
- Defensive to missing/partial data

Most callers should prefer importing from `analytics.stats` (or the small
`stats_util.py` facade) rather than reaching into module internals.
"""

from __future__ import annotations

from .cache import get_or_build_cached_leaderboards
from .leaders import compute_leaderboards

__all__ = [
    "compute_leaderboards",
    "get_or_build_cached_leaderboards",
]
