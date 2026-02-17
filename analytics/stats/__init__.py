"""Statistical analytics: leaderboards, advanced metrics, and awards.

This package is designed to be:
- Deterministic
- Defensive to missing/partial data
- Backwards compatible with existing server endpoints

Most callers should use `stats_util.py` (facade) which re-exports stable functions.
"""

from __future__ import annotations

from .cache import get_or_build_cached_leaderboards
from .leaders import compute_flat_legacy_leaders, compute_leaderboards

__all__ = [
    "compute_leaderboards",
    "compute_flat_legacy_leaders",
    "get_or_build_cached_leaderboards",
]
