# db_schema/init.py
"""Public entrypoint for applying the SQLite schema.

NOTE: Pure refactor split from league_repo.py (no functional changes).
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

from . import core, training, fatigue, injury, readiness, agency, trade_assets, draft, gm, college, scouting
from .registry import EnsureColumnsFn, apply_all


# Order matters:
# - core must come first (players table is referenced by other modules)
# - training references core.players
DEFAULT_MODULES = (core, training, fatigue, injury, readiness, agency, trade_assets, draft, gm, college, scouting)


def apply_schema(
    cur: sqlite3.Cursor,
    *,
    now: str,
    schema_version: str,
    ensure_columns: EnsureColumnsFn,
    modules: Iterable[object] = DEFAULT_MODULES,
) -> None:
    """Apply the schema and migrations.

    The default module order matches the intended split:
    core -> trade_assets -> draft -> gm
    """
    apply_all(
        cur,
        modules=modules,  # type: ignore[arg-type]
        now=now,
        schema_version=schema_version,
        ensure_columns=ensure_columns,
    )
