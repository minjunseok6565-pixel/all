# db_schema/ego.py
"""SQLite SSOT schema: player ego.

This module adds a new table `player_ego` storing per-player personality traits
and mutable mood/issue state. Both columns are JSON blobs.

The ego table is designed for:
- durability (commercial save files)
- backwards compatibility (JSON version fields, tolerant reads)
- minimal coupling (no runtime state columns)

This module contains only DDL + migrations. It must not import LeagueRepo.
"""

from __future__ import annotations

import sqlite3
from typing import Callable, Mapping


# Signature compatible with LeagueRepo._ensure_table_columns(cur, table, columns)
EnsureColumnsFn = Callable[[sqlite3.Cursor, str, Mapping[str, str]], None]


def ddl(*, now: str, schema_version: str) -> str:
    # schema_version is unused but kept for registry signature.
    _ = schema_version
    return f"""
                CREATE TABLE IF NOT EXISTS player_ego (
                    player_id TEXT PRIMARY KEY,
                    traits_json TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(player_id) REFERENCES players(player_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_player_ego_updated_at ON player_ego(updated_at);
    """


def migrate(cur: sqlite3.Cursor, *, ensure_columns: EnsureColumnsFn) -> None:
    # Future-proofing: allow additional columns without breaking older DBs.
    ensure_columns(
        cur,
        "player_ego",
        {
            "traits_json": "TEXT NOT NULL DEFAULT '{}'",
            "state_json": "TEXT NOT NULL DEFAULT '{}'",
            "created_at": "TEXT",
            "updated_at": "TEXT",
        },
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_player_ego_updated_at ON player_ego(updated_at);")
