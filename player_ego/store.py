# player_ego/store.py
"""SQLite persistence layer for Player Ego.

This module is intentionally small:
- Read/write the player_ego table (traits_json, state_json).
- Seed missing rows deterministically.

It is written to match the codebase's philosophy:
- SQLite is the single source of truth (SSOT) for persisted league data.
- No OS clock: use game_time.now_utc_like_iso() or explicit date args.

The repo methods in league_repo.py provide transaction() with nested SAVEPOINT.
We accept an optional sqlite3.Cursor to allow callers (LeagueService) to update ego
within existing transactions.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TypeVar

import game_time

from . import archetypes
from .logic import make_default_state, normalize_state, normalize_traits
from .types import EgoRow


T = TypeVar("T")


def _json_dumps(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _with_cursor(repo: Any, cur: Optional[sqlite3.Cursor], fn: Callable[[sqlite3.Cursor], T]) -> T:
    """Run fn with a cursor.

    - If cur is provided, reuse it.
    - Otherwise, open a new repo.transaction() (supports nested SAVEPOINT).
    """
    if cur is not None:
        return fn(cur)

    tx = getattr(repo, "transaction", None)
    if tx is None:
        raise TypeError("repo must provide transaction()")

    with tx() as c:
        return fn(c)


def with_cursor(repo: Any, cur: Optional[sqlite3.Cursor], fn: Callable[[sqlite3.Cursor], T]) -> T:
    """Public wrapper for internal cursor helper."""

    return _with_cursor(repo, cur, fn)


def get_player_ego(
    repo: Any,
    player_id: str,
    *,
    cur: Optional[sqlite3.Cursor] = None,
) -> Optional[EgoRow]:
    """Read ego row for player_id. Returns None if missing."""
    pid = str(player_id)

    def _impl(c: sqlite3.Cursor) -> Optional[EgoRow]:
        row = c.execute(
            "SELECT player_id, traits_json, state_json FROM player_ego WHERE player_id=?;",
            (pid,),
        ).fetchone()
        if not row:
            return None
        traits = normalize_traits(_json_loads(row["traits_json"], {}))
        state = normalize_state(_json_loads(row["state_json"], {}))
        return {"player_id": str(row["player_id"]), "traits": traits, "state": state}

    return _with_cursor(repo, cur, _impl)


def upsert_player_ego(
    repo: Any,
    player_id: str,
    traits: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    cur: Optional[sqlite3.Cursor] = None,
) -> None:
    """Insert/update ego row."""
    pid = str(player_id)
    now = game_time.now_utc_like_iso()

    def _impl(c: sqlite3.Cursor) -> None:
        c.execute(
            """
            INSERT INTO player_ego(player_id, traits_json, state_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(player_id) DO UPDATE SET
                traits_json=excluded.traits_json,
                state_json=excluded.state_json,
                updated_at=excluded.updated_at;
            """,
            (pid, _json_dumps(dict(traits)), _json_dumps(dict(state)), now, now),
        )

    _with_cursor(repo, cur, _impl)


def get_or_create_player_ego(
    repo: Any,
    player_id: str,
    *,
    team_id: Optional[str] = None,
    date_iso: Optional[str] = None,
    cur: Optional[sqlite3.Cursor] = None,
) -> EgoRow:
    """Read ego row; if missing, create it deterministically."""
    existing = get_player_ego(repo, player_id, cur=cur)
    if existing is not None:
        return existing

    pid = str(player_id)

    def _impl(c: sqlite3.Cursor) -> EgoRow:
        prow = c.execute(
            "SELECT player_id, name, pos, age, ovr FROM players WHERE player_id=?;",
            (pid,),
        ).fetchone()
        if not prow:
            raise KeyError(f"player not found: {player_id}")

        player = {
            "player_id": str(prow["player_id"]),
            "name": prow["name"],
            "pos": prow["pos"],
            "age": prow["age"],
            "ovr": prow["ovr"],
        }

        traits = archetypes.generate_traits(player)
        state = make_default_state(player=player, traits=traits, team_id=team_id, date_iso=date_iso)

        upsert_player_ego(repo, pid, traits, state, cur=c)
        return {"player_id": pid, "traits": normalize_traits(traits), "state": normalize_state(state)}

    return _with_cursor(repo, cur, _impl)


def bulk_get_player_ego(
    repo: Any,
    player_ids: Sequence[str],
    *,
    cur: Optional[sqlite3.Cursor] = None,
) -> Dict[str, EgoRow]:
    """Bulk get ego rows. Missing ids are omitted."""
    ids = [str(p) for p in (player_ids or []) if p is not None and str(p).strip()]
    if not ids:
        return {}

    def _impl(c: sqlite3.Cursor) -> Dict[str, EgoRow]:
        placeholders = ",".join(["?"] * len(ids))
        rows = c.execute(
            f"SELECT player_id, traits_json, state_json FROM player_ego WHERE player_id IN ({placeholders});",
            ids,
        ).fetchall()
        out: Dict[str, EgoRow] = {}
        for r in rows:
            pid = str(r["player_id"])
            traits = normalize_traits(_json_loads(r["traits_json"], {}))
            state = normalize_state(_json_loads(r["state_json"], {}))
            out[pid] = {"player_id": pid, "traits": traits, "state": state}
        return out

    return _with_cursor(repo, cur, _impl)


def ensure_seeded_for_all_players(
    repo: Any,
    *,
    date_iso: Optional[str] = None,
    cur: Optional[sqlite3.Cursor] = None,
) -> int:
    """Ensure every player in players table has a player_ego row.

    Returns
    -------
    int
        Number of newly created rows.
    """
    date_iso = str(date_iso)[:10] if date_iso else game_time.game_date_iso()
    now = game_time.now_utc_like_iso()

    def _impl(c: sqlite3.Cursor) -> int:
        existing_rows = c.execute("SELECT player_id FROM player_ego;").fetchall()
        existing = {str(r["player_id"]) for r in existing_rows}

        players = c.execute("SELECT player_id, name, pos, age, ovr FROM players ORDER BY player_id;").fetchall()

        to_insert: List[tuple[str, str, str, str, str]] = []
        created = 0

        roster_team: Dict[str, str] = {}
        try:
            rrows = c.execute("SELECT player_id, team_id FROM roster WHERE status='active';").fetchall()
            roster_team = {str(r["player_id"]): str(r["team_id"]).upper() for r in rrows}
        except Exception:
            roster_team = {}

        for p in players:
            pid = str(p["player_id"])
            if pid in existing:
                continue

            player = {
                "player_id": pid,
                "name": p["name"],
                "pos": p["pos"],
                "age": p["age"],
                "ovr": p["ovr"],
            }
            traits = archetypes.generate_traits(player)
            state = make_default_state(
                player=player,
                traits=traits,
                team_id=roster_team.get(pid),
                date_iso=date_iso,
            )
            to_insert.append((pid, _json_dumps(traits), _json_dumps(state), now, now))
            created += 1

        if to_insert:
            c.executemany(
                """
                INSERT OR IGNORE INTO player_ego(player_id, traits_json, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?);
                """,
                to_insert,
            )

        return created

    return _with_cursor(repo, cur, _impl)
