from __future__ import annotations

"""DB access layer for the agency subsystem.

This module is intentionally *pure DB I/O*:
- No imports from sim/matchengine to avoid circular dependencies.
- No business logic besides defensive normalization.

Tables (SSOT):
- player_agency_state
- agency_events

Dates are stored as ISO strings.
"""

import sqlite3
from typing import Any, Dict, Mapping, Optional

from .utils import json_dumps, json_loads, norm_date_iso, norm_month_key, safe_float, safe_int


def _uniq_str_ids(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in ids:
        s = str(x)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def get_player_agency_states(
    cur: sqlite3.Cursor,
    player_ids: list[str],
) -> Dict[str, Dict[str, Any]]:
    """Bulk-load agency states for the given players.

    Returns:
        dict[player_id] -> state dict (JSON fields decoded).

    Notes:
    - Missing rows are omitted.
    - Invalid JSON is tolerated.
    """
    uniq = _uniq_str_ids([str(pid) for pid in (player_ids or []) if str(pid)])
    if not uniq:
        return {}

    placeholders = ",".join(["?"] * len(uniq))
    rows = cur.execute(
        f"""
        SELECT
            player_id,
            team_id,
            season_year,
            role_bucket,
            leverage,
            minutes_expected_mpg,
            minutes_actual_mpg,
            minutes_frustration,
            team_frustration,
            trust,
            trade_request_level,
            cooldown_minutes_until,
            cooldown_trade_until,
            cooldown_help_until,
            cooldown_contract_until,
            last_processed_month,
            context_json
        FROM player_agency_state
        WHERE player_id IN ({placeholders});
        """,
        uniq,
    ).fetchall()

    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        pid = str(r[0])
        out[pid] = {
            "player_id": pid,
            "team_id": str(r[1] or "").upper(),
            "season_year": safe_int(r[2], 0),
            "role_bucket": str(r[3] or "UNKNOWN"),
            "leverage": safe_float(r[4], 0.0),
            "minutes_expected_mpg": safe_float(r[5], 0.0),
            "minutes_actual_mpg": safe_float(r[6], 0.0),
            "minutes_frustration": safe_float(r[7], 0.0),
            "team_frustration": safe_float(r[8], 0.0),
            "trust": safe_float(r[9], 0.5),
            "trade_request_level": safe_int(r[10], 0),
            "cooldown_minutes_until": norm_date_iso(r[11]),
            "cooldown_trade_until": norm_date_iso(r[12]),
            "cooldown_help_until": norm_date_iso(r[13]),
            "cooldown_contract_until": norm_date_iso(r[14]),
            "last_processed_month": norm_month_key(r[15]),
            "context": json_loads(r[16], default={}) or {},
        }

    return out


def upsert_player_agency_states(
    cur: sqlite3.Cursor,
    states_by_pid: Mapping[str, Mapping[str, Any]],
    *,
    now: str,
) -> None:
    """Upsert agency states.

    Args:
        states_by_pid: mapping[player_id] -> state dict.
        now: timestamp string (UTC-like in-game time)

    Commercial safety:
    - invalid entries are skipped silently.
    """
    if not states_by_pid:
        return

    rows: list[tuple[Any, ...]] = []
    for pid, st in states_by_pid.items():
        pid_s = str(pid)
        if not pid_s:
            continue
        try:
            team_id = str(st.get("team_id") or "").upper()
            if not team_id:
                continue
            season_year = safe_int(st.get("season_year"), 0)
            if season_year <= 0:
                continue

            role_bucket = str(st.get("role_bucket") or "UNKNOWN")
            leverage = float(safe_float(st.get("leverage"), 0.0))

            exp_mpg = float(safe_float(st.get("minutes_expected_mpg"), 0.0))
            act_mpg = float(safe_float(st.get("minutes_actual_mpg"), 0.0))

            minutes_fr = float(safe_float(st.get("minutes_frustration"), 0.0))
            team_fr = float(safe_float(st.get("team_frustration"), 0.0))
            trust = float(safe_float(st.get("trust"), 0.5))

            tr_level = safe_int(st.get("trade_request_level"), 0)

            cd_minutes = norm_date_iso(st.get("cooldown_minutes_until"))
            cd_trade = norm_date_iso(st.get("cooldown_trade_until"))
            cd_help = norm_date_iso(st.get("cooldown_help_until"))
            cd_contract = norm_date_iso(st.get("cooldown_contract_until"))

            last_month = norm_month_key(st.get("last_processed_month"))

            context = st.get("context") or {}
            if not isinstance(context, Mapping):
                context = {}

        except Exception:
            continue

        rows.append(
            (
                pid_s,
                team_id,
                int(season_year),
                role_bucket,
                float(leverage),
                float(exp_mpg),
                float(act_mpg),
                float(minutes_fr),
                float(team_fr),
                float(trust),
                int(tr_level),
                cd_minutes,
                cd_trade,
                cd_help,
                cd_contract,
                last_month,
                json_dumps(dict(context)),
                str(now),
                str(now),
            )
        )

    if not rows:
        return

    cur.executemany(
        """
        INSERT INTO player_agency_state(
            player_id,
            team_id,
            season_year,
            role_bucket,
            leverage,
            minutes_expected_mpg,
            minutes_actual_mpg,
            minutes_frustration,
            team_frustration,
            trust,
            trade_request_level,
            cooldown_minutes_until,
            cooldown_trade_until,
            cooldown_help_until,
            cooldown_contract_until,
            last_processed_month,
            context_json,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_id) DO UPDATE SET
            team_id=excluded.team_id,
            season_year=excluded.season_year,
            role_bucket=excluded.role_bucket,
            leverage=excluded.leverage,
            minutes_expected_mpg=excluded.minutes_expected_mpg,
            minutes_actual_mpg=excluded.minutes_actual_mpg,
            minutes_frustration=excluded.minutes_frustration,
            team_frustration=excluded.team_frustration,
            trust=excluded.trust,
            trade_request_level=excluded.trade_request_level,
            cooldown_minutes_until=excluded.cooldown_minutes_until,
            cooldown_trade_until=excluded.cooldown_trade_until,
            cooldown_help_until=excluded.cooldown_help_until,
            cooldown_contract_until=excluded.cooldown_contract_until,
            last_processed_month=excluded.last_processed_month,
            context_json=excluded.context_json,
            updated_at=excluded.updated_at;
        """,
        rows,
    )


def insert_agency_events(
    cur: sqlite3.Cursor,
    events: list[Mapping[str, Any]],
    *,
    now: str,
) -> None:
    """Insert agency events (append-only).

    Args:
        events: list of dict rows matching agency_events columns.
        now: created_at timestamp

    Invalid entries are skipped silently.
    """
    if not events:
        return

    rows: list[tuple[Any, ...]] = []
    for e in events:
        try:
            event_id = str(e.get("event_id") or "")
            if not event_id:
                continue
            player_id = str(e.get("player_id") or "")
            team_id = str(e.get("team_id") or "").upper()
            if not player_id or not team_id:
                continue
            season_year = safe_int(e.get("season_year"), 0)
            if season_year <= 0:
                continue
            date_iso = norm_date_iso(e.get("date"))
            if not date_iso:
                continue
            event_type = str(e.get("event_type") or "")
            if not event_type:
                continue
            severity = float(safe_float(e.get("severity"), 0.0))
            payload = e.get("payload") or {}
            if not isinstance(payload, Mapping):
                payload = {}
        except Exception:
            continue

        rows.append(
            (
                event_id,
                player_id,
                team_id,
                int(season_year),
                date_iso,
                event_type,
                float(severity),
                json_dumps(dict(payload)),
                str(now),
            )
        )

    if not rows:
        return

    cur.executemany(
        """
        INSERT OR IGNORE INTO agency_events(
            event_id,
            player_id,
            team_id,
            season_year,
            date,
            event_type,
            severity,
            payload_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        rows,
    )


def list_agency_events(
    cur: sqlite3.Cursor,
    *,
    team_id: Optional[str] = None,
    player_id: Optional[str] = None,
    season_year: Optional[int] = None,
    event_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Dict[str, Any]]:
    """List events for UI feeds.

    This is intentionally simple (SQLite). For large-scale needs, add proper
    pagination and richer filtering.
    """
    where: list[str] = []
    args: list[Any] = []

    if team_id:
        where.append("team_id = ?")
        args.append(str(team_id).upper())
    if player_id:
        where.append("player_id = ?")
        args.append(str(player_id))
    if season_year is not None:
        where.append("season_year = ?")
        args.append(int(season_year))
    if event_type:
        where.append("event_type = ?")
        args.append(str(event_type))

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    lim = int(limit) if int(limit) > 0 else 50
    off = int(offset) if int(offset) >= 0 else 0

    rows = cur.execute(
        f"""
        SELECT
            event_id,
            player_id,
            team_id,
            season_year,
            date,
            event_type,
            severity,
            payload_json,
            created_at
        FROM agency_events
        {where_sql}
        ORDER BY date DESC, created_at DESC
        LIMIT ? OFFSET ?;
        """,
        args + [lim, off],
    ).fetchall()

    out: list[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "event_id": str(r[0]),
                "player_id": str(r[1]),
                "team_id": str(r[2]),
                "season_year": safe_int(r[3], 0),
                "date": norm_date_iso(r[4]),
                "event_type": str(r[5]),
                "severity": safe_float(r[6], 0.0),
                "payload": json_loads(r[7], default={}) or {},
                "created_at": r[8],
            }
        )

    return out


