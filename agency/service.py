from __future__ import annotations

"""DB-backed orchestration for the agency subsystem.

This module wires together:
- DB reads (players/roster + existing agency state)
- expectation computation (role/leverage/expected minutes)
- monthly tick logic (tick.apply_monthly_player_tick)
- DB writes (upsert state + append events)

It is designed to be called from a checkpoint trigger (see agency/checkpoints.py)
when a month finishes.
"""

import logging
from typing import Any, Dict, Mapping, Optional

from league_repo import LeagueRepo

from .config import AgencyConfig, DEFAULT_CONFIG
from .expectations import compute_expectations_for_league
from .repo import get_player_agency_states, insert_agency_events, upsert_player_agency_states
from .tick import apply_monthly_player_tick
from .types import MonthlyPlayerInputs
from .utils import extract_mental_from_attrs, safe_float, safe_int


logger = logging.getLogger(__name__)


def _meta_key_for_month(month_key: str) -> str:
    return f"nba_agency_tick_done_{str(month_key)}"


def _best_effort_injury_status_by_pid(repo: LeagueRepo, player_ids: list[str]) -> Dict[str, str]:
    """Return mapping pid -> injury status (HEALTHY/OUT/RETURNING).

    This is best-effort: if injury tables are missing, returns empty.
    """
    if not player_ids:
        return {}

    try:
        placeholders = ",".join(["?"] * len(player_ids))
        rows = repo._conn.execute(
            f"""
            SELECT player_id, status
            FROM player_injury_state
            WHERE player_id IN ({placeholders});
            """,
            [*player_ids],
        ).fetchall()
        out: Dict[str, str] = {}
        for pid, status in rows:
            pid_s = str(pid)
            if not pid_s:
                continue
            out[pid_s] = str(status or "HEALTHY").upper()
        return out
    except Exception:
        return {}


def apply_monthly_agency_tick(
    *,
    db_path: str,
    season_year: int,
    month_key: str,
    minutes_by_player: Mapping[str, float],
    games_by_player: Optional[Mapping[str, int]] = None,
    team_win_pct_by_team: Optional[Mapping[str, float]] = None,
    now_iso: str,
    cfg: AgencyConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Apply a league-wide monthly agency tick.

    Idempotent via meta key `nba_agency_tick_done_{month_key}`.

    Args:
        db_path: SQLite path
        season_year: current season year
        month_key: YYYY-MM (the month being processed)
        minutes_by_player: total minutes played in that month (regular-season finals)
        games_by_player: games played in that month (optional but recommended)
        team_win_pct_by_team: team win% for that month (optional)
        now_iso: UTC-like timestamp string for created_at/updated_at/meta
        cfg: AgencyConfig
    """

    sy = int(season_year)
    mk = str(month_key)
    meta_key = _meta_key_for_month(mk)

    minutes_map = {str(k): float(v) for k, v in (minutes_by_player or {}).items() if str(k)}
    games_map = {str(k): int(v) for k, v in (games_by_player or {}).items() if str(k)} if games_by_player else {}
    team_win_map = {str(k).upper(): float(v) for k, v in (team_win_pct_by_team or {}).items() if str(k)}

    with LeagueRepo(db_path) as repo:
        repo.init_db()

        # Idempotency
        row = repo._conn.execute("SELECT value FROM meta WHERE key=?;", (meta_key,)).fetchone()
        if row and str(row[0]) == "1":
            return {"ok": True, "skipped": True, "reason": "already_done", "month": mk, "meta_key": meta_key}

        # Ensure agency tables exist. Fail loud with a helpful message.
        try:
            repo._conn.execute("SELECT 1 FROM player_agency_state LIMIT 1;")
        except Exception as exc:
            raise RuntimeError(
                "Agency schema is missing. Did you add db_schema.agency to db_schema.init.DEFAULT_MODULES?"
            ) from exc

        # Load roster + player data.
        rows = repo._conn.execute(
            """
            SELECT
                p.player_id,
                r.team_id,
                r.salary_amount,
                p.ovr,
                p.age,
                p.attrs_json
            FROM players p
            JOIN roster r ON r.player_id = p.player_id
            WHERE r.status='active'
            ORDER BY r.team_id ASC, p.player_id ASC;
            """
        ).fetchall()

        roster_rows: list[Dict[str, Any]] = []
        player_ids: list[str] = []
        for r in rows:
            pid = str(r[0] or "")
            tid = str(r[1] or "").upper()
            if not pid or not tid:
                continue
            player_ids.append(pid)
            roster_rows.append(
                {
                    "player_id": pid,
                    "team_id": tid,
                    "salary_amount": safe_float(r[2], 0.0),
                    "ovr": safe_int(r[3], 0),
                    "age": safe_int(r[4], 0),
                    "attrs_json": r[5],
                }
            )

        if not roster_rows:
            # Still write meta key to avoid reprocessing empty months.
            with repo.transaction() as cur:
                cur.execute(
                    "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                    (meta_key, "1"),
                )
            return {"ok": True, "skipped": True, "reason": "no_active_roster", "month": mk, "meta_key": meta_key}

        # Expectations for role/leverage/expected minutes.
        expectations = compute_expectations_for_league(roster_rows, config=cfg.expectations)

        # Injury status (best-effort)
        injury_status_by_pid = _best_effort_injury_status_by_pid(repo, player_ids)

        # Load previous agency states
        with repo.transaction() as cur:
            prev_states = get_player_agency_states(cur, player_ids)

        # Run tick
        new_states: Dict[str, Dict[str, Any]] = {}
        events: list[Dict[str, Any]] = []

        for rr in roster_rows:
            pid = str(rr["player_id"])
            tid = str(rr["team_id"]).upper()

            exp = expectations.get(pid)
            if exp is None:
                role_bucket = "UNKNOWN"
                leverage = 0.0
                expected_mpg = float(cfg.expectations.expected_mpg_by_role.get("UNKNOWN", 12.0))
            else:
                role_bucket = str(exp.role_bucket)
                leverage = float(exp.leverage)
                expected_mpg = float(exp.expected_mpg)

            mental = extract_mental_from_attrs(rr.get("attrs_json"), keys=cfg.mental_attr_keys)

            inp = MonthlyPlayerInputs(
                player_id=pid,
                team_id=tid,
                season_year=sy,
                month_key=mk,
                now_date_iso=str(now_iso)[:10],
                expected_mpg=float(expected_mpg),
                actual_minutes=float(minutes_map.get(pid, 0.0)),
                games_played=int(games_map.get(pid, 0)),
                role_bucket=role_bucket,  # type: ignore[arg-type]
                leverage=float(leverage),
                team_win_pct=float(team_win_map.get(tid, 0.5)),
                injury_status=injury_status_by_pid.get(pid),
                ovr=safe_int(rr.get("ovr"), 0),
                age=safe_int(rr.get("age"), 0),
                mental=mental,
            )

            prev = prev_states.get(pid)
            new_state, new_events = apply_monthly_player_tick(prev, inputs=inp, cfg=cfg)

            new_states[pid] = new_state
            events.extend(new_events)

        # Persist
        with repo.transaction() as cur:
            upsert_player_agency_states(cur, new_states, now=str(now_iso))
            insert_agency_events(cur, events, now=str(now_iso))
            cur.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                (meta_key, "1"),
            )

        return {
            "ok": True,
            "skipped": False,
            "month": mk,
            "meta_key": meta_key,
            "players_processed": len(roster_rows),
            "states_upserted": len(new_states),
            "events_emitted": len(events),
        }


