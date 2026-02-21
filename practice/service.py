from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, Mapping, Optional, Tuple

import game_time
from league_repo import LeagueRepo

from . import ai as p_ai
from . import defaults as p_defaults
from . import repo as p_repo
from . import types as p_types
from .config import PRACTICE_TYPES, SCRIMMAGE_MAX_PLAYERS, SCRIMMAGE_MIN_PLAYERS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulation hook (v1)
# ---------------------------------------------------------------------------


def apply_practice_before_game(
    repo: LeagueRepo,
    *,
    game_date_iso: str,
    season_year: int,
    home_team_id: str,
    away_team_id: str,
    home_tactics: Optional[Mapping[str, Any]] = None,  # noqa: ARG001
    away_tactics: Optional[Mapping[str, Any]] = None,  # noqa: ARG001
) -> None:
    """Between-game practice hook invoked by the simulation pipeline.

    Current scope (v1):
      - Safe placeholder so the match pipeline can call a single practice hook
        before readiness/fatigue/injury.

    Future scope (v2+):
      - Apply practice sessions to readiness SSOT:
          * team scheme familiarity updates (off/def)
          * player sharpness daily adjustments (incl. scrimmage participants)

    Commercial safety:
      - Must never crash the sim. Errors are logged and ignored.
    """

    # NOTE: `repo` is intentionally unused in v1. Keep signature stable.
    _ = repo

    try:
        _ = game_time.require_date_iso(game_date_iso, field="game_date_iso")
        _ = int(season_year)
        _ = str(home_team_id)
        _ = str(away_team_id)
    except Exception:
        logger.warning("PRACTICE_APPLY_INVALID_INPUTS", exc_info=True)
        return

    # Intentionally no-op.
    return


# ---------------------------------------------------------------------------
# High-level CRUD (repo wrappers)
# ---------------------------------------------------------------------------


def get_or_default_team_practice_plan(
    *,
    repo: LeagueRepo,
    team_id: str,
    season_year: int,
) -> Tuple[Dict[str, Any], bool]:
    """Return (plan, is_default)."""
    with repo.transaction() as cur:
        raw = p_repo.get_team_practice_plan(cur, team_id=str(team_id).upper(), season_year=int(season_year))
    if raw is None:
        return (p_defaults.default_team_practice_plan(team_id=str(team_id).upper(), season_year=int(season_year)), True)
    return (p_types.normalize_plan(raw), False)


def set_team_practice_plan(
    *,
    db_path: str,
    team_id: str,
    season_year: int,
    plan: Mapping[str, Any],
    now_iso: str,
) -> Dict[str, Any]:
    """Upsert a team practice plan."""
    with LeagueRepo(str(db_path)) as repo:
        repo.init_db()
        with repo.transaction() as cur:
            p_repo.upsert_team_practice_plan(
                cur,
                team_id=str(team_id).upper(),
                season_year=int(season_year),
                plan=p_types.normalize_plan(plan),
                now=str(now_iso),
            )
    return {"ok": True, "team_id": str(team_id).upper(), "season_year": int(season_year)}


def get_team_practice_session(
    *,
    repo: LeagueRepo,
    team_id: str,
    season_year: int,
    date_iso: str,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Return (session, is_user_set) without auto-generation."""
    d = game_time.require_date_iso(date_iso, field="date_iso")
    with repo.transaction() as cur:
        raw, is_user_set = p_repo.get_team_practice_session(
            cur, team_id=str(team_id).upper(), season_year=int(season_year), date_iso=d
        )
    if raw is None:
        return (None, False)
    return (p_types.normalize_session(raw), bool(is_user_set))


def set_team_practice_session(
    *,
    db_path: str,
    team_id: str,
    season_year: int,
    date_iso: str,
    session: Mapping[str, Any],
    now_iso: str,
    is_user_set: bool = True,
) -> Dict[str, Any]:
    """Upsert a daily practice session (typically user-authored)."""
    d = game_time.require_date_iso(date_iso, field="date_iso")
    sess = p_types.normalize_session(session)
    with LeagueRepo(str(db_path)) as repo:
        repo.init_db()
        with repo.transaction() as cur:
            p_repo.upsert_team_practice_session(
                cur,
                team_id=str(team_id).upper(),
                season_year=int(season_year),
                date_iso=d,
                session=sess,
                now=str(now_iso),
                is_user_set=bool(is_user_set),
            )
    return {"ok": True, "team_id": str(team_id).upper(), "season_year": int(season_year), "date_iso": d}


def list_team_practice_sessions(
    *,
    repo: LeagueRepo,
    team_id: str,
    season_year: int,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """List stored sessions (no auto-generation)."""
    df = game_time.require_date_iso(date_from, field="date_from") if date_from else None
    dt = game_time.require_date_iso(date_to, field="date_to") if date_to else None
    with repo.transaction() as cur:
        rows = p_repo.list_team_practice_sessions(
            cur,
            team_id=str(team_id).upper(),
            season_year=int(season_year),
            date_from=df,
            date_to=dt,
        )
    # Normalize payloads
    out: Dict[str, Dict[str, Any]] = {}
    for d, payload in (rows or {}).items():
        sess = p_types.normalize_session(payload.get("session") or {})
        out[str(d)[:10]] = {"session": sess, "is_user_set": bool(payload.get("is_user_set"))}
    return out


# ---------------------------------------------------------------------------
# Core: resolve a per-day session (DB-backed, deterministic)
# ---------------------------------------------------------------------------


def _autofill_scrimmage_participants(
    session: Dict[str, Any],
    *,
    roster_pids: list[str],
    sharpness_by_pid: Optional[Mapping[str, float]],
) -> Dict[str, Any]:
    """Ensure SCRIMMAGE has a reasonable participant list.

    Strategy:
      - If sharpness_by_pid is provided: pick lowest sharpness first.
      - Else: pick from roster order.

    The list is clamped to [SCRIMMAGE_MIN_PLAYERS, SCRIMMAGE_MAX_PLAYERS].
    """
    if str(session.get("type") or "").upper() != "SCRIMMAGE":
        return session

    existing = session.get("participant_pids") or []
    if isinstance(existing, list) and len(existing) >= SCRIMMAGE_MIN_PLAYERS:
        # Still clamp max to keep it sane.
        session["participant_pids"] = [str(x) for x in existing][:SCRIMMAGE_MAX_PLAYERS]
        return session

    roster = [str(pid) for pid in (roster_pids or []) if str(pid)]
    if not roster:
        session["participant_pids"] = []
        return session

    # Rank by sharpness if available.
    if sharpness_by_pid:
        ranked = sorted(roster, key=lambda pid: float(sharpness_by_pid.get(pid, 50.0)))
    else:
        ranked = roster

    n = max(SCRIMMAGE_MIN_PLAYERS, min(SCRIMMAGE_MAX_PLAYERS, len(ranked)))
    session["participant_pids"] = ranked[:n]
    return session


def _finalize_session_fields(
    session: Dict[str, Any],
    *,
    fallback_off_scheme: Optional[str],
    fallback_def_scheme: Optional[str],
    roster_pids: list[str],
    sharpness_by_pid: Optional[Mapping[str, float]],
) -> Dict[str, Any]:
    """Fill required fields based on session type."""
    typ = str(session.get("type") or "FILM").upper()

    if typ in ("OFF_TACTICS", "FILM") and not session.get("offense_scheme_key"):
        if fallback_off_scheme:
            session["offense_scheme_key"] = str(fallback_off_scheme)

    if typ in ("DEF_TACTICS", "FILM") and not session.get("defense_scheme_key"):
        if fallback_def_scheme:
            session["defense_scheme_key"] = str(fallback_def_scheme)

    if typ == "SCRIMMAGE":
        session = _autofill_scrimmage_participants(session, roster_pids=roster_pids, sharpness_by_pid=sharpness_by_pid)

    # Ensure non_participant_type is always a known type.
    nonp = str(session.get("non_participant_type") or "").upper()
    if nonp not in PRACTICE_TYPES:
        session["non_participant_type"] = "RECOVERY"

    return session


def resolve_practice_session(
    cur: sqlite3.Cursor,
    *,
    team_id: str,
    season_year: int,
    date_iso: str,
    fallback_off_scheme: Optional[str] = None,
    fallback_def_scheme: Optional[str] = None,
    roster_pids: Optional[list[str]] = None,
    sharpness_by_pid: Optional[Mapping[str, float]] = None,
    # Optional hints for AUTO mode (used by AI). All optional.
    days_to_next_game: Optional[int] = None,
    off_fam: Optional[float] = None,
    def_fam: Optional[float] = None,
    low_sharp_count: Optional[int] = None,
    now_iso: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve a per-day practice session.

    Behavior:
      1) If a session exists in DB, return it (normalized).
      2) Else, generate based on team_practice_plan:
         - AUTO: choose via practice.ai
         - MANUAL: default to FILM
      3) Store the generated session with is_user_set=0.

    Notes:
      - This function is cursor-based so it can be used inside other subsystem
        transactions (fatigue/injury/readiness) without nested transactions.
      - now_iso defaults to date_iso at midnight (UTC-like) to avoid OS time.
    """
    d = game_time.require_date_iso(date_iso, field="date_iso")
    tid = str(team_id).upper()
    sy = int(season_year)
    roster = [str(pid) for pid in (roster_pids or []) if str(pid)]

    raw, is_user_set = p_repo.get_team_practice_session(cur, team_id=tid, season_year=sy, date_iso=d)
    if raw is not None:
        # Stored session wins, even if not user-set.
        return p_types.normalize_session(raw)

    # Plan policy (AUTO/MANUAL).
    plan_raw = p_repo.get_team_practice_plan(cur, team_id=tid, season_year=sy)
    if plan_raw is None:
        plan = p_defaults.default_team_practice_plan(team_id=tid, season_year=sy)
    else:
        plan = p_types.normalize_plan(plan_raw)

    mode = str(plan.get("mode") or "AUTO").upper()

    # Generate a raw session.
    try:
        if mode == "AUTO":
            raw_sess = p_ai.choose_session_for_date(
                date_iso=d,
                days_to_next_game=days_to_next_game,
                off_fam=off_fam,
                def_fam=def_fam,
                low_sharp_count=low_sharp_count,
                fallback_off_scheme=fallback_off_scheme,
                fallback_def_scheme=fallback_def_scheme,
            )
        else:
            raw_sess = p_defaults.default_session(
                typ="FILM",
                offense_scheme_key=fallback_off_scheme,
                defense_scheme_key=fallback_def_scheme,
            )
    except Exception:
        logger.warning("PRACTICE_AI_FAILED", exc_info=True)
        raw_sess = p_defaults.default_session(
            typ="FILM",
            offense_scheme_key=fallback_off_scheme,
            defense_scheme_key=fallback_def_scheme,
        )

    sess = p_types.normalize_session(raw_sess)
    sess = _finalize_session_fields(
        sess,
        fallback_off_scheme=fallback_off_scheme,
        fallback_def_scheme=fallback_def_scheme,
        roster_pids=roster,
        sharpness_by_pid=sharpness_by_pid,
    )
    sess = p_types.normalize_session(sess)

    # Persist auto session for determinism and debuggability.
    now = str(now_iso) if now_iso else game_time.utc_like_from_date_iso(d, field="date_iso")
    p_repo.upsert_team_practice_session(
        cur,
        team_id=tid,
        season_year=sy,
        date_iso=d,
        session=sess,
        now=now,
        is_user_set=False,
    )

    return sess
