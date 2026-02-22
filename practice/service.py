from __future__ import annotations

import datetime as _dt
import logging
import sqlite3
from typing import Any, Dict, Mapping, Optional, Tuple

import game_time
from league_repo import LeagueRepo
from readiness import config as r_cfg
from readiness import formulas as r_f
from readiness import repo as r_repo
from sim.roster_adapter import resolve_effective_schemes

from . import ai as p_ai
from . import config as p_cfg
from . import defaults as p_defaults
from . import repo as p_repo
from . import types as p_types

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
    home_tactics: Optional[Mapping[str, Any]] = None,
    away_tactics: Optional[Mapping[str, Any]] = None,
) -> None:
    """Between-game practice hook invoked by the simulation pipeline.

    This hook is called **after** injury.prepare and **before** readiness.prepare.

    Responsibilities (v2):
      - Resolve / persist daily practice sessions for off-days.
      - Apply those sessions to the *readiness SSOT*:
          * player sharpness: base daily decay + practice delta (incl. scrimmage participants)
          * scheme familiarity: practice gain (diminishing returns) + correct decay via last_date gaps

    Commercial safety:
      - Must never crash the sim. Errors are logged and ignored.
      - Never uses host OS clock; uses in-game date ISO only.
    """
    try:
        d_iso = game_time.require_date_iso(game_date_iso, field="game_date_iso")
        gdate = _dt.date.fromisoformat(d_iso)
        sy = int(season_year)
        hid = str(home_team_id).upper()
        aid = str(away_team_id).upper()
    except Exception:
        logger.warning("PRACTICE_APPLY_INVALID_INPUTS", exc_info=True)
        return

    try:
        with repo.transaction() as cur:
            try:
                _apply_team_practice_to_readiness_ssot(
                    cur,
                    repo=repo,
                    team_id=hid,
                    season_year=sy,
                    game_date=gdate,
                    game_date_iso=d_iso,
                    raw_tactics=home_tactics,
                )
            except Exception:
                logger.warning("PRACTICE_APPLY_TEAM_FAILED team=%s date=%s", hid, d_iso, exc_info=True)

            try:
                _apply_team_practice_to_readiness_ssot(
                    cur,
                    repo=repo,
                    team_id=aid,
                    season_year=sy,
                    game_date=gdate,
                    game_date_iso=d_iso,
                    raw_tactics=away_tactics,
                )
            except Exception:
                logger.warning("PRACTICE_APPLY_TEAM_FAILED team=%s date=%s", aid, d_iso, exc_info=True)
    except Exception:
        # Transaction-level failure should never take down the sim.
        logger.warning("PRACTICE_APPLY_TRANSACTION_FAILED date=%s", d_iso, exc_info=True)
        return


def _apply_team_practice_to_readiness_ssot(
    cur: sqlite3.Cursor,
    *,
    repo: LeagueRepo,
    team_id: str,
    season_year: int,
    game_date: _dt.date,
    game_date_iso: str,
    raw_tactics: Optional[Mapping[str, Any]],
) -> None:
    """Apply between-game practice sessions to readiness SSOT for one team.

    We only process **off-days**: (anchor_date, game_date) exclusive of game_date.
    anchor_date is derived from the most recent readiness last_date seen for this team
    (sharpness and/or current effective scheme familiarity). This avoids accidentally
    applying practice to distant past dates (e.g., due to newly-acquired players).

    Persistence:
      - Uses ``resolve_practice_session`` (cursor-based) so sessions are deterministic.
      - Writes readiness SSOT via readiness.repo bulk upserts.
    """
    tid = str(team_id).upper()
    sy = int(season_year)

    # Resolve effective (off, def) schemes using roster_adapter SSOT.
    try:
        eff_off, eff_def = resolve_effective_schemes(tid, raw_tactics)
        eff_off = str(eff_off or "")
        eff_def = str(eff_def or "")
    except Exception:
        logger.warning("PRACTICE_EFFECTIVE_SCHEME_RESOLVE_FAILED team=%s", tid, exc_info=True)
        eff_off = ""
        eff_def = ""

    # Load roster (from SSOT roster table).
    try:
        roster_rows = repo.get_team_roster(tid)
    except Exception:
        logger.warning("PRACTICE_TEAM_ROSTER_LOAD_FAILED team=%s", tid, exc_info=True)
        return

    roster_pids: list[str] = []
    for row in roster_rows or []:
        pid = str((row or {}).get("player_id") or "")
        if pid:
            roster_pids.append(pid)

    if not roster_pids:
        return

    # Load readiness SSOT (sharpness) for roster players.
    sharp_rows = r_repo.get_player_sharpness_states(cur, roster_pids, season_year=sy)

    # Seed familiarity rows for the current effective schemes (if known).
    fam_seed: list[r_repo.SchemeKey] = []
    if eff_off:
        fam_seed.append(("offense", eff_off))
    if eff_def:
        fam_seed.append(("defense", eff_def))

    fam_rows = (
        r_repo.get_team_scheme_familiarity_states(cur, team_id=tid, season_year=sy, schemes=fam_seed) if fam_seed else {}
    )

    # Determine an anchor date: the most recent last_date seen among:
    # - roster player sharpness rows
    # - current effective scheme familiarity rows (if any)
    anchor_candidates: list[_dt.date] = []
    for row in (sharp_rows or {}).values():
        dt = r_f.parse_date_iso((row or {}).get("last_date"))
        if dt is not None:
            anchor_candidates.append(dt)
    for row in (fam_rows or {}).values():
        dt = r_f.parse_date_iso((row or {}).get("last_date"))
        if dt is not None:
            anchor_candidates.append(dt)

    if not anchor_candidates:
        # No SSOT baseline yet (e.g., first game of a new league/season).
        return

    anchor = max(anchor_candidates)
    end_day = game_date - _dt.timedelta(days=1)
    if anchor >= end_day:
        return  # no off-days to process

    # In-memory sharpness state (always kept for entire roster).
    s_default = float(getattr(r_cfg, "SHARPNESS_DEFAULT", 50.0) or 50.0)
    sharp_val: Dict[str, float] = {}
    sharp_last: Dict[str, _dt.date] = {}

    for pid in roster_pids:
        row = sharp_rows.get(pid)
        if row is None:
            sharp_val[pid] = s_default
            sharp_last[pid] = anchor
            continue
        try:
            sharp_val[pid] = float((row or {}).get("sharpness", s_default))
        except Exception:
            sharp_val[pid] = s_default
        dt = r_f.parse_date_iso((row or {}).get("last_date"))
        sharp_last[pid] = dt if dt is not None else anchor

    # In-memory familiarity state (only for scheme keys that receive practice gains).
    f_default = float(getattr(r_cfg, "FAMILIARITY_DEFAULT", 50.0) or 50.0)
    fam_val: Dict[r_repo.SchemeKey, float] = {}
    fam_last: Dict[r_repo.SchemeKey, Optional[_dt.date]] = {}
    touched_fam: set[r_repo.SchemeKey] = set()

    def _ensure_fam_key(key: r_repo.SchemeKey) -> None:
        if key in fam_val:
            return
        row = fam_rows.get(key)
        if row is None:
            # Load on-demand (small-N) for non-effective scheme practice targets.
            try:
                fetched = r_repo.get_team_scheme_familiarity_states(cur, team_id=tid, season_year=sy, schemes=[key])
                row = fetched.get(key)
            except Exception:
                row = None
        if row is None:
            fam_val[key] = f_default
            fam_last[key] = None
            return
        try:
            fam_val[key] = float((row or {}).get("value", f_default))
        except Exception:
            fam_val[key] = f_default
        fam_last[key] = r_f.parse_date_iso((row or {}).get("last_date"))

    touched_pids: set[str] = set()

    # Iterate off-days: (anchor, game_date) exclusive of game_date.
    day = anchor + _dt.timedelta(days=1)
    while day <= end_day:
        day_iso = day.isoformat()

        # Resolve practice session deterministically. This may persist an AUTO session if missing.
        try:
            sess = resolve_practice_session(
                cur,
                team_id=tid,
                season_year=sy,
                date_iso=day_iso,
                fallback_off_scheme=eff_off or None,
                fallback_def_scheme=eff_def or None,
                roster_pids=roster_pids,
                sharpness_by_pid=sharp_val,  # used only for SCRIMMAGE participant autofill ranking
                now_iso=game_time.utc_like_from_date_iso(day_iso, field="date_iso"),
            )
        except Exception:
            logger.warning("PRACTICE_RESOLVE_FAILED team=%s date=%s", tid, day_iso, exc_info=True)
            sess = p_defaults.default_session(
                typ="FILM",
                offense_scheme_key=eff_off or None,
                defense_scheme_key=eff_def or None,
            )

        typ = str((sess or {}).get("type") or "FILM").upper()

        # --- Sharpness daily adjustment ---
        for pid in roster_pids:
            last_dt = sharp_last.get(pid, anchor)
            if last_dt >= day:
                continue

            eff_typ = str(p_types.effective_type_for_pid(sess, pid) or "").upper()
            try:
                delta = float(p_cfg.SHARPNESS_DELTA.get(eff_typ, 0.0) or 0.0)
            except Exception:
                delta = 0.0

            s0 = float(sharp_val.get(pid, s_default))
            try:
                gap_days = int((day - last_dt).days)
            except Exception:
                gap_days = 1
            if gap_days <= 0:
                gap_days = 1
            s1 = r_f.decay_sharpness_linear(s0, days=gap_days)
            sharp_val[pid] = float(r_f.clamp100(s1 + delta))
            sharp_last[pid] = day
            touched_pids.add(pid)

        # --- Familiarity practice gain (lazy; only when gain>0 and type supports it) ---
        try:
            gain = float(p_cfg.FAMILIARITY_GAIN.get(typ, 0.0) or 0.0)
        except Exception:
            gain = 0.0

        if gain > 0.0:
            if typ in ("OFF_TACTICS", "FILM", "SCRIMMAGE"):
                sk = str((sess or {}).get("offense_scheme_key") or eff_off or "")
                if sk:
                    key = ("offense", sk)
                    _ensure_fam_key(key)
                    last = fam_last.get(key)
                    gap = int((day - last).days) if last is not None else 0
                    fam_val[key] = float(r_f.apply_diminishing_gain(r_f.decay_familiarity_exp(fam_val[key], days=gap), gain=gain))
                    fam_last[key] = day
                    touched_fam.add(key)

            if typ in ("DEF_TACTICS", "FILM", "SCRIMMAGE"):
                sk = str((sess or {}).get("defense_scheme_key") or eff_def or "")
                if sk:
                    key = ("defense", sk)
                    _ensure_fam_key(key)
                    last = fam_last.get(key)
                    gap = int((day - last).days) if last is not None else 0
                    fam_val[key] = float(r_f.apply_diminishing_gain(r_f.decay_familiarity_exp(fam_val[key], days=gap), gain=gain))
                    fam_last[key] = day
                    touched_fam.add(key)

        day += _dt.timedelta(days=1)

    # Persist SSOT updates (bulk).
    now_iso = game_time.utc_like_from_date_iso(game_date_iso, field="game_date_iso")

    if touched_pids:
        up_rows: Dict[str, Dict[str, Any]] = {}
        for pid in touched_pids:
            up_rows[pid] = {
                "sharpness": float(sharp_val.get(pid, s_default)),
                "last_date": sharp_last.get(pid, anchor).isoformat(),
            }
        try:
            r_repo.upsert_player_sharpness_states(cur, up_rows, season_year=sy, now=str(now_iso))
        except Exception:
            logger.warning("PRACTICE_SHARPNESS_UPSERT_FAILED team=%s date=%s", tid, game_date_iso, exc_info=True)

    if touched_fam:
        fam_up: Dict[r_repo.SchemeKey, Dict[str, Any]] = {}
        for key in touched_fam:
            fam_up[key] = {
                "value": float(fam_val.get(key, f_default)),
                "last_date": (fam_last.get(key) or anchor).isoformat(),
            }
        try:
            r_repo.upsert_team_scheme_familiarity_states(cur, fam_up, team_id=tid, season_year=sy, now=str(now_iso))
        except Exception:
            logger.warning("PRACTICE_FAMILIARITY_UPSERT_FAILED team=%s date=%s", tid, game_date_iso, exc_info=True)


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

    The list is clamped to [p_cfg.SCRIMMAGE_MIN_PLAYERS, p_cfg.SCRIMMAGE_MAX_PLAYERS].
    """
    if str(session.get("type") or "").upper() != "SCRIMMAGE":
        return session

    existing = session.get("participant_pids") or []
    if isinstance(existing, list) and len(existing) >= p_cfg.SCRIMMAGE_MIN_PLAYERS:
        # Still clamp max to keep it sane.
        session["participant_pids"] = [str(x) for x in existing][:p_cfg.SCRIMMAGE_MAX_PLAYERS]
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

    n = max(p_cfg.SCRIMMAGE_MIN_PLAYERS, min(p_cfg.SCRIMMAGE_MAX_PLAYERS, len(ranked)))
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

    if typ in ("OFF_TACTICS", "FILM", "SCRIMMAGE") and not session.get("offense_scheme_key"):
        if fallback_off_scheme:
            session["offense_scheme_key"] = str(fallback_off_scheme)

    if typ in ("DEF_TACTICS", "FILM", "SCRIMMAGE") and not session.get("defense_scheme_key"):
        if fallback_def_scheme:
            session["defense_scheme_key"] = str(fallback_def_scheme)

    if typ == "SCRIMMAGE":
        session = _autofill_scrimmage_participants(session, roster_pids=roster_pids, sharpness_by_pid=sharpness_by_pid)

    # Ensure non_participant_type is always a known type.
    nonp = str(session.get("non_participant_type") or "").upper()
    if nonp not in p_cfg.PRACTICE_TYPES:
        session["non_participant_type"] = str(p_cfg.SCRIMMAGE_NON_PARTICIPANT_DEFAULT)

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
