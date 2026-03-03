from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

import game_time
import state
from league_repo import LeagueRepo
from app.schemas.practice import (
    TeamPracticePlanRequest,
    TeamPracticePreviewRequest,
    TeamPracticePreviewRangeRequest,
    TeamPracticeSessionRequest,
)
from app.schemas.training import PlayerTrainingPlanRequest, TeamTrainingPlanRequest

router = APIRouter()
logger = logging.getLogger(__name__)

def _resolve_season_year(explicit: Optional[int]) -> int:
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(explicit or (league_ctx.get("season_year") or 0))
    if sy <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")
    return sy


def _resolve_as_of_date(as_of_date: Optional[str]) -> str:
    if as_of_date:
        return game_time.require_date_iso(as_of_date, field="as_of_date")
    return state.get_current_date_as_date().isoformat()


def _normalize_preview_session_payload(req: Any) -> Dict[str, Any]:
    from practice import types as p_types

    return p_types.normalize_session({
        "type": req.type,
        "offense_scheme_key": req.offense_scheme_key,
        "defense_scheme_key": req.defense_scheme_key,
        "participant_pids": req.participant_pids or [],
        "non_participant_type": req.non_participant_type,
    })



@router.get("/api/training/team/{team_id}")
async def api_get_team_training_plan(team_id: str, season_year: Optional[int] = None):
    """Get a team training plan (default if missing)."""
    db_path = state.get_db_path()
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(season_year or (league_ctx.get("season_year") or 0))
    if sy <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    from training.service import get_or_default_team_plan

    with LeagueRepo(db_path) as repo:
        repo.init_db()
        plan, is_default = get_or_default_team_plan(repo=repo, team_id=str(team_id).upper(), season_year=sy)
    return {"team_id": str(team_id).upper(), "season_year": sy, "plan": plan, "is_default": bool(is_default)}


@router.post("/api/training/team/set")
async def api_set_team_training_plan(req: TeamTrainingPlanRequest):
    """Set a team training plan."""
    db_path = state.get_db_path()
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(req.season_year or (league_ctx.get("season_year") or 0))
    if sy <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    from training.service import set_team_plan

    now_iso = state.get_current_date_as_date().isoformat()
    plan = {"focus": req.focus, "intensity": req.intensity, "weights": req.weights or {}}
    return set_team_plan(
        db_path=str(db_path),
        team_id=str(req.team_id).upper(),
        season_year=sy,
        plan=plan,
        now_iso=now_iso,
    )


@router.get("/api/training/player/{player_id}")
async def api_get_player_training_plan(player_id: str, season_year: Optional[int] = None):
    """Get a player training plan (default if missing)."""
    db_path = state.get_db_path()
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(season_year or (league_ctx.get("season_year") or 0))
    if sy <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    from training.service import get_or_default_player_plan

    with LeagueRepo(db_path) as repo:
        repo.init_db()
        try:
            p = repo.get_player(str(player_id))
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        attrs = p.get("attrs") or {}
        plan, is_default = get_or_default_player_plan(repo=repo, player_id=str(player_id), season_year=sy, attrs=attrs)
    return {"player_id": str(player_id), "season_year": sy, "plan": plan, "is_default": bool(is_default)}


@router.post("/api/training/player/set")
async def api_set_player_training_plan(req: PlayerTrainingPlanRequest):
    """Set a player training plan."""
    db_path = state.get_db_path()
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(req.season_year or (league_ctx.get("season_year") or 0))
    if sy <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    from training.service import set_player_plan

    now_iso = state.get_current_date_as_date().isoformat()
    plan = {"primary": req.primary, "secondary": req.secondary, "intensity": req.intensity}
    return set_player_plan(
        db_path=str(db_path),
        player_id=str(req.player_id),
        season_year=sy,
        plan=plan,
        now_iso=now_iso,
        is_user_set=True,
    )


# -------------------------------------------------------------------------
# Practice API (team sessions)
# -------------------------------------------------------------------------


@router.get("/api/practice/team/{team_id}/plan")
async def api_get_team_practice_plan(team_id: str, season_year: Optional[int] = None):
    """Get a team practice plan (default if missing)."""
    db_path = state.get_db_path()
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(season_year or (league_ctx.get("season_year") or 0))
    if sy <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    from practice.service import get_or_default_team_practice_plan

    with LeagueRepo(db_path) as repo:
        repo.init_db()
        plan, is_default = get_or_default_team_practice_plan(repo=repo, team_id=str(team_id).upper(), season_year=sy)
    return {"team_id": str(team_id).upper(), "season_year": sy, "plan": plan, "is_default": bool(is_default)}


@router.post("/api/practice/team/plan/set")
async def api_set_team_practice_plan(req: TeamPracticePlanRequest):
    """Set a team practice plan."""
    db_path = state.get_db_path()
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(req.season_year or (league_ctx.get("season_year") or 0))
    if sy <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    from practice.service import set_team_practice_plan

    now_iso = state.get_current_date_as_date().isoformat()
    plan = {"mode": req.mode}
    return set_team_practice_plan(
        db_path=str(db_path),
        team_id=str(req.team_id).upper(),
        season_year=sy,
        plan=plan,
        now_iso=now_iso,
    )


@router.get("/api/practice/team/{team_id}/session")
async def api_get_team_practice_session(
    team_id: str,
    date_iso: str,
    season_year: Optional[int] = None,
):
    """Get (and auto-resolve) a practice session for a specific date."""
    db_path = state.get_db_path()
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(season_year or (league_ctx.get("season_year") or 0))
    if sy <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    tid = str(team_id).upper()
    d = game_time.require_date_iso(date_iso, field="date_iso")
    now_iso = game_time.utc_like_from_date_iso(d, field="date_iso")

    # Schedule-context hint for AUTO practice AI (best-effort).
    d2g: Optional[int] = None
    try:
        d2g = state.get_days_to_next_game(team_id=tid, date_iso=d)
    except Exception:
        logger.exception("state.get_days_to_next_game failed (practice session). team=%s date=%s", tid, d)
        d2g = None

    # Best-effort fallback schemes from coach presets.
    fb_off = None
    fb_def = None
    try:
        from sim import roster_adapter as _roster_adapter
        from matchengine_v3.tactics import canonical_defense_scheme

        cfg = _roster_adapter._build_tactics_config(None)
        _roster_adapter._apply_default_coach_preset(tid, cfg)
        _roster_adapter._apply_coach_preset_tactics(tid, cfg, None)
        fb_off = str(cfg.offense_scheme)
        fb_def = canonical_defense_scheme(cfg.defense_scheme)
    except Exception:
        fb_off, fb_def = (None, None)

    with LeagueRepo(db_path) as repo:
        repo.init_db()

        # Stable roster pid ordering for scrimmage autofill.
        roster_rows = repo.get_team_roster(tid)
        roster_pids = [str(r.get("player_id")) for r in (roster_rows or []) if r.get("player_id")]

        # Resolve inside a single transaction for determinism.
        from practice import repo as p_repo
        from practice import types as p_types
        from practice.service import resolve_practice_session

        with repo.transaction() as cur:
            raw, is_user_set = p_repo.get_team_practice_session(cur, team_id=tid, season_year=sy, date_iso=d)
            if raw is None:
                sess = resolve_practice_session(
                    cur,
                    team_id=tid,
                    season_year=sy,
                    date_iso=d,
                    fallback_off_scheme=fb_off,
                    fallback_def_scheme=fb_def,
                    roster_pids=roster_pids,
                    days_to_next_game=d2g,
                    now_iso=now_iso,
                )
                is_user_set = False
            else:
                sess = p_types.normalize_session(raw)

    return {"team_id": tid, "season_year": sy, "date_iso": d, "session": sess, "is_user_set": bool(is_user_set)}


@router.get("/api/practice/team/{team_id}/sessions")
async def api_list_team_practice_sessions(
    team_id: str,
    season_year: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    """List stored practice sessions (does not auto-generate missing dates)."""
    db_path = state.get_db_path()
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(season_year or (league_ctx.get("season_year") or 0))
    if sy <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    from practice.service import list_team_practice_sessions

    with LeagueRepo(db_path) as repo:
        repo.init_db()
        rows = list_team_practice_sessions(
            repo=repo,
            team_id=str(team_id).upper(),
            season_year=sy,
            date_from=date_from,
            date_to=date_to,
        )
    return {"team_id": str(team_id).upper(), "season_year": sy, "sessions": rows}





@router.get("/api/practice/team/{team_id}/dashboard-summary")
async def api_get_team_practice_dashboard_summary(
    team_id: str,
    season_year: Optional[int] = None,
    as_of_date: Optional[str] = None,
):
    """Aggregated header summary for the training dashboard."""
    db_path = state.get_db_path()
    sy = _resolve_season_year(season_year)
    tid = str(team_id).upper()
    as_of = _resolve_as_of_date(as_of_date)
    as_of_dt = _dt.date.fromisoformat(as_of)

    league = (state.export_full_state_snapshot() or {}).get("league", {})
    games = ((league.get("master_schedule") or {}).get("games") or [])

    team_games = []
    for g in games:
        if g.get("home_team_id") != tid and g.get("away_team_id") != tid:
            continue
        d = str(g.get("date") or "")[:10]
        if len(d) != 10:
            continue
        opp = g.get("away_team_id") if g.get("home_team_id") == tid else g.get("home_team_id")
        is_home = bool(g.get("home_team_id") == tid)
        team_games.append({"date": d, "opponent_team_id": str(opp or "").upper(), "is_home": is_home})
    team_games.sort(key=lambda x: x["date"])

    next_game = None
    for g in team_games:
        gdt = _dt.date.fromisoformat(g["date"])
        if gdt >= as_of_dt:
            next_game = {
                "date": g["date"],
                "opponent_team_id": g["opponent_team_id"],
                "is_home": g["is_home"],
                "days_until": int((gdt - as_of_dt).days),
            }
            break

    recent_games = 0
    from_dt = as_of_dt - _dt.timedelta(days=6)
    for g in team_games:
        gdt = _dt.date.fromisoformat(g["date"])
        if from_dt <= gdt <= as_of_dt:
            recent_games += 1

    from practice.service import list_team_practice_sessions

    with LeagueRepo(db_path) as repo:
        repo.init_db()
        sessions = list_team_practice_sessions(
            repo=repo,
            team_id=tid,
            season_year=sy,
            date_from=from_dt.isoformat(),
            date_to=as_of,
        )

    recent_practices = 0
    for d, payload in (sessions or {}).items():
        try:
            ddt = _dt.date.fromisoformat(str(d)[:10])
        except ValueError:
            continue
        if from_dt <= ddt <= as_of_dt and str((payload.get("session") or {}).get("type") or "").upper() != "REST":
            recent_practices += 1

    from readiness import formulas as r_f
    from readiness import repo as r_repo
    from injury.status import status_for_date

    with LeagueRepo(db_path) as repo:
        repo.init_db()
        roster_rows = repo.get_team_roster(tid)
        pids = [str(r.get("player_id")) for r in (roster_rows or []) if r.get("player_id")]
        with repo.transaction() as cur:
            sharp_rows = r_repo.get_player_sharpness_states(cur, pids, season_year=sy) if pids else {}
            from injury import repo as i_repo
            injury_rows = i_repo.get_player_injury_states(cur, pids) if pids else {}

    sharp_values = []
    for pid in pids:
        row = sharp_rows.get(pid) or {}
        base = float(row.get("sharpness", 50.0) or 50.0)
        last_date = row.get("last_date")
        last_dt = r_f.parse_date_iso(last_date)
        days = max(0, int((as_of_dt - last_dt).days)) if last_dt is not None else 0
        sharp_values.append(float(r_f.decay_sharpness_linear(base, days=days)))

    low_sharp_count = sum(1 for v in sharp_values if v < 45.0)
    sharp_avg = float(sum(sharp_values) / len(sharp_values)) if sharp_values else 0.0

    out_count = 0
    returning_count = 0
    for pid in pids:
        status = str(status_for_date(injury_rows.get(pid) or {}, on_date_iso=as_of) or "HEALTHY").upper()
        if status == "OUT":
            out_count += 1
        elif status == "RETURNING":
            returning_count += 1

    overview = {"summary": {"risk_tier_counts": {"HIGH": 0}}}
    try:
        # Reuse existing route logic via direct import fallback is intentionally avoided;
        # keep dashboard independent and fail-soft.
        from app.api.routes import core as core_routes
        req = await core_routes.api_medical_team_overview(team_id=tid, season_year=sy)
        if isinstance(req, dict):
            overview = req
    except Exception:
        overview = {"summary": {"risk_tier_counts": {"HIGH": 0}}}

    risk_high_count = int((((overview.get("summary") or {}).get("risk_tier_counts") or {}).get("HIGH") or 0))

    return {
        "team_id": tid,
        "season_year": sy,
        "as_of_date": as_of,
        "next_game": next_game,
        "recent_load": {
            "games_last_7_days": recent_games,
            "practices_last_7_days": recent_practices,
        },
        "readiness": {
            "sharpness_avg": sharp_avg,
            "low_sharp_count": low_sharp_count,
        },
        "medical": {
            "risk_high_count": risk_high_count,
            "out_count": out_count,
            "returning_count": returning_count,
        },
    }


@router.get("/api/practice/team/{team_id}/calendar-window")
async def api_get_team_practice_calendar_window(
    team_id: str,
    date_from: str,
    date_to: str,
    season_year: Optional[int] = None,
):
    """Aggregated day-by-day practice/schedule/risk window for training calendar UI."""
    db_path = state.get_db_path()
    sy = _resolve_season_year(season_year)
    tid = str(team_id).upper()
    df = game_time.require_date_iso(date_from, field="date_from")
    dt = game_time.require_date_iso(date_to, field="date_to")
    df_dt = _dt.date.fromisoformat(df)
    dt_dt = _dt.date.fromisoformat(dt)
    if dt_dt < df_dt:
        raise HTTPException(status_code=400, detail="date_to must be >= date_from")

    league = (state.export_full_state_snapshot() or {}).get("league", {})
    games = ((league.get("master_schedule") or {}).get("games") or [])
    game_by_date: Dict[str, Dict[str, Any]] = {}
    for g in games:
        if g.get("home_team_id") != tid and g.get("away_team_id") != tid:
            continue
        d = str(g.get("date") or "")[:10]
        if d < df or d > dt:
            continue
        opp = g.get("away_team_id") if g.get("home_team_id") == tid else g.get("home_team_id")
        game_by_date[d] = {"opponent_team_id": str(opp or "").upper(), "is_home": bool(g.get("home_team_id") == tid)}

    from practice.service import list_team_practice_sessions, resolve_practice_session
    from practice import types as p_types

    with LeagueRepo(db_path) as repo:
        repo.init_db()
        roster_rows = repo.get_team_roster(tid)
        roster_pids = [str(r.get("player_id")) for r in (roster_rows or []) if r.get("player_id")]

        stored = list_team_practice_sessions(
            repo=repo,
            team_id=tid,
            season_year=sy,
            date_from=df,
            date_to=dt,
        )

        from readiness import repo as r_repo
        with repo.transaction() as cur:
            sharp_rows = r_repo.get_player_sharpness_states(cur, roster_pids, season_year=sy) if roster_pids else {}

        high_medical_risk_global = False
        try:
            from app.api.routes import core as core_routes
            overview = await core_routes.api_medical_team_overview(team_id=tid, season_year=sy)
            high_medical_risk_global = int((((overview.get("summary") or {}).get("risk_tier_counts") or {}).get("HIGH") or 0)) > 0
        except Exception:
            high_medical_risk_global = False

        days: List[Dict[str, Any]] = []
        cur_dt = df_dt
        while cur_dt <= dt_dt:
            d = cur_dt.isoformat()
            game_info = game_by_date.get(d)
            is_game_day = game_info is not None

            sess_info = stored.get(d)
            if sess_info is None and not is_game_day:
                with repo.transaction() as cur:
                    resolved = resolve_practice_session(
                        cur,
                        team_id=tid,
                        season_year=sy,
                        date_iso=d,
                        fallback_off_scheme=None,
                        fallback_def_scheme=None,
                        roster_pids=roster_pids,
                        days_to_next_game=None,
                        now_iso=d,
                    )
                session = p_types.normalize_session(resolved)
                is_user_set = False
            elif sess_info is None:
                session = None
                is_user_set = False
            else:
                session = p_types.normalize_session((sess_info.get("session") or {}))
                is_user_set = bool(sess_info.get("is_user_set"))

            low_sharpness_cluster = False
            if sharp_rows:
                vals = [float((sharp_rows.get(pid) or {}).get("sharpness", 50.0) or 50.0) for pid in roster_pids]
                low_sharpness_cluster = sum(1 for v in vals if v < 45.0) >= 3

            high_medical_risk = high_medical_risk_global

            ui_tags: List[str] = []
            if is_game_day:
                ui_tags.append("GAME")
            elif session is not None:
                st = str(session.get("type") or "REST").upper()
                if st == "REST":
                    ui_tags.append("REST")
                ui_tags.append("USER_SET" if is_user_set else "AUTO")
            if high_medical_risk or low_sharpness_cluster:
                ui_tags.append("RISK_HIGH")

            days.append({
                "date_iso": d,
                "is_game_day": is_game_day,
                "opponent_team_id": (game_info or {}).get("opponent_team_id"),
                "is_home": (game_info or {}).get("is_home"),
                "session": None if session is None else {
                    "type": str(session.get("type") or "REST").upper(),
                    "is_user_set": bool(is_user_set),
                },
                "risk_flags": {
                    "high_medical_risk": bool(high_medical_risk),
                    "low_sharpness_cluster": bool(low_sharpness_cluster),
                },
                "ui_tags": ui_tags,
            })
            cur_dt = cur_dt + _dt.timedelta(days=1)

    return {
        "team_id": tid,
        "season_year": sy,
        "date_from": df,
        "date_to": dt,
        "days": days,
    }


@router.post("/api/practice/team/{team_id}/preview-range")
async def api_preview_team_practice_effect_range(team_id: str, req: TeamPracticePreviewRangeRequest):
    """Preview practice effects across multiple dates for a single session template without writes."""
    db_path = state.get_db_path()
    sy = _resolve_season_year(req.season_year)
    tid = str(team_id).upper()

    dates = sorted({game_time.require_date_iso(d, field="dates") for d in (req.dates or [])})
    if not dates:
        raise HTTPException(status_code=400, detail="dates must not be empty")

    session = _normalize_preview_session_payload(req)

    from practice import config as p_cfg
    from practice import types as p_types

    with LeagueRepo(db_path) as repo:
        repo.init_db()
        roster_rows = repo.get_team_roster(tid)
        roster_pids = [str(r.get("player_id")) for r in (roster_rows or []) if r.get("player_id")]

    per_date: List[Dict[str, Any]] = []
    offense_gain_sum = 0.0
    defense_gain_sum = 0.0
    avg_sharpness_values: List[float] = []

    for d in dates:
        by_pid: Dict[str, Dict[str, Any]] = {}
        for pid in roster_pids:
            eff_type = p_types.effective_type_for_pid(session, pid)
            by_pid[pid] = {
                "effective_type": eff_type,
                "intensity_mult": float(p_types.intensity_for_session_type(eff_type)),
                "sharpness_delta": float(p_cfg.SHARPNESS_DELTA.get(eff_type, 0.0) or 0.0),
            }

        sess_type = str(session.get("type") or "FILM").upper()
        fam_gain = float(p_cfg.FAMILIARITY_GAIN.get(sess_type, 0.0) or 0.0)
        offense_gain = fam_gain if sess_type in ("OFF_TACTICS", "FILM", "SCRIMMAGE") else 0.0
        defense_gain = fam_gain if sess_type in ("DEF_TACTICS", "FILM", "SCRIMMAGE") else 0.0
        avg_sharpness_delta = (
            float(sum(float(x.get("sharpness_delta", 0.0)) for x in by_pid.values()) / len(by_pid))
            if by_pid else 0.0
        )

        offense_gain_sum += float(offense_gain)
        defense_gain_sum += float(defense_gain)
        avg_sharpness_values.append(avg_sharpness_delta)

        per_date.append({
            "date_iso": d,
            "familiarity_gain": {
                "offense_gain": float(offense_gain),
                "defense_gain": float(defense_gain),
            },
            "avg_sharpness_delta": float(avg_sharpness_delta),
        })

    return {
        "team_id": tid,
        "season_year": sy,
        "dates": dates,
        "session": session,
        "per_date": per_date,
        "aggregate": {
            "offense_gain_sum": float(offense_gain_sum),
            "defense_gain_sum": float(defense_gain_sum),
            "avg_sharpness_delta_mean": float(sum(avg_sharpness_values) / len(avg_sharpness_values)) if avg_sharpness_values else 0.0,
        },
    }


@router.post("/api/practice/team/{team_id}/preview")
async def api_preview_team_practice_effect(team_id: str, req: TeamPracticePreviewRequest):
    """Preview practice effects for one date/session without any writes."""
    db_path = state.get_db_path()
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(req.season_year or (league_ctx.get("season_year") or 0))
    if sy <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    tid = str(team_id).upper()
    d = game_time.require_date_iso(req.date_iso, field="date_iso")

    from practice import config as p_cfg
    from practice import types as p_types

    session = p_types.normalize_session({
        "type": req.type,
        "offense_scheme_key": req.offense_scheme_key,
        "defense_scheme_key": req.defense_scheme_key,
        "participant_pids": req.participant_pids or [],
        "non_participant_type": req.non_participant_type,
    })

    with LeagueRepo(db_path) as repo:
        repo.init_db()
        roster_rows = repo.get_team_roster(tid)
        roster_pids = [str(r.get("player_id")) for r in (roster_rows or []) if r.get("player_id")]

    by_pid: Dict[str, Dict[str, Any]] = {}
    for pid in roster_pids:
        eff_type = p_types.effective_type_for_pid(session, pid)
        by_pid[pid] = {
            "effective_type": eff_type,
            "intensity_mult": float(p_types.intensity_for_session_type(eff_type)),
            "sharpness_delta": float(p_cfg.SHARPNESS_DELTA.get(eff_type, 0.0) or 0.0),
        }

    sess_type = str(session.get("type") or "FILM").upper()
    fam_gain = float(p_cfg.FAMILIARITY_GAIN.get(sess_type, 0.0) or 0.0)

    return {
        "team_id": tid,
        "season_year": sy,
        "date_iso": d,
        "session": session,
        "preview": {
            "intensity_mult_by_pid": by_pid,
            "sharpness_delta_by_type": dict(p_cfg.SHARPNESS_DELTA),
            "familiarity_gain": {
                "session_type": sess_type,
                "offense_gain": fam_gain if sess_type in ("OFF_TACTICS", "FILM", "SCRIMMAGE") else 0.0,
                "defense_gain": fam_gain if sess_type in ("DEF_TACTICS", "FILM", "SCRIMMAGE") else 0.0,
            },
        },
    }


@router.get("/api/readiness/team/{team_id}/familiarity")
async def api_get_team_familiarity_status(
    team_id: str,
    season_year: Optional[int] = None,
    scheme_type: Optional[str] = None,
    as_of_date: Optional[str] = None,
):
    """Read-only familiarity status for all/specific scheme types."""
    db_path = state.get_db_path()
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(season_year or (league_ctx.get("season_year") or 0))
    if sy <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    tid = str(team_id).upper()
    st_filter = str(scheme_type).strip().lower() if scheme_type is not None else None
    if st_filter not in (None, "offense", "defense"):
        raise HTTPException(status_code=400, detail="scheme_type must be one of: offense, defense")

    as_of = game_time.require_date_iso(as_of_date, field="as_of_date") if as_of_date else None

    from readiness import formulas as r_f
    from readiness import repo as r_repo

    with LeagueRepo(db_path) as repo:
        repo.init_db()
        with repo.transaction() as cur:
            rows = r_repo.list_team_scheme_familiarity_states(
                cur,
                team_id=tid,
                season_year=sy,
                scheme_type=st_filter,
            )

    items: List[Dict[str, Any]] = []
    for (st, sk), row in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        value = float((row or {}).get("value", 50.0) or 50.0)
        last_date = (row or {}).get("last_date")
        item: Dict[str, Any] = {
            "scheme_type": st,
            "scheme_key": sk,
            "value": value,
            "last_date": last_date,
        }
        if as_of is not None:
            last_dt = r_f.parse_date_iso(last_date)
            days = 0
            if last_dt is not None:
                days = max(0, int((_dt.date.fromisoformat(as_of) - last_dt).days))
            item["value_as_of"] = float(r_f.decay_familiarity_exp(value, days=days))
            item["as_of_date"] = as_of
        items.append(item)

    return {"team_id": tid, "season_year": sy, "scheme_type": st_filter, "items": items}


@router.get("/api/readiness/team/{team_id}/sharpness")
async def api_get_team_sharpness_distribution(
    team_id: str,
    season_year: Optional[int] = None,
    as_of_date: Optional[str] = None,
    include_players: bool = False,
):
    """Read-only team sharpness distribution for roster players."""
    db_path = state.get_db_path()
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(season_year or (league_ctx.get("season_year") or 0))
    if sy <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    tid = str(team_id).upper()
    as_of = game_time.require_date_iso(as_of_date, field="as_of_date") if as_of_date else None

    from readiness import formulas as r_f
    from readiness import repo as r_repo

    with LeagueRepo(db_path) as repo:
        repo.init_db()
        roster_rows = repo.get_team_roster(tid)
        roster_items = [
            {"player_id": str(r.get("player_id")), "name": r.get("name")}
            for r in (roster_rows or [])
            if r.get("player_id")
        ]

        pids = [it["player_id"] for it in roster_items]
        with repo.transaction() as cur:
            sharp_rows = r_repo.get_player_sharpness_states(cur, pids, season_year=sy) if pids else {}

    values: List[float] = []
    players: List[Dict[str, Any]] = []
    as_of_dt = _dt.date.fromisoformat(as_of) if as_of else None

    for item in roster_items:
        pid = item["player_id"]
        row = sharp_rows.get(pid) or {}
        base = float(row.get("sharpness", 50.0) or 50.0)
        last_date = row.get("last_date")
        val = base
        if as_of_dt is not None:
            last_dt = r_f.parse_date_iso(last_date)
            days = max(0, int((as_of_dt - last_dt).days)) if last_dt is not None else 0
            val = float(r_f.decay_sharpness_linear(base, days=days))
        values.append(val)
        if include_players:
            players.append({
                "player_id": pid,
                "name": item.get("name"),
                "sharpness": base,
                "sharpness_as_of": val if as_of_dt is not None else None,
                "last_date": last_date,
            })

    n = len(values)
    avg = float(sum(values) / n) if n > 0 else 0.0
    vmin = float(min(values)) if n > 0 else 0.0
    vmax = float(max(values)) if n > 0 else 0.0

    buckets = {"0_39": 0, "40_49": 0, "50_59": 0, "60_69": 0, "70_plus": 0}
    for v in values:
        if v < 40:
            buckets["0_39"] += 1
        elif v < 50:
            buckets["40_49"] += 1
        elif v < 60:
            buckets["50_59"] += 1
        elif v < 70:
            buckets["60_69"] += 1
        else:
            buckets["70_plus"] += 1

    return {
        "team_id": tid,
        "season_year": sy,
        "as_of_date": as_of,
        "distribution": {
            "count": n,
            "avg": avg,
            "min": vmin,
            "max": vmax,
            "low_sharp_count": sum(1 for v in values if v < 45.0),
            "buckets": buckets,
        },
        "players": players if include_players else None,
    }


@router.post("/api/practice/team/session/set")
async def api_set_team_practice_session(req: TeamPracticeSessionRequest):
    """Set a daily practice session (user-authored)."""
    db_path = state.get_db_path()
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(req.season_year or (league_ctx.get("season_year") or 0))
    if sy <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    from practice.service import set_team_practice_session

    now_iso = state.get_current_date_as_date().isoformat()

    session = {
        "type": req.type,
        "offense_scheme_key": req.offense_scheme_key,
        "defense_scheme_key": req.defense_scheme_key,
        "participant_pids": req.participant_pids or [],
        "non_participant_type": req.non_participant_type,
    }

    try:
        d = game_time.require_date_iso(req.date_iso, field="date_iso")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return set_team_practice_session(
        db_path=str(db_path),
        team_id=str(req.team_id).upper(),
        season_year=sy,
        date_iso=d,
        session=session,
        now_iso=now_iso,
        is_user_set=True,
    )
