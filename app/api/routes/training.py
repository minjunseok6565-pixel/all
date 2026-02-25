from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

import game_time
import state
from league_repo import LeagueRepo
from app.schemas.practice import TeamPracticePlanRequest, TeamPracticeSessionRequest
from app.schemas.training import PlayerTrainingPlanRequest, TeamTrainingPlanRequest

router = APIRouter()
logger = logging.getLogger(__name__)








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
