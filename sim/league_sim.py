from __future__ import annotations

import logging
import random
from contextlib import contextmanager
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import schema

from league_repo import LeagueRepo
from matchengine_v2_adapter import (
    adapt_matchengine_result_to_v2,
    build_context_from_master_schedule_entry,
    build_context_from_team_ids,
)
from matchengine_v3.sim_game import simulate_game
import fatigue
import injury
from state import (
    export_full_state_snapshot,
    get_db_path,
    get_league_context_snapshot,
    get_current_date_as_date,
    ingest_game_result,
    set_current_date,
)
from sim.roster_adapter import build_team_state_from_db

logger = logging.getLogger(__name__)

@contextmanager
def _repo_ctx() -> LeagueRepo:
    db_path = get_db_path()
    with LeagueRepo(db_path) as repo:
        yield repo


def _run_match(
    *,
    home_team_id: str,
    away_team_id: str,
    game_date: str,
    home_tactics: Optional[Dict[str, Any]] = None,
    away_tactics: Optional[Dict[str, Any]] = None,
    context: schema.GameContext,
) -> Dict[str, Any]:
    rng = random.Random()
    with _repo_ctx() as repo:
        # Ensure schema is applied (idempotent). This guarantees fatigue/injury tables exist
        # even if the DB was created before the modules were added.
        repo.init_db()

        season_year = fatigue.season_year_from_season_id(str(getattr(context, "season_id", "") or ""))

        # ------------------------------------------------------------
        # Injury: prepare between-game state (OUT players + returning debuffs)
        # - must run BEFORE building TeamState so roster_adapter can exclude/apply debuffs
        # ------------------------------------------------------------
        prepared_inj = None
        unavailable_by_team: Dict[str, Set[str]] = {}
        attrs_mods_by_pid = None
        try:
            prepared_inj = injury.prepare_game_injuries(
                repo,
                game_id=str(getattr(context, "game_id", "") or ""),
                game_date_iso=str(game_date),
                season_year=int(season_year),
                home_team_id=str(home_team_id),
                away_team_id=str(away_team_id),
            )
            unavailable_by_team = dict(prepared_inj.unavailable_pids_by_team or {})
            attrs_mods_by_pid = prepared_inj.attrs_mods_by_pid
        except Exception:
            logger.warning(
                "INJURY_PREPARE_FAILED game_date=%s home=%s away=%s",
                game_date,
                str(home_team_id),
                str(away_team_id),
                exc_info=True,
            )
            prepared_inj = None

        hid = schema.normalize_team_id(str(home_team_id)).upper()
        aid = schema.normalize_team_id(str(away_team_id)).upper()

        home = build_team_state_from_db(
            repo=repo,
            team_id=home_team_id,
            tactics=home_tactics,
            exclude_pids=set(unavailable_by_team.get(hid, set()) or set()),
            attrs_mods_by_pid=attrs_mods_by_pid,
        )
        away = build_team_state_from_db(
            repo=repo,
            team_id=away_team_id,
            tactics=away_tactics,
            exclude_pids=set(unavailable_by_team.get(aid, set()) or set()),
            attrs_mods_by_pid=attrs_mods_by_pid,
        )

        # ------------------------------------------------------------
        # Fatigue: prepare between-game condition (start_energy + energy_cap)
        # ------------------------------------------------------------
        prepared_fat = None
        try:
            prepared_fat = fatigue.prepare_game_fatigue(
                repo,
                game_date_iso=game_date,
                season_year=int(season_year),
                home=home,
                away=away,
            )
        except Exception:
            logger.warning(
                "FATIGUE_PREPARE_FAILED game_date=%s home=%s away=%s",
                game_date,
                str(home_team_id),
                str(away_team_id),
                exc_info=True,
            )

        # ------------------------------------------------------------
        # Injury: in-game hook (segment-level) + simulate
        # ------------------------------------------------------------
        injury_hook = None
        if prepared_inj is not None:
            try:
                injury_hook = injury.make_in_game_injury_hook(prepared_inj, context=context, home=home, away=away)
            except Exception:
                logger.warning(
                    "INJURY_HOOK_BUILD_FAILED game_date=%s home=%s away=%s",
                    game_date,
                    str(home_team_id),
                    str(away_team_id),
                    exc_info=True,
                )
                injury_hook = None

        raw_result = simulate_game(rng, home, away, context=context, injury_hook=injury_hook)

        # ------------------------------------------------------------
        # Post-game finalize: fatigue + injuries
        # ------------------------------------------------------------
        if prepared_fat is not None:
            try:
                fatigue.finalize_game_fatigue(
                    repo,
                    prepared=prepared_fat,
                    home=home,
                    away=away,
                    raw_result=raw_result,
                )
            except Exception:
                logger.warning(
                    "FATIGUE_FINALIZE_FAILED game_date=%s home=%s away=%s",
                    game_date,
                    str(home_team_id),
                    str(away_team_id),
                    exc_info=True,
                )

        if prepared_inj is not None:
            try:
                injury.finalize_game_injuries(
                    repo,
                    prepared=prepared_inj,
                    home=home,
                    away=away,
                    raw_result=raw_result,
                )
            except Exception:
                logger.warning(
                    "INJURY_FINALIZE_FAILED game_date=%s home=%s away=%s",
                    game_date,
                    str(home_team_id),
                    str(away_team_id),
                    exc_info=True,
                )
    v2_result = adapt_matchengine_result_to_v2(
        raw_result,
        context,
        engine_name="matchengine_v3",
    )
    return ingest_game_result(game_result=v2_result, game_date=game_date)

def advance_league_until(
    target_date_str: str,
    user_team_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    league_full = export_full_state_snapshot().get("league", {})
    master_schedule = league_full.get("master_schedule", {})
    by_date: Dict[str, List[str]] = master_schedule.get("by_date") or {}
    games: List[Dict[str, Any]] = master_schedule.get("games") or []

    if not by_date or not games:
        raise RuntimeError(
            "Master schedule is not initialized. Expected state.startup_init_state() to run before calling advance_league_until()."
        )

    
    try:
        target_date = date.fromisoformat(target_date_str)
    except ValueError as exc:
        raise ValueError(f"invalid target_date: {target_date_str}") from exc

    league_context = get_league_context_snapshot()
    current_date_str = league_context.get("current_date")
    if current_date_str:
        try:
            current_date = date.fromisoformat(current_date_str)
        except ValueError:
            current_date = target_date
    else:
        if league_context.get("season_start"):
            try:
                season_start = date.fromisoformat(league_context["season_start"])
            except ValueError:
                season_start = target_date
        else:
            season_start = target_date
        current_date = season_start - timedelta(days=1)

    simulated_game_objs: List[Dict[str, Any]] = []
    user_team_upper = user_team_id.upper() if user_team_id else None

    day = current_date + timedelta(days=1)
    last_month_key: Optional[str] = None
    while day <= target_date:
        day_str = day.isoformat()

        # Monthly NBA growth tick (process previous month once we enter a new month).
        month_key = day_str[:7]
        if month_key != last_month_key:
            from training.checkpoints import maybe_run_monthly_growth_tick

            maybe_run_monthly_growth_tick(db_path=get_db_path(), game_date_iso=day_str)
            last_month_key = month_key

        game_ids = by_date.get(day_str, [])
        if not game_ids:
            day += timedelta(days=1)
            continue

        for gid in game_ids:
            g = next((x for x in games if x.get("game_id") == gid), None)
            if not g:
                continue
            if g.get("status") == "final":
                continue

            home_id = str(g["home_team_id"]).upper()
            away_id = str(g["away_team_id"]).upper()

            if user_team_upper and (home_id == user_team_upper or away_id == user_team_upper):
                continue

            # SSOT: context date must come from entry["date"] (no override arg in builder).
            entry_for_ctx = dict(g)
            entry_for_ctx["date"] = day_str
            context = build_context_from_master_schedule_entry(
                entry=entry_for_ctx,
                league_state=league_context,
                phase=str(g.get("phase") or "regular"),
            )

            game_obj = _run_match(
                home_team_id=home_id,
                away_team_id=away_id,
                game_date=day_str,
                context=context,
            )
            simulated_game_objs.append(game_obj)

        day += timedelta(days=1)

    set_current_date(target_date_str)
    return simulated_game_objs


def simulate_single_game(
    home_team_id: str,
    away_team_id: str,
    game_date: Optional[str] = None,
    home_tactics: Optional[Dict[str, Any]] = None,
    away_tactics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    league_context = get_league_context_snapshot()
    game_date_str = game_date or get_current_date_as_date().isoformat()

    # Ensure monthly growth tick is applied (idempotent).
    try:
        from training.checkpoints import maybe_run_monthly_growth_tick

        maybe_run_monthly_growth_tick(db_path=get_db_path(), game_date_iso=game_date_str)
    except Exception:
        # Growth tick must never crash games.
        logger.warning("MONTHLY_GROWTH_TICK_FAILED date=%s", game_date_str, exc_info=True)

    # Ensure monthly agency tick is applied (idempotent).
    try:
        from agency.checkpoints import maybe_run_monthly_agency_tick

        maybe_run_monthly_agency_tick(db_path=get_db_path(), game_date_iso=game_date_str)
    except Exception:
        # Agency tick must never crash games.
        logger.warning("MONTHLY_AGENCY_TICK_FAILED date=%s", game_date_str, exc_info=True)

    game_id = f"single_{home_team_id}_{away_team_id}_{uuid4().hex[:8]}"

    context = build_context_from_team_ids(
        game_id,
        game_date_str,
        home_team_id,
        away_team_id,
        league_context,
        phase="regular",
    )

    return _run_match(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        game_date=game_date_str,
        home_tactics=home_tactics,
        away_tactics=away_tactics,
        context=context,
    )
