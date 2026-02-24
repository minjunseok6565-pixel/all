from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import date, timedelta
from typing import Any, Dict, Optional, List, Literal
from uuid import uuid4

import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import game_time
from config import BASE_DIR, ALL_TEAM_IDS
from league_repo import LeagueRepo
from league_service import LeagueService, CapViolationError
from schema import normalize_team_id, normalize_player_id
import state
from sim.league_sim import simulate_single_game, advance_league_until
from postseason.director import (
    auto_advance_current_round,
    advance_my_team_one_game,
    build_postseason_field,
    initialize_postseason,
    play_my_team_play_in_game,
    reset_postseason_state,
)
from news_ai import refresh_playoff_news, refresh_weekly_news
from analytics.stats.leaders import compute_leaderboards
from team_utils import get_conference_standings, get_team_cards, get_team_detail, ui_cache_rebuild_all, ui_cache_refresh_players

from college.ui import (
    get_college_meta,
    get_college_team_cards,
    get_college_team_detail,
    list_college_players,
    get_college_player_detail,
    get_college_draft_pool,
)
from season_report_ai import generate_season_report
from trades.errors import TradeError
from trades.models import canonicalize_deal, parse_deal, serialize_deal
from trades.validator import validate_deal
from trades.apply import apply_deal_to_db
from trades import agreements
from trades import negotiation_store
from save_service import (
    SaveError,
    create_new_game,
    get_save_slot_detail,
    list_save_slots,
    load_game,
    save_game,
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# FastAPI 앱 생성 및 기본 설정
# -------------------------------------------------------------------------
app = FastAPI(title="느바 시뮬 GM 서버")

@app.on_event("startup")
def _startup_init_state() -> None:
    # 1) DB init + seed once (per db_path)
    # 2) SSOT state init: season/schedule + cap model
    # 3) repo integrity validate once (per db_path)
    # 4) ingest_turn backfill once (per state instance)
    # 5) UI-only cache bootstrap (derived, non-authoritative)
    db_path = os.environ.get("LEAGUE_DB_PATH")
    if not db_path:
        raise RuntimeError("LEAGUE_DB_PATH is required (no default db_path).")
    state.set_db_path(db_path)

    state.startup_init_state()

    # Explicit UI-only cache bootstrap (derived, non-authoritative).
    # Ensures team/player UI metadata exists from server boot without requiring any read path to "init".
    try:
        ui_cache_rebuild_all()
    except Exception as e:
        raise RuntimeError(f"ui_cache_rebuild_all() failed during startup: {e}") from e

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _auth_guard_middleware(request: Request, call_next):
    """Optional commercial auth guard.

    If NBA_SIM_ADMIN_TOKEN is configured, require it on state-changing API calls.
    """
    required_token = (os.environ.get("NBA_SIM_ADMIN_TOKEN") or "").strip()
    if not required_token:
        return await call_next(request)

    path = request.url.path or ""
    method = (request.method or "GET").upper()
    if method != "POST" or not path.startswith("/api/"):
        return await call_next(request)

    # Keep health/auth bootstrap available.
    if path in {"/api/validate-key"}:
        return await call_next(request)

    provided = (request.headers.get("X-Admin-Token") or "").strip()
    if provided != required_token:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized: invalid X-Admin-Token"})

    return await call_next(request)

# static/NBA.html 서빙
static_dir = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """간단한 헬스체크 및 NBA.html 링크 안내."""
    index_path = os.path.join(static_dir, "NBA.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "느바 시뮬 GM 서버입니다. /static/NBA.html 을 확인하세요."}


# -------------------------------------------------------------------------
# Pydantic 모델 정의
# -------------------------------------------------------------------------
class SimGameRequest(BaseModel):
    home_team_id: str
    away_team_id: str
    home_tactics: Optional[Dict[str, Any]] = None
    away_tactics: Optional[Dict[str, Any]] = None
    game_date: Optional[str] = None  # 인게임 날짜 (YYYY-MM-DD)


class TeamTrainingPlanRequest(BaseModel):
    team_id: str
    season_year: Optional[int] = None
    focus: Optional[str] = None
    intensity: Optional[str] = None
    weights: Optional[Dict[str, float]] = None


class PlayerTrainingPlanRequest(BaseModel):
    player_id: str
    season_year: Optional[int] = None
    primary: Optional[str] = None
    secondary: Optional[str] = None
    intensity: Optional[str] = None


class TeamPracticePlanRequest(BaseModel):
    team_id: str
    season_year: Optional[int] = None
    mode: Optional[str] = None  # AUTO | MANUAL


class TeamPracticeSessionRequest(BaseModel):
    team_id: str
    season_year: Optional[int] = None
    date_iso: str  # YYYY-MM-DD
    type: Optional[str] = None
    offense_scheme_key: Optional[str] = None
    defense_scheme_key: Optional[str] = None
    participant_pids: Optional[List[str]] = None
    non_participant_type: Optional[str] = None


class ChatMainRequest(BaseModel):
    apiKey: str
    userInput: str = Field(..., alias="userMessage")
    mainPrompt: Optional[str] = ""
    context: Any = ""

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True
        fields = {"userInput": "userMessage"}


class AdvanceLeagueRequest(BaseModel):
    target_date: str  # YYYY-MM-DD, 이 날짜까지 리그를 자동 진행
    user_team_id: Optional[str] = None
    apiKey: Optional[str] = None  # Optional: used for month-end scouting LLM generation


class PostseasonSetupRequest(BaseModel):
    my_team_id: str
    use_random_field: bool = False


class EmptyRequest(BaseModel):
    pass




class GameNewRequest(BaseModel):
    slot_name: str
    slot_id: Optional[str] = None
    user_team_id: Optional[str] = None
    season_year: Optional[int] = None
    overwrite_if_exists: bool = False


class GameSaveRequest(BaseModel):
    slot_id: str
    save_name: Optional[str] = None
    note: Optional[str] = None


class GameLoadRequest(BaseModel):
    slot_id: str
    strict: bool = True
    expected_save_version: Optional[int] = None


class OffseasonContractsProcessRequest(BaseModel):
    user_team_id: str


class TeamOptionPendingRequest(BaseModel):
    user_team_id: str


class TeamOptionDecisionItem(BaseModel):
    contract_id: str
    decision: Literal["EXERCISE", "DECLINE"]


class TeamOptionDecideRequest(BaseModel):
    user_team_id: str
    decisions: List[TeamOptionDecisionItem] = Field(default_factory=list)


class DraftCombineRequest(BaseModel):
    rng_seed: Optional[int] = None


class DraftWorkoutsRequest(BaseModel):
    # User-controlled workouts:
    # - If invited_prospect_temp_ids is empty => skip (no DB writes)
    # - Otherwise, run only for this team (no league-wide generation)
    team_id: str
    invited_prospect_temp_ids: List[str] = Field(default_factory=list)
    max_invites: int = 12
    rng_seed: Optional[int] = None


class DraftInterviewItem(BaseModel):
    prospect_temp_id: str
    selected_question_ids: List[str] = Field(default_factory=list)


class DraftInterviewsRequest(BaseModel):
    # User-controlled interviews:
    # - 'interviews' contains per-prospect selected questions (typically 3)
    # - This endpoint only writes results for this team (private info)
    team_id: str
    interviews: List[DraftInterviewItem] = Field(default_factory=list)
    rng_seed: Optional[int] = None


class DraftAutoSelectionsRequest(BaseModel):
    max_picks: Optional[int] = None
    stop_on_user_controlled_team_ids: Optional[List[str]] = None
    allow_autopick_user_team: bool = False


class DraftRecordPickRequest(BaseModel):
    prospect_temp_id: str
    source: str = "draft_user"
    meta: Optional[Dict[str, Any]] = None


class DraftWatchRecomputeRequest(BaseModel):
    draft_year: Optional[int] = None
    as_of_date: Optional[str] = None       # YYYY-MM-DD (default: current in-game date)
    period_key: Optional[str] = None       # YYYY-MM (default: as_of_date[:7])
    season_year: Optional[int] = None      # stats season used (default: draft_year - 1)
    min_inclusion_prob: Optional[float] = None  # default: 0.35
    force: bool = False


class ScoutingAssignRequest(BaseModel):
    team_id: str
    scout_id: str
    player_id: str
    assigned_date: Optional[str] = None  # YYYY-MM-DD (default: in-game date)
    target_kind: Literal["COLLEGE"] = "COLLEGE"


class ScoutingUnassignRequest(BaseModel):
    team_id: str
    assignment_id: Optional[str] = None
    scout_id: Optional[str] = None
    ended_date: Optional[str] = None  # YYYY-MM-DD (default: in-game date)


class WeeklyNewsRequest(BaseModel):
    apiKey: str


class ApiKeyRequest(BaseModel):
    apiKey: str


class SeasonReportRequest(BaseModel):
    apiKey: str
    user_team_id: str


class TradeSubmitRequest(BaseModel):
    deal: Dict[str, Any]


class TradeSubmitCommittedRequest(BaseModel):
    deal_id: str


class TradeNegotiationStartRequest(BaseModel):
    user_team_id: str
    other_team_id: str


class TradeNegotiationCommitRequest(BaseModel):
    session_id: str
    deal: Dict[str, Any]

class TradeEvaluateRequest(BaseModel):
    deal: Dict[str, Any]
    team_id: str
    include_breakdown: bool = True


# -------------------------------------------------------------------------
# Contracts / Roster Write API models
# -------------------------------------------------------------------------
class ReleaseToFARequest(BaseModel):
    player_id: str
    released_date: Optional[str] = None  # YYYY-MM-DD (default: in-game date)


class SignFreeAgentRequest(BaseModel):
    session_id: str  # must reference an ACCEPTED contract negotiation session
    team_id: str
    player_id: str
    signed_date: Optional[str] = None  # YYYY-MM-DD (default: in-game date)
    years: int = 1
    salary_by_year: Optional[Dict[int, int]] = None  # {season_year: salary}
    team_option_years: Optional[List[int]] = None  # Absolute season_years; must be tail-consecutive and include final year
    # Deprecated shorthand; prefer team_option_years.
    team_option_last_year: bool = False  # If True, last year is a TEAM option (PENDING)


class ReSignOrExtendRequest(BaseModel):
    session_id: str  # must reference an ACCEPTED contract negotiation session
    team_id: str
    player_id: str
    signed_date: Optional[str] = None  # YYYY-MM-DD (default: in-game date)
    years: int = 1
    salary_by_year: Optional[Dict[int, int]] = None  # {season_year: salary}
    team_option_years: Optional[List[int]] = None  # Absolute season_years; must be tail-consecutive and include final year
    # Deprecated shorthand; prefer team_option_years.
    team_option_last_year: bool = False  # If True, last year is a TEAM option (PENDING)


class ContractNegotiationStartRequest(BaseModel):
    team_id: str
    player_id: str
    mode: str = "SIGN_FA"  # SIGN_FA | RE_SIGN | EXTEND
    valid_days: Optional[int] = 7  # in-game days the offer window stays open (best-effort)


class ContractNegotiationOfferRequest(BaseModel):
    session_id: str
    offer: Dict[str, Any]  # see contracts.negotiation.types.ContractOffer.from_payload


class ContractNegotiationAcceptCounterRequest(BaseModel):
    session_id: str


class ContractNegotiationCommitRequest(BaseModel):
    session_id: str
    signed_date: Optional[str] = None  # YYYY-MM-DD (default: in-game date)


class ContractNegotiationCancelRequest(BaseModel):
    session_id: str
    reason: Optional[str] = None


class AgencyEventRespondRequest(BaseModel):
    user_team_id: str
    event_id: str
    response_type: str
    response_payload: Optional[Dict[str, Any]] = None
    now_date: Optional[str] = None  # YYYY-MM-DD (default: in-game date)


class AgencyUserActionRequest(BaseModel):
    user_team_id: str
    player_id: str
    action_type: str
    action_payload: Optional[Dict[str, Any]] = None
    now_date: Optional[str] = None  # YYYY-MM-DD (default: in-game date)


# -------------------------------------------------------------------------
# 유틸: Gemini 응답 텍스트 추출
# -------------------------------------------------------------------------
def extract_text_from_gemini_response(resp: Any) -> str:
    """google-generativeai 응답 객체에서 텍스트만 안전하게 뽑아낸다."""
    text = getattr(resp, "text", None)
    if text:
        return text

    try:
        parts = resp.candidates[0].content.parts
        texts = []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                texts.append(t)
        if texts:
            return "\n".join(texts)
    except Exception:
        pass

    return str(resp)


# -------------------------------------------------------------------------
# 경기 시뮬레이션 API
# -------------------------------------------------------------------------
@app.post("/api/simulate-game")
async def api_simulate_game(req: SimGameRequest):
    """matchengine_v3를 사용해 한 경기를 시뮬레이션한다.

    NOTE (SSOT 계약):
    - Home/Away SSOT는 league_sim.simulate_single_game 내부에서 GameContext로 생성/주입된다.
    - server는 엔진을 직접 호출하지 않으며(직접 호출 금지), 결과는 어댑터+validator 관문을 통과한 V2만 반환한다.
    """
    try:
        result = simulate_single_game(
            home_team_id=req.home_team_id,
            away_team_id=req.away_team_id,
            game_date=req.game_date,
            home_tactics=req.home_tactics,
            away_tactics=req.away_tactics,
        )
        return result
    except ValueError as e:
        # 팀을 찾지 못한 경우 등
        raise HTTPException(status_code=404, detail=str(e))


# -------------------------------------------------------------------------
# 리그 자동 진행 API (다른 팀 경기 일괄 시뮬레이션)
# -------------------------------------------------------------------------
@app.post("/api/advance-league")
async def api_advance_league(req: AdvanceLeagueRequest):
    """target_date까지 (유저 팀 경기를 제외한) 리그 전체 경기를 자동 시뮬레이션."""
    prev_date = state.get_current_date_as_date().isoformat()
    try:
        simulated = advance_league_until(
            target_date_str=req.target_date,
            user_team_id=req.user_team_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    db_path = state.get_db_path()

    # 2차: 월별 대학 스탯 스냅샷(변동성 모델) + watch-run(사전 빅보드) 체크포인트 갱신
    try:
        from college.service import run_monthly_watch_and_stats_checkpoints

        college_checkpoints = run_monthly_watch_and_stats_checkpoints(
            str(db_path),
            from_date=str(prev_date),
            to_date=str(req.target_date),
            min_inclusion_prob=0.35,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"college monthly checkpoints failed: {e}") from e

    # 3차: 월별 스카우팅 리포트 체크포인트(유저 선택 기반)
    # - ACTIVE assignment가 없으면 no-op이어야 한다.
    # - 월말 기준 14일 이내 배정된 스카우터는 해당 월 리포트를 작성하지 않는다.
    try:
        from scouting.service import run_monthly_scouting_checkpoints

        scouting_checkpoints = run_monthly_scouting_checkpoints(
            str(db_path),
            from_date=str(prev_date),
            to_date=str(req.target_date),
            api_key=req.apiKey,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"scouting monthly checkpoints failed: {e}") from e
    
    return {
        "target_date": req.target_date,
        "simulated_count": len(simulated),
        "simulated_games": simulated,
        "college_checkpoints": college_checkpoints,
        "scouting_checkpoints": scouting_checkpoints,
    }


# -------------------------------------------------------------------------
# Training / Growth API (plans)
# -------------------------------------------------------------------------


@app.get("/api/training/team/{team_id}")
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


@app.post("/api/training/team/set")
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


@app.get("/api/training/player/{player_id}")
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


@app.post("/api/training/player/set")
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


@app.get("/api/practice/team/{team_id}/plan")
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


@app.post("/api/practice/team/plan/set")
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


@app.get("/api/practice/team/{team_id}/session")
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


@app.get("/api/practice/team/{team_id}/sessions")
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


@app.post("/api/practice/team/session/set")
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


# -------------------------------------------------------------------------
# 리그 리더 / 스탠딩 / 팀 API
# -------------------------------------------------------------------------


@app.get("/api/stats/leaders")
async def api_stats_leaders():
    """Regular-season per-game leaders.

    Note:
        - We intentionally keep the payload small (top 5, no ties) because this endpoint
          is commonly used as a quick "at-a-glance" widget.
        - This endpoint no longer depends on legacy `stats_util.py` facades.
    """
    workflow_state = state.export_workflow_state() or {}
    if not isinstance(workflow_state, dict):
        workflow_state = {}

    player_stats = workflow_state.get("player_stats") or {}
    team_stats = workflow_state.get("team_stats") or {}

    cfg = {
        "top_n": 5,
        "include_ties": False,
        "modes": ["per_game"],
        "metric_keys": ["PTS", "AST", "REB", "3PM"],
    }
    bundle = compute_leaderboards(player_stats, team_stats, phase="regular", config=cfg)
    leaders = bundle.get("per_game") or {}

    current_date = state.get_current_date()
    return {"leaders": leaders, "updated_at": current_date}


@app.get("/api/stats/playoffs/leaders")
async def api_playoff_stats_leaders():
    """Playoff per-game leaders (same small payload as regular season)."""
    workflow_state = state.export_workflow_state() or {}
    if not isinstance(workflow_state, dict):
        workflow_state = {}

    phase_results = workflow_state.get("phase_results") or {}
    if not isinstance(phase_results, dict):
        phase_results = {}

    playoffs = phase_results.get("playoffs") or {}
    if not isinstance(playoffs, dict):
        playoffs = {}

    player_stats = playoffs.get("player_stats") or {}
    team_stats = playoffs.get("team_stats") or {}

    cfg = {
        "top_n": 5,
        "include_ties": False,
        "modes": ["per_game"],
        "metric_keys": ["PTS", "AST", "REB", "3PM"],
    }
    bundle = compute_leaderboards(player_stats, team_stats, phase="playoffs", config=cfg)
    leaders = bundle.get("per_game") or {}
    current_date = state.get_current_date()
    return {"leaders": leaders, "updated_at": current_date}


@app.get("/api/standings")
async def api_standings():
    return get_conference_standings()


@app.get("/api/teams")
async def api_teams():
    return get_team_cards()


@app.get("/api/team-detail/{team_id}")
async def api_team_detail(team_id: str):
    try:
        return get_team_detail(team_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# -------------------------------------------------------------------------
# College (Read-only / UI) API
# -------------------------------------------------------------------------


@app.get("/api/college/meta")
async def api_college_meta():
    try:
        return get_college_meta()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/college/teams")
async def api_college_teams(season_year: Optional[int] = None):
    try:
        return get_college_team_cards(season_year=season_year)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/college/team-detail/{college_team_id}")
async def api_college_team_detail(
    college_team_id: str,
    season_year: Optional[int] = None,
    include_attrs: bool = False,
):
    try:
        return get_college_team_detail(
            college_team_id,
            season_year=season_year,
            include_attrs=include_attrs,
        )
    except ValueError as e:
        msg = str(e)
        status = 404 if "not found" in msg.lower() else 400
        raise HTTPException(status_code=status, detail=msg)


@app.get("/api/college/players")
async def api_college_players(
    season_year: Optional[int] = None,
    status: Optional[str] = None,
    college_team_id: Optional[str] = None,
    draft_year: Optional[int] = None,
    declared_only: bool = False,
    q: Optional[str] = None,
    sort: str = "pts",
    order: str = "desc",
    include_attrs: bool = False,
    include_decision: bool = False,
    limit: int = 200,
    offset: int = 0,
):
    try:
        return list_college_players(
            season_year=season_year,
            status=status,
            college_team_id=college_team_id,
            draft_year=draft_year,
            declared_only=declared_only,
            q=q,
            sort=sort,
            order=order,
            include_attrs=include_attrs,
            include_decision=include_decision,
            limit=limit,
            offset=offset,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/college/player/{player_id}")
async def api_college_player(
    player_id: str,
    draft_year: Optional[int] = None,
    include_stats_history: bool = True,
):
    try:
        return get_college_player_detail(
            player_id,
            draft_year=draft_year,
            include_stats_history=include_stats_history,
        )
    except ValueError as e:
        msg = str(e)
        status = 404 if "not found" in msg.lower() else 400
        raise HTTPException(status_code=status, detail=msg)


@app.get("/api/college/draft-pool/{draft_year}")
async def api_college_draft_pool(
    draft_year: int,
    season_year: Optional[int] = None,
    limit: Optional[int] = None,
    pool_mode: Optional[str] = "auto",
    watch_run_id: Optional[str] = None,
    watch_min_prob: Optional[float] = None,
):
    try:
        return get_college_draft_pool(
            draft_year,
            season_year=season_year,
            limit=limit,
            pool_mode=pool_mode or "auto",
            watch_run_id=watch_run_id,
            watch_min_prob=watch_min_prob,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/college/draft-watch/recompute")
async def api_college_draft_watch_recompute(req: DraftWatchRecomputeRequest):
    """(Dev/Admin) Recompute a pre-declaration watch snapshot for a given draft_year/period.

    This writes:
      - draft_watch_runs
      - draft_watch_probs

    It does NOT affect the declared pool (college_draft_entries).
    """
    # draft_year default: state.season_year + 1
    if req.draft_year is None:
        league_ctx = state.get_league_context_snapshot() or {}
        try:
            from_year = int(league_ctx.get("season_year") or 0)
        except Exception:
            from_year = 0
        if from_year <= 0:
            raise HTTPException(status_code=500, detail="Invalid season_year in state (draft_year not provided).")
        dy = int(from_year) + 1
    else:
        try:
            dy = int(req.draft_year)
        except Exception:
            raise HTTPException(status_code=400, detail="draft_year must be an integer.")
        if dy <= 0:
            raise HTTPException(status_code=400, detail="draft_year must be > 0.")

    # as_of_date default: current in-game date
    as_of = str(req.as_of_date or state.get_current_date_as_date().isoformat())
    period_key = str(req.period_key or as_of[:7])

    # season_year default: draft_year - 1
    sy = int(req.season_year) if req.season_year is not None else (dy - 1)
    if sy <= 0:
        raise HTTPException(status_code=400, detail="season_year must be > 0.")

    min_prob = float(req.min_inclusion_prob) if req.min_inclusion_prob is not None else 0.35
    force = bool(req.force)

    try:
        from college.service import recompute_draft_watch_run

        db_path = state.get_db_path()
        return recompute_draft_watch_run(
            str(db_path),
            draft_year=int(dy),
            as_of_date=as_of,
            period_key=period_key,
            season_year=int(sy),
            min_inclusion_prob=float(min_prob),
            force=force,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to recompute draft watch run: {e}") from e


# -------------------------------------------------------------------------
# Scouting (In-season; user-controlled; private to viewer team)
# -------------------------------------------------------------------------


@app.get("/api/scouting/scouts/{team_id}")
async def api_scouting_list_scouts(team_id: str):
    """List scouts for a given team (seeded staff) + current ACTIVE assignment if any."""
    try:
        tid = str(normalize_team_id(team_id, strict=True))
        db_path = state.get_db_path()

        scouts: List[Dict[str, Any]] = []
        with LeagueRepo(db_path) as repo:
            repo.init_db()
            rows = repo._conn.execute(
                """
                SELECT
                    s.scout_id, s.display_name, s.specialty_key, s.profile_json, s.is_active,
                    a.assignment_id, a.target_player_id, a.assigned_date
                FROM scouting_scouts s
                LEFT JOIN scouting_assignments a
                    ON a.scout_id = s.scout_id
                   AND a.status = 'ACTIVE'
                WHERE s.team_id = ?
                ORDER BY s.specialty_key ASC, s.display_name ASC;
                """,
                (tid,),
            ).fetchall()

        for r in rows:
            scout_id = str(r[0])
            display_name = str(r[1] or "Scout")
            specialty_key = str(r[2] or "GENERAL")
            profile_raw = r[3]
            is_active = int(r[4] or 0)

            assignment_id = r[5]
            target_player_id = r[6]
            assigned_date = r[7]

            profile: Dict[str, Any] = {}
            try:
                if profile_raw:
                    profile = json.loads(str(profile_raw))
            except Exception:
                profile = {}

            # Expose only high-level profile info (avoid leaking raw tuning numbers).
            profile_public = {
                "focus_axes": profile.get("focus_axes") if isinstance(profile.get("focus_axes"), list) else [],
                "style_tags": profile.get("style_tags") if isinstance(profile.get("style_tags"), list) else [],
            }

            scouts.append(
                {
                    "scout_id": scout_id,
                    "display_name": display_name,
                    "specialty_key": specialty_key,
                    "is_active": bool(is_active),
                    "profile": profile_public,
                    "active_assignment": (
                        {
                            "assignment_id": str(assignment_id),
                            "target_player_id": str(target_player_id),
                            "assigned_date": str(assigned_date)[:10] if assigned_date else None,
                        }
                        if assignment_id
                        else None
                    ),
                }
            )

        return {"ok": True, "team_id": tid, "scouts": scouts}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list scouts: {e}") from e


@app.post("/api/scouting/assign")
async def api_scouting_assign(req: ScoutingAssignRequest):
    """Assign a scout to a college player (user-driven).

    Policy:
      - If the scout already has an ACTIVE assignment, return 409 (user should unassign first).
      - This endpoint does NOT generate reports immediately; reports are generated at month end.
    """
    try:
        tid = str(normalize_team_id(req.team_id, strict=True))
        scout_id = str(req.scout_id or "").strip()
        if not scout_id:
            raise HTTPException(status_code=400, detail="scout_id is required.")

        pid = str(normalize_player_id(req.player_id, strict=True))

        # assigned_date defaults to in-game date
        assigned_date = game_time.require_date_iso(req.assigned_date or state.get_current_date_as_date().isoformat(), field="assigned_date")
        now = game_time.now_utc_like_iso()
        db_path = state.get_db_path()

        created_assignment: Dict[str, Any] = {}

        with LeagueRepo(db_path) as repo:
            repo.init_db()
            with repo.transaction() as cur:
                # Validate scout exists + belongs to team
                srow = cur.execute(
                    """
                    SELECT scout_id, team_id, is_active
                    FROM scouting_scouts
                    WHERE scout_id=?
                    LIMIT 1;
                    """,
                    (scout_id,),
                ).fetchone()
                if not srow:
                    raise HTTPException(status_code=404, detail=f"Scout not found: {scout_id}")
                if str(srow[1]) != tid:
                    raise HTTPException(status_code=400, detail="Scout does not belong to the given team_id.")
                if int(srow[2] or 0) != 1:
                    raise HTTPException(status_code=409, detail="Scout is inactive.")

                # Guard: one active assignment per scout
                arow = cur.execute(
                    """
                    SELECT assignment_id, target_player_id
                    FROM scouting_assignments
                    WHERE team_id=? AND scout_id=? AND status='ACTIVE'
                    LIMIT 1;
                    """,
                    (tid, scout_id),
                ).fetchone()
                if arow:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Scout already assigned (assignment_id={arow[0]}, target_player_id={arow[1]}). Unassign first.",
                    )

                # Validate player exists (college only for now)
                prow = cur.execute(
                    """
                    SELECT player_id, name, pos, college_team_id, class_year, status
                    FROM college_players
                    WHERE player_id=?
                    LIMIT 1;
                    """,
                    (pid,),
                ).fetchone()
                if not prow:
                    raise HTTPException(status_code=404, detail=f"College player not found: {pid}")

                assignment_id = f"SASN_{uuid4().hex}"
                # Assignment progress state (scouting v2):
                #   - "signals": per-signal mu/sigma updated by monthly checkpoints
                progress = {
                    "schema_version": 2,
                    "signals": {},
                    "last_obs_date": None,
                    "total_obs_days": 0,
                }

                try:
                    cur.execute(
                        """
                        INSERT INTO scouting_assignments(
                            assignment_id, team_id, scout_id, target_player_id, target_kind,
                            assigned_date, status, ended_date, progress_json,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', NULL, ?, ?, ?);
                        """,
                        (
                            assignment_id,
                            tid,
                            scout_id,
                            pid,
                            str(req.target_kind or "COLLEGE"),
                            assigned_date,
                            json.dumps(progress, ensure_ascii=False),
                            now,
                            now,
                        ),
                    )
                except sqlite3.IntegrityError as e:
                    # Covers the partial unique index: uq_scouting_active_assignment_per_scout
                    raise HTTPException(status_code=409, detail=f"Assignment conflict: {e}") from e

                created_assignment = {
                    "assignment_id": assignment_id,
                    "team_id": tid,
                    "scout_id": scout_id,
                    "target_player_id": pid,
                    "target_kind": str(req.target_kind or "COLLEGE"),
                    "assigned_date": assigned_date,
                    "player": {
                        "player_id": str(prow[0]),
                        "name": str(prow[1] or ""),
                        "pos": str(prow[2] or ""),
                        "college_team_id": str(prow[3] or ""),
                        "class_year": int(prow[4] or 0),
                        "status": str(prow[5] or ""),
                    },
                }

        return {"ok": True, "assignment": created_assignment}

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to assign scout: {e}") from e


@app.post("/api/scouting/unassign")
async def api_scouting_unassign(req: ScoutingUnassignRequest):
    """End an ACTIVE scouting assignment (user-driven).

    You can specify either:
      - assignment_id
      - (team_id + scout_id) to end the active assignment for that scout
    """
    try:
        tid = str(normalize_team_id(req.team_id, strict=True))
        ended_date = game_time.require_date_iso(req.ended_date or state.get_current_date_as_date().isoformat(), field="ended_date")
        now = game_time.now_utc_like_iso()
        db_path = state.get_db_path()

        assignment_id = str(req.assignment_id).strip() if req.assignment_id else ""
        scout_id = str(req.scout_id).strip() if req.scout_id else ""
        if not assignment_id and not scout_id:
            raise HTTPException(status_code=400, detail="assignment_id or scout_id is required.")

        ended: Dict[str, Any] = {}

        with LeagueRepo(db_path) as repo:
            repo.init_db()
            with repo.transaction() as cur:
                if assignment_id:
                    row = cur.execute(
                        """
                        SELECT assignment_id, scout_id, target_player_id, status
                        FROM scouting_assignments
                        WHERE assignment_id=? AND team_id=?
                        LIMIT 1;
                        """,
                        (assignment_id, tid),
                    ).fetchone()
                    if not row:
                        raise HTTPException(status_code=404, detail=f"Assignment not found: {assignment_id}")
                    if str(row[3]) != "ACTIVE":
                        raise HTTPException(status_code=409, detail="Assignment is not ACTIVE.")

                    cur.execute(
                        """
                        UPDATE scouting_assignments
                        SET status='ENDED', ended_date=?, updated_at=?
                        WHERE assignment_id=? AND team_id=?;
                        """,
                        (ended_date, now, assignment_id, tid),
                    )
                    ended = {
                        "assignment_id": str(row[0]),
                        "scout_id": str(row[1]),
                        "target_player_id": str(row[2]),
                        "ended_date": ended_date,
                    }
                else:
                    # End active assignment for scout_id
                    row = cur.execute(
                        """
                        SELECT assignment_id, target_player_id
                        FROM scouting_assignments
                        WHERE team_id=? AND scout_id=? AND status='ACTIVE'
                        LIMIT 1;
                        """,
                        (tid, scout_id),
                    ).fetchone()
                    if not row:
                        raise HTTPException(status_code=404, detail="No ACTIVE assignment found for this scout.")

                    assignment_id2 = str(row[0])
                    cur.execute(
                        """
                        UPDATE scouting_assignments
                        SET status='ENDED', ended_date=?, updated_at=?
                        WHERE assignment_id=? AND team_id=?;
                        """,
                        (ended_date, now, assignment_id2, tid),
                    )
                    ended = {
                        "assignment_id": assignment_id2,
                        "scout_id": scout_id,
                        "target_player_id": str(row[1]),
                        "ended_date": ended_date,
                    }

        return {"ok": True, "ended": ended}

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unassign scout: {e}") from e


@app.get("/api/scouting/reports")
async def api_scouting_reports(
    team_id: str,
    player_id: Optional[str] = None,
    scout_id: Optional[str] = None,
    period_key: Optional[str] = None,  # YYYY-MM
    include_payload: bool = True,
    include_text: bool = True,
    limit: int = 50,
    offset: int = 0,
):
    """List scouting reports for a team (private). Supports filters."""
    try:
        tid = str(normalize_team_id(team_id, strict=True))

        pid: Optional[str] = None
        if player_id:
            pid = str(normalize_player_id(player_id, strict=True))

        sid: Optional[str] = str(scout_id).strip() if scout_id else None
        pk: Optional[str] = str(period_key).strip() if period_key else None

        lim = int(limit)
        off = int(offset)
        if lim <= 0:
            lim = 50
        if lim > 200:
            lim = 200
        if off < 0:
            off = 0

        db_path = state.get_db_path()

        where = ["r.team_id = ?"]
        params: List[Any] = [tid]
        if pid:
            where.append("r.target_player_id = ?")
            params.append(pid)
        if sid:
            where.append("r.scout_id = ?")
            params.append(sid)
        if pk:
            where.append("r.period_key = ?")
            params.append(pk)

        sql = f"""
            SELECT
                r.report_id, r.assignment_id, r.scout_id,
                s.display_name, s.specialty_key,
                r.target_player_id, r.target_kind,
                r.season_year, r.period_key, r.as_of_date,
                r.days_covered, r.player_snapshot_json,
                r.payload_json, r.report_text, r.status,
                r.created_at, r.updated_at
            FROM scouting_reports r
            LEFT JOIN scouting_scouts s ON s.scout_id = r.scout_id
            WHERE {' AND '.join(where)}
            ORDER BY r.as_of_date DESC, r.scout_id ASC
            LIMIT ? OFFSET ?;
        """
        params.extend([lim, off])

        out_reports: List[Dict[str, Any]] = []
        with LeagueRepo(db_path) as repo:
            repo.init_db()
            rows = repo._conn.execute(sql, tuple(params)).fetchall()

        for r in rows:
            snapshot = {}
            payload = {}
            try:
                snapshot = json.loads(str(r[11] or "{}"))
            except Exception:
                snapshot = {}
            if include_payload:
                try:
                    payload = json.loads(str(r[12] or "{}"))
                except Exception:
                    payload = {}

            out_reports.append(
                {
                    "report_id": str(r[0]),
                    "assignment_id": str(r[1]),
                    "scout": {
                        "scout_id": str(r[2]),
                        "display_name": str(r[3] or ""),
                        "specialty_key": str(r[4] or ""),
                    },
                    "target_player_id": str(r[5]),
                    "target_kind": str(r[6] or ""),
                    "season_year": int(r[7] or 0),
                    "period_key": str(r[8] or ""),
                    "as_of_date": str(r[9] or "")[:10],
                    "days_covered": int(r[10] or 0),
                    "player_snapshot": snapshot,
                    "payload": payload if include_payload else None,
                    "report_text": (str(r[13]) if (include_text and r[13] is not None) else None),
                    "status": str(r[14] or ""),
                    "created_at": str(r[15] or ""),
                    "updated_at": str(r[16] or ""),
                }
            )

        return {
            "ok": True,
            "team_id": tid,
            "filters": {
                "player_id": pid,
                "scout_id": sid,
                "period_key": pk,
                "limit": lim,
                "offset": off,
                "include_payload": bool(include_payload),
                "include_text": bool(include_text),
            },
            "count": len(out_reports),
            "reports": out_reports,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list scouting reports: {e}") from e


# -------------------------------------------------------------------------
# 플레이-인 / 플레이오프
# -------------------------------------------------------------------------


@app.get("/api/postseason/field")
async def api_postseason_field():
    return build_postseason_field()


@app.get("/api/postseason/state")
async def api_postseason_state():
    return state.get_postseason_snapshot()


@app.post("/api/postseason/reset")
async def api_postseason_reset():
    return reset_postseason_state()


@app.post("/api/postseason/setup")
async def api_postseason_setup(req: PostseasonSetupRequest):
    try:
        return initialize_postseason(req.my_team_id, use_random_field=req.use_random_field)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/postseason/play-in/my-team-game")
async def api_play_in_my_team_game(req: EmptyRequest):
    try:
        return play_my_team_play_in_game()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/postseason/playoffs/advance-my-team-game")
async def api_playoffs_advance_my_team_game(req: EmptyRequest):
    try:
        return advance_my_team_one_game()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/postseason/playoffs/auto-advance-round")
async def api_playoffs_auto_advance_round(req: EmptyRequest):
    try:
        return auto_advance_current_round()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


 # -------------------------------------------------------------------------
# 시즌 전환 (오프시즌 진입 / 정규시즌 시작)
# -------------------------------------------------------------------------


@app.post("/api/season/enter-offseason")
async def api_enter_offseason(req: EmptyRequest):
    """플레이오프 우승 확정 이후, 다음 시즌으로 전환하고 오프시즌(날짜 구간)으로 진입한다.

    Design notes:
    - 이 엔드포인트는 '오프시즌 진입(날짜 이동)'만 수행한다.
    - 대학 시즌 마감/선언 생성, 오프시즌 계약 처리, 드래프트(로터리/정산/지명 기록/적용)는
      아래의 stepwise 오프시즌 API를 단계별로 호출해 실행한다.
    - 실제 시즌 전환(state.start_new_season)은 드래프트 적용 이후에만 수행한다.
    """
    post = state.get_postseason_snapshot() or {}
    champion = post.get("champion")
    if not champion:
        raise HTTPException(status_code=400, detail="Champion not decided yet.")

    league_ctx = state.get_league_context_snapshot() or {}
    try:
        season_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        season_year = 0
    if season_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    next_year = season_year + 1

    # Skeleton offseason: move to an offseason date window where there are no scheduled games.
    offseason_start = f"{next_year}-07-01"
    state.set_current_date(offseason_start)

    # Best-effort UI cache rebuild (derived, non-authoritative).
    try:
        ui_cache_rebuild_all()
    except Exception:
        pass

    return {
        "ok": True,
        "prev_champion": champion,
        "from_season_year": int(season_year),
        "draft_year": int(next_year),
        "offseason_start": offseason_start,
        "steps": [
            "/api/offseason/college/finalize",
            "/api/offseason/contracts/process",
            "/api/offseason/retirement/process",
            "/api/offseason/training/apply-growth",
            "/api/offseason/draft/lottery",
            "/api/offseason/draft/settle",
            "/api/offseason/draft/combine",
            "/api/offseason/draft/workouts",
            "/api/offseason/draft/interviews",
            "/api/offseason/draft/withdrawals",
            "/api/offseason/draft/selections/auto",
            "/api/offseason/draft/selections/pick",
            "/api/offseason/draft/apply",
        ],
    }


# -------------------------------------------------------------------------
# Offseason (stepwise) API: College finalize -> Contracts -> Draft -> Apply
# -------------------------------------------------------------------------


@app.post("/api/offseason/college/finalize")
async def api_offseason_college_finalize(req: EmptyRequest):
    """대학 시즌 마감(스탯 생성) + 드래프트 선언 생성(SSOT=DB).

    NOTE:
    - season_year: 현재 league_ctx 기준(막 끝난 시즌)
    - draft_year: season_year+1
    - 구현은 idempotent 하도록 college.service가 내부에서 guard 한다.
    """
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        season_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        season_year = 0
    if season_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    draft_year = season_year + 1
    try:
        db_path = state.get_db_path()
        from college.service import finalize_season_and_generate_entries

        finalize_season_and_generate_entries(db_path=db_path, season_year=season_year, draft_year=draft_year)
        return {"ok": True, "season_year": int(season_year), "draft_year": int(draft_year)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/offseason/contracts/process")
async def api_offseason_contracts_process(req: OffseasonContractsProcessRequest):
    """오프시즌 계약 처리(만료/옵션/연장/트레이드 정산 등)."""
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        from_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        from_year = 0
    if from_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    to_year = from_year + 1

    # Hard gate: user's TEAM options must be decided before running offseason contracts processing.
    team_id = normalize_team_id(req.user_team_id)
    if not team_id:
        raise HTTPException(status_code=400, detail="Invalid user_team_id.")
    try:
        db_path = state.get_db_path()
        with LeagueRepo(db_path) as repo:
            repo.init_db()
            svc = LeagueService(repo)
            pending = svc.list_pending_team_options(str(team_id), season_year=int(to_year))
        if pending:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "TEAM_OPTION_DECISION_REQUIRED",
                    "message": "Pending TEAM options must be decided before offseason contracts processing.",
                    "team_id": str(team_id),
                    "season_year": int(to_year),
                    "pending_team_options": list(pending),
                    "count": int(len(pending)),
                },
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check pending TEAM options: {e}")

    try:
        from contracts.offseason import process_offseason
        from contracts.options_policy import make_ai_team_option_decision_policy

        # Use in-game date for all contract/offseason decisions & transaction logs.
        # Fail-loud: do NOT fall back to OS date (timeline immersion).
        try:
            in_game_date = state.get_current_date_as_date()
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to read in-game date from state.") from e
        if not hasattr(in_game_date, "isoformat"):
            raise HTTPException(status_code=500, detail="Invalid in-game date object in state.")
        decision_date_iso = in_game_date.isoformat()

        # process_offseason은 DB를 갱신하지만 state를 직접 mutate 하진 않는다.
        snap = state.export_full_state_snapshot()

        # Best-effort: ensure the final regular-season month agency tick is applied
        # before offseason contract processing (idempotent).
        try:
            from agency.checkpoints import ensure_last_regular_month_agency_tick

            final_agency_tick = ensure_last_regular_month_agency_tick(
                db_path=str(db_path),
                now_date_iso=str(decision_date_iso),
                state_snapshot=snap,
            )
        except Exception:
            final_agency_tick = {"ok": True, "skipped": True, "reason": "tick_check_failed"}

        result = process_offseason(
            snap,
            from_season_year=int(from_year),
            to_season_year=int(to_year),
            decision_date_iso=str(decision_date_iso),
            decision_policy=make_ai_team_option_decision_policy(user_team_id=str(team_id)),
        )
        return {
            "ok": True,
            "from_season_year": int(from_year),
            "to_season_year": int(to_year),
            "final_regular_month_agency_tick": final_agency_tick,
            "result": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/offseason/retirement/preview")
async def api_offseason_retirement_preview(req: EmptyRequest):
    """오프시즌 은퇴 결정 미리보기(확정 전)."""
    _ = req
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        from_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        from_year = 0
    if from_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")
    to_year = int(from_year) + 1

    try:
        in_game_date = state.get_current_date_as_date().isoformat()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read in-game date from state: {e}")

    try:
        from retirement.service import preview_offseason_retirement

        out = preview_offseason_retirement(
            db_path=str(state.get_db_path()),
            season_year=int(to_year),
            decision_date_iso=str(in_game_date),
        )
        return out
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/offseason/retirement/process")
async def api_offseason_retirement_process(req: EmptyRequest):
    """오프시즌 은퇴 확정 처리(해당 시즌 1회, idempotent)."""
    _ = req
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        from_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        from_year = 0
    if from_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")
    to_year = int(from_year) + 1

    try:
        in_game_date = state.get_current_date_as_date().isoformat()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read in-game date from state: {e}")

    try:
        from retirement.service import process_offseason_retirement

        out = process_offseason_retirement(
            db_path=str(state.get_db_path()),
            season_year=int(to_year),
            decision_date_iso=str(in_game_date),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"offseason retirement failed: {e}")

    # Best-effort UI cache rebuild.
    try:
        ui_cache_rebuild_all()
    except Exception:
        pass

    return out


@app.post("/api/offseason/training/apply-growth")
async def api_offseason_training_apply_growth(req: EmptyRequest):
    """오프시즌 성장/훈련 적용 (Step 2).

    - SSOT: players.attrs_json 업데이트
    - players.age +1
    - Growth profile 생성/업데이트
    - Idempotent: 같은 시즌 전환에 대해 1회만 적용
    """
    _ = req
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        from_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        from_year = 0
    if from_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")
    to_year = int(from_year) + 1

    try:
        in_game_date = state.get_current_date_as_date().isoformat()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read in-game date from state: {e}")

    db_path = state.get_db_path()
    workflow_state = state.export_workflow_state()

    # If the last regular-season month tick hasn't been applied yet (common when
    # jumping into offseason), apply it now (idempotent).
    snap = state.export_full_state_snapshot()

    try:
        from training.checkpoints import ensure_last_regular_month_tick
        final_month_tick = ensure_last_regular_month_tick(
            db_path=str(db_path),
            now_date_iso=str(in_game_date),
            state_snapshot=snap,
        )
    except Exception:
        final_month_tick = {"ok": True, "skipped": True, "reason": "tick_check_failed"}

    # Same parity checkpoint for player agency (idempotent).
    try:
        from agency.checkpoints import ensure_last_regular_month_agency_tick

        final_agency_tick = ensure_last_regular_month_agency_tick(
            db_path=str(db_path),
            now_date_iso=str(in_game_date),
            state_snapshot=snap,
        )
    except Exception:
        final_agency_tick = {"ok": True, "skipped": True, "reason": "tick_check_failed"}

    from training.service import apply_offseason_growth

    try:
        result = apply_offseason_growth(
            db_path=str(db_path),
            from_season_year=int(from_year),
            to_season_year=int(to_year),
            in_game_date_iso=str(in_game_date),
            workflow_state=workflow_state,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"offseason growth failed: {e}")

    # Best-effort UI cache rebuild.
    try:
        ui_cache_rebuild_all()
    except Exception:
        pass

    result["final_regular_month_tick"] = final_month_tick
    result["final_regular_month_agency_tick"] = final_agency_tick
    return result


# -------------------------------------------------------------------------
# Agency (player autonomy) API
# -------------------------------------------------------------------------


@app.get("/api/agency/player/{player_id}")
async def api_agency_get_player(player_id: str, limit: int = 50, offset: int = 0, season_year: Optional[int] = None):
    """Get a player's current agency state + recent events."""
    pid = str(normalize_player_id(player_id, strict=False, allow_legacy_numeric=True))
    if not pid:
        raise HTTPException(status_code=400, detail="Invalid player_id")

    db_path = state.get_db_path()
    with LeagueRepo(db_path) as repo:
        repo.init_db()
        try:
            from agency import repo as agency_repo

            with repo.transaction() as cur:
                st_map = agency_repo.get_player_agency_states(cur, [pid])
                st = st_map.get(pid)
                events = agency_repo.list_agency_events(
                    cur,
                    player_id=pid,
                    season_year=int(season_year) if season_year is not None else None,
                    limit=int(limit),
                    offset=int(offset),
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read agency data: {e}")

    return {"ok": True, "player_id": pid, "state": st, "events": events}


@app.get("/api/agency/team/{team_id}/events")
async def api_agency_get_team_events(
    team_id: str,
    limit: int = 50,
    offset: int = 0,
    season_year: Optional[int] = None,
    event_type: Optional[str] = None,
):
    """List agency events for a team (UI feed)."""
    tid = str(normalize_team_id(team_id)).upper()
    if not tid:
        raise HTTPException(status_code=400, detail="Invalid team_id")

    db_path = state.get_db_path()
    with LeagueRepo(db_path) as repo:
        repo.init_db()
        try:
            from agency import repo as agency_repo

            with repo.transaction() as cur:
                events = agency_repo.list_agency_events(
                    cur,
                    team_id=tid,
                    season_year=int(season_year) if season_year is not None else None,
                    event_type=str(event_type) if event_type else None,
                    limit=int(limit),
                    offset=int(offset),
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list agency events: {e}")

    return {"ok": True, "team_id": tid, "events": events}


@app.get("/api/agency/events")
async def api_agency_get_events(
    limit: int = 50,
    offset: int = 0,
    season_year: Optional[int] = None,
    event_type: Optional[str] = None,
):
    """List league-wide agency events (debug / commissioner feed)."""
    db_path = state.get_db_path()
    with LeagueRepo(db_path) as repo:
        repo.init_db()
        try:
            from agency import repo as agency_repo

            with repo.transaction() as cur:
                events = agency_repo.list_agency_events(
                    cur,
                    season_year=int(season_year) if season_year is not None else None,
                    event_type=str(event_type) if event_type else None,
                    limit=int(limit),
                    offset=int(offset),
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list agency events: {e}")

    return {"ok": True, "events": events}


@app.post("/api/agency/events/respond")
async def api_agency_events_respond(req: AgencyEventRespondRequest):
    """Respond to an agency event (user chooses how to handle demands/promises).

    This is the *mandatory* user-facing response path:
    it records the user's response, mutates player agency state, and (optionally)
    creates promises to be resolved later.
    """
    db_path = state.get_db_path()
    in_game_date = state.get_current_date_as_date().isoformat()
    now_date = req.now_date or in_game_date

    try:
        from agency.interaction_service import AgencyInteractionError, respond_to_agency_event
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agency interaction module import failed: {exc}")

    try:
        out = respond_to_agency_event(
            db_path=str(db_path),
            user_team_id=req.user_team_id,
            event_id=req.event_id,
            response_type=req.response_type,
            response_payload=req.response_payload,
            now_date_iso=str(now_date),
            strict_promises=True,
        )
        pid = out.get("player_id")
        if pid:
            _try_ui_cache_refresh_players([str(pid)], context="agency.events.respond")
        return out
    except AgencyInteractionError as e:
        # Map well-known error codes to stable HTTP semantics.
        code = str(e.code or "")
        if code in {"AGENCY_EVENT_NOT_FOUND"}:
            raise HTTPException(status_code=404, detail={"code": code, "message": e.message, "details": e.details})
        if code in {"AGENCY_EVENT_TEAM_MISMATCH", "AGENCY_PLAYER_NOT_ON_TEAM"}:
            raise HTTPException(status_code=409, detail={"code": code, "message": e.message, "details": e.details})
        if code in {"AGENCY_PROMISE_SCHEMA_MISSING"}:
            raise HTTPException(status_code=500, detail={"code": code, "message": e.message, "details": e.details})
        raise HTTPException(status_code=400, detail={"code": code, "message": e.message, "details": e.details})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/agency/actions/apply")
async def api_agency_actions_apply(req: AgencyUserActionRequest):
    """User-initiated agency actions (proactive management).

    Examples: meet player, praise, warn, set expectations, start extension talks.
    This records an agency event, updates player agency state, and may create a promise.
    """
    db_path = state.get_db_path()
    in_game_date = state.get_current_date_as_date().isoformat()
    now_date = req.now_date or in_game_date

    league_ctx = state.get_league_context_snapshot() or {}
    try:
        sy = int(league_ctx.get("season_year") or 0)
    except Exception:
        sy = 0

    try:
        from agency.interaction_service import AgencyInteractionError, apply_user_agency_action
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agency interaction module import failed: {exc}")

    try:
        out = apply_user_agency_action(
            db_path=str(db_path),
            user_team_id=req.user_team_id,
            player_id=req.player_id,
            season_year=int(sy),
            action_type=req.action_type,
            action_payload=req.action_payload,
            now_date_iso=str(now_date),
            strict_promises=True,
        )
        pid = out.get("player_id")
        if pid:
            _try_ui_cache_refresh_players([str(pid)], context="agency.actions.apply")
        return out
    except AgencyInteractionError as e:
        code = str(e.code or "")
        if code in {"AGENCY_PLAYER_NOT_ON_TEAM"}:
            raise HTTPException(status_code=409, detail={"code": code, "message": e.message, "details": e.details})
        if code in {"AGENCY_PROMISE_SCHEMA_MISSING"}:
            raise HTTPException(status_code=500, detail={"code": code, "message": e.message, "details": e.details})
        raise HTTPException(status_code=400, detail={"code": code, "message": e.message, "details": e.details})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/offseason/options/team/pending")
async def api_offseason_team_options_pending(req: TeamOptionPendingRequest):
    """유저 팀의 다음 시즌 TEAM 옵션(PENDING) 목록 조회."""
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        from_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        from_year = 0
    if from_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    to_year = from_year + 1

    team_id = normalize_team_id(req.user_team_id)
    if not team_id:
        raise HTTPException(status_code=400, detail="Invalid user_team_id.")

    try:
        db_path = state.get_db_path()
        with LeagueRepo(db_path) as repo:
            repo.init_db()
            svc = LeagueService(repo)
            pending = svc.list_pending_team_options(str(team_id), season_year=int(to_year))
        return {
            "ok": True,
            "team_id": str(team_id),
            "season_year": int(to_year),
            "count": int(len(pending)),
            "pending_team_options": list(pending),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list pending TEAM options: {e}")


@app.post("/api/offseason/options/team/decide")
async def api_offseason_team_options_decide(req: TeamOptionDecideRequest):
    """유저 팀 TEAM 옵션 행사/거절 결정 커밋(DB write)."""
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        from_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        from_year = 0
    if from_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    to_year = from_year + 1

    team_id = normalize_team_id(req.user_team_id)
    if not team_id:
        raise HTTPException(status_code=400, detail="Invalid user_team_id.")

    decisions = list(req.decisions or [])
    if not decisions:
        raise HTTPException(status_code=400, detail="decisions must not be empty.")

    # Use in-game date for all contract/offseason decisions & transaction logs.
    # Fail-loud: do NOT fall back to OS date (timeline immersion).
    try:
        in_game_date = state.get_current_date_as_date()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to read in-game date from state.") from e
    if not hasattr(in_game_date, "isoformat"):
        raise HTTPException(status_code=500, detail="Invalid in-game date object in state.")
    decision_date_iso = in_game_date.isoformat()

    try:
        db_path = state.get_db_path()
        events: List[Dict[str, Any]] = []
        affected_player_ids: List[str] = []

        with LeagueRepo(db_path) as repo:
            repo.init_db()
            svc = LeagueService(repo)

            # All-or-nothing: apply all decisions within one DB transaction.
            with repo.transaction():
                for item in decisions:
                    ev = svc.apply_team_option_decision(
                        contract_id=str(item.contract_id),
                        season_year=int(to_year),
                        decision=str(item.decision),
                        expected_team_id=str(team_id),
                        decision_date=str(decision_date_iso),
                    )
                    evd = ev.to_dict()
                    events.append(evd)
                    pid = evd.get("player_id")
                    if pid:
                        affected_player_ids.append(str(pid))

            remaining = svc.list_pending_team_options(str(team_id), season_year=int(to_year))

        _validate_repo_integrity(db_path)
        _try_ui_cache_refresh_players(list(sorted(set(affected_player_ids))), context="offseason.team_options.decide")

        return {
            "ok": True,
            "team_id": str(team_id),
            "season_year": int(to_year),
            "applied": int(len(events)),
            "events": events,
            "remaining_pending_count": int(len(remaining)),
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply TEAM option decisions: {e}")


@app.post("/api/offseason/draft/lottery")
async def api_offseason_draft_lottery(req: EmptyRequest):
    """드래프트 1~4픽 로터리(플랜 생성/저장)."""
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        from_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        from_year = 0
    if from_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")
    draft_year = from_year + 1

    try:
        from draft.pipeline import run_lottery

        db_path = state.get_db_path()
        snap = state.export_full_state_snapshot()
        plan = run_lottery(state_snapshot=snap, db_path=db_path, draft_year=int(draft_year))
        return {"ok": True, "draft_year": int(draft_year), "plan": plan.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/offseason/draft/settle")
async def api_offseason_draft_settle(req: EmptyRequest):
    """픽 정산(보호/스왑) + 최종 지명 턴 생성."""
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        from_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        from_year = 0
    if from_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")
    draft_year = from_year + 1

    try:
        from draft.pipeline import run_settlement

        db_path = state.get_db_path()
        events, turns = run_settlement(db_path=db_path, draft_year=int(draft_year))
        return {
            "ok": True,
            "draft_year": int(draft_year),
            "settlement_events": list(events),
            "turns": [t.to_dict() for t in turns],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/offseason/draft/combine")
async def api_offseason_draft_combine(req: DraftCombineRequest):
    """드래프트 컴바인 실행 + 결과 DB 저장."""
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        from_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        from_year = 0
    if from_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")
    draft_year = from_year + 1

    try:
        from draft.events import run_combine

        db_path = state.get_db_path()
        result = run_combine(db_path=db_path, draft_year=int(draft_year), rng_seed=req.rng_seed)
        return {"ok": True, "draft_year": int(draft_year), "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/offseason/draft/workouts")
async def api_offseason_draft_workouts(req: DraftWorkoutsRequest):
    """팀 워크아웃 실행 + 결과 DB 저장."""
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        from_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        from_year = 0
    if from_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")
    draft_year = from_year + 1

    try:
        from draft.events import run_workouts

        db_path = state.get_db_path()
        team_id = normalize_team_id(req.team_id)
        if not team_id:
            raise HTTPException(status_code=400, detail="Invalid team_id.")

        try:
            max_invites = int(req.max_invites)
        except Exception:
            max_invites = 0
        if max_invites < 1:
            raise HTTPException(status_code=400, detail="max_invites must be >= 1.")

        invited = list(req.invited_prospect_temp_ids or [])
        result = run_workouts(
            db_path=db_path,
            draft_year=int(draft_year),
            team_id=str(team_id),
            invited_prospect_temp_ids=invited,
            max_invites=int(max_invites),
            rng_seed=req.rng_seed,
        )
        return {
            "ok": True,
            "draft_year": int(draft_year),
            "team_id": str(team_id),
            "invited_count": int(len(invited)),
            "max_invites": int(max_invites),
            "result": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------------------------------------------------
# Draft Interviews (user-controlled; private to viewer team)
# -------------------------------------------------------------------------


@app.get("/api/offseason/draft/interviews/questions")
async def api_offseason_draft_interview_questions():
    """인터뷰 질문 목록(서버 정의)을 반환한다. (UI 미구현이어도 API만 준비)"""
    try:
        from draft.interviews import list_interview_questions

        questions = list_interview_questions()
        return {"ok": True, "questions": questions}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/offseason/draft/interviews")
async def api_offseason_draft_interviews(req: DraftInterviewsRequest):
    """팀 인터뷰 실행 + 결과 DB 저장. (유저가 선택한 질문 기반)"""
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        from_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        from_year = 0
    if from_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")
    draft_year = from_year + 1

    try:
        from draft.interviews import run_interviews

        db_path = state.get_db_path()
        team_id = normalize_team_id(req.team_id)
        if not team_id:
            raise HTTPException(status_code=400, detail="Invalid team_id.")

        items = list(req.interviews or [])
        if len(items) == 0:
            # Allow skipping (no DB writes), consistent with workouts endpoint behavior.
            return {
                "ok": True,
                "draft_year": int(draft_year),
                "team_id": str(team_id),
                "skipped": True,
                "result": {"written": 0, "skipped": 0},
            }

        interviews: List[Dict[str, Any]] = []
        for it in items:
            pid = str(it.prospect_temp_id or "").strip()
            if not pid:
                raise HTTPException(status_code=400, detail="prospect_temp_id is required.")
            qids = [str(x).strip() for x in (it.selected_question_ids or []) if str(x).strip()]
            # For v1, enforce exactly 3 picks here (UI can still choose 3).
            if len(qids) != 3:
                raise HTTPException(status_code=400, detail="selected_question_ids must have exactly 3 items.")
            if len(set(qids)) != len(qids):
                raise HTTPException(status_code=400, detail="selected_question_ids must be unique.")
            interviews.append({"prospect_temp_id": pid, "selected_question_ids": qids})

        result = run_interviews(
            db_path=db_path,
            draft_year=int(draft_year),
            team_id=str(team_id),
            interviews=interviews,
            rng_seed=req.rng_seed,
        )
        return {
            "ok": True,
            "draft_year": int(draft_year),
            "team_id": str(team_id),
            "requested": int(len(interviews)),
            "result": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/offseason/draft/withdrawals")
async def api_offseason_draft_withdrawals(req: EmptyRequest):
    """드래프트 철회(언더클래스만 복귀) 단계 실행 + DB 반영."""
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        from_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        from_year = 0
    if from_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")
    draft_year = from_year + 1

    try:
        from draft.withdrawals import run_withdrawals

        db_path = state.get_db_path()
        result = run_withdrawals(db_path=db_path, draft_year=int(draft_year))
        return {"ok": True, "draft_year": int(draft_year), "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------------------------------------------------
# Draft Experts Big Board (Sam Vecenie-style)
# -------------------------------------------------------------------------


@app.get("/api/offseason/draft/experts")
async def api_offseason_draft_experts():
    """드래프트 전문가(외부 빅보드 작성자) 목록."""
    try:
        # Local import (avoid extra import work at server boot).
        from draft.expert_bigboard import list_experts, PHASE_PRE_COMBINE, PHASE_POST_COMBINE, PHASE_AUTO

        return {
            "ok": True,
            "experts": list_experts(),
            "phases": [PHASE_PRE_COMBINE, PHASE_POST_COMBINE],
            "default_phase": PHASE_AUTO,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list draft experts: {e}") from e


@app.get("/api/offseason/draft/bigboard/expert")
async def api_offseason_draft_bigboard_expert(
    expert_id: str,
    phase: Optional[str] = None,
    draft_year: Optional[int] = None,
    limit: Optional[int] = None,
    pool_mode: Optional[str] = "auto",
    watch_run_id: Optional[str] = None,
    watch_min_prob: Optional[float] = None,
):
    """특정 전문가의 Big Board 생성(불완전 정보 + 바이어스 + 상단 수렴 앵커링).

    Query:
      - expert_id (required)
      - phase: "pre_combine" | "post_combine" | "auto" (default: auto)
      - draft_year: override (default: state.season_year + 1)
      - limit: optional number of prospects in output
      - pool_mode: "declared" | "watch" | "auto" (default: auto)
      - watch_run_id: explicit watch run id (only for pool_mode=watch/auto fallback)
      - watch_min_prob: inclusion threshold for watch pool (declare_prob >= threshold)
    """
    eid = str(expert_id or "").strip()
    if not eid:
        raise HTTPException(status_code=400, detail="expert_id is required.")

    # draft_year default is state.season_year + 1, but allow override even if state is not ready.
    if draft_year is None:
        league_ctx = state.get_league_context_snapshot() or {}
        try:
            from_year = int(league_ctx.get("season_year") or 0)
        except Exception:
            from_year = 0
        if from_year <= 0:
            raise HTTPException(status_code=500, detail="Invalid season_year in state (draft_year not provided).")
        dy = int(from_year) + 1
    else:
        try:
            dy = int(draft_year)
        except Exception:
            raise HTTPException(status_code=400, detail="draft_year must be an integer.")
        if dy <= 0:
            raise HTTPException(status_code=400, detail="draft_year must be > 0.")

    lim: Optional[int] = None
    if limit is not None:
        try:
            lim = int(limit)
        except Exception:
            raise HTTPException(status_code=400, detail="limit must be an integer.")
        if lim <= 0:
            raise HTTPException(status_code=400, detail="limit must be > 0.")

    ph = str(phase or "auto").strip()

    try:
        from draft.expert_bigboard import generate_expert_bigboard

        db_path = state.get_db_path()
        result = generate_expert_bigboard(
            db_path=str(db_path),
            draft_year=int(dy),
            expert_id=str(eid),
            phase=str(ph),
            limit=lim,
            include_debug_axes=False,
            pool_mode=pool_mode or "auto",
            watch_run_id=watch_run_id,
            watch_min_prob=watch_min_prob,
        )
        return result
    except KeyError as e:
        # Unknown expert_id
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate expert bigboard: {e}") from e


@app.get("/api/offseason/draft/bundle")
async def api_offseason_draft_bundle(
    draft_year: Optional[int] = None,
    pool_season_year: Optional[int] = None,
    pool_limit: Optional[int] = None,
    viewer_team_id: Optional[str] = None,
):
    """현재 저장된 플랜(로터리 결과) 기반으로 드래프트 번들(턴/세션/풀) 생성."""
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        from_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        from_year = 0
    if from_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    dy = int(draft_year) if draft_year is not None else int(from_year) + 1
    pysy = int(pool_season_year) if pool_season_year is not None else int(from_year)

    try:
        from draft.engine import prepare_bundle_from_saved_plan

        db_path = state.get_db_path()

        # Viewer team is optional. If provided, normalize it so per-team workout visibility works.
        vt = None
        if viewer_team_id is not None:
            vt = normalize_team_id(viewer_team_id)
            if not vt:
                raise HTTPException(status_code=400, detail="Invalid viewer_team_id.")
        
        snap = state.export_full_state_snapshot()
        bundle = prepare_bundle_from_saved_plan(
            snap,
            db_path=db_path,
            draft_year=int(dy),
            pool_season_year=int(pysy),
            pool_limit=pool_limit,
            session_meta={"trigger": "api_bundle", "viewer_team_id": (str(vt) if vt else None)},
        )
        # Fog-of-war: return ONLY public payload (no ovr/attrs/potential leak, no db_path leak).
        # viewer_team_id controls whether team-private workout results are included.
        return bundle.to_public_dict(viewer_team_id=(str(vt) if vt else None))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/offseason/draft/selections/auto")
async def api_offseason_draft_selections_auto(req: DraftAutoSelectionsRequest):
    """저장된 플랜 기반으로 남은 픽을 자동 선택(draft_selections에 기록)."""
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        from_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        from_year = 0
    if from_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")
    draft_year = from_year + 1

    try:
        from draft.engine import prepare_bundle_from_saved_plan, auto_run_selections

        # Use in-game date for selection timestamps (draft_selections.selected_at).
        # Fail-loud: do NOT fall back to OS date (timeline immersion).
        try:
            in_game_date = state.get_current_date_as_date()
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to read in-game date from state.") from e
        if not hasattr(in_game_date, "isoformat"):
            raise HTTPException(status_code=500, detail="Invalid in-game date object in state.")
        selected_at_iso = in_game_date.isoformat()

        db_path = state.get_db_path()
        snap = state.export_full_state_snapshot()
        bundle = prepare_bundle_from_saved_plan(
            snap,
            db_path=db_path,
            draft_year=int(draft_year),
            pool_season_year=int(from_year),
            pool_limit=None,
            session_meta={"trigger": "api_auto"},
        )

        team_ids = None
        if req.stop_on_user_controlled_team_ids is not None:
            cleaned = [normalize_team_id(t) for t in (req.stop_on_user_controlled_team_ids or [])]
            cleaned = [t for t in cleaned if t]
            if cleaned:
                team_ids = cleaned

        # Fail-closed: 유저팀 목록이 없으면 기본은 "멈춤".
        # (명시적으로 allow_autopick_user_team=true를 준 경우만 예외)
        if (not req.allow_autopick_user_team) and (not team_ids):
            raise ValueError(
                "stop_on_user_controlled_team_ids is required unless allow_autopick_user_team=true"
            )

        picks = auto_run_selections(
            bundle=bundle,
            selected_at_iso=str(selected_at_iso),
            max_picks=req.max_picks,
            stop_on_user_controlled_team_ids=team_ids,
            allow_autopick_user_team=bool(req.allow_autopick_user_team),
            source="draft_auto",
        )
        # Fog-of-war: return only public-safe pick payloads.
        out_picks = []
        for p in (picks or []):
            out_picks.append(p.to_public_dict() if hasattr(p, "to_public_dict") else p.to_dict())
        return {"ok": True, "draft_year": int(draft_year), "picks": out_picks}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/offseason/draft/selections/pick")
async def api_offseason_draft_selections_pick(req: DraftRecordPickRequest):
    """현재 커서(온더클락) 픽을 1개 기록(draft_selections에 저장)."""
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        from_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        from_year = 0
    if from_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")
    draft_year = from_year + 1

    try:
        from draft.engine import prepare_bundle_from_saved_plan, record_pick_and_save_selection

        # Use in-game date for selection timestamps (draft_selections.selected_at).
        # Fail-loud: do NOT fall back to OS date (timeline immersion).
        try:
            in_game_date = state.get_current_date_as_date()
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to read in-game date from state.") from e
        if not hasattr(in_game_date, "isoformat"):
            raise HTTPException(status_code=500, detail="Invalid in-game date object in state.")
        selected_at_iso = in_game_date.isoformat()

        db_path = state.get_db_path()
        snap = state.export_full_state_snapshot()
        bundle = prepare_bundle_from_saved_plan(
            snap,
            db_path=db_path,
            draft_year=int(draft_year),
            pool_season_year=int(from_year),
            pool_limit=None,
            session_meta={"trigger": "api_pick"},
        )

        pick = record_pick_and_save_selection(
            bundle=bundle,
            prospect_temp_id=str(req.prospect_temp_id),
            selected_at_iso=str(selected_at_iso),
            source=str(req.source or "draft_user"),
            meta=(dict(req.meta) if isinstance(req.meta, dict) else None),
        )
        # Fog-of-war: scrub meta defensively (even if AI/meta accidentally includes sensitive keys).
        pick_payload = pick.to_public_dict() if hasattr(pick, "to_public_dict") else pick.to_dict()
        return {"ok": True, "draft_year": int(draft_year), "pick": pick_payload}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/offseason/draft/apply")
async def api_offseason_draft_apply(req: EmptyRequest):
    """draft_selections -> 실제 DB 적용(draft_results/roster/contract/tx), 이후 시즌 전환."""
    league_ctx = state.get_league_context_snapshot() or {}
    try:
        from_year = int(league_ctx.get("season_year") or 0)
    except Exception:
        from_year = 0
    if from_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    to_year = from_year + 1
    draft_year = int(to_year)

    try:
        db_path = state.get_db_path()

        # Hard gate: contracts offseason processing must be completed before draft apply.
        required_meta_key = f"contracts_offseason_done_{to_year}"
        with LeagueRepo(db_path) as _repo:
            _repo.init_db()
            row = _repo._conn.execute("SELECT value FROM meta WHERE key=?;", (required_meta_key,)).fetchone()
            ok = bool(row is not None and str(row["value"]) == "1")
        if not ok:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "CONTRACTS_OFFSEASON_NOT_PROCESSED",
                    "message": "Run /api/offseason/contracts/process (and decide TEAM options if required) before draft apply.",
                    "required_meta_key": str(required_meta_key),
                    "season_year": int(to_year),
                },
            )

        # Hard gate: retirement offseason processing should be completed before draft apply.
        retirement_meta_key = f"retirement_processed_{to_year}"
        with LeagueRepo(db_path) as _repo:
            _repo.init_db()
            row = _repo._conn.execute("SELECT value FROM meta WHERE key=?;", (retirement_meta_key,)).fetchone()
            retirement_ok = bool(row is not None and str(row["value"]) == "1")
        if not retirement_ok:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "RETIREMENT_OFFSEASON_NOT_PROCESSED",
                    "message": "Run /api/offseason/retirement/process before draft apply.",
                    "required_meta_key": str(retirement_meta_key),
                    "season_year": int(to_year),
                },
            )

        # 1) Apply picks (SSOT: draft_results)
        from draft.pipeline import apply_selections

        # Use in-game date for transaction logs (avoid OS date mismatch).
        # Fail-loud: do NOT fall back to OS date (timeline immersion).
        try:
            in_game_date = state.get_current_date_as_date()
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to read in-game date from state.") from e
        if not hasattr(in_game_date, "isoformat"):
            raise HTTPException(status_code=500, detail="Invalid in-game date object in state.")
        tx_date_iso = in_game_date.isoformat()

        # Inject CapModel built from SSOT (state.trade_rules) to avoid duplicated cap math.
        try:
            from cap_model import CapModel
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"CapModel import failed: {exc}")

        trade_rules = league_ctx.get("trade_rules") if isinstance(league_ctx, dict) else None
        if not isinstance(trade_rules, dict):
            trade_rules = {}
        cap_model = CapModel.from_trade_rules(trade_rules, current_season_year=int(from_year))

        applied_count = int(
            apply_selections(
                db_path=db_path,
                draft_year=int(draft_year),
                tx_date_iso=tx_date_iso,
                cap_model=cap_model,
            )
        )

        # 2) Resolve undrafted declared players into pro routes (FA / retirement)
        from draft.undrafted import resolve_undrafted_to_pro

        undrafted_result = resolve_undrafted_to_pro(
            db_path=db_path,
            draft_year=int(draft_year),
            tx_date_iso=str(tx_date_iso),
        )

        # Mark draft completed AFTER undrafted resolution to avoid leaving DECLARED players behind.
        meta_key = f"draft_completed_{draft_year}"
        with LeagueRepo(db_path) as repo:
            repo.init_db()
            repo._conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                (meta_key, "1"),
            )
            repo._conn.commit()

        # 3) College offseason advance (hidden; auto after NBA draft)
        from college.service import advance_offseason as _college_advance_offseason
        _college_advance_offseason(db_path=db_path, from_season_year=int(from_year), to_season_year=int(to_year))

        # 4) Now perform the actual season transition (NO auto-offseason hooks)
        transition = state.start_new_season(int(to_year), rebuild_schedule=True)

        # Best-effort UI cache rebuild after apply + season transition.
        try:
            ui_cache_rebuild_all()
        except Exception:
            pass

        return {
            "ok": True,
            "draft_year": int(draft_year),
            "applied_count": int(applied_count),
            "undrafted": undrafted_result,
            "college_advanced_to": int(to_year),
            "season_transition": transition,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/season/start-regular-season")
async def api_start_regular_season(req: EmptyRequest):
    """오프시즌(또는 임의 시점)에서 정규시즌 시작 직전으로 날짜를 이동한다.

    IMPORTANT:
    - advance_league_until()은 current_date+1부터 진행하므로, 개막일 게임을 스킵하지 않게
      season_start '전날'로 세팅한다.
    """
    league_ctx = state.get_league_context_snapshot() or {}
    season_start = league_ctx.get("season_start")
    if not season_start:
        raise HTTPException(status_code=500, detail="season_start is missing. Schedule not initialized?")

    try:
        ss = date.fromisoformat(str(season_start))
    except ValueError:
        raise HTTPException(status_code=500, detail=f"Invalid season_start format: {season_start}")

    start_day_minus_1 = (ss - timedelta(days=1)).isoformat()
    state.set_current_date(start_day_minus_1)

    return {
        "ok": True,
        "current_date": state.get_current_date(),
        "season_start": str(season_start),
    }


# -------------------------------------------------------------------------
# 주간 뉴스 (LLM 요약)
# -------------------------------------------------------------------------


@app.post("/api/news/week")
async def api_news_week(req: WeeklyNewsRequest):
    if not req.apiKey:
        raise HTTPException(status_code=400, detail="apiKey is required")
    try:
        payload = refresh_weekly_news(req.apiKey)

        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weekly news generation failed: {e}")


@app.post("/api/news/playoffs")
async def api_playoff_news(req: EmptyRequest):
    try:
        return refresh_playoff_news()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Playoff news generation failed: {e}")


@app.post("/api/season-report")
async def api_season_report(req: SeasonReportRequest):
    """정규 시즌 종료 후, LLM을 이용해 시즌 결산 리포트를 생성한다."""
    if not req.apiKey:
        raise HTTPException(status_code=400, detail="apiKey is required")

    try:
        report_text = generate_season_report(req.apiKey, req.user_team_id)
        return {"report_markdown": report_text}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Season report generation failed: {e}")


@app.post("/api/validate-key")
async def api_validate_key(req: ApiKeyRequest):
    """주어진 Gemini API 키를 간단히 검증한다."""
    if not req.apiKey:
        raise HTTPException(status_code=400, detail="apiKey is required")

    try:
        genai.configure(api_key=req.apiKey)
        # 최소 호출로 키 유효성 확인 (토큰 카운트 호출)
        model = genai.GenerativeModel("gemini-3-pro-preview")
        model.count_tokens("ping")
        return {"valid": True}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid API key: {e}")


# -------------------------------------------------------------------------
# 메인 LLM (Home 대화) API
# -------------------------------------------------------------------------
@app.post("/api/chat-main")
async def chat_main(req: ChatMainRequest):
    """메인 프롬프트 + 컨텍스트 + 유저 입력을 가지고 Gemini를 호출."""
    if not req.apiKey:
        raise HTTPException(status_code=400, detail="apiKey is required")

    try:
        genai.configure(api_key=req.apiKey)
        model = genai.GenerativeModel(
            model_name="gemini-3-pro-preview",
            system_instruction=req.mainPrompt or "",
        )

        context_text = req.context
        if isinstance(req.context, (dict, list)):
            context_text = json.dumps(req.context, ensure_ascii=False)

        prompt = f"{context_text}\n\n[USER]\n{req.userInput}"
        resp = model.generate_content(prompt)
        text = extract_text_from_gemini_response(resp)
        return {"reply": text, "answer": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini main chat error: {e}")


@app.post("/api/main-llm")
async def chat_main_legacy(req: ChatMainRequest):
    return await chat_main(req)


# -------------------------------------------------------------------------
# 트레이드 API
# -------------------------------------------------------------------------
def _trade_error_response(error: TradeError) -> JSONResponse:
    payload = {
        "ok": False,
        "error": {
            "code": error.code,
            "message": error.message,
            "details": error.details,
        },
    }
    return JSONResponse(status_code=400, content=payload)

def _validate_repo_integrity(db_path: str) -> None:
    with LeagueRepo(db_path) as repo:
        # DB schema is guaranteed during server startup (state.startup_init_state()).
        repo.validate_integrity()


def _commit_accepted_contract_negotiation(
    *,
    db_path: str,
    session_id: str,
    expected_team_id: str,
    expected_player_id: str,
    signed_date_iso: str,
    allowed_modes: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Commit an ACCEPTED contract negotiation session by applying the SSOT contract write.

    This is the **enforcement gate** for commercial-grade 'player agency':
    - the signing endpoints cannot bypass the negotiation outcome
    - the contract terms used are always the session's agreed_offer

    Raises HTTPException on failure.
    """
    sid = str(session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail={"code": "MISSING_SESSION_ID", "message": "session_id is required"})

    try:
        from contracts.negotiation.store import close_session, get_session
        from contracts.negotiation.types import ContractOffer
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Negotiation module import failed: {exc}")

    # Load and validate session
    try:
        session = get_session(sid)
    except Exception as exc:
        raise HTTPException(status_code=404, detail={"code": "NEGOTIATION_NOT_FOUND", "message": str(exc)})

    if str(session.get("kind") or "").upper() != "CONTRACT":
        raise HTTPException(status_code=409, detail={"code": "NEGOTIATION_KIND_MISMATCH", "session_id": sid})

    mode = str(session.get("mode") or "").upper()
    if allowed_modes is not None and mode not in allowed_modes:
        raise HTTPException(
            status_code=409,
            detail={"code": "NEGOTIATION_MODE_MISMATCH", "session_id": sid, "mode": mode, "allowed": sorted(allowed_modes)},
        )

    if str(session.get("status") or "").upper() != "ACTIVE":
        raise HTTPException(
            status_code=409,
            detail={"code": "NEGOTIATION_NOT_ACTIVE", "session_id": sid, "status": session.get("status")},
        )

    if str(session.get("phase") or "").upper() != "ACCEPTED":
        raise HTTPException(
            status_code=409,
            detail={"code": "NEGOTIATION_NOT_ACCEPTED", "session_id": sid, "phase": session.get("phase")},
        )

    team_norm = str(normalize_team_id(expected_team_id)).upper()
    pid_norm = str(normalize_player_id(expected_player_id, strict=False, allow_legacy_numeric=True))

    if str(session.get("team_id") or "").upper() != team_norm:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "NEGOTIATION_TEAM_MISMATCH",
                "session_id": sid,
                "expected_team_id": team_norm,
                "session_team_id": session.get("team_id"),
            },
        )

    if str(session.get("player_id") or "") != pid_norm:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "NEGOTIATION_PLAYER_MISMATCH",
                "session_id": sid,
                "expected_player_id": pid_norm,
                "session_player_id": session.get("player_id"),
            },
        )

    offer_payload = session.get("agreed_offer")
    if not isinstance(offer_payload, dict):
        raise HTTPException(
            status_code=409,
            detail={"code": "NEGOTIATION_NO_AGREED_OFFER", "session_id": sid},
        )

    # Normalize offer
    try:
        offer = ContractOffer.from_payload(offer_payload)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "NEGOTIATION_BAD_OFFER", "session_id": sid, "message": str(exc)},
        )

    # Apply SSOT write via LeagueService (DB-backed).
    with LeagueRepo(db_path) as repo:
        svc = LeagueService(repo)
        try:
            if mode == "SIGN_FA":
                event = svc.sign_free_agent(
                    team_id=team_norm,
                    player_id=pid_norm,
                    signed_date=signed_date_iso,
                    years=int(offer.years),
                    salary_by_year=offer.salary_by_year,
                    options=[dict(x) for x in (offer.options or [])],
                )
            else:
                # RE_SIGN and EXTEND both map to the same SSOT operation.
                event = svc.re_sign_or_extend(
                    team_id=team_norm,
                    player_id=pid_norm,
                    signed_date=signed_date_iso,
                    years=int(offer.years),
                    salary_by_year=offer.salary_by_year,
                    options=[dict(x) for x in (offer.options or [])],
                )
        except CapViolationError as exc:
            # Rule-based rejection: return 409 instead of 500.
            raise HTTPException(
                status_code=409,
                detail={
                    "code": getattr(exc, "code", "CAP_VIOLATION"),
                    "message": getattr(exc, "message", str(exc)),
                    "details": getattr(exc, "details", None),
                },
            )

    # Close the session (idempotent-ish; no side effects on DB).
    try:
        close_session(sid, phase="ACCEPTED", status="CLOSED")
    except Exception:
        # Never fail contract commit due to in-memory session closure.
        pass

    event_dict = event.to_dict()
    affected = event_dict.get("affected_player_ids") or []
    _try_ui_cache_refresh_players(list(affected), context="contracts.negotiation.commit")
    return {"ok": True, "session_id": sid, "mode": mode, "team_id": team_norm, "player_id": pid_norm, "event": event_dict}


def _try_ui_cache_refresh_players(player_ids: List[str], *, context: str) -> None:
    """Best-effort UI cache refresh. Never fails the API call.

    Policy: DB SSOT write APIs should succeed even if UI cache refresh fails.
    """
    try:
        if not player_ids:
            return
        ui_cache_refresh_players(player_ids)
    except Exception:
        logger.warning(
            "UI cache refresh failed (%s): player_ids=%r",
            context,
            player_ids,
            exc_info=True,
        )

# -------------------------------------------------------------------------
# Contracts / Roster Write API
# -------------------------------------------------------------------------
@app.post("/api/contracts/release-to-fa")
async def api_contracts_release_to_fa(req: ReleaseToFARequest):
    """Release a player to free agency (DB write)."""
    try:
        db_path = state.get_db_path()
        in_game_date = state.get_current_date_as_date()
        with LeagueRepo(db_path) as repo:
            svc = LeagueService(repo)
            event = svc.release_player_to_free_agency(
                player_id=req.player_id,
                released_date=req.released_date or in_game_date,
            )
        _validate_repo_integrity(db_path)
        event_dict = event.to_dict()
        affected = event_dict.get("affected_player_ids") or []
        _try_ui_cache_refresh_players(list(affected), context="contracts.release_to_fa")
        return {"ok": True, "event": event_dict}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Release-to-FA failed: {e}")


# -------------------------------------------------------------------------
# Contract Negotiation API (player agency - mandatory path)
# -------------------------------------------------------------------------


@app.post("/api/contracts/negotiation/start")
async def api_contracts_negotiation_start(req: ContractNegotiationStartRequest):
    """Start a contract negotiation session (state-backed)."""
    try:
        from contracts.negotiation.service import start_contract_negotiation
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Negotiation module import failed: {exc}")

    try:
        db_path = state.get_db_path()
        now_iso = game_time.now_utc_like_iso()
        out = start_contract_negotiation(
            db_path=str(db_path),
            team_id=req.team_id,
            player_id=req.player_id,
            mode=req.mode,
            valid_days=req.valid_days,
            now_iso=str(now_iso),
        )
        return out
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/contracts/negotiation/offer")
async def api_contracts_negotiation_offer(req: ContractNegotiationOfferRequest):
    """Submit a team offer; player may ACCEPT / COUNTER / REJECT / WALK."""
    try:
        from contracts.negotiation.service import submit_contract_offer
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Negotiation module import failed: {exc}")

    try:
        db_path = state.get_db_path()
        now_iso = game_time.now_utc_like_iso()
        out = submit_contract_offer(
            db_path=str(db_path),
            session_id=req.session_id,
            offer_payload=req.offer,
            now_iso=str(now_iso),
        )
        return out
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/contracts/negotiation/accept-counter")
async def api_contracts_negotiation_accept_counter(req: ContractNegotiationAcceptCounterRequest):
    """Accept the last counter offer proposed by the player."""
    try:
        from contracts.negotiation.service import accept_last_counter
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Negotiation module import failed: {exc}")

    try:
        db_path = state.get_db_path()
        now_iso = game_time.now_utc_like_iso()
        out = accept_last_counter(
            db_path=str(db_path),
            session_id=req.session_id,
            now_iso=str(now_iso),
        )
        return out
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/contracts/negotiation/commit")
async def api_contracts_negotiation_commit(req: ContractNegotiationCommitRequest):
    """Commit an ACCEPTED session (SSOT contract write)."""
    try:
        from contracts.negotiation.store import get_session
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Negotiation module import failed: {exc}")

    db_path = state.get_db_path()
    signed_date_iso = req.signed_date or state.get_current_date_as_date().isoformat()

    try:
        session = get_session(str(req.session_id))
    except Exception as exc:
        raise HTTPException(status_code=404, detail={"code": "NEGOTIATION_NOT_FOUND", "message": str(exc)})

    return _commit_accepted_contract_negotiation(
        db_path=str(db_path),
        session_id=str(req.session_id),
        expected_team_id=str(session.get("team_id") or ""),
        expected_player_id=str(session.get("player_id") or ""),
        signed_date_iso=str(signed_date_iso),
        allowed_modes=None,
    )


@app.post("/api/contracts/negotiation/cancel")
async def api_contracts_negotiation_cancel(req: ContractNegotiationCancelRequest):
    """Cancel/close a negotiation session (no SSOT DB write)."""
    try:
        from contracts.negotiation.store import close_session, get_session
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Negotiation module import failed: {exc}")

    try:
        session = get_session(str(req.session_id))
        close_session(str(req.session_id), phase="WALKED", status="CLOSED")
        return {"ok": True, "session_id": str(req.session_id), "team_id": session.get("team_id"), "player_id": session.get("player_id")}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/contracts/sign-free-agent")
async def api_contracts_sign_free_agent(req: SignFreeAgentRequest):
    """Sign a free agent (DB write).

    Commercial enforcement:
    - This endpoint cannot bypass negotiation.
    - It requires a contract negotiation session_id in ACCEPTED phase.
    - The signed terms are taken from the session's agreed_offer (not from request payload).
    """
    try:
        db_path = state.get_db_path()
        signed_date_iso = req.signed_date or state.get_current_date_as_date().isoformat()

        out = _commit_accepted_contract_negotiation(
            db_path=str(db_path),
            session_id=str(req.session_id),
            expected_team_id=req.team_id,
            expected_player_id=req.player_id,
            signed_date_iso=str(signed_date_iso),
            allowed_modes={"SIGN_FA"},
        )
        _validate_repo_integrity(str(db_path))
        return out
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sign-free-agent failed: {e}")


@app.post("/api/contracts/re-sign-or-extend")
async def api_contracts_re_sign_or_extend(req: ReSignOrExtendRequest):
    """Re-sign / extend a player (DB write).

    Commercial enforcement:
    - This endpoint cannot bypass negotiation.
    - It requires a contract negotiation session_id in ACCEPTED phase.
    - The signed terms are taken from the session's agreed_offer (not from request payload).
    """
    try:
        db_path = state.get_db_path()
        signed_date_iso = req.signed_date or state.get_current_date_as_date().isoformat()

        out = _commit_accepted_contract_negotiation(
            db_path=str(db_path),
            session_id=str(req.session_id),
            expected_team_id=req.team_id,
            expected_player_id=req.player_id,
            signed_date_iso=str(signed_date_iso),
            allowed_modes={"RE_SIGN", "EXTEND"},
        )
        _validate_repo_integrity(str(db_path))
        return out
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Re-sign/extend failed: {e}")


@app.post("/api/trade/submit")
async def api_trade_submit(req: TradeSubmitRequest):
    try:
        in_game_date = state.get_current_date_as_date()
        db_path = state.get_db_path()
        agreements.gc_expired_agreements(current_date=in_game_date)
        deal = canonicalize_deal(parse_deal(req.deal))
        validate_deal(deal, current_date=in_game_date)
        transaction = apply_deal_to_db(
            db_path=db_path,
            deal=deal,
            source="menu",
            deal_id=None,
            trade_date=in_game_date,
            dry_run=False,
        )
        _validate_repo_integrity(db_path)
        moved_ids: List[str] = []
        for mv in (transaction.get("player_moves") or []):
            if isinstance(mv, dict):
                pid = mv.get("player_id")
                if pid:
                    moved_ids.append(str(pid))
        _try_ui_cache_refresh_players(moved_ids, context="trade.submit")
        return {
            "ok": True,
            "deal": serialize_deal(deal),
            "transaction": transaction,
        }
    except TradeError as exc:
        return _trade_error_response(exc)


@app.post("/api/trade/submit-committed")
async def api_trade_submit_committed(req: TradeSubmitCommittedRequest):
    try:
        in_game_date = state.get_current_date_as_date()
        db_path = state.get_db_path()
        agreements.gc_expired_agreements(current_date=in_game_date)
        deal = agreements.verify_committed_deal(req.deal_id, current_date=in_game_date)
        validate_deal(
            deal,
            current_date=in_game_date,
            allow_locked_by_deal_id=req.deal_id,
        )
        transaction = apply_deal_to_db(
            db_path=db_path,
            deal=deal,
            source="negotiation",
            deal_id=req.deal_id,
            trade_date=in_game_date,
            dry_run=False,
        )
        _validate_repo_integrity(db_path)
        agreements.mark_executed(req.deal_id)
        moved_ids: List[str] = []
        for mv in (transaction.get("player_moves") or []):
            if isinstance(mv, dict):
                pid = mv.get("player_id")
                if pid:
                    moved_ids.append(str(pid))
        _try_ui_cache_refresh_players(moved_ids, context="trade.submit_committed")
        return {"ok": True, "deal_id": req.deal_id, "transaction": transaction}
    except TradeError as exc:
        return _trade_error_response(exc)


@app.post("/api/trade/negotiation/start")
async def api_trade_negotiation_start(req: TradeNegotiationStartRequest):
    try:
        in_game_date = state.get_current_date_as_date()
        session = negotiation_store.create_session(
            user_team_id=req.user_team_id, other_team_id=req.other_team_id
        )
        # Ensure sessions naturally expire even if the user ignores them,
        # so they don't permanently consume the active-session cap.
        valid_until = (in_game_date + timedelta(days=2)).isoformat()
        negotiation_store.set_valid_until(session["session_id"], valid_until)
        # Keep response consistent with stored session
        session["valid_until"] = valid_until
        return {"ok": True, "session": session}
    except TradeError as exc:
        return _trade_error_response(exc)


@app.post("/api/trade/negotiation/commit")
async def api_trade_negotiation_commit(req: TradeNegotiationCommitRequest):
    try:
        in_game_date = state.get_current_date_as_date()
        db_path = state.get_db_path()
        session = negotiation_store.get_session(req.session_id)
        deal = canonicalize_deal(parse_deal(req.deal))
        team_ids = {session["user_team_id"].upper(), session["other_team_id"].upper()}
        if set(deal.teams) != team_ids or len(deal.teams) != 2:
            raise TradeError(
                "DEAL_INVALIDATED",
                "Deal teams must match negotiation session",
                {"session_id": req.session_id, "teams": deal.teams},
            )
        # Hot path: negotiation UI calls this endpoint repeatedly.
        # DB integrity is already guaranteed at startup and after any write APIs.
        # Avoid running full repo integrity check on every offer update.
        validate_deal(deal, current_date=in_game_date, db_path=db_path, integrity_check=False)
        
        # Always persist the latest valid offer payload
        negotiation_store.set_draft_deal(req.session_id, serialize_deal(deal))

        # Local imports to keep integration flexible.
        from trades.valuation.service import evaluate_deal_for_team as eval_service  # type: ignore
        from trades.valuation.types import (
            to_jsonable,
            DealVerdict,
            DealDecision,
            DecisionReason,
        )  # type: ignore
        from trades.generation.dealgen.dedupe import dedupe_hash  # type: ignore

        # ------------------------------------------------------------------
        # Fast path: if the user submits the exact last AI counter-offer, accept
        # immediately.
        # - Prevents the frustrating UX where the AI "rejects its own counter".
        # - Only active while the session is in COUNTER_PENDING phase.
        # ------------------------------------------------------------------
        try:
            phase = str(session.get("phase") or "").upper()
            last_counter = session.get("last_counter")
            expected_hash = last_counter.get("counter_hash") if isinstance(last_counter, dict) else None
            if phase == "COUNTER_PENDING" and isinstance(expected_hash, str) and expected_hash.strip():
                if dedupe_hash(deal) == expected_hash.strip():
                    committed = agreements.create_committed_deal(
                        deal,
                        valid_days=2,
                        current_date=in_game_date,
                        validate=False,   # already validated above
                        db_path=db_path,
                    )
                    negotiation_store.set_committed(req.session_id, committed["deal_id"])
                    negotiation_store.set_status(req.session_id, "CLOSED")
                    negotiation_store.set_phase(req.session_id, "ACCEPTED")
                    negotiation_store.set_valid_until(req.session_id, committed["expires_at"])

                    fast_decision = DealDecision(
                        verdict=DealVerdict.ACCEPT,
                        required_surplus=0.0,
                        overpay_allowed=0.0,
                        confidence=1.0,
                        reasons=(
                            DecisionReason(
                                code="COUNTER_ACCEPTED",
                                message="Accepted last counter offer",
                            ),
                        ),
                        counter=None,
                        meta={"fast_accept": True},
                    )

                    # Preserve any cached evaluation summary if present
                    fast_eval: Dict[str, Any] = {}
                    if isinstance(last_counter, dict):
                        ev = last_counter.get("ai_evaluation")
                        if isinstance(ev, dict):
                            fast_eval = dict(ev)

                    return {
                        "ok": True,
                        "accepted": True,
                        "fast_accept": True,
                        "deal_id": committed["deal_id"],
                        "expires_at": committed["expires_at"],
                        "deal": serialize_deal(deal),
                        "ai_verdict": to_jsonable(fast_decision.verdict),
                        "ai_decision": to_jsonable(fast_decision),
                        "ai_evaluation": fast_eval,
                    }
        except Exception:
            # Fast-accept should never crash the commit flow.
            pass


        # ------------------------------------------------------------------
        # AI evaluation (other team perspective)
        # NOTE:
        # - legality is already checked by validate_deal above
        # - valuation service will build DecisionContext internally (team_situation + gm profile)
        # ------------------------------------------------------------------
        other_team_id = session["other_team_id"].upper()

        decision, evaluation = eval_service(
            deal=deal,
            team_id=other_team_id,
            current_date=in_game_date,
            db_path=db_path,
            include_breakdown=False,   # keep negotiation response light
            include_package_effects=True,
            allow_counter=True,
            validate=False,            # already validated above
        )

        eval_summary = {
            "team_id": other_team_id,
            "incoming_total": float(evaluation.incoming_total),
            "outgoing_total": float(evaluation.outgoing_total),
            "net_surplus": float(evaluation.net_surplus),
            "surplus_ratio": float(evaluation.surplus_ratio),
        }

        # Record the latest offer evaluation in-session (do NOT overwrite last_counter).
        # - last_counter is reserved for the actual counter deal payload (for fast-accept).
        try:
            negotiation_store.set_last_offer(
                req.session_id,
                {
                    "offer": serialize_deal(deal),
                    "ai_verdict": to_jsonable(decision.verdict),
                    "ai_decision": to_jsonable(decision),
                    "ai_evaluation": eval_summary,
                },
            )
        except Exception:
            pass

        # Decide action
        verdict = decision.verdict

        if verdict == DealVerdict.ACCEPT:
            committed = agreements.create_committed_deal(
                deal,
                valid_days=2,
                current_date=in_game_date,
                validate=False,   # already validated above
                db_path=db_path,  # keep hash/locking based on the same db snapshot
            )
            negotiation_store.set_committed(req.session_id, committed["deal_id"])
            # Once committed, this negotiation should no longer consume "ACTIVE" capacity.
            negotiation_store.set_status(req.session_id, "CLOSED")
            # Optional but useful for UI/debugging.
            negotiation_store.set_phase(req.session_id, "ACCEPTED")
            # Keep session expiry aligned with the committed deal expiry.
            negotiation_store.set_valid_until(req.session_id, committed["expires_at"])
            return {
                "ok": True,
                "accepted": True,
                "deal_id": committed["deal_id"],
                "expires_at": committed["expires_at"],
                "deal": serialize_deal(deal),
                "ai_verdict": to_jsonable(decision.verdict),
                "ai_decision": to_jsonable(decision),
                "ai_evaluation": eval_summary,
            }

        # ------------------------------------------------------------------
        # COUNTER: build an actual counter proposal (NBA-like minimal edits)
        # ------------------------------------------------------------------
        if verdict == DealVerdict.COUNTER:
            counter_prop = None
            try:
                from trades.counter_offer.init import build_counter_offer  # type: ignore

                counter_prop = build_counter_offer(
                    offer=deal,
                    user_team_id=session["user_team_id"],
                    other_team_id=session["other_team_id"],
                    current_date=in_game_date,
                    db_path=db_path,
                    session=session,
                )
            except Exception:
                counter_prop = None

            if counter_prop is not None and getattr(counter_prop, "deal", None) is not None:
                # Attach the generated counter proposal to the decision (SSOT).
                decision = DealDecision(
                    verdict=decision.verdict,
                    required_surplus=float(decision.required_surplus),
                    overpay_allowed=float(decision.overpay_allowed),
                    confidence=float(decision.confidence),
                    reasons=decision.reasons,
                    counter=counter_prop,
                    meta=dict(decision.meta or {}),
                )

                # Persist counter offer in-session (for UI + fast-accept).
                try:
                    counter_hash = counter_prop.meta.get("counter_hash") if isinstance(counter_prop.meta, dict) else None
                    deal_payload = None
                    if isinstance(counter_prop.meta, dict):
                        deal_payload = counter_prop.meta.get("deal_serialized")
                    if not isinstance(deal_payload, dict):
                        # Defensive fallback
                        deal_payload = serialize_deal(counter_prop.deal)

                    negotiation_store.set_last_counter(
                        req.session_id,
                        {
                            "counter_hash": counter_hash,
                            "counter_deal": deal_payload,
                            "strategy": counter_prop.meta.get("strategy") if isinstance(counter_prop.meta, dict) else None,
                            "diff": counter_prop.meta.get("diff") if isinstance(counter_prop.meta, dict) else None,
                            "message": counter_prop.meta.get("message") if isinstance(counter_prop.meta, dict) else None,
                            "generated_at": in_game_date.isoformat(),
                            "base_hash": counter_prop.meta.get("base_hash") if isinstance(counter_prop.meta, dict) else None,
                            "ai_evaluation": eval_summary,
                        },
                    )
                except Exception:
                    pass

                # Push a GM-style message for the counter.
                try:
                    msg = ""
                    if isinstance(counter_prop.meta, dict):
                        msg = str(counter_prop.meta.get("message") or "")
                    msg = msg.strip() if msg else ""
                    if not msg:
                        msg = f"[{other_team_id}] COUNTER"
                    negotiation_store.append_message(req.session_id, speaker="OTHER_GM", text=msg)
                    negotiation_store.set_phase(req.session_id, "COUNTER_PENDING")
                except Exception:
                    pass

                # Response: counter details are embedded in ai_decision.counter (SSOT).

                return {
                    "ok": True,
                    "accepted": False,
                    "counter_unimplemented": False,
                    "deal": serialize_deal(deal),
                    "ai_verdict": to_jsonable(decision.verdict),
                    "ai_decision": to_jsonable(decision),
                    "ai_evaluation": eval_summary,
                }

            # If we couldn't build a legal/acceptable counter, fall back conservatively to REJECT.
            decision = DealDecision(
                verdict=DealVerdict.REJECT,
                required_surplus=float(decision.required_surplus),
                overpay_allowed=float(decision.overpay_allowed),
                confidence=float(decision.confidence),
                reasons=tuple(decision.reasons)
                + (
                    DecisionReason(
                        code="COUNTER_BUILD_FAILED",
                        message="Could not generate a legal counter offer",
                    ),
                ),
                counter=None,
                meta=dict(decision.meta or {}),
            )
            verdict = DealVerdict.REJECT


        # Build a short reason string for UI
        try:
            reason_lines = []
            for r in (decision.reasons or [])[:4]:
                if isinstance(r, dict):
                    msg = r.get("message") or r.get("code") or ""
                else:
                    msg = getattr(r, "message", None) or getattr(r, "code", None) or ""
                if msg:
                    reason_lines.append(str(msg))
            reason_text = " | ".join(reason_lines) if reason_lines else "AI rejected the offer."
        except Exception:
            reason_text = "AI rejected the offer."

        # Record rejection in session (message + phase)
        try:
            negotiation_store.append_message(
                req.session_id,
                speaker="OTHER_GM",
                text=f"[{other_team_id}] {verdict}: {reason_text}",
            )
            negotiation_store.set_phase(req.session_id, "REJECTED")
        except Exception:
            pass

        return {
            "ok": True,
            "accepted": False,
            "counter_unimplemented": False,
            "deal": serialize_deal(deal),
            "ai_verdict": to_jsonable(decision.verdict),
            "ai_decision": to_jsonable(decision),
            "ai_evaluation": eval_summary,
        }
    except TradeError as exc:
        return _trade_error_response(exc)


@app.post("/api/trade/evaluate")
async def api_trade_evaluate(req: TradeEvaluateRequest):
    """
    Debug endpoint: evaluate a proposed deal from a single team's perspective.
    Flow:
      deal = canonicalize_deal(parse_deal(req.deal))
      validate_deal(deal, current_date=in_game_date)
      trades.valuation.service.evaluate_deal_for_team(...)
      return decision + breakdown
    """
    try:
        in_game_date = state.get_current_date_as_date()
        db_path = state.get_db_path()

        deal = canonicalize_deal(parse_deal(req.deal))
        # Hot path: debug / UI-driven repeated calls.
        # Integrity is checked at startup and after any DB writes.
        validate_deal(deal, current_date=in_game_date, db_path=db_path, integrity_check=False)

        # Local import to avoid hard dependency during incremental integration.
        from trades.valuation.service import evaluate_deal_for_team as eval_service  # type: ignore
        from trades.valuation.types import to_jsonable  # type: ignore

        decision, evaluation = eval_service(
            deal=deal,
            team_id=req.team_id,
            current_date=in_game_date,
            db_path=db_path,
            include_breakdown=bool(req.include_breakdown),
            # We already validated above; avoid duplicate validate_deal in service.
            validate=False,
        )

        return {
            "ok": True,
            "team_id": str(req.team_id).upper(),
            "deal": serialize_deal(deal),
            "decision": to_jsonable(decision),
            "evaluation": to_jsonable(evaluation),
        }
    except TradeError as exc:
        return _trade_error_response(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Trade evaluation failed: {exc}")


# -------------------------------------------------------------------------
# 로스터 요약 API (LLM 컨텍스트용)
# -------------------------------------------------------------------------
@app.get("/api/roster-summary/{team_id}")
async def roster_summary(team_id: str):
    """특정 팀의 로스터를 LLM이 보기 좋은 형태로 요약해서 돌려준다."""
    db_path = state.get_db_path()
    team_id = str(normalize_team_id(team_id, strict=True))
    with LeagueRepo(db_path) as repo:
        # DB schema is guaranteed during server startup (state.startup_init_state()).
        roster = repo.get_team_roster(team_id)

    if not roster:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found in roster")

    players: List[Dict[str, Any]] = []
    for row in roster:
        players.append({
            "player_id": row.get("player_id"),
            "name": row.get("name"),
            "pos": str(row.get("pos") or ""),
            "overall": float(row.get("ovr") or 0.0),
        })

    players = sorted(players, key=lambda x: x["overall"], reverse=True)

    return {
        "team_id": team_id,
        "players": players[:12],
    }


# -------------------------------------------------------------------------
# 팀별 시즌 스케줄 조회 API
# -------------------------------------------------------------------------
@app.get("/api/team-schedule/{team_id}")
async def team_schedule(team_id: str):
    """마스터 스케줄 기준으로 특정 팀의 전체 시즌 일정을 반환."""
    team_id = team_id.upper()
    if team_id not in ALL_TEAM_IDS:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found in league")

    # (startup 보장 전제) 마스터 스케줄은 이미 초기화되어 있어야 함
    league = state.export_full_state_snapshot().get("league", {})
    master_schedule = league.get("master_schedule", {})
    games = master_schedule.get("games") or []

    if not games:
        raise HTTPException(
            status_code=500,
            detail="Master schedule is not initialized. Expected server startup_init_state() to run.",
        )
        

    team_games: List[Dict[str, Any]] = [
        g for g in games
        if g.get("home_team_id") == team_id or g.get("away_team_id") == team_id
    ]
    team_games.sort(key=lambda g: (g.get("date"), g.get("game_id")))

    formatted_games: List[Dict[str, Any]] = []
    for g in team_games:
        home_score = g.get("home_score")
        away_score = g.get("away_score")
        result_for_team = None
        if home_score is not None and away_score is not None:
            if team_id == g.get("home_team_id"):
                result_for_team = "W" if home_score > away_score else "L"
            else:
                result_for_team = "W" if away_score > home_score else "L"

        formatted_games.append({
            "game_id": g.get("game_id"),
            "date": g.get("date"),
            "home_team_id": g.get("home_team_id"),
            "away_team_id": g.get("away_team_id"),
            "home_score": home_score,
            "away_score": away_score,
            "result_for_user_team": result_for_team,
        })

    return {
        "team_id": team_id,
        "games": formatted_games,
    }


# -------------------------------------------------------------------------
# STATE 요약 조회 API (프론트/디버그용)
# -------------------------------------------------------------------------

@app.get("/api/state/summary")
async def state_summary():
    workflow_state: Dict[str, Any] = state.export_workflow_state()
    for k in (
        # Trade assets ledger (DB SSOT)
        "draft_picks",
        "swap_rights",
        "fixed_assets",
        # Transactions ledger (DB SSOT)
        "transactions",
        # Contracts/FA ledger (DB SSOT)
        "contracts",
        "player_contracts",
        "active_contract_id_by_player",
        "free_agents",
        # GM profiles (DB SSOT)
        "gm_profiles",
    ):
        workflow_state.pop(k, None)

    # 2) DB snapshot (SSOT). Fail loud on DB path/schema issues.
    db_path = state.get_db_path()
    try:
        with LeagueRepo(db_path) as repo:
            # DB schema is guaranteed during server startup (state.startup_init_state()).
            db_snapshot: Dict[str, Any] = {
                "ok": True,
                "db_path": db_path,
                "trade_assets": repo.get_trade_assets_snapshot(),
                "contracts_ledger": repo.get_contract_ledger_snapshot(),
                "transactions": repo.list_transactions(limit=200),
                "gm_profiles": repo.get_all_gm_profiles(),
            }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "DB snapshot failed",
                "db_path": db_path,
                "error": str(exc),
            },
        )

    return {
        "workflow_state": workflow_state,
        "db_snapshot": db_snapshot,
    }


@app.post("/api/game/new")
async def api_game_new(req: GameNewRequest):
    try:
        return create_new_game(
            slot_name=req.slot_name,
            slot_id=req.slot_id,
            season_year=req.season_year,
            user_team_id=req.user_team_id,
            overwrite_if_exists=bool(req.overwrite_if_exists),
        )
    except SaveError as exc:
        msg = str(exc)
        status = 409 if "already exists" in msg else 400
        raise HTTPException(status_code=status, detail=msg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create new game: {exc}")


@app.post("/api/game/save")
async def api_game_save(req: GameSaveRequest):
    try:
        return save_game(slot_id=req.slot_id, save_name=req.save_name, note=req.note)
    except SaveError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save game: {exc}")


@app.get("/api/game/saves")
async def api_game_saves():
    try:
        return list_save_slots()
    except SaveError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list saves: {exc}")


@app.get("/api/game/saves/{slot_id}")
async def api_game_save_detail(slot_id: str, strict: bool = False):
    try:
        return get_save_slot_detail(slot_id=slot_id, strict=bool(strict))
    except SaveError as exc:
        msg = str(exc)
        status = 404 if "not found" in msg else 400
        raise HTTPException(status_code=status, detail=msg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get save detail: {exc}")


@app.post("/api/game/load")
async def api_game_load(req: GameLoadRequest):
    try:
        return load_game(
            slot_id=req.slot_id,
            strict=bool(req.strict),
            expected_save_version=req.expected_save_version,
        )
    except SaveError as exc:
        msg = str(exc)
        status = 404 if "not found" in msg else 409 if "mismatch" in msg else 400
        raise HTTPException(status_code=status, detail=msg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load game: {exc}")


@app.get("/api/debug/schedule-summary")
async def debug_schedule_summary():
    """마스터 스케줄 생성/검증용 디버그 엔드포인트."""
    return state.get_schedule_summary()























