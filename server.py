from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from typing import Any, Dict, Optional, List, Literal

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import BASE_DIR, ALL_TEAM_IDS
from league_repo import LeagueRepo
from league_service import LeagueService
from schema import normalize_team_id
import state
from sim.league_sim import simulate_single_game, advance_league_until
from playoffs import (
    auto_advance_current_round,
    advance_my_team_one_game,
    build_postseason_field,
    initialize_postseason,
    play_my_team_play_in_game,
    reset_postseason_state,
)
from news_ai import refresh_playoff_news, refresh_weekly_news
from stats_util import compute_league_leaders, compute_playoff_league_leaders
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


class PostseasonSetupRequest(BaseModel):
    my_team_id: str
    use_random_field: bool = False


class EmptyRequest(BaseModel):
    pass


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
    team_id: str
    player_id: str
    signed_date: Optional[str] = None  # YYYY-MM-DD (default: in-game date)
    years: int = 1
    salary_by_year: Optional[Dict[int, int]] = None  # {season_year: salary}


class ReSignOrExtendRequest(BaseModel):
    team_id: str
    player_id: str
    signed_date: Optional[str] = None  # YYYY-MM-DD (default: in-game date)
    years: int = 1
    salary_by_year: Optional[Dict[int, int]] = None  # {season_year: salary}


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
# 리그 리더 / 스탠딩 / 팀 API
# -------------------------------------------------------------------------


@app.get("/api/stats/leaders")
async def api_stats_leaders():
    # The frontend expects a flat object with an uppercase stat key (e.g., PTS)
    # under `data.leaders`. Some previous iterations of the API wrapped this
    # structure under stats.leaderboards with lowercase keys, which caused the
    # UI to break. Normalize here so the client always receives
    # `{ leaders: { PTS: [...], AST: [...], ... }, updated_at: <iso date> }`.
    workflow_state = state.export_workflow_state()
    leaders = compute_league_leaders(workflow_state.get("player_stats") or {})
    current_date = state.get_current_date()
    return {"leaders": leaders, "updated_at": current_date}


@app.get("/api/stats/playoffs/leaders")
async def api_playoff_stats_leaders():
    workflow_state = state.export_workflow_state()
    playoff_stats = (workflow_state.get("phase_results") or {}).get("playoffs", {}).get("player_stats") or {}
    leaders = compute_playoff_league_leaders(playoff_stats)
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
            "/api/offseason/training/apply-growth",
            "/api/offseason/draft/lottery",
            "/api/offseason/draft/settle",
            "/api/offseason/draft/combine",
            "/api/offseason/draft/workouts",
            "/api/offseason/draft/interviews",
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
        result = process_offseason(
            snap,
            from_season_year=int(from_year),
            to_season_year=int(to_year),
            decision_date_iso=str(decision_date_iso),
            decision_policy=None,
        )
        return {"ok": True, "from_season_year": int(from_year), "to_season_year": int(to_year), "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    try:
        from training.checkpoints import ensure_last_regular_month_tick

        snap = state.export_full_state_snapshot()
        final_month_tick = ensure_last_regular_month_tick(
            db_path=str(db_path),
            now_date_iso=str(in_game_date),
            state_snapshot=snap,
        )
    except Exception:
        final_month_tick = {"ok": True, "skipped": True, "reason": "tick_check_failed"}

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
    return result


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

        applied_count = int(apply_selections(db_path=db_path, draft_year=int(draft_year), tx_date_iso=tx_date_iso))

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

        # Some endpoints previously wrapped the news payload like
        # `{ "news": { "current_date": ..., "items": [...] } }`, which the
        # frontend does not expect. Normalize it back to the raw shape.
        if isinstance(payload, dict) and "news" in payload and isinstance(
            payload["news"], dict
        ):
            payload = payload["news"]

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


@app.post("/api/contracts/sign-free-agent")
async def api_contracts_sign_free_agent(req: SignFreeAgentRequest):
    """Sign a free agent (DB write): roster.team_id + contract + active contract."""
    try:
        db_path = state.get_db_path()
        in_game_date = state.get_current_date_as_date()
        with LeagueRepo(db_path) as repo:
            svc = LeagueService(repo)
            event = svc.sign_free_agent(
                team_id=req.team_id,
                player_id=req.player_id,
                signed_date=req.signed_date or in_game_date,
                years=req.years,
                salary_by_year=req.salary_by_year,
            )
        _validate_repo_integrity(db_path)
        event_dict = event.to_dict()
        affected = event_dict.get("affected_player_ids") or []
        _try_ui_cache_refresh_players(list(affected), context="contracts.sign_free_agent")
        return {"ok": True, "event": event_dict}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sign-free-agent failed: {e}")


@app.post("/api/contracts/re-sign-or-extend")
async def api_contracts_re_sign_or_extend(req: ReSignOrExtendRequest):
    """Re-sign / extend a player (DB write): contract + active contract (+ roster salary sync)."""
    try:
        db_path = state.get_db_path()
        in_game_date = state.get_current_date_as_date()
        with LeagueRepo(db_path) as repo:
            svc = LeagueService(repo)
            event = svc.re_sign_or_extend(
                team_id=req.team_id,
                player_id=req.player_id,
                signed_date=req.signed_date or in_game_date,
                years=req.years,
                salary_by_year=req.salary_by_year,
            )
        _validate_repo_integrity(db_path)
        event_dict = event.to_dict()
        affected = event_dict.get("affected_player_ids") or []
        _try_ui_cache_refresh_players(list(affected), context="contracts.re_sign_or_extend")
        return {"ok": True, "event": event_dict}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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

        # ------------------------------------------------------------------
        # AI evaluation (other team perspective)
        # NOTE:
        # - legality is already checked by validate_deal above
        # - valuation service will build DecisionContext internally (team_situation + gm profile)
        # ------------------------------------------------------------------
        other_team_id = session["other_team_id"].upper()

        # Local imports to keep integration flexible.
        from trades.valuation.service import evaluate_deal_for_team as eval_service  # type: ignore
        from trades.valuation.types import to_jsonable, DealVerdict  # type: ignore

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

        # Record AI response in-session for debugging / UI explanations
        try:
            negotiation_store.set_last_counter(
                req.session_id,
                {
                    "verdict": to_jsonable(decision.verdict),
                    "decision": to_jsonable(decision),
                    "evaluation": eval_summary,
                },
            )
        except Exception:
            # Session logging failure should not crash commit flow
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

        # COUNTER is not implemented yet -> treat as reject (or return explicit marker)
        counter_unimplemented = (verdict == DealVerdict.COUNTER)

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
            negotiation_store.set_phase(req.session_id, "REJECTED" if not counter_unimplemented else "COUNTER_PENDING")
        except Exception:
            pass

        return {
            "ok": True,
            "accepted": False,
            "counter_unimplemented": bool(counter_unimplemented),
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


@app.get("/api/debug/schedule-summary")
async def debug_schedule_summary():
    """마스터 스케줄 생성/검증용 디버그 엔드포인트."""
    return state.get_schedule_summary()



























