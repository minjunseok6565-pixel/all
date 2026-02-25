from __future__ import annotations

import os
import sqlite3
from datetime import date
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import BASE_DIR
import state
from analytics.stats.leaders import compute_leaderboards
from team_utils import get_conference_standings, get_team_cards, get_team_detail

router = APIRouter()

static_dir = os.path.join(BASE_DIR, "static")

@router.get("/")
async def root():
    """간단한 헬스체크 및 NBA.html 링크 안내."""
    index_path = os.path.join(static_dir, "NBA.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "느바 시뮬 GM 서버입니다. /static/NBA.html 을 확인하세요."}

@router.get("/api/stats/leaders")
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


@router.get("/api/stats/playoffs/leaders")
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


@router.get("/api/standings")
async def api_standings():
    return get_conference_standings()


@router.get("/api/teams")
async def api_teams():
    return get_team_cards()


@router.get("/api/team-detail/{team_id}")
async def api_team_detail(team_id: str):
    try:
        return get_team_detail(team_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# -------------------------------------------------------------------------
# College (Read-only / UI) API
# -------------------------------------------------------------------------




@router.get("/api/player-detail/{player_id}")
async def api_player_detail(player_id: str, season_year: Optional[int] = None):
    """Return rich player detail for My Team UI."""
    pid = str(normalize_player_id(player_id, strict=False, allow_legacy_numeric=True))
    if not pid:
        raise HTTPException(status_code=400, detail="Invalid player_id")

    db_path = state.get_db_path()
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(season_year or (league_ctx.get("season_year") or 0))

    workflow_state = state.export_workflow_state()
    season_player_stats = (workflow_state.get("player_stats") or {}) if isinstance(workflow_state, dict) else {}
    season_stats_entry = season_player_stats.get(pid) or {}

    with LeagueRepo(db_path) as repo:
        repo.init_db()
        try:
            player = repo.get_player(pid)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

        with repo.transaction() as cur:
            roster_row = cur.execute(
                """
                SELECT team_id, salary_amount, status
                FROM roster
                WHERE player_id=?
                LIMIT 1;
                """,
                (pid,),
            ).fetchone()

            active_contract_row = cur.execute(
                """
                SELECT *
                FROM contracts
                WHERE player_id=? AND COALESCE(is_active,0)=1
                ORDER BY updated_at DESC
                LIMIT 1;
                """,
                (pid,),
            ).fetchone()

            contract_rows = cur.execute(
                """
                SELECT *
                FROM contracts
                WHERE player_id=?
                ORDER BY start_season_year DESC, updated_at DESC;
                """,
                (pid,),
            ).fetchall()

            two_way_row = cur.execute(
                """
                SELECT COALESCE(contract_json,'') AS contract_json
                FROM contracts
                WHERE player_id=?
                  AND UPPER(COALESCE(contract_type,''))='TWO_WAY'
                  AND UPPER(COALESCE(status,''))='ACTIVE'
                  AND COALESCE(is_active, 0)=1
                ORDER BY updated_at DESC
                LIMIT 1;
                """,
                (pid,),
            ).fetchone()

            two_way = {"is_two_way": False, "game_limit": None, "games_used": 0, "games_remaining": None}
            if two_way_row:
                contract_data: Dict[str, Any] = {}
                raw_cd = two_way_row["contract_json"]
                if raw_cd:
                    try:
                        contract_data = json.loads(str(raw_cd))
                    except Exception:
                        contract_data = {}
                game_limit = int(contract_data.get("two_way_game_limit") or 50)
                used = cur.execute(
                    "SELECT COUNT(1) AS n FROM two_way_appearances WHERE player_id=? AND season_year=?;",
                    (pid, sy),
                ).fetchone()
                games_used = int((used["n"] if used is not None else 0) or 0)
                two_way = {
                    "is_two_way": True,
                    "game_limit": game_limit,
                    "games_used": games_used,
                    "games_remaining": max(0, int(game_limit) - int(games_used)),
                }

            from agency import repo as agency_repo
            agency_state = agency_repo.get_player_agency_states(cur, [pid]).get(pid)

            from injury import repo as injury_repo
            injury_state = injury_repo.get_player_injury_states(cur, [pid]).get(pid)

            from fatigue import repo as fatigue_repo
            fatigue_state = fatigue_repo.get_player_fatigue_states(cur, [pid]).get(pid)

            from readiness import repo as readiness_repo
            sharpness_state = None
            if sy > 0:
                sharpness_state = readiness_repo.get_player_sharpness_states(
                    cur,
                    [pid],
                    season_year=int(sy),
                ).get(pid)

    from contract_codec import contract_from_row

    active_contract = contract_from_row(active_contract_row) if active_contract_row else None
    contracts = [contract_from_row(r) for r in contract_rows]

    dissatisfaction = {
        "is_dissatisfied": False,
        "state": agency_state,
    }
    if isinstance(agency_state, dict):
        axes = [
            float(agency_state.get("team_frustration") or 0.0),
            float(agency_state.get("role_frustration") or 0.0),
            float(agency_state.get("contract_frustration") or 0.0),
            float(agency_state.get("health_frustration") or 0.0),
            float(agency_state.get("chemistry_frustration") or 0.0),
            float(agency_state.get("usage_frustration") or 0.0),
        ]
        trade_request_level = int(agency_state.get("trade_request_level") or 0)
        dissatisfaction["is_dissatisfied"] = trade_request_level > 0 or any(v >= 0.5 for v in axes)

    injury = {
        "is_injured": False,
        "status": "HEALTHY",
        "state": injury_state,
    }
    fatigue = fatigue_state or {}
    st_fatigue = float(fatigue.get("st", 0.0) or 0.0)
    lt_fatigue = float(fatigue.get("lt", 0.0) or 0.0)
    sharpness = float((sharpness_state or {}).get("sharpness", 50.0) or 50.0)
    if isinstance(injury_state, dict):
        status = str(injury_state.get("status") or "HEALTHY").upper()
        injury["status"] = status
        injury["is_injured"] = status in {"OUT", "RETURNING"}

    return {
        "ok": True,
        "player": {
            "player_id": pid,
            "name": player.get("name"),
            "pos": player.get("pos"),
            "age": player.get("age"),
            "height_in": player.get("height_in"),
            "weight_lb": player.get("weight_lb"),
            "ovr": player.get("ovr"),
            "attrs": player.get("attrs") or {},
        },
        "roster": {
            "team_id": (str(roster_row["team_id"]).upper() if roster_row and roster_row["team_id"] else None),
            "status": (str(roster_row["status"]) if roster_row and roster_row["status"] else None),
            "salary_amount": int(roster_row["salary_amount"] or 0) if roster_row else None,
        },
        "contract": {
            "active": active_contract,
            "all": contracts,
        },
        "dissatisfaction": dissatisfaction,
        "season_stats": season_stats_entry,
        "two_way": two_way,
        "condition": {
            "short_term_fatigue": st_fatigue,
            "long_term_fatigue": lt_fatigue,
            "short_term_stamina": max(0.0, 1.0 - st_fatigue),
            "long_term_stamina": max(0.0, 1.0 - lt_fatigue),
            "sharpness": sharpness,
            "fatigue_state": fatigue_state,
            "sharpness_state": sharpness_state,
        },
        "injury": injury,
    }













@router.get("/api/roster-summary/{team_id}")
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


@router.get("/api/two-way/summary/{team_id}")
async def two_way_summary(team_id: str):
    """특정 팀의 투웨이 슬롯/출전 가능 경기 수를 요약해서 반환한다."""
    db_path = state.get_db_path()
    tid = str(normalize_team_id(team_id, strict=True))
    season_year = int((state.export_full_state_snapshot().get("league", {}) or {}).get("season_year") or 0)

    with LeagueRepo(db_path) as repo:
        with repo.transaction() as cur:
            rows = cur.execute(
                """
                SELECT c.player_id, p.name,
                       COALESCE(c.contract_type,'') AS contract_type,
                       COALESCE(c.status,'') AS status,
                       COALESCE(c.contract_json,'') AS contract_json
                FROM contracts c
                LEFT JOIN players p ON p.player_id = c.player_id
                WHERE c.team_id=?
                  AND UPPER(COALESCE(c.contract_type,''))='TWO_WAY'
                  AND UPPER(COALESCE(c.status,''))='ACTIVE'
                  AND COALESCE(c.is_active, 0)=1
                ORDER BY p.name ASC;
                """,
                (tid,),
            ).fetchall()

            players: List[Dict[str, Any]] = []
            for r in rows:
                player_id = str(r["player_id"])
                contract_data_raw = r["contract_json"]
                contract_data: Dict[str, Any] = {}
                if contract_data_raw:
                    try:
                        contract_data = json.loads(str(contract_data_raw))
                    except Exception:
                        contract_data = {}

                limit = int(contract_data.get("two_way_game_limit") or 50)
                used = cur.execute(
                    "SELECT COUNT(1) AS n FROM two_way_appearances WHERE player_id=? AND season_year=?;",
                    (player_id, season_year),
                ).fetchone()
                used_i = int((used["n"] if used is not None else 0) or 0)
                players.append(
                    {
                        "player_id": player_id,
                        "name": r["name"],
                        "contract_type": "TWO_WAY",
                        "game_limit": limit,
                        "games_used": used_i,
                        "games_remaining": max(0, int(limit) - int(used_i)),
                    }
                )

    max_slots = 3
    return {
        "team_id": tid,
        "season_year": season_year,
        "max_two_way_slots": max_slots,
        "used_two_way_slots": len(players),
        "open_two_way_slots": max(0, max_slots - len(players)),
        "players": players,
    }


# -------------------------------------------------------------------------
# 팀별 시즌 스케줄 조회 API
# -------------------------------------------------------------------------
@router.get("/api/team-schedule/{team_id}")
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

@router.get("/api/state/summary")
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
