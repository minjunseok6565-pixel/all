from __future__ import annotations

import json
import os
import sqlite3
import hashlib
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import BASE_DIR, ALL_TEAM_IDS
from league_repo import LeagueRepo
from schema import normalize_player_id, normalize_team_id
import state
from analytics.stats.leaders import compute_leaderboards
from team_utils import get_conference_standings, get_conference_standings_table, get_team_cards, get_team_detail

router = APIRouter()

static_dir = os.path.join(BASE_DIR, "static")

TEAM_FULL_NAMES: Dict[str, str] = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons", "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans", "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


def _format_mmdd(date_value: Any) -> str:
    raw = str(date_value or "")[:10]
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        return f"{raw[5:7]}/{raw[8:10]}"
    return "--/--"


def _deterministic_tipoff_time(game_id: Any) -> str:
    slots = ("07:00 PM", "07:30 PM", "08:00 PM", "08:30 PM", "09:00 PM", "09:30 PM")
    digest = hashlib.md5(str(game_id or "").encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(slots)
    return slots[idx]


def _num_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_leader(rows: List[Dict[str, Any]], stat_keys: List[str]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_value = float("-inf")
    for row in rows:
        value = None
        for k in stat_keys:
            value = _num_or_none(row.get(k))
            if value is not None:
                break
        if value is None:
            continue
        if value > best_value:
            best_value = value
            best = {
                "player_id": str(row.get("PlayerID") or ""),
                "name": row.get("Name") or "",
                "value": int(value),
            }
    return best


def _attr_float(attrs: Any, *keys: str) -> Optional[float]:
    if not isinstance(attrs, dict):
        return None
    lower_map = {str(k).lower(): v for k, v in attrs.items()}
    for key in keys:
        raw = lower_map.get(str(key).lower())
        try:
            if raw is None:
                continue
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _risk_tier_from_inputs(
    *,
    st_fatigue: float,
    lt_fatigue: float,
    age: int,
    injury_freq: Optional[float],
    durability: Optional[float],
    reinjury_total: int,
) -> Dict[str, Any]:
    score = 0.0
    score += _clamp(st_fatigue, 0.0, 1.0) * 30.0
    score += _clamp(lt_fatigue, 0.0, 1.0) * 30.0
    score += _clamp((int(age) - 28) / 10.0, 0.0, 1.0) * 15.0
    if injury_freq is not None:
        score += _clamp((float(injury_freq) - 1.0) / 9.0, 0.0, 1.0) * 15.0
    if durability is not None:
        score += (1.0 - _clamp(float(durability) / 100.0, 0.0, 1.0)) * 10.0
    score += _clamp(float(reinjury_total) / 5.0, 0.0, 1.0) * 10.0
    final = int(round(_clamp(score, 0.0, 100.0)))
    if final >= 67:
        tier = "HIGH"
    elif final >= 34:
        tier = "MEDIUM"
    else:
        tier = "LOW"
    return {"risk_score": final, "risk_tier": tier}



def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _build_risk_profile(
    *,
    row: Mapping[str, Any],
    injury_state: Mapping[str, Any],
    fatigue_state: Mapping[str, Any],
    sharpness_state: Mapping[str, Any],
    as_of_date: str,
) -> Dict[str, Any]:
    from injury.status import status_for_date

    attrs = row.get("attrs") if isinstance(row.get("attrs"), dict) else {}
    st = float((fatigue_state or {}).get("st", 0.0) or 0.0)
    lt = float((fatigue_state or {}).get("lt", 0.0) or 0.0)
    sharpness = float((sharpness_state or {}).get("sharpness", 50.0) or 50.0)
    age = int(row.get("age") or 0)
    injury_freq = _attr_float(attrs, "injury_freq", "injuryfrequency", "injury_frequency")
    durability = _attr_float(attrs, "durability", "dur")
    reinjury = injury_state.get("reinjury_count") if isinstance(injury_state, dict) else {}
    reinjury_total = sum(_safe_int(v, 0) for v in (reinjury or {}).values()) if isinstance(reinjury, dict) else 0
    risk = _risk_tier_from_inputs(
        st_fatigue=st,
        lt_fatigue=lt,
        age=age,
        injury_freq=injury_freq,
        durability=durability,
        reinjury_total=reinjury_total,
    )
    status = status_for_date(injury_state, on_date_iso=as_of_date)
    return {
        "player_id": str(row.get("player_id") or ""),
        "name": row.get("name"),
        "pos": row.get("pos"),
        "age": age,
        "injury_status": status,
        "injury_state": dict(injury_state or {}),
        "condition": {
            "short_term_fatigue": st,
            "long_term_fatigue": lt,
            "short_term_stamina": max(0.0, 1.0 - st),
            "long_term_stamina": max(0.0, 1.0 - lt),
            "sharpness": sharpness,
        },
        "risk_inputs": {
            "injury_freq": injury_freq,
            "durability": durability,
            "age": age,
            "lt_wear_proxy": lt,
            "energy_proxy": max(0.0, 1.0 - st),
            "reinjury_count": reinjury if isinstance(reinjury, dict) else {},
        },
        "risk_score": int(risk["risk_score"]),
        "risk_tier": risk["risk_tier"],
    }


def _row_stat_float(row: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        try:
            raw = row.get(key)
        except AttributeError:
            raw = None
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return float(default)


def _row_stat_int(row: Mapping[str, Any], *keys: str, default: int = 0) -> int:
    return int(round(_row_stat_float(row, *keys, default=float(default))))


def _row_player_id(row: Mapping[str, Any]) -> str:
    candidates = (
        row.get("PlayerID"),
        row.get("player_id"),
        row.get("PLAYER_ID"),
        row.get("id"),
    )
    for value in candidates:
        if value is None:
            continue
        sval = str(value).strip()
        if sval:
            return sval
    return ""


def _estimate_game_score(row: Mapping[str, Any]) -> float:
    pts = _row_stat_float(row, "PTS", "points")
    fgm = _row_stat_float(row, "FGM")
    fga = _row_stat_float(row, "FGA")
    ftm = _row_stat_float(row, "FTM")
    fta = _row_stat_float(row, "FTA")
    orb = _row_stat_float(row, "OREB", "ORB")
    drb = _row_stat_float(row, "DREB", "DRB")
    stl = _row_stat_float(row, "STL")
    ast = _row_stat_float(row, "AST")
    blk = _row_stat_float(row, "BLK")
    pf = _row_stat_float(row, "PF")
    tov = _row_stat_float(row, "TOV", "TO")
    score = (
        pts
        + 0.4 * fgm
        - 0.7 * fga
        - 0.4 * (fta - ftm)
        + 0.7 * orb
        + 0.3 * drb
        + stl
        + 0.7 * ast
        + 0.7 * blk
        - 0.4 * pf
        - tov
    )
    return round(score, 1)


def _ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


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


@router.get("/api/standings/table")
async def api_standings_table():
    return get_conference_standings_table()


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


@router.get("/api/players/{player_id}/game-log")
async def api_player_game_log(player_id: str, limit: int = 10):
    """Return latest completed game logs for a player from workflow game_results."""
    pid = str(normalize_player_id(player_id, strict=False, allow_legacy_numeric=True))
    if not pid:
        raise HTTPException(status_code=400, detail="Invalid player_id")

    limit = max(1, min(int(limit or 10), 30))

    workflow_state = state.export_workflow_state() or {}
    if not isinstance(workflow_state, dict):
        workflow_state = {}

    full_snapshot = state.export_full_state_snapshot()
    league = full_snapshot.get("league", {}) if isinstance(full_snapshot, dict) else {}
    season_id = str((league or {}).get("active_season_id") or "")
    game_results = workflow_state.get("game_results") or {}
    if not isinstance(game_results, dict):
        game_results = {}

    items: List[Dict[str, Any]] = []
    for game_id, gr in game_results.items():
        if not isinstance(gr, dict):
            continue
        game_date = str(gr.get("date") or "")[:10]
        teams = gr.get("teams")
        if not isinstance(teams, dict):
            continue

        found_row: Optional[Mapping[str, Any]] = None
        owner_team_id: Optional[str] = None
        for team_id, team_box in teams.items():
            players = (team_box or {}).get("players") if isinstance(team_box, dict) else None
            if not isinstance(players, list):
                continue
            for row in players:
                if not isinstance(row, dict):
                    continue
                if _row_player_id(row) == pid:
                    found_row = row
                    owner_team_id = str(team_id)
                    break
            if found_row is not None:
                break

        if found_row is None or owner_team_id is None:
            continue

        home_team_id = str(gr.get("home_team_id") or "")
        away_team_id = str(gr.get("away_team_id") or "")
        is_home = owner_team_id == home_team_id
        opponent_team_id = away_team_id if is_home else home_team_id

        fgm = _row_stat_int(found_row, "FGM")
        fga = _row_stat_int(found_row, "FGA")
        tpm = _row_stat_int(found_row, "3PM", "TPM")
        tpa = _row_stat_int(found_row, "3PA", "TPA")
        ftm = _row_stat_int(found_row, "FTM")
        fta = _row_stat_int(found_row, "FTA")
        pts = _row_stat_int(found_row, "PTS")
        reb = _row_stat_int(found_row, "REB", "TRB")
        ast = _row_stat_int(found_row, "AST")

        items.append(
            {
                "game_id": str(game_id),
                "date": game_date,
                "team_id": owner_team_id,
                "opponent_team_id": opponent_team_id,
                "is_home": bool(is_home),
                "minutes": round(_row_stat_float(found_row, "MIN", "minutes"), 1),
                "pts": pts,
                "reb": reb,
                "ast": ast,
                "stl": _row_stat_int(found_row, "STL"),
                "blk": _row_stat_int(found_row, "BLK"),
                "tov": _row_stat_int(found_row, "TOV", "TO"),
                "fgm": fgm,
                "fga": fga,
                "tpm": tpm,
                "tpa": tpa,
                "ftm": ftm,
                "fta": fta,
                "fg_pct": round(_ratio(fgm, fga), 3),
                "tp_pct": round(_ratio(tpm, tpa), 3),
                "ft_pct": round(_ratio(ftm, fta), 3),
                "plus_minus": _row_stat_int(found_row, "+/-", "PLUS_MINUS", "plus_minus"),
                "game_score": _estimate_game_score(found_row),
            }
        )

    items_sorted = sorted(items, key=lambda x: (str(x.get("date") or ""), str(x.get("game_id") or "")), reverse=True)
    recent = items_sorted[:limit]

    last_n = recent[: min(5, len(recent))]
    if last_n:
        ts_num = sum(r.get("pts", 0.0) for r in last_n)
        ts_den = 2.0 * sum((r.get("fga", 0.0) + 0.44 * r.get("fta", 0.0)) for r in last_n)
        summary_last_5 = {
            "games": len(last_n),
            "pts": round(sum(r.get("pts", 0.0) for r in last_n) / len(last_n), 1),
            "reb": round(sum(r.get("reb", 0.0) for r in last_n) / len(last_n), 1),
            "ast": round(sum(r.get("ast", 0.0) for r in last_n) / len(last_n), 1),
            "game_score": round(sum(r.get("game_score", 0.0) for r in last_n) / len(last_n), 1),
            "ts_pct": round(_ratio(ts_num, ts_den), 3),
        }
    else:
        summary_last_5 = {"games": 0, "pts": 0.0, "reb": 0.0, "ast": 0.0, "game_score": 0.0, "ts_pct": 0.0}

    return {
        "player_id": pid,
        "season_id": season_id,
        "limit": limit,
        "items": recent,
        "summary_last_5": summary_last_5,
    }


@router.get("/api/team/{team_id}/roster-insights")
async def api_team_roster_insights(team_id: str, top_n: int = 3):
    """Roster decision-support summary for UI cards."""
    tid = str(normalize_team_id(team_id, strict=True))
    top_n = max(1, min(int(top_n or 3), 10))

    detail = get_team_detail(tid)
    roster = detail.get("roster") or []
    if not isinstance(roster, list):
        roster = []

    medical = await api_medical_team_overview(team_id=tid, top_n=max(top_n, 5))
    summary = (detail.get("summary") or {}) if isinstance(detail, dict) else {}
    as_of_date = str((medical or {}).get("as_of_date") or state.get_current_date_as_date().isoformat())

    def position_bucket(pos: str) -> str:
        p = str(pos or "").upper()
        if p in {"PG", "SG", "G"}:
            return "guards"
        if p in {"SF", "PF", "F"}:
            return "forwards"
        if p in {"C"}:
            return "centers"
        return "unknown"

    pos_counts = {"guards": 0, "forwards": 0, "centers": 0, "unknown": 0}
    value_rows: List[Dict[str, Any]] = []
    for row in roster:
        if not isinstance(row, dict):
            continue
        pos_counts[position_bucket(str(row.get("pos") or ""))] += 1
        salary_m = max(_row_stat_float(row, "salary", default=0.0) / 1_000_000.0, 0.0)
        production = (
            _row_stat_float(row, "pts")
            + 0.7 * _row_stat_float(row, "reb")
            + 0.7 * _row_stat_float(row, "ast")
        )
        sharpness_adj = 0.7 + 0.3 * _clamp(_row_stat_float(row, "sharpness", default=50.0) / 100.0, 0.0, 1.0)
        durability_adj = 0.8 + 0.2 * _clamp(_row_stat_float(row, "long_term_stamina", default=0.5), 0.0, 1.0)
        value_score = (production * sharpness_adj * durability_adj) / max(0.2, salary_m)
        value_rows.append(
            {
                "player_id": str(row.get("player_id") or ""),
                "name": row.get("name"),
                "pos": row.get("pos"),
                "salary": _row_stat_float(row, "salary"),
                "production_proxy": round(production, 2),
                "value_score": round(value_score * 10.0, 1),
            }
        )

    value_rows_sorted = sorted(value_rows, key=lambda x: (float(x.get("value_score") or 0.0), str(x.get("name") or "")), reverse=True)
    top_positive = value_rows_sorted[:top_n]
    top_negative = list(reversed(value_rows_sorted[-top_n:])) if value_rows_sorted else []
    team_value_score = round(
        (sum(float(r.get("value_score") or 0.0) for r in value_rows_sorted) / len(value_rows_sorted)) if value_rows_sorted else 0.0,
        1,
    )

    high_risk_count = int((((medical or {}).get("summary") or {}).get("risk_tier_counts") or {}).get("HIGH") or 0)
    unavailable_count = len((((medical or {}).get("watchlists") or {}).get("currently_unavailable") or []))
    health_score = _clamp((high_risk_count * 22.0) + (unavailable_count * 18.0), 0.0, 100.0)
    health_risk_index = {
        "score": int(round(health_score)),
        "tier": "HIGH" if health_score >= 67 else ("MEDIUM" if health_score >= 34 else "LOW"),
        "high_risk_count": high_risk_count,
        "unavailable_count": unavailable_count,
    }

    need_flags: List[str] = []
    if pos_counts["centers"] <= 1:
        need_flags.append("BACKUP_CENTER_THIN")
    if pos_counts["guards"] <= 2:
        need_flags.append("BALL_HANDLER_THIN")
    if pos_counts["forwards"] <= 2:
        need_flags.append("WING_DEPTH_THIN")
    if not need_flags:
        need_flags.append("BALANCED")

    return {
        "team_id": tid,
        "as_of_date": as_of_date,
        "record": {
            "wins": int(summary.get("wins") or 0),
            "losses": int(summary.get("losses") or 0),
            "rank": summary.get("rank"),
        },
        "health_risk_index": health_risk_index,
        "salary_efficiency": {
            "team_value_score": team_value_score,
            "top_positive": top_positive,
            "top_negative": top_negative,
        },
        "rotation_balance": {
            "guards": pos_counts["guards"],
            "forwards": pos_counts["forwards"],
            "centers": pos_counts["centers"],
            "unknown": pos_counts["unknown"],
            "need_flags": need_flags,
        },
    }


@router.get("/api/meta/attribute-groups")
async def api_meta_attribute_groups():
    """Attribute grouping metadata for UI rendering.

    Uses only existing attribute key names already present in player attrs.
    """
    return {
        "version": 1,
        "groups": [
            {
                "group_key": "finishing",
                "label": "마무리",
                "attr_keys": ["DRIVING_LAYUP", "DRIVING_DUNK", "STANDING_DUNK", "CLOSE_SHOT", "DRAW_FOUL"],
            },
            {
                "group_key": "shooting",
                "label": "슈팅",
                "attr_keys": ["MID-RANGE_SHOT", "THREE_POINT_SHOT", "FREE_THROW", "SHOT_IQ", "OFFENSIVE_CONSISTENCY"],
            },
            {
                "group_key": "playmaking",
                "label": "플레이메이킹",
                "attr_keys": ["PASS_ACCURACY", "PASS_VISION", "PASS_IQ", "BALL_HANDLE", "HANDS"],
            },
            {
                "group_key": "defense",
                "label": "수비",
                "attr_keys": ["INTERIOR_DEFENSE", "PERIMETER_DEFENSE", "STEAL", "BLOCK", "HELP_DEFENSE_IQ", "DEFENSIVE_CONSISTENCY"],
            },
            {
                "group_key": "rebounding",
                "label": "리바운드",
                "attr_keys": ["OFFENSIVE_REBOUND", "DEFENSIVE_REBOUND"],
            },
            {
                "group_key": "physical",
                "label": "피지컬",
                "attr_keys": ["SPEED", "ACCELERATION", "STRENGTH", "VERTICAL", "STAMINA", "HUSTLE", "OVERALL_DURABILITY", "L_INJURYFREQ"],
            },
            {
                "group_key": "mental",
                "label": "멘탈",
                "attr_keys": ["INTANGIBLES", "M_ADAPTABILITY", "M_AMBITION", "M_LOYALTY", "M_EGO", "M_COACHABILITY", "M_WORKETHIC"],
            },
        ],
    }


@router.get("/api/medical/team/{team_id}/injury-risk")
async def api_medical_injury_risk(
    team_id: str,
    season_year: Optional[int] = None,
    min_risk_tier: Optional[str] = None,
    include_healthy_only: bool = True,
):
    db_path = state.get_db_path()
    tid = str(normalize_team_id(team_id, strict=True))
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(season_year or (league_ctx.get("season_year") or 0))
    as_of_date = state.get_current_date_as_date().isoformat()

    tier_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    min_tier = str(min_risk_tier or "LOW").upper()
    if min_tier not in tier_order:
        raise HTTPException(status_code=400, detail="min_risk_tier must be one of: LOW, MEDIUM, HIGH")

    with LeagueRepo(db_path) as repo:
        repo.init_db()
        roster_rows = repo.get_team_roster(tid)
        pids = [str(r.get("player_id")) for r in (roster_rows or []) if r.get("player_id")]
        fatigue_by_pid: Dict[str, Dict[str, Any]] = {}
        sharpness_by_pid: Dict[str, Dict[str, Any]] = {}
        injury_by_pid: Dict[str, Dict[str, Any]] = {}

        if pids:
            from fatigue import repo as fatigue_repo
            from readiness import repo as readiness_repo
            from injury import repo as injury_repo

            with repo.transaction() as cur:
                fatigue_by_pid = fatigue_repo.get_player_fatigue_states(cur, pids)
                if sy > 0:
                    sharpness_by_pid = readiness_repo.get_player_sharpness_states(cur, pids, season_year=sy)
                injury_by_pid = injury_repo.get_player_injury_states(cur, pids)

    from injury.status import status_for_date

    items: List[Dict[str, Any]] = []
    for row in roster_rows:
        pid = str(row.get("player_id"))
        attrs = row.get("attrs") if isinstance(row.get("attrs"), dict) else {}
        injury_state = injury_by_pid.get(pid) or {}
        normalized_status = status_for_date(injury_state, on_date_iso=as_of_date)
        if include_healthy_only and normalized_status != "HEALTHY":
            continue

        fatigue_row = fatigue_by_pid.get(pid) or {}
        st = float(fatigue_row.get("st", 0.0) or 0.0)
        lt = float(fatigue_row.get("lt", 0.0) or 0.0)
        sharpness = float((sharpness_by_pid.get(pid) or {}).get("sharpness", 50.0) or 50.0)
        age = int(row.get("age") or 0)
        injury_freq = _attr_float(attrs, "injury_freq", "injuryfrequency", "injury_frequency")
        durability = _attr_float(attrs, "durability", "dur")
        reinjury = injury_state.get("reinjury_count") if isinstance(injury_state, dict) else {}
        reinjury_total = sum(int(v or 0) for v in (reinjury or {}).values()) if isinstance(reinjury, dict) else 0
        risk = _risk_tier_from_inputs(
            st_fatigue=st,
            lt_fatigue=lt,
            age=age,
            injury_freq=injury_freq,
            durability=durability,
            reinjury_total=reinjury_total,
        )
        if tier_order[risk["risk_tier"]] < tier_order[min_tier]:
            continue

        items.append(
            {
                "player_id": pid,
                "name": row.get("name"),
                "pos": row.get("pos"),
                "age": age,
                "injury_status": normalized_status,
                "injury_state": injury_state,
                "condition": {
                    "short_term_fatigue": st,
                    "long_term_fatigue": lt,
                    "short_term_stamina": max(0.0, 1.0 - st),
                    "long_term_stamina": max(0.0, 1.0 - lt),
                    "sharpness": sharpness,
                },
                "risk_inputs": {
                    "injury_freq": injury_freq,
                    "durability": durability,
                    "age": age,
                    "lt_wear_proxy": lt,
                    "energy_proxy": max(0.0, 1.0 - st),
                    "reinjury_count": reinjury if isinstance(reinjury, dict) else {},
                },
                "risk_score": int(risk["risk_score"]),
                "risk_tier": risk["risk_tier"],
            }
        )

    items.sort(key=lambda x: (int(x.get("risk_score") or 0), str(x.get("name") or "")), reverse=True)
    return {
        "team_id": tid,
        "season_year": sy,
        "as_of_date": as_of_date,
        "min_risk_tier": min_tier,
        "include_healthy_only": bool(include_healthy_only),
        "items": items,
    }


@router.get("/api/medical/team/{team_id}/injured")
async def api_medical_injured_players(
    team_id: str,
    season_year: Optional[int] = None,
    include_returning: bool = True,
    include_event_history: bool = True,
    history_days: int = 180,
):
    db_path = state.get_db_path()
    tid = str(normalize_team_id(team_id, strict=True))
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(season_year or (league_ctx.get("season_year") or 0))
    as_of = state.get_current_date_as_date()
    as_of_iso = as_of.isoformat()
    history_days = max(1, min(int(history_days or 180), 730))
    start_iso = (as_of - timedelta(days=history_days)).isoformat()
    end_iso = (as_of + timedelta(days=1)).isoformat()

    with LeagueRepo(db_path) as repo:
        repo.init_db()
        roster_rows = repo.get_team_roster(tid)
        pids = [str(r.get("player_id")) for r in (roster_rows or []) if r.get("player_id")]

        from injury import repo as injury_repo
        from injury.status import status_for_date

        with repo.transaction() as cur:
            injury_by_pid = injury_repo.get_player_injury_states(cur, pids) if pids else {}
            events = (
                injury_repo.get_overlapping_injury_events(
                    cur,
                    pids,
                    start_date=start_iso,
                    end_date=end_iso,
                )
                if (include_event_history and pids)
                else []
            )

    events_by_pid: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        pid = str(e.get("player_id") or "")
        if not pid:
            continue
        events_by_pid.setdefault(pid, []).append(e)

    items: List[Dict[str, Any]] = []
    for row in roster_rows:
        pid = str(row.get("player_id"))
        state_row = injury_by_pid.get(pid) or {}
        recovery_status = status_for_date(state_row, on_date_iso=as_of_iso)
        if recovery_status == "HEALTHY":
            continue
        if recovery_status == "RETURNING" and not include_returning:
            continue

        items.append(
            {
                "player_id": pid,
                "name": row.get("name"),
                "pos": row.get("pos"),
                "recovery_status": recovery_status,
                "is_injured": recovery_status in {"OUT", "RETURNING"},
                "injury_current": {
                    "injury_id": state_row.get("injury_id"),
                    "start_date": state_row.get("start_date"),
                    "body_part": state_row.get("body_part"),
                    "injury_type": state_row.get("injury_type"),
                    "severity": state_row.get("severity"),
                    "out_until_date": state_row.get("out_until_date"),
                    "returning_until_date": state_row.get("returning_until_date"),
                    "temp_debuff": state_row.get("temp_debuff") or {},
                    "perm_drop": state_row.get("perm_drop") or {},
                    "reinjury_count": state_row.get("reinjury_count") or {},
                    "last_processed_date": state_row.get("last_processed_date"),
                },
                "availability": {
                    "out_until_date": state_row.get("out_until_date"),
                    "returning_until_date": state_row.get("returning_until_date"),
                },
                "history": events_by_pid.get(pid, []) if include_event_history else None,
            }
        )

    items.sort(key=lambda x: (str(x.get("availability", {}).get("out_until_date") or ""), str(x.get("name") or "")))
    return {
        "team_id": tid,
        "season_year": sy,
        "as_of_date": as_of_iso,
        "include_returning": bool(include_returning),
        "include_event_history": bool(include_event_history),
        "history_days": history_days,
        "items": items,
    }













@router.get("/api/medical/team/{team_id}/overview")
async def api_medical_team_overview(
    team_id: str,
    season_year: Optional[int] = None,
    history_days: int = 180,
    top_n: int = 5,
):
    db_path = state.get_db_path()
    tid = str(normalize_team_id(team_id, strict=True))
    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(season_year or (league_ctx.get("season_year") or 0))
    as_of = state.get_current_date_as_date()
    as_of_iso = as_of.isoformat()
    history_days = max(1, min(int(history_days or 180), 730))
    top_n = max(1, min(int(top_n or 5), 20))
    health_high_threshold = 0.5

    from agency import repo as agency_repo
    from fatigue import repo as fatigue_repo
    from injury import repo as injury_repo
    from readiness import repo as readiness_repo
    with LeagueRepo(db_path) as repo:
        repo.init_db()
        roster_rows = repo.get_team_roster(tid)
        pids = [str(r.get("player_id")) for r in (roster_rows or []) if r.get("player_id")]

        with repo.transaction() as cur:
            fatigue_by_pid = fatigue_repo.get_player_fatigue_states(cur, pids) if pids else {}
            sharp_by_pid = readiness_repo.get_player_sharpness_states(cur, pids, season_year=sy) if (pids and sy > 0) else {}
            injury_by_pid = injury_repo.get_player_injury_states(cur, pids) if pids else {}
            agency_by_pid = agency_repo.get_player_agency_states(cur, pids) if pids else {}
            start_iso = (as_of - timedelta(days=history_days)).isoformat()
            end_iso = (as_of + timedelta(days=1)).isoformat()
            recent_events = injury_repo.get_overlapping_injury_events(cur, pids, start_date=start_iso, end_date=end_iso) if pids else []

    risk_items: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, Any]] = []
    health_rows: List[Dict[str, Any]] = []

    status_counts = {"OUT": 0, "RETURNING": 0, "HEALTHY": 0}
    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for row in roster_rows:
        pid = str(row.get("player_id"))
        prof = _build_risk_profile(
            row=row,
            injury_state=injury_by_pid.get(pid) or {},
            fatigue_state=fatigue_by_pid.get(pid) or {},
            sharpness_state=sharp_by_pid.get(pid) or {},
            as_of_date=as_of_iso,
        )
        risk_items.append(prof)
        status = str(prof.get("injury_status") or "HEALTHY")
        status_counts[status] = status_counts.get(status, 0) + 1
        tier = str(prof.get("risk_tier") or "LOW")
        risk_counts[tier] = risk_counts.get(tier, 0) + 1

        if status in {"OUT", "RETURNING"}:
            st = prof.get("injury_state") or {}
            unavailable.append({
                "player_id": pid,
                "name": row.get("name"),
                "pos": row.get("pos"),
                "recovery_status": status,
                "injury_current": {
                    "body_part": st.get("body_part"),
                    "injury_type": st.get("injury_type"),
                    "severity": st.get("severity"),
                    "out_until_date": st.get("out_until_date"),
                    "returning_until_date": st.get("returning_until_date"),
                },
            })

        agency_state = agency_by_pid.get(pid) or {}
        hf = _safe_float(agency_state.get("health_frustration"), 0.0)
        if agency_state:
            health_rows.append({
                "player_id": pid,
                "name": row.get("name"),
                "pos": row.get("pos"),
                "health_frustration": hf,
                "trade_request_level": _safe_int(agency_state.get("trade_request_level"), 0),
                "cooldown_health_until": agency_state.get("cooldown_health_until"),
                "escalation_health": _safe_int(agency_state.get("escalation_health"), 0),
            })

    risk_items_sorted = sorted(risk_items, key=lambda x: (int(x.get("risk_score") or 0), str(x.get("name") or "")), reverse=True)
    unavailable_sorted = sorted(unavailable, key=lambda x: (str((x.get("injury_current") or {}).get("out_until_date") or ""), str(x.get("name") or "")))
    health_sorted = sorted(health_rows, key=lambda x: (float(x.get("health_frustration") or 0.0), str(x.get("name") or "")), reverse=True)

    # attach player names to events
    name_by_pid = {str(r.get("player_id")): r.get("name") for r in roster_rows}
    recent_events_sorted = sorted(recent_events, key=lambda x: str(x.get("date") or ""), reverse=True)
    recent_event_items: List[Dict[str, Any]] = []
    for e in recent_events_sorted[:top_n]:
        pid = str(e.get("player_id") or "")
        recent_event_items.append({
            "injury_id": e.get("injury_id"),
            "player_id": pid,
            "name": name_by_pid.get(pid),
            "date": e.get("date"),
            "context": e.get("context"),
            "body_part": e.get("body_part"),
            "injury_type": e.get("injury_type"),
            "severity": e.get("severity"),
            "out_until_date": e.get("out_until_date"),
            "returning_until_date": e.get("returning_until_date"),
        })

    hf_values = [float(r.get("health_frustration") or 0.0) for r in health_rows]
    hf_count = len(hf_values)

    return {
        "team_id": tid,
        "season_year": sy,
        "as_of_date": as_of_iso,
        "history_days": history_days,
        "top_n": top_n,
        "summary": {
            "roster_count": len(roster_rows),
            "injury_status_counts": status_counts,
            "risk_tier_counts": risk_counts,
            "health_frustration": {
                "count_with_state": hf_count,
                "high_count": sum(1 for v in hf_values if v >= health_high_threshold),
                "max": max(hf_values) if hf_values else 0.0,
                "avg": (sum(hf_values) / hf_count) if hf_values else 0.0,
            },
        },
        "watchlists": {
            "highest_risk": risk_items_sorted[:top_n],
            "currently_unavailable": unavailable_sorted[:top_n],
            "health_frustration_high": [r for r in health_sorted if float(r.get("health_frustration") or 0.0) >= health_high_threshold][:top_n],
            "recent_injury_events": recent_event_items,
        },
    }


@router.get("/api/medical/team/{team_id}/players/{player_id}/timeline")
async def api_medical_player_timeline(
    team_id: str,
    player_id: str,
    season_year: Optional[int] = None,
    history_days: int = 365,
    include_event_history: bool = True,
):
    db_path = state.get_db_path()
    tid = str(normalize_team_id(team_id, strict=True))
    pid = str(normalize_player_id(player_id, strict=False, allow_legacy_numeric=True))
    if not pid:
        raise HTTPException(status_code=400, detail="Invalid player_id")

    league_ctx = state.get_league_context_snapshot() or {}
    sy = int(season_year or (league_ctx.get("season_year") or 0))
    as_of = state.get_current_date_as_date()
    as_of_iso = as_of.isoformat()
    history_days = max(1, min(int(history_days or 365), 730))

    from agency import repo as agency_repo
    from fatigue import repo as fatigue_repo
    from injury import repo as injury_repo
    from readiness import repo as readiness_repo
    from injury.status import status_for_date

    with LeagueRepo(db_path) as repo:
        repo.init_db()
        roster_rows = repo.get_team_roster(tid)
        row_by_pid = {str(r.get("player_id")): r for r in roster_rows}
        row = row_by_pid.get(pid)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Player '{pid}' is not on team '{tid}' active roster")

        with repo.transaction() as cur:
            fatigue_state = fatigue_repo.get_player_fatigue_states(cur, [pid]).get(pid)
            sharp_state = readiness_repo.get_player_sharpness_states(cur, [pid], season_year=sy).get(pid) if sy > 0 else None
            injury_state = injury_repo.get_player_injury_states(cur, [pid]).get(pid)
            agency_state = agency_repo.get_player_agency_states(cur, [pid]).get(pid)
            events = []
            if include_event_history:
                start_iso = (as_of - timedelta(days=history_days)).isoformat()
                end_iso = (as_of + timedelta(days=1)).isoformat()
                events = injury_repo.get_overlapping_injury_events(cur, [pid], start_date=start_iso, end_date=end_iso)

    prof = _build_risk_profile(
        row=row,
        injury_state=injury_state or {},
        fatigue_state=fatigue_state or {},
        sharpness_state=sharp_state or {},
        as_of_date=as_of_iso,
    )
    recovery_status = status_for_date(injury_state or {}, on_date_iso=as_of_iso)

    events_sorted = sorted(events, key=lambda x: str(x.get("date") or ""), reverse=True) if include_event_history else []

    return {
        "team_id": tid,
        "player_id": pid,
        "season_year": sy,
        "as_of_date": as_of_iso,
        "history_days": history_days,
        "include_event_history": bool(include_event_history),
        "player": {
            "name": row.get("name"),
            "pos": row.get("pos"),
            "age": int(row.get("age") or 0),
        },
        "status": {
            "recovery_status": recovery_status,
            "is_injured": recovery_status in {"OUT", "RETURNING"},
            "injury_status": prof.get("injury_status"),
        },
        "current": {
            "injury_state": prof.get("injury_state") or {},
            "availability": {
                "out_until_date": (prof.get("injury_state") or {}).get("out_until_date"),
                "returning_until_date": (prof.get("injury_state") or {}).get("returning_until_date"),
            },
            "condition": prof.get("condition") or {},
            "risk": {
                "risk_score": prof.get("risk_score"),
                "risk_tier": prof.get("risk_tier"),
                "risk_inputs": prof.get("risk_inputs") or {},
            },
            "health_psychology": {
                "health_frustration": _safe_float((agency_state or {}).get("health_frustration"), 0.0),
                "trade_request_level": _safe_int((agency_state or {}).get("trade_request_level"), 0),
                "cooldown_health_until": (agency_state or {}).get("cooldown_health_until"),
                "escalation_health": _safe_int((agency_state or {}).get("escalation_health"), 0),
            },
        },
        "timeline": {
            "events": events_sorted,
        },
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
    """마스터 스케줄 기준으로 특정 팀의 전체 시즌 일정을 UI 친화 포맷으로 반환."""
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

    workflow_state = state.export_workflow_state() or {}
    game_results = workflow_state.get("game_results") or {}
    current_date = str(league.get("current_date") or "")[:10]
    season_id = str(league.get("active_season_id") or "")

    formatted_games: List[Dict[str, Any]] = []
    wins = 0
    losses = 0
    for g in team_games:
        game_id = g.get("game_id")
        home_team_id = str(g.get("home_team_id") or "")
        away_team_id = str(g.get("away_team_id") or "")
        is_home = team_id == home_team_id
        opponent_team_id = away_team_id if is_home else home_team_id
        home_score = g.get("home_score")
        away_score = g.get("away_score")
        status = str(g.get("status") or "")
        is_completed = home_score is not None and away_score is not None
        result_for_team = None
        record_after_game: Optional[Dict[str, Any]] = None
        leaders: Optional[Dict[str, Any]] = None
        result: Optional[Dict[str, Any]] = None

        if is_completed:
            if is_home:
                result_for_team = "W" if home_score > away_score else "L"
            else:
                result_for_team = "W" if away_score > home_score else "L"

            if result_for_team == "W":
                wins += 1
            else:
                losses += 1

            score_for = int(home_score if is_home else away_score)
            score_against = int(away_score if is_home else home_score)
            result = {
                "wl": result_for_team,
                "score_for": score_for,
                "score_against": score_against,
                "display": f"{result_for_team} {score_for}-{score_against}",
            }
            record_after_game = {
                "wins": wins,
                "losses": losses,
                "display": f"{wins}-{losses}",
            }

            gr = game_results.get(str(game_id)) if isinstance(game_results, dict) else None
            team_box = ((gr or {}).get("teams") or {}).get(team_id) if isinstance(gr, dict) else None
            rows = team_box.get("players") if isinstance(team_box, dict) else []
            rows = rows if isinstance(rows, list) else []
            leaders = {
                "points": _pick_leader(rows, ["PTS"]),
                "rebounds": _pick_leader(rows, ["REB", "TRB"]),
                "assists": _pick_leader(rows, ["AST"]),
            }

        if not status:
            status = "final" if is_completed else "scheduled"

        formatted_games.append({
            "game_id": game_id,
            "date": g.get("date"),
            "date_mmdd": _format_mmdd(g.get("date")),
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "is_home": is_home,
            "opponent_team_id": opponent_team_id,
            "opponent_team_name": TEAM_FULL_NAMES.get(opponent_team_id, opponent_team_id),
            "opponent_label": f"{'vs' if is_home else '@'} {opponent_team_id}",
            "status": status,
            "is_completed": is_completed,
            "home_score": home_score,
            "away_score": away_score,
            "result_for_user_team": result_for_team,
            "result": result,
            "record_after_game": record_after_game,
            "leaders": leaders,
            "tipoff_time": None if is_completed else _deterministic_tipoff_time(game_id),
        })

    return {
        "team_id": team_id,
        "season_id": season_id,
        "current_date": current_date,
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
