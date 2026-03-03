from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

import state
from league_repo import LeagueRepo
from college.ui import (
    get_college_draft_pool,
    get_college_meta,
    get_college_player_detail,
    get_college_team_cards,
    get_college_team_detail,
    list_college_players,
)
from app.schemas.draft import DraftWatchRecomputeRequest

router = APIRouter()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _tier_from_pick(projected_pick: Optional[int]) -> str:
    if projected_pick is None or projected_pick <= 0:
        return "Unranked"
    if projected_pick <= 5:
        return "Tier 1"
    if projected_pick <= 14:
        return "Lottery"
    if projected_pick <= 30:
        return "Tier 2"
    return "Tier 3"


def _build_bigboard_signals(item: Dict[str, Any]) -> Dict[str, float]:
    stats = item.get("season_stats") if isinstance(item.get("season_stats"), dict) else {}

    pts = max(0.0, _safe_float(stats.get("pts"), 0.0))
    reb = max(0.0, _safe_float(stats.get("reb"), 0.0))
    ast = max(0.0, _safe_float(stats.get("ast"), 0.0))
    stl = max(0.0, _safe_float(stats.get("stl"), 0.0))
    blk = max(0.0, _safe_float(stats.get("blk"), 0.0))
    ts_pct = max(0.0, min(1.0, _safe_float(stats.get("ts_pct"), 0.0)))
    usg = max(0.0, _safe_float(stats.get("usg"), 0.0))
    class_year = _safe_int((item.get("college") or {}).get("class_year"), 1)

    production = min(1.0, (0.40 * (pts / 30.0)) + (0.20 * (reb / 12.0)) + (0.25 * (ast / 8.0)) + (0.15 * ((stl + blk) / 4.0)))
    efficiency = min(1.0, (0.75 * ts_pct) + (0.25 * min(1.0, usg / 35.0)))
    experience_risk = 1.0 if class_year <= 1 else (0.6 if class_year == 2 else 0.35)
    low_eff_risk = 1.0 - efficiency
    risk = min(1.0, (0.65 * experience_risk) + (0.35 * low_eff_risk))

    return {
        "production_score": round(production, 3),
        "efficiency_score": round(efficiency, 3),
        "risk_score": round(risk, 3),
    }


def _bigboard_summary(item: Dict[str, Any], signals: Dict[str, float]) -> str:
    stats = item.get("season_stats") if isinstance(item.get("season_stats"), dict) else {}
    strengths: List[str] = []
    if _safe_float(stats.get("pts"), 0.0) >= 18.0:
        strengths.append("Scoring")
    if _safe_float(stats.get("ast"), 0.0) >= 5.0:
        strengths.append("Playmaking")
    if _safe_float(stats.get("reb"), 0.0) >= 8.0:
        strengths.append("Rebounding")
    if _safe_float(stats.get("stl"), 0.0) + _safe_float(stats.get("blk"), 0.0) >= 2.2:
        strengths.append("Defensive Activity")
    if not strengths:
        strengths.append("Balanced Production")

    concerns: List[str] = []
    if signals.get("risk_score", 0.0) >= 0.75:
        concerns.append("Projection Volatility")
    if _safe_float(stats.get("ts_pct"), 0.0) < 0.54:
        concerns.append("Efficiency")
    if not concerns:
        concerns.append("Role Translation")

    return f"Strengths: {', '.join(strengths[:2])}. Concern: {concerns[0]}."



@router.get("/api/college/meta")
async def api_college_meta():
    try:
        return get_college_meta()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/college/teams")
async def api_college_teams(season_year: Optional[int] = None):
    try:
        return get_college_team_cards(season_year=season_year)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/college/team-detail/{college_team_id}")
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


@router.get("/api/college/players")
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


@router.get("/api/college/player/{player_id}")
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


@router.get("/api/college/draft-pool/{draft_year}")
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


@router.get("/api/college/dashboard-summary")
async def api_college_dashboard_summary(
    season_year: Optional[int] = None,
    draft_year: Optional[int] = None,
    team_id: Optional[str] = None,
):
    try:
        meta = get_college_meta()
        sy = int(season_year) if season_year is not None else int(meta.get("default_stats_season_year") or 0)
        if sy <= 0:
            raise ValueError("invalid season_year")

        dy = int(draft_year) if draft_year is not None else int(meta.get("upcoming_draft_year") or 0)
        if dy <= 0:
            raise ValueError("invalid draft_year")

        teams = get_college_team_cards(season_year=sy)
        team_valid = [t for t in teams if t.get("srs") is not None]
        top_team = None
        avg_srs = None
        if team_valid:
            team_valid.sort(key=lambda x: _safe_float(x.get("srs"), -999.0), reverse=True)
            t0 = team_valid[0]
            top_team = {
                "college_team_id": str(t0.get("college_team_id") or ""),
                "name": str(t0.get("name") or ""),
                "srs": round(_safe_float(t0.get("srs"), 0.0), 2),
            }
            avg_srs = round(sum(_safe_float(t.get("srs"), 0.0) for t in team_valid) / max(1, len(team_valid)), 2)

        declared_total = sum(_safe_int(t.get("declared_count"), 0) for t in teams)

        leaders: Dict[str, Dict[str, Any]] = {}
        for key in ("pts", "reb", "ast"):
            board = list_college_players(season_year=sy, sort=key, order="desc", limit=1, offset=0)
            rows = list(board.get("players") or [])
            if rows:
                p0 = rows[0]
                stat = p0.get("stats") if isinstance(p0.get("stats"), dict) else {}
                leaders[f"{key}_leader"] = {
                    "player_id": str(p0.get("player_id") or ""),
                    "name": str(p0.get("name") or ""),
                    "value": round(_safe_float(stat.get(key), 0.0), 2),
                }
            else:
                leaders[f"{key}_leader"] = {"player_id": "", "name": "", "value": 0.0}

        pool = get_college_draft_pool(dy, season_year=sy, pool_mode="auto")
        prospects = list(pool.get("prospects") or [])
        projected = []
        for p in prospects:
            if not isinstance(p, dict):
                continue
            consensus = p.get("consensus") if isinstance(p.get("consensus"), dict) else {}
            pick = consensus.get("projected_pick")
            if pick is None:
                continue
            try:
                projected.append(int(pick))
            except Exception:
                continue

        tier1_count = sum(1 for v in projected if v <= 5)
        avg_projected = round(sum(projected) / len(projected), 2) if projected else None

        scouting_block = None
        if team_id:
            db_path = state.get_db_path()
            with LeagueRepo(db_path) as repo:
                repo.init_db()
                scouts = repo._conn.execute(
                    """
                    SELECT COUNT(*) AS total,
                           SUM(CASE WHEN is_active=1 THEN 1 ELSE 0 END) AS active
                    FROM scouting_scouts
                    WHERE team_id=?;
                    """,
                    (str(team_id),),
                ).fetchone()
                active_assignments = repo._conn.execute(
                    """
                    SELECT COUNT(*) AS n
                    FROM scouting_assignments
                    WHERE team_id=? AND status='ACTIVE';
                    """,
                    (str(team_id),),
                ).fetchone()
                period_key = str(meta.get("current_date") or "")[:7]
                reports = repo._conn.execute(
                    """
                    SELECT COUNT(*) AS n
                    FROM scouting_reports
                    WHERE team_id=? AND period_key=?;
                    """,
                    (str(team_id), period_key),
                ).fetchone()

            total_scouts = _safe_int(scouts[0] if scouts else 0, 0)
            active_scouts = _safe_int(scouts[1] if scouts else 0, 0)
            scouting_block = {
                "team_id": str(team_id),
                "active_assignments": _safe_int(active_assignments[0] if active_assignments else 0, 0),
                "idle_scouts": max(0, active_scouts - _safe_int(active_assignments[0] if active_assignments else 0, 0)),
                "reports_this_period": _safe_int(reports[0] if reports else 0, 0),
                "total_scouts": int(total_scouts),
            }

        return {
            "ok": True,
            "season_year": int(sy),
            "draft_year": int(dy),
            "team_ranking": {
                "top_team": top_team,
                "avg_srs": avg_srs,
                "declared_total": int(declared_total),
            },
            "leaderboard": leaders,
            "bigboard": {
                "pool_mode_used": str(pool.get("pool_mode_used") or ""),
                "tier1_count": int(tier1_count),
                "lottery_cut_pick": 14,
                "avg_projected_pick": avg_projected,
            },
            "scouting": scouting_block,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build dashboard summary: {e}") from e


@router.get("/api/college/bigboard")
async def api_college_bigboard(
    draft_year: int,
    season_year: Optional[int] = None,
    expert_profile: str = "consensus",
    limit: int = 100,
    offset: int = 0,
):
    try:
        dy = int(draft_year)
        if dy <= 0:
            raise ValueError("draft_year must be > 0")

        sy = int(season_year) if season_year is not None else (dy - 1)
        if sy <= 0:
            raise ValueError("season_year must be > 0")

        lim = max(1, min(500, int(limit)))
        off = max(0, int(offset))

        profile = str(expert_profile or "consensus").strip().lower()
        if profile not in {"consensus", "efficiency", "upside", "defense"}:
            raise ValueError("expert_profile must be one of: consensus, efficiency, upside, defense")

        pool = get_college_draft_pool(dy, season_year=sy, pool_mode="auto")
        prospects = [p for p in list(pool.get("prospects") or []) if isinstance(p, dict)]

        records: List[Dict[str, Any]] = []
        for p in prospects:
            signals = _build_bigboard_signals(p)
            consensus = p.get("consensus") if isinstance(p.get("consensus"), dict) else {}
            projected_pick = consensus.get("projected_pick")
            projected_pick_i = _safe_int(projected_pick, 999) if projected_pick is not None else 999

            if profile == "consensus":
                rank_score = float(projected_pick_i)
            elif profile == "efficiency":
                rank_score = -signals["efficiency_score"]
            elif profile == "upside":
                rank_score = -(0.65 * signals["production_score"] + 0.35 * (1.0 - signals["risk_score"]))
            else:  # defense
                stats = p.get("season_stats") if isinstance(p.get("season_stats"), dict) else {}
                rank_score = -(_safe_float(stats.get("stl"), 0.0) + _safe_float(stats.get("blk"), 0.0))

            records.append(
                {
                    "temp_id": str(p.get("temp_id") or ""),
                    "name": str(p.get("name") or ""),
                    "pos": str(p.get("pos") or ""),
                    "age": _safe_int(p.get("age"), 0),
                    "college": p.get("college") if isinstance(p.get("college"), dict) else {},
                    "consensus_projected_pick": (None if projected_pick is None else _safe_int(projected_pick, 0)),
                    "tier": _tier_from_pick((None if projected_pick is None else _safe_int(projected_pick, 0))),
                    "summary": _bigboard_summary(p, signals),
                    "signals": signals,
                    "_rank_score": rank_score,
                }
            )

        records.sort(key=lambda x: (x.get("_rank_score", 999999.0), x.get("name", "")))
        total = len(records)
        page = records[off : off + lim]

        items: List[Dict[str, Any]] = []
        for idx, r in enumerate(page, start=off + 1):
            college = r.get("college") if isinstance(r.get("college"), dict) else {}
            items.append(
                {
                    "rank": int(idx),
                    "temp_id": str(r.get("temp_id") or ""),
                    "player": {
                        "name": str(r.get("name") or ""),
                        "pos": str(r.get("pos") or ""),
                        "age": _safe_int(r.get("age"), 0),
                        "college_team_id": str(college.get("college_team_id") or ""),
                        "college_team_name": str(college.get("college_team_name") or ""),
                        "class_year": _safe_int(college.get("class_year"), 0),
                    },
                    "consensus_projected_pick": r.get("consensus_projected_pick"),
                    "tier": str(r.get("tier") or "Unranked"),
                    "summary": str(r.get("summary") or ""),
                    "signals": dict(r.get("signals") or {}),
                }
            )

        return {
            "ok": True,
            "draft_year": int(dy),
            "season_year": int(sy),
            "expert_profile": profile,
            "pool_mode_used": str(pool.get("pool_mode_used") or ""),
            "count": int(total),
            "limit": int(lim),
            "offset": int(off),
            "items": items,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build bigboard: {e}") from e


@router.post("/api/college/draft-watch/recompute")
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
