from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import state
from league_repo import LeagueRepo
from schema import normalize_team_id

router = APIRouter()


TACTICS_OFFENSE_SCHEMES: List[Dict[str, str]] = [
    {"key": "Spread_HeavyPnR", "label": "헤비 PnR", "description": "볼핸들러 중심의 2대2 전개를 강화합니다."},
    {"key": "Drive_Kick", "label": "드라이브&킥", "description": "돌파 후 외곽 분배 비중을 높입니다."},
    {"key": "FiveOut", "label": "5-Out", "description": "모든 포지션을 외곽에 배치해 공간을 극대화합니다."},
    {"key": "Motion_SplitCut", "label": "모션 스플릿", "description": "오프볼 무브와 컷을 활용한 연계 플레이를 늘립니다."},
    {"key": "DHO_Chicago", "label": "DHO 시카고", "description": "핸드오프와 핀다운 연계를 자주 사용합니다."},
    {"key": "Post_InsideOut", "label": "인사이드-아웃", "description": "포스트 터치 기반으로 외곽 찬스를 만듭니다."},
    {"key": "Horns_Elbow", "label": "혼즈 엘보", "description": "엘보 지역 허브 플레이로 옵션을 다양화합니다."},
    {"key": "Transition_Early", "label": "얼리 트랜지션", "description": "초반 공격 전개 속도를 높여 속공을 노립니다."},
]

TACTICS_DEFENSE_SCHEMES: List[Dict[str, str]] = [
    {"key": "Drop", "label": "드롭", "description": "빅맨이 림 근처를 지키며 2대2를 수비합니다."},
    {"key": "Switch_Everything", "label": "올 스위치", "description": "대부분의 스크린 상황에서 스위치를 선택합니다."},
    {"key": "Switch_1_4", "label": "1-4 스위치", "description": "가드~포워드 구간 위주로 스위치를 적용합니다."},
    {"key": "Hedge_ShowRecover", "label": "헤지-리커버", "description": "빅맨이 일시적으로 볼 압박 후 복귀합니다."},
    {"key": "Blitz_TrapPnR", "label": "블리츠", "description": "볼핸들러에게 적극적인 더블팀을 시도합니다."},
    {"key": "AtTheLevel", "label": "앳더레벨", "description": "스크린 레벨에서 볼 압박 강도를 높입니다."},
    {"key": "Zone", "label": "존", "description": "지역 수비로 페인트 보호와 로테이션을 운영합니다."},
]

TACTICS_OFFENSE_ROLES: List[Dict[str, str]] = [
    {"key": "Engine_Primary", "label": "1차 볼핸들러", "description": "주요 공격 시작점"},
    {"key": "Engine_Secondary", "label": "2차 볼핸들러", "description": "보조 볼운반 및 창출"},
    {"key": "Transition_Engine", "label": "트랜지션 엔진", "description": "속공 전개 주도"},
    {"key": "Shot_Creator", "label": "샷 크리에이터", "description": "자체 득점 창출"},
    {"key": "Rim_Pressure", "label": "림 압박", "description": "페인트 침투 중심"},
    {"key": "SpotUp_Spacer", "label": "스팟업 슈터", "description": "코너/윙 공간 유지"},
    {"key": "Movement_Shooter", "label": "무브먼트 슈터", "description": "오프스크린 캐치앤슈트"},
    {"key": "Cutter_Finisher", "label": "커터", "description": "백도어/컷 마무리"},
    {"key": "Connector", "label": "커넥터", "description": "볼 흐름 연결"},
    {"key": "Roll_Man", "label": "롤맨", "description": "스크린 후 림 다이브"},
    {"key": "ShortRoll_Hub", "label": "쇼트롤 허브", "description": "쇼트롤 의사결정"},
    {"key": "Pop_Threat", "label": "팝 위협", "description": "스크린 후 외곽 벌림"},
    {"key": "Post_Anchor", "label": "포스트 앵커", "description": "저지점 허브"},
]

TACTICS_DEFENSE_ROLE_BY_SCHEME: Dict[str, List[Dict[str, str]]] = {
    "Drop": [
        {"key": "PnR_POA_Defender", "label": "POA 수비", "description": "볼핸들러 1선 압박"},
        {"key": "PnR_Cover_Big_Drop", "label": "드롭 빅", "description": "드롭 커버 핵심"},
        {"key": "Lowman_Helper", "label": "로우맨", "description": "헬프/림 커버"},
        {"key": "Nail_Helper", "label": "네일 헬퍼", "description": "중앙 도움수비"},
        {"key": "Weakside_Rotator", "label": "약측 로테이터", "description": "약측 로테이션"},
    ],
    "Switch_Everything": [
        {"key": "PnR_POA_Switch", "label": "POA 스위치", "description": "1선 스위치"},
        {"key": "PnR_Cover_Big_Switch", "label": "빅 스위치", "description": "빅맨 스위치 대응"},
        {"key": "Switch_Wing_Strong", "label": "강측 윙", "description": "강측 미스매치 대응"},
        {"key": "Switch_Wing_Weak", "label": "약측 윙", "description": "약측 로테이션"},
        {"key": "Backline_Anchor", "label": "백라인 앵커", "description": "최종 림 보호"},
    ],
    "Switch_1_4": [
        {"key": "PnR_POA_Switch_1_4", "label": "POA 1-4 스위치", "description": "가드/윙 스위치"},
        {"key": "PnR_Cover_Big_Switch_1_4", "label": "빅 1-4 커버", "description": "빅맨 리커버"},
        {"key": "Switch_Wing_Strong_1_4", "label": "강측 윙 1-4", "description": "강측 보조"},
        {"key": "Switch_Wing_Weak_1_4", "label": "약측 윙 1-4", "description": "약측 보조"},
        {"key": "Backline_Anchor", "label": "백라인 앵커", "description": "최종 림 보호"},
    ],
    "Hedge_ShowRecover": [
        {"key": "PnR_POA_Defender", "label": "POA 수비", "description": "볼핸들러 1선 압박"},
        {"key": "PnR_Cover_Big_HedgeRecover", "label": "헤지 빅", "description": "쇼 후 복귀"},
        {"key": "Lowman_Helper", "label": "로우맨", "description": "헬프/림 커버"},
        {"key": "Nail_Helper", "label": "네일 헬퍼", "description": "중앙 도움수비"},
        {"key": "Weakside_Rotator", "label": "약측 로테이터", "description": "약측 로테이션"},
    ],
    "Blitz_TrapPnR": [
        {"key": "PnR_POA_Blitz", "label": "POA 블리츠", "description": "강한 트랩 유도"},
        {"key": "PnR_Cover_Big_Blitz", "label": "빅 블리츠", "description": "더블팀 합류"},
        {"key": "Lowman_Helper", "label": "로우맨", "description": "헬프/림 커버"},
        {"key": "Nail_Helper", "label": "네일 헬퍼", "description": "중앙 도움수비"},
        {"key": "Weakside_Rotator", "label": "약측 로테이터", "description": "약측 로테이션"},
    ],
    "AtTheLevel": [
        {"key": "PnR_POA_AtTheLevel", "label": "POA 앳더레벨", "description": "스크린 레벨 압박"},
        {"key": "PnR_Cover_Big_AtTheLevel", "label": "빅 앳더레벨", "description": "빅 압박 후 복귀"},
        {"key": "Lowman_Helper", "label": "로우맨", "description": "헬프/림 커버"},
        {"key": "Nail_Helper", "label": "네일 헬퍼", "description": "중앙 도움수비"},
        {"key": "Weakside_Rotator", "label": "약측 로테이터", "description": "약측 로테이션"},
    ],
    "Zone": [
        {"key": "Zone_Top_Left", "label": "존 탑 레프트", "description": "상단 왼쪽"},
        {"key": "Zone_Top_Right", "label": "존 탑 라이트", "description": "상단 오른쪽"},
        {"key": "Zone_Bottom_Left", "label": "존 바텀 레프트", "description": "하단 왼쪽"},
        {"key": "Zone_Bottom_Right", "label": "존 바텀 라이트", "description": "하단 오른쪽"},
        {"key": "Zone_Bottom_Center", "label": "존 바텀 센터", "description": "하단 중앙"},
    ],
}


class TacticsLineupItem(BaseModel):
    slot: int
    player_id: str
    offense_role_key: str
    defense_role_key: str
    target_minutes: int = Field(ge=0, le=48)


class TacticsConstraints(BaseModel):
    minutes_total_target: int = 240
    max_minutes_per_player: int = 48
    allow_duplicate_defense_role: bool = False


class TacticsPlanModel(BaseModel):
    plan_id: str = "default"
    plan_name: str = "정규 시즌 기본"
    is_active: bool = True
    offense_scheme_key: str
    defense_scheme_key: str
    starters: List[TacticsLineupItem]
    rotation: List[TacticsLineupItem]
    constraints: TacticsConstraints = Field(default_factory=TacticsConstraints)


class ImpactPreviewRequest(BaseModel):
    plan: TacticsPlanModel
    context: Optional[Dict[str, Any]] = None


def _league_season_year() -> int:
    league_ctx = state.get_league_context_snapshot() or {}
    return int(league_ctx.get("season_year") or 0)


def _ensure_table(cur: Any) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tactics_plans (
            team_id TEXT NOT NULL,
            season_year INTEGER NOT NULL,
            plan_id TEXT NOT NULL,
            plan_name TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (team_id, season_year, plan_id)
        );
        """
    )


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _default_plan_from_roster(roster_rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    names = [str(r.get("player_id") or "") for r in roster_rows if r.get("player_id")]
    default_def_roles = [x["key"] for x in TACTICS_DEFENSE_ROLE_BY_SCHEME["Drop"]]
    off_roles = [x["key"] for x in TACTICS_OFFENSE_ROLES]

    starters: List[Dict[str, Any]] = []
    rotation: List[Dict[str, Any]] = []
    for i in range(5):
        pid = names[i] if i < len(names) else ""
        starters.append(
            {
                "slot": i + 1,
                "player_id": pid,
                "offense_role_key": off_roles[i % len(off_roles)],
                "defense_role_key": default_def_roles[i % len(default_def_roles)],
                "target_minutes": 32 - i,
            }
        )
    for i in range(5):
        idx = i + 5
        pid = names[idx] if idx < len(names) else ""
        rotation.append(
            {
                "slot": idx + 1,
                "player_id": pid,
                "offense_role_key": off_roles[idx % len(off_roles)],
                "defense_role_key": default_def_roles[i % len(default_def_roles)],
                "target_minutes": 18 - i,
            }
        )

    return {
        "plan_id": "default",
        "plan_name": "정규 시즌 기본",
        "is_active": True,
        "offense_scheme_key": "Spread_HeavyPnR",
        "defense_scheme_key": "Drop",
        "starters": starters,
        "rotation": rotation,
        "constraints": {
            "minutes_total_target": 240,
            "max_minutes_per_player": 48,
            "allow_duplicate_defense_role": False,
        },
    }


def _plan_from_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    payload_raw = row.get("payload_json")
    payload: Dict[str, Any]
    try:
        payload = json.loads(str(payload_raw or "{}"))
    except Exception:
        payload = {}
    payload.setdefault("plan_id", str(row.get("plan_id") or "default"))
    payload.setdefault("plan_name", str(row.get("plan_name") or "정규 시즌 기본"))
    payload.setdefault("is_active", bool(int(row.get("is_active") or 0)))
    payload["updated_at"] = str(row.get("updated_at") or "")
    return payload


def _lineup_items(plan: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    starters = plan.get("starters") if isinstance(plan.get("starters"), list) else []
    rotation = plan.get("rotation") if isinstance(plan.get("rotation"), list) else []
    return list(starters) + list(rotation)


def _validate_plan_structure(plan: Mapping[str, Any]) -> Dict[str, Any]:
    offense_keys = {x["key"] for x in TACTICS_OFFENSE_SCHEMES}
    defense_keys = {x["key"] for x in TACTICS_DEFENSE_SCHEMES}
    offense_role_keys = {x["key"] for x in TACTICS_OFFENSE_ROLES}

    warnings: List[str] = []
    offense_scheme = str(plan.get("offense_scheme_key") or "")
    defense_scheme = str(plan.get("defense_scheme_key") or "")
    if offense_scheme not in offense_keys:
        raise HTTPException(status_code=400, detail=f"invalid offense_scheme_key: {offense_scheme}")
    if defense_scheme not in defense_keys:
        raise HTTPException(status_code=400, detail=f"invalid defense_scheme_key: {defense_scheme}")

    defense_roles_allowed = {x["key"] for x in TACTICS_DEFENSE_ROLE_BY_SCHEME.get(defense_scheme, [])}
    items = _lineup_items(plan)
    if len(items) != 10:
        raise HTTPException(status_code=400, detail="plan must contain 5 starters and 5 rotation players")

    minutes_total = 0
    seen_player_ids: Dict[str, int] = {}
    seen_def_roles: Dict[str, int] = {}

    for item in items:
        pid = str(item.get("player_id") or "").strip()
        off_role = str(item.get("offense_role_key") or "")
        def_role = str(item.get("defense_role_key") or "")
        minutes = _safe_int(item.get("target_minutes"), -1)
        if off_role not in offense_role_keys:
            raise HTTPException(status_code=400, detail=f"invalid offense_role_key: {off_role}")
        if def_role not in defense_roles_allowed:
            raise HTTPException(status_code=400, detail=f"invalid defense_role_key for scheme {defense_scheme}: {def_role}")
        if minutes < 0 or minutes > 48:
            raise HTTPException(status_code=400, detail="target_minutes must be between 0 and 48")
        minutes_total += minutes
        if pid:
            seen_player_ids[pid] = seen_player_ids.get(pid, 0) + 1
        seen_def_roles[def_role] = seen_def_roles.get(def_role, 0) + 1

    duplicate_players = sorted([pid for pid, cnt in seen_player_ids.items() if cnt > 1])
    duplicate_def_roles = sorted([role for role, cnt in seen_def_roles.items() if cnt > 1])
    if duplicate_players:
        warnings.append(f"duplicate players: {', '.join(duplicate_players)}")

    allow_dup_def = bool((plan.get("constraints") or {}).get("allow_duplicate_defense_role", False))
    has_dup_def_roles = len(duplicate_def_roles) > 0
    if has_dup_def_roles and not allow_dup_def:
        warnings.append("duplicate defense roles detected while allow_duplicate_defense_role=false")

    return {
        "minutes_total": minutes_total,
        "has_duplicate_player": len(duplicate_players) > 0,
        "has_invalid_role": False,
        "has_duplicate_defense_role": has_dup_def_roles,
        "warnings": warnings,
    }


def _fetch_or_default_plan(team_id: str, plan_id: str, season_year: int) -> Dict[str, Any]:
    db_path = state.get_db_path()
    with LeagueRepo(db_path) as repo:
        repo.init_db()
        roster_rows = repo.get_team_roster(team_id)
        with repo.transaction() as cur:
            _ensure_table(cur)
            row = cur.execute(
                """
                SELECT team_id, season_year, plan_id, plan_name, is_active, payload_json, created_at, updated_at
                FROM tactics_plans
                WHERE team_id=? AND season_year=? AND plan_id=?
                LIMIT 1
                """,
                (team_id, season_year, plan_id),
            ).fetchone()

    if row:
        return _plan_from_row(dict(row))

    default_plan = _default_plan_from_roster(roster_rows)
    default_plan["updated_at"] = ""
    return default_plan


def _player_strength(plan: Mapping[str, Any], roster_map: Mapping[str, Mapping[str, Any]]) -> float:
    items = _lineup_items(plan)
    if not items:
        return 0.0
    minute_sum = 0.0
    weighted = 0.0
    for item in items:
        pid = str(item.get("player_id") or "")
        minutes = max(0, _safe_int(item.get("target_minutes"), 0))
        p = roster_map.get(pid) or {}
        ovr = _safe_float(p.get("ovr"), 60.0)
        st_stamina = max(0.0, min(1.0, _safe_float(p.get("short_term_stamina"), 1.0)))
        lt_stamina = max(0.0, min(1.0, _safe_float(p.get("long_term_stamina"), 1.0)))
        sharp = _safe_float(p.get("sharpness"), 50.0)
        player_score = (ovr * 0.65 + sharp * 0.35) * (0.55 + 0.45 * st_stamina) * (0.55 + 0.45 * lt_stamina)
        weighted += player_score * float(minutes)
        minute_sum += float(minutes)
    if minute_sum <= 0:
        return 0.0
    return weighted / minute_sum


def _estimate_metrics(plan: Mapping[str, Any], roster_rows: List[Mapping[str, Any]], opp_rows: Optional[List[Mapping[str, Any]]] = None) -> Dict[str, float]:
    roster_map = {str(r.get("player_id") or ""): r for r in roster_rows}
    team_strength = _player_strength(plan, roster_map)

    opp_strength = 75.0
    if opp_rows:
        opp_roster_map = {str(r.get("player_id") or ""): r for r in opp_rows}
        opp_default_plan = _default_plan_from_roster(opp_rows)
        opp_strength = _player_strength(opp_default_plan, opp_roster_map)

    offense_bonus = {
        "Spread_HeavyPnR": 1.8,
        "Drive_Kick": 1.5,
        "FiveOut": 1.7,
        "Motion_SplitCut": 1.2,
        "DHO_Chicago": 1.1,
        "Post_InsideOut": 0.9,
        "Horns_Elbow": 0.8,
        "Transition_Early": 1.6,
    }
    defense_bonus = {
        "Drop": 1.4,
        "Switch_Everything": 1.2,
        "Switch_1_4": 1.0,
        "Hedge_ShowRecover": 0.9,
        "Blitz_TrapPnR": 0.7,
        "AtTheLevel": 0.8,
        "Zone": 0.6,
    }
    pace_bonus = {
        "Transition_Early": 2.2,
        "FiveOut": 0.9,
        "Drive_Kick": 0.8,
        "Spread_HeavyPnR": 0.5,
    }

    off_scheme = str(plan.get("offense_scheme_key") or "Spread_HeavyPnR")
    def_scheme = str(plan.get("defense_scheme_key") or "Drop")

    offense_rating = 88.0 + team_strength * 0.36 + offense_bonus.get(off_scheme, 0.0) - max(0.0, (opp_strength - 75.0) * 0.06)
    defense_rating = 116.0 - team_strength * 0.22 - defense_bonus.get(def_scheme, 0.0) + max(0.0, (opp_strength - 75.0) * 0.04)
    pace = 94.0 + pace_bonus.get(off_scheme, 0.0)

    return {
        "offense_rating": round(offense_rating, 1),
        "defense_rating": round(defense_rating, 1),
        "pace": round(pace, 1),
    }


def _fit_warnings(plan: Mapping[str, Any]) -> List[Dict[str, str]]:
    items = _lineup_items(plan)
    minutes_total = sum(max(0, _safe_int(x.get("target_minutes"), 0)) for x in items)
    warnings: List[Dict[str, str]] = []

    engine_roles = {"Engine_Primary", "Engine_Secondary", "Shot_Creator"}
    engine_count = sum(1 for x in items if str(x.get("offense_role_key") or "") in engine_roles)
    if engine_count >= 6:
        warnings.append({"code": "TOO_MANY_CREATION_ROLES", "message": "볼 창출 역할이 과다해 오프볼 밸런스가 무너질 수 있습니다."})

    def_roles = {str(x.get("defense_role_key") or "") for x in items}
    rim_keywords = ("Lowman", "Backline", "Bottom_Center", "Big")
    if not any(any(k in role for k in rim_keywords) for role in def_roles):
        warnings.append({"code": "LOW_RIM_PROTECTION", "message": "림 보호 역할이 부족합니다."})

    if minutes_total != 240:
        warnings.append({"code": "MINUTES_NOT_240", "message": f"총 출전시간이 240분이 아닙니다. 현재 {minutes_total}분"})

    return warnings


def _metric_delta(before: float, after: float) -> Dict[str, float]:
    return {
        "before": round(before, 1),
        "after": round(after, 1),
        "delta": round(after - before, 1),
    }


def _load_team_roster(team_id: str) -> List[Mapping[str, Any]]:
    db_path = state.get_db_path()
    with LeagueRepo(db_path) as repo:
        repo.init_db()
        return repo.get_team_roster(team_id)


@router.get("/api/tactics/meta")
async def api_tactics_meta() -> Dict[str, Any]:
    return {
        "offense_schemes": TACTICS_OFFENSE_SCHEMES,
        "defense_schemes": TACTICS_DEFENSE_SCHEMES,
        "offense_roles": TACTICS_OFFENSE_ROLES,
        "defense_roles_by_scheme": TACTICS_DEFENSE_ROLE_BY_SCHEME,
    }


@router.get("/api/tactics/team/{team_id}")
async def api_get_tactics_plan(team_id: str, plan_id: str = "default") -> Dict[str, Any]:
    tid = str(normalize_team_id(team_id, strict=True))
    season_year = _league_season_year()
    if season_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    plan = _fetch_or_default_plan(tid, str(plan_id or "default"), season_year)
    return {
        "team_id": tid,
        "season_year": season_year,
        **plan,
    }


@router.put("/api/tactics/team/{team_id}")
async def api_put_tactics_plan(team_id: str, payload: TacticsPlanModel) -> Dict[str, Any]:
    tid = str(normalize_team_id(team_id, strict=True))
    season_year = _league_season_year()
    if season_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    db_path = state.get_db_path()
    with LeagueRepo(db_path) as repo:
        repo.init_db()
        roster_rows = repo.get_team_roster(tid)
        roster_pids = {str(r.get("player_id") or "") for r in roster_rows if r.get("player_id")}

    plan_dict = payload.model_dump()
    validation = _validate_plan_structure(plan_dict)

    for item in _lineup_items(plan_dict):
        pid = str(item.get("player_id") or "")
        if pid and pid not in roster_pids:
            raise HTTPException(status_code=400, detail=f"player_id not in team roster: {pid}")

    now_iso = _now_iso()
    with LeagueRepo(db_path) as repo:
        repo.init_db()
        with repo.transaction() as cur:
            _ensure_table(cur)
            cur.execute(
                """
                INSERT INTO tactics_plans(team_id, season_year, plan_id, plan_name, is_active, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team_id, season_year, plan_id) DO UPDATE SET
                    plan_name=excluded.plan_name,
                    is_active=excluded.is_active,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    tid,
                    season_year,
                    payload.plan_id,
                    payload.plan_name,
                    1 if payload.is_active else 0,
                    json.dumps(plan_dict, ensure_ascii=False),
                    now_iso,
                    now_iso,
                ),
            )
            if payload.is_active:
                cur.execute(
                    """
                    UPDATE tactics_plans
                    SET is_active=0, updated_at=?
                    WHERE team_id=? AND season_year=? AND plan_id<>?
                    """,
                    (now_iso, tid, season_year, payload.plan_id),
                )

    return {
        "ok": True,
        "team_id": tid,
        "season_year": season_year,
        "plan_id": payload.plan_id,
        "saved_at": now_iso,
        "validation": validation,
    }


@router.post("/api/tactics/team/{team_id}/impact-preview")
async def api_tactics_impact_preview(team_id: str, req: ImpactPreviewRequest) -> Dict[str, Any]:
    tid = str(normalize_team_id(team_id, strict=True))
    season_year = _league_season_year()
    if season_year <= 0:
        raise HTTPException(status_code=500, detail="Invalid season_year in state.")

    after_plan = req.plan.model_dump()
    after_validation = _validate_plan_structure(after_plan)
    before_plan = _fetch_or_default_plan(tid, str(after_plan.get("plan_id") or "default"), season_year)

    team_roster = _load_team_roster(tid)

    context = req.context or {}
    opponent_team_id = context.get("opponent_team_id")
    opp_roster: Optional[List[Mapping[str, Any]]] = None
    if opponent_team_id:
        opp_tid = str(normalize_team_id(str(opponent_team_id), strict=True))
        opponent_team_id = opp_tid
        opp_roster = _load_team_roster(opp_tid)

    before_metrics = _estimate_metrics(before_plan, team_roster, opp_roster)
    after_metrics = _estimate_metrics(after_plan, team_roster, opp_roster)

    minutes_target = _safe_int((after_plan.get("constraints") or {}).get("minutes_total_target"), 240)
    minutes_current = _safe_int(after_validation.get("minutes_total"), 0)

    return {
        "team_id": tid,
        "season_year": season_year,
        "opponent_team_id": opponent_team_id,
        "metrics": {
            "offense_rating": _metric_delta(before_metrics["offense_rating"], after_metrics["offense_rating"]),
            "defense_rating": _metric_delta(before_metrics["defense_rating"], after_metrics["defense_rating"]),
            "pace": _metric_delta(before_metrics["pace"], after_metrics["pace"]),
        },
        "fit_warnings": _fit_warnings(after_plan),
        "minutes_check": {
            "target": minutes_target,
            "current_total": minutes_current,
            "delta": minutes_current - minutes_target,
        },
    }
