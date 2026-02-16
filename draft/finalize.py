from __future__ import annotations

"""Draft finalization (DB-integrated).

This module bridges:
  - (pure) standings/lottery/order computations
  - (DB) pick settlement (protections + swap rights) via LeagueService.settle_draft_year
  - (DB) resolving drafting team (pick owner_team after settlement)
  - turning the plan into a turn list (DraftTurn)

Output is still primarily *ephemeral* (in-memory): turns, plan, and settlement events can be
stored in UI/cache if desired. For stepwise offseason flows (lottery -> settlement -> draft),
the order plan can also be persisted (draft_order_plans) and reloaded.
"""

import json
import game_time
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from config import ALL_TEAM_IDS

from .types import DraftOrderPlan, DraftTurn, LotteryResult, TeamId, TeamRecord, make_pick_id, norm_team_id
from .standings import compute_team_records_from_master_schedule
from .order import compute_draft_order_plan_from_records


def is_order_plan_settled(plan: DraftOrderPlan) -> bool:
    meta = plan.meta if isinstance(getattr(plan, "meta", None), dict) else {}
    # Strict: only True counts as settled
    return meta.get("settled") is True


def mark_order_plan_settled(
    plan: DraftOrderPlan,
    *,
    settled: bool,
    settled_at_utc: Optional[str] = None,
) -> DraftOrderPlan:
    meta = dict(plan.meta) if isinstance(getattr(plan, "meta", None), dict) else {}
    meta["settled"] = bool(settled)
    if settled_at_utc is not None:
        meta["settled_at_utc"] = str(settled_at_utc)
    return replace(plan, meta=meta)


def require_order_plan_settled(plan: DraftOrderPlan, *, where: str = "") -> None:
    if is_order_plan_settled(plan):
        return
    suffix = f" ({where})" if where else ""
    raise ValueError(
        f"Draft picks are not settled for draft_year={int(plan.draft_year)}. "
        f"Call /api/offseason/draft/settle before using bundle/selections/apply.{suffix}"
    )


def build_turns_from_plan(
    *,
    db_path: str,
    plan: DraftOrderPlan,
    turn_attr_settled: Optional[bool] = None,
) -> List[DraftTurn]:
    """Build the 60 DraftTurn list (read-only).

    Reads draft_picks.owner_team for the given draft_year and maps each plan slot
    to the current drafting team. Does NOT perform settlement or any other writes.
    """
    draft_year = int(plan.draft_year)

    from league_repo import LeagueRepo

    with LeagueRepo(str(db_path)) as repo:
        repo.init_db()
        picks_map = repo.get_draft_picks_map()

    turns: List[DraftTurn] = []
    overall = 0
    base_attrs: Dict[str, Any] = {}
    if turn_attr_settled is not None:
        base_attrs["settled"] = bool(turn_attr_settled)

    # Round 1
    for slot, original_team in enumerate(plan.round1_slot_to_original_team, start=1):
        overall += 1
        pick_id = make_pick_id(draft_year, 1, original_team)
        pick = picks_map.get(pick_id)
        if not isinstance(pick, dict):
            raise ValueError(
                f"draft_picks row missing for pick_id={pick_id!r} (draft_year={draft_year}). "
                "Run /api/offseason/draft/settle first."
            )
        owner = pick.get("owner_team")
        owner_norm = norm_team_id(owner)
        if not owner_norm or owner_norm in ("NONE", "NULL"):
            raise ValueError(
                f"draft_picks.owner_team invalid for pick_id={pick_id!r} (draft_year={draft_year}). "
                "Run /api/offseason/draft/settle first."
            )
        drafting_team = owner_norm
        turns.append(
            DraftTurn(
                round=1,
                slot=int(slot),
                overall_no=int(overall),
                pick_id=pick_id,
                original_team=original_team,
                drafting_team=drafting_team,
                attrs=dict(base_attrs),
            )
        )

    # Round 2
    for slot, original_team in enumerate(plan.round2_slot_to_original_team, start=1):
        overall += 1
        pick_id = make_pick_id(draft_year, 2, original_team)
        pick = picks_map.get(pick_id)
        if not isinstance(pick, dict):
            raise ValueError(
                f"draft_picks row missing for pick_id={pick_id!r} (draft_year={draft_year}). "
                "Run /api/offseason/draft/settle first."
            )
        owner = pick.get("owner_team")
        owner_norm = norm_team_id(owner)
        if not owner_norm or owner_norm in ("NONE", "NULL"):
            raise ValueError(
                f"draft_picks.owner_team invalid for pick_id={pick_id!r} (draft_year={draft_year}). "
                "Run /api/offseason/draft/settle first."
            )
        drafting_team = owner_norm
        turns.append(
            DraftTurn(
                round=2,
                slot=int(slot),
                overall_no=int(overall),
                pick_id=pick_id,
                original_team=original_team,
                drafting_team=drafting_team,
                attrs=dict(base_attrs),
            )
        )
    return turns


def infer_draft_year_from_state(state_snapshot: Mapping[str, Any]) -> int:
    league = state_snapshot.get("league", {}) if isinstance(state_snapshot, Mapping) else {}
    if isinstance(league, Mapping):
        dy = league.get("draft_year")
        if dy is not None and str(dy) != "":
            try:
                return int(dy)
            except (TypeError, ValueError):
                pass
        sy = league.get("season_year")
        if sy is not None and str(sy) != "":
            try:
                return int(sy) + 1
            except (TypeError, ValueError):
                pass
    raise ValueError("Cannot infer draft_year from state_snapshot (expected league.draft_year or league.season_year)")


def infer_db_path_from_state(state_snapshot: Mapping[str, Any]) -> str:
    league = state_snapshot.get("league", {}) if isinstance(state_snapshot, Mapping) else {}
    if isinstance(league, Mapping):
        db_path = league.get("db_path")
        if db_path:
            return str(db_path)
    raise ValueError("Cannot infer db_path from state_snapshot (expected league.db_path)")


def compute_plan_from_state(
    state_snapshot: Mapping[str, Any],
    *,
    draft_year: Optional[int] = None,
    rng_seed: int,
    tie_break_seed: Optional[int] = None,
    use_lottery: bool = True,
) -> DraftOrderPlan:
    dy = int(draft_year) if draft_year is not None else infer_draft_year_from_state(state_snapshot)

    records = compute_team_records_from_master_schedule(
        state_snapshot,
        team_ids=list(ALL_TEAM_IDS),
        require_initialized_schedule=True,
    )

    plan = compute_draft_order_plan_from_records(
        draft_year=dy,
        records=records,
        rng_seed=int(rng_seed),
        tie_break_seed=tie_break_seed,
        use_lottery=bool(use_lottery),
        meta={
            "rng_seed": int(rng_seed),
            "tie_break_seed": int(tie_break_seed) if tie_break_seed is not None else None,
            "use_lottery": bool(use_lottery),
        },
    )
    return plan


def settle_and_build_turns(
    *,
    db_path: str,
    plan: DraftOrderPlan,
    settle_db: bool = True,
) -> Dict[str, Any]:
    """Settle picks (protections+swaps) in DB and build the 60-turn list."""
    draft_year = int(plan.draft_year)

    # Ensure baseline pick rows exist (idempotent)
    from league_service import LeagueService

    settlement_events: List[Dict[str, Any]] = []
    if settle_db:
        with LeagueService.open(str(db_path)) as svc:
            # years_ahead is irrelevant for a single settlement call; keep minimal.
            svc.ensure_draft_picks_seeded(draft_year, list(ALL_TEAM_IDS), years_ahead=0)
            settlement_events = svc.settle_draft_year(draft_year, plan.pick_order_by_pick_id)

    turns = build_turns_from_plan(db_path=str(db_path), plan=plan, turn_attr_settled=bool(settle_db))
    return {"draft_year": draft_year, "settlement_events": settlement_events, "turns": turns}


def finalize_draft_year(
    state_snapshot: Mapping[str, Any],
    *,
    db_path: Optional[str] = None,
    draft_year: Optional[int] = None,
    rng_seed: int,
    tie_break_seed: Optional[int] = None,
    use_lottery: bool = True,
    settle_db: bool = True,
) -> Dict[str, Any]:
    """Convenience: compute plan from state, settle DB, build turns."""
    plan = compute_plan_from_state(
        state_snapshot,
        draft_year=draft_year,
        rng_seed=int(rng_seed),
        tie_break_seed=tie_break_seed,
        use_lottery=use_lottery,
    )
    dbp = str(db_path) if db_path is not None else infer_db_path_from_state(state_snapshot)
    out = settle_and_build_turns(db_path=dbp, plan=plan, settle_db=bool(settle_db))
    return {
        "draft_year": int(plan.draft_year),
        "plan": plan,
        "settlement_events": out["settlement_events"],
        "turns": out["turns"],
    }


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _json_loads(s: str) -> Any:
    try:
        return json.loads(s or "{}")
    except Exception:
        return {}


def _team_record_from_dict(team_id_fallback: str, d: Mapping[str, Any]) -> TeamRecord:
    tid = str(d.get("team_id") or team_id_fallback)
    return TeamRecord(
        team_id=tid,
        wins=int(d.get("wins") or 0),
        losses=int(d.get("losses") or 0),
        pf=int(d.get("pf") or 0),
        pa=int(d.get("pa") or 0),
    )


def _lottery_result_from_dict(d: Mapping[str, Any]) -> LotteryResult:
    winners = d.get("winners_top4") or []
    seed_order = d.get("seed_order") or []
    odds_by_team = d.get("odds_by_team") or {}
    audit = d.get("audit") or {}
    return LotteryResult(
        rng_seed=int(d.get("rng_seed") or 0),
        seed_order=tuple(str(x) for x in seed_order),
        odds_by_team={str(k): float(v) for k, v in dict(odds_by_team).items()},
        winners_top4=tuple(str(x) for x in winners[:4]),
        audit=dict(audit) if isinstance(audit, dict) else {},
    )


def _draft_order_plan_from_dict(d: Mapping[str, Any]) -> DraftOrderPlan:
    dy = int(d.get("draft_year") or 0)
    records_in = d.get("records") or {}
    records: Dict[TeamId, TeamRecord] = {}
    if isinstance(records_in, dict):
        for tid, rec in records_in.items():
            if isinstance(rec, Mapping):
                records[str(tid)] = _team_record_from_dict(str(tid), rec)

    lr = d.get("lottery_result")
    lottery_result: Optional[LotteryResult] = None
    if isinstance(lr, Mapping):
        lottery_result = _lottery_result_from_dict(lr)

    meta = d.get("meta") or {}
    return DraftOrderPlan(
        draft_year=dy,
        records=records,
        rank_worst_to_best=tuple(str(x) for x in (d.get("rank_worst_to_best") or [])),
        round1_slot_to_original_team=tuple(str(x) for x in (d.get("round1_slot_to_original_team") or [])),
        round2_slot_to_original_team=tuple(str(x) for x in (d.get("round2_slot_to_original_team") or [])),
        pick_order_by_pick_id={str(k): int(v) for k, v in dict(d.get("pick_order_by_pick_id") or {}).items()},
        lottery_result=lottery_result,
        meta=dict(meta) if isinstance(meta, dict) else {},
    )


def has_order_plan(db_path: str, draft_year: int) -> bool:
    """Return True if a persisted DraftOrderPlan exists for draft_year."""
    from league_repo import LeagueRepo

    dy = int(draft_year)
    with LeagueRepo(str(db_path)) as repo:
        repo.init_db()
        cur = repo._conn.cursor()
        row = cur.execute("SELECT 1 FROM draft_order_plans WHERE draft_year=? LIMIT 1;", (dy,)).fetchone()
        return row is not None


def store_order_plan(db_path: str, draft_year: int, plan: DraftOrderPlan) -> None:
    """Upsert the DraftOrderPlan for draft_year into draft_order_plans."""
    from league_repo import LeagueRepo

    dy = int(draft_year)
    now = game_time.now_utc_like_iso()
    plan_json = _json_dumps(plan.to_dict())

    with LeagueRepo(str(db_path)) as repo:
        repo.init_db()
        with repo.transaction() as cur:
            cur.execute(
                """
                INSERT INTO draft_order_plans(draft_year, plan_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(draft_year) DO UPDATE SET
                    plan_json=excluded.plan_json,
                    updated_at=excluded.updated_at;
                """,
                (dy, plan_json, now, now),
            )


def load_order_plan(db_path: str, draft_year: int) -> DraftOrderPlan:
    """Load a persisted DraftOrderPlan for draft_year from draft_order_plans."""
    from league_repo import LeagueRepo

    dy = int(draft_year)
    with LeagueRepo(str(db_path)) as repo:
        repo.init_db()
        cur = repo._conn.cursor()
        row = cur.execute("SELECT plan_json FROM draft_order_plans WHERE draft_year=? LIMIT 1;", (dy,)).fetchone()
        if row is None:
            raise ValueError(f"No draft_order_plan found for draft_year={dy}")
        plan_dict = _json_loads(row["plan_json"] if isinstance(row, dict) else row[0])
        if not isinstance(plan_dict, Mapping):
            raise ValueError(f"Invalid draft_order_plan payload for draft_year={dy}")
        return _draft_order_plan_from_dict(plan_dict)
