from __future__ import annotations

"""Promise evaluation logic for player agency.

This module is *pure business logic* (no DB I/O).

A "promise" is created when a user responds to a player's demand/complaint
(e.g., "I'll give you 32 MPG", "We'll shop a trade", "We'll get help").

Promises are evaluated later (typically during the monthly agency tick) to
produce trust/frustration adjustments and loggable evidence.

Design principles
-----------------
- Promises must be explainable: every resolution returns reason codes + evidence.
- Promises should never resolve incorrectly due to missing data. When the
  evaluation context lacks required evidence, the promise can be deferred
  (due_month extended) with an explicit reason.
- Mental traits modulate *strength* of reactions, not the binary outcome.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

from .utils import clamp, clamp01, mental_norm, norm_date_iso, norm_month_key, safe_float


PromiseType = Literal[
    "MINUTES",
    "HELP",
    "SHOP_TRADE",
    "ROLE",
]

PromiseStatus = Literal[
    "ACTIVE",
    "FULFILLED",
    "BROKEN",
    "EXPIRED",
    "CANCELLED",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromiseConfig:
    """Tunable parameters for promise resolution."""

    # Base trust deltas (scaled by leverage and mental)
    trust_gain_fulfilled: float = 0.08
    trust_loss_broken: float = 0.12

    # Frustration deltas (scaled)
    minutes_frustration_relief: float = 0.06
    minutes_frustration_bump: float = 0.06
    team_frustration_relief: float = 0.05
    team_frustration_bump: float = 0.07

    # Minutes promise evaluation
    minutes_tolerance_mpg: float = 1.5
    minutes_injury_defer_months: int = 1

    # HELP promise evaluation
    help_defer_months_if_missing_evidence: int = 1

    # General
    max_auto_defer_months: int = 3


DEFAULT_PROMISE_CONFIG = PromiseConfig()


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromiseSpec:
    """A promise to be persisted (created by a response)."""

    promise_type: PromiseType
    due_month: str  # YYYY-MM

    # Optional numeric target (e.g., promised MPG)
    target_value: Optional[float] = None

    # Optional structured target (e.g., need_tag, roster criteria)
    target: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PromiseEvaluationContext:
    """Context needed to evaluate an ACTIVE promise for a given month."""

    now_date_iso: str
    month_key: str

    player_id: str
    # NOTE: this value is intentionally named "current" for v1 compatibility, but in a
    # month-based tick it should be interpreted as:
    #   - the player's team at the end of the *processed month* (EOM), or
    #   - the relevant team context for the promise being evaluated.
    # Do NOT pass the real-time roster team if you are processing a past month.
    team_id_current: str

    # Player month stats (optional; depends on promise type)
    actual_mpg: Optional[float] = None
    injury_status: Optional[str] = None  # HEALTHY/OUT/RETURNING

    # Relationship / profile
    leverage: float = 0.0  # 0..1
    mental: Mapping[str, Any] = field(default_factory=dict)

    # Team context (optional)
    team_win_pct: Optional[float] = None
    team_transactions: Optional[List[Mapping[str, Any]]] = None


@dataclass(frozen=True, slots=True)
class PromiseEvaluationResult:
    """Resolution output for a promise evaluation."""

    due: bool
    resolved: bool

    new_status: PromiseStatus

    # Promise row updates the DB layer should apply.
    promise_updates: Dict[str, Any] = field(default_factory=dict)

    # State deltas to apply to player_agency_state.
    # These are *deltas*; the DB layer should clamp final values.
    state_deltas: Dict[str, float] = field(default_factory=dict)

    # Explainability
    reasons: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Month helpers
# ---------------------------------------------------------------------------


def add_months(month_key: str, delta_months: int) -> str:
    """Add delta months to YYYY-MM."""
    mk = norm_month_key(month_key) or None
    if not mk:
        # Fail safe: treat as Jan 2000.
        mk = "2000-01"
    y_s, m_s = mk.split("-")
    y = int(y_s)
    m = int(m_s)

    total = y * 12 + (m - 1) + int(delta_months)
    if total < 0:
        total = 0
    ny = total // 12
    nm = total % 12 + 1
    return f"{ny:04d}-{nm:02d}"


def due_month_from_now(now_date_iso: str, months_ahead: int) -> str:
    """Compute due_month (YYYY-MM) from a now_date_iso."""
    d = norm_date_iso(now_date_iso) or "2000-01-01"
    return add_months(d[:7], int(months_ahead))


def _month_ge(a: str, b: str) -> bool:
    """Return True if month key a >= b."""
    aa = norm_month_key(a) or "0000-00"
    bb = norm_month_key(b) or "0000-00"
    return aa >= bb


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_promise(
    promise: Mapping[str, Any],
    *,
    ctx: PromiseEvaluationContext,
    cfg: PromiseConfig = DEFAULT_PROMISE_CONFIG,
) -> PromiseEvaluationResult:
    """Evaluate an ACTIVE promise.

    Args:
        promise: promise row dict (from DB). Expected keys:
          - promise_type, status, due_month, team_id, target_value, target_json/target
        ctx: evaluation context for the current month

    Returns:
        PromiseEvaluationResult

    Notes:
        - This function is deterministic and side-effect free.
        - It will *not* mark a promise BROKEN purely due to missing evidence.
          Instead it may defer (extend due_month) with a reason.
    """

    # Normalize inputs
    ptype = str(promise.get("promise_type") or promise.get("type") or "").upper()
    status = str(promise.get("status") or "ACTIVE").upper()
    due_month = norm_month_key(promise.get("due_month") or promise.get("due_month_key"))

    if ptype not in {"MINUTES", "HELP", "SHOP_TRADE", "ROLE"}:
        return PromiseEvaluationResult(
            due=False,
            resolved=False,
            new_status="ACTIVE",
            reasons=[{"code": "PROMISE_UNKNOWN_TYPE", "evidence": {"promise_type": ptype}}],
        )

    # Ignore non-active promises
    if status != "ACTIVE":
        return PromiseEvaluationResult(
            due=False,
            resolved=False,
            new_status=status if status in {"FULFILLED", "BROKEN", "EXPIRED", "CANCELLED"} else "ACTIVE",
            reasons=[{"code": "PROMISE_NOT_ACTIVE", "evidence": {"status": status}}],
        )

    if not due_month:
        # Missing due_month => treat as due now (but explain it)
        due_month = norm_month_key(ctx.month_key) or (norm_date_iso(ctx.now_date_iso) or "2000-01-01")[:7]

    month_now = norm_month_key(ctx.month_key) or due_month

    if not _month_ge(month_now, due_month):
        return PromiseEvaluationResult(
            due=False,
            resolved=False,
            new_status="ACTIVE",
            meta={"month_now": month_now, "due_month": due_month},
        )

    # Due: dispatch by type
    if ptype == "MINUTES":
        return _eval_minutes(promise, ctx=ctx, due_month=due_month, cfg=cfg)
    if ptype == "SHOP_TRADE":
        return _eval_shop_trade(promise, ctx=ctx, due_month=due_month, cfg=cfg)
    if ptype == "HELP":
        return _eval_help(promise, ctx=ctx, due_month=due_month, cfg=cfg)

    # ROLE (placeholder): treat as deferred if we can't evaluate.
    return _defer(
        promise,
        ctx=ctx,
        due_month=due_month,
        cfg=cfg,
        reason_code="PROMISE_ROLE_UNSUPPORTED",
        evidence={"promise_type": ptype},
    )


# ---------------------------------------------------------------------------
# Type-specific evaluators
# ---------------------------------------------------------------------------


def _impact_scale(ctx: PromiseEvaluationContext) -> Dict[str, float]:
    lev = clamp01(ctx.leverage)
    ego = mental_norm(ctx.mental, "ego")
    loy = mental_norm(ctx.mental, "loyalty")
    coach = mental_norm(ctx.mental, "coachability")

    # Base scaling: high leverage => stronger reaction; low leverage still reacts somewhat.
    base = 0.45 + 0.55 * lev

    pos_mult = clamp(0.85 + 0.35 * coach + 0.25 * loy - 0.15 * ego, 0.50, 1.60)
    neg_mult = clamp(0.90 + 0.45 * ego - 0.25 * loy - 0.10 * coach, 0.55, 2.00)

    return {"base": float(base), "pos_mult": float(pos_mult), "neg_mult": float(neg_mult)}


def _fulfilled(
    *,
    promise_type: str,
    ctx: PromiseEvaluationContext,
    cfg: PromiseConfig,
    evidence: Mapping[str, Any],
) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, Any]]:
    scale = _impact_scale(ctx)
    base = scale["base"]
    pos_mult = scale["pos_mult"]

    trust_delta = float(cfg.trust_gain_fulfilled) * base * pos_mult

    state: Dict[str, float] = {"trust": trust_delta}
    if promise_type == "MINUTES":
        state["minutes_frustration"] = -float(cfg.minutes_frustration_relief) * base
    elif promise_type in {"HELP", "SHOP_TRADE"}:
        state["team_frustration"] = -float(cfg.team_frustration_relief) * base

    reasons = [
        {
            "code": "PROMISE_FULFILLED",
            "evidence": dict(evidence),
        }
    ]

    meta = {"scale": scale, "promise_type": promise_type}
    return state, reasons, meta


def _broken(
    *,
    promise_type: str,
    ctx: PromiseEvaluationContext,
    cfg: PromiseConfig,
    evidence: Mapping[str, Any],
    severe: bool = False,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, Any]]:
    scale = _impact_scale(ctx)
    base = scale["base"]
    neg_mult = scale["neg_mult"]

    # Broken promises should hurt more than fulfilled promises help.
    trust_delta = -float(cfg.trust_loss_broken) * base * neg_mult * (1.20 if severe else 1.00)

    state: Dict[str, float] = {"trust": trust_delta}
    if promise_type == "MINUTES":
        state["minutes_frustration"] = float(cfg.minutes_frustration_bump) * base
    elif promise_type in {"HELP", "SHOP_TRADE"}:
        state["team_frustration"] = float(cfg.team_frustration_bump) * base

    # Broken trade/shop promises can also increase minutes frustration slightly.
    if promise_type == "SHOP_TRADE":
        state["minutes_frustration"] = state.get("minutes_frustration", 0.0) + 0.02 * base

    reasons = [
        {
            "code": "PROMISE_BROKEN",
            "evidence": dict(evidence),
        }
    ]
    meta = {"scale": scale, "promise_type": promise_type, "severe": bool(severe)}
    return state, reasons, meta


def _defer(
    promise: Mapping[str, Any],
    *,
    ctx: PromiseEvaluationContext,
    due_month: str,
    cfg: PromiseConfig,
    reason_code: str,
    evidence: Mapping[str, Any],
    months: Optional[int] = None,
) -> PromiseEvaluationResult:
    # Defer by a bounded number of months.
    defer_m = int(months) if months is not None else 1
    defer_m = max(1, min(defer_m, int(cfg.max_auto_defer_months)))

    new_due = add_months(due_month, defer_m)

    return PromiseEvaluationResult(
        due=True,
        resolved=False,
        new_status="ACTIVE",
        promise_updates={"due_month": new_due},
        reasons=[{"code": reason_code, "evidence": {**dict(evidence), "old_due_month": due_month, "new_due_month": new_due}}],
        meta={"defer_months": defer_m},
    )


def _eval_minutes(
    promise: Mapping[str, Any],
    *,
    ctx: PromiseEvaluationContext,
    due_month: str,
    cfg: PromiseConfig,
) -> PromiseEvaluationResult:
    target = safe_float(promise.get("target_value"), 0.0)
    actual = safe_float(ctx.actual_mpg, 0.0)
    tol = max(0.0, float(cfg.minutes_tolerance_mpg))

    injury = str(ctx.injury_status or "").upper()

    # If player was out, do not punish; defer instead.
    if injury == "OUT":
        return _defer(
            promise,
            ctx=ctx,
            due_month=due_month,
            cfg=cfg,
            reason_code="PROMISE_DEFER_INJURY_OUT",
            evidence={"injury_status": injury, "target_mpg": target, "actual_mpg": actual},
            months=int(cfg.minutes_injury_defer_months),
        )

    if actual >= (target - tol):
        state_delta, reasons, meta = _fulfilled(
            promise_type="MINUTES",
            ctx=ctx,
            cfg=cfg,
            evidence={"target_mpg": target, "actual_mpg": actual, "tolerance": tol, "injury_status": injury},
        )
        return PromiseEvaluationResult(
            due=True,
            resolved=True,
            new_status="FULFILLED",
            promise_updates={"status": "FULFILLED", "resolved_at": norm_date_iso(ctx.now_date_iso)},
            state_deltas=state_delta,
            reasons=reasons,
            meta=meta,
        )

    # Returning from injury: allow one defer if they played very little.
    if injury == "RETURNING" and actual <= 1.0:
        return _defer(
            promise,
            ctx=ctx,
            due_month=due_month,
            cfg=cfg,
            reason_code="PROMISE_DEFER_INJURY_RETURNING",
            evidence={"injury_status": injury, "target_mpg": target, "actual_mpg": actual},
            months=1,
        )

    state_delta, reasons, meta = _broken(
        promise_type="MINUTES",
        ctx=ctx,
        cfg=cfg,
        evidence={"target_mpg": target, "actual_mpg": actual, "tolerance": tol, "injury_status": injury},
        severe=(target >= 30.0 and actual < 10.0),
    )
    return PromiseEvaluationResult(
        due=True,
        resolved=True,
        new_status="BROKEN",
        promise_updates={"status": "BROKEN", "resolved_at": norm_date_iso(ctx.now_date_iso)},
        state_deltas=state_delta,
        reasons=reasons,
        meta=meta,
    )


def _eval_shop_trade(
    promise: Mapping[str, Any],
    *,
    ctx: PromiseEvaluationContext,
    due_month: str,
    cfg: PromiseConfig,
) -> PromiseEvaluationResult:
    original_team = str(promise.get("team_id") or "").upper()
    current_team = str(ctx.team_id_current or "").upper()

    if original_team and current_team and current_team != original_team:
        state_delta, reasons, meta = _fulfilled(
            promise_type="SHOP_TRADE",
            ctx=ctx,
            cfg=cfg,
            evidence={"original_team": original_team, "current_team": current_team},
        )
        return PromiseEvaluationResult(
            due=True,
            resolved=True,
            new_status="FULFILLED",
            promise_updates={"status": "FULFILLED", "resolved_at": norm_date_iso(ctx.now_date_iso)},
            state_deltas=state_delta,
            reasons=reasons,
            meta=meta,
        )

    state_delta, reasons, meta = _broken(
        promise_type="SHOP_TRADE",
        ctx=ctx,
        cfg=cfg,
        evidence={"original_team": original_team, "current_team": current_team},
        severe=True,
    )
    # A broken trade-shop promise should often escalate a trade request.
    state_delta["trade_request_level_min"] = 2.0

    return PromiseEvaluationResult(
        due=True,
        resolved=True,
        new_status="BROKEN",
        promise_updates={"status": "BROKEN", "resolved_at": norm_date_iso(ctx.now_date_iso)},
        state_deltas=state_delta,
        reasons=reasons,
        meta=meta,
    )


def _eval_help(
    promise: Mapping[str, Any],
    *,
    ctx: PromiseEvaluationContext,
    due_month: str,
    cfg: PromiseConfig,
) -> PromiseEvaluationResult:
    # HELP is evaluated by evidence: transactions or explicit evaluation score.
    tx = ctx.team_transactions

    # If no evidence was provided, defer instead of penalizing.
    if not tx:
        return _defer(
            promise,
            ctx=ctx,
            due_month=due_month,
            cfg=cfg,
            reason_code="PROMISE_DEFER_MISSING_EVIDENCE",
            evidence={"promise_type": "HELP"},
            months=int(cfg.help_defer_months_if_missing_evidence),
        )

    # Minimal heuristic: if there was *any* roster-changing transaction, assume progress.
    # Higher fidelity evaluation can be added later by providing richer tx payloads.
    meaningful = 0
    for e in tx:
        if not isinstance(e, Mapping):
            continue
        t = str(e.get("type") or e.get("tx_type") or "").upper()
        if not t:
            continue
        if any(k in t for k in ("TRADE", "SIGN", "RE_SIGN", "EXTEND", "WAIVE", "CLAIM", "BUYOUT")):
            meaningful += 1

    if meaningful > 0:
        state_delta, reasons, meta = _fulfilled(
            promise_type="HELP",
            ctx=ctx,
            cfg=cfg,
            evidence={"meaningful_transactions": meaningful},
        )
        return PromiseEvaluationResult(
            due=True,
            resolved=True,
            new_status="FULFILLED",
            promise_updates={"status": "FULFILLED", "resolved_at": norm_date_iso(ctx.now_date_iso)},
            state_deltas=state_delta,
            reasons=reasons,
            meta=meta,
        )

    state_delta, reasons, meta = _broken(
        promise_type="HELP",
        ctx=ctx,
        cfg=cfg,
        evidence={"meaningful_transactions": meaningful},
        severe=False,
    )
    return PromiseEvaluationResult(
        due=True,
        resolved=True,
        new_status="BROKEN",
        promise_updates={"status": "BROKEN", "resolved_at": norm_date_iso(ctx.now_date_iso)},
        state_deltas=state_delta,
        reasons=reasons,
        meta=meta,
    )
