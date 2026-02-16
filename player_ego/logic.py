# player_ego/logic.py
"""Player Ego subsystem: pure decision logic.

This module contains *no database I/O*. It operates on plain dicts and returns
updated copies.

The ego model is intentionally conservative:
- It should create life-like friction without turning the game into a
  "complaint simulator".
- Bench players should rarely create high-severity issues.
- Stars have more leverage but still differ by personality archetype.

The numeric model is designed to be easy to tune.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .types import (
    ContractOffer,
    DesiredRole,
    EgoIssue,
    EgoState,
    OfferDecision,
    PlayerTraits,
)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


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


def _clamp(x: float, lo: float, hi: float) -> float:
    xf = _safe_float(x, lo)
    if xf < lo:
        return float(lo)
    if xf > hi:
        return float(hi)
    return float(xf)


def _clamp01(x: float) -> float:
    return _clamp(x, 0.0, 1.0)


def _stable_u32(*parts: str) -> int:
    h = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _stable_unit_float(*parts: str) -> float:
    return (_stable_u32(*parts) % 10_000_000) / 10_000_000.0


def _sigmoid(x: float) -> float:
    # stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def normalize_traits(traits: Mapping[str, Any]) -> PlayerTraits:
    """Return a safe traits dict with defaults + clamp."""
    t: PlayerTraits = {
        "version": str(traits.get("version") or "1.0"),
        "archetype": str(traits.get("archetype") or "UNKNOWN"),
    }

    def _get01(k: str, d: float) -> float:
        return _clamp01(_safe_float(traits.get(k), d))

    t["ego"] = _get01("ego", 0.5)
    t["loyalty"] = _get01("loyalty", 0.5)
    t["money_focus"] = _get01("money_focus", 0.5)
    t["win_focus"] = _get01("win_focus", 0.5)
    t["ambition"] = _get01("ambition", 0.5)
    t["patience"] = _get01("patience", 0.5)
    t["professionalism"] = _get01("professionalism", 0.6)
    t["volatility"] = _get01("volatility", 0.4)
    t["privacy"] = _get01("privacy", 0.6)
    t["risk_tolerance"] = _get01("risk_tolerance", 0.5)
    return t


def normalize_state(state: Mapping[str, Any]) -> EgoState:
    """Return a safe state dict with defaults + clamp."""
    s: EgoState = {
        "version": str(state.get("version") or "1.0"),
        "happiness": _clamp01(_safe_float(state.get("happiness"), 0.72)),
        "trust_team": _clamp01(_safe_float(state.get("trust_team"), 0.65)),
        "desired_minutes": _clamp(_safe_float(state.get("desired_minutes"), 18.0), 0.0, 48.0),
        "desired_role": str(state.get("desired_role") or "ROTATION"),  # type: ignore[assignment]
        "recent_minutes": list(state.get("recent_minutes") or []),
        "open_issues": list(state.get("open_issues") or []),
        "cooldowns": dict(state.get("cooldowns") or {}),
        "last_team_id": state.get("last_team_id"),
        "last_updated_date": state.get("last_updated_date"),
    }

    # Sanitize desired_role
    if s["desired_role"] not in {"STARTER", "ROTATION", "BENCH"}:
        s["desired_role"] = "ROTATION"  # type: ignore[assignment]

    # Sanitize recent_minutes buffer
    rm: List[float] = []
    for v in s["recent_minutes"]:
        try:
            rm.append(float(v))
        except Exception:
            continue
    s["recent_minutes"] = rm[-10:]

    # Sanitize issues: keep only dicts
    issues: List[EgoIssue] = []
    for it in s["open_issues"]:
        if isinstance(it, dict):
            issues.append(dict(it))  # shallow copy
    s["open_issues"] = issues

    # cooldowns -> ensure str dates
    cds: Dict[str, str] = {}
    for k, v in s["cooldowns"].items():
        if k is None or v is None:
            continue
        cds[str(k)] = str(v)[:10]
    s["cooldowns"] = cds

    return s


def default_desired_minutes(
    *,
    ovr: Optional[int],
    age: Optional[int],
    traits: PlayerTraits,
    leverage_hint: Optional[float] = None,
) -> float:
    """Compute a starting desired_minutes for a player."""
    o = int(ovr or 0)
    a = int(age or 0)
    ambition = float(traits.get("ambition", 0.5))

    # Base minutes by ability
    if o >= 88:
        base = 34.0
    elif o >= 82:
        base = 30.0
    elif o >= 76:
        base = 24.0
    elif o >= 70:
        base = 16.0
    else:
        base = 10.0

    # Youth tends to want more, vets may accept less
    if a and a <= 23:
        base += 2.0
    elif a and a >= 33:
        base -= 1.5

    # Ambition pushes target up
    base += (ambition - 0.5) * 6.0

    # Leverage hint nudges target further (star expects star minutes)
    if leverage_hint is not None:
        base += (float(leverage_hint) - 0.5) * 4.0

    return float(_clamp(base, 6.0, 40.0))


def default_desired_role(*, ovr: Optional[int], traits: PlayerTraits) -> DesiredRole:
    o = int(ovr or 0)
    ambition = float(traits.get("ambition", 0.5))

    if o >= 86:
        return "STARTER"
    if o >= 79:
        return "STARTER" if ambition >= 0.65 else "ROTATION"
    if o >= 74:
        return "ROTATION"
    return "BENCH"


def make_default_state(
    *,
    player: Mapping[str, Any],
    traits: Mapping[str, Any],
    team_id: Optional[str] = None,
    date_iso: Optional[str] = None,
) -> EgoState:
    """Create a fresh state blob for a new player ego row."""
    t = normalize_traits(traits)
    ovr = _safe_int(player.get("ovr"), 0)
    age = _safe_int(player.get("age"), 0)

    desired_role = default_desired_role(ovr=ovr, traits=t)
    desired_minutes = default_desired_minutes(ovr=ovr, age=age, traits=t, leverage_hint=None)

    s: EgoState = {
        "version": "1.0",
        # Start slightly positive, but not maxed.
        "happiness": 0.74,
        "trust_team": 0.66,
        "desired_minutes": float(desired_minutes),
        "desired_role": desired_role,
        "recent_minutes": [],
        "open_issues": [],
        "cooldowns": {},
        "last_team_id": str(team_id).upper() if team_id else None,
        "last_updated_date": str(date_iso)[:10] if date_iso else None,
    }
    return s


# -----------------------------------------------------------------------------
# Team context helpers (pure)
# -----------------------------------------------------------------------------


def compute_team_win_pct_from_master_schedule(
    state_snapshot: Mapping[str, Any],
    *,
    team_id: str,
) -> Optional[float]:
    """Compute a team's win% from master_schedule finals."""
    try:
        from draft.standings import compute_team_records_from_master_schedule

        records = compute_team_records_from_master_schedule(
            state_snapshot, team_ids=[str(team_id).upper()], require_initialized_schedule=False
        )
        rec = records.get(str(team_id).upper())
        if rec is None:
            return None
        games = int(rec.wins + rec.losses)
        if games <= 0:
            return None
        return float(rec.wins) / float(games)
    except Exception:
        return None


def classify_team_competitiveness(win_pct: Optional[float]) -> str:
    """Bucket team direction used for negotiation logic."""
    if win_pct is None:
        return "unknown"
    if win_pct >= 0.58:
        return "contender"
    if win_pct >= 0.48:
        return "competitive"
    if win_pct >= 0.40:
        return "fringe"
    return "rebuilding"


# -----------------------------------------------------------------------------
# Leverage model (pure)
# -----------------------------------------------------------------------------


def calc_leverage(
    *,
    player_id: str,
    player_ovr: Optional[int],
    team_roster: Optional[List[Mapping[str, Any]]] = None,
    salary_amount: Optional[int] = None,
) -> float:
    """Compute a 0..1 leverage score."""
    ovr = _safe_int(player_ovr, 0)
    ability = _clamp01((ovr - 65.0) / 25.0)  # 65->0, 90->1

    rank_factor = 0.5
    if team_roster:
        ovs: List[Tuple[str, int]] = []
        for r in team_roster:
            pid = str(r.get("player_id") or "")
            ovs.append((pid, _safe_int(r.get("ovr"), 0)))
        ovs.sort(key=lambda kv: (-kv[1], kv[0]))
        ids = [pid for pid, _ in ovs]
        if player_id in ids:
            idx = ids.index(player_id)
            n = max(1, len(ids))
            rank_factor = 1.0 - (idx / max(1, n - 1))

    salary_factor = 0.5
    if salary_amount is not None:
        s = float(max(0, int(salary_amount)))
        salary_factor = _clamp01((s - 1_500_000.0) / 35_000_000.0)

    leverage = 0.55 * ability + 0.35 * rank_factor + 0.10 * salary_factor
    return float(_clamp01(leverage))


# -----------------------------------------------------------------------------
# Salary expectation (pure, consistent with trades/valuation)
# -----------------------------------------------------------------------------


def expected_salary_from_ovr(ovr: int) -> float:
    """Market-expected salary based on OVR."""
    try:
        from trades.valuation.market_pricing import MarketPricingConfig

        cfg = MarketPricingConfig()
        x = (float(ovr) - cfg.expected_salary_ovr_center) / max(cfg.expected_salary_ovr_scale, cfg.eps)
        s = _sigmoid(x)
        lo = cfg.expected_salary_midpoint - cfg.expected_salary_span
        hi = cfg.expected_salary_midpoint + cfg.expected_salary_span
        return float(lo + (hi - lo) * s)
    except Exception:
        return float(_clamp(ovr, 60, 95) - 60) / 35.0 * 28_000_000.0 + 2_000_000.0


# -----------------------------------------------------------------------------
# Happiness + issue model (pure)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EgoUpdateResult:
    traits: PlayerTraits
    state: EgoState
    new_issues: List[EgoIssue]
    escalated_issues: List[EgoIssue]


def _issue_id() -> str:
    return uuid.uuid4().hex


def _get_issue(open_issues: List[EgoIssue], issue_type: str) -> Optional[EgoIssue]:
    for it in open_issues:
        if not isinstance(it, dict):
            continue
        if it.get("status") != "OPEN":
            continue
        if it.get("type") == issue_type:
            return it
    return None


def _cooldown_ok(state: EgoState, issue_type: str, *, date_iso: str) -> bool:
    cd = (state.get("cooldowns") or {}).get(issue_type)
    if not cd:
        return True
    return str(date_iso)[:10] >= str(cd)[:10]


def _set_cooldown(state: EgoState, issue_type: str, *, until_date_iso: str) -> None:
    cds = state.setdefault("cooldowns", {})
    cds[str(issue_type)] = str(until_date_iso)[:10]


def update_after_game(
    *,
    player: Mapping[str, Any],
    traits_in: Mapping[str, Any],
    state_in: Mapping[str, Any],
    date_iso: str,
    team_id: str,
    minutes_played: float,
    team_win_pct: Optional[float],
    salary_amount: Optional[int],
    leverage: float,
    recent_window: int = 8,
) -> EgoUpdateResult:
    """Update ego state after a game."""
    traits = normalize_traits(traits_in)
    state = normalize_state(state_in)

    rm = list(state.get("recent_minutes") or [])
    rm.append(float(_clamp(minutes_played, 0.0, 60.0)))
    state["recent_minutes"] = rm[-10:]

    desired = float(state.get("desired_minutes") or 18.0)
    window = max(1, int(recent_window))
    recent = state["recent_minutes"][-window:]
    avg_min = float(sum(recent) / max(1, len(recent)))

    gap = max(0.0, desired - avg_min)
    minutes_sat = _clamp01(1.0 - (gap / max(1.0, desired)))

    if team_win_pct is None:
        team_sat = 0.55
    else:
        team_sat = _clamp01((float(team_win_pct) - 0.30) / 0.40)

    money_sat = 0.55
    if salary_amount is not None:
        ovr = _safe_int(player.get("ovr"), 0)
        exp = expected_salary_from_ovr(ovr)
        ratio = float(salary_amount) / max(1.0, float(exp))
        if ratio >= 1.0:
            money_sat = _clamp01(0.65 + 0.35 * min(1.0, ratio - 1.0))
        else:
            money_sat = _clamp01(0.25 + 0.75 * ratio)

    w_minutes = 0.25 + 0.45 * float(traits.get("ambition", 0.5))
    w_team = 0.15 + 0.40 * float(traits.get("win_focus", 0.5))
    w_money = 0.10 + 0.40 * float(traits.get("money_focus", 0.5))
    total_w = max(1e-6, w_minutes + w_team + w_money)

    target = (w_minutes * minutes_sat + w_team * team_sat + w_money * money_sat) / total_w

    inertia = 0.86
    vol = float(traits.get("volatility", 0.4))
    jitter = (
        _stable_unit_float("ego_mood", str(player.get("player_id") or ""), str(date_iso)[:10]) * 2.0 - 1.0
    )
    jitter *= 0.04 * vol

    prev_h = float(state.get("happiness") or 0.7)
    new_h = inertia * prev_h + (1.0 - inertia) * target + jitter
    state["happiness"] = float(_clamp01(new_h))

    prev_t = float(state.get("trust_team") or 0.65)
    trust_target = 0.55 + 0.35 * float(traits.get("professionalism", 0.6))
    state["trust_team"] = float(_clamp01(0.93 * prev_t + 0.07 * trust_target))

    state["last_team_id"] = str(team_id).upper() if team_id else None
    state["last_updated_date"] = str(date_iso)[:10]

    new_issues: List[EgoIssue] = []
    escalated: List[EgoIssue] = []

    if len(state["recent_minutes"]) >= 4:
        min_issue = _get_issue(state["open_issues"], "MINUTES_COMPLAINT")

        patience = float(traits.get("patience", 0.5))
        ego = float(traits.get("ego", 0.5))

        threshold = 6.0 + 5.0 * (1.0 - patience) + 3.0 * (1.0 - leverage)
        if leverage >= 0.75:
            threshold -= 1.5
        threshold = max(4.0, threshold)

        if gap > threshold and _cooldown_ok(state, "MINUTES_COMPLAINT", date_iso=date_iso):
            sev = _clamp01((gap - threshold) / 12.0)
            sev *= 0.70 + 0.30 * ego

            if leverage < 0.35:
                sev *= 0.50
            if leverage < 0.20:
                sev *= 0.35

            p_gate = _clamp01(0.35 + 0.50 * sev + 0.25 * (1.0 - patience))
            if _stable_unit_float("ego_issue_gate", str(player.get("player_id") or ""), str(date_iso)[:10]) <= p_gate:
                if min_issue is None:
                    issue: EgoIssue = {
                        "issue_id": _issue_id(),
                        "type": "MINUTES_COMPLAINT",
                        "status": "OPEN",
                        "severity": float(sev),
                        "created_date": str(date_iso)[:10],
                        "updated_date": str(date_iso)[:10],
                        "title": "Playing time frustration",
                        "summary": f"Wants ~{desired:.0f} MPG, recently averaging ~{avg_min:.1f} MPG.",
                        "meta": {
                            "desired_minutes": float(desired),
                            "avg_recent_minutes": float(avg_min),
                            "gap": float(gap),
                            "window": int(window),
                        },
                    }
                    state["open_issues"].append(issue)
                    new_issues.append(issue)
                    _set_cooldown(state, "MINUTES_COMPLAINT", until_date_iso=str(date_iso)[:10])
                else:
                    old = float(min_issue.get("severity") or 0.0)
                    new_sev = float(_clamp01(max(old, sev) + 0.05))
                    if new_sev > old + 1e-6:
                        min_issue["severity"] = new_sev
                        min_issue["updated_date"] = str(date_iso)[:10]
                        escalated.append(min_issue)

    if (
        float(state.get("happiness") or 0.0) < 0.34
        and leverage >= 0.70
        and float(traits.get("loyalty", 0.5)) < 0.42
        and _cooldown_ok(state, "TRADE_REQUEST", date_iso=date_iso)
    ):
        minutes_issue = _get_issue(state["open_issues"], "MINUTES_COMPLAINT")
        if minutes_issue is not None and float(minutes_issue.get("severity") or 0.0) >= 0.75:
            u = _stable_unit_float("ego_trade_gate", str(player.get("player_id") or ""), str(date_iso)[:10])
            base_p = 0.10 + 0.45 * (1.0 - float(traits.get("loyalty", 0.5)))
            base_p += 0.20 * (1.0 - float(traits.get("patience", 0.5)))
            base_p += 0.10 * leverage
            base_p = _clamp01(base_p)
            if u <= base_p:
                issue = {
                    "issue_id": _issue_id(),
                    "type": "TRADE_REQUEST",
                    "status": "OPEN",
                    "severity": float(_clamp01(0.65 + 0.35 * leverage)),
                    "created_date": str(date_iso)[:10],
                    "updated_date": str(date_iso)[:10],
                    "title": "Trade request",
                    "summary": "Unhappy with role and wants a change of scenery.",
                    "meta": {
                        "trigger": "minutes",
                        "happiness": float(state.get("happiness") or 0.0),
                        "leverage": float(leverage),
                    },
                }
                state["open_issues"].append(issue)
                new_issues.append(issue)
                _set_cooldown(state, "TRADE_REQUEST", until_date_iso=str(date_iso)[:10])

    return EgoUpdateResult(traits=traits, state=state, new_issues=new_issues, escalated_issues=escalated)


# -----------------------------------------------------------------------------
# Option decision (pure)
# -----------------------------------------------------------------------------


def decide_player_option(
    *,
    option: Mapping[str, Any],
    contract: Mapping[str, Any],
    player: Mapping[str, Any],
    traits_in: Mapping[str, Any],
    state_in: Mapping[str, Any],
    team_win_pct: Optional[float],
    decision_date_iso: str,
) -> str:
    """Return "EXERCISE" or "DECLINE" for PLAYER/ETO options."""
    traits = normalize_traits(traits_in)
    state = normalize_state(state_in)

    opt_type = str(option.get("type") or "").upper()
    season_year = _safe_int(option.get("season_year"), 0)

    salary_by_year = contract.get("salary_by_year") or {}
    opt_salary = None
    if isinstance(salary_by_year, dict) and season_year:
        v = salary_by_year.get(str(season_year))
        if v is None:
            v = salary_by_year.get(int(season_year))
        if v is not None:
            try:
                opt_salary = float(v)
            except Exception:
                opt_salary = None

    ovr = _safe_int(player.get("ovr"), 0)
    age = _safe_int(player.get("age"), 0)

    if opt_salary is None:
        return "EXERCISE"

    exp = expected_salary_from_ovr(ovr)
    market_ratio = float(opt_salary) / max(1.0, float(exp))

    underpaid = max(0.0, 1.0 - market_ratio)

    risk = 0.0
    if age >= 32:
        risk += 0.12
    if ovr <= 74:
        risk += 0.10

    risk += 0.15 * (1.0 - float(traits.get("risk_tolerance", 0.5)))
    risk += 0.10 * float(traits.get("money_focus", 0.5))

    if team_win_pct is not None:
        contending_bonus = 0.10 * float(traits.get("win_focus", 0.5)) * max(0.0, 0.50 - float(team_win_pct))
    else:
        contending_bonus = 0.0

    u = _stable_unit_float(
        "ego_option",
        str(player.get("player_id") or ""),
        str(season_year),
        str(decision_date_iso)[:10],
        opt_type,
    )

    p_optout = 0.05
    p_optout += 0.75 * underpaid
    p_optout += contending_bonus

    if opt_type == "ETO":
        p_optout += 0.08

    p_optout -= risk

    p_optout -= 0.10 * float(state.get("happiness") or 0.7)
    p_optout -= 0.08 * float(traits.get("loyalty", 0.5))

    p_optout = _clamp01(p_optout)

    if u <= p_optout:
        return "DECLINE"
    return "EXERCISE"


# -----------------------------------------------------------------------------
# Offer evaluation (pure)
# -----------------------------------------------------------------------------


def _avg_annual_value(salary_by_year: Mapping[str, Any]) -> Optional[float]:
    vals: List[float] = []
    for v in salary_by_year.values():
        try:
            vals.append(float(v))
        except Exception:
            continue
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def evaluate_re_sign_offer(
    *,
    player: Mapping[str, Any],
    team_id: str,
    offer: ContractOffer,
    traits_in: Mapping[str, Any],
    state_in: Mapping[str, Any],
    team_win_pct: Optional[float],
    decision_date_iso: str,
    leverage: float,
) -> OfferDecision:
    """Evaluate a re-sign/extension offer."""
    traits = normalize_traits(traits_in)
    state = normalize_state(state_in)

    pid = str(player.get("player_id") or "")
    ovr = _safe_int(player.get("ovr"), 0)
    age = _safe_int(player.get("age"), 0)

    salary_by_year = offer.get("salary_by_year") or {}
    if not isinstance(salary_by_year, dict) or not salary_by_year:
        return {
            "decision": "REJECT",
            "reason_code": "INVALID_OFFER",
            "reason": "Offer is missing salary_by_year.",
            "counter": None,
        }

    aav = _avg_annual_value(salary_by_year)
    if aav is None:
        return {
            "decision": "REJECT",
            "reason_code": "INVALID_OFFER",
            "reason": "Offer salary values are invalid.",
            "counter": None,
        }

    expected = expected_salary_from_ovr(ovr)

    money_focus = float(traits.get("money_focus", 0.5))
    win_focus = float(traits.get("win_focus", 0.5))
    loyalty = float(traits.get("loyalty", 0.5))

    happiness = float(state.get("happiness") or 0.7)

    pay_ratio = float(aav) / max(1.0, float(expected))

    if pay_ratio >= 1.05:
        pay_score = _clamp01(0.65 + 0.35 * min(1.0, pay_ratio - 1.05))
    elif pay_ratio >= 0.92:
        pay_score = _clamp01(0.50 + 0.15 * (pay_ratio - 0.92) / 0.13)
    else:
        pay_score = _clamp01(0.15 + 0.35 * (pay_ratio / 0.92))

    comp = classify_team_competitiveness(team_win_pct)
    if comp in {"contender", "competitive"}:
        team_score = 0.80 if comp == "contender" else 0.65
    elif comp == "fringe":
        team_score = 0.45
    elif comp == "rebuilding":
        team_score = 0.25
    else:
        team_score = 0.50

    role_score = 0.55
    promised_role = offer.get("promised_role")
    if promised_role in {"STARTER", "ROTATION", "BENCH"}:
        desired_role = str(state.get("desired_role") or "ROTATION")
        if promised_role == desired_role:
            role_score = 0.80
        elif desired_role == "STARTER" and promised_role != "STARTER":
            role_score = 0.35
        else:
            role_score = 0.55

    w_pay = 0.40 + 0.45 * money_focus
    w_team = 0.15 + 0.45 * win_focus
    w_role = 0.10 + 0.20 * float(traits.get("ambition", 0.5))
    w_loyal = 0.05 + 0.15 * loyalty

    total_w = max(1e-6, w_pay + w_team + w_role + w_loyal)
    accept_score = (
        w_pay * pay_score
        + w_team * team_score
        + w_role * role_score
        + w_loyal * (0.55 + 0.35 * loyalty)
    ) / total_w

    accept_score += 0.10 * (happiness - 0.6)

    if age >= 32:
        accept_score += 0.03

    accept_score -= 0.10 * max(0.0, leverage - 0.55)

    accept_score = _clamp01(accept_score)

    u = _stable_unit_float("ego_offer", pid, str(team_id).upper(), str(decision_date_iso)[:10])
    noise = (u * 2.0 - 1.0) * (0.03 + 0.05 * float(traits.get("volatility", 0.4)))
    accept_score = _clamp01(accept_score + noise)

    if accept_score >= 0.70:
        return {
            "decision": "ACCEPT",
            "reason_code": "ACCEPTED",
            "reason": "Offer meets expectations.",
            "counter": None,
        }

    if accept_score >= 0.52:
        premium = 0.05 + 0.10 * money_focus + 0.08 * max(0.0, leverage - 0.5)
        if pay_ratio < 0.92:
            premium += 0.08

        target_aav = max(float(aav) * (1.0 + premium), float(expected) * (0.95 + 0.25 * money_focus))

        years = int(offer.get("years") or len(salary_by_year) or 1)
        keys = sorted(str(k) for k in salary_by_year.keys())
        if len(keys) != years:
            years = len(keys)

        counter_salary: Dict[str, float] = {}
        for k in keys:
            counter_salary[k] = float(target_aav)

        counter: ContractOffer = {
            "years": years,
            "salary_by_year": counter_salary,
            "promised_role": offer.get("promised_role"),
            "promised_minutes": offer.get("promised_minutes"),
        }

        return {
            "decision": "COUNTER",
            "reason_code": "COUNTER_OFFER",
            "reason": "Wants improved terms.",
            "counter": counter,
        }

    reason_code = "UNDERPAID" if pay_ratio < 0.90 else "UNHAPPY"
    if win_focus >= 0.70 and team_win_pct is not None and team_win_pct < 0.42:
        reason_code = "TEAM_NOT_COMPETITIVE"

    return {
        "decision": "REJECT",
        "reason_code": reason_code,
        "reason": "Offer does not match expectations.",
        "counter": None,
    }
