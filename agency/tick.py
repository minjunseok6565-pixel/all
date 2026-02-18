from __future__ import annotations

"""Monthly agency tick logic.

This module contains the *behavioral core*:
- Update frustrations + trust (state variables)
- Emit events (complaints, demands, trade requests)

Design notes
------------
1) Mental traits are modulators, not absolute rules.
2) Strong actions are gated by leverage (role/importance).
3) Deterministic randomness (stable hashing) provides variety without breaking
   reproducibility.

This file is intentionally free of DB I/O.
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple

from .config import AgencyConfig
from .types import MonthlyPlayerInputs
from .utils import (
    clamp,
    clamp01,
    date_add_days,
    make_event_id,
    mental_norm,
    norm_date_iso,
    stable_u01,
)


def _cooldown_active(until_iso: Optional[str], *, now_date_iso: str) -> bool:
    u = norm_date_iso(until_iso)
    if not u:
        return False
    # ISO dates compare lexicographically.
    return str(u) > str(now_date_iso)[:10]


def _compute_minutes_tolerance_mpg(mental: Mapping[str, Any], cfg: AgencyConfig) -> float:
    fcfg = cfg.frustration
    coach = mental_norm(mental, "coachability")
    loy = mental_norm(mental, "loyalty")
    adapt = mental_norm(mental, "adaptability")
    ego = mental_norm(mental, "ego")
    amb = mental_norm(mental, "ambition")

    tol = (
        float(fcfg.tolerance_base_mpg)
        + float(fcfg.tolerance_coachability_bonus) * coach
        + float(fcfg.tolerance_loyalty_bonus) * loy
        + float(fcfg.tolerance_adaptability_bonus) * adapt
        - float(fcfg.tolerance_ego_penalty) * ego
        - float(fcfg.tolerance_ambition_penalty) * amb
    )
    return float(clamp(tol, float(fcfg.tolerance_min_mpg), float(fcfg.tolerance_max_mpg)))


def _injury_multiplier(status: Optional[str], cfg: AgencyConfig) -> float:
    fcfg = cfg.frustration
    s = str(status or "").upper()
    if s == "OUT":
        return float(clamp01(fcfg.injury_out_multiplier))
    if s == "RETURNING":
        return float(clamp01(fcfg.injury_returning_multiplier))
    return 1.0


def _update_minutes_frustration(
    *,
    prev: float,
    expected_mpg: float,
    actual_mpg: float,
    mental: Mapping[str, Any],
    leverage: float,
    injury_status: Optional[str],
    cfg: AgencyConfig,
) -> Tuple[float, Dict[str, Any]]:
    """Update minutes frustration using a smooth EMA."""
    fcfg = cfg.frustration

    expected = max(0.0, float(expected_mpg))
    actual = max(0.0, float(actual_mpg))

    gap = max(0.0, expected - actual)

    tol = _compute_minutes_tolerance_mpg(mental, cfg)
    gap_pressure = clamp01(gap / max(tol, 1e-9))

    coach = mental_norm(mental, "coachability")
    loy = mental_norm(mental, "loyalty")
    ego = mental_norm(mental, "ego")
    amb = mental_norm(mental, "ambition")

    gain_mult = clamp(
        0.60 + 0.70 * ego + 0.40 * amb - 0.50 * coach - 0.30 * loy,
        0.25,
        1.75,
    )

    inj_mult = _injury_multiplier(injury_status, cfg)

    # If the player is "close enough" to expectation, let frustration cool down faster.
    within = gap <= (0.50 * tol)
    decay = float(fcfg.minutes_decay) * (1.4 if within else 1.0)

    updated = float(prev) * max(0.0, 1.0 - decay) + gap_pressure * float(fcfg.minutes_base_gain) * gain_mult * inj_mult
    updated = float(clamp01(updated))

    meta = {
        "expected_mpg": expected,
        "actual_mpg": actual,
        "gap": gap,
        "tolerance_mpg": tol,
        "gap_pressure": gap_pressure,
        "gain_mult": float(gain_mult),
        "injury_mult": float(inj_mult),
        "within": bool(within),
    }
    return updated, meta


def _update_team_frustration(
    *,
    prev: float,
    team_win_pct: float,
    mental: Mapping[str, Any],
    leverage: float,
    cfg: AgencyConfig,
) -> Tuple[float, Dict[str, Any]]:
    fcfg = cfg.frustration

    win_pct = clamp(team_win_pct, 0.0, 1.0)

    # "Badness" starts accumulating below team_good_win_pct.
    target = clamp(fcfg.team_good_win_pct, 0.35, 0.75)
    badness = clamp01((target - win_pct) / max(target, 1e-9))

    amb = mental_norm(mental, "ambition")
    loy = mental_norm(mental, "loyalty")
    lev = clamp01(leverage)

    pressure = badness * (0.35 + 0.65 * amb) * (0.40 + 0.60 * lev) * (1.10 - 0.80 * loy)

    # If team is doing well, decay slightly faster.
    decay = float(fcfg.team_decay) * (1.35 if win_pct >= target else 1.0)

    updated = float(prev) * max(0.0, 1.0 - decay) + float(pressure) * float(fcfg.team_base_gain)
    updated = float(clamp01(updated))

    meta = {
        "team_win_pct": float(win_pct),
        "target_win_pct": float(target),
        "badness": float(badness),
        "pressure": float(pressure),
    }
    return updated, meta


def _update_trust(
    *,
    prev: float,
    minutes_frustration: float,
    team_frustration: float,
    mental: Mapping[str, Any],
    cfg: AgencyConfig,
) -> Tuple[float, Dict[str, Any]]:
    fcfg = cfg.frustration

    trust = clamp01(prev)
    fr_avg = clamp01((float(minutes_frustration) + float(team_frustration)) / 2.0)

    coach = mental_norm(mental, "coachability")
    loy = mental_norm(mental, "loyalty")
    ego = mental_norm(mental, "ego")
    amb = mental_norm(mental, "ambition")
    adapt = mental_norm(mental, "adaptability")

    bad_th = clamp(fcfg.trust_bad_frustration_threshold, 0.3, 0.9)

    # Degrade trust when frustration is high.
    if fr_avg > bad_th:
        severity = (fr_avg - bad_th) / max(1.0 - bad_th, 1e-9)
        weight = clamp(0.70 + 0.60 * ego + 0.30 * amb - 0.20 * coach, 0.25, 1.75)
        trust = clamp01(trust - float(fcfg.trust_decay) * float(severity) * float(weight))
        return trust, {"fr_avg": fr_avg, "mode": "decay", "severity": float(severity), "weight": float(weight)}

    # Recover trust slowly when things are calm.
    calm_th = bad_th * 0.55
    if fr_avg < calm_th:
        calmness = (calm_th - fr_avg) / max(calm_th, 1e-9)
        weight = clamp(0.55 + 0.60 * loy + 0.35 * coach + 0.25 * adapt - 0.25 * ego, 0.15, 1.75)
        trust = clamp01(trust + float(fcfg.trust_recovery) * float(calmness) * float(weight))
        return trust, {"fr_avg": fr_avg, "mode": "recover", "calmness": float(calmness), "weight": float(weight)}

    return trust, {"fr_avg": fr_avg, "mode": "stable"}


def _maybe_emit_minutes_complaint(
    *,
    state: Dict[str, Any],
    inputs: MonthlyPlayerInputs,
    cfg: AgencyConfig,
    context: Dict[str, Any],
    sample_weight: float,
) -> Optional[Dict[str, Any]]:
    ecfg = cfg.events

    fr = clamp01(state.get("minutes_frustration"))
    if fr < float(ecfg.minutes_complaint_threshold):
        return None

    now_date = str(inputs.now_date_iso)[:10]
    if _cooldown_active(state.get("cooldown_minutes_until"), now_date_iso=now_date):
        return None

    ego = mental_norm(inputs.mental, "ego")
    lev = clamp01(inputs.leverage)

    # Gate: either meaningful leverage, or very high ego.
    if lev < float(ecfg.minutes_complaint_min_leverage) and ego < float(ecfg.minutes_complaint_ego_override):
        return None

    # Very low-leverage bench/garbage should almost never trigger.
    role = str(inputs.role_bucket or "UNKNOWN")
    low_role = role in {"GARBAGE", "BENCH"}
    if low_role and lev < 0.20 and ego < 0.90:
        return None

    # Small-sample gating: avoid immediate drama after a 1-game cameo on a new team.
    # DNP months have games_played=0 and are handled via sample_weight override in apply_monthly_player_tick.
    gp = int(inputs.games_played or 0)
    try:
        min_g = int(ecfg.min_games_for_events)
    except Exception:
        min_g = 2
    if gp > 0 and gp < max(1, min_g):
        # If the player played *some* minutes in a tiny sample, don't trigger events yet.
        # (Still accumulates frustration; it may trigger next month.)
        if float(inputs.actual_minutes or 0.0) > 0.0:
            return None

    # Probabilistic trigger (stable).
    softness = max(1e-6, float(ecfg.minutes_complaint_softness))
    base_p = clamp01((fr - float(ecfg.minutes_complaint_threshold)) / softness)
    p = base_p * (0.40 + 0.60 * lev) * (0.80 + 0.40 * ego)
    # Scale by sample weight (mid-month attribution confidence proxy)
    p *= clamp(0.35 + 0.65 * float(clamp01(sample_weight)), 0.20, 1.00)

    roll = stable_u01(inputs.player_id, inputs.month_key, "minutes_complaint")
    if roll >= p:
        return None

    # Emit event
    event_type = cfg.event_types.get("minutes_complaint", "MINUTES_COMPLAINT")
    event_id = make_event_id("agency", inputs.player_id, inputs.month_key, event_type)

    severity = clamp01(0.50 * fr + 0.30 * ego + 0.20 * lev)
    severity *= clamp(0.60 + 0.40 * float(clamp01(sample_weight)), 0.60, 1.00)

    payload = {
        "role_bucket": role,
        "expected_mpg": float(inputs.expected_mpg),
        "actual_mpg": float(context.get("minutes", {}).get("actual_mpg") or state.get("minutes_actual_mpg") or 0.0),
        "gap": float(context.get("minutes", {}).get("gap") or 0.0),
        "leverage": float(lev),
        "ego": float(ego),
        "frustration": float(fr),
        "sample_games_played": int(inputs.games_played or 0),
        "sample_weight": float(clamp01(sample_weight)),
    }

    # Cooldown
    state["cooldown_minutes_until"] = date_add_days(now_date, int(ecfg.cooldown_minutes_days))

    return {
        "event_id": event_id,
        "player_id": inputs.player_id,
        "team_id": inputs.team_id,
        "season_year": int(inputs.season_year),
        "date": now_date,
        "event_type": event_type,
        "severity": float(severity),
        "payload": payload,
    }


def _maybe_emit_help_demand(
    *,
    state: Dict[str, Any],
    inputs: MonthlyPlayerInputs,
    cfg: AgencyConfig,
    sample_weight: float,
) -> Optional[Dict[str, Any]]:
    ecfg = cfg.events

    lev = clamp01(inputs.leverage)
    if lev < float(ecfg.help_demand_min_leverage):
        return None

    amb = mental_norm(inputs.mental, "ambition")
    if amb < float(ecfg.help_demand_ambition_threshold):
        return None

    fr_team = clamp01(state.get("team_frustration"))
    if fr_team < float(ecfg.help_demand_team_frustration_threshold):
        return None

    now_date = str(inputs.now_date_iso)[:10]
    if _cooldown_active(state.get("cooldown_help_until"), now_date_iso=now_date):
        return None

    # Small-sample gating (mid-month trade safety): require either DNP month (gp==0)
    # or at least a minimal sample of games played with the evaluated team.
    gp = int(inputs.games_played or 0)
    try:
        min_g = int(ecfg.min_games_for_events)
    except Exception:
        min_g = 2
    if gp > 0 and gp < max(1, min_g):
        return None

    softness = max(1e-6, float(ecfg.help_demand_softness))
    base_p = clamp01((fr_team - float(ecfg.help_demand_team_frustration_threshold)) / softness)
    p = base_p * (0.55 + 0.45 * amb) * (0.50 + 0.50 * lev)
    p *= clamp(0.35 + 0.65 * float(clamp01(sample_weight)), 0.20, 1.00)

    roll = stable_u01(inputs.player_id, inputs.month_key, "help_demand")
    if roll >= p:
        return None

    event_type = cfg.event_types.get("help_demand", "HELP_DEMAND")
    event_id = make_event_id("agency", inputs.player_id, inputs.month_key, event_type)

    severity = clamp01(0.55 * fr_team + 0.25 * amb + 0.20 * lev)
    severity *= clamp(0.60 + 0.40 * float(clamp01(sample_weight)), 0.60, 1.00)

    payload = {
        "role_bucket": str(inputs.role_bucket or "UNKNOWN"),
        "team_win_pct": float(inputs.team_win_pct),
        "leverage": float(lev),
        "ambition": float(amb),
        "team_frustration": float(fr_team),
        "sample_games_played": int(inputs.games_played or 0),
        "sample_weight": float(clamp01(sample_weight)),
    }

    state["cooldown_help_until"] = date_add_days(now_date, int(ecfg.cooldown_help_days))

    return {
        "event_id": event_id,
        "player_id": inputs.player_id,
        "team_id": inputs.team_id,
        "season_year": int(inputs.season_year),
        "date": now_date,
        "event_type": event_type,
        "severity": float(severity),
        "payload": payload,
    }


def _maybe_emit_trade_request(
    *,
    state: Dict[str, Any],
    inputs: MonthlyPlayerInputs,
    cfg: AgencyConfig,
    sample_weight: float,
) -> Optional[Dict[str, Any]]:
    ecfg = cfg.events

    now_date = str(inputs.now_date_iso)[:10]
    if _cooldown_active(state.get("cooldown_trade_until"), now_date_iso=now_date):
        return None

    lev = clamp01(inputs.leverage)
    # Very low leverage: never request trade (can still complain).
    if lev < 0.30:
        return None

    # Small-sample gating: avoid a trade request after a 1-game cameo on a new team.
    gp = int(inputs.games_played or 0)
    try:
        min_g = int(ecfg.min_games_for_events)
    except Exception:
        min_g = 2
    if gp > 0 and gp < max(1, min_g):
        return None

    ego = mental_norm(inputs.mental, "ego")
    amb = mental_norm(inputs.mental, "ambition")
    loy = mental_norm(inputs.mental, "loyalty")

    fr_m = clamp01(state.get("minutes_frustration"))
    fr_t = clamp01(state.get("team_frustration"))

    request_score = (0.45 * fr_m + 0.45 * fr_t + 0.10 * ego) * (0.40 + 0.60 * lev)

    base = float(ecfg.trade_request_threshold_base)
    threshold = base + float(ecfg.trade_request_threshold_loyalty_bonus) * loy + float(ecfg.trade_request_threshold_ambition_bonus) * amb

    # Trust can delay a request slightly.
    trust = clamp01(state.get("trust"))
    threshold += 0.05 * (trust - 0.5)

    softness = max(1e-6, float(ecfg.trade_request_softness))
    p = clamp01((request_score - threshold) / softness)

    # High ego/ambition increases chance of pulling the trigger.
    p *= clamp(0.75 + 0.55 * ego + 0.35 * amb - 0.10 * loy, 0.10, 2.00)
    p *= clamp(0.35 + 0.65 * float(clamp01(sample_weight)), 0.20, 1.00)
    p = clamp01(p)

    roll = stable_u01(inputs.player_id, inputs.month_key, "trade_request", int(state.get("trade_request_level") or 0))
    if roll >= p:
        return None

    prev_level = int(state.get("trade_request_level") or 0)
    new_level = 1 if prev_level <= 0 else prev_level

    # Escalate to public if already private and pressure is significantly above threshold.
    event_key = "trade_request"
    event_type = cfg.event_types.get("trade_request", "TRADE_REQUEST")
    if prev_level == 1 and (request_score - threshold) >= float(ecfg.trade_request_public_escalate_delta):
        event_key = "trade_request_public"
        event_type = cfg.event_types.get("trade_request_public", "TRADE_REQUEST_PUBLIC")
        new_level = 2

    state["trade_request_level"] = int(new_level)
    state["cooldown_trade_until"] = date_add_days(now_date, int(ecfg.cooldown_trade_days))

    event_id = make_event_id("agency", inputs.player_id, inputs.month_key, event_type)

    severity = clamp01(0.55 * request_score + 0.20 * ego + 0.25 * lev)
    severity *= clamp(0.60 + 0.40 * float(clamp01(sample_weight)), 0.60, 1.00)

    payload = {
        "role_bucket": str(inputs.role_bucket or "UNKNOWN"),
        "leverage": float(lev),
        "trust": float(trust),
        "minutes_frustration": float(fr_m),
        "team_frustration": float(fr_t),
        "request_score": float(request_score),
        "threshold": float(threshold),
        "ego": float(ego),
        "ambition": float(amb),
        "loyalty": float(loy),
        "public": bool(new_level >= 2),
        "level": int(new_level),
        "sample_games_played": int(inputs.games_played or 0),
        "sample_weight": float(clamp01(sample_weight)),
    }

    return {
        "event_id": event_id,
        "player_id": inputs.player_id,
        "team_id": inputs.team_id,
        "season_year": int(inputs.season_year),
        "date": now_date,
        "event_type": event_type,
        "severity": float(severity),
        "payload": payload,
    }


def apply_monthly_player_tick(
    prev_state: Optional[Mapping[str, Any]],
    *,
    inputs: MonthlyPlayerInputs,
    cfg: AgencyConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Apply one player's monthly tick.

    Args:
        prev_state: existing SSOT state dict (or None for new).
        inputs: MonthlyPlayerInputs
        cfg: AgencyConfig

    Returns:
        (new_state_dict, events)

    The returned state dict matches columns in player_agency_state, but uses
    Python-native types:
      - context: dict
    """

    # Initialize state with safe defaults.
    st: Dict[str, Any] = {
        "player_id": str(inputs.player_id),
        "team_id": str(inputs.team_id).upper(),
        "season_year": int(inputs.season_year),
        "role_bucket": str(inputs.role_bucket or "UNKNOWN"),
        "leverage": float(clamp01(inputs.leverage)),
        "minutes_expected_mpg": float(max(0.0, inputs.expected_mpg)),
        "minutes_actual_mpg": 0.0,
        "minutes_frustration": 0.0,
        "team_frustration": 0.0,
        "trust": 0.5,
        "trade_request_level": 0,
        "cooldown_minutes_until": None,
        "cooldown_trade_until": None,
        "cooldown_help_until": None,
        "cooldown_contract_until": None,
        "last_processed_month": str(inputs.month_key),
        "context": {},
    }

    if prev_state:
        # copy known fields (defensive)
        for k in list(st.keys()):
            if k in prev_state and prev_state.get(k) is not None:
                st[k] = prev_state.get(k)

        # Ensure identity fields reflect current roster.
        st["player_id"] = str(inputs.player_id)
        st["team_id"] = str(inputs.team_id).upper()
        st["season_year"] = int(inputs.season_year)

    # Expectations (current month)
    st["role_bucket"] = str(inputs.role_bucket or st.get("role_bucket") or "UNKNOWN")
    st["leverage"] = float(clamp01(inputs.leverage))
    st["minutes_expected_mpg"] = float(max(0.0, inputs.expected_mpg))

    # Actuals
    gp = int(inputs.games_played or 0)
    mins = float(max(0.0, inputs.actual_minutes))
    actual_mpg = mins / float(gp) if gp > 0 else 0.0
    st["minutes_actual_mpg"] = float(actual_mpg)

    # Sample weight (mid-month trade attribution safety)
    # - If gp > 0: scale with games played on the evaluated team.
    # - If gp == 0 and the player expected minutes but got 0, treat as a full DNP sample.
    try:
        full = int(cfg.month_context.full_weight_games)
    except Exception:
        full = 10
    full = max(1, full)
    if gp > 0:
        sample_weight = float(clamp01(gp / float(full)))
    else:
        exp_mpg = float(st.get("minutes_expected_mpg") or 0.0)
        # DNP month: allow complaints/requests to trigger; injury multiplier controls accumulation.
        sample_weight = 1.0 if mins <= 0.0 and exp_mpg > 0.0 else 0.0

    # Update frustrations and trust.
    context: Dict[str, Any] = {}

    new_m_fr, meta_m = _update_minutes_frustration(
        prev=float(st.get("minutes_frustration") or 0.0),
        expected_mpg=float(st.get("minutes_expected_mpg") or 0.0),
        actual_mpg=float(actual_mpg),
        mental=inputs.mental,
        leverage=float(st.get("leverage") or 0.0),
        injury_status=inputs.injury_status,
        cfg=cfg,
    )
    st["minutes_frustration"] = float(new_m_fr)

    new_t_fr, meta_t = _update_team_frustration(
        prev=float(st.get("team_frustration") or 0.0),
        team_win_pct=float(inputs.team_win_pct),
        mental=inputs.mental,
        leverage=float(st.get("leverage") or 0.0),
        cfg=cfg,
    )
    st["team_frustration"] = float(new_t_fr)

    new_trust, meta_trust = _update_trust(
        prev=float(st.get("trust") or 0.5),
        minutes_frustration=float(st.get("minutes_frustration") or 0.0),
        team_frustration=float(st.get("team_frustration") or 0.0),
        mental=inputs.mental,
        cfg=cfg,
    )
    st["trust"] = float(new_trust)

    context["minutes"] = meta_m
    context["team"] = meta_t
    context["trust"] = meta_trust

    # Keep some player context for debugging/telemetry.
    context["player"] = {
        "ovr": inputs.ovr,
        "age": inputs.age,
        "role_bucket": str(inputs.role_bucket or "UNKNOWN"),
        "leverage": float(clamp01(inputs.leverage)),
        "games_played": int(gp),
        "month_key": str(inputs.month_key),
        "injury_status": str(inputs.injury_status or ""),
    }

    context["sample"] = {
        "games_played": int(gp),
        "full_weight_games": int(full),
        "sample_weight": float(sample_weight),
    }

    st["context"] = context
    st["last_processed_month"] = str(inputs.month_key)

    # Events
    events: List[Dict[str, Any]] = []

    ev = _maybe_emit_minutes_complaint(state=st, inputs=inputs, cfg=cfg, context=context, sample_weight=sample_weight)
    if ev is not None:
        events.append(ev)

    ev = _maybe_emit_help_demand(state=st, inputs=inputs, cfg=cfg, sample_weight=sample_weight)
    if ev is not None:
        events.append(ev)

    ev = _maybe_emit_trade_request(state=st, inputs=inputs, cfg=cfg, sample_weight=sample_weight)
    if ev is not None:
        events.append(ev)

    return st, events


