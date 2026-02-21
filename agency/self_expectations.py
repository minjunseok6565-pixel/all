from __future__ import annotations

"""Self-expectation modeling for the agency subsystem.

A player's dissatisfaction is driven by *their own* standards, not only by
team-computed expectations.

This module maintains three primary self-standards:
- self_expected_mpg
- self_expected_starts_rate
- self_expected_closes_rate

These values:
- are persisted in player_agency_state (SSOT)
- update monthly via a smooth, explainable rule
- are influenced by both team expectations and reality, modulated by temperament

Design principles
-----------------
- Pure business logic: no DB I/O.
- Explainable: return meta with intermediate values.
- Robust: tolerates missing fields and can run even before full config wiring.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .metrics import role_expected_rates
from .temperament import Temperament, compute_temperament, entitlement_bias
from .utils import clamp, clamp01, safe_float


@dataclass(frozen=True, slots=True)
class SelfExpectationUpdate:
    """Updated self-expectations for a player (absolute values, not deltas)."""

    self_expected_mpg: float
    self_expected_starts_rate: float
    self_expected_closes_rate: float

    meta: Dict[str, Any]


# ---------------------------------------------------------------------------
# Fallback tuning
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SelfExpectationsTuning:
    """Internal fallback tuning.

    The project later wires these into AgencyConfig.self_expectations.
    """

    bias_mpg_max: float = 6.0
    bias_starts_rate_max: float = 0.35
    bias_closes_rate_max: float = 0.30

    adapt_rate_min: float = 0.10
    adapt_rate_max: float = 0.35

    performance_weight_mpg: float = 0.25
    performance_weight_status: float = 0.35

    mpg_min: float = 0.0
    mpg_max: float = 40.0


def _get_tuning(cfg: Any) -> _SelfExpectationsTuning:
    se = getattr(cfg, "self_expectations", None)
    if se is None:
        return _SelfExpectationsTuning()

    # Allow partial configs; fall back field-by-field.
    d = _SelfExpectationsTuning()
    return _SelfExpectationsTuning(
        bias_mpg_max=float(getattr(se, "bias_mpg_max", d.bias_mpg_max)),
        bias_starts_rate_max=float(getattr(se, "bias_starts_rate_max", d.bias_starts_rate_max)),
        bias_closes_rate_max=float(getattr(se, "bias_closes_rate_max", d.bias_closes_rate_max)),
        adapt_rate_min=float(getattr(se, "adapt_rate_min", d.adapt_rate_min)),
        adapt_rate_max=float(getattr(se, "adapt_rate_max", d.adapt_rate_max)),
        performance_weight_mpg=float(getattr(se, "performance_weight_mpg", d.performance_weight_mpg)),
        performance_weight_status=float(getattr(se, "performance_weight_status", d.performance_weight_status)),
        mpg_min=float(getattr(se, "mpg_min", d.mpg_min)),
        mpg_max=float(getattr(se, "mpg_max", d.mpg_max)),
    )


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Get key from mapping or attribute."""

    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _compute_actual_mpg(*, inputs: Any) -> Optional[float]:
    gp = int(_get_attr(inputs, "games_played", 0) or 0)
    if gp <= 0:
        return None
    mins = float(safe_float(_get_attr(inputs, "actual_minutes", 0.0), 0.0))
    return float(max(0.0, mins / float(gp)))


def _compute_rate(*, numerator: Any, games_played: int) -> float:
    gp = max(0, int(games_played or 0))
    if gp <= 0:
        return 0.0
    return float(clamp01(float(safe_float(numerator, 0.0)) / float(gp)))


def update_self_expectations(
    *,
    state: Mapping[str, Any],
    inputs: Any,
    mental: Mapping[str, Any],
    cfg: Any,
) -> SelfExpectationUpdate:
    """Update player's self-expectations for the processed month.

    Args:
        state: current player_agency_state mapping (SSOT row as dict)
        inputs: MonthlyPlayerInputs (or mapping with the same keys)
        mental: normalized-ish mental mapping (0..100 or 0..1)
        cfg: AgencyConfig-like

    Returns:
        SelfExpectationUpdate with new values and explainable meta.

    Key idea
    --------
    - Team expectations provide a baseline.
    - The player's temperament adds entitlement bias and determines adaptation speed.
    - Actual usage nudges the target, but cannot fully override temperament.
    """

    t = _get_tuning(cfg)

    # Inputs
    team_expected_mpg = float(max(0.0, safe_float(_get_attr(inputs, "expected_mpg", 0.0), 0.0)))

    gp = int(_get_attr(inputs, "games_played", 0) or 0)

    # Prefer the service-layer computed rates when available.
    starts_rate_in = _get_attr(inputs, "starts_rate", None)
    closes_rate_in = _get_attr(inputs, "closes_rate", None)
    if starts_rate_in is None:
        starts = _get_attr(inputs, "starts", 0)
        starts_rate = _compute_rate(numerator=starts, games_played=gp)
    else:
        starts_rate = float(clamp01(safe_float(starts_rate_in, 0.0)))

    if closes_rate_in is None:
        closes = _get_attr(inputs, "closes", 0)
        closes_rate = _compute_rate(numerator=closes, games_played=gp)
    else:
        closes_rate = float(clamp01(safe_float(closes_rate_in, 0.0)))

    role_bucket = str(_get_attr(inputs, "role_bucket", state.get("role_bucket") or "UNKNOWN") or "UNKNOWN").upper()

    # Expectations for this role bucket (team baseline)
    exp_s_team, exp_c_team = role_expected_rates(role_bucket, cfg=cfg)

    # Context
    trust = float(clamp01(safe_float(state.get("trust", 0.5), 0.5)))
    leverage = float(clamp01(safe_float(state.get("leverage", _get_attr(inputs, "leverage", 0.0)), 0.0)))
    cred_damage = float(clamp01(safe_float(state.get("credibility_damage", 0.0), 0.0)))

    temp: Temperament = compute_temperament(mental=mental, leverage=leverage, trust=trust, credibility_damage=cred_damage)
    ent = entitlement_bias(temp)  # [-1..+1]

    # -----------------------------
    # MPG self expectation
    # -----------------------------

    base_mpg = team_expected_mpg + ent * float(t.bias_mpg_max)
    base_mpg = float(clamp(base_mpg, float(t.mpg_min), float(t.mpg_max)))

    actual_mpg = _compute_actual_mpg(inputs=inputs)

    perf_term = 0.0
    if actual_mpg is not None:
        perf_term = float(t.performance_weight_mpg) * float(actual_mpg - team_expected_mpg)

    target_mpg = float(clamp(base_mpg + perf_term, float(t.mpg_min), float(t.mpg_max)))

    lr = float(clamp(float(t.adapt_rate_min) + float(temp.adaptation) * (float(t.adapt_rate_max) - float(t.adapt_rate_min)), 0.0, 1.0))

    prev_mpg_raw = safe_float(state.get("self_expected_mpg", 0.0), 0.0)
    prev_mpg = float(prev_mpg_raw) if prev_mpg_raw > 0.0 else float(base_mpg)

    new_mpg = float(clamp(prev_mpg + lr * (target_mpg - prev_mpg), float(t.mpg_min), float(t.mpg_max)))

    # -----------------------------
    # Starts / closes self expectation
    # -----------------------------

    base_s = float(clamp01(exp_s_team + ent * float(t.bias_starts_rate_max)))
    base_c = float(clamp01(exp_c_team + ent * float(t.bias_closes_rate_max)))

    target_s = float(clamp01(base_s + float(t.performance_weight_status) * (starts_rate - exp_s_team)))
    target_c = float(clamp01(base_c + float(t.performance_weight_status) * (closes_rate - exp_c_team)))

    prev_s_raw = safe_float(state.get("self_expected_starts_rate", 0.0), 0.0)
    prev_c_raw = safe_float(state.get("self_expected_closes_rate", 0.0), 0.0)

    prev_s = float(prev_s_raw) if prev_s_raw > 0.0 else float(base_s)
    prev_c = float(prev_c_raw) if prev_c_raw > 0.0 else float(base_c)

    new_s = float(clamp01(prev_s + lr * (target_s - prev_s)))
    new_c = float(clamp01(prev_c + lr * (target_c - prev_c)))

    meta: Dict[str, Any] = {
        "team_expected": {
            "mpg": float(team_expected_mpg),
            "starts_rate": float(exp_s_team),
            "closes_rate": float(exp_c_team),
            "role_bucket": role_bucket,
        },
        "actual": {
            "mpg": None if actual_mpg is None else float(actual_mpg),
            "starts_rate": float(starts_rate),
            "closes_rate": float(closes_rate),
            "games_played": int(gp),
        },
        "self": {
            "prev_mpg": float(prev_mpg),
            "prev_starts_rate": float(prev_s),
            "prev_closes_rate": float(prev_c),
            "new_mpg": float(new_mpg),
            "new_starts_rate": float(new_s),
            "new_closes_rate": float(new_c),
        },
        "model": {
            "base_mpg": float(base_mpg),
            "target_mpg": float(target_mpg),
            "perf_term_mpg": float(perf_term),
            "base_starts_rate": float(base_s),
            "base_closes_rate": float(base_c),
            "target_starts_rate": float(target_s),
            "target_closes_rate": float(target_c),
            "lr": float(lr),
            "entitlement_bias": float(ent),
            "temperament": temp.to_dict(),
        },
    }

    return SelfExpectationUpdate(
        self_expected_mpg=float(new_mpg),
        self_expected_starts_rate=float(new_s),
        self_expected_closes_rate=float(new_c),
        meta=meta,
    )
