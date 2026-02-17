from __future__ import annotations

"""Configuration for the player agency subsystem.

All numbers here are intended to be *tunable* without rewriting logic.

Important:
- These defaults are conservative to avoid "everyone is angry" syndrome.
- For commercial quality, treat these as initial values; then tune with
  playtest telemetry.
"""

from dataclasses import dataclass, field
from typing import Dict, Mapping


ROLE_BUCKETS: tuple[str, ...] = (
    "UNKNOWN",
    "FRANCHISE",
    "STAR",
    "STARTER",
    "ROTATION",
    "BENCH",
    "GARBAGE",
)


# ---------------------------------------------------------------------------
# Expectations / leverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExpectationsConfig:
    """How we compute role bucket / leverage / expected minutes."""

    expected_mpg_by_role: Mapping[str, float] = field(
        default_factory=lambda: {
            "FRANCHISE": 36.0,
            "STAR": 34.0,
            "STARTER": 30.0,
            "ROTATION": 24.0,
            "BENCH": 16.0,
            "GARBAGE": 6.0,
            "UNKNOWN": 12.0,
        }
    )

    # Leverage = w_ovr * ovr_rank_score + w_salary * salary_score
    leverage_weight_ovr: float = 0.75
    leverage_weight_salary: float = 0.25

    # If a player takes ~20% of team payroll, salary_score ~ 1.0
    salary_share_star: float = 0.20


# ---------------------------------------------------------------------------
# Frustration update (monthly EMA)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrustrationConfig:
    """Monthly update parameters for frustration and trust."""

    # minutes frustration
    minutes_base_gain: float = 0.55
    minutes_decay: float = 0.15

    # team frustration
    team_base_gain: float = 0.35
    team_decay: float = 0.12

    # Trust updates (simple v1)
    trust_decay: float = 0.05
    trust_recovery: float = 0.03
    trust_bad_frustration_threshold: float = 0.60

    # Expected tolerance window (in minutes) depends on mental traits
    tolerance_base_mpg: float = 4.0
    tolerance_coachability_bonus: float = 4.0
    tolerance_loyalty_bonus: float = 2.0
    tolerance_adaptability_bonus: float = 2.0
    tolerance_ego_penalty: float = 4.0
    tolerance_ambition_penalty: float = 2.0
    tolerance_min_mpg: float = 1.0
    tolerance_max_mpg: float = 12.0

    # Injury multipliers for minutes frustration accumulation
    injury_out_multiplier: float = 0.05
    injury_returning_multiplier: float = 0.40

    # Team win% target (below this contributes to "badness")
    team_good_win_pct: float = 0.55


# ---------------------------------------------------------------------------
# Event thresholds / cooldowns
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventConfig:
    """When to emit events and how long to cool down."""

    # Minutes complaint
    minutes_complaint_threshold: float = 0.60
    minutes_complaint_min_leverage: float = 0.35
    minutes_complaint_ego_override: float = 0.75
    minutes_complaint_softness: float = 0.12  # how "probabilistic" the trigger is
    cooldown_minutes_days: int = 28

    # Help demand ("get help" roster request)
    help_demand_min_leverage: float = 0.70
    help_demand_ambition_threshold: float = 0.65
    help_demand_team_frustration_threshold: float = 0.55
    help_demand_softness: float = 0.15
    cooldown_help_days: int = 60

    # Trade request
    trade_request_softness: float = 0.10
    trade_request_threshold_base: float = 0.82
    trade_request_threshold_loyalty_bonus: float = 0.12
    trade_request_threshold_ambition_bonus: float = -0.08

    trade_request_public_escalate_delta: float = 0.08  # additional score needed to go public

    cooldown_trade_days: int = 90


# ---------------------------------------------------------------------------
# Player option / ETO decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OptionsConfig:
    """Decision logic settings for PLAYER option and ETO."""

    # Expected salary curve (sigmoid mapping)
    expected_salary_ovr_center: float = 75.0
    expected_salary_ovr_scale: float = 7.0
    expected_salary_midpoint: float = 18_000_000.0
    expected_salary_span: float = 16_000_000.0

    # Hard edges for deterministic decisions
    hard_exercise_ratio: float = 1.10  # option >= 110% market => exercise
    hard_decline_ratio: float = 0.90  # option <= 90% market => decline

    # Probabilistic zone (between hard edges)
    ambiguous_value_center: float = 0.98  # slight bias towards exercising when close

    # Logit weights (tunable)
    w_value: float = 6.0
    w_ambition: float = 1.2
    w_loyalty: float = -0.8
    w_ego: float = 0.3
    w_age: float = -0.6
    w_injury_risk: float = -0.8


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgencyConfig:
    expectations: ExpectationsConfig = field(default_factory=ExpectationsConfig)
    frustration: FrustrationConfig = field(default_factory=FrustrationConfig)
    events: EventConfig = field(default_factory=EventConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)

    # Names for mental attributes in attrs_json
    mental_attr_keys: Mapping[str, str] = field(
        default_factory=lambda: {
            "work_ethic": "M_WorkEthic",
            "coachability": "M_Coachability",
            "ambition": "M_Ambition",
            "loyalty": "M_Loyalty",
            "ego": "M_Ego",
            "adaptability": "M_Adaptability",
        }
    )

    # Event type strings (UI + analytics)
    event_types: Dict[str, str] = field(
        default_factory=lambda: {
            "minutes_complaint": "MINUTES_COMPLAINT",
            "help_demand": "HELP_DEMAND",
            "trade_request": "TRADE_REQUEST",
            "trade_request_public": "TRADE_REQUEST_PUBLIC",
        }
    )


DEFAULT_CONFIG = AgencyConfig()
