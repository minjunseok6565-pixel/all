from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetirementConfig:
    # Consideration stage
    consider_bias: float = -2.8
    consider_w_age: float = 2.2
    consider_w_injury: float = 1.5
    consider_w_teamless: float = 1.1
    consider_w_ambition: float = -0.9
    consider_w_work_ethic: float = -0.8
    consider_w_adaptability: float = -0.4

    # Final decision stage
    decision_bias: float = -1.4
    decision_w_age: float = 2.6
    decision_w_injury: float = 1.8
    decision_w_teamless: float = 1.4
    decision_w_loyalty: float = 0.3
    decision_w_ego: float = 0.5
    decision_w_ambition: float = -1.1
    decision_w_work_ethic: float = -0.9
    decision_w_adaptability: float = -0.7
    decision_w_coachability: float = -0.4

    # Interactions
    interaction_age_injury: float = 0.7
    interaction_teamless_loyalty: float = 0.4
    interaction_ambition_adaptability: float = -0.5

    # Guards
    youth_age_guard: int = 31
    youth_prob_cap: float = 0.08
    elite_ovr_guard: int = 88
    elite_ovr_z_penalty: float = 0.35


DEFAULT_RETIREMENT_CONFIG = RetirementConfig()
