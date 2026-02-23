from __future__ import annotations

from typing import Any, Dict

from agency.utils import clamp01, mental_norm, sigmoid, stable_u01

from .config import DEFAULT_RETIREMENT_CONFIG, RetirementConfig
from .types import RetirementDecision, RetirementInputs


def _age_factor(age: int) -> float:
    return clamp01((float(age) - 28.0) / 12.0)


def _injury_factor(status: str, severity: int) -> float:
    st = str(status or "HEALTHY").upper()
    base = 0.0
    if st == "RETURNING":
        base = 0.35
    elif st == "OUT":
        base = 0.65
    sev_bonus = clamp01(float(max(0, int(severity))) / 20.0) * 0.35
    return clamp01(base + sev_bonus)


def evaluate_retirement_candidate(
    inp: RetirementInputs,
    *,
    cfg: RetirementConfig = DEFAULT_RETIREMENT_CONFIG,
) -> RetirementDecision:
    age_f = _age_factor(int(inp.age))
    inj_f = _injury_factor(inp.injury_status, int(inp.injury_severity))
    teamless_f = 1.0 if str(inp.team_id).upper() == "FA" else 0.0

    work = mental_norm(inp.mental, "work_ethic")
    coach = mental_norm(inp.mental, "coachability")
    amb = mental_norm(inp.mental, "ambition")
    loy = mental_norm(inp.mental, "loyalty")
    ego = mental_norm(inp.mental, "ego")
    adapt = mental_norm(inp.mental, "adaptability")

    consider_z = (
        float(cfg.consider_bias)
        + float(cfg.consider_w_age) * age_f
        + float(cfg.consider_w_injury) * inj_f
        + float(cfg.consider_w_teamless) * teamless_f
        + float(cfg.consider_w_ambition) * amb
        + float(cfg.consider_w_work_ethic) * work
        + float(cfg.consider_w_adaptability) * adapt
    )
    consider_prob = clamp01(sigmoid(float(consider_z)))
    consider_roll = stable_u01("retire.consider", inp.player_id, int(inp.season_year))
    considered = bool(consider_roll < consider_prob)

    decision_z = (
        float(cfg.decision_bias)
        + float(cfg.decision_w_age) * age_f
        + float(cfg.decision_w_injury) * inj_f
        + float(cfg.decision_w_teamless) * teamless_f
        + float(cfg.decision_w_loyalty) * loy
        + float(cfg.decision_w_ego) * ego
        + float(cfg.decision_w_ambition) * amb
        + float(cfg.decision_w_work_ethic) * work
        + float(cfg.decision_w_adaptability) * adapt
        + float(cfg.decision_w_coachability) * coach
        + float(cfg.interaction_age_injury) * (age_f * inj_f)
        + float(cfg.interaction_teamless_loyalty) * (teamless_f * loy)
        + float(cfg.interaction_ambition_adaptability) * (amb * adapt)
    )
    if int(inp.ovr) >= int(cfg.elite_ovr_guard):
        decision_z -= float(cfg.elite_ovr_z_penalty)

    retirement_prob = clamp01(sigmoid(float(decision_z)))
    if int(inp.age) <= int(cfg.youth_age_guard):
        retirement_prob = min(float(retirement_prob), float(cfg.youth_prob_cap))

    if not considered:
        retirement_prob = 0.0

    final_roll = stable_u01("retire.final", inp.player_id, int(inp.season_year))
    retired = bool(considered and final_roll < retirement_prob)

    explanation: Dict[str, Any] = {
        "age_factor": float(age_f),
        "injury_factor": float(inj_f),
        "teamless_factor": float(teamless_f),
        "mental": {
            "work_ethic": float(work),
            "coachability": float(coach),
            "ambition": float(amb),
            "loyalty": float(loy),
            "ego": float(ego),
            "adaptability": float(adapt),
        },
        "consider_z": float(consider_z),
        "decision_z": float(decision_z),
    }
    inputs_json: Dict[str, Any] = {
        "age": int(inp.age),
        "ovr": int(inp.ovr),
        "team_id": str(inp.team_id),
        "injury_status": str(inp.injury_status),
        "injury_severity": int(inp.injury_severity),
        "mental": dict(inp.mental or {}),
    }

    return RetirementDecision(
        player_id=str(inp.player_id),
        season_year=int(inp.season_year),
        considered=bool(considered),
        decision="RETIRED" if retired else "STAY",
        consider_prob=float(consider_prob),
        retirement_prob=float(retirement_prob),
        random_roll=float(final_roll),
        age=int(inp.age),
        team_id=str(inp.team_id),
        injury_status=str(inp.injury_status),
        inputs=inputs_json,
        explanation=explanation,
    )
