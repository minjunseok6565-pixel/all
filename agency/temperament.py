from __future__ import annotations

"""Temperament modeling for the agency subsystem.

This module converts a player's mental traits (M_*) into a compact set of
behavioral parameters used across the agency system:

- Self-expectation drift (how quickly the player adjusts their own standards)
- Negotiation behavior (patience, concession)
- Escalation style (publicness, insult sensitivity)
- Relationship repair (forgiveness)

Design principles
-----------------
- Pure business logic: no DB I/O.
- Deterministic: no randomness.
- Mental traits are modulators, not absolute rules.

Notes
-----
- Inputs are expected to be normalized with mental_norm() (0..1), but this
  module tolerates raw 0..100 ints as well.
- WorkEthic is intentionally used here to differentiate "hard-working" players
  from merely "high-ego" players.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from .utils import clamp, clamp01, mental_norm


@dataclass(frozen=True, slots=True)
class Temperament:
    """A compact set of behavioral parameters derived from mental traits.

    All fields are in [0..1] unless stated otherwise.

    Fields
    ------
    entitlement:
        Tendency to inflate self-worth / desired status.
    adaptation:
        Speed of adjusting self-expectations to reality.
    patience:
        How long the player is willing to negotiate or tolerate short-term issues.
    concession:
        How much the player yields on each negotiation counter (0.12..0.82).
    publicness:
        Likelihood to escalate to agent/public steps.
    insult_sensitivity:
        Likelihood to treat a low offer as insulting (lowball strike).
    forgiveness:
        How quickly relationship damage recovers after fulfilled promises.
    """

    entitlement: float
    adaptation: float
    patience: float
    concession: float
    publicness: float
    insult_sensitivity: float
    forgiveness: float

    def to_dict(self) -> dict:
        return {
            "entitlement": float(self.entitlement),
            "adaptation": float(self.adaptation),
            "patience": float(self.patience),
            "concession": float(self.concession),
            "publicness": float(self.publicness),
            "insult_sensitivity": float(self.insult_sensitivity),
            "forgiveness": float(self.forgiveness),
        }


def compute_temperament(
    *,
    mental: Mapping[str, Any],
    leverage: float,
    trust: float,
    credibility_damage: float,
) -> Temperament:
    """Compute temperament parameters from mental traits + context.

    Args:
        mental: mapping containing logical mental keys (0..100 or 0..1)
        leverage: player's leverage (0..1)
        trust: relationship trust (0..1)
        credibility_damage: accumulated promise credibility damage (0..1)

    Returns:
        Temperament

    Implementation
    --------------
    The formulae here are intentionally simple and tunable. They are chosen to:
    - Make ego/ambition drive entitlement and publicness
    - Make coachability/adaptability/work_ethic drive patience and adaptation
    - Make broken-credibility reduce patience and increase publicness
    """

    lev = float(clamp01(leverage))
    tr = float(clamp01(trust))
    cd = float(clamp01(credibility_damage))

    ego = mental_norm(mental, "ego")
    amb = mental_norm(mental, "ambition")
    loy = mental_norm(mental, "loyalty")
    coach = mental_norm(mental, "coachability")
    adapt = mental_norm(mental, "adaptability")
    work = mental_norm(mental, "work_ethic")

    # NOTE: These are the "v1" formulas from the design spec.
    entitlement = clamp01(
        0.50
        + 0.70 * (ego - 0.5)
        + 0.55 * (amb - 0.5)
        - 0.40 * (coach - 0.5)
        - 0.30 * (loy - 0.5)
        - 0.25 * (work - 0.5)
        - 0.20 * (adapt - 0.5)
    )

    adaptation = clamp01(0.20 + 0.55 * coach + 0.45 * adapt + 0.25 * work - 0.50 * ego - 0.20 * amb)

    patience = clamp01(
        0.45
        + 0.35 * coach
        + 0.25 * loy
        + 0.25 * adapt
        + 0.15 * work
        - 0.45 * ego
        - 0.20 * amb
        - 0.25 * cd
        + 0.10 * tr
    )

    concession = float(
        clamp(
            0.45 + 0.35 * coach + 0.35 * adapt + 0.15 * loy + 0.10 * work - 0.55 * ego - 0.25 * amb,
            0.12,
            0.82,
        )
    )

    publicness = clamp01(
        0.25
        + 0.55 * ego
        + 0.35 * amb
        + 0.20 * lev
        + 0.20 * cd
        - 0.35 * coach
        - 0.35 * loy
        - 0.15 * adapt
    )

    insult_sensitivity = clamp01(
        0.35 + 0.65 * ego + 0.20 * lev + 0.25 * cd - 0.25 * coach - 0.20 * loy
    )

    forgiveness = clamp01(
        0.40 + 0.45 * loy + 0.35 * coach + 0.25 * adapt + 0.20 * work - 0.55 * ego - 0.20 * amb
    )

    return Temperament(
        entitlement=float(entitlement),
        adaptation=float(adaptation),
        patience=float(patience),
        concession=float(concession),
        publicness=float(publicness),
        insult_sensitivity=float(insult_sensitivity),
        forgiveness=float(forgiveness),
    )


def entitlement_bias(temp: Temperament) -> float:
    """Convert entitlement (0..1) into a symmetric bias in [-1..+1]."""

    return float(clamp((float(temp.entitlement) - 0.5) * 2.0, -1.0, 1.0))


def compute_minutes_tolerance_mpg(*, mental: Mapping[str, Any], cfg: Any) -> float:
    """Compute the tolerated MPG gap before frustration/negotiation intensifies.

    This mirrors the existing tick.py tolerance formula but lives in a shared
    module so negotiation and tick can reuse it.

    Args:
        mental: mental traits mapping
        cfg: AgencyConfig-like object with cfg.frustration tolerance fields

    Returns:
        tolerance minutes in [tolerance_min_mpg, tolerance_max_mpg]
    """

    fcfg = getattr(cfg, "frustration", None)
    if fcfg is None:
        # Safe fallback.
        base = 4.0
        return float(clamp(base, 1.0, 12.0))

    coach = mental_norm(mental, "coachability")
    loy = mental_norm(mental, "loyalty")
    adapt = mental_norm(mental, "adaptability")
    ego = mental_norm(mental, "ego")
    amb = mental_norm(mental, "ambition")

    tol = (
        float(getattr(fcfg, "tolerance_base_mpg", 4.0))
        + float(getattr(fcfg, "tolerance_coachability_bonus", 4.0)) * coach
        + float(getattr(fcfg, "tolerance_loyalty_bonus", 2.0)) * loy
        + float(getattr(fcfg, "tolerance_adaptability_bonus", 2.0)) * adapt
        - float(getattr(fcfg, "tolerance_ego_penalty", 4.0)) * ego
        - float(getattr(fcfg, "tolerance_ambition_penalty", 2.0)) * amb
    )

    return float(
        clamp(
            tol,
            float(getattr(fcfg, "tolerance_min_mpg", 1.0)),
            float(getattr(fcfg, "tolerance_max_mpg", 12.0)),
        )
    )
