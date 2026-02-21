from __future__ import annotations

"""Promise credibility (trustworthiness) scoring.

In the agency system, *trust* captures the general relationship with the player,
while *credibility* answers a narrower question:

  "If I promise something of type X, will the player believe I will deliver?"

Credibility is type-specific and is influenced by:
- trust (baseline)
- credibility_damage (accumulated damage from broken promises)
- fulfilled/broken history by PromiseType
- mental traits (skeptical/forgiving tendencies)

This module is pure business logic with explainable outputs.
"""

import math
from typing import Any, Dict, Mapping, Optional, Tuple

from .promises import PromiseType
from .utils import clamp01, mental_norm, safe_int


def _mem_dict(mem: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(mem, Mapping):
        return {}
    v = mem.get(key)
    if isinstance(v, Mapping):
        return v
    return {}


def compute_credibility(
    *,
    trust: float,
    credibility_damage: float,
    mem: Optional[Mapping[str, Any]],
    mental: Mapping[str, Any],
    promise_type: PromiseType,
    cfg: Any = None,
) -> Tuple[float, Dict[str, Any]]:
    """Compute a credibility score in [0..1] for a given promise type.

    Args:
        trust: current relationship trust (0..1)
        credibility_damage: accumulated promise credibility damage (0..1)
        mem: state.context['mem'] mapping
        mental: player's mental traits mapping
        promise_type: PromiseType being evaluated
        cfg: AgencyConfig-like (unused for now; reserved for future tuning)

    Returns:
        (score, meta)

    Scoring model (v1 spec)
    ----------------------
    broken_penalty  = 1 - exp(-0.55 * broken_count)
    fulfilled_bonus = 1 - exp(-0.35 * fulfilled_count)

    score = trust*(1 - 0.75*broken_penalty) + 0.20*fulfilled_bonus - 0.35*credibility_damage
    score += 0.08*(coach + loy + work - ego - amb)

    clamped to [0..1].
    """

    tr = float(clamp01(trust))
    cd = float(clamp01(credibility_damage))

    mem0: Mapping[str, Any] = mem or {}

    broken_by = _mem_dict(mem0, "broken_promises_by_type")
    fulfilled_by = _mem_dict(mem0, "fulfilled_promises_by_type")

    ptype = str(promise_type).upper()

    broken = safe_int(broken_by.get(ptype), 0)
    fulfilled = safe_int(fulfilled_by.get(ptype), 0)

    broken_penalty = 1.0 - math.exp(-0.55 * float(max(0, broken)))
    fulfilled_bonus = 1.0 - math.exp(-0.35 * float(max(0, fulfilled)))

    ego = mental_norm(mental, "ego")
    amb = mental_norm(mental, "ambition")
    coach = mental_norm(mental, "coachability")
    loy = mental_norm(mental, "loyalty")
    work = mental_norm(mental, "work_ethic")

    mental_adj = 0.08 * (coach + loy + work - ego - amb)

    raw = tr * (1.0 - 0.75 * broken_penalty) + 0.20 * fulfilled_bonus - 0.35 * cd + mental_adj
    score = float(clamp01(raw))

    meta: Dict[str, Any] = {
        "promise_type": ptype,
        "trust": float(tr),
        "credibility_damage": float(cd),
        "broken": int(broken),
        "fulfilled": int(fulfilled),
        "broken_penalty": float(broken_penalty),
        "fulfilled_bonus": float(fulfilled_bonus),
        "mental_adj": float(mental_adj),
        "raw": float(raw),
        "score": float(score),
    }

    return score, meta
