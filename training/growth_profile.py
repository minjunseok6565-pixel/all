from __future__ import annotations

import math
import random
from typing import Any, Dict, Mapping

from ratings_2k import potential_grade_to_scalar, compute_ovr_proxy

from .types import stable_seed


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def build_growth_profile(
    *,
    player_id: str,
    attrs: Mapping[str, Any],
    pos: str,
    age: int,
) -> Dict[str, Any]:
    """Create a deterministic per-player career curve profile.

    - ceiling_proxy: max OVR-proxy the player can realistically reach
    - peak_age: age where growth starts slowing sharply
    - decline_start_age: age where decline starts
    - late_decline_age: age where decline accelerates

    Determinism: profile is a pure function of player_id + current snapshot,
    so a save reload yields the same curve.
    """

    pid = str(player_id)
    rng = random.Random(stable_seed("growth_profile", pid))

    pot = potential_grade_to_scalar(attrs.get("Potential"))  # 0.40..1.00
    pot = _clamp(pot, 0.40, 1.00)

    try:
        cur_proxy = float(compute_ovr_proxy(attrs, pos=str(pos)))
    except Exception:
        cur_proxy = 60.0

    # Younger players tend to have more runway.
    # Age factor: 19 -> 1.00, 27 -> ~0.55, 31 -> ~0.35
    age_f = _clamp(1.10 - 0.06 * float(max(0, age - 19)), 0.25, 1.10)

    # Headroom: 6..18ish depending on potential.
    base_headroom = 6.0 + 14.0 * pot
    noise = 0.85 + 0.30 * rng.random()
    headroom = base_headroom * age_f * noise

    ceiling = _clamp(cur_proxy + headroom, cur_proxy + 1.0, 99.0)

    # Peak age: higher potential tends to peak slightly later.
    peak = 24.0 + 4.0 * pot + rng.gauss(0.0, 0.8)
    peak = _clamp(peak, 23.0, 29.5)

    # Decline starts after peak.
    decline_start = peak + 3.5 + (1.0 - pot) * 2.0 + rng.gauss(0.0, 0.7)
    decline_start = _clamp(decline_start, 27.5, 34.0)

    late_decline = decline_start + 4.5 + rng.gauss(0.0, 0.8)
    late_decline = _clamp(late_decline, max(31.0, decline_start + 2.0), 38.0)

    return {
        "player_id": pid,
        "ceiling_proxy": float(ceiling),
        "peak_age": float(peak),
        "decline_start_age": float(decline_start),
        "late_decline_age": float(late_decline),
    }


def ensure_profile(
    *,
    existing: Dict[str, Any] | None,
    player_id: str,
    attrs: Mapping[str, Any],
    pos: str,
    age: int,
) -> Dict[str, Any]:
    if existing:
        return dict(existing)
    return build_growth_profile(player_id=player_id, attrs=attrs, pos=pos, age=age)
