# player_ego/archetypes.py
"""Deterministic personality archetypes for Player Ego.

Why archetypes?
--------------
If every player uses the same complaint/negotiation logic with only random noise,
the league quickly feels "all players are divas". Archetypes ensure:
- Bench players usually keep their heads down
- Glue guys are professional and internal
- Stars have leverage but still diverse (loyal vs mercenary)

Commercial release goals:
- Deterministic across machines: same save => same personalities.
- Realistic distribution: not everyone demands trades.

Implementation notes
--------------------
We avoid Python's built-in hash() because it is randomized per process.
We use SHA256 to derive stable pseudo-random numbers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple


def _clamp01(x: float) -> float:
    try:
        xf = float(x)
    except Exception:
        return 0.0
    if xf < 0.0:
        return 0.0
    if xf > 1.0:
        return 1.0
    return xf


def _stable_u32(*parts: str) -> int:
    h = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _stable_unit_float(*parts: str) -> float:
    # Map stable u32 to [0,1).
    return (_stable_u32(*parts) % 10_000_000) / 10_000_000.0


def _jitter(base: float, *, seed_parts: Tuple[str, ...], amp: float = 0.08) -> float:
    """Small stable noise to avoid identical clones."""
    u = _stable_unit_float(*seed_parts)
    # u in [0,1) -> [-1,1)
    delta = (u * 2.0 - 1.0) * float(amp)
    return _clamp01(float(base) + delta)


@dataclass(frozen=True, slots=True)
class Archetype:
    key: str
    weights: Dict[str, float]


# -----------------------------------------------------------------------------
# Archetype library
# -----------------------------------------------------------------------------

# Values are anchors (0..1). We add small stable jitter per player.
ARCHETYPES: Dict[str, Archetype] = {
    # Loyal veteran: low drama, accepts role, values stability.
    "LOYAL_VET": Archetype(
        "LOYAL_VET",
        {
            "ego": 0.35,
            "loyalty": 0.85,
            "money_focus": 0.45,
            "win_focus": 0.55,
            "ambition": 0.40,
            "patience": 0.80,
            "professionalism": 0.85,
            "volatility": 0.25,
            "privacy": 0.75,
            "risk_tolerance": 0.35,
        },
    ),
    # Mercenary: follows the money, can bounce in FA.
    "MERCENARY": Archetype(
        "MERCENARY",
        {
            "ego": 0.55,
            "loyalty": 0.25,
            "money_focus": 0.90,
            "win_focus": 0.45,
            "ambition": 0.55,
            "patience": 0.45,
            "professionalism": 0.60,
            "volatility": 0.45,
            "privacy": 0.55,
            "risk_tolerance": 0.70,
        },
    ),
    # Win-now star: wants to compete and be treated as a centerpiece.
    "WIN_NOW_STAR": Archetype(
        "WIN_NOW_STAR",
        {
            "ego": 0.80,
            "loyalty": 0.45,
            "money_focus": 0.55,
            "win_focus": 0.95,
            "ambition": 0.85,
            "patience": 0.45,
            "professionalism": 0.65,
            "volatility": 0.55,
            "privacy": 0.40,
            "risk_tolerance": 0.55,
        },
    ),
    # Young gun: volatile, wants minutes and a big role.
    "YOUNG_GUN": Archetype(
        "YOUNG_GUN",
        {
            "ego": 0.70,
            "loyalty": 0.45,
            "money_focus": 0.55,
            "win_focus": 0.55,
            "ambition": 0.90,
            "patience": 0.40,
            "professionalism": 0.55,
            "volatility": 0.80,
            "privacy": 0.50,
            "risk_tolerance": 0.65,
        },
    ),
    # Glue guy: pro's pro, team-first, internal.
    "GLUE_GUY": Archetype(
        "GLUE_GUY",
        {
            "ego": 0.35,
            "loyalty": 0.70,
            "money_focus": 0.45,
            "win_focus": 0.60,
            "ambition": 0.45,
            "patience": 0.75,
            "professionalism": 0.95,
            "volatility": 0.25,
            "privacy": 0.85,
            "risk_tolerance": 0.35,
        },
    ),
    # End-of-bench: low leverage, low drama.
    "END_OF_BENCH": Archetype(
        "END_OF_BENCH",
        {
            "ego": 0.20,
            "loyalty": 0.55,
            "money_focus": 0.45,
            "win_focus": 0.45,
            "ambition": 0.25,
            "patience": 0.85,
            "professionalism": 0.80,
            "volatility": 0.20,
            "privacy": 0.80,
            "risk_tolerance": 0.25,
        },
    ),
}


def _tier_from_player(player: Mapping[str, Any]) -> str:
    """Rough tier for distribution: STAR / STARTER / ROTATION / BENCH."""
    try:
        ovr = int(player.get("ovr") or 0)
    except Exception:
        ovr = 0

    if ovr >= 88:
        return "STAR"
    if ovr >= 80:
        return "STARTER"
    if ovr >= 74:
        return "ROTATION"
    return "BENCH"


def choose_archetype(player: Mapping[str, Any]) -> str:
    """Choose a deterministic archetype for a player.

    Uses a stable hash derived from player_id and tier.
    """
    pid = str(player.get("player_id") or "")
    tier = _tier_from_player(player)

    # Distribution by tier. Bench-heavy stability is intentional.
    if tier == "STAR":
        choices = [
            ("WIN_NOW_STAR", 0.45),
            ("MERCENARY", 0.25),
            ("LOYAL_VET", 0.20),
            ("GLUE_GUY", 0.10),
        ]
    elif tier == "STARTER":
        choices = [
            ("GLUE_GUY", 0.30),
            ("LOYAL_VET", 0.25),
            ("YOUNG_GUN", 0.20),
            ("MERCENARY", 0.15),
            ("WIN_NOW_STAR", 0.10),
        ]
    elif tier == "ROTATION":
        choices = [
            ("GLUE_GUY", 0.35),
            ("END_OF_BENCH", 0.25),
            ("LOYAL_VET", 0.20),
            ("YOUNG_GUN", 0.15),
            ("MERCENARY", 0.05),
        ]
    else:  # BENCH
        choices = [
            ("END_OF_BENCH", 0.55),
            ("GLUE_GUY", 0.25),
            ("LOYAL_VET", 0.15),
            ("YOUNG_GUN", 0.05),
        ]

    u = _stable_unit_float("ego_archetype", pid, tier)
    acc = 0.0
    for key, p in choices:
        acc += float(p)
        if u <= acc:
            return key
    return choices[-1][0]


def generate_traits(player: Mapping[str, Any]) -> Dict[str, float | str]:
    """Generate deterministic traits for a given player.

    Output is JSON-serializable.
    Values are clamped to [0,1].
    """
    pid = str(player.get("player_id") or "")
    archetype_key = choose_archetype(player)
    arch = ARCHETYPES.get(archetype_key, ARCHETYPES["GLUE_GUY"])

    traits: Dict[str, float | str] = {
        "version": "1.0",
        "archetype": archetype_key,
    }

    # Add per-trait stable jitter so two players in same archetype are not identical.
    for k, base in arch.weights.items():
        traits[k] = float(_jitter(float(base), seed_parts=("ego_trait", pid, archetype_key, k)))

    # Light conditioning by age: younger players tend to be more ambitious/volatile.
    try:
        age = int(player.get("age") or 0)
    except Exception:
        age = 0

    if age and age <= 24:
        traits["ambition"] = _clamp01(float(traits.get("ambition", 0.5)) + 0.06)
        traits["volatility"] = _clamp01(float(traits.get("volatility", 0.5)) + 0.05)
        traits["patience"] = _clamp01(float(traits.get("patience", 0.5)) - 0.04)
    elif age and age >= 32:
        traits["professionalism"] = _clamp01(float(traits.get("professionalism", 0.5)) + 0.05)
        traits["patience"] = _clamp01(float(traits.get("patience", 0.5)) + 0.05)
        traits["risk_tolerance"] = _clamp01(float(traits.get("risk_tolerance", 0.5)) - 0.04)

    return traits
