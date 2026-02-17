from __future__ import annotations

"""matchengine_v3.offense_roles

Single source of truth (SSOT) for the *offensive* role key system.

Design goals:
- Canonical keys reflect modern NBA offensive roles.
- Provide compatibility helpers for the legacy 12-role keys used by older configs.

This module is intentionally dependency-light so it can be imported broadly.
"""

from typing import Dict, Sequence, Tuple

# ---------------------------------------------------------------------------
# Canonical C13 role keys (Modern NBA Offensive Role System v1)
# ---------------------------------------------------------------------------

ROLE_ENGINE_PRIMARY = "Engine_Primary"
ROLE_ENGINE_SECONDARY = "Engine_Secondary"
ROLE_TRANSITION_ENGINE = "Transition_Engine"
ROLE_SHOT_CREATOR = "Shot_Creator"
ROLE_RIM_PRESSURE = "Rim_Pressure"

ROLE_SPOTUP_SPACER = "SpotUp_Spacer"
ROLE_MOVEMENT_SHOOTER = "Movement_Shooter"
ROLE_CUTTER_FINISHER = "Cutter_Finisher"
ROLE_CONNECTOR = "Connector"

ROLE_ROLL_MAN = "Roll_Man"
ROLE_SHORTROLL_HUB = "ShortRoll_Hub"
ROLE_POP_THREAT = "Pop_Threat"
ROLE_POST_ANCHOR = "Post_Anchor"


ALL_OFFENSE_ROLES: Tuple[str, ...] = (
    ROLE_ENGINE_PRIMARY,
    ROLE_ENGINE_SECONDARY,
    ROLE_TRANSITION_ENGINE,
    ROLE_SHOT_CREATOR,
    ROLE_RIM_PRESSURE,
    ROLE_SPOTUP_SPACER,
    ROLE_MOVEMENT_SHOOTER,
    ROLE_CUTTER_FINISHER,
    ROLE_CONNECTOR,
    ROLE_ROLL_MAN,
    ROLE_SHORTROLL_HUB,
    ROLE_POP_THREAT,
    ROLE_POST_ANCHOR,
)


# ---------------------------------------------------------------------------
# Legacy 12-role compatibility
# ---------------------------------------------------------------------------

# Legacy -> Canonical
LEGACY_12ROLE_TO_CANONICAL: Dict[str, str] = {
    "Initiator_Primary": ROLE_ENGINE_PRIMARY,
    "Initiator_Secondary": ROLE_ENGINE_SECONDARY,
    "Transition_Handler": ROLE_TRANSITION_ENGINE,
    "Shot_Creator": ROLE_SHOT_CREATOR,
    "Rim_Attacker": ROLE_RIM_PRESSURE,
    "Spacer_CatchShoot": ROLE_SPOTUP_SPACER,
    "Spacer_Movement": ROLE_MOVEMENT_SHOOTER,
    "Connector_Playmaker": ROLE_CONNECTOR,
    "Roller_Finisher": ROLE_ROLL_MAN,
    "ShortRoll_Playmaker": ROLE_SHORTROLL_HUB,
    "Pop_Spacer_Big": ROLE_POP_THREAT,
    "Post_Hub": ROLE_POST_ANCHOR,
}

# Canonical -> Legacy (when a 1:1 legacy key exists)
CANONICAL_TO_LEGACY_12ROLE: Dict[str, str] = {v: k for k, v in LEGACY_12ROLE_TO_CANONICAL.items()}


def canonical_offense_role(role_key: str) -> str:
    """Return the canonical C13 key for a role key.

    If the key is already canonical (or unknown), it is returned unchanged.
    """
    k = str(role_key or "").strip()
    if not k:
        return ""
    return LEGACY_12ROLE_TO_CANONICAL.get(k, k)


def expand_role_keys_for_lookup(role_keys: Sequence[str]) -> Tuple[str, ...]:
    """Expand a list of role keys to include canonical/legacy synonyms.

    Useful during migration when TeamState.roles may contain either:
    - canonical C13 keys
    - legacy 12-role keys
    - or a mix of both

    The expansion preserves order and removes duplicates.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(k: str) -> None:
        kk = str(k or "").strip()
        if not kk:
            return
        if kk in seen:
            return
        seen.add(kk)
        out.append(kk)

    for k in role_keys:
        _add(k)

        # If the provided key is legacy, add canonical.
        canon = LEGACY_12ROLE_TO_CANONICAL.get(str(k))
        if canon:
            _add(canon)

        # If the provided key is canonical, add its legacy alias (if any).
        legacy = CANONICAL_TO_LEGACY_12ROLE.get(str(k))
        if legacy:
            _add(legacy)

        # Also handle the canonical form of the key (if different from original).
        canon2 = canonical_offense_role(str(k))
        if canon2 and canon2 != str(k):
            _add(canon2)
            legacy2 = CANONICAL_TO_LEGACY_12ROLE.get(canon2)
            if legacy2:
                _add(legacy2)

    return tuple(out)


# ---------------------------------------------------------------------------
# Optional group tags (used by rotation/fatigue systems)
# ---------------------------------------------------------------------------

ROLE_GROUP_HANDLER = "Handler"
ROLE_GROUP_WING = "Wing"
ROLE_GROUP_BIG = "Big"

ROLE_TO_GROUPS: Dict[str, Tuple[str, ...]] = {
    ROLE_ENGINE_PRIMARY: (ROLE_GROUP_HANDLER,),
    ROLE_ENGINE_SECONDARY: (ROLE_GROUP_HANDLER, ROLE_GROUP_WING),
    ROLE_TRANSITION_ENGINE: (ROLE_GROUP_HANDLER,),
    ROLE_SHOT_CREATOR: (ROLE_GROUP_WING, ROLE_GROUP_HANDLER),
    ROLE_RIM_PRESSURE: (ROLE_GROUP_WING, ROLE_GROUP_HANDLER),
    ROLE_SPOTUP_SPACER: (ROLE_GROUP_WING,),
    ROLE_MOVEMENT_SHOOTER: (ROLE_GROUP_WING,),
    ROLE_CUTTER_FINISHER: (ROLE_GROUP_WING,),
    ROLE_CONNECTOR: (ROLE_GROUP_WING, ROLE_GROUP_HANDLER),
    ROLE_ROLL_MAN: (ROLE_GROUP_BIG,),
    ROLE_SHORTROLL_HUB: (ROLE_GROUP_BIG,),
    ROLE_POP_THREAT: (ROLE_GROUP_BIG,),
    ROLE_POST_ANCHOR: (ROLE_GROUP_BIG,),
}


def role_groups(role_key: str) -> Tuple[str, ...]:
    """Return group tags for a role key.

    The input may be canonical or legacy.
    """
    canon = canonical_offense_role(role_key)
    return ROLE_TO_GROUPS.get(canon, ())
