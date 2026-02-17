from __future__ import annotations

"""participants_roles.py

Legacy location for offensive role keys.

Historically this module defined the engine's legacy 12-role keys (e.g.,
"Initiator_Primary"). The project is migrating to a modern, canonical 13-role
system (C13) defined in :mod:`matchengine_v3.offense_roles`.

This file now *re-exports* the canonical role keys (and helpers) so the
participants subsystem can import locally without duplicating strings.
"""

from ..offense_roles import (
    # Canonical C13 keys
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
    ALL_OFFENSE_ROLES,
    # Legacy compatibility + helpers
    LEGACY_12ROLE_TO_CANONICAL,
    CANONICAL_TO_LEGACY_12ROLE,
    canonical_offense_role,
    expand_role_keys_for_lookup,
)

# Backwards-compatible names (only used within participants subsystem today).
# These are kept so callers can transition gradually.
ROLE_INITIATOR_PRIMARY = ROLE_ENGINE_PRIMARY
ROLE_INITIATOR_SECONDARY = ROLE_ENGINE_SECONDARY
ROLE_TRANSITION_HANDLER = ROLE_TRANSITION_ENGINE
ROLE_RIM_ATTACKER = ROLE_RIM_PRESSURE
ROLE_SPACER_CS = ROLE_SPOTUP_SPACER
ROLE_SPACER_MOVE = ROLE_MOVEMENT_SHOOTER
ROLE_ROLLER = ROLE_ROLL_MAN
ROLE_SHORTROLL = ROLE_SHORTROLL_HUB
ROLE_POP_BIG = ROLE_POP_THREAT
ROLE_POST_HUB = ROLE_POST_ANCHOR
