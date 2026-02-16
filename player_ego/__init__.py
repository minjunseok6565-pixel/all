"""Player Ego subsystem.

Public surface area:
- PlayerEgoService: orchestration API used by LeagueService / sim hooks
- Types are under player_ego.types

This package is designed to be added incrementally:
- Schema creation: db_schema/ego.py (include in db_schema/init.DEFAULT_MODULES)
- Service hooks: sim/league_sim.py + league_service.py integration

The implementation is deterministic and tunable, aiming for realistic NBA-like
player behavior without excessive disruption.
"""

from __future__ import annotations

from .service import PlayerEgoService

__all__ = ["PlayerEgoService"]
