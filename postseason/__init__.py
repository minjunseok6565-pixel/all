"""Postseason (Play-In + Playoffs) subsystem.

This package implements the play-in / playoffs director for the NBA simulation game.

Design goals
------------
- Deterministic IDs for postseason games/series (stable for replay, debugging, news de-dup)
- Correct phase ingestion: play-in => phase='play_in', playoffs => phase='playoffs'
- Keeps existing UI/server contract by preserving the legacy bracket/series/game dict shapes
- Safe, tolerant behavior: fail-loud on SSOT mismatches, but avoid partial state corruption

Public entry points
-------------------
The `playoffs.py` module at project root re-exports:
- build_postseason_field
- reset_postseason_state
- initialize_postseason
- play_my_team_play_in_game
- advance_my_team_one_game
- auto_advance_current_round

Those functions are implemented in `postseason.director`.
"""

from .director import (
    build_postseason_field,
    reset_postseason_state,
    initialize_postseason,
    play_my_team_play_in_game,
    advance_my_team_one_game,
    auto_advance_current_round,
)

__all__ = [
    "build_postseason_field",
    "reset_postseason_state",
    "initialize_postseason",
    "play_my_team_play_in_game",
    "advance_my_team_one_game",
    "auto_advance_current_round",
]
