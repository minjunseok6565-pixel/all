from __future__ import annotations

"""Legacy facade for postseason endpoints.

The original project exposed postseason features via `playoffs.py`.
To keep backward compatibility with server routes and existing imports, this
module re-exports the public API implemented in `postseason.director`.

Do not put core logic here.
"""

from postseason.director import (
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
