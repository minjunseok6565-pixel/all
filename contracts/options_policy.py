"""Contract option decision policies.

This module contains the *default* option policy used by the offseason contract
processor, plus optional helper policies.

Design constraint:
- The default policy must be conservative/stable (never surprising).
- More "gamey"/AI policies should be opt-in from the caller (e.g. server flow).

Option decision hook signature (used by LeagueService.expire_contracts_for_season_transition):
    (option: dict, player_id: str, contract: dict, game_state: dict) -> "EXERCISE" | "DECLINE"

NOTE:
- When called via contracts.offseason.process_offseason(), the policy receives the
  *full* exported game_state snapshot (so it can use ui_cache, stats, etc.).
"""

from __future__ import annotations

import hashlib
from typing import Callable, Literal, Optional

from config import (
    CAP_ANNUAL_GROWTH_RATE,
    CAP_BASE_SALARY_CAP,
    CAP_BASE_SEASON_YEAR,
    CAP_ROUND_UNIT,
)
from schema import normalize_player_id

Decision = Literal["EXERCISE", "DECLINE"]


def normalize_option_type(value: str) -> str:
    normalized = str(value).strip().upper()
    if normalized not in {"TEAM", "PLAYER", "ETO"}:
        raise ValueError(f"Invalid option type: {value}")
    return normalized


def default_option_decision_policy(
    option: dict,
    player_id: str,
    contract: dict,
    game_state: dict,
) -> Decision:
    """Minimal default policy for stability.

    - TEAM option: EXERCISE
    - PLAYER option: EXERCISE
    - ETO: EXERCISE

    This exists to keep DB transitions deterministic even if no AI logic is wired.
    """
    normalize_player_id(player_id, strict=False, allow_legacy_numeric=True)
    return "EXERCISE"


# ---------------------------------------------------------------------------
# Optional: AI TEAM option policy helpers
# ---------------------------------------------------------------------------

def _stable_u32(text: str) -> int:
    """Deterministic 32-bit unsigned integer from text."""
    h = hashlib.blake2b(str(text).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "little", signed=False)


def _stable_rand01(*parts: object) -> float:
    """Deterministic pseudo-random float in [0, 1)."""
    key = "|".join(str(p) for p in parts)
    return float(_stable_u32(key) % 1_000_000) / 1_000_000.0


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        x = float(value)  # type: ignore[arg-type]
        if x != x:  # NaN
            return float(default)
        return x
    except Exception:
        return float(default)


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return int(default)


def _cap_for_season_year_from_state(game_state: dict, season_year: int) -> int:
    """Compute salary cap for a given season_year using state trade_rules if present."""
    y = int(season_year)
    league = game_state.get("league") if isinstance(game_state, dict) else None
    trade_rules = league.get("trade_rules") if isinstance(league, dict) else None
    if not isinstance(trade_rules, dict):
        trade_rules = {}

    base_year = _coerce_int(trade_rules.get("cap_base_season_year"), int(CAP_BASE_SEASON_YEAR))
    base_cap = _coerce_float(trade_rules.get("cap_base_salary_cap"), float(CAP_BASE_SALARY_CAP))
    growth = _coerce_float(trade_rules.get("cap_annual_growth_rate"), float(CAP_ANNUAL_GROWTH_RATE))
    round_unit = _coerce_int(trade_rules.get("cap_round_unit"), int(CAP_ROUND_UNIT) or 1)
    if round_unit <= 0:
        round_unit = int(CAP_ROUND_UNIT) or 1

    years_passed = y - int(base_year)
    mult = (1.0 + float(growth)) ** years_passed
    raw = float(base_cap) * float(mult)
    return int(round(raw / float(round_unit)) * int(round_unit))


def make_ai_team_option_decision_policy(
    *,
    user_team_id: Optional[str] = None,
    baseline_decline_threshold: float = 2.2,
) -> Callable[[dict, str, dict, dict], Decision]:
    """Create an opt-in policy: AI teams evaluate TEAM options and may DECLINE.

    Scope:
    - Only affects TEAM options.
    - Only affects AI teams (team_id != user_team_id).
    - Leaves PLAYER/ETO options as default (EXERCISE) for now.

    The scoring is intentionally simple and uses only the passed-in game_state
    snapshot (ui_cache + season stats). It is deterministic and cheap.

    Tuning tip:
    - Increase baseline_decline_threshold to make AI *more* likely to decline.
    - Decrease it to make AI *more* likely to exercise.
    """

    user_team_norm = str(user_team_id).strip().upper() if user_team_id else None
    base_threshold = float(baseline_decline_threshold)

    def _policy(option: dict, player_id: str, contract: dict, game_state: dict) -> Decision:
        # Defensive: keep default stability.
        try:
            opt_type = normalize_option_type(option.get("type"))
        except Exception:
            return "EXERCISE"
        if opt_type != "TEAM":
            return "EXERCISE"

        team_id = str(contract.get("team_id") or "").strip().upper()
        if not team_id or team_id == "FA":
            return "EXERCISE"
        if user_team_norm and team_id == user_team_norm:
            # Safety: user TEAM options should be decided by the user flow (hard gate).
            return "EXERCISE"

        opt_year = _coerce_int(option.get("season_year"), 0)
        salary = _coerce_float((contract.get("salary_by_year") or {}).get(str(opt_year)), 0.0)
        salary_m = float(salary) / 1_000_000.0

        # Player UI meta (derived from DB; kept in runtime state for UI).
        ui_players = ((game_state.get("ui_cache") or {}).get("players") or {})
        p = ui_players.get(str(player_id)) if isinstance(ui_players, dict) else None
        if not isinstance(p, dict):
            return "EXERCISE"  # conservative fallback

        ovr = _coerce_float(p.get("overall"), _coerce_float(p.get("ovr"), 0.0))
        age = _coerce_int(p.get("age"), 0)
        potential = _coerce_float(p.get("potential"), 0.6)  # 0.4~1.0-ish

        # Last-season usage signal (minutes per game).
        mpg = 0.0
        ps = (game_state.get("player_stats") or {}).get(str(player_id))
        if isinstance(ps, dict):
            g = _coerce_int(ps.get("games"), 0)
            totals = ps.get("totals")
            if g > 0 and isinstance(totals, dict):
                mpg = _coerce_float(totals.get("MIN"), 0.0) / float(g)

        cap = _cap_for_season_year_from_state(game_state, opt_year)
        salary_pct = (float(salary) / float(cap)) if cap > 0 else 0.0

        # Contract type biases (only present when written into contract_json).
        ct = str(contract.get("contract_type") or "").strip().upper()
        start_year = _coerce_int(contract.get("start_season_year"), 0)
        option_offset = opt_year - start_year  # e.g. 2 => 3rd season of deal
        bias = 0.0
        if ct == "ROOKIE_SCALE":
            # NBA-like feel: 3rd-year option should be exercised most of the time.
            bias += 0.8
            if option_offset == 2:
                bias += 0.6
            elif option_offset == 3:
                bias += 0.2
        elif ct == "SECOND_ROUND_EXCEPTION":
            bias += 0.4

        # Simple quality proxy.
        quality = (ovr - 55.0)
        quality += (potential - 0.6) * 12.0
        quality += (mpg - 10.0) * 0.2
        quality -= max(0.0, float(age) - 25.0) * 0.5

        # Expensive deals get a harsher penalty (esp. top picks who bust).
        quality -= max(0.0, salary_pct - 0.03) * 200.0  # penalty starts above ~3% of cap

        value_score = quality / max(float(salary_m), 0.75)
        value_score += bias

        # Hard keep / hard cut rules (helps avoid bizarre outcomes).
        if ovr >= 78.0:
            return "EXERCISE"
        if (ovr <= 60.0 and potential < 0.65 and mpg < 6.0 and salary_pct > 0.015):
            return "DECLINE"

        # Team-level patience tweak (if present; UI-only but good enough for flavor).
        threshold = base_threshold
        ui_teams = ((game_state.get("ui_cache") or {}).get("teams") or {})
        t = ui_teams.get(team_id) if isinstance(ui_teams, dict) else None
        if isinstance(t, dict):
            pat = _coerce_float(t.get("patience"), 0.5)  # 0~1
            # patient => lower threshold => more EXERCISE
            threshold -= (pat - 0.5) * 0.6

        # Add small deterministic "human error" noise for borderline cases.
        noise = (_stable_rand01(team_id, player_id, opt_year) - 0.5) * 0.6  # [-0.3, +0.3]

        return "EXERCISE" if (value_score + noise) >= threshold else "DECLINE"

    return _policy
