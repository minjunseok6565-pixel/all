from __future__ import annotations

from typing import Any, Dict

from config import (
    CAP_ANNUAL_GROWTH_RATE,
    CAP_BASE_FIRST_APRON,
    CAP_BASE_SALARY_CAP,
    CAP_BASE_SECOND_APRON,
    CAP_BASE_SEASON_YEAR,
    CAP_ROUND_UNIT,
)


def _apply_cap_model_for_season(league: Dict[str, Any], season_year: int) -> None:
    """Apply the season-specific cap/apron values to trade rules.

    This is the SSOT for season-based cap numbers. We also piggy-back salary
    matching (below-1st-apron) parameters here because, in the real NBA, those
    thresholds scale with the cap over time.
    """
    trade_rules = league.setdefault("trade_rules", {})
    if trade_rules.get("cap_auto_update") is False:
        return
    try:
        base_season_year = int(
            trade_rules.get("cap_base_season_year", CAP_BASE_SEASON_YEAR)
        )
    except (TypeError, ValueError):
        base_season_year = CAP_BASE_SEASON_YEAR
    try:
        base_salary_cap = float(
            trade_rules.get("cap_base_salary_cap", CAP_BASE_SALARY_CAP)
        )
    except (TypeError, ValueError):
        base_salary_cap = float(CAP_BASE_SALARY_CAP)
    try:
        base_first_apron = float(
            trade_rules.get("cap_base_first_apron", CAP_BASE_FIRST_APRON)
        )
    except (TypeError, ValueError):
        base_first_apron = float(CAP_BASE_FIRST_APRON)
    try:
        base_second_apron = float(
            trade_rules.get("cap_base_second_apron", CAP_BASE_SECOND_APRON)
        )
    except (TypeError, ValueError):
        base_second_apron = float(CAP_BASE_SECOND_APRON)
    try:
        annual_growth_rate = float(
            trade_rules.get("cap_annual_growth_rate", CAP_ANNUAL_GROWTH_RATE)
        )
    except (TypeError, ValueError):
        annual_growth_rate = float(CAP_ANNUAL_GROWTH_RATE)
    try:
        round_unit = int(trade_rules.get("cap_round_unit", CAP_ROUND_UNIT) or 1)
    except (TypeError, ValueError):
        round_unit = CAP_ROUND_UNIT
    if round_unit <= 0:
        round_unit = CAP_ROUND_UNIT or 1

    years_passed = season_year - base_season_year
    multiplier = (1.0 + annual_growth_rate) ** years_passed

    def _round_to_unit(value: float) -> int:
        return int(round(value / round_unit) * round_unit)

    salary_cap = _round_to_unit(base_salary_cap * multiplier)
    first_apron = _round_to_unit(base_first_apron * multiplier)
    second_apron = _round_to_unit(base_second_apron * multiplier)

    if first_apron < salary_cap:
        first_apron = salary_cap
    if second_apron < first_apron:
        second_apron = first_apron

    trade_rules["salary_cap"] = salary_cap
    trade_rules["first_apron"] = first_apron
    trade_rules["second_apron"] = second_apron

    # --- Salary matching parameters (below 1st apron) ---
    # The matching brackets are intentionally derived from (mid_add, buffer)
    # to guarantee continuity at thresholds:
    #   2*out + buffer  == out + mid_add  at out = mid_add - buffer
    #   out + mid_add   == 1.25*out + buffer at out = 4*(mid_add - buffer)
    #
    # We scale mid_add by the cap ratio each season when match_auto_update is enabled.
    if trade_rules.get("match_auto_update") is not False:
        # base_mid_add corresponds to the base season cap values.
        try:
            base_mid_add = float(trade_rules.get("match_base_mid_add", 8_527_000))
        except (TypeError, ValueError):
            base_mid_add = 8_527_000.0

        # Keep base_salary_cap consistent with what we used above.
        base_salary_cap_for_match = base_salary_cap
        if base_salary_cap_for_match <= 0:
            base_salary_cap_for_match = float(CAP_BASE_SALARY_CAP)
        if base_salary_cap_for_match <= 0:
            base_salary_cap_for_match = 1.0

        # match_buffer is not scaled (CBA uses a fixed $250k buffer).
        try:
            match_buffer_d = int(round(float(trade_rules.get("match_buffer", 250_000))))
        except (TypeError, ValueError):
            match_buffer_d = 250_000
        if match_buffer_d < 0:
            match_buffer_d = 0

        # Scale by cap ratio, then round to the same unit used for the cap.
        scaled_mid_add = base_mid_add * (float(salary_cap) / float(base_salary_cap_for_match))
        match_mid_add_d = _round_to_unit(scaled_mid_add)
        if match_mid_add_d < match_buffer_d:
            # Prevent negative/invalid thresholds.
            match_mid_add_d = match_buffer_d

        # Derive bracket thresholds.
        match_small_out_max_d = max(0, int(match_mid_add_d) - match_buffer_d)
        match_mid_out_max_d = int(match_small_out_max_d * 4)

        trade_rules["match_mid_add"] = int(match_mid_add_d)
        trade_rules["match_small_out_max"] = int(match_small_out_max_d)
        trade_rules["match_mid_out_max"] = int(match_mid_out_max_d)

