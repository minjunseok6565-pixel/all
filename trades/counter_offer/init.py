from __future__ import annotations

"""trades/counter_offer/init.py

Public entrypoint for the counter-offer subsystem.

Why init.py (not __init__.py)?
------------------------------
This project sometimes keeps a lightweight `init.py` as an explicit import target
(e.g., db_schema/init.py). The `counter_offer` folder can still be imported as a
namespace package if __init__.py is absent.

Typical usage
-------------
    from trades.counter_offer.init import build_counter_offer

    counter = build_counter_offer(
        offer=deal,
        user_team_id=session["user_team_id"],
        other_team_id=session["other_team_id"],
        current_date=in_game_date,
        db_path=db_path,
        session=session,
    )

The returned object is trades.valuation.types.CounterProposal.
"""

from dataclasses import replace
from datetime import date
from typing import Any, Mapping, Optional

from ..models import Deal
from ..generation.generation_tick import TradeGenerationTickContext
from ..valuation.types import CounterProposal, DealDecision, DealVerdict, DecisionReason

from .config import CounterOfferConfig
from .builder import CounterOfferBuilder


def build_counter_offer(
    *,
    offer: Deal,
    user_team_id: str,
    other_team_id: str,
    current_date: date,
    db_path: str,
    session: Optional[Mapping[str, Any]] = None,
    allow_locked_by_deal_id: Optional[str] = None,
    tick_ctx: Optional[TradeGenerationTickContext] = None,
    config: Optional[CounterOfferConfig] = None,
) -> Optional[CounterProposal]:
    """One-shot helper to build a counter offer."""

    builder = CounterOfferBuilder(config=config)
    return builder.build(
        offer=offer,
        user_team_id=user_team_id,
        other_team_id=other_team_id,
        current_date=current_date,
        db_path=db_path,
        session=session,
        allow_locked_by_deal_id=allow_locked_by_deal_id,
        tick_ctx=tick_ctx,
    )


def attach_counter_or_downgrade(
    *,
    decision: DealDecision,
    offer: Deal,
    user_team_id: str,
    other_team_id: str,
    current_date: date,
    db_path: str,
    session: Optional[Mapping[str, Any]] = None,
    allow_locked_by_deal_id: Optional[str] = None,
    tick_ctx: Optional[TradeGenerationTickContext] = None,
    config: Optional[CounterOfferConfig] = None,
) -> DealDecision:
    """If decision is COUNTER, attach a real counter offer or downgrade to REJECT.

    Rationale
    ---------
    The valuation DecisionPolicy may return DealVerdict.COUNTER to express willingness
    to negotiate in a gray zone. However, the *actual* counter proposal generation
    lives in the trades.counter_offer subsystem.

    To avoid SSOT drift and UX bugs ("COUNTER but no proposal"), any API path
    that wants to expose counter-offers should run its decision through this helper.

    Behavior
    --------
    - If decision.verdict != COUNTER: return decision unchanged.
    - If decision already includes a counter with a Deal: return decision unchanged.
    - Else attempt to build a counter.
      - Success: return decision with counter attached.
      - Failure: return decision downgraded to REJECT with an extra reason.
    """

    if decision.verdict != DealVerdict.COUNTER:
        return decision

    # Idempotency: if a real counter is already attached, do nothing.
    try:
        if decision.counter is not None and getattr(decision.counter, "deal", None) is not None:
            return decision
    except Exception:
        pass

    counter: Optional[CounterProposal] = None
    try:
        counter = build_counter_offer(
            offer=offer,
            user_team_id=user_team_id,
            other_team_id=other_team_id,
            current_date=current_date,
            db_path=db_path,
            session=session,
            allow_locked_by_deal_id=allow_locked_by_deal_id,
            tick_ctx=tick_ctx,
            config=config,
        )
    except Exception:
        counter = None

    if counter is not None and getattr(counter, "deal", None) is not None:
        return replace(decision, counter=counter)

    # If we can't build a counter, never leak a COUNTER verdict without an offer.
    fail_reason = DecisionReason(
        code="COUNTER_BUILD_FAILED",
        message="Could not generate a legal counter offer",
    )
    try:
        reasons = tuple(decision.reasons) + (fail_reason,)
    except Exception:
        reasons = (fail_reason,)

    return replace(
        decision,
        verdict=DealVerdict.REJECT,
        counter=None,
        reasons=reasons,
    )


__all__ = [
    "CounterOfferConfig",
    "CounterOfferBuilder",
    "attach_counter_or_downgrade",
    "build_counter_offer",
]
