# player_ego/types.py
"""Player Ego subsystem: shared types.

We persist ego data in SQLite as JSON blobs (traits_json, state_json).
This module defines the expected shapes.

Goals
-----
- Stable, versioned storage format suitable for commercial saves.
- Backwards-compatible reads (missing keys => defaults).
- Minimal coupling: callers pass small context dicts rather than importing runtime state.

This file contains no database I/O.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


EGO_TRAITS_VERSION: str = "1.0"
EGO_STATE_VERSION: str = "1.0"


IssueType = Literal[
    "MINUTES_COMPLAINT",
    "ROLE_COMPLAINT",
    "CONTRACT_DISPUTE",
    "TEAM_DIRECTION",
    "TRADE_REQUEST",
]

IssueStatus = Literal["OPEN", "RESOLVED", "ESCALATED"]

DesiredRole = Literal["STARTER", "ROTATION", "BENCH"]

OfferDecisionType = Literal["ACCEPT", "COUNTER", "REJECT"]


class PlayerTraits(TypedDict, total=False):
    """Mostly fixed personality values in [0,1]."""

    version: str
    archetype: str

    # Core personality axes
    ego: float  # pride / respect sensitivity
    loyalty: float  # attachment to team
    money_focus: float  # cares about dollars
    win_focus: float  # cares about contending
    ambition: float  # minutes/role hunger
    patience: float  # slower to complain
    professionalism: float  # keeps it internal, plays hard even unhappy
    volatility: float  # mood swings
    privacy: float  # keeps issues private vs public
    risk_tolerance: float  # opt-out / FA risk appetite


class EgoIssue(TypedDict, total=False):
    issue_id: str
    type: IssueType
    status: IssueStatus

    severity: float  # 0..1
    created_date: str  # YYYY-MM-DD
    updated_date: str  # YYYY-MM-DD

    # Optional details for UI / logging
    title: str
    summary: str
    meta: Dict[str, Any]


class EgoState(TypedDict, total=False):
    """Mutable state that changes through the season."""

    version: str

    happiness: float  # 0..1
    trust_team: float  # 0..1

    desired_minutes: float
    desired_role: DesiredRole

    # Rolling buffer of per-game minutes (most recent last)
    recent_minutes: List[float]

    open_issues: List[EgoIssue]
    cooldowns: Dict[str, str]  # issue_type -> YYYY-MM-DD

    last_team_id: Optional[str]
    last_updated_date: Optional[str]


class ContractOffer(TypedDict, total=False):
    """A re-sign/extension offer (minimal normalized shape)."""

    years: int
    salary_by_year: Dict[str, float]
    # Optional negotiation context
    promised_role: Optional[DesiredRole]
    promised_minutes: Optional[float]


class OfferDecision(TypedDict, total=False):
    decision: OfferDecisionType
    reason_code: str
    reason: str
    counter: Optional[ContractOffer]


class EgoRow(TypedDict):
    """Convenience wrapper for store reads."""

    player_id: str
    traits: PlayerTraits
    state: EgoState
