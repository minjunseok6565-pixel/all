from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel


class TradeSubmitRequest(BaseModel):
    deal: Dict[str, Any]


class TradeSubmitCommittedRequest(BaseModel):
    deal_id: str


class TradeNegotiationStartRequest(BaseModel):
    user_team_id: str
    other_team_id: str


class TradeNegotiationCommitRequest(BaseModel):
    session_id: str
    deal: Dict[str, Any]


class TradeEvaluateRequest(BaseModel):
    deal: Dict[str, Any]
    team_id: str
    include_breakdown: bool = True
