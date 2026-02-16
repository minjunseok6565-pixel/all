# player_ego/events.py
"""Transaction payload builders for Player Ego.

The project already has a transactions_log SSOT table. This ego subsystem emits
structured entries so:
- UI can display them as a feed
- news_ai can summarize them in weekly news

We intentionally keep the payload flat and rule-friendly:
- top-level `type`, `date`, `season_year`, `teams` are always present
- include player_id, team_id for easy filtering

The tx hash is computed when inserted into the transactions_log table
(via LeagueRepo.insert_transactions or LeagueService._insert_transactions_in_cur).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .types import OfferDecision


def build_issue_tx(
    *,
    issue: Mapping[str, Any],
    player_id: str,
    team_id: str,
    date_iso: str,
    season_year: Optional[int] = None,
    source: str = "player_ego",
) -> Dict[str, Any]:
    it = dict(issue)
    title = it.get("title") or "Player issue"
    summary = it.get("summary") or ""
    return {
        "type": "PLAYER_ISSUE",
        "date": str(date_iso),
        "season_year": int(season_year) if season_year is not None else None,
        "source": source,
        "teams": [str(team_id).upper()],
        "team_id": str(team_id).upper(),
        "player_id": str(player_id),
        "issue_type": str(it.get("type") or ""),
        "issue_id": str(it.get("issue_id") or ""),
        "severity": float(it.get("severity") or 0.0),
        "status": str(it.get("status") or ""),
        "title": str(title),
        "summary": str(summary),
        "meta": it.get("meta") or {},
    }


def build_trade_request_tx(
    *,
    issue: Mapping[str, Any],
    player_id: str,
    team_id: str,
    date_iso: str,
    season_year: Optional[int] = None,
) -> Dict[str, Any]:
    it = dict(issue)
    return {
        "type": "TRADE_REQUEST",
        "date": str(date_iso),
        "season_year": int(season_year) if season_year is not None else None,
        "source": "player_ego",
        "teams": [str(team_id).upper()],
        "team_id": str(team_id).upper(),
        "player_id": str(player_id),
        "issue_id": str(it.get("issue_id") or ""),
        "severity": float(it.get("severity") or 0.0),
        "title": str(it.get("title") or "Trade request"),
        "summary": str(it.get("summary") or "Player has requested a trade."),
        "meta": it.get("meta") or {},
    }


def build_offer_response_tx(
    *,
    player_id: str,
    team_id: str,
    date_iso: str,
    decision: OfferDecision,
    offered: Mapping[str, Any],
    season_year: Optional[int] = None,
) -> Dict[str, Any]:
    dec = dict(decision)
    return {
        "type": "PLAYER_OFFER_RESPONSE",
        "date": str(date_iso),
        "season_year": int(season_year) if season_year is not None else None,
        "source": "player_ego",
        "teams": [str(team_id).upper()],
        "team_id": str(team_id).upper(),
        "player_id": str(player_id),
        "decision": str(dec.get("decision") or ""),
        "reason_code": str(dec.get("reason_code") or ""),
        "reason": str(dec.get("reason") or ""),
        "offered": dict(offered),
        "counter": dec.get("counter"),
    }
