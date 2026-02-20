from __future__ import annotations

"""DB-backed orchestration for responding to agency events.

This module is the bridge between:
- agency_events (player-generated events)
- user responses (acknowledge/promise/refuse)
- player_agency_state (trust/frustration updates)
- promises (optional, persisted and evaluated later)

It is designed for server APIs:
- validate inputs (never trust payload)
- idempotent writes (safe on retries)
- deterministic IDs (reproducible, conflict-safe)

NOTE
----
This file intentionally does not depend on FastAPI.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from league_repo import LeagueRepo

from .config import AgencyConfig, DEFAULT_CONFIG
from .repo import get_player_agency_states, insert_agency_events, upsert_player_agency_states
from .responses import DEFAULT_RESPONSE_CONFIG, ResponseConfig, apply_user_response
from .utils import (
    clamp01,
    extract_mental_from_attrs,
    json_dumps,
    json_loads,
    make_event_id,
    norm_date_iso,
    safe_float,
    safe_int,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgencyInteractionError(Exception):
    code: str
    message: str
    details: Dict[str, Any]

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.code}: {self.message} ({self.details})"


EVENT_NOT_FOUND = "AGENCY_EVENT_NOT_FOUND"
EVENT_TEAM_MISMATCH = "AGENCY_EVENT_TEAM_MISMATCH"
PLAYER_NOT_ON_TEAM = "AGENCY_PLAYER_NOT_ON_TEAM"
BAD_RESPONSE = "AGENCY_BAD_RESPONSE"
PROMISE_SCHEMA_MISSING = "AGENCY_PROMISE_SCHEMA_MISSING"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_exists(cur, name: str) -> bool:
    row = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;",
        (str(name),),
    ).fetchone()
    return bool(row)


def _now_utc_like_iso_from_date(date_iso: str) -> str:
    """Best-effort convert YYYY-MM-DD into UTC-like timestamp using game_time."""
    d = norm_date_iso(date_iso) or "2000-01-01"
    try:
        import game_time

        return game_time.utc_like_from_date_iso(d)
    except Exception:
        # Never use OS clock.
        return f"{d}T00:00:00Z"


def _load_agency_event(cur, event_id: str) -> Dict[str, Any]:
    r = cur.execute(
        """
        SELECT
            event_id,
            player_id,
            team_id,
            season_year,
            date,
            event_type,
            severity,
            payload_json,
            created_at
        FROM agency_events
        WHERE event_id = ?
        LIMIT 1;
        """,
        (str(event_id),),
    ).fetchone()

    if not r:
        raise AgencyInteractionError(EVENT_NOT_FOUND, "Agency event not found", {"event_id": event_id})

    payload = json_loads(r[7], default={})
    if not isinstance(payload, Mapping):
        payload = {}

    return {
        "event_id": str(r[0]),
        "player_id": str(r[1]),
        "team_id": str(r[2]).upper(),
        "season_year": safe_int(r[3], 0),
        "date": str(r[4])[:10],
        "event_type": str(r[5]).upper(),
        "severity": safe_float(r[6], 0.0),
        "payload": dict(payload),
        "created_at": r[8],
    }


def _load_player_profile(cur, player_id: str, *, cfg: AgencyConfig) -> Dict[str, Any]:
    r = cur.execute(
        "SELECT player_id, ovr, age, attrs_json FROM players WHERE player_id=? LIMIT 1;",
        (str(player_id),),
    ).fetchone()

    if not r:
        return {
            "player_id": str(player_id),
            "ovr": None,
            "age": None,
            "mental": {},
        }

    attrs_json = r[3]
    mental = extract_mental_from_attrs(attrs_json, keys=cfg.mental_attr_keys)

    return {
        "player_id": str(r[0]),
        "ovr": safe_int(r[1], 0),
        "age": safe_int(r[2], 0),
        "mental": mental,
    }


def _default_state_for_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    """Create a safe default agency state if missing."""
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        payload = {}

    lev = safe_float(payload.get("leverage"), 0.0)

    return {
        "player_id": str(event.get("player_id") or ""),
        "team_id": str(event.get("team_id") or "").upper(),
        "season_year": safe_int(event.get("season_year"), 0),
        "role_bucket": str(payload.get("role_bucket") or "UNKNOWN"),
        "leverage": float(clamp01(lev)),
        "minutes_expected_mpg": safe_float(payload.get("expected_mpg"), 0.0),
        "minutes_actual_mpg": safe_float(payload.get("actual_mpg"), 0.0),
        "minutes_frustration": 0.0,
        "team_frustration": 0.0,
        "trust": 0.5,

        # v2 axes
        "role_frustration": 0.0,
        "contract_frustration": 0.0,
        "health_frustration": 0.0,
        "chemistry_frustration": 0.0,
        "usage_frustration": 0.0,

        # v2 evidence caches (best-effort)
        "starts_rate": safe_float(payload.get("starts_rate"), 0.0),
        "closes_rate": safe_float(payload.get("closes_rate"), 0.0),
        "usage_share": safe_float(payload.get("usage_share"), 0.0),
        "trade_request_level": 0,
        "cooldown_minutes_until": None,
        "cooldown_trade_until": None,
        "cooldown_help_until": None,
        "cooldown_contract_until": None,

        "cooldown_role_until": None,
        "cooldown_health_until": None,
        "cooldown_chemistry_until": None,

        "escalation_role": 0,
        "escalation_contract": 0,
        "escalation_team": 0,
        "escalation_health": 0,
        "escalation_chemistry": 0,
        "last_processed_month": None,
        "context": {},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def respond_to_agency_event(
    *,
    db_path: str,
    user_team_id: str,
    event_id: str,
    response_type: str,
    response_payload: Optional[Mapping[str, Any]] = None,
    now_date_iso: Optional[str] = None,
    cfg: AgencyConfig = DEFAULT_CONFIG,
    rcfg: ResponseConfig = DEFAULT_RESPONSE_CONFIG,
    strict_promises: bool = True,
) -> Dict[str, Any]:
    """Respond to a single agency event.

    Idempotency
    ----------
    - We write a deterministic response event into agency_events:
        event_id = make_event_id('agency', 'response', source_event_id)
      If it already exists, we return skipped=True without applying state changes.

    Promise persistence
    -------------------
    - If the response produces a PromiseSpec, we attempt to persist it into
      player_agency_promises.
    - If strict_promises=True and the table is missing, we raise.

    Returns:
        dict payload suitable for API responses.
    """

    if not str(event_id or "").strip():
        raise AgencyInteractionError(BAD_RESPONSE, "event_id is required", {"event_id": event_id})
    if not str(user_team_id or "").strip():
        raise AgencyInteractionError(BAD_RESPONSE, "user_team_id is required", {"user_team_id": user_team_id})

    user_tid = str(user_team_id).upper()

    # Determine now date
    if now_date_iso is None:
        try:
            import game_time

            now_date_iso = game_time.game_date_iso()
        except Exception:
            now_date_iso = "2000-01-01"

    now_date = norm_date_iso(now_date_iso) or "2000-01-01"
    now_iso = _now_utc_like_iso_from_date(now_date)

    with LeagueRepo(db_path) as repo:
        repo.init_db()

        with repo.transaction() as cur:
            # Load the source event
            ev = _load_agency_event(cur, str(event_id))

            if str(ev.get("team_id") or "").upper() != user_tid:
                raise AgencyInteractionError(
                    EVENT_TEAM_MISMATCH,
                    "Cannot respond to another team's agency event",
                    {"event_id": ev.get("event_id"), "event_team_id": ev.get("team_id"), "user_team_id": user_tid},
                )

            # Safety: prevent responding to stale events after a trade/waive.
            # Even if the event belongs to this team historically, only the
            # player's *current* active roster team may mutate their agency state.
            row_roster = cur.execute(
                "SELECT team_id FROM roster WHERE player_id=? AND status='active' LIMIT 1;",
                (str(ev.get("player_id") or ""),),
            ).fetchone()
            current_tid = str(row_roster[0]).upper() if row_roster and row_roster[0] is not None else None
            if current_tid != user_tid:
                raise AgencyInteractionError(
                    PLAYER_NOT_ON_TEAM,
                    "Cannot respond: player is no longer on this team",
                    {
                        "event_id": ev.get("event_id"),
                        "player_id": ev.get("player_id"),
                        "event_team_id": ev.get("team_id"),
                        "user_team_id": user_tid,
                        "current_team_id": current_tid,
                    },
                )

            # Deterministic response event ID (one response per source event)
            response_event_id = make_event_id("agency", "response", ev["event_id"])

            # Idempotency: if response event already exists, skip safely.
            already = cur.execute(
                "SELECT 1 FROM agency_events WHERE event_id=? LIMIT 1;",
                (response_event_id,),
            ).fetchone()
            if already:
                # Best effort: return current state
                st_map = get_player_agency_states(cur, [ev["player_id"]])
                return {
                    "ok": True,
                    "skipped": True,
                    "reason": "already_responded",
                    "event_id": ev["event_id"],
                    "response_event_id": response_event_id,
                    "player_id": ev["player_id"],
                    "team_id": ev["team_id"],
                    "season_year": ev["season_year"],
                    "state": st_map.get(ev["player_id"]),
                }

            # Load player state
            prev_states = get_player_agency_states(cur, [ev["player_id"]])
            prev_state = prev_states.get(ev["player_id"]) or _default_state_for_event(ev)

            # Load player mental
            prof = _load_player_profile(cur, ev["player_id"], cfg=cfg)
            mental = prof.get("mental") or {}

            # Apply response (pure logic)
            outcome = apply_user_response(
                event=ev,
                state=prev_state,
                mental=mental,
                response_type=response_type,
                response_payload=response_payload,
                now_date_iso=now_date,
                cfg=cfg,
                rcfg=rcfg,
            )

            if not outcome.ok:
                raise AgencyInteractionError(
                    BAD_RESPONSE,
                    "Invalid response for this event",
                    {
                        "event_id": ev["event_id"],
                        "event_type": ev["event_type"],
                        "response_type": str(response_type),
                        "reasons": outcome.reasons,
                    },
                )

            # Promise table optional (enforced only when needed)
            promise_table_exists = _table_exists(cur, "player_agency_promises")
            if outcome.promise is not None and not promise_table_exists and strict_promises:
                raise AgencyInteractionError(
                    PROMISE_SCHEMA_MISSING,
                    "player_agency_promises table is missing (required for promise responses)",
                    {"required_table": "player_agency_promises"},
                )

            # Persist response marker (optional table; nice-to-have)
            response_table_exists = _table_exists(cur, "agency_event_responses")

            response_id = make_event_id("agency_resp", ev["event_id"])  # deterministic per source event
            if response_table_exists:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO agency_event_responses(
                        response_id,
                        source_event_id,
                        player_id,
                        team_id,
                        season_year,
                        response_type,
                        response_payload_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        response_id,
                        ev["event_id"],
                        ev["player_id"],
                        ev["team_id"],
                        int(ev["season_year"]),
                        str(outcome.response_type),
                        json_dumps(dict(response_payload or {})),
                        str(now_iso),
                    ),
                )

            # Update player agency state (keep all other fields intact)
            new_state = dict(prev_state)
            new_state.update(outcome.state_updates or {})
            new_state["player_id"] = ev["player_id"]
            new_state["team_id"] = ev["team_id"]
            new_state["season_year"] = int(ev["season_year"])

            # Defensive clamp
            new_state["trust"] = float(clamp01(new_state.get("trust", 0.5)))
            new_state["minutes_frustration"] = float(clamp01(new_state.get("minutes_frustration", 0.0)))
            new_state["team_frustration"] = float(clamp01(new_state.get("team_frustration", 0.0)))
            try:
                new_state["trade_request_level"] = int(max(0, min(2, int(new_state.get("trade_request_level") or 0))))
            except Exception:
                new_state["trade_request_level"] = int(prev_state.get("trade_request_level") or 0)

            upsert_player_agency_states(cur, {ev["player_id"]: new_state}, now=str(now_iso))

            # Insert response event into agency_events (UI + analytics)
            trust_delta = safe_float(outcome.meta.get("deltas", {}).get("trust"), 0.0) if isinstance(outcome.meta, Mapping) else 0.0
            sev = clamp01(abs(trust_delta) * 4.0)

            response_event = {
                "event_id": response_event_id,
                "player_id": ev["player_id"],
                "team_id": ev["team_id"],
                "season_year": int(ev["season_year"]),
                "date": str(now_date)[:10],
                "event_type": "USER_RESPONSE",
                "severity": float(sev),
                "payload": {
                    "source_event_id": ev["event_id"],
                    "source_event_type": ev["event_type"],
                    "response_id": response_id,
                    "response_type": str(outcome.response_type),
                    "response_payload": dict(response_payload or {}),
                    "tone": outcome.tone,
                    "player_reply": outcome.player_reply,
                    "reasons": outcome.reasons,
                    "state_updates": outcome.state_updates,
                    "meta": outcome.meta,
                },
            }

            events_to_insert = [response_event]

            promise_row: Optional[Dict[str, Any]] = None
            promise_id: Optional[str] = None

            if outcome.promise is not None:
                promise_spec = outcome.promise
                promise_id = make_event_id("agency_promise", response_id, promise_spec.promise_type)

                # Persist promise row if schema exists
                if promise_table_exists:
                    promise_row = {
                        "promise_id": promise_id,
                        "player_id": ev["player_id"],
                        "team_id": ev["team_id"],
                        "season_year": int(ev["season_year"]),
                        "source_event_id": ev["event_id"],
                        "response_id": response_id,
                        "promise_type": str(promise_spec.promise_type),
                        "status": "ACTIVE",
                        "created_date": str(now_date)[:10],
                        "due_month": str(promise_spec.due_month),
                        "target_value": promise_spec.target_value,
                        "target_json": dict(promise_spec.target or {}),
                        "evidence_json": {},
                        "resolved_at": None,
                    }

                    cur.execute(
                        """
                        INSERT OR IGNORE INTO player_agency_promises(
                            promise_id,
                            player_id,
                            team_id,
                            season_year,
                            source_event_id,
                            response_id,
                            promise_type,
                            status,
                            created_date,
                            due_month,
                            target_value,
                            target_json,
                            evidence_json,
                            resolved_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            promise_row["promise_id"],
                            promise_row["player_id"],
                            promise_row["team_id"],
                            int(promise_row["season_year"]),
                            promise_row["source_event_id"],
                            promise_row["response_id"],
                            promise_row["promise_type"],
                            promise_row["status"],
                            promise_row["created_date"],
                            promise_row["due_month"],
                            promise_row["target_value"],
                            json_dumps(promise_row["target_json"]),
                            json_dumps(promise_row["evidence_json"]),
                            promise_row["resolved_at"],
                        ),
                    )

                # Log promise creation as an event (even if promise table is missing)
                promise_event_id = make_event_id("agency", "promise", promise_id)
                events_to_insert.append(
                    {
                        "event_id": promise_event_id,
                        "player_id": ev["player_id"],
                        "team_id": ev["team_id"],
                        "season_year": int(ev["season_year"]),
                        "date": str(now_date)[:10],
                        "event_type": "PROMISE_CREATED",
                        "severity": float(clamp01(0.20 + 0.60 * safe_float(ev.get("severity"), 0.0))),
                        "payload": {
                            "promise_id": promise_id,
                            "promise_type": str(promise_spec.promise_type),
                            "due_month": str(promise_spec.due_month),
                            "target_value": promise_spec.target_value,
                            "target": dict(promise_spec.target or {}),
                            "source_event_id": ev["event_id"],
                            "response_event_id": response_event_id,
                        },
                    }
                )

            insert_agency_events(cur, events_to_insert, now=str(now_iso))

            return {
                "ok": True,
                "skipped": False,
                "event_id": ev["event_id"],
                "response_event_id": response_event_id,
                "response_id": response_id,
                "player_id": ev["player_id"],
                "team_id": ev["team_id"],
                "season_year": ev["season_year"],
                "outcome": {
                    "event_type": outcome.event_type,
                    "response_type": outcome.response_type,
                    "tone": outcome.tone,
                    "player_reply": outcome.player_reply,
                    "reasons": outcome.reasons,
                    "meta": outcome.meta,
                },
                "promise": {
                    "promise_id": promise_id,
                    "persisted": bool(promise_table_exists and promise_row is not None),
                    "promise_type": outcome.promise.promise_type if outcome.promise else None,
                    "due_month": outcome.promise.due_month if outcome.promise else None,
                }
                if outcome.promise is not None
                else None,
                "state": {
                    "trust": new_state.get("trust"),
                    "minutes_frustration": new_state.get("minutes_frustration"),
                    "team_frustration": new_state.get("team_frustration"),
                    "trade_request_level": new_state.get("trade_request_level"),
                },
            }
