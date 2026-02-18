# db_schema/agency.py
"""SQLite SSOT schema: player agency.

This module introduces:
  - player_agency_state (current player "agency" state: frustrations, expectations, cooldowns)
  - agency_events (append-only log of agency-related events)

Notes
-----
* These tables reference `players` via player_id foreign key, so this module must
  be applied after db_schema.core.
* Dates are stored as ISO strings (YYYY-MM-DD). Month keys are stored as YYYY-MM.
* agency_events is append-only and keyed by event_id for idempotency.

Design goals
------------
- Keep the schema small but extensible.
- Make reads cheap for UI (team feed, player feed).
- Make writes safe and idempotent (event_id primary key + INSERT OR IGNORE).
"""

from __future__ import annotations


def ddl(*, now: str, schema_version: str) -> str:  # noqa: ARG001
    """Return DDL SQL for agency tables (as a single executescript string)."""

    return """

                CREATE TABLE IF NOT EXISTS player_agency_state (
                    player_id TEXT PRIMARY KEY,
                    team_id TEXT NOT NULL,
                    season_year INTEGER NOT NULL,

                    role_bucket TEXT NOT NULL DEFAULT 'UNKNOWN'
                        CHECK(role_bucket IN ('UNKNOWN','FRANCHISE','STAR','STARTER','ROTATION','BENCH','GARBAGE')),
                    leverage REAL NOT NULL DEFAULT 0.0
                        CHECK(leverage >= 0.0 AND leverage <= 1.0),

                    minutes_expected_mpg REAL NOT NULL DEFAULT 0.0,
                    minutes_actual_mpg REAL NOT NULL DEFAULT 0.0,

                    minutes_frustration REAL NOT NULL DEFAULT 0.0
                        CHECK(minutes_frustration >= 0.0 AND minutes_frustration <= 1.0),
                    team_frustration REAL NOT NULL DEFAULT 0.0
                        CHECK(team_frustration >= 0.0 AND team_frustration <= 1.0),
                    trust REAL NOT NULL DEFAULT 0.5
                        CHECK(trust >= 0.0 AND trust <= 1.0),

                    trade_request_level INTEGER NOT NULL DEFAULT 0
                        CHECK(trade_request_level IN (0,1,2)),

                    cooldown_minutes_until TEXT,
                    cooldown_trade_until TEXT,
                    cooldown_help_until TEXT,
                    cooldown_contract_until TEXT,

                    last_processed_month TEXT,

                    context_json TEXT NOT NULL DEFAULT '{}',

                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    FOREIGN KEY(player_id) REFERENCES players(player_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_player_agency_state_team
                    ON player_agency_state(team_id);

                CREATE INDEX IF NOT EXISTS idx_player_agency_state_team_tradelevel
                    ON player_agency_state(team_id, trade_request_level);

                CREATE INDEX IF NOT EXISTS idx_player_agency_state_team_frustration
                    ON player_agency_state(team_id, minutes_frustration, team_frustration);


                CREATE TABLE IF NOT EXISTS agency_events (
                    event_id TEXT PRIMARY KEY,
                    player_id TEXT NOT NULL,
                    team_id TEXT NOT NULL,
                    season_year INTEGER NOT NULL,
                    date TEXT NOT NULL,

                    event_type TEXT NOT NULL,
                    severity REAL NOT NULL DEFAULT 0.0,

                    payload_json TEXT NOT NULL DEFAULT '{}',

                    created_at TEXT NOT NULL,

                    FOREIGN KEY(player_id) REFERENCES players(player_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_agency_events_player_date
                    ON agency_events(player_id, date);

                CREATE INDEX IF NOT EXISTS idx_agency_events_team_date
                    ON agency_events(team_id, date);

                CREATE INDEX IF NOT EXISTS idx_agency_events_type_date
                    ON agency_events(event_type, date);


                -- User responses to agency events (idempotency + UI state)
                CREATE TABLE IF NOT EXISTS agency_event_responses (
                    response_id TEXT PRIMARY KEY,
                    source_event_id TEXT NOT NULL UNIQUE,
                    player_id TEXT NOT NULL,
                    team_id TEXT NOT NULL,
                    season_year INTEGER NOT NULL,
                    response_type TEXT NOT NULL,
                    response_payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,

                    FOREIGN KEY(player_id) REFERENCES players(player_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_agency_event_responses_player
                    ON agency_event_responses(player_id);

                CREATE INDEX IF NOT EXISTS idx_agency_event_responses_team
                    ON agency_event_responses(team_id);

                CREATE INDEX IF NOT EXISTS idx_agency_event_responses_source
                    ON agency_event_responses(source_event_id);


                -- Promises created by user responses, resolved later (monthly tick)
                CREATE TABLE IF NOT EXISTS player_agency_promises (
                    promise_id TEXT PRIMARY KEY,
                    player_id TEXT NOT NULL,
                    team_id TEXT NOT NULL,
                    season_year INTEGER NOT NULL,

                    source_event_id TEXT,
                    response_id TEXT,

                    promise_type TEXT NOT NULL
                        CHECK(promise_type IN ('MINUTES','HELP','SHOP_TRADE','ROLE')),

                    status TEXT NOT NULL DEFAULT 'ACTIVE'
                        CHECK(status IN ('ACTIVE','FULFILLED','BROKEN','EXPIRED','CANCELLED')),

                    created_date TEXT NOT NULL,
                    due_month TEXT NOT NULL,

                    target_value REAL,
                    target_json TEXT NOT NULL DEFAULT '{}',
                    evidence_json TEXT NOT NULL DEFAULT '{}',

                    resolved_at TEXT,

                    FOREIGN KEY(player_id) REFERENCES players(player_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_player_agency_promises_player_status_due
                    ON player_agency_promises(player_id, status, due_month);

                CREATE INDEX IF NOT EXISTS idx_player_agency_promises_team_status_due
                    ON player_agency_promises(team_id, status, due_month);

                CREATE INDEX IF NOT EXISTS idx_player_agency_promises_due
                    ON player_agency_promises(status, due_month);

"""
