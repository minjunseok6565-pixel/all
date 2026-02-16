# player_ego/service.py
"""Player Ego subsystem: orchestration service.

This service ties together:
- SQLite persistence (player_ego/store.py)
- pure logic (player_ego/logic.py)
- transactions_log emission (player_ego/events.py)

It is designed to be called from existing simulation flows:
- After ingest_game_result (per-game updates)
- During offseason contract option processing (player option decisions)
- When evaluating re-sign/extension offers

The service is safe to call either:
- within an existing DB transaction (pass `cur`), or
- standalone (it will open its own repo.transaction())

Notes
-----
- This module avoids importing LeagueService to prevent circular imports.
- To keep the system usable even before integration, some parameters are optional.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import game_time

from . import events, store
from .logic import (
    calc_leverage,
    compute_team_win_pct_from_master_schedule,
    decide_player_option as decide_player_option_logic,
    evaluate_re_sign_offer as evaluate_offer_logic,
    update_after_game,
)
from .types import ContractOffer, EgoIssue, OfferDecision


def _json_dumps(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


def _insert_transactions_in_cur(cur: sqlite3.Cursor, entries: Sequence[Mapping[str, Any]]) -> None:
    """Insert into transactions_log within an existing cursor.

    We mirror the hashing/shape from LeagueRepo.insert_transactions.
    """
    if not entries:
        return
    now = game_time.now_utc_like_iso()
    rows: List[Tuple[str, str, Optional[str], Optional[int], Optional[str], Optional[str], str, str, str]] = []

    for e in entries:
        if not isinstance(e, dict):
            e = dict(e)
        payload = _json_dumps(e)
        tx_hash = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        sy = e.get("season_year")
        try:
            sy_i = int(sy) if sy is not None and str(sy) != "" else None
        except Exception:
            sy_i = None

        rows.append(
            (
                tx_hash,
                str(e.get("type") or "unknown"),
                str(e.get("date") or "") if e.get("date") is not None else None,
                sy_i,
                str(e.get("deal_id") or "") if e.get("deal_id") is not None else None,
                str(e.get("source") or "") if e.get("source") is not None else None,
                _json_dumps(e.get("teams") or []),
                payload,
                now,
            )
        )

    cur.executemany(
        """
        INSERT OR IGNORE INTO transactions_log(
            tx_hash, tx_type, tx_date, season_year, deal_id, source, teams_json, payload_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        rows,
    )


def _extract_minutes_map(team_game: Mapping[str, Any]) -> Dict[str, float]:
    """Extract player_id -> minutes from a v2 game_result team object."""
    out: Dict[str, float] = {}
    players = team_game.get("players") or []
    if not isinstance(players, list):
        return out
    for row in players:
        if not isinstance(row, Mapping):
            continue
        pid = row.get("PlayerID") or row.get("player_id")
        if pid is None:
            continue
        try:
            mins = float(row.get("MIN") or 0.0)
        except Exception:
            mins = 0.0
        out[str(pid)] = max(0.0, mins)
    return out


class PlayerEgoService:
    """High-level API for the Player Ego subsystem."""

    def __init__(self, repo: Any):
        self.repo = repo

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def ensure_seeded(self, *, date_iso: Optional[str] = None, cur: Optional[sqlite3.Cursor] = None) -> int:
        """Ensure all players have ego rows. Returns number of created rows."""
        return store.ensure_seeded_for_all_players(self.repo, date_iso=date_iso, cur=cur)

    # ------------------------------------------------------------------
    # Core hooks
    # ------------------------------------------------------------------

    def on_game_result(
        self,
        game_result_v2: Mapping[str, Any],
        *,
        game_date_iso: Optional[str] = None,
        game_state_context: Optional[Mapping[str, Any]] = None,
        season_year: Optional[int] = None,
        cur: Optional[sqlite3.Cursor] = None,
    ) -> Dict[str, Any]:
        """Update ego for players involved in a game.

        Parameters
        ----------
        game_result_v2:
            state_modules/state_results validated v2 game result.
        game_date_iso:
            If omitted, uses game_result_v2['game']['date'].
        game_state_context:
            Optional full state snapshot. If None, will try state.export_full_state_snapshot().
            Used only for team record (win%).
        season_year:
            Optional season year for logging.
        cur:
            Optional existing DB cursor (recommended when called from LeagueService).

        Returns
        -------
        dict:
            Summary with counts and new tx payloads.
        """
        game = game_result_v2.get("game") or {}
        if not isinstance(game, Mapping):
            game = {}
        date_iso = str(game_date_iso or game.get("date") or game_time.game_date_iso())[:10]

        if season_year is None:
            # Infer from state snapshot when available.
            sy = None
            if isinstance(game_state_context, Mapping):
                league = game_state_context.get("league")
                if isinstance(league, Mapping):
                    sy = league.get("season_year")
            if sy is not None:
                try:
                    season_year = int(sy)
                except Exception:
                    season_year = None

        # Lazy state snapshot fallback
        if game_state_context is None:
            try:
                import state

                game_state_context = state.export_full_state_snapshot()
            except Exception:
                game_state_context = None

        teams = game_result_v2.get("teams") or {}
        if not isinstance(teams, Mapping):
            teams = {}

        # Determine team ids
        home_id = str(game.get("home_team_id") or "").upper()
        away_id = str(game.get("away_team_id") or "").upper()
        team_ids = [tid for tid in (home_id, away_id) if tid]

        new_txs: List[Dict[str, Any]] = []
        updated_players: int = 0
        new_issues: int = 0

        def _impl(c: sqlite3.Cursor) -> Dict[str, Any]:
            nonlocal updated_players, new_issues

            for tid in team_ids:
                team_game = teams.get(tid) or {}
                if not isinstance(team_game, Mapping):
                    team_game = {}

                minutes_map = _extract_minutes_map(team_game)

                # Load roster snapshot for leverage computation (join players for ovr)
                roster_rows = c.execute(
                    """
                    SELECT p.player_id, p.ovr, r.salary_amount
                    FROM roster r
                    JOIN players p ON p.player_id = r.player_id
                    WHERE r.team_id=? AND r.status='active'
                    ORDER BY p.ovr DESC, p.player_id ASC;
                    """,
                    (tid,),
                ).fetchall()

                roster: List[Dict[str, Any]] = []
                for r in roster_rows:
                    roster.append(
                        {
                            "player_id": str(r["player_id"]),
                            "ovr": r["ovr"],
                            "salary_amount": r["salary_amount"],
                        }
                    )

                win_pct = None
                if isinstance(game_state_context, Mapping):
                    win_pct = compute_team_win_pct_from_master_schedule(game_state_context, team_id=tid)

                # Update each roster player (including DNP -> 0 minutes)
                for r in roster:
                    pid = str(r["player_id"])
                    mins = float(minutes_map.get(pid, 0.0))
                    salary_amount = r.get("salary_amount")
                    ovr = r.get("ovr")

                    ego = store.get_or_create_player_ego(
                        self.repo,
                        pid,
                        team_id=tid,
                        date_iso=date_iso,
                        cur=c,
                    )

                    leverage = calc_leverage(
                        player_id=pid,
                        player_ovr=ovr,
                        team_roster=roster,
                        salary_amount=int(salary_amount) if salary_amount is not None else None,
                    )

                    result = update_after_game(
                        player={"player_id": pid, "ovr": ovr},
                        traits_in=ego["traits"],
                        state_in=ego["state"],
                        date_iso=date_iso,
                        team_id=tid,
                        minutes_played=mins,
                        team_win_pct=win_pct,
                        salary_amount=int(salary_amount) if salary_amount is not None else None,
                        leverage=float(leverage),
                    )

                    store.upsert_player_ego(self.repo, pid, result.traits, result.state, cur=c)
                    updated_players += 1

                    for issue in result.new_issues:
                        new_issues += 1
                        if issue.get("type") == "TRADE_REQUEST":
                            tx = events.build_trade_request_tx(
                                issue=issue,
                                player_id=pid,
                                team_id=tid,
                                date_iso=date_iso,
                                season_year=season_year,
                            )
                        else:
                            tx = events.build_issue_tx(
                                issue=issue,
                                player_id=pid,
                                team_id=tid,
                                date_iso=date_iso,
                                season_year=season_year,
                            )
                        new_txs.append(tx)

            if new_txs:
                _insert_transactions_in_cur(c, new_txs)

            return {
                "date": date_iso,
                "season_year": season_year,
                "updated_players": updated_players,
                "new_issues": new_issues,
                "new_transactions": new_txs,
            }

        if cur is not None:
            return _impl(cur)

        tx = getattr(self.repo, "transaction", None)
        if tx is None:
            raise TypeError("repo must provide transaction()")
        with tx() as c:
            return _impl(c)

    # ------------------------------------------------------------------
    # Contracts
    # ------------------------------------------------------------------

    def decide_player_option(
        self,
        *,
        option: Mapping[str, Any],
        contract: Mapping[str, Any],
        player_id: str,
        team_id: Optional[str] = None,
        game_state_context: Optional[Mapping[str, Any]] = None,
        decision_date_iso: Optional[str] = None,
        cur: Optional[sqlite3.Cursor] = None,
    ) -> str:
        """Decide PLAYER/ETO option ("EXERCISE" or "DECLINE")."""
        pid = str(player_id)
        date_iso = str(decision_date_iso or game_time.game_date_iso())[:10]

        if game_state_context is None:
            try:
                import state

                game_state_context = state.export_full_state_snapshot()
            except Exception:
                game_state_context = None

        def _impl(c: sqlite3.Cursor) -> str:
            team = team_id
            if team is None:
                try:
                    row = c.execute(
                        "SELECT team_id FROM roster WHERE player_id=? AND status='active';",
                        (pid,),
                    ).fetchone()
                    if row:
                        team = str(row["team_id"]).upper()
                except Exception:
                    team = None

            prow = c.execute("SELECT player_id, age, ovr FROM players WHERE player_id=?;", (pid,)).fetchone()
            if not prow:
                raise KeyError(f"player not found: {player_id}")
            player = {"player_id": pid, "age": prow["age"], "ovr": prow["ovr"]}

            ego = store.get_or_create_player_ego(self.repo, pid, team_id=team, date_iso=date_iso, cur=c)

            win_pct = None
            if team and isinstance(game_state_context, Mapping):
                win_pct = compute_team_win_pct_from_master_schedule(game_state_context, team_id=team)

            return decide_player_option_logic(
                option=option,
                contract=contract,
                player=player,
                traits_in=ego["traits"],
                state_in=ego["state"],
                team_win_pct=win_pct,
                decision_date_iso=date_iso,
            )

        return store.with_cursor(self.repo, cur, _impl)

    def evaluate_re_sign_offer(
        self,
        *,
        team_id: str,
        player_id: str,
        offer: ContractOffer,
        game_state_context: Optional[Mapping[str, Any]] = None,
        decision_date_iso: Optional[str] = None,
        season_year: Optional[int] = None,
        cur: Optional[sqlite3.Cursor] = None,
        log_transaction: bool = True,
    ) -> OfferDecision:
        """Evaluate a re-sign / extension offer and optionally log a transaction."""
        pid = str(player_id)
        tid = str(team_id).upper()
        date_iso = str(decision_date_iso or game_time.game_date_iso())[:10]

        if game_state_context is None:
            try:
                import state

                game_state_context = state.export_full_state_snapshot()
            except Exception:
                game_state_context = None

        def _impl(c: sqlite3.Cursor) -> OfferDecision:
            prow = c.execute("SELECT player_id, age, ovr FROM players WHERE player_id=?;", (pid,)).fetchone()
            if not prow:
                raise KeyError(f"player not found: {player_id}")
            player = {"player_id": pid, "age": prow["age"], "ovr": prow["ovr"]}

            roster_rows = c.execute(
                """
                SELECT p.player_id, p.ovr, r.salary_amount
                FROM roster r
                JOIN players p ON p.player_id = r.player_id
                WHERE r.team_id=? AND r.status='active'
                ORDER BY p.ovr DESC, p.player_id ASC;
                """,
                (tid,),
            ).fetchall()
            roster: List[Dict[str, Any]] = [
                {"player_id": str(r["player_id"]), "ovr": r["ovr"], "salary_amount": r["salary_amount"]}
                for r in roster_rows
            ]

            salary_amount = None
            for r in roster:
                if r.get("player_id") == pid:
                    salary_amount = r.get("salary_amount")
                    break

            leverage = calc_leverage(
                player_id=pid,
                player_ovr=player.get("ovr"),
                team_roster=roster,
                salary_amount=int(salary_amount) if salary_amount is not None else None,
            )

            ego = store.get_or_create_player_ego(self.repo, pid, team_id=tid, date_iso=date_iso, cur=c)

            win_pct = None
            if isinstance(game_state_context, Mapping):
                win_pct = compute_team_win_pct_from_master_schedule(game_state_context, team_id=tid)

            decision = evaluate_offer_logic(
                player=player,
                team_id=tid,
                offer=offer,
                traits_in=ego["traits"],
                state_in=ego["state"],
                team_win_pct=win_pct,
                decision_date_iso=date_iso,
                leverage=float(leverage),
            )

            if log_transaction:
                tx_payload = events.build_offer_response_tx(
                    player_id=pid,
                    team_id=tid,
                    date_iso=date_iso,
                    season_year=season_year,
                    decision=decision,
                    offered=offer,
                )
                _insert_transactions_in_cur(c, [tx_payload])

            return decision

        return store.with_cursor(self.repo, cur, _impl)

    # ------------------------------------------------------------------
    # Team changes
    # ------------------------------------------------------------------

    def on_team_changed(
        self,
        *,
        player_id: str,
        from_team: Optional[str],
        to_team: Optional[str],
        date_iso: Optional[str] = None,
        cur: Optional[sqlite3.Cursor] = None,
    ) -> None:
        """Adjust ego state when a player changes teams.

        Current behavior:
        - resolves open TRADE_REQUEST issues
        - nudges happiness up slightly (fresh start)
        """
        pid = str(player_id)
        d = str(date_iso or game_time.game_date_iso())[:10]

        def _impl(c: sqlite3.Cursor) -> None:
            ego = store.get_or_create_player_ego(self.repo, pid, team_id=to_team, date_iso=d, cur=c)
            state = dict(ego["state"])

            new_open: List[EgoIssue] = []
            for it in state.get("open_issues") or []:
                if not isinstance(it, dict):
                    continue
                if it.get("type") == "TRADE_REQUEST" and it.get("status") == "OPEN":
                    it = dict(it)
                    it["status"] = "RESOLVED"
                    it["updated_date"] = d
                    continue
                new_open.append(it)
            state["open_issues"] = new_open

            try:
                h = float(state.get("happiness") or 0.7)
            except Exception:
                h = 0.7
            state["happiness"] = min(1.0, h + 0.04)
            state["last_team_id"] = str(to_team).upper() if to_team else None
            state["last_updated_date"] = d

            store.upsert_player_ego(self.repo, pid, ego["traits"], state, cur=c)

        store.with_cursor(self.repo, cur, _impl)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_team_open_issues(
        self,
        *,
        team_id: str,
        cur: Optional[sqlite3.Cursor] = None,
    ) -> List[Dict[str, Any]]:
        """List open issues for a team (for UI)."""
        tid = str(team_id).upper()

        def _impl(c: sqlite3.Cursor) -> List[Dict[str, Any]]:
            rows = c.execute(
                "SELECT player_id FROM roster WHERE team_id=? AND status='active' ORDER BY player_id;",
                (tid,),
            ).fetchall()
            pids = [str(r["player_id"]) for r in rows]
            ego_map = store.bulk_get_player_ego(self.repo, pids, cur=c)

            out: List[Dict[str, Any]] = []
            for pid in pids:
                ego = ego_map.get(pid)
                if not ego:
                    continue
                for issue in ego["state"].get("open_issues") or []:
                    if not isinstance(issue, dict):
                        continue
                    if issue.get("status") != "OPEN":
                        continue
                    out.append({"team_id": tid, "player_id": pid, "issue": dict(issue)})

            def _key(x: Dict[str, Any]) -> Tuple[int, float, str]:
                it = x.get("issue") or {}
                t = it.get("type")
                pri = 0 if t == "TRADE_REQUEST" else 1
                try:
                    sev = float(it.get("severity") or 0.0)
                except Exception:
                    sev = 0.0
                return (pri, -sev, str(x.get("player_id")))

            out.sort(key=_key)
            return out

        return store.with_cursor(self.repo, cur, _impl)
