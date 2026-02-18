from __future__ import annotations

"""DB-backed orchestration for the agency subsystem.

This module wires together:
- DB reads (players/roster + existing agency state)
- expectation computation (role/leverage/expected minutes)
- monthly tick logic (tick.apply_monthly_player_tick)
- DB writes (upsert state + append events)

It is designed to be called from a checkpoint trigger (see agency/checkpoints.py)
when a month finishes.
"""

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

from league_repo import LeagueRepo

from .config import AgencyConfig, DEFAULT_CONFIG
from .expectations import compute_expectations_for_league
from .expectations_month import compute_month_expectations
from .month_context import PlayerMonthSplit, build_split_summary, players_by_team_from_splits
from .promises import DEFAULT_PROMISE_CONFIG, PromiseEvaluationContext, evaluate_promise
from .repo import (
    get_player_agency_states,
    insert_agency_events,
    list_active_promises_due,
    update_promises,
    upsert_player_agency_states,
)
from .team_transition import apply_team_transition
from .tick import apply_monthly_player_tick
from .types import MonthlyPlayerInputs
from .utils import clamp01, extract_mental_from_attrs, json_loads, make_event_id, norm_date_iso, safe_float, safe_int


logger = logging.getLogger(__name__)


def _meta_key_for_month(month_key: str) -> str:
    return f"nba_agency_tick_done_{str(month_key)}"


def _best_effort_injury_status_by_pid(repo: LeagueRepo, player_ids: list[str]) -> Dict[str, str]:
    """Return mapping pid -> injury status (HEALTHY/OUT/RETURNING).

    This is best-effort: if injury tables are missing, returns empty.
    """
    if not player_ids:
        return {}

    try:
        placeholders = ",".join(["?"] * len(player_ids))
        rows = repo._conn.execute(
            f"""
            SELECT player_id, status
            FROM player_injury_state
            WHERE player_id IN ({placeholders});
            """,
            [*player_ids],
        ).fetchall()
        out: Dict[str, str] = {}
        for pid, status in rows:
            pid_s = str(pid)
            if not pid_s:
                continue
            out[pid_s] = str(status or "HEALTHY").upper()
        return out
    except Exception:
        return {}


def _typed_splits(month_splits_by_player: Optional[Mapping[str, Any]]) -> Dict[str, PlayerMonthSplit]:
    out: Dict[str, PlayerMonthSplit] = {}
    for pid, sp in (month_splits_by_player or {}).items():
        if isinstance(sp, PlayerMonthSplit):
            out[str(pid)] = sp
    return out


def _slice_minutes_games(split: Optional[PlayerMonthSplit], team_id: str) -> Tuple[float, int]:
    """Return (minutes, games_played) for player on a specific team in the month."""
    if split is None:
        return (0.0, 0)
    tid = str(team_id or "").upper()
    if not tid:
        return (0.0, 0)
    sl = (split.teams or {}).get(tid)
    if sl is None:
        return (0.0, 0)
    return (float(sl.minutes), int(sl.games_played))


def apply_monthly_agency_tick(
    *,
    db_path: str,
    season_year: int,
    month_key: str,
    month_splits_by_player: Optional[Mapping[str, Any]] = None,
    # Backward-compatible fallback (deprecated): if month_splits are not provided
    # callers may provide aggregated per-player minutes/games.
    minutes_by_player: Optional[Mapping[str, float]] = None,
    games_by_player: Optional[Mapping[str, int]] = None,
    team_win_pct_by_team: Optional[Mapping[str, float]] = None,
    now_iso: str,
    cfg: AgencyConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Apply a league-wide monthly agency tick.

    Idempotent via meta key `nba_agency_tick_done_{month_key}`.

    Args:
        db_path: SQLite path
        season_year: current season year
        month_key: YYYY-MM (the month being processed)
        month_splits_by_player: per-player month/team splits (preferred)
        minutes_by_player: total minutes played in that month (fallback)
        games_by_player: games played in that month (fallback)
        team_win_pct_by_team: team win% for that month (optional)
        now_iso: UTC-like timestamp string for created_at/updated_at/meta
        cfg: AgencyConfig
    """

    sy = int(season_year)
    mk = str(month_key)
    meta_key = _meta_key_for_month(mk)

    minutes_map = {str(k): float(v) for k, v in (minutes_by_player or {}).items() if str(k)} if minutes_by_player else {}
    games_map = {str(k): int(v) for k, v in (games_by_player or {}).items() if str(k)} if games_by_player else {}
    team_win_map = {str(k).upper(): float(v) for k, v in (team_win_pct_by_team or {}).items() if str(k)}

    splits_by_pid = _typed_splits(month_splits_by_player)
    month_players_by_team = players_by_team_from_splits(splits_by_pid, min_games_present=1) if splits_by_pid else {}

    with LeagueRepo(db_path) as repo:
        repo.init_db()

        # Idempotency
        row = repo._conn.execute("SELECT value FROM meta WHERE key=?;", (meta_key,)).fetchone()
        if row and str(row[0]) == "1":
            return {"ok": True, "skipped": True, "reason": "already_done", "month": mk, "meta_key": meta_key}

        # Ensure agency tables exist. Fail loud with a helpful message.
        try:
            repo._conn.execute("SELECT 1 FROM player_agency_state LIMIT 1;")
        except Exception as exc:
            raise RuntimeError(
                "Agency schema is missing. Did you add db_schema.agency to db_schema.init.DEFAULT_MODULES?"
            ) from exc

        # Load roster + player data.
        rows = repo._conn.execute(
            """
            SELECT
                p.player_id,
                r.team_id,
                r.salary_amount,
                p.ovr,
                p.age,
                p.attrs_json
            FROM players p
            JOIN roster r ON r.player_id = p.player_id
            WHERE r.status='active'
            ORDER BY r.team_id ASC, p.player_id ASC;
            """
        ).fetchall()

        roster_rows: list[Dict[str, Any]] = []
        player_ids: list[str] = []
        for r in rows:
            pid = str(r[0] or "")
            tid = str(r[1] or "").upper()
            if not pid or not tid:
                continue
            player_ids.append(pid)
            roster_rows.append(
                {
                    "player_id": pid,
                    "team_id": tid,
                    "salary_amount": safe_float(r[2], 0.0),
                    "ovr": safe_int(r[3], 0),
                    "age": safe_int(r[4], 0),
                    "attrs_json": r[5],
                }
            )

        if not roster_rows:
            # Still write meta key to avoid reprocessing empty months.
            with repo.transaction() as cur:
                cur.execute(
                    "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                    (meta_key, "1"),
                )
            return {"ok": True, "skipped": True, "reason": "no_active_roster", "month": mk, "meta_key": meta_key}

        # Expectations for role/leverage/expected minutes.
        expectations_current = compute_expectations_for_league(roster_rows, config=cfg.expectations)

        # Month expectations (role/leverage/expected minutes in the roster context actually experienced).
        # If month split data is unavailable, this will be empty and we will fall back to current expectations.
        with repo.transaction() as cur:
            month_expectations = compute_month_expectations(
                cur,
                players_by_team=month_players_by_team,
                config=cfg.expectations,
            )

        # Injury status (best-effort)
        injury_status_by_pid = _best_effort_injury_status_by_pid(repo, player_ids)

        # Load previous agency states
        with repo.transaction() as cur:
            prev_states = get_player_agency_states(cur, player_ids)

        # Run tick (pass 1): evaluate each player's month under the team they actually played for.
        # We may later apply a team transition to move the state onto the current roster team.
        states_eval: Dict[str, Dict[str, Any]] = {}
        events: list[Dict[str, Any]] = []
        mental_by_pid: Dict[str, Dict[str, int]] = {}

        eval_team_by_pid: Dict[str, str] = {}
        roster_team_by_pid: Dict[str, str] = {}
        team_end_by_pid: Dict[str, str] = {}
        split_summary_by_pid: Dict[str, Dict[str, Any]] = {}

        for rr in roster_rows:
            pid = str(rr["player_id"])
            roster_tid = str(rr["team_id"]).upper()
            roster_team_by_pid[pid] = roster_tid

            split = splits_by_pid.get(pid)
            eval_tid = (split.primary_team if split and split.primary_team else roster_tid) or roster_tid
            eval_tid = str(eval_tid).upper() if eval_tid else roster_tid
            eval_team_by_pid[pid] = eval_tid

            # Team at end-of-month for promise evaluation (trade/shop promises) & explainability.
            team_end = None
            if split and split.team_last:
                team_end = str(split.team_last).upper()
            elif split and split.primary_team:
                team_end = str(split.primary_team).upper()
            else:
                team_end = eval_tid
            team_end_by_pid[pid] = str(team_end or eval_tid).upper()

            if split:
                split_summary_by_pid[pid] = build_split_summary(split)

            # Month expectation for evaluated team (preferred)
            exp_m = month_expectations.get((pid, eval_tid))

            # Fallback: current expectations (useful for players with no appearances in month splits)
            exp_c = expectations_current.get(pid)

            if exp_m is not None:
                role_bucket = str(exp_m.role_bucket)
                leverage = float(exp_m.leverage)
                expected_mpg = float(exp_m.expected_mpg)
            elif exp_c is not None:
                role_bucket = str(exp_c.role_bucket)
                leverage = float(exp_c.leverage)
                expected_mpg = float(exp_c.expected_mpg)
            else:
                role_bucket = "UNKNOWN"
                leverage = 0.0
                expected_mpg = float(cfg.expectations.expected_mpg_by_role.get("UNKNOWN", 12.0))

            mental = extract_mental_from_attrs(rr.get("attrs_json"), keys=cfg.mental_attr_keys)
            mental_by_pid[pid] = dict(mental)

            # Month actuals for evaluated team.
            if split is not None:
                mins_eval, gp_eval = _slice_minutes_games(split, eval_tid)
            else:
                # Deprecated fallback (pre-attribution): aggregated per-player minutes/games (may misattribute).
                mins_eval = float(minutes_map.get(pid, 0.0))
                gp_eval = int(games_map.get(pid, 0))

            inp = MonthlyPlayerInputs(
                player_id=pid,
                team_id=eval_tid,
                season_year=sy,
                month_key=mk,
                now_date_iso=str(now_iso)[:10],
                expected_mpg=float(expected_mpg),
                actual_minutes=float(mins_eval),
                games_played=int(gp_eval),
                role_bucket=role_bucket,  # type: ignore[arg-type]
                leverage=float(leverage),
                team_win_pct=float(team_win_map.get(eval_tid, 0.5)),
                injury_status=injury_status_by_pid.get(pid),
                ovr=safe_int(rr.get("ovr"), 0),
                age=safe_int(rr.get("age"), 0),
                mental=mental,
            )

            prev = prev_states.get(pid)
            new_state, new_events = apply_monthly_player_tick(prev, inputs=inp, cfg=cfg)

            # Attach attribution context for UI / debugging.
            ctx = new_state.get("context") if isinstance(new_state.get("context"), dict) else {}
            ctx.setdefault("month_attribution", split_summary_by_pid.get(pid) or {})
            ctx.setdefault("evaluation_team_id", eval_tid)
            ctx.setdefault("current_roster_team_id", roster_tid)
            ctx.setdefault("team_end_of_month_id", team_end_by_pid.get(pid))
            new_state["context"] = ctx

            states_eval[pid] = new_state
            events.extend(new_events)

        # Resolve due promises (best-effort; requires promise schema).
        # This runs *inside the same monthly meta idempotency gate*.
        promise_stats: Dict[str, Any] = {
            "due": 0,
            "resolved": 0,
            "fulfilled": 0,
            "broken": 0,
            "deferred": 0,
            "cancelled": 0,
            "skipped_missing_schema": False,
        }

        promise_events: list[Dict[str, Any]] = []
        promise_updates: list[Dict[str, Any]] = []


        # Persist
        with repo.transaction() as cur:
            # Promise evaluation uses current-month context (minutes/games, injury, team context).
            # If schema is missing (older DB), we skip safely.
            try:
                due_promises = list_active_promises_due(cur, month_key=mk, limit=2000)
            except Exception:
                due_promises = []

            if due_promises:
                promise_stats["due"] = int(len(due_promises))

                # Preload team transactions evidence for HELP promises (best-effort).
                help_since_by_team: Dict[str, str] = {}
                for p in due_promises:
                    if str(p.get("promise_type") or "").upper() != "HELP":
                        continue
                    tid0 = str(p.get("team_id") or "").upper()
                    cd = norm_date_iso(p.get("created_date")) or now_iso[:10]
                    if not tid0:
                        continue
                    prev = help_since_by_team.get(tid0)
                    if prev is None or str(cd) < str(prev):
                        help_since_by_team[tid0] = str(cd)

                team_tx_cache: Dict[str, list[Dict[str, Any]]] = {}
                if help_since_by_team:
                    for tid0, since_d in help_since_by_team.items():
                        try:
                            like_pat = f'%"{tid0}"%'
                            rows_tx = cur.execute(
                                """
                                SELECT payload_json
                                FROM transactions_log
                                WHERE tx_date >= ?
                                  AND teams_json LIKE ?
                                ORDER BY COALESCE(tx_date,'') DESC, created_at DESC
                                LIMIT 1000;
                                """,
                                (str(since_d), str(like_pat)),
                            ).fetchall()
                            tx_list: list[Dict[str, Any]] = []
                            for (payload_json,) in rows_tx:
                                payload = json_loads(payload_json, default=None)
                                if isinstance(payload, dict):
                                    tx_list.append(payload)
                            team_tx_cache[tid0] = tx_list
                        except Exception:
                            team_tx_cache[tid0] = []

                for p in due_promises:
                    try:
                        promise_id = str(p.get("promise_id") or "")
                        ptype = str(p.get("promise_type") or "").upper()
                        pid = str(p.get("player_id") or "")
                        promised_team = str(p.get("team_id") or "").upper()
                        if not promise_id or not pid or not promised_team:
                            continue

                        st = states_eval.get(pid)
                        if not st:
                            continue

                        split = splits_by_pid.get(pid)

                        # Determine the team at end-of-month (EOM) for this processed month.
                        # If we cannot infer it (missing boxscore), anchor to promised team to avoid false cancellations.
                        team_eom = str(team_end_by_pid.get(pid) or "").upper()
                        if not team_eom:
                            if split and split.team_last:
                                team_eom = str(split.team_last).upper()
                            else:
                                team_eom = promised_team

                        # Cancellation: for non-trade promises, leaving the promised team makes the promise moot.
                        if ptype in {"MINUTES", "HELP", "ROLE"} and team_eom and promised_team and team_eom != promised_team:
                            promise_updates.append(
                                {
                                    "promise_id": promise_id,
                                    "status": "CANCELLED",
                                    "resolved_at": str(now_iso)[:10],
                                    "evidence": {
                                        "code": "PROMISE_CANCELLED_TEAM_CHANGED",
                                        "promised_team_id": promised_team,
                                        "team_end_of_month_id": team_eom,
                                        "month_key": mk,
                                    },
                                }
                            )
                            promise_events.append(
                                {
                                    "event_id": make_event_id("agency", "promise_cancelled", promise_id, mk),
                                    "player_id": pid,
                                    "team_id": team_eom or str(st.get("team_id") or promised_team).upper(),
                                    "season_year": sy,
                                    "date": str(now_iso)[:10],
                                    "event_type": "PROMISE_CANCELLED",
                                    "severity": 0.10,
                                    "payload": {
                                        "promise_id": promise_id,
                                        "promise_type": ptype,
                                        "promised_team_id": promised_team,
                                        "team_end_of_month_id": team_eom,
                                        "month_key": mk,
                                    },
                                }
                            )
                            promise_stats["cancelled"] = int(promise_stats["cancelled"]) + 1
                            continue

                        mental = mental_by_pid.get(pid) or {}

                        # Build context for evaluation.
                        actual_mpg = None
                        if ptype == "MINUTES":
                            # Evaluate against the promised team slice (not necessarily the month evaluation team).
                            mins_p, gp_p = _slice_minutes_games(split, promised_team) if split else (0.0, 0)
                            actual_mpg = float(mins_p / float(gp_p) if gp_p > 0 else 0.0)

                        ctx = PromiseEvaluationContext(
                            now_date_iso=str(now_iso)[:10],
                            month_key=mk,
                            player_id=pid,
                            # IMPORTANT: this is the team at end of *processed month*, not current roster at evaluation time.
                            team_id_current=team_eom or promised_team,
                            actual_mpg=actual_mpg,
                            injury_status=injury_status_by_pid.get(pid),
                            leverage=float(safe_float(st.get("leverage"), 0.0)),
                            mental=mental,
                            team_win_pct=float(safe_float(team_win_map.get(promised_team, 0.5), 0.5)),
                            team_transactions=team_tx_cache.get(promised_team) if ptype == "HELP" else None,
                        )

                        res = evaluate_promise(p, ctx=ctx, cfg=DEFAULT_PROMISE_CONFIG)
                        if not res.due:
                            continue

                        deltas = res.state_deltas or {}
                        if deltas:
                            if "trust" in deltas:
                                st["trust"] = float(clamp01(safe_float(st.get("trust"), 0.5) + safe_float(deltas.get("trust"), 0.0)))
                            if "minutes_frustration" in deltas:
                                st["minutes_frustration"] = float(
                                    clamp01(
                                        safe_float(st.get("minutes_frustration"), 0.0)
                                        + safe_float(deltas.get("minutes_frustration"), 0.0)
                                    )
                                )
                            if "team_frustration" in deltas:
                                st["team_frustration"] = float(
                                    clamp01(
                                        safe_float(st.get("team_frustration"), 0.0)
                                        + safe_float(deltas.get("team_frustration"), 0.0)
                                    )
                                )
                            if "trade_request_level_min" in deltas:
                                try:
                                    floor_v = int(safe_float(deltas.get("trade_request_level_min"), 0.0))
                                except Exception:
                                    floor_v = 0
                                try:
                                    cur_tr = int(st.get("trade_request_level") or 0)
                                except Exception:
                                    cur_tr = 0
                                st["trade_request_level"] = int(max(cur_tr, floor_v))

                        upd: Dict[str, Any] = {"promise_id": promise_id}
                        pu = res.promise_updates or {}
                        if "status" in pu and pu.get("status") is not None:
                            upd["status"] = str(pu.get("status")).upper()
                        if "due_month" in pu and pu.get("due_month") is not None:
                            upd["due_month"] = pu.get("due_month")
                        if "resolved_at" in pu:
                            upd["resolved_at"] = pu.get("resolved_at")

                        upd["evidence"] = {
                            "month_key": mk,
                            "now_date": str(now_iso)[:10],
                            "promised_team_id": promised_team,
                            "team_end_of_month_id": team_eom,
                            "result": {"due": res.due, "resolved": res.resolved, "new_status": res.new_status},
                            "reasons": res.reasons,
                            "meta": res.meta,
                        }
                        promise_updates.append(upd)

                        sev = float(clamp01(0.10 + abs(float(safe_float(deltas.get("trust"), 0.0))) * 4.0))
                        if res.resolved and str(res.new_status).upper() == "FULFILLED":
                            ev_type = "PROMISE_FULFILLED"
                            promise_stats["fulfilled"] = int(promise_stats["fulfilled"]) + 1
                            promise_stats["resolved"] = int(promise_stats["resolved"]) + 1
                        elif res.resolved and str(res.new_status).upper() == "BROKEN":
                            ev_type = "PROMISE_BROKEN"
                            promise_stats["broken"] = int(promise_stats["broken"]) + 1
                            promise_stats["resolved"] = int(promise_stats["resolved"]) + 1
                        elif not res.resolved and res.due:
                            ev_type = "PROMISE_DEFERRED"
                            promise_stats["deferred"] = int(promise_stats["deferred"]) + 1
                        else:
                            ev_type = "PROMISE_DUE"

                        if ev_type in {"PROMISE_FULFILLED", "PROMISE_BROKEN", "PROMISE_DEFERRED"}:
                            promise_events.append(
                                {
                                    "event_id": make_event_id("agency", "promise", promise_id, mk, ev_type),
                                    "player_id": pid,
                                    "team_id": team_eom or promised_team,
                                    "season_year": sy,
                                    "date": str(now_iso)[:10],
                                    "event_type": ev_type,
                                    "severity": sev,
                                    "payload": {
                                        "promise_id": promise_id,
                                        "promise_type": ptype,
                                        "promised_team_id": promised_team,
                                        "team_end_of_month_id": team_eom,
                                        "month_key": mk,
                                        "due_month": p.get("due_month"),
                                        "new_status": res.new_status,
                                        "state_deltas": deltas,
                                        "reasons": res.reasons,
                                        "meta": res.meta,
                                    },
                                }
                            )
                    except Exception:
                        continue

            # Persist promise row updates first (atomic)
            if promise_updates:
                try:
                    update_promises(cur, promise_updates)
                except Exception:
                    promise_stats["skipped_missing_schema"] = True

        if promise_events:
            events.extend(promise_events)

        # ------------------------------------------------------------------
        # Pass 3: apply team transitions for players whose evaluated team != current roster team.
        # ------------------------------------------------------------------
        multi_team_count = 0
        transitioned_count = 0
        states_final: Dict[str, Dict[str, Any]] = {}

        for rr in roster_rows:
            pid = str(rr["player_id"])
            st = states_eval.get(pid)
            if not st:
                continue

            roster_tid = roster_team_by_pid.get(pid) or str(rr.get("team_id") or "").upper()
            eval_tid = str(eval_team_by_pid.get(pid) or st.get("team_id") or "").upper()

            split = splits_by_pid.get(pid)
            if split and split.multi_team():
                multi_team_count += 1

            if roster_tid and eval_tid and roster_tid != eval_tid:
                transitioned_count += 1
                out = apply_team_transition(
                    st,
                    player_id=pid,
                    season_year=sy,
                    from_team_id=eval_tid,
                    to_team_id=roster_tid,
                    month_key=mk,
                    now_date_iso=str(now_iso)[:10],
                    mental=mental_by_pid.get(pid) or {},
                    trade_request_level_before=safe_int(st.get("trade_request_level"), 0),
                    split_summary=split_summary_by_pid.get(pid),
                    reason="POST_MONTH_TRADE",
                    cfg=cfg.transition,
                )
                st2 = out.state_after
                if out.event:
                    events.append(out.event)
                # Keep transition metadata for explainability.
                ctx2 = st2.get("context") if isinstance(st2.get("context"), dict) else {}
                ctx2.setdefault("transition", out.meta)
                st2["context"] = ctx2
                states_final[pid] = st2
            else:
                states_final[pid] = st

        # Persist
        with repo.transaction() as cur:
            upsert_player_agency_states(cur, states_final, now=str(now_iso))
            insert_agency_events(cur, events, now=str(now_iso))
            cur.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                (meta_key, "1"),
            )

        return {
            "ok": True,
            "skipped": False,
            "month": mk,
            "meta_key": meta_key,
            "players_processed": len(roster_rows),
            "states_upserted": len(states_final),
            "events_emitted": len(events),
            "month_attribution": {
                "multi_team_players": int(multi_team_count),
                "transitions_applied": int(transitioned_count),
                "splits_count": int(len(splits_by_pid)),
            },
            "promise_stats": promise_stats,
        }


