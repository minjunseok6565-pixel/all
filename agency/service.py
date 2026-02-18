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

import calendar
import logging
from typing import Any, Dict, Mapping, Optional, Tuple

from league_repo import LeagueRepo

from .config import AgencyConfig, DEFAULT_CONFIG
from .expectations import compute_expectations_for_league
from .promises import DEFAULT_PROMISE_CONFIG, PromiseEvaluationContext, evaluate_promise
from .expectations_month import compute_month_expectations
from .month_context import (
    PlayerMonthSplit,
    TeamSlice,
    build_split_summary,
    finalize_player_month_split,
    players_by_team_from_splits,
)
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


def _month_start_end_dates(month_key: str) -> Tuple[str, str]:
    """Return (month_start_date_iso, month_end_date_iso) for a YYYY-MM key."""
    mk = str(month_key or "")
    try:
        y_s, m_s = mk.split("-", 1)
        y = int(y_s)
        m = int(m_s)
        last = int(calendar.monthrange(y, m)[1])
        return (f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{last:02d}")
    except Exception:
        # Defensive fallback; caller should still handle invalid mk.
        return (f"{mk}-01", f"{mk}-28")


def _team_move_events_by_pid_since(
    repo: LeagueRepo,
    *,
    player_ids: list[str],
    since_date_iso: str,
    limit: int = 20000,
) -> Dict[str, list[Tuple[str, str, str]]]:
    """Collect team-change events (trade/sign/release) from SSOT transactions_log.

    Returns mapping pid -> list[(event_date_iso, from_team, to_team)] sorted DESC by event_date_iso.
    """
    pids = [str(pid) for pid in (player_ids or []) if str(pid)]
    if not pids:
        return {}

    pid_set = set(pids)
    since_d = str(since_date_iso or "")[:10]
    if not since_d:
        return {}

    # NOTE: transactions_log.tx_type is the SSOT discriminator.
    # We only need types that can change roster.team_id.
    tx_types = [
        "trade",
        "TRADE",
        "SIGN_FREE_AGENT",
        "RELEASE_TO_FA",
        # legacy/dev spellings (safety)
        "signing",
        "release_to_free_agency",
    ]
    placeholders = ",".join(["?"] * len(tx_types))

    try:
        rows = repo._conn.execute(
            f"""
            SELECT payload_json
            FROM transactions_log
            WHERE tx_date IS NOT NULL
              AND substr(tx_date, 1, 10) >= ?
              AND tx_type IN ({placeholders})
            ORDER BY COALESCE(tx_date,'') DESC, created_at DESC
            LIMIT ?;
            """,
            [since_d, *tx_types, max(1, int(limit))],
        ).fetchall()
    except Exception:
        return {}

    out: Dict[str, list[Tuple[str, str, str]]] = {}

    def _tx_date(tx: Mapping[str, Any]) -> Optional[str]:
        v = tx.get("action_date")
        if v is None:
            v = tx.get("date")
        if v is None:
            v = tx.get("trade_date")
        if v is None:
            return None
        s = str(v).strip()
        if not s or len(s) < 10:
            return None
        if str(s)[:10] < since_d:
            return None
        return s

    for (payload_json,) in rows:
        tx = json_loads(payload_json, default=None)
        if not isinstance(tx, Mapping):
            continue
        tdate = _tx_date(tx)
        if tdate is None:
            continue

        ttype = str(tx.get("type") or tx.get("tx_type") or "")
        if str(ttype).lower() == "trade":
            pm = tx.get("player_moves")
            if not isinstance(pm, list):
                continue
            for m in pm:
                if not isinstance(m, Mapping):
                    continue
                pid = str(m.get("player_id") or "")
                if not pid or pid not in pid_set:
                    continue
                ft = str(m.get("from_team") or "").upper()
                tt = str(m.get("to_team") or "").upper()
                if not ft or not tt:
                    continue
                out.setdefault(pid, []).append((str(tdate), ft, tt))
            continue

        # Contract-style tx: top-level player_id + from_team/to_team
        pid = tx.get("player_id")
        if pid is None:
            continue
        pid_s = str(pid)
        if not pid_s or pid_s not in pid_set:
            continue
        ft = str(tx.get("from_team") or "").upper()
        tt = str(tx.get("to_team") or "").upper()
        if not ft or not tt:
            continue
        out.setdefault(pid_s, []).append((str(tdate), ft, tt))

    # Sort desc for stable reverse application.
    for pid in list(out.keys()):
        out[pid].sort(key=lambda e: str(e[0]), reverse=True)
    return out


def _teams_as_of_dates(
    *,
    roster_team_by_pid: Mapping[str, str],
    move_events_by_pid: Mapping[str, list[Tuple[str, str, str]]],
    target_dates_desc: list[str],
) -> Dict[str, Dict[str, str]]:
    """Compute team_id as-of end-of-day for each target date, per player."""

    def _reverse_team(cur_team: str, ft: str, tt: str) -> str:
        cur = str(cur_team or "").upper()
        if not cur:
            return cur
        if str(tt or "").upper() == cur:
            return str(ft or "").upper()
        return cur

    out: Dict[str, Dict[str, str]] = {}
    for pid, team_now in (roster_team_by_pid or {}).items():
        cur_team = str(team_now or "").upper()
        ev = list(move_events_by_pid.get(pid) or [])
        i = 0
        by_date: Dict[str, str] = {}
        for d in target_dates_desc:
            cutoff = f"{str(d)[:10]}T23:59:59Z"
            while i < len(ev) and str(ev[i][0]) > cutoff:
                _dt, ft, tt = ev[i]
                cur_team = _reverse_team(cur_team, ft, tt)
                i += 1
            by_date[str(d)[:10]] = str(cur_team)
        out[str(pid)] = by_date
    return out


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
    # date_iso -> {team_id: games_count} for the processed month.
    # Used to synthesize DNP presence for players not appearing in boxscores.
    team_games_by_date: Optional[Mapping[str, Mapping[str, int]]] = None,
    now_iso: str,
    cfg: AgencyConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Apply a league-wide monthly agency tick.

    Idempotent via meta key `nba_agency_tick_done_{month_key}`.

    Args:
        db_path: SQLite path
        season_year: current season year
        month_key: YYYY-MM (the month being processed)
        minutes_by_player: total minutes played in that month (regular-season finals)
        games_by_player: games played in that month (optional but recommended)
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
                p.attrs_json,
                r.updated_at
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
                    "roster_updated_at": r[6],
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

        # Load previous agency states early (used for defensive fallbacks in month attribution).
        with repo.transaction() as cur:
            prev_states = get_player_agency_states(cur, player_ids)

        # Expectations for role/leverage/expected minutes.
        expectations_current = compute_expectations_for_league(roster_rows, config=cfg.expectations)

        # ------------------------------------------------------------------
        # Month context helpers
        # ------------------------------------------------------------------
        month_start_date, month_end_date = _month_start_end_dates(mk)

        # Normalize schedule-derived team games map (optional but recommended).
        team_games_map: Dict[str, Dict[str, int]] = {}
        for d_raw, by_team_raw in (team_games_by_date or {}).items():
            d = norm_date_iso(d_raw) or str(d_raw or "")[:10]
            if not d or not str(d).startswith(str(mk)):
                continue
            if not isinstance(by_team_raw, Mapping):
                continue
            for tid_raw, n_raw in by_team_raw.items():
                tid = str(tid_raw or "").upper()
                if not tid:
                    continue
                try:
                    n = int(n_raw)
                except Exception:
                    continue
                if n <= 0:
                    continue
                team_games_map.setdefault(str(d), {})[tid] = int(n)

        month_game_dates = sorted(team_games_map.keys())

        # Team at end-of-month (calendar) derived from SSOT transactions.
        roster_team_by_pid_now: Dict[str, str] = {str(rr["player_id"]): str(rr["team_id"]).upper() for rr in roster_rows}

        # Target dates: month game dates (for DNP synthesis) + month end (for promises).
        target_dates = list(month_game_dates)
        if month_end_date and month_end_date not in target_dates:
            target_dates.append(month_end_date)
        target_dates_desc = sorted({str(d)[:10] for d in target_dates if str(d)[:10]}, reverse=True)

        move_events_by_pid = _team_move_events_by_pid_since(
            repo,
            player_ids=player_ids,
            since_date_iso=month_start_date,
        )
        teams_asof_by_pid = _teams_as_of_dates(
            roster_team_by_pid=roster_team_by_pid_now,
            move_events_by_pid=move_events_by_pid,
            target_dates_desc=target_dates_desc,
        )

        # Schedule presence: how many team games each player could have appeared in during the month,
        # per team, based on SSOT transactions + the schedule-derived team_games_by_date.
        # This is used to compute DNP frequency pressure while keeping MPG = minutes/games_played.
        schedule_games_by_pid_by_team: Dict[str, Dict[str, int]] = {}
        if team_games_map and month_game_dates:
            for pid in player_ids:
                by_date = teams_asof_by_pid.get(pid) or {}
                by_team: Dict[str, int] = schedule_games_by_pid_by_team.setdefault(pid, {})
                for d in month_game_dates:
                    tid = str(by_date.get(d) or roster_team_by_pid_now.get(pid) or "").upper()
                    if not tid or tid == "FA":
                        continue
                    n = int((team_games_map.get(d) or {}).get(tid, 0) or 0)
                    if n <= 0:
                        continue
                    by_team[tid] = int(by_team.get(tid, 0) + n)

        roster_updated_date_by_pid: Dict[str, str] = {}
        for rr in roster_rows:
            pid = str(rr.get("player_id") or "")
            if not pid:
                continue
            ud = norm_date_iso(rr.get("roster_updated_at")) or str(rr.get("roster_updated_at") or "")[:10]
            if ud:
                roster_updated_date_by_pid[pid] = str(ud)

        # Derive end-of-month team_id per player (used for promise evaluation).
        team_eom_by_pid: Dict[str, str] = {}
        for pid in player_ids:
            team_eom = str((teams_asof_by_pid.get(pid) or {}).get(month_end_date) or roster_team_by_pid_now.get(pid) or "").upper()

            # Defensive fallback: if there are *no* move events since month start and roster was updated
            # after month end, the player likely joined after the processed month.
            # In that case, prefer the previous agency_state team to avoid false promise fulfilment.
            if not move_events_by_pid.get(pid):
                upd = roster_updated_date_by_pid.get(pid)
                if upd and str(upd) > str(month_end_date):
                    prev = prev_states.get(pid)
                    prev_tid = str(prev.get("team_id") or "").upper() if isinstance(prev, Mapping) else ""
                    cur_tid = str(roster_team_by_pid_now.get(pid) or "").upper()
                    if prev_tid and cur_tid and prev_tid != cur_tid:
                        team_eom = prev_tid

            if team_eom:
                team_eom_by_pid[pid] = team_eom

        # ------------------------------------------------------------------
        # Synthesize month splits for players who never appear in boxscores.
        # ------------------------------------------------------------------
        if team_games_map and target_dates_desc:
            for pid in player_ids:
                if pid in splits_by_pid:
                    continue

                # If we cannot infer historical team membership and roster was updated after month end,
                # skip DNP synthesis to avoid blaming the current team for a month the player didn't play.
                if not move_events_by_pid.get(pid):
                    upd = roster_updated_date_by_pid.get(pid)
                    if upd and str(upd) > str(month_end_date):
                        prev = prev_states.get(pid)
                        prev_tid = str(prev.get("team_id") or "").upper() if isinstance(prev, Mapping) else ""
                        cur_tid = str(roster_team_by_pid_now.get(pid) or "").upper()
                        if not prev_tid or (prev_tid and cur_tid and prev_tid != cur_tid):
                            continue

                team_slices: Dict[str, TeamSlice] = {}
                by_date = teams_asof_by_pid.get(pid) or {}
                for d in month_game_dates:
                    tid = str(by_date.get(d) or roster_team_by_pid_now.get(pid) or "").upper()
                    if not tid or tid == "FA":
                        continue
                    n = int((team_games_map.get(d) or {}).get(tid, 0) or 0)
                    if n <= 0:
                        continue
                    sl = team_slices.get(tid)
                    if sl is None:
                        sl = TeamSlice(team_id=tid)
                        team_slices[tid] = sl
                    for _ in range(n):
                        sl.add_game(game_date_iso=str(d), minutes=0.0)

                if team_slices:
                    splits_by_pid[pid] = finalize_player_month_split(
                        player_id=pid,
                        month_key=mk,
                        team_slices=team_slices,
                        cfg=cfg.month_context,
                    )

        # Month expectations (role/leverage/expected minutes in the roster context actually experienced).
        # If month split data is unavailable, this will be empty and we will fall back to current expectations.
        month_players_by_team = players_by_team_from_splits(splits_by_pid, min_games_present=1) if splits_by_pid else {}
        with repo.transaction() as cur:
            month_expectations = compute_month_expectations(
                cur,
                players_by_team=month_players_by_team,
                config=cfg.expectations,
            )


        # Injury status (best-effort)
        injury_status_by_pid = _best_effort_injury_status_by_pid(repo, player_ids)


        # Run tick (pass 1): evaluate each player's month under the team they actually played for.
        # We may later apply a team transition to move the state onto the current roster team.
        states_eval: Dict[str, Dict[str, Any]] = {}
        events: list[Dict[str, Any]] = []
        mental_by_pid: Dict[str, Dict[str, int]] = {}

        eval_team_by_pid: Dict[str, str] = {}
        roster_team_by_pid: Dict[str, str] = {}
        split_summary_by_pid: Dict[str, Dict[str, Any]] = {}

        for rr in roster_rows:
            pid = str(rr["player_id"])
            roster_tid = str(rr["team_id"]).upper()
            roster_team_by_pid[pid] = roster_tid

            prev = prev_states.get(pid)

            split = splits_by_pid.get(pid)
            eval_tid = (split.primary_team if split and split.primary_team else roster_tid) or roster_tid

            # Choose the team context we evaluate this month under.
            # - Prefer primary_team when month attribution exists.
            # - If attribution exists but primary_team is None (small sample), fall back to last/dominant.
            # - If the player has *no* month sample (e.g., joined after month end), anchor to prev_state.team_id
            #   to avoid blaming the current roster team for a month they didn't play.
            eval_tid = None
            if split is not None:
                eval_tid = split.primary_team or split.team_last or split.team_dominant
            if not eval_tid and prev and prev.get("team_id"):
                eval_tid = prev.get("team_id")
            if not eval_tid:
                eval_tid = roster_tid
            
            eval_tid = str(eval_tid).upper() if eval_tid else roster_tid
            eval_team_by_pid[pid] = eval_tid


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

            # Guardrail: players with no processed-month sample (no boxscore rows)
            # should not be treated as a full DNP month. This avoids phantom
            # "last month I didn't play" complaints for players who joined after
            # the month boundary (trade/signing).
            legacy_mins = float(minutes_map.get(pid, 0.0))
            legacy_gp = int(games_map.get(pid, 0))
            has_month_sample = bool(split is not None or legacy_mins > 0.0 or legacy_gp > 0)
            if not has_month_sample:
                if pid not in split_summary_by_pid:
                    split_summary_by_pid[pid] = {
                        "player_id": pid,
                        "month_key": mk,
                        "missing": True,
                        "reason": "NO_MONTH_SAMPLE",
                    }

                # Keep prior state intact; only attach explainability context.
                # (We still allow pass-3 team transitions to reconcile roster changes.)
                base_state: Dict[str, Any]
                if prev and isinstance(prev, Mapping):
                    base_state = dict(prev)
                else:
                    base_state = {
                        "player_id": pid,
                        "team_id": str(eval_tid).upper(),
                        "season_year": int(sy),
                        "role_bucket": str(role_bucket or "UNKNOWN"),
                        "leverage": float(clamp01(leverage)),
                        "minutes_expected_mpg": float(max(0.0, expected_mpg)),
                        "minutes_actual_mpg": 0.0,
                        "minutes_frustration": 0.0,
                        "team_frustration": 0.0,
                        "trust": 0.5,
                        "trade_request_level": 0,
                        "cooldown_minutes_until": None,
                        "cooldown_trade_until": None,
                        "cooldown_help_until": None,
                        "cooldown_contract_until": None,
                        "last_processed_month": None,
                        "context": {},
                    }

                # Normalize identity fields + mark the month as considered.
                base_state["player_id"] = pid
                base_state["team_id"] = str(eval_tid).upper()
                base_state["season_year"] = int(sy)
                base_state["last_processed_month"] = mk

                ctx = base_state.get("context") if isinstance(base_state.get("context"), dict) else {}
                ctx.setdefault("month_attribution", split_summary_by_pid.get(pid) or {})
                ctx.setdefault("month_sample_missing", True)
                ctx.setdefault("evaluation_team_id", eval_tid)
                ctx.setdefault("current_roster_team_id", roster_tid)
                ctx.setdefault("team_end_of_month_id", team_eom_by_pid.get(pid) or str(eval_tid).upper())
                base_state["context"] = ctx

                states_eval[pid] = base_state
                continue
 
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
                games_possible=int((schedule_games_by_pid_by_team.get(pid) or {}).get(eval_tid, 0) or 0),
                role_bucket=role_bucket,  # type: ignore[arg-type]
                leverage=float(leverage),
                team_win_pct=float(team_win_map.get(eval_tid, 0.5)),
                injury_status=injury_status_by_pid.get(pid),
                ovr=safe_int(rr.get("ovr"), 0),
                age=safe_int(rr.get("age"), 0),
                mental=mental,
            )

            new_state, new_events = apply_monthly_player_tick(prev, inputs=inp, cfg=cfg)

            # Attach attribution context for UI / debugging.
            ctx = new_state.get("context") if isinstance(new_state.get("context"), dict) else {}
            ctx.setdefault("month_attribution", split_summary_by_pid.get(pid) or {})
            ctx.setdefault("evaluation_team_id", eval_tid)
            ctx.setdefault("current_roster_team_id", roster_tid)
            ctx.setdefault("team_end_of_month_id", team_eom_by_pid.get(pid) or str(eval_tid).upper())
            new_state["context"] = ctx

            states_eval[pid] = new_state
            events.extend(new_events)

        # ------------------------------------------------------------------
        # Pass 2: resolve due promises using *month-end team context*, not current roster.
        # ------------------------------------------------------------------
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

        with repo.transaction() as cur:
            # Promise tables are optional for older saves. If missing, we skip cleanly.
            has_promise_schema = bool(
                cur.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='player_agency_promises' LIMIT 1;"
                ).fetchone()
            )
            if not has_promise_schema:
                promise_stats["skipped_missing_schema"] = True
                due_promises = []
            else:
                due_promises = list_active_promises_due(cur, month_key=mk, limit=2000)

            if due_promises:
                promise_stats["due"] = int(len(due_promises))

                # Preload team transactions evidence for HELP promises (best-effort).
                help_since_by_team: Dict[str, str] = {}
                for p in due_promises:
                    if str(p.get("promise_type") or "").upper() != "HELP":
                        continue
                    tid0 = str(p.get("team_id") or "").upper()
                    cd = norm_date_iso(p.get("created_date")) or str(now_iso)[:10]
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
                        team_eom = str(team_eom_by_pid.get(pid) or "").upper()
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


        # ------------------------------------------------------------------
        # Persist: write promise updates + state + events + meta key in ONE transaction.
        # ------------------------------------------------------------------
        with repo.transaction() as cur:
            # Apply promise row updates (if any). This must happen in the same transaction
            # as state/events/meta to avoid SSOT inconsistencies on crash/retry.
            if promise_updates:
                update_promises(cur, promise_updates)

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

