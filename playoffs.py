from __future__ import annotations

import logging
import random
from contextlib import contextmanager
from copy import deepcopy
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import schema

from config import TEAM_TO_CONF_DIV
from league_repo import LeagueRepo
import fatigue
import injury
from matchengine_v2_adapter import adapt_matchengine_result_to_v2, build_context_from_team_ids
from matchengine_v3.sim_game import simulate_game
from sim.roster_adapter import build_team_state_from_db
from state import (
    export_full_state_snapshot,
    get_cached_playoff_news_snapshot,
    get_cached_stats_snapshot,
    get_current_date_as_date,
    get_db_path,
    get_postseason_snapshot,
    ingest_game_result,
    postseason_reset,
    postseason_set_champion,
    postseason_set_dates,
    postseason_set_field,
    postseason_set_my_team_id,
    postseason_set_play_in,
    postseason_set_playoffs,
    set_cached_playoff_news_snapshot,
    set_cached_stats_snapshot,
    set_current_date,
)
from team_utils import get_conference_standings

logger = logging.getLogger(__name__)

HomePattern = [True, True, False, False, True, False, True]


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _accumulate_player_rows(rows: List[Dict[str, Any]], season_player_stats: Dict[str, Any]) -> None:
    for row in rows:
        player_id = str(row.get("PlayerID"))
        team_id = str(row.get("TeamID"))

        entry = season_player_stats.setdefault(
            player_id,
            {"player_id": player_id, "name": row.get("Name"), "team_id": team_id, "games": 0, "totals": {}},
        )
        entry["name"] = row.get("Name", entry.get("name"))
        entry["team_id"] = team_id
        entry["games"] = int(entry.get("games", 0) or 0) + 1

        totals = entry.setdefault("totals", {})
        for k, v in row.items():
            if k in {"PlayerID", "TeamID", "Name", "Pos", "Position"}:
                continue
            if _is_number(v):
                try:
                    totals[k] = float(totals.get(k, 0.0)) + float(v)
                except (TypeError, ValueError):
                    continue


# ---------------------------------------------------------------------------
# 상태 helpers
# ---------------------------------------------------------------------------

@contextmanager
def _repo_ctx() -> LeagueRepo:
    db_path = get_db_path()
    with LeagueRepo(db_path) as repo:
        yield repo

def _ensure_postseason_state() -> Dict[str, Any]:
    postseason = get_postseason_snapshot()
    if not isinstance(postseason, dict):
        postseason_reset()
        return get_postseason_snapshot()
    return postseason


def _safe_date_fromisoformat(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    try:
        return date.fromisoformat(str(date_str))
    except ValueError:
        return None


def _regular_season_end_date() -> date:
    league = export_full_state_snapshot().get("league", {})
    master_schedule = league.get("master_schedule") or {}
    by_date = master_schedule.get("by_date") or {}

    latest: Optional[date] = None
    for ds in by_date.keys():
        parsed = _safe_date_fromisoformat(ds)
        if parsed and (latest is None or parsed > latest):
            latest = parsed

    if latest:
        return latest

    season_start = _safe_date_fromisoformat(league.get("season_start"))
    if season_start:
        return season_start + timedelta(days=180)

    return get_current_date_as_date()


def _play_in_schedule_window() -> Tuple[date, date]:
    season_end = _regular_season_end_date()
    start = season_end + timedelta(days=2)
    final_day = start + timedelta(days=2)
    return start, final_day


def _play_in_end_date(play_in_state: Dict[str, Any]) -> Optional[date]:
    latest: Optional[date] = None
    for conf_state in play_in_state.values():
        matchups = conf_state.get("matchups") or {}
        for key in ("seven_vs_eight", "nine_vs_ten", "final"):
            d = _safe_date_fromisoformat((matchups.get(key) or {}).get("date"))
            if d and (latest is None or d > latest):
                latest = d
    return latest


def _round_latest_end(series_list: List[Dict[str, Any]]) -> Optional[date]:
    latest: Optional[date] = None
    for s in series_list:
        games = s.get("games") or []
        if not games:
            continue
        d = _safe_date_fromisoformat(games[-1].get("date"))
        if d and (latest is None or d > latest):
            latest = d
    return latest


def _next_round_start(series_list: List[Dict[str, Any]], buffer_days: int = 2) -> Optional[str]:
    latest = _round_latest_end(series_list)
    if not latest:
        return None
    return (latest + timedelta(days=buffer_days)).isoformat()


def reset_postseason_state() -> Dict[str, Any]:
    postseason_reset()
    playoff_news = get_cached_playoff_news_snapshot() or {}
    playoff_news["series_game_counts"] = {}
    playoff_news["items"] = []
    set_cached_playoff_news_snapshot(playoff_news)
    stats_cache = get_cached_stats_snapshot() or {}
    stats_cache.pop("playoff_leaders", None)
    set_cached_stats_snapshot(stats_cache)
    return get_postseason_snapshot()


# ---------------------------------------------------------------------------
# 로스터 / 경기 헬퍼
# ---------------------------------------------------------------------------

def _seed_entry(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "team_id": row.get("team_id"),
        "seed": row.get("rank"),
        "conference": row.get("conference"),
        "division": row.get("division"),
        "wins": row.get("wins"),
        "losses": row.get("losses"),
        "win_pct": row.get("win_pct"),
        "point_diff": row.get("point_diff"),
    }


def _random_seed_entry(team_id: str, seed: Optional[int], conf_key: str) -> Dict[str, Any]:
    info = TEAM_TO_CONF_DIV.get(team_id, {})
    division = info.get("division")
    # 높은 시드가 더 높은 승률을 갖도록 약간의 편차를 둔다.
    base_win_pct = 0.78 - max(seed - 1, 0) * 0.035 if seed else 0.42
    win_pct = max(0.35, min(0.78, base_win_pct + random.uniform(-0.01, 0.02)))
    wins = int(round(win_pct * 82))
    wins = min(max(wins, 32), 62)
    losses = 82 - wins
    point_diff = int((0.8 - (seed or 12) * 0.2) + random.uniform(-3, 5))

    return {
        "team_id": team_id,
        "seed": seed,
        "conference": conf_key,
        "division": division,
        "wins": wins,
        "losses": losses,
        "win_pct": wins / 82 if 82 else 0,
        "games_played": 82,
        "point_diff": point_diff,
    }


def _build_random_conf_field(conf_key: str, my_team_id: Optional[str]) -> Dict[str, Any]:
    conf_teams = [
        tid
        for tid, meta in TEAM_TO_CONF_DIV.items()
        if (meta.get("conference") or "").lower() == conf_key
    ]
    random.shuffle(conf_teams)

    auto_slots = list(range(1, 7))
    play_in_slots = list(range(7, 11))
    auto_bids: List[Dict[str, Any]] = []
    play_in: List[Dict[str, Any]] = []
    eliminated: List[Dict[str, Any]] = []

    remaining = [tid for tid in conf_teams if tid != my_team_id]
    random.shuffle(remaining)

    if my_team_id:
        my_seed = random.choice(auto_slots)
        auto_slots.remove(my_seed)
        auto_bids.append(_random_seed_entry(my_team_id, my_seed, conf_key))

    for seed in auto_slots:
        if not remaining:
            break
        auto_bids.append(_random_seed_entry(remaining.pop(), seed, conf_key))

    for seed in play_in_slots:
        if not remaining:
            break
        play_in.append(_random_seed_entry(remaining.pop(), seed, conf_key))

    seed_counter = 11
    while remaining:
        eliminated.append(_random_seed_entry(remaining.pop(), seed_counter, conf_key))
        seed_counter += 1

    auto_bids = sorted(auto_bids, key=lambda r: r.get("seed") or 99)
    play_in = sorted(play_in, key=lambda r: r.get("seed") or 99)

    return {
        "auto_bids": auto_bids,
        "play_in": play_in,
        "eliminated": eliminated,
    }


def _simulate_postseason_game(
    home_team_id: str, away_team_id: str, game_date: Optional[str] = None
) -> Dict[str, Any]:
    if game_date:
        try:
            game_date = date.fromisoformat(str(game_date)).isoformat()
        except ValueError:
            game_date = str(game_date)
    else:
        game_date = get_current_date_as_date().isoformat()

    set_current_date(game_date)

    league = export_full_state_snapshot().get("league", {})
    game_id = f"playoffs_{home_team_id}_{away_team_id}_{uuid4().hex[:8]}"
    context = build_context_from_team_ids(
        game_id,
        game_date,
        home_team_id,
        away_team_id,
        league,
        phase="playoffs",
    )

    rng = random.Random()
    with _repo_ctx() as repo:
        # Ensure schema is applied (idempotent). This guarantees fatigue/injury tables exist
        # even if the DB was created before the modules were added.
        repo.init_db()

        season_year = fatigue.season_year_from_season_id(str(getattr(context, "season_id", "") or ""))

        # ------------------------------------------------------------
        # Injury: prepare between-game state (OUT players + returning debuffs)
        # ------------------------------------------------------------
        prepared_inj = None
        unavailable_by_team: Dict[str, set[str]] = {}
        attrs_mods_by_pid = None
        try:
            prepared_inj = injury.prepare_game_injuries(
                repo,
                game_id=str(getattr(context, "game_id", "") or ""),
                game_date_iso=str(game_date),
                season_year=int(season_year),
                home_team_id=str(home_team_id),
                away_team_id=str(away_team_id),
            )
            unavailable_by_team = dict(prepared_inj.unavailable_pids_by_team or {})
            attrs_mods_by_pid = prepared_inj.attrs_mods_by_pid
        except Exception:
            logger.warning(
                "INJURY_PREPARE_FAILED (playoffs) date=%s home=%s away=%s",
                str(game_date),
                str(home_team_id),
                str(away_team_id),
                exc_info=True,
            )
            prepared_inj = None

        hid = schema.normalize_team_id(str(home_team_id)).upper()
        aid = schema.normalize_team_id(str(away_team_id)).upper()

        home_team = build_team_state_from_db(
            repo=repo,
            team_id=home_team_id,
            exclude_pids=set(unavailable_by_team.get(hid, set()) or set()),
            attrs_mods_by_pid=attrs_mods_by_pid,
        )
        away_team = build_team_state_from_db(
            repo=repo,
            team_id=away_team_id,
            exclude_pids=set(unavailable_by_team.get(aid, set()) or set()),
            attrs_mods_by_pid=attrs_mods_by_pid,
        )

        # ------------------------------------------------------------
        # Fatigue: prepare between-game condition (start_energy + energy_cap)
        # ------------------------------------------------------------
        prepared_fat = None
        try:
            prepared_fat = fatigue.prepare_game_fatigue(
                repo,
                game_date_iso=str(game_date),
                season_year=int(season_year),
                home=home_team,
                away=away_team,
            )
        except Exception:
            logger.warning(
                "FATIGUE_PREPARE_FAILED (playoffs) date=%s home=%s away=%s",
                str(game_date),
                str(home_team_id),
                str(away_team_id),
                exc_info=True,
            )

        # In-game injury hook
        injury_hook = None
        if prepared_inj is not None:
            try:
                injury_hook = injury.make_in_game_injury_hook(prepared_inj, context=context, home=home_team, away=away_team)
            except Exception:
                logger.warning(
                    "INJURY_HOOK_BUILD_FAILED (playoffs) date=%s home=%s away=%s",
                    str(game_date),
                    str(home_team_id),
                    str(away_team_id),
                    exc_info=True,
                )
                injury_hook = None

        raw_result = simulate_game(rng, home_team, away_team, context=context, injury_hook=injury_hook)

        # Post-game finalize: fatigue + injuries
        if prepared_fat is not None:
            try:
                fatigue.finalize_game_fatigue(
                    repo,
                    prepared=prepared_fat,
                    home=home_team,
                    away=away_team,
                    raw_result=raw_result,
                )
            except Exception:
                logger.warning(
                    "FATIGUE_FINALIZE_FAILED (playoffs) date=%s home=%s away=%s",
                    str(game_date),
                    str(home_team_id),
                    str(away_team_id),
                    exc_info=True,
                )

        if prepared_inj is not None:
            try:
                injury.finalize_game_injuries(
                    repo,
                    prepared=prepared_inj,
                    home=home_team,
                    away=away_team,
                    raw_result=raw_result,
                )
            except Exception:
                logger.warning(
                    "INJURY_FINALIZE_FAILED (playoffs) date=%s home=%s away=%s",
                    str(game_date),
                    str(home_team_id),
                    str(away_team_id),
                    exc_info=True,
                )

    v2_result = adapt_matchengine_result_to_v2(
        raw_result,
        context,
        engine_name="matchengine_v3",
    )
    ingest_game_result(game_result=v2_result, game_date=game_date)

    _ensure_postseason_state()

    final = v2_result.get("final") or {}
    home_score = int(final.get(home_team_id, 0))
    away_score = int(final.get(away_team_id, 0))
    winner = home_team_id if home_score > away_score else away_team_id

    return {
        "date": game_date,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_score": home_score,
        "away_score": away_score,
        "winner": winner,
        "status": "final",
        "final_score": final,
        "boxscore": v2_result.get("teams"),
    }


def _simulate_play_in_game(
    home_entry: Optional[Dict[str, Any]],
    away_entry: Optional[Dict[str, Any]],
    game_date: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not home_entry or not away_entry:
        return None
    return _simulate_postseason_game(
        home_entry["team_id"], away_entry["team_id"], game_date=game_date
    )


def _auto_play_in_conf(conf_state: Dict[str, Any], my_team_id: Optional[str]) -> None:
    matchups = conf_state.get("matchups", {})

    for key in ("seven_vs_eight", "nine_vs_ten"):
        matchup = matchups.get(key)
        if matchup and not matchup.get("result"):
            home = matchup.get("home")
            away = matchup.get("away")
            if not home or not away:
                continue
            if my_team_id in {home.get("team_id"), away.get("team_id")}:
                continue
            matchup["result"] = _simulate_play_in_game(
                home, away, matchup.get("date")
            )

    _apply_play_in_results(conf_state)

    final_matchup = matchups.get("final")
    if final_matchup:
        home = final_matchup.get("home")
        away = final_matchup.get("away")
        if home and away and not final_matchup.get("result"):
            if my_team_id not in {home.get("team_id"), away.get("team_id")}:
                final_matchup["result"] = _simulate_play_in_game(
                    home, away, final_matchup.get("date")
                )
        _apply_play_in_results(conf_state)


def play_my_team_play_in_game() -> Dict[str, Any]:
    postseason = _ensure_postseason_state()
    my_team_id = postseason.get("my_team_id")
    play_in = deepcopy(postseason.get("play_in"))
    if not my_team_id or not play_in:
        raise ValueError("Play-in state is not initialized with a user team")

    target_conf = None
    for conf_key, conf_state in play_in.items():
        participants = conf_state.get("participants", {})
        if any(p.get("team_id") == my_team_id for p in participants.values()):
            target_conf = conf_key
            break
    if target_conf is None:
        raise ValueError("User team is not part of the play-in field")

    conf_state = play_in[target_conf]
    matchups = conf_state.get("matchups", {})

    for key in ("seven_vs_eight", "nine_vs_ten", "final"):
        matchup = matchups.get(key)
        if not matchup or matchup.get("result"):
            continue
        home = matchup.get("home")
        away = matchup.get("away")
        if home and away and my_team_id in {home.get("team_id"), away.get("team_id")}:
            matchup["result"] = _simulate_play_in_game(
                home, away, matchup.get("date")
            )
            _apply_play_in_results(conf_state)
            _auto_play_in_conf(conf_state, my_team_id)
            postseason_set_play_in(play_in)
            _maybe_start_playoffs_from_play_in()
            return get_postseason_snapshot()

    raise ValueError("No pending play-in game for the user team")


# ---------------------------------------------------------------------------
# 플레이오프 시리즈
# ---------------------------------------------------------------------------

def _series_template(
    home_adv: Dict[str, Any],
    road: Dict[str, Any],
    round_name: str,
    matchup_label: str,
    start_date: str,
    best_of: int = 7,
) -> Dict[str, Any]:
    return {
        "round": round_name,
        "matchup": matchup_label,
        "home_court": home_adv.get("team_id"),
        "road": road.get("team_id"),
        "home_entry": home_adv,
        "road_entry": road,
        "games": [],
        "wins": {home_adv.get("team_id"): 0, road.get("team_id"): 0},
        "best_of": best_of,
        "winner": None,
        "start_date": start_date,
    }


def _is_series_finished(series: Dict[str, Any]) -> bool:
    winner = series.get("winner")
    if winner:
        return True
    wins = series.get("wins") or {}
    best_of = series.get("best_of", 7)
    needed = best_of // 2 + 1
    return any(v >= needed for v in wins.values())


def _simulate_one_series_game(series: Dict[str, Any]) -> Dict[str, Any]:
    if _is_series_finished(series):
        return series

    game_idx = len(series.get("games", []))
    best_of = series.get("best_of", 7)
    if game_idx >= best_of:
        return series

    higher_is_home = HomePattern[game_idx]
    home_id = series["home_court"] if higher_is_home else series["road"]
    away_id = series["road"] if higher_is_home else series["home_court"]

    if game_idx == 0:
        next_game_date = series.get("start_date") or get_current_date_as_date().isoformat()
    else:
        last_game = series.get("games", [])[-1]
        last_date = _safe_date_fromisoformat(last_game.get("date")) or get_current_date_as_date()
        prev_home_flag = HomePattern[game_idx - 1]
        rest_days = 1 if prev_home_flag == higher_is_home else 2
        next_game_date = (last_date + timedelta(days=rest_days)).isoformat()

    game_result = _simulate_postseason_game(home_id, away_id, game_date=next_game_date)
    series.setdefault("games", []).append(game_result)

    wins = series.setdefault("wins", {})
    wins[game_result["winner"]] = wins.get(game_result["winner"], 0) + 1

    needed = best_of // 2 + 1
    if wins[game_result["winner"]] >= needed:
        series["winner"] = series["home_entry"] if series["home_entry"].get("team_id") == game_result["winner"] else series["road_entry"]
    return series


def _round_series(bracket: Dict[str, Any], round_name: str) -> List[Dict[str, Any]]:
    if round_name == "Conference Quarterfinals":
        return (bracket.get("east", {}).get("quarterfinals") or []) + (bracket.get("west", {}).get("quarterfinals") or [])
    if round_name == "Conference Semifinals":
        return (bracket.get("east", {}).get("semifinals") or []) + (bracket.get("west", {}).get("semifinals") or [])
    if round_name == "Conference Finals":
        finals = []
        if bracket.get("east", {}).get("finals"):
            finals.append(bracket["east"]["finals"])
        if bracket.get("west", {}).get("finals"):
            finals.append(bracket["west"]["finals"])
        return finals
    if round_name == "NBA Finals":
        return [bracket.get("finals")]
    return []


# ---------------------------------------------------------------------------
# 플레이오프 브래킷 생성
# ---------------------------------------------------------------------------

def _conference_quarterfinals(
    seeds: Dict[int, Dict[str, Any]], start_date: str
) -> List[Dict[str, Any]]:
    qf_pairs = [(1, 8), (4, 5), (3, 6), (2, 7)]
    results = []
    for high, low in qf_pairs:
        team_high = seeds.get(high)
        team_low = seeds.get(low)
        if not team_high or not team_low:
            continue
        home, road = _pick_home_advantage(team_high, team_low)
        results.append(
            _series_template(
                home,
                road,
                "Conference Quarterfinals",
                f"{high} vs {low}",
                start_date,
            )
        )
    return results


def _conference_semifinals_from_qf(
    qf_list: List[Dict[str, Any]], start_date: str
) -> List[Dict[str, Any]]:
    def _find_winner(matchup_prefix: str) -> Optional[Dict[str, Any]]:
        for s in qf_list:
            if s.get("matchup", "").startswith(matchup_prefix):
                return s.get("winner")
        return None

    inputs = [(_find_winner("1 vs 8"), _find_winner("4 vs 5")), (_find_winner("2 vs 7"), _find_winner("3 vs 6"))]
    results = []
    for idx, (a, b) in enumerate(inputs, start=1):
        if not a or not b:
            continue
        home, road = _pick_home_advantage(a, b)
        results.append(
            _series_template(
                home,
                road,
                "Conference Semifinals",
                f"SF{idx}",
                start_date,
            )
        )
    return results


def _conference_finals_from_sf(
    sf_list: List[Dict[str, Any]], start_date: str
) -> Optional[Dict[str, Any]]:
    if len(sf_list) < 2:
        return None
    if not all(s.get("winner") for s in sf_list):
        return None
    home, road = _pick_home_advantage(sf_list[0]["winner"], sf_list[1]["winner"])
    return _series_template(home, road, "Conference Finals", "CF", start_date)


def _finals_from_conf(
    east: Optional[Dict[str, Any]], west: Optional[Dict[str, Any]], start_date: str
) -> Optional[Dict[str, Any]]:
    if not east or not west:
        return None
    if not east.get("winner") or not west.get("winner"):
        return None
    home, road = _pick_home_advantage(east["winner"], west["winner"])
    return _series_template(home, road, "NBA Finals", "FINALS", start_date)


def _initialize_playoffs(
    seeds_by_conf: Dict[str, Dict[int, Dict[str, Any]]], start_date: date
) -> None:
    start_date_str = start_date.isoformat()
    bracket = {
        "east": {
            "quarterfinals": _conference_quarterfinals(
                seeds_by_conf.get("east", {}), start_date_str
            ),
            "semifinals": [],
            "finals": None,
        },
        "west": {
            "quarterfinals": _conference_quarterfinals(
                seeds_by_conf.get("west", {}), start_date_str
            ),
            "semifinals": [],
            "finals": None,
        },
        "finals": None,
    }

    postseason_set_playoffs(
        {
            "seeds": seeds_by_conf,
            "bracket": bracket,
            "current_round": "Conference Quarterfinals",
            "start_date": start_date_str,
        }
    )


def _advance_round_if_ready() -> None:
    postseason = _ensure_postseason_state()
    playoffs = deepcopy(postseason.get("playoffs"))
    if not playoffs:
        return

    bracket = playoffs.get("bracket", {})
    current_round = playoffs.get("current_round", "Conference Quarterfinals")

    if current_round == "Conference Quarterfinals":
        qf_series = _round_series(bracket, current_round)
        if qf_series and all(_is_series_finished(s) for s in qf_series):
            start_date = _next_round_start(qf_series) or playoffs.get("start_date") or get_current_date_as_date().isoformat()
            bracket["east"]["semifinals"] = _conference_semifinals_from_qf(
                bracket["east"].get("quarterfinals", []), start_date
            )
            bracket["west"]["semifinals"] = _conference_semifinals_from_qf(
                bracket["west"].get("quarterfinals", []), start_date
            )
            playoffs["current_round"] = "Conference Semifinals"
            postseason_set_playoffs(playoffs)
            return

    if current_round == "Conference Semifinals":
        sf_series = _round_series(bracket, current_round)
        if sf_series and all(_is_series_finished(s) for s in sf_series):
            start_date = _next_round_start(sf_series) or playoffs.get("start_date") or get_current_date_as_date().isoformat()
            bracket["east"]["finals"] = _conference_finals_from_sf(
                bracket["east"].get("semifinals", []), start_date
            )
            bracket["west"]["finals"] = _conference_finals_from_sf(
                bracket["west"].get("semifinals", []), start_date
            )
            playoffs["current_round"] = "Conference Finals"
            postseason_set_playoffs(playoffs)
            return

    if current_round == "Conference Finals":
        cf_series = _round_series(bracket, current_round)
        if cf_series and all(_is_series_finished(s) for s in cf_series):
            start_date = _next_round_start(cf_series) or playoffs.get("start_date") or get_current_date_as_date().isoformat()
            bracket["finals"] = _finals_from_conf(
                bracket.get("east", {}).get("finals"),
                bracket.get("west", {}).get("finals"),
                start_date,
            )
            playoffs["current_round"] = "NBA Finals"
            postseason_set_playoffs(playoffs)
            return

    if current_round == "NBA Finals":
        finals = bracket.get("finals")
        if finals and _is_series_finished(finals):
            postseason_set_champion(finals.get("winner"))


# ---------------------------------------------------------------------------
# 사용자 팀 기준 진행
# ---------------------------------------------------------------------------

def _find_my_series(playoffs: Dict[str, Any], my_team_id: str) -> Optional[Dict[str, Any]]:
    bracket = playoffs.get("bracket", {})
    round_name = playoffs.get("current_round", "Conference Quarterfinals")
    for series in _round_series(bracket, round_name):
        if not series:
            continue
        if my_team_id in {series.get("home_court"), series.get("road")}:
            return series
    return None


def advance_my_team_one_game() -> Dict[str, Any]:
    postseason = _ensure_postseason_state()
    my_team_id = postseason.get("my_team_id")
    playoffs = deepcopy(postseason.get("playoffs"))
    if not my_team_id or not playoffs:
        raise ValueError("Playoffs are not initialized with a user team")

    bracket = playoffs.get("bracket", {})
    round_name = playoffs.get("current_round", "Conference Quarterfinals")
    my_series = _find_my_series(playoffs, my_team_id)
    if not my_series:
        raise ValueError("User team is not in an active playoff series")
    if _is_series_finished(my_series):
        raise ValueError("User team series has already finished")

    _simulate_one_series_game(my_series)

    for series in _round_series(bracket, round_name):
        if not series or series is my_series:
            continue
        if _is_series_finished(series):
            continue
        _simulate_one_series_game(series)

    postseason_set_playoffs(playoffs)
    _advance_round_if_ready()
    return get_postseason_snapshot()


def auto_advance_current_round() -> Dict[str, Any]:
    postseason = _ensure_postseason_state()
    playoffs = deepcopy(postseason.get("playoffs"))
    if not playoffs:
        raise ValueError("Playoffs are not initialized")

    bracket = playoffs.get("bracket", {})
    round_name = playoffs.get("current_round", "Conference Quarterfinals")
    for series in _round_series(bracket, round_name):
        if not series:
            continue
        while not _is_series_finished(series):
            _simulate_one_series_game(series)

    postseason_set_playoffs(playoffs)
    _advance_round_if_ready()
    return get_postseason_snapshot()


# ---------------------------------------------------------------------------
# 초기화 흐름
# ---------------------------------------------------------------------------

def _build_playoff_seeds(field: Dict[str, Any], play_in: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    seeds_for_bracket: Dict[str, Dict[int, Dict[str, Any]]] = {"east": {}, "west": {}}
    for conf_key in ("east", "west"):
        conf_field = field.get(conf_key, {})
        conf_seeds = {entry["seed"]: entry for entry in conf_field.get("auto_bids", []) if entry.get("seed")}
        play_in_conf = play_in.get(conf_key) or {}
        seed7 = play_in_conf.get("seed7")
        seed8 = play_in_conf.get("seed8")
        if seed7:
            seed7_fixed = dict(seed7)
            seed7_fixed["seed"] = 7
            conf_seeds[7] = seed7_fixed

        if seed8:
            seed8_fixed = dict(seed8)
            seed8_fixed["seed"] = 8
            conf_seeds[8] = seed8_fixed
        seeds_for_bracket[conf_key] = conf_seeds
    return seeds_for_bracket


def _maybe_start_playoffs_from_play_in() -> None:
    postseason = _ensure_postseason_state()
    field = postseason.get("field")
    play_in = postseason.get("play_in")
    if not field or not play_in:
        return

    for conf_state in play_in.values():
        if not conf_state.get("seed7") or not conf_state.get("seed8"):
            return

    seeds = _build_playoff_seeds(field, play_in)
    play_in_end = _safe_date_fromisoformat(
        postseason.get("play_in_end_date")
    ) or _play_in_end_date(play_in)
    playoff_start = (play_in_end + timedelta(days=3)) if play_in_end else get_current_date_as_date()
    postseason_set_dates(
        postseason.get("play_in_start_date"),
        postseason.get("play_in_end_date"),
        playoff_start.isoformat(),
    )
    _initialize_playoffs(seeds, playoff_start)


def _prepare_play_in(field: Dict[str, Any], my_team_id: Optional[str]) -> Dict[str, Any]:
    play_in_state: Dict[str, Any] = {}
    start_date, final_date = _play_in_schedule_window()
    start_date_str = start_date.isoformat()
    final_date_str = final_date.isoformat()

    for conf_key in ("east", "west"):
        conf_state = _conference_play_in_template(conf_key, field)
        conf_matchups = conf_state.get("matchups", {})
        for key in ("seven_vs_eight", "nine_vs_ten"):
            if key in conf_matchups:
                conf_matchups[key]["date"] = start_date_str
        if "final" in conf_matchups:
            conf_matchups["final"]["date"] = final_date_str
        play_in_state[conf_key] = conf_state

    postseason_set_play_in(play_in_state)
    postseason_set_dates(start_date_str, final_date_str, _ensure_postseason_state().get("playoffs_start_date"))

    my_conf = None
    my_seed = None
    for conf_key, conf_field in field.items():
        for entry in conf_field.get("auto_bids", []) + conf_field.get("play_in", []):
            if entry.get("team_id") == my_team_id:
                my_conf = conf_key
                my_seed = entry.get("seed")
                break

    for conf_key, conf_state in play_in_state.items():
        if my_seed and my_seed <= 6:
            _auto_play_in_conf(conf_state, None)
        elif my_conf == conf_key:
            _auto_play_in_conf(conf_state, my_team_id)
        else:
            _auto_play_in_conf(conf_state, None)

    for conf_state in play_in_state.values():
        _apply_play_in_results(conf_state)

    postseason_set_play_in(play_in_state)
    if my_seed and my_seed <= 6:
        _maybe_start_playoffs_from_play_in()

    return play_in_state


def initialize_postseason(my_team_id: str, use_random_field: bool = False) -> Dict[str, Any]:
    reset_postseason_state()
    postseason_set_my_team_id(my_team_id)
    if use_random_field:
        field = build_random_postseason_field(my_team_id)
    else:
        field = build_postseason_field()

    play_in_state = _prepare_play_in(field, my_team_id)

    # 사용자가 플레이인을 건너뛴 경우 이미 플레이오프가 세팅됨
    if not get_postseason_snapshot().get("playoffs"):
        _maybe_start_playoffs_from_play_in()

    return get_postseason_snapshot()


__all__ = [
    "build_postseason_field",
    "reset_postseason_state",
    "initialize_postseason",
    "play_my_team_play_in_game",
    "advance_my_team_one_game",
    "auto_advance_current_round",
]
