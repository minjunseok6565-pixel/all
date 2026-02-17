from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from league_repo import LeagueRepo

from state import (
    export_workflow_state,
    get_cached_playoff_news_snapshot,
    get_cached_weekly_news_snapshot,
    get_current_date_as_date,
    get_db_path,
    get_postseason_snapshot,
    set_cached_playoff_news_snapshot,
    set_cached_weekly_news_snapshot,
)

from news.cache import get_playoff_cache, get_weekly_cache, set_playoff_cache, set_weekly_cache
from news.editorial import select_top_events
from news.extractors.playoffs import extract_playoff_events
from news.extractors.weekly import build_week_window, extract_weekly_events
from news.ids import make_event_id
from news.models import NewsArticle, NewsEvent
from news.scoring import apply_importance
from news.render.template_ko import render_article
from news.render.gemini_rewrite import rewrite_article_with_gemini

logger = logging.getLogger(__name__)


def _iso(d: date) -> str:
    return d.isoformat()


def _as_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if len(s) >= 10:
        s = s[:10]
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def _extract_transaction_events(*, start: date, end: date) -> List[NewsEvent]:
    """Fetch SSOT transactions from SQLite and convert to NewsEvent."""
    db_path = get_db_path()
    events: List[NewsEvent] = []
    try:
        with LeagueRepo(db_path) as repo:
            repo.init_db()
            rows = repo.list_transactions(limit=500, since_date=_iso(start))
    except Exception as exc:
        logger.warning("transactions fetch failed: %s", exc, exc_info=True)
        return []

    for t in rows:
        if not isinstance(t, dict):
            continue
        d = _as_date(t.get("date") or t.get("action_date") or t.get("created_at"))
        if not d:
            continue
        if not (start <= d <= end):
            continue

        tx_type = str(t.get("type") or t.get("tx_type") or "transaction")
        title = str(t.get("title") or "").strip()
        summary = str(t.get("summary") or "").strip()
        if not title:
            if tx_type.lower() == "trade":
                title = "트레이드 소식"
            else:
                title = "로스터 변동"
        if not summary:
            summary = str(t)

        teams = t.get("teams") or []
        if not isinstance(teams, list):
            teams = []
        team_ids = [str(x) for x in teams if x]

        events.append(
            {
                "event_id": make_event_id("TX", d.isoformat(), tx_type, t.get("deal_id") or ""),
                "date": d.isoformat(),
                "type": "TRANSACTION",
                "importance": 0.0,
                "facts": {
                    "tx_type": tx_type,
                    "title": title,
                    "summary": summary,
                    "teams": team_ids,
                    "payload": t,
                },
                "related_team_ids": team_ids,
                "related_player_ids": [],
                "related_player_names": [],
                "tags": ["transaction", tx_type],
            }
        )

    return events


def refresh_weekly_news(api_key: str) -> Dict[str, Any]:
    """Generate (or return cached) weekly news.

    Output shape is compatible with the existing frontend:
      {"current_date": "YYYY-MM-DD", "items": [ {title, summary, ...}, ... ]}

    Notes on cache:
    - Current state_schema (4.0) restricts cached_views.weekly_news keys.
      We therefore cache per-week only (week_start).
    - Later schema upgrades can add "as_of_date" / "built_from_turn" etc.
    """
    if not api_key:
        raise ValueError("apiKey is required")

    current = get_current_date_as_date()  # fail-loud if in-game date missing
    ws, we, week_key = build_week_window(current)

    cache = get_weekly_cache()
    if cache.get("last_generated_week_start") == week_key and cache.get("items"):
        return {"current_date": _iso(current), "items": cache.get("items", [])}

    snapshot = export_workflow_state()

    events = extract_weekly_events(snapshot, start_date=_iso(ws), end_date=_iso(we))
    events += _extract_transaction_events(start=ws, end=we)

    apply_importance(events)
    selected = select_top_events(events, min_count=3, max_count=6)

    articles: List[NewsArticle] = [render_article(e) for e in selected]

    # Optional style rewrite; safe fallback to templates
    if api_key:
        rewritten: List[NewsArticle] = []
        for a in articles:
            rewritten.append(rewrite_article_with_gemini(api_key, a))
        articles = rewritten

    cache["last_generated_week_start"] = week_key
    cache["items"] = articles
    set_weekly_cache(cache)
    return {"current_date": _iso(current), "items": articles}


def _build_playoffs_boxscore_lookup(workflow: Dict[str, Any]) -> Dict[Tuple[str, str, str, int, int], Dict[str, Any]]:
    """Index playoff phase game_results for quick matching."""
    out: Dict[Tuple[str, str, str, int, int], Dict[str, Any]] = {}
    pr = (workflow.get("phase_results") or {}).get("playoffs") or {}
    games = pr.get("games") or []
    game_results = pr.get("game_results") or {}

    if not isinstance(games, list) or not isinstance(game_results, dict):
        return out

    for g in games:
        if not isinstance(g, dict):
            continue
        try:
            d = str(g.get("date") or "")[:10]
            home = str(g.get("home_team_id") or "")
            away = str(g.get("away_team_id") or "")
            hs = int(g.get("home_score"))
            as_ = int(g.get("away_score"))
            gid = str(g.get("game_id") or "")
        except Exception:
            continue
        gr = game_results.get(gid)
        if isinstance(gr, dict) and d and home and away:
            out[(d, home, away, hs, as_)] = gr

    return out


def refresh_playoff_news() -> Dict[str, Any]:
    """Append newly completed playoff games as news articles.

    Output shape is compatible with the existing frontend:
      {"items": [...], "new_items": [...]}
    """
    postseason = get_postseason_snapshot()
    playoffs = postseason.get("playoffs")
    if not playoffs:
        raise ValueError("플레이오프 진행 중이 아닙니다.")

    cache = get_playoff_cache()
    prev_counts = cache.get("series_game_counts") or {}
    items = cache.get("items") or []

    workflow = export_workflow_state()
    box_lookup = _build_playoffs_boxscore_lookup(workflow)

    events, new_counts = extract_playoff_events(
        playoffs,
        previous_counts=prev_counts,
        boxscore_lookup=box_lookup,
    )

    apply_importance(events)

    # For playoffs, we publish ALL newly generated events, but keep them readable.
    # Group by date and keep a capped number per refresh.
    events_sorted = sorted(events, key=lambda e: (str(e.get("date") or ""), float(e.get("importance") or 0.0)))
    # A single refresh can produce multiple events per game (recap+swing+matchpoint+elimination).
    # Limit to avoid spamming: keep top 12 by importance.
    events_sorted = sorted(events, key=lambda e: float(e.get("importance") or 0.0), reverse=True)[:12]
    events_sorted = sorted(events_sorted, key=lambda e: str(e.get("date") or ""))

    new_articles: List[NewsArticle] = [render_article(e) for e in events_sorted]

    items_out = list(items) + new_articles

    cache["series_game_counts"] = new_counts
    cache["items"] = items_out
    set_playoff_cache(cache)

    return {"items": items_out, "new_items": new_articles}
