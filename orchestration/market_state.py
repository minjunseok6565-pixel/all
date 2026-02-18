from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, Iterable, Optional, Set

import state

from .types import CleanupReport, OrchestrationConfig


def _ensure_trade_market_schema(m: Dict[str, Any]) -> Dict[str, Any]:
    # IMPORTANT: keep in-place mutation when m is already a dict (state persistence depends on this)
    if not isinstance(m, dict):
        m = dict(m or {})
    m.setdefault("last_tick_date", None)
    m.setdefault("last_fail_closed_date", None)
    m.setdefault("last_fail_closed_reason", None)
    m.setdefault("listings", {})
    m.setdefault("threads", {})
    m.setdefault("cooldowns", {})
    m.setdefault("events", [])
    m.setdefault("human_controlled_team_ids", [])
    m.setdefault("tick_nonce", 0)
    if not isinstance(m.get("listings"), dict):
        m["listings"] = {}
    if not isinstance(m.get("threads"), dict):
        m["threads"] = {}
    if not isinstance(m.get("cooldowns"), dict):
        m["cooldowns"] = {}
    if not isinstance(m.get("events"), list):
        m["events"] = []
    if not isinstance(m.get("human_controlled_team_ids"), (list, tuple, set, str, type(None))):
        m["human_controlled_team_ids"] = []
    try:
        m["tick_nonce"] = int(m.get("tick_nonce") or 0)
    except Exception:
        m["tick_nonce"] = 0
    return m


def _ensure_trade_memory_schema(mem: Dict[str, Any]) -> Dict[str, Any]:
    # IMPORTANT: keep in-place mutation when mem is already a dict (state persistence depends on this)
    if not isinstance(mem, dict):
        mem = dict(mem or {})
    mem.setdefault("relationships", {})
    if not isinstance(mem.get("relationships"), dict):
        mem["relationships"] = {}
    return mem


def load_trade_market() -> Dict[str, Any]:
    return _ensure_trade_market_schema(state.trade_market_get() or {})


def save_trade_market(market: Dict[str, Any]) -> None:
    state.trade_market_set(_ensure_trade_market_schema(market))


def load_trade_memory() -> Dict[str, Any]:
    return _ensure_trade_memory_schema(state.trade_memory_get() or {})


def save_trade_memory(mem: Dict[str, Any]) -> None:
    state.trade_memory_set(_ensure_trade_memory_schema(mem))


def get_human_controlled_team_ids(
    trade_market: Dict[str, Any],
    *,
    state_key: str = "human_controlled_team_ids",
) -> Set[str]:
    m = _ensure_trade_market_schema(trade_market)
    raw = m.get(state_key)
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {str(x).upper() for x in raw if x}
    if isinstance(raw, str):
        return {s.strip().upper() for s in raw.split(",") if s.strip()}
    return set()


def set_human_controlled_team_ids(
    trade_market: Dict[str, Any],
    team_ids: Iterable[str],
    *,
    state_key: str = "human_controlled_team_ids",
) -> None:
    m = _ensure_trade_market_schema(trade_market)
    m[state_key] = sorted({str(x).upper() for x in team_ids if x})


def prune_market_events(trade_market: Dict[str, Any], *, max_kept: int) -> int:
    m = _ensure_trade_market_schema(trade_market)
    ev = m.get("events") if isinstance(m.get("events"), list) else []
    if not isinstance(max_kept, int) or max_kept <= 0:
        max_kept = 200
    if len(ev) > max_kept:
        prune_n = len(ev) - max_kept
        m["events"] = ev[prune_n:]
        return prune_n
    return 0


def pre_tick_cleanup(
    *,
    today: date,
    trade_market: Dict[str, Any],
    trade_memory: Dict[str, Any],
    config: OrchestrationConfig,
) -> CleanupReport:
    """
    tick_ctx build 전에 호출 권장.

    cooldown 기간 정의(명확화):
    - cooldown_days = "오늘 액션 이후, 다음날부터 N일 동안 막는다"
    - 저장 필드 expires_on(YYYY-MM-DD): tick 시작 시점에 today >= expires_on 이면 cooldown 키 제거(배타적 종료)
      예) today=3/1, days=1  -> 3/2 하루 막힘 -> expires_on=3/3, 3/3 tick 시작에 키 제거
    """
    report = CleanupReport()
    m = _ensure_trade_market_schema(trade_market)
    _ensure_trade_memory_schema(trade_memory)

    cd = m.get("cooldowns") if isinstance(m.get("cooldowns"), dict) else {}
    removed = 0
    kept: Dict[str, Any] = {}

    for k, v in (cd or {}).items():
        tid = str(k).upper()

        expires_on = None
        # 신규 스키마
        if isinstance(v, dict):
            expires_on = v.get("expires_on")

        # 레거시 지원: until(inclusive last active date) 또는 until(ambiguous) 필드를 발견하면 변환
        if isinstance(expires_on, str):
            try:
                exp = date.fromisoformat(expires_on[:10])
                if today >= exp:
                    removed += 1
                    continue
            except Exception:
                # 파싱 실패: 보수적으로 유지(키 존재=active)
                kept[tid] = v
                continue

        else:
            # expires_on이 없을 때: 레거시 'until'/'expires_at'/'until_inclusive' 등을 최대한 해석
            legacy_until = None
            if isinstance(v, dict):
                legacy_until = v.get("until_inclusive") or v.get("until") or v.get("expires_at")

            if isinstance(legacy_until, str):
                try:
                    d_until = date.fromisoformat(legacy_until[:10])
                    # 레거시 until을 inclusive last active로 간주하고 expires_on = until + 1 day
                    exp = d_until + timedelta(days=1)
                    if today >= exp:
                        removed += 1
                        continue
                    # 보존하면서 새 필드도 추가(점진적 마이그레이션)
                    nv = dict(v)
                    nv["expires_on"] = exp.isoformat()
                    kept[tid] = nv
                    continue
                except Exception:
                    kept[tid] = v
                    continue

            # 어떤 날짜 정보도 없으면: 안전하게 유지(개발자/툴로 정리 필요)
            kept[tid] = v

    m["cooldowns"] = kept
    report.removed_cooldowns = removed

    # --- threads cleanup (expiry + size cap)
    th = m.get("threads") if isinstance(m.get("threads"), dict) else {}
    removed_th = 0
    kept_th: Dict[str, Any] = {}

    for k, v in (th or {}).items():
        key = str(k)
        if not isinstance(v, dict):
            # 오염된 엔트리는 제거(상업용 상태 안정성)
            removed_th += 1
            continue

        expires_on = v.get("expires_on")
        if isinstance(expires_on, str):
            try:
                exp = date.fromisoformat(expires_on[:10])
                if today >= exp:
                    removed_th += 1
                    continue
            except Exception:
                # 파싱 실패: 보수적으로 유지(키 존재=active)
                pass

        kept_th[key] = v

    m["threads"] = kept_th
    report.removed_threads_expired = removed_th

    # Keep only the most recent N threads to avoid state bloat
    try:
        max_threads = int(getattr(config, "max_threads_kept", 50) or 50)
    except Exception:
        max_threads = 50

    if max_threads > 0 and isinstance(m.get("threads"), dict) and len(m["threads"]) > max_threads:

        def _thread_last_at(entry: Dict[str, Any]) -> date:
            for k2 in ("last_at", "started_at"):
                s = entry.get(k2)
                if isinstance(s, str):
                    try:
                        return date.fromisoformat(s[:10])
                    except Exception:
                        continue
            return date.min

        items = [(k2, v2) for k2, v2 in m["threads"].items() if isinstance(v2, dict)]
        items.sort(key=lambda kv: _thread_last_at(kv[1]), reverse=True)
        keep_keys = {k2 for k2, _ in items[:max_threads]}

        if len(keep_keys) < len(m["threads"]):
            before_n = len(m["threads"])
            m["threads"] = {k2: v2 for k2, v2 in m["threads"].items() if k2 in keep_keys}
            report.pruned_threads_limit = before_n - len(m["threads"])

    report.pruned_events = prune_market_events(m, max_kept=int(config.max_market_events_kept))
    return report


def add_team_cooldown(
    trade_market: Dict[str, Any],
    *,
    team_id: str,
    today: date,
    days: int,
    reason: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    cooldown 추가.

    days 정의(명확):
    - days <= 0: cooldown을 만들지 않는다.
    - days = N: 오늘 액션 이후 "다음날부터 N일" 막는다.
      expires_on = today + (N + 1)
    """
    m = _ensure_trade_market_schema(trade_market)
    try:
        days_i = int(days)
    except Exception:
        days_i = 0

    if days_i <= 0:
        return

    tid = str(team_id).upper()
    expires_on = date.fromordinal(today.toordinal() + days_i + 1).isoformat()

    m["cooldowns"][tid] = {
        "expires_on": expires_on,
        "reason": str(reason),
        "meta": dict(meta or {}),
        "created_at": today.isoformat(),
    }


def record_market_event(
    trade_market: Dict[str, Any],
    *,
    today: date,
    event_type: str,
    payload: Dict[str, Any],
) -> None:
    m = _ensure_trade_market_schema(trade_market)
    m["events"].append(
        {"at": today.isoformat(), "type": str(event_type), "payload": dict(payload or {})}
    )




# -----------------------------------------------------------------------------
# Threads (minimal "talks ongoing" market memory)
# -----------------------------------------------------------------------------


def make_pair_key(team_a: str, team_b: str) -> str:
    a = str(team_a).upper()
    b = str(team_b).upper()
    if a <= b:
        return f"{a}|{b}"
    return f"{b}|{a}"


def _compute_expires_on(today: date, days: int) -> str:
    """
    threads TTL 정의:
    - days = N: 오늘 이후 "다음날부터 N일" 동안 접촉 상태를 유지한다.
      expires_on = today + (N + 1)
    - days <= 0: 최소 1일은 유지(당일만 존재하면 체감이 약함)
    """
    try:
        d = int(days)
    except Exception:
        d = 0
    if d <= 0:
        d = 1
    return date.fromordinal(today.toordinal() + d + 1).isoformat()


def touch_thread(
    trade_market: Dict[str, Any],
    *,
    today: date,
    team_a: str,
    team_b: str,
    deal_id: str,
    score: float,
    reason_code: str,
    ttl_days: int,
) -> Dict[str, Any]:
    """
    Upsert/touch a thread for (team_a, team_b). Returns the entry dict with an
    additional transient key "_created" (bool) indicating whether it was newly created.
    """
    m = _ensure_trade_market_schema(trade_market)
    threads = m.get("threads") if isinstance(m.get("threads"), dict) else {}
    m["threads"] = threads

    a = str(team_a).upper()
    b = str(team_b).upper()
    if a == b:
        return {}

    key = make_pair_key(a, b)
    now_iso = today.isoformat()
    expires_on = _compute_expires_on(today, ttl_days)

    entry = threads.get(key)
    created = False
    if not isinstance(entry, dict):
        created = True
        entry = {
            "pair_key": key,
            "team_a": key.split("|")[0],
            "team_b": key.split("|")[1],
            "started_at": now_iso,
            "rumor_count": 0,
        }

    entry["last_at"] = now_iso
    entry["expires_on"] = expires_on
    entry["last_deal_id"] = str(deal_id)
    try:
        entry["last_score"] = float(score or 0.0)
    except Exception:
        entry["last_score"] = 0.0
    entry["last_reason_code"] = str(reason_code)
    try:
        entry["rumor_count"] = int(entry.get("rumor_count") or 0) + 1
    except Exception:
        entry["rumor_count"] = 1

    threads[key] = entry

    # Return a view that contains creation info without persisting it into state.
    out = dict(entry)
    out["_created"] = created
    return out


def get_active_thread_team_ids(
    trade_market: Dict[str, Any],
    *,
    today: date,
    excluded_team_ids: Optional[Set[str]] = None,
) -> Set[str]:
    m = _ensure_trade_market_schema(trade_market)
    threads = m.get("threads")
    if not isinstance(threads, dict):
        return set()

    excluded = {str(x).upper() for x in (excluded_team_ids or set()) if x}
    out: Set[str] = set()

    for v in threads.values():
        if not isinstance(v, dict):
            continue
        exp = v.get("expires_on")
        if not isinstance(exp, str):
            continue
        try:
            d_exp = date.fromisoformat(exp[:10])
        except Exception:
            continue
        if today >= d_exp:
            continue

        a = str(v.get("team_a") or "").upper()
        b = str(v.get("team_b") or "").upper()
        if a and a not in excluded:
            out.add(a)
        if b and b not in excluded:
            out.add(b)

    return out


def bump_relationship(
    trade_memory: Dict[str, Any],
    *,
    team_a: str,
    team_b: str,
    today: date,
    patch: Dict[str, Any],
) -> None:
    mem = _ensure_trade_memory_schema(trade_memory)
    rel = mem["relationships"]

    a = str(team_a).upper()
    b = str(team_b).upper()
    if a == b:
        return

    rel.setdefault(a, {})
    rel.setdefault(b, {})
    rel[a].setdefault(b, {"last_at": None, "counts": {}, "meta": {}})
    rel[b].setdefault(a, {"last_at": None, "counts": {}, "meta": {}})

    def _apply(entry: Dict[str, Any]) -> None:
        entry["last_at"] = today.isoformat()
        counts = entry.setdefault("counts", {})
        if isinstance(patch.get("counts"), dict):
            for ck, cv in patch["counts"].items():
                try:
                    counts[str(ck)] = int(counts.get(str(ck), 0)) + int(cv)
                except Exception:
                    continue
        meta = entry.setdefault("meta", {})
        if isinstance(patch.get("meta"), dict):
            meta.update(patch["meta"])

    _apply(rel[a][b])
    _apply(rel[b][a])


# -----------------------------------------------------------------------------
# Relationship read helpers (NO implicit creation; safe for dry_run / analysis)
# -----------------------------------------------------------------------------

def get_relationship_entry(
    trade_memory: Dict[str, Any],
    *,
    team_a: str,
    team_b: str,
) -> Optional[Dict[str, Any]]:
    """
    Read-only accessor for trade_memory.relationships[team_a][team_b].

    IMPORTANT:
    - Do NOT create missing keys (no setdefault) to avoid state pollution in dry_run / analysis.
    - Returns the entry dict if it exists and is a dict, otherwise None.
    """
    if not isinstance(trade_memory, dict):
        return None
    rel = trade_memory.get("relationships")
    if not isinstance(rel, dict):
        return None

    a = str(team_a).upper()
    b = str(team_b).upper()
    if a == b:
        return None

    a_map = rel.get(a)
    if not isinstance(a_map, dict):
        return None
    entry = a_map.get(b)
    if isinstance(entry, dict):
        return entry
    return None


def get_rel_meta_date_iso(
    trade_memory: Dict[str, Any],
    *,
    team_a: str,
    team_b: str,
    key: str,
) -> Optional[str]:
    """
    Convenience: return entry.meta[key] if it looks like an ISO date (YYYY-MM-DD).
    Returns None if missing or invalid.
    """
    entry = get_relationship_entry(trade_memory, team_a=team_a, team_b=team_b)
    if not entry:
        return None
    meta = entry.get("meta")
    if not isinstance(meta, dict):
        return None
    v = meta.get(key)
    if not isinstance(v, str):
        return None
    s = v[:10]
    try:
        # Validate format
        date.fromisoformat(s)
        return s
    except Exception:
        return None
