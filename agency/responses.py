from __future__ import annotations

"""User response logic for player agency events.

This module is *pure business logic* (no DB I/O). It turns:
  - a player-generated agency event (complaint/demand/request)
  - the player's current agency state
  - player mental traits
  - a user-chosen response

into:
  - immediate agency state adjustments (trust/frustrations/trade_request_level)
  - an optional PromiseSpec to be persisted and evaluated later
  - explainable reasons and meta for UI/analytics

Important
---------
- Mental traits are modulators, not absolute rules.
- Leverage gates the *strength* of reactions.
- Effects are intentionally conservative by default; tune in playtests.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional

from .config import AgencyConfig, DEFAULT_CONFIG
from .promises import PromiseSpec, PromiseType, due_month_from_now
from .utils import clamp, clamp01, mental_norm, norm_date_iso, safe_float, safe_float_opt


AgencyEventType = Literal[
    "MINUTES_COMPLAINT",
    "HELP_DEMAND",
    "TRADE_REQUEST",
    "TRADE_REQUEST_PUBLIC",

    # v2 issue families
    "ROLE_PRIVATE",
    "ROLE_AGENT",
    "ROLE_PUBLIC",

    "CONTRACT_PRIVATE",
    "CONTRACT_AGENT",
    "CONTRACT_PUBLIC",

    "HEALTH_PRIVATE",
    "HEALTH_AGENT",
    "HEALTH_PUBLIC",

    "TEAM_PRIVATE",
    "TEAM_PUBLIC",

    "CHEMISTRY_PRIVATE",
    "CHEMISTRY_AGENT",
    "CHEMISTRY_PUBLIC",

    # team pass
    "LOCKER_ROOM_MEETING",
]

ResponseType = Literal[
    # Generic
    "ACKNOWLEDGE",
    "DISMISS",

    # Minutes complaint
    "PROMISE_MINUTES",

    # v2 axis promises
    "PROMISE_ROLE",
    "PROMISE_LOAD",
    "PROMISE_EXTENSION_TALKS",

    # Help demand
    "PROMISE_HELP",
    "REFUSE_HELP",

    # Trade request
    "SHOP_TRADE",
    "REFUSE_TRADE",
    "PROMISE_COMPETE",

    # Team meeting
    "TEAM_TALK_CALM",
    "TEAM_TALK_FIRM",
    "TEAM_TALK_IGNORE",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResponseConfig:
    """Tunable parameters for immediate response impacts."""

    # Trust deltas (base; scaled by leverage + mental + event severity)
    trust_acknowledge: float = 0.03
    trust_promise: float = 0.06
    trust_dismiss_penalty: float = 0.06
    trust_refuse_help_penalty: float = 0.06
    trust_refuse_trade_penalty: float = 0.10

    # Frustration deltas (base; scaled)
    minutes_relief_acknowledge: float = 0.04
    minutes_relief_promise: float = 0.06
    minutes_bump_dismiss: float = 0.03

    team_relief_acknowledge: float = 0.02
    team_relief_promise: float = 0.03
    team_bump_refuse_help: float = 0.03
    team_bump_refuse_trade: float = 0.04

    # Secondary frustration bump when trade is refused
    minutes_bump_refuse_trade: float = 0.02


    # v2: generic deltas for non-v1 axis events (role/contract/health/chemistry)
    axis_relief_acknowledge: float = 0.03
    axis_relief_promise: float = 0.05
    axis_bump_dismiss: float = 0.03

    # v2: team-level locker room meeting talk
    team_talk_trust_calm: float = 0.03
    team_talk_trust_firm: float = -0.01
    team_talk_trust_ignore: float = -0.06

    team_talk_chem_relief_calm: float = 0.05
    team_talk_chem_relief_firm: float = 0.02
    team_talk_chem_bump_ignore: float = 0.03

    team_talk_team_relief_calm: float = 0.02
    team_talk_team_relief_firm: float = 0.01
    team_talk_team_bump_ignore: float = 0.02

    # Promise due months
    promise_minutes_due_months: int = 1
    promise_role_due_months: int = 1
    promise_load_due_months: int = 1
    promise_extension_due_months: int = 1
    promise_help_due_months: int = 2
    promise_trade_due_months: int = 2

    # v2 promise defaults
    promise_role_min_starts_rate: float = 0.60
    promise_role_min_closes_rate: float = 0.35
    promise_load_default_max_mpg: float = 30.0

    # Minutes promise target bounds (safety)
    promise_minutes_min_mpg: float = 8.0
    promise_minutes_max_mpg: float = 40.0

    # Escalation thresholds
    dismiss_escalate_trade_fr_threshold: float = 0.80
    refuse_trade_public_ego_threshold: float = 0.72
    refuse_trade_public_leverage_threshold: float = 0.65


DEFAULT_RESPONSE_CONFIG = ResponseConfig()


# ---------------------------------------------------------------------------
# Public output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResponseOutcome:
    ok: bool

    event_type: str
    response_type: str

    # New absolute values to write to player_agency_state (only for affected fields).
    # The DB layer should clamp once more.
    state_updates: Dict[str, Any] = field(default_factory=dict)

    # Team events may affect multiple players (deltas, not absolutes).
    bulk_state_deltas: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    promise: Optional[PromiseSpec] = None

    # Explainability
    tone: Literal["CALM", "FIRM", "ANGRY"] = "CALM"
    player_reply: str = ""
    reasons: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def apply_user_response(
    *,
    event: Mapping[str, Any],
    state: Mapping[str, Any],
    mental: Mapping[str, Any],
    response_type: str,
    response_payload: Optional[Mapping[str, Any]] = None,
    now_date_iso: Optional[str] = None,
    cfg: AgencyConfig = DEFAULT_CONFIG,
    rcfg: ResponseConfig = DEFAULT_RESPONSE_CONFIG,
) -> ResponseOutcome:
    """Apply a user response to a player agency event (pure logic)."""

    et = str(event.get("event_type") or "").upper()
    rt = str(response_type or "").upper()
    payload = dict(response_payload or {})

    # Normalize time
    if now_date_iso is None:
        now_date_iso = str(event.get("date") or "")
    now_d = norm_date_iso(now_date_iso) or (str(event.get("date") or "")[:10] or "2000-01-01")

    # Validate supported event types
    supported_events = {
        "MINUTES_COMPLAINT",
        "HELP_DEMAND",
        "TRADE_REQUEST",
        "TRADE_REQUEST_PUBLIC",

        # v2 families
        "ROLE_PRIVATE", "ROLE_AGENT", "ROLE_PUBLIC",
        "CONTRACT_PRIVATE", "CONTRACT_AGENT", "CONTRACT_PUBLIC",
        "HEALTH_PRIVATE", "HEALTH_AGENT", "HEALTH_PUBLIC",
        "TEAM_PRIVATE", "TEAM_PUBLIC",
        "CHEMISTRY_PRIVATE", "CHEMISTRY_AGENT", "CHEMISTRY_PUBLIC",

        # team pass
        "LOCKER_ROOM_MEETING",
    }
    if et not in supported_events:
        return ResponseOutcome(
            ok=False,
            event_type=et,
            response_type=rt,
            reasons=[{"code": "RESPONSE_UNSUPPORTED_EVENT", "evidence": {"event_type": et}}],
        )

    # Validate response type
    allowed = _allowed_responses_for_event(et)
    if rt not in allowed:
        return ResponseOutcome(
            ok=False,
            event_type=et,
            response_type=rt,
            reasons=[
                {
                    "code": "RESPONSE_INVALID_TYPE",
                    "evidence": {"event_type": et, "response_type": rt, "allowed": sorted(list(allowed))},
                }
            ],
        )

    # Inputs
    trust0 = clamp01(state.get("trust", 0.5))
    mfr0 = clamp01(state.get("minutes_frustration", 0.0))
    tfr0 = clamp01(state.get("team_frustration", 0.0))
    tr_level0 = int(state.get("trade_request_level") or 0)


    # v2 axis state (safe defaults; older saves may not have these keys)
    rfr0 = clamp01(state.get("role_frustration", 0.0))
    cfr0 = clamp01(state.get("contract_frustration", 0.0))
    hfr0 = clamp01(state.get("health_frustration", 0.0))
    chfr0 = clamp01(state.get("chemistry_frustration", 0.0))
    ufr0 = clamp01(state.get("usage_frustration", 0.0))

    lev = _extract_leverage(state=state, event=event)

    ego = mental_norm(mental, "ego")
    amb = mental_norm(mental, "ambition")
    loy = mental_norm(mental, "loyalty")
    coach = mental_norm(mental, "coachability")
    adapt = mental_norm(mental, "adaptability")

    sev = clamp01(event.get("severity", 0.0))

    # Scaling
    impact = 0.45 + 0.55 * lev
    pos_mult = clamp(0.85 + 0.35 * coach + 0.25 * loy + 0.10 * adapt - 0.15 * ego, 0.55, 1.65)
    neg_mult = clamp(0.90 + 0.50 * ego + 0.20 * amb - 0.30 * loy - 0.10 * coach, 0.55, 2.10)
    sev_mult = 0.80 + 0.40 * sev

    # Defaults
    trust1 = trust0
    mfr1 = mfr0
    tfr1 = tfr0
    tr_level1 = tr_level0


    rfr1 = rfr0
    cfr1 = cfr0
    hfr1 = hfr0
    chfr1 = chfr0
    ufr1 = ufr0

    promise: Optional[PromiseSpec] = None
    reasons: List[Dict[str, Any]] = []

    bulk_state_deltas: Dict[str, Dict[str, Any]] = {}

    # Apply
    if et == "MINUTES_COMPLAINT":
        trust1, mfr1, tfr1, tr_level1, promise, reasons = _apply_minutes_complaint(
            rt,
            payload,
            now_d,
            trust0=trust0,
            mfr0=mfr0,
            tfr0=tfr0,
            tr_level0=tr_level0,
            lev=lev,
            ego=ego,
            amb=amb,
            loy=loy,
            coach=coach,
            impact=impact,
            pos_mult=pos_mult,
            neg_mult=neg_mult,
            sev_mult=sev_mult,
            rcfg=rcfg,
            event=event,
            state=state,
        )

    elif et == "HELP_DEMAND":
        trust1, mfr1, tfr1, tr_level1, promise, reasons = _apply_help_demand(
            rt,
            payload,
            now_d,
            trust0=trust0,
            mfr0=mfr0,
            tfr0=tfr0,
            tr_level0=tr_level0,
            lev=lev,
            ego=ego,
            amb=amb,
            loy=loy,
            coach=coach,
            impact=impact,
            pos_mult=pos_mult,
            neg_mult=neg_mult,
            sev_mult=sev_mult,
            rcfg=rcfg,
            event=event,
        )

    elif et in {"TRADE_REQUEST", "TRADE_REQUEST_PUBLIC"}:
        trust1, mfr1, tfr1, tr_level1, promise, reasons = _apply_trade_request(
            et,
            rt,
            payload,
            now_d,
            trust0=trust0,
            mfr0=mfr0,
            tfr0=tfr0,
            tr_level0=tr_level0,
            lev=lev,
            ego=ego,
            amb=amb,
            loy=loy,
            coach=coach,
            impact=impact,
            pos_mult=pos_mult,
            neg_mult=neg_mult,
            sev_mult=sev_mult,
            rcfg=rcfg,
            event=event,
        )


    elif et == "LOCKER_ROOM_MEETING":
        # Team meeting: affects multiple players. We return per-player deltas and
        # let the DB orchestration layer apply them to each current state.
        bulk_state_deltas = {}

        # Base deltas (scaled by severity and leverage of the spokesperson)
        if rt == "TEAM_TALK_CALM":
            dt_trust = float(rcfg.team_talk_trust_calm) * impact * pos_mult * sev_mult
            dt_chem = -float(rcfg.team_talk_chem_relief_calm) * impact * pos_mult * sev_mult
            dt_team = -float(rcfg.team_talk_team_relief_calm) * impact * pos_mult * sev_mult
            reasons = [{"code": "TEAM_TALK_CALM", "evidence": {"trust_delta": dt_trust}}]

        elif rt == "TEAM_TALK_FIRM":
            dt_trust = float(rcfg.team_talk_trust_firm) * impact * neg_mult * sev_mult
            dt_chem = -float(rcfg.team_talk_chem_relief_firm) * impact * pos_mult * sev_mult
            dt_team = -float(rcfg.team_talk_team_relief_firm) * impact * pos_mult * sev_mult
            reasons = [{"code": "TEAM_TALK_FIRM", "evidence": {"trust_delta": dt_trust}}]

        else:  # TEAM_TALK_IGNORE
            dt_trust = float(rcfg.team_talk_trust_ignore) * impact * neg_mult * sev_mult
            dt_chem = float(rcfg.team_talk_chem_bump_ignore) * impact * neg_mult * sev_mult
            dt_team = float(rcfg.team_talk_team_bump_ignore) * impact * neg_mult * sev_mult
            reasons = [{"code": "TEAM_TALK_IGNORE", "evidence": {"trust_delta": dt_trust}}]

        # Affected players list (best-effort).
        affected = []
        evp = event.get("payload")
        if isinstance(evp, Mapping):
            ap = evp.get("affected_player_ids")
            if isinstance(ap, list):
                affected = [str(x) for x in ap if str(x).strip()]

        if not affected:
            affected = [str(event.get("player_id") or "")]
            affected = [x for x in affected if x]

        per = {"trust": float(dt_trust), "chemistry_frustration": float(dt_chem), "team_frustration": float(dt_team)}
        for pid in affected:
            bulk_state_deltas[str(pid)] = dict(per)

        # Apply to the spokesperson's state locally so meta/after are meaningful.
        trust1 = trust0 + float(dt_trust)
        tfr1 = tfr0 + float(dt_team)
        chfr1 = chfr0 + float(dt_chem)

    else:
        # v2 issue families (axis-based). Some axes allow promises.
        axis = _axis_for_v2_event(et)

        if rt == "ACKNOWLEDGE":
            trust1 = trust0 + rcfg.trust_acknowledge * impact * pos_mult * sev_mult
            if axis == "TEAM":
                delta = -rcfg.team_relief_acknowledge * impact * pos_mult * sev_mult
            else:
                delta = -rcfg.axis_relief_acknowledge * impact * pos_mult * sev_mult
            reasons = [{"code": "V2_ACKNOWLEDGE", "evidence": {"axis": axis}}]

        elif rt == "DISMISS":
            trust1 = trust0 - rcfg.trust_dismiss_penalty * impact * neg_mult * sev_mult
            if axis == "TEAM":
                delta = rcfg.team_bump_refuse_trade * 0.35 * impact * neg_mult * sev_mult
            else:
                delta = rcfg.axis_bump_dismiss * impact * neg_mult * sev_mult
            reasons = [{"code": "V2_DISMISS", "evidence": {"axis": axis}}]

        elif rt == "PROMISE_ROLE" and axis == "ROLE":
            dt = float(rcfg.trust_promise) * impact * pos_mult * sev_mult
            trust1 = trust0 + dt
            delta = -rcfg.axis_relief_promise * impact * pos_mult * sev_mult

            # Target role can be provided by UI; default to STARTER.
            target_role = str(payload.get("target_role") or payload.get("role") or "STARTER").upper()
            if target_role not in {"STARTER", "CLOSER", "SIXTH_MAN", "SIXTH", "SIXTHMAN"}:
                target_role = "STARTER"
            if target_role in {"SIXTH", "SIXTHMAN"}:
                target_role = "SIXTH_MAN"

            due = due_month_from_now(now_d, int(rcfg.promise_role_due_months))
            promise = PromiseSpec(
                promise_type="ROLE",
                due_month=due,
                target={
                    "role": target_role,
                    "min_starts_rate": float(rcfg.promise_role_min_starts_rate),
                    "min_closes_rate": float(rcfg.promise_role_min_closes_rate),
                    "source": "role_issue",
                },
            )
            reasons = [{"code": "V2_PROMISE_ROLE_CREATED", "evidence": {"due_month": due, "target_role": target_role, "trust_delta": dt}}]

        elif rt == "PROMISE_LOAD" and axis == "HEALTH":
            dt = float(rcfg.trust_promise) * impact * pos_mult * sev_mult * 0.85
            trust1 = trust0 + dt
            delta = -rcfg.axis_relief_promise * impact * pos_mult * sev_mult

            max_mpg = safe_float_opt(payload.get("max_mpg"))
            if max_mpg is None:
                max_mpg = safe_float_opt(payload.get("target_mpg"))
            if max_mpg is None:
                max_mpg = float(rcfg.promise_load_default_max_mpg)

            max_mpg = float(clamp(max_mpg, 8.0, 40.0))

            due = due_month_from_now(now_d, int(rcfg.promise_load_due_months))
            promise = PromiseSpec(
                promise_type="LOAD",
                due_month=due,
                target_value=float(max_mpg),
                target={"mode": "MAX_MPG", "source": "health_issue"},
            )
            reasons = [{"code": "V2_PROMISE_LOAD_CREATED", "evidence": {"due_month": due, "max_mpg": float(max_mpg), "trust_delta": dt}}]

        elif rt == "PROMISE_EXTENSION_TALKS" and axis == "CONTRACT":
            dt = float(rcfg.trust_promise) * impact * pos_mult * sev_mult * 0.80
            trust1 = trust0 + dt
            delta = -rcfg.axis_relief_promise * impact * pos_mult * sev_mult

            base_cid = None
            base_end = None
            ev_payload = event.get("payload")
            if isinstance(ev_payload, Mapping):
                base_cid = ev_payload.get("active_contract_id")
                base_end = ev_payload.get("contract_end_season_id")

            due = due_month_from_now(now_d, int(rcfg.promise_extension_due_months))
            promise = PromiseSpec(
                promise_type="EXTENSION_TALKS",
                due_month=due,
                target={
                    "baseline_active_contract_id": str(base_cid) if base_cid else None,
                    "baseline_end_season_id": str(base_end) if base_end else None,
                    "source": "contract_issue",
                },
            )
            reasons = [{"code": "V2_PROMISE_EXTENSION_TALKS_CREATED", "evidence": {"due_month": due, "trust_delta": dt}}]

        elif rt == "PROMISE_HELP" and axis == "TEAM":
            dt = float(rcfg.trust_promise) * impact * pos_mult * sev_mult * 0.85
            trust1 = trust0 + dt
            delta = -rcfg.team_relief_promise * impact * sev_mult

            due = due_month_from_now(now_d, int(rcfg.promise_help_due_months))
            tgt: Dict[str, Any] = {"source": "team_issue"}
            need_tags = payload.get("need_tags")
            need_tag = payload.get("need_tag")
            if (not isinstance(need_tags, list) or not need_tags) and (need_tag is None or not str(need_tag).strip()):
                evp2 = event.get("payload")
                if isinstance(evp2, Mapping):
                    need_tags = evp2.get("need_tags")
                    if need_tag is None:
                        need_tag = evp2.get("need_tag")

            norm_tags: List[str] = []
            if isinstance(need_tags, list):
                for x in need_tags:
                    if x is None:
                        continue
                    s = str(x).strip().upper()
                    if s:
                        norm_tags.append(s)

            if not norm_tags and need_tag is not None:
                s = str(need_tag).strip().upper()
                if s:
                    norm_tags = [s]

            if norm_tags:
                tgt["need_tags"] = norm_tags
            promise = PromiseSpec(promise_type="HELP", due_month=due, target=tgt)
            reasons = [{"code": "V2_PROMISE_HELP_CREATED", "evidence": {"due_month": due, "trust_delta": dt, "need_tags": tgt.get("need_tags")}}]

        else:
            return ResponseOutcome(
                ok=False,
                event_type=et,
                response_type=rt,
                reasons=[{"code": "RESPONSE_INVALID_TYPE", "evidence": {"event_type": et, "response_type": rt}}],
            )

        # Apply axis deltas
        if axis == "ROLE":
            rfr1 = rfr0 + delta
        elif axis == "CONTRACT":
            cfr1 = cfr0 + delta
        elif axis == "HEALTH":
            hfr1 = hfr0 + delta
        elif axis == "CHEMISTRY":
            chfr1 = chfr0 + delta
        elif axis == "TEAM":
            tfr1 = tfr0 + delta
    trust1 = clamp01(trust1)
    mfr1 = clamp01(mfr1)
    tfr1 = clamp01(tfr1)

    rfr1 = clamp01(rfr1)
    cfr1 = clamp01(cfr1)
    hfr1 = clamp01(hfr1)
    chfr1 = clamp01(chfr1)
    ufr1 = clamp01(ufr1)

    tr_level1 = int(max(0, min(2, tr_level1)))

    # Tone + player reply (simple v1; keep UI-friendly)
    tone = _tone_for_response(et, rt, ego=ego, sev=sev)
    reply = _player_reply(et, rt, tone=tone)

    meta = {
        "inputs": {
            "event_type": et,
            "response_type": rt,
            "severity": float(sev),
            "leverage": float(lev),
            "impact": float(impact),
            "pos_mult": float(pos_mult),
            "neg_mult": float(neg_mult),
            "sev_mult": float(sev_mult),
            "mental": {
                "ego": float(ego),
                "ambition": float(amb),
                "loyalty": float(loy),
                "coachability": float(coach),
                "adaptability": float(adapt),
            },
        },
        "before": {
            "trust": float(trust0),
            "minutes_frustration": float(mfr0),
            "team_frustration": float(tfr0),
            "role_frustration": float(rfr0),
            "contract_frustration": float(cfr0),
            "health_frustration": float(hfr0),
            "chemistry_frustration": float(chfr0),
            "usage_frustration": float(ufr0),
            "trade_request_level": int(tr_level0),
        },
        "after": {
            "trust": float(trust1),
            "minutes_frustration": float(mfr1),
            "team_frustration": float(tfr1),
            "role_frustration": float(rfr1),
            "contract_frustration": float(cfr1),
            "health_frustration": float(hfr1),
            "chemistry_frustration": float(chfr1),
            "usage_frustration": float(ufr1),
            "trade_request_level": int(tr_level1),
        },
        "deltas": {
            "trust": float(trust1 - trust0),
            "minutes_frustration": float(mfr1 - mfr0),
            "team_frustration": float(tfr1 - tfr0),
            "role_frustration": float(rfr1 - rfr0),
            "contract_frustration": float(cfr1 - cfr0),
            "health_frustration": float(hfr1 - hfr0),
            "chemistry_frustration": float(chfr1 - chfr0),
            "usage_frustration": float(ufr1 - ufr0),
        },
        "bulk": {
            "player_count": int(len(bulk_state_deltas or {})),
        },
    }

    updates: Dict[str, Any] = {
        "trust": float(trust1),
        "minutes_frustration": float(mfr1),
        "team_frustration": float(tfr1),
        "role_frustration": float(rfr1),
        "contract_frustration": float(cfr1),
        "health_frustration": float(hfr1),
        "chemistry_frustration": float(chfr1),
        "usage_frustration": float(ufr1),
        "trade_request_level": int(tr_level1),
    }

    return ResponseOutcome(
        ok=True,
        event_type=et,
        response_type=rt,
        state_updates=updates,
        bulk_state_deltas=bulk_state_deltas,
        promise=promise,
        tone=tone,
        player_reply=reply,
        reasons=reasons,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _allowed_responses_for_event(event_type: str) -> set[str]:
    et = str(event_type or "").upper()
    if et == "MINUTES_COMPLAINT":
        return {"ACKNOWLEDGE", "PROMISE_MINUTES", "DISMISS"}
    if et == "HELP_DEMAND":
        return {"ACKNOWLEDGE", "PROMISE_HELP", "REFUSE_HELP"}
    if et in {"TRADE_REQUEST", "TRADE_REQUEST_PUBLIC"}:
        return {"ACKNOWLEDGE", "SHOP_TRADE", "REFUSE_TRADE", "PROMISE_COMPETE"}

    # v2 issue families
    if et.startswith("ROLE_"):
        return {"ACKNOWLEDGE", "PROMISE_ROLE", "DISMISS"}
    if et.startswith("CONTRACT_"):
        return {"ACKNOWLEDGE", "PROMISE_EXTENSION_TALKS", "DISMISS"}
    if et.startswith("HEALTH_"):
        return {"ACKNOWLEDGE", "PROMISE_LOAD", "DISMISS"}
    if et.startswith("TEAM_"):
        return {"ACKNOWLEDGE", "PROMISE_HELP", "DISMISS"}
    if et.startswith("CHEMISTRY_"):
        return {"ACKNOWLEDGE", "DISMISS"}

    if et == "LOCKER_ROOM_MEETING":
        return {"TEAM_TALK_CALM", "TEAM_TALK_FIRM", "TEAM_TALK_IGNORE"}

    return {"ACKNOWLEDGE"}


def _axis_for_v2_event(event_type: str) -> str:
    et = str(event_type or "").upper()
    if et.startswith("ROLE_"):
        return "ROLE"
    if et.startswith("CONTRACT_"):
        return "CONTRACT"
    if et.startswith("HEALTH_"):
        return "HEALTH"
    if et.startswith("CHEMISTRY_"):
        return "CHEMISTRY"
    if et.startswith("TEAM_"):
        return "TEAM"
    return "UNKNOWN"


def _extract_leverage(*, state: Mapping[str, Any], event: Mapping[str, Any]) -> float:
    # Prefer state leverage (current roster), then event payload leverage.
    lev = safe_float_opt(state.get("leverage"))
    if lev is None:
        payload = event.get("payload")
        if isinstance(payload, Mapping):
            lev = safe_float(payload.get("leverage"), 0.0)
        else:
            lev = 0.0
    return float(clamp01(lev))


def _tone_for_response(event_type: str, response_type: str, *, ego: float, sev: float) -> Literal["CALM", "FIRM", "ANGRY"]:
    rt = str(response_type).upper()

    # Promises and positive team talks
    if rt in {
        "PROMISE_MINUTES",
        "PROMISE_HELP",
        "SHOP_TRADE",
        "PROMISE_COMPETE",
        "PROMISE_ROLE",
        "PROMISE_LOAD",
        "PROMISE_EXTENSION_TALKS",
        "TEAM_TALK_CALM",
    }:
        return "CALM"

    if rt == "TEAM_TALK_FIRM":
        return "FIRM"
    if rt == "TEAM_TALK_IGNORE":
        return "ANGRY"

    if rt in {"ACKNOWLEDGE"}:
        return "FIRM" if sev > 0.65 else "CALM"

    # Negative responses
    if ego > 0.70 or sev > 0.75:
        return "ANGRY"
    return "FIRM"


def _player_reply(event_type: str, response_type: str, *, tone: str) -> str:
    # Keep this short; UI can localize or override.
    rt = str(response_type).upper()

    if rt == "ACKNOWLEDGE":
        return "Alright. I hear you." if tone != "ANGRY" else "You better mean that."

    if rt in {
        "PROMISE_MINUTES",
        "PROMISE_HELP",
        "SHOP_TRADE",
        "PROMISE_COMPETE",
        "PROMISE_ROLE",
        "PROMISE_LOAD",
        "PROMISE_EXTENSION_TALKS",
    }:
        return "Okay. I'll hold you to that." if tone != "ANGRY" else "Don't waste my time."

    if rt == "TEAM_TALK_CALM":
        return "Okay. Let's reset." if tone != "ANGRY" else "We'll see."
    if rt == "TEAM_TALK_FIRM":
        return "We'll respond the right way." if tone != "ANGRY" else "Don't lecture me."
    if rt == "TEAM_TALK_IGNORE":
        return "..." if tone != "ANGRY" else "Unbelievable."

    if rt in {"DISMISS", "REFUSE_HELP", "REFUSE_TRADE"}:
        return "So that's how it is." if tone != "ANGRY" else "This is disrespectful."

    return "Understood."


# ---------------------------------------------------------------------------
# Event-specific appliers
# ---------------------------------------------------------------------------


def _apply_minutes_complaint(
    rt: str,
    payload: Mapping[str, Any],
    now_date_iso: str,
    *,
    trust0: float,
    mfr0: float,
    tfr0: float,
    tr_level0: int,
    lev: float,
    ego: float,
    amb: float,
    loy: float,
    coach: float,
    impact: float,
    pos_mult: float,
    neg_mult: float,
    sev_mult: float,
    rcfg: ResponseConfig,
    event: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[float, float, float, int, Optional[PromiseSpec], List[Dict[str, Any]]]:
    reasons: List[Dict[str, Any]] = []
    trust1, mfr1, tfr1 = trust0, mfr0, tfr0
    tr_level1 = tr_level0
    promise: Optional[PromiseSpec] = None

    if rt == "ACKNOWLEDGE":
        dt = float(rcfg.trust_acknowledge) * impact * pos_mult * sev_mult
        trust1 += dt
        mfr1 -= float(rcfg.minutes_relief_acknowledge) * impact * (0.65 + 0.35 * coach) * sev_mult
        reasons.append({"code": "ACKNOWLEDGED_MINUTES", "evidence": {"trust_delta": dt}})

    elif rt == "PROMISE_MINUTES":
        dt = float(rcfg.trust_promise) * impact * pos_mult * sev_mult
        trust1 += dt
        mfr1 -= float(rcfg.minutes_relief_promise) * impact * sev_mult

        # Target MPG
        target = safe_float_opt(payload.get("target_mpg"))
        if target is None:
            # Try event/state expected mpg
            evp = event.get("payload")
            target = None
            if isinstance(evp, Mapping):
                target = safe_float_opt(evp.get("expected_mpg"))
            if target is None:
                target = safe_float(state.get("minutes_expected_mpg"), 24.0)

        target = float(clamp(target, float(rcfg.promise_minutes_min_mpg), float(rcfg.promise_minutes_max_mpg)))

        due = due_month_from_now(now_date_iso, int(rcfg.promise_minutes_due_months))
        promise = PromiseSpec(
            promise_type="MINUTES",
            due_month=due,
            target_value=float(target),
            target={"target_mpg": float(target)},
        )

        reasons.append(
            {
                "code": "PROMISE_MINUTES_CREATED",
                "evidence": {"due_month": due, "target_mpg": float(target), "trust_delta": dt},
            }
        )

    elif rt == "DISMISS":
        dt = -float(rcfg.trust_dismiss_penalty) * impact * neg_mult * sev_mult
        trust1 += dt
        mfr1 += float(rcfg.minutes_bump_dismiss) * impact * (0.70 + 0.30 * ego) * sev_mult

        # Dismissal can accelerate a trade request *only* if frustration is already extreme and leverage is meaningful.
        if mfr1 >= float(rcfg.dismiss_escalate_trade_fr_threshold) and lev >= 0.60 and (ego >= 0.65 or amb >= 0.65):
            tr_level1 = max(tr_level1, 1)
            reasons.append({"code": "DISMISS_ESCALATED_TRADE_PRESSURE", "evidence": {"trade_request_level": tr_level1}})

        reasons.append({"code": "DISMISSED_MINUTES", "evidence": {"trust_delta": dt}})

    return trust1, mfr1, tfr1, tr_level1, promise, reasons


def _apply_help_demand(
    rt: str,
    payload: Mapping[str, Any],
    now_date_iso: str,
    *,
    trust0: float,
    mfr0: float,
    tfr0: float,
    tr_level0: int,
    lev: float,
    ego: float,
    amb: float,
    loy: float,
    coach: float,
    impact: float,
    pos_mult: float,
    neg_mult: float,
    sev_mult: float,
    rcfg: ResponseConfig,
    event: Mapping[str, Any],
) -> tuple[float, float, float, int, Optional[PromiseSpec], List[Dict[str, Any]]]:
    reasons: List[Dict[str, Any]] = []
    trust1, mfr1, tfr1 = trust0, mfr0, tfr0
    tr_level1 = tr_level0
    promise: Optional[PromiseSpec] = None

    if rt == "ACKNOWLEDGE":
        dt = float(rcfg.trust_acknowledge) * impact * pos_mult * sev_mult * 0.70
        trust1 += dt
        tfr1 -= float(rcfg.team_relief_acknowledge) * impact * sev_mult
        reasons.append({"code": "ACKNOWLEDGED_HELP", "evidence": {"trust_delta": dt}})

    elif rt == "PROMISE_HELP":
        dt = float(rcfg.trust_promise) * impact * pos_mult * sev_mult * 0.85
        trust1 += dt
        tfr1 -= float(rcfg.team_relief_promise) * impact * sev_mult

        due = due_month_from_now(now_date_iso, int(rcfg.promise_help_due_months))
      
        need_tags = payload.get("need_tags")
        need_tag = payload.get("need_tag")

        if (not isinstance(need_tags, list) or not need_tags) and (need_tag is None or not str(need_tag).strip()):
            evp = event.get("payload")
            if isinstance(evp, Mapping):
                if not isinstance(need_tags, list) or not need_tags:
                    need_tags = evp.get("need_tags")
                if need_tag is None:
                    need_tag = evp.get("need_tag")

        norm_tags: List[str] = []
        if isinstance(need_tags, list):
            for t in need_tags:
                if t is None:
                    continue
                s = str(t).strip().upper()
                if s:
                    norm_tags.append(s)

        if not norm_tags and need_tag is not None:
            s = str(need_tag).strip().upper()
            if s:
                norm_tags = [s]
      
        target: Dict[str, Any] = {}
        if norm_tags:
            target["need_tags"] = norm_tags

        promise = PromiseSpec(
            promise_type="HELP",
            due_month=due,
            target_value=None,
            target=target,
        )
        reasons.append({"code": "PROMISE_HELP_CREATED", "evidence": {"due_month": due, "need_tags": norm_tags or None}})

    elif rt == "REFUSE_HELP":
        dt = -float(rcfg.trust_refuse_help_penalty) * impact * neg_mult * sev_mult
        trust1 += dt
        tfr1 += float(rcfg.team_bump_refuse_help) * impact * (0.60 + 0.40 * amb) * sev_mult

        # Strong stars may move towards trade talk when team refuses help.
        if lev >= 0.70 and amb >= 0.70 and trust1 < 0.40:
            tr_level1 = max(tr_level1, 1)
            reasons.append({"code": "REFUSE_HELP_INCREASED_TRADE_PRESSURE", "evidence": {"trade_request_level": tr_level1}})

        reasons.append({"code": "REFUSED_HELP", "evidence": {"trust_delta": dt}})

    return trust1, mfr1, tfr1, tr_level1, promise, reasons


def _apply_trade_request(
    event_type: str,
    rt: str,
    payload: Mapping[str, Any],
    now_date_iso: str,
    *,
    trust0: float,
    mfr0: float,
    tfr0: float,
    tr_level0: int,
    lev: float,
    ego: float,
    amb: float,
    loy: float,
    coach: float,
    impact: float,
    pos_mult: float,
    neg_mult: float,
    sev_mult: float,
    rcfg: ResponseConfig,
    event: Mapping[str, Any],
) -> tuple[float, float, float, int, Optional[PromiseSpec], List[Dict[str, Any]]]:
    reasons: List[Dict[str, Any]] = []
    trust1, mfr1, tfr1 = trust0, mfr0, tfr0
    tr_level1 = max(tr_level0, 1)  # a trade request implies at least private level
    promise: Optional[PromiseSpec] = None

    if rt == "ACKNOWLEDGE":
        # Talking without committing: small trust gain, little/no frustration relief.
        dt = float(rcfg.trust_acknowledge) * impact * pos_mult * sev_mult * 0.55
        trust1 += dt
        reasons.append({"code": "ACKNOWLEDGED_TRADE_REQUEST", "evidence": {"trust_delta": dt}})

    elif rt == "SHOP_TRADE":
        dt = float(rcfg.trust_promise) * impact * pos_mult * sev_mult * 0.85
        trust1 += dt
        # Slight relief: being heard matters.
        tfr1 -= float(rcfg.team_relief_promise) * impact * sev_mult * 0.70
        mfr1 -= float(rcfg.minutes_relief_acknowledge) * impact * sev_mult * 0.35

        due_months = int(rcfg.promise_trade_due_months)
        # If already public, shorten the window slightly.
        if str(event_type).upper() == "TRADE_REQUEST_PUBLIC":
            due_months = max(1, due_months - 1)

        due = due_month_from_now(now_date_iso, due_months)
        promise = PromiseSpec(
            promise_type="SHOP_TRADE",
            due_month=due,
            target={"source": "trade_request", "public": str(event_type).upper() == "TRADE_REQUEST_PUBLIC"},
        )
        reasons.append({"code": "PROMISE_SHOP_TRADE_CREATED", "evidence": {"due_month": due, "trust_delta": dt}})

    elif rt == "PROMISE_COMPETE":
        dt = float(rcfg.trust_promise) * impact * pos_mult * sev_mult * 0.75
        trust1 += dt
        tfr1 -= float(rcfg.team_relief_promise) * impact * sev_mult

        due = due_month_from_now(now_date_iso, int(rcfg.promise_help_due_months))
        promise = PromiseSpec(
            promise_type="HELP",
            due_month=due,
            target={"source": "promise_compete"},
        )
        reasons.append({"code": "PROMISE_COMPETE_CREATED_HELP", "evidence": {"due_month": due, "trust_delta": dt}})

    elif rt == "REFUSE_TRADE":
        dt = -float(rcfg.trust_refuse_trade_penalty) * impact * neg_mult * sev_mult
        trust1 += dt
        tfr1 += float(rcfg.team_bump_refuse_trade) * impact * (0.55 + 0.45 * ego) * sev_mult
        mfr1 += float(rcfg.minutes_bump_refuse_trade) * impact * (0.55 + 0.45 * ego) * sev_mult

        # Escalate to public if high-ego/high-leverage and refusal feels like disrespect.
        if ego >= float(rcfg.refuse_trade_public_ego_threshold) and lev >= float(rcfg.refuse_trade_public_leverage_threshold):
            tr_level1 = max(tr_level1, 2)
            reasons.append({"code": "REFUSE_TRADE_ESCALATED_PUBLIC", "evidence": {"trade_request_level": tr_level1}})

        reasons.append({"code": "REFUSED_TRADE", "evidence": {"trust_delta": dt}})

    # Once public, never de-escalate automatically.
    if str(event_type).upper() == "TRADE_REQUEST_PUBLIC":
        tr_level1 = max(tr_level1, 2)

    return trust1, mfr1, tfr1, tr_level1, promise, reasons
