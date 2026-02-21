from __future__ import annotations

"""Issue negotiation (offer evaluation) for agency interactions.

This module enables Football-Manager-like interaction depth:
- Users do not "auto-succeed" by clicking PROMISE.
- Players evaluate the offer against their own standards (self expectations)
  and the manager's credibility.
- Outcomes are deterministic and explainable:
    ACCEPT  -> create a PromiseSpec
    COUNTER -> create a follow-up dialogue event (no promise yet)
    REJECT  -> relationship damage / escalation

The module is pure business logic:
- No DB I/O
- No direct event writing

It returns structured data that the DB/service layer can persist.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

from .credibility import compute_credibility
from .promises import PromiseType
from .temperament import compute_minutes_tolerance_mpg, compute_temperament
from .utils import clamp, clamp01, safe_float, safe_float_opt, safe_int


Axis = Literal[
    "MINUTES",
    "ROLE",
    "HEALTH",
    "CONTRACT",
    "TEAM",
    "CHEMISTRY",
    "USAGE",
]

NegotiationDecision = Literal["ACCEPT", "COUNTER", "REJECT"]


@dataclass(frozen=True, slots=True)
class OfferEvalResult:
    """Result of evaluating a user's offer for a specific issue."""

    decision: NegotiationDecision

    # If COUNTER: what the player asks for next (in Promise target_json shape).
    counter_target: Optional[Dict[str, Any]] = None

    # Whether the offer is considered insulting (lowball).
    is_insult: bool = False

    # Explainability
    reasons: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # If COUNTER: updated dialogue payload to attach to the follow-up event.
    next_dialogue: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Fallback tuning
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _NegotiationTuning:
    """Internal fallback tuning.

    The project later wires these into AgencyConfig.negotiation.
    """

    max_rounds_min: int = 1
    max_rounds_max: int = 3

    # Minimum credibility needed to accept a promise that otherwise meets the ask.
    cred_accept_min: float = 0.40

    # Credibility premium: when credibility is low, the effective ask is higher.
    cred_premium_mpg_max: float = 2.0
    cred_premium_rate_max: float = 0.10

    mpg_round_step: float = 0.5

    # Minutes negotiation bands
    mpg_floor_gap_tolerance_weight: float = 0.40
    mpg_floor_gap_min: float = 1.0
    mpg_floor_gap_max: float = 5.0
    mpg_insult_extra: float = 2.5
    mpg_insult_gap_max: float = 10.0

    # Role negotiation bands (rates in 0..1)
    rate_floor_gap_base: float = 0.10
    rate_floor_gap_max: float = 0.25
    rate_insult_extra: float = 0.10
    rate_insult_gap_max: float = 0.40

    # Load negotiation (max MPG: LOWER is better)
    load_cap_min: float = 10.0
    load_cap_max: float = 36.0
    load_reduction_base: float = 1.0
    load_reduction_scale: float = 4.0
    load_floor_extra: float = 1.5
    load_insult_extra: float = 5.0

    # Extension talks (months ahead: LOWER is better)
    ext_floor_extra_months: int = 2
    ext_insult_extra_months: int = 4


def _get_tuning(cfg: Any) -> _NegotiationTuning:
    ng = getattr(cfg, "negotiation", None)
    if ng is None:
        return _NegotiationTuning()

    d = _NegotiationTuning()

    def gi(name: str, default: int) -> int:
        try:
            return int(getattr(ng, name, default))
        except Exception:
            return int(default)

    def gf(name: str, default: float) -> float:
        try:
            return float(getattr(ng, name, default))
        except Exception:
            return float(default)

    return _NegotiationTuning(
        max_rounds_min=gi("max_rounds_min", d.max_rounds_min),
        max_rounds_max=gi("max_rounds_max", d.max_rounds_max),
        cred_accept_min=gf("cred_accept_min", d.cred_accept_min),
        cred_premium_mpg_max=gf("cred_premium_mpg_max", d.cred_premium_mpg_max),
        cred_premium_rate_max=gf("cred_premium_rate_max", d.cred_premium_rate_max),
        mpg_round_step=gf("mpg_round_step", d.mpg_round_step),
        mpg_floor_gap_tolerance_weight=gf("mpg_floor_gap_tolerance_weight", d.mpg_floor_gap_tolerance_weight),
        mpg_floor_gap_min=gf("mpg_floor_gap_min", d.mpg_floor_gap_min),
        mpg_floor_gap_max=gf("mpg_floor_gap_max", d.mpg_floor_gap_max),
        mpg_insult_extra=gf("mpg_insult_extra", d.mpg_insult_extra),
        mpg_insult_gap_max=gf("mpg_insult_gap_max", d.mpg_insult_gap_max),
        rate_floor_gap_base=gf("rate_floor_gap_base", d.rate_floor_gap_base),
        rate_floor_gap_max=gf("rate_floor_gap_max", d.rate_floor_gap_max),
        rate_insult_extra=gf("rate_insult_extra", d.rate_insult_extra),
        rate_insult_gap_max=gf("rate_insult_gap_max", d.rate_insult_gap_max),
        load_cap_min=gf("load_cap_min", d.load_cap_min),
        load_cap_max=gf("load_cap_max", d.load_cap_max),
        load_reduction_base=gf("load_reduction_base", d.load_reduction_base),
        load_reduction_scale=gf("load_reduction_scale", d.load_reduction_scale),
        load_floor_extra=gf("load_floor_extra", d.load_floor_extra),
        load_insult_extra=gf("load_insult_extra", d.load_insult_extra),
        ext_floor_extra_months=gi("ext_floor_extra_months", d.ext_floor_extra_months),
        ext_insult_extra_months=gi("ext_insult_extra_months", d.ext_insult_extra_months),
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _axis(axis: Any) -> str:
    return str(axis or "").upper() or "MINUTES"


def _get_mem_from_state(state: Mapping[str, Any]) -> Mapping[str, Any]:
    ctx = state.get("context")
    if isinstance(ctx, Mapping):
        mem = ctx.get("mem")
        if isinstance(mem, Mapping):
            return mem
    return {}


def _round_step(x: float, step: float) -> float:
    s = float(step) if float(step) > 1e-9 else 1.0
    return float(round(float(x) / s) * s)


def _clamp_rate(x: Any) -> float:
    return float(clamp01(safe_float(x, 0.0)))


def _clamp_mpg(x: Any, lo: float = 0.0, hi: float = 48.0) -> float:
    return float(clamp(safe_float(x, 0.0), float(lo), float(hi)))


def _clamp_months(x: Any, lo: int = 0, hi: int = 24) -> int:
    try:
        v = int(x)
    except Exception:
        v = int(lo)
    if v < lo:
        v = int(lo)
    if v > hi:
        v = int(hi)
    return int(v)


def _can_counter(round_idx: int, max_rounds: int) -> bool:
    # round is 0-indexed; allow counter while round < max_rounds - 1
    return int(round_idx) < int(max_rounds) - 1


# ---------------------------------------------------------------------------
# Dialogue building
# ---------------------------------------------------------------------------


def build_initial_dialogue(
    *,
    axis: str,
    promise_type: PromiseType,
    state: Mapping[str, Any],
    mental: Mapping[str, Any],
    cfg: Any,
    now_date_iso: str,
    thread_id: Optional[str] = None,
    phase: str = "OPEN",
) -> Dict[str, Any]:
    """Build the initial negotiation dialogue payload for an issue.

    The caller should ideally pass thread_id as the anchor event_id.
    If missing, it is left blank and should be filled in by the caller.
    """

    ng = _get_tuning(cfg)

    ax = _axis(axis)

    trust = float(clamp01(safe_float(state.get("trust", 0.5), 0.5)))
    leverage = float(clamp01(safe_float(state.get("leverage", 0.0), 0.0)))
    cred_damage = float(clamp01(safe_float(state.get("credibility_damage", 0.0), 0.0)))

    mem = _get_mem_from_state(state)

    cred, _cred_meta = compute_credibility(
        trust=trust,
        credibility_damage=cred_damage,
        mem=mem,
        mental=mental,
        promise_type=promise_type,
        cfg=cfg,
    )

    temp = compute_temperament(mental=mental, leverage=leverage, trust=trust, credibility_damage=cred_damage)

    # Rounds are 0-indexed. max_rounds is the number of rounds allowed.
    max_rounds = int(round(float(ng.max_rounds_min) + float(temp.patience) * (float(ng.max_rounds_max) - float(ng.max_rounds_min))))
    if max_rounds < int(ng.max_rounds_min):
        max_rounds = int(ng.max_rounds_min)
    if max_rounds > int(ng.max_rounds_max):
        max_rounds = int(ng.max_rounds_max)

    ask, floor, insult = _build_terms(
        axis=ax,
        promise_type=promise_type,
        state=state,
        mental=mental,
        cfg=cfg,
        ng=ng,
        credibility=float(cred),
        temperament=temp,
    )

    return {
        "thread_id": str(thread_id or ""),
        "round": 0,
        "max_rounds": int(max_rounds),
        "phase": str(phase or "OPEN").upper(),
        "axis": ax,
        "promise_type": str(promise_type).upper(),
        "ask": ask,
        "floor": floor,
        "insult": insult,
        "credibility": float(cred),
        "credibility_damage": float(cred_damage),
        "temperament": {
            "patience": float(temp.patience),
            "concession": float(temp.concession),
            "publicness": float(temp.publicness),
            "insult_sensitivity": float(temp.insult_sensitivity),
        },
        "now_date": str(now_date_iso or ""),
    }


# ---------------------------------------------------------------------------
# Offer evaluation
# ---------------------------------------------------------------------------


def evaluate_offer(
    *,
    axis: str,
    promise_type: PromiseType,
    offer_target: Optional[Mapping[str, Any]],
    state: Mapping[str, Any],
    mental: Mapping[str, Any],
    mem: Optional[Mapping[str, Any]],
    cfg: Any,
    now_date_iso: str,
    dialogue_from_event: Optional[Mapping[str, Any]] = None,
) -> OfferEvalResult:
    """Evaluate an offer and return ACCEPT/COUNTER/REJECT.

    Args:
        axis: issue axis (MINUTES/ROLE/HEALTH/CONTRACT/...)
        promise_type: promise type being offered
        offer_target: proposed promise target_json (may be empty)
        state: current agency state mapping
        mental: player's mental traits
        mem: memory mapping (state.context['mem']); if None, derived from state
        cfg: AgencyConfig-like
        now_date_iso: used for explainability/consistency
        dialogue_from_event: existing dialogue payload, if the offer is in response
            to a prior dialogue event.
    """

    ax = _axis(axis)
    ng = _get_tuning(cfg)

    # Load dialogue. If missing, construct a new one.
    if isinstance(dialogue_from_event, Mapping):
        dialogue = dict(dialogue_from_event)
    else:
        dialogue = build_initial_dialogue(
            axis=ax,
            promise_type=promise_type,
            state=state,
            mental=mental,
            cfg=cfg,
            now_date_iso=now_date_iso,
            thread_id=None,
            phase="OPEN",
        )

    round_idx = safe_int(dialogue.get("round"), 0)
    max_rounds = safe_int(dialogue.get("max_rounds"), int(ng.max_rounds_min))
    if max_rounds < 1:
        max_rounds = 1

    can_counter = _can_counter(round_idx, max_rounds)

    # Credibility/temperament should be computed fresh (mem could have changed).
    trust = float(clamp01(safe_float(state.get("trust", 0.5), 0.5)))
    leverage = float(clamp01(safe_float(state.get("leverage", 0.0), 0.0)))
    cred_damage = float(clamp01(safe_float(state.get("credibility_damage", 0.0), 0.0)))

    mem0 = mem if isinstance(mem, Mapping) else _get_mem_from_state(state)

    cred, cred_meta = compute_credibility(
        trust=trust,
        credibility_damage=cred_damage,
        mem=mem0,
        mental=mental,
        promise_type=promise_type,
        cfg=cfg,
    )

    temp = compute_temperament(mental=mental, leverage=leverage, trust=trust, credibility_damage=cred_damage)

    # If dialogue lacks ask/floor/insult, rebuild terms.
    ask = dialogue.get("ask") if isinstance(dialogue.get("ask"), Mapping) else None
    floor = dialogue.get("floor") if isinstance(dialogue.get("floor"), Mapping) else None
    insult = dialogue.get("insult") if isinstance(dialogue.get("insult"), Mapping) else None

    if ask is None or floor is None or insult is None:
        ask, floor, insult = _build_terms(
            axis=ax,
            promise_type=promise_type,
            state=state,
            mental=mental,
            cfg=cfg,
            ng=ng,
            credibility=float(cred),
            temperament=temp,
        )
        dialogue["ask"] = ask
        dialogue["floor"] = floor
        dialogue["insult"] = insult

    offer = dict(offer_target or {})

    # Normalize and dispatch by promise type.
    ptype = str(promise_type).upper()

    if ptype == "MINUTES":
        return _eval_minutes(
            axis=ax,
            offer=offer,
            dialogue=dialogue,
            credibility=float(cred),
            cred_meta=cred_meta,
            temperament=temp,
            can_counter=can_counter,
            ng=ng,
        )

    if ptype == "ROLE":
        return _eval_role(
            axis=ax,
            offer=offer,
            dialogue=dialogue,
            credibility=float(cred),
            cred_meta=cred_meta,
            temperament=temp,
            can_counter=can_counter,
            ng=ng,
            cfg=cfg,
        )

    if ptype == "LOAD":
        return _eval_load(
            axis=ax,
            offer=offer,
            dialogue=dialogue,
            credibility=float(cred),
            cred_meta=cred_meta,
            temperament=temp,
            can_counter=can_counter,
            ng=ng,
        )

    if ptype == "EXTENSION_TALKS":
        return _eval_extension_talks(
            axis=ax,
            offer=offer,
            dialogue=dialogue,
            credibility=float(cred),
            cred_meta=cred_meta,
            temperament=temp,
            can_counter=can_counter,
            ng=ng,
        )

    if ptype == "HELP":
        return _eval_help(
            axis=ax,
            offer=offer,
            dialogue=dialogue,
            credibility=float(cred),
            cred_meta=cred_meta,
            temperament=temp,
            can_counter=can_counter,
            ng=ng,
        )

    # Unsupported promise type => be safe (reject, explain).
    return OfferEvalResult(
        decision="REJECT",
        counter_target=None,
        is_insult=False,
        reasons=[{"code": "NEGOTIATION_UNSUPPORTED_PROMISE_TYPE", "evidence": {"promise_type": ptype}}],
        meta={"axis": ax, "promise_type": ptype, "credibility": float(cred)},
        next_dialogue=None,
    )


# ---------------------------------------------------------------------------
# Term building
# ---------------------------------------------------------------------------


def _build_terms(
    *,
    axis: str,
    promise_type: PromiseType,
    state: Mapping[str, Any],
    mental: Mapping[str, Any],
    cfg: Any,
    ng: _NegotiationTuning,
    credibility: float,
    temperament: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Build (ask, floor, insult) dicts for the promise type."""

    ptype = str(promise_type).upper()
    cred = float(clamp01(credibility))

    # Use self expectations if present.
    self_mpg = safe_float(state.get("self_expected_mpg"), 0.0)
    team_mpg = safe_float(state.get("minutes_expected_mpg"), 0.0)
    base_mpg = float(self_mpg) if float(self_mpg) > 0.0 else float(team_mpg)
    base_mpg = float(clamp(base_mpg, 0.0, 48.0))

    if ptype == "MINUTES":
        # Credibility premium makes the ask slightly higher when credibility is low.
        ask_mpg = float(clamp(base_mpg + (1.0 - cred) * float(ng.cred_premium_mpg_max), 0.0, 48.0))

        tol = compute_minutes_tolerance_mpg(mental=mental, cfg=cfg)
        floor_gap = float(ng.mpg_floor_gap_min) + float(ng.mpg_floor_gap_tolerance_weight) * float(tol)
        floor_gap = float(clamp(floor_gap, float(ng.mpg_floor_gap_min), float(ng.mpg_floor_gap_max)))

        floor_mpg = float(clamp(ask_mpg - floor_gap, 0.0, 48.0))

        insult_gap = float(min(float(ng.mpg_insult_gap_max), floor_gap + float(ng.mpg_insult_extra)))
        insult_mpg = float(clamp(ask_mpg - insult_gap, 0.0, 48.0))

        return {"mpg": float(ask_mpg)}, {"mpg": float(floor_mpg)}, {"mpg": float(insult_mpg)}

    if ptype == "ROLE":
        # Determine desired focus (STARTER vs CLOSER) based on self-expected gaps.
        self_s = safe_float(state.get("self_expected_starts_rate"), 0.0)
        self_c = safe_float(state.get("self_expected_closes_rate"), 0.0)
        self_s = float(clamp01(self_s))
        self_c = float(clamp01(self_c))

        # Use last known actual rates when available to pick focus.
        actual_s = float(clamp01(safe_float(state.get("starts_rate"), 0.0)))
        actual_c = float(clamp01(safe_float(state.get("closes_rate"), 0.0)))

        gap_s = max(0.0, self_s - actual_s)
        gap_c = max(0.0, self_c - actual_c)

        desired = "STARTER"
        if gap_c >= gap_s + 0.08 and self_c >= 0.20:
            desired = "CLOSER"

        # Apply credibility premium to the relevant ask rate.
        ask_s = float(clamp01(self_s + (1.0 - cred) * float(ng.cred_premium_rate_max)))
        ask_c = float(clamp01(self_c + (1.0 - cred) * float(ng.cred_premium_rate_max)))

        # Band width: base + small modulation from concession.
        floor_gap = float(ng.rate_floor_gap_base) + 0.10 * float(1.0 - float(getattr(temperament, "concession", 0.45)))
        floor_gap = float(clamp(floor_gap, float(ng.rate_floor_gap_base), float(ng.rate_floor_gap_max)))
        insult_gap = float(min(float(ng.rate_insult_gap_max), floor_gap + float(ng.rate_insult_extra)))

        if desired == "STARTER":
            floor_s = float(clamp01(ask_s - floor_gap))
            insult_s = float(clamp01(ask_s - insult_gap))
            ask = {"role": desired, "min_starts_rate": float(ask_s)}
            floor = {"role": desired, "min_starts_rate": float(floor_s)}
            insult = {"role": desired, "min_starts_rate": float(insult_s)}
            return ask, floor, insult

        # CLOSER focuses on closes rate.
        floor_c = float(clamp01(ask_c - floor_gap))
        insult_c = float(clamp01(ask_c - insult_gap))
        ask = {"role": desired, "min_closes_rate": float(ask_c)}
        floor = {"role": desired, "min_closes_rate": float(floor_c)}
        insult = {"role": desired, "min_closes_rate": float(insult_c)}
        return ask, floor, insult

    if ptype == "LOAD":
        # Ask for a lower max MPG cap when health frustration is higher.
        hfr = float(clamp01(safe_float(state.get("health_frustration"), 0.0)))
        reduction = float(ng.load_reduction_base) + float(ng.load_reduction_scale) * hfr

        ask_max = float(clamp(base_mpg - reduction, float(ng.load_cap_min), float(ng.load_cap_max)))

        # Low credibility -> more strict (lower cap).
        ask_max = float(clamp(ask_max - (1.0 - cred) * float(ng.cred_premium_mpg_max), float(ng.load_cap_min), float(ng.load_cap_max)))

        floor_max = float(clamp(ask_max + float(ng.load_floor_extra), float(ng.load_cap_min), float(ng.load_cap_max)))
        insult_max = float(clamp(ask_max + float(ng.load_insult_extra), float(ng.load_cap_min), float(ng.load_cap_max)))

        return {"max_mpg": float(ask_max), "mode": "MAX_MPG"}, {"max_mpg": float(floor_max)}, {"max_mpg": float(insult_max)}

    if ptype == "EXTENSION_TALKS":
        cfr = float(clamp01(safe_float(state.get("contract_frustration"), 0.0)))

        # Higher frustration => sooner talks.
        base_months = float(clamp(2.0 - 2.5 * cfr, 0.0, 6.0))
        ask_months = int(round(base_months))

        # Low credibility -> ask for even sooner talks (slightly).
        if cred < 0.50:
            ask_months = max(0, ask_months - 1)

        floor_months = int(min(24, ask_months + int(ng.ext_floor_extra_months)))
        insult_months = int(min(24, ask_months + int(ng.ext_insult_extra_months)))

        return {"months_ahead": int(ask_months)}, {"months_ahead": int(floor_months)}, {"months_ahead": int(insult_months)}

    if ptype == "HELP":
        # HELP is not numeric; it negotiates required tags.
        # Ask should be provided by the triggering event, but we fall back safely.
        ask_tags: List[str] = []
        ctx = state.get("context")
        if isinstance(ctx, Mapping):
            # tick.py stores help needs in context["team"]["help_need_tags"] sometimes
            tctx = ctx.get("team")
            if isinstance(tctx, Mapping):
                tags = tctx.get("help_need_tags")
                if isinstance(tags, list):
                    ask_tags = [str(x).upper() for x in tags if str(x).strip()]

        ask = {"need_tags": ask_tags}
        return ask, dict(ask), dict(ask)

    # Default: minimal terms.
    return {}, {}, {}


# ---------------------------------------------------------------------------
# Type-specific evaluators
# ---------------------------------------------------------------------------


def _accept_allowed(*, credibility: float, ask: float, offer: float, higher_is_better: bool, ng: _NegotiationTuning) -> bool:
    """Return True if credibility allows accepting an offer meeting ask.

    We allow "generous" offers to overcome low credibility.
    """

    cred = float(clamp01(credibility))
    if cred >= float(ng.cred_accept_min):
        return True

    # Generous override margins (fixed constants for now).
    if higher_is_better:
        return float(offer) >= float(ask) + 1.0
    return float(offer) <= float(ask) - 1.0


def _counter_dialogue(
    *,
    dialogue: Mapping[str, Any],
    new_ask: Mapping[str, Any],
    new_floor: Mapping[str, Any],
    new_insult: Mapping[str, Any],
    offer_norm: Mapping[str, Any],
) -> Dict[str, Any]:
    out = dict(dialogue)
    out["round"] = safe_int(out.get("round"), 0) + 1
    out["phase"] = "COUNTER"
    out["ask"] = dict(new_ask)
    out["floor"] = dict(new_floor)
    out["insult"] = dict(new_insult)
    out["last_offer"] = dict(offer_norm)
    return out


def _eval_minutes(
    *,
    axis: str,
    offer: Mapping[str, Any],
    dialogue: Mapping[str, Any],
    credibility: float,
    cred_meta: Mapping[str, Any],
    temperament: Any,
    can_counter: bool,
    ng: _NegotiationTuning,
) -> OfferEvalResult:
    ask_mpg = _clamp_mpg(((dialogue.get("ask") or {}) if isinstance(dialogue.get("ask"), Mapping) else {}).get("mpg"), 0.0, 48.0)
    floor_mpg = _clamp_mpg(((dialogue.get("floor") or {}) if isinstance(dialogue.get("floor"), Mapping) else {}).get("mpg"), 0.0, 48.0)
    insult_mpg = _clamp_mpg(((dialogue.get("insult") or {}) if isinstance(dialogue.get("insult"), Mapping) else {}).get("mpg"), 0.0, 48.0)

    off = safe_float_opt(offer.get("target_mpg"))
    if off is None:
        off = safe_float_opt(offer.get("mpg"))
    if off is None:
        # If user didn't specify, interpret as offering the current ask.
        off = float(ask_mpg)

    offer_mpg = _clamp_mpg(off, 0.0, 48.0)

    is_insult = bool(offer_mpg < float(insult_mpg) - 1e-9)

    evidence = {
        "ask_mpg": float(ask_mpg),
        "floor_mpg": float(floor_mpg),
        "insult_mpg": float(insult_mpg),
        "offer_mpg": float(offer_mpg),
        "credibility": float(credibility),
    }

    # Decision
    if offer_mpg >= ask_mpg - 1e-9:
        if _accept_allowed(credibility=credibility, ask=ask_mpg, offer=offer_mpg, higher_is_better=True, ng=ng):
            return OfferEvalResult(
                decision="ACCEPT",
                counter_target=None,
                is_insult=False,
                reasons=[{"code": "NEGOTIATION_ACCEPT", "evidence": evidence}],
                meta={"axis": axis, "promise_type": "MINUTES", "cred_meta": dict(cred_meta)},
                next_dialogue=None,
            )
        # Low credibility: require a counter round if possible.
        if can_counter:
            # Ask stays the same; the player wants commitment that feels real.
            nd = _counter_dialogue(
                dialogue=dialogue,
                new_ask={"mpg": float(ask_mpg)},
                new_floor={"mpg": float(floor_mpg)},
                new_insult={"mpg": float(insult_mpg)},
                offer_norm={"target_mpg": float(offer_mpg)},
            )
            return OfferEvalResult(
                decision="COUNTER",
                counter_target={"target_mpg": float(ask_mpg)},
                is_insult=False,
                reasons=[{"code": "NEGOTIATION_COUNTER_LOW_CREDIBILITY", "evidence": evidence}],
                meta={"axis": axis, "promise_type": "MINUTES", "cred_meta": dict(cred_meta)},
                next_dialogue=nd,
            )
        return OfferEvalResult(
            decision="REJECT",
            counter_target=None,
            is_insult=False,
            reasons=[{"code": "NEGOTIATION_REJECT_LOW_CREDIBILITY", "evidence": evidence}],
            meta={"axis": axis, "promise_type": "MINUTES", "cred_meta": dict(cred_meta)},
            next_dialogue=None,
        )

    if offer_mpg >= floor_mpg - 1e-9 and can_counter and not is_insult:
        # Counter with a softened ask between ask and offer.
        conc = float(getattr(temperament, "concession", 0.45))
        counter = float(ask_mpg - conc * (ask_mpg - offer_mpg))

        # Keep the band widths stable.
        floor_gap = float(ask_mpg - floor_mpg)
        insult_gap = float(ask_mpg - insult_mpg)

        counter = float(clamp(counter, float(floor_mpg), float(ask_mpg)))
        counter = _round_step(counter, float(ng.mpg_round_step))

        new_ask = {"mpg": float(counter)}
        new_floor = {"mpg": float(clamp(counter - floor_gap, 0.0, 48.0))}
        new_insult = {"mpg": float(clamp(counter - insult_gap, 0.0, 48.0))}

        nd = _counter_dialogue(
            dialogue=dialogue,
            new_ask=new_ask,
            new_floor=new_floor,
            new_insult=new_insult,
            offer_norm={"target_mpg": float(offer_mpg)},
        )

        return OfferEvalResult(
            decision="COUNTER",
            counter_target={"target_mpg": float(counter)},
            is_insult=False,
            reasons=[{"code": "NEGOTIATION_COUNTER", "evidence": {**evidence, "counter_mpg": float(counter)}}],
            meta={"axis": axis, "promise_type": "MINUTES", "cred_meta": dict(cred_meta)},
            next_dialogue=nd,
        )

    # Reject
    code = "NEGOTIATION_REJECT_BELOW_FLOOR" if offer_mpg < floor_mpg - 1e-9 else "NEGOTIATION_REJECT"
    if is_insult:
        code = "NEGOTIATION_REJECT_INSULT"

    return OfferEvalResult(
        decision="REJECT",
        counter_target=None,
        is_insult=bool(is_insult),
        reasons=[{"code": code, "evidence": evidence}],
        meta={"axis": axis, "promise_type": "MINUTES", "cred_meta": dict(cred_meta)},
        next_dialogue=None,
    )


def _eval_role(
    *,
    axis: str,
    offer: Mapping[str, Any],
    dialogue: Mapping[str, Any],
    credibility: float,
    cred_meta: Mapping[str, Any],
    temperament: Any,
    can_counter: bool,
    ng: _NegotiationTuning,
    cfg: Any,
) -> OfferEvalResult:
    ask_d = dialogue.get("ask") if isinstance(dialogue.get("ask"), Mapping) else {}
    floor_d = dialogue.get("floor") if isinstance(dialogue.get("floor"), Mapping) else {}
    insult_d = dialogue.get("insult") if isinstance(dialogue.get("insult"), Mapping) else {}

    desired = str(ask_d.get("role") or "STARTER").upper()

    higher_is_better = True
    metric_key = "min_starts_rate"

    if desired in {"CLOSER", "CLOSE", "CLOSING"}:
        metric_key = "min_closes_rate"
    elif desired in {"SIXTH", "SIXTH_MAN", "BENCH"}:
        # For sixth-man, lower starts-rate is better. (Not the common path.)
        higher_is_better = False
        metric_key = "max_starts_rate"

    ask_v = _clamp_rate(ask_d.get(metric_key))
    floor_v = _clamp_rate(floor_d.get(metric_key))
    insult_v = _clamp_rate(insult_d.get(metric_key))

    # Parse offer.
    off_v = safe_float_opt(offer.get(metric_key))

    if off_v is None:
        # Allow role label offers.
        role_label = str(offer.get("role") or offer.get("target_role") or "").upper()
        if role_label:
            if role_label in {"STARTER", "START"}:
                off_v = 0.60
            elif role_label in {"CLOSER", "CLOSE", "CLOSING"}:
                off_v = 0.35
            elif role_label in {"SIXTH", "SIXTH_MAN", "BENCH"}:
                off_v = 0.25

    if off_v is None:
        off_v = float(ask_v)  # default to "accept your ask"

    offer_v = _clamp_rate(off_v)

    is_insult = bool((offer_v < insult_v - 1e-9) if higher_is_better else (offer_v > insult_v + 1e-9))

    evidence = {
        "desired": desired,
        "metric": metric_key,
        "ask": float(ask_v),
        "floor": float(floor_v),
        "insult": float(insult_v),
        "offer": float(offer_v),
        "credibility": float(credibility),
    }

    # Accept condition
    accept_cond = (offer_v >= ask_v - 1e-9) if higher_is_better else (offer_v <= ask_v + 1e-9)

    if accept_cond:
        if _accept_allowed(credibility=credibility, ask=ask_v, offer=offer_v, higher_is_better=higher_is_better, ng=ng):
            return OfferEvalResult(
                decision="ACCEPT",
                counter_target=None,
                is_insult=False,
                reasons=[{"code": "NEGOTIATION_ACCEPT", "evidence": evidence}],
                meta={"axis": axis, "promise_type": "ROLE", "cred_meta": dict(cred_meta)},
                next_dialogue=None,
            )
        if can_counter:
            nd = _counter_dialogue(
                dialogue=dialogue,
                new_ask={"role": desired, metric_key: float(ask_v)},
                new_floor={"role": desired, metric_key: float(floor_v)},
                new_insult={"role": desired, metric_key: float(insult_v)},
                offer_norm={"role": desired, metric_key: float(offer_v)},
            )
            return OfferEvalResult(
                decision="COUNTER",
                counter_target={"role": desired, metric_key: float(ask_v)},
                is_insult=False,
                reasons=[{"code": "NEGOTIATION_COUNTER_LOW_CREDIBILITY", "evidence": evidence}],
                meta={"axis": axis, "promise_type": "ROLE", "cred_meta": dict(cred_meta)},
                next_dialogue=nd,
            )
        return OfferEvalResult(
            decision="REJECT",
            counter_target=None,
            is_insult=False,
            reasons=[{"code": "NEGOTIATION_REJECT_LOW_CREDIBILITY", "evidence": evidence}],
            meta={"axis": axis, "promise_type": "ROLE", "cred_meta": dict(cred_meta)},
            next_dialogue=None,
        )

    # Counter condition
    within_floor = (offer_v >= floor_v - 1e-9) if higher_is_better else (offer_v <= floor_v + 1e-9)
    if within_floor and can_counter and not is_insult:
        conc = float(getattr(temperament, "concession", 0.45))

        if higher_is_better:
            counter = float(ask_v - conc * (ask_v - offer_v))
            floor_gap = float(ask_v - floor_v)
            insult_gap = float(ask_v - insult_v)
            counter = float(clamp(counter, float(floor_v), float(ask_v)))
            counter = float(clamp01(_round_step(counter, 0.01)))
            new_ask = {"role": desired, metric_key: float(counter)}
            new_floor = {"role": desired, metric_key: float(clamp01(counter - floor_gap))}
            new_insult = {"role": desired, metric_key: float(clamp01(counter - insult_gap))}
        else:
            counter = float(ask_v + conc * (offer_v - ask_v))
            floor_gap = float(floor_v - ask_v)
            insult_gap = float(insult_v - ask_v)
            counter = float(clamp(counter, float(ask_v), float(floor_v)))
            counter = float(clamp01(_round_step(counter, 0.01)))
            new_ask = {"role": desired, metric_key: float(counter)}
            new_floor = {"role": desired, metric_key: float(clamp01(counter + floor_gap))}
            new_insult = {"role": desired, metric_key: float(clamp01(counter + insult_gap))}

        nd = _counter_dialogue(
            dialogue=dialogue,
            new_ask=new_ask,
            new_floor=new_floor,
            new_insult=new_insult,
            offer_norm={"role": desired, metric_key: float(offer_v)},
        )

        return OfferEvalResult(
            decision="COUNTER",
            counter_target=dict(new_ask),
            is_insult=False,
            reasons=[{"code": "NEGOTIATION_COUNTER", "evidence": {**evidence, "counter": float(counter)}}],
            meta={"axis": axis, "promise_type": "ROLE", "cred_meta": dict(cred_meta)},
            next_dialogue=nd,
        )

    code = "NEGOTIATION_REJECT_BELOW_FLOOR" if not within_floor else "NEGOTIATION_REJECT"
    if is_insult:
        code = "NEGOTIATION_REJECT_INSULT"

    return OfferEvalResult(
        decision="REJECT",
        counter_target=None,
        is_insult=bool(is_insult),
        reasons=[{"code": code, "evidence": evidence}],
        meta={"axis": axis, "promise_type": "ROLE", "cred_meta": dict(cred_meta)},
        next_dialogue=None,
    )


def _eval_load(
    *,
    axis: str,
    offer: Mapping[str, Any],
    dialogue: Mapping[str, Any],
    credibility: float,
    cred_meta: Mapping[str, Any],
    temperament: Any,
    can_counter: bool,
    ng: _NegotiationTuning,
) -> OfferEvalResult:
    ask_d = dialogue.get("ask") if isinstance(dialogue.get("ask"), Mapping) else {}
    floor_d = dialogue.get("floor") if isinstance(dialogue.get("floor"), Mapping) else {}
    insult_d = dialogue.get("insult") if isinstance(dialogue.get("insult"), Mapping) else {}

    ask_max = _clamp_mpg(ask_d.get("max_mpg"), float(ng.load_cap_min), float(ng.load_cap_max))
    floor_max = _clamp_mpg(floor_d.get("max_mpg"), float(ng.load_cap_min), float(ng.load_cap_max))
    insult_max = _clamp_mpg(insult_d.get("max_mpg"), float(ng.load_cap_min), float(ng.load_cap_max))

    off = safe_float_opt(offer.get("max_mpg"))
    if off is None:
        off = safe_float_opt(offer.get("target_mpg"))
    if off is None:
        off = float(ask_max)

    offer_max = _clamp_mpg(off, float(ng.load_cap_min), float(ng.load_cap_max))

    # LOWER is better.
    is_insult = bool(offer_max > float(insult_max) + 1e-9)

    evidence = {
        "ask_max_mpg": float(ask_max),
        "floor_max_mpg": float(floor_max),
        "insult_max_mpg": float(insult_max),
        "offer_max_mpg": float(offer_max),
        "credibility": float(credibility),
    }

    accept_cond = offer_max <= ask_max + 1e-9

    if accept_cond:
        if _accept_allowed(credibility=credibility, ask=ask_max, offer=offer_max, higher_is_better=False, ng=ng):
            return OfferEvalResult(
                decision="ACCEPT",
                counter_target=None,
                is_insult=False,
                reasons=[{"code": "NEGOTIATION_ACCEPT", "evidence": evidence}],
                meta={"axis": axis, "promise_type": "LOAD", "cred_meta": dict(cred_meta)},
                next_dialogue=None,
            )
        if can_counter:
            nd = _counter_dialogue(
                dialogue=dialogue,
                new_ask={"max_mpg": float(ask_max), "mode": "MAX_MPG"},
                new_floor={"max_mpg": float(floor_max)},
                new_insult={"max_mpg": float(insult_max)},
                offer_norm={"max_mpg": float(offer_max)},
            )
            return OfferEvalResult(
                decision="COUNTER",
                counter_target={"max_mpg": float(ask_max), "mode": "MAX_MPG"},
                is_insult=False,
                reasons=[{"code": "NEGOTIATION_COUNTER_LOW_CREDIBILITY", "evidence": evidence}],
                meta={"axis": axis, "promise_type": "LOAD", "cred_meta": dict(cred_meta)},
                next_dialogue=nd,
            )
        return OfferEvalResult(
            decision="REJECT",
            counter_target=None,
            is_insult=False,
            reasons=[{"code": "NEGOTIATION_REJECT_LOW_CREDIBILITY", "evidence": evidence}],
            meta={"axis": axis, "promise_type": "LOAD", "cred_meta": dict(cred_meta)},
            next_dialogue=None,
        )

    within_floor = offer_max <= floor_max + 1e-9
    if within_floor and can_counter and not is_insult:
        conc = float(getattr(temperament, "concession", 0.45))
        counter = float(ask_max + conc * (offer_max - ask_max))

        floor_gap = float(floor_max - ask_max)
        insult_gap = float(insult_max - ask_max)

        counter = float(clamp(counter, float(ask_max), float(floor_max)))
        counter = _round_step(counter, float(ng.mpg_round_step))

        new_ask = {"max_mpg": float(counter), "mode": "MAX_MPG"}
        new_floor = {"max_mpg": float(clamp(counter + floor_gap, float(ng.load_cap_min), float(ng.load_cap_max)))}
        new_insult = {"max_mpg": float(clamp(counter + insult_gap, float(ng.load_cap_min), float(ng.load_cap_max)))}

        nd = _counter_dialogue(
            dialogue=dialogue,
            new_ask=new_ask,
            new_floor=new_floor,
            new_insult=new_insult,
            offer_norm={"max_mpg": float(offer_max)},
        )

        return OfferEvalResult(
            decision="COUNTER",
            counter_target={"max_mpg": float(counter), "mode": "MAX_MPG"},
            is_insult=False,
            reasons=[{"code": "NEGOTIATION_COUNTER", "evidence": {**evidence, "counter_max_mpg": float(counter)}}],
            meta={"axis": axis, "promise_type": "LOAD", "cred_meta": dict(cred_meta)},
            next_dialogue=nd,
        )

    code = "NEGOTIATION_REJECT_ABOVE_FLOOR" if not within_floor else "NEGOTIATION_REJECT"
    if is_insult:
        code = "NEGOTIATION_REJECT_INSULT"

    return OfferEvalResult(
        decision="REJECT",
        counter_target=None,
        is_insult=bool(is_insult),
        reasons=[{"code": code, "evidence": evidence}],
        meta={"axis": axis, "promise_type": "LOAD", "cred_meta": dict(cred_meta)},
        next_dialogue=None,
    )


def _eval_extension_talks(
    *,
    axis: str,
    offer: Mapping[str, Any],
    dialogue: Mapping[str, Any],
    credibility: float,
    cred_meta: Mapping[str, Any],
    temperament: Any,
    can_counter: bool,
    ng: _NegotiationTuning,
) -> OfferEvalResult:
    ask_d = dialogue.get("ask") if isinstance(dialogue.get("ask"), Mapping) else {}
    floor_d = dialogue.get("floor") if isinstance(dialogue.get("floor"), Mapping) else {}
    insult_d = dialogue.get("insult") if isinstance(dialogue.get("insult"), Mapping) else {}

    ask_m = _clamp_months(ask_d.get("months_ahead"), 0, 24)
    floor_m = _clamp_months(floor_d.get("months_ahead"), 0, 24)
    insult_m = _clamp_months(insult_d.get("months_ahead"), 0, 24)

    off = offer.get("months_ahead")
    if off is None:
        off = offer.get("due_months")
    if off is None:
        off = ask_m

    offer_m = _clamp_months(off, 0, 24)

    # LOWER is better.
    is_insult = bool(offer_m > int(insult_m))

    evidence = {
        "ask_months_ahead": int(ask_m),
        "floor_months_ahead": int(floor_m),
        "insult_months_ahead": int(insult_m),
        "offer_months_ahead": int(offer_m),
        "credibility": float(credibility),
    }

    accept_cond = offer_m <= ask_m

    if accept_cond:
        if float(clamp01(credibility)) >= float(ng.cred_accept_min) or offer_m <= max(0, ask_m - 1):
            return OfferEvalResult(
                decision="ACCEPT",
                counter_target=None,
                is_insult=False,
                reasons=[{"code": "NEGOTIATION_ACCEPT", "evidence": evidence}],
                meta={"axis": axis, "promise_type": "EXTENSION_TALKS", "cred_meta": dict(cred_meta)},
                next_dialogue=None,
            )
        if can_counter:
            nd = _counter_dialogue(
                dialogue=dialogue,
                new_ask={"months_ahead": int(ask_m)},
                new_floor={"months_ahead": int(floor_m)},
                new_insult={"months_ahead": int(insult_m)},
                offer_norm={"months_ahead": int(offer_m)},
            )
            return OfferEvalResult(
                decision="COUNTER",
                counter_target={"months_ahead": int(ask_m)},
                is_insult=False,
                reasons=[{"code": "NEGOTIATION_COUNTER_LOW_CREDIBILITY", "evidence": evidence}],
                meta={"axis": axis, "promise_type": "EXTENSION_TALKS", "cred_meta": dict(cred_meta)},
                next_dialogue=nd,
            )
        return OfferEvalResult(
            decision="REJECT",
            counter_target=None,
            is_insult=False,
            reasons=[{"code": "NEGOTIATION_REJECT_LOW_CREDIBILITY", "evidence": evidence}],
            meta={"axis": axis, "promise_type": "EXTENSION_TALKS", "cred_meta": dict(cred_meta)},
            next_dialogue=None,
        )

    within_floor = offer_m <= floor_m
    if within_floor and can_counter and not is_insult:
        conc = float(getattr(temperament, "concession", 0.45))
        # Ask is smaller; offer is bigger.
        counter = int(round(float(ask_m) + conc * float(offer_m - ask_m)))

        floor_gap = int(floor_m - ask_m)
        insult_gap = int(insult_m - ask_m)

        counter = int(clamp(counter, ask_m, floor_m))

        new_ask = {"months_ahead": int(counter)}
        new_floor = {"months_ahead": int(min(24, counter + floor_gap))}
        new_insult = {"months_ahead": int(min(24, counter + insult_gap))}

        nd = _counter_dialogue(
            dialogue=dialogue,
            new_ask=new_ask,
            new_floor=new_floor,
            new_insult=new_insult,
            offer_norm={"months_ahead": int(offer_m)},
        )

        return OfferEvalResult(
            decision="COUNTER",
            counter_target={"months_ahead": int(counter)},
            is_insult=False,
            reasons=[{"code": "NEGOTIATION_COUNTER", "evidence": {**evidence, "counter_months_ahead": int(counter)}}],
            meta={"axis": axis, "promise_type": "EXTENSION_TALKS", "cred_meta": dict(cred_meta)},
            next_dialogue=nd,
        )

    code = "NEGOTIATION_REJECT_ABOVE_FLOOR" if not within_floor else "NEGOTIATION_REJECT"
    if is_insult:
        code = "NEGOTIATION_REJECT_INSULT"

    return OfferEvalResult(
        decision="REJECT",
        counter_target=None,
        is_insult=bool(is_insult),
        reasons=[{"code": code, "evidence": evidence}],
        meta={"axis": axis, "promise_type": "EXTENSION_TALKS", "cred_meta": dict(cred_meta)},
        next_dialogue=None,
    )


def _eval_help(
    *,
    axis: str,
    offer: Mapping[str, Any],
    dialogue: Mapping[str, Any],
    credibility: float,
    cred_meta: Mapping[str, Any],
    temperament: Any,
    can_counter: bool,
    ng: _NegotiationTuning,
) -> OfferEvalResult:
    ask_d = dialogue.get("ask") if isinstance(dialogue.get("ask"), Mapping) else {}

    ask_tags = ask_d.get("need_tags")
    if isinstance(ask_tags, list):
        ask_set = {str(x).upper() for x in ask_tags if str(x).strip()}
    else:
        ask_set = set()

    off_tags = offer.get("need_tags")
    if isinstance(off_tags, list):
        offer_set = {str(x).upper() for x in off_tags if str(x).strip()}
    else:
        # If user didn't specify, interpret as agreeing to the ask.
        offer_set = set(ask_set)

    evidence = {
        "ask_need_tags": sorted(list(ask_set)),
        "offer_need_tags": sorted(list(offer_set)),
        "credibility": float(credibility),
    }

    if not ask_set:
        # No concrete ask => accept if credibility is okay.
        if float(clamp01(credibility)) >= float(ng.cred_accept_min) or not can_counter:
            return OfferEvalResult(
                decision="ACCEPT",
                counter_target=None,
                is_insult=False,
                reasons=[{"code": "NEGOTIATION_ACCEPT", "evidence": evidence}],
                meta={"axis": axis, "promise_type": "HELP", "cred_meta": dict(cred_meta)},
                next_dialogue=None,
            )
        nd = _counter_dialogue(
            dialogue=dialogue,
            new_ask={"need_tags": sorted(list(ask_set))},
            new_floor={"need_tags": sorted(list(ask_set))},
            new_insult={"need_tags": sorted(list(ask_set))},
            offer_norm={"need_tags": sorted(list(offer_set))},
        )
        return OfferEvalResult(
            decision="COUNTER",
            counter_target={"need_tags": sorted(list(ask_set))},
            is_insult=False,
            reasons=[{"code": "NEGOTIATION_COUNTER_LOW_CREDIBILITY", "evidence": evidence}],
            meta={"axis": axis, "promise_type": "HELP", "cred_meta": dict(cred_meta)},
            next_dialogue=nd,
        )

    # Offer covers ask.
    if ask_set.issubset(offer_set):
        if float(clamp01(credibility)) >= float(ng.cred_accept_min) or not can_counter:
            return OfferEvalResult(
                decision="ACCEPT",
                counter_target=None,
                is_insult=False,
                reasons=[{"code": "NEGOTIATION_ACCEPT", "evidence": evidence}],
                meta={"axis": axis, "promise_type": "HELP", "cred_meta": dict(cred_meta)},
                next_dialogue=None,
            )
        nd = _counter_dialogue(
            dialogue=dialogue,
            new_ask={"need_tags": sorted(list(ask_set))},
            new_floor={"need_tags": sorted(list(ask_set))},
            new_insult={"need_tags": sorted(list(ask_set))},
            offer_norm={"need_tags": sorted(list(offer_set))},
        )
        return OfferEvalResult(
            decision="COUNTER",
            counter_target={"need_tags": sorted(list(ask_set))},
            is_insult=False,
            reasons=[{"code": "NEGOTIATION_COUNTER_LOW_CREDIBILITY", "evidence": evidence}],
            meta={"axis": axis, "promise_type": "HELP", "cred_meta": dict(cred_meta)},
            next_dialogue=nd,
        )

    # Partial overlap -> counter.
    overlap = ask_set.intersection(offer_set)
    if overlap and can_counter:
        nd = _counter_dialogue(
            dialogue=dialogue,
            new_ask={"need_tags": sorted(list(ask_set))},
            new_floor={"need_tags": sorted(list(ask_set))},
            new_insult={"need_tags": sorted(list(ask_set))},
            offer_norm={"need_tags": sorted(list(offer_set))},
        )
        return OfferEvalResult(
            decision="COUNTER",
            counter_target={"need_tags": sorted(list(ask_set))},
            is_insult=False,
            reasons=[{"code": "NEGOTIATION_COUNTER", "evidence": {**evidence, "overlap": sorted(list(overlap))}}],
            meta={"axis": axis, "promise_type": "HELP", "cred_meta": dict(cred_meta)},
            next_dialogue=nd,
        )

    # Reject.
    return OfferEvalResult(
        decision="REJECT",
        counter_target=None,
        is_insult=False,
        reasons=[{"code": "NEGOTIATION_REJECT", "evidence": evidence}],
        meta={"axis": axis, "promise_type": "HELP", "cred_meta": dict(cred_meta)},
        next_dialogue=None,
    )
