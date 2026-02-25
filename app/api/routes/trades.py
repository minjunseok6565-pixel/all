from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

import state
from trades import agreements, negotiation_store
from trades.apply import apply_deal_to_db
from trades.errors import TradeError
from trades.models import canonicalize_deal, parse_deal, serialize_deal
from trades.validator import validate_deal
from app.schemas.trades import (
    TradeEvaluateRequest,
    TradeNegotiationCommitRequest,
    TradeNegotiationStartRequest,
    TradeSubmitCommittedRequest,
    TradeSubmitRequest,
)
from app.services.cache_facade import _try_ui_cache_refresh_players
from app.services.contract_facade import _validate_repo_integrity
from app.services.trade_facade import _trade_error_response

router = APIRouter()












@router.post("/api/trade/submit")
async def api_trade_submit(req: TradeSubmitRequest):
    try:
        in_game_date = state.get_current_date_as_date()
        db_path = state.get_db_path()
        agreements.gc_expired_agreements(current_date=in_game_date)
        deal = canonicalize_deal(parse_deal(req.deal))
        validate_deal(deal, current_date=in_game_date)
        transaction = apply_deal_to_db(
            db_path=db_path,
            deal=deal,
            source="menu",
            deal_id=None,
            trade_date=in_game_date,
            dry_run=False,
        )
        _validate_repo_integrity(db_path)
        moved_ids: List[str] = []
        for mv in (transaction.get("player_moves") or []):
            if isinstance(mv, dict):
                pid = mv.get("player_id")
                if pid:
                    moved_ids.append(str(pid))
        _try_ui_cache_refresh_players(moved_ids, context="trade.submit")
        return {
            "ok": True,
            "deal": serialize_deal(deal),
            "transaction": transaction,
        }
    except TradeError as exc:
        return _trade_error_response(exc)


@router.post("/api/trade/submit-committed")
async def api_trade_submit_committed(req: TradeSubmitCommittedRequest):
    try:
        in_game_date = state.get_current_date_as_date()
        db_path = state.get_db_path()
        agreements.gc_expired_agreements(current_date=in_game_date)
        deal = agreements.verify_committed_deal(req.deal_id, current_date=in_game_date)
        validate_deal(
            deal,
            current_date=in_game_date,
            allow_locked_by_deal_id=req.deal_id,
        )
        transaction = apply_deal_to_db(
            db_path=db_path,
            deal=deal,
            source="negotiation",
            deal_id=req.deal_id,
            trade_date=in_game_date,
            dry_run=False,
        )
        _validate_repo_integrity(db_path)
        agreements.mark_executed(req.deal_id)
        moved_ids: List[str] = []
        for mv in (transaction.get("player_moves") or []):
            if isinstance(mv, dict):
                pid = mv.get("player_id")
                if pid:
                    moved_ids.append(str(pid))
        _try_ui_cache_refresh_players(moved_ids, context="trade.submit_committed")
        return {"ok": True, "deal_id": req.deal_id, "transaction": transaction}
    except TradeError as exc:
        return _trade_error_response(exc)


@router.post("/api/trade/negotiation/start")
async def api_trade_negotiation_start(req: TradeNegotiationStartRequest):
    try:
        in_game_date = state.get_current_date_as_date()
        session = negotiation_store.create_session(
            user_team_id=req.user_team_id, other_team_id=req.other_team_id
        )
        # Ensure sessions naturally expire even if the user ignores them,
        # so they don't permanently consume the active-session cap.
        valid_until = (in_game_date + timedelta(days=2)).isoformat()
        negotiation_store.set_valid_until(session["session_id"], valid_until)
        # Keep response consistent with stored session
        session["valid_until"] = valid_until
        return {"ok": True, "session": session}
    except TradeError as exc:
        return _trade_error_response(exc)


@router.post("/api/trade/negotiation/commit")
async def api_trade_negotiation_commit(req: TradeNegotiationCommitRequest):
    try:
        in_game_date = state.get_current_date_as_date()
        db_path = state.get_db_path()
        session = negotiation_store.get_session(req.session_id)
        deal = canonicalize_deal(parse_deal(req.deal))
        team_ids = {session["user_team_id"].upper(), session["other_team_id"].upper()}
        if set(deal.teams) != team_ids or len(deal.teams) != 2:
            raise TradeError(
                "DEAL_INVALIDATED",
                "Deal teams must match negotiation session",
                {"session_id": req.session_id, "teams": deal.teams},
            )
        # Hot path: negotiation UI calls this endpoint repeatedly.
        # DB integrity is already guaranteed at startup and after any write APIs.
        # Avoid running full repo integrity check on every offer update.
        validate_deal(deal, current_date=in_game_date, db_path=db_path, integrity_check=False)
        
        # Always persist the latest valid offer payload
        negotiation_store.set_draft_deal(req.session_id, serialize_deal(deal))

        # Local imports to keep integration flexible.
        from trades.valuation.service import evaluate_deal_for_team as eval_service  # type: ignore
        from trades.valuation.types import (
            to_jsonable,
            DealVerdict,
            DealDecision,
            DecisionReason,
        )  # type: ignore
        from trades.generation.dealgen.dedupe import dedupe_hash  # type: ignore

        # ------------------------------------------------------------------
        # Fast path: if the user submits the exact last AI counter-offer, accept
        # immediately.
        # - Prevents the frustrating UX where the AI "rejects its own counter".
        # - Only active while the session is in COUNTER_PENDING phase.
        # ------------------------------------------------------------------
        try:
            phase = str(session.get("phase") or "").upper()
            last_counter = session.get("last_counter")
            expected_hash = last_counter.get("counter_hash") if isinstance(last_counter, dict) else None
            if phase == "COUNTER_PENDING" and isinstance(expected_hash, str) and expected_hash.strip():
                if dedupe_hash(deal) == expected_hash.strip():
                    committed = agreements.create_committed_deal(
                        deal,
                        valid_days=2,
                        current_date=in_game_date,
                        validate=False,   # already validated above
                        db_path=db_path,
                    )
                    negotiation_store.set_committed(req.session_id, committed["deal_id"])
                    negotiation_store.set_status(req.session_id, "CLOSED")
                    negotiation_store.set_phase(req.session_id, "ACCEPTED")
                    negotiation_store.set_valid_until(req.session_id, committed["expires_at"])

                    fast_decision = DealDecision(
                        verdict=DealVerdict.ACCEPT,
                        required_surplus=0.0,
                        overpay_allowed=0.0,
                        confidence=1.0,
                        reasons=(
                            DecisionReason(
                                code="COUNTER_ACCEPTED",
                                message="Accepted last counter offer",
                            ),
                        ),
                        counter=None,
                        meta={"fast_accept": True},
                    )

                    # Preserve any cached evaluation summary if present
                    fast_eval: Dict[str, Any] = {}
                    if isinstance(last_counter, dict):
                        ev = last_counter.get("ai_evaluation")
                        if isinstance(ev, dict):
                            fast_eval = dict(ev)

                    return {
                        "ok": True,
                        "accepted": True,
                        "fast_accept": True,
                        "deal_id": committed["deal_id"],
                        "expires_at": committed["expires_at"],
                        "deal": serialize_deal(deal),
                        "ai_verdict": to_jsonable(fast_decision.verdict),
                        "ai_decision": to_jsonable(fast_decision),
                        "ai_evaluation": fast_eval,
                    }
        except Exception:
            # Fast-accept should never crash the commit flow.
            pass


        # ------------------------------------------------------------------
        # AI evaluation (other team perspective)
        # NOTE:
        # - legality is already checked by validate_deal above
        # - valuation service will build DecisionContext internally (team_situation + gm profile)
        # ------------------------------------------------------------------
        other_team_id = session["other_team_id"].upper()

        decision, evaluation = eval_service(
            deal=deal,
            team_id=other_team_id,
            current_date=in_game_date,
            db_path=db_path,
            include_breakdown=False,   # keep negotiation response light
            include_package_effects=True,
            allow_counter=True,
            validate=False,            # already validated above
        )

        eval_summary = {
            "team_id": other_team_id,
            "incoming_total": float(evaluation.incoming_total),
            "outgoing_total": float(evaluation.outgoing_total),
            "net_surplus": float(evaluation.net_surplus),
            "surplus_ratio": float(evaluation.surplus_ratio),
        }

        # Record the latest offer evaluation in-session (do NOT overwrite last_counter).
        # - last_counter is reserved for the actual counter deal payload (for fast-accept).
        try:
            negotiation_store.set_last_offer(
                req.session_id,
                {
                    "offer": serialize_deal(deal),
                    "ai_verdict": to_jsonable(decision.verdict),
                    "ai_decision": to_jsonable(decision),
                    "ai_evaluation": eval_summary,
                },
            )
        except Exception:
            pass

        # Decide action
        verdict = decision.verdict

        if verdict == DealVerdict.ACCEPT:
            committed = agreements.create_committed_deal(
                deal,
                valid_days=2,
                current_date=in_game_date,
                validate=False,   # already validated above
                db_path=db_path,  # keep hash/locking based on the same db snapshot
            )
            negotiation_store.set_committed(req.session_id, committed["deal_id"])
            # Once committed, this negotiation should no longer consume "ACTIVE" capacity.
            negotiation_store.set_status(req.session_id, "CLOSED")
            # Optional but useful for UI/debugging.
            negotiation_store.set_phase(req.session_id, "ACCEPTED")
            # Keep session expiry aligned with the committed deal expiry.
            negotiation_store.set_valid_until(req.session_id, committed["expires_at"])
            return {
                "ok": True,
                "accepted": True,
                "deal_id": committed["deal_id"],
                "expires_at": committed["expires_at"],
                "deal": serialize_deal(deal),
                "ai_verdict": to_jsonable(decision.verdict),
                "ai_decision": to_jsonable(decision),
                "ai_evaluation": eval_summary,
            }

        # ------------------------------------------------------------------
        # COUNTER: build an actual counter proposal (NBA-like minimal edits)
        # ------------------------------------------------------------------
        if verdict == DealVerdict.COUNTER:
            counter_prop = None
            try:
                from trades.counter_offer.init import build_counter_offer  # type: ignore

                counter_prop = build_counter_offer(
                    offer=deal,
                    user_team_id=session["user_team_id"],
                    other_team_id=session["other_team_id"],
                    current_date=in_game_date,
                    db_path=db_path,
                    session=session,
                )
            except Exception:
                counter_prop = None

            if counter_prop is not None and getattr(counter_prop, "deal", None) is not None:
                # Attach the generated counter proposal to the decision (SSOT).
                decision = DealDecision(
                    verdict=decision.verdict,
                    required_surplus=float(decision.required_surplus),
                    overpay_allowed=float(decision.overpay_allowed),
                    confidence=float(decision.confidence),
                    reasons=decision.reasons,
                    counter=counter_prop,
                    meta=dict(decision.meta or {}),
                )

                # Persist counter offer in-session (for UI + fast-accept).
                try:
                    counter_hash = counter_prop.meta.get("counter_hash") if isinstance(counter_prop.meta, dict) else None
                    deal_payload = None
                    if isinstance(counter_prop.meta, dict):
                        deal_payload = counter_prop.meta.get("deal_serialized")
                    if not isinstance(deal_payload, dict):
                        # Defensive fallback
                        deal_payload = serialize_deal(counter_prop.deal)

                    negotiation_store.set_last_counter(
                        req.session_id,
                        {
                            "counter_hash": counter_hash,
                            "counter_deal": deal_payload,
                            "strategy": counter_prop.meta.get("strategy") if isinstance(counter_prop.meta, dict) else None,
                            "diff": counter_prop.meta.get("diff") if isinstance(counter_prop.meta, dict) else None,
                            "message": counter_prop.meta.get("message") if isinstance(counter_prop.meta, dict) else None,
                            "generated_at": in_game_date.isoformat(),
                            "base_hash": counter_prop.meta.get("base_hash") if isinstance(counter_prop.meta, dict) else None,
                            "ai_evaluation": eval_summary,
                        },
                    )
                except Exception:
                    pass

                # Push a GM-style message for the counter.
                try:
                    msg = ""
                    if isinstance(counter_prop.meta, dict):
                        msg = str(counter_prop.meta.get("message") or "")
                    msg = msg.strip() if msg else ""
                    if not msg:
                        msg = f"[{other_team_id}] COUNTER"
                    negotiation_store.append_message(req.session_id, speaker="OTHER_GM", text=msg)
                    negotiation_store.set_phase(req.session_id, "COUNTER_PENDING")
                except Exception:
                    pass

                # Response: counter details are embedded in ai_decision.counter (SSOT).

                return {
                    "ok": True,
                    "accepted": False,
                    "counter_unimplemented": False,
                    "deal": serialize_deal(deal),
                    "ai_verdict": to_jsonable(decision.verdict),
                    "ai_decision": to_jsonable(decision),
                    "ai_evaluation": eval_summary,
                }

            # If we couldn't build a legal/acceptable counter, fall back conservatively to REJECT.
            decision = DealDecision(
                verdict=DealVerdict.REJECT,
                required_surplus=float(decision.required_surplus),
                overpay_allowed=float(decision.overpay_allowed),
                confidence=float(decision.confidence),
                reasons=tuple(decision.reasons)
                + (
                    DecisionReason(
                        code="COUNTER_BUILD_FAILED",
                        message="Could not generate a legal counter offer",
                    ),
                ),
                counter=None,
                meta=dict(decision.meta or {}),
            )
            verdict = DealVerdict.REJECT


        # Build a short reason string for UI
        try:
            reason_lines = []
            for r in (decision.reasons or [])[:4]:
                if isinstance(r, dict):
                    msg = r.get("message") or r.get("code") or ""
                else:
                    msg = getattr(r, "message", None) or getattr(r, "code", None) or ""
                if msg:
                    reason_lines.append(str(msg))
            reason_text = " | ".join(reason_lines) if reason_lines else "AI rejected the offer."
        except Exception:
            reason_text = "AI rejected the offer."

        # Record rejection in session (message + phase)
        try:
            negotiation_store.append_message(
                req.session_id,
                speaker="OTHER_GM",
                text=f"[{other_team_id}] {verdict}: {reason_text}",
            )
            negotiation_store.set_phase(req.session_id, "REJECTED")
        except Exception:
            pass

        return {
            "ok": True,
            "accepted": False,
            "counter_unimplemented": False,
            "deal": serialize_deal(deal),
            "ai_verdict": to_jsonable(decision.verdict),
            "ai_decision": to_jsonable(decision),
            "ai_evaluation": eval_summary,
        }
    except TradeError as exc:
        return _trade_error_response(exc)


@router.post("/api/trade/evaluate")
async def api_trade_evaluate(req: TradeEvaluateRequest):
    """
    Debug endpoint: evaluate a proposed deal from a single team's perspective.
    Flow:
      deal = canonicalize_deal(parse_deal(req.deal))
      validate_deal(deal, current_date=in_game_date)
      trades.valuation.service.evaluate_deal_for_team(...)
      return decision + breakdown
    """
    try:
        in_game_date = state.get_current_date_as_date()
        db_path = state.get_db_path()

        deal = canonicalize_deal(parse_deal(req.deal))
        # Hot path: debug / UI-driven repeated calls.
        # Integrity is checked at startup and after any DB writes.
        validate_deal(deal, current_date=in_game_date, db_path=db_path, integrity_check=False)

        # Local import to avoid hard dependency during incremental integration.
        from trades.valuation.service import evaluate_deal_for_team as eval_service  # type: ignore
        from trades.valuation.types import to_jsonable  # type: ignore

        decision, evaluation = eval_service(
            deal=deal,
            team_id=req.team_id,
            current_date=in_game_date,
            db_path=db_path,
            include_breakdown=bool(req.include_breakdown),
            # We already validated above; avoid duplicate validate_deal in service.
            validate=False,
        )

        return {
            "ok": True,
            "team_id": str(req.team_id).upper(),
            "deal": serialize_deal(deal),
            "decision": to_jsonable(decision),
            "evaluation": to_jsonable(evaluation),
        }
    except TradeError as exc:
        return _trade_error_response(exc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Trade evaluation failed: {exc}")


# -------------------------------------------------------------------------
# 로스터 요약 API (LLM 컨텍스트용)
