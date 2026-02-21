# Agency v3 Implementation Review (Code-based)

This document summarizes a neutral code audit of the FM-style agency upgrade implementation.

## High-level verdict

- The core v3 architecture is present (self expectations, credibility, negotiation, stances, follow-up events, broken-promise reactions).
- Several important gaps/regressions remain that prevent calling it "fully complete" against the plan.

## Confirmed implemented

- DB schema/state fields for self expectations + stance.
- Config blocks for SelfExpectations/Credibility/Negotiation/Stance.
- Self-expectation monthly logic and personality-driven targeting.
- Promise credibility module using trust + memory.
- Negotiation flow with ACCEPT/COUNTER/REJECT/WALKOUT and follow-up PROMISE_NEGOTIATION events.
- Service-layer memory accumulation for broken/fulfilled outcomes and broken-promise reaction events.

## Major gaps / risks found

1. **Tick self-expectation update uses previous self expectation as input baseline**
   - update_self_expectations_monthly is fed `self_expected_mpg or minutes_expected_mpg` instead of team expected MPG source.
2. **Role frustration update does not pass self expected starts/closes overrides**
   - callsite omits expected_starts_rate/expected_closes_rate even though function supports them.
3. **ROLE event payload misses `role_focus` and gap fields from design**
   - no `role_focus`, `gap_starts`, `gap_closes` in payload.
4. **MINUTES promise acceptance path increases frustration instead of reducing it**
   - sign bug in one branch: `mfr1 += minutes_relief_promise`.
5. **User-action path baseline expectation bootstrapping helper not integrated**
   - state can be created with minutes_expected_mpg=0 and no bootstrap self expectations.

## SSOT assessment

- SSOT direction is mostly respected:
  - persistent mutable state is in player_agency_state
  - promises stored only on ACCEPT
  - events append-only
  - credibility derived (not persisted)
- The biggest SSOT-adjacent concern is incomplete baseline injection for user actions, which can cause unstable behavior when state is missing.

