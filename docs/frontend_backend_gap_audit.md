# Frontend vs Backend Gap Audit (Commercial-readiness)

## Scope
- Compared FastAPI backend endpoints and request contracts in `server.py` with the current production UI (`static/NBA.html`, `static/app.js`).
- Focused only on verifiable, user-impacting functional gaps (not style/preferences).

## Verified critical gaps

### 1) Offseason pipeline buttons include endpoints that will fail from current UI payloads
- The offseason screen auto-generates one-click buttons for `/api/offseason/draft/workouts` and `/api/offseason/draft/interviews`.
- Current button handler sends `{}` for those endpoints (no `team_id`).
- Backend requires `team_id` for both requests via `DraftWorkoutsRequest.team_id` and `DraftInterviewsRequest.team_id`.
- Result: the advertised pipeline is not actually executable from the “오프시즌 파이프라인” button grid alone (422 validation failure).

### 2) Contract action panel is inconsistent with backend-required negotiation flow
- Backend enforces that `sign-free-agent` and `re-sign-or-extend` must use an accepted negotiation session (`session_id` required in request models).
- But the transactions panel’s generic payload placeholder only shows `{"team_id":"LAL","player_id":"..."}`, which omits the required `session_id`.
- Result: likely runtime validation failures for normal users unless they already infer hidden contract workflow details.

### 3) Major gameplay domains exist in backend but have no dedicated first-class UI flows
The backend already exposes operational APIs for these game systems, but the main product UX has no dedicated screens/workflows:
- Training (team/player)
- Practice plans/sessions
- Scouting assignment/reporting
- Postseason setup/advance flow
- Agency event feed/respond/apply
- Trade negotiation session flow
- Single game simulation endpoint

Notes:
- Yes, API Studio can technically call GET/POST endpoints, but that is a developer tool, not a commercial-user gameplay UX.
- For commercial release, these systems need guided UI states (selection, validation, sequencing, user feedback), not raw JSON forms.

## Coverage snapshot
- Backend GET/POST API routes discovered: 90
- Frontend hard-referenced API routes discovered: 47
- Routes not explicitly wired in the current UI code: 45

(Computed by static extraction script; see terminal command history.)
