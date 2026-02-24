# NBA 시뮬레이션 상용화 관점 프론트엔드-백엔드 갭 점검 (재검증)

## 점검 범위
- 백엔드: `server.py`의 실제 요청 모델/엔드포인트/핸들러
- 프론트엔드: `static/NBA.html`, `static/app.js`의 전용 UI·placeholder·호출 바인딩
- 기준: "실패를 재현 가능한 불일치" 또는 "상용 운영 리스크가 명확한 항목"만 유지

---

## 재검증 결론 (요약)
- 이전 리포트의 핵심 3건 중, **실제 문제로 재확정 3건 / 철회 0건**.
- 다만 표현을 더 엄밀히 조정:
  - "미구현" 대신 "백엔드 대비 전용 UX 미노출"로 정의 (API 스튜디오 우회는 가능).

---

## 1) 즉시 수정이 필요한 기능 결함 (재검증 확정)

### A. Agency 입력 예시(JSON)와 백엔드 요청 스키마 불일치
**프론트 예시**
- `agencyRespondPayload`: `{"event_id":"...","team_id":"LAL","action":"ACCEPT"}`
- `agencyApplyPayload`: `{"team_id":"LAL","actions":[...]}`

**백엔드 요구 스키마**
- `/api/agency/events/respond` → `user_team_id`, `event_id`, `response_type`, `response_payload`
- `/api/agency/actions/apply` → `user_team_id`, `player_id`, `action_type`, `action_payload`

**판정**
- UI placeholder를 그대로 사용하면 422 가능성이 매우 높음(필수 필드명 불일치).
- 상용 기준에서 "초기 사용 실패"로 직결되는 진짜 결함.

---

### B. Scouting assign 입력 예시(JSON)와 백엔드 요청 스키마 불일치
**프론트 예시**
- `{"team_id":"LAL","scout_id":"...","target_type":"COLLEGE_PLAYER","target_id":"..."}`

**백엔드 요구 스키마**
- `/api/scouting/assign` → `team_id`, `scout_id`, `player_id`, `target_kind="COLLEGE"`

**판정**
- placeholder 기준으로 요청 시 `player_id` 누락 + 키 불일치로 실패 가능.
- 사용자/QA가 화면 예시를 따라가면 막히는 실결함.

---

### C. 상용 운영 기준에서 인증/권한 제어 부재
**확인 사항**
- CORS `allow_origins=["*"]`, `allow_credentials=True`.
- 서버 레벨 인증/인가(토큰 검증, 사용자 식별 미들웨어/Depends) 확인되지 않음.
- 쓰기 API(리그 진행, 계약, 트레이드, 저장/로드 등)가 다수 노출.

**판정**
- 개발 단계에서는 허용될 수 있으나, 상용 배포 조건으로는 명확한 운영 결함.

---

## 2) 백엔드 대비 프론트 전용 UX 미노출 (재검증 확정)

> 주의: "기능 자체 없음"이 아니라, **전용 화면 플로우가 없어 일반 사용자가 접근하기 어려운 상태**를 의미.

### 확인된 미노출 영역
- Practice API: 조회/설정 엔드포인트 존재하나 전용 화면 컨트롤 부재
- News 생성 API: `/api/news/week`, `/api/news/playoffs` 전용 버튼/폼 부재
- Season report API: `/api/season-report` 전용 화면 부재
- Draft insight 일부: recompute/experts/bigboard 전용 화면 부재
- Trade negotiation API(start/commit): 전용 flow 부재 (현재 트레이드 섹션은 evaluate/submit 계열 위주)

**판정**
- API 스튜디오로 호출은 가능하지만, 상용 사용자 UX 기준에서는 미흡.

---

## 3) 출시 전 우선순위
1. **P0**: Agency/Scouting placeholder와 실제 요청 스키마 즉시 정합화.
2. **P0**: 인증/권한 도입(최소 토큰 기반) + 상태변경 API 보호.
3. **P1**: Practice/News/Season Report/Draft insight/Trade negotiation 전용 UX 추가.
4. **P1**: API 스튜디오를 운영자 모드로 분리(일반 사용자 동선 분리).

---

## 4) 최종 코멘트
- 재검증 후에도 "억지 이슈"는 추가하지 않았고, 실제 실패 가능성이 높은 항목만 유지했습니다.
- 특히 A/B는 필드명 불일치라 재현 가능성이 높아, 상용 전 반드시 수정 권장.
