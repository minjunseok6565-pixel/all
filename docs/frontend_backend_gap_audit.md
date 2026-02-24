# 백엔드 vs 프론트엔드 구현 갭 점검 리포트

## 점검 범위
- 백엔드 API 정의: `server.py`
- 현재 프론트 UI/행동: `static/NBA.html`, `static/app.js`
- API 문서 동기화 상태: `docs/api_endpoints_inventory.md`

## 결론 요약
아래 항목만 **실제 상용 출시 관점에서 문제로 확인**되었습니다.

1. **투웨이 계약 체결(협상/결정/커밋) 전용 UI 플로우 미구현**
2. **운영 문서(`api_endpoints_inventory`)가 실제 서버 API와 불일치 (누락 존재)**

그 외 대부분의 핵심 루프(시즌 진행, 오프시즌 파이프라인, 일반 계약 협상, 트레이드, 스카우팅, 대학/드래프트 조회)는 프론트에서 호출 경로가 존재하거나 API 스튜디오로 우회 실행 가능해, 이번 점검에서는 “실결함”으로 분류하지 않았습니다.

---

## 1) [High] 투웨이 계약 플로우 프론트 미구현

### 백엔드 구현 상태
서버에는 투웨이 전용 협상 API 3종이 구현되어 있습니다.
- `POST /api/contracts/two-way/negotiation/start`
- `POST /api/contracts/two-way/negotiation/decision`
- `POST /api/contracts/two-way/negotiation/commit`

### 프론트 구현 상태
- 계약/트랜잭션 화면의 액션 버튼은 다음 3개만 제공됩니다.
  - `POST /api/contracts/sign-free-agent`
  - `POST /api/contracts/re-sign-or-extend`
  - `POST /api/contracts/release-to-fa`
- 투웨이 전용 협상/커밋을 수행하는 버튼/폼/가이드가 없습니다.

### 왜 실문제인가
- 현재 UI 기준으로 유저가 투웨이 계약을 정상 절차(협상→결정→커밋)로 수행하기 어렵습니다.
- API 스튜디오 우회는 가능하지만, 상용 UX 기준에서는 핵심 트랜잭션 기능이 일반 사용자 동선에 없는 상태입니다.
- 특히 로스터 화면에 투웨이 슬롯 현황은 노출되는데, 이를 실제로 채우는 대표 동작이 전용 UI에서 닫혀 있어 기능 단절이 발생합니다.

### 권고
- `transactionsView`에 투웨이 협상 전용 Wizard 추가
  - start(팀/선수/유효일)
  - decision(accept/reject)
  - commit(session_id)
- 결과를 기존 `negotiationResult`와 분리(예: `twoWayNegotiationResult`)해 에러 분석 가능성 강화

---

## 2) [Medium] API 인벤토리 문서와 서버 코드 불일치

### 확인 내용
- `docs/api_endpoints_inventory.md`는 POST 총계를 56개로 명시하지만,
- 서버 코드에는 해당 문서에 없는 투웨이 협상 API 3종이 추가로 존재합니다.

### 왜 실문제인가
- 상용 운영 시 QA 체크리스트/모니터링/연동 문서가 소스와 어긋나면 누락 테스트가 발생합니다.
- 특히 계약 관련 엔드포인트 누락은 규정 위반/정합성 이슈를 조기에 검출하지 못하게 만듭니다.

### 권고
- 릴리즈 전 문서 자동 생성 또는 CI 검증 추가
  - 예: `server.py` 라우트 스캔 결과와 문서 diff 검사

---

## 실결함으로 보지 않은 항목 (검증 후 제외)
- `POST` 보호를 위한 `X-Admin-Token` 미들웨어는 의도된 상용 보호장치이며, 프론트에도 토큰 입력/저장 동선이 존재함.
- 오프시즌 단계별 API는 프론트 버튼 및 보조 툴(팀 옵션 Pending/Decide, 드래프트 툴)로 실행 경로가 존재함.
- 일반 계약 협상(`start/offer/accept-counter/commit/cancel`)과 트레이드 협상(`start/commit`)은 전용 UI가 존재함.

