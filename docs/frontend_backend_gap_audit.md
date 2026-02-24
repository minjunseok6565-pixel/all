# Frontend ↔ Backend 기능 격차 점검 (상업 출시 관점)

## 범위
- 프론트엔드: `static/NBA.html`, `static/app.js`
- 백엔드: `server.py`
- 기준: 사용자 관점에서 "실제로 클릭해서 진행 가능한 게임 루프"와 "백엔드가 제공하는 실제 기능"의 일치 여부

## 핵심 결론
- 백엔드 API는 매우 넓게 구현되어 있으나(`server.py` 기준 `/api/*` 89개), 프론트의 실제 전용 UX는 런처/기본 조회/일부 계약·트레이드/오프시즌 버튼 수준에 머물러 있습니다.
- 특히 **계약 협상(negotiation) 선행이 필수인 경로를 프론트가 우회하도록 UI를 제공**하고 있어, 클릭 시 실패가 발생하는 구조적 결함이 있습니다.
- 오프시즌 버튼들 중 일부는 **필수 payload 없이 호출되도록 하드코딩**되어 있어 정상 진행이 불가능합니다.

---

## 1) 즉시 수정 필요 (출시 차단급)

### 1-1. Team Option 결정 플로우가 프론트에서 실패하도록 고정됨
- 프론트는 `/api/offseason/options/team/decide` 호출 시 `decisions: []`를 고정 전송합니다.
- 백엔드는 `decisions`가 비어 있으면 400 에러를 반환하도록 강제하고 있습니다.
- 결과: 버튼을 누르면 실패하는 UX가 기본 동작으로 노출됩니다.

**영향**
- 오프시즌 핵심 단계(옵션 결정)가 프론트에서 완료 불가.
- 사용자 신뢰 하락 + 진행 막힘(blocking).

### 1-2. 드래프트 자동선택 버튼이 기본적으로 실패할 가능성이 큼
- 프론트는 `/api/offseason/draft/selections/auto`를 기본 `{}`로 호출합니다.
- 백엔드는 `allow_autopick_user_team=false`이고 `stop_on_user_controlled_team_ids`가 없으면 에러를 던지도록 설계되어 있습니다.
- 결과: 기본 버튼 동작이 실패 경로.

### 1-3. 단일 픽 선택 버튼은 필수 값 입력 UX가 없음
- 프론트는 `/api/offseason/draft/selections/pick`에 대해 필수 값(`prospect_temp_id`) 입력 컴포넌트를 제공하지 않고 빈 body 호출 경로만 존재합니다.
- 백엔드는 `prospect_temp_id`를 필수로 요구합니다.
- 결과: API 스튜디오를 수동 사용하지 않으면 정상적인 유저 드래프트 진행이 불가.

### 1-4. 계약 체결 버튼이 협상 선행 조건을 충족할 수 없음
- 프론트 "FA 계약", "재계약/연장" 버튼은 곧바로 `/api/contracts/sign-free-agent`, `/api/contracts/re-sign-or-extend`를 호출하도록 노출되어 있습니다.
- 백엔드 모델상 두 API는 `session_id`(협상 세션) 기반이며, 협상 시작/오퍼/카운터 수락/커밋의 별도 플로우가 존재합니다.
- 결과: 프론트 UX만으로는 정상 계약 체결이 거의 불가능(실패율 높음).

---

## 2) 기능 미구현/미흡 영역 (게임 완성도 관점)

### 2-1. 시즌 핵심 서브시스템 전용 UI 부재
백엔드에는 존재하지만 프론트 전용 화면/상호작용이 없는 영역:
- Training/Practice 전체
- Scouting 전체
- Agency Interaction(선수 에이전시 이벤트 응답)
- Postseason 진행(플레이-인/PO 라운드 진행)
- Draft 세부 조회(질문 리스트, 전문가 보드, 번들 확인 등)

현재 프론트는 메뉴 기준으로 런처/대시보드/시즌 운영/로스터/트랜잭션/오프시즌/어시스턴트/API 스튜디오만 제공하며, 많은 고급 시스템은 API 스튜디오에서 수동 JSON으로만 접근 가능합니다. 상업 제품 UX로는 부족합니다.

### 2-2. 거래/계약 UX가 “JSON 수동 입력” 중심
- 트레이드/계약 영역이 textarea JSON 직접 입력에 의존합니다.
- 백엔드는 검증 규칙이 촘촘해 실패 케이스가 많은데, 프론트는 스키마 가이드/폼 검증/필수 필드 유도/자동 완성이 거의 없습니다.
- 결과: 초중급 사용자에게는 사실상 사용 불가 수준의 진입장벽.

### 2-3. 에러 메시지 UX 미흡
- 프론트는 에러를 JSON 덤프 위주로 보여주고, 필드 단위 피드백(어떤 값이 왜 틀렸는지)에 대한 UI 가이드가 없습니다.
- 상업 출시 시 CS/이탈률에 직접적인 악영향.

---

## 3) 상업 출시 리스크 (아키텍처/보안/운영)

### 3-1. 인증/권한 모델 부재
- 현재 주요 API 호출 경로는 별도의 사용자 인증/권한 분리 없이 열려 있는 구조입니다(서버 전역 CORS도 wildcard).
- 싱글 유저 로컬 툴이라면 괜찮지만, 상업 서비스(멀티 유저/과금)에는 필수 보안 요건 미충족.

### 3-2. API Key 취급 UX
- 프론트에서 LLM API Key를 localStorage에 저장합니다.
- 상업 환경에서는 키 탈취/브라우저 취약점 리스크를 고려해 서버 측 키 프록시/세션 토큰 방식으로 전환이 일반적입니다.

---

## 4) 우선순위 개선 백로그 (추천)

1. **오프시즌 실패 버튼 3종 수정**
   - options/decide: pending 목록 기반 체크박스 결정 UI 생성
   - draft/selections/auto: 유저팀 ID 자동 주입 또는 allow_autopick 토글 제공
   - draft/selections/pick: 온더클락 후보 리스트 + 필수 값 입력 UI

2. **계약 협상 전용 Wizard 추가 (필수)**
   - start → offer → accept-counter/commit → sign/re-sign 연결
   - 현재의 “바로 체결” 버튼은 숨기거나 고급모드로 분리

3. **트레이드 빌더 UI 도입**
   - 팀 선택, 자산 선택, 실시간 검증, 에러 번역
   - deal JSON 직접 입력은 디버그 모드로 격리

4. **Postseason/Scouting/Training 전용 화면 확장**
   - 이미 백엔드가 있는 만큼 UI만 보강하면 체감 완성도 급상승

5. **상용화 대응 보안 레이어**
   - 인증(계정), 세션/권한, rate limit, 감사 로그
   - CORS/키 관리 재설계

---

## 5) 참고 근거 (코드 위치)
- 프론트 메뉴/기능 구성: `static/NBA.html`
- 프론트 오프시즌 버튼 payload 하드코딩: `static/app.js`
- 계약/트레이드 버튼 endpoint 노출: `static/NBA.html`
- 백엔드 요청 모델(필수 필드): `server.py` 상단 Pydantic 모델 섹션
- options/decide 빈 배열 금지 로직: `server.py`의 `/api/offseason/options/team/decide`
- draft auto/pick 요구사항: `server.py`의 `/api/offseason/draft/selections/auto`, `/api/offseason/draft/selections/pick`
- CORS wildcard 설정: `server.py` 초기화 구간
