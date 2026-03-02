# 트레이드 요청/트레이드 블록 우선순위 개편 설계안 (코드 작업 전)

## 1) 목적

요구사항을 아래 3가지로 분리해 반영한다.

1. 가치평가에서 `team_frustration`, `role_frustration` 직접 반영 제거
2. `trade_request_level`은 "레벨별 선형 스케일"이 아니라 **2단계(공개 요청)에서만 할인** 적용
3. 탐색 우선순위에서 트레이드 블록 가산점 강화 + `trade_request_level=2` 연계 가산점 추가

---

## 2) 현재 구조 점검 결과 (이미 구현된 것 / 미흡한 것)

### A. 이미 구현된 것

- `trade_request_level`은 0/1/2 스키마로 관리됨.
- 가치평가 엔진에서 agency state를 읽어 보정하는 경로가 이미 존재함.
- 트레이드 블록은 공개오퍼/유출 기반 자동 등록뿐 아니라,
  API(`/api/trade/block/list`)로 사용자 수동 등록도 이미 가능함.

### B. 요구사항 대비 미흡한 것

- 가치평가에서 `team_frustration`, `role_frustration`이 현재 직접 가중합에 포함됨.
- `trade_request_level`도 현재 `level/3` 형태의 연속 스케일로 반영됨.
- 탐색 우선순위는 "팀이 active listing을 보유했는지" 수준의 배수만 있고,
  - 트레이드 블록 선수 수/우선도/사유 기반 세분 가점,
  - `trade_request_level=2` 연계 가점
  이 없음.

---

## 3) 목표 동작(타깃 정책)

## 3.1 가치평가 정책

- 입력 요소: `trade_request_level`만 사용
- 할인 조건: `trade_request_level == 2` 일 때만 적용
- 의미: 선수 자체 능력 하락이 아니라 "판매팀 협상력 약화"를 반영한 거래조건 할인

권장식(예시):

- `discount_ratio = agency_public_request_discount if tr_level == 2 else 0.0`
- Incoming(내가 받는 선수): `delta -= value * discount_ratio`
- Outgoing(내가 보내는 선수): `delta += value * discount_ratio`

> 위 식은 기존 `AGENCY_DISTRESS_VALUE_ADJUST` 스텝의 부호 관례를 유지하면서,
> 공개 요청시에만 효과가 생기도록 단순화한다.

## 3.2 탐색 우선순위 정책

팀 단위 액터 가중치에서 아래 항목을 추가한다.

1. **트레이드 블록 기본 가산점(강화)**
   - active listing 팀에 대한 multiplier를 기존보다 크게(예: 1.20 → 1.45)
   - 또는 multiplier 대신 additive bonus를 병행

2. **공개 요청(레벨2) + 트레이드 블록 동시 가산점(소폭)**
   - listing 중 agency state `trade_request_level==2` 선수 수가 있으면 추가 multiplier
   - 예: `listing_with_public_req_mult = 1.10`

3. **공개 요청(레벨2) 선수 존재 시, listing이 없어도 최소 가산점 부여**
   - 예: `public_req_presence_add = +0.10` (activity_score에 더함)
   - 목적: 공개 트레이드 요청팀이 시장 탐색에서 뒤로 밀리지 않도록 보장

---

## 4) 모듈별 설계 변경안

## 4.1 가치평가 (`trades/valuation/package_effects.py`)

### 변경 포인트

- `_agency_distress_adjustment()`를 "distress" 개념에서 "public trade request leverage" 개념으로 단순화
- 제거:
  - `agency_team_frustration_weight`
  - `agency_role_frustration_weight`
  - `trade_request_level / 3.0` 스케일
- 유지/신규:
  - `agency_public_request_discount` (예: 0.12)
  - 조건: `int(trade_request_level) >= 2` 일 때만 할인

### 메타/텔레메트리

- step code를 유지하거나 신규(`AGENCY_PUBLIC_REQUEST_DISCOUNT`)로 분리
- meta에 다음 포함:
  - `incoming_public_request_count`
  - `outgoing_public_request_count`
  - `public_request_discount`

## 4.2 데이터 컨텍스트 (`trades/valuation/data_context.py`)

### 선택지

- A안(권장): 현재처럼 `trade_request_level/team_frustration/role_frustration` 로드 유지
  - 이유: 다른 화면/시스템에서 재사용 가능, 하위호환 용이
  - 가치평가 함수만 `trade_request_level`만 읽도록 제한
- B안: SQL select에서 frustration 필드 제거
  - 장점: 의도 명확
  - 단점: 확장성/호환성 저하 위험

## 4.3 오케스트레이션 우선순위 (`trades/orchestration/actor_selection.py`)

### 추가 입력

- `trade_market` listing 상세 + agency snapshot(팀별 public request count)
- helper 함수 신설(예):
  - `_team_listing_score(team_id, trade_market, today)`
  - `_team_public_request_score(team_id, tick_ctx)`

### 가중치 적용 순서(권장)

1. 기존 `activity_score` 계산
2. 압력 tier/threads/listing multiplier 적용
3. **listing 강가산 적용**
4. **public request 연계 가산 적용**

권장 공식(예시):

- `w = base_activity`
- `w *= listing_mult_strong` if has_active_listing
- `w *= listing_public_req_mult` if active_listing_with_tr2_count > 0
- `w += public_req_presence_add` if team_public_req_count > 0

## 4.4 설정값 (`trades/orchestration/types.py`, valuation config)

### 신규/변경 권장 파라미터

- valuation
  - `agency_public_request_discount: float = 0.12`
- orchestration
  - `trade_block_actor_weight_multiplier: float = 1.45` (기존값 상향)
  - `trade_block_public_request_multiplier: float = 1.10`
  - `public_trade_request_actor_add: float = 0.10`

> 숫자는 예시이며 시뮬 로그 기반 튜닝 필요.

---

## 5) "트레이드 블록 수동 등록" 요구사항 점검

요구사항 중
"AI/유저가 별다른 제안 없이도 직접 트레이드 블록에 올릴 수 있게"는
**이미 부분 구현됨**.

- 유저: `/api/trade/block/list`로 수동 등록 가능
- AI: tick 루프에서 생성 제안 샘플을 바탕으로 자동 listing 수행

즉, 유저 수동 등록은 신규 개발 불필요.
AI의 "완전 수동"(제안 생성과 무관한 독립 listing 판단)은 개선 여지가 있음.

---

## 6) AI 독립 트레이드 블록 등록(선택 확장)

현재 AI listing은 "생성된 제안 props[0]의 outgoing"에 종속.
이를 분리하려면:

1. `tick_loop` 초반 또는 actor별 처리 전에
2. 팀 상황 + agency state + 로스터 과잉 포지션을 기반으로
3. 독립적으로 listing 후보를 선정

예시 기준:

- `trade_request_level==2` 우선
- 팀 needs와 미스핏이 큰 선수
- 출전시간/역할 불만 고점 선수

이 확장은 요구사항의 본질(공개 요청 시장 노출 강화)에 가장 직접적이다.

---

## 7) 구현 순서 제안

1. 가치평가 단순화(리스크 적음, 효과 명확)
2. actor_selection 가중치 확장
3. (선택) AI 독립 listing 로직 추가
4. 텔레메트리/로그 키 추가 후 시뮬레이션 튜닝

---

## 8) 테스트/검증 시나리오 설계

## 8.1 가치평가

- Case A: `tr=1, tf/rf 높음` → 보정 0이어야 함
- Case B: `tr=2, tf/rf 낮음` → 할인 보정 발생해야 함
- Case C: incoming/outgoing 부호 방향 확인

## 8.2 우선순위

- 동일 팀상황에서
  - listing 없음 < listing 있음
  - listing+tr2 > listing만 있음
  - listing 없어도 tr2 존재팀이 baseline보다 높음

## 8.3 회귀

- 공개오퍼/유출 자동 listing 기존 동작 유지
- `/api/trade/block/list` 수동 등록 API 정상

---

## 9) 요약

- "불만축 직접 반영 제거 + 공개요청(2단계) 단일 할인"은 현재 구조에서 간단히 실현 가능.
- 트레이드 블록 우선순위 강화와 공개요청 연계 가산은 actor_selection 레벨에서 설계적으로 자연스럽게 확장 가능.
- 유저 수동 트레이드 블록 등록은 이미 구현되어 있어 신규 개발보다
  "AI 독립 수동 listing" 추가 여부가 핵심 선택지다.
