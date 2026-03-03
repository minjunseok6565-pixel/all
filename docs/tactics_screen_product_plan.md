# 전술 설정 화면 고도화 기획안 (상업용 품질 목표)

## 1) 문서 목적/원칙
- 목적: 현재 `전술 설정` 화면을 상업용 게임 수준의 정보 구조/비주얼/인터랙션으로 고도화한다.
- 원칙 A: **현재 코드/서버에서 조회 가능한 데이터만 우선 사용**한다.
- 원칙 B: 화면에 필요한데 현재 없는 데이터는 **신규 API를 설계한 뒤에만** 붙인다.
- 원칙 C: 내부 식별자(`Engine_Primary` 등)를 그대로 노출하지 않고 표시 라벨을 분리한다.

---

## 2) 현재 구현 기준(AS-IS) 요약
- 전술 화면 진입 시 프론트는 `/api/team-detail/{team_id}`를 호출해 `detail.roster`를 가져오고, 이를 기반으로 임시 `tacticsDraft`를 생성한다.
- 공격/수비 스킴, 공격 역할, 수비 역할은 현재 프론트 상수(`TACTICS_*`)에 하드코딩되어 있다.
- 전술 저장/불러오기 전용 API가 없어, 새로고침/세션 단위에서 전술 편집 상태가 안정적으로 유지되지 않는다.

---

## 3) 목표 UX (TO-BE) — 화면 구조

### 3-1. 레이아웃(데스크톱 기준)
- 상단 글로벌 헤더(고정)
  - 팀명, 시즌일자, 저장 상태, CTA(저장/되돌리기/자동밸런스).
- 본문 3열 레이아웃
  - 좌측(전술 플랜/스킴 선택): 플랜 프리셋 카드, 공격/수비 스킴 선택, 스킴 설명.
  - 중앙(라인업 에디터): 선발 5명 + 로테이션 5명 편집, 역할·분배 편집.
  - 우측(영향 분석): 분배 검증(240분), 역할 충돌 경고, 전술 영향 프리뷰.
- 하단 확장 패널
  - 전체 로스터 검색/필터(포지션, OVR, 컨디션).

### 3-2. 사용자 플로우
1. 전술 플랜 선택(정규/클러치/스몰볼 등) →
2. 스킴 변경(공/수) →
3. 선발/로테이션 선수 및 역할 조정 →
4. 출전시간 총합/경고 확인 →
5. 저장 및 다음 경기 반영.

---

## 4) 화면 컴포넌트 상세 스펙

### 4-1. 상단 상태 바
- 필수 표시
  - 팀명
  - 현재 게임 날짜
  - 전술 플랜명
  - 저장 상태(`저장됨`, `수정됨`)
- 액션
  - `저장`
  - `되돌리기`
  - `자동 밸런스(분배 240분 자동 보정)`

### 4-2. 스킴 선택 카드
- 공격/수비 스킴을 카드/드롭다운 혼합 UI로 제공.
- 각 스킴 항목은 2줄 표기
  - 1행: 사용자 라벨(예: `헤비 PnR`)
  - 2행: 짧은 설명(예: `볼핸들러 중심 2:2 빈도 증가`)
- 주의: 내부 키(예: `Spread_HeavyPnR`)는 디버그 모드가 아니면 비노출.

### 4-3. 선발/로테이션 에디터
- 행 구성
  - 선수(썸네일/이름/포지션)
  - 공격 역할
  - 수비 역할
  - 출전시간(숫자 + 슬라이더)
- 실시간 규칙
  - 동일 선수 중복 배치 금지
  - 수비 역할 중복 정책은 기획 옵션(현재는 중복 금지)
  - 총합 240분 기준으로 초과/부족 시 즉시 경고

### 4-4. 우측 영향 분석 패널
- 기본 섹션
  - `분배 검증`: 스타터/벤치 합계, 총합, 포지션 편중
  - `역할 충돌`: 볼핸들러 과다, 림프로텍터 부재 등
  - `전술 영향 프리뷰`: 변경 전/후 지표 비교
- 전/후 비교는 숫자 + 방향 아이콘(▲/▼)으로 표현.

### 4-5. 로스터 패널
- 검색 + 필터(포지션, OVR 구간, 컨디션)
- 드래그앤드롭 또는 선택-적용 방식으로 선발/로테이션 교체

---

## 5) 데이터 소스 매핑 ("어디서 가져오는지")

## 5-1. **기존 API로 즉시 충당 가능한 정보**

| 화면 정보 | 소스 API | 사용 필드 |
|---|---|---|
| 팀 기본 정보 | `GET /api/team-detail/{team_id}` | `summary.team_id`, `summary.wins/losses`, `summary.rank`, `summary.cap_space` |
| 전술 편집 대상 로스터 | `GET /api/team-detail/{team_id}` | `roster[].player_id/name/pos/ovr/short_term_fatigue/long_term_fatigue/sharpness` |
| 날짜/리그 상태 | `GET /api/state/summary` | `workflow_state.league.current_date`(또는 league 컨텍스트) |
| 다음 경기 컨텍스트 | `GET /api/team-schedule/{team_id}` | `games[]` 중 미완료 경기의 `date/opponent_team_id/is_home/tipoff_time` |
| 스킴 숙련도(훈련 연계) | `GET /api/readiness/team/{team_id}/familiarity` | `items[].scheme_type/scheme_key/value/value_as_of` |
| 샤프니스 팀 분포 | `GET /api/readiness/team/{team_id}/sharpness` | `distribution.avg/min/max/low_sharp_count` |

> 위 항목은 현재 코드베이스에 이미 존재하는 API/필드 기반이다.

## 5-2. **현재 없음 → 신규 API 설계 필요**

### A. 전술 플랜 조회 API (신규)
- 목적: 팀 전술 플랜(스킴 + 선발/로테이션 + 역할 + 분배)을 저장/재조회.
- Endpoint: `GET /api/tactics/team/{team_id}`
- Query: `plan_id`(optional, 기본 `default`)
- Response(예시):
```json
{
  "team_id": "GSW",
  "season_year": 2026,
  "plan_id": "default",
  "plan_name": "정규 시즌 기본",
  "is_active": true,
  "offense_scheme_key": "Spread_HeavyPnR",
  "defense_scheme_key": "Drop",
  "starters": [
    {
      "slot": 1,
      "player_id": "201939",
      "offense_role_key": "Engine_Primary",
      "defense_role_key": "PnR_POA_Defender",
      "target_minutes": 32
    }
  ],
  "rotation": [
    {
      "slot": 6,
      "player_id": "201143",
      "offense_role_key": "SpotUp_Spacer",
      "defense_role_key": "PnR_POA_Defender",
      "target_minutes": 18
    }
  ],
  "constraints": {
    "minutes_total_target": 240,
    "max_minutes_per_player": 48,
    "allow_duplicate_defense_role": false
  },
  "updated_at": "2026-03-03T10:00:00Z"
}
```

### B. 전술 플랜 저장 API (신규)
- Endpoint: `PUT /api/tactics/team/{team_id}`
- Request body: GET 응답과 동일 shape(단, `updated_at` 제외)
- Response:
```json
{
  "ok": true,
  "team_id": "GSW",
  "plan_id": "default",
  "saved_at": "2026-03-03T10:05:00Z",
  "validation": {
    "minutes_total": 240,
    "has_duplicate_player": false,
    "has_invalid_role": false,
    "warnings": []
  }
}
```

### C. 전술 메타(라벨/설명) API (신규)
- 목적: 내부 키와 표시 라벨 분리.
- Endpoint: `GET /api/tactics/meta`
- Response:
```json
{
  "offense_schemes": [
    {"key": "Spread_HeavyPnR", "label": "헤비 PnR", "description": "볼핸들러 중심 2대2 빈도 증가"}
  ],
  "defense_schemes": [
    {"key": "Drop", "label": "드롭", "description": "빅맨은 페인트 근처 유지"}
  ],
  "offense_roles": [
    {"key": "Engine_Primary", "label": "1차 볼핸들러", "description": "주요 공격 시작점"}
  ],
  "defense_roles_by_scheme": {
    "Drop": [
      {"key": "PnR_POA_Defender", "label": "POA 수비", "description": "볼핸들러 압박"}
    ]
  }
}
```

### D. 전술 영향 프리뷰 API (신규)
- 목적: 전술 변경 전/후 비교 지표 제공(우측 패널).
- Endpoint: `POST /api/tactics/team/{team_id}/impact-preview`
- Request:
```json
{
  "plan": {
    "offense_scheme_key": "Spread_HeavyPnR",
    "defense_scheme_key": "Drop",
    "starters": [],
    "rotation": []
  },
  "context": {
    "opponent_team_id": "LAL",
    "as_of_date": "2026-03-03"
  }
}
```
- Response:
```json
{
  "team_id": "GSW",
  "opponent_team_id": "LAL",
  "metrics": {
    "offense_rating": {"before": 112.4, "after": 114.1, "delta": 1.7},
    "defense_rating": {"before": 110.8, "after": 109.9, "delta": -0.9},
    "pace": {"before": 98.2, "after": 99.0, "delta": 0.8}
  },
  "fit_warnings": [
    {"code": "LOW_RIM_PROTECTION", "message": "림 보호 역할이 부족합니다."}
  ],
  "minutes_check": {
    "target": 240,
    "current_total": 238,
    "delta": -2
  }
}
```

---

## 6) 정보 사용 제한(중요)
- 아래 정보는 **현재 소스 확인 전까지 UI에 노출 금지**
  - 선수 초상화 URL(현재 `/api/team-detail`에 없음)
  - 선수별 코트 핫존/샷차트(현재 전술 UI 경로에서 조회 API 없음)
  - 실시간 부상 복귀 확률 수치(메디컬 overview 외 전술 전용 지표 없음)
- 즉, 위 정보가 필요하면 반드시 신규 API 또는 기존 API 확장 후 사용한다.

---

## 7) 단계별 구현 로드맵
- Phase 1 (빠른 고도화)
  - 라벨 체계 분리(내부 키 비노출)
  - 3열 레이아웃/상태 바/분배 검증 UI
  - 기존 API만으로 구성 가능한 카드 우선 구현
- Phase 2 (기능 완성)
  - `GET/PUT /api/tactics/team/{team_id}` 도입
  - 플랜 저장/불러오기 + 저장 상태 표시
- Phase 3 (전략성 강화)
  - `POST /impact-preview` 도입
  - 전/후 비교 지표 및 자동 밸런스 추천

---

## 8) QA 체크리스트
- [ ] `총 출전시간 == 240` 검증이 항상 동작한다.
- [ ] 저장하지 않고 화면 이탈 시 경고가 노출된다.
- [ ] 스킴 변경 시 허용되지 않는 수비 역할이 자동 보정/경고 처리된다.
- [ ] API 실패 시 기존 플랜/캐시를 유지하고 사용자에게 복구 가능한 메시지를 보여준다.
- [ ] 내부 키가 사용자에게 직접 노출되지 않는다.
