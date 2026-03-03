# NBA 시뮬레이션 게임 메인 화면 UI 사양서 (실제 제작용)

## 0) 문서 목적 / 적용 범위
- 목적: 현재 Home 화면을 상업 게임 수준으로 끌어올리기 위한 **실행 가능한 UI/UX 제작 사양** 정의.
- 범위: 메인 화면(Home) 정보 구조, 비주얼 시스템, 인터랙션, 데이터 계약, API 계획.
- 원칙: 모든 화면 데이터는 **게임 코드가 보유한 상태/DB 기반 데이터만 사용**한다.

---

## 1) 제품 목표와 성공 기준
### 1.1 제품 목표
1. 첫 진입 3초 안에 “다음 경기 + 팀 상태 + 당장 해야 할 액션”을 이해시킨다.
2. NBA 스포츠 감성(중계 그래픽 + 팀 브랜딩)을 강화해 “프로덕션 퀄리티” 인상을 준다.
3. 사용자의 행동 유도(경기 시작, 전술 조정, 로스터/의무 확인) 클릭률을 높인다.

### 1.2 정량 KPI (권장)
- Home 첫 화면에서 `경기 시작` 버튼 클릭률: +20%.
- Home 진입 후 60초 내 액션(전술/훈련/의무센터 진입) 비율: +25%.
- 사용자 주관 평가(비주얼 완성도): 5점 만점 기준 4.2 이상.

---

## 2) 디자인 방향성 (컨셉)
### 2.1 컨셉명: Broadcast Premium + Arena Luxury
- Base Tone: 딥 네이비/차콜 기반 다크 테마.
- Accent: 사용자 팀 컬러를 주요 포인트(CTA, 배지, 차트 라인)에 동적 반영.
- Material: 가벼운 글래스/패널 + 섬세한 라인 하이라이트.
- Density: 정보 밀도를 높이되, 계층을 분명히 하여 복잡하지 않게 보이도록 설계.

### 2.2 피해야 할 것
- 기본 HTML 버튼/테이블 느낌.
- 의미 없는 대면적 빈 공간.
- 모든 요소를 동일 강조로 배치하는 평면적 레이아웃.

---

## 3) 정보 구조 (Information Architecture)

## 3.1 화면 그리드
- Desktop 기준 12-column grid.
- 좌우 여백 32px, 거터 24px.
- 주요 섹션 사이 간격 24~32px.

## 3.2 섹션 구성
1. **Top Navigation Bar**
   - 좌: 게임/팀 아이덴티티.
   - 중: 전역 메뉴(Home, 내 팀, 전술, 스케줄, 훈련, 순위, 대학, 시장, 메디컬).
   - 우: 저장 상태, 게임 날짜, 설정/프로필.

2. **Hero Matchup Panel (최상단 핵심 카드)**
   - 다음 경기 상대/우리 팀, 경기 시각, 홈/원정, 간단 예측 지표.
   - Primary CTA: 경기 시작.
   - Secondary CTA: 전술 조정.

3. **Team Snapshot Row (3~4개 카드)**
   - 팀 컨디션(체력/샤프니스/케미), 부상 현황, 최근 폼, 순위 변동.

4. **Action & Alerts Row**
   - 처리 필요 항목(계약 만료 임박, 투웨이 슬롯, 메디컬 리스크 상위).

5. **News & Schedule Row**
   - 팀 관련 뉴스/이슈.
   - 향후 5경기 난이도 및 일정.

---

## 4) 화면 상세 사양

## 4.1 Top Navigation
### 필수 요소
- 활성 메뉴는 언더라인 + 액센트 glow.
- 저장 버튼은 텍스트 버튼 대신 아이콘+툴팁(hover) 사용.
- 현재 게임 날짜 표기: `2025-10-19 (D-2 Next Tipoff)` 형태.

### 상태
- Default / Hover / Active / Focus-visible / Disabled.
- 키보드 접근성: `Tab` 이동 시 2px 외곽 포커스 링.

## 4.2 Hero Matchup Panel
### 레이아웃
- 좌: 상대 팀(로고, 팀명, 최근 5경기 W/L 스트립).
- 중: VS, 경기 중요 배지(정규/플옵/라이벌), 팁오프 카운트다운.
- 우: 우리 팀(로고, 팀명, 홈/원정, 부상자 수).
- 하단 액션 바:
  - Primary: `경기 시작`
  - Secondary: `전술 조정`
  - Tertiary: `빠른 진행`

### 표시 데이터
- 경기 일시, 홈/원정.
- 양 팀 최근 5경기 결과.
- 간단 승률 예측(선택, 있으면 표시 / 없으면 숨김).
- `Back-to-Back`, `Rest Day +2` 등 컨디션 배지.

### 인터랙션
- 카드 hover 시 외곽 라이트/미세 확대(1.01).
- 버튼 클릭 즉시 pressed feedback (80~120ms).

## 4.3 Team Snapshot Cards
1. **Condition Card**
   - 팀 평균 short-term stamina, long-term stamina, sharpness.
2. **Medical Card**
   - OUT/RETURNING 인원 수, 최고 리스크 선수 TOP1.
3. **Standings Card**
   - 컨퍼런스 순위, 최근 10경기, 연승/연패.
4. **Roster Attention Card**
   - OVR 상위 3명, 투웨이 슬롯 사용 현황.

## 4.4 Alerts Panel
- 우선순위 높은 알림을 최대 5개 노출.
- 예: `주전 PG 부상 복귀 임박`, `투웨이 슬롯 1개 비어 있음`, `건강 불만 HIGH 2명`.
- 알림 심각도: Info / Warning / Critical.

## 4.5 Schedule Mini Timeline
- 다음 5경기 상대 + 홈/원정 + 난이도 색상.
- 날짜 구분점과 연속 원정/백투백 시각 강조.

---

## 5) 디자인 시스템 토큰

## 5.1 Color Tokens
- `bg.base = #0B1220`
- `bg.elevated = #111827`
- `text.primary = #E5E7EB`
- `text.secondary = #94A3B8`
- `border.soft = #233047`
- `state.success = #22C55E`
- `state.warning = #F59E0B`
- `state.critical = #EF4444`
- `team.accent.primary = 팀 컬러 동적`
- `team.accent.secondary = 팀 보조 컬러 동적`

## 5.2 Typography
- Display(매치업 팀명): 48/56, 700~800.
- Heading(섹션): 24/32, 700.
- Body: 15/22, 500.
- Caption: 12/16, 500.
- 숫자(스코어/기록): tabular-nums 사용.

## 5.3 Spacing / Radius / Shadow
- Spacing: 4, 8, 12, 16, 24, 32, 40.
- Radius: 카드 16, 버튼 12, 칩 999.
- Shadow:
  - card: `0 6 24 rgba(0,0,0,0.28)`
  - hover: `0 10 28 rgba(0,0,0,0.34)`

## 5.4 Motion
- Hover transition: 160~200ms / ease-out.
- Panel enter: 220~280ms / cubic-bezier(0.2, 0.8, 0.2, 1).
- Tab change: 140~180ms.
- 원칙: motion은 의미 전달용, 장식 과다 금지.

---

## 6) 컴포넌트 상태 정의 (바이브 코딩 티 제거 핵심)
- 모든 interactive component에 대해 아래 상태를 반드시 제공:
  - default
  - hover
  - active(pressed)
  - focus-visible
  - disabled
  - loading(필요 시)
- Skeleton UI:
  - Hero 로딩 1개, Snapshot 카드 3개, Alerts 3줄.
- Empty State:
  - 데이터 없음 시 `—` 표기 대신 설명형 메시지 + 유도 액션.

---

## 7) 메인 화면 데이터 요구사항 (게임 코드 기반)

## 7.1 이미 존재하는 API 재사용
1. `/api/team-schedule/{team_id}`
   - 다음 경기, 홈/원정, 상대, 완료 경기 결과/리더 추출에 사용.
2. `/api/standings/table`
   - 컨퍼런스 순위/승패 지표 카드 구성.
3. `/api/medical/team/{team_id}/overview`
   - 부상 요약, 리스크 티어, 복귀 상태, health frustration.
4. `/api/roster-summary/{team_id}`
   - OVR 상위 선수 스냅샷.
5. `/api/two-way/summary/{team_id}`
   - 투웨이 슬롯 현황.
6. `/api/state/summary`
   - 현재 날짜/시뮬레이션 상태 확인(과다 사용은 지양).

## 7.2 추가 API 제안 (Home 전용 Aggregation)
> 목적: 프런트에서 N개 API를 병렬 조합하는 복잡도/지연을 줄이고, Home 전용 ViewModel을 서버에서 생성.

### 제안 엔드포인트
- `GET /api/home/dashboard/{team_id}`

### Query
- `include_prediction=true|false` (default false)
- `top_n=5` (알림/워치리스트 길이)

### Response (초안)
```json
{
  "meta": {
    "team_id": "GSW",
    "as_of_date": "2025-10-19",
    "season_id": "2025-26"
  },
  "hero": {
    "next_game": {
      "game_id": "...",
      "date": "2025-10-19",
      "tipoff_time": "09:30 PM",
      "is_home": false,
      "opponent_team_id": "NYK",
      "opponent_team_name": "New York Knicks",
      "labels": ["REG", "AWAY"]
    },
    "user_team": { "team_id": "GSW", "name": "Golden State Warriors" },
    "opponent_team": { "team_id": "NYK", "name": "New York Knicks" },
    "form": {
      "user_last5": ["W","L","W","W","L"],
      "opp_last5": ["W","W","L","L","W"]
    },
    "prediction": {
      "available": false,
      "user_win_prob": null,
      "model": null
    }
  },
  "snapshots": {
    "condition": {
      "stamina_short_avg": 0.84,
      "stamina_long_avg": 0.78,
      "sharpness_avg": 0.71
    },
    "medical": {
      "out_count": 1,
      "returning_count": 2,
      "high_risk_count": 3,
      "top_risk_player": {"player_id": "...", "name": "...", "risk_score": 79}
    },
    "standing": {
      "conference": "West",
      "rank": 4,
      "wins": 12,
      "losses": 7,
      "last10": "7-3",
      "streak": "W2"
    },
    "roster": {
      "top_players": [
        {"player_id": "...", "name": "...", "overall": 91}
      ],
      "two_way": {"used": 2, "max": 3, "open": 1}
    }
  },
  "alerts": [
    {"severity": "warning", "code": "INJURY_HIGH", "title": "부상 리스크 HIGH 3명", "cta": "메디컬 센터"}
  ],
  "schedule_preview": [
    {"date": "2025-10-19", "opponent": "NYK", "is_home": false, "difficulty": "medium"}
  ]
}
```

### 서버 집계 로직
- Source fan-in:
  - team-schedule + standings/table + medical/overview + roster-summary + two-way/summary.
- 알림 규칙은 서버에서 계산:
  - `out_count >= 2` → Warning.
  - `high_risk_count >= 3` → Warning.
  - `open_two_way_slots > 0` → Info.
  - `health_frustration.high_count >= 1` → Warning.

### 캐싱/성능
- Home API는 15~30초 TTL 인메모리 캐시 권장.
- 시뮬레이션 진행/날짜 변경 이벤트 시 캐시 무효화.

---

## 7.3 구현 완료 API (서버 기준)
- 엔드포인트: `GET /api/home/dashboard/{team_id}`
- Query:
  - `include_prediction` (bool, 기본 `false`)
  - `top_n` (int, 기본 `5`, 범위 `1~10`)
- 서버 동작:
  - 기존 조회 API들을 서버에서 집계해 Home 화면 전용 ViewModel로 반환.
  - 20초 TTL 인메모리 캐시 사용(동일 team/query 조합).
- 포함 데이터:
  - `meta`, `hero(next_game/form/prediction)`, `snapshots(condition/medical/standing/roster)`, `alerts`, `schedule_preview`.

## 8) 데이터 매핑 상세 (어디서 무엇을 끌어오는가)

## 8.1 Hero `next_game`
- Source: `/api/team-schedule/{team_id}`의 `games` 중 `is_completed=false` 첫 항목.
- 필드 매핑:
  - `opponent_team_name` ← `opponent_team_name`
  - `tipoff_time` ← `tipoff_time`
  - `is_home` ← `is_home`

## 8.2 최근 5경기 폼
- Source: `/api/team-schedule/{team_id}`
- 로직:
  - `is_completed=true` 경기의 `result_for_user_team` 최근 5개.
  - 상대팀 폼은 상대팀 team_id로 동일 API 호출 1회 추가(집계 API에서 처리).

## 8.3 메디컬 카드
- Source: `/api/medical/team/{team_id}/overview`
- 사용 필드:
  - `summary.injury_status_counts`
  - `summary.risk_tier_counts`
  - `watchlists.highest_risk[0]`

## 8.4 순위 카드
- Source: `/api/standings/table`
- 사용 필드:
  - 해당 팀 row의 conference/rank/W/L/streak.
- 주의: standings의 row schema 확정 후 타입 고정 필요.

## 8.5 로스터 카드
- Source: `/api/roster-summary/{team_id}`, `/api/two-way/summary/{team_id}`
- 사용 필드:
  - OVR 상위 3명.
  - 투웨이 슬롯 used/open.

## 8.6 게임 날짜/상태
- Source: `/api/state/summary` 또는 league context 공용 endpoint.
- Home에서 필요한 최소 필드만 가져오도록 경량화 권장.

---

## 9) 반응형 규칙
- ≥1440px: 12col 풀 레이아웃.
- 1024~1439px: Hero 유지, Snapshot 2x2.
- 768~1023px: Hero 세로 재배치(상대/VS/우리 순), 카드 1열.
- ≤767px: 정보 우선순위 재정렬(경기 시작 CTA > 다음 경기 > 알림 > 스냅샷).

---

## 10) 카피라이팅 가이드 (한국어)
- 버튼은 동사형 + 짧게:
  - `경기 시작`, `전술 조정`, `라인업 관리`, `의무센터 확인`.
- 상태 텍스트는 숫자+의미 동시 제공:
  - `OUT 1명 · 복귀 관리 2명`
  - `최근 5경기 3승 2패`
- 시스템 메시지는 행동 유도형:
  - `부상 위험이 높은 선수가 있습니다. 의무센터에서 확인하세요.`

---

## 11) 구현 순서 (2주 스프린트 예시)

### Sprint 1 (기반)
1. 다크 테마 토큰/타이포/버튼 상태 정리.
2. Hero 패널/Top Nav/기본 카드 컴포넌트 구현.
3. 기존 API로 최소 Home 조합 연결.

### Sprint 2 (완성)
1. `/api/home/dashboard/{team_id}` 집계 API 추가.
2. 알림 규칙/스케줄 미니 타임라인/로딩/빈 상태 추가.
3. 마이크로 인터랙션 + 성능 최적화 + QA.

---

## 12) QA 체크리스트
- [ ] 키보드 접근성(탭 이동/포커스 표시) 동작.
- [ ] 데이터 지연 시 Skeleton 노출.
- [ ] API 실패 시 섹션 단위 graceful fallback.
- [ ] 1920/1440/1280/1024/768 해상도 QA 완료.
- [ ] 팀 컬러 변경 시 대비(contrast) 기준 충족.
- [ ] 애니메이션 reduce-motion 대응.

---

## 13) Definition of Done
- Home 첫 화면에서 다음 경기/상태/액션이 즉시 인지 가능.
- 디자인 시스템 토큰과 컴포넌트 상태가 문서화/재사용 가능.
- 데이터는 게임 코드 기반으로만 구성되고, Home 집계 API 계약이 확정됨.
- 시각 완성도/일관성/인터랙션 측면에서 “프로토타입 느낌”이 제거됨.
