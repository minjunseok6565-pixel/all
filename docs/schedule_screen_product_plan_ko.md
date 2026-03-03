# NBA 스케줄 화면 상업화 기획안 (실행 스펙)

작성일: 2026-03-03  
대상 화면: `static/NBA.html`의 `#schedule-screen`  
목표: 현재 스케줄 화면을 "상업용 게임 품질"로 고도화하되, **현재 코드베이스에 존재하는 데이터만 사용**한다.

---

## 1) 범위와 원칙

### 1.1 데이터 사용 원칙 (필수)
- 외부 스포츠 API(실 NBA 일정/중계사/선수 프로필 이미지)는 **사용 금지**.
- 게임 내 이미 존재하는 서버 API/상태에서만 조회.
- 화면에 새 정보를 붙일 때는 아래 둘 중 하나만 허용:
  1. 이미 존재하는 조회 API 필드 사용
  2. 현재 서버 상태(`master_schedule`, `workflow_state`)를 원천으로 한 **신규 내부 API 설계**

### 1.2 이번 문서의 산출물
- 화면 IA(정보구조), 섹션별 UI 스펙, 상태/인터랙션 규칙
- **필드 단위 데이터 매핑표**(어디서 가져오는지 명시)
- 신규 API 필요 시 응답 스키마 제안

---

## 2) 화면 목표 (유저 관점)

1. "다음 경기 준비"를 5초 안에 파악
   - 상대, 홈/원정, 시간, 최근 폼, 위험 신호
2. "지난 경기 리뷰"를 10초 안에 파악
   - 결과, 누적 전적, 팀 내 리더(PTS/REB/AST)
3. "중장기 일정 난이도"를 빠르게 파악
   - 연전/원정 연속/강팀 구간

---

## 3) 권장 화면 구조 (실제 배치안)

## 3.1 상단 Hero + 시즌 컨텍스트 바
- 좌측: 팀명 + 시즌 라벨
- 우측: 현재 인게임 날짜, 컨퍼런스 순위, 최근 10경기, 현재 연승/연패

**사용 데이터(기존 API로 충족 가능)**
- 팀명: 클라이언트 `TEAM_FULL_NAMES` 또는 `/api/team-detail/{team_id}`
- 현재 날짜: `/api/state/summary` → `workflow_state.league.current_date`
- 순위/최근10/연승연패: `/api/standings/table`의 팀 row (`rank`, `l10`, `strk`)

---

## 3.2 Next Game Focus 카드 (최상단 우선 정보)
- 표시 항목
  - 날짜(`YYYY-MM-DD`) + 시간(`tipoff_time`)
  - 상대 표기(`@ BOS`/`vs NYK`) + 상대 풀네임
  - 홈/원정
  - 빠른 액션 버튼(경기 진행, 전술)

**사용 데이터(기존 API로 충족 가능)**
- `/api/team-schedule/{team_id}`의 `games[]`에서
  - `is_completed == false`이면서 `date >= current_date`인 첫 경기
  - `date`, `tipoff_time`, `opponent_label`, `opponent_team_name`, `is_home`

---

## 3.3 본문 2단: 완료 경기 / 예정 경기

### A. 완료 경기 리스트
- 컬럼(유지): DATE / OPPONENT / RESULT / W-L / Hi PTS / Hi REB / Hi AST
- 행 강조:
  - 승리(W): 블루-그린 계열 배지
  - 패배(L): 오렌지-레드 계열 배지
- 보조 정보:
  - `record_after_game.display`를 누적 전적으로 표시

**사용 데이터(기존 API로 충족 가능)**
- `/api/team-schedule/{team_id}`
  - `date_mmdd`, `opponent_label`, `opponent_team_name`
  - `result.display`, `result.wl`
  - `record_after_game.display`
  - `leaders.points|rebounds|assists` (name, value)

### B. 예정 경기 리스트
- 컬럼(유지): DATE / OPPONENT / TIME
- 가독성 개선:
  - TIME은 pill 컴포넌트 유지
  - 당일 경기(today)는 강조 테두리

**사용 데이터(기존 API로 충족 가능)**
- `/api/team-schedule/{team_id}`
  - `date_mmdd`, `opponent_label`, `opponent_team_name`, `tipoff_time`

---

## 3.4 우측 보조 패널(선택): 일정 리스크 인사이트

> 이 패널은 "새로운 외부 데이터" 없이도 계산 가능.  
> 일정 자체와 기존 API를 조합해 유저의 의사결정(훈련 강도, 로테이션)에 도움.

- `Back-to-Back`: 날짜 차이 1일 경기 여부(클라이언트 계산 가능)
- `Road Trip`: 연속 원정 경기 수(클라이언트 계산 가능)
- `의료 리스크 요약`(선택): 팀 위험도 high/medium/low 카운트
  - `/api/medical/team/{team_id}/overview` 사용

---

## 4) 컴포넌트/상태 스펙

## 4.1 상태 종류
- Loading: 스켈레톤 6행 + 헤더 플레이스홀더
- Empty(완료): "완료된 경기가 없습니다"
- Empty(예정): "예정된 경기가 없습니다"
- Error: API 실패 사유 표시 + 재시도 버튼

## 4.2 인터랙션
- 정렬: 기본 `date ASC` (서버 정렬 유지)
- hover: 행 배경 +2~4% 명도 상승, 우측 얇은 강조 라인
- focus(키보드): 2px 포커스 링
- 트랜지션: 160~200ms ease-out

## 4.3 타이포/밀도
- 숫자(스코어/시간/전적): tabular nums
- 행 높이: desktop 48px, compact 40px
- 테이블 헤더: 본문 대비 1단계 작은 캡션 스타일

---

## 5) 데이터 매핑표 ("어디서 가져오나")

| UI 항목 | 데이터 필드 | 조회 API | 비고 |
|---|---|---|---|
| 현재 날짜 | `workflow_state.league.current_date` | `GET /api/state/summary` | 상단 헤더 |
| 팀명/요약 | `summary.team_id`, `wins`, `losses`, `rank` | `GET /api/team-detail/{team_id}` | 헤더 보조 텍스트 |
| 순위/L10/STRK | `rank`, `l10`, `strk` | `GET /api/standings/table` | 상단 컨텍스트 바 |
| 완료/예정 경기 리스트 | `games[]` | `GET /api/team-schedule/{team_id}` | 화면 핵심 데이터 |
| 결과 배지 | `result.wl`, `result.display` | `GET /api/team-schedule/{team_id}` | 승/패 스타일 분기 |
| 누적 전적 | `record_after_game.display` | `GET /api/team-schedule/{team_id}` | 완료 경기 전용 |
| 경기 리더 | `leaders.points/rebounds/assists` | `GET /api/team-schedule/{team_id}` | 완료 경기 전용 |
| 다음 경기 시간 | `tipoff_time` | `GET /api/team-schedule/{team_id}` | 랜덤 시간 fallback 제거 권장 |
| 부상/컨디션 경고(선택) | `risk_counts`, `status_counts` | `GET /api/medical/team/{team_id}/overview` | 우측 인사이트 패널 |

---

## 6) "현재 UI에 없는 정보" 추가 시 정책

## 6.1 바로 추가 가능한 정보(신규 API 불필요)
1. 오늘 경기 여부 배지
- 소스: `team-schedule.games[].date` vs `current_date`
- 계산 위치: 클라이언트

2. Back-to-Back 경고
- 소스: 연속 경기의 `date`
- 계산 위치: 클라이언트(정렬된 일정 기준)

3. 연속 원정(road trip) 길이
- 소스: `games[].is_home`
- 계산 위치: 클라이언트

## 6.2 신규 API가 필요한 정보(설계 포함)

아래 정보는 현재 API 응답에 없으므로, 붙이려면 신규 API가 필요함.

### A) 월별 요약 블록 (예: 11월 9승 4패, 평균 득실차)

**신규 API 제안**  
`GET /api/team-schedule/monthly-summary/{team_id}`

**응답 필드 제안**
```json
{
  "team_id": "GSW",
  "season_id": "2025-26",
  "months": [
    {
      "month": "2025-10",
      "games": 6,
      "wins": 4,
      "losses": 2,
      "win_pct": 0.667,
      "avg_score_for": 112.3,
      "avg_score_against": 107.8,
      "avg_point_diff": 4.5,
      "completed_games": 6
    }
  ]
}
```

**서버 원천 데이터**
- `league.master_schedule.games`의 날짜/점수
- 팀 기준 home/away 변환 후 월 단위 집계

### B) 구간 난이도 지표 (다음 10경기 상대 강도)

**신규 API 제안**  
`GET /api/team-schedule/segment-strength/{team_id}?window=10`

**응답 필드 제안**
```json
{
  "team_id": "GSW",
  "window": 10,
  "as_of_date": "2025-11-12",
  "segment": {
    "start_game_id": "2025-26-RS-00124",
    "end_game_id": "2025-26-RS-00133",
    "avg_opponent_win_pct": 0.548,
    "home_games": 4,
    "away_games": 6,
    "back_to_back_sets": 2,
    "strength_tier": "HARD"
  }
}
```

**서버 원천 데이터**
- 일정: `master_schedule.games`
- 상대 승률: 기존 standings 계산 로직(`team_utils`의 승/패 집계) 재사용

---

## 7) 단계별 구현 우선순위 (기획 기준)

### Phase 1 (즉시 가능, 기존 API만)
- 상단 컨텍스트 바 추가(현재 날짜/순위/L10/STRK)
- Next Game 포커스 카드 강화
- 완료/예정 테이블 시각 고도화(배지/상태/밀도)
- 랜덤 tipoff fallback 제거(가능하면 API 값만 사용)

### Phase 2 (선택, 신규 API 1~2개)
- 월별 요약 블록
- 다음 10경기 난이도 패널

### Phase 3 (아트 고도화)
- 팀 브랜딩 테마(색/텍스처/애니메이션)
- 컴포넌트 토큰 정리(라운드/섀도우/간격)

---

## 8) 품질 기준 ("바이브 코딩 티" 제거 체크리스트)

- 동일 유형 컴포넌트의 radius/padding/border 두께가 화면 전체에서 일관적인가
- 숫자 정렬이 흔들리지 않는가(tabular nums)
- 상태 표현이 색상 하나에만 의존하지 않는가(텍스트+아이콘+배지)
- 에러/빈 상태/로딩 상태가 디자인된 화면으로 제공되는가
- 같은 데이터가 위치마다 다른 의미로 보이지 않는가(예: 다음 경기 시간)

---

## 9) 최종 결론
- 현재 코드베이스 기준으로도 상업용에 가까운 스케줄 UX를 만드는 데 필요한 핵심 데이터는 이미 확보되어 있다.
- 특히 `/api/team-schedule/{team_id}`가 결과/리더/누적 전적/예정 시간까지 포함하므로, **Phase 1은 신규 API 없이 진행 가능**.
- 다만 "월별 퍼포먼스 요약"과 "구간 난이도"는 서버 집계 API를 추가하면 프론트 복잡도와 품질 리스크를 크게 줄일 수 있다.
