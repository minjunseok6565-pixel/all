# 팀 훈련 화면 상업화 기획안 (실행 설계 기준)

## 0) 전제 및 데이터 사용 원칙
- 본 문서는 **현재 코드베이스에 이미 존재하는 데이터만 사용**하는 것을 원칙으로 한다.
- 화면 고도화를 위해 새 API를 추가하더라도, 응답 필드는 아래 소스에서만 계산/조합한다.
  - `master_schedule.games` 기반 팀 일정 (`/api/team-schedule/{team_id}`)
  - 팀/선수 로스터 및 컨디션 (`/api/team-detail/{team_id}`)
  - 날짜별 팀 훈련 세션 (`/api/practice/team/{team_id}/session`, `/sessions`)
  - 훈련 효과 프리뷰 (`/api/practice/team/{team_id}/preview`)
  - 전술 익숙도/샤프니스 (`/api/readiness/team/{team_id}/familiarity`, `/sharpness`)
  - 메디컬 리스크 요약 (`/api/medical/team/{team_id}/overview`)
  - 게임 기준 날짜 (`/api/state/summary` -> `workflow_state.league.current_date`)
- **금지**: DB/코드에 없는 감성 지표(예: 팀 케미 점수, 락커룸 분위기 점수, 코치 성향 점수)를 임의 생성해서 UI에 노출하지 않는다.

---

## 1) 화면 목표 (훈련 UI를 게임다운 운영 콘솔로 재정의)

### 1-1. UX 목표
1. 일정과 훈련을 동시에 의사결정할 수 있어야 한다.
2. 선택한 훈련의 결과를 적용 전에 수치로 확인할 수 있어야 한다.
3. 부상/피로 리스크를 즉시 인지해 “안전한 편성”을 유도해야 한다.

### 1-2. 비주얼 목표
- 현재 단순 박스형 UI에서 벗어나 `상단 KPI + 메인 캘린더 + 우측 상세/프리뷰`의 3축 레이아웃으로 전환.
- 캘린더 셀을 이벤트 중심(경기/훈련/휴식/경고)으로 시각적 계층화.
- 훈련 버튼은 단순 나열이 아닌 “행동 바(Action Bar)”로 재구성.

---

## 2) IA(정보 구조) 및 화면 레이아웃

## 2-1. 최종 레이아웃
- **A. 상단 상태 바 (Header KPI Bar)**
  - 팀명, 시즌/주차, 기준일
  - 팀 평균 샤프니스, 저샤프 선수 수, 고위험 선수 수
  - 다음 경기까지 남은 일수, 최근 7일 경기 수
- **B. 중앙 캘린더 패널**
  - 4주 캘린더(현재 유지)
  - 각 날짜의 상태 배지(경기 / AUTO / 지정 / 휴식 / 경고)
- **C. 우측 상세 패널**
  - 선택 날짜 목록
  - 선택 훈련 세션 설정(공격/수비 스킴, 스크리미지 참가자)
  - 적용 전 효과 프리뷰(익숙도/샤프니스 델타)
- **D. 하단 액션 바**
  - 훈련 타입 선택 버튼 + 적용/초기화 버튼

## 2-2. 날짜 셀 상태 분류
- `GAME`: 경기일 (상대팀 표기)
- `AUTO`: 자동 결정된 훈련
- `USER_SET`: 유저 지정 훈련
- `REST`: 훈련 없음
- `RISK_HIGH`: 고위험(부상/저샤프 경고 배지)

> `RISK_HIGH`는 새 데이터가 아니라 기존 `medical overview` 및 `sharpness` 집계의 시각적 파생 상태다.

---

## 3) 컴포넌트별 데이터 매핑 (기존 API 우선)

## 3-1. 상단 상태 바
| UI 항목 | 데이터 소스 | 타입 |
|---|---|---|
| 팀명/팀 ID | `/api/team-detail/{team_id}` (`team.name`, `team.team_id`) | 기존 |
| 기준일 | `/api/state/summary` (`workflow_state.league.current_date`) | 기존 |
| 평균 샤프니스/저샤프 인원 | `/api/readiness/team/{team_id}/sharpness` (`distribution.avg`, `distribution.low_sharp_count`) | 기존 |
| 고위험 인원 | `/api/medical/team/{team_id}/overview` (`summary.risk_tier_counts.HIGH`) | 기존 |
| 다음 경기 D-day | **신규 API A** (아래 설계) | 신규 |

## 3-2. 캘린더 셀
| UI 항목 | 데이터 소스 | 타입 |
|---|---|---|
| 경기 여부/상대팀 | `/api/team-schedule/{team_id}` (`games[].date`, `opponent_team_id`) | 기존 |
| 저장된 세션 | `/api/practice/team/{team_id}/sessions?date_from&date_to` (`sessions[date]`) | 기존 |
| 미저장 날짜 AUTO 세션 | `/api/practice/team/{team_id}/session?date_iso=...` | 기존 |
| 셀 위험 배지 | **신규 API B** (기존 medical/readiness를 일자 단위 집계) | 신규 |

## 3-3. 우측 상세 패널
| UI 항목 | 데이터 소스 | 타입 |
|---|---|---|
| 선택 세션 기본값 | 프론트 state + `/api/readiness/.../familiarity` | 기존 |
| 세션 프리뷰(익숙도/샤프니스) | `/api/practice/team/{team_id}/preview` | 기존 |
| 다중 날짜 총합 프리뷰 | **신규 API C** (preview 반복 호출 서버 집계) | 신규 |

---

## 4) 인터랙션 시나리오 (화면 기획 수준)

## 시나리오 1: 경기 전 훈련 편성
1. 유저가 캘린더에서 경기 전 2일 선택.
2. 액션 바에서 `FILM` 선택.
3. 우측 패널에 `선택일 2일`, `공/수 익숙도 gain`, `평균 sharpness delta` 즉시 표시.
4. 고위험 인원이 임계치 이상이면 경고 배지와 함께 `RECOVERY` 권고 문구 표시.
5. 적용 버튼 클릭 시 날짜별 세션 저장.

## 시나리오 2: 청백전 참가자 편집
1. `SCRIMMAGE` 선택 시 현재 로스터와 단기/장기 체력, 샤프니스 표시.
2. 참가자 PID 편집.
3. 프리뷰에서 참가자/비참가자의 sharpness 변화량 분리 노출.
4. 저장.

---

## 5) 신규 API 설계

> 목적: 기존 API를 억지로 프론트에서 다중 호출/조합하던 부분을 서버 집계로 정리.
> 제약: 응답 필드는 기존 데이터(스케줄, practice, readiness, medical, roster)에서만 계산.

## API A) 헤더 요약
### `GET /api/practice/team/{team_id}/dashboard-summary`
#### Query
- `season_year?: int`
- `as_of_date?: YYYY-MM-DD` (없으면 현재 게임 날짜)

#### Response
```json
{
  "team_id": "HOU",
  "season_year": 2025,
  "as_of_date": "2025-10-20",
  "next_game": {
    "date": "2025-10-22",
    "opponent_team_id": "BOS",
    "is_home": true,
    "days_until": 2
  },
  "recent_load": {
    "games_last_7_days": 2,
    "practices_last_7_days": 4
  },
  "readiness": {
    "sharpness_avg": 56.2,
    "low_sharp_count": 3
  },
  "medical": {
    "risk_high_count": 2,
    "out_count": 1,
    "returning_count": 1
  }
}
```
#### 데이터 생성 근거
- next_game/recent_load: `team-schedule.games` + `as_of_date`
- readiness: `readiness sharpness distribution`
- medical: `medical overview summary`

## API B) 캘린더 윈도우
### `GET /api/practice/team/{team_id}/calendar-window`
#### Query
- `season_year?: int`
- `date_from: YYYY-MM-DD`
- `date_to: YYYY-MM-DD`

#### Response
```json
{
  "team_id": "HOU",
  "season_year": 2025,
  "date_from": "2025-10-13",
  "date_to": "2025-11-09",
  "days": [
    {
      "date_iso": "2025-10-20",
      "is_game_day": false,
      "opponent_team_id": null,
      "session": {
        "type": "RECOVERY",
        "is_user_set": false
      },
      "risk_flags": {
        "high_medical_risk": true,
        "low_sharpness_cluster": false
      },
      "ui_tags": ["AUTO", "RISK_HIGH"]
    }
  ]
}
```
#### 데이터 생성 근거
- is_game_day/opponent: `team-schedule`
- session: 기존 `sessions` + 누락일 `session` 자동해결
- risk_flags:
  - `high_medical_risk`: `medical overview.watchlists.highest_risk` 기반(당일 기준)
  - `low_sharpness_cluster`: `sharpness distribution.low_sharp_count` 임계치 기반

## API C) 다중 날짜 프리뷰
### `POST /api/practice/team/{team_id}/preview-range`
#### Request
```json
{
  "season_year": 2025,
  "dates": ["2025-10-20", "2025-10-21"],
  "session": {
    "type": "FILM",
    "offense_scheme_key": "PACE_5OUT",
    "defense_scheme_key": "MAN_TO_MAN",
    "participant_pids": [],
    "non_participant_type": "RECOVERY"
  }
}
```

#### Response
```json
{
  "team_id": "HOU",
  "season_year": 2025,
  "dates": ["2025-10-20", "2025-10-21"],
  "per_date": [
    {
      "date_iso": "2025-10-20",
      "familiarity_gain": { "offense_gain": 0.8, "defense_gain": 0.8 },
      "avg_sharpness_delta": 1.2
    }
  ],
  "aggregate": {
    "offense_gain_sum": 1.6,
    "defense_gain_sum": 1.6,
    "avg_sharpness_delta_mean": 1.2
  }
}
```
#### 데이터 생성 근거
- 기존 `/api/practice/team/{team_id}/preview`를 날짜별 반복 계산 후 서버 집계.

---

## 6) 새로 추가되는 정보의 “출처 명세”

## 6-1. 현재 UI에 없는 정보, 그러나 추가 가능한 항목
1. `다음 경기까지 남은 일수`  
   - 출처: `team-schedule.games` + `as_of_date` 비교
2. `최근 7일 경기/훈련 수`  
   - 출처: `team-schedule` + `practice sessions`
3. `고위험 선수 수`  
   - 출처: `medical overview.summary.risk_tier_counts.HIGH`
4. `저샤프 군집 경고`  
   - 출처: `sharpness distribution.low_sharp_count`

## 6-2. 추가 금지 항목 (현 코드 기준)
- 팀 케미스트리 점수 (별도 저장/계산 없음)
- 코치 성향 점수 (명시 API/DB 없음)
- 선수 심리 상태 지수(health frustration 외 확장 정서 지표 없음)

---

## 7) 프론트 연동 순서 (구현 리스크 최소화)
1. **1단계(기존 API만 사용)**: 레이아웃 재구성 + 현재 데이터로 KPI/캘린더/상세 패널 구성.
2. **2단계(API A/B 추가)**: 다중 호출 축소, 헤더/캘린더 집계 정확도 개선.
3. **3단계(API C 추가)**: 다중 날짜 프리뷰 UX 고도화.

---

## 8) 품질 체크리스트 (상업용 완성도 기준)
- 정보 우선순위: 헤더 KPI -> 캘린더 선택 -> 우측 프리뷰 순으로 3클릭 이내 의사결정 가능.
- 상태 가시성: `경기일/지정/AUTO/위험`을 셀에서 1초 이내 구분 가능.
- 데이터 정합성: 모든 수치는 기존 API 응답 또는 그 조합으로 재현 가능.
- 성능: 초기 진입 API 호출 수(현재 다건 병렬) 대비 집계 API 도입 후 30% 이상 감소 목표.

