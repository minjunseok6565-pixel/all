# 대학 리그 UI 상업화 화면 기획안 (데이터 소스/API 설계 포함)

## 1) 문서 목적 / 원칙

- 목적: 현재 대학 리그 화면(팀 순위, 리더보드, 전문가 빅보드, 스카우팅)을 **상업용 게임 품질**로 끌어올리기 위한 실제 화면 기획안을 정의한다.
- 핵심 원칙:
  1. **현재 게임 코드에 존재하는 정보만 사용**한다.
  2. 화면에서 새로 필요한 집계/상태 값은 기존 API 조합으로 가능한지 먼저 판단한다.
  3. 기존 API로 불가능한 경우에만 **신규 API를 명시적으로 설계**한다.
  4. 신규 API는 반드시 요청/응답 필드를 정의한다.

---

## 2) 현재 사용 가능한 데이터(코드 기준 SSOT)

### 2-1. 대학 리그 기본 메타
- API: `GET /api/college/meta`
- 사용 가능 정보:
  - `season_year`, `current_date`
  - `upcoming_draft_year`
  - 대학 팀 수/선수 수/상태별 선수 수
  - 드래프트 클래스 강도(`class_strength.strength`)

### 2-2. 대학 팀 순위/카드
- API: `GET /api/college/teams?season_year=`
- 사용 가능 정보:
  - 팀 식별/이름/컨퍼런스
  - `wins`, `losses`, `srs`, `pace`, `off_ppg`, `def_ppg`
  - `roster_count`, `declared_count`

### 2-3. 팀 상세 + 로스터
- API: `GET /api/college/team-detail/{college_team_id}?season_year=`
- 사용 가능 정보:
  - 팀 시즌 성적(승패/SRS/PACE/공수 득점)
  - 로스터: 선수 기본정보 + 시즌 stats JSON

### 2-4. 대학 선수 목록(리더보드 기반)
- API: `GET /api/college/players`
- 필터/정렬 가능 정보:
  - 필터: `status`, `college_team_id`, `draft_year`, `declared_only`, `q`
  - 정렬: `pts`, `reb`, `ast`, `stl`, `blk`, `mpg`, `games`, `ts_pct`, `usg`, `fg_pct`, `age`, `class_year`, `name`, `player_id`
- 주의:
  - 숨김 정보(ovr/potential/attrs)는 노출 금지 정책.

### 2-5. 드래프트/빅보드 풀
- API: `GET /api/college/draft-pool/{draft_year}`
- 사용 가능 정보:
  - `pool_mode_used` (declared/watch/auto)
  - `prospects[]`의 공개 정보: 기본 프로필, 대학 정보, 시즌 스탯, 컨센서스 projected pick, combine/workout/interview(팀별 가시성 정책 적용)

### 2-6. 스카우팅
- 스카우트 목록: `GET /api/scouting/scouts/{team_id}`
- 배정: `POST /api/scouting/assign`
- 배정 해제: `POST /api/scouting/unassign`
- 리포트 목록: `GET /api/scouting/reports?team_id=...`
- 사용 가능 정보:
  - 스카우트 전문분야/활성여부/현재 배정
  - 리포트의 기간, 대상 선수, 스냅샷, payload, 텍스트

---

## 3) 화면 구조 개편안 (실행 가능한 수준)

## 3-1. 공통 레이아웃

### A. 상단 리그 컨텍스트 바
- 좌측: `COLLEGE LEAGUE` + 시즌/드래프트 연도
- 중앙: 상태 칩
  - 예: `시즌 2025`, `예정 드래프트 2026`, `클래스 강도 0.73`
- 우측: 빠른 액션
  - `메인으로`
  - `스카우팅 인박스 (완료 리포트 수)`

### B. 1행 KPI 요약 카드(탭 공통)
- 카드 3~4개 고정:
  1) 팀 순위 탭: 최고 SRS 팀 / 평균 SRS / DECLARED 선수 총합
  2) 리더보드 탭: 득점 1위 / 리바운드 1위 / 어시스트 1위
  3) 빅보드 탭: Tier1 수 / Lottery 컷 / 평균 projected pick 변동
  4) 스카우팅 탭: ACTIVE 배정 수 / 이번 period 리포트 수 / 미배정 스카우트 수

### C. 메인 컨텐츠
- 좌측: 리스트/테이블
- 우측: 선택 대상의 디테일 패널(고정 폭)
- 가로폭이 좁아지면 우측 패널은 Drawer로 전환

---

## 3-2. 화면별 상세 기획

### (1) 팀 순위 화면

#### 정보 구조
1. KPI 카드
2. 팀 순위 테이블
3. 선택 팀 로스터/시즌 상세 패널

#### 테이블 컬럼
- `RK`, `팀`, `컨퍼런스`, `W`, `L`, `SRS`, `PACE`, `OFF PPG`, `DEF PPG`, `DECLARED`

#### 인터랙션
- 행 클릭 시 우측 패널에서 해당 팀 로스터 표시
- 정렬: SRS 기본, 승률/PACE/OFF/DEF 전환 가능

#### 데이터 소스
- 테이블 본문: `/api/college/teams`
- 우측 패널: `/api/college/team-detail/{college_team_id}`

---

### (2) 대학 리더보드 화면

#### 정보 구조
1. 상단 KPI(현재 정렬 기준 1위 선수 카드 포함)
2. 선수 리더보드 테이블
3. 선수 상세 사이드 패널

#### 테이블 컬럼
- `RK`, `선수`, `팀`, `POS`, `PTS`, `REB`, `AST`, `STL`, `BLK`, `TS%`, `USG%`, `CLASS`

#### 인터랙션
- 정렬 드롭다운: 기존 sort 파라미터만 사용
- 선수 행 클릭 → 우측 패널에서 선수 기본 정보 + stats history

#### 데이터 소스
- 리스트: `/api/college/players` (`sort`, `order`, `limit`, `offset` 활용)
- 선수 상세: `/api/college/player/{player_id}`

---

### (3) 전문가 빅보드 화면

#### 정보 구조
1. 상단: 전문가 프로필/평가 철학 선택
2. 빅보드 리스트
3. 우측: 선수 비교/리스크 메모

#### 리스트 컬럼
- `RANK`, `선수`, `POS`, `대학`, `CLASS`, `Consensus Pick`, `Tier`, `요약`

#### 인터랙션
- 전문가 선택 시 리스트 재정렬/요약문 갱신
- 선수 선택 시 `스카우트 배정` CTA 노출

#### 데이터 소스
- 기본 데이터: `/api/college/draft-pool/{draft_year}`
- 전문가 요약문/티어/랭크는 기존 응답만으로는 불충분하므로 신규 API 필요(아래 4장 참조)

---

### (4) 스카우팅 화면

#### 정보 구조
- 좌측 상단: 스카우트 카드 리스트(전문분야, 배정 상태)
- 좌측 하단: 배정 액션(스카우트/대상선수/배정 버튼)
- 우측: 스카우팅 리포트 타임라인

#### 주요 인터랙션
- `assign`: 성공 시 해당 스카우트 상태를 ACTIVE로 즉시 반영
- `unassign`: ACTIVE 종료 후 상태 갱신
- 보고서 클릭 시 payload + report_text 상세 모달

#### 데이터 소스
- 스카우트 목록: `/api/scouting/scouts/{team_id}`
- 배정/해제: `/api/scouting/assign`, `/api/scouting/unassign`
- 리포트: `/api/scouting/reports`

---

## 4) “새로운 정보” 도입 시 API 전략

본 장은 "현재 UI에 없는 정보"를 붙일 때, 기존 API 재사용/신규 API 필요 여부를 구분한다.

## 4-1. 기존 API 조합으로 가능한 항목

1) **탭 상단 시즌 문맥 정보**
- 출처: `/api/college/meta`
- 필드: `season_year`, `upcoming_draft_year`, `class_strength.strength`

2) **팀 순위 보강 지표(OFF/DEF/PACE/DECLARED)**
- 출처: `/api/college/teams`
- 필드: `off_ppg`, `def_ppg`, `pace`, `declared_count`

3) **리더보드 다중 정렬 및 페이지네이션**
- 출처: `/api/college/players`
- 파라미터: `sort`, `order`, `limit`, `offset`

4) **스카우팅 배정 상태 및 보고서 리스트**
- 출처: `/api/scouting/scouts/{team_id}`, `/api/scouting/reports`

## 4-2. 신규 API가 필요한 항목

아래 정보는 현재 엔드포인트에서 직접 제공되지 않거나 클라이언트에서 합성 시 비용/일관성 문제가 커서 신규 API를 권장한다.

### 신규 API A: 대학 리그 대시보드 요약
- 목적: 상단 KPI 카드를 탭별로 일관된 서버 집계로 제공
- Method/Path: `GET /api/college/dashboard-summary`
- Query:
  - `season_year` (optional, int)
  - `draft_year` (optional, int; default: season_year+1)
  - `team_id` (optional, str; 스카우팅 KPI용)

#### 응답 스키마 (제안)
```json
{
  "ok": true,
  "season_year": 2025,
  "draft_year": 2026,
  "team_ranking": {
    "top_team": {
      "college_team_id": "COL_001",
      "name": "Atlantic Heights",
      "srs": 22.1
    },
    "avg_srs": 14.38,
    "declared_total": 87
  },
  "leaderboard": {
    "pts_leader": {"player_id": "CP_001", "name": "A", "value": 24.1},
    "reb_leader": {"player_id": "CP_002", "name": "B", "value": 11.4},
    "ast_leader": {"player_id": "CP_003", "name": "C", "value": 7.9}
  },
  "bigboard": {
    "pool_mode_used": "watch",
    "tier1_count": 5,
    "lottery_cut_pick": 14,
    "avg_projected_pick": 28.4
  },
  "scouting": {
    "team_id": "HOU",
    "active_assignments": 3,
    "idle_scouts": 1,
    "reports_this_period": 2
  }
}
```

### 신규 API B: 빅보드 뷰 모델(전문가 관점 포함)
- 목적: 현재 draft-pool 공개 정보에 `tier`, `summary`, `expert_rank`를 안정적으로 공급
- Method/Path: `GET /api/college/bigboard`
- Query:
  - `draft_year` (required, int)
  - `season_year` (optional, int)
  - `expert_profile` (optional, enum: `consensus|efficiency|upside|defense`)
  - `limit`, `offset`

#### 응답 스키마 (제안)
```json
{
  "ok": true,
  "draft_year": 2026,
  "season_year": 2025,
  "expert_profile": "efficiency",
  "pool_mode_used": "watch",
  "count": 100,
  "items": [
    {
      "rank": 1,
      "temp_id": "DP_001",
      "player": {
        "name": "Aurelio Hawkins",
        "pos": "SG",
        "age": 20,
        "college_team_id": "COL_001",
        "college_team_name": "Atlantic Heights",
        "class_year": 2
      },
      "consensus_projected_pick": 2,
      "tier": "Tier 1",
      "summary": "Strengths: Touch A, Handle A. Concern: Tools D.",
      "signals": {
        "production_score": 0.88,
        "efficiency_score": 0.84,
        "risk_score": 0.31
      }
    }
  ]
}
```

> 주의: `signals`는 공개 가능한 통계 기반 점수만 사용(숨김 ratings 불가).

### 신규 API C: 스카우팅 인박스 요약
- 목적: 화면 우측 상단/탭 배지에서 "읽지 않은 리포트" 등 즉시 사용
- Method/Path: `GET /api/scouting/inbox-summary`
- Query:
  - `team_id` (required, str)
  - `period_key` (optional, YYYY-MM)

#### 응답 스키마 (제안)
```json
{
  "ok": true,
  "team_id": "HOU",
  "period_key": "2025-11",
  "reports_total": 12,
  "reports_this_period": 3,
  "active_assignments": 4,
  "idle_scouts": 1,
  "last_report": {
    "report_id": "SR_123",
    "as_of_date": "2025-11-30",
    "target_player_id": "CP_9001",
    "scout_id": "SC_01"
  }
}
```

---

## 5) 화면-데이터 매핑 표 (개발 handoff)

| 화면 블록 | 필요 데이터 | 우선 소스 | 대체/신규 |
|---|---|---|---|
| 상단 시즌 문맥 | 시즌 연도, 드래프트 연도, 클래스 강도 | `/api/college/meta` | - |
| 팀 순위 테이블 | W/L/SRS/PACE/OFF/DEF/DECLARED | `/api/college/teams` | - |
| 팀 우측 패널 | 팀 시즌 + 로스터 | `/api/college/team-detail/{id}` | - |
| 리더보드 테이블 | 선수 통계/정렬 | `/api/college/players` | - |
| 선수 디테일 | 선수 히스토리 | `/api/college/player/{player_id}` | - |
| 빅보드 기본 리스트 | draft prospects 공개 정보 | `/api/college/draft-pool/{draft_year}` | `/api/college/bigboard` |
| 스카우트 리스트 | 전문분야/ACTIVE 상태 | `/api/scouting/scouts/{team_id}` | - |
| 스카우팅 리포트 | report list + payload/text | `/api/scouting/reports` | - |
| 탭 KPI 카드 | 종합 집계 | 기존 API 조합 가능(클라) | `/api/college/dashboard-summary` 권장 |
| 스카우팅 배지 | 기간별 리포트/미확인 요약 | `/api/scouting/reports` 집계 | `/api/scouting/inbox-summary` 권장 |

---

## 6) 절대 금지/검증 체크리스트

1. UI 표출 데이터가 아래 금지 목록을 참조하지 않도록 검증
   - `ovr`, `attrs`, `Potential`, `potential_*`, 민감 성향/부상 내부값
2. 빅보드 summary 생성 시 숨김 값 기반 문구 생성 금지
3. 모든 신규 API는 "통계 기반 공개값" 또는 "기존 공개 엔드포인트 재가공"만 사용
4. QA 시, 기존 화면 값과 신규 화면 값의 원천 불일치 여부를 스냅샷 비교

---

## 7) 구현 상태 (API 준비 완료)

다음 조회 API가 실제 서버 라우트에 구현되어 UI 개발에서 바로 호출 가능하다.

- `GET /api/college/dashboard-summary`
- `GET /api/college/bigboard`
- `GET /api/scouting/inbox-summary`

> 참고: 빅보드의 `tier`, `summary`, `signals`는 공개 통계 기반 파생값으로 계산되며 숨김 rating/attrs를 사용하지 않는다.

---

## 8) 1차 릴리스 범위 제안 (2주)

### Week 1
- 레이아웃/컴포넌트 재구성(상단 컨텍스트 바, KPI 카드, 고정 패널)
- 기존 API만 사용해 팀 순위/리더보드/스카우팅 화면 개선

### Week 2
- 신규 API A/B/C 중 A 우선 구현
- 빅보드 전문화는 API B를 붙여 2차 반영
- 인박스 배지는 API C 또는 reports 집계 방식 중 택1

