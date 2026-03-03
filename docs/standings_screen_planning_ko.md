# NBA 시뮬레이션 순위표 화면 기획안 (상업용 품질 목표)

## 0) 전제 / 범위
- 본 문서는 `순위표(Standings) 화면`의 기획안이다.
- **원칙:** 현재 코드에 존재하는 데이터만 우선 사용한다.
- 현재 코드에 없는 정보를 추가하려면, 본 문서에 명시한 신규 API를 먼저 구현한 뒤 UI에 연결한다.

---

## 1) 목표 UX (한 줄 정의)
"숫자 테이블"이 아니라, 시즌 경쟁 구도와 플레이오프 긴장감을 즉시 읽을 수 있는 **게임형 지휘실 대시보드**로 전환한다.

### 핵심 사용자 질문
1. 내 팀은 지금 안전권인가? (직행/플레이인/탈락)
2. 최근 폼은 좋은가? (연승/연패, 최근 10경기)
3. 상위권/경쟁팀과의 격차는 얼마나 되는가? (GB, 득실차, 공격/수비 지표)
4. 당장 어떤 액션을 해야 하는가? (다음 경기 난이도, 목표 달성률)

---

## 2) 정보 구조 (IA)

## 2.1 화면 구조
1. **상단 HUD 바**
   - 시즌, 현재 날짜, 사용자 팀, 동/서부 토글, 정렬 기준
2. **핵심 KPI 스트립**
   - 내 팀 랭크, 승률, GB, 최근 10경기, 연속 기록
3. **메인 스탠딩 테이블**
   - 구간 강조(1~6, 7~10, 11~15)
   - 고정 핵심 컬럼 + 확장 통계 컬럼
4. **우측 컨텍스트 패널**
   - 내 팀 최근 10경기 추세
   - 다음 경기 정보
   - (선택) 샐러리/캡 여유

## 2.2 기본/확장 컬럼 분리
- **기본 컬럼(항상 표시):** RK, TEAM, W, L, PCT, GB, STRK, L10
- **확장 컬럼(토글):** HOME, AWAY, DIV, CONF, PPG, OPP PPG, DIFF

---

## 3) 시각 디자인 시스템 제안

## 3.1 톤 앤 무드
- 베이스: 딥 네이비/차콜 다크 테마
- 강조색: 컨퍼런스/상태 구간용 2~3개만 사용
- 목적: 방송 그래픽 + 게임 HUD 감성

## 3.2 타이포/간격
- 숫자 컬럼은 tabular figures 적용(자릿수 흔들림 제거)
- 타이틀/서브타이틀/본문/숫자 스타일을 토큰화
- 4/8px spacing scale로 여백 리듬 통일

## 3.3 상태 표현
- 1~6위: 플레이오프 직행(강조 바 + 은은한 하이라이트)
- 7~10위: 플레이인(중간 강조)
- 11~15위: 저채도 처리
- 6위/10위 경계선 라벨: `PLAY-IN CUT`

## 3.4 모션 원칙
- 정렬/필터/행 hover 전환 150~220ms
- opacity + translateY 4px 중심
- `prefers-reduced-motion` 대응

---

## 4) 컴포넌트 상세 기획

## 4.1 Standings HUD
- 좌측: `STANDINGS` / `2025-26 REGULAR SEASON`
- 중앙: East/West segmented control
- 우측: `정렬 기준`, `업데이트 시각`

## 4.2 KPI 카드 (내 팀)
- Rank (Conference)
- Record (W-L)
- PCT
- GB
- L10 / STRK

## 4.3 Team Row 컴포넌트
- 팀 셀: 팀명(현재는 전체명 텍스트) + 선택 상태 강조
- 행 hover: 배경 미세 상승
- Diff 컬럼: +/- 색상 일관 규칙

## 4.4 우측 컨텍스트 패널
- 최근 10경기 시퀀스(예: W W L W ...)
- 다음 경기: 상대, 홈/원정, 날짜/시간
- 선택 확장: 팀 Payroll / Cap Space

---

## 5) 데이터 소스 명세 (현재 코드 기준)

## 5.1 기존 API만으로 가능한 정보 (즉시 구현 가능)

| UI 항목 | 데이터 소스 | 사용 필드 |
|---|---|---|
| 컨퍼런스 순위표 본문 | `GET /api/standings/table` | `east[]/west[]`의 `rank, team_id, wins, losses, pct, gb_display, home, away, div, conf, ppg, opp_ppg, diff, strk, l10` |
| 동/서부 분리 | `GET /api/standings/table` | 응답 루트 `east`, `west` |
| 플레이오프/플레이인 구간 표시 | `GET /api/standings/table` | `rank`(UI에서 1~6/7~10/11~15 판정) |
| 내 팀 KPI | `GET /api/team-detail/{team_id}` | `summary.rank, wins, losses, win_pct, gb, point_diff` |
| 다음 경기 카드 | `GET /api/team-schedule/{team_id}` | `games[]`의 `date, is_home, opponent_team_id, opponent_team_name, tipoff_time, status, is_completed` |
| 최근 10경기 추세 문자열 | `GET /api/team-schedule/{team_id}` | `games[]`의 완료 경기에서 승/패 시퀀스 계산 |
| 화면 상단 현재 날짜 | `GET /api/state/summary` | `workflow_state.league.current_date` |
| 팀 재정 요약(선택) | `GET /api/team-detail/{team_id}` | `summary.payroll, summary.cap_space` |

> 주의: 팀 로고/팀 대표색은 현재 API 필드에 없다. 팀 로고 자리는 텍스트/모노그램 fallback으로 설계하거나, 신규 API를 추가해야 한다.

## 5.2 "현재 코드에 없는 정보"를 추가하려면 필요한 신규 API

아래 2개는 상업용 고급 연출에 유용하지만, 현재 API 응답에 없는 정보다.

### A. 팀 브랜딩 메타 API
- **Endpoint:** `GET /api/teams/branding`
- **용도:** 팀 로고, 팀 컬러, 약어 표기 통일
- **응답 예시:**
```json
{
  "teams": [
    {
      "team_id": "BOS",
      "display_name": "Boston Celtics",
      "short_name": "BOS",
      "logo_url": "/static/assets/team_logos/bos.svg",
      "colors": {
        "primary": "#007A33",
        "secondary": "#BA9653",
        "text_on_primary": "#FFFFFF"
      }
    }
  ],
  "updated_at": "2026-03-03"
}
```
- **필드 정의:**
  - `team_id: string`
  - `display_name: string`
  - `short_name: string`
  - `logo_url: string | null`
  - `colors.primary: string(hex)`
  - `colors.secondary: string(hex)`
  - `colors.text_on_primary: string(hex)`
  - `updated_at: string(YYYY-MM-DD)`

### B. 타이브레이커/레이스 인사이트 API
- **Endpoint:** `GET /api/standings/race-insights?conference=East&team_id=HOU`
- **용도:** "컷라인까지 몇 게임", "바로 위/아래 경쟁팀" 같은 설명형 인사이트 제공
- **응답 예시:**
```json
{
  "conference": "East",
  "team_id": "HOU",
  "as_of_date": "2026-03-03",
  "current_rank": 8,
  "cutlines": {
    "playoff_direct_rank": 6,
    "playin_rank": 10,
    "gb_to_direct_cut": 1.5,
    "gb_to_playin_cut": -2.0
  },
  "neighbors": {
    "above": {"team_id": "MIA", "rank": 7, "gb_gap": 0.5},
    "below": {"team_id": "CHI", "rank": 9, "gb_gap": 0.5}
  },
  "tiebreaker_context": {
    "vs_above_head_to_head": "1-2",
    "vs_below_head_to_head": "2-1"
  }
}
```
- **필드 정의:**
  - `conference: "East" | "West"`
  - `team_id: string`
  - `as_of_date: string(YYYY-MM-DD)`
  - `current_rank: int`
  - `cutlines.playoff_direct_rank: int`
  - `cutlines.playin_rank: int`
  - `cutlines.gb_to_direct_cut: float`
  - `cutlines.gb_to_playin_cut: float`
  - `neighbors.above|below.team_id: string | null`
  - `neighbors.above|below.rank: int | null`
  - `neighbors.above|below.gb_gap: float | null`
  - `tiebreaker_context.vs_above_head_to_head: string | null`
  - `tiebreaker_context.vs_below_head_to_head: string | null`

---

## 6) 단계별 구현 우선순위 (기획 관점)

### Phase 1 (기존 API 100% 활용)
1. HUD + KPI + 구간 강조 + 컬럼 우선순위 재배치
2. 내 팀 컨텍스트 패널(다음 경기/최근 10경기)
3. 모션/타이포/컬러 토큰 정리

### Phase 2 (신규 API 연동)
1. 팀 브랜딩 메타(로고/컬러)
2. 레이스 인사이트(컷라인/경쟁팀/타이브레이커)

### Phase 3 (연출 고도화)
1. 행 선택 시 우측 패널 팀 전환
2. 미니 차트/애니메이션 강화

---

## 7) 품질 체크리스트 ("바이브 코딩 티" 제거용)
- 시각 계층(제목/핵심/보조)이 3단계 이상 명확한가?
- 숫자 정렬/자릿수/소수점 규칙이 화면 전체에서 일관적인가?
- 동일 상태에 동일 색/아이콘/모션 규칙을 쓰는가?
- 데이터 로딩/빈 상태/에러 상태 컴포넌트가 분리되어 있는가?
- 필드명이 UI 문구와 1:1 매칭되어 유지보수 가능한가?

---

## 8) 최종 요약
- 지금도 `standings/table`, `team-detail`, `team-schedule`, `state/summary`만으로 상업용에 가까운 1차 완성본이 가능하다.
- 로고/팀 컬러, 고급 레이스 인사이트는 현재 코드 범위를 넘으므로 신규 API 2종을 별도 설계해 붙인다.
- 이 순서대로 가면 "표를 예쁘게 만든 수준"이 아니라, 실제 스포츠 매니지먼트 게임의 핵심 화면 수준으로 발전시킬 수 있다.
