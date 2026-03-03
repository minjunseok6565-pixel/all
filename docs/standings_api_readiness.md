# Standings UI API Readiness

순위표 UI 구현을 바로 시작할 수 있도록, 실제 구현된 조회 API 계약을 정리한다.

## 1) 신규 구현 API

## 1.1 GET `/api/teams/branding`
- 목적: 팀 브랜딩 메타(표시명/약어/컬러/로고 URL 슬롯) 제공
- 쿼리 파라미터: 없음
- 응답 필드:
  - `teams[]`
    - `team_id: string`
    - `display_name: string`
    - `short_name: string`
    - `logo_url: string | null` (현재 `null` 반환)
    - `colors.primary: string(hex)`
    - `colors.secondary: string(hex)`
    - `colors.text_on_primary: string(hex)`
  - `updated_at: string(YYYY-MM-DD)`

## 1.2 GET `/api/standings/race-insights?conference=East&team_id=BOS`
- 목적: 순위 경쟁 인사이트(컷라인 격차/인접 팀/맞대결 전적) 제공
- 쿼리 파라미터:
  - `conference: East | West` (필수)
  - `team_id: string` (필수)
- 응답 필드:
  - `conference: East | West`
  - `team_id: string`
  - `as_of_date: string(YYYY-MM-DD)`
  - `current_rank: int`
  - `cutlines.playoff_direct_rank: int` (6)
  - `cutlines.playin_rank: int` (10)
  - `cutlines.gb_to_direct_cut: float`
  - `cutlines.gb_to_playin_cut: float`
  - `neighbors.above.team_id | rank | gb_gap`
  - `neighbors.below.team_id | rank | gb_gap`
  - `tiebreaker_context.vs_above_head_to_head: string | null` (`W-L`)
  - `tiebreaker_context.vs_below_head_to_head: string | null` (`W-L`)

---

## 2) UI 연동 체크리스트
- [ ] 순위 행 hover/선택 시 `team_id`를 race-insights 조회 파라미터로 전달
- [ ] 동/서부 토글 상태를 `conference` 쿼리로 전달
- [ ] 팀 셀 브랜딩 배경/보더는 `teams/branding.colors.*` 사용
- [ ] 로고 에셋 준비 전까지 `logo_url == null`이면 약어 모노그램 fallback 렌더

---

## 3) 기존 API와의 역할 분담
- `GET /api/standings/table`: 메인 테이블 원천 데이터
- `GET /api/standings/race-insights`: 설명형 인사이트 패널 데이터
- `GET /api/teams/branding`: 시각 브랜딩 메타
- `GET /api/team-detail/{team_id}`: 내 팀 KPI 보강 데이터
- `GET /api/team-schedule/{team_id}`: 다음 경기/최근 추세 데이터
- `GET /api/state/summary`: 현재 인게임 날짜
