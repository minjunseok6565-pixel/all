# Schedule 탭 고도화 기획안 (상업용 게임 품질 목표)

## 1) 목적/범위
- 목적: 현재 `schedule-screen`을 운영 툴형 테이블 UI에서 **상업용 스포츠 게임 UI**로 고도화한다.
- 범위: 다음 구현 작업에서 **`static/NBA.html`, `static/NBA.css`, `static/NBA.js` 내 스케쥴 탭 관련 영역만 수정**한다.
- 비범위:
  - 다른 탭(내 팀/전술/훈련/순위/대학/메디컬) UI 구조 변경 금지
  - 신규 API 추가 금지 (기존 `/api/team-schedule/{teamId}` 및 이미 페이지에서 사용하는 조회 API만 사용)

---

## 2) 현재 구조 기반 제약 정리

### 2-1. HTML 구조(현행)
- `#schedule-screen` 내부는 크게 아래 두 카드 구조:
  - 완료 경기 카드 (`#schedule-completed-body`)
  - 예정 경기 카드 (`#schedule-upcoming-body`)
- 상단 헤더는 `schedule-title`, `schedule-back-btn` 중심의 단순 정보형.

### 2-2. JS 데이터 바인딩(현행)
- 스케쥴 렌더 함수: `renderScheduleTables(games)`
- 완료/예정 분기: `g.is_completed`
- 주요 사용 필드:
  - 공통: `date_mmdd`, `opponent_label`, `opponent_team_name`, `tipoff_time`
  - 완료 경기: `result.display`, `result.wl`, `record_after_game.display`, `leaders.points/rebounds/assists`
- 스케쥴 탭 진입: `showScheduleScreen()`에서 `/api/team-schedule/{teamId}` 조회 후 제목/테이블 렌더.

### 2-3. CSS 특성(현행)
- `#schedule-screen`은 현재 라이트 테마 기반 카드+테이블 구조.
- 결과/시간 배지(`.schedule-result-badge`, `.schedule-time-chip`)는 존재하나 시각 계층이 약함.

---

## 3) 목표 UX 원칙
1. **First Focus 원칙**: 탭 진입 즉시 “다음 경기” 행동을 유도한다.
2. **정보 위계 원칙**: 핵심(다음 경기/결과 상태)과 보조(세부기록)를 시각적으로 분리한다.
3. **게임 브랜딩 원칙**: 팀 운영 콘솔 느낌(긴장감+프리미엄)을 유지한다.
4. **재사용/확장 원칙**: 스케쥴 컴포넌트 규칙을 만들어 다음 시즌/모바일에서도 유지 가능하게 한다.

---

## 4) 정보 구조(IA) 제안

### 상단 A영역: Next Game Hero
- 탭 헤더 하단에 신규 Hero 블록 추가.
- 표시 요소:
  - 대진: `@ NYK` / `vs MIA` 형태
  - 상대 팀명
  - 경기일시 (`date_mmdd + tipoff_time`)
  - 상태 배지: `UPCOMING`, `HOME`, `AWAY`
  - 서브 카피: "다음 경기 준비를 완료하세요"
- 액션(버튼):
  - `경기 진행`
  - `전술로 이동`
  - `빠른 진행`
- 데이터 소스: `games`에서 `!is_completed` 중 가장 빠른 경기.

### 중단 B영역: Schedule Control Bar
- 필터/정렬 바(신규):
  - 세그먼트: 전체 / 완료 / 예정
  - 조건: 홈만 / 원정만
  - 기간: 이번 주 / 이번 달 / 전체
- 정렬 기준:
  - 기본: 날짜 오름차순
  - 옵션: 난이도(placeholder 제거, 실제 적용은 향후)

### 하단 C영역: 게임 리스트 2단
- 좌측(넓게): 예정 경기 리스트 (카드형 행)
- 우측(좁게): 완료 경기 요약 카드 또는 최근 완료 경기 5개
- 데스크탑에서는 2단, 작은 해상도에서는 1단 스택.

### 보조 D영역: Schedule Insights (선택)
- 기존 데이터 가공만으로 계산 가능한 지표:
  - 월별 경기 수
  - 홈/원정 비율
  - 백투백 횟수(날짜 간격 1일)
  - 연속 원정 구간 길이

---

## 5) 컴포넌트 스펙 (디자인 시스템 수준)

### 5-1. 카드 계층
- `surface-1`: 전체 배경 카드
- `surface-2`: 리스트 행 카드
- `surface-3`: 선택/호버 카드
- 각 레벨별 그림자/보더/채도 차이를 토큰화.

### 5-2. 배지 시스템
- 상태 배지: `W`, `L`, `UPCOMING`, `FINAL`
- 위치 배지: `HOME`, `AWAY`
- 강조 배지: `B2B`, `RIVAL` (데이터 없으면 숨김)
- 규칙:
  - 텍스트 길이 변화에도 높이 고정
  - 색상만으로 의미 전달하지 않고 라벨 텍스트 포함

### 5-3. 타이포 스케일
- H1(탭 제목), H2(섹션), Body, Meta, Label 5단계 제한
- 숫자(시간/점수/기록)는 tabular 숫자 스타일 적용

### 5-4. 인터랙션
- Hover: 배경 1단 상승 + 보더 강조
- Active/Selected: 좌측 포인트 바 + 미세 확대(1.01)
- Transition: 120~180ms 범위 통일

---

## 6) 데이터 매핑 정의 (기존 API만 사용)

### 6-1. Row ViewModel
- 공통 파생값:
  - `gameType`: `completed | upcoming`
  - `venueType`: `home | away` (`opponent_label`의 `vs`/`@`로 파생)
  - `displayDate`: `date_mmdd`
  - `displayOpponent`: `opponent_team_name`
- 완료 경기 파생값:
  - `resultText`: `result.display`
  - `resultWL`: `result.wl`
  - `recordAfter`: `record_after_game.display`
  - `leaderPts/Reb/Ast`: `leaders.*`
- 예정 경기 파생값:
  - `tipoffText`: `tipoff_time`

### 6-2. Insight 계산
- `totalCompleted`, `totalUpcoming`
- `homeCount`, `awayCount`
- `b2bCount`: 인접 경기 날짜 차가 1일인 횟수
- `longestRoadTrip`: 연속 원정 경기 최대 길이

### 6-3. 빈 상태 규칙
- 완료 0건: 시즌 시작 전 안내 카피 + 전술 이동 CTA
- 예정 0건: "전체 일정 종료" 안내 + 시즌 보고서 이동 CTA(버튼 비활성 허용)

---

## 7) 구현 단위(다음 작업용) — HTML/CSS/JS 분할

### 7-1. HTML 작업
1. `#schedule-screen` 내 Hero 컨테이너 추가
2. 필터 바/인사이트 블록 마크업 추가
3. 기존 테이블은 단계적 유지:
   - 1차: 테이블 유지 + 상단 Hero/필터만 추가
   - 2차: 카드형 리스트로 전환(필요 시 테이블 제거)

### 7-2. CSS 작업
1. 스케쥴 전용 토큰(`--schedule-*`) 선언
2. Hero/필터/리스트 카드 스타일 추가
3. 반응형 분기:
   - >=1440: 2열
   - 1024~1439: 1열 + 인사이트 접기
   - <=768: 필터를 수평 스크롤 칩으로 전환
4. 접근성:
   - 명도 대비 4.5:1 이상
   - `prefers-reduced-motion` 대응

### 7-3. JS 작업
1. 기존 `renderScheduleTables`를 확장 또는 분리:
   - `buildScheduleViewModel(games)`
   - `renderScheduleHero(vm)`
   - `renderScheduleControls(vm)`
   - `renderScheduleLists(vm)`
   - `renderScheduleInsights(vm)`
2. 필터 상태 저장:
   - `state.scheduleFilter` 신설 (`segment`, `venue`, `period`)
3. 이벤트 핸들러:
   - 필터 클릭 시 재렌더(재조회 없이 클라이언트 파생)

---

## 8) 품질 기준 (상업용 품질 체크리스트)

### Visual Polish
- 간격 오차 2px 이상 금지 (8px 그리드)
- 라운드/보더/그림자 규칙 통일
- 한 화면 내 블루톤/레드톤 과다 혼용 금지 (포인트 색 1개 원칙)

### UX 품질
- 탭 진입 후 3초 안에 "다음 행동" 인지 가능
- 경기 상태(W/L/예정)가 스캔 1초 내 판별 가능
- 빈 상태/로딩/에러 상태 디자인 각각 별도 정의

### 코드 품질
- 스케쥴 전용 클래스 네이밍 prefix 유지(`schedule-*`)
- 기존 다른 탭 스타일에 영향 없도록 `#schedule-screen` 스코프 우선
- 렌더 함수 단일 책임 유지(가공/렌더/이벤트 분리)

---

## 9) 단계별 로드맵

### Phase 1 (핵심 체감)
- Next Game Hero
- 필터 바(전체/완료/예정)
- 결과 배지 및 행 위계 개선

### Phase 2 (완성도)
- 홈/원정/기간 필터
- 인사이트 패널
- 빈 상태/로딩 상태 고도화

### Phase 3 (출시 직전 폴리시)
- 미세 모션/키보드 접근성
- 반응형 최적화
- 문구 톤 일관화(한국어/영문 표기 룰)

---

## 10) 구현 시 주의사항 (필수)
- 스케쥴 탭 외 DOM/CSS/JS 동작 변경 금지
- 전역 공용 클래스 수정 시 반드시 `#schedule-screen` 범위 제한
- 기존 API 필드 없는 정보는 절대 하드코딩으로 가장하지 않기
- 데이터 누락 시 반드시 graceful fallback 제공 (`-`, `예정 없음`, `기록 없음`)

---

## 11) 인수 기준(Definition of Done)
1. 스케쥴 탭 진입 시 Hero/필터/리스트/인사이트가 일관된 시각 체계로 렌더된다.
2. 완료/예정/홈/원정/기간 필터가 정상 작동한다.
3. 데이터 0건/부분 누락/로딩 실패 시 UI가 깨지지 않는다.
4. 다른 탭 화면의 레이아웃/스타일/스크립트 동작에 회귀가 없다.
5. 1440/1024/768 해상도에서 레이아웃 붕괴가 없다.

