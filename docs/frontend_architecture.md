# NBA 시뮬 GM 프론트엔드 설계안 (완성형 v2)

## 핵심 설계
- **게임 런처 우선 진입**: 새 게임 생성/저장 로드가 첫 화면에서 동작.
- **세션 중심 UX**: `slot_id`, `user_team_id`, 시즌/날짜를 전역 세션으로 유지.
- **실행 가능한 전체 API 접근성**: 일반 사용용 화면 + 범용 `API 스튜디오`를 통해 프로젝트의 GET/POST 엔드포인트를 전부 호출 가능.

## 화면 모듈
1. 런처: `/api/game/new`, `/api/game/load`, `/api/game/saves`, `/api/game/save`
2. 대시보드: `/api/state/summary`, `/api/teams`, `/api/team-detail/{team_id}`
3. 시즌 운영: `/api/advance-league`, `/api/team-schedule/{team_id}`, 순위/리더 API
4. 로스터/팀: `/api/roster-summary/{team_id}`, `/api/college/players`
5. 계약/트레이드: `/api/contracts/*`, `/api/trade/*`
6. 오프시즌: `/api/offseason/*`, `/api/season/*`
7. AI 어시스턴트: `/api/validate-key`, `/api/chat-main`
8. API 스튜디오: `/openapi.json` 기반 전체 GET/POST 동적 실행기

## 구현 포인트
- **정적 파일 분리**: `NBA.html`(구조), `style.css`(스타일), `app.js`(동작)
- **OpenAPI 동적 바인딩**: 경로 파라미터/쿼리/바디를 템플릿으로 생성해 즉시 호출 가능
- **로컬 API 키 보관**: `localStorage['nba_sim_api_key']`
- **게임 느낌 강화**: 사이드바 내비게이션 + GM 콘솔 스타일 다크 테마

## 확장 전략
- 모듈별 JS 파일 분리(`modules/launcher.js`, `modules/offseason.js` 등)
- API 스튜디오 템플릿 저장(엔드포인트별 샘플 payload)
- 사용자 팀 기반 자동 payload 보정(예: `user_team_id` 자동 주입)
