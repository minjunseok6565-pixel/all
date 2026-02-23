# 경기 진행/스탯 누적 SSOT 감사 보고서

## 범위
- 대상: `state.ingest_game_result()` 기반의 경기 결과 반영, 시즌 누적 스탯 집계.
- 가정: 서버 스타트업이 정상 완료되어 `active_season_id`, `master_schedule`이 유효한 상태.

## 결론 요약
- **치명적 SSOT 위반 2건 확인**
  1. 스케줄에 없는 `game_id`도 ingest가 허용되어 누적 스탯/경기 로그가 오염됨.
  2. 동일 `game_id`를 중복 ingest하면 경기 수/누적 스탯이 2배 반영되며, `game_results`는 마지막 1건만 남아 내부 SSOT 불일치 발생.

## 근거 코드
- ingest 시 결과 저장은 무조건 append/덮어쓰기 수행:
  - `container["games"].append(game_obj)`
  - `container["game_results"][game_id] = game_result`
- 스케줄 final 반영 함수는 `game_id` 미존재 시 예외 없이 종료(no-op):
  - `mark_master_schedule_game_final()`에서 `by_id`/`games` 탐색 실패 시 return/raise 없음.

## 재현 (내부 시뮬레이션 3회)

### 시나리오 A: 스케줄 미존재 game_id ingest 허용
- 절차: 서버 초기화와 동일하게 `startup_init_state()` 호출 후, `FAKE-*` game_id ingest.
- 기대: 스케줄 SSOT 기준이면 reject.
- 실제: ingest 성공, `games` 증가, `master_schedule.by_id`에는 해당 game_id 없음.

### 시나리오 B: 동일 game_id 중복 ingest
- 절차: 동일 payload(`game_id=DUP-1`)를 연속 2회 ingest.
- 기대: idempotent 처리 또는 duplicate reject.
- 실제: `games`는 2건, `game_results`는 키 기반이라 1건, 팀 누적 PTS/게임수는 2배.

### 3회 반복 결과
- run1: `games=2`, `game_results=1`, `ATL.games=2`, `ATL.PTS=200.0`
- run2: `games=2`, `game_results=1`, `ATL.games=2`, `ATL.PTS=200.0`
- run3: `games=2`, `game_results=1`, `ATL.games=2`, `ATL.PTS=200.0`

## 영향도
- 리그 진행 무결성: 스케줄과 결과 저장소의 참조 일관성 파괴.
- 누적 통계: 경기 수, 팀/선수 누적치 과대집계 가능.
- 파생 뷰(리더보드/일정 캐시) 신뢰도 저하.

## 권고
1. ingest 전에 `master_schedule.by_id[game_id]` 존재와 팀/날짜 일치 강제 검증.
2. `game_id` 중복 ingest 차단(정상 재처리 필요 시 명시적 idempotency 정책 분리).
3. `mark_master_schedule_game_final()`에서 미존재 game_id를 예외 처리하도록 fail-fast 적용.
