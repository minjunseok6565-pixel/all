# API Endpoint Functional Summary

Source: `server.py` route handlers

- Total GET: 36
- Total POST: 59

| Method | Endpoint | Handler | 기능 요약(한 문장) |
|---|---|---|---|
| GET | `/` | `root` (`server.py:128`) | 간단한 헬스체크 및 NBA.html 링크 안내. |
| POST | `/api/simulate-game` | `api_simulate_game` (`server.py:465`) | matchengine_v3를 사용해 한 경기를 시뮬레이션한다. |
| POST | `/api/advance-league` | `api_advance_league` (`server.py:490`) | target_date까지 (유저 팀 경기를 제외한) 리그 전체 경기를 자동 시뮬레이션. |
| GET | `/api/training/team/{team_id}` | `api_get_team_training_plan` (`server.py:550`) | Get a team training plan (default if missing). |
| POST | `/api/training/team/set` | `api_set_team_training_plan` (`server.py:567`) | Set a team training plan. |
| GET | `/api/training/player/{player_id}` | `api_get_player_training_plan` (`server.py:589`) | Get a player training plan (default if missing). |
| POST | `/api/training/player/set` | `api_set_player_training_plan` (`server.py:611`) | Set a player training plan. |
| GET | `/api/practice/team/{team_id}/plan` | `api_get_team_practice_plan` (`server.py:639`) | Get a team practice plan (default if missing). |
| POST | `/api/practice/team/plan/set` | `api_set_team_practice_plan` (`server.py:656`) | Set a team practice plan. |
| GET | `/api/practice/team/{team_id}/session` | `api_get_team_practice_session` (`server.py:678`) | Get (and auto-resolve) a practice session for a specific date. |
| GET | `/api/practice/team/{team_id}/sessions` | `api_list_team_practice_sessions` (`server.py:751`) | List stored practice sessions (does not auto-generate missing dates). |
| POST | `/api/practice/team/session/set` | `api_set_team_practice_session` (`server.py:779`) | Set a daily practice session (user-authored). |
| GET | `/api/stats/leaders` | `api_stats_leaders` (`server.py:821`) | Regular-season per-game leaders. |
| GET | `/api/stats/playoffs/leaders` | `api_playoff_stats_leaders` (`server.py:850`) | Playoff per-game leaders (same small payload as regular season). |
| GET | `/api/standings` | `api_standings` (`server.py:880`) | 동/서부 컨퍼런스 순위를 계산해 프런트의 순위표 화면 데이터로 반환한다. |
| GET | `/api/teams` | `api_teams` (`server.py:885`) | 리그 팀 카드 목록(요약 정보)을 조회해 팀 선택/목록 UI에 제공한다. |
| GET | `/api/team-detail/{team_id}` | `api_team_detail` (`server.py:890`) | 선택한 팀의 상세 정보(로스터/상태 포함)를 조회해 팀 상세 화면에 제공한다. |
| GET | `/api/college/meta` | `api_college_meta` (`server.py:903`) | 대학 리그 시즌/설정 메타 정보를 조회해 대학 탭 기본 컨텍스트를 구성한다. |
| GET | `/api/college/teams` | `api_college_teams` (`server.py:911`) | 대학 팀 카드 목록을 시즌 기준으로 조회해 대학 팀 리스트 화면에 제공한다. |
| GET | `/api/college/team-detail/{college_team_id}` | `api_college_team_detail` (`server.py:919`) | 특정 대학 팀의 상세 정보를 조회해 대학 팀 상세 화면에 제공한다. |
| GET | `/api/college/players` | `api_college_players` (`server.py:937`) | 필터/정렬 조건으로 대학 선수 목록을 조회해 스카우팅·드래프트 탐색 UI에 제공한다. |
| GET | `/api/college/player/{player_id}` | `api_college_player` (`server.py:971`) | 특정 대학 선수의 상세 정보를 조회해 스카우팅/선수 상세 화면에 제공한다. |
| GET | `/api/college/draft-pool/{draft_year}` | `api_college_draft_pool` (`server.py:989`) | 지정 드래프트 연도의 참가 풀을 조회해 드래프트 준비 화면에 제공한다. |
| POST | `/api/college/draft-watch/recompute` | `api_college_draft_watch_recompute` (`server.py:1011`) | (Dev/Admin) Recompute a pre-declaration watch snapshot for a given draft_year/period. |
| GET | `/api/scouting/scouts/{team_id}` | `api_scouting_list_scouts` (`server.py:1075`) | List scouts for a given team (seeded staff) + current ACTIVE assignment if any. |
| POST | `/api/scouting/assign` | `api_scouting_assign` (`server.py:1151`) | Assign a scout to a college player (user-driven). |
| POST | `/api/scouting/unassign` | `api_scouting_unassign` (`server.py:1285`) | End an ACTIVE scouting assignment (user-driven). |
| GET | `/api/scouting/reports` | `api_scouting_reports` (`server.py:1378`) | List scouting reports for a team (private). Supports filters. |
| GET | `/api/postseason/field` | `api_postseason_field` (`server.py:1509`) | 포스트시즌 진출 팀 필드를 구성해 플레이인·플레이오프 브래킷 초기 화면에 제공한다. |
| GET | `/api/postseason/state` | `api_postseason_state` (`server.py:1514`) | 현재 포스트시즌 진행 상태 스냅샷을 반환해 브래킷/라운드 UI를 동기화한다. |
| POST | `/api/postseason/reset` | `api_postseason_reset` (`server.py:1519`) | 포스트시즌 상태를 초기화해 테스트/재시작 시나리오를 지원한다. |
| POST | `/api/postseason/setup` | `api_postseason_setup` (`server.py:1524`) | 포스트시즌 대진·상태를 초기 세팅해 플레이인/플레이오프를 시작 가능 상태로 만든다. |
| POST | `/api/postseason/play-in/my-team-game` | `api_play_in_my_team_game` (`server.py:1532`) | 내 팀의 플레이인 경기를 1회 진행해 결과와 상태 변화를 반영한다. |
| POST | `/api/postseason/playoffs/advance-my-team-game` | `api_playoffs_advance_my_team_game` (`server.py:1540`) | 내 팀 플레이오프 경기를 한 경기씩 진행해 시리즈 상태를 업데이트한다. |
| POST | `/api/postseason/playoffs/auto-advance-round` | `api_playoffs_auto_advance_round` (`server.py:1548`) | 현재 플레이오프 라운드를 자동 진행해 다음 라운드 진출팀을 확정한다. |
| POST | `/api/season/enter-offseason` | `api_enter_offseason` (`server.py:1561`) | 플레이오프 우승 확정 이후, 다음 시즌으로 전환하고 오프시즌(날짜 구간)으로 진입한다. |
| POST | `/api/offseason/college/finalize` | `api_offseason_college_finalize` (`server.py:1625`) | 대학 시즌 마감(스탯 생성) + 드래프트 선언 생성(SSOT=DB). |
| POST | `/api/offseason/contracts/process` | `api_offseason_contracts_process` (`server.py:1653`) | 오프시즌 계약 처리(만료/옵션/연장/트레이드 정산 등). |
| POST | `/api/offseason/retirement/preview` | `api_offseason_retirement_preview` (`server.py:1741`) | 오프시즌 은퇴 결정 미리보기(확정 전). |
| POST | `/api/offseason/retirement/process` | `api_offseason_retirement_process` (`server.py:1772`) | 오프시즌 은퇴 확정 처리(해당 시즌 1회, idempotent). |
| POST | `/api/offseason/training/apply-growth` | `api_offseason_training_apply_growth` (`server.py:1812`) | 오프시즌 성장/훈련 적용 (Step 2). |
| GET | `/api/agency/player/{player_id}` | `api_agency_get_player` (`server.py:1896`) | Get a player's current agency state + recent events. |
| GET | `/api/agency/team/{team_id}/events` | `api_agency_get_team_events` (`server.py:1925`) | List agency events for a team (UI feed). |
| GET | `/api/agency/events` | `api_agency_get_events` (`server.py:1959`) | List league-wide agency events (debug / commissioner feed). |
| POST | `/api/agency/events/respond` | `api_agency_events_respond` (`server.py:1987`) | Respond to an agency event (user chooses how to handle demands/promises). |
| POST | `/api/agency/actions/apply` | `api_agency_actions_apply` (`server.py:2033`) | User-initiated agency actions (proactive management). |
| POST | `/api/offseason/options/team/pending` | `api_offseason_team_options_pending` (`server.py:2080`) | 유저 팀의 다음 시즌 TEAM 옵션(PENDING) 목록 조회. |
| POST | `/api/offseason/options/team/decide` | `api_offseason_team_options_decide` (`server.py:2116`) | 유저 팀 TEAM 옵션 행사/거절 결정 커밋(DB write). |
| POST | `/api/offseason/draft/lottery` | `api_offseason_draft_lottery` (`server.py:2193`) | 드래프트 1~4픽 로터리(플랜 생성/저장). |
| POST | `/api/offseason/draft/settle` | `api_offseason_draft_settle` (`server.py:2216`) | 픽 정산(보호/스왑) + 최종 지명 턴 생성. |
| POST | `/api/offseason/draft/combine` | `api_offseason_draft_combine` (`server.py:2243`) | 드래프트 컴바인 실행 + 결과 DB 저장. |
| POST | `/api/offseason/draft/workouts` | `api_offseason_draft_workouts` (`server.py:2265`) | 팀 워크아웃 실행 + 결과 DB 저장. |
| GET | `/api/offseason/draft/interviews/questions` | `api_offseason_draft_interview_questions` (`server.py:2318`) | 인터뷰 질문 목록(서버 정의)을 반환한다. (UI 미구현이어도 API만 준비) |
| POST | `/api/offseason/draft/interviews` | `api_offseason_draft_interviews` (`server.py:2330`) | 팀 인터뷰 실행 + 결과 DB 저장. (유저가 선택한 질문 기반) |
| POST | `/api/offseason/draft/withdrawals` | `api_offseason_draft_withdrawals` (`server.py:2392`) | 드래프트 철회(언더클래스만 복귀) 단계 실행 + DB 반영. |
| GET | `/api/offseason/draft/experts` | `api_offseason_draft_experts` (`server.py:2419`) | 드래프트 전문가(외부 빅보드 작성자) 목록. |
| GET | `/api/offseason/draft/bigboard/expert` | `api_offseason_draft_bigboard_expert` (`server.py:2436`) | 특정 전문가의 Big Board 생성(불완전 정보 + 바이어스 + 상단 수렴 앵커링). |
| GET | `/api/offseason/draft/bundle` | `api_offseason_draft_bundle` (`server.py:2515`) | 현재 저장된 플랜(로터리 결과) 기반으로 드래프트 번들(턴/세션/풀) 생성. |
| POST | `/api/offseason/draft/selections/auto` | `api_offseason_draft_selections_auto` (`server.py:2562`) | 저장된 플랜 기반으로 남은 픽을 자동 선택(draft_selections에 기록). |
| POST | `/api/offseason/draft/selections/pick` | `api_offseason_draft_selections_pick` (`server.py:2629`) | 현재 커서(온더클락) 픽을 1개 기록(draft_selections에 저장). |
| POST | `/api/offseason/draft/apply` | `api_offseason_draft_apply` (`server.py:2679`) | draft_selections -> 실제 DB 적용(draft_results/roster/contract/tx), 이후 시즌 전환. |
| POST | `/api/season/start-regular-season` | `api_start_regular_season` (`server.py:2808`) | 오프시즌(또는 임의 시점)에서 정규시즌 시작 직전으로 날짜를 이동한다. |
| POST | `/api/news/week` | `api_news_week` (`server.py:2841`) | 주간 뉴스를 재생성/갱신해 인게임 뉴스 피드에 반영한다. |
| POST | `/api/news/playoffs` | `api_playoff_news` (`server.py:2853`) | 플레이오프 전용 뉴스를 생성해 포스트시즌 뉴스 피드를 갱신한다. |
| POST | `/api/season-report` | `api_season_report` (`server.py:2863`) | 정규 시즌 종료 후, LLM을 이용해 시즌 결산 리포트를 생성한다. |
| POST | `/api/validate-key` | `api_validate_key` (`server.py:2878`) | 주어진 Gemini API 키를 간단히 검증한다. |
| POST | `/api/chat-main` | `chat_main` (`server.py:2897`) | 메인 프롬프트 + 컨텍스트 + 유저 입력을 가지고 Gemini를 호출. |
| POST | `/api/main-llm` | `chat_main_legacy` (`server.py:2922`) | 레거시 메인 LLM 엔드포인트로, 내부적으로 chat_main을 호출해 동일 응답을 반환한다. |
| GET | `/api/contracts/free-agents` | `api_contracts_free_agents` (`server.py:3112`) | List free-agent candidates (players without an active team assignment). |
| POST | `/api/contracts/release-to-fa` | `api_contracts_release_to_fa` (`server.py:3172`) | Release a player to free agency (DB write). |
| POST | `/api/contracts/negotiation/start` | `api_contracts_negotiation_start` (`server.py:3202`) | Start a contract negotiation session (state-backed). |
| POST | `/api/contracts/negotiation/offer` | `api_contracts_negotiation_offer` (`server.py:3230`) | Submit a team offer; player may ACCEPT / COUNTER / REJECT / WALK. |
| POST | `/api/contracts/negotiation/accept-counter` | `api_contracts_negotiation_accept_counter` (`server.py:3256`) | Accept the last counter offer proposed by the player. |
| POST | `/api/contracts/negotiation/commit` | `api_contracts_negotiation_commit` (`server.py:3281`) | Commit an ACCEPTED session (SSOT contract write). |
| POST | `/api/contracts/negotiation/cancel` | `api_contracts_negotiation_cancel` (`server.py:3307`) | Cancel/close a negotiation session (no SSOT DB write). |
| POST | `/api/contracts/sign-free-agent` | `api_contracts_sign_free_agent` (`server.py:3323`) | Sign a free agent (DB write). |
| POST | `/api/contracts/re-sign-or-extend` | `api_contracts_re_sign_or_extend` (`server.py:3352`) | Re-sign / extend a player (DB write). |
| POST | `/api/contracts/two-way/negotiation/start` | `api_two_way_negotiation_start` (`server.py:3381`) | 투웨이 계약 협상 세션을 시작해 제안/결정 단계의 기준 상태를 만든다. |
| POST | `/api/contracts/two-way/negotiation/decision` | `api_two_way_negotiation_decision` (`server.py:3402`) | 투웨이 협상에 대한 선수 측 수락/거절 결정을 처리해 세션 상태를 갱신한다. |
| POST | `/api/contracts/two-way/negotiation/commit` | `api_two_way_negotiation_commit` (`server.py:3417`) | 합의된 투웨이 협상을 DB에 커밋해 계약을 확정 반영한다. |
| POST | `/api/trade/submit` | `api_trade_submit` (`server.py:3437`) | 트레이드 딜을 검증 후 즉시 적용해 로스터/픽/장부를 실제 상태에 반영한다. |
| POST | `/api/trade/submit-committed` | `api_trade_submit_committed` (`server.py:3470`) | 사전 커밋 토큰이 있는 트레이드를 검증·실행하고 실행 완료로 마킹한다. |
| POST | `/api/trade/negotiation/start` | `api_trade_negotiation_start` (`server.py:3504`) | 트레이드 협상 세션을 생성하고 유효기간을 설정해 왕복 제안 플로우를 시작한다. |
| POST | `/api/trade/negotiation/commit` | `api_trade_negotiation_commit` (`server.py:3522`) | 협상 세션의 최종 딜을 검증·적용해 실제 트레이드를 확정한다. |
| POST | `/api/trade/evaluate` | `api_trade_evaluate` (`server.py:3826`) | Debug endpoint: evaluate a proposed deal from a single team's perspective. |
| GET | `/api/roster-summary/{team_id}` | `roster_summary` (`server.py:3875`) | 특정 팀의 로스터를 LLM이 보기 좋은 형태로 요약해서 돌려준다. |
| GET | `/api/two-way/summary/{team_id}` | `two_way_summary` (`server.py:3904`) | 특정 팀의 투웨이 슬롯/출전 가능 경기 수를 요약해서 반환한다. |
| GET | `/api/team-schedule/{team_id}` | `team_schedule` (`server.py:3972`) | 마스터 스케줄 기준으로 특정 팀의 전체 시즌 일정을 반환. |
| GET | `/api/state/summary` | `state_summary` (`server.py:4028`) | 대용량 SSOT 항목을 제외한 워크플로 상태와 DB 스냅샷을 묶어 프런트 상태 동기화용으로 반환한다. |
| POST | `/api/game/new` | `api_game_new` (`server.py:4077`) | 새 세이브 슬롯을 생성해 새 커리어/리그 시작점을 만든다. |
| POST | `/api/game/save` | `api_game_save` (`server.py:4095`) | 현재 진행 상태를 지정 슬롯에 저장해 이후 이어하기가 가능하도록 한다. |
| GET | `/api/game/saves` | `api_game_saves` (`server.py:4105`) | 사용 가능한 저장 슬롯 목록을 조회해 로드/저장 화면에 제공한다. |
| GET | `/api/game/saves/{slot_id}` | `api_game_save_detail` (`server.py:4115`) | 특정 저장 슬롯의 상세 메타를 조회해 로드 전 검토 정보를 제공한다. |
| POST | `/api/game/load` | `api_game_load` (`server.py:4127`) | 지정 저장 슬롯을 로드해 서버의 현재 게임 상태를 해당 세이브로 전환한다. |
| GET | `/api/debug/schedule-summary` | `debug_schedule_summary` (`server.py:4143`) | 마스터 스케줄 생성/검증용 디버그 엔드포인트. |
