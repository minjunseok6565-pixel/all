const TEAM_FULL_NAMES = {
  ATL: "애틀랜타 호크스", BOS: "보스턴 셀틱스", BKN: "브루클린 네츠", CHA: "샬럿 호네츠",
  CHI: "시카고 불스", CLE: "클리블랜드 캐벌리어스", DAL: "댈러스 매버릭스", DEN: "덴버 너기츠",
  DET: "디트로이트 피스톤스", GSW: "골든 스테이트 워리어스", HOU: "휴스턴 로키츠", IND: "인디애나 페이서스",
  LAC: "LA 클리퍼스", LAL: "LA 레이커스", MEM: "멤피스 그리즐리스", MIA: "마이애미 히트",
  MIL: "밀워키 벅스", MIN: "미네소타 팀버울브스", NOP: "뉴올리언스 펠리컨스", NYK: "뉴욕 닉스",
  OKC: "오클라호마시티 썬더", ORL: "올랜도 매직", PHI: "필라델피아 세븐티식서스", PHX: "피닉스 선즈",
  POR: "포틀랜드 트레일블레이저스", SAC: "새크라멘토 킹스", SAS: "샌안토니오 스퍼스", TOR: "토론토 랩터스",
  UTA: "유타 재즈", WAS: "워싱턴 위저즈"
};

const TACTICS_OFFENSE_SCHEMES = [
  { key: "Spread_HeavyPnR", label: "헤비 PnR", description: "주볼핸들러 중심 2:2 빈도를 높입니다." },
  { key: "Drive_Kick", label: "드라이브 앤 킥", description: "돌파 후 외곽 킥아웃으로 찬스를 만듭니다." },
  { key: "FiveOut", label: "파이브 아웃", description: "5-Out 간격으로 페인트 존을 넓게 사용합니다." },
  { key: "Motion_SplitCut", label: "모션 스플릿", description: "오프볼 움직임과 컷 위주로 전개합니다." },
  { key: "DHO_Chicago", label: "DHO 시카고", description: "핸드오프 + 스크린 조합으로 슈터를 살립니다." },
  { key: "Post_InsideOut", label: "포스트 인사이드-아웃", description: "포스트 터치 이후 외곽으로 전개합니다." },
  { key: "Horns_Elbow", label: "혼즈 엘보우", description: "하이포스트 2빅 셋에서 의사결정합니다." },
  { key: "Transition_Early", label: "얼리 트랜지션", description: "세트 오펜스 전 빠른 찬스 창출을 노립니다." }
];

const TACTICS_DEFENSE_SCHEMES = [
  { key: "Drop", label: "드롭", description: "빅맨을 림 근처에 두고 드라이브를 억제합니다." },
  { key: "Switch_Everything", label: "올 스위치", description: "스크린 상황에서 전원 스위칭합니다." },
  { key: "Switch_1_4", label: "1-4 스위치", description: "가드-윙 구간 중심으로 스위칭합니다." },
  { key: "Hedge_ShowRecover", label: "헤지 & 리커버", description: "빅맨 쇼업 후 원위치 복귀를 우선합니다." },
  { key: "Blitz_TrapPnR", label: "블리츠 트랩", description: "PnR 볼핸들러에 강한 더블팀을 가합니다." },
  { key: "AtTheLevel", label: "앳 더 레벨", description: "빅맨이 스크린 레벨까지 올라와 저지합니다." },
  { key: "Zone", label: "존", description: "지역 방어로 드라이브 각도와 패싱 레인을 관리합니다." }
];

const TACTICS_OFFENSE_ROLES = [
  "Engine_Primary", "Engine_Secondary", "Transition_Engine", "Shot_Creator", "Rim_Pressure",
  "SpotUp_Spacer", "Movement_Shooter", "Cutter_Finisher", "Connector",
  "Roll_Man", "ShortRoll_Hub", "Pop_Threat", "Post_Anchor"
];

const TACTICS_DEFENSE_ROLE_BY_SCHEME = {
  Drop: ["PnR_POA_Defender", "PnR_Cover_Big_Drop", "Lowman_Helper", "Nail_Helper", "Weakside_Rotator"],
  Switch_Everything: ["PnR_POA_Switch", "PnR_Cover_Big_Switch", "Switch_Wing_Strong", "Switch_Wing_Weak", "Backline_Anchor"],
  Switch_1_4: ["PnR_POA_Switch_1_4", "PnR_Cover_Big_Switch_1_4", "Switch_Wing_Strong_1_4", "Switch_Wing_Weak_1_4", "Backline_Anchor"],
  Hedge_ShowRecover: ["PnR_POA_Defender", "PnR_Cover_Big_HedgeRecover", "Lowman_Helper", "Nail_Helper", "Weakside_Rotator"],
  Blitz_TrapPnR: ["PnR_POA_Blitz", "PnR_Cover_Big_Blitz", "Lowman_Helper", "Nail_Helper", "Weakside_Rotator"],
  AtTheLevel: ["PnR_POA_AtTheLevel", "PnR_Cover_Big_AtTheLevel", "Lowman_Helper", "Nail_Helper", "Weakside_Rotator"],
  Zone: ["Zone_Top_Left", "Zone_Top_Right", "Zone_Bottom_Left", "Zone_Bottom_Right", "Zone_Bottom_Center"]
};

const TACTICS_ROLE_LABELS = {
  Engine_Primary: "1차 볼핸들러",
  Engine_Secondary: "2차 볼핸들러",
  Transition_Engine: "트랜지션 전개",
  Shot_Creator: "크리에이터",
  Rim_Pressure: "림 압박",
  SpotUp_Spacer: "스팟업 스페이서",
  Movement_Shooter: "무브먼트 슈터",
  Cutter_Finisher: "커터/피니셔",
  Connector: "커넥터",
  Roll_Man: "롤맨",
  ShortRoll_Hub: "쇼트롤 허브",
  Pop_Threat: "팝 위협",
  Post_Anchor: "포스트 앵커",
  PnR_POA_Defender: "POA 디펜더",
  PnR_Cover_Big_Drop: "드롭 커버 빅",
  Lowman_Helper: "로우맨 헬퍼",
  Nail_Helper: "네일 헬퍼",
  Weakside_Rotator: "약측 로테이터",
  PnR_POA_Switch: "POA 스위치",
  PnR_Cover_Big_Switch: "빅 스위치",
  Switch_Wing_Strong: "윙 스위치(강)",
  Switch_Wing_Weak: "윙 스위치(약)",
  Backline_Anchor: "백라인 앵커",
  PnR_POA_Switch_1_4: "POA 스위치 1-4",
  PnR_Cover_Big_Switch_1_4: "빅 스위치 1-4",
  Switch_Wing_Strong_1_4: "윙 스위치 강 1-4",
  Switch_Wing_Weak_1_4: "윙 스위치 약 1-4",
  PnR_Cover_Big_HedgeRecover: "헤지 리커버 빅",
  PnR_POA_Blitz: "POA 블리츠",
  PnR_Cover_Big_Blitz: "빅 블리츠",
  PnR_POA_AtTheLevel: "POA 앳더레벨",
  PnR_Cover_Big_AtTheLevel: "빅 앳더레벨",
  Zone_Top_Left: "존 탑 좌",
  Zone_Top_Right: "존 탑 우",
  Zone_Bottom_Left: "존 바텀 좌",
  Zone_Bottom_Right: "존 바텀 우",
  Zone_Bottom_Center: "존 바텀 중앙",
};

const state = {
  lastSaveSlotId: null,
  selectedTeamId: null,
  selectedTeamName: "",
  currentDate: "",
  rosterRows: [],
  selectedPlayerId: null,
  trainingSelectedDates: new Set(),
  trainingCalendarDays: [],
  trainingSessionsByDate: {},
  trainingRoster: [],
  trainingFamiliarity: { offense: [], defense: [] },
  trainingDraftSession: null,
  standingsData: null,
  standingsView: { conference: "east", sortKey: "pct", sortDir: "desc", showAdvanced: false },
  tacticsDraft: null,
  tacticsSavedSnapshot: "",
  tacticsDirty: false,
  medicalOverview: null,
  medicalSelectedPlayerId: null,
  scheduleFilter: { segment: "all", venue: "all" },
  scheduleGames: [],
};

const els = {
  startScreen: document.getElementById("start-screen"),
  teamScreen: document.getElementById("team-screen"),
  mainScreen: document.getElementById("main-screen"),
  scheduleScreen: document.getElementById("schedule-screen"),
  myTeamScreen: document.getElementById("my-team-screen"),
  playerDetailScreen: document.getElementById("player-detail-screen"),
  newGameBtn: document.getElementById("new-game-btn"),
  continueBtn: document.getElementById("continue-btn"),
  continueHint: document.getElementById("continue-hint"),
  teamGrid: document.getElementById("team-grid"),
  mainTeamTitle: document.getElementById("main-team-title"),
  mainCurrentDate: document.getElementById("main-current-date"),
  teamAName: document.getElementById("team-a-name"),
  teamBName: document.getElementById("team-b-name"),
  nextGameDatetime: document.getElementById("next-game-datetime"),
  myTeamTitle: document.getElementById("my-team-title"),
  myTeamBtn: document.getElementById("my-team-btn"),
  tacticsMenuBtn: document.getElementById("tactics-menu-btn"),
  nextGameTacticsBtn: document.getElementById("next-game-tactics-btn"),
  scheduleBtn: document.getElementById("schedule-btn"),
  scheduleBackBtn: document.getElementById("schedule-back-btn"),
  scheduleTitle: document.getElementById("schedule-title"),
  scheduleCompletedBody: document.getElementById("schedule-completed-body"),
  scheduleUpcomingBody: document.getElementById("schedule-upcoming-body"),
  scheduleHeroMatchup: document.getElementById("schedule-hero-matchup"),
  scheduleHeroDatetime: document.getElementById("schedule-hero-datetime"),
  scheduleHeroStatus: document.getElementById("schedule-hero-status"),
  scheduleHeroVenue: document.getElementById("schedule-hero-venue"),
  scheduleHeroTacticsBtn: document.getElementById("schedule-hero-tactics-btn"),
  scheduleInsightCompleted: document.getElementById("schedule-insight-completed"),
  scheduleInsightUpcoming: document.getElementById("schedule-insight-upcoming"),
  scheduleInsightHomeAway: document.getElementById("schedule-insight-home-away"),
  scheduleInsightB2b: document.getElementById("schedule-insight-b2b"),
  trainingMenuBtn: document.getElementById("training-menu-btn"),
  tacticsScreen: document.getElementById("tactics-screen"),
  tacticsBackBtn: document.getElementById("tactics-back-btn"),
  tacticsOffenseBtn: document.getElementById("tactics-offense-btn"),
  tacticsDefenseBtn: document.getElementById("tactics-defense-btn"),
  tacticsOffenseOptions: document.getElementById("tactics-offense-options"),
  tacticsDefenseOptions: document.getElementById("tactics-defense-options"),
  tacticsOffenseCurrent: document.getElementById("tactics-offense-current"),
  tacticsDefenseCurrent: document.getElementById("tactics-defense-current"),
  tacticsTeamTitle: document.getElementById("tactics-team-title"),
  tacticsCurrentDate: document.getElementById("tactics-current-date"),
  tacticsStatusPill: document.getElementById("tactics-status-pill"),
  tacticsSaveBtn: document.getElementById("tactics-save-btn"),
  tacticsAutobalanceBtn: document.getElementById("tactics-autobalance-btn"),
  tacticsMinutesTotal: document.getElementById("tactics-minutes-total"),
  tacticsMinutesDelta: document.getElementById("tactics-minutes-delta"),
  tacticsMinutesSplit: document.getElementById("tactics-minutes-split"),
  tacticsStarterMinutes: document.getElementById("tactics-starter-minutes"),
  tacticsOffRoleCounts: document.getElementById("tactics-off-role-counts"),
  tacticsDefRoleCounts: document.getElementById("tactics-def-role-counts"),
  tacticsRiskList: document.getElementById("tactics-risk-list"),
  tacticsInsightCopy: document.getElementById("tactics-insight-copy"),
  tacticsStarters: document.getElementById("tactics-starters"),
  tacticsRotation: document.getElementById("tactics-rotation"),
  tacticsRosterList: document.getElementById("tactics-roster-list"),
  standingsMenuBtn: document.getElementById("standings-menu-btn"),
  trainingScreen: document.getElementById("training-screen"),
  standingsScreen: document.getElementById("standings-screen"),
  collegeScreen: document.getElementById("college-screen"),
  medicalScreen: document.getElementById("medical-screen"),
  trainingBackBtn: document.getElementById("training-back-btn"),
  standingsBackBtn: document.getElementById("standings-back-btn"),
  collegeMenuBtn: document.getElementById("college-menu-btn"),
  medicalMenuBtn: document.getElementById("medical-menu-btn"),
  medicalBackBtn: document.getElementById("medical-back-btn"),
  collegeBackBtn: document.getElementById("college-back-btn"),
  collegeMetaLine: document.getElementById("college-meta-line"),
  collegeTabTeams: document.getElementById("college-tab-teams"),
  collegeTabLeaders: document.getElementById("college-tab-leaders"),
  collegeTabBigboard: document.getElementById("college-tab-bigboard"),
  collegeTabScouting: document.getElementById("college-tab-scouting"),
  collegePanelTeams: document.getElementById("college-panel-teams"),
  collegePanelLeaders: document.getElementById("college-panel-leaders"),
  collegePanelBigboard: document.getElementById("college-panel-bigboard"),
  collegePanelScouting: document.getElementById("college-panel-scouting"),
  collegeTeamsBody: document.getElementById("college-teams-body"),
  collegeRosterTitle: document.getElementById("college-roster-title"),
  collegeRosterBody: document.getElementById("college-roster-body"),
  collegeLeaderSort: document.getElementById("college-leader-sort"),
  collegeLeadersBody: document.getElementById("college-leaders-body"),
  collegeExpertSelect: document.getElementById("college-expert-select"),
  collegeBigboardBody: document.getElementById("college-bigboard-body"),
  collegeScoutSelect: document.getElementById("college-scout-select"),
  collegeScoutPlayerSelect: document.getElementById("college-scout-player-select"),
  collegeAssignBtn: document.getElementById("college-assign-btn"),
  collegeUnassignBtn: document.getElementById("college-unassign-btn"),
  collegeReportsBody: document.getElementById("college-reports-body"),
  teamTrainingTabBtn: document.getElementById("team-training-tab-btn"),
  playerTrainingTabBtn: document.getElementById("player-training-tab-btn"),
  trainingCalendarGrid: document.getElementById("training-calendar-grid"),
  trainingTypeButtons: document.getElementById("training-type-buttons"),
  trainingDetailPanel: document.getElementById("training-detail-panel"),
  standingsBody: document.getElementById("standings-body"),
  standingsCardTitle: document.getElementById("standings-card-title"),
  standingsTable: document.getElementById("standings-table"),
  standingsConferenceToggle: document.getElementById("standings-conference-toggle"),
  standingsSortKey: document.getElementById("standings-sort-key"),
  standingsSortDirBtn: document.getElementById("standings-sort-dir-btn"),
  standingsAdvancedToggle: document.getElementById("standings-advanced-toggle"),
  standingsSummaryOffenseTeam: document.getElementById("standings-summary-offense-team"),
  standingsSummaryOffenseValue: document.getElementById("standings-summary-offense-value"),
  standingsSummaryDefenseTeam: document.getElementById("standings-summary-defense-team"),
  standingsSummaryDefenseValue: document.getElementById("standings-summary-defense-value"),
  standingsSummaryDiffTeam: document.getElementById("standings-summary-diff-team"),
  standingsSummaryDiffValue: document.getElementById("standings-summary-diff-value"),
  standingsSummaryRaceTeam: document.getElementById("standings-summary-race-team"),
  standingsSummaryRaceValue: document.getElementById("standings-summary-race-value"),
  backToMainBtn: document.getElementById("back-to-main-btn"),
  backToRosterBtn: document.getElementById("back-to-roster-btn"),
  rosterBody: document.getElementById("my-team-roster-body"),
  playerDetailTitle: document.getElementById("player-detail-title"),
  playerDetailPanel: document.getElementById("player-detail-panel"),
  playerDetailContent: document.getElementById("player-detail-content"),
  medicalTitle: document.getElementById("medical-title"),
  medicalAsOf: document.getElementById("medical-as-of"),
  medicalRosterCount: document.getElementById("medical-roster-count"),
  medicalOutCount: document.getElementById("medical-out-count"),
  medicalReturningCount: document.getElementById("medical-returning-count"),
  medicalHighRiskCount: document.getElementById("medical-high-risk-count"),
  medicalHealthFrustrationCount: document.getElementById("medical-health-frustration-count"),
  medicalRiskBody: document.getElementById("medical-risk-body"),
  medicalInjuredBody: document.getElementById("medical-injured-body"),
  medicalHealthBody: document.getElementById("medical-health-body"),
  medicalTimelineTitle: document.getElementById("medical-timeline-title"),
  medicalTimelineList: document.getElementById("medical-timeline-list"),
  medicalAlertBar: document.getElementById("medical-alert-bar"),
  medicalAlertText: document.getElementById("medical-alert-text"),
  medicalAlertMeta: document.getElementById("medical-alert-meta"),
  medicalAlertLevel: document.getElementById("medical-alert-level"),
  medicalAlertOpenPlayer: document.getElementById("medical-alert-open-player"),
  medicalAlertOpenAction: document.getElementById("medical-alert-open-action"),
  medicalRosterDelta: document.getElementById("medical-roster-delta"),
  medicalOutDelta: document.getElementById("medical-out-delta"),
  medicalHighRiskDelta: document.getElementById("medical-high-risk-delta"),
  medicalHealthDelta: document.getElementById("medical-health-delta"),
  medicalRiskCalendarList: document.getElementById("medical-risk-calendar-list"),
  medicalActionList: document.getElementById("medical-action-list"),
  loadingOverlay: document.getElementById("loading-overlay"),
  loadingText: document.getElementById("loading-text")
};

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `요청 실패: ${url}`);
  return data;
}

function setLoading(show, msg = "") {
  els.loadingOverlay.classList.toggle("hidden", !show);
  if (msg) els.loadingText.textContent = msg;
}

function activateScreen(target) {
  [
    els.startScreen,
    els.teamScreen,
    els.mainScreen,
    els.scheduleScreen,
    els.myTeamScreen,
    els.playerDetailScreen,
    els.tacticsScreen,
    els.trainingScreen,
    els.standingsScreen,
    els.collegeScreen,
    els.medicalScreen,
  ].forEach((screen) => {
    const active = screen === target;
    screen.classList.toggle("active", active);
    screen.setAttribute("aria-hidden", active ? "false" : "true");
  });
}

function renderCollegeEmpty(tbody, colspan, msg) {
  tbody.innerHTML = `<tr><td class="schedule-empty" colspan="${colspan}">${msg}</td></tr>`;
}

function collegeStat(player, key) {
  const stats = player?.stats || {};
  const n = Number(stats?.[key]);
  return Number.isFinite(n) ? n : 0;
}

function switchCollegeTab(tab) {
  const mapping = {
    teams: [els.collegeTabTeams, els.collegePanelTeams],
    leaders: [els.collegeTabLeaders, els.collegePanelLeaders],
    bigboard: [els.collegeTabBigboard, els.collegePanelBigboard],
    scouting: [els.collegeTabScouting, els.collegePanelScouting],
  };
  Object.values(mapping).forEach(([btn, panel]) => {
    const active = btn === mapping[tab][0];
    btn.classList.toggle("is-active", active);
    panel.classList.toggle("active", active);
    panel.setAttribute("aria-hidden", active ? "false" : "true");
  });
}

function renderCollegeTeams(teams) {
  if (!teams.length) {
    renderCollegeEmpty(els.collegeTeamsBody, 6, "대학 팀 데이터가 없습니다.");
    return;
  }
  const sorted = [...teams].sort((a, b) => {
    const wa = Number(a?.wins ?? -9999);
    const wb = Number(b?.wins ?? -9999);
    if (wb !== wa) return wb - wa;
    const la = Number(a?.losses ?? 9999);
    const lb = Number(b?.losses ?? 9999);
    if (la !== lb) return la - lb;
    return Number(b?.srs ?? -9999) - Number(a?.srs ?? -9999);
  });
  els.collegeTeamsBody.innerHTML = "";
  sorted.forEach((team, idx) => {
    const tr = document.createElement("tr");
    tr.className = "roster-row";
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td class="standings-team-cell">${team?.name || team?.college_team_id || "-"}</td>
      <td>${team?.conference || "-"}</td>
      <td>${team?.wins ?? "-"}</td>
      <td>${team?.losses ?? "-"}</td>
      <td>${Number(team?.srs ?? 0).toFixed(2)}</td>
    `;
    tr.addEventListener("click", () => loadCollegeTeamDetail(team?.college_team_id).catch((e) => alert(e.message)));
    els.collegeTeamsBody.appendChild(tr);
  });
  if (!state.selectedCollegeTeamId && sorted[0]?.college_team_id) {
    state.selectedCollegeTeamId = sorted[0].college_team_id;
  }
}

async function loadCollegeTeamDetail(teamId) {
  if (!teamId) return;
  const payload = await fetchJson(`/api/college/team-detail/${encodeURIComponent(teamId)}`);
  const teamName = payload?.team?.name || teamId;
  const roster = payload?.roster || [];
  state.selectedCollegeTeamId = teamId;
  els.collegeRosterTitle.textContent = `${teamName} 로스터`;
  els.collegeRosterBody.innerHTML = roster.length ? roster.map((p) => `
    <tr>
      <td>${p?.name || "-"}</td>
      <td>${p?.pos || "-"}</td>
      <td>${p?.class_year || "-"}</td>
      <td>${collegeStat(p, "pts").toFixed(1)}</td>
      <td>${collegeStat(p, "reb").toFixed(1)}</td>
      <td>${collegeStat(p, "ast").toFixed(1)}</td>
    </tr>
  `).join("") : `<tr><td class="schedule-empty" colspan="6">로스터 데이터가 없습니다.</td></tr>`;
}

async function loadCollegeLeaders() {
  const sort = state.collegeLeadersSort || "pts";
  const payload = await fetchJson(`/api/college/players?sort=${encodeURIComponent(sort)}&order=desc&limit=100`);
  const players = payload?.players || [];
  els.collegeLeadersBody.innerHTML = players.length ? players.map((p, idx) => `
    <tr>
      <td>${idx + 1}</td>
      <td>${p?.name || "-"}</td>
      <td>${p?.college_team_name || p?.college_team_id || "-"}</td>
      <td>${p?.pos || "-"}</td>
      <td>${collegeStat(p, "pts").toFixed(1)}</td>
      <td>${collegeStat(p, "reb").toFixed(1)}</td>
      <td>${collegeStat(p, "ast").toFixed(1)}</td>
      <td>${collegeStat(p, "stl").toFixed(1)}</td>
      <td>${collegeStat(p, "blk").toFixed(1)}</td>
    </tr>
  `).join("") : `<tr><td class="schedule-empty" colspan="9">리더보드 데이터가 없습니다.</td></tr>`;
}

async function loadCollegeBigboard() {
  const expertId = state.selectedCollegeExpertId;
  if (!expertId) {
    renderCollegeEmpty(els.collegeBigboardBody, 5, "전문가를 선택하세요.");
    return;
  }
  const payload = await fetchJson(`/api/offseason/draft/bigboard/expert?expert_id=${encodeURIComponent(expertId)}&pool_mode=auto`);
  const board = payload?.board || [];
  els.collegeBigboardBody.innerHTML = board.length ? board.map((r) => `
    <tr>
      <td>${r?.rank ?? "-"}</td>
      <td>${r?.name || "-"}</td>
      <td>${r?.pos || "-"}</td>
      <td>${r?.tier || "-"}</td>
      <td>${r?.summary || "-"}</td>
    </tr>
  `).join("") : `<tr><td class="schedule-empty" colspan="5">빅보드 데이터가 없습니다.</td></tr>`;
}

async function loadCollegeScouting() {
  if (!state.selectedTeamId) return;
  const [scoutsPayload, playersPayload, reportsPayload] = await Promise.all([
    fetchJson(`/api/scouting/scouts/${encodeURIComponent(state.selectedTeamId)}`),
    fetchJson("/api/college/players?sort=pts&order=desc&limit=200"),
    fetchJson(`/api/scouting/reports?team_id=${encodeURIComponent(state.selectedTeamId)}&limit=50`),
  ]);
  state.scoutingScouts = scoutsPayload?.scouts || [];
  state.scoutingReports = reportsPayload?.reports || [];
  const players = playersPayload?.players || [];

  els.collegeScoutSelect.innerHTML = state.scoutingScouts.map((s) => `<option value="${s.scout_id}">${s.display_name} (${s.specialty_key})</option>`).join("");
  els.collegeScoutPlayerSelect.innerHTML = players.map((p) => `<option value="${p.player_id}">${p.name} · ${p.college_team_name || p.college_team_id}</option>`).join("");

  els.collegeReportsBody.innerHTML = state.scoutingReports.length ? state.scoutingReports.map((r) => `
    <tr>
      <td>${String(r?.as_of_date || "-").slice(0, 10)}</td>
      <td>${r?.scout?.display_name || r?.scout?.scout_id || "-"}</td>
      <td>${r?.player_snapshot?.name || r?.target_player_id || "-"}</td>
      <td>${r?.status || "-"}</td>
      <td>${(r?.report_text || "").slice(0, 80) || "(텍스트 리포트 없음)"}</td>
    </tr>
  `).join("") : `<tr><td class="schedule-empty" colspan="5">리포트가 없습니다. 배정 후 월말 진행 시 생성됩니다.</td></tr>`;
}

async function showCollegeScreen() {
  if (!state.selectedTeamId) {
    alert("먼저 팀을 선택해주세요.");
    return;
  }
  setLoading(true, "대학 리그 정보를 불러오는 중입니다...");
  try {
    const [meta, teams, experts] = await Promise.all([
      fetchJson("/api/college/meta"),
      fetchJson("/api/college/teams"),
      fetchJson("/api/offseason/draft/experts"),
    ]);
    state.collegeMeta = meta;
    state.collegeTeams = teams || [];
    state.collegeExperts = experts?.experts || [];

    els.collegeMetaLine.textContent = `시즌 ${meta?.season_year || "-"} · 대학팀 ${meta?.college?.teams || 0}개 · 예정 드래프트 ${meta?.upcoming_draft_year || "-"}`;
    renderCollegeTeams(state.collegeTeams);
    if (state.selectedCollegeTeamId) {
      await loadCollegeTeamDetail(state.selectedCollegeTeamId);
    }

    const sortOptions = ["pts", "reb", "ast", "stl", "blk", "mpg", "games", "ts_pct", "usg", "fg_pct"];
    els.collegeLeaderSort.innerHTML = sortOptions.map((k) => `<option value="${k}">${k.toUpperCase()}</option>`).join("");
    els.collegeLeaderSort.value = state.collegeLeadersSort;
    await loadCollegeLeaders();

    els.collegeExpertSelect.innerHTML = state.collegeExperts.map((e) => `<option value="${e.expert_id}">${e.display_name}</option>`).join("");
    if (!state.selectedCollegeExpertId && state.collegeExperts[0]?.expert_id) {
      state.selectedCollegeExpertId = state.collegeExperts[0].expert_id;
    }
    els.collegeExpertSelect.value = state.selectedCollegeExpertId;
    await loadCollegeBigboard();

    await loadCollegeScouting();
    switchCollegeTab("teams");
    activateScreen(els.collegeScreen);
  } finally {
    setLoading(false);
  }
}

function showTeamSelection() { activateScreen(els.teamScreen); }

function showMainScreen() {
  activateScreen(els.mainScreen);
  const teamName = state.selectedTeamName || state.selectedTeamId || "선택 팀";
  els.mainTeamTitle.textContent = teamName;
  void refreshMainDashboard();
}

function formatIsoDate(dateString) {
  const raw = String(dateString || "").slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(raw) ? raw : "YYYY-MM-DD";
}

function randomTipoffTime() {
  const hour24 = 14 + Math.floor(Math.random() * 6);
  const minute = Math.floor(Math.random() * 60);
  const hour12 = String(hour24 > 12 ? hour24 - 12 : hour24).padStart(2, "0");
  return `${hour12}:${String(minute).padStart(2, "0")} PM`;
}

function isCompletedGame(game) {
  return game?.home_score != null && game?.away_score != null;
}

async function fetchInGameDate() {
  const summary = await fetchJson("/api/state/summary");
  const currentDate = summary?.workflow_state?.league?.current_date;
  return formatIsoDate(currentDate);
}

function resetNextGameCard() {
  els.teamAName.textContent = "Team A";
  els.teamBName.textContent = "Team B";
  els.nextGameDatetime.textContent = "YYYY-MM-DD --:-- PM";
}

function formatLeader(leader) {
  if (!leader || !leader.name) return "-";
  return `${leader.name} ${num(leader.value, 0)}`;
}

function renderEmptyScheduleRow(colSpan, text) {
  return `<tr><td colspan="${colSpan}" class="schedule-empty">${text}</td></tr>`;
}


function scheduleVenueType(game) {
  const label = String(game?.opponent_label || "").trim().toLowerCase();
  if (label.startsWith("vs")) return "home";
  if (label.startsWith("@")) return "away";
  return "all";
}

function scheduleGameDate(game) {
  const iso = String(game?.date || "").slice(0, 10);
  if (/^\d{4}-\d{2}-\d{2}$/.test(iso)) return new Date(`${iso}T00:00:00`);
  const mmdd = String(game?.date_mmdd || "");
  const m = mmdd.match(/^(\d{1,2})\/(\d{1,2})$/);
  if (!m) return null;
  const month = Number(m[1]);
  const day = Number(m[2]);
  const seasonYear = month >= 10 ? 2025 : 2026;
  return new Date(seasonYear, month - 1, day);
}

function dateDiffDays(gameA, gameB) {
  const a = scheduleGameDate(gameA);
  const b = scheduleGameDate(gameB);
  if (!a || !b) return null;
  return Math.round((b - a) / (1000 * 60 * 60 * 24));
}

function applyScheduleFilter(games) {
  const segment = state.scheduleFilter?.segment || "all";
  const venue = state.scheduleFilter?.venue || "all";
  return (games || []).filter((g) => {
    const completed = !!g?.is_completed;
    const passSegment = segment === "all" || (segment === "completed" && completed) || (segment === "upcoming" && !completed);
    const gameVenue = scheduleVenueType(g);
    const passVenue = venue === "all" || gameVenue === venue;
    return passSegment && passVenue;
  });
}

function renderScheduleHero(games) {
  const upcoming = (games || []).filter((g) => !g?.is_completed);
  const nextGame = upcoming[0];
  if (!nextGame) {
    els.scheduleHeroMatchup.textContent = "예정된 다음 경기가 없습니다.";
    els.scheduleHeroDatetime.textContent = "-";
    els.scheduleHeroStatus.textContent = "IDLE";
    els.scheduleHeroVenue.textContent = "-";
    return;
  }
  const short = nextGame.opponent_label || "-";
  const team = nextGame.opponent_team_name || nextGame.opponent_team_id || "상대팀";
  els.scheduleHeroMatchup.textContent = `${short} ${team}`;
  els.scheduleHeroDatetime.textContent = `${nextGame.date_mmdd || "--/--"} · ${nextGame.tipoff_time || "--:-- --"}`;
  els.scheduleHeroStatus.textContent = "UPCOMING";
  els.scheduleHeroVenue.textContent = scheduleVenueType(nextGame) === "home" ? "HOME" : "AWAY";
}

function renderScheduleInsights(games) {
  const list = games || [];
  const completed = list.filter((g) => g?.is_completed);
  const upcoming = list.filter((g) => !g?.is_completed);
  const home = list.filter((g) => scheduleVenueType(g) === "home").length;
  const away = list.filter((g) => scheduleVenueType(g) === "away").length;
  const sorted = [...list].sort((a, b) => {
    const ad = scheduleGameDate(a)?.getTime() || 0;
    const bd = scheduleGameDate(b)?.getTime() || 0;
    return ad - bd;
  });
  let b2b = 0;
  for (let i = 1; i < sorted.length; i += 1) {
    const d = dateDiffDays(sorted[i - 1], sorted[i]);
    if (d === 1) b2b += 1;
  }
  els.scheduleInsightCompleted.textContent = String(completed.length);
  els.scheduleInsightUpcoming.textContent = String(upcoming.length);
  els.scheduleInsightHomeAway.textContent = `${home} / ${away}`;
  els.scheduleInsightB2b.textContent = `${b2b}회`;
}

function refreshScheduleFilterButtons() {
  document.querySelectorAll('#schedule-screen .schedule-filter-btn').forEach((btn) => {
    const group = btn.dataset.filterGroup;
    const value = btn.dataset.filterValue;
    const active = state.scheduleFilter?.[group] === value;
    btn.classList.toggle('is-active', active);
  });
}

function renderScheduleScreenData() {
  const source = state.scheduleGames || [];
  renderScheduleHero(source);
  renderScheduleInsights(source);
  const filtered = applyScheduleFilter(source);
  const completed = filtered.filter((g) => g?.is_completed);
  const upcoming = filtered.filter((g) => !g?.is_completed);
  renderScheduleTables(filtered);
  if ((state.scheduleFilter?.segment === 'all' || state.scheduleFilter?.segment === 'completed') && completed.length === 0) {
    els.scheduleCompletedBody.innerHTML = renderEmptyScheduleRow(7, '조건에 맞는 완료 경기가 없습니다.');
  }
  if ((state.scheduleFilter?.segment === 'all' || state.scheduleFilter?.segment === 'upcoming') && upcoming.length === 0) {
    els.scheduleUpcomingBody.innerHTML = renderEmptyScheduleRow(3, '조건에 맞는 예정 경기가 없습니다.');
  }
  refreshScheduleFilterButtons();
}

function bindScheduleFilters() {
  document.querySelectorAll('#schedule-screen .schedule-filter-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const group = btn.dataset.filterGroup;
      const value = btn.dataset.filterValue;
      if (!group || !value) return;
      state.scheduleFilter[group] = value;
      renderScheduleScreenData();
    });
  });
}

function renderScheduleTables(games) {
  const completed = (games || []).filter((g) => g?.is_completed);
  const upcoming = (games || []).filter((g) => !g?.is_completed);

  els.scheduleCompletedBody.innerHTML = completed.length
    ? completed.map((g) => {
      const result = g.result || {};
      const record = g.record_after_game || {};
      const leaders = g.leaders || {};
      return `
        <tr>
          <td>${g.date_mmdd || "--/--"}</td>
          <td class="schedule-opponent-cell">${g.opponent_label || "-"} <span class="schedule-opponent-name">${g.opponent_team_name || g.opponent_team_id || ""}</span></td>
          <td><span class="schedule-result-badge ${result.wl === "W" ? "schedule-result-win" : "schedule-result-loss"}">${result.display || "-"}</span></td>
          <td>${record.display || "-"}</td>
          <td>${formatLeader(leaders.points)}</td>
          <td>${formatLeader(leaders.rebounds)}</td>
          <td>${formatLeader(leaders.assists)}</td>
        </tr>
      `;
    }).join("")
    : renderEmptyScheduleRow(7, "완료된 경기가 없습니다.");

  els.scheduleUpcomingBody.innerHTML = upcoming.length
    ? upcoming.map((g) => `
      <tr>
        <td>${g.date_mmdd || "--/--"}</td>
        <td class="schedule-opponent-cell">${g.opponent_label || "-"} <span class="schedule-opponent-name">${g.opponent_team_name || g.opponent_team_id || ""}</span></td>
        <td><span class="schedule-time-chip">${g.tipoff_time || "--:-- --"}</span></td>
      </tr>
    `).join("")
    : renderEmptyScheduleRow(3, "예정된 경기가 없습니다.");
}

async function showScheduleScreen() {
  if (!state.selectedTeamId) {
    alert("먼저 팀을 선택해주세요.");
    return;
  }

  setLoading(true, "스케줄 정보를 불러오는 중...");
  try {
    const schedule = await fetchJson(`/api/team-schedule/${encodeURIComponent(state.selectedTeamId)}`);
    const teamName = state.selectedTeamName || TEAM_FULL_NAMES[state.selectedTeamId] || state.selectedTeamId;
    els.scheduleTitle.textContent = `${teamName} 정규 시즌 일정`;
    state.scheduleGames = schedule?.games || [];
    renderScheduleScreenData();
    activateScreen(els.scheduleScreen);
  } catch (e) {
    els.scheduleCompletedBody.innerHTML = renderEmptyScheduleRow(7, `스케줄 로딩 실패: ${e.message}`);
    els.scheduleUpcomingBody.innerHTML = renderEmptyScheduleRow(3, "-");
    activateScreen(els.scheduleScreen);
  } finally {
    setLoading(false);
  }
}

async function refreshMainDashboard() {
  if (!state.selectedTeamId) return;

  try {
    const currentDate = await fetchInGameDate();
    state.currentDate = currentDate;
    els.mainCurrentDate.textContent = currentDate;

    const schedule = await fetchJson(`/api/team-schedule/${encodeURIComponent(state.selectedTeamId)}`);
    const games = schedule?.games || [];
    const nextGame = games.find((g) => {
      const date = String(g?.date || "").slice(0, 10);
      return date >= currentDate && !isCompletedGame(g);
    });

    if (!nextGame) {
      resetNextGameCard();
      els.nextGameDatetime.textContent = "예정된 다음 경기가 없습니다.";
      return;
    }

    const homeId = String(nextGame.home_team_id || "").toUpperCase();
    const awayId = String(nextGame.away_team_id || "").toUpperCase();
    const gameDate = formatIsoDate(nextGame.date);
    els.teamAName.textContent = TEAM_FULL_NAMES[homeId] || homeId || "Team A";
    els.teamBName.textContent = TEAM_FULL_NAMES[awayId] || awayId || "Team B";
    const tipoffTime = nextGame.tipoff_time || randomTipoffTime();
    els.nextGameDatetime.textContent = `${gameDate} ${tipoffTime}`;
  } catch (e) {
    resetNextGameCard();
    els.mainCurrentDate.textContent = "YYYY-MM-DD";
    els.nextGameDatetime.textContent = `다음 경기 정보를 불러오지 못했습니다: ${e.message}`;
  }
}

function num(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(v, min, max) {
  return Math.min(max, Math.max(min, v));
}

function formatHeightIn(inches) {
  const inch = Math.max(0, Math.round(num(inches, 0)));
  const feet = Math.floor(inch / 12);
  const rem = inch % 12;
  return `${feet}'${String(rem).padStart(2, "0")}"`;
}

function formatWeightLb(lb) { return `${Math.round(num(lb, 0))} lb`; }

function formatMoney(n) {
  return `$${Math.round(num(n, 0)).toLocaleString("en-US")}`;
}

function formatPercent(value) {
  return `${Math.round(clamp(num(value, 0), 0, 1) * 100)}%`;
}

function seasonLabelByYear(year) {
  const y = Number(year);
  if (!Number.isFinite(y)) return "시즌 미정";
  const start = String(y).slice(-2);
  const end = String(y + 1).slice(-2).padStart(2, "0");
  return `${start}-${end} 시즌`;
}

function getOptionTypeLabel(optionType) {
  if (optionType === "PLAYER") return "플레이어 옵션";
  if (optionType === "TEAM") return "팀 옵션";
  return "옵션";
}

function ratioToColor(ratio) {
  const r = clamp(num(ratio, 0), 0, 1);
  const hue = Math.round(r * 120);
  return `hsl(${hue} 80% 36%)`;
}

function renderConditionRing(longStamina, shortStamina) {
  const longPct = clamp(num(longStamina, 0), 0, 1) * 100;
  const shortPct = clamp(num(shortStamina, 0), 0, 1) * 100;
  const longColor = ratioToColor(longStamina);
  const shortColor = ratioToColor(shortStamina);
  return `<div class="condition-ring" style="--long-pct:${longPct};--short-pct:${shortPct};--long-color:${longColor};--short-color:${shortColor};" title="장기 ${Math.round(longPct)}% · 단기 ${Math.round(shortPct)}%"></div>`;
}

function renderRosterRows(rows) {
  els.rosterBody.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.className = "roster-row";
    tr.dataset.playerId = row.player_id;

    const shortStamina = row.short_term_stamina ?? (1 - num(row.short_term_fatigue, 0));
    const longStamina = row.long_term_stamina ?? (1 - num(row.long_term_fatigue, 0));
    const sharpness = clamp(num(row.sharpness, 50), 0, 100);

    tr.innerHTML = `
      <td>${row.name || "-"}</td>
      <td>${row.pos || "-"}</td>
      <td>${num(row.age, 0)}</td>
      <td>${formatHeightIn(row.height_in)}</td>
      <td>${formatWeightLb(row.weight_lb)}</td>
      <td>${formatMoney(row.salary)}</td>
      <td class="condition-cell">${renderConditionRing(longStamina, shortStamina)}</td>
      <td><span class="sharpness-badge" style="background:${ratioToColor(sharpness / 100)}">${Math.round(sharpness)}%</span></td>
    `;

    tr.addEventListener("click", () => {
      state.selectedPlayerId = row.player_id;
      loadPlayerDetail(row.player_id).catch((e) => alert(e.message));
    });

    els.rosterBody.appendChild(tr);
  }
}

function getDissatisfactionSummary(d) {
  if (!d || !d.is_dissatisfied) return { text: "불만: 없음", details: [] };
  const st = d.state || {};
  const axes = [
    ["팀", num(st.team_frustration, 0)],
    ["역할", num(st.role_frustration, 0)],
    ["계약", num(st.contract_frustration, 0)],
    ["건강", num(st.health_frustration, 0)],
    ["케미", num(st.chemistry_frustration, 0)],
    ["사용률", num(st.usage_frustration, 0)],
  ].sort((a, b) => b[1] - a[1]);

  const top = axes.filter(([, v]) => v > 0.1).slice(0, 3).map(([k, v]) => `${k} ${Math.round(v * 100)}%`);
  const level = clamp(num(st.trade_request_level, 0), 0, 10);
  return {
    text: `불만: 있음 (강도 ${Math.round(axes[0][1] * 100)}%, TR ${level})`,
    details: top,
  };
}

function renderAttrGrid(attrs) {
  const entries = Object.entries(attrs || {}).sort((a, b) => String(a[0]).localeCompare(String(b[0])));
  if (!entries.length) return '<p class="empty-copy">능력치 데이터가 없습니다.</p>';
  return entries
    .map(([k, v]) => {
      const value = typeof v === "number" ? (Math.abs(v) <= 1 ? `${Math.round(v * 100)}` : `${Math.round(v)}`) : String(v);
      return `
        <div class="attr-card">
          <span class="attr-name">${k}</span>
          <strong class="attr-value">${value}</strong>
        </div>
      `;
    })
    .join("");
}

function buildContractRows(contractActive, fallbackSalary) {
  if (!contractActive) {
    return [{ label: "계약", value: "활성 계약 정보 없음", emphasis: true }];
  }

  const salaryByYear = contractActive.salary_by_year || {};
  const salaryYears = Object.keys(salaryByYear)
    .map((y) => Number(y))
    .filter((y) => Number.isFinite(y))
    .sort((a, b) => a - b);

  const optionByYear = new Map((contractActive.options || []).map((opt) => [Number(opt.season_year), opt]));
  const rows = [];

  const initialSalary = salaryYears.length ? salaryByYear[salaryYears[0]] : fallbackSalary;
  rows.push({ label: "샐러리", value: formatMoney(initialSalary), emphasis: true });

  salaryYears.forEach((year, idx) => {
    if (idx === 0) return;
    const option = optionByYear.get(year);
    const optionText = option ? ` (${getOptionTypeLabel(option.type)})` : "";
    rows.push({
      label: seasonLabelByYear(year),
      value: `${formatMoney(salaryByYear[year])}${optionText}`,
      emphasis: false,
    });
  });

  const outstandingOptionRows = (contractActive.options || [])
    .map((option) => ({
      year: Number(option.season_year),
      option,
    }))
    .filter(({ year }) => Number.isFinite(year) && !(year in salaryByYear))
    .sort((a, b) => a.year - b.year)
    .map(({ year, option }) => ({
      label: seasonLabelByYear(year),
      value: `${getOptionTypeLabel(option.type)} (${option.status || "PENDING"})`,
      emphasis: false,
    }));

  return rows.concat(outstandingOptionRows);
}

function renderPlayerDetail(detail) {
  const p = detail.player || {};
  const contract = detail.contract || {};
  const diss = getDissatisfactionSummary(detail.dissatisfaction);
  const injury = detail.injury || {};
  const condition = detail.condition || {};
  const seasonStats = detail.season_stats || {};
  const totals = seasonStats.totals || {};
  const twoWay = detail.two_way || {};
  const contractActive = contract.active || null;
  const contractRows = buildContractRows(contractActive, detail.roster?.salary_amount);
  const dissatisfactionDescription = (detail.dissatisfaction?.reasons || []).length
    ? detail.dissatisfaction.reasons
    : diss.details;

  const injuryState = injury.state || {};
  const injuryDetails = [
    injuryState.injury_type && `부상 유형: ${injuryState.injury_type}`,
    injuryState.body_part && `부위: ${injuryState.body_part}`,
    injuryState.games_remaining != null && `복귀 예상: ${num(injuryState.games_remaining, 0)}경기 후`,
    injuryState.note && `메모: ${injuryState.note}`,
  ].filter(Boolean);

  const totalsEntries = Object.entries(totals || {});
  const statsSummary = totalsEntries.length
    ? `<div class="stats-grid">${totalsEntries
      .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
      .map(([k, v]) => `<div class="stat-chip"><span>${k}</span><strong>${typeof v === "number" ? (Math.round(v * 100) / 100) : v}</strong></div>`)
      .join("")}</div>`
    : '<p class="empty-copy">누적 스탯 데이터가 없습니다.</p>';

  const healthText = injury.is_injured
    ? `${injury.status || "부상"} · ${(injury.state?.injury_type || "")}`
    : "건강함";


  const playerName = p.name || "선수";
  els.playerDetailTitle.textContent = `${playerName} 상세 정보`;
  els.playerDetailContent.innerHTML = `
    <div class="player-layout">
      <section class="detail-card detail-card-header">
        <div class="detail-head detail-head-main">
          <div>
            <h3>${playerName}</h3>
            <p class="detail-subline">${p.pos || "-"} · ${num(p.age, 0)}세 · ${formatHeightIn(p.height_in)} / ${formatWeightLb(p.weight_lb)}</p>
          </div>
          <span class="sharpness-badge" style="background:${ratioToColor(num(condition.sharpness, 50) / 100)}">경기력 ${Math.round(num(condition.sharpness, 50))}%</span>
        </div>
      </section>

      <section class="detail-card detail-card-contract">
        <h4>계약 정보</h4>
        <ul class="compact-kv-list">
          ${contractRows.map((row) => `<li><span>${row.label}</span><strong${row.emphasis ? ' class="text-accent"' : ""}>${row.value}</strong></li>`).join("")}
        </ul>
        ${twoWay.is_two_way ? `<p class="section-note">투웨이 계약 · 남은 경기 ${num(twoWay.games_remaining, 0)} / ${num(twoWay.game_limit, 0)}</p>` : ""}
      </section>

      <section class="detail-card detail-card-dissatisfaction">
        <h4>불만 여부</h4>
        <p class="status-line ${detail.dissatisfaction?.is_dissatisfied ? "status-danger" : "status-ok"}">${detail.dissatisfaction?.is_dissatisfied ? "불만 있음" : "불만 없음"}</p>
        <p class="section-copy">${diss.text}</p>
        ${dissatisfactionDescription.length ? `<ul class="kv-list">${dissatisfactionDescription.map((x) => `<li>${x}</li>`).join("")}</ul>` : ""}
      </section>

      <section class="detail-card detail-card-attr">
        <h4>능력치 (ATTR)</h4>
        <div class="attr-grid">${renderAttrGrid(p.attrs || {})}</div>
      </section>

      <section class="detail-card detail-card-health">
        <h4>건강 상태</h4>
        <ul class="compact-kv-list compact-kv-list-health">
          <li><span>장기 체력</span><strong>${formatPercent(condition.long_term_stamina)}</strong></li>
          <li><span>단기 체력</span><strong>${formatPercent(condition.short_term_stamina)}</strong></li>
          <li><span>부상 여부</span><strong>${injury.is_injured ? "부상" : "정상"}</strong></li>
        </ul>
        <p class="section-copy">${healthText}</p>
        ${injuryDetails.length ? `<ul class="kv-list">${injuryDetails.map((item) => `<li>${item}</li>`).join("")}</ul>` : ""}
      </section>

      <section class="detail-card detail-card-stats">
        <h4>누적 스탯</h4>
        <p class="section-copy">출전 경기 수: ${num(seasonStats.games, 0)}경기</p>
        ${statsSummary}
      </section>
    </div>
  `;
}

async function loadPlayerDetail(playerId) {
  setLoading(true, "선수 상세 정보를 불러오는 중...");
  try {
    const detail = await fetchJson(`/api/player-detail/${encodeURIComponent(playerId)}`);
    renderPlayerDetail(detail);
    activateScreen(els.playerDetailScreen);
  } finally {
    setLoading(false);
  }
}

async function showMyTeamScreen() {
  if (!state.selectedTeamId) {
    alert("먼저 팀을 선택해주세요.");
    return;
  }

  setLoading(true, "내 팀 로스터를 불러오는 중...");
  try {
    const detail = await fetchJson(`/api/team-detail/${encodeURIComponent(state.selectedTeamId)}`);
    state.rosterRows = detail.roster || [];
    state.selectedPlayerId = null;

    const teamName = state.selectedTeamName || TEAM_FULL_NAMES[state.selectedTeamId] || state.selectedTeamId;
    els.myTeamTitle.textContent = `${teamName} 선수단`;

    renderRosterRows(state.rosterRows);
    els.playerDetailContent.innerHTML = "";
    els.playerDetailTitle.textContent = "선수 상세 정보";
    activateScreen(els.myTeamScreen);
  } finally {
    setLoading(false);
  }
}

async function confirmTeamSelection(teamId, fullName) {
  const confirmed = window.confirm(`${fullName}을(를) 선택하시겠습니까?`);
  if (!confirmed) return;

  state.selectedTeamId = teamId;
  state.selectedTeamName = fullName;

  if (state.lastSaveSlotId) {
    await fetchJson("/api/game/set-user-team", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slot_id: state.lastSaveSlotId, user_team_id: teamId })
    });
  }

  showMainScreen();
}


function dateToIso(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function parseIsoDate(iso) {
  const v = String(iso || "").slice(0, 10);
  const d = new Date(`${v}T00:00:00`);
  return Number.isNaN(d.getTime()) ? null : d;
}

function startOfWeek(date) {
  const d = new Date(date.getTime());
  const day = d.getDay();
  const diff = day === 0 ? -6 : 1 - day;
  d.setDate(d.getDate() + diff);
  return d;
}

function addDays(date, n) {
  const d = new Date(date.getTime());
  d.setDate(d.getDate() + n);
  return d;
}

function trainingTypeLabel(t) {
  const m = {
    OFF_TACTICS: "공격",
    DEF_TACTICS: "수비",
    FILM: "필름",
    SCRIMMAGE: "청백전",
    RECOVERY: "휴식",
    REST: "없음"
  };
  return m[String(t || "").toUpperCase()] || "-";
}

function buildCalendar4Weeks(currentDateIso) {
  const today = parseIsoDate(currentDateIso) || new Date();
  const first = startOfWeek(today);
  const days = [];
  for (let i = 0; i < 28; i += 1) {
    const date = addDays(first, i);
    days.push(dateToIso(date));
  }
  return days;
}

async function loadTrainingData() {
  if (!state.selectedTeamId) return;
  const currentDate = state.currentDate || await fetchInGameDate();
  state.currentDate = currentDate;
  const allDays = buildCalendar4Weeks(currentDate);
  state.trainingCalendarDays = allDays;

  const schedule = await fetchJson(`/api/team-schedule/${encodeURIComponent(state.selectedTeamId)}`);
  const gameByDate = {};
  (schedule.games || []).forEach((g) => {
    const d = String(g.date || "").slice(0, 10);
    if (!d) return;
    const opp = g.home_team_id === state.selectedTeamId ? g.away_team_id : g.home_team_id;
    gameByDate[d] = String(opp || "").toUpperCase();
  });

  const from = allDays[0];
  const to = allDays[allDays.length - 1];
  const stored = await fetchJson(`/api/practice/team/${encodeURIComponent(state.selectedTeamId)}/sessions?date_from=${encodeURIComponent(from)}&date_to=${encodeURIComponent(to)}`);
  const sessions = { ...(stored.sessions || {}) };

  const previewDates = allDays.filter((d) => d >= currentDate && !gameByDate[d]);
  await Promise.all(previewDates.map(async (d) => {
    if (sessions[d]) return;
    try {
      const res = await fetchJson(`/api/practice/team/${encodeURIComponent(state.selectedTeamId)}/session?date_iso=${encodeURIComponent(d)}`);
      sessions[d] = { session: res.session, is_user_set: res.is_user_set };
    } catch (e) {
      // fail-soft
    }
  }));

  const teamDetail = await fetchJson(`/api/team-detail/${encodeURIComponent(state.selectedTeamId)}`);
  state.trainingRoster = teamDetail.roster || [];

  const [offFam, defFam] = await Promise.all([
    fetchJson(`/api/readiness/team/${encodeURIComponent(state.selectedTeamId)}/familiarity?scheme_type=offense`).catch(() => ({ items: [] })),
    fetchJson(`/api/readiness/team/${encodeURIComponent(state.selectedTeamId)}/familiarity?scheme_type=defense`).catch(() => ({ items: [] })),
  ]);
  state.trainingFamiliarity = { offense: offFam.items || [], defense: defFam.items || [] };

  state.trainingSessionsByDate = sessions;
  state.trainingGameByDate = gameByDate;
}

function renderTrainingCalendar() {
  const container = els.trainingCalendarGrid;
  const today = state.currentDate;
  container.innerHTML = "";

  state.trainingCalendarDays.forEach((iso) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "training-day-cell";

    const d = parseIsoDate(iso);
    const label = `${d.getMonth() + 1}/${d.getDate()}`;
    const gameOpp = state.trainingGameByDate?.[iso];
    const isPast = iso < today;
    const isGameDay = !!gameOpp;
    const selectable = !isPast && !isGameDay;

    if (isGameDay) btn.classList.add("is-game");
    if (state.trainingSelectedDates.has(iso)) btn.classList.add("is-selected");

    const sessInfo = state.trainingSessionsByDate?.[iso];
    const sessType = sessInfo?.session?.type;
    const sessionLine = sessInfo
      ? (sessInfo.is_user_set ? `지정 · ${trainingTypeLabel(sessType)}` : `AUTO · ${trainingTypeLabel(sessType)}`)
      : "";

    btn.innerHTML = `
      <div class="training-day-date">${label}</div>
      <div class="training-day-note">${gameOpp ? `vs ${gameOpp}` : ""}</div>
      <div class="training-day-sub">${!gameOpp ? sessionLine : ""}</div>
    `;

    if (!selectable) {
      btn.disabled = true;
    } else {
      btn.addEventListener("click", () => {
        if (state.trainingSelectedDates.has(iso)) state.trainingSelectedDates.delete(iso);
        else state.trainingSelectedDates.add(iso);
        renderTrainingCalendar();
      });
    }

    container.appendChild(btn);
  });
}

function optionsHtml(list, fallback = []) {
  const merged = [...new Set([...(list || []), ...fallback])];
  return merged.map((x) => `<option value="${x}">${x}</option>`).join("");
}

async function renderTrainingDetail(type) {
  const selected = [...state.trainingSelectedDates].sort();
  if (!selected.length) {
    els.trainingDetailPanel.innerHTML = '<p class="empty-copy">적용할 날짜를 먼저 선택하세요.</p>';
    return;
  }

  const baseSession = {
    type,
    offense_scheme_key: null,
    defense_scheme_key: null,
    participant_pids: [],
    non_participant_type: "RECOVERY"
  };

  const offSchemes = state.trainingFamiliarity.offense.map((x) => x.scheme_key);
  const defSchemes = state.trainingFamiliarity.defense.map((x) => x.scheme_key);

  if (type === "OFF_TACTICS") baseSession.offense_scheme_key = offSchemes[0] || "PACE_5OUT";
  if (type === "DEF_TACTICS") baseSession.defense_scheme_key = defSchemes[0] || "MAN_TO_MAN";
  if (type === "FILM") {
    baseSession.offense_scheme_key = offSchemes[0] || "PACE_5OUT";
    baseSession.defense_scheme_key = defSchemes[0] || "MAN_TO_MAN";
  }
  if (type === "SCRIMMAGE") {
    baseSession.participant_pids = state.trainingRoster.slice(0, 10).map((r) => String(r.player_id));
    baseSession.non_participant_type = "RECOVERY";
  }

  state.trainingDraftSession = baseSession;

  const firstDate = selected[0];
  const preview = await fetchJson(`/api/practice/team/${encodeURIComponent(state.selectedTeamId)}/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ season_year: null, date_iso: firstDate, ...baseSession })
  }).catch(() => null);

  const famRows = (type === "OFF_TACTICS" ? state.trainingFamiliarity.offense : type === "DEF_TACTICS" ? state.trainingFamiliarity.defense : []);
  const famHtml = famRows.length
    ? `<ul class="kv-list">${famRows.map((r) => `<li>${r.scheme_key}: ${Math.round(Number(r.value || 0))}</li>`).join("")}</ul>`
    : '<p class="empty-copy">숙련도 데이터가 없습니다.</p>';

  let extra = "";
  if (type === "OFF_TACTICS") {
    extra = `<div class="training-inline-row"><label>공격 스킴</label><select id="training-off-scheme">${optionsHtml(offSchemes, ["PACE_5OUT"])}</select></div>${famHtml}`;
  } else if (type === "DEF_TACTICS") {
    extra = `<div class="training-inline-row"><label>수비 스킴</label><select id="training-def-scheme">${optionsHtml(defSchemes, ["MAN_TO_MAN"])}</select></div>${famHtml}`;
  } else if (type === "SCRIMMAGE") {
    const rosterRows = state.trainingRoster.map((r) => `
      <tr>
        <td>${r.name || r.player_id}</td>
        <td>${Math.round(Number((r.short_term_stamina ?? 1) * 100))}%</td>
        <td>${Math.round(Number((r.long_term_stamina ?? 1) * 100))}%</td>
        <td>${Math.round(Number(r.sharpness ?? 50))}</td>
      </tr>
    `).join("");
    extra = `
      <p>5대5 라인업(참가자 PID 콤마 구분, 기본 10명):</p>
      <textarea id="training-scrimmage-pids" rows="3" style="width:100%;">${baseSession.participant_pids.join(",")}</textarea>
      <table class="training-player-table">
        <thead><tr><th>선수</th><th>단기 체력</th><th>장기 체력</th><th>샤프니스</th></tr></thead>
        <tbody>${rosterRows}</tbody>
      </table>
    `;
  }

  const prevText = preview
    ? `<ul class="kv-list"><li>공격 익숙도 gain: ${preview.preview?.familiarity_gain?.offense_gain ?? 0}</li><li>수비 익숙도 gain: ${preview.preview?.familiarity_gain?.defense_gain ?? 0}</li><li>평균 샤프니스 delta: ${Object.values(preview.preview?.intensity_mult_by_pid || {}).length ? (Object.values(preview.preview.intensity_mult_by_pid).reduce((a, x) => a + Number(x.sharpness_delta || 0), 0) / Object.values(preview.preview.intensity_mult_by_pid).length).toFixed(2) : "0.00"}</li></ul>`
    : '<p class="empty-copy">효과 프리뷰를 불러오지 못했습니다.</p>';

  els.trainingDetailPanel.innerHTML = `
    <div class="training-detail-grid">
      <h3>${trainingTypeLabel(type)} 훈련 설정</h3>
      <p>선택 날짜: ${selected.join(", ")}</p>
      ${extra}
      <div><strong>연습 효과 프리뷰</strong>${prevText}</div>
      <div class="training-inline-row"><button id="training-apply-btn" class="btn btn-primary" type="button">선택 날짜에 적용</button></div>
    </div>
  `;

  const offSel = document.getElementById("training-off-scheme");
  const defSel = document.getElementById("training-def-scheme");
  const scrimmagePids = document.getElementById("training-scrimmage-pids");
  if (offSel) offSel.addEventListener("change", () => { state.trainingDraftSession.offense_scheme_key = offSel.value; });
  if (defSel) defSel.addEventListener("change", () => { state.trainingDraftSession.defense_scheme_key = defSel.value; });
  if (scrimmagePids) scrimmagePids.addEventListener("input", () => {
    state.trainingDraftSession.participant_pids = scrimmagePids.value.split(",").map((x) => x.trim()).filter(Boolean);
  });

  const applyBtn = document.getElementById("training-apply-btn");
  applyBtn.addEventListener("click", async () => {
    const dates = [...state.trainingSelectedDates];
    await Promise.all(dates.map((dateIso) => fetchJson("/api/practice/team/session/set", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        team_id: state.selectedTeamId,
        date_iso: dateIso,
        ...state.trainingDraftSession
      })
    })));
    await loadTrainingData();
    renderTrainingCalendar();
    alert(`${dates.length}일에 훈련을 적용했습니다.`);
  });
}

async function showTrainingScreen() {
  if (!state.selectedTeamId) {
    alert("먼저 팀을 선택해주세요.");
    return;
  }
  setLoading(true, "훈련 화면 데이터를 불러오는 중...");
  try {
    state.trainingSelectedDates = new Set();
    await loadTrainingData();
    renderTrainingCalendar();
    els.trainingDetailPanel.innerHTML = '<p class="empty-copy">캘린더에서 날짜를 선택하고 훈련 버튼을 눌러 세부 설정을 확인하세요.</p>';
    activateScreen(els.trainingScreen);
  } finally {
    setLoading(false);
  }
}

async function loadSavesStatus() {
  try {
    const saveResult = await fetchJson("/api/game/saves");
    const slots = saveResult.slots || [];
    if (slots.length > 0) {
      state.lastSaveSlotId = slots[0].slot_id;
      els.continueBtn.disabled = false;
      els.continueBtn.setAttribute("aria-disabled", "false");
      els.continueHint.textContent = `저장된 게임 ${slots.length}개를 찾았습니다.`;
    } else {
      els.continueBtn.disabled = true;
      els.continueBtn.setAttribute("aria-disabled", "true");
      els.continueHint.textContent = "저장된 게임이 없습니다. 새 게임으로 시작해주세요.";
    }
  } catch (e) {
    els.continueBtn.disabled = true;
    els.continueBtn.setAttribute("aria-disabled", "true");
    els.continueHint.textContent = `저장 상태 확인 실패: ${e.message}`;
  }
}

async function renderTeams() {
  const result = await fetchJson("/api/teams");
  const teams = (result || []).slice(0, 30);
  els.teamGrid.innerHTML = "";

  const conferenceOrder = ["East", "West"];
  const divisionOrder = {
    East: ["Atlantic", "Central", "Southeast"],
    West: ["Northwest", "Pacific", "Southwest"],
  };

  const grouped = { East: {}, West: {} };
  teams.forEach((team) => {
    const conference = team.conference === "West" ? "West" : "East";
    const division = String(team.division || "");
    if (!grouped[conference][division]) grouped[conference][division] = [];
    grouped[conference][division].push(team);
  });

  conferenceOrder.forEach((conference) => {
    const conferenceSection = document.createElement("section");
    conferenceSection.className = "team-conference";

    const conferenceTitle = document.createElement("h3");
    conferenceTitle.className = "team-conference-title";
    conferenceTitle.textContent = conference === "East" ? "동부 컨퍼런스" : "서부 컨퍼런스";
    conferenceSection.appendChild(conferenceTitle);

    (divisionOrder[conference] || Object.keys(grouped[conference])).forEach((division) => {
      const divisionTeams = (grouped[conference][division] || []).sort((a, b) => {
        const aName = TEAM_FULL_NAMES[String(a.team_id || "").toUpperCase()] || String(a.team_id || "");
        const bName = TEAM_FULL_NAMES[String(b.team_id || "").toUpperCase()] || String(b.team_id || "");
        return aName.localeCompare(bName);
      });
      if (!divisionTeams.length) return;

      const divisionSection = document.createElement("div");
      divisionSection.className = "team-division";
      divisionSection.innerHTML = `<h4 class="team-division-title">${division}</h4>`;

      const divisionGrid = document.createElement("div");
      divisionGrid.className = "team-division-grid";

      divisionTeams.forEach((team) => {
        const id = String(team.team_id || "").toUpperCase();
        const fullName = TEAM_FULL_NAMES[id] || id;
        const card = document.createElement("button");
        card.className = "team-card";
        card.type = "button";
        card.innerHTML = `<strong>${fullName}</strong><small>${conference} · ${division}</small>`;
        card.addEventListener("click", () => {
          confirmTeamSelection(id, fullName).catch((e) => alert(e.message));
        });
        divisionGrid.appendChild(card);
      });

      divisionSection.appendChild(divisionGrid);
      conferenceSection.appendChild(divisionSection);
    });

    els.teamGrid.appendChild(conferenceSection);
  });
}


function formatSignedDiff(value) {
  const n = Number(value || 0);
  if (!Number.isFinite(n)) return "0.0";
  if (Math.abs(n) < 0.05) return "0.0";
  return `${n > 0 ? "+" : ""}${n.toFixed(1)}`;
}

function parseStreakScore(streak) {
  const value = String(streak || "-").trim().toUpperCase();
  const match = value.match(/^([WL])(\d+)$/);
  if (!match) return 0;
  const score = Number(match[2] || 0);
  return match[1] === "W" ? score : -score;
}

function parseL10Score(l10) {
  const value = String(l10 || "0-0").trim();
  const match = value.match(/^(\d+)-(\d+)$/);
  if (!match) return 0;
  return Number(match[1] || 0) - Number(match[2] || 0);
}

function standingsTier(rank) {
  const r = Number(rank || 999);
  if (r <= 6) return "playoff";
  if (r <= 10) return "playin";
  return "out";
}

function sortStandingsRows(rows, sortKey, sortDir) {
  const dir = sortDir === "asc" ? 1 : -1;
  const safe = [...(rows || [])];
  safe.sort((a, b) => {
    const rankA = Number(a?.rank || 999);
    const rankB = Number(b?.rank || 999);
    let va = 0;
    let vb = 0;

    switch (sortKey) {
      case "wins":
        va = Number(a?.wins || 0);
        vb = Number(b?.wins || 0);
        break;
      case "diff":
        va = Number(a?.diff || 0);
        vb = Number(b?.diff || 0);
        break;
      case "strk":
        va = parseStreakScore(a?.strk);
        vb = parseStreakScore(b?.strk);
        break;
      case "l10":
        va = parseL10Score(a?.l10);
        vb = parseL10Score(b?.l10);
        break;
      case "ppg":
        va = Number(a?.ppg || 0);
        vb = Number(b?.ppg || 0);
        break;
      case "opp_ppg":
        va = Number(a?.opp_ppg || 0);
        vb = Number(b?.opp_ppg || 0);
        break;
      case "pct":
      default:
        va = Number(a?.pct || 0);
        vb = Number(b?.pct || 0);
        break;
    }

    if (va === vb) return rankA - rankB;
    return (va - vb) * dir;
  });
  return safe;
}

function buildStandingsSummary(rows) {
  const source = [...(rows || [])];
  if (!source.length) {
    return {
      offense: { team: "-", value: "PPG -" },
      defense: { team: "-", value: "OPP PPG -" },
      diff: { team: "-", value: "DIFF -" },
      race: { team: "-", value: "GB -" }
    };
  }
  const by = (getter, mode) => source.reduce((best, row) => {
    if (!best) return row;
    const bv = getter(best);
    const rv = getter(row);
    if (mode === "max") return rv > bv ? row : best;
    return rv < bv ? row : best;
  }, null);

  const offense = by((r) => Number(r?.ppg || 0), "max");
  const defense = by((r) => Number(r?.opp_ppg || 0), "min");
  const diff = by((r) => Number(r?.diff || 0), "max");
  const racePool = source.filter((r) => Number(r?.rank || 99) >= 7 && Number(r?.rank || 99) <= 10);
  const race = racePool.sort((a, b) => Number(a?.rank || 99) - Number(b?.rank || 99))[0] || source[0];
  const name = (r) => TEAM_FULL_NAMES[String(r?.team_id || "").toUpperCase()] || String(r?.team_id || "-");

  return {
    offense: { team: name(offense), value: `PPG ${Number(offense?.ppg || 0).toFixed(1)}` },
    defense: { team: name(defense), value: `OPP PPG ${Number(defense?.opp_ppg || 0).toFixed(1)}` },
    diff: { team: name(diff), value: `DIFF ${formatSignedDiff(diff?.diff)}` },
    race: { team: name(race), value: `GB ${race?.gb_display ?? "-"}` }
  };
}

function renderStandingsSummary(summary) {
  els.standingsSummaryOffenseTeam.textContent = summary.offense.team;
  els.standingsSummaryOffenseValue.textContent = summary.offense.value;
  els.standingsSummaryDefenseTeam.textContent = summary.defense.team;
  els.standingsSummaryDefenseValue.textContent = summary.defense.value;
  els.standingsSummaryDiffTeam.textContent = summary.diff.team;
  els.standingsSummaryDiffValue.textContent = summary.diff.value;
  els.standingsSummaryRaceTeam.textContent = summary.race.team;
  els.standingsSummaryRaceValue.textContent = summary.race.value;
}

function renderStandingsTable() {
  const conf = state.standingsView.conference;
  const rows = conf === "west" ? (state.standingsData?.west || []) : (state.standingsData?.east || []);
  const sorted = sortStandingsRows(rows, state.standingsView.sortKey, state.standingsView.sortDir);
  els.standingsBody.innerHTML = "";
  els.standingsCardTitle.textContent = conf === "west" ? "Western Conference" : "Eastern Conference";

  sorted.forEach((row) => {
    const tr = document.createElement("tr");
    const teamId = String(row?.team_id || "").toUpperCase();
    const diff = Number(row?.diff || 0);
    const diffClass = diff > 0 ? "standings-diff-positive" : diff < 0 ? "standings-diff-negative" : "";
    const strk = String(row?.strk || "-").toUpperCase();
    const strkClass = strk.startsWith("W") ? "is-positive" : strk.startsWith("L") ? "is-negative" : "";
    tr.className = `standings-row-tier-${standingsTier(row?.rank)}`;
    tr.innerHTML = `
      <td>${row?.rank ?? "-"}</td>
      <td class="standings-team-cell"><span class="standings-team-pill">${TEAM_FULL_NAMES[teamId] || teamId || "-"}</span></td>
      <td>${row?.wins ?? 0}-${row?.losses ?? 0}</td>
      <td>${row?.pct || ".000"}</td>
      <td>${row?.gb_display ?? "-"}</td>
      <td><span class="standings-strk-badge ${strkClass}">${row?.strk || "-"}</span></td>
      <td>${row?.l10 || "0-0"}</td>
      <td class="${diffClass}">${formatSignedDiff(row?.diff)}</td>
      <td class="standings-advanced-col">${row?.home || "0-0"}</td>
      <td class="standings-advanced-col">${row?.away || "0-0"}</td>
      <td class="standings-advanced-col">${row?.div || "0-0"}</td>
      <td class="standings-advanced-col">${row?.conf || "0-0"}</td>
      <td class="standings-advanced-col">${Number(row?.ppg || 0).toFixed(1)}</td>
      <td class="standings-advanced-col">${Number(row?.opp_ppg || 0).toFixed(1)}</td>
    `;
    els.standingsBody.appendChild(tr);
  });

  const summary = buildStandingsSummary(rows);
  renderStandingsSummary(summary);

  if (els.standingsConferenceToggle) {
    Array.from(els.standingsConferenceToggle.querySelectorAll("button")).forEach((btn) => {
      const active = btn.dataset.conference === conf;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });
  }

  els.standingsSortDirBtn.textContent = state.standingsView.sortDir === "asc" ? "↑" : "↓";
  els.standingsSortKey.value = state.standingsView.sortKey;
  els.standingsTable.dataset.showAdvanced = state.standingsView.showAdvanced ? "true" : "false";
  els.standingsAdvancedToggle.textContent = state.standingsView.showAdvanced ? "고급 지표 숨기기" : "고급 지표 보기";
}

async function showStandingsScreen() {
  setLoading(true, "순위 데이터를 불러오는 중입니다...");
  try {
    const payload = await fetchJson("/api/standings/table");
    state.standingsData = payload;
    renderStandingsTable();
    activateScreen(els.standingsScreen);
  } finally {
    setLoading(false);
  }
}

async function createNewGame() {
  setLoading(true, "새 게임을 준비하는 중입니다. 엑셀 로스터를 DB로 부팅하고 있습니다...");
  const slotId = `slot_${new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14)}`;
  try {
    const response = await fetchJson("/api/game/new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        slot_name: `새 게임 ${new Date().toLocaleString("ko-KR")}`,
        slot_id: slotId,
        overwrite_if_exists: false
      })
    });

    state.lastSaveSlotId = response.slot_id;
    await renderTeams();
    showTeamSelection();
  } finally {
    setLoading(false);
  }
}

async function continueGame() {
  if (!state.lastSaveSlotId) return;
  setLoading(true, "저장된 게임을 불러오는 중입니다...");
  try {
    const loaded = await fetchJson("/api/game/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slot_id: state.lastSaveSlotId, strict: true })
    });

    const savedTeamId = String(loaded.user_team_id || "").toUpperCase();
    if (savedTeamId) {
      state.selectedTeamId = savedTeamId;
      state.selectedTeamName = TEAM_FULL_NAMES[savedTeamId] || savedTeamId;
      showMainScreen();
      return;
    }

    await renderTeams();
    showTeamSelection();
  } finally {
    setLoading(false);
  }
}


function tacticsSchemeLabel(schemes, key) {
  const found = (schemes || []).find((x) => x.key === key);
  return found ? found.label : key;
}

function tacticsRoleLabel(key) {
  return TACTICS_ROLE_LABELS[key] || key;
}

function getDefenseRolesForScheme(key) {
  return TACTICS_DEFENSE_ROLE_BY_SCHEME[key] || TACTICS_DEFENSE_ROLE_BY_SCHEME.Drop;
}

function buildTacticsDraft(roster) {
  const names = (roster || []).map((r) => ({ id: String(r.player_id || ""), name: String(r.name || r.player_id || "-") })).filter((x) => x.id);
  const starters = [];
  const rotation = [];
  for (let i = 0; i < 5; i += 1) {
    const p = names[i];
    starters.push({
      pid: p?.id || "",
      offenseRole: TACTICS_OFFENSE_ROLES[i % TACTICS_OFFENSE_ROLES.length],
      defenseRole: getDefenseRolesForScheme("Drop")[i],
      minutes: 32 - i
    });
  }
  for (let i = 5; i < 10; i += 1) {
    const p = names[i];
    rotation.push({
      pid: p?.id || "",
      offenseRole: TACTICS_OFFENSE_ROLES[i % TACTICS_OFFENSE_ROLES.length],
      defenseRole: getDefenseRolesForScheme("Drop")[i - 5],
      minutes: 18 - (i - 5)
    });
  }
  return { offenseScheme: "Spread_HeavyPnR", defenseScheme: "Drop", starters, rotation };
}

function tacticsSnapshot() {
  return JSON.stringify(state.tacticsDraft || {});
}

function setTacticsDirty(nextDirty) {
  state.tacticsDirty = !!nextDirty;
  if (!els.tacticsStatusPill) return;
  els.tacticsStatusPill.textContent = state.tacticsDirty ? "수정됨" : "저장됨";
  els.tacticsStatusPill.className = `tactics-status-pill ${state.tacticsDirty ? "dirty" : "saved"}`;
}

function markTacticsDirty() {
  if (!state.tacticsSavedSnapshot) {
    setTacticsDirty(true);
    return;
  }
  setTacticsDirty(tacticsSnapshot() !== state.tacticsSavedSnapshot);
}

function renderSchemeOptions(kind) {
  const isOff = kind === "offense";
  const optionsEl = isOff ? els.tacticsOffenseOptions : els.tacticsDefenseOptions;
  const list = isOff ? TACTICS_OFFENSE_SCHEMES : TACTICS_DEFENSE_SCHEMES;
  const selected = isOff ? state.tacticsDraft.offenseScheme : state.tacticsDraft.defenseScheme;
  optionsEl.innerHTML = list.map((s) => `
    <button type="button" data-key="${s.key}" class="${s.key === selected ? "is-selected" : ""}">
      <strong>${s.label}</strong>
      <span>${s.description || s.key}</span>
    </button>
  `).join("");
  optionsEl.querySelectorAll("button[data-key]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (isOff) state.tacticsDraft.offenseScheme = btn.dataset.key;
      else {
        state.tacticsDraft.defenseScheme = btn.dataset.key;
        const defRoles = getDefenseRolesForScheme(btn.dataset.key);
        [...state.tacticsDraft.starters, ...state.tacticsDraft.rotation].forEach((row, idx) => {
          if (!defRoles.includes(row.defenseRole)) row.defenseRole = defRoles[idx % defRoles.length];
        });
      }
      optionsEl.classList.add("hidden");
      markTacticsDirty();
      renderTacticsScreen();
    });
  });
}

function buildLineupRowHtml(group, idx, row, defenseRoles) {
  const players = state.rosterRows || [];
  const playerOptions = ['<option value="">- 선택 -</option>', ...players.map((r) => `<option value="${r.player_id}" ${String(r.player_id) === String(row.pid) ? "selected" : ""}>${r.name || r.player_id}</option>`)].join("");
  const offOptions = TACTICS_OFFENSE_ROLES.map((role) => `<option value="${role}" ${role === row.offenseRole ? "selected" : ""}>${tacticsRoleLabel(role)}</option>`).join("");
  const defOptions = defenseRoles.map((role) => `<option value="${role}" ${role === row.defenseRole ? "selected" : ""}>${tacticsRoleLabel(role)}</option>`).join("");
  return `
    <div class="tactics-lineup-row" data-group="${group}" data-idx="${idx}">
      <select data-field="pid">${playerOptions}</select>
      <select data-field="offenseRole">${offOptions}</select>
      <select data-field="defenseRole">${defOptions}</select>
      <input data-field="minutes" type="number" min="0" max="48" value="${Number(row.minutes || 0)}" />
    </div>
  `;
}

function validateDefenseRoleUnique(changedEl, nextValue) {
  const all = [...document.querySelectorAll('.tactics-lineup-row select[data-field="defenseRole"]')];
  const dup = all.find((el) => el !== changedEl && el.value === nextValue);
  return !dup;
}

function computeTacticsSummary() {
  const starters = state.tacticsDraft?.starters || [];
  const rotation = state.tacticsDraft?.rotation || [];
  const rows = [...starters, ...rotation];
  const starterMinutes = starters.reduce((sum, x) => sum + Number(x.minutes || 0), 0);
  const benchMinutes = rotation.reduce((sum, x) => sum + Number(x.minutes || 0), 0);
  const totalMinutes = starterMinutes + benchMinutes;

  const offCounts = {};
  const defCounts = {};
  rows.forEach((r) => {
    if (r.offenseRole) offCounts[r.offenseRole] = (offCounts[r.offenseRole] || 0) + 1;
    if (r.defenseRole) defCounts[r.defenseRole] = (defCounts[r.defenseRole] || 0) + 1;
  });

  const risks = [];
  if (totalMinutes !== 240) risks.push(`출전 시간 총합이 ${240 - totalMinutes > 0 ? `${240 - totalMinutes}분 부족` : `${totalMinutes - 240}분 초과`}입니다.`);
  if ((offCounts.Engine_Primary || 0) + (offCounts.Engine_Secondary || 0) < 2) risks.push("볼핸들러 역할이 부족합니다.");
  if ((offCounts.SpotUp_Spacer || 0) + (offCounts.Movement_Shooter || 0) < 2) risks.push("스페이싱 자원이 부족합니다.");
  if (!Object.keys(defCounts).some((k) => String(k).includes("Lowman") || String(k).includes("Anchor"))) risks.push("백라인 헬프/앵커 역할이 약합니다.");
  if (!risks.length) risks.push("주요 리스크 없음. 현재 밸런스가 안정적입니다.");

  let insight = "스타터와 벤치의 역할을 분리해 48분 내내 공격 방향성을 유지하세요.";
  if (totalMinutes > 240) insight = "총합 시간이 초과되었습니다. 벤치 유닛 분배를 우선 조정하세요.";
  else if (totalMinutes < 240) insight = "총합 시간이 부족합니다. 주축 볼핸들러/빅맨의 시간을 소폭 늘리세요.";

  return { starterMinutes, benchMinutes, totalMinutes, offCounts, defCounts, risks, insight };
}

function renderRoleCounts(container, counts) {
  const items = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([role, cnt]) => `<span class="tactics-chip"><strong>${tacticsRoleLabel(role)}</strong><em>${cnt}</em></span>`);
  container.innerHTML = items.length ? items.join("") : '<span class="tactics-chip muted">데이터 없음</span>';
}

function renderTacticsSummaryPanel() {
  const summary = computeTacticsSummary();
  els.tacticsMinutesTotal.textContent = `${summary.totalMinutes}분`;
  els.tacticsMinutesSplit.textContent = `${summary.starterMinutes} / ${summary.benchMinutes}`;
  els.tacticsStarterMinutes.textContent = `스타터 ${summary.starterMinutes}분`;

  const delta = 240 - summary.totalMinutes;
  if (delta === 0) {
    els.tacticsMinutesDelta.textContent = "목표 240분 달성";
    els.tacticsMinutesDelta.className = "kpi-sub is-good";
  } else if (delta > 0) {
    els.tacticsMinutesDelta.textContent = `${delta}분 부족`;
    els.tacticsMinutesDelta.className = "kpi-sub is-warn";
  } else {
    els.tacticsMinutesDelta.textContent = `${Math.abs(delta)}분 초과`;
    els.tacticsMinutesDelta.className = "kpi-sub is-danger";
  }

  renderRoleCounts(els.tacticsOffRoleCounts, summary.offCounts);
  renderRoleCounts(els.tacticsDefRoleCounts, summary.defCounts);
  els.tacticsRiskList.innerHTML = summary.risks.map((r) => `<li>${r}</li>`).join("");
  els.tacticsInsightCopy.textContent = summary.insight;
}

function bindLineupEvents() {
  document.querySelectorAll('.tactics-lineup-row').forEach((rowEl) => {
    const group = rowEl.dataset.group;
    const idx = Number(rowEl.dataset.idx || 0);
    rowEl.querySelectorAll('select, input').forEach((control) => {
      control.addEventListener('change', () => {
        const field = control.dataset.field;
        const target = group === 'starters' ? state.tacticsDraft.starters[idx] : state.tacticsDraft.rotation[idx];
        if (!target || !field) return;
        if (field === 'defenseRole') {
          if (!validateDefenseRoleUnique(control, control.value)) {
            alert('수비 역할은 중복 선택할 수 없습니다.');
            renderTacticsScreen();
            return;
          }
          target.defenseRole = control.value;
        } else if (field === 'minutes') {
          target.minutes = Math.max(0, Math.min(48, Number(control.value || 0)));
        } else {
          target[field] = control.value;
        }
        markTacticsDirty();
        renderTacticsSummaryPanel();
      });
    });
  });
}

function renderTacticsRosterList() {
  els.tacticsRosterList.innerHTML = (state.rosterRows || []).length
    ? state.rosterRows.map((r) => `<div class="tactics-roster-item"><strong>${r.name || r.player_id}</strong><span>${r.pos || "-"} · OVR ${Number(r.ovr || 0)}</span></div>`).join("")
    : '<p class="empty-copy">로스터 데이터가 없습니다.</p>';
}

function renderTacticsScreen() {
  if (!state.tacticsDraft) return;
  const teamName = state.selectedTeamName || TEAM_FULL_NAMES[state.selectedTeamId] || state.selectedTeamId || "팀";
  els.tacticsTeamTitle.textContent = `${teamName} 전술 보드`;
  els.tacticsCurrentDate.textContent = state.currentDate || "시즌 진행 중";

  const offMeta = TACTICS_OFFENSE_SCHEMES.find((x) => x.key === state.tacticsDraft.offenseScheme);
  const defMeta = TACTICS_DEFENSE_SCHEMES.find((x) => x.key === state.tacticsDraft.defenseScheme);
  const defRoles = getDefenseRolesForScheme(state.tacticsDraft.defenseScheme);

  els.tacticsOffenseCurrent.textContent = `현재 공격 스킴: ${offMeta?.label || state.tacticsDraft.offenseScheme}`;
  els.tacticsDefenseCurrent.textContent = `현재 수비 스킴: ${defMeta?.label || state.tacticsDraft.defenseScheme}`;
  els.tacticsStarters.innerHTML = state.tacticsDraft.starters.map((r, i) => buildLineupRowHtml('starters', i, r, defRoles)).join('');
  els.tacticsRotation.innerHTML = state.tacticsDraft.rotation.map((r, i) => buildLineupRowHtml('rotation', i, r, defRoles)).join('');

  renderTacticsRosterList();
  bindLineupEvents();
  renderTacticsSummaryPanel();
}

function autoBalanceTacticsMinutes() {
  if (!state.tacticsDraft) return;
  const rows = [...state.tacticsDraft.starters, ...state.tacticsDraft.rotation];
  const total = rows.reduce((sum, r) => sum + Number(r.minutes || 0), 0);
  let diff = 240 - total;
  if (diff === 0) return;
  const direction = diff > 0 ? 1 : -1;
  diff = Math.abs(diff);

  let guard = 0;
  while (diff > 0 && guard < 500) {
    for (let i = 0; i < rows.length && diff > 0; i += 1) {
      const cur = Number(rows[i].minutes || 0);
      const next = cur + direction;
      if (next < 0 || next > 48) continue;
      rows[i].minutes = next;
      diff -= 1;
    }
    guard += 1;
  }

  markTacticsDirty();
  renderTacticsScreen();
}

function saveTacticsDraft() {
  if (!state.tacticsDraft) return;
  state.tacticsSavedSnapshot = tacticsSnapshot();
  setTacticsDirty(false);
}


function renderMedicalEmpty(tbody, colSpan, text) {
  tbody.innerHTML = `<tr><td colspan="${colSpan}" class="schedule-empty">${text}</td></tr>`;
}

function riskTierClass(tier) {
  const t = String(tier || '').toUpperCase();
  if (t === 'HIGH') return 'status-danger';
  if (t === 'MEDIUM') return 'status-warn';
  return 'status-ok';
}

function formatSignedDelta(v) {
  const n = num(v, 0);
  if (!n) return { text: '지난 7일 대비 변동 없음', cls: '' };
  return {
    text: `지난 7일 대비 ${n > 0 ? '+' : ''}${n}`,
    cls: n > 0 ? 'pos' : 'neg',
  };
}

function renderMedicalHero(alerts = {}) {
  const p = alerts?.primary_alert_player;
  const load = alerts?.team_load_context || {};
  const level = String(alerts?.alert_level || 'info').toUpperCase();

  els.medicalAlertLevel.textContent = level;
  els.medicalAlertLevel.className = `medical-alert-badge ${level === 'CRITICAL' ? 'level-critical' : level === 'WARN' ? 'level-warn' : ''}`;

  if (!p) {
    els.medicalAlertText.textContent = '현재 주요 경고가 없습니다.';
    els.medicalAlertMeta.textContent = `다음 7일 경기 ${num(load?.next_7d_game_count, 0)}회 · B2B ${num(load?.next_7d_back_to_back_count, 0)}회`;
    return;
  }

  els.medicalAlertText.textContent = `${p.name || '-'} 리스크 ${p.risk_tier || '-'} (${num(p.risk_score, 0)})`;
  els.medicalAlertMeta.textContent = `${p.injury_status || '-'} · OUT ${p.out_until_date || '-'} / RETURNING ${p.returning_until_date || '-'} · 다음 7일 ${num(load?.next_7d_game_count, 0)}경기 (B2B ${num(load?.next_7d_back_to_back_count, 0)}회)`;
}

function renderMedicalTimeline(playerName, events) {
  els.medicalTimelineTitle.textContent = playerName ? `${playerName} 최근 부상 타임라인` : '워치리스트에서 선수를 선택하세요.';
  if (!events || !events.length) {
    els.medicalTimelineList.innerHTML = '<p class="empty-copy">최근 이벤트가 없습니다.</p>';
    return;
  }
  els.medicalTimelineList.innerHTML = events.map((e) => `
    <article class="medical-timeline-item">
      <p><strong>${e.date || '-'}</strong> · ${e.context || '-'}</p>
      <p>${e.body_part || '-'} / ${e.injury_type || '-'} / severity ${num(e.severity, 0)}</p>
      <p>OUT ~ ${e.out_until_date || '-'} · RETURNING ~ ${e.returning_until_date || '-'}</p>
    </article>
  `).join('');
}

function renderMedicalActionRecommendations(payload, playerName) {
  const items = payload?.recommendations || [];
  if (!items.length) {
    els.medicalActionList.innerHTML = '<p class="empty-copy">권고안이 없습니다.</p>';
    return;
  }
  els.medicalActionList.innerHTML = items.map((it) => {
    const d = it.expected_delta || {};
    const riskDelta = num(d.risk_score, 0);
    const stDelta = num(d.short_term_fatigue, 0);
    const ltDelta = num(d.long_term_fatigue, 0);
    const sharpDelta = num(d.sharpness, 0);
    return `
      <article class="medical-action-item">
        <strong>${it.label || it.action_id || '-'}</strong>
        <p>${playerName || '-'} 예상 변화 · Risk ${riskDelta > 0 ? '+' : ''}${riskDelta} · ST ${stDelta > 0 ? '+' : ''}${stDelta.toFixed(3)} · LT ${ltDelta > 0 ? '+' : ''}${ltDelta.toFixed(3)} · Sharp ${sharpDelta > 0 ? '+' : ''}${sharpDelta.toFixed(2)}</p>
      </article>
    `;
  }).join('');
}

function renderMedicalRiskCalendar(payload) {
  const days = payload?.days || [];
  if (!days.length) {
    els.medicalRiskCalendarList.innerHTML = '<p class="empty-copy">캘린더 데이터가 없습니다.</p>';
    return;
  }
  els.medicalRiskCalendarList.innerHTML = days.map((d) => `
    <article class="medical-day-card ${d.is_game_day ? 'is-game' : ''} ${d.is_back_to_back ? 'is-b2b' : ''}">
      <div class="date">${d.date || '-'}</div>
      <div class="meta">${d.is_game_day ? `vs/@ ${d.opponent_team_id || '-'}` : 'No Game'} · ${d.practice_session_type || '훈련 미정'}</div>
      <div class="badges">
        <span class="badge">HIGH ${num(d.high_risk_player_count, 0)}</span>
        <span class="badge">OUT ${num(d.out_player_count, 0)}</span>
        <span class="badge">RET ${num(d.returning_player_count, 0)}</span>
        <span class="badge">EVT ${num(d.injury_event_count, 0)}</span>
      </div>
    </article>
  `).join('');
}

async function loadMedicalPlayerContext(playerId, playerName) {
  if (!playerId || !state.selectedTeamId) return;
  setLoading(true, '선수 메디컬 컨텍스트를 불러오는 중...');
  try {
    const [timelinePayload, actionPayload] = await Promise.all([
      fetchJson(`/api/medical/team/${encodeURIComponent(state.selectedTeamId)}/players/${encodeURIComponent(playerId)}/timeline`),
      fetchJson(`/api/medical/team/${encodeURIComponent(state.selectedTeamId)}/players/${encodeURIComponent(playerId)}/action-recommendations`),
    ]);
    const resolvedName = playerName || timelinePayload?.player?.name || '-';
    renderMedicalTimeline(resolvedName, timelinePayload?.timeline?.events || []);
    renderMedicalActionRecommendations(actionPayload, resolvedName);
  } catch (e) {
    renderMedicalTimeline(playerName || '-', []);
    els.medicalActionList.innerHTML = `<p class="empty-copy">권고안 로딩 실패: ${e.message}</p>`;
  } finally {
    setLoading(false);
  }
}

function renderMedicalOverview(overview, alerts) {
  const summary = overview?.summary || {};
  const statusCounts = summary?.injury_status_counts || {};
  const riskCounts = summary?.risk_tier_counts || {};
  const watch = overview?.watchlists || {};
  const delta = alerts?.kpi_delta_7d || {};

  els.medicalAsOf.textContent = `기준일 ${overview?.as_of_date || '-'}`;
  els.medicalRosterCount.textContent = num(summary?.roster_count, 0);
  els.medicalOutCount.textContent = num(statusCounts?.OUT, 0);
  els.medicalReturningCount.textContent = `복귀 관리: ${num(statusCounts?.RETURNING, 0)}명`;
  els.medicalHighRiskCount.textContent = num(riskCounts?.HIGH, 0);
  els.medicalHealthFrustrationCount.textContent = num(summary?.health_frustration?.high_count, 0);

  const rosterDelta = formatSignedDelta(0);
  const outDelta = formatSignedDelta(delta?.out_count_delta);
  const hrDelta = formatSignedDelta(delta?.high_risk_count_delta);
  const healthDelta = formatSignedDelta(delta?.health_high_count_delta);
  els.medicalRosterDelta.textContent = rosterDelta.text;
  els.medicalOutDelta.textContent = outDelta.text;
  els.medicalOutDelta.className = `medical-delta ${outDelta.cls}`;
  els.medicalHighRiskDelta.textContent = hrDelta.text;
  els.medicalHighRiskDelta.className = `medical-delta ${hrDelta.cls}`;
  els.medicalHealthDelta.textContent = healthDelta.text;
  els.medicalHealthDelta.className = `medical-delta ${healthDelta.cls}`;

  const riskRows = watch?.highest_risk || [];
  if (!riskRows.length) {
    renderMedicalEmpty(els.medicalRiskBody, 6, '위험 데이터가 없습니다.');
  } else {
    els.medicalRiskBody.innerHTML = '';
    riskRows.forEach((r) => {
      const tr = document.createElement('tr');
      tr.className = 'roster-row';
      const riskScore = num(r.risk_score, 0);
      const reinjuryTotal = Object.values(r?.risk_inputs?.reinjury_count || {}).reduce((acc, v) => acc + num(v, 0), 0);
      tr.innerHTML = `
        <td>${r.name || '-'} <span class="schedule-opponent-name">${r.pos || '-'} · ${num(r.age, 0)}세</span></td>
        <td><span class="status-line ${riskTierClass(r.injury_status)}">${r.injury_status || '-'}</span></td>
        <td>
          <strong class="${riskTierClass(r.risk_tier)}">${r.risk_tier || '-'} (${riskScore})</strong>
          <div class="medical-risk-meter"><span style="width:${clamp(riskScore, 0, 100)}%"></span></div>
        </td>
        <td>${formatPercent(1 - num(r.condition?.short_term_fatigue, 0))} / ${formatPercent(1 - num(r.condition?.long_term_fatigue, 0))}</td>
        <td>${Math.round(num(r.condition?.sharpness, 0))}</td>
        <td>${reinjuryTotal}</td>
      `;
      tr.addEventListener('click', () => {
        state.medicalSelectedPlayerId = r.player_id;
        loadMedicalPlayerContext(r.player_id, r.name).catch((e) => alert(e.message));
      });
      els.medicalRiskBody.appendChild(tr);
    });
  }

  const injuredRows = watch?.currently_unavailable || [];
  els.medicalInjuredBody.innerHTML = injuredRows.length ? injuredRows.map((r) => `
    <tr>
      <td>${r.name || '-'} <span class="schedule-opponent-name">${r.pos || '-'}</span></td>
      <td><span class="status-line ${riskTierClass(r.recovery_status)}">${r.recovery_status || '-'}</span></td>
      <td>${r.injury_current?.body_part || '-'} (${r.injury_current?.injury_type || '-'})</td>
      <td>${r.injury_current?.out_until_date || '-'} ~ ${r.injury_current?.returning_until_date || '-'}</td>
    </tr>
  `).join('') : renderEmptyScheduleRow(4, '결장/복귀 관리 대상이 없습니다.');

  const healthRows = watch?.health_frustration_high || [];
  els.medicalHealthBody.innerHTML = healthRows.length ? healthRows.map((r) => `
    <tr>
      <td>${r.name || '-'} <span class="schedule-opponent-name">${r.pos || '-'}</span></td>
      <td>${num(r.health_frustration, 2)}</td>
      <td>${num(r.trade_request_level, 0)}</td>
      <td>${num(r.escalation_health, 0)}</td>
    </tr>
  `).join('') : renderEmptyScheduleRow(4, '건강 불만 상위 선수가 없습니다.');
}

async function showMedicalScreen() {
  if (!state.selectedTeamId) {
    alert('먼저 팀을 선택해주세요.');
    return;
  }
  setLoading(true, '메디컬 센터 데이터를 불러오는 중...');
  try {
    const [overview, alerts, calendar] = await Promise.all([
      fetchJson(`/api/medical/team/${encodeURIComponent(state.selectedTeamId)}/overview`),
      fetchJson(`/api/medical/team/${encodeURIComponent(state.selectedTeamId)}/alerts`).catch(() => ({})),
      fetchJson(`/api/medical/team/${encodeURIComponent(state.selectedTeamId)}/risk-calendar?days=14`).catch(() => ({ days: [] })),
    ]);
    state.medicalOverview = overview;
    const teamName = state.selectedTeamName || TEAM_FULL_NAMES[state.selectedTeamId] || state.selectedTeamId;
    els.medicalTitle.textContent = `${teamName} 메디컬 센터`;

    renderMedicalHero(alerts);
    renderMedicalOverview(overview, alerts);
    renderMedicalRiskCalendar(calendar);

    const primaryPlayerId = alerts?.primary_alert_player?.player_id;
    const primaryPlayerName = alerts?.primary_alert_player?.name;
    els.medicalAlertOpenPlayer.onclick = () => {
      if (!primaryPlayerId) return;
      state.medicalSelectedPlayerId = primaryPlayerId;
      loadMedicalPlayerContext(primaryPlayerId, primaryPlayerName).catch(() => {});
    };
    els.medicalAlertOpenAction.onclick = els.medicalAlertOpenPlayer.onclick;

    const first = primaryPlayerId ? { player_id: primaryPlayerId, name: primaryPlayerName } : (overview?.watchlists?.highest_risk || [])[0];
    if (first?.player_id) {
      state.medicalSelectedPlayerId = first.player_id;
      await loadMedicalPlayerContext(first.player_id, first.name);
    } else {
      renderMedicalTimeline(null, []);
      els.medicalActionList.innerHTML = '<p class="empty-copy">권고안이 없습니다.</p>';
    }

    activateScreen(els.medicalScreen);
  } finally {
    setLoading(false);
  }
}


async function showTacticsScreen() {
  if (!state.selectedTeamId) {
    alert('먼저 팀을 선택해주세요.');
    return;
  }
  setLoading(true, '전술 데이터를 불러오는 중...');
  try {
    const [detail, summary] = await Promise.all([
      fetchJson(`/api/team-detail/${encodeURIComponent(state.selectedTeamId)}`),
      fetchJson('/api/state/summary').catch(() => ({})),
    ]);
    state.rosterRows = detail.roster || [];
    if (!state.tacticsDraft) state.tacticsDraft = buildTacticsDraft(state.rosterRows);
    const currentDate = summary?.workflow_state?.league?.current_date || summary?.league?.current_date || '';
    if (currentDate) state.currentDate = currentDate;

    if (!state.tacticsSavedSnapshot) state.tacticsSavedSnapshot = tacticsSnapshot();
    setTacticsDirty(tacticsSnapshot() !== state.tacticsSavedSnapshot);

    renderSchemeOptions('offense');
    renderSchemeOptions('defense');
    renderTacticsScreen();
    activateScreen(els.tacticsScreen);
  } finally {
    setLoading(false);
  }
}

function toggleTacticsOptions(kind) {
  const target = kind === 'offense' ? els.tacticsOffenseOptions : els.tacticsDefenseOptions;
  const other = kind === 'offense' ? els.tacticsDefenseOptions : els.tacticsOffenseOptions;
  other.classList.add('hidden');
  target.classList.toggle('hidden');
}

els.newGameBtn.addEventListener("click", () => createNewGame().catch((e) => alert(e.message)));
els.continueBtn.addEventListener("click", () => continueGame().catch((e) => alert(e.message)));
els.myTeamBtn.addEventListener("click", () => showMyTeamScreen().catch((e) => alert(e.message)));
els.tacticsMenuBtn.addEventListener("click", () => showTacticsScreen().catch((e) => alert(e.message)));
els.nextGameTacticsBtn.addEventListener("click", () => showTacticsScreen().catch((e) => alert(e.message)));
els.scheduleBtn.addEventListener("click", () => showScheduleScreen().catch((e) => alert(e.message)));
els.scheduleBackBtn.addEventListener("click", () => showMainScreen());
if (els.scheduleHeroTacticsBtn) els.scheduleHeroTacticsBtn.addEventListener("click", () => showTacticsScreen().catch((e) => alert(e.message)));
bindScheduleFilters();
els.trainingMenuBtn.addEventListener("click", () => showTrainingScreen().catch((e) => alert(e.message)));
els.tacticsBackBtn.addEventListener("click", () => showMainScreen());
els.tacticsOffenseBtn.addEventListener("click", () => toggleTacticsOptions("offense"));
els.tacticsDefenseBtn.addEventListener("click", () => toggleTacticsOptions("defense"));
els.tacticsAutobalanceBtn?.addEventListener("click", () => autoBalanceTacticsMinutes());
els.tacticsSaveBtn?.addEventListener("click", () => saveTacticsDraft());
els.standingsMenuBtn.addEventListener("click", () => showStandingsScreen().catch((e) => alert(e.message)));
if (els.standingsConferenceToggle) {
  els.standingsConferenceToggle.addEventListener("click", (event) => {
    const btn = event.target.closest("button[data-conference]");
    if (!btn) return;
    state.standingsView.conference = btn.dataset.conference === "west" ? "west" : "east";
    renderStandingsTable();
  });
}
if (els.standingsSortKey) {
  els.standingsSortKey.addEventListener("change", () => {
    state.standingsView.sortKey = els.standingsSortKey.value || "pct";
    renderStandingsTable();
  });
}
if (els.standingsSortDirBtn) {
  els.standingsSortDirBtn.addEventListener("click", () => {
    state.standingsView.sortDir = state.standingsView.sortDir === "desc" ? "asc" : "desc";
    renderStandingsTable();
  });
}
if (els.standingsAdvancedToggle) {
  els.standingsAdvancedToggle.addEventListener("click", () => {
    state.standingsView.showAdvanced = !state.standingsView.showAdvanced;
    renderStandingsTable();
  });
}
els.collegeMenuBtn.addEventListener("click", () => showCollegeScreen().catch((e) => alert(e.message)));
els.medicalMenuBtn.addEventListener("click", () => showMedicalScreen().catch((e) => alert(e.message)));
els.trainingBackBtn.addEventListener("click", () => showMainScreen());
els.medicalBackBtn.addEventListener("click", () => showMainScreen());
els.standingsBackBtn.addEventListener("click", () => showMainScreen());
els.collegeBackBtn.addEventListener("click", () => showMainScreen());
els.collegeTabTeams.addEventListener("click", () => switchCollegeTab("teams"));
els.collegeTabLeaders.addEventListener("click", () => switchCollegeTab("leaders"));
els.collegeTabBigboard.addEventListener("click", () => switchCollegeTab("bigboard"));
els.collegeTabScouting.addEventListener("click", () => switchCollegeTab("scouting"));
els.collegeLeaderSort.addEventListener("change", () => {
  state.collegeLeadersSort = els.collegeLeaderSort.value || "pts";
  loadCollegeLeaders().catch((e) => alert(e.message));
});
els.collegeExpertSelect.addEventListener("change", () => {
  state.selectedCollegeExpertId = els.collegeExpertSelect.value || "";
  loadCollegeBigboard().catch((e) => alert(e.message));
});
els.collegeAssignBtn.addEventListener("click", async () => {
  const scoutId = els.collegeScoutSelect.value;
  const playerId = els.collegeScoutPlayerSelect.value;
  if (!scoutId || !playerId) {
    alert("스카우터와 선수를 선택하세요.");
    return;
  }
  await fetchJson("/api/scouting/assign", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ team_id: state.selectedTeamId, scout_id: scoutId, player_id: playerId, target_kind: "COLLEGE" })
  });
  await loadCollegeScouting();
  alert("스카우터를 배정했습니다. 리포트는 월말 진행 시 생성됩니다.");
});
els.collegeUnassignBtn.addEventListener("click", async () => {
  const scoutId = els.collegeScoutSelect.value;
  if (!scoutId) {
    alert("해제할 스카우터를 선택하세요.");
    return;
  }
  await fetchJson("/api/scouting/unassign", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ team_id: state.selectedTeamId, scout_id: scoutId })
  });
  await loadCollegeScouting();
  alert("배정을 해제했습니다.");
});
els.trainingTypeButtons.querySelectorAll("button[data-training-type]").forEach((btn) => {
  btn.addEventListener("click", () => renderTrainingDetail(btn.dataset.trainingType).catch((e) => alert(e.message)));
});
els.backToMainBtn.addEventListener("click", () => showMainScreen());
els.backToRosterBtn.addEventListener("click", () => activateScreen(els.myTeamScreen));

loadSavesStatus();

window.__debugRenderMyTeam = function __debugRenderMyTeam() {
  state.selectedTeamId = "BOS";
  state.selectedTeamName = "보스턴 셀틱스";
  state.rosterRows = [
    { player_id: "p1", name: "J. Tatum", pos: "SF", age: 27, height_in: 80, weight_lb: 210, salary: 34000000, short_term_stamina: 0.72, long_term_stamina: 0.86, sharpness: 89 },
    { player_id: "p2", name: "J. Brown", pos: "SG", age: 28, height_in: 78, weight_lb: 223, salary: 32000000, short_term_stamina: 0.51, long_term_stamina: 0.78, sharpness: 61 },
    { player_id: "p3", name: "K. Porzingis", pos: "C", age: 29, height_in: 87, weight_lb: 240, salary: 36000000, short_term_stamina: 0.33, long_term_stamina: 0.62, sharpness: 42 }
  ];
  els.myTeamTitle.textContent = `${state.selectedTeamName} 선수단`;
  renderRosterRows(state.rosterRows);
  els.playerDetailTitle.textContent = "선수 상세 정보";
  els.playerDetailContent.innerHTML = "";
  activateScreen(els.myTeamScreen);
};
