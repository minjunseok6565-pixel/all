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
};

const els = {
  startScreen: document.getElementById("start-screen"),
  teamScreen: document.getElementById("team-screen"),
  mainScreen: document.getElementById("main-screen"),
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
  trainingMenuBtn: document.getElementById("training-menu-btn"),
  standingsMenuBtn: document.getElementById("standings-menu-btn"),
  trainingScreen: document.getElementById("training-screen"),
  standingsScreen: document.getElementById("standings-screen"),
  trainingBackBtn: document.getElementById("training-back-btn"),
  standingsBackBtn: document.getElementById("standings-back-btn"),
  teamTrainingTabBtn: document.getElementById("team-training-tab-btn"),
  playerTrainingTabBtn: document.getElementById("player-training-tab-btn"),
  trainingCalendarGrid: document.getElementById("training-calendar-grid"),
  trainingTypeButtons: document.getElementById("training-type-buttons"),
  trainingDetailPanel: document.getElementById("training-detail-panel"),
  standingsEastBody: document.getElementById("standings-east-body"),
  standingsWestBody: document.getElementById("standings-west-body"),
  backToMainBtn: document.getElementById("back-to-main-btn"),
  backToRosterBtn: document.getElementById("back-to-roster-btn"),
  rosterBody: document.getElementById("my-team-roster-body"),
  playerDetailTitle: document.getElementById("player-detail-title"),
  playerDetailPanel: document.getElementById("player-detail-panel"),
  playerDetailContent: document.getElementById("player-detail-content"),
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
  [els.startScreen, els.teamScreen, els.mainScreen, els.myTeamScreen, els.playerDetailScreen, els.trainingScreen, els.standingsScreen].forEach((screen) => {
    const active = screen === target;
    screen.classList.toggle("active", active);
    screen.setAttribute("aria-hidden", active ? "false" : "true");
  });
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
    els.nextGameDatetime.textContent = `${gameDate} ${randomTipoffTime()}`;
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

  teams.forEach((team) => {
    const id = String(team.team_id || "").toUpperCase();
    const fullName = TEAM_FULL_NAMES[id] || id;
    const card = document.createElement("button");
    card.className = "team-card";
    card.type = "button";
    card.innerHTML = `<strong>${fullName}</strong><small>${team.conference || ""} · ${team.division || ""}</small>`;
    card.addEventListener("click", () => {
      confirmTeamSelection(id, fullName).catch((e) => alert(e.message));
    });
    els.teamGrid.appendChild(card);
  });
}


function formatSignedDiff(value) {
  const n = Number(value || 0);
  if (!Number.isFinite(n)) return "0.0";
  if (Math.abs(n) < 0.05) return "0.0";
  return `${n > 0 ? "+" : ""}${n.toFixed(1)}`;
}

function renderStandingsRows(tbody, rows) {
  tbody.innerHTML = "";
  (rows || []).forEach((row) => {
    const tr = document.createElement("tr");
    const teamId = String(row?.team_id || "").toUpperCase();
    const diff = Number(row?.diff || 0);
    const diffClass = diff > 0 ? "standings-diff-positive" : diff < 0 ? "standings-diff-negative" : "";
    tr.innerHTML = `
      <td>${row?.rank ?? "-"}</td>
      <td class="standings-team-cell">${TEAM_FULL_NAMES[teamId] || teamId || "-"}</td>
      <td>${row?.wins ?? 0}</td>
      <td>${row?.losses ?? 0}</td>
      <td>${row?.pct || ".000"}</td>
      <td>${row?.gb_display ?? "-"}</td>
      <td>${row?.home || "0-0"}</td>
      <td>${row?.away || "0-0"}</td>
      <td>${row?.div || "0-0"}</td>
      <td>${row?.conf || "0-0"}</td>
      <td>${Number(row?.ppg || 0).toFixed(1)}</td>
      <td>${Number(row?.opp_ppg || 0).toFixed(1)}</td>
      <td class="${diffClass}">${formatSignedDiff(row?.diff)}</td>
      <td>${row?.strk || "-"}</td>
      <td>${row?.l10 || "0-0"}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function showStandingsScreen() {
  setLoading(true, "순위 데이터를 불러오는 중입니다...");
  try {
    const payload = await fetchJson("/api/standings/table");
    state.standingsData = payload;
    renderStandingsRows(els.standingsEastBody, payload?.east || []);
    renderStandingsRows(els.standingsWestBody, payload?.west || []);
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

els.newGameBtn.addEventListener("click", () => createNewGame().catch((e) => alert(e.message)));
els.continueBtn.addEventListener("click", () => continueGame().catch((e) => alert(e.message)));
els.myTeamBtn.addEventListener("click", () => showMyTeamScreen().catch((e) => alert(e.message)));
els.trainingMenuBtn.addEventListener("click", () => showTrainingScreen().catch((e) => alert(e.message)));
els.standingsMenuBtn.addEventListener("click", () => showStandingsScreen().catch((e) => alert(e.message)));
els.trainingBackBtn.addEventListener("click", () => showMainScreen());
els.standingsBackBtn.addEventListener("click", () => showMainScreen());
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
