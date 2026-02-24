const appState = {
  teams: [],
  saves: [],
  session: null,
  openApi: null,
  endpoints: [],
  selectedEndpoint: null,
  pendingTeamOptions: [],
  lastNegotiationSessionId: null,
  lastNegotiationMode: 'SIGN_FA',
  adminToken: '',
};

const OFFSEASON_ENDPOINTS = [
  '/api/season/enter-offseason',
  '/api/offseason/college/finalize',
  '/api/offseason/contracts/process',
  '/api/offseason/retirement/preview',
  '/api/offseason/retirement/process',
  '/api/offseason/training/apply-growth',
  '/api/offseason/options/team/pending',
  '/api/offseason/options/team/decide',
  '/api/offseason/draft/lottery',
  '/api/offseason/draft/settle',
  '/api/offseason/draft/combine',
  '/api/offseason/draft/workouts',
  '/api/offseason/draft/interviews',
  '/api/offseason/draft/withdrawals',
  '/api/offseason/draft/selections/auto',
  '/api/offseason/draft/selections/pick',
  '/api/offseason/draft/apply',
  '/api/season/start-regular-season',
];

const $ = (id) => document.getElementById(id);

function log(message, payload = null) {
  const line = `[${new Date().toISOString()}] ${message}`;
  const body = payload ? `${line}\n${JSON.stringify(payload, null, 2)}` : line;
  $('logBox').textContent = `${body}\n\n${$('logBox').textContent}`;
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

async function request(path, method = 'GET', data = null, query = null) {
  const url = new URL(path, window.location.origin);
  if (query) {
    Object.entries(query).forEach(([k, v]) => {
      if (v !== undefined && v !== null && String(v).trim() !== '') url.searchParams.set(k, v);
    });
  }

  const headers = { 'Content-Type': 'application/json' };
  const adminToken = (appState.adminToken || '').trim();
  if (adminToken) headers['X-Admin-Token'] = adminToken;
  const options = { method, headers };
  if (data !== null) options.body = JSON.stringify(data);

  const res = await fetch(url.toString(), options);
  const json = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(json.detail || `${method} ${path} failed (${res.status})`);
    err.status = res.status;
    err.payload = json;
    throw err;
  }
  return json;
}

function currentTeamId() {
  return appState.session?.user_team_id || $('userTeamSelect').value;
}

function setSession(sessionObj) {
  appState.session = sessionObj;
  renderSession();
}

function renderSession() {
  const s = appState.session;
  const badge = $('sessionBadge');
  if (!s) {
    $('sessionInfo').innerHTML = '<div>세션 없음</div>';
    badge.textContent = '세션 없음';
    return;
  }

  $('sessionInfo').innerHTML = `
    <div><strong>슬롯:</strong> ${s.slot_name || '-'}</div>
    <div><strong>slot_id:</strong> ${s.slot_id || '-'}</div>
    <div><strong>유저팀:</strong> ${s.user_team_id || '-'}</div>
    <div><strong>시즌:</strong> ${s.season_year || '-'}</div>
    <div><strong>현재 날짜:</strong> ${s.current_date || '-'}</div>
    <div><strong>저장 버전:</strong> ${s.save_version || '-'}</div>
  `;
  badge.textContent = `${s.slot_name || s.slot_id} | ${s.user_team_id || 'TEAM?'}`;
}

function bindMenu() {
  const titles = {
    launcherView: ['게임 런처', '새 게임 생성 또는 저장 불러오기'],
    dashboardView: ['대시보드', '리그/세션 현황 요약'],
    seasonView: ['시즌 운영', '리그 진행/일정/순위/리더'],
    rosterView: ['로스터/팀', '팀 상세와 대학 선수 정보'],
    transactionsView: ['계약/트레이드', '선수 계약 및 트레이드 실행'],
    operationsView: ['운영 도구', '시뮬/트레이닝/스카우팅/포스트시즌/에이전시'],
    offseasonView: ['오프시즌', '오프시즌 전체 파이프라인 실행'],
    insightsView: ['인사이트/뉴스', 'Practice/뉴스/리포트/드래프트/트레이드 협상'],
    assistantView: ['AI 어시스턴트', 'API 키 검증 및 챗 요청'],
    apiStudioView: ['API 스튜디오', '전체 엔드포인트 범용 실행 도구'],
  };

  document.querySelectorAll('.menu-item').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.menu-item').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const view = btn.dataset.view;
      document.querySelectorAll('.view').forEach((v) => v.classList.remove('active'));
      $(view).classList.add('active');
      $('viewTitle').textContent = titles[view][0];
      $('viewSubtitle').textContent = titles[view][1];
    });
  });
}

function renderTeamOptions() {
  const select = $('userTeamSelect');
  select.innerHTML = '';
  appState.teams.forEach((t) => {
    const opt = document.createElement('option');
    const id = t.team_id || t.id;
    opt.value = id;
    opt.textContent = `${id} - ${t.team_name || t.name || 'Unknown'}`;
    select.appendChild(opt);
  });
}

function renderTeamCards() {
  const box = $('teamCards');
  box.innerHTML = '';
  appState.teams.forEach((t) => {
    const id = t.team_id || t.id;
    const div = document.createElement('div');
    div.className = 'team-card';
    div.innerHTML = `
      <strong>${t.team_name || t.name || id}</strong>
      <div class="muted">${id}</div>
      <button data-team-id="${id}">팀 상세</button>
    `;
    div.querySelector('button').addEventListener('click', async () => {
      try {
        const data = await request(`/api/team-detail/${id}`);
        $('myTeamDetailBox').textContent = pretty(data);
        activateView('rosterView');
      } catch (e) {
        log('팀 상세 조회 실패', { id, error: String(e), detail: e.payload || null });
      }
    });
    box.appendChild(div);
  });
}

function renderSaves() {
  const wrap = $('saveSlots');
  wrap.innerHTML = '';
  if (!appState.saves.length) {
    wrap.innerHTML = '<div class="muted">저장 슬롯이 없습니다.</div>';
    return;
  }

  appState.saves.forEach((s) => {
    const div = document.createElement('div');
    div.className = 'save-item';
    div.innerHTML = `
      <div><strong>${s.slot_name || s.slot_id}</strong></div>
      <div class="muted">ID: ${s.slot_id}</div>
      <div class="muted">시즌: ${s.season_year || '-'}, 날짜: ${s.current_date || '-'}</div>
      <div class="inline-actions">
        <button data-load="${s.slot_id}">로드</button>
        <button data-detail="${s.slot_id}">상세</button>
        <button data-save="${s.slot_id}">현재 상태 저장</button>
      </div>
    `;

    div.querySelector('[data-load]').addEventListener('click', () => loadGame(s.slot_id));
    div.querySelector('[data-detail]').addEventListener('click', () => loadSaveDetail(s.slot_id, true));
    div.querySelector('[data-save]').addEventListener('click', () => saveGame(s.slot_id));
    wrap.appendChild(div);
  });
}

function activateView(viewId) {
  document.querySelector(`.menu-item[data-view="${viewId}"]`)?.click();
}

async function loadTeams() {
  const data = await request('/api/teams');
  appState.teams = Array.isArray(data) ? data : data.teams || [];
  renderTeamOptions();
  renderTeamCards();
}

async function loadSaves() {
  const data = await request('/api/game/saves');
  appState.saves = data.slots || [];
  renderSaves();
}

async function createNewGame(e) {
  e.preventDefault();
  const payload = {
    slot_name: $('slotName').value.trim(),
    slot_id: $('slotId').value.trim() || null,
    season_year: $('seasonYear').value ? Number($('seasonYear').value) : null,
    user_team_id: $('userTeamSelect').value,
  };

  const data = await request('/api/game/new', 'POST', payload);
  setSession(data);
  await loadSaves();
  activateView('dashboardView');
  log('새 게임 생성 완료', data);
}

async function loadGame(slotId) {
  const data = await request('/api/game/load', 'POST', { slot_id: slotId, strict: true });
  setSession(data);
  activateView('dashboardView');
  log('게임 로드 완료', data);
}

async function saveGame(slotId = null) {
  const sid = slotId || appState.session?.slot_id;
  if (!sid) return log('저장 실패: slot_id가 없습니다.');
  const data = await request('/api/game/save', 'POST', { slot_id: sid, save_name: 'manual_save' });
  log('게임 저장 완료', data);
  await loadSaves();
}

async function loadSaveDetail(slotId, strict = true) {
  if (!slotId) return log('슬롯 상세 조회 실패: slot_id가 없습니다.');
  await runSimple(`/api/game/saves/${encodeURIComponent(String(slotId))}`, 'saveDetailBox', 'GET', null, { strict: strict ? 'true' : 'false' });
}

async function runSimple(path, targetEl, method = 'GET', body = null, query = null) {
  try {
    const data = await request(path, method, body, query);
    $(targetEl).textContent = pretty(data);
  } catch (e) {
    $(targetEl).textContent = pretty({ error: String(e), detail: e.payload || null });
    log(`${path} 호출 실패`, { error: String(e), detail: e.payload || null });
  }
}

function parseJsonOrEmpty(value) {
  if (!value || !value.trim()) return {};
  return JSON.parse(value);
}


function parseListInput(value) {
  if (!value || !value.trim()) return [];
  return value.split(/[\n,]/g).map((x) => x.trim()).filter(Boolean);
}

function normalizeTeamInput(value) {
  return value?.trim() || currentTeamId();
}

function renderTwoWaySlots(summary) {
  const box = $('twoWaySlotsBox');
  if (!box) return;
  const players = Array.isArray(summary?.players) ? summary.players : [];
  const maxSlots = Number(summary?.max_two_way_slots || 3);

  const cards = [];
  for (let i = 0; i < maxSlots; i += 1) {
    const p = players[i];
    if (!p) {
      cards.push(`<div class="two-way-slot-card empty">슬롯 ${i + 1}: 비어 있음</div>`);
      continue;
    }
    cards.push(`
      <div class="two-way-slot-card">
        <div><strong>슬롯 ${i + 1}</strong> · ${p.name || p.player_id}</div>
        <div class="muted">player_id: ${p.player_id}</div>
        <div>남은 출전 가능 경기: <strong>${p.games_remaining}</strong> / ${p.game_limit}</div>
      </div>
    `);
  }

  box.innerHTML = `
    <div class="muted">사용 슬롯: ${summary?.used_two_way_slots || 0} / ${maxSlots}</div>
    ${cards.join('')}
  `;
}

async function loadTwoWaySummary() {
  const teamId = normalizeTeamInput(null);
  if (!teamId) return log('투웨이 슬롯 조회 실패: team_id가 필요합니다.');
  const data = await request(`/api/two-way/summary/${teamId}`);
  renderTwoWaySlots(data);
}

function findTwoWayTradeRuleViolation(payload, currentTeam) {
  const err = payload?.error || payload?.detail?.error || null;
  if (!err) return null;
  const msg = String(err.message || payload?.detail?.message || '').toLowerCase();
  const details = err.details || payload?.detail?.details || {};
  if (!msg.includes('two-way trade-time limit exceeded')) return null;

  const targetTeam = String(details.team_id || '').toUpperCase();
  const userTeam = String(currentTeam || '').toUpperCase();
  const roleText = targetTeam && userTeam && targetTeam === userTeam ? '받는 입장' : '보내는 입장';
  return `투웨이 트레이드 규정 위반으로 트레이드 불가 (${roleText}). team=${targetTeam || '-'}, 보유 예정 투웨이=${details.two_way_count ?? '-'} (최대 ${details.trade_time_max_two_way ?? 2})`;
}

function clearTwoWayTradeRuleMessage() {
  const el = $('twoWayTradeRuleMsg');
  if (!el) return;
  el.hidden = true;
  el.textContent = '';
}

function showTwoWayTradeRuleMessage(text) {
  const el = $('twoWayTradeRuleMsg');
  if (!el) return;
  el.hidden = false;
  el.textContent = text;
}

async function runTradeAction(endpoint) {
  clearTwoWayTradeRuleMessage();
  try {
    const payload = parseJsonOrEmpty($('tradePayload').value);
    if (endpoint === '/api/trade/evaluate') {
      const teamId = normalizeTeamInput($('tradeEvaluateTeamId')?.value);
      await runSimple(endpoint, 'tradeResult', 'POST', { ...payload, team_id: teamId });
      return;
    }
    await runSimple(endpoint, 'tradeResult', 'POST', payload);
  } catch (e) {
    const errorObj = { error: String(e), detail: e.payload || null };
    $('tradeResult').textContent = pretty(errorObj);
    const violation = findTwoWayTradeRuleViolation(e.payload || {}, normalizeTeamInput(null));
    if (violation) showTwoWayTradeRuleMessage(violation);
  }
}

async function runDraftWorkouts() {
  const teamId = normalizeTeamInput($('draftWorkoutsTeamId')?.value);
  if (!teamId) return log('워크아웃 실행 실패: team_id가 필요합니다.');
  const maxInvites = Number($('draftWorkoutsMaxInvites')?.value || 12);
  const rngSeedRaw = $('draftWorkoutsRngSeed')?.value?.trim();
  const invited = parseListInput($('draftWorkoutProspects')?.value || '');
  await runSimple('/api/offseason/draft/workouts', 'draftToolsResult', 'POST', {
    team_id: teamId,
    max_invites: maxInvites,
    invited_prospect_temp_ids: invited,
    rng_seed: rngSeedRaw ? Number(rngSeedRaw) : null,
  });
}

async function loadDraftInterviewQuestions() {
  await runSimple('/api/offseason/draft/interviews/questions', 'draftToolsResult');
}

async function runDraftInterviews() {
  const teamId = normalizeTeamInput($('draftWorkoutsTeamId')?.value);
  if (!teamId) return log('인터뷰 실행 실패: team_id가 필요합니다.');
  let interviews = [];
  try {
    const parsed = JSON.parse($('draftInterviewsPayload')?.value || '[]');
    interviews = Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    return log('인터뷰 실행 실패: interviews JSON 파싱 오류', { error: String(e) });
  }
  for (const item of interviews) {
    const qids = Array.isArray(item.selected_question_ids) ? item.selected_question_ids.filter(Boolean) : [];
    if (qids.length !== 3) return log('인터뷰 실행 실패: 각 항목의 selected_question_ids는 정확히 3개여야 합니다.');
  }
  const rngSeedRaw = $('draftInterviewsRngSeed')?.value?.trim();
  await runSimple('/api/offseason/draft/interviews', 'draftToolsResult', 'POST', {
    team_id: teamId,
    interviews,
    rng_seed: rngSeedRaw ? Number(rngSeedRaw) : null,
  });
}

async function runSingleSimulation() {
  const home = $('simHomeTeamId')?.value.trim();
  const away = $('simAwayTeamId')?.value.trim();
  if (!home || !away) return log('경기 시뮬레이션 실패: home_team_id, away_team_id가 필요합니다.');
  await runSimple('/api/simulate-game', 'simulateGameResult', 'POST', {
    home_team_id: home,
    away_team_id: away,
    game_date: $('simGameDate')?.value.trim() || null,
  });
}

async function loadTeamTraining() {
  const teamId = normalizeTeamInput($('trainingTeamId')?.value);
  if (!teamId) return log('팀 트레이닝 조회 실패: team_id가 필요합니다.');
  await runSimple(`/api/training/team/${teamId}`, 'trainingResult');
}

async function setTeamTraining() {
  const teamId = normalizeTeamInput($('trainingTeamId')?.value);
  if (!teamId) return log('팀 트레이닝 저장 실패: team_id가 필요합니다.');
  let payload = {};
  try { payload = parseJsonOrEmpty($('teamTrainingPayload').value); } catch (e) { return log('팀 트레이닝 JSON 오류', { error: String(e) }); }
  await runSimple('/api/training/team/set', 'trainingResult', 'POST', { team_id: teamId, ...payload });
}

async function loadPlayerTraining() {
  const playerId = $('trainingPlayerId')?.value.trim();
  if (!playerId) return log('선수 트레이닝 조회 실패: player_id가 필요합니다.');
  await runSimple(`/api/training/player/${playerId}`, 'trainingResult');
}

async function setPlayerTraining() {
  const playerId = $('trainingPlayerId')?.value.trim();
  if (!playerId) return log('선수 트레이닝 저장 실패: player_id가 필요합니다.');
  let payload = {};
  try { payload = parseJsonOrEmpty($('playerTrainingPayload').value); } catch (e) { return log('선수 트레이닝 JSON 오류', { error: String(e) }); }
  await runSimple('/api/training/player/set', 'trainingResult', 'POST', { player_id: playerId, ...payload });
}

async function loadScouts() {
  const teamId = normalizeTeamInput($('scoutingTeamId')?.value);
  if (!teamId) return log('스카우트 조회 실패: team_id가 필요합니다.');
  await runSimple(`/api/scouting/scouts/${teamId}`, 'scoutingResult');
}

async function loadScoutingReports() {
  const teamId = normalizeTeamInput($('scoutingTeamId')?.value);
  if (!teamId) return log('리포트 조회 실패: team_id가 필요합니다.');
  await runSimple('/api/scouting/reports', 'scoutingResult', 'GET', null, { team_id: teamId });
}

async function scoutingAssign() {
  try {
    const payload = parseJsonOrEmpty($('scoutingAssignPayload').value);
    await runSimple('/api/scouting/assign', 'scoutingResult', 'POST', payload);
  } catch (e) { log('스카우팅 assign JSON 오류', { error: String(e) }); }
}

async function scoutingUnassign() {
  try {
    const payload = parseJsonOrEmpty($('scoutingAssignPayload').value);
    await runSimple('/api/scouting/unassign', 'scoutingResult', 'POST', payload);
  } catch (e) { log('스카우팅 unassign JSON 오류', { error: String(e) }); }
}

async function loadPostseasonState() { await runSimple('/api/postseason/state', 'postseasonResult'); }
async function loadPostseasonField() { await runSimple('/api/postseason/field', 'postseasonResult'); }
async function resetPostseason() { await runSimple('/api/postseason/reset', 'postseasonResult', 'POST', {}); }
async function setupPostseason() {
  const teamId = normalizeTeamInput($('postseasonTeamId')?.value);
  if (!teamId) return log('포스트시즌 세팅 실패: my_team_id가 필요합니다.');
  await runSimple('/api/postseason/setup', 'postseasonResult', 'POST', {
    my_team_id: teamId,
    use_random_field: $('postseasonRandomField')?.checked || false,
  });
}
async function playInMyTeam() { await runSimple('/api/postseason/play-in/my-team-game', 'postseasonResult', 'POST', {}); }
async function advanceMyTeamPlayoff() { await runSimple('/api/postseason/playoffs/advance-my-team-game', 'postseasonResult', 'POST', {}); }
async function autoAdvanceRound() { await runSimple('/api/postseason/playoffs/auto-advance-round', 'postseasonResult', 'POST', {}); }

async function loadAgencyEvents() { await runSimple('/api/agency/events', 'agencyResult'); }
async function loadAgencyTeamEvents() {
  const teamId = normalizeTeamInput($('agencyTeamId')?.value);
  if (!teamId) return log('팀 에이전시 이벤트 조회 실패: team_id가 필요합니다.');
  await runSimple(`/api/agency/team/${teamId}/events`, 'agencyResult');
}
async function loadAgencyPlayer() {
  const playerId = $('agencyPlayerId')?.value.trim();
  if (!playerId) return log('에이전시 선수 조회 실패: player_id가 필요합니다.');
  await runSimple(`/api/agency/player/${playerId}`, 'agencyResult');
}
async function respondAgencyEvent() {
  try {
    const payload = parseJsonOrEmpty($('agencyRespondPayload').value);
    await runSimple('/api/agency/events/respond', 'agencyResult', 'POST', payload);
  } catch (e) { log('에이전시 응답 JSON 오류', { error: String(e) }); }
}
async function applyAgencyActions() {
  try {
    const payload = parseJsonOrEmpty($('agencyApplyPayload').value);
    await runSimple('/api/agency/actions/apply', 'agencyResult', 'POST', payload);
  } catch (e) { log('에이전시 액션 JSON 오류', { error: String(e) }); }
}

function buildOffseasonButtons() {
  const box = $('offseasonButtons');
  box.innerHTML = '';
  OFFSEASON_ENDPOINTS.forEach((ep) => {
    const btn = document.createElement('button');
    btn.textContent = ep;
    btn.addEventListener('click', async () => {
      const teamId = currentTeamId();
      let payload = {};
      if (ep.includes('/contracts/process')) payload = { user_team_id: teamId };
      if (ep.includes('/options/team/pending')) payload = { user_team_id: teamId };
      if (ep.includes('/draft/workouts')) payload = { team_id: teamId };
      if (ep.includes('/draft/interviews')) payload = { team_id: teamId, interviews: [] };
      if (ep.includes('/options/team/decide')) {
        return log('TEAM 옵션은 아래 전용 UI에서 선택 후 제출해주세요.');
      }
      if (ep.includes('/draft/selections/auto')) {
        return log('드래프트 자동선택은 아래 드래프트 진행 도구를 사용해주세요.');
      }
      if (ep.includes('/draft/selections/pick')) {
        return log('단일 픽 선택은 prospect_temp_id 입력 후 아래 도구에서 실행해주세요.');
      }

      await runSimple(ep, 'offseasonResult', 'POST', payload);
    });
    box.appendChild(btn);
  });
}

function renderPendingTeamOptions(items = []) {
  appState.pendingTeamOptions = Array.isArray(items) ? items : [];
  const box = $('teamOptionPendingList');
  if (!box) return;

  box.innerHTML = '';
  if (!appState.pendingTeamOptions.length) {
    box.innerHTML = '<div class="muted">처리할 TEAM 옵션이 없습니다.</div>';
    return;
  }

  appState.pendingTeamOptions.forEach((item, idx) => {
    const div = document.createElement('div');
    div.className = 'save-item';
    const contractId = item.contract_id || item.id || `contract_${idx}`;
    const playerName = item.player_name || item.player_id || 'Unknown Player';
    const seasonYear = item.season_year || '-';
    const salary = item.salary ?? item.option_salary ?? '-';

    div.innerHTML = `
      <div><strong>${playerName}</strong></div>
      <div class="muted">contract_id: ${contractId}</div>
      <div class="muted">season: ${seasonYear}, salary: ${salary}</div>
      <div class="inline-form">
        <label><input type="radio" name="opt_${idx}" value="EXERCISE" checked /> 행사</label>
        <label><input type="radio" name="opt_${idx}" value="DECLINE" /> 거절</label>
      </div>
    `;
    div.dataset.contractId = String(contractId);
    box.appendChild(div);
  });
}

async function loadPendingTeamOptions() {
  const teamId = currentTeamId();
  if (!teamId) return log('TEAM 옵션 조회 실패: 팀 ID가 없습니다.');

  try {
    const data = await request('/api/offseason/options/team/pending', 'POST', { user_team_id: teamId });
    renderPendingTeamOptions(data.pending_team_options || []);
    $('teamOptionResult').textContent = pretty(data);
  } catch (e) {
    $('teamOptionResult').textContent = pretty({ error: String(e), detail: e.payload || null });
    log('TEAM 옵션 pending 조회 실패', { error: String(e), detail: e.payload || null });
  }
}

async function submitTeamOptionDecisions() {
  const teamId = currentTeamId();
  if (!teamId) return log('TEAM 옵션 제출 실패: 팀 ID가 없습니다.');

  const cards = Array.from(document.querySelectorAll('#teamOptionPendingList .save-item'));
  const decisions = cards.map((card, idx) => {
    const checked = card.querySelector(`input[name="opt_${idx}"]:checked`);
    return {
      contract_id: card.dataset.contractId,
      decision: checked?.value || 'EXERCISE',
    };
  }).filter((d) => d.contract_id);

  if (!decisions.length) {
    return log('TEAM 옵션 제출 실패: 먼저 Pending 목록을 불러오고 결정을 선택하세요.');
  }

  await runSimple('/api/offseason/options/team/decide', 'teamOptionResult', 'POST', {
    user_team_id: teamId,
    decisions,
  });
  await loadPendingTeamOptions();
}

async function runDraftAutoPick() {
  const teamId = currentTeamId();
  if (!teamId) return log('드래프트 자동 진행 실패: 팀 ID가 없습니다.');
  const allowAutopick = $('allowAutopickUserTeam')?.checked || false;
  const payload = allowAutopick
    ? { allow_autopick_user_team: true }
    : { stop_on_user_controlled_team_ids: [teamId], allow_autopick_user_team: false };

  await runSimple('/api/offseason/draft/selections/auto', 'draftToolsResult', 'POST', payload);
}

async function runDraftPickOne() {
  const prospectTempId = $('draftPickProspectId')?.value.trim();
  if (!prospectTempId) {
    return log('단일 픽 선택 실패: prospect_temp_id를 입력하세요.');
  }

  await runSimple('/api/offseason/draft/selections/pick', 'draftToolsResult', 'POST', {
    prospect_temp_id: prospectTempId,
    source: 'draft_user_ui',
  });
}

async function loadDraftBundle() {
  await runSimple('/api/offseason/draft/bundle', 'draftToolsResult');
}



async function loadFreeAgentCandidates() {
  const q = $('freeAgentQuery')?.value?.trim() || '';
  const limitRaw = $('freeAgentLimit')?.value?.trim();
  const limit = limitRaw ? Number(limitRaw) : 200;
  await runSimple('/api/contracts/free-agents', 'freeAgentListResult', 'GET', null, { q, limit });
}

async function startContractNegotiation() {
  const teamId = $('negoTeamId').value.trim() || currentTeamId();
  const playerId = $('negoPlayerId').value.trim();
  const mode = $('negoMode').value;
  const validDays = Number($('negoValidDays').value || 7);

  if (!teamId || !playerId) {
    return log('협상 시작 실패: team_id, player_id를 입력하세요.');
  }

  try {
    const isTwoWay = mode === 'TWO_WAY';
    const endpoint = isTwoWay
      ? '/api/contracts/two-way/negotiation/start'
      : '/api/contracts/negotiation/start';
    const payload = isTwoWay
      ? { team_id: teamId, player_id: playerId, valid_days: validDays }
      : { team_id: teamId, player_id: playerId, mode: 'SIGN_FA', valid_days: validDays };

    const data = await request(endpoint, 'POST', payload);
    const sid = data.session_id || data.session?.session_id || null;
    if (sid) {
      appState.lastNegotiationSessionId = sid;
      appState.lastNegotiationMode = mode;
      $('negotiationSessionId').value = sid;
    }
    $('negotiationResult').textContent = pretty(data);
  } catch (e) {
    $('negotiationResult').textContent = pretty({ error: String(e), detail: e.payload || null });
    log('협상 시작 실패', { error: String(e), detail: e.payload || null });
  }
}

function getNegotiationSessionId() {
  return $('negotiationSessionId').value.trim() || appState.lastNegotiationSessionId;
}


function currentNegotiationMode() {
  const sidInput = $('negotiationSessionId')?.value?.trim();
  if (sidInput && sidInput === appState.lastNegotiationSessionId) {
    return appState.lastNegotiationMode || $('negoMode')?.value || 'SIGN_FA';
  }
  return $('negoMode')?.value || 'SIGN_FA';
}


async function sendNegotiationOffer() {
  const sid = getNegotiationSessionId();
  if (!sid) return log('오퍼 전송 실패: session_id가 없습니다.');

  const mode = currentNegotiationMode();
  if (mode === 'TWO_WAY') {
    return log('투웨이 협상은 오퍼 전송을 지원하지 않습니다. 수락/거절 버튼을 사용하세요.');
  }

  let offer;
  try {
    offer = parseJsonOrEmpty($('negotiationOfferPayload').value);
  } catch (e) {
    return log('오퍼 전송 실패: offer JSON 파싱 오류', { error: String(e) });
  }

  await runSimple('/api/contracts/negotiation/offer', 'negotiationResult', 'POST', {
    session_id: sid,
    offer,
  });
}

async function acceptNegotiationCounter() {
  const sid = getNegotiationSessionId();
  if (!sid) return log('카운터 수락 실패: session_id가 없습니다.');
  const mode = currentNegotiationMode();
  if (mode === 'TWO_WAY') {
    return log('투웨이 협상은 카운터 수락을 지원하지 않습니다. 투웨이 수락 버튼을 사용하세요.');
  }
  await runSimple('/api/contracts/negotiation/accept-counter', 'negotiationResult', 'POST', { session_id: sid });
}

async function decideTwoWayNegotiation(accept) {
  const sid = getNegotiationSessionId();
  if (!sid) return log('투웨이 의사결정 실패: session_id가 없습니다.');
  await runSimple('/api/contracts/two-way/negotiation/decision', 'negotiationResult', 'POST', {
    session_id: sid,
    accept: Boolean(accept),
  });
}

async function commitNegotiation() {
  const sid = getNegotiationSessionId();
  if (!sid) return log('협상 커밋 실패: session_id가 없습니다.');
  const mode = currentNegotiationMode();
  if (mode === 'TWO_WAY') {
    await runSimple('/api/contracts/two-way/negotiation/commit', 'negotiationResult', 'POST', { session_id: sid });
    return;
  }
  await runSimple('/api/contracts/negotiation/commit', 'negotiationResult', 'POST', { session_id: sid });
}

async function cancelNegotiation() {
  const sid = getNegotiationSessionId();
  if (!sid) return log('협상 취소 실패: session_id가 없습니다.');
  const mode = currentNegotiationMode();
  if (mode === 'TWO_WAY') {
    return log('투웨이 협상 취소 API는 제공되지 않습니다. 필요 시 새 협상을 시작하세요.');
  }
  await runSimple('/api/contracts/negotiation/cancel', 'negotiationResult', 'POST', { session_id: sid });
}

async function loadCollegeMeta() {
  await runSimple('/api/college/meta', 'collegeDetailBox');
}

async function loadCollegeTeams() {
  await runSimple('/api/college/teams', 'collegeDetailBox');
}

async function loadCollegeTeamDetail() {
  const teamId = $('collegeTeamDetailId')?.value?.trim();
  if (!teamId) return log('대학 팀 상세 조회 실패: college_team_id를 입력하세요.');
  await runSimple(`/api/college/team-detail/${encodeURIComponent(teamId)}`, 'collegeDetailBox');
}

async function loadCollegePlayerDetail() {
  const playerId = $('collegePlayerDetailId')?.value?.trim();
  if (!playerId) return log('대학 선수 상세 조회 실패: player_id를 입력하세요.');
  await runSimple(`/api/college/player/${encodeURIComponent(playerId)}`, 'collegeDetailBox');
}

async function loadCollegeDraftPool() {
  const yearRaw = $('collegeDraftPoolYear')?.value?.trim();
  if (!yearRaw) return log('드래프트 풀 조회 실패: draft_year를 입력하세요.');
  const draftYear = Number(yearRaw);
  if (!Number.isInteger(draftYear) || draftYear < 1900) return log('드래프트 풀 조회 실패: 유효한 draft_year가 필요합니다.');
  await runSimple(`/api/college/draft-pool/${draftYear}`, 'collegeDetailBox');
}

async function loadOpenApi() {
  const doc = await request('/openapi.json');
  appState.openApi = doc;
  const endpoints = [];
  Object.entries(doc.paths || {}).forEach(([path, methods]) => {
    Object.entries(methods).forEach(([method, spec]) => {
      const m = method.toUpperCase();
      if (m !== 'GET' && m !== 'POST') return;
      endpoints.push({
        key: `${m} ${path}`,
        method: m,
        path,
        summary: spec.summary || '',
        description: spec.description || '',
        requestBody: spec.requestBody || null,
        parameters: spec.parameters || [],
      });
    });
  });
  appState.endpoints = endpoints.sort((a, b) => a.path.localeCompare(b.path));
  renderEndpointList();
  log('OpenAPI 로딩 완료', { endpoints: appState.endpoints.length });
}

function renderEndpointList(filter = '') {
  const box = $('endpointList');
  box.innerHTML = '';
  const f = filter.trim().toLowerCase();
  const list = appState.endpoints.filter((ep) => !f || ep.key.toLowerCase().includes(f));

  list.forEach((ep) => {
    const div = document.createElement('div');
    div.className = `endpoint-item ${appState.selectedEndpoint?.key === ep.key ? 'active' : ''}`;
    div.innerHTML = `<span class="badge ${ep.method.toLowerCase()}">${ep.method}</span>${ep.path}`;
    div.addEventListener('click', () => selectEndpoint(ep));
    box.appendChild(div);
  });
}

function selectEndpoint(ep) {
  appState.selectedEndpoint = ep;
  $('studioTitle').textContent = `${ep.method} ${ep.path}`;
  $('studioMeta').textContent = ep.summary || ep.description || '설명 없음';

  const pathParams = (ep.path.match(/\{[^}]+\}/g) || []).map((x) => x.slice(1, -1));
  const queryParams = ep.parameters.filter((p) => p.in === 'query').map((p) => p.name);

  const payloadTemplate = {
    pathParams: Object.fromEntries(pathParams.map((k) => [k, ''])),
    query: Object.fromEntries(queryParams.map((k) => [k, ''])),
    body: {},
  };

  if (ep.path.includes('{team_id}')) payloadTemplate.pathParams.team_id = currentTeamId() || '';
  if (ep.path.includes('{player_id}')) payloadTemplate.pathParams.player_id = '';

  $('studioPayload').value = pretty(payloadTemplate);
  renderEndpointList($('endpointFilter').value);
}

async function sendStudioRequest() {
  const ep = appState.selectedEndpoint;
  if (!ep) return;

  try {
    const parsed = parseJsonOrEmpty($('studioPayload').value);
    let path = ep.path;
    Object.entries(parsed.pathParams || {}).forEach(([k, v]) => {
      path = path.replace(`{${k}}`, encodeURIComponent(String(v || '')));
    });

    const data = await request(
      path,
      ep.method,
      ep.method === 'POST' ? (parsed.body || {}) : null,
      parsed.query || null,
    );
    $('studioResult').textContent = pretty(data);
  } catch (e) {
    $('studioResult').textContent = pretty({ error: String(e), detail: e.payload || null });
    log('API 스튜디오 요청 실패', { error: String(e), detail: e.payload || null });
  }
}


function currentSeasonYearInput(id) {
  const raw = $(id)?.value?.trim();
  return raw ? Number(raw) : null;
}

async function loadPracticePlan() {
  const teamId = normalizeTeamInput($('practiceTeamId')?.value);
  if (!teamId) return log('Practice 플랜 조회 실패: team_id 필요');
  await runSimple(`/api/practice/team/${teamId}/plan`, 'practiceResult', 'GET', null, { season_year: currentSeasonYearInput('practiceSeasonYear') });
}
async function setPracticePlan() {
  const teamId = normalizeTeamInput($('practiceTeamId')?.value);
  if (!teamId) return log('Practice 플랜 저장 실패: team_id 필요');
  let payload = {};
  try { payload = parseJsonOrEmpty($('practicePlanPayload')?.value || '{}'); } catch (e) { return log('Practice plan JSON 오류', { error: String(e) }); }
  await runSimple('/api/practice/team/plan/set', 'practiceResult', 'POST', { team_id: teamId, season_year: currentSeasonYearInput('practiceSeasonYear'), ...payload });
}
async function loadPracticeSession() {
  const teamId = normalizeTeamInput($('practiceTeamId')?.value);
  const dateIso = $('practiceDateIso')?.value?.trim();
  if (!teamId || !dateIso) return log('Practice 세션 조회 실패: team_id/date_iso 필요');
  await runSimple(`/api/practice/team/${teamId}/session`, 'practiceResult', 'GET', null, { date_iso: dateIso, season_year: currentSeasonYearInput('practiceSeasonYear') });
}
async function setPracticeSession() {
  const teamId = normalizeTeamInput($('practiceTeamId')?.value);
  const dateIso = $('practiceDateIso')?.value?.trim();
  if (!teamId || !dateIso) return log('Practice 세션 저장 실패: team_id/date_iso 필요');
  let payload = {};
  try { payload = parseJsonOrEmpty($('practiceSessionPayload')?.value || '{}'); } catch (e) { return log('Practice session JSON 오류', { error: String(e) }); }
  await runSimple('/api/practice/team/session/set', 'practiceResult', 'POST', { team_id: teamId, season_year: currentSeasonYearInput('practiceSeasonYear'), date_iso: dateIso, ...payload });
}
async function listPracticeSessions() {
  const teamId = normalizeTeamInput($('practiceTeamId')?.value);
  if (!teamId) return log('Practice 세션 목록 실패: team_id 필요');
  await runSimple(`/api/practice/team/${teamId}/sessions`, 'practiceResult', 'GET', null, { season_year: currentSeasonYearInput('practiceSeasonYear') });
}

async function generateWeeklyNews() {
  await runSimple('/api/news/week', 'newsReportResult', 'POST', { apiKey: $('apiKeyInput')?.value?.trim() || '' });
}
async function generatePlayoffNews() {
  await runSimple('/api/news/playoffs', 'newsReportResult', 'POST', {});
}
async function generateSeasonReport() {
  const teamId = normalizeTeamInput($('seasonReportTeamId')?.value);
  if (!teamId) return log('시즌 리포트 실패: user_team_id 필요');
  await runSimple('/api/season-report', 'newsReportResult', 'POST', { apiKey: $('apiKeyInput')?.value?.trim() || '', user_team_id: teamId });
}

async function runDraftWatchRecompute() {
  let payload = {};
  try { payload = parseJsonOrEmpty($('draftWatchPayload')?.value || '{}'); } catch (e) { return log('Draft watch JSON 오류', { error: String(e) }); }
  await runSimple('/api/college/draft-watch/recompute', 'draftInsightResult', 'POST', payload);
}
async function loadDraftExperts() {
  await runSimple('/api/offseason/draft/experts', 'draftInsightResult');
}
async function loadDraftBigboardExpert() {
  await runSimple('/api/offseason/draft/bigboard/expert', 'draftInsightResult', 'GET', null, { expert_id: $('draftBigboardExpertId')?.value?.trim(), viewer_team_id: $('draftBigboardTeamId')?.value?.trim() || null });
}

async function startTradeNegotiation() {
  const userTeamId = normalizeTeamInput($('tradeNegoUserTeamId')?.value);
  const otherTeamId = $('tradeNegoOtherTeamId')?.value?.trim();
  if (!userTeamId || !otherTeamId) return log('트레이드 협상 시작 실패: 팀 ID 필요');
  const data = await request('/api/trade/negotiation/start', 'POST', { user_team_id: userTeamId, other_team_id: otherTeamId });
  $('tradeNegoResult').textContent = pretty(data);
  $('tradeNegoSessionId').value = data?.session?.session_id || '';
}
async function commitTradeNegotiation() {
  const sid = $('tradeNegoSessionId')?.value?.trim();
  if (!sid) return log('트레이드 협상 커밋 실패: session_id 필요');
  let payload = {};
  try { payload = parseJsonOrEmpty($('tradeNegoDealPayload')?.value || '{}'); } catch (e) { return log('트레이드 deal JSON 오류', { error: String(e) }); }
  await runSimple('/api/trade/negotiation/commit', 'tradeNegoResult', 'POST', { session_id: sid, ...(payload || {}) });
}

function bindAssistant() {
  $('saveApiKeyBtn').addEventListener('click', () => {
    const key = $('apiKeyInput').value.trim();
    localStorage.setItem('nba_sim_api_key', key);
    appState.adminToken = $('adminTokenInput')?.value?.trim() || '';
    localStorage.setItem('nba_sim_admin_token', appState.adminToken);
    log('API/Admin 토큰 저장 완료');
  });

  $('clearApiKeyBtn').addEventListener('click', () => {
    $('apiKeyInput').value = '';
    localStorage.removeItem('nba_sim_api_key');
    if ($('adminTokenInput')) $('adminTokenInput').value = '';
    appState.adminToken = '';
    localStorage.removeItem('nba_sim_admin_token');
    log('API/Admin 토큰 삭제 완료');
  });

  $('validateApiKeyBtn').addEventListener('click', async () => {
    await runSimple('/api/validate-key', 'apiKeyResult', 'POST', { apiKey: $('apiKeyInput').value.trim() });
  });

  $('askAssistantBtn').addEventListener('click', async () => {
    const body = {
      apiKey: $('apiKeyInput').value.trim(),
      userMessage: $('assistantPrompt').value,
      mainPrompt: 'You are the GM assistant.',
      context: { team_id: currentTeamId(), session: appState.session || null },
    };
    await runSimple('/api/chat-main', 'assistantResult', 'POST', body);
  });
}

function bindEvents() {
  $('newGameForm').addEventListener('submit', async (e) => {
    try { await createNewGame(e); } catch (err) { log('새 게임 생성 실패', { error: String(err), detail: err.payload || null }); }
  });
  $('refreshSavesBtn').addEventListener('click', async () => {
    try { await loadSaves(); } catch (e) { log('세이브 목록 조회 실패', { error: String(e) }); }
  });
  $('quickRefreshBtn').addEventListener('click', async () => {
    try {
      await Promise.all([loadTeams(), loadSaves(), loadOpenApi()]);
      log('전체 새로고침 완료');
    } catch (e) {
      log('전체 새로고침 실패', { error: String(e), detail: e.payload || null });
    }
  });

  $('loadStateSummaryBtn').addEventListener('click', () => runSimple('/api/state/summary', 'stateSummaryBox'));
  $('loadStandingsBtn').addEventListener('click', () => runSimple('/api/standings', 'seasonDataBox'));
  $('loadLeadersBtn').addEventListener('click', () => runSimple('/api/stats/leaders', 'seasonDataBox'));
  $('loadPlayoffLeadersBtn').addEventListener('click', () => runSimple('/api/stats/playoffs/leaders', 'seasonDataBox'));

  $('loadScheduleBtn').addEventListener('click', () => {
    const tid = currentTeamId();
    runSimple(`/api/team-schedule/${tid}`, 'teamScheduleBox');
  });

  $('advanceLeagueBtn').addEventListener('click', () => {
    runSimple('/api/advance-league', 'advanceResult', 'POST', {
      target_date: $('advanceDate').value.trim(),
      user_team_id: currentTeamId(),
      apiKey: $('apiKeyInput').value.trim() || null,
    });
  });

  $('loadMyTeamDetailBtn').addEventListener('click', () => runSimple(`/api/team-detail/${currentTeamId()}`, 'myTeamDetailBox'));
  $('loadRosterSummaryBtn').addEventListener('click', () => runSimple(`/api/roster-summary/${currentTeamId()}`, 'rosterSummaryBox'));
  $('loadTwoWaySummaryBtn')?.addEventListener('click', () => {
    loadTwoWaySummary().catch((e) => log('투웨이 슬롯 조회 실패', { error: String(e), detail: e.payload || null }));
  });

  $('searchCollegePlayersBtn').addEventListener('click', () => {
    runSimple('/api/college/players', 'collegePlayersBox', 'GET', null, { q: $('collegeQuery').value.trim(), limit: 50 });
  });
  $('loadCollegeMetaBtn')?.addEventListener('click', loadCollegeMeta);
  $('loadCollegeTeamsBtn')?.addEventListener('click', loadCollegeTeams);
  $('loadCollegeTeamDetailBtn')?.addEventListener('click', loadCollegeTeamDetail);
  $('loadCollegePlayerDetailBtn')?.addEventListener('click', loadCollegePlayerDetail);
  $('loadCollegeDraftPoolBtn')?.addEventListener('click', loadCollegeDraftPool);

  document.querySelectorAll('.tx-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      try {
        const payload = parseJsonOrEmpty($('contractPayload').value);
        await runSimple(btn.dataset.endpoint, 'contractResult', 'POST', payload);
      } catch (e) {
        $('contractResult').textContent = pretty({ error: String(e) });
      }
    });
  });

  document.querySelectorAll('.trade-btn').forEach((btn) => {
    btn.addEventListener('click', async () => runTradeAction(btn.dataset.endpoint));
  });

  $('reloadOpenApiBtn').addEventListener('click', async () => {
    try { await loadOpenApi(); } catch (e) { log('OpenAPI 재로딩 실패', { error: String(e) }); }
  });
  $('endpointFilter').addEventListener('input', (e) => renderEndpointList(e.target.value));
  $('sendStudioBtn').addEventListener('click', sendStudioRequest);

  $('loadPendingOptionsBtn')?.addEventListener('click', loadPendingTeamOptions);
  $('submitTeamOptionDecisionsBtn')?.addEventListener('click', submitTeamOptionDecisions);
  $('draftAutoPickBtn')?.addEventListener('click', runDraftAutoPick);
  $('draftPickOneBtn')?.addEventListener('click', runDraftPickOne);
  $('loadDraftBundleBtn')?.addEventListener('click', loadDraftBundle);
  $('runDraftWorkoutsBtn')?.addEventListener('click', runDraftWorkouts);
  $('loadInterviewQuestionsBtn')?.addEventListener('click', loadDraftInterviewQuestions);
  $('runDraftInterviewsBtn')?.addEventListener('click', runDraftInterviews);

  $('simulateGameBtn')?.addEventListener('click', runSingleSimulation);
  $('loadTeamTrainingBtn')?.addEventListener('click', loadTeamTraining);
  $('setTeamTrainingBtn')?.addEventListener('click', setTeamTraining);
  $('loadPlayerTrainingBtn')?.addEventListener('click', loadPlayerTraining);
  $('setPlayerTrainingBtn')?.addEventListener('click', setPlayerTraining);

  $('loadScoutsBtn')?.addEventListener('click', loadScouts);
  $('loadScoutingReportsBtn')?.addEventListener('click', loadScoutingReports);
  $('scoutingAssignBtn')?.addEventListener('click', scoutingAssign);
  $('scoutingUnassignBtn')?.addEventListener('click', scoutingUnassign);

  $('loadPostseasonStateBtn')?.addEventListener('click', loadPostseasonState);
  $('loadPostseasonFieldBtn')?.addEventListener('click', loadPostseasonField);
  $('resetPostseasonBtn')?.addEventListener('click', resetPostseason);
  $('setupPostseasonBtn')?.addEventListener('click', setupPostseason);
  $('playInMyTeamBtn')?.addEventListener('click', playInMyTeam);
  $('advanceMyTeamPlayoffBtn')?.addEventListener('click', advanceMyTeamPlayoff);
  $('autoAdvanceRoundBtn')?.addEventListener('click', autoAdvanceRound);

  $('loadAgencyEventsBtn')?.addEventListener('click', loadAgencyEvents);
  $('loadAgencyTeamEventsBtn')?.addEventListener('click', loadAgencyTeamEvents);
  $('loadAgencyPlayerBtn')?.addEventListener('click', loadAgencyPlayer);
  $('agencyRespondBtn')?.addEventListener('click', respondAgencyEvent);
  $('agencyApplyBtn')?.addEventListener('click', applyAgencyActions);

  $('loadPracticePlanBtn')?.addEventListener('click', loadPracticePlan);
  $('setPracticePlanBtn')?.addEventListener('click', setPracticePlan);
  $('loadPracticeSessionBtn')?.addEventListener('click', loadPracticeSession);
  $('setPracticeSessionBtn')?.addEventListener('click', setPracticeSession);
  $('listPracticeSessionsBtn')?.addEventListener('click', listPracticeSessions);

  $('generateWeeklyNewsBtn')?.addEventListener('click', generateWeeklyNews);
  $('generatePlayoffNewsBtn')?.addEventListener('click', generatePlayoffNews);
  $('generateSeasonReportBtn')?.addEventListener('click', generateSeasonReport);

  $('runDraftWatchBtn')?.addEventListener('click', runDraftWatchRecompute);
  $('loadDraftExpertsBtn')?.addEventListener('click', loadDraftExperts);
  $('loadDraftBigboardBtn')?.addEventListener('click', loadDraftBigboardExpert);

  $('startTradeNegoBtn')?.addEventListener('click', async () => { try { await startTradeNegotiation(); } catch (e) { log('트레이드 협상 시작 실패', { error: String(e), detail: e.payload || null }); } });
  $('commitTradeNegoBtn')?.addEventListener('click', commitTradeNegotiation);

  $('loadFreeAgentsBtn')?.addEventListener('click', loadFreeAgentCandidates);
  $('startNegotiationBtn')?.addEventListener('click', startContractNegotiation);
  $('sendOfferBtn')?.addEventListener('click', sendNegotiationOffer);
  $('acceptCounterBtn')?.addEventListener('click', acceptNegotiationCounter);
  $('twoWayAcceptBtn')?.addEventListener('click', () => decideTwoWayNegotiation(true));
  $('twoWayRejectBtn')?.addEventListener('click', () => decideTwoWayNegotiation(false));
  $('cancelNegotiationBtn')?.addEventListener('click', cancelNegotiation);
  $('commitNegotiationBtn')?.addEventListener('click', commitNegotiation);
  $('useLastSessionBtn')?.addEventListener('click', () => {
    if (appState.lastNegotiationSessionId) {
      $('negotiationSessionId').value = appState.lastNegotiationSessionId;
      if ($('negoMode')) $('negoMode').value = appState.lastNegotiationMode || 'SIGN_FA';
      log('최근 협상 session_id를 불러왔습니다.', { session_id: appState.lastNegotiationSessionId, mode: appState.lastNegotiationMode || 'SIGN_FA' });
    } else {
      log('최근 협상 session_id가 없습니다. 먼저 협상을 시작하세요.');
    }
  });
}

async function init() {
  bindMenu();
  bindEvents();
  bindAssistant();
  buildOffseasonButtons();

  $('apiKeyInput').value = localStorage.getItem('nba_sim_api_key') || '';
  appState.adminToken = localStorage.getItem('nba_sim_admin_token') || '';
  if ($('adminTokenInput')) $('adminTokenInput').value = appState.adminToken;

  try { await loadTeams(); } catch (e) { log('/api/teams 로딩 실패', { error: String(e) }); }
  try { await loadSaves(); } catch (e) { log('/api/game/saves 로딩 실패', { error: String(e) }); }
  try { await loadOpenApi(); } catch (e) { log('/openapi.json 로딩 실패', { error: String(e) }); }

  renderSession();
  log('프론트엔드 초기화 완료');
}

init();
