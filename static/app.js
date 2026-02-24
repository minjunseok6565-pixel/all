const appState = {
  teams: [],
  saves: [],
  session: null,
  openApi: null,
  endpoints: [],
  selectedEndpoint: null,
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

  const options = { method, headers: { 'Content-Type': 'application/json' } };
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
    offseasonView: ['오프시즌', '오프시즌 전체 파이프라인 실행'],
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
        <button data-save="${s.slot_id}">현재 상태 저장</button>
      </div>
    `;

    div.querySelector('[data-load]').addEventListener('click', () => loadGame(s.slot_id));
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
      if (ep.includes('/options/team/decide')) payload = { user_team_id: teamId, decisions: [] };

      await runSimple(ep, 'offseasonResult', 'POST', payload);
    });
    box.appendChild(btn);
  });
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

function bindAssistant() {
  $('saveApiKeyBtn').addEventListener('click', () => {
    const key = $('apiKeyInput').value.trim();
    localStorage.setItem('nba_sim_api_key', key);
    log('API key 저장 완료');
  });

  $('clearApiKeyBtn').addEventListener('click', () => {
    $('apiKeyInput').value = '';
    localStorage.removeItem('nba_sim_api_key');
    log('API key 삭제 완료');
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

  $('searchCollegePlayersBtn').addEventListener('click', () => {
    runSimple('/api/college/players', 'collegePlayersBox', 'GET', null, { q: $('collegeQuery').value.trim(), limit: 50 });
  });

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
    btn.addEventListener('click', async () => {
      try {
        const payload = parseJsonOrEmpty($('tradePayload').value);
        await runSimple(btn.dataset.endpoint, 'tradeResult', 'POST', payload);
      } catch (e) {
        $('tradeResult').textContent = pretty({ error: String(e) });
      }
    });
  });

  $('reloadOpenApiBtn').addEventListener('click', async () => {
    try { await loadOpenApi(); } catch (e) { log('OpenAPI 재로딩 실패', { error: String(e) }); }
  });
  $('endpointFilter').addEventListener('input', (e) => renderEndpointList(e.target.value));
  $('sendStudioBtn').addEventListener('click', sendStudioRequest);
}

async function init() {
  bindMenu();
  bindEvents();
  bindAssistant();
  buildOffseasonButtons();

  $('apiKeyInput').value = localStorage.getItem('nba_sim_api_key') || '';

  try { await loadTeams(); } catch (e) { log('/api/teams 로딩 실패', { error: String(e) }); }
  try { await loadSaves(); } catch (e) { log('/api/game/saves 로딩 실패', { error: String(e) }); }
  try { await loadOpenApi(); } catch (e) { log('/openapi.json 로딩 실패', { error: String(e) }); }

  renderSession();
  log('프론트엔드 초기화 완료');
}

init();
