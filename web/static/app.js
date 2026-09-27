const $ = id => document.getElementById(id);
const columns = 'ABCDEFGHJKLMNOPQRSTUVWXYZ';
const rules = {freestyle: '自由五子棋', standard: '标准五子棋', renju: '连珠规则'};
const notes = {freestyle: '自由规则下，五子或长连均获胜。', standard: '恰好五子获胜，长连不计胜。', renju: '黑棋三三、四四、长连禁手判负；恰好五子优先。白棋五子或长连获胜。'};
let state = null;
let catalog = null;
let pending = false;
let formKey = '';
let online = false;
let notice = '';

function coordinate(action, size) {
  return columns[action % size] + (size - Math.floor(action / size));
}

function connection(value) {
  online = value;
  $('connection').textContent = value ? '本地引擎已连接' : '连接中断，正在重连';
  $('connection-dot').classList.toggle('online', value);
}

async function api(path, body) {
  const response = await fetch(path, body ? {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  } : {});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `请求失败 (${response.status})`);
  return result;
}

function accept(next) {
  if (state && next.instance === state.instance && next.version < state.version) return;
  state = next;
  render();
}

async function command(operation, values = {}) {
  if (!state || pending || state.busy || !online) return;
  pending = true;
  notice = '';
  render();
  try {
    accept(await api(`/api/${operation}`, {version: state.version, ...values}));
  } catch (error) {
    notice = error.message;
    try { accept(await api('/api/state')); } catch { connection(false); }
  } finally {
    pending = false;
    render();
  }
}

function renderBoard(game, disabled) {
  const size = game?.board_size || Number($('size').value) || catalog.default_size;
  const step = 100 / (size + 1);
  const end = step * size;
  let drawing = `<svg viewBox="0 0 100 100" aria-hidden="true"><g stroke="#927b59" stroke-width=".12">`;
  for (let i = 1; i <= size; i++) {
    const p = i * step;
    drawing += `<path d="M${step},${p}H${end} M${p},${step}V${end}"/>`;
  }
  drawing += '</g><g fill="#796b52" font-family="system-ui" font-size="1.55" text-anchor="middle">';
  for (let i = 0; i < size; i++) {
    drawing += `<text x="${(i + 1) * step}" y="${100 - step * .37}">${columns[i]}</text><text x="${step * .38}" y="${(i + 1) * step + .55}">${size - i}</text>`;
  }
  drawing += '</g><g fill="#877454">';
  if (size % 2 === 1) {
    const stars = size >= 11 ? [3, (size + 1) / 2, size - 2] : [(size + 1) / 2];
    for (const x of stars) for (const y of stars) {
      if (x === y || x + y === size + 1) drawing += `<circle cx="${x * step}" cy="${y * step}" r=".36"/>`;
    }
  }
  drawing += '</g></svg>';
  const numbers = new Map((game?.moves || []).map((action, i) => [action, i + 1]));
  for (let action = 0; action < size * size; action++) {
    const stone = game?.board[action] || 0;
    const number = numbers.get(action);
    const last = number && number === game.turn;
    const color = stone === 1 ? 'black' : 'white';
    const label = coordinate(action, size) + (stone ? `，${stone === 1 ? '黑' : '白'}棋，第 ${number} 手` : '，空位');
    drawing += `<button class="board-point" data-action="${action}" aria-label="${label}" style="left:${(action % size + 1) * step}%;top:${(Math.floor(action / size) + 1) * step}%;width:${step * .96}%;height:${step * .96}%" ${disabled || stone ? 'disabled' : ''}>${stone ? `<span class="stone ${color} ${last ? 'last' : ''}">${number}</span>` : ''}</button>`;
  }
  $('board').innerHTML = drawing;
  $('board').classList.toggle('white-turn', state.human === -1);
  $('board').classList.toggle('hide-numbers', !$('numbers').checked);
}

function renderHeatmap(id, field, maximumId, analysis) {
  const size = analysis.board_size;
  const candidates = new Map(analysis.candidates.map(candidate => [candidate.action, candidate]));
  const maximum = Math.max(...analysis.candidates.map(candidate => candidate[field]));
  let html = '<span></span>';
  for (let x = 0; x < size; x++) html += `<span class="heat-axis">${columns[x]}</span>`;
  for (let y = 0; y < size; y++) {
    html += `<span class="heat-axis">${size - y}</span>`;
    for (let x = 0; x < size; x++) {
      const action = y * size + x;
      const stone = analysis.board[action];
      const candidate = candidates.get(action);
      const value = candidate ? candidate[field] : 0;
      const ratio = maximum > 0 ? value / maximum : 0;
      const color = value > 0 ? `hsl(${160 - 125 * ratio} 55% ${94 - 38 * ratio}%)` : '#edf0e8';
      const detail = `${coordinate(action, size)} · ` + (stone ? `${stone === 1 ? '黑' : '白'}棋` :
        `网络先验 ${(candidate.network_prior * 100).toFixed(3)}% · 访问 ${candidate.visits} 次（${(candidate.visit_policy * 100).toFixed(3)}%） · 选择权重 ${(candidate.selection_weight * 100).toFixed(3)}%`);
      const percent = value === 0 ? '·' : value < .001 ? '&lt;0.1' : (value * 100).toFixed(1);
      html += `<div class="heat-cell ${action === analysis.action ? 'chosen-point' : ''}" data-action="${action}" data-value="${value}" data-detail="${detail}" role="img" aria-label="${detail}" title="${detail}" tabindex="0" style="background:${color}">${stone ? `<i class="heat-stone ${stone === 1 ? 'black' : 'white'}"></i>` : `<span>${percent}</span>`}</div>`;
    }
  }
  const map = $(id);
  map.style.gridTemplateColumns = `16px repeat(${size}, minmax(0, 1fr))`;
  map.classList.toggle('dense', size > 15);
  map.innerHTML = html;
  $(maximumId).textContent = `${(maximum * 100).toFixed(2)}%`;
}

function renderHeatmaps() {
  const analysis = state.analysis;
  $('heatmap-empty').hidden = Boolean(analysis);
  $('heatmap-data').hidden = !analysis;
  $('search-map-kind').disabled = !analysis;
  $('heatmap-position').textContent = analysis ? `第 ${analysis.turn + 1} 手落子前 · ${analysis.player === 1 ? '黑' : '白'}方` : '等待 AI 搜索';
  $('heatmap-detail').textContent = '悬停或聚焦点位，查看数值；描边标记 AI 的实际落点。';
  if (!analysis) {
    $('prior-map').replaceChildren();
    $('search-map').replaceChildren();
    return;
  }
  renderHeatmap('prior-map', 'network_prior', 'prior-max', analysis);
  renderHeatmap('search-map', $('search-map-kind').value, 'search-max', analysis);
}

function render() {
  if (!state || !catalog) return;
  const game = state.game;
  const busy = pending || state.busy || !online;
  const key = game ? [state.model, game.board_size, state.rule, state.human, state.visits].join('|') : '';
  if (key && key !== formKey) {
    $('model').value = state.model;
    $('size').value = game.board_size;
    $('rule').value = state.rule;
    $('visits').value = state.visits;
    document.querySelector(`input[name="human"][value="${state.human}"]`).checked = true;
    formKey = key;
  }
  for (const input of $('settings').elements) input.disabled = busy;
  const humanBlack = state.human === 1;
  $('human-stone').className = `player-stone ${humanBlack ? 'black' : 'white'}`;
  $('ai-stone').className = `player-stone ${humanBlack ? 'white' : 'black'}`;
  $('human-label').textContent = humanBlack ? '执黑 · 先行' : '执白 · 后行';
  $('ai-label').textContent = humanBlack ? '执白 · 后行' : '执黑 · 先行';
  $('rule-badge').textContent = rules[state.rule];
  $('rule-note').textContent = notes[state.rule];
  $('move-count').textContent = `第 ${game?.turn || 0} 手`;
  $('new-game').innerHTML = `${game ? '重新开始一局' : '开始新局'} <span>↗</span>`;
  $('undo').disabled = busy || !game?.moves.some((_, i) => (i % 2 === 0 ? 1 : -1) === state.human);
  $('retry').hidden = busy || !game || game.finished || game.player === state.human;
  $('error').textContent = notice || state.error || '';
  $('error').hidden = !$('error').textContent;
  renderBoard(game, busy || !game || game.finished || game.player !== state.human);
  renderStatus();
  const history = $('history');
  history.replaceChildren();
  if (!game?.turn) {
    const empty = document.createElement('span');
    empty.className = 'empty-inline';
    empty.textContent = '每一步，都从这里开始。';
    history.append(empty);
  } else {
    game.moves.forEach((action, i) => {
      const item = document.createElement('span');
      item.className = 'history-item';
      item.innerHTML = `<small>${i + 1}</small><i class="tiny-stone ${i % 2 === 0 ? 'black' : 'white'}"></i><span>${coordinate(action, game.board_size)}</span>`;
      history.append(item);
    });
    history.scrollLeft = history.scrollWidth;
  }
  $('history-count').textContent = game?.turn ? `${game.turn} 手` : '尚未落子';
  const analysis = state.analysis;
  renderHeatmaps();
  $('analysis-empty').hidden = Boolean(analysis);
  $('analysis-data').hidden = !analysis;
  $('analysis-move').textContent = analysis ? `第 ${analysis.turn + 1} 手 · ${analysis.player === 1 ? '黑' : '白'}` : '—';
  if (analysis) {
    $('seconds').textContent = `${analysis.seconds.toFixed(2)} s`;
    $('completed').textContent = analysis.completed_visits.toLocaleString();
    $('value-label').textContent = `AI（${analysis.player === 1 ? '黑' : '白'}方）局面估值`;
    $('value').textContent = `${analysis.root_value >= 0 ? '+' : ''}${analysis.root_value.toFixed(3)}`;
    $('value-fill').style.width = `${Math.max(0, Math.min(100, (analysis.root_value + 1) * 50))}%`;
    $('candidates').replaceChildren();
    analysis.candidates.slice(0, 5).forEach((candidate, i) => {
      const row = document.createElement('tr');
      row.innerHTML = `<td class="${candidate.action === analysis.action ? 'chosen' : ''}"><span class="rank">${i + 1}</span>${coordinate(candidate.action, analysis.board_size)}${candidate.action === analysis.action ? ' · 落子' : ''}</td><td>${candidate.visits}</td><td>${(candidate.selection_weight * 100).toFixed(1)}%</td>`;
      $('candidates').append(row);
    });
  }
}

function renderStatus() {
  if (!state) return;
  const game = state.game;
  let text = '选择模型，开始一局';
  if (!online) text = '等待服务连接';
  else if (state.busy) {
    const elapsed = state.started_at ? Math.max(0, Date.now() / 1000 - state.started_at).toFixed(1) : '0.0';
    text = `${state.phase === 'loading' ? '正在准备模型' : 'AI 思考中'} · ${elapsed}s`;
  } else if (game?.finished) text = game.winner === 0 ? '本局和棋' : `${game.winner === 1 ? '黑棋' : '白棋'}获胜 · ${game.winner === state.human ? '你赢了' : 'MuZero 获胜'}`;
  else if (game) text = game.player === state.human ? '轮到你落子' : '等待 AI 落子';
  $('status').textContent = text;
  $('status').parentElement.classList.toggle('thinking', state.busy);
}

$('settings').addEventListener('submit', event => {
  event.preventDefault();
  command('new', {model: $('model').value, size: Number($('size').value), rule: $('rule').value,
    human: Number(document.querySelector('input[name="human"]:checked').value), visits: Number($('visits').value)});
});
$('board').addEventListener('click', event => {
  const button = event.target.closest('button[data-action]');
  if (button && !button.disabled) command('play', {action: Number(button.dataset.action)});
});
$('undo').addEventListener('click', () => command('undo'));
$('retry').addEventListener('click', () => command('retry'));
$('numbers').addEventListener('change', () => $('board').classList.toggle('hide-numbers', !$('numbers').checked));
$('search-map-kind').addEventListener('change', renderHeatmaps);
for (const event of ['mouseover', 'focusin']) $('heatmap-data').addEventListener(event, event => {
  const cell = event.target.closest('[data-detail]');
  if (cell) $('heatmap-detail').textContent = cell.dataset.detail;
});
$('size').addEventListener('change', () => { if (!state?.game) render(); });
setInterval(renderStatus, 200);

async function start() {
  for (;;) {
    try {
      catalog = await api('/api/catalog');
      for (const model of catalog.models) {
        const label = model.label.replace(/^.*\/data\//, '').replace(/\/models\/model_0*(\d+)\.pt$/, ' · 第 $1 代');
        $('model').add(new Option(label, model.id));
      }
      for (let size = 5; size <= 25; size++) $('size').add(new Option(`${size} × ${size}`, size));
      $('size').value = catalog.default_size;
      $('visits').value = catalog.default_visits;
      for (const rule of catalog.rules) $('rule').add(new Option(rules[rule], rule));
      $('engine-info').textContent = `${catalog.search_threads} 线程 · 虚拟损失 ${catalog.virtual_loss} · ${catalog.device}`;
      break;
    } catch {
      connection(false);
      await new Promise(resolve => setTimeout(resolve, 1500));
    }
  }
  for (;;) {
    try {
      const next = await api(`/api/state${state ? `?since=${state.version}` : ''}`);
      connection(true);
      accept(next);
    } catch {
      connection(false);
      render();
      await new Promise(resolve => setTimeout(resolve, 1500));
    }
  }
}
start();
