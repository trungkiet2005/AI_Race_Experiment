(() => {
  'use strict';

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const config = {
    safeProgress: 1,
    unsafeProgress: 1.5,
    payoffs: { safe: { safe: 1, unsafe: .6 }, unsafe: { safe: 2.4, unsafe: 2 } },
    minRounds: 5,
    stopProbability: .2,
    prize: 100,
    maxRounds: 100
  };

  let state;
  let selected = [null, null];
  let randomState = 20260731;

  function seededRandom() {
    randomState |= 0;
    randomState = randomState + 0x6D2B79F5 | 0;
    let t = Math.imul(randomState ^ randomState >>> 15, 1 | randomState);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }

  function initialState() {
    return { round: 1, progress: [0, 0], payoffs: [0, 0], unsafe: [0, 0], history: [], finished: false, maxRisk: Number($('#riskCap').value) / 100 };
  }

  function actionLabel(action) {
    return `<span class="action-pill ${action}">${action.toUpperCase()}</span>`;
  }

  function privateRisk(player) {
    return state.history.length ? state.maxRisk * state.unsafe[player] / state.history.length : 0;
  }

  function selectAction(player, action, button) {
    if (state.finished) return;
    selected[player] = action;
    const group = button.closest('.action-buttons');
    $$('button', group).forEach(item => item.classList.toggle('selected', item === button));
    const card = button.closest('.player-card');
    card.classList.remove('selected-safe', 'selected-unsafe');
    card.classList.add(`selected-${action}`);
    updateCommit();
  }

  function updateCommit() {
    const button = $('#commitRound');
    const ready = selected.every(Boolean) && !state.finished;
    button.disabled = !ready;
    $('span', button).textContent = state.finished ? 'Race complete' : ready ? 'Commit both actions' : 'Commit both actions';
    $('small', button).textContent = state.finished ? 'Reset to run another race' : ready ? `${selected[0].toUpperCase()} × ${selected[1].toUpperCase()} · reveal simultaneously` : 'Waiting for Company A and Company B';
  }

  function commitRound() {
    if (!selected.every(Boolean) || state.finished) return;
    const actions = [...selected];
    const increments = actions.map(action => action === 'unsafe' ? config.unsafeProgress : config.safeProgress);
    const payoffs = [config.payoffs[actions[0]][actions[1]], config.payoffs[actions[1]][actions[0]]];
    actions.forEach((action, player) => {
      state.progress[player] += increments[player];
      state.payoffs[player] += payoffs[player];
      state.unsafe[player] += Number(action === 'unsafe');
    });

    let stopDraw = null;
    let stopped = false;
    if (state.round >= config.minRounds) {
      stopDraw = seededRandom();
      stopped = stopDraw < config.stopProbability || state.round >= config.maxRounds;
    }
    state.history.push({ round: state.round, actions, increments, payoffs, stopDraw, stopped });
    state.finished = stopped;
    state.round += 1;
    selected = [null, null];
    $$('.action-buttons button').forEach(button => button.classList.remove('selected'));
    $$('.player-card').forEach(card => card.classList.remove('selected-safe', 'selected-unsafe'));
    render();
    if (stopped) setTimeout(showResult, 250);
  }

  function render() {
    $('#roundNumber').textContent = state.finished ? state.history.length : state.round;
    [0, 1].forEach(player => {
      const suffix = player ? 'B' : 'A';
      $(`#progress${suffix}`).textContent = state.progress[player].toFixed(1);
      $(`#payoff${suffix}`).textContent = state.payoffs[player].toFixed(1);
      $(`#risk${suffix}`).textContent = `${Math.round(privateRisk(player) * 100)}%`;
    });
    const difference = state.progress[0] - state.progress[1];
    const labels = difference === 0 ? ['TIED', 'TIED'] : difference > 0 ? ['LEADING', 'TRAILING'] : ['TRAILING', 'LEADING'];
    ['A', 'B'].forEach((suffix, player) => {
      const badge = $(`#lead${suffix}`);
      badge.textContent = labels[player];
      badge.classList.toggle('leading', labels[player] === 'LEADING');
    });
    const activeRound = state.finished ? state.history.length : state.round;
    $('#stopLabel').textContent = state.finished ? 'Terminal state reached' : activeRound <= config.minRounds ? 'Guaranteed to continue' : 'Stochastic horizon active';
    $('#stopDetail').textContent = state.finished ? `Race ended after ${state.history.length} rounds` : activeRound <= config.minRounds ? 'Stop draws begin after round 5' : '20% stop chance after this round';
    $('#historyCount').textContent = `${state.history.length} ROUND${state.history.length === 1 ? '' : 'S'}`;
    const treatmentLocked = state.history.length > 0;
    $('#riskCap').disabled = treatmentLocked;
    $$('.presets button').forEach(button => { button.disabled = treatmentLocked; });
    renderHistory();
    renderChart();
    updateCommit();
  }

  function renderHistory() {
    const body = $('#historyBody');
    if (!state.history.length) {
      body.innerHTML = '<tr class="empty-row"><td colspan="6">Choose two actions to begin the race.</td></tr>';
      return;
    }
    body.innerHTML = [...state.history].reverse().map(item => `<tr>
      <td class="mono">${String(item.round).padStart(2, '0')}</td>
      <td>${actionLabel(item.actions[0])}</td><td>${actionLabel(item.actions[1])}</td>
      <td>+${item.increments[0].toFixed(1)} / +${item.increments[1].toFixed(1)}</td>
      <td>${item.payoffs[0].toFixed(1)} / ${item.payoffs[1].toFixed(1)}</td>
      <td class="mono">${item.stopDraw === null ? '—' : item.stopDraw.toFixed(3)}${item.stopped ? ' · STOP' : ''}</td>
    </tr>`).join('');
  }

  function renderChart() {
    const svg = $('#progressChart');
    const width = 760, height = 230, left = 36, right = 12, top = 12, bottom = 28;
    const rounds = Math.max(5, state.history.length);
    const maxProgress = Math.max(6, ...state.progress) * 1.12;
    const x = round => left + round / rounds * (width - left - right);
    const y = value => height - bottom - value / maxProgress * (height - top - bottom);
    const series = [[0], [0]];
    const totals = [0, 0];
    state.history.forEach(item => item.increments.forEach((value, player) => { totals[player] += value; series[player].push(totals[player]); }));
    let markup = '';
    for (let i = 0; i <= 4; i += 1) {
      const value = maxProgress * i / 4;
      const yPos = y(value);
      markup += `<line class="chart-grid" x1="${left}" x2="${width - right}" y1="${yPos}" y2="${yPos}"/><text class="chart-label" x="0" y="${yPos + 4}">${value.toFixed(1)}</text>`;
    }
    for (let round = 0; round <= rounds; round += 1) {
      if (round === 0 || round === rounds || round % Math.ceil(rounds / 5) === 0) markup += `<text class="chart-label" text-anchor="middle" x="${x(round)}" y="${height - 5}">R${round}</text>`;
    }
    series.forEach((values, player) => {
      const points = values.map((value, round) => `${x(round)},${y(value)}`).join(' ');
      markup += `<polyline class="chart-line-${player ? 'b' : 'a'}" points="${points}"/>`;
      values.forEach((value, round) => markup += `<circle class="chart-point-${player ? 'b' : 'a'}" cx="${x(round)}" cy="${y(value)}" r="${round === values.length - 1 ? 5 : 3}"/>`);
    });
    svg.innerHTML = markup;
  }

  function showResult() {
    const outcomes = Math.abs(state.progress[0] - state.progress[1]) < 1e-9 ? ['tie', 'tie'] : state.progress[0] > state.progress[1] ? ['winner', 'loser'] : ['loser', 'winner'];
    const draws = [seededRandom(), seededRandom()];
    const risks = [privateRisk(0), privateRisk(1)];
    const prizes = outcomes.map(outcome => outcome === 'tie' ? config.prize / 2 : outcome === 'winner' ? config.prize : 0);
    const setbacks = outcomes.map((outcome, player) => outcome !== 'loser' && draws[player] < risks[player]);
    const finals = state.payoffs.map((payoff, player) => setbacks[player] ? 0 : payoff + prizes[player]);
    const winner = outcomes[0] === 'tie' ? null : outcomes.indexOf('winner');
    $('#resultTitle').textContent = winner === null ? 'The race ends in a tie' : `Company ${winner ? 'B' : 'A'} reaches the frontier first`;
    $('#resultSubtitle').textContent = `${state.history.length} rounds completed. Terminal setback draws are applied only to the winner or tied winners.`;
    $('#resultCards').innerHTML = [0, 1].map(player => `<article><small>COMPANY ${player ? 'B' : 'A'} · ${outcomes[player].toUpperCase()}</small><strong>${finals[player].toFixed(1)} ECU</strong><small>${Math.round(risks[player] * 100)}% risk · draw ${draws[player].toFixed(3)} · ${setbacks[player] ? 'SETBACK' : 'NO SETBACK'}</small></article>`).join('');
    $('#resultDialog').showModal();
  }

  function reset() {
    randomState = 20260731;
    selected = [null, null];
    state = initialState();
    $$('.action-buttons button').forEach(button => button.classList.remove('selected'));
    $$('.player-card').forEach(card => card.classList.remove('selected-safe', 'selected-unsafe'));
    render();
  }

  $$('.action-buttons button').forEach(button => button.addEventListener('click', () => selectAction(Number(button.closest('.action-buttons').dataset.player), button.dataset.action, button)));
  $('#commitRound').addEventListener('click', commitRound);
  $('#resetGame').addEventListener('click', reset);
  $('#riskCap').addEventListener('input', event => {
    const value = Number(event.target.value);
    $('#riskValue').textContent = `${value}%`;
    state.maxRisk = value / 100;
    $$('.presets button').forEach(button => button.classList.toggle('active', Number(button.dataset.risk) === value));
    render();
  });
  $$('.presets button').forEach(button => button.addEventListener('click', () => {
    $('#riskCap').value = button.dataset.risk;
    $('#riskCap').dispatchEvent(new Event('input'));
  }));
  $('.dialog-close').addEventListener('click', () => $('#resultDialog').close());
  $('#playAgain').addEventListener('click', () => { $('#resultDialog').close(); reset(); });
  reset();
})();
