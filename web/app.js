(() => {
  'use strict';

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const { AIRaceSimulation, DEFAULT_CONFIG, POLICIES, expectedRounds } = window.AIRaceCore;
  let game;
  let selected = [null, null];
  let autoRunning = false;
  let phaseBusy = false;
  let simMode = 'protocol';
  const phaseOrder = ['observe', 'decide', 'reveal', 'update', 'stop'];

  function actionLabel(action) {
    return `<span class="action-pill ${action}">${action.toUpperCase()}</span>`;
  }

  function policyValue(player) {
    return $(`#policy${player ? 'B' : 'A'}`).value;
  }

  function manualActionMap() {
    const actions = {};
    [0, 1].forEach(player => {
      if (policyValue(player) === 'manual' && selected[player]) actions[player] = selected[player];
    });
    return actions;
  }

  function manualReady() {
    return [0, 1].every(player => policyValue(player) !== 'manual' || Boolean(selected[player]));
  }

  function applyCanonicalParameters() {
    $('#safeProgress').value = DEFAULT_CONFIG.safeProgress;
    $('#unsafeProgress').value = DEFAULT_CONFIG.unsafeProgress;
    $('#stopProbability').value = DEFAULT_CONFIG.stopProbability * 100;
    $('#minRounds').value = DEFAULT_CONFIG.minRounds;
    $('#racePrize').value = DEFAULT_CONFIG.prize;
    $('#maxRounds').value = DEFAULT_CONFIG.maxRounds;
    $('#payoffSS').value = DEFAULT_CONFIG.payoffs.safe.safe;
    $('#payoffSU').value = DEFAULT_CONFIG.payoffs.safe.unsafe;
    $('#payoffUS').value = DEFAULT_CONFIG.payoffs.unsafe.safe;
    $('#payoffUU').value = DEFAULT_CONFIG.payoffs.unsafe.unsafe;
  }

  function readConfig() {
    return {
      safeProgress: Number($('#safeProgress').value),
      unsafeProgress: Number($('#unsafeProgress').value),
      stopProbability: Number($('#stopProbability').value) / 100,
      minRounds: Number($('#minRounds').value),
      prize: Number($('#racePrize').value),
      maxRounds: Number($('#maxRounds').value),
      payoffs: {
        safe: { safe: Number($('#payoffSS').value), unsafe: Number($('#payoffSU').value) },
        unsafe: { safe: Number($('#payoffUS').value), unsafe: Number($('#payoffUU').value) }
      }
    };
  }

  function updateParameterSummary(config) {
    $('#stopRule').textContent = `${(config.stopProbability * 100).toFixed(0)}% after R${config.minRounds}`;
    const expectation = expectedRounds(config);
    $('#expectedRule').textContent = `${Number.isInteger(expectation) ? expectation : expectation.toFixed(1)} rounds`;
    $('#prizeRule').textContent = `${config.prize.toFixed(0)} ECU`;
    $('#parameterLock').textContent = simMode === 'protocol' ? 'LOCKED BY PROTOCOL' : 'EXPLORATORY';
    $('#modeNote').innerHTML = simMode === 'protocol'
      ? '<b>Protocol faithful.</b> Only the three risk treatments can change; every other game rule is locked.'
      : '<b>Sandbox extension.</b> Changed mechanics are exploratory and must not be interpreted as protocol evidence.';
  }

  function switchMode(mode) {
    if (game && (game.history.length || phaseBusy)) return;
    simMode = mode;
    $$('.mode-switch button').forEach(button => button.classList.toggle('active', button.dataset.mode === mode));
    if (mode === 'protocol') applyCanonicalParameters();
    reset();
  }

  function wait(milliseconds) {
    return new Promise(resolve => window.setTimeout(resolve, milliseconds));
  }

  function setPhase(phase, title, body, facts = [], tone = 'cyan') {
    const activeIndex = phaseOrder.indexOf(phase);
    $$('.round-stepper li').forEach((item, index) => {
      item.classList.toggle('active', index === activeIndex);
      item.classList.toggle('complete', activeIndex >= 0 && index < activeIndex);
    });
    $('#narrativeStep').textContent = phase ? `${activeIndex + 1} / 5 · ${phase.toUpperCase()}` : 'READY';
    $('#narrativeTitle').textContent = title;
    $('#narrativeBody').textContent = body;
    $('#narrativeFacts').innerHTML = facts.map(fact => `<span>${fact}</span>`).join('');
    $('#roundNarrative').dataset.tone = tone;
  }

  function selectAction(player, action, button) {
    if (game.finished || autoRunning || policyValue(player) !== 'manual') return;
    selected[player] = action;
    const group = button.closest('.action-buttons');
    $$('button', group).forEach(item => item.classList.toggle('selected', item === button));
    const card = button.closest('.player-card');
    card.classList.remove('selected-safe', 'selected-unsafe');
    card.classList.add(`selected-${action}`);
    $(`#decision${player ? 'B' : 'A'}`).textContent = `Manual ${action.toUpperCase()} action queued. It remains hidden until both decisions commit.`;
    updateControls();
  }

  function setDecisionVisuals(record) {
    record.decisions.forEach((decision, player) => {
      const suffix = player ? 'B' : 'A';
      const card = $(`.player-${suffix.toLowerCase()}`);
      card.classList.remove('selected-safe', 'selected-unsafe');
      card.classList.add(`selected-${decision.action}`);
      $$('button', $(`.action-buttons[data-player="${player}"]`)).forEach(button => button.classList.toggle('selected', button.dataset.action === decision.action));
      $(`#decision${suffix}`).textContent = decision.reason;
    });
  }

  function clearQueuedActions() {
    selected = [null, null];
    [0, 1].forEach(player => {
      if (policyValue(player) === 'manual') {
        $(`#decision${player ? 'B' : 'A'}`).textContent = 'Choose a manual action for the next round.';
      }
    });
  }

  async function runRound() {
    if (game.finished || !manualReady() || phaseBusy) return null;
    phaseBusy = true;
    const speed = autoRunning ? 180 : 520;
    const before = game.snapshot();
    const previous = before.previousActions[0] ? `${before.previousActions[0].toUpperCase()} / ${before.previousActions[1].toUpperCase()}` : 'none yet';
    const observeTitle = before.round === 1
      ? 'Opening move: the race begins tied'
      : before.round === game.config.minRounds
        ? 'The hidden horizon activates this round'
        : `Both companies observe round ${before.round}`;
    setPhase(
      'observe',
      observeTitle,
      `They receive the same pre-action snapshot: progress ${before.progress[0].toFixed(1)} vs ${before.progress[1].toFixed(1)}, with previous actions ${previous}.`,
      [`A risk ${(game.privateRisk(0) * 100).toFixed(0)}%`, `B risk ${(game.privateRisk(1) * 100).toFixed(0)}%`, 'Same snapshot']
    );
    render();
    await wait(speed);

    setPhase(
      'decide',
      'Choices are generated privately',
      'Each policy reads only the shared pre-round state. Neither company can see the rival’s current choice.',
      [POLICIES[game.policies[0]], POLICIES[game.policies[1]], 'Actions hidden'],
      'amber'
    );
    await wait(speed);

    const record = game.step(manualActionMap());
    setPhase(
      'reveal',
      `${record.actions[0].toUpperCase()} meets ${record.actions[1].toUpperCase()}`,
      'Both committed decisions are revealed at the same time. There is no same-round reaction advantage.',
      [`Company A · ${record.actions[0].toUpperCase()}`, `Company B · ${record.actions[1].toUpperCase()}`, 'Simultaneous reveal'],
      'amber'
    );
    setDecisionVisuals(record);
    await wait(speed);

    setPhase(
      'update',
      'The environment updates exact state',
      `Progress changes by +${record.increments[0].toFixed(1)} / +${record.increments[1].toFixed(1)} and stage payoff by +${record.payoffs[0].toFixed(1)} / +${record.payoffs[1].toFixed(1)}.`,
      [`Progress ${record.progressAfter[0].toFixed(1)} / ${record.progressAfter[1].toFixed(1)}`, `Payoff ${record.stagePayoffsAfter[0].toFixed(1)} / ${record.stagePayoffsAfter[1].toFixed(1)}`, 'Risk recalculated'],
      'lime'
    );
    render();
    await wait(speed);

    const threshold = game.config.stopProbability;
    const stopMessage = record.stopDraw === null
      ? `Round ${record.round} is below the minimum horizon. No stop draw is allowed yet.`
      : record.stopped
        ? `The stop draw is ${record.stopDraw.toFixed(3)}, below ${threshold.toFixed(3)}. The race ends now.`
        : `The stop draw is ${record.stopDraw.toFixed(3)}, at least ${threshold.toFixed(3)}. The race continues.`;
    setPhase(
      'stop',
      record.stopped ? 'The hidden horizon stops the race' : 'Check whether the race continues',
      stopMessage,
      record.stopDraw === null ? [`Minimum ${game.config.minRounds} rounds`, 'No draw this round'] : [`Draw ${record.stopDraw.toFixed(3)}`, `Stop if draw < ${threshold.toFixed(3)}`, record.stopped ? 'STOP' : 'CONTINUE'],
      record.stopped ? 'amber' : 'lime'
    );
    await wait(speed);

    clearQueuedActions();
    phaseBusy = false;
    render();
    if (game.finished) window.setTimeout(showResult, 180);
    return record;
  }

  async function autoRun() {
    if (!game.canAutoRun() || game.finished || autoRunning) return;
    autoRunning = true;
    render();
    while (!game.finished && autoRunning) {
      await runRound();
      if (!game.finished && autoRunning) await wait(260);
    }
    autoRunning = false;
    render();
  }

  function stopAutoRun() {
    autoRunning = false;
    render();
  }

  function updateControls() {
    const stepButton = $('#commitRound');
    const autoButton = $('#autoRun');
    const ready = manualReady() && !game.finished && !autoRunning && !phaseBusy;
    stepButton.disabled = !ready;
    $('span', stepButton).textContent = game.finished ? 'Race complete' : phaseBusy ? 'Explaining round…' : autoRunning ? 'Auto-running…' : 'Run next round';
    $('small', stepButton).textContent = game.finished ? 'Reset to start a new race' : manualReady() ? 'Both decisions use the same state snapshot' : 'Waiting for manual action';
    autoButton.disabled = game.finished || !game.canAutoRun() || (phaseBusy && !autoRunning);
    $('span', autoButton).textContent = autoRunning ? 'Pause auto-run' : 'Auto-run race';
    $('small', autoButton).textContent = game.canAutoRun() ? 'Run until the hidden stop' : 'Available when both seats use bots';
    $('#simulationStatus').textContent = game.finished ? 'Race complete' : phaseBusy ? 'Explaining current round' : autoRunning ? 'Bots are racing' : game.history.length ? 'Race in progress' : 'Simulation ready';
    $('#resetGame').disabled = phaseBusy;
  }

  function render() {
    const displayedRound = game.finished ? game.history.length : game.round;
    $('#roundNumber').textContent = displayedRound;
    [0, 1].forEach(player => {
      const suffix = player ? 'B' : 'A';
      $(`#progress${suffix}`).textContent = game.progress[player].toFixed(1);
      $(`#payoff${suffix}`).textContent = game.payoffs[player].toFixed(1);
      $(`#risk${suffix}`).textContent = `${Math.round(game.privateRisk(player) * 100)}%`;
      const manual = policyValue(player) === 'manual';
      const group = $(`.action-buttons[data-player="${player}"]`);
      group.classList.toggle('manual-enabled', manual);
      $$('button', group).forEach(button => { button.disabled = !manual || game.finished || autoRunning || phaseBusy; });
    });

    const difference = game.progress[0] - game.progress[1];
    const labels = difference === 0 ? ['TIED', 'TIED'] : difference > 0 ? ['LEADING', 'TRAILING'] : ['TRAILING', 'LEADING'];
    ['A', 'B'].forEach((suffix, player) => {
      const badge = $(`#lead${suffix}`);
      badge.textContent = labels[player];
      badge.classList.toggle('leading', labels[player] === 'LEADING');
    });

    $('#stopLabel').textContent = game.finished ? 'Terminal state reached' : displayedRound < game.config.minRounds ? 'Guaranteed to continue' : 'Stochastic horizon active';
    $('#stopDetail').textContent = game.finished ? `Race ended after ${game.history.length} rounds` : displayedRound < game.config.minRounds ? `A stop draw occurs after round ${game.config.minRounds}` : `${(game.config.stopProbability * 100).toFixed(0)}% stop chance after this round`;
    $('#historyCount').textContent = `${game.history.length} ROUND${game.history.length === 1 ? '' : 'S'}`;
    const locked = game.history.length > 0 || autoRunning || phaseBusy;
    $('#riskCap').disabled = locked || simMode === 'protocol';
    $('#gameSeed').disabled = locked;
    $$('.presets button').forEach(button => { button.disabled = locked; });
    $$('.policy-select').forEach(select => { select.disabled = locked; });
    $$('.mode-switch button').forEach(button => { button.disabled = locked; });
    $$('[data-parameter]').forEach(input => { input.disabled = locked || simMode === 'protocol'; });
    updateParameterSummary(game.config);
    renderHistory();
    renderChart();
    updateControls();
  }

  function renderHistory() {
    const body = $('#historyBody');
    if (!game.history.length) {
      body.innerHTML = '<tr class="empty-row"><td colspan="6">Run the bots or choose manual actions to begin.</td></tr>';
      return;
    }
    body.innerHTML = [...game.history].reverse().map(item => `<tr title="A: ${item.decisions[0].reason} B: ${item.decisions[1].reason}">
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
    const rounds = Math.max(5, game.history.length);
    const maxProgress = Math.max(6, ...game.progress) * 1.12;
    const x = round => left + round / rounds * (width - left - right);
    const y = value => height - bottom - value / maxProgress * (height - top - bottom);
    const series = [[0], [0]];
    game.history.forEach(item => item.progressAfter.forEach((value, player) => series[player].push(value)));
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
      const variant = player ? 'b' : 'a';
      const points = values.map((value, round) => `${x(round)},${y(value)}`).join(' ');
      markup += `<polyline class="chart-line-${variant}" points="${points}"/>`;
      values.forEach((value, round) => { markup += `<circle class="chart-point-${variant}" cx="${x(round)}" cy="${y(value)}" r="${round === values.length - 1 ? 5 : 3}"/>`; });
    });
    svg.innerHTML = markup;
  }

  function showResult() {
    const result = game.terminal;
    const winner = result.outcomes[0] === 'tie' ? null : result.outcomes.indexOf('winner');
    $('#resultTitle').textContent = winner === null ? 'The race ends in a tie' : `Company ${winner ? 'B' : 'A'} reaches the frontier first`;
    $('#resultSubtitle').textContent = `${game.history.length} rounds completed with seed ${game.seed}. Terminal risk applies only to the winner or tied winners.`;
    $('#resultCards').innerHTML = [0, 1].map(player => `<article class="${result.setbacks[player] ? 'setback-card' : ''}">
      <small>COMPANY ${player ? 'B' : 'A'} · ${result.outcomes[player].toUpperCase()}</small>
      <strong>${result.finalPayoffs[player].toFixed(1)} ECU</strong>
      <small>Stage ${game.payoffs[player].toFixed(1)} + prize ${result.prizes[player].toFixed(1)}</small>
      <small>${(result.risks[player] * 100).toFixed(1)}% risk · draw ${result.setbackDraws[player].toFixed(3)} · ${result.setbacks[player] ? 'SETBACK' : 'NO SETBACK'}</small>
    </article>`).join('');
    $('#resultDialog').showModal();
  }

  function reset() {
    const config = readConfig();
    let nextGame;
    try {
      nextGame = new AIRaceSimulation({
        seed: Number($('#gameSeed').value),
        maxRisk: Number($('#riskCap').value) / 100,
        policies: [policyValue(0), policyValue(1)],
        config
      });
    } catch (error) {
      $('#parameterError').textContent = error.message;
      return;
    }
    $('#parameterError').textContent = '';
    autoRunning = false;
    phaseBusy = false;
    selected = [null, null];
    game = nextGame;
    $$('.action-buttons button').forEach(button => button.classList.remove('selected'));
    $$('.player-card').forEach(card => card.classList.remove('selected-safe', 'selected-unsafe'));
    [0, 1].forEach(player => {
      const suffix = player ? 'B' : 'A';
      const policy = policyValue(player);
      $(`#decision${suffix}`).textContent = policy === 'manual' ? 'Choose a manual action for round 1.' : `${POLICIES[policy]} is ready to evaluate the shared pre-round state.`;
    });
    setPhase(null, 'The next round is ready', 'Run one round to see the complete decision cycle explained here.', ['Same snapshot', 'Hidden choices', 'Exact arithmetic']);
    render();
  }

  $$('.action-buttons button').forEach(button => button.addEventListener('click', () => selectAction(Number(button.closest('.action-buttons').dataset.player), button.dataset.action, button)));
  $('#commitRound').addEventListener('click', runRound);
  $('#autoRun').addEventListener('click', () => autoRunning ? stopAutoRun() : autoRun());
  $('#resetGame').addEventListener('click', reset);
  $('#riskCap').addEventListener('input', event => {
    const value = Number(event.target.value);
    $('#riskValue').textContent = `${value}%`;
    $$('.presets button').forEach(button => button.classList.toggle('active', Number(button.dataset.risk) === value));
    reset();
  });
  $$('.presets button').forEach(button => button.addEventListener('click', () => {
    $('#riskCap').value = button.dataset.risk;
    $('#riskCap').dispatchEvent(new Event('input'));
  }));
  $$('.policy-select').forEach(select => select.addEventListener('change', reset));
  $$('.mode-switch button').forEach(button => button.addEventListener('click', () => switchMode(button.dataset.mode)));
  $$('[data-parameter]').forEach(input => input.addEventListener('change', reset));
  $('#gameSeed').addEventListener('change', reset);
  $('.dialog-close').addEventListener('click', () => $('#resultDialog').close());
  $('#playAgain').addEventListener('click', () => { $('#resultDialog').close(); reset(); });
  reset();
})();
