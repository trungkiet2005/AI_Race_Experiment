(() => {
  'use strict';
  const data = window.AI_RACE_TRAJECTORY_DATA;
  const $ = id => document.getElementById(id);
  let activeCase = data.cases[0];
  let roundIndex = -1;
  let timer = null;

  const pct = value => `${(100 * value).toFixed(1)}%`;
  const pp = value => `${(100 * value).toFixed(1)} pp`;
  const actionClass = action => action.toLowerCase();

  function stop() {
    if (timer) window.clearInterval(timer);
    timer = null;
    $('playButton').textContent = 'Play trajectory';
  }

  function setAction(element, action) {
    element.textContent = action || '—';
    element.className = `action-display ${action ? actionClass(action) : ''}`;
  }

  function buildTimeline() {
    $('timeline').innerHTML = activeCase.rounds.map(row => {
      const match = row.reference_action === row.context_action;
      return `<div class="round-cell ${match ? 'match' : 'flip'}" data-round="${row.round}"><span>R${String(row.round).padStart(2, '0')}</span><b>${row.reference_action[0]} · ${row.context_action[0]}</b></div>`;
    }).join('');
  }

  function render(index) {
    roundIndex = index;
    document.querySelectorAll('.round-cell').forEach(node => node.classList.toggle('active', Number(node.dataset.round) === index + 1));
    if (index < 0) {
      $('roundNumber').textContent = '0';
      setAction($('referenceAction'));
      setAction($('contextAction'));
      ['referenceProgress','contextProgress'].forEach(id => $(id).textContent = '0.0');
      ['referenceRisk','contextRisk'].forEach(id => $(id).textContent = '0.0%');
      ['referenceProgressBar','contextProgressBar','referenceRiskBar','contextRiskBar'].forEach(id => $(id).style.width = '0%');
      $('syncState').textContent = 'SYNCED';
      $('syncState').className = 'sync';
      return;
    }
    const row = activeCase.rounds[index];
    const diverged = activeCase.rounds.slice(0, index + 1).some(item => item.reference_action !== item.context_action);
    const maxProgress = Math.max(...activeCase.rounds.flatMap(item => [item.reference_progress, item.context_progress]), 1);
    $('roundNumber').textContent = row.round;
    setAction($('referenceAction'), row.reference_action);
    setAction($('contextAction'), row.context_action);
    $('referenceProgress').textContent = row.reference_progress.toFixed(1);
    $('contextProgress').textContent = row.context_progress.toFixed(1);
    $('referenceRisk').textContent = pct(row.reference_risk);
    $('contextRisk').textContent = pct(row.context_risk);
    $('referenceProgressBar').style.width = `${100 * row.reference_progress / maxProgress}%`;
    $('contextProgressBar').style.width = `${100 * row.context_progress / maxProgress}%`;
    $('referenceRiskBar').style.width = `${100 * row.reference_risk / activeCase.risk}%`;
    $('contextRiskBar').style.width = `${100 * row.context_risk / activeCase.risk}%`;
    $('syncState').textContent = diverged ? 'DIVERGED' : 'SYNCED';
    $('syncState').className = `sync ${diverged ? 'diverged' : ''}`;
  }

  function selectCase(id) {
    stop();
    activeCase = data.cases.find(item => item.id === id) || data.cases[0];
    $('referenceName').textContent = activeCase.reference_label;
    $('contextName').textContent = activeCase.context_label;
    $('caseRisk').textContent = `RISK ${(100 * activeCase.risk).toFixed(0)}%`;
    $('caseMapping').textContent = activeCase.mapping.replace('_', '=').toUpperCase();
    $('caseRep').textContent = `REP ${activeCase.rep} · PLAYER ${activeCase.player_index + 1}`;
    $('caseDivergence').textContent = `FIRST DIVERGENCE R${activeCase.first_divergence_round} · PAYOFF Δ ${activeCase.final_payoff_delta.toFixed(1)}`;
    buildTimeline();
    render(-1);
  }

  function play() {
    if (timer) { stop(); return; }
    if (roundIndex >= activeCase.rounds.length - 1) render(-1);
    $('playButton').textContent = 'Pause';
    timer = window.setInterval(() => {
      if (roundIndex >= activeCase.rounds.length - 1) { stop(); return; }
      render(roundIndex + 1);
    }, 760);
  }

  function buildMapping() {
    const byContext = new Map();
    data.mapping_interaction.forEach(row => {
      if (!byContext.has(row.context)) byContext.set(row.context, {});
      byContext.get(row.context)[row.mapping] = 100 * row.mean_unsafe_delta;
    });
    const labels = Object.fromEntries(data.contexts.map(row => [row.context, row.context_label]));
    $('mappingGrid').innerHTML = [...byContext.entries()]
      .sort(([, a], [, b]) => b.safe_p - a.safe_p)
      .map(([context, values]) => {
        const safeP = values.safe_p || 0;
        const safeQ = values.safe_q || 0;
        return `<div class="mapping-row"><span>${labels[context]}</span><div class="mapping-bars"><div class="mapping-bar safe-p" title="Safe=P"><i style="width:${100 * safeP / 70}%"></i></div><div class="mapping-bar safe-q" title="Safe=Q"><i style="width:${100 * safeQ / 70}%"></i></div></div><b>${safeP.toFixed(1)} pp</b></div>`;
      }).join('');
  }

  $('caseSelect').innerHTML = data.cases.map(item => `<option value="${item.id}">${item.context_label} · risk ${(100 * item.risk).toFixed(0)}% · ${item.mapping.replace('_', '=')}</option>`).join('');
  $('caseSelect').addEventListener('change', event => selectCase(event.target.value));
  $('playButton').addEventListener('click', play);
  $('resetButton').addEventListener('click', () => { stop(); render(-1); });
  buildMapping();
  selectCase(data.cases[0].id);
})();
