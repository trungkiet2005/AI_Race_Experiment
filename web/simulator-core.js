(function exposeAIRaceCore(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.AIRaceCore = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function buildAIRaceCore() {
  'use strict';

  const DEFAULT_CONFIG = Object.freeze({
    safeProgress: 1,
    unsafeProgress: 1.5,
    payoffs: Object.freeze({
      safe: Object.freeze({ safe: 1, unsafe: 0.6 }),
      unsafe: Object.freeze({ safe: 2.4, unsafe: 2 })
    }),
    minRounds: 5,
    stopProbability: 0.2,
    prize: 100,
    maxRounds: 100
  });

  const POLICIES = Object.freeze({
    normal: 'Normal bot',
    myopic: 'Myopic payoff bot',
    always_safe: 'AS - Always Safe',
    always_unsafe: 'AU - Always Unsafe',
    copycat_safe: 'CS - Safe then copy',
    copycat_unsafe: 'CAS - Unsafe then copy',
    manual: 'Manual control'
  });

  function assertAction(action) {
    if (action !== 'safe' && action !== 'unsafe') throw new Error(`Invalid action: ${action}`);
    return action;
  }

  function validateConfig(config, maxRisk) {
    const errors = [];
    if (!(config.safeProgress > 0)) errors.push('Safe progress must be positive.');
    if (!(config.unsafeProgress > config.safeProgress)) errors.push('Unsafe progress must exceed Safe progress.');
    if (!(config.stopProbability > 0 && config.stopProbability <= 1)) errors.push('Stop probability must be in (0, 1].');
    if (!(Number.isInteger(config.minRounds) && config.minRounds >= 1)) errors.push('Minimum rounds must be a positive integer.');
    if (!(Number.isInteger(config.maxRounds) && config.maxRounds >= config.minRounds)) errors.push('Safety cap must be an integer at least as large as minimum rounds.');
    if (!(config.prize >= 0)) errors.push('Race prize cannot be negative.');
    const payoffs = config.payoffs || DEFAULT_CONFIG.payoffs;
    const payoffValues = [payoffs.safe.safe, payoffs.safe.unsafe, payoffs.unsafe.safe, payoffs.unsafe.unsafe];
    if (payoffValues.some(value => !Number.isFinite(value))) errors.push('Every stage payoff must be a finite number.');
    if (!(maxRisk >= 0 && maxRisk <= 1)) errors.push('Maximum private risk must be between 0 and 1.');
    if (errors.length) throw new Error(errors.join(' '));
    return true;
  }

  function expectedRounds(config = DEFAULT_CONFIG) {
    return config.minRounds - 1 + 1 / config.stopProbability;
  }

  function deriveSeed(baseSeed, stream) {
    const modulus = 2147483647n;
    return Number((BigInt(Math.trunc(baseSeed)) * 1000003n + BigInt(stream) * 97409n + 11729n) % modulus);
  }

  function makeRng(seed) {
    let value = Math.trunc(seed) | 0;
    return function random() {
      value = value + 0x6D2B79F5 | 0;
      let mixed = Math.imul(value ^ value >>> 15, 1 | value);
      mixed = mixed + Math.imul(mixed ^ mixed >>> 7, 61 | mixed) ^ mixed;
      return ((mixed ^ mixed >>> 14) >>> 0) / 4294967296;
    };
  }

  function jointRoundOutcome(actions, config = DEFAULT_CONFIG) {
    const normalized = actions.map(assertAction);
    if (normalized.length !== 2) throw new Error('Exactly two actions are required');
    return {
      payoffs: [
        config.payoffs[normalized[0]][normalized[1]],
        config.payoffs[normalized[1]][normalized[0]]
      ],
      increments: normalized.map(action => action === 'unsafe' ? config.unsafeProgress : config.safeProgress)
    };
  }

  function effectivePrivateRisk(maxRisk, unsafeCount, roundsPlayed) {
    if (roundsPlayed <= 0) return 0;
    return Math.min(1, Math.max(0, maxRisk * unsafeCount / roundsPlayed));
  }

  function raceOutcomes(progress) {
    if (Math.abs(progress[0] - progress[1]) <= 1e-9) return ['tie', 'tie'];
    return progress[0] > progress[1] ? ['winner', 'loser'] : ['loser', 'winner'];
  }

  function terminalScoring({ payoffs, progress, unsafeCounts, roundsPlayed, maxRisk, prize, setbackDraws }) {
    const outcomes = raceOutcomes(progress);
    const tied = outcomes.every(outcome => outcome === 'tie');
    const prizes = outcomes.map(outcome => tied ? prize / 2 : outcome === 'winner' ? prize : 0);
    const risks = unsafeCounts.map(count => effectivePrivateRisk(maxRisk, count, roundsPlayed));
    const eligible = outcomes.map(outcome => outcome === 'winner' || outcome === 'tie');
    const setbacks = eligible.map((isEligible, player) => isEligible && setbackDraws[player] < risks[player]);
    const finalPayoffs = payoffs.map((payoff, player) => setbacks[player] ? 0 : payoff + prizes[player]);
    return { outcomes, prizes, risks, eligible, setbackDraws: [...setbackDraws], setbacks, finalPayoffs };
  }

  function copySnapshot(state) {
    return {
      round: state.round,
      progress: [...state.progress],
      payoffs: [...state.payoffs],
      unsafeCounts: [...state.unsafeCounts],
      previousActions: state.history.length ? [...state.history[state.history.length - 1].actions] : [null, null],
      roundsPlayed: state.history.length,
      maxRisk: state.maxRisk
    };
  }

  function decidePolicy(policy, player, snapshot, random) {
    const opponent = 1 - player;
    if (policy === 'always_safe') return { action: 'safe', reason: 'AS policy: always choose Safe.' };
    if (policy === 'always_unsafe') return { action: 'unsafe', reason: 'AU policy: always choose Unsafe.' };
    if (policy === 'myopic') return { action: 'unsafe', reason: 'Unsafe strictly dominates in immediate payoff and progress.' };
    if (policy === 'copycat_safe') {
      const action = snapshot.round === 1 ? 'safe' : snapshot.previousActions[opponent];
      return { action, reason: snapshot.round === 1 ? 'CS policy opens Safe.' : `CS policy copies the rival's previous ${action.toUpperCase()} action.` };
    }
    if (policy === 'copycat_unsafe') {
      const action = snapshot.round === 1 ? 'unsafe' : snapshot.previousActions[opponent];
      return { action, reason: snapshot.round === 1 ? 'CAS policy opens Unsafe.' : `CAS policy copies the rival's previous ${action.toUpperCase()} action.` };
    }
    if (policy !== 'normal') throw new Error(`Unknown bot policy: ${policy}`);

    const gap = snapshot.progress[player] - snapshot.progress[opponent];
    const opponentWasUnsafe = snapshot.previousActions[opponent] === 'unsafe';
    let unsafeProbability = 0.45 - 0.20 * snapshot.maxRisk;
    if (gap < 0) unsafeProbability += 0.25;
    if (gap > 0) unsafeProbability -= 0.10;
    if (opponentWasUnsafe) unsafeProbability += 0.20;
    unsafeProbability = Math.min(0.95, Math.max(0.05, unsafeProbability));
    const draw = random();
    const action = draw < unsafeProbability ? 'unsafe' : 'safe';
    const signals = [
      gap < 0 ? 'behind' : gap > 0 ? 'ahead' : 'tied',
      opponentWasUnsafe ? 'rival previously Unsafe' : snapshot.round === 1 ? 'no action history' : 'rival previously Safe'
    ];
    return {
      action,
      probability: unsafeProbability,
      draw,
      reason: `Normal heuristic: ${signals.join(', ')}; P(Unsafe) ${(unsafeProbability * 100).toFixed(0)}%, draw ${draw.toFixed(3)}.`
    };
  }

  class AIRaceSimulation {
    constructor(options = {}) {
      this.config = { ...DEFAULT_CONFIG, ...(options.config || {}) };
      this.seed = Number.isFinite(Number(options.seed)) ? Math.trunc(Number(options.seed)) : 20260731;
      this.policies = options.policies ? [...options.policies] : ['normal', 'normal'];
      this.maxRisk = Number(options.maxRisk ?? 0.6);
      if (this.policies.length !== 2 || this.policies.some(policy => !POLICIES[policy])) throw new Error('Exactly two valid policies are required.');
      validateConfig(this.config, this.maxRisk);
      this.horizonRandom = makeRng(deriveSeed(this.seed, 17));
      this.setbackRandom = makeRng(deriveSeed(this.seed, 29));
      this.decisionRandom = [makeRng(deriveSeed(this.seed, 101)), makeRng(deriveSeed(this.seed, 102))];
      this.round = 1;
      this.progress = [0, 0];
      this.payoffs = [0, 0];
      this.unsafeCounts = [0, 0];
      this.history = [];
      this.finished = false;
      this.terminal = null;
      this.stopForced = false;
    }

    snapshot() {
      return copySnapshot(this);
    }

    privateRisk(player) {
      return effectivePrivateRisk(this.maxRisk, this.unsafeCounts[player], this.history.length);
    }

    canAutoRun() {
      return this.policies.every(policy => policy !== 'manual');
    }

    step(manualActions = {}) {
      if (this.finished) throw new Error('Cannot step a finished race');
      const before = this.snapshot();
      const decisions = this.policies.map((policy, player) => {
        if (policy === 'manual') {
          const action = assertAction(manualActions[player]);
          return { action, reason: 'Manual action selected by the user.' };
        }
        return decidePolicy(policy, player, before, this.decisionRandom[player]);
      });
      const actions = decisions.map(decision => decision.action);
      const outcome = jointRoundOutcome(actions, this.config);
      actions.forEach((action, player) => {
        this.progress[player] += outcome.increments[player];
        this.payoffs[player] += outcome.payoffs[player];
        this.unsafeCounts[player] += Number(action === 'unsafe');
      });

      let stopDraw = null;
      let stopped = false;
      if (this.round >= this.config.minRounds) {
        stopDraw = this.horizonRandom();
        const stochasticStop = stopDraw < this.config.stopProbability;
        const forcedStop = this.round >= this.config.maxRounds;
        this.stopForced = forcedStop && !stochasticStop;
        stopped = stochasticStop || forcedStop;
      }
      const record = {
        round: this.round,
        actions,
        decisions,
        increments: outcome.increments,
        payoffs: outcome.payoffs,
        progressAfter: [...this.progress],
        stagePayoffsAfter: [...this.payoffs],
        unsafeCountsAfter: [...this.unsafeCounts],
        stopDraw,
        stopped
      };
      this.history.push(record);
      this.finished = stopped;
      this.round += 1;
      if (stopped) this.finalize();
      return record;
    }

    finalize() {
      if (this.terminal) return this.terminal;
      if (!this.finished) throw new Error('Cannot finalize before the race stops');
      const setbackDraws = [this.setbackRandom(), this.setbackRandom()];
      this.terminal = terminalScoring({
        payoffs: this.payoffs,
        progress: this.progress,
        unsafeCounts: this.unsafeCounts,
        roundsPlayed: this.history.length,
        maxRisk: this.maxRisk,
        prize: this.config.prize,
        setbackDraws
      });
      return this.terminal;
    }
  }

  return {
    AIRaceSimulation,
    DEFAULT_CONFIG,
    POLICIES,
    decidePolicy,
    deriveSeed,
    effectivePrivateRisk,
    expectedRounds,
    jointRoundOutcome,
    makeRng,
    raceOutcomes,
    terminalScoring,
    validateConfig
  };
}));
