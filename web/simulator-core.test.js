'use strict';

const assert = require('node:assert/strict');
const {
  AIRaceSimulation,
  expectedRounds,
  effectivePrivateRisk,
  jointRoundOutcome,
  terminalScoring,
  validateConfig
} = require('./simulator-core.js');

assert.deepEqual(jointRoundOutcome(['safe', 'safe']), { payoffs: [1, 1], increments: [1, 1] });
assert.deepEqual(jointRoundOutcome(['safe', 'unsafe']), { payoffs: [0.6, 2.4], increments: [1, 1.5] });
assert.deepEqual(jointRoundOutcome(['unsafe', 'safe']), { payoffs: [2.4, 0.6], increments: [1.5, 1] });
assert.deepEqual(jointRoundOutcome(['unsafe', 'unsafe']), { payoffs: [2, 2], increments: [1.5, 1.5] });
assert.deepEqual(
  jointRoundOutcome(['safe', 'unsafe'], {
    safeProgress: 2, unsafeProgress: 3,
    payoffs: { safe: { safe: 4, unsafe: -1 }, unsafe: { safe: 5, unsafe: 2 } }
  }),
  { payoffs: [-1, 5], increments: [2, 3] }
);
assert.equal(effectivePrivateRisk(0.6, 3, 5), 0.36);
assert.equal(expectedRounds(), 9);
assert.equal(validateConfig({ safeProgress: 1, unsafeProgress: 1.5, stopProbability: 0.2, minRounds: 5, maxRounds: 100, prize: 100 }, 0.6), true);
assert.throws(() => validateConfig({ safeProgress: 1, unsafeProgress: 0.8, stopProbability: 0.2, minRounds: 5, maxRounds: 100, prize: 100 }, 0.6), /Unsafe progress/);
assert.throws(() => validateConfig({ safeProgress: 1, unsafeProgress: 1.5, stopProbability: 0, minRounds: 5, maxRounds: 100, prize: 100 }, 0.6), /Stop probability/);
assert.throws(() => validateConfig({ safeProgress: 1, unsafeProgress: Infinity, stopProbability: 0.2, minRounds: 5, maxRounds: 100, prize: 100 }, 0.6), /Unsafe progress/);
assert.throws(() => validateConfig({ safeProgress: 1, unsafeProgress: 1.5, stopProbability: 0.2, minRounds: 5, maxRounds: 100, prize: Infinity }, 0.6), /Race prize/);

const terminal = terminalScoring({
  payoffs: [8, 7], progress: [6, 5], unsafeCounts: [2, 0], roundsPlayed: 5,
  maxRisk: 0.6, prize: 100, setbackDraws: [0.1, 0.1]
});
assert.deepEqual(terminal.outcomes, ['winner', 'loser']);
assert.deepEqual(terminal.prizes, [100, 0]);
assert.deepEqual(terminal.setbacks, [true, false]);
assert.deepEqual(terminal.finalPayoffs, [0, 7]);
assert.throws(() => terminalScoring({
  payoffs: [8, 7], progress: [6, NaN], unsafeCounts: [2, 0], roundsPlayed: 5,
  maxRisk: 0.6, prize: 100, setbackDraws: [0.1, 0.1]
}), /finite/);
assert.throws(() => terminalScoring({
  payoffs: [8, 7], progress: [6, 5], unsafeCounts: [6, 0], roundsPlayed: 5,
  maxRisk: 0.6, prize: 100, setbackDraws: [0.1, 0.1]
}), /Unsafe counts/);
assert.throws(() => terminalScoring({
  payoffs: [8, 7], progress: [6, 5], unsafeCounts: [2, 0], roundsPlayed: 5,
  maxRisk: 0.6, prize: 100, setbackDraws: [1.1, 0.1]
}), /Setback draws/);

const copyRace = new AIRaceSimulation({ seed: 42, policies: ['copycat_safe', 'copycat_unsafe'] });
const first = copyRace.step();
assert.deepEqual(first.actions, ['safe', 'unsafe']);
const second = copyRace.step();
assert.deepEqual(second.actions, ['unsafe', 'safe']);
assert.deepEqual(first.progressAfter, [1, 1.5]);
assert.deepEqual(second.progressAfter, [2.5, 2.5]);
assert.equal(copyRace.history.slice(0, 4).every(record => record.stopDraw === null), true);

const safeRace = new AIRaceSimulation({ seed: 7, maxRisk: 0.9, policies: ['always_safe', 'always_safe'] });
while (!safeRace.finished) safeRace.step();
assert.ok(safeRace.history.length >= 5);
assert.deepEqual(safeRace.unsafeCounts, [0, 0]);
assert.deepEqual(safeRace.terminal.risks, [0, 0]);
assert.deepEqual(safeRace.terminal.setbacks, [false, false]);

const raceA = new AIRaceSimulation({ seed: 99, policies: ['normal', 'normal'] });
const raceB = new AIRaceSimulation({ seed: 99, policies: ['always_safe', 'always_unsafe'] });
while (!raceA.finished) raceA.step();
while (!raceB.finished) raceB.step();
assert.deepEqual(raceA.history.map(record => record.stopDraw), raceB.history.map(record => record.stopDraw));

console.log('AI Race simulator core: all theory fixtures passed.');
