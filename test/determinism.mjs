// node test/determinism.mjs
// 1) Same inputs => byte-identical event logs across repeated runs.
// 2) verifyReplay() agrees with a freshly simulated record.
// 3) Balance report: round-robin win rates per frame at equal (zero and mid) investment.
import { simulateBattle, verifyReplay } from '../js/engine/battle.js';
import { generateRival, powerRating } from '../js/data/economy.js';
import { FRAMES, STATS } from '../js/data/frames.js';
import { abilitiesForFrame } from '../js/data/abilities.js';
import { ENVIRONMENTS } from '../js/data/environments.js';
import { createRng } from '../js/engine/rng.js';

let fail = 0;
const N = 500;
for (let i = 0; i < N; i++) {
  const a = generateRival(1000 + i, 100 + (i % 30) * 10);
  const b = generateRival(5000 + i, 100 + ((i * 7) % 30) * 10);
  const env = ENVIRONMENTS[i % ENVIRONMENTS.length].id;
  const seed = (i * 2654435761) >>> 0;
  const r1 = simulateBattle({ seed, env, a, b });
  const r2 = simulateBattle({ seed, env, a: JSON.parse(JSON.stringify(a)), b: JSON.parse(JSON.stringify(b)) });
  const stored = JSON.parse(JSON.stringify(r1));
  if (r1.hash !== r2.hash || JSON.stringify(r1.events) !== JSON.stringify(r2.events)) { fail++; console.log('MISMATCH run', i); }
  if (!verifyReplay(stored).ok) { fail++; console.log('REPLAY FAIL', i); }
}
console.log(`determinism: ${N - fail}/${N} identical`);

// Balance: every frame vs every frame, all envs, many seeds, equal build.
function build(frame, lvl, nAb, seed) {
  const rng = createRng(seed);
  const pool = abilitiesForFrame(frame).map(a => a.id);
  const abilities = [frame.exclusive[0]];
  while (abilities.length < nAb) { const p = rng.pick(pool); if (!abilities.includes(p)) abilities.push(p); }
  return { name: frame.name, frame: frame.id, levels: Object.fromEntries(STATS.map(s => [s, lvl])), abilities };
}
for (const [lvl, nAb] of [[0, 1], [5, 4]]) {
  const wins = {}, games = {}, turns = [];
  let draws = 0;
  for (const fa of FRAMES) for (const fb of FRAMES) {
    if (fa === fb) continue;
    for (let s = 0; s < 24; s++) {
      const env = ENVIRONMENTS[s % ENVIRONMENTS.length].id;
      const r = simulateBattle({ seed: s * 7919 + 13, env, a: build(fa, lvl, nAb, s), b: build(fb, lvl, nAb, s + 99) });
      games[fa.id] = (games[fa.id] || 0) + 1; games[fb.id] = (games[fb.id] || 0) + 1;
      if (r.winner === 0) wins[fa.id] = (wins[fa.id] || 0) + 1;
      else if (r.winner === 1) wins[fb.id] = (wins[fb.id] || 0) + 1;
      else draws++;
      turns.push(r.turns);
    }
  }
  turns.sort((x, y) => x - y);
  console.log(`\nbalance @ level ${lvl}, ${nAb} abilities  (median actions ${turns[turns.length >> 1]}, draws ${draws})`);
  for (const f of FRAMES) console.log(`  ${f.name.padEnd(12)} ${((100 * (wins[f.id] || 0)) / games[f.id]).toFixed(1)}%`);
}
process.exit(fail ? 1 : 0);
