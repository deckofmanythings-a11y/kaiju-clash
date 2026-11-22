// node test/frame-balance.mjs [frameId,...]
// Round-robin win rate of each playable frame against every other playable frame, at two investment levels.
// Faster than determinism.mjs and lets you focus on a few frames (they still fight the whole roster).
import { simulateBattle } from '../js/engine/battle.js';
import { FRAMES, STATS } from '../js/data/frames.js';
import { abilitiesForFrame } from '../js/data/abilities.js';
import { ENVIRONMENTS } from '../js/data/environments.js';
import { createRng } from '../js/engine/rng.js';

const focus = (process.argv[2] || '').split(',').filter(Boolean);
const SEEDS = Number(process.env.SEEDS || 24);

function build(frame, lvl, nAb, seed) {
  const rng = createRng(seed);
  const pool = abilitiesForFrame(frame).map(a => a.id);
  const abilities = [frame.exclusive[0]];
  while (abilities.length < nAb && abilities.length < pool.length) { const p = rng.pick(pool); if (!abilities.includes(p)) abilities.push(p); }
  return { name: frame.name, frame: frame.id, levels: Object.fromEntries(STATS.map(s => [s, lvl])), abilities };
}

for (const [lvl, nAb] of [[0, 1], [5, 4], [9, 4]]) {
  const rows = [];
  for (const A of FRAMES) {
    if (focus.length && !focus.includes(A.id)) continue;
    let w = 0, g = 0;
    for (const B of FRAMES) {
      if (A === B) continue;
      for (let s = 0; s < SEEDS; s++) {
        const env = ENVIRONMENTS[s % ENVIRONMENTS.length].id;
        const seed = ((s + 1) * 2654435761 + A.id.length * 97 + B.id.length * 13) >>> 0;
        const flip = s % 2;
        const a = build(A, lvl, nAb, seed + 1), b = build(B, lvl, nAb, seed + 2);
        const r = simulateBattle({ seed, env, a: flip ? b : a, b: flip ? a : b });
        const me = flip ? 1 : 0;
        g++; if (r.winner === me) w++; else if (r.winner === null) w += 0.5;
      }
    }
    rows.push([A.name, (100 * w / g).toFixed(1)]);
  }
  console.log(`\nlevel ${lvl}, ${nAb} abilities`);
  for (const [n, p] of rows) console.log('  ' + n.padEnd(15) + p + '%');
}
