// node test/arena-balance.mjs [--quick]
// 1) Structure: every arena favors and hinders exactly ARENA_SIDE playable frames; every playable frame is favored as
//    often as it is hindered; favored/hindered counts differ by at most 1 between frames.
// 2) Effect: each frame's win rate in each arena minus its win rate in a modifier-free arena (same opponents, seeds,
//    builds). Favored should be clearly positive, hindered clearly negative, neutral near zero.
import { simulateBattle } from '../js/engine/battle.js';
import { FRAMES, STATS } from '../js/data/frames.js';
import { abilitiesForFrame } from '../js/data/abilities.js';
import { ENVIRONMENTS, ENV_BY_ID, ARENA_SIDE, arenaSide } from '../js/data/environments.js';
import { createRng } from '../js/engine/rng.js';

let fail = 0;
const bad = (m) => { fail++; console.log('FAIL', m); };
const ids = new Set(FRAMES.map(f => f.id));
const fav = Object.fromEntries(FRAMES.map(f => [f.id, 0])), hin = { ...fav };
for (const e of ENVIRONMENTS) {
  for (const x of [...e.favored, ...e.hindered]) if (!ids.has(x)) bad(`${e.id}: unknown frame ${x}`);
  if (new Set(e.favored).size !== ARENA_SIDE) bad(`${e.id}: favors ${new Set(e.favored).size}, want ${ARENA_SIDE}`);
  if (new Set(e.hindered).size !== ARENA_SIDE) bad(`${e.id}: hinders ${new Set(e.hindered).size}, want ${ARENA_SIDE}`);
  for (const x of e.favored) { fav[x]++; if (e.hindered.includes(x)) bad(`${e.id}: ${x} both favored and hindered`); }
  for (const x of e.hindered) hin[x]++;
}
for (const f of FRAMES) if (fav[f.id] !== hin[f.id]) bad(`${f.name}: favored ${fav[f.id]}x but hindered ${hin[f.id]}x`);
const fc = Object.values(fav);
if (Math.max(...fc) - Math.min(...fc) > 1) bad(`favored counts range ${Math.min(...fc)}..${Math.max(...fc)}`);
console.log(`structure: ${ENVIRONMENTS.length} arenas, ${ARENA_SIDE} favored + ${ARENA_SIDE} hindered each`);
console.log('  ' + FRAMES.map(f => `${f.name} +${fav[f.id]}/-${hin[f.id]}`).join(' · '));

// ---- Effect ----
const SEEDS = process.argv.includes('--quick') ? 8 : Number(process.env.SEEDS || 20);
ENV_BY_ID.__neutral = { id: '__neutral', favored: [], hindered: [], mods: {}, hazard: null };
function build(frame, lvl, nAb, seed) {
  const rng = createRng(seed);
  const pool = abilitiesForFrame(frame).map(a => a.id);
  const abilities = [frame.exclusive[0]];
  while (abilities.length < nAb && abilities.length < pool.length) { const p = rng.pick(pool); if (!abilities.includes(p)) abilities.push(p); }
  return { name: frame.name, frame: frame.id, levels: Object.fromEntries(STATS.map(s => [s, lvl])), abilities };
}
function winRate(A, env) {
  let w = 0, g = 0;
  for (const B of FRAMES) {
    if (A === B) continue;
    for (let s = 0; s < SEEDS; s++) {
      const [lvl, nAb] = [[0, 1], [5, 4], [9, 4]][s % 3];
      const seed = ((s + 1) * 2654435761 + A.id.length * 97 + B.id.length * 13) >>> 0;
      const flip = s % 2;
      const a = build(A, lvl, nAb, seed + 1), b = build(B, lvl, nAb, seed + 2);
      const r = simulateBattle({ seed, env, a: flip ? b : a, b: flip ? a : b });
      g++; if (r.winner === (flip ? 1 : 0)) w++; else if (r.winner === null) w += 0.5;
    }
  }
  return 100 * w / g;
}
const base = Object.fromEntries(FRAMES.map(f => [f.id, winRate(f, '__neutral')]));
const sums = { 1: [], 0: [], '-1': [] };
const net = Object.fromEntries(FRAMES.map(f => [f.id, 0]));
console.log('\nwin-rate change vs a modifier-free arena (points)');
console.log('  ' + 'arena'.padEnd(20) + 'favored avg  hindered avg  neutral avg   (min favored / max hindered)');
for (const e of ENVIRONMENTS) {
  const d = { 1: [], 0: [], '-1': [] };
  for (const f of FRAMES) { const x = winRate(f, e.id) - base[f.id]; d[arenaSide(e, f.id)].push(x); sums[arenaSide(e, f.id)].push(x); net[f.id] += x; }
  if (process.argv.includes('--detail')) console.log('    ' + FRAMES.map(f => `${f.name}${['-','','+'][arenaSide(e, f.id) + 1]} ${(winRate(f, e.id) - base[f.id]).toFixed(0)}`).join(' · '));
  const avg = (a) => a.reduce((s, x) => s + x, 0) / (a.length || 1);
  console.log('  ' + e.name.padEnd(20) + avg(d[1]).toFixed(1).padStart(8) + avg(d[-1]).toFixed(1).padStart(14) + avg(d[0]).toFixed(1).padStart(13)
    + `      (${Math.min(...d[1]).toFixed(1)} / ${Math.max(...d[-1]).toFixed(1)})`);
  if (Math.min(...d[1]) <= 0) bad(`${e.id}: a favored frame does worse than neutral`);
  if (Math.max(...d[-1]) >= 0) bad(`${e.id}: a hindered frame does better than neutral`);
}
console.log('\nnet change summed over all arenas, per frame (should be near 0)');
console.log('  ' + FRAMES.map(f => `${f.name} ${net[f.id] >= 0 ? '+' : ''}${net[f.id].toFixed(1)}`).join(' · '));
console.log(fail ? `\n${fail} FAILED` : '\nall arena checks passed');
process.exit(fail ? 1 : 0);
