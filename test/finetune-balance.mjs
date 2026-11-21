// Prices Fine Tune against normal upgrades: win-rate gained per Biomass spent.
// For late-game builds (~Elo 1200 territory), compare the BEST next normal stat level, the best
// single tuning point, and the best single skill mod, each vs the same rival pool and seeds.
// Run: node test/finetune-balance.mjs [rating=700] [builds=24] [pool=60]
import { simulateBattle } from '../js/engine/battle.js';
import { generateRival, statUpgradeCost } from '../js/data/economy.js';
import { STATS, MAX_STAT_LEVEL } from '../js/data/frames.js';
import { MOD_IDS, modAllowed, tunePointCost, MOD_COST, FT_UNLOCK_COST, TUNE_MAX_PTS } from '../js/data/finetune.js';
import { ABILITY_BY_ID } from '../js/data/abilities.js';
import { FRAME_BY_ID } from '../js/data/frames.js';

const RATING = +(process.argv[2] || 700), BUILDS = +(process.argv[3] || 24), POOL = +(process.argv[4] || 60);
const ENVS = ['city', 'ocean', 'volcano', 'arctic', 'desert', 'storm'];
const pool = Array.from({ length: POOL }, (_, i) => generateRival(900000 + i * 13, RATING));

function winRate(k, set = 0) {
  let w = 0, n = 0;
  for (let i = 0; i < pool.length; i++) for (let s = 0; s < 2; s++) {
    const seed = 7000 + set * 1000003 + i * 31 + s, env = ENVS[(i + s) % ENVS.length];
    const r = simulateBattle({ seed, env, a: k, b: pool[i] });
    w += r.winner === 0 ? 1 : r.winner === null ? 0.5 : 0; n++;
  }
  return (w * 100) / n;
}
const tuned = (k, t) => ({ ...k, tune: { on: true, pts: TUNE_MAX_PTS, alloc: {}, mods: {}, ...t } });

const eff = { level: [], point: [], mod: [] };
for (let b = 0; b < BUILDS; b++) {
  const k = generateRival(31337 + b * 101, RATING);
  const base = winRate(k), base1 = winRate(k, 1);
  // Best next normal level (what a player would buy).
  let bestL = null;
  for (const s of STATS) {
    const lv = k.levels[s]; if (lv >= MAX_STAT_LEVEL) continue;
    const g = winRate({ ...k, levels: { ...k.levels, [s]: lv + 1 } }) - base;
    const e = g / statUpgradeCost(lv);
    if (!bestL || e > bestL.e) bestL = { s, g, cost: statUpgradeCost(lv), e, k2: { ...k, levels: { ...k.levels, [s]: lv + 1 } } };
  }
  // Best single tuning point (first point price).
  let bestP = null;
  for (const s of STATS) {
    const g = winRate(tuned(k, { alloc: { [s]: 1 } })) - base;
    if (!bestP || g > bestP.g) bestP = { s, g, k2: tuned(k, { alloc: { [s]: 1 } }) };
  }
  // Best single mod.
  let bestM = null;
  const kit = [FRAME_BY_ID[k.frame].basic, ...k.abilities];
  for (const id of kit) for (const m of MOD_IDS) {
    if (!modAllowed(ABILITY_BY_ID[id], m)) continue;
    const g = winRate(tuned(k, { mods: { [id]: m } })) - base;
    if (!bestM || g > bestM.g) bestM = { id, m, g, k2: tuned(k, { mods: { [id]: m } }) };
  }
  for (const x of [bestL, bestP, bestM]) if (x) x.g = winRate(x.k2, 1) - base1;   // unbiased holdout gain
  if (bestL) eff.level.push(bestL.g / bestL.cost);
  eff.point.push(bestP.g / tunePointCost(0));
  eff.mod.push(bestM.g / MOD_COST);
  console.log(`${k.frame.padEnd(12)} base ${base.toFixed(1)}%  level(${bestL?.s}) +${bestL?.g.toFixed(2)} for ${bestL?.cost}  point(${bestP.s}) +${bestP.g.toFixed(2)}  mod(${bestM.m} on ${bestM.id}) +${bestM.g.toFixed(2)}`);
}
const avg = a => a.reduce((x, y) => x + y, 0) / a.length;
const L = avg(eff.level), P = avg(eff.point), M = avg(eff.mod);
console.log(`\nwin-rate points per 100 BIO:  normal level ${(L * 100).toFixed(3)} | tune point ${(P * 100).toFixed(3)} (${(P / L * 100).toFixed(0)}% of level) | mod ${(M * 100).toFixed(3)} (${(M / L * 100).toFixed(0)}% of level)`);
console.log(`break-even prices (100% of level value): point ≈ ${Math.round(avg(eff.point) * tunePointCost(0) / L)} , mod ≈ ${Math.round(avg(eff.mod) * MOD_COST / L)}   (unlock ${FT_UNLOCK_COST} not included)`);
