// Fine Tune: a late-game, per-Kaiju upgrade layer bought with Biomass once that Kaiju reaches FT_ELO.
//  - Tuning: granular stat steps. Each stat can go TUNE_MAX steps up or down; lowering a stat refunds
//    its step as a free point, so you can trade (e.g. -2 Speed for +2 Might). Extra points are bought.
//  - Skill mods: one mod per ability (basic included), changing its numbers.
// Engine-relevant: tuned snapshots carry `tune: { pts, alloc, mods }`, sanitized by sanitizeTune().
// Prices were tuned with test/finetune-balance.mjs to sit just below the value of normal top-level upgrades.
const STATS = ['hp', 'pow', 'enr', 'arm', 'spd', 'fcs'];   // same order as frames.js STATS (no import: frames.js imports this file)

export const FT_ELO = 1200;            // Kaiju Elo needed to BUY Fine Tune (it stays unlocked if Elo drops later)
export const FT_UNLOCK_COST = 1500;    // one-time per Kaiju
export const TUNE_STEP = { hp: 40, pow: 1, enr: 1, arm: 1, spd: 1, fcs: 1 };
export const TUNE_MAX = 3;             // max steps up or down per stat
export const TUNE_MAX_PTS = 6;         // max bought points per Kaiju
export const tunePointCost = (owned) => 320 + 80 * owned;   // price of the next point (1st ≈ break-even, later points worse)
export const MOD_COST = 1150;          // applying (or swapping) a skill mod (≈ 93% of measured value)
export const RETUNE_COST = 50;         // re-allocating points (free while buying)

export const MODS = {
  overcharge: { name: 'Overcharged', desc: 'Power +15%, costs 10 more energy.',
    ok: a => a.kind === 'attack', apply: a => { a.power = Math.floor(a.power * 115 / 100); a.cost = (a.cost || 0) + 10; } },
  efficient:  { name: 'Efficient',   desc: 'Energy cost -30%.',
    ok: a => (a.cost || 0) > 0, apply: a => { a.cost = Math.floor(a.cost * 70 / 100); } },
  rapid:      { name: 'Rapid',       desc: 'Cooldown -1 turn; attacks lose 10% power.',
    ok: a => (a.cd || 0) >= 2, apply: a => { a.cd -= 1; if (a.kind === 'attack') a.power = Math.floor(a.power * 90 / 100); } },
  precise:    { name: 'Precise',     desc: 'Accuracy +10; attacks also get +5 crit chance.',
    ok: a => a.kind === 'attack' || a.kind === 'debuff', apply: a => { a.acc = (a.acc ?? 100) + 10; if (a.kind === 'attack') a.critBonus = (a.critBonus || 0) + 5; } },
  potent:     { name: 'Potent',      desc: 'Effect chances +20 (max 100%).',
    ok: a => (a.effects || []).some(e => (e.chance ?? 100) < 100), apply: a => { a.effects = a.effects.map(e => ({ ...e, chance: Math.min(100, (e.chance ?? 100) + 20) })); } },
  lasting:    { name: 'Lasting',     desc: 'Status effects last 1 turn longer; costs 10 more energy.',
    ok: a => [...(a.effects || []), ...(a.selfEffects || [])].length > 0,
    apply: a => { const t = e => ({ ...e, turns: (e.turns || 1) + 1 }); if (a.effects) a.effects = a.effects.map(t); if (a.selfEffects) a.selfEffects = a.selfEffects.map(t); a.cost = (a.cost || 0) + 10; } },
};
export const MOD_IDS = Object.keys(MODS);
export const modAllowed = (ab, id) => !!(ab && MODS[id] && MODS[id].ok(ab));

// A modified copy of an ability (the original catalogue entry is never touched).
export function modAbility(ab, id) {
  if (!modAllowed(ab, id)) return ab;
  const c = { ...ab, mod: id };
  if (c.effects) c.effects = c.effects.map(e => ({ ...e }));
  if (c.selfEffects) c.selfEffects = c.selfEffects.map(e => ({ ...e }));
  MODS[id].apply(c);
  return c;
}

const clampI = (v, lo, hi) => Math.max(lo, Math.min(hi, Math.floor(Number(v) || 0)));
// Points spent by an allocation: raised steps cost 1 each, lowered steps give 1 back.
export const allocNet = (alloc = {}) => STATS.reduce((s, k) => s + (alloc[k] || 0), 0);

// Normalize untrusted tune data. `kit` = ability ids the Kaiju can carry mods on (basic + equipped).
export function sanitizeTune(t, kit = []) {
  if (!t || typeof t !== 'object' || !t.on) return null;
  const pts = clampI(t.pts, 0, TUNE_MAX_PTS);
  const alloc = {};
  for (const s of STATS) { const v = clampI(t.alloc?.[s], -TUNE_MAX, TUNE_MAX); if (v) alloc[s] = v; }
  // Over budget: trim raised steps (in STATS order, from the end) until it fits.
  for (let i = STATS.length - 1; allocNet(alloc) > pts && i >= 0; i--) {
    const s = STATS[i];
    while (allocNet(alloc) > pts && (alloc[s] || 0) > 0) { alloc[s]--; if (!alloc[s]) delete alloc[s]; }
  }
  const mods = {};
  if (t.mods && typeof t.mods === 'object') for (const id of kit) if (MODS[t.mods[id]]) mods[id] = t.mods[id];
  return { on: true, pts, alloc, mods };
}

export function applyTune(stats, tune) {
  if (!tune?.alloc) return stats;
  const out = { ...stats };
  for (const s of STATS) if (tune.alloc[s]) out[s] = Math.max(1, out[s] + tune.alloc[s] * TUNE_STEP[s]);
  return out;
}

// Power contribution (matchmaking + rewards): 5 per net raised step, 10 per mod.
export function tunePower(tune) {
  if (!tune?.on) return 0;
  return 5 * Math.max(0, allocNet(tune.alloc)) + 10 * Object.keys(tune.mods || {}).length;
}
