// Injuries persist between fights. Each one lowers stats (percent penalties) until it heals
// naturally (healMin real minutes, 60 max) or is healed instantly for Biomass.
// severity: 1 = minor, 2 = major. A Kaiju's condition is the sum of its active injuries' severity.
// part: hint for the renderer (where to show the scar / damage).
import { STATS } from './frames.js';

export const INJURIES = [
  { id: 'fractured_claw', name: 'Fractured Claw',  severity: 1, part: 'claw',   healMin: 10, mods: { pow: -15 } },
  { id: 'torn_hide',      name: 'Torn Hide',       severity: 1, part: 'hide',   healMin: 10, mods: { arm: -15 } },
  { id: 'burned_glands',  name: 'Burned Glands',   severity: 1, part: 'glands', healMin: 15, mods: { enr: -15 } },
  { id: 'hamstrung',      name: 'Hamstrung',       severity: 1, part: 'leg',    healMin: 15, mods: { spd: -20 } },
  { id: 'scorched_eye',   name: 'Scorched Eye',    severity: 1, part: 'eye',    healMin: 12, mods: { fcs: -20 } },
  { id: 'deep_gash',      name: 'Deep Gash',       severity: 1, part: 'body',   healMin: 20, mods: { hp: -10 } },
  { id: 'shattered_limb', name: 'Shattered Limb',  severity: 2, part: 'leg',    healMin: 45, mods: { pow: -20, spd: -20 } },
  { id: 'ruptured_core',  name: 'Ruptured Core',   severity: 2, part: 'core',   healMin: 60, mods: { enr: -30, hp: -10 } },
  { id: 'cracked_skull',  name: 'Cracked Skull',   severity: 2, part: 'head',   healMin: 50, mods: { fcs: -30, spd: -10 } },
  { id: 'gored_belly',    name: 'Gored Belly',     severity: 2, part: 'body',   healMin: 60, mods: { hp: -20, arm: -15 } },
];
export const INJURY_BY_ID = Object.fromEntries(INJURIES.map(i => [i.id, i]));
export const MINOR = INJURIES.filter(i => i.severity === 1).map(i => i.id);
export const MAJOR = INJURIES.filter(i => i.severity === 2).map(i => i.id);

export const MAX_HEAL_MIN = 60;
// Natural healing time for an injury, in ms.
export const healMs = (id) => Math.min(MAX_HEAL_MIN, INJURY_BY_ID[id]?.healMin || 15) * 60e3;
export const HEAL_COST = { 1: 25, 2: 75 };     // instant heal at full duration (pro-rated by time left)
export const MAX_INJURIES = 5;

export const CONDITIONS = [
  { min: 0, id: 'healthy',  name: 'Healthy',    color: '#4ade80' },
  { min: 1, id: 'wounded',  name: 'Wounded',    color: '#facc15' },
  { min: 3, id: 'battered', name: 'Battered',   color: '#fb923c' },
  { min: 5, id: 'critical', name: 'Critical',   color: '#ef4444' },
];
export const CRITICAL_AT = 5;

export function severityOf(ids = []) {
  let s = 0; for (const id of ids) s += INJURY_BY_ID[id]?.severity || 0; return s;
}
export function conditionOf(ids = []) {
  const s = severityOf(ids);
  let c = CONDITIONS[0]; for (const x of CONDITIONS) if (s >= x.min) c = x;
  return { ...c, severity: s };
}
// Chance (%) that a Kaiju with these injuries dies if it is knocked out in a fight it started.
export function deathChance(ids = []) {
  const s = severityOf(ids);
  return s < CRITICAL_AT ? 0 : Math.min(50, 20 + 10 * (s - CRITICAL_AT));
}
// Integer stat penalties. Penalties for the same stat stack additively, floored at -60%.
export function applyInjuries(stats, ids = []) {
  const total = {};
  for (const id of ids) for (const [k, v] of Object.entries(INJURY_BY_ID[id]?.mods || {})) total[k] = (total[k] || 0) + v;
  const out = { ...stats };
  for (const s of STATS) if (total[s]) out[s] = Math.max(1, Math.floor((out[s] * (100 + Math.max(-60, total[s]))) / 100));
  return out;
}
// Instant-heal price for one injury given ms of healing left.
export function healCost(injury, msLeft) {
  const def = INJURY_BY_ID[injury.id]; if (!def) return 0;
  const full = healMs(injury.id);
  return Math.max(5, Math.ceil((HEAL_COST[def.severity] * Math.min(full, Math.max(0, msLeft))) / full));
}
