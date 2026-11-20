// Biomass (BIO) economy.
import { STATS, MAX_STAT_LEVEL, FRAMES, FRAME_BY_ID, WILD_FRAMES } from './frames.js';
import { ABILITY_BY_ID, abilitiesForFrame, MAX_EQUIPPED } from './abilities.js';
import { ENVIRONMENTS } from './environments.js';
import { createRng } from '../engine/rng.js';

export const STARTING_BIO = 300;
export const REWARD = { win: 120, loss: 45, draw: 75 };

export function statUpgradeCost(level) { return level >= MAX_STAT_LEVEL ? null : 60 + 40 * level; }

export function powerRating(k) {
  let r = 100;
  for (const s of STATS) r += 10 * (k.levels?.[s] || 0);
  for (const id of k.abilities || []) r += Math.floor((ABILITY_BY_ID[id]?.price || 150) / 20);
  return r;
}

// Reward scales with how strong the opponent was relative to you.
export function battleReward(result, myRating, oppRating) {
  const base = REWARD[result];
  const ratio = Math.max(60, Math.min(160, Math.floor((oppRating * 100) / myRating)));
  return Math.floor((base * ratio) / 100);
}

const NAME_A = ['Gor', 'Zil', 'Mek', 'Rath', 'Vor', 'Kai', 'Bara', 'Tor', 'Gya', 'Oro', 'Dra', 'Mos', 'Hed', 'Sky', 'Ul'];
const NAME_B = ['gax', 'dora', 'thul', 'goth', 'mora', 'zan', 'ruga', 'lith', 'ra', 'nox', 'kor', 'vex', 'don', 'mar'];
const TITLES = ['the Unbroken', 'of the Deep', 'Ashmaw', 'the Eternal', 'Stormborn', 'the Hungry', 'Cityeater', 'the Pale', 'Ironhide', ''];

function hslHex(h, s, l) {
  s /= 100; l /= 100;
  const k = n => (n + h / 30) % 12, a = s * Math.min(l, 1 - l);
  const f = n => Math.round(255 * (l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1))))).toString(16).padStart(2, '0');
  return '#' + f(0) + f(8) + f(4);
}

// ---------- Rare Wild encounters ----------
export const RARE_CHANCE = 6;          // % chance each Wild slot is a rare encounter
export const RARE_POWER_PCT = 110;     // % of the slot's rating (tuned: rares win ~57-66%, see test/rare-balance.mjs)
export const RARE_REWARD_MULT = 2;     // Biomass multiplier for beating one
const RARE_NAMES = {
  superhero: ['Captain Dawnstrike', 'The Crimson Comet', 'Ironheart', 'Major Valor', 'Skyguard', 'The Paragon'],
  heroteam: ['The Vanguard Five', 'Aegis Squadron', 'The Dawn Legion', 'Team Bulwark', 'The Sentinel Corps'],
  army: ['7th Armored Division', 'Task Force Hammer', 'Iron Rain Battalion', 'Operation Last Line', 'Joint Strike Group Kilo'],
  magicalgirl: ['Magical Girl Stellar Lumi', 'Starlight Mirai', 'Pretty Guardian Hoshi', 'Miracle Ribbon Aiko', 'Comet Princess Yuna'],
};
// Weighted pick of a rare frame id (weights = frame.rarity).
export function pickRare(rng) {
  let total = 0; for (const f of WILD_FRAMES) total += f.rarity;
  let r = rng.int(total);
  for (const f of WILD_FRAMES) { if (r < f.rarity) return f.id; r -= f.rarity; }
  return WILD_FRAMES[0].id;
}

// Deterministic rival with roughly `rating` power. Same (seed, rating, frame) => same rival.
// Pass a rare frame id to build a rare Wild-only enemy instead of a random Kaiju.
export function generateRival(seed, rating, forceFrame = null) {
  const rng = createRng(seed);
  const frame = forceFrame && FRAME_BY_ID[forceFrame] ? FRAME_BY_ID[forceFrame] : rng.pick(FRAMES);
  const levels = Object.fromEntries(STATS.map(s => [s, 0]));
  const pool = abilitiesForFrame(frame);
  const abilities = [frame.exclusive[0]];
  let budget = Math.max(0, rating - 100 - Math.floor((ABILITY_BY_ID[frame.exclusive[0]].price || 0) / 20));
  // Favour the frame's strongest growth stats a bit.
  const weights = STATS.map(s => (s === 'hp' ? 3 : frame.growth[s] + 1));
  let guard = 200;
  while (budget >= 7 && guard--) {
    if (abilities.length < MAX_EQUIPPED && rng.roll(25)) {
      const opts = pool.filter(a => !abilities.includes(a.id));
      if (opts.length) {
        const a = rng.pick(opts); const c = Math.floor((a.price || 150) / 20);
        if (c <= budget) { abilities.push(a.id); budget -= c; continue; }
      }
    }
    let total = 0; for (const w of weights) total += w;
    let r = rng.int(total), idx = 0;
    while (r >= weights[idx]) { r -= weights[idx]; idx++; }
    const s = STATS[idx];
    if (levels[s] < MAX_STAT_LEVEL) { levels[s]++; budget -= 10; }
  }
  if (frame.wildOnly) {
    return {
      id: 'rare-' + seed, name: rng.pick(RARE_NAMES[frame.id] || [frame.name]), frame: frame.id, levels, abilities,
      colors: { ...frame.colors }, owner: 'Defenders of Humanity', rare: true,
    };
  }
  const hue = rng.int(360);
  const name = rng.pick(NAME_A) + rng.pick(NAME_B);
  const title = rng.pick(TITLES);
  return {
    id: 'rival-' + seed, name: title ? `${name} ${title}` : name, frame: frame.id, levels, abilities,
    colors: { body: hslHex(hue, 30, 22), belly: hslHex((hue + 30) % 360, 25, 50), accent: hslHex((hue + 180) % 360, 90, 60) },
    owner: 'Wild Kaiju',
  };
}

export function pickEnvironment(rng) { return rng.pick(ENVIRONMENTS).id; }

export function newKaiju(frameId, name) {
  const f = FRAME_BY_ID[frameId];
  return {
    id: 'k' + Date.now().toString(36),
    name, frame: f.id,
    levels: Object.fromEntries(STATS.map(s => [s, 0])),
    abilities: [f.exclusive[0]],
    unlocked: [f.exclusive[0]],
    colors: { ...f.colors },
    record: { w: 0, l: 0, d: 0 },
  };
}
