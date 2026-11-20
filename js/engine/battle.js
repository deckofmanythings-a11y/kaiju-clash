// Deterministic auto-battle simulator. Pure: no DOM, no Math.random, no Date, integer math only.
// simulateBattle({ seed, env, a, b }) -> BattleRecord (see DESIGN.md).
import { createRng, fnv1a } from './rng.js';
import { FRAME_BY_ID, STATS, effectiveStats, PASSIVES } from '../data/frames.js';
import { ABILITY_BY_ID, STATUSES, abilityAvailableTo, MAX_EQUIPPED } from '../data/abilities.js';
import { ENV_BY_ID } from '../data/environments.js';
import { INJURY_BY_ID, MAX_INJURIES, applyInjuries } from '../data/injuries.js';

export const ENGINE_VERSION = 3; // v2: persistent injuries lower stats. v3: frame passives (rare Wild enemies)
const MAX_ACTIONS = 80;          // 40 rounds
const GAUGE = 1000;              // initiative threshold
const START_ENERGY = 20;

const NEGATIVE = new Set(['burn', 'poison', 'bleed', 'stun', 'bind', 'slow', 'blind', 'weaken', 'sunder']);
const STAT_MODS = {
  slow: { spd: 65 }, haste: { spd: 135 },
  weaken: { pow: 75, enr: 75 }, enrage: { pow: 130 }, charged: { enr: 130 },
  fortify: { arm: 150 }, sunder: { arm: 60 },
};

const pct = (v, p) => Math.floor((v * p) / 100);
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

// Normalize a snapshot so bad/foreign data can't break or bias the sim.
export function sanitizeSnapshot(s) {
  const frame = FRAME_BY_ID[s.frame] || FRAME_BY_ID.saurian;
  const levels = {};
  for (const st of STATS) levels[st] = clamp(Math.floor(Number(s.levels?.[st]) || 0), 0, 10);
  const seen = new Set();
  const abilities = [];
  for (const id of Array.isArray(s.abilities) ? s.abilities : []) {
    const ab = ABILITY_BY_ID[id];
    if (!ab || ab.basic || seen.has(id) || !abilityAvailableTo(ab, frame)) continue;
    seen.add(id); abilities.push(id);
    if (abilities.length >= MAX_EQUIPPED) break;
  }
  const injuries = [...new Set(Array.isArray(s.injuries) ? s.injuries : [])].filter(id => INJURY_BY_ID[id]).slice(0, MAX_INJURIES);
  return {
    id: String(s.id || ''), name: String(s.name || 'Unnamed').slice(0, 24), frame: frame.id,
    levels, abilities, injuries, colors: { ...frame.colors, ...(s.colors || {}) }, owner: s.owner ? String(s.owner).slice(0, 32) : undefined,
  };
}

function envStats(snapshot, env) {
  const frame = FRAME_BY_ID[snapshot.frame];
  const st = applyInjuries(effectiveStats(snapshot), snapshot.injuries);
  const mods = env.tagMods || {};
  let matched = false;
  for (const tag of Object.keys(mods)) {
    if (tag === '_other' || !frame.tags.includes(tag)) continue;
    matched = true;
    for (const k of Object.keys(mods[tag])) st[k] = pct(st[k], mods[tag][k]);
  }
  if (!matched && mods._other) for (const k of Object.keys(mods._other)) st[k] = pct(st[k], mods._other[k]);
  return st;
}

export function simulateBattle({ seed, env: envId, a, b }) {
  seed = seed >>> 0;
  const env = ENV_BY_ID[envId] || ENV_BY_ID.city;
  const rng = createRng(seed);
  const snaps = [sanitizeSnapshot(a), sanitizeSnapshot(b)];
  const events = [];
  const emit = (e) => events.push(e);

  const F = snaps.map((s, i) => {
    const frame = FRAME_BY_ID[s.frame];
    const stats = envStats(s, env);
    return {
      i, s, frame, stats, maxHp: stats.hp, hp: stats.hp, energy: START_ENERGY,
      gauge: 0, statuses: [], cds: {}, passive: frame.passive || null, pUsed: false, pPending: false,
      kit: [frame.basic, ...s.abilities].map(id => ABILITY_BY_ID[id]),
    };
  });

  const has = (f, id) => f.statuses.find(x => x.id === id);
  const stat = (f, k) => {
    let v = f.stats[k];
    for (const x of f.statuses) { const m = STAT_MODS[x.id]?.[k]; if (m) v = pct(v, m); }
    return Math.max(1, v);
  };
  const resistOf = (f, type) => f.frame.resist?.[type] ?? 100;
  const typeMult = (type) => env.typeMult?.[type] ?? 100;

  function applyStatus(src, tg, eff) {
    if (!rng.roll(eff.chance ?? 100)) return;
    if (tg.passive === 'swarm' && (eff.status === 'stun' || eff.status === 'bind')) {
      emit({ t: 'passive', tg: tg.i, id: 'swarm', name: PASSIVES.swarm.name, resisted: eff.status });
      return;
    }
    const existing = has(tg, eff.status);
    const rec = existing || { id: eff.status, turns: 0, val: 0 };
    rec.turns = Math.max(rec.turns, eff.turns || 1);
    const def = STATUSES[eff.status];
    if (def?.dot) {
      // DoT damage is locked in at application from the source's Energy/Might.
      const s = Math.max(stat(src, 'pow'), stat(src, 'enr'));
      rec.val = Math.max(rec.val, pct(Math.floor(((eff.power || 30) * s) / 10), resistOf(tg, def.dot)));
    } else if (eff.status === 'shield') {
      rec.val = Math.max(rec.val, pct(tg.maxHp, eff.power || 10));
    } else if (eff.status === 'regen') {
      rec.val = Math.max(rec.val, pct(tg.maxHp, eff.power || 4));
    }
    if (!existing) tg.statuses.push(rec);
    emit({ t: 'status', tg: tg.i, st: eff.status, turns: rec.turns });
  }

  function damage(tg, amount, absorbable = true) {
    let absorbed = 0;
    const sh = absorbable && has(tg, 'shield');
    if (sh && amount > 0) {
      absorbed = Math.min(sh.val, amount);
      sh.val -= absorbed; amount -= absorbed;
      if (sh.val <= 0) { tg.statuses = tg.statuses.filter(x => x !== sh); emit({ t: 'expire', tg: tg.i, st: 'shield' }); }
    }
    if (tg.passive === 'last_stand' && !tg.pUsed && amount >= tg.hp && tg.hp > 1) {
      amount = tg.hp - 1; tg.pUsed = true; tg.pPending = true;
    }
    tg.hp = Math.max(0, tg.hp - amount);
    return { dealt: amount, absorbed };
  }

  // Passives that react to taking damage. Called right after each hit/dot/hazard event is emitted.
  function afterDamage(tg) {
    if (tg.hp <= 0) return;
    if (tg.pPending) {
      tg.pPending = false;
      emit({ t: 'passive', tg: tg.i, id: 'last_stand', name: PASSIVES.last_stand.name });
      applyStatus(tg, tg, { status: 'enrage', chance: 100, turns: 2 });
      return;
    }
    if (tg.pUsed || tg.hp * 2 >= tg.maxHp) return;
    if (tg.passive !== 'tag_team' && tg.passive !== 'transformation') return;
    tg.pUsed = true;
    emit({ t: 'passive', tg: tg.i, id: tg.passive, name: PASSIVES[tg.passive].name });
    for (const x of tg.statuses.filter(x => NEGATIVE.has(x.id))) emit({ t: 'expire', tg: tg.i, st: x.id });
    tg.statuses = tg.statuses.filter(x => !NEGATIVE.has(x.id));
    if (tg.passive === 'tag_team') {
      const amt = Math.min(tg.maxHp - tg.hp, pct(tg.maxHp, 30));
      tg.hp += amt;
      emit({ t: 'heal', tg: tg.i, amt, hp: tg.hp });
    } else {
      tg.energy = 100;
      applyStatus(tg, tg, { status: 'shield', chance: 100, turns: 3, power: 10 });
      applyStatus(tg, tg, { status: 'charged', chance: 100, turns: 3 });
      applyStatus(tg, tg, { status: 'haste', chance: 100, turns: 3 });
    }
  }

  function koCheck() {
    for (const f of F) if (f.hp <= 0 && !f.ko) { f.ko = true; emit({ t: 'ko', tg: f.i }); }
    return F.some(f => f.ko);
  }

  // ---------- AI: score each usable ability, weighted-random pick among the best ----------
  function estimate(f, tg, ab) {
    if (ab.kind === 'attack') {
      const atk = stat(f, ab.stat || 'pow');
      let v = Math.floor((ab.power * (ab.hits || 1) * atk) / 20);
      v = pct(v, ab.type === 'random' ? 100 : resistOf(tg, ab.type));
      v = pct(v, ab.type === 'random' ? 100 : typeMult(ab.type));
      v = pct(v, ab.acc);
      if (ab.execute && tg.hp * 100 < tg.maxHp * ab.execute) v *= 2;
      for (const e of ab.effects || []) if (!has(tg, e.status)) v += pct(40, e.chance);
      return v;
    }
    if (ab.kind === 'heal') {
      const missing = 100 - Math.floor((f.hp * 100) / f.maxHp);
      return missing < 30 ? 0 : missing * 3;
    }
    const list = ab.kind === 'buff' ? ab.selfEffects : ab.effects;
    const target = ab.kind === 'buff' ? f : tg;
    const fresh = (list || []).filter(e => !has(target, e.status)).length;
    let v = fresh * 70;
    if (ab.energyGain && f.energy < 40) v += 60;
    return v;
  }

  function choose(f, tg) {
    const ready = f.kit.filter(ab => !(f.cds[ab.id] > 0) && f.energy >= (ab.cost || 0));
    const scored = ready.map(ab => ({ ab, sc: Math.max(1, estimate(f, tg, ab)) }));
    scored.sort((x, y) => y.sc - x.sc);           // stable sort, kit order breaks ties
    const top = scored.slice(0, 3);
    let total = 0;
    const w = top.map(x => { const v = Math.floor((x.sc * x.sc) / 100) + 1; total += v; return v; });
    let r = rng.int(total);
    for (let k = 0; k < top.length; k++) { if (r < w[k]) return top[k].ab; r -= w[k]; }
    return top[0].ab;
  }

  // ---------- Resolve an ability ----------
  function useAbility(f, tg, ab) {
    f.energy -= ab.cost || 0;
    if (ab.cd) f.cds[ab.id] = ab.cd + 1; // +1 because it ticks down at end of this turn
    emit({ t: 'use', a: f.i, tg: tg.i, ab: ab.id, fx: ab.fx, name: ab.name });

    if (ab.energyGain) f.energy = Math.min(100, f.energy + ab.energyGain);
    if (ab.selfDamage) {
      const d = pct(f.maxHp, ab.selfDamage);
      f.hp = Math.max(1, f.hp - d);
      emit({ t: 'selfdmg', a: f.i, dmg: d, hp: f.hp });
    }
    if (ab.kind === 'heal') {
      const amt = Math.min(f.maxHp - f.hp, pct(f.maxHp, ab.heal));
      f.hp += amt;
      emit({ t: 'heal', tg: f.i, amt, hp: f.hp });
      if (ab.cleanse) {
        for (const x of f.statuses.filter(x => NEGATIVE.has(x.id))) emit({ t: 'expire', tg: f.i, st: x.id });
        f.statuses = f.statuses.filter(x => !NEGATIVE.has(x.id));
      }
    }
    for (const e of ab.selfEffects || []) applyStatus(f, f, e);

    const accFor = () => clamp(
      (ab.acc ?? 100) + Math.floor((stat(f, 'fcs') - stat(tg, 'spd')) / 2) + (env.accMod || 0)
        - (has(f, 'blind') ? 30 : 0) - (has(tg, 'evasive') ? 25 : 0),
      15, 100);

    if (ab.kind === 'debuff') {
      if (rng.roll(accFor())) for (const e of ab.effects || []) applyStatus(f, tg, e);
      else emit({ t: 'miss', a: f.i, tg: tg.i, hitNo: 1, hits: 1 });
      return;
    }
    if (ab.kind !== 'attack') return;

    const hits = ab.hits || 1;
    let landed = 0;
    for (let h = 1; h <= hits; h++) {
      if (!rng.roll(accFor())) { emit({ t: 'miss', a: f.i, tg: tg.i, hitNo: h, hits }); continue; }
      landed++;
      const type = ab.type === 'random' ? rng.pick(ab.randomTypes) : ab.type;
      let raw = Math.floor((ab.power * stat(f, ab.stat || 'pow')) / 20);
      raw = Math.floor((raw * 100) / (100 + stat(tg, 'arm') * 2));
      raw = pct(raw, rng.range(90, 110));
      const crit = rng.roll(clamp(5 + Math.floor(stat(f, 'fcs') / 2) + (ab.critBonus || 0), 0, 60));
      if (crit) raw = pct(raw, 150);
      raw = pct(raw, typeMult(type));
      raw = pct(raw, resistOf(tg, type));
      if (env.fxMult?.[ab.fx]) raw = pct(raw, env.fxMult[ab.fx]);
      if (has(tg, 'bind')) raw = pct(raw, 120);
      if (ab.execute && tg.hp * 100 < tg.maxHp * ab.execute) raw *= 2;
      const { dealt, absorbed } = damage(tg, Math.max(1, raw));
      emit({ t: 'hit', a: f.i, tg: tg.i, dmg: dealt, crit, type, hp: tg.hp, absorbed, hitNo: h, hits });
      afterDamage(tg);
      if (tg.hp <= 0) return;
    }
    if (landed) for (const e of ab.effects || []) applyStatus(f, tg, e);
  }

  function hazard(actions) {
    const hz = env.hazard;
    if (!hz || actions % (hz.every * 2) !== 0) return;
    let targets = F.filter(f => !f.frame.tags.some(t => (hz.immuneTags || []).includes(t)));
    if (hz.random && targets.length > 1) targets = [rng.pick(targets)];
    if (!targets.length) return;
    const hits = targets.map(f => {
      const d = pct(pct(pct(f.maxHp, hz.pct), typeMult(hz.type)), resistOf(f, hz.type));
      const { dealt } = damage(f, Math.max(1, d));
      return { tg: f.i, dmg: dealt, hp: f.hp };
    });
    emit({ t: 'hazard', name: hz.name, type: hz.type, hits });
    for (const f of targets) afterDamage(f);
    if (hz.status) for (const f of targets) if (f.hp > 0) applyStatus(f, f, hz.status);
  }

  // ---------- Main loop ----------
  emit({ t: 'start' });
  let actions = 0, reason = 'decision', winner = null;
  while (actions < MAX_ACTIONS) {
    // Advance initiative until someone reaches the threshold.
    let ready = F.filter(f => f.gauge >= GAUGE);
    while (!ready.length) {
      for (const f of F) f.gauge += stat(f, 'spd') * 5 + 100;
      ready = F.filter(f => f.gauge >= GAUGE);
    }
    let f = ready[0];
    if (ready.length > 1) f = ready[0].gauge === ready[1].gauge ? rng.pick(ready) : (ready[0].gauge > ready[1].gauge ? ready[0] : ready[1]);
    f.gauge -= GAUGE;
    const tg = F[1 - f.i];
    actions++;

    f.energy = Math.min(100, f.energy + 12 + Math.floor(stat(f, 'enr') / 3));
    emit({ t: 'turn', n: actions, a: f.i, en: f.energy });

    // Start-of-turn ticks: DoTs and regen.
    for (const x of [...f.statuses]) {
      if (!f.statuses.includes(x)) continue;   // cleansed by a passive earlier this tick
      if (STATUSES[x.id]?.dot && x.val > 0) {
        const { dealt } = damage(f, pct(x.val, typeMult(STATUSES[x.id].dot)), false);
        emit({ t: 'dot', tg: f.i, st: x.id, dmg: dealt, hp: f.hp });
        afterDamage(f);
      } else if (x.id === 'regen' && f.hp < f.maxHp) {
        const amt = Math.min(f.maxHp - f.hp, x.val);
        f.hp += amt;
        emit({ t: 'heal', tg: f.i, amt, hp: f.hp });
      }
    }
    if (koCheck()) { reason = 'ko'; break; }

    let skipped = null;
    if (has(f, 'stun')) skipped = 'stun';
    else if (has(f, 'bind') && rng.roll(50)) skipped = 'bind';
    if (skipped) {
      emit({ t: 'skip', a: f.i, reason: skipped });
      if (skipped === 'stun') { f.statuses = f.statuses.filter(x => x.id !== 'stun'); emit({ t: 'expire', tg: f.i, st: 'stun' }); }
    } else {
      useAbility(f, tg, choose(f, tg));
      if (koCheck()) { reason = 'ko'; break; }
    }

    // End of turn: cooldowns and status durations tick down.
    for (const k of Object.keys(f.cds)) if (f.cds[k] > 0) f.cds[k]--;
    for (const x of f.statuses) x.turns--;
    for (const x of f.statuses.filter(x => x.turns <= 0)) emit({ t: 'expire', tg: f.i, st: x.id });
    f.statuses = f.statuses.filter(x => x.turns > 0);

    hazard(actions);
    if (koCheck()) { reason = 'ko'; break; }
  }

  const alive = F.filter(f => !f.ko);
  if (reason === 'ko') {
    winner = alive.length === 1 ? alive[0].i : null;
    if (winner === null) reason = 'draw';
  } else {
    const p0 = Math.floor((F[0].hp * 1000) / F[0].maxHp), p1 = Math.floor((F[1].hp * 1000) / F[1].maxHp);
    winner = p0 === p1 ? null : p0 > p1 ? 0 : 1;
    if (winner === null) reason = 'draw';
  }
  emit({ t: 'end', winner, reason });

  return {
    v: ENGINE_VERSION, seed, env: env.id, fighters: snaps,
    meta: F.map(f => ({ maxHp: f.maxHp, stats: f.stats })),
    events, winner, reason, turns: actions, finalHp: F.map(f => f.hp), hash: fnv1a(JSON.stringify(events)),
  };
}

// Re-simulate a stored record and confirm it matches. Returns { ok, record }.
export function verifyReplay(record) {
  if (record.v !== ENGINE_VERSION) return { ok: false, stale: true, record };
  const re = simulateBattle({ seed: record.seed, env: record.env, a: record.fighters[0], b: record.fighters[1] });
  return { ok: re.hash === record.hash, record: re };
}
