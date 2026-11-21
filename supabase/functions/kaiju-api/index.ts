// kaiju-api — the whole Kaiju Clash backend, dispatched on `action`.
// Server-authoritative: costs, matchmaking, the battle simulation itself, rewards and Elo all
// happen here. The battle engine is the SAME file the browser runs (synced into _shared/kaiju by
// scripts/sync-engine.mjs), so the client can re-simulate any fight and check the hash.
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.45.4';
import { simulateBattle, sanitizeSnapshot, ENGINE_VERSION } from '../_shared/kaiju/engine/battle.js';
import { createRng, randomSeed } from '../_shared/kaiju/engine/rng.js';
import { FRAME_BY_ID, STATS, MAX_STAT_LEVEL, RARE_FRAME_IDS } from '../_shared/kaiju/data/frames.js';
import { ABILITY_BY_ID, MAX_EQUIPPED, abilityAvailableTo } from '../_shared/kaiju/data/abilities.js';
import { ENVIRONMENTS } from '../_shared/kaiju/data/environments.js';
import { statUpgradeCost, powerRating, battleReward, generateRival, pickRare, RARE_CHANCE, RARE_POWER_PCT, RARE_REWARD_MULT } from '../_shared/kaiju/data/economy.js';
import { INJURY_BY_ID, healMs, healCost } from '../_shared/kaiju/data/injuries.js';
import { rollAftermath } from '../_shared/kaiju/engine/aftermath.js';
import { FT_ELO, FT_UNLOCK_COST, TUNE_MAX, TUNE_MAX_PTS, tunePointCost, MOD_COST, RETUNE_COST, MODS, modAllowed, sanitizeTune, allocNet } from '../_shared/kaiju/data/finetune.js';

const MAX_KAIJU = 6;
const RATED_FIGHTS_PER_DAY = 40;
const REROLL_COST = 15;
const DEFENSE_WIN_BIO = 25;
const K_PVP = 32, K_WILD = 16;
const OFFER_TTL_MS = 60 * 60 * 1000;
const SALVAGE_PCT = 20; // % of invested Biomass returned when a Kaiju dies

const cors = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { ...cors, 'Content-Type': 'application/json' } });
class Fail extends Error { constructor(msg: string, public status = 400) { super(msg); } }

function serviceKey(): string {
  const legacy = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
  if (legacy) return legacy;
  const raw = Deno.env.get('SUPABASE_SECRET_KEYS');
  if (raw) {
    const p = JSON.parse(raw);
    if (typeof p === 'string') return p;
    if (p.default) return p.default;
    const first = Object.values(p)[0];
    if (typeof first === 'string') return first;
  }
  throw new Error('No service role key available.');
}
const db = createClient(Deno.env.get('SUPABASE_URL')!, serviceKey(), { auth: { persistSession: false } });

async function userId(req: Request): Promise<string> {
  const jwt = (req.headers.get('Authorization') || '').replace(/^Bearer\s+/i, '').trim();
  if (!jwt) throw new Fail('Not signed in.', 401);
  const { data, error } = await db.auth.getUser(jwt);
  if (error || !data?.user) throw new Fail('Session expired, please sign in again.', 401);
  return data.user.id;
}
const must = <T>(r: { data: T; error: { message: string } | null }, what: string): T => {
  if (r.error) throw new Fail(`${what}: ${r.error.message}`, 500);
  return r.data;
};

type Kaiju = {
  id: string; owner: string; name: string; frame: string; levels: Record<string, number>;
  abilities: string[]; unlocked: string[]; colors: Record<string, string>; elo: number; version: number;
  wins: number; losses: number; draws: number; def_wins: number; def_losses: number; retired: boolean;
  injuries: Injury[]; died_at?: string | null; death?: unknown; tune?: any;
};
type Injury = { id: string; at: string; heals_at: string };
// Injuries that haven't healed yet. Healed ones are dropped on the next write.
const activeInjuries = (k: Kaiju, now = Date.now()): Injury[] =>
  (Array.isArray(k.injuries) ? k.injuries : []).filter(i => INJURY_BY_ID[i?.id] && Date.parse(i.heals_at) > now);
const withActive = (k: Kaiju) => ({ ...k, injuries: activeInjuries(k) });
function invested(k: Kaiju) {
  let total = 0;
  for (const s of STATS) for (let l = 0; l < (k.levels[s] || 0); l++) total += statUpgradeCost(l) || 0;
  for (const id of k.unlocked || []) total += ABILITY_BY_ID[id]?.price || 0;
  if (k.tune?.on) {
    total += FT_UNLOCK_COST + MOD_COST * Object.keys(k.tune.mods || {}).length;
    for (let p = 0; p < (k.tune.pts || 0); p++) total += tunePointCost(p);
  }
  return total;
}
const snapshot = (k: Kaiju, ownerName?: string) =>
  sanitizeSnapshot({ id: k.id, name: k.name, frame: k.frame, levels: k.levels, abilities: k.abilities, colors: k.colors, owner: ownerName, injuries: activeInjuries(k).map(i => i.id), tune: k.tune });
const cleanColors = (c: unknown, fallback: Record<string, string>) => {
  const out = { ...fallback };
  if (c && typeof c === 'object') for (const key of ['body', 'belly', 'accent']) {
    const v = (c as Record<string, unknown>)[key];
    if (typeof v === 'string' && /^#[0-9a-f]{6}$/i.test(v)) out[key] = v;
  }
  return out;
};
const cleanName = (s: unknown, max: number) => String(s ?? '').replace(/[\u0000-\u001f<>]/g, '').trim().slice(0, max);
// The daily rated-fight counter resets at midnight Pacific (DST-aware). Must match kb_settle's day expression.
const RESET_TZ = 'America/Los_Angeles';
const gameDay = () => new Intl.DateTimeFormat('en-CA', { timeZone: RESET_TZ, year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());

async function getPlayer(uid: string) {
  return must(await db.from('kb_players').select('*').eq('user_id', uid).maybeSingle(), 'player lookup');
}
async function requirePlayer(uid: string) {
  const p = await getPlayer(uid);
  if (!p) throw new Fail('Pick a display name first.', 409);
  return p;
}
async function myKaiju(uid: string): Promise<Kaiju[]> {
  return (must(await db.from('kb_kaiju').select('*').eq('owner', uid).eq('retired', false).order('created_at'), 'kaiju list') as Kaiju[]).map(withActive);
}
async function ownKaiju(uid: string, id: unknown): Promise<Kaiju> {
  const k = must(await db.from('kb_kaiju').select('*').eq('id', String(id)).eq('owner', uid).eq('retired', false).maybeSingle(), 'kaiju') as Kaiju | null;
  if (!k) throw new Fail('Kaiju not found.', 404);
  return k;
}
async function spend(uid: string, k: Kaiju, cost: number, patch: Partial<Kaiju>) {
  const n = { ...k, ...patch };
  const { data, error } = await db.rpc('kb_spend', {
    p_user: uid, p_kaiju: k.id, p_version: k.version, p_cost: cost, p_levels: n.levels,
    p_abilities: n.abilities, p_unlocked: n.unlocked, p_colors: n.colors, p_name: n.name,
    p_injuries: patch.injuries ?? activeInjuries(k), p_tune: n.tune ?? null,
  });
  if (error) throw new Fail(error.message.replace(/^.*?: /, ''), 409);
  return data as Kaiju;
}

// ---------- Matchmaking ----------
async function ensureOffers(uid: string, player: any, k: Kaiju, force = false) {
  let offers = must(await db.from('kb_offers').select('*').eq('user_id', uid).order('slot'), 'offers') as any[];
  const fresh = offers.length === 3 && offers.every(o => o.kaiju_id === k.id && Date.now() - Date.parse(o.created_at) < OFFER_TTL_MS);
  if (fresh && !force) return offers;

  const rng = createRng(Number(BigInt(player.rival_seed) & 0xffffffffn) ^ 0x2545f491);
  // Real opponents: other players' active kaiju near this Elo, widening the window if needed.
  let pool: Kaiju[] = [];
  for (const win of [150, 350, 800, 5000]) {
    pool = must(await db.from('kb_kaiju').select('*').neq('owner', uid).eq('retired', false)
      .gte('elo', k.elo - win).lte('elo', k.elo + win).limit(60), 'opponents') as Kaiju[];
    if (pool.length >= 2) break;
  }
  pool.sort((a, b) => (a.elo - b.elo) || a.id.localeCompare(b.id));
  const picks: (Kaiju | null)[] = [];
  const used = new Set<string>(), usedOwners = new Set<string>();
  // At most 2 real players: one slot is always a Wild Kaiju (sometimes a rare one).
  for (let tries = 0; picks.length < 2 && tries < 40 && pool.length; tries++) {
    const c = pool[rng.int(pool.length)];
    if (used.has(c.id) || (usedOwners.has(c.owner) && pool.length > 3)) continue;
    used.add(c.id); usedOwners.add(c.owner); picks.push(c);
  }
  picks.sort((a, b) => (a ? a.elo : 0) - (b ? b.elo : 0));
  // Players stay in Elo order; the guaranteed Wild slot lands at a random position (so its difficulty varies).
  picks.splice(rng.int(picks.length + 1), 0, null);
  while (picks.length < 3) picks.push(null);

  const myPower = powerRating(k);
  const rows = picks.map((opp, slot) => {
    // Wild slots occasionally roll a rare Wild-only enemy instead.
    const rare = !opp && rng.int(100) < RARE_CHANCE ? pickRare(rng) : null;
    const base = Math.floor((myPower * [85, 100, 120][slot]) / 100);
    return {
    user_id: uid, slot, kaiju_id: k.id,
    opponent: opp ? opp.id : null,
    rival_seed: opp ? null : (rng.next() >>> 0),
    rival_rating: opp ? null : (rare ? Math.floor((base * RARE_POWER_PCT) / 100) : base),
    rival_frame: rare,
    env: rng.pick(ENVIRONMENTS).id,
    created_at: new Date().toISOString(),
  }; });
  must(await db.from('kb_offers').delete().eq('user_id', uid), 'clear offers');
  must(await db.from('kb_offers').insert(rows), 'save offers');
  return rows;
}

async function describeOffers(offers: any[], me: Kaiju) {
  const ids = offers.map(o => o.opponent).filter(Boolean);
  const opps = ids.length ? must(await db.from('kb_kaiju').select('*').in('id', ids), 'offer kaiju') as Kaiju[] : [];
  const owners = await ownerNames(opps.map(o => o.owner));
  return offers.map(o => {
    const opp = opps.find(x => x.id === o.opponent);
    if (opp) return { slot: o.slot, env: o.env, kind: 'player', elo: opp.elo, record: [opp.wins, opp.losses, opp.draws], snapshot: snapshot(opp, owners[opp.owner]) };
    const rv = generateRival(Number(o.rival_seed), o.rival_rating, o.rival_frame);
    return { slot: o.slot, env: o.env, kind: o.rival_frame ? 'rare' : 'wild', elo: me.elo + [-100, 0, 120][o.slot], snapshot: rv };
  });
}
async function ownerNames(ids: string[]) {
  const uniq = [...new Set(ids)];
  if (!uniq.length) return {} as Record<string, string>;
  const rows = must(await db.from('kb_players').select('user_id, display_name').in('user_id', uniq), 'names') as any[];
  return Object.fromEntries(rows.map(r => [r.user_id, r.display_name])) as Record<string, string>;
}

function eloDelta(ra: number, rb: number, score: number, k: number) {
  const exp = 1 / (1 + Math.pow(10, (rb - ra) / 400));
  return Math.round(k * (score - exp));
}

// ---------- Actions ----------
async function bootstrap(uid: string) {
  const player = await getPlayer(uid);
  if (!player) return { needsName: true };
  const kaiju = await myKaiju(uid);
  let active = kaiju.find(k => k.id === player.active_kaiju) || kaiju[0] || null;
  // Defense report: fights other players started against you since you last looked.
  const defenses = must(await db.from('kb_battles').select('id, created_at, fighters, winner, env, defense_reward, elo_delta_d, aftermath')
    .eq('defender_owner', uid).gt('created_at', player.defense_seen_at).order('created_at', { ascending: false }).limit(20), 'defenses');
  if (defenses.length) await db.from('kb_players').update({ defense_seen_at: new Date().toISOString() }).eq('user_id', uid);
  return {
    player: { name: player.display_name, bio: player.bio, active: active?.id || null,
      fightsToday: player.fights_day === gameDay() ? player.fights_today : 0, ratedPerDay: RATED_FIGHTS_PER_DAY },
    kaiju, defenses, engine: ENGINE_VERSION,
    trophies: Object.entries((must(await db.from('kb_battles').select('rare:fighters->1->>frame')
      .eq('attacker_owner', uid).eq('winner', 0).is('defender_kaiju', null).in('fighters->1->>frame', RARE_FRAME_IDS).limit(2000), 'trophies') as any[])
      .reduce((acc: Record<string, number>, r) => { acc[r.rare] = (acc[r.rare] || 0) + 1; return acc; }, {})),
    fallen: must(await db.from('kb_kaiju').select('id, name, frame, levels, abilities, colors, injuries, wins, losses, elo, died_at, death, tune')
      .eq('owner', uid).not('died_at', 'is', null).order('died_at', { ascending: false }).limit(20), 'fallen'),
  };
}

const actions: Record<string, (uid: string, b: any) => Promise<unknown>> = {
  async me(uid) { return bootstrap(uid); },

  async register(uid, b) {
    const name = cleanName(b.display_name, 20);
    if (name.length < 2) throw new Fail('Name must be 2-20 characters.');
    if (await getPlayer(uid)) throw new Fail('Already registered.');
    const { error } = await db.from('kb_players').insert({ user_id: uid, display_name: name, rival_seed: randomSeed() });
    if (error) throw new Fail(error.code === '23505' ? 'That name is taken.' : error.message);
    return bootstrap(uid);
  },

  async create_kaiju(uid, b) {
    await requirePlayer(uid);
    const frame = FRAME_BY_ID[b.frame];
    if (!frame || frame.wildOnly) throw new Fail('Unknown frame.');
    if ((await myKaiju(uid)).length >= MAX_KAIJU) throw new Fail(`Max ${MAX_KAIJU} Kaiju. Release one first.`);
    const name = cleanName(b.name, 24) || frame.name;
    const starter = frame.exclusive[0];
    const k = must(await db.from('kb_kaiju').insert({
      owner: uid, name, frame: frame.id, levels: Object.fromEntries(STATS.map((s: string) => [s, 0])),
      abilities: [starter], unlocked: [starter], colors: cleanColors(b.colors, frame.colors),
    }).select().single(), 'create kaiju') as Kaiju;
    await db.from('kb_players').update({ active_kaiju: k.id }).eq('user_id', uid);
    return bootstrap(uid);
  },

  async upgrade(uid, b) {
    const k = await ownKaiju(uid, b.kaiju_id);
    if (!STATS.includes(b.stat)) throw new Fail('Unknown stat.');
    const lv = k.levels[b.stat] || 0;
    const cost = statUpgradeCost(lv);
    if (cost === null || lv >= MAX_STAT_LEVEL) throw new Fail('Already maxed.');
    await spend(uid, k, cost, { levels: { ...k.levels, [b.stat]: lv + 1 } });
    return bootstrap(uid);
  },

  async unlock(uid, b) {
    const k = await ownKaiju(uid, b.kaiju_id);
    const ab = ABILITY_BY_ID[b.ability];
    if (!ab || ab.basic || !abilityAvailableTo(ab, FRAME_BY_ID[k.frame])) throw new Fail('Ability not available to this frame.');
    if (k.unlocked.includes(ab.id)) throw new Fail('Already unlocked.');
    const abilities = k.abilities.length < MAX_EQUIPPED ? [...k.abilities, ab.id] : k.abilities;
    await spend(uid, k, ab.price || 0, { unlocked: [...k.unlocked, ab.id], abilities });
    return bootstrap(uid);
  },

  async equip(uid, b) {
    const k = await ownKaiju(uid, b.kaiju_id);
    const list = [...new Set((Array.isArray(b.abilities) ? b.abilities : []).map(String))] as string[];
    if (list.length > MAX_EQUIPPED || list.some(id => !k.unlocked.includes(id))) throw new Fail('Invalid loadout.');
    await spend(uid, k, 0, { abilities: list });
    return bootstrap(uid);
  },

  async cosmetic(uid, b) {
    const k = await ownKaiju(uid, b.kaiju_id);
    await spend(uid, k, 0, { colors: cleanColors(b.colors, k.colors), name: cleanName(b.name, 24) || k.name });
    return bootstrap(uid);
  },

  // ---------- Fine Tune ----------
  async ft_unlock(uid, b) {
    const k = await ownKaiju(uid, b.kaiju_id);
    if (k.tune?.on) throw new Fail('Fine Tune is already unlocked for this Kaiju.');
    if (k.elo < FT_ELO) throw new Fail(`Fine Tune needs this Kaiju at ${FT_ELO} Elo (it has ${k.elo}).`);
    await spend(uid, k, FT_UNLOCK_COST, { tune: { on: true, pts: 0, alloc: {}, mods: {} } });
    return bootstrap(uid);
  },

  // Buy one tuning point (optionally placing it on a stat right away).
  async ft_point(uid, b) {
    const k = await ownKaiju(uid, b.kaiju_id);
    const t = k.tune;
    if (!t?.on) throw new Fail('Unlock Fine Tune first.');
    if ((t.pts || 0) >= TUNE_MAX_PTS) throw new Fail(`Max ${TUNE_MAX_PTS} tuning points.`);
    const alloc = { ...(t.alloc || {}) };
    if (STATS.includes(b.stat) && (alloc[b.stat] || 0) < TUNE_MAX) alloc[b.stat] = (alloc[b.stat] || 0) + 1;
    await spend(uid, k, tunePointCost(t.pts || 0), { tune: { ...t, pts: (t.pts || 0) + 1, alloc } });
    return bootstrap(uid);
  },

  // Re-allocate points. Free if it only places unspent points; otherwise costs RETUNE_COST.
  async ft_alloc(uid, b) {
    const k = await ownKaiju(uid, b.kaiju_id);
    const t = k.tune;
    if (!t?.on) throw new Fail('Unlock Fine Tune first.');
    const alloc: Record<string, number> = {};
    for (const st of STATS) {
      const v = Math.trunc(Number(b.alloc?.[st]) || 0);
      if (v < -TUNE_MAX || v > TUNE_MAX) throw new Fail(`Each stat can be tuned ${TUNE_MAX} steps up or down.`);
      if (v) alloc[st] = v;
    }
    if (allocNet(alloc) > (t.pts || 0)) throw new Fail('Not enough tuning points.');
    const old = t.alloc || {};
    const onlyAdds = STATS.every((st: string) => (alloc[st] || 0) >= (old[st] || 0));
    await spend(uid, k, onlyAdds ? 0 : RETUNE_COST, { tune: { ...t, alloc } });
    return bootstrap(uid);
  },

  // Apply, swap or remove a skill mod. Removing is free; applying/swapping costs MOD_COST.
  async ft_mod(uid, b) {
    const k = await ownKaiju(uid, b.kaiju_id);
    const t = k.tune;
    if (!t?.on) throw new Fail('Unlock Fine Tune first.');
    const id = String(b.ability || ''), ab = ABILITY_BY_ID[id];
    const owns = ab && (FRAME_BY_ID[k.frame].basic === id || k.unlocked.includes(id));
    if (!owns) throw new Fail('This Kaiju does not have that ability.');
    const mods = { ...(t.mods || {}) };
    if (!b.mod) { delete mods[id]; await spend(uid, k, 0, { tune: { ...t, mods } }); return bootstrap(uid); }
    if (!MODS[b.mod] || !modAllowed(ab, b.mod)) throw new Fail('That mod does not fit this ability.');
    if (mods[id] === b.mod) throw new Fail('Already has that mod.');
    mods[id] = b.mod;
    await spend(uid, k, MOD_COST, { tune: { ...t, mods } });
    return bootstrap(uid);
  },

  async set_active(uid, b) {
    const k = await ownKaiju(uid, b.kaiju_id);
    must(await db.from('kb_players').update({ active_kaiju: k.id }).eq('user_id', uid), 'set active');
    return bootstrap(uid);
  },

  async retire(uid, b) {
    const k = await ownKaiju(uid, b.kaiju_id);
    if ((await myKaiju(uid)).length <= 1) throw new Fail('You need at least one Kaiju.');
    must(await db.from('kb_kaiju').update({ retired: true }).eq('id', k.id), 'retire');
    return bootstrap(uid);
  },

  async heal(uid, b) {
    const k = await ownKaiju(uid, b.kaiju_id);
    const now = Date.now(), list = activeInjuries(k, now);
    const target = b.injury === 'all' ? list : list.filter(i => i.id === b.injury);
    if (!target.length) throw new Fail('Nothing to heal.');
    const cost = target.reduce((sum, i) => sum + healCost(i, Date.parse(i.heals_at) - now), 0);
    await spend(uid, k, cost, { injuries: list.filter(i => !target.includes(i)) });
    return bootstrap(uid);
  },

  async offers(uid, b) {
    const player = await requirePlayer(uid);
    const k = await ownKaiju(uid, b.kaiju_id || player.active_kaiju || (await myKaiju(uid))[0]?.id);
    if (b.reroll) {
      const { data, error } = await db.from('kb_players').update({ bio: player.bio - REROLL_COST, rival_seed: randomSeed() })
        .eq('user_id', uid).gte('bio', REROLL_COST).eq('bio', player.bio).select().maybeSingle();
      if (error || !data) throw new Fail('Not enough Biomass.');
      return { offers: await describeOffers(await ensureOffers(uid, data, k, true), k), bio: data.bio };
    }
    return { offers: await describeOffers(await ensureOffers(uid, player, k), k), bio: player.bio };
  },

  async fight(uid, b) {
    const player = await requirePlayer(uid);
    const k = await ownKaiju(uid, player.active_kaiju || (await myKaiju(uid))[0]?.id);
    const slot = Number(b.slot);
    const offer = must(await db.from('kb_offers').select('*').eq('user_id', uid).eq('slot', slot).maybeSingle(), 'offer') as any;
    if (!offer || offer.kaiju_id !== k.id) throw new Fail('Offer expired, refresh opponents.', 409);

    let opp: Kaiju | null = null, oppSnap: any, oppElo: number, oppOwnerName: string | undefined;
    if (offer.opponent) {
      opp = must(await db.from('kb_kaiju').select('*').eq('id', offer.opponent).eq('retired', false).maybeSingle(), 'opponent') as Kaiju | null;
      if (!opp) throw new Fail('That opponent retired, refresh opponents.', 409);
      oppOwnerName = (await ownerNames([opp.owner]))[opp.owner];
      oppSnap = snapshot(opp, oppOwnerName);
      oppElo = opp.elo;
    } else {
      oppSnap = generateRival(Number(offer.rival_seed), offer.rival_rating, offer.rival_frame);
      oppElo = k.elo + [-100, 0, 120][slot];
    }

    const seed = randomSeed();
    const record = simulateBattle({ seed, env: offer.env, a: snapshot(k, player.display_name), b: oppSnap });
    const result = record.winner === 0 ? 'win' : record.winner === 1 ? 'loss' : 'draw';
    const score = result === 'win' ? 1 : result === 'loss' ? 0 : 0.5;
    const fightsToday = player.fights_day === gameDay() ? player.fights_today : 0;
    const rated = fightsToday < RATED_FIGHTS_PER_DAY;
    const rareWin = !!offer.rival_frame && result === 'win';
    const reward = rated ? battleReward(result, powerRating(k), powerRating(record.fighters[1])) * (rareWin ? RARE_REWARD_MULT : 1) : 0;
    const deltaA = rated ? eloDelta(k.elo, oppElo, score, opp ? K_PVP : K_WILD) : 0;
    const deltaD = rated && opp ? -eloDelta(k.elo, oppElo, score, K_PVP) : 0;
    const defReward = rated && opp && result === 'loss' ? DEFENSE_WIN_BIO : 0;

    // Aftermath: only the attacker can be injured or die. Defenders weren't there to choose the fight,
    // so they never come back to an injured (or dead) Kaiju.
    const aftermath = rollAftermath(record, { injurable: [true, false], canDie: [true, false] });
    const nowMs = Date.now(), nowIso = new Date(nowMs).toISOString();
    const entry = (id: string | null) => id ? [{ id, at: nowIso, heals_at: new Date(nowMs + healMs(id)).toISOString() }] : [];
    const died = aftermath[0].died;
    const salvage = died ? Math.floor((invested(k) * SALVAGE_PCT) / 100) : 0;
    const death = died ? { killer: oppSnap.name, killer_owner: oppOwnerName || 'Wild Kaiju', env: record.env, salvage } : null;

    const { data: battleId, error } = await db.rpc('kb_settle', {
      p_user: uid, p_slot: slot, p_attacker: k.id, p_defender: opp?.id ?? null, p_defender_owner: opp?.owner ?? null,
      p_engine_v: record.v, p_seed: seed, p_env: record.env, p_fighters: record.fighters, p_winner: record.winner,
      p_reason: record.reason, p_turns: record.turns, p_hash: record.hash, p_rated: rated, p_reward: reward,
      p_def_reward: defReward, p_delta_a: deltaA, p_delta_d: deltaD, p_new_rival_seed: randomSeed(),
      p_aftermath: aftermath, p_new_inj_a: entry(aftermath[0].injury), p_new_inj_d: entry(aftermath[1].injury),
      p_died_a: died, p_death: death, p_salvage: salvage,
    });
    if (error) throw new Fail(error.message.replace(/^.*?: /, ''), 409);

    // Events are NOT sent: the client re-simulates from (v, seed, env, fighters) and checks the hash.
    return {
      battle: { id: battleId, v: record.v, seed, env: record.env, fighters: record.fighters, winner: record.winner,
        reason: record.reason, turns: record.turns, hash: record.hash },
      result, rated, reward, eloDelta: deltaA, opponentKind: opp ? 'player' : offer.rival_frame ? 'rare' : 'wild', aftermath, salvage,
      ...(await bootstrap(uid)),
    };
  },

  async history(uid) {
    const rows = must(await db.from('kb_battles').select('id, created_at, attacker_owner, defender_owner, engine_v, seed, env, fighters, winner, reason, reward, defense_reward, elo_delta_a, elo_delta_d, rated, aftermath')
      .or(`attacker_owner.eq.${uid},defender_owner.eq.${uid}`).order('created_at', { ascending: false }).limit(30), 'history') as any[];
    return { history: rows.map(r => ({ ...r, side: r.attacker_owner === uid ? 0 : 1, attacker_owner: undefined, defender_owner: undefined })) };
  },

  async battle(_uid, b) {
    const r = must(await db.from('kb_battles').select('id, engine_v, seed, env, fighters, winner, reason, turns, hash, aftermath').eq('id', String(b.id)).maybeSingle(), 'battle') as any;
    if (!r) throw new Fail('Battle not found.', 404);
    return { battle: { id: r.id, v: r.engine_v, seed: Number(r.seed), env: r.env, fighters: r.fighters, winner: r.winner, reason: r.reason, turns: r.turns, hash: r.hash }, aftermath: r.aftermath };
  },

  async leaderboard() {
    const rows = must(await db.from('kb_kaiju').select('*').eq('retired', false).order('elo', { ascending: false }).limit(50), 'leaderboard') as Kaiju[];
    const names = await ownerNames(rows.map(r => r.owner));
    return { ladder: rows.map(k => ({ id: k.id, elo: k.elo, record: [k.wins, k.losses, k.draws], defense: [k.def_wins, k.def_losses], power: powerRating(k), snapshot: snapshot(k, names[k.owner]) })) };
  },
};
const PUBLIC = new Set(['leaderboard', 'battle']);

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: cors });
  try {
    const body = await req.json().catch(() => ({}));
    const fn = actions[body.action];
    if (!fn) throw new Fail('Unknown action.');
    const uid = PUBLIC.has(body.action) ? '' : await userId(req);
    return json({ ok: true, ...(await fn(uid, body) as object) });
  } catch (e) {
    const status = e instanceof Fail ? e.status : 500;
    if (status >= 500) console.error(e);
    return json({ ok: false, error: (e as Error).message || String(e) }, status);
  }
});
