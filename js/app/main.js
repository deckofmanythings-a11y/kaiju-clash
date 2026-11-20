import { FRAMES, FRAME_BY_ID, STATS, STAT_NAMES, STAT_BLURBS, MAX_STAT_LEVEL, effectiveStats, PASSIVES, WILD_FRAMES } from '../data/frames.js';
import { ABILITY_BY_ID, STATUSES, MAX_EQUIPPED, abilitiesForFrame, abilityScope } from '../data/abilities.js';
import { ENV_BY_ID, ENVIRONMENTS } from '../data/environments.js';
import { statUpgradeCost, powerRating, battleReward, RARE_REWARD_MULT } from '../data/economy.js';
import { simulateBattle, verifyReplay, ENGINE_VERSION } from '../engine/battle.js';
import { randomSeed, createRng } from '../engine/rng.js';
import { INJURY_BY_ID, conditionOf, deathChance, healCost, applyInjuries } from '../data/injuries.js';
import * as codes from './codes.js';
import * as api from './api.js';
// Shown in the top bar. Bump on every push, together with the ?v= on main.js in index.html.
// Minor = feature milestone (1.1 online ladder, 1.2 injuries, 1.3 rare enemies), patch = fixes.
const APP_VERSION = '1.5.0';

document.getElementById('ver').textContent = `Ver. ${APP_VERSION} · engine v${ENGINE_VERSION}`;

const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const panel = $('#panel'), hud = $('#hud'), modal = $('#modal');
const NEG = new Set(['burn', 'poison', 'bleed', 'stun', 'bind', 'slow', 'blind', 'weaken', 'sunder']);

let me = null;                 // server bootstrap: { player, kaiju, defenses }
let screen = null, offers = null, ladder = null, history = null;
let draft = { frame: 'saurian', name: '', colors: { ...FRAME_BY_ID.saurian.colors } };
let stage = null, playBattle = null, playing = null, currentEnv = null;
let authMode = 'signin';
let showcaseOverride = null;   // e.g. a fallen Kaiju's statue in the Hangar

// ---------- helpers ----------
function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 3000);
}
const active = () => me?.kaiju?.find(k => k.id === me.player.active) || me?.kaiju?.[0] || null;
// ---------- limits ----------
const MAX_KAIJU = 6;
const LOW_RATED = 5;   // start warning when this few rated fights are left
const ratedLeft = () => (me?.player ? Math.max(0, me.player.ratedPerDay - me.player.fightsToday) : 0);
function resetIn() {
  const n = new Date(), m = Math.max(1, Math.ceil((Date.UTC(n.getUTCFullYear(), n.getUTCMonth(), n.getUTCDate() + 1) - n) / 60000));
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m`;
}
// Big, impossible-to-miss explanation of the daily rated-fight limit (or '' when there's plenty left).
function limitBanner() {
  if (!me?.player) return '';
  const left = ratedLeft(), per = me.player.ratedPerDay;
  if (!left) return `<div class="limit">
      <div class="limit-h">⛔ Daily Biomass limit reached</div>
      <p>You've used all <b>${per} rated fights</b> for today. You can keep fighting, but fights are now <b>unrated</b>: they pay <b>no Biomass and no Elo</b>, and your Kaiju can still be injured.</p>
      <p>Rated fights reset in <b>${resetIn()}</b> (midnight UTC). You can still spend the Biomass you have in the Lab.</p>
    </div>`;
  if (left <= LOW_RATED) return `<div class="limit soft">
      <div class="limit-h">⚠ Only ${left} rated fight${left === 1 ? '' : 's'} left today</div>
      <p>Each day you get ${per} rated fights. After that, fights pay no Biomass or Elo until the reset in <b>${resetIn()}</b> (midnight UTC).</p>
    </div>`;
  return '';
}
function updateTop() {
  $('#bio').textContent = me?.player ? me.player.bio.toLocaleString() : '–';
  const chip = $('#limchip'), left = ratedLeft();
  chip.hidden = !me?.player || left > LOW_RATED;
  if (!chip.hidden) {
    chip.classList.toggle('out', !left);
    chip.textContent = left ? `⚠ ${left} rated left` : '⛔ No BIO until reset';
    chip.title = left ? `${left} rated fights left today. After that, fights pay no Biomass or Elo until midnight UTC.` : `Daily limit of ${me.player.ratedPerDay} rated fights reached. Fights pay no Biomass or Elo until midnight UTC (in ${resetIn()}).`;
  }
  const a = $('#acct');
  a.hidden = !me?.player; if (me?.player) a.textContent = `${me.player.name} · Sign out`;
  $('#nav').style.visibility = me?.player ? 'visible' : 'hidden';
}
function applyBoot(b) {
  if (b.player) me = { player: b.player, kaiju: b.kaiju, defenses: b.defenses || [], fallen: b.fallen || [], trophies: Object.fromEntries(b.trophies || []) };
  if (b.needsName) me = { needsName: true };
  updateTop();
}
// Every server mutation goes through here: lock the panel, apply the fresh bootstrap, re-render.
async function act(action, payload, { rerender = true } = {}) {
  panel.classList.add('busy');
  try {
    const res = await api.call(action, payload);
    applyBoot(res);
    if (rerender) { const y = panel.scrollTop; render(); panel.scrollTop = y; refreshShowcase(); }
    return res;
  } catch (e) {
    toast(e.message);
    if (e.status === 401) { me = null; go('auth'); }
    return null;
  } finally { panel.classList.remove('busy'); }
}

// ---------- injuries ----------
const liveInjuries = (k) => (k?.injuries || []).filter(i => INJURY_BY_ID[i.id] && Date.parse(i.heals_at) > Date.now());
const injIds = (k) => liveInjuries(k).map(i => i.id);
const snap = (k, extra = {}) => ({ ...k, injuries: injIds(k), ...extra });   // what the engine/renderer see
const injText = (id) => Object.entries(INJURY_BY_ID[id].mods).map(([k, v]) => `${v}% ${STAT_NAMES[k]}`).join(', ');
const timeLeft = (iso) => { const m = Math.max(0, Math.ceil((Date.parse(iso) - Date.now()) / 60000)); return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m`; };
function condBadge(ids) {
  const c = conditionOf(ids);
  return `<span class="cond" style="--c:${c.color}">${c.name}${c.severity ? ' · ' + c.severity : ''}</span>`;
}
function aftermathHtml(am, fighters, { mySide = null } = {}) {
  if (!am) return '';
  return am.map((x, i) => {
    const nm = esc(fighters[i].name), bits = [];
    if (x.died) bits.push(`<div class="death">☠ ${nm} has died.</div>`);
    if (x.injury) bits.push(`<div class="inj">🩸 ${nm} suffered <b>${INJURY_BY_ID[x.injury].name}</b> <span class="rec">(${injText(x.injury)})</span></div>`);
    else if (x.deathChance && !x.died && mySide === i) bits.push(`<div class="rec">${nm} survived a ${x.deathChance}% death roll.</div>`);
    return bits.join('');
  }).join('');
}

function describe(ab) {
  const bits = [];
  if (ab.kind === 'attack') bits.push(`${ab.type === 'random' ? ab.randomTypes.join('/') : ab.type} · power ${ab.power}${ab.hits ? ' ×' + ab.hits : ''} · scales ${STAT_NAMES[ab.stat || 'pow']}`);
  if (ab.heal) bits.push(`heals ${ab.heal}% HP${ab.cleanse ? ', cleanses debuffs' : ''}`);
  for (const e of ab.effects || []) bits.push(`${e.chance < 100 ? e.chance + '% ' : ''}${STATUSES[e.status].name} ${e.turns}t`);
  for (const e of ab.selfEffects || []) bits.push(`self: ${STATUSES[e.status].name} ${e.turns}t`);
  if (ab.energyGain) bits.push(`+${ab.energyGain} energy`);
  if (ab.selfDamage) bits.push(`costs ${ab.selfDamage}% HP`);
  if (ab.execute) bits.push(`double damage below ${ab.execute}% HP`);
  if (ab.critBonus) bits.push(`+${ab.critBonus}% crit`);
  return bits.join(' · ');
}

// ---------- 3D stage (optional) ----------
function setEnv(id) { if (stage && currentEnv !== id) { stage.setEnvironment(id); currentEnv = id; } }
function refreshShowcase() {
  if (!stage) return;
  try {
    if (screen === 'creator') stage.showcase({ name: draft.name, frame: draft.frame, levels: {}, abilities: [FRAME_BY_ID[draft.frame].exclusive[0]], colors: draft.colors });
    else if (showcaseOverride) stage.showcase(showcaseOverride);
    else { const k = active(); stage.showcase(k ? snap(k) : { frame: 'saurian', levels: {}, abilities: [], colors: FRAME_BY_ID.saurian.colors }); }
  } catch (e) { console.warn(e); }
}
async function initStage() {
  try {
    const [st, pb] = await Promise.all([import('../render/stage.js'), import('../render/playback.js')]);
    stage = st.createStage($('#stage'));
    playBattle = pb.playBattle;
    await import('../render/wild-models.js').catch((e) => console.warn('rare enemy models unavailable', e));
    window.addEventListener('resize', () => stage.resize());
    setEnv('city'); refreshShowcase();
  } catch (e) { console.warn('3D renderer unavailable, running text-only', e); }
}

// ---------- Router ----------
const SCREENS = { auth, name: nameScreen, hangar, creator, lab, arena, ladder: ladderScreen, replays };
function go(s) {
  if (playing) { playing.stop(); playing = null; }
  applyPendingBoot();   // leaving a fight early still settles the displayed state
  if (!me) s = 'auth';
  else if (me.needsName) s = 'name';
  else if (!active() && !['creator', 'ladder', 'replays'].includes(s)) s = 'creator';
  screen = s; showcaseOverride = null;
  hud.hidden = true; panel.hidden = false; modal.hidden = true;
  for (const b of document.querySelectorAll('#nav button')) b.classList.toggle('on', b.dataset.go === s);
  updateTop();
  render();
  setEnv('city'); refreshShowcase();
}
function render() { SCREENS[screen](); }
document.addEventListener('click', (e) => {
  const g = e.target.closest('[data-go]'); if (g) go(g.dataset.go);
  if (e.target.closest('#acct')) signOut();
});
async function signOut() { await api.signOut(); me = null; offers = ladder = history = null; go('auth'); }

// ---------- Auth ----------
function auth() {
  const recovering = authMode === 'recover';
  panel.innerHTML = `<div class="auth">
    <h2>${recovering ? 'Set a new password' : authMode === 'signup' ? 'Create account' : 'Sign in'}</h2>
    <p class="sub">${recovering ? 'Choose a new password for your account.' : 'Build a Kaiju, then climb a ladder against other players’ Kaiju. Fights run on the server and replay identically.'}</p>
    <form id="authf">
      ${recovering ? '' : '<label>Email</label><input type="email" id="em" autocomplete="email" required>'}
      ${authMode === 'forgot' ? '' : `<label>${recovering ? 'New password' : 'Password'}</label><input type="password" id="pw" minlength="6" autocomplete="${authMode === 'signin' ? 'current-password' : 'new-password'}" required>`}
      <button class="btn big" style="margin-top:14px">${({ signin: 'Sign in', signup: 'Create account', forgot: 'Send reset email', recover: 'Save password' })[authMode]}</button>
      <div class="err" id="aerr"></div>
    </form>
    ${recovering ? '' : `<div class="row" style="justify-content:space-between">
      <button class="linkbtn" data-mode="${authMode === 'signin' ? 'signup' : 'signin'}">${authMode === 'signin' ? 'New here? Create an account' : 'Have an account? Sign in'}</button>
      ${authMode === 'signin' ? '<button class="linkbtn" data-mode="forgot">Forgot password</button>' : ''}</div>`}
    <h3>Watch a replay</h3>
    <textarea id="rcode" placeholder="KR1.… replay code"></textarea>
    <button class="btn ghost small" id="watch" style="margin-top:6px">Watch</button></div>`;
  panel.onclick = (e) => {
    const m = e.target.closest('[data-mode]'); if (m) { authMode = m.dataset.mode; auth(); }
    if (e.target.id === 'watch') watchCode($('#rcode').value);
  };
  $('#authf').onsubmit = async (e) => {
    e.preventDefault();
    const em = $('#em')?.value.trim(), pw = $('#pw')?.value, err = $('#aerr');
    err.textContent = ''; panel.classList.add('busy');
    try {
      let r;
      if (authMode === 'signin') r = await api.signIn(em, pw);
      else if (authMode === 'signup') r = await api.signUp(em, pw);
      else if (authMode === 'forgot') { r = await api.resetPassword(em); if (!r.error) { err.style.color = 'var(--good)'; err.textContent = 'Check your email for a reset link.'; return; } }
      else r = await api.sb.auth.updateUser({ password: pw });
      if (r.error) { err.textContent = r.error.message; return; }
      if (authMode === 'signup' && !r.data.session) { err.style.color = 'var(--good)'; err.textContent = 'Check your email to confirm, then sign in.'; return; }
      authMode = 'signin';
      await boot();
    } finally { panel.classList.remove('busy'); }
  };
}

function nameScreen() {
  panel.innerHTML = `<div class="auth"><h2>Choose a display name</h2>
    <p class="sub">Shown on the ladder and to everyone your Kaiju fights. 2–20 characters.</p>
    <form id="nf"><input type="text" id="dn" maxlength="20" minlength="2" required placeholder="Display name">
    <button class="btn big" style="margin-top:12px">Continue</button><div class="err" id="nerr"></div></form>
    <button class="linkbtn" id="so" style="margin-top:10px">Sign out</button></div>`;
  $('#nf').onsubmit = async (e) => {
    e.preventDefault();
    const r = await act('register', { display_name: $('#dn').value }, { rerender: false });
    if (r) go('creator');
  };
  panel.onclick = (e) => { if (e.target.id === 'so') signOut(); };
}

// ---------- Hangar ----------
function hangar() {
  const k = active();
  panel.innerHTML = `
    <h2>Hangar</h2><p class="sub">Your roster. Biomass is shared across all of them. The <b>active</b> Kaiju fights in the Arena and defends you on the ladder while you're away.</p>
    ${limitBanner()}
    ${me.kaiju.map(x => `
      <div class="card click ${x.id === k.id ? 'sel' : ''}" data-pick="${x.id}">
        <div class="row"><b class="grow">${esc(x.name)}${x.id === k.id ? ' <span class="tag ex">Active</span>' : ''}</b><span class="elo">${x.elo}</span></div>
        <div class="rec">${condBadge(injIds(x))} ${FRAME_BY_ID[x.frame].name} · Power ${powerRating(x)} · <span class="w">${x.wins}W</span> <span class="l">${x.losses}L</span> ${x.draws}D · Defense ${x.def_wins}–${x.def_losses}</div>
      </div>`).join('')}
    <div class="row" style="margin-top:12px"><button class="btn ghost grow" data-go="creator" ${me.kaiju.length >= MAX_KAIJU ? 'disabled' : ''}>+ New Kaiju</button><button class="btn grow" data-go="arena">To the Arena</button></div>
    ${me.kaiju.length >= MAX_KAIJU ? `<div class="limit soft"><div class="limit-h">Roster full (${MAX_KAIJU}/${MAX_KAIJU})</div><p>You can have at most ${MAX_KAIJU} Kaiju. Release one (button below) to make room for a new one. Biomass is kept.</p></div>` : ''}
    <h3>Friendly code for ${esc(k.name)}</h3>
    <p class="sub">A friend can paste this in the Arena for an unrated sparring match.</p>
    <textarea readonly id="mycode">${codes.kaijuCode(snap(k, { owner: me.player.name }))}</textarea>
    <div class="row" style="margin-top:6px"><button class="btn ghost small" id="copycode">Copy code</button><span class="grow"></span>
    ${me.kaiju.length > 1 ? `<button class="btn ghost small" id="release">Release ${esc(k.name)}</button>` : ''}</div>
    <h3>Trophies</h3>
    <div class="trophies">${WILD_FRAMES.map(w => { const n = me.trophies[w.id] || 0; return `<div class="trophy ${n ? 'got' : ''}" title="${n ? esc(w.blurb) : 'Undiscovered. Defeat this rare Wild encounter to reveal it.'}"><b>${n ? '×' + n : '?'}</b><span>${n ? w.name : 'Undiscovered'}</span></div>`; }).join('')}</div>
    <p class="rec" style="margin-top:4px">Rare Wild encounters: defenders of humanity who sometimes appear in the Arena. Beat one for double Biomass and a trophy.</p>
    ${me.fallen.length ? '<h3>Hall of the Fallen</h3>' + me.fallen.map((f, i) => `
      <div class="card click fallen" data-fallen="${i}">
        <div class="row"><b class="grow">☠ ${esc(f.name)}</b><span class="rec">${new Date(f.died_at).toLocaleDateString()}</span></div>
        <div class="rec">${FRAME_BY_ID[f.frame].name} · ${f.wins}W ${f.losses}L · Elo ${f.elo} · slain by ${esc(f.death?.killer || 'unknown')} in ${ENV_BY_ID[f.death?.env]?.name || 'battle'}</div>
      </div>`).join('') : ''}
    <p style="margin-top:24px"><button class="linkbtn" id="so">Sign out (${esc(me.player.name)})</button></p>`;
  panel.onclick = async (e) => {
    const p = e.target.closest('[data-pick]');
    if (p && p.dataset.pick !== k.id) { offers = null; await act('set_active', { kaiju_id: p.dataset.pick }); }
    if (e.target.id === 'copycode') navigator.clipboard?.writeText($('#mycode').value).then(() => toast('Code copied'));
    if (e.target.id === 'release' && confirm(`Release ${k.name}? It leaves the ladder for good (Biomass stays).`)) { offers = null; await act('retire', { kaiju_id: k.id }); }
    if (e.target.id === 'so') signOut();
    const wb = e.target.closest('[data-watch]'); if (wb) return watchBattleId(wb.dataset.watch);
    const fl = e.target.closest('[data-fallen]');
    if (fl) {
      const f = me.fallen[+fl.dataset.fallen];
      showcaseOverride = { ...f, injuries: (f.injuries || []).map(i => i.id), fallen: true };
      refreshShowcase();
      for (const c of document.querySelectorAll('[data-fallen]')) c.classList.toggle('sel', c === fl);
      if (f.death?.battle && !fl.querySelector('[data-watch]')) fl.insertAdjacentHTML('beforeend', `<button class="linkbtn" data-watch="${f.death.battle}">Watch the final battle</button>`);
    }
  };
}

// ---------- Creator ----------
function creator() {
  const f = FRAME_BY_ID[draft.frame];
  const maxBase = { hp: 1300, pow: 24, enr: 24, arm: 24, spd: 18, fcs: 18 };
  panel.innerHTML = `
    <h2>Create a Kaiju</h2><p class="sub">Pick a frame. Each one has its own exclusive abilities and a different body.</p>
    ${me?.kaiju?.length >= MAX_KAIJU ? `<div class="limit"><div class="limit-h">Roster full (${MAX_KAIJU}/${MAX_KAIJU})</div><p>You already have the maximum of ${MAX_KAIJU} Kaiju. Release one in the Hangar before creating another.</p></div>` : ''}
    <div class="frames">${FRAMES.map(x => `<div class="card click ${x.id === f.id ? 'sel' : ''}" data-frame="${x.id}"><b>${x.name}</b><small>${x.tags.slice(0, 2).join(' · ')}</small></div>`).join('')}</div>
    <h3>${f.name}</h3>
    <p class="sub" style="margin:0">${f.blurb}</p>
    <div class="tags">${f.tags.map(t => `<span class="tag">${t}</span>`).join('')}</div>
    ${STATS.map(s => `<div class="row" style="margin:3px 0"><span style="width:70px" class="rec">${STAT_NAMES[s]}</span><div class="bar grow"><i style="width:${Math.min(100, (100 * f.base[s]) / maxBase[s])}%"></i></div></div>`).join('')}
    <h3>Exclusive abilities</h3>
    ${f.exclusive.map((id, i) => { const a = ABILITY_BY_ID[id]; return `<div class="card ab"><b>${a.name}</b><span class="tag ex">${i === 0 ? 'Starter' : a.price + ' BIO'}</span><div class="desc">${describe(a)}</div></div>`; }).join('')}
    <h3>Name & colors</h3>
    <input type="text" id="kname" maxlength="24" placeholder="Name your Kaiju" value="${esc(draft.name)}">
    <div class="row" style="margin-top:8px">
      <label class="row rec">Body <input type="color" data-col="body" value="${draft.colors.body}"></label>
      <label class="row rec">Belly <input type="color" data-col="belly" value="${draft.colors.belly}"></label>
      <label class="row rec">Glow <input type="color" data-col="accent" value="${draft.colors.accent}"></label>
    </div>
    <button class="btn big" id="create" style="margin-top:16px">Create ${f.name}</button>`;
  panel.onclick = async (e) => {
    const fr = e.target.closest('[data-frame]');
    if (fr) { draft.frame = fr.dataset.frame; draft.colors = { ...FRAME_BY_ID[draft.frame].colors }; creator(); refreshShowcase(); }
    if (e.target.id === 'create') {
      const name = $('#kname').value.trim() || f.name;
      const r = await act('create_kaiju', { frame: draft.frame, name, colors: draft.colors }, { rerender: false });
      if (r) { draft.name = ''; offers = null; toast(`${name} has awakened`); go('lab'); }
    }
  };
  panel.oninput = (e) => {
    if (e.target.id === 'kname') draft.name = e.target.value;
    if (e.target.dataset.col) { draft.colors[e.target.dataset.col] = e.target.value; refreshShowcase(); }
  };
}

// ---------- Lab ----------
function lab() {
  const k = active(), f = FRAME_BY_ID[k.frame], base = effectiveStats(k), st = applyInjuries(base, injIds(k)), bio = me.player.bio;
  const inj = liveInjuries(k), now = Date.now(), dc = deathChance(injIds(k));
  const healAll = inj.reduce((sum, i) => sum + healCost(i, Date.parse(i.heals_at) - now), 0);
  const pool = abilitiesForFrame(f);
  const groups = [['Exclusive', pool.filter(a => a.frames)], ['Shared', pool.filter(a => a.tags)], ['Universal', pool.filter(a => a.universal)]];
  const basic = ABILITY_BY_ID[f.basic];
  panel.innerHTML = `
    <h2>${esc(k.name)}</h2><p class="sub">${f.name} · Power ${powerRating(k)} · Elo <span class="elo">${k.elo}</span><br>Upgrades at levels 3, 6 and 9 change how it looks.</p>
    <h3>Condition: ${condBadge(injIds(k))}</h3>
    ${dc ? `<div class="warn">⚠ <b>${esc(k.name)} is Critical.</b> If it's knocked out in a fight you start, there's a <b>${dc}%</b> chance it dies for good. Heal it, or let it rest.</div>` : ''}
    ${inj.length ? inj.map(i => { const c = healCost(i, Date.parse(i.heals_at) - now), d = INJURY_BY_ID[i.id]; return `<div class="card ab"><div><b>${d.name}</b> <span class="tag ${d.severity === 2 ? 'ex' : ''}">${d.severity === 2 ? 'Major' : 'Minor'}</span>
      <div class="meta">${injText(i.id)} · heals on its own in ${timeLeft(i.heals_at)}</div></div><button class="btn small" data-heal="${i.id}" ${bio < c ? 'disabled' : ''}>Heal ${c}</button></div>`; }).join('')
      + (inj.length > 1 ? `<button class="btn ghost small" data-heal="all" style="margin-top:6px" ${bio < healAll ? 'disabled' : ''}>Heal all (${healAll} BIO)</button>` : '')
      : '<p class="rec" style="margin:0">No injuries. Losses (and narrow wins) can injure your Kaiju; injuries lower its stats until healed.</p>'}
    <h3>Stats</h3>
    ${STATS.map(s => {
      const lv = k.levels[s] || 0, cost = statUpgradeCost(lv);
      return `<div class="stat" title="${STAT_BLURBS[s]}"><span class="nm">${STAT_NAMES[s]}</span>
        <div class="pips">${Array.from({ length: MAX_STAT_LEVEL }, (_, i) => `<i class="pip ${i < lv ? 'on' : ''} ${[2, 5, 8].includes(i) ? 't' : ''}"></i>`).join('')}</div>
        <span class="val" ${st[s] < base[s] ? `style="color:var(--bad)" title="${base[s]} when healed"` : ''}>${st[s]}</span>
        ${cost === null ? '<span class="rec" style="text-align:center">MAX</span>' : `<button class="btn small" data-up="${s}" ${bio < cost ? 'disabled' : ''}>+ ${cost}</button>`}</div>`;
    }).join('')}
    <h3>Abilities <span style="font-size:14px">(${k.abilities.length}/${MAX_EQUIPPED} equipped)</span></h3>
    <div class="card ab"><b>${basic.name}</b><span class="tag">Basic · always on</span><div class="desc">${describe(basic)}</div></div>
    ${groups.map(([label, list]) => list.length ? `<h3>${label}</h3>` + list.map(a => {
      const owned = k.unlocked.includes(a.id), eq = k.abilities.includes(a.id);
      const cls = a.frames ? 'ex' : a.universal ? 'un' : '';
      const action = !owned
        ? `<button class="btn small" data-buy="${a.id}" ${bio < a.price ? 'disabled' : ''}>Unlock ${a.price}</button>`
        : `<button class="btn small ${eq ? '' : 'ghost'}" data-eq="${a.id}">${eq ? 'Equipped' : 'Equip'}</button>`;
      return `<div class="card ab ${eq ? 'sel' : ''}"><div><b>${a.name}</b> <span class="tag ${cls}">${abilityScope(a)}</span>
        <div class="meta">${a.cost ? a.cost + ' energy' : 'free'}${a.cd ? ' · ' + a.cd + ' turn cooldown' : ''}${a.vis ? ' · changes look' : ''}</div></div>${action}<div class="desc">${describe(a)}</div></div>`;
    }).join('') : '').join('')}
    <h3>Name & colors</h3>
    <input type="text" id="kname" maxlength="24" value="${esc(k.name)}">
    <div class="row" style="margin-top:8px">
      <label class="row rec">Body <input type="color" data-col="body" value="${k.colors.body}"></label>
      <label class="row rec">Belly <input type="color" data-col="belly" value="${k.colors.belly}"></label>
      <label class="row rec">Glow <input type="color" data-col="accent" value="${k.colors.accent}"></label>
      <span class="grow"></span><button class="btn ghost small" id="savecos" hidden>Save</button>
    </div>
    <button class="btn big" data-go="arena" style="margin-top:18px">Fight</button>`;
  panel.onclick = async (e) => {
    const up = e.target.closest('[data-up]'), buy = e.target.closest('[data-buy]'), eq = e.target.closest('[data-eq]');
    if (up) {
      const r = await act('upgrade', { kaiju_id: k.id, stat: up.dataset.up });
      if (r && [3, 6, 9].includes(active().levels[up.dataset.up])) toast(`${k.name} is mutating`);
    }
    if (buy) await act('unlock', { kaiju_id: k.id, ability: buy.dataset.buy });
    const hl = e.target.closest('[data-heal]');
    if (hl) { const r = await act('heal', { kaiju_id: k.id, injury: hl.dataset.heal }); if (r) toast('Healed'); }
    if (eq) {
      const id = eq.dataset.eq;
      let list = k.abilities.includes(id) ? k.abilities.filter(x => x !== id) : [...k.abilities, id];
      if (list.length > MAX_EQUIPPED) return toast(`Max ${MAX_EQUIPPED} equipped: unequip one first`);
      await act('equip', { kaiju_id: k.id, abilities: list });
    }
    if (e.target.id === 'savecos') await act('cosmetic', { kaiju_id: k.id, name: $('#kname').value, colors: k.colors });
  };
  panel.oninput = (e) => {
    if (e.target.dataset.col) { k.colors = { ...k.colors, [e.target.dataset.col]: e.target.value }; refreshShowcase(); }
    if (e.target.dataset.col || e.target.id === 'kname') $('#savecos').hidden = false;
  };
}

// ---------- Arena ----------
async function arena() {
  const k = active();
  panel.innerHTML = `<h2>Arena</h2><p class="sub">Loading opponents…</p>`;
  if (!offers || offers.kaijuId !== k.id) {
    const r = await act('offers', { kaiju_id: k.id }, { rerender: false });
    if (!r) return;
    offers = { kaijuId: k.id, list: r.offers };
  }
  if (screen !== 'arena') return;
  const my = powerRating(k), left = Math.max(0, me.player.ratedPerDay - me.player.fightsToday), dc = deathChance(injIds(k));
  panel.innerHTML = `
    <h2>Arena</h2><p class="sub">${esc(k.name)} · Elo <span class="elo">${k.elo}</span> · Power ${my}<br>
    Condition: ${condBadge(injIds(k))}${injIds(k).length ? ' <button class="linkbtn" data-go="lab">heal</button>' : ''}</p>
    ${dc ? `<div class="warn">⚠ <b>Critical.</b> If ${esc(k.name)} is knocked out, there's a <b>${dc}%</b> chance it dies permanently.</div>` : ''}
    <p class="sub">Fights run on the server and are automatic. Defenders are never injured or killed. ${left > LOW_RATED ? `${left} of ${me.player.ratedPerDay} rated fights left today.` : ''}</p>
    ${limitBanner()}
    ${offers.list.map((o, i) => {
      const e = ENV_BY_ID[o.env], s = o.snapshot, rr = powerRating(s);
      const tag = o.kind === 'player' ? `<span class="tag ex">Player</span>` : o.kind === 'rare' ? `<span class="tag rare">★ Rare</span>` : `<span class="tag">Wild</span>`;
      const pas = o.kind === 'rare' ? PASSIVES[FRAME_BY_ID[s.frame]?.passive] : null;
      const mult = o.kind === 'rare' ? RARE_REWARD_MULT : 1;
      return `<div class="card ${o.kind === 'rare' ? 'rarecard' : ''}">
        ${o.kind === 'rare' ? `<div class="rarebanner">★ Rare encounter: ${FRAME_BY_ID[s.frame].name}</div>` : ''}
        <div class="row"><b class="grow">${esc(s.name)}</b>${tag}<span class="elo" style="margin-left:6px">${o.elo}</span></div>
        <div class="rec">${(s.injuries || []).length ? condBadge(s.injuries) + ' ' : ''}${FRAME_BY_ID[s.frame].name}${o.kind === 'player' ? ' · ' + esc(s.owner) + (o.record ? ` · ${o.record[0]}W ${o.record[1]}L` : '') : ''} · Power ${rr}</div>
        <div class="rec">${s.abilities.map(a => ABILITY_BY_ID[a]?.name).filter(Boolean).join(', ') || 'Basic attack only'}</div>
        ${pas ? `<div class="rec" style="margin-top:4px"><b style="color:var(--gold)">${pas.name}:</b> ${pas.desc}</div>` : ''}
        <div class="rec" style="margin:4px 0 8px">📍 ${e.name}: ${e.blurb}</div>
        <div class="row"><span class="rec grow">${left ? `Win ≈ ${battleReward('win', my, rr) * mult}${mult > 1 ? ' (×' + mult + ')' : ''} · Loss ${battleReward('loss', my, rr)} BIO${rr < my * 0.85 ? ' <span title="Rewards scale with opponent strength: weaker opponents pay less (down to 60%).">(weaker foe: reduced pay)</span>' : ''}` : '<b class="l">Unrated: 0 BIO</b>'}</span><button class="btn small" data-fight="${i}">Fight</button></div>
      </div>`;
    }).join('')}
    <p class="rec" style="margin-top:6px">One opponent is always a Wild Kaiju (others are real players near your Elo when available). Wild fights move your Elo at half rate.</p>
    <button class="btn ghost small" id="reroll" ${me.player.bio < 15 ? 'disabled title="Not enough Biomass (needs 15)"' : ''}>New opponents (15 BIO)</button>
    <h3>Friendly sparring</h3>
    <p class="sub">Paste a friend's Kaiju code (KJ1…) for an unrated match, or a replay code (KR1…) to watch.</p>
    <textarea id="code" placeholder="KJ1.… or KR1.…"></textarea>
    <div class="row" style="margin-top:6px">
      <select id="envsel" class="grow" style="background:#0e1320;color:inherit;border:1px solid var(--line);border-radius:8px;padding:8px">${ENVIRONMENTS.map(e => `<option value="${e.id}">${e.name}</option>`).join('')}</select>
      <button class="btn small" id="usecode">Go</button>
    </div>`;
  panel.onclick = async (e) => {
    const f = e.target.closest('[data-fight]');
    if (f) {
      if (dc && !confirm(`${k.name} is Critical. If it is knocked out there is a ${dc}% chance it dies permanently.\n\nFight anyway?`)) return;
      if (!left && !arena._unratedOk) {
        if (!confirm(`Daily limit reached: this fight is UNRATED.\n\nIt pays no Biomass and no Elo, but ${k.name} can still be injured. Rated fights reset in ${resetIn()} (midnight UTC).\n\nFight anyway?`)) return;
        arena._unratedOk = true;
      }
      return rankedFight(+f.dataset.fight);
    }
    if (e.target.id === 'reroll') {
      panel.classList.add('busy');
      try { const r = await api.call('offers', { kaiju_id: k.id, reroll: true }); me.player.bio = r.bio; offers = { kaijuId: k.id, list: r.offers }; updateTop(); arena(); }
      catch (err) { toast(err.message); } finally { panel.classList.remove('busy'); }
    }
    if (e.target.id === 'usecode') {
      try {
        const { kind, data } = codes.parseCode($('#code').value);
        if (kind === 'kaiju') spar(data, $('#envsel').value); else watchReplay(data);
      } catch (err) { toast(err.message || 'Bad code'); }
    }
  };
}

// The fight response already contains the post-fight state (Biomass, Elo, injuries, deaths).
// Hold it back until the battle finishes playing, so the top bar doesn't spoil the result.
let pendingBoot = null;
function applyPendingBoot() {
  if (!pendingBoot) return;
  applyBoot(pendingBoot); pendingBoot = null;
}

async function rankedFight(slot) {
  panel.classList.add('busy');
  let r = null;
  try { r = await api.call('fight', { slot }); }
  catch (e) { toast(e.message); if (e.status === 401) { me = null; go('auth'); return; } }
  finally { panel.classList.remove('busy'); }
  offers = null;
  if (!r) { if (screen === 'arena') arena(); return; }
  pendingBoot = r;
  // The server sent no events. Re-simulate locally and confirm we got the exact same fight.
  const { ok, stale, record } = verifyReplay(r.battle);
  if (stale) toast(`Server is on engine v${r.battle.v}, this page is v${ENGINE_VERSION}. Reload to update.`);
  showBattle(record, { result: r.result, reward: r.reward, elo: r.eloDelta, rated: r.rated, verified: ok, battleId: r.battle.id, aftermath: r.aftermath, salvage: r.salvage, mySide: 0, rare: r.opponentKind === 'rare' });
}

function spar(opp, env) {
  const k = active();
  const record = simulateBattle({ seed: randomSeed(), env, a: snap(k, { owner: me.player.name }), b: opp });
  const result = record.winner === 0 ? 'win' : record.winner === 1 ? 'loss' : 'draw';
  showBattle(record, { result, friendly: true });
}

function watchReplay(data, aftermath) {
  const { ok, stale, record } = verifyReplay(data);
  if (stale) toast(`Replay is from engine v${data.v}; re-simulated on v${ENGINE_VERSION}, result may differ`);
  else if (!ok) toast('Warning: replay did not match its original hash');
  showBattle(record, { replay: true, verified: ok, aftermath });
}
function watchCode(code) {
  try { const { kind, data } = codes.parseCode(code); if (kind !== 'replay') throw new Error('That is a Kaiju code, use it in the Arena'); watchReplay(data); }
  catch (err) { toast(err.message); }
}
async function watchBattleId(id) {
  try { const r = await api.call('battle', { id }); watchReplay(r.battle, r.aftermath); }
  catch (e) { toast(e.message); }
}

// ---------- Ladder ----------
async function ladderScreen() {
  panel.innerHTML = `<h2>Ladder</h2><p class="sub">Loading…</p>`;
  try { ladder = (await api.call('leaderboard')).ladder; } catch (e) { toast(e.message); return; }
  if (screen !== 'ladder') return;
  const mine = new Set(me.kaiju.map(k => k.id));
  panel.innerHTML = `<h2>Ladder</h2><p class="sub">Top Kaiju by Elo. You gain Elo by attacking in the Arena; your active Kaiju also defends while you're offline and earns ${25} BIO per defense win.</p>
    ${ladder.length ? ladder.map((l, i) => `
      <div class="lad ${mine.has(l.id) ? 'me' : ''}">
        <span class="rk">${i + 1}</span>
        <div><b>${esc(l.snapshot.name)}</b> <span class="rec">· ${esc(l.snapshot.owner || '')}</span>
          <div class="rec">${FRAME_BY_ID[l.snapshot.frame].name} · Power ${l.power} · ${l.record[0]}W ${l.record[1]}L · Def ${l.defense[0]}–${l.defense[1]}</div></div>
        <div style="text-align:right"><div class="elo">${l.elo}</div>${mine.has(l.id) ? '' : `<button class="linkbtn" data-spar="${i}">Spar</button>`}</div>
      </div>`).join('') : '<p class="rec">Nobody on the ladder yet. Create a Kaiju and you are #1.</p>'}`;
  panel.onclick = (e) => {
    const s = e.target.closest('[data-spar]');
    if (s && active()) { const l = ladder[+s.dataset.spar]; spar(l.snapshot, createRng(randomSeed()).pick(ENVIRONMENTS).id); }
  };
}

// ---------- Replays ----------
async function replays() {
  panel.innerHTML = `<h2>Replays</h2><p class="sub">Loading…</p>`;
  try { history = (await api.call('history')).history; } catch (e) { toast(e.message); return; }
  if (screen !== 'replays') return;
  panel.innerHTML = `
    <h2>Replays</h2><p class="sub">Every fight is stored as a seed plus two Kaiju snapshots. Replaying re-runs the engine and checks it matches the server's result.</p>
    ${history.length ? history.map((h, i) => {
      const res = h.winner === null ? 'draw' : h.winner === h.side ? 'win' : 'loss';
      const reward = h.side === 0 ? h.reward : h.defense_reward, delta = h.side === 0 ? h.elo_delta_a : h.elo_delta_d;
      return `<div class="card click" data-rep="${i}">
        <div class="row"><b class="grow">${esc(h.fighters[0].name)} vs ${esc(h.fighters[1].name)}</b>
          <span class="${res === 'win' ? 'w' : res === 'loss' ? 'l' : 'rec'}">${res.toUpperCase()}</span></div>
        <div class="rec">${h.side === 0 ? 'Attack' : 'Defense'} · ${ENV_BY_ID[h.env]?.name || h.env} · ${new Date(h.created_at).toLocaleString()}${reward ? ' · +' + reward + ' BIO' : ''}${delta ? ` · <span class="${delta > 0 ? 'delta-up' : 'delta-dn'}">${delta > 0 ? '+' : ''}${delta} Elo</span>` : ''}${h.rated ? '' : ' · unrated'}${h.aftermath?.[h.side]?.died ? ' · <b class="l">☠ died</b>' : h.aftermath?.[h.side]?.injury ? ' · 🩸 ' + INJURY_BY_ID[h.aftermath[h.side].injury].name : ''}</div>
      </div>`;
    }).join('') : '<p class="rec">No fights yet.</p>'}
    <h3>Watch a replay code</h3>
    <textarea id="rcode" placeholder="KR1.…"></textarea>
    <button class="btn small" id="watch" style="margin-top:6px">Watch</button>`;
  panel.onclick = (e) => {
    const r = e.target.closest('[data-rep]');
    if (r) watchBattleId(history[+r.dataset.rep].id);
    if (e.target.id === 'watch') watchCode($('#rcode').value);
  };
}

// ---------- Defense report ----------
function defenseReport() {
  const d = me?.defenses || [];
  if (!d.length) return;
  me.defenses = [];
  const total = d.reduce((s, x) => s + (x.defense_reward || 0), 0);
  modal.innerHTML = `<div class="dialog" style="text-align:left">
    <h2>While you were away</h2><p class="sub">${d.length} attack${d.length > 1 ? 's' : ''} on your Kaiju${total ? ` · <b style="color:var(--gold)">+${total} BIO</b> from defense wins` : ''}</p>
    ${d.map(x => {
      const held = x.winner === 1;
      return `<div class="card click" data-def="${x.id}"><div class="row"><b class="grow">${esc(x.fighters[0].name)} <span class="rec">(${esc(x.fighters[0].owner || '')})</span> attacked ${esc(x.fighters[1].name)}</b>
        <span class="${held ? 'w' : x.winner === 0 ? 'l' : 'rec'}">${held ? 'HELD' : x.winner === 0 ? 'FELL' : 'DRAW'}</span></div>
        <div class="rec">${x.aftermath?.[1]?.injury ? '🩸 ' + INJURY_BY_ID[x.aftermath[1].injury].name + ' · ' : ''}${ENV_BY_ID[x.env]?.name || x.env}${x.elo_delta_d ? ` · ${x.elo_delta_d > 0 ? '+' : ''}${x.elo_delta_d} Elo` : ''} · tap to watch</div></div>`;
    }).join('')}
    <button class="btn big" id="dclose" style="margin-top:12px">OK</button></div>`;
  modal.hidden = false;
  modal.onclick = (e) => {
    const c = e.target.closest('[data-def]');
    if (c) { modal.hidden = true; watchBattleId(c.dataset.def); }
    if (e.target.id === 'dclose') modal.hidden = true;
  };
}

// ---------- Battle playback ----------
function showBattle(record, { result, reward, elo, rated, replay, friendly, verified, battleId, aftermath, salvage, mySide = null, rare = false } = {}) {
  if (playing) { playing.stop(); playing = null; }
  panel.hidden = true; hud.hidden = false; modal.hidden = true;
  const env = ENV_BY_ID[record.env];
  const hp = record.meta.map(m => m.maxHp), en = [20, 20];
  const sts = [new Map(), new Map()];
  hud.innerHTML = `
    ${record.fighters.map((f, i) => `
      <div class="fighter f${i}">
        <div class="nm">${esc(f.name)}</div>
        <div class="fr">${FRAME_BY_ID[f.frame].name}${f.owner ? ' · ' + esc(f.owner) : ''}</div>
        <div class="hp"><u></u><i></i><span></span></div>
        <div class="en"><i></i></div>
        <div class="sts"></div>
        <div class="sts injs">${(f.injuries || []).map(id => `<span class="st inj" title="${injText(id)}">${INJURY_BY_ID[id]?.name || id}</span>`).join('')}</div>
      </div>`).join('')}
    <div class="envtag">${env.name}<br><small style="font-family:Inter;font-size:11px">${env.blurb}</small></div>
    <div class="callout" style="opacity:0"></div>
    <div id="log"></div>
    <div class="ctrls">
      <button class="btn ghost small" data-speed="1">1×</button>
      <button class="btn ghost small" data-speed="2">2×</button>
      <button class="btn ghost small" data-speed="4">4×</button>
      <button class="btn small" id="skip">Skip ▶▶</button>
    </div>`;
  const fx = [0, 1].map(i => $(`.f${i}`, hud));
  const name = (i) => esc(record.fighters[i].name);
  const draw = (i) => {
    const p = Math.max(0, (100 * hp[i]) / record.meta[i].maxHp);
    $('.hp > i', fx[i]).style.width = p + '%'; $('.hp > u', fx[i]).style.width = p + '%';
    $('.hp > span', fx[i]).textContent = `${Math.max(0, hp[i])} / ${record.meta[i].maxHp}`;
    $('.en > i', fx[i]).style.width = en[i] + '%';
    $('.sts', fx[i]).innerHTML = [...sts[i].keys()].map(s => `<span class="st ${NEG.has(s) ? 'neg' : 'pos'}">${STATUSES[s].name}</span>`).join('');
  };
  draw(0); draw(1);
  const log = (html) => { const d = document.createElement('div'); d.innerHTML = html; $('#log').prepend(d); };
  const callout = (txt, color) => { const c = $('.callout', hud); c.textContent = txt; c.style.color = color || '#fff'; c.style.opacity = 1; clearTimeout(callout._t); callout._t = setTimeout(() => (c.style.opacity = 0), 900); };

  const onEvent = (ev) => {
    switch (ev.t) {
      case 'turn': en[ev.a] = ev.en; draw(ev.a); break;
      case 'use': en[ev.a] = Math.max(0, en[ev.a] - (ABILITY_BY_ID[ev.ab]?.cost || 0)); draw(ev.a); log(`<b>${name(ev.a)}</b> uses <b>${esc(ev.name)}</b>`); if (!ABILITY_BY_ID[ev.ab]?.basic) callout(ev.name); break;
      case 'hit': hp[ev.tg] = ev.hp; draw(ev.tg); log(`&nbsp;&nbsp;${ev.crit ? '<b style="color:var(--gold)">CRIT</b> ' : ''}${ev.dmg} ${ev.type}${ev.absorbed ? ` (${ev.absorbed} shielded)` : ''}`); break;
      case 'miss': log(`&nbsp;&nbsp;<span class="rec">miss</span>`); break;
      case 'status': sts[ev.tg].set(ev.st, ev.turns); draw(ev.tg); log(`&nbsp;&nbsp;${name(ev.tg)} is <b>${STATUSES[ev.st].name}</b>`); break;
      case 'expire': sts[ev.tg].delete(ev.st); draw(ev.tg); break;
      case 'dot': hp[ev.tg] = ev.hp; draw(ev.tg); log(`${name(ev.tg)} takes ${ev.dmg} from ${STATUSES[ev.st].name.toLowerCase()}`); break;
      case 'heal': hp[ev.tg] = ev.hp; draw(ev.tg); log(`${name(ev.tg)} heals <span class="w">${ev.amt}</span>`); break;
      case 'selfdmg': hp[ev.a] = ev.hp; draw(ev.a); break;
      case 'skip': log(`${name(ev.a)} is ${ev.reason === 'stun' ? 'stunned' : 'held fast'} and can't act`); break;
      case 'hazard': callout(ev.name, '#ffb347'); for (const h of ev.hits) { hp[h.tg] = h.hp; draw(h.tg); log(`<b style="color:#ffb347">${esc(ev.name)}</b> hits ${name(h.tg)} for ${h.dmg}`); } break;
      case 'passive': callout(ev.name, '#ffd23f'); log(`<b style="color:#ffd23f">${name(ev.tg)}: ${esc(ev.name)}</b>${ev.resisted ? ` (shrugs off ${STATUSES[ev.resisted].name.toLowerCase()})` : ''}`); break;
      case 'ko': callout('K.O.!', '#ff3a2a'); log(`<b>${name(ev.tg)} is down!</b>`); break;
    }
  };
  const onDone = () => {
    setTimeout(() => {
      applyPendingBoot();   // reveal Biomass/Elo only now, together with the result
      const w = record.winner;
      const big = replay ? (w === null ? 'DRAW' : `${record.fighters[w].name} WINS`) : ({ win: 'VICTORY', loss: 'DEFEAT', draw: 'DRAW' })[result];
      const sub = friendly ? 'Friendly · unrated' : replay ? 'Replay' : rated === false ? 'Unrated (daily limit reached)' : 'Ranked';
      modal.innerHTML = `<div class="dialog">
        <div class="big" style="color:${result === 'loss' ? 'var(--bad)' : result === 'win' ? 'var(--good)' : '#fff'}">${esc(big)}</div>
        <div class="rec">${sub} · ${record.reason === 'ko' ? 'Knockout' : record.reason === 'decision' ? 'Time: decision on remaining HP' : 'Double knockout'} · ${record.turns} actions</div>
        ${verified !== undefined ? `<div class="${verified ? 'verified' : 'l'}" style="font-size:11px;margin-top:4px">${verified ? '✓ Re-simulated locally: identical to the server result' : '⚠ Local re-simulation did not match'}</div>` : ''}
        ${rare && result === 'win' ? `<div class="rarebanner" style="margin-top:10px">★ Trophy: ${FRAME_BY_ID[record.fighters[1].frame].name} defeated (×${RARE_REWARD_MULT} Biomass)</div>` : ''}
        ${!friendly && !replay && rated === false ? `<div class="limit" style="text-align:left;margin-top:10px"><div class="limit-h">No Biomass: daily limit reached</div><p>You've used all ${me?.player?.ratedPerDay ?? 40} rated fights for today, so this fight paid <b>0 Biomass and 0 Elo</b>. Rated fights reset in <b>${resetIn()}</b> (midnight UTC).</p></div>` : ''}
        ${!friendly && !replay && rated && me?.player && ratedLeft() === 0 ? `<div class="limit soft" style="text-align:left;margin-top:10px"><div class="limit-h">That was your last rated fight today</div><p>Fights after this pay no Biomass or Elo until the reset in <b>${resetIn()}</b> (midnight UTC).</p></div>` : ''}
        ${!friendly && !replay && rated && me?.player && ratedLeft() > 0 && ratedLeft() <= LOW_RATED ? `<div class="rec" style="margin-top:8px;color:#fcd34d">⚠ ${ratedLeft()} rated fight${ratedLeft() === 1 ? '' : 's'} left today.</div>` : ''}
        ${aftermathHtml(aftermath, record.fighters, { mySide })}
        ${salvage ? `<div class="rec">Salvaged <b style="color:var(--gold)">${salvage} BIO</b> from the remains.</div>` : ''}
        ${reward || elo ? `<div class="reward">${reward ? `+${reward} Biomass` : ''}${elo ? ` <span class="${elo > 0 ? 'delta-up' : 'delta-dn'}" style="font-size:22px">${elo > 0 ? '+' : ''}${elo} Elo</span>` : ''}</div>` : '<div style="height:18px"></div>'}
        <div class="row"><button class="btn ghost grow" id="rewatch">Replay</button><button class="btn ghost grow" id="share">Copy replay code</button></div>
        <div class="row" style="margin-top:8px">${replay || !me?.player ? `<button class="btn grow" data-go="${me?.player ? 'replays' : 'auth'}">Back</button>` : '<button class="btn ghost grow" data-go="lab">Upgrade</button><button class="btn grow" data-go="arena">Next fight</button>'}</div>
      </div>`;
      modal.hidden = false;
      modal.onclick = (e) => {
        if (e.target.closest('[data-go]')) modal.hidden = true;
        if (e.target.id === 'rewatch') showBattle(verifyReplay(record).record, { result, replay: replay || !friendly, friendly, verified, aftermath, mySide });
        if (e.target.id === 'share') navigator.clipboard?.writeText(codes.replayCode(record)).then(() => toast('Replay code copied'));
      };
    }, 900);
  };

  if (stage && playBattle) {
    try {
      setEnv(record.env);
      stage.setFighters(record.fighters[0], record.fighters[1]);
      playing = playBattle(stage, record, { speed: 1, onEvent, onDone });
    } catch (e) { console.error(e); playing = textPlayback(record, onEvent, onDone); }
  } else playing = textPlayback(record, onEvent, onDone);

  hud.onclick = (e) => {
    const sp = e.target.closest('[data-speed]');
    if (sp) playing?.setSpeed(+sp.dataset.speed);
    if (e.target.id === 'skip') playing?.skipToEnd();
  };
}

function textPlayback(record, onEvent, onDone) {
  let i = 0, speed = 1, timer = null, stopped = false;
  const step = () => {
    if (stopped) return;
    if (i >= record.events.length) return onDone();
    const ev = record.events[i]; onEvent(ev, i++);
    timer = setTimeout(step, (ev.t === 'turn' ? 500 : ev.t === 'use' ? 350 : 120) / speed);
  };
  step();
  return {
    pause() { clearTimeout(timer); }, resume() { step(); }, setSpeed(x) { speed = x; },
    skipToEnd() { clearTimeout(timer); while (i < record.events.length) onEvent(record.events[i], i++); onDone(); },
    stop() { stopped = true; clearTimeout(timer); },
  };
}

// ---------- Boot ----------
async function boot() {
  const s = await api.session();
  if (!s) { me = null; go('auth'); return; }
  panel.innerHTML = '<p class="sub">Loading your Kaiju…</p>';
  try { applyBoot(await api.call('me')); }
  catch (e) { toast(e.message); if (e.status === 401) { await api.signOut(); me = null; } go('auth'); return; }
  if (authMode === 'recover') { me = null; go('auth'); return; }   // reset link: set a new password before anything else
  go(me.needsName ? 'name' : active() ? 'hangar' : 'creator');
  defenseReport();
}

api.sb.auth.onAuthStateChange((evt) => { if (evt === 'PASSWORD_RECOVERY') { authMode = 'recover'; me = null; go('auth'); } });
initStage();
if (api.isRecoveryLink) authMode = 'recover';
if (api.authLinkError) setTimeout(() => toast(`Reset link problem: ${api.authLinkError}. Request a new one.`), 300);
const sharedBattle = new URLSearchParams(location.search).get('battle');
boot().then(() => { if (sharedBattle) watchBattleId(sharedBattle); });
