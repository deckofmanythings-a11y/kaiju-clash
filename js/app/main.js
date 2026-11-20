import { FRAMES, FRAME_BY_ID, STATS, STAT_NAMES, STAT_BLURBS, MAX_STAT_LEVEL, effectiveStats } from '../data/frames.js';
import { ABILITY_BY_ID, STATUSES, MAX_EQUIPPED, abilitiesForFrame, abilityScope } from '../data/abilities.js';
import { ENV_BY_ID } from '../data/environments.js';
import { statUpgradeCost, powerRating, battleReward, generateRival, newKaiju } from '../data/economy.js';
import { simulateBattle, verifyReplay, ENGINE_VERSION } from '../engine/battle.js';
import { createRng, hashString, randomSeed } from '../engine/rng.js';
import * as store from './store.js';

const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const panel = $('#panel'), hud = $('#hud'), modal = $('#modal');

store.load();
let screen = 'hangar';
let draft = { frame: 'saurian', name: '', colors: { ...FRAME_BY_ID.saurian.colors } };
let stage = null, playBattle = null, playing = null;

// ---------- 3D stage (optional: app still works text-only if the renderer fails to load) ----------
async function initStage() {
  try {
    const [st, pb] = await Promise.all([import('../render/stage.js'), import('../render/playback.js')]);
    stage = st.createStage($('#stage'));
    playBattle = pb.playBattle;
    window.addEventListener('resize', () => stage.resize());
    setEnv('city');
    refreshShowcase();
  } catch (e) {
    console.warn('3D renderer unavailable, running text-only', e);
  }
}

let currentEnv = null;
function setEnv(id) { if (stage && currentEnv !== id) { stage.setEnvironment(id); currentEnv = id; } }

function refreshShowcase() {
  if (!stage) return;
  try {
    if (screen === 'creator') stage.showcase({ name: draft.name, frame: draft.frame, levels: {}, abilities: [FRAME_BY_ID[draft.frame].exclusive[0]], colors: draft.colors });
    else { const k = store.active(); if (k) stage.showcase(k); }
  } catch (e) { console.warn(e); }
}

function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 2600);
}
function updateBio() { $('#bio').textContent = store.S().bio.toLocaleString(); }

function describe(ab) {
  const bits = [];
  if (ab.kind === 'attack') {
    bits.push(`${ab.type === 'random' ? ab.randomTypes.join('/') : ab.type} · power ${ab.power}${ab.hits ? ' ×' + ab.hits : ''} · scales ${STAT_NAMES[ab.stat || 'pow']}`);
  }
  if (ab.heal) bits.push(`heals ${ab.heal}% HP${ab.cleanse ? ', cleanses debuffs' : ''}`);
  for (const e of ab.effects || []) bits.push(`${e.chance < 100 ? e.chance + '% ' : ''}${STATUSES[e.status].name} ${e.turns}t`);
  for (const e of ab.selfEffects || []) bits.push(`self: ${STATUSES[e.status].name} ${e.turns}t`);
  if (ab.energyGain) bits.push(`+${ab.energyGain} energy`);
  if (ab.selfDamage) bits.push(`costs ${ab.selfDamage}% HP`);
  if (ab.execute) bits.push(`double damage below ${ab.execute}% HP`);
  if (ab.critBonus) bits.push(`+${ab.critBonus}% crit`);
  return bits.join(' · ');
}

// ---------- Router ----------
function go(s) {
  if (playing) { playing.stop(); playing = null; }
  const k = store.active();
  if (!k && s !== 'creator' && s !== 'replays') s = 'creator';
  screen = s;
  hud.hidden = true; panel.hidden = false;
  for (const b of document.querySelectorAll('#nav button')) b.classList.toggle('on', b.dataset.go === s);
  updateBio();
  ({ hangar, creator, lab, arena, replays })[s]();
  if (stage) { setEnv('city'); refreshShowcase(); }
}
document.addEventListener('click', (e) => {
  const g = e.target.closest('[data-go]'); if (g) go(g.dataset.go);
});

// ---------- Hangar ----------
function hangar() {
  const S = store.S(), k = store.active();
  panel.innerHTML = `
    <h2>Hangar</h2><p class="sub">Your Kaiju roster. Biomass is shared across all of them.</p>
    ${S.kaiju.map(x => `
      <div class="card click ${x.id === k.id ? 'sel' : ''}" data-pick="${x.id}">
        <div class="row"><b class="grow">${esc(x.name)}</b><span class="rec"><span class="w">${x.record.w}W</span> · <span class="l">${x.record.l}L</span> · ${x.record.d}D</span></div>
        <div class="rec">${FRAME_BY_ID[x.frame].name} · Power ${powerRating(x)}</div>
      </div>`).join('')}
    <div class="row" style="margin-top:12px"><button class="btn ghost grow" data-go="creator">+ New Kaiju</button><button class="btn grow" data-go="arena">To the Arena</button></div>
    <h3>Share ${esc(k.name)}</h3>
    <p class="sub">Send this code to a friend. They can paste it in the Arena to fight your Kaiju.</p>
    <textarea readonly id="mycode">${store.kaijuCode(k)}</textarea>
    <button class="btn ghost small" id="copycode" style="margin-top:6px">Copy code</button>
    ${S.kaiju.length > 1 ? `<button class="btn ghost small" id="release" style="margin-top:6px;float:right">Release ${esc(k.name)}</button>` : ''}`;
  panel.onclick = (e) => {
    const p = e.target.closest('[data-pick]');
    if (p) { S.activeId = p.dataset.pick; store.save(); hangar(); refreshShowcase(); }
    if (e.target.id === 'copycode') navigator.clipboard?.writeText($('#mycode').value).then(() => toast('Code copied'));
    if (e.target.id === 'release' && confirm(`Release ${k.name}? It is gone forever (Biomass stays).`)) {
      S.kaiju = S.kaiju.filter(x => x.id !== k.id); S.activeId = S.kaiju[0].id; store.save(); hangar(); refreshShowcase();
    }
  };
}

// ---------- Creator ----------
function creator() {
  const f = FRAME_BY_ID[draft.frame];
  const maxBase = { hp: 1300, pow: 24, enr: 24, arm: 24, spd: 18, fcs: 18 };
  panel.innerHTML = `
    <h2>Create a Kaiju</h2><p class="sub">Pick a frame. Each one has its own exclusive abilities and a different body.</p>
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
  panel.onclick = (e) => {
    const fr = e.target.closest('[data-frame]');
    if (fr) { draft.frame = fr.dataset.frame; draft.colors = { ...FRAME_BY_ID[draft.frame].colors }; creator(); refreshShowcase(); }
    if (e.target.id === 'create') {
      const name = $('#kname').value.trim() || `${f.name} ${store.S().kaiju.length + 1}`;
      const k = newKaiju(draft.frame, name);
      k.colors = { ...draft.colors };
      store.S().kaiju.push(k); store.S().activeId = k.id; store.save();
      draft.name = '';
      toast(`${name} has awakened`);
      go('lab');
    }
  };
  panel.oninput = (e) => {
    if (e.target.id === 'kname') draft.name = e.target.value;
    if (e.target.dataset.col) { draft.colors[e.target.dataset.col] = e.target.value; refreshShowcase(); }
  };
}

// ---------- Lab (upgrades) ----------
function lab() {
  const S = store.S(), k = store.active(), f = FRAME_BY_ID[k.frame], st = effectiveStats(k);
  const pool = abilitiesForFrame(f);
  const groups = [
    ['Exclusive', pool.filter(a => a.frames)],
    ['Shared', pool.filter(a => a.tags)],
    ['Universal', pool.filter(a => a.universal)],
  ];
  const basic = ABILITY_BY_ID[f.basic];
  panel.innerHTML = `
    <h2>${esc(k.name)}</h2><p class="sub">${f.name} · Power ${powerRating(k)} · Upgrades at levels 3, 6 and 9 change how it looks.</p>
    <h3>Stats</h3>
    ${STATS.map(s => {
      const lv = k.levels[s], cost = statUpgradeCost(lv);
      return `<div class="stat" title="${STAT_BLURBS[s]}"><span class="nm">${STAT_NAMES[s]}</span>
        <div class="pips">${Array.from({ length: MAX_STAT_LEVEL }, (_, i) => `<i class="pip ${i < lv ? 'on' : ''} ${[2, 5, 8].includes(i) ? 't' : ''}"></i>`).join('')}</div>
        <span class="val">${st[s]}</span>
        ${cost === null ? '<span class="rec" style="text-align:center">MAX</span>' : `<button class="btn small" data-up="${s}" ${S.bio < cost ? 'disabled' : ''}>+ ${cost}</button>`}</div>`;
    }).join('')}
    <h3>Abilities <span style="font-size:14px">(${k.abilities.length}/${MAX_EQUIPPED} equipped)</span></h3>
    <div class="card ab"><b>${basic.name}</b><span class="tag">Basic · always on</span><div class="desc">${describe(basic)}</div></div>
    ${groups.map(([label, list]) => list.length ? `<h3>${label}</h3>` + list.map(a => {
      const owned = k.unlocked.includes(a.id), eq = k.abilities.includes(a.id);
      const cls = a.frames ? 'ex' : a.universal ? 'un' : '';
      const action = !owned
        ? `<button class="btn small" data-buy="${a.id}" ${S.bio < a.price ? 'disabled' : ''}>Unlock ${a.price}</button>`
        : `<button class="btn small ${eq ? '' : 'ghost'}" data-eq="${a.id}">${eq ? 'Equipped' : 'Equip'}</button>`;
      return `<div class="card ab ${eq ? 'sel' : ''}"><div><b>${a.name}</b> <span class="tag ${cls}">${abilityScope(a)}</span>
        <div class="meta">${a.cost ? a.cost + ' energy' : 'free'}${a.cd ? ' · ' + a.cd + ' turn cooldown' : ''}${a.vis ? ' · changes look' : ''}</div></div>${action}<div class="desc">${describe(a)}</div></div>`;
    }).join('') : '').join('')}
    <h3>Colors</h3>
    <div class="row">
      <label class="row rec">Body <input type="color" data-col="body" value="${k.colors.body}"></label>
      <label class="row rec">Belly <input type="color" data-col="belly" value="${k.colors.belly}"></label>
      <label class="row rec">Glow <input type="color" data-col="accent" value="${k.colors.accent}"></label>
    </div>
    <button class="btn big" data-go="arena" style="margin-top:18px">Fight</button>`;
  panel.onclick = (e) => {
    const up = e.target.closest('[data-up]'), buy = e.target.closest('[data-buy]'), eq = e.target.closest('[data-eq]');
    if (up) {
      const s = up.dataset.up, cost = statUpgradeCost(k.levels[s]);
      if (cost !== null && S.bio >= cost) { S.bio -= cost; k.levels[s]++; store.save(); if ([3, 6, 9].includes(k.levels[s])) toast(`${k.name} is mutating`); }
    }
    if (buy) {
      const a = ABILITY_BY_ID[buy.dataset.buy];
      if (S.bio >= a.price && !k.unlocked.includes(a.id)) {
        S.bio -= a.price; k.unlocked.push(a.id);
        if (k.abilities.length < MAX_EQUIPPED) k.abilities.push(a.id);
        store.save();
      }
    }
    if (eq) {
      const id = eq.dataset.eq;
      if (k.abilities.includes(id)) k.abilities = k.abilities.filter(x => x !== id);
      else if (k.abilities.length < MAX_EQUIPPED) k.abilities.push(id);
      else return toast(`Max ${MAX_EQUIPPED} equipped: unequip one first`);
      store.save();
    }
    if (up || buy || eq) { const y = panel.scrollTop; updateBio(); lab(); panel.scrollTop = y; refreshShowcase(); }
  };
  panel.oninput = (e) => {
    if (e.target.dataset.col) { k.colors[e.target.dataset.col] = e.target.value; store.save(); refreshShowcase(); }
  };
}

// ---------- Arena ----------
function rivals() {
  const S = store.S(), k = store.active(), my = powerRating(k);
  return [85, 100, 120].map((p, i) => {
    const seed = hashString(`${S.rivalSeed}:${k.id}:${i}`);
    const rv = generateRival(seed, Math.floor((my * p) / 100));
    const env = createRng(seed ^ 0x5bd1e995).pick(Object.keys(ENV_BY_ID));
    return { rv, env, label: ['Warm-up', 'Even match', 'Dangerous'][i] };
  });
}
function arena() {
  const k = store.active(), my = powerRating(k);
  const list = rivals();
  panel.innerHTML = `
    <h2>Arena</h2><p class="sub">${esc(k.name)} · Power ${my}. Fights are automatic. Win for more Biomass; tougher rivals pay more.</p>
    ${list.map((r, i) => {
      const e = ENV_BY_ID[r.env], rr = powerRating(r.rv);
      return `<div class="card">
        <div class="row"><b class="grow">${esc(r.rv.name)}</b><span class="tag ${i === 2 ? 'ex' : i === 1 ? 'un' : ''}">${r.label}</span></div>
        <div class="rec">${FRAME_BY_ID[r.rv.frame].name} · Power ${rr} · ${r.rv.abilities.map(a => ABILITY_BY_ID[a].name).join(', ')}</div>
        <div class="rec" style="margin:4px 0 8px">📍 ${e.name}: ${e.blurb}</div>
        <div class="row"><span class="rec grow">Win ≈ ${battleReward('win', my, rr)} · Loss ${battleReward('loss', my, rr)} BIO</span><button class="btn small" data-fight="${i}">Fight</button></div>
      </div>`;
    }).join('')}
    <button class="btn ghost small" id="reroll" style="margin-top:8px">New rivals (15 BIO)</button>
    <h3>Challenge a friend</h3>
    <p class="sub">Paste a Kaiju code (KJ1…) to fight it, or a replay code (KR1…) to watch a fight.</p>
    <textarea id="code" placeholder="KJ1.… or KR1.…"></textarea>
    <div class="row" style="margin-top:6px">
      <select id="envsel" class="grow" style="background:#0e1320;color:inherit;border:1px solid var(--line);border-radius:8px;padding:8px">${Object.values(ENV_BY_ID).map(e => `<option value="${e.id}">${e.name}</option>`).join('')}</select>
      <button class="btn small" id="usecode">Go</button>
    </div>`;
  panel.onclick = (e) => {
    const f = e.target.closest('[data-fight]');
    if (f) { const r = list[+f.dataset.fight]; startFight(r.rv, r.env, { rated: true }); }
    if (e.target.id === 'reroll') {
      const S = store.S(); if (S.bio < 15) return toast('Not enough Biomass');
      S.bio -= 15; S.rivalSeed = randomSeed(); store.save(); updateBio(); arena();
    }
    if (e.target.id === 'usecode') {
      try {
        const { kind, data } = store.parseCode($('#code').value);
        if (kind === 'kaiju') startFight(data, $('#envsel').value, { rated: true, friend: true });
        else watchReplay(data);
      } catch (err) { toast(err.message || 'Bad code'); }
    }
  };
}

function startFight(opp, env, { rated, friend } = {}) {
  const S = store.S(), k = store.active();
  const record = simulateBattle({ seed: randomSeed(), env, a: { ...k, owner: 'You' }, b: opp });
  // Rewards are settled now (before playback) so refreshing mid-fight can't reroll a loss.
  let reward = 0, result = record.winner === 0 ? 'win' : record.winner === 1 ? 'loss' : 'draw';
  if (rated) {
    reward = battleReward(result, powerRating(k), powerRating(record.fighters[1]));
    S.bio += reward;
    k.record[result[0]]++;
    if (!friend) S.rivalSeed = randomSeed();
  }
  store.addHistory(record, { kaijuId: k.id, result, reward, friend: !!friend });
  store.save();
  showBattle(record, { result, reward });
}

function watchReplay(data) {
  const { ok, stale, record } = verifyReplay({ ...data, hash: data.hash });
  if (stale) toast(`Replay is from engine v${data.v}; re-simulated on v${ENGINE_VERSION}, result may differ`);
  else if (!ok) toast('Warning: replay did not match its original hash');
  showBattle(record, { replay: true });
}

// ---------- Battle playback ----------
function showBattle(record, { result, reward, replay } = {}) {
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
    $('.sts', fx[i]).innerHTML = [...sts[i].keys()].map(s => `<span class="st ${['burn', 'poison', 'bleed', 'stun', 'bind', 'slow', 'blind', 'weaken', 'sunder'].includes(s) ? 'neg' : 'pos'}">${STATUSES[s].name}</span>`).join('');
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
      case 'ko': callout('K.O.!', '#ff3a2a'); log(`<b>${name(ev.tg)} is down!</b>`); break;
      case 'end': break;
    }
  };
  const onDone = () => {
    setTimeout(() => {
      const w = record.winner;
      const big = replay ? (w === null ? 'DRAW' : `${record.fighters[w].name} WINS`) : ({ win: 'VICTORY', loss: 'DEFEAT', draw: 'DRAW' })[result];
      modal.innerHTML = `<div class="dialog">
        <div class="big" style="color:${result === 'loss' ? 'var(--bad)' : result === 'win' ? 'var(--good)' : '#fff'}">${esc(big)}</div>
        <div class="rec">${record.reason === 'ko' ? 'Knockout' : record.reason === 'decision' ? 'Time: decision on remaining HP' : 'Double knockout'} · ${record.turns} actions · seed ${record.seed}</div>
        ${reward ? `<div class="reward">+${reward} Biomass</div>` : '<div style="height:18px"></div>'}
        <div class="row"><button class="btn ghost grow" id="rewatch">Replay</button><button class="btn ghost grow" id="share">Copy replay code</button></div>
        <div class="row" style="margin-top:8px">${replay ? '<button class="btn grow" data-go="replays">Back</button>' : '<button class="btn ghost grow" data-go="lab">Upgrade</button><button class="btn grow" data-go="arena">Next fight</button>'}</div>
      </div>`;
      modal.hidden = false;
      modal.onclick = (e) => {
        if (e.target.closest('[data-go]')) modal.hidden = true;
        if (e.target.id === 'rewatch') showBattle(verifyReplay(record).record, { result, reward: 0, replay });
        if (e.target.id === 'share') navigator.clipboard?.writeText(store.replayCode(record)).then(() => toast('Replay code copied'));
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

// Fallback when WebGL/renderer is unavailable: step through events on a timer.
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

// ---------- Replays ----------
function replays() {
  const H = store.S().history;
  panel.innerHTML = `
    <h2>Replays</h2><p class="sub">Every fight is stored as a seed and two Kaiju snapshots. Re-running it gives the exact same battle.</p>
    ${H.length ? H.map((h, i) => `
      <div class="card click" data-rep="${i}">
        <div class="row"><b class="grow">${esc(h.fighters[0].name)} vs ${esc(h.fighters[1].name)}</b>
          <span class="${h.result === 'win' ? 'w' : h.result === 'loss' ? 'l' : 'rec'}">${(h.result || '').toUpperCase()}</span></div>
        <div class="rec">${ENV_BY_ID[h.env]?.name || h.env} · ${new Date(h.at).toLocaleString()}${h.reward ? ' · +' + h.reward + ' BIO' : ''}</div>
      </div>`).join('') : '<p class="rec">No fights yet.</p>'}
    <h3>Watch a replay code</h3>
    <textarea id="rcode" placeholder="KR1.…"></textarea>
    <button class="btn small" id="watch" style="margin-top:6px">Watch</button>`;
  panel.onclick = (e) => {
    const r = e.target.closest('[data-rep]');
    if (r) watchReplay(H[+r.dataset.rep]);
    if (e.target.id === 'watch') {
      try { const { kind, data } = store.parseCode($('#rcode').value); if (kind !== 'replay') throw new Error('That is a Kaiju code, use it in the Arena'); watchReplay(data); }
      catch (err) { toast(err.message); }
    }
  };
}

go(store.active() ? 'hangar' : 'creator');
initStage();
