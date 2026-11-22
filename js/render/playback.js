// Plays a BattleRecord as 3D animation on a stage.
//
// playBattle(stage, record, { speed = 1, onEvent(ev, index), onDone() })
//   -> { pause(), resume(), setSpeed(x), skipToEnd(), stop() }
//
// Events are animated strictly in order on a playback clock (scaled by speed, frozen while paused).
// onEvent(ev, i) fires once per event, at the moment its visual lands (impact, number pop, etc).
//
// Rare Wild plans (js/render/wild-models.js) expose optional hooks on rig.extra.wild (reset, setTarget, onHit,
// onStrike, origin, strafe, kneel/flare, beginSwap/swapK, reinforce/reinK, setTransformed, beamSources,
// twinBeam). Every hook is optional, so ordinary Kaiju play exactly as before. `passive` events
// ({ t:'passive', tg, id, name }) get a visual per id; unknown ids fall back to a generic aura pulse.

import * as THREE from 'three';
import { ABILITY_BY_ID, STATUSES } from '../data/abilities.js?v=5';
import { softTexture } from './arena.js';

const V3 = THREE.Vector3;
const PI = Math.PI;
const FX_KINDS = new Set(['melee', 'beam', 'projectile', 'aoe', 'grab', 'roar', 'heal', 'shield', 'buff', 'multi']);
const TYPE_COLORS = { physical: '#fff2d8', fire: '#ff7a2a', electric: '#bfe4ff', acid: '#9aff3a', cold: '#8fe8ff', sonic: '#e0b0ff' };
const STATUS_COLORS = {
  burn: '#ff7a2a', poison: '#6aff4a', bleed: '#ff3a3a', stun: '#ffe84a', bind: '#e8dcc0', slow: '#6ab8ff', blind: '#b8b8b8',
  weaken: '#b07aff', sunder: '#ffaa6a', enrage: '#ff4a4a', charged: '#fff07a', fortify: '#aac0d8', haste: '#7affff',
  evasive: '#d0d0ff', regen: '#6aff9a', shield: '#7ad8ff',
};

const ease = {
  linear: k => k,
  inOut: k => k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2,
  out: k => 1 - (1 - k) * (1 - k),
  in: k => k * k,
  backOut: k => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(k - 1, 3) + c1 * Math.pow(k - 1, 2); },
};
const lerp = (a, b, t) => a + (b - a) * t;
const clampSpeed = (x) => Math.max(0.1, Math.min(16, Number(x) || 1));

function disposeTree(obj) {
  if (!obj) return;
  obj.traverse(o => {
    if (o.geometry && !o.geometry.userData?.shared) o.geometry.dispose();
    const mats = Array.isArray(o.material) ? o.material : (o.material ? [o.material] : []);
    for (const m of mats) {
      if (m.userData?.shared) continue;
      if (m.map && !m.map.userData?.shared) m.map.dispose();
      m.dispose();
    }
  });
}

function textSprite(text, color, size) {
  const cv = document.createElement('canvas');
  const ctx = cv.getContext('2d');
  const font = '900 96px Impact, "Arial Black", "Segoe UI", sans-serif';
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width + 36);
  cv.width = w; cv.height = 132;
  ctx.font = font; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round'; ctx.lineWidth = 14; ctx.strokeStyle = 'rgba(0,0,0,0.88)';
  ctx.strokeText(text, w / 2, 70);
  ctx.fillStyle = color; ctx.fillText(text, w / 2, 70);
  const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false, depthWrite: false, transparent: true, fog: false, toneMapped: false });
  const sp = new THREE.Sprite(mat);
  sp.userData.aspect = w / 132;
  sp.scale.set(size * sp.userData.aspect, size, 1);
  sp.renderOrder = 1000;
  return sp;
}

function addMat(color, mult = 2, opacity = 1) {
  const m = new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(mult), transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false, fog: false });
  return m;
}

export function playBattle(stage, record, opts = {}) {
  let speed = clampSpeed(opts.speed ?? 1);
  const onEvent = typeof opts.onEvent === 'function' ? opts.onEvent : () => {};
  const onDone = typeof opts.onDone === 'function' ? opts.onDone : () => {};
  record = record || {};
  const events = Array.isArray(record.events) ? record.events : [];
  const N = events.length;

  // ---------- scene setup
  if (record.env && stage.envId !== record.env) stage.setEnvironment(record.env);
  const fs = record.fighters || [];
  if (!stage.fighters[0] || !stage.fighters[1] || stage._snaps[0] !== fs[0] || stage._snaps[1] !== fs[1]) {
    stage.setFighters(fs[0] || {}, fs[1] || {});
  }
  const scene = stage.scene, dir = stage.director;
  const F = stage.fighters.slice(), R = F.map(m => m.userData.rig);
  const home = stage.homes.map(h => h.clone());
  const fwd = [new V3(Math.sign(home[1].x - home[0].x) || 1, 0, 0), null];
  fwd[1] = fwd[0].clone().negate();
  const maxHp = [0, 1].map(k => record.meta?.[k]?.maxHp || 1000);
  const arena = () => stage.arena;
  const accent = R.map((r, k) => new THREE.Color(fs[k]?.colors?.accent || r.mats.find(m => m.userData.glow)?.color || '#5fd6ff'));
  const resetAnim = (r) => Object.assign(r.anim, { lunge: 0, recoil: 0, roar: 0, bite: 0, charge: 0, walk: 0, lift: 0, ko: 0, victory: 0, flash: 0, dodge: 0, shieldPulse: 0 });
  R.forEach((r, k) => { resetAnim(r); r.clearStatuses(); F[k].position.copy(home[k]); });
  // wild-plan hooks (all optional) + generic glow boost used by passives on ordinary rigs
  const WL = R.map(r => r.extra?.wild || null);
  // human-scale wild fighters: frame toward them, push in when they act, make their hits land big
  const tinyIdx = R.findIndex(r => r.extra?.tiny);
  const isTiny = (k) => !!R[k]?.extra?.tiny;
  const biasX = (a) => tinyIdx >= 0 ? home[tinyIdx].x * 0.35 : home[a].x * 0.15;
  const fsc = (k, s) => isTiny(k) ? s * 0.6 : s;
  const homeRotY = F.map(f => f.rotation.y);
  const setGlowBoost = (k, f) => {
    const r = R[k];
    if (!r.__pbGlowBase) { if (f === 1) return; r.__pbGlowBase = r.mats.filter(m => m.userData.glow && m.userData.e0).map(m => [m, m.userData.e0.clone()]); }
    for (const [m, e] of r.__pbGlowBase) m.userData.e0.copy(e).multiplyScalar(f);
  };
  const resetWild = (k) => {
    try { WL[k]?.reset?.(); } catch (e) { console.warn('[playback] wild reset failed', e); }
    setGlowBoost(k, 1);
    F[k].rotation.y = homeRotY[k];
  };
  R.forEach((r, k) => {
    resetWild(k);
    try { WL[k]?.setTarget?.((out) => R[1 - k].getCenterWorld(out)); } catch { /* optional */ }
  });

  // ---------- clock / scheduling
  let clock = 0, paused = false, stopped = false, skipping = false, done = false;
  const alive = () => !(stopped || skipping);
  let timers = [], tweens = [];
  const fx = new Set();
  const fired = new Uint8Array(N);
  const fire = (i) => {
    if (i < 0 || i >= N || fired[i]) return;
    fired[i] = 1;
    try { onEvent(events[i], i); } catch (e) { console.error('[playback] onEvent error', e); }
  };
  const wait = (ms) => new Promise(res => { if (!alive()) return res(); timers.push({ at: clock + ms, res }); });
  const tween = (ms, fn, e = ease.inOut) => new Promise(res => {
    if (!alive()) return res();
    tweens.push({ t0: clock, dur: Math.max(1, ms), fn, e, res });
  });
  const W = async (ms) => { await wait(ms); return !alive(); };
  const T = async (ms, fn, e) => { await tween(ms, fn, e); return !alive(); };
  const tw = (ms, fn, e) => { tween(ms, fn, e); };            // fire and forget
  const realSec = (ms) => ms / 1000 / speed;

  const spawn = (obj, update, onEnd) => {
    if (!alive()) { disposeTree(obj); onEnd?.(); return null; }
    if (obj) scene.add(obj);
    const e = { obj, update, onEnd, t: 0 };
    fx.add(e);
    return e;
  };
  const kill = (e) => {
    if (!e || !fx.has(e)) return;
    fx.delete(e);
    if (e.obj) { scene.remove(e.obj); disposeTree(e.obj); }
    try { e.onEnd?.(); } catch { /* ignore */ }
  };

  // ---------- particle pool (sparks, trails, heal motes)
  const PMAX = 1200;
  const pPos = new Float32Array(PMAX * 3), pCol = new Float32Array(PMAX * 3), pVel = new Float32Array(PMAX * 3);
  const pLife = new Float32Array(PMAX), pMax = new Float32Array(PMAX), pGrav = new Float32Array(PMAX), pBase = new Float32Array(PMAX * 3);
  for (let i = 0; i < PMAX; i++) pPos[i * 3 + 1] = -9999;
  const pGeo = new THREE.BufferGeometry();
  pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
  pGeo.setAttribute('color', new THREE.BufferAttribute(pCol, 3));
  const pMat = new THREE.PointsMaterial({ size: 1.4, map: softTexture(), vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false });
  const points = new THREE.Points(pGeo, pMat); points.frustumCulled = false; points.renderOrder = 10;
  scene.add(points);
  let pNext = 0;
  const _c = new THREE.Color();
  const emit = (pos, vel, color, life = 0.8, grav = 0) => {
    const i = pNext; pNext = (pNext + 1) % PMAX;
    pPos[i * 3] = pos.x; pPos[i * 3 + 1] = pos.y; pPos[i * 3 + 2] = pos.z;
    pVel[i * 3] = vel.x; pVel[i * 3 + 1] = vel.y; pVel[i * 3 + 2] = vel.z;
    _c.set(color);
    pBase[i * 3] = _c.r * 1.6; pBase[i * 3 + 1] = _c.g * 1.6; pBase[i * 3 + 2] = _c.b * 1.6;
    pLife[i] = life; pMax[i] = life; pGrav[i] = grav;
  };
  const burst = (pos, color, n = 30, spd = 10, life = 0.6, grav = 10, dirBias = null) => {
    const vv = new V3();
    for (let k = 0; k < n; k++) {
      vv.set(Math.random() - 0.5, Math.random() - 0.3, Math.random() - 0.5).normalize().multiplyScalar(spd * (0.4 + Math.random() * 0.8));
      if (dirBias) vv.addScaledVector(dirBias, spd * 0.6);
      emit(pos, vv, color, life * (0.6 + Math.random() * 0.6), grav);
    }
  };
  const updateParticles = (dt) => {
    let any = false;
    for (let i = 0; i < PMAX; i++) {
      if (pLife[i] <= 0) continue;
      any = true;
      pLife[i] -= dt;
      if (pLife[i] <= 0) { pPos[i * 3 + 1] = -9999; pCol[i * 3] = pCol[i * 3 + 1] = pCol[i * 3 + 2] = 0; continue; }
      pVel[i * 3 + 1] -= pGrav[i] * dt;
      pPos[i * 3] += pVel[i * 3] * dt; pPos[i * 3 + 1] += pVel[i * 3 + 1] * dt; pPos[i * 3 + 2] += pVel[i * 3 + 2] * dt;
      const k = pLife[i] / pMax[i];
      pCol[i * 3] = pBase[i * 3] * k; pCol[i * 3 + 1] = pBase[i * 3 + 1] * k; pCol[i * 3 + 2] = pBase[i * 3 + 2] * k;
    }
    if (any) { pGeo.attributes.position.needsUpdate = true; pGeo.attributes.color.needsUpdate = true; }
  };

  // ---------- shared geometry
  const beamGeo = new THREE.CylinderGeometry(1, 1, 1, 14, 1, true).rotateX(PI / 2); beamGeo.userData.shared = true;
  const sphGeo = new THREE.SphereGeometry(1, 20, 14); sphGeo.userData.shared = true;
  const ringGeo = new THREE.RingGeometry(0.86, 1, 72).rotateX(-PI / 2); ringGeo.userData.shared = true;
  const sharedGeos = [beamGeo, sphGeo, ringGeo];

  // ---------- ticker
  const removeTicker = stage.addTicker((dt) => {
    if (stopped) return;
    const pdt = paused ? 0 : dt * speed;
    const ar = arena(); if (ar) ar.timeScale = paused ? 0 : speed;
    clock += pdt * 1000;
    if (tweens.length) {
      const list = tweens; tweens = [];
      for (const tw of list) {
        const k = Math.min(1, (clock - tw.t0) / tw.dur);
        try { tw.fn(tw.e(k)); } catch (e) { console.error('[playback] tween error', e); }
        if (k >= 1) tw.res(); else tweens.push(tw);
      }
    }
    if (timers.length) {
      const due = timers.filter(x => x.at <= clock);
      if (due.length) { timers = timers.filter(x => x.at > clock); due.forEach(x => x.res()); }
    }
    for (const e of [...fx]) {
      e.t += pdt;
      let keep = true;
      try { keep = e.update ? e.update(e.t, pdt) : false; } catch (err) { console.error('[playback] fx error', err); keep = false; }
      if (keep === false) kill(e);
    }
    updateParticles(pdt);
  });

  // ---------- helpers
  const centerOf = (i) => R[i].getCenterWorld(new V3());
  const mouthOf = (i, k = 0) => R[i].getMouthWorld(k, new V3());
  const topOf = (i) => R[i].getTopWorld(new V3());
  const gy = () => arena()?.fighterY ?? 0;
  const contactReach = (a) => {
    const b = 1 - a;
    const dist = Math.abs(F[b].position.x - F[a].position.x);
    const gapWorld = dist - R[a].front - R[b].front * 0.55;
    return Math.max(0.5, gapWorld + R[a].front * 0.2) / R[a].scale;
  };
  const statusColor = (st) => STATUS_COLORS[st] || '#ffffff';
  const beamColor = (a, ev) => {
    const ab = ABILITY_BY_ID[act?.ab];
    if (ab?.type === 'random' && TYPE_COLORS[ev?.type]) return new THREE.Color(TYPE_COLORS[ev.type]);
    return accent[a].clone();
  };
  const fxColor = (a, ev) => {
    const ab = ABILITY_BY_ID[act?.ab];
    const type = ev?.type || ab?.type;
    if (type && type !== 'physical' && TYPE_COLORS[type]) return new THREE.Color(TYPE_COLORS[type]);
    return accent[a].clone();
  };

  const floatText = (pos, text, color, size = 2.4, life = 1.25, rise = 4) => {
    const sp = textSprite(String(text), color, size);
    const p0 = pos.clone().add(new V3((Math.random() - 0.5) * 2.5, 0, 0));
    sp.position.copy(p0);
    spawn(sp, (t) => {
      const k = t / life;
      const pop = k < 0.12 ? ease.backOut(k / 0.12) : 1;
      sp.scale.set(size * sp.userData.aspect * pop, size * pop, 1);
      sp.position.set(p0.x, p0.y + ease.out(Math.min(1, k)) * rise, p0.z);
      sp.material.opacity = k > 0.65 ? Math.max(0, 1 - (k - 0.65) / 0.35) : 1;
      return k < 1;
    });
  };

  const recoil = (tg, amt = 1) => {
    const an = R[tg].anim;
    tw(90, k => { an.recoil = Math.max(an.recoil * (1 - k), amt * k); }, ease.out);
    wait(90).then(() => tw(380, k => { an.recoil = amt * (1 - k); }, ease.inOut));
  };
  const knockback = (tg, dist) => {
    const p0 = F[tg].position.clone();
    const p1 = p0.clone().addScaledVector(fwd[tg], -dist);
    tw(260, k => F[tg].position.lerpVectors(p0, p1, k), ease.out);
  };

  function impact(ev, i, o = {}) {
    const tg = ev.tg ?? (1 - (ev.a ?? 0));
    const a = ev.a ?? (1 - tg);
    fire(i);
    const r = R[tg];
    const crit = !!ev.crit, dmg = Math.max(0, ev.dmg | 0);
    const big = crit || dmg >= maxHp[tg] * 0.12 || o.thrown;
    const tinyA = isTiny(a) && !isTiny(tg);
    const absorbedOnly = (ev.absorbed | 0) > 0 && dmg === 0;
    r.hit(absorbedOnly ? '#9ad8ff' : '#ff2a1a', 1);
    if (!absorbedOnly) {
      let wp = null;
      try { wp = WL[tg]?.onHit?.(big, new V3()) || null; } catch { wp = null; }
      if (wp) explode(wp, big ? 1.3 : 0.9);
    }
    if ((ev.absorbed | 0) > 0) { r.anim.shieldPulse = 1; tw(600, k => { r.anim.shieldPulse = 1 - k; }); }
    if (!o.thrown) recoil(tg, tinyA ? (crit ? 2.0 : o.small ? 1.0 : 1.4) : o.small ? 0.5 : big ? 1.3 : 0.8);
    if ((big || tinyA) && !o.thrown) knockback(tg, tinyA ? (crit ? 5.5 : o.small ? 1.6 : 3) : crit ? 3.5 : 2);
    const hp = centerOf(tg).addScaledVector(fwd[tg], R[tg].front * 0.4);
    const col = fxColor(a, ev);
    burst(hp, col, crit ? 60 : 30, crit ? 16 : 11, 0.55, 12, fwd[a].clone().multiplyScalar(0.5));
    if (tinyA) shockwave(hp, fwd[a], col, crit);
    burst(hp, '#ffffff', crit ? 20 : 10, 8, 0.25, 0);
    // flash sphere
    const fl = new THREE.Mesh(sphGeo, addMat(col, 2.5, 0.9));
    fl.position.copy(hp);
    spawn(fl, (t) => { const k = t / 0.25; fl.scale.setScalar(1 + k * (crit ? 6 : 3.5)); fl.material.opacity = 0.9 * (1 - k); return k < 1; });
    const top = topOf(tg);
    if (absorbedOnly) floatText(top, 'BLOCKED', '#9ad8ff', 2.0);
    else floatText(top, dmg, crit ? '#ffe23a' : '#ffffff', crit ? 3.8 : 2.5, crit ? 1.5 : 1.2);
    if (crit) floatText(top.clone().add(new V3(0, 2.6, 0)), 'CRIT!', '#ffb02a', 1.6, 1.3, 3);
    if ((ev.absorbed | 0) > 0 && !absorbedOnly) floatText(top.clone().add(new V3(0, -1.6, 0)), `-${ev.absorbed} shield`, '#9ad8ff', 1.2);
    if (big) dir.focus(tinyA ? centerOf(tg).lerp(centerOf(a), 0.4) : centerOf(tg), crit ? 0.55 : 0.7, realSec(crit ? 1100 : 800));
    dir.shake(tinyA ? (crit ? 1.5 : 0.65) : crit ? 0.9 : big ? 0.45 : 0.15);
    if (crit || o.thrown || (tinyA && big)) {
      const behind = F[tg].position.clone().addScaledVector(fwd[tg], -(R[tg].back + 10));
      arena()?.crumbleNear(behind, 10);
    }
  }

  function missFx(ev, i, o = {}) {
    const tg = ev.tg ?? (1 - (ev.a ?? 0));
    fire(i);
    floatText(topOf(tg), 'MISS', '#cfd8e0', 2.2);
    const an = R[tg].anim;
    if (o.hop) {
      tw(380, k => { an.lift = Math.sin(k * PI) * 2.5 / R[tg].scale; });
    } else {
      const side = Math.random() < 0.5 ? -1 : 1;
      tw(180, k => { an.dodge = side * 2.6 * k; }, ease.out);
      wait(260).then(() => tw(300, k => { an.dodge = side * 2.6 * (1 - k); }));
    }
  }

  // ---------- fx builders
  function chargeOrb(a, k = 0, color, sizeMul = 1) {
    const g = new THREE.Group();
    const core = new THREE.Mesh(sphGeo, addMat('#ffffff', 2.5, 1));
    const halo = new THREE.Mesh(sphGeo, addMat(color, 2.2, 0.55));
    halo.scale.setScalar(2.2);
    g.add(core, halo);
    const size = Math.max(0.5, R[a].height / 20) * sizeMul;
    const p = new V3();
    return spawn(g, (t) => {
      R[a].getMouthWorld(k, g.position);
      const s = Math.min(1, t / 0.7) * size * (1 + 0.12 * Math.sin(t * 45));
      g.scale.setScalar(s);
      if (Math.random() < 0.7) {
        p.set(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).normalize().multiplyScalar(size * 6);
        emit(g.position.clone().add(p), p.clone().multiplyScalar(-2.2), color, 0.42, 0);
      }
      return true;
    });
  }

  function beamFx(a, k, toFn, color, life = 0.45, opts = {}) {
    const w = (R[a].extra?.tiny ? 1.5 : Math.max(0.55, R[a].height / 13)) * (opts.w ?? 1);
    const g = new THREE.Group();
    const core = new THREE.Mesh(beamGeo, addMat('#ffffff', 2.4, 1));
    const outer = new THREE.Mesh(beamGeo, addMat(color, 2.0, 0.6));
    const flare = new THREE.Mesh(sphGeo, addMat(color, 3, 0.85));
    const muzzle = new THREE.Mesh(sphGeo, addMat(color, 3, 0.85));
    g.add(core, outer, flare, muzzle);
    const from = new V3(), to = new V3(), end = new V3(), mid = new V3();
    return spawn(g, (t) => {
      if (opts.fromFn) opts.fromFn(from); else R[a].getMouthWorld(k, from);
      toFn(to);
      const grow = Math.min(1, t / 0.08);
      const fade = t > life - 0.15 ? Math.max(0, (life - t) / 0.15) : 1;
      end.copy(from).lerp(to, grow);
      mid.lerpVectors(from, end, 0.5);
      const L = Math.max(0.01, from.distanceTo(end));
      const wob = 1 + Math.sin(t * 70) * 0.12;
      for (const m of [core, outer]) { m.position.copy(mid); m.lookAt(end); }
      core.scale.set(w * 0.33 * fade * wob, w * 0.33 * fade * wob, L);
      outer.scale.set(w * fade * wob, w * fade * wob, L);
      flare.position.copy(end); flare.scale.setScalar(w * 2.2 * fade * wob * grow);
      muzzle.position.copy(from); muzzle.scale.setScalar(w * 1.3 * fade);
      if (grow >= 1 && Math.random() < 0.8) burst(end, color, 2, 9, 0.4, 6);
      return t < life;
    });
  }

  function makeProjectile(ab, a, ev) {
    const id = String(ab || '');
    const col = fxColor(a, ev);
    const g = new THREE.Group();
    let kind = 'orb';
    if (id.includes('missile')) {
      kind = 'missile';
      const body = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 1.8, 10).rotateX(PI / 2), new THREE.MeshStandardMaterial({ color: '#c8ccd2', metalness: 0.6, roughness: 0.4 }));
      const tip = new THREE.Mesh(new THREE.ConeGeometry(0.28, 0.6, 10).rotateX(PI / 2), new THREE.MeshStandardMaterial({ color: '#d02a1a', roughness: 0.4 }));
      tip.position.z = 1.2;
      const flame = new THREE.Mesh(new THREE.ConeGeometry(0.3, 1.2, 8).rotateX(-PI / 2), addMat('#ffa040', 2.5, 0.9));
      flame.position.z = -1.4;
      g.add(body, tip, flame);
    } else if (id.includes('toss') || id.includes('boulder') || id.includes('mountain')) {
      kind = 'rock';
      const geo = new THREE.IcosahedronGeometry(2.6, 1);
      const p = geo.attributes.position;
      for (let i = 0; i < p.count; i++) { const k = 0.8 + 0.4 * Math.abs(Math.sin(p.getX(i) * 9 + p.getY(i) * 5)); p.setXYZ(i, p.getX(i) * k, p.getY(i) * k, p.getZ(i) * k); }
      geo.computeVertexNormals();
      const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: '#6a5a4a', roughness: 0.95, flatShading: true }));
      m.castShadow = true; g.add(m);
    } else if (id.includes('rocket') || id.includes('punch')) {
      kind = 'fist';
      const bodyCol = R[a].mats[0]?.color || new THREE.Color('#777');
      const m = new THREE.Mesh(new THREE.BoxGeometry(1.4, 1.2, 1.5), new THREE.MeshStandardMaterial({ color: bodyCol, metalness: 0.6, roughness: 0.35 }));
      const flame = new THREE.Mesh(new THREE.ConeGeometry(0.5, 1.6, 8).rotateX(-PI / 2), addMat('#7ac8ff', 2.5, 0.9));
      flame.position.z = -1.5;
      m.castShadow = true; g.add(m, flame);
    } else if (id.includes('web')) {
      kind = 'web';
      g.add(new THREE.Mesh(sphGeo, new THREE.MeshStandardMaterial({ color: '#f0f0e8', roughness: 0.8, transparent: true, opacity: 0.85 })));
      g.add(new THREE.Mesh(new THREE.IcosahedronGeometry(1.6, 1), new THREE.MeshBasicMaterial({ color: '#ffffff', wireframe: true })));
      g.children[0].scale.setScalar(0.9);
    } else {
      const core = new THREE.Mesh(sphGeo, addMat('#ffffff', 2.5, 1)); core.scale.setScalar(0.5);
      const halo = new THREE.Mesh(sphGeo, addMat(col, 2.2, 0.6)); halo.scale.setScalar(1.1);
      if (id.includes('barb')) { halo.scale.set(0.6, 0.6, 1.8); core.scale.set(0.3, 0.3, 1.2); }
      g.add(core, halo);
    }
    g.userData.kind = kind; g.userData.col = col;
    return g;
  }

  function travel(obj, from, toFn, ms, arcH = 0) {
    return new Promise(res => {
      const kind = obj.userData.kind, col = obj.userData.col;
      const prev = from.clone(), to = new V3(), wob = new V3((Math.random() - 0.5) * 6, Math.random() * 3, (Math.random() - 0.5) * 8);
      const e = spawn(obj, (t) => {
        const k = Math.min(1, t * 1000 / ms);
        toFn(to);
        prev.copy(obj.position);
        obj.position.lerpVectors(from, to, k);
        obj.position.y += Math.sin(k * PI) * arcH;
        if (kind === 'missile') obj.position.addScaledVector(wob, Math.sin(k * PI) * 0.6);
        if (k > 0.01) {
          const ahead = obj.position.clone().add(obj.position.clone().sub(prev));
          if (kind === 'rock') { obj.rotation.x += 0.15; obj.rotation.z += 0.1; } else obj.lookAt(ahead);
        }
        if (kind === 'missile') emit(obj.position, new V3((Math.random() - 0.5), 1, (Math.random() - 0.5)), '#ffb070', 0.5, -2);
        else if (kind === 'fist') emit(obj.position, new V3(0, 0.5, 0), '#7ac8ff', 0.3, 0);
        else if (kind === 'orb') emit(obj.position, new V3(0, 0, 0), col, 0.3, 0);
        return k < 1;
      }, res);
      if (!e) res();
    });
  }

  async function aoeBlast(a, tg, colOverride) {
    const r = R[a];
    const ab = ABILITY_BY_ID[act?.ab];
    let col = colOverride || fxColor(a, { type: ab?.type });
    if (act?.ab === 'ink_cloud') col = new THREE.Color('#2a1a3a');
    if (act?.ab === 'gale_wings' || act?.ab === 'wing_buffet') col = new THREE.Color('#e8f0ff');
    const l0 = r.anim.lift;
    if (await T(120, k => { r.anim.lift = l0 * (1 - k); r.anim.roar = 0.5 + 0.5 * k; }, ease.in)) return true;
    const p = F[a].position.clone(); p.y = gy() + 0.3;
    arena()?.groundImpact(p, 1.6);
    dir.shake(0.55);
    const reach = Math.abs(F[1 - a].position.x - F[a].position.x) + R[1 - a].radius + 4;
    for (let n = 0; n < 2; n++) {
      const ring = new THREE.Mesh(ringGeo, addMat(col, act?.ab === 'ink_cloud' ? 0.6 : 2.2, 0.9));
      ring.position.copy(p).add(new V3(0, n * 0.4, 0));
      const delay = n * 0.12;
      spawn(ring, (t) => { const k = Math.max(0, (t - delay) / 0.55); ring.scale.setScalar(1 + k * reach); ring.material.opacity = 0.9 * (1 - k); ring.visible = t >= delay; return k < 1; });
    }
    const dome = new THREE.Mesh(sphGeo, addMat(col, act?.ab === 'ink_cloud' ? 0.5 : 1.4, 0.25));
    dome.position.copy(p);
    spawn(dome, (t) => { const k = t / 0.6; dome.scale.set(1 + k * reach * 0.8, (1 + k * reach * 0.8) * 0.45, 1 + k * reach * 0.8); dome.material.opacity = 0.25 * (1 - k); return k < 1; });
    for (let k = 0; k < 40; k++) {
      const ang = Math.random() * PI * 2;
      emit(p.clone().add(new V3(Math.cos(ang) * 2, 0.5, Math.sin(ang) * 2)), new V3(Math.cos(ang) * 20, 2 + Math.random() * 4, Math.sin(ang) * 20), col, 0.6, 4);
    }
    tw(300, k => { r.anim.roar = 1 - k; });
    if (tg != null) { if (await W(Math.min(400, reach * 18))) return true; }
    return false;
  }

  async function roarFx(a) {
    const r = R[a];
    dir.focus(mouthOf(a), 0.75, realSec(1100));
    if (await T(260, k => { r.anim.roar = k; }, ease.out)) return true;
    dir.shake(0.6);
    for (let n = 0; n < 3; n++) {
      const s = new THREE.Mesh(sphGeo, addMat(accent[a], 1.2, 0.3));
      const delay = n * 0.14;
      spawn(s, (t) => {
        const k = Math.max(0, (t - delay) / 0.6);
        r.getMouthWorld(0, s.position);
        s.scale.setScalar(1 + k * 18); s.material.opacity = 0.3 * (1 - k); s.visible = t >= delay;
        return k < 1;
      });
    }
    if (await W(620)) return true;
    tw(300, k => { r.anim.roar = 1 - k; });
    return false;
  }

  async function healFx(a, color = '#5aff8a') {
    const r = R[a];
    r.hit(color, 0.8);
    const base = F[a].position.clone(); base.y = gy() + 0.4;
    const ring = new THREE.Mesh(ringGeo, addMat(color, 2, 0.8));
    ring.position.copy(base);
    const rad = Math.max(4, r.radius * 0.9);
    spawn(ring, (t) => { const k = t / 0.9; ring.scale.setScalar(rad * (0.6 + k * 0.6)); ring.material.opacity = 0.8 * (1 - k); return k < 1; });
    spawn(null, (t) => {
      for (let n = 0; n < 4; n++) {
        const ang = Math.random() * PI * 2, rr = Math.random() * rad;
        emit(base.clone().add(new V3(Math.cos(ang) * rr, Math.random() * r.height * 0.3, Math.sin(ang) * rr)), new V3(0, 5 + Math.random() * 6, 0), color, 1.0, -2);
      }
      return t < 0.8;
    });
    return W(800);
  }

  async function shieldFx(a) {
    const r = R[a];
    r.hit(accent[a].getStyle(), 0.5);
    if (await T(300, k => { r.anim.shieldPulse = k * 1.4; }, ease.out)) return true;
    tw(600, k => { r.anim.shieldPulse = 1.4 * (1 - k); });
    return W(400);
  }

  async function buffFx(a, color) {
    const r = R[a];
    const ab = ABILITY_BY_ID[act?.ab];
    const st = ab?.selfEffects?.[0]?.status;
    const col = new THREE.Color(color || (st ? statusColor(st) : accent[a]));
    r.hit(col.getStyle(), 0.7);
    const rad = Math.max(3.5, r.radius * 0.7);
    const aura = new THREE.Mesh(new THREE.CylinderGeometry(rad, rad * 1.1, r.height * 1.1, 32, 1, true), addMat(col, 1.6, 0.35));
    aura.material.side = THREE.DoubleSide;
    const base = F[a].position.clone();
    spawn(aura, (t) => {
      const k = t / 0.9;
      aura.position.set(base.x, gy() + r.height * 0.55 + k * 2, base.z);
      aura.scale.set(1 + k * 0.3, 1, 1 + k * 0.3);
      aura.material.opacity = 0.35 * Math.sin(Math.min(1, k) * PI);
      if (Math.random() < 0.8) { const ang = Math.random() * PI * 2; emit(new V3(base.x + Math.cos(ang) * rad, gy() + Math.random() * r.height, base.z + Math.sin(ang) * rad), new V3(0, 8, 0), col, 0.6, 0); }
      return k < 1;
    });
    tw(250, k => { r.anim.roar = 0.4 * k; });
    if (await W(700)) return true;
    tw(250, k => { r.anim.roar = 0.4 * (1 - k); });
    return false;
  }

  // ---------- wild-plan fx (army explosions / muzzle flashes / air strikes, ribbons, passives)
  // outsized impact for tiny attackers: a vertical shockwave ring facing the blow, ground ring, sparks, debris
  function shockwave(p, dirV, col, crit) {
    const s = crit ? 1.6 : 1;
    for (let j = 0; j < (crit ? 3 : 2); j++) {
      const ring = new THREE.Mesh(ringGeo, addMat(j ? '#ffffff' : col, 2.4, 0.9));
      ring.quaternion.setFromUnitVectors(new V3(0, 1, 0), dirV.clone().normalize());
      ring.position.copy(p);
      const delay = j * 0.07;
      spawn(ring, (t) => { const k = Math.max(0, (t - delay) / 0.45); ring.visible = t >= delay; ring.scale.setScalar(0.5 + k * 9 * s); ring.material.opacity = 0.9 * (1 - k); return k < 1; });
    }
    groundRings(p.clone().setY(gy()), col, 9 * s, 2, 0.55);
    burst(p, '#ffffff', Math.round(40 * s), 22 * s, 0.4, 4, dirV.clone().multiplyScalar(0.6));
    burst(p, '#8a7a68', Math.round(24 * s), 9, 1.0, 14);
    if (crit) { arena()?.flash?.('#ffffff', 0.22, 220); const fl = new THREE.Mesh(sphGeo, addMat(col, 2.6, 0.9)); fl.position.copy(p); spawn(fl, (t) => { const k = t / 0.3; fl.scale.setScalar(2 + k * 10); fl.material.opacity = 0.9 * (1 - k); return k < 1; }); }
  }
  function explode(p, s = 1) {
    const fl = new THREE.Mesh(sphGeo, addMat('#ffb050', 2.6, 0.95));
    fl.position.copy(p);
    spawn(fl, (t) => { const k = t / 0.35; fl.scale.setScalar(s * (0.8 + k * 3.2)); fl.material.opacity = 0.95 * (1 - k); return k < 1; });
    burst(p, '#ff8a2a', Math.round(26 * s), 10 * s, 0.6, 6);
    burst(p, '#ffe08a', Math.round(14 * s), 14 * s, 0.35, 4);
    burst(p, '#5a5048', Math.round(12 * s), 4, 1.2, -3);
  }
  function muzzleFlash(p, s = 1) {
    const fl = new THREE.Mesh(sphGeo, addMat('#ffd27a', 3, 1));
    fl.position.copy(p);
    spawn(fl, (t) => { const k = t / 0.12; fl.scale.setScalar(s * (0.4 + k * 0.9)); fl.material.opacity = 1 - k; return k < 1; });
    burst(p, '#ffb050', Math.max(3, Math.round(8 * s)), 6 * s, 0.25, 0);
    burst(p, '#6a625a', Math.max(2, Math.round(5 * s)), 2, 0.9, -2);
  }
  function pillar(pos, color, h, life = 1, r0 = 2) {
    const g = new THREE.Group();
    const col = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 24, 1, true), addMat(color, 1.5, 0.4));
    col.material.side = THREE.DoubleSide;
    const core = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 16, 1, true), addMat('#ffffff', 1.6, 0.45));
    g.add(col, core);
    g.position.copy(pos);
    spawn(g, (t) => {
      const k = t / life, open = Math.min(1, t / 0.12), fade = k > 0.55 ? Math.max(0, 1 - (k - 0.55) / 0.45) : 1;
      col.scale.set(r0 * open * (1 + k * 0.4), h, r0 * open * (1 + k * 0.4)); col.position.y = h / 2;
      core.scale.set(r0 * 0.3 * open, h, r0 * 0.3 * open); core.position.y = h / 2;
      col.material.opacity = 0.4 * fade; core.material.opacity = 0.45 * fade;
      return k < 1;
    });
  }
  function groundRings(pos, color, rad, n = 2, life = 0.7) {
    for (let j = 0; j < n; j++) {
      const ring = new THREE.Mesh(ringGeo, addMat(color, 2.2, 0.9));
      ring.position.copy(pos).add(new V3(0, 0.3 + j * 0.3, 0));
      const delay = j * 0.14;
      spawn(ring, (t) => { const k = Math.max(0, (t - delay) / life); ring.visible = t >= delay; ring.scale.setScalar(1 + k * rad); ring.material.opacity = 0.9 * (1 - k); return k < 1; });
    }
  }

  async function airStrike(a, tg) {
    const w = WL[a];
    const tgt = tg != null ? F[tg].position.clone() : F[a].position.clone().addScaledVector(fwd[a], 20);
    tgt.y = gy();
    w.strafe(tgt);
    dir.focus(tgt.clone().setY(gy() + 6), 0.95, realSec(2000));
    const p = new V3(), q = new V3();
    if (await T(820, k => {
      w.strafeK = k * 0.5;
      if (k > 0.5) for (let j = 0; j < w.jetCount; j++) {
        if (Math.random() > 0.6) continue;
        w.jetWorld(j, p);
        q.set(tgt.x + (Math.random() - 0.5) * 8, tgt.y + Math.random() * 2, tgt.z + (Math.random() - 0.5) * 8).sub(p).normalize().multiplyScalar(80);
        emit(p, q, '#ffe08a', 0.3, 0);
      }
    }, ease.linear)) return true;
    for (let j = 0; j < w.jetCount; j++) {
      w.jetWorld(j, p);
      for (let b = 0; b < 2; b++) {
        const bomb = new THREE.Mesh(sphGeo, new THREE.MeshStandardMaterial({ color: '#2a2a26', roughness: 0.6 }));
        bomb.scale.set(0.35, 0.35, 0.7);
        const to = tgt.clone().add(new V3((Math.random() - 0.5) * 9, 0, (Math.random() - 0.5) * 9));
        travel(bomb, p.clone(), (out) => out.copy(to), 380 + b * 110, 0).then(() => { if (!alive()) return; explode(to.clone().setY(gy() + 1), 1.5); arena()?.groundImpact(to, 1.1); dir.shake(0.5); });
      }
    }
    tw(820, k => { w.strafeK = 0.5 + k * 0.5; }, ease.linear);
    wait(840).then(() => { if (w.strafeK >= 0.999) w.strafeK = -1; });
    if (await W(540)) return true;
    arena()?.flash?.('#ffb060', 0.18, 250);
    dir.shake(0.8);
    return false;
  }

  async function ribbonFx(a, tg) {
    const r = R[a];
    const cols = [accent[a].clone(), new THREE.Color(fs[a]?.colors?.body || '#ff8fc7')];
    const from = mouthOf(a), c0 = centerOf(tg);
    const rad = Math.max(3, R[tg].radius * 0.55), h = R[tg].height;
    dir.focus(c0.clone().lerp(from, 0.3), 0.85, realSec(1500));
    if (await T(220, k => { r.anim.lunge = -0.12 * k; r.anim.roar = 0.3 * k; })) return true;
    tw(200, k => { r.anim.lunge = -0.12 + 0.5 * k; });
    for (let n = 0; n < 2; n++) {
      const pts = [from.clone()];
      const mid = from.clone().lerp(c0, 0.5); mid.y += 4 + n * 2.5; mid.z += (n ? -1 : 1) * 3;
      pts.push(mid);
      for (let s = 0; s <= 28; s++) {
        const u = s / 28, an = u * PI * 6 + n * PI;
        pts.push(new V3(c0.x + Math.cos(an) * rad, gy() + h * (0.12 + 0.72 * u), c0.z + Math.sin(an) * rad));
      }
      const geo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 220, 0.28, 6, false);
      const mesh = new THREE.Mesh(geo, addMat(cols[n], 1.8, 0.9));
      const total = geo.index.count;
      geo.setDrawRange(0, 0);
      const pp = new V3();
      spawn(mesh, (t) => {
        const k = Math.min(1, t / 0.7);
        geo.setDrawRange(0, Math.floor(total * k / 3) * 3);
        mesh.material.opacity = t > 1.3 ? Math.max(0, 0.9 * (1 - (t - 1.3) / 0.45)) : 0.9;
        if (k < 1 && Math.random() < 0.8) { geo.parameters.path.getPointAt(k, pp); emit(pp, new V3((Math.random() - 0.5) * 3, Math.random() * 3, (Math.random() - 0.5) * 3), Math.random() < 0.5 ? '#ffffff' : cols[n], 0.6, 0); }
        return t < 1.75;
      });
    }
    if (await W(700)) return true;
    R[tg].hit('#ffc0e8', 0.8);
    burst(c0, cols[0], 40, 10, 0.7, 1);
    burst(c0, '#ffffff', 20, 8, 0.5, 0);
    dir.shake(0.25);
    tw(300, k => { r.anim.roar = 0.3 * (1 - k); });
    return W(380);
  }

  function spiralFx(tg, colors, life = 1.9) {
    const r = R[tg];
    const g = new THREE.Group();
    const rad = isTiny(tg) ? 1.4 : Math.max(3, r.radius * 0.7), h = isTiny(tg) ? 4.2 : r.height * 1.25;
    const curves = [];
    for (let n = 0; n < colors.length; n++) {
      const pts = [];
      for (let s = 0; s <= 60; s++) { const u = s / 60, an = u * PI * 5 + n * PI * 2 / colors.length; pts.push(new V3(Math.cos(an) * rad * (1 - u * 0.35), u * h, Math.sin(an) * rad * (1 - u * 0.35))); }
      const curve = new THREE.CatmullRomCurve3(pts);
      curves.push(curve);
      g.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 200, 0.2, 6, false), addMat(colors[n], 1.25, 0.6)));
    }
    const base = F[tg].position.clone(); base.y = gy();
    g.position.copy(base);
    const pp = new V3();
    spawn(g, (t) => {
      const k = t / life;
      g.rotation.y = t * 5;
      g.scale.set(1 + k * 0.25, Math.min(1, t / 0.5), 1 + k * 0.25);
      const fade = k > 0.75 ? Math.max(0, 1 - (k - 0.75) / 0.25) : 1;
      g.children.forEach(m => { m.material.opacity = 0.6 * fade; });
      for (let j = 0; j < 3; j++) {
        const c = curves[j % curves.length]; c.getPointAt(Math.random(), pp);
        pp.multiply(g.scale).applyAxisAngle(new V3(0, 1, 0), g.rotation.y).add(base);
        emit(pp, new V3(0, 3 + Math.random() * 3, 0), Math.random() < 0.4 ? '#ffffff' : colors[j % colors.length], 0.6, 0);
      }
      return k < 1;
    });
  }

  // ---------- action context
  let act = null;
  const hasHits = (i, a) => {
    for (let j = i + 1; j < N; j++) {
      const e = events[j];
      if (e.t === 'hit' || e.t === 'miss') { if ((e.a ?? a) === a) return true; continue; }
      if (e.t === 'status' || e.t === 'expire' || e.t === 'heal' || e.t === 'selfdmg' || e.t === 'passive') continue;
      return false;
    }
    return false;
  };
  const hitCount = (i, a) => { let n = 0; for (let j = i + 1; j < N; j++) { const e = events[j]; if (e.t === 'hit' || e.t === 'miss') n++; else if (!['status', 'expire', 'heal', 'selfdmg', 'passive'].includes(e.t)) break; } return n; };

  async function endAction() {
    if (act?.orb) kill(act.orb);
    if (act?.orb2) kill(act.orb2);
    act = null;
    for (let k = 0; k < 2; k++) F[k].rotation.y = homeRotY[k];
    const ps = [];
    for (let k = 0; k < 2; k++) {
      const rr = R[k], an = rr.anim;
      if (an.ko > 0) continue;
      const l0 = an.lunge, c0 = an.charge, li0 = an.lift, b0 = an.bite, ro0 = an.roar;
      const needsMove = F[k].position.distanceToSquared(home[k]) > 0.04;
      if (!needsMove && Math.abs(l0) < 0.01 && c0 < 0.01 && Math.abs(li0) < 0.01 && ro0 < 0.01 && b0 < 0.01) continue;
      const p0 = F[k].position.clone();
      if (needsMove) an.walk = 1;
      ps.push(tween(needsMove ? 480 : 220, kk => {
        an.lunge = l0 * (1 - kk); an.charge = c0 * (1 - kk); an.lift = li0 * (1 - kk); an.bite = b0 * (1 - kk); an.roar = ro0 * (1 - kk);
        if (needsMove) F[k].position.lerpVectors(p0, home[k], kk);
      }).then(() => { if (alive()) an.walk = 0; }));
    }
    if (ps.length) await Promise.all(ps);
  }

  // ---------- event handlers
  async function onStart(i) {
    const start = [0, 1].map(k => home[k].clone().addScaledVector(fwd[k], -16));
    R.forEach((r, k) => { F[k].position.copy(start[k]); r.anim.walk = 1; });
    dir.clearFocus();
    if (await T(1700, k => { for (let n = 0; n < 2; n++) F[n].position.lerpVectors(start[n], home[n], k); }, ease.out)) return;
    R.forEach(r => { r.anim.walk = 0; });
    if (await T(350, k => R.forEach(r => { r.anim.roar = k; }), ease.out)) return;
    dir.shake(0.4);
    for (let n = 0; n < 2; n++) arena()?.groundImpact(F[n].position, 0.8);
    fire(i);
    if (await W(450)) return;
    await T(300, k => R.forEach(r => { r.anim.roar = 1 - k; }));
  }

  async function onUse(ev, i) {
    const a = ev.a ?? 0, tg = ev.tg ?? 1 - a;
    const strikes = hasHits(i, a);
    const kind = FX_KINDS.has(ev.fx) ? ev.fx : (strikes ? 'melee' : 'buff');
    act = { a, tg, kind, ab: ev.ab, hits: 0, hasHits: strikes, nHits: hitCount(i, a) };
    fire(i);
    const r = R[a];
    r.anim.reach = contactReach(a);
    dir.setBias(biasX(a));
    if (isTiny(a)) dir.focus(centerOf(a).lerp(centerOf(tg), 0.3), 0.55, realSec(1000));
    switch (kind) {
      case 'melee': case 'multi':
        if (await T(220, k => { r.anim.lunge = -0.12 * k; })) return;
        break;
      case 'beam': {
        const col = beamColor(a, null);
        const WH = WL[a];
        dir.focus(mouthOf(a).lerp(centerOf(tg), 0.3), 0.85, realSec(1300));
        act.orb = chargeOrb(a, 0, col, WH?.twinBeam ? 0.45 : 1);
        if (WH?.twinBeam) act.orb2 = chargeOrb(a, 1, col, 0.45);
        if (WH?.beamSources) {
          const cs = WH.memberColors?.() || [];
          WH.beamSources().forEach((o, j) => beamFx(a, 0, (out) => R[a].getMouthWorld(0, out), cs[j] || col, 0.7, { fromFn: (out) => o.getWorldPosition(out), w: 0.18 }));
        }
        if (await T(700, k => { r.anim.charge = k; })) return;
        break;
      }
      case 'projectile':
        if (!WL[a]?.noLunge && await T(240, k => { r.anim.lunge = -0.15 * k; })) return;
        if (!strikes) {
          // debuff projectiles (e.g. web) have no hit events: still throw something at the target
          const obj = makeProjectile(ev.ab, a, null);
          const from = mouthOf(a);
          await travel(obj, from, (out) => R[tg].getCenterWorld(out), 480, from.distanceTo(centerOf(tg)) * 0.1);
          if (!alive()) return;
          burst(centerOf(tg), fxColor(a, null), 30, 9, 0.5, 6);
          R[tg].hit('#ffffff', 0.5);
          tw(160, k => { r.anim.lunge = -0.15 * (1 - k); });
        }
        break;
      case 'aoe':
        if (await T(320, k => { if (!WL[a]?.noLift) r.anim.lift = Math.sin(k * PI / 2) * 2.2 / r.scale; r.anim.roar = k * 0.5; }, ease.out)) return;
        if (!strikes) { if (WL[a]?.strafe) await airStrike(a, null); else await aoeBlast(a, null); if (!alive()) return; await W(300); }
        break;
      case 'grab': {
        if (ev.ab === 'ribbon_bind' || WL[a]?.ribbonGrab) { await ribbonFx(a, tg); break; }
        const from = F[a].position.clone();
        const to = F[tg].position.clone().addScaledVector(fwd[a], -(R[a].front * 0.75 + R[tg].front * 0.35));
        if (from.distanceTo(home[a]) < 0.1) {
          r.anim.walk = 1;
          if (await T(460, k => F[a].position.lerpVectors(from, to, k))) return;
          r.anim.walk = 0;
          r.anim.reach = contactReach(a);
        }
        break;
      }
      case 'roar': await roarFx(a); break;
      case 'heal': await healFx(a); break;
      case 'shield': await shieldFx(a); break;
      case 'buff': await buffFx(a); break;
    }
  }

  async function onStrike(ev, i, hit) {
    const a = ev.a ?? act?.a ?? 0, tg = ev.tg ?? 1 - a;
    if (!act || act.a !== a) act = { a, tg, kind: 'melee', hits: 0, hasHits: true, nHits: 1 };
    act.hits++;
    const kind = act.kind, r = R[a], first = act.hits === 1;
    const WH = WL[a];
    if (kind !== 'grab') r.anim.reach = contactReach(a);
    try { WH?.onStrike?.(act.hits, act.nHits, kind, act.ab); } catch { /* optional */ }
    switch (kind) {
      case 'multi': {
        const from = r.anim.lunge;
        if (String(act.ab || '').includes('spin')) {
          // spinning multi-hit (e.g. Hurricane Spin): one full turn per hit, ends facing the same way
          const n = act.hits;
          tw(240, k => { F[a].rotation.y = homeRotY[a] + (n - 1 + k) * PI * 2; }, ease.linear);
          groundRings(F[a].position.clone().setY(gy()), '#e8f4ff', R[a].radius * 0.9, 1, 0.4);
          burst(centerOf(a), '#e8f4ff', 14, 12, 0.4, 0);
        }
        if (await T(110, k => { r.anim.lunge = lerp(from, 1, k); r.anim.bite = k; }, ease.in)) return;
        if (hit) impact(ev, i, { small: true }); else missFx(ev, i);
        if (await T(130, k => { r.anim.lunge = lerp(1, 0.4, k); r.anim.bite = 1 - k; })) return;
        break;
      }
      case 'beam': {
        if (act.orb) { kill(act.orb); act.orb = null; }
        if (act.orb2) { kill(act.orb2); act.orb2 = null; }
        const k = act.hits - 1;
        const col = beamColor(a, ev);
        const side = Math.random() < 0.5 ? -1 : 1;
        const toFn = hit ? (out) => R[tg].getCenterWorld(out)
          : (out) => R[tg].getCenterWorld(out).add(new V3(0, R[tg].height * 0.45, side * (R[tg].radius + 4)));
        dir.shake(0.25);
        const life = act.nHits > 1 ? 0.35 : 0.5;
        if (WH?.twinBeam) { beamFx(a, 0, toFn, col, life, { w: 0.45 }); beamFx(a, 1, toFn, col, life, { w: 0.45 }); }
        else beamFx(a, act.nHits > 1 ? k : 0, toFn, col, life);
        if (WH?.beamSources) {
          const cs = WH.memberColors?.() || [];
          WH.beamSources().forEach((o, j) => beamFx(a, 0, (out) => R[a].getMouthWorld(0, out), cs[j] || col, life + 0.1, { fromFn: (out) => o.getWorldPosition(out), w: 0.28 }));
        }
        if (await W(90)) return;
        if (hit) impact(ev, i); else missFx(ev, i);
        if (await W(act.nHits > 1 ? 260 : 380)) return;
        r.anim.charge = act.hits < act.nHits ? 0.6 : 0;
        break;
      }
      case 'projectile': {
        const obj = makeProjectile(act.ab, a, ev);
        const id = String(act.ab || '');
        const from = new V3();
        let o = null;
        try { o = WH?.origin?.(act.ab, act.hits, act.nHits, from) || null; } catch { o = null; }
        if (!o) {
          const origin = id.includes('punch') ? (r.attach.handR || r.mouth) : (id.includes('missile') && r.attach.shoulderL ? (act.hits % 2 ? r.attach.shoulderL : r.attach.shoulderR) : r.mouth);
          (origin || r.mouth).getWorldPosition(from);
          if (id.includes('toss')) from.y += R[a].height * 0.4;
        } else {
          if (o.scale) obj.scale.setScalar(o.scale);
          muzzleFlash(from, o.flash ?? 1);
        }
        if (!WH?.noLunge) tw(120, k => { r.anim.lunge = -0.15 + 0.6 * k; });
        const missOff = new V3(0, 0, (Math.random() < 0.5 ? -1 : 1) * (R[tg].radius + 5));
        const toFn = hit ? (out) => R[tg].getCenterWorld(out) : (out) => { R[tg].getCenterWorld(out).add(missOff); out.y = gy(); return out; };
        const dist = from.distanceTo(centerOf(tg));
        const ms = o?.ms ?? (act.nHits > 2 ? 360 : id.includes('toss') ? 700 : 480);
        const arc = o ? dist * (o.arcK ?? 0.05) : id.includes('toss') ? dist * 0.35 : id.includes('missile') ? dist * 0.18 : id.includes('web') ? dist * 0.1 : dist * 0.05;
        // cosmetic salvo from the other guns (tank column, rifle squads)
        if (o?.extras) for (const p of o.extras) {
          muzzleFlash(p, (o.flash ?? 1) * 0.8);
          const ex = makeProjectile(act.ab, a, ev);
          if (o.scale) ex.scale.setScalar(o.scale);
          const jit = new V3((Math.random() - 0.5) * 3, (Math.random() - 0.5) * 3, (Math.random() - 0.5) * 3);
          travel(ex, p.clone(), (out) => toFn(out).add(jit), ms * (0.9 + Math.random() * 0.25), p.distanceTo(centerOf(tg)) * (o.arcK ?? 0.05))
            .then(() => { if (!alive()) return; const q = centerOf(tg).add(jit); burst(q, fxColor(a, ev), 10, 8, 0.4, 6); });
        }
        await travel(obj, from, toFn, ms, arc);
        if (!alive()) return;
        if (hit) impact(ev, i); else { missFx(ev, i, { hop: true }); const p = new V3(); toFn(p); arena()?.groundImpact(p, 0.8); }
        if (id.includes('toss') || id.includes('missile') || (o && (o.arcK ?? 0) > 0.2)) arena()?.groundImpact(centerOf(tg).setY(gy()), 0.6);
        if (!WH?.noLunge) tw(160, k => { r.anim.lunge = 0.45 * (1 - k); });
        if (await W(act.nHits > 2 ? 60 : 160)) return;
        break;
      }
      case 'aoe': {
        if (first) { if (await (WH?.strafe ? airStrike(a, tg) : aoeBlast(a, tg))) return; }
        if (hit) impact(ev, i); else missFx(ev, i, { hop: true });
        if (await W(first ? 260 : 140)) return;
        break;
      }
      case 'grab': {
        if (!hit) {
          if (await T(160, k => { r.anim.lunge = k; }, ease.in)) return;
          missFx(ev, i);
          if (await W(300)) return;
          break;
        }
        const tr = R[tg], t0 = F[tg].position.clone();
        const liftH = Math.max(3, R[a].height * 0.55);
        if (await T(160, k => { r.anim.lunge = 0.7 * k; r.anim.bite = k; }, ease.in)) return;
        tr.hit('#ff2a1a', 0.5);
        const held = t0.clone().addScaledVector(fwd[a], -R[a].front * 0.25);
        if (await T(340, k => { F[tg].position.lerpVectors(t0, held, k); tr.anim.lift = liftH * k / tr.scale; tr.anim.recoil = k; r.anim.lunge = 0.7 - 0.9 * k; }, ease.out)) return;
        const land = home[tg].clone().addScaledVector(fwd[a], 9);
        const l0 = tr.anim.lift;
        if (await T(480, k => {
          F[tg].position.lerpVectors(held, land, k);
          tr.anim.lift = l0 * (1 - k) + Math.sin(k * PI) * liftH * 0.6 / tr.scale;
          r.anim.lunge = -0.2 + 1.0 * k;
        }, ease.in)) return;
        tr.anim.lift = 0;
        impact(ev, i, { thrown: true });
        arena()?.groundImpact(F[tg].position, 2.2);
        dir.shake(1.0);
        tw(400, k => { tr.anim.recoil = 1 - k; });
        if (await W(420)) return;
        break;
      }
      case 'roar': {
        if (first) { const s = new THREE.Mesh(sphGeo, addMat(accent[a], 1.2, 0.3)); s.position.copy(centerOf(tg)); spawn(s, (t) => { s.scale.setScalar(1 + t * 20); s.material.opacity = 0.3 * (1 - t / 0.4); return t < 0.4; }); }
        if (hit) impact(ev, i); else missFx(ev, i);
        if (await W(260)) return;
        break;
      }
      default: { // melee (and fallbacks for heal/shield/buff kinds that somehow hit)
        if (await T(160, k => { r.anim.lunge = lerp(Math.min(0, r.anim.lunge), 1, k); r.anim.bite = k; }, ease.in)) return;
        if (hit) impact(ev, i); else missFx(ev, i);
        if (await W(140)) return;
        tw(220, k => { r.anim.lunge = 1 - 0.6 * k; r.anim.bite = 1 - k; });
        if (await W(170)) return;
      }
    }
  }

  async function onHazard(ev, i) {
    const hits = Array.isArray(ev.hits) ? ev.hits : [];
    const targets = hits.map(h => (F[h.tg] ? F[h.tg].position.clone().setY(gy()) : new V3()));
    dir.clearFocus();
    dir.shake(0.3);
    let ms = 900;
    try { ms = arena()?.hazard?.(ev.name, ev.type, targets) ?? 900; } catch (e) { console.warn('[playback] hazard visual failed', e); }
    floatText(new V3(0, Math.max(R[0].height, R[1].height) + 6, 0).add(home[0].clone().add(home[1]).multiplyScalar(0.5)).setY(Math.max(R[0].height, R[1].height) + 6 + gy()),
      String(ev.name || 'HAZARD').toUpperCase(), '#ffcf6a', 2.2, 1.8, 2);
    if (await W(ms)) return;
    fire(i);
    dir.shake(0.6);
    for (const h of hits) {
      if (!R[h.tg]) continue;
      const r = R[h.tg];
      r.hit(TYPE_COLORS[ev.type] || '#ff3a1a', 1);
      recoil(h.tg, 0.8);
      floatText(topOf(h.tg), h.dmg | 0, '#ffd0a0', 2.4);
      burst(centerOf(h.tg), TYPE_COLORS[ev.type] || '#ffffff', 25, 10, 0.5, 8);
    }
    await W(650);
  }

  async function onKO(ev, i) {
    const tg = ev.tg ?? 0;
    if (!R[tg]) { fire(i); return; }
    const r = R[tg];
    r.clearStatuses();
    dir.focus(centerOf(tg), 0.62, realSec(2400));
    r.hit('#ffffff', 0.6);
    if (await T(950, k => { r.anim.ko = k; }, ease.in)) return;
    fire(i);
    dir.shake(1.2);
    arena()?.groundImpact(F[tg].position, 2.4);
    arena()?.crumbleNear(F[tg].position.clone().addScaledVector(fwd[tg], -(R[tg].back + 8)), 12);
    arena()?.flash('#ffffff', 0.3, 300);
    if (await T(260, k => { r.anim.ko = 1 - Math.sin(k * PI) * 0.07; })) return;
    await W(600);
  }

  // ---------- passives ({ t:'passive', tg, id, name }); they can land mid-action, so no endAction here
  const passiveTitle = (ev) => String(ev.name || ev.id || 'Passive').toUpperCase();

  async function passiveLastStand(tg, ev, i) {
    const r = R[tg], w = WL[tg], an = r.anim;
    const canKneel = w && 'kneel' in w;
    const gold = new THREE.Color('#ffd23f');
    dir.focus(centerOf(tg), fsc(tg, 0.68), realSec(2400));
    r.hit('#ffffff', 0.7);
    // stagger down to one knee
    if (await T(520, k => { if (canKneel) w.kneel = k; else an.ko = 0.3 * k; an.recoil = Math.sin(k * PI) * 0.7; }, ease.out)) return;
    dir.shake(0.4);
    arena()?.groundImpact(F[tg].position, 0.9);
    floatText(topOf(tg).add(new V3(0, -1.5, 0)), '...', '#cfd8e0', 1.6, 0.8, 1);
    if (await W(480)) return;
    // ...and rise with a golden flare
    fire(i);
    const base = F[tg].position.clone(); base.y = gy();
    pillar(base, gold, r.height * 1.7, 1.1, Math.max(1.6, r.radius * 0.3));
    groundRings(base, gold, Math.max(8, r.radius * 2), 3, 0.8);
    burst(centerOf(tg), '#ffe27a', 90, 16, 0.9, -2);
    burst(centerOf(tg), '#ffffff', 30, 10, 0.5, 0);
    arena()?.flash?.('#ffd860', 0.18, 350);
    dir.shake(0.6);
    r.hit('#ffd23f', 1);
    floatText(topOf(tg), passiveTitle(ev), '#ffd23f', 2.3, 1.9, 3);
    if (w && 'flare' in w) { w.flare = 1; tw(1700, k => { w.flare = 1 - k; }, ease.in); }
    if (await T(460, k => { if (canKneel) w.kneel = 1 - k; else an.ko = 0.3 * (1 - k); an.lift = Math.sin(k * PI) * 1.4 / r.scale; an.roar = Math.sin(k * PI) * 0.6; }, ease.out)) return;
    an.lift = 0; an.roar = 0;
    await W(450);
  }

  async function passiveTagTeam(tg, ev, i) {
    const r = R[tg], w = WL[tg];
    const p = new V3();
    if (w?.frontWorld) w.frontWorld(p); else p.copy(centerOf(tg));
    dir.focus(p, fsc(tg, 0.75), realSec(2000));
    const col = accent[tg];
    if (w?.beginSwap) {
      w.beginSwap();
      let landed = false;
      if (await T(1000, k => {
        w.swapK = k;
        if (!landed && k >= 0.55) {
          landed = true;
          fire(i);
          const g = p.clone(); g.y = gy();
          pillar(g, col, r.height * 1.4, 0.8, 2.2);
          groundRings(g, col, 10, 2, 0.6);
          const fl = new THREE.Mesh(sphGeo, addMat('#ffffff', 2.6, 1)); fl.position.copy(p);
          spawn(fl, (t) => { const kk = t / 0.3; fl.scale.setScalar(1 + kk * 7); fl.material.opacity = 1 - kk; return kk < 1; });
          burst(p, col, 50, 14, 0.6, 4); burst(p, '#ffffff', 20, 10, 0.4, 0);
          dir.shake(0.5);
          r.hit('#ffffff', 0.8);
          floatText(topOf(tg), 'TAG IN!', '#ffffff', 2.4, 1.5, 3);
          floatText(topOf(tg).add(new V3(0, -1.6, 0)), passiveTitle(ev), col.getStyle(), 1.5, 1.6, 2.5);
        }
      }, ease.linear)) return;
      w.swapK = 1;
      await W(250);
    } else {
      fire(i);
      floatText(topOf(tg), passiveTitle(ev), col.getStyle(), 2.2, 1.6, 3);
      await buffFx(tg, col.getStyle());
    }
  }

  async function passiveSwarm(tg, ev, i) {
    const r = R[tg], w = WL[tg];
    fire(i);
    floatText(topOf(tg), 'RESISTED', '#ffe08a', 2.6, 1.6, 3);
    floatText(topOf(tg).add(new V3(0, -1.6, 0)), ev.resisted ? `${passiveTitle(ev)}: ${String(ev.resisted).toUpperCase()}` : passiveTitle(ev), '#ffcf8a', 1.3, 1.6, 2.5);
    r.hit('#ffe08a', 0.6);
    if (w?.reinforce) {
      w.reinforce();
      dir.focus(centerOf(tg).addScaledVector(fwd[tg], -6), 0.85, realSec(1500));
      if (w.strafe && w.strafeK < 0) {
        w.strafe(F[tg].position.clone().addScaledVector(fwd[tg], 4));
        tw(1500, k => { w.strafeK = k; }, ease.linear);
        wait(1520).then(() => { if (w.strafeK >= 0.999) w.strafeK = -1; });
      }
      const k0 = w.reinK || 0;
      if (await T(1100, k => { w.reinK = Math.max(k0, k); if (Math.random() < 0.5) burst(F[tg].position.clone().addScaledVector(fwd[tg], -8 + Math.random() * 4).setY(gy() + 0.5), '#9a8a70', 2, 3, 0.8, -1); }, ease.out)) return;
      await W(200);
    } else {
      groundRings(F[tg].position.clone().setY(gy()), accent[tg], Math.max(6, r.radius * 1.5), 2, 0.6);
      await W(650);
    }
  }

  async function passiveTransform(tg, ev, i) {
    const r = R[tg], w = WL[tg], an = r.anim;
    const cA = accent[tg].clone(), cB = new THREE.Color(fs[tg]?.colors?.body || '#ffffff'), cW = new THREE.Color('#ffffff');
    dir.focus(centerOf(tg), fsc(tg, 0.62), realSec(2800));
    // ribbon / light spiral while she rises and spins
    spiralFx(tg, [cA, cB], 1.9);
    const rot0 = homeRotY[tg];
    if (await T(1500, k => {
      an.lift = Math.sin(Math.min(1, k * 1.5) * PI / 2) * 2.4 / r.scale;
      an.roar = 0.5 * k;
      F[tg].rotation.y = rot0 + ease.inOut(k) * PI * 4;
      r.hit(cA.getStyle(), 0.25 + 0.3 * k);
    }, ease.linear)) return;
    F[tg].rotation.y = rot0;
    // burst of sparkles, then the brighter form for the rest of the fight
    fire(i);
    const c = centerOf(tg);
    burst(c, cA, 110, 20, 1.1, 1.5);
    burst(c, cB, 70, 15, 1.0, 1);
    burst(c, cW, 50, 12, 0.7, 0);
    const fl = new THREE.Mesh(sphGeo, addMat(cA, 1.6, 0.7)); fl.position.copy(c);
    spawn(fl, (t) => { const k = t / 0.4; fl.scale.setScalar(1 + k * r.height * 0.45); fl.material.opacity = 0.7 * (1 - k); return k < 1; });
    groundRings(F[tg].position.clone().setY(gy()), cA, Math.max(10, r.radius * 2.5), 3, 0.9);
    arena()?.flash?.(cA.getStyle(), 0.22, 400);
    dir.shake(0.7);
    r.hit('#ffffff', 1);
    floatText(topOf(tg), passiveTitle(ev) + '!', cA.getStyle(), 2.6, 2.0, 3);
    if (w?.setTransformed) tw(650, k => w.setTransformed(k), ease.out);
    else tw(650, k => setGlowBoost(tg, 1 + 1.2 * k), ease.out);
    if (await W(550)) return;
    await T(420, k => { an.lift = 2.4 / r.scale * (1 - k); an.roar = 0.5 * (1 - k); });
  }

  async function passiveGeneric(tg, ev, i) {
    fire(i);
    floatText(topOf(tg), passiveTitle(ev), accent[tg].getStyle(), 2.0, 1.6, 3);
    await buffFx(tg, accent[tg].getStyle());
  }

  async function onPassive(ev, i) {
    const tg = ev.tg ?? 0;
    if (!R[tg]) { fire(i); await W(80); return; }
    switch (ev.id) {
      case 'last_stand': return passiveLastStand(tg, ev, i);
      case 'tag_team': return passiveTagTeam(tg, ev, i);
      case 'swarm': return passiveSwarm(tg, ev, i);
      case 'transformation': return passiveTransform(tg, ev, i);
      default: return passiveGeneric(tg, ev, i);
    }
  }

  async function onEnd(ev, i) {
    fire(i);
    const w = ev.winner ?? record.winner;
    if (w === 0 || w === 1) {
      dir.focus(centerOf(w), fsc(w, 0.72), realSec(4000), w === 0 ? -0.45 : 0.45);
      dir.shake(0.3);
      await T(600, k => { R[w].anim.victory = k; });
    } else {
      dir.clearFocus();
    }
    await W(2200);
  }

  async function handle(ev, i) {
    switch (ev?.t) {
      case 'start': await onStart(i); break;
      case 'turn':
        await endAction(); if (!alive()) return;
        fire(i);
        if (R[ev.a]) dir.setBias(biasX(ev.a));
        await W(240);
        break;
      case 'skip': {
        await endAction(); if (!alive()) return;
        fire(i);
        const a = ev.a ?? 0;
        if (R[a]) {
          floatText(topOf(a), ev.reason === 'bind' ? 'BOUND!' : 'STUNNED!', statusColor(ev.reason === 'bind' ? 'bind' : 'stun'), 1.8);
          const an = R[a].anim;
          tw(700, k => { an.recoil = Math.sin(k * PI * 3) * 0.3; });
        }
        await W(750);
        break;
      }
      case 'use': await endAction(); if (!alive()) return; await onUse(ev, i); break;
      case 'hit': await onStrike(ev, i, true); break;
      case 'miss': await onStrike(ev, i, false); break;
      case 'status': {
        fire(i);
        const r = R[ev.tg];
        if (r) {
          r.setStatus(ev.st, true);
          floatText(topOf(ev.tg).add(new V3(0, -1.2, 0)), (STATUSES[ev.st]?.name || ev.st || '').toUpperCase(), statusColor(ev.st), 1.5, 1.3, 2.5);
          if (ev.st === 'shield') { r.anim.shieldPulse = 1.4; tw(500, k => { r.anim.shieldPulse = 1.4 * (1 - k); }); }
        }
        await W(320);
        break;
      }
      case 'expire': fire(i); R[ev.tg]?.setStatus(ev.st, false); await W(140); break;
      case 'dot': {
        await endAction(); if (!alive()) return;
        fire(i);
        const r = R[ev.tg];
        if (r) {
          const c = statusColor(ev.st);
          r.hit(c, 0.9);
          floatText(topOf(ev.tg), ev.dmg | 0, c, 2.1);
          const p = centerOf(ev.tg);
          for (let n = 0; n < 24; n++) emit(p.clone().add(new V3((Math.random() - 0.5) * r.width * 0.6, (Math.random() - 0.5) * r.height * 0.5, (Math.random() - 0.5) * r.width * 0.6)), new V3(0, 4 + Math.random() * 4, 0), c, 0.8, -1);
          recoil(ev.tg, 0.3);
        }
        await W(520);
        break;
      }
      case 'heal': {
        const tg = ev.tg ?? act?.a ?? 0;
        if (R[tg]) { if (act?.kind !== 'heal') await healFx(tg); else { const r = R[tg]; r.hit('#5aff8a', 0.6); } }
        if (!alive()) return;
        fire(i);
        if (R[tg]) floatText(topOf(tg), `+${ev.amt | 0}`, '#6aff8a', 2.4);
        await W(420);
        break;
      }
      case 'selfdmg': {
        fire(i);
        const a = ev.a ?? 0;
        if (R[a]) {
          R[a].hit('#ff6a2a', 0.9);
          recoil(a, 0.5);
          floatText(topOf(a), ev.dmg | 0, '#ff9a6a', 2.0);
          burst(centerOf(a), '#ff8a3a', 20, 8, 0.5, 6);
        }
        await W(450);
        break;
      }
      case 'passive': await onPassive(ev, i); break;
      case 'hazard': await endAction(); if (!alive()) return; await onHazard(ev, i); break;
      case 'ko': await endAction(); if (!alive()) return; await onKO(ev, i); break;
      case 'end': await endAction(); if (!alive()) return; await onEnd(ev, i); break;
      default: fire(i); await W(80);
    }
  }

  // ---------- final pose (skip / completion)
  function applyFinal() {
    for (const e of [...fx]) kill(e);
    if (act?.orb) kill(act.orb);
    act = null;
    const koed = new Set(events.filter(e => e?.t === 'ko').map(e => e.tg));
    const st = [new Set(), new Set()];
    for (const e of events) {
      if (e?.t === 'status' && st[e.tg]) st[e.tg].add(e.st);
      if (e?.t === 'expire' && st[e.tg]) st[e.tg].delete(e.st);
    }
    const endEv = events.find(e => e?.t === 'end');
    const w = endEv ? (endEv.winner ?? record.winner) : null;
    // persistent passive states: transformed glow, re-formed team, reinforcements that rolled in
    for (let k = 0; k < 2; k++) resetWild(k);
    for (const e of events) {
      if (e?.t !== 'passive' || !R[e.tg]) continue;
      const w = WL[e.tg];
      try {
        if (e.id === 'transformation') { if (w?.setTransformed) w.setTransformed(1); else setGlowBoost(e.tg, 2.2); }
        else if (e.id === 'tag_team' && w?.beginSwap) { w.beginSwap(); w.swapK = 1; }
        else if (e.id === 'swarm' && w?.reinforce) w.reinforce(true);
      } catch (err) { console.warn('[playback] passive final state failed', err); }
    }
    for (let k = 0; k < 2; k++) {
      resetAnim(R[k]);
      F[k].position.copy(home[k]);
      R[k].clearStatuses();
      if (!koed.has(k)) st[k].forEach(s => R[k].setStatus(s, true));
      R[k].anim.ko = koed.has(k) ? 1 : 0;
      R[k].anim.victory = endEv && w === k ? 1 : 0;
    }
    if (w === 0 || w === 1) dir.focus(centerOf(w), fsc(w, 0.72), 4, w === 0 ? -0.45 : 0.45);
    else dir.clearFocus();
  }

  function flushScheduling() {
    const ts = timers, tws = tweens;
    timers = []; tweens = [];
    ts.forEach(x => x.res());
    tws.forEach(x => x.res());
  }

  function finish() {
    if (done) return;
    done = true;
    try { onDone(); } catch (e) { console.error('[playback] onDone error', e); }
    // release per-playback resources once trailing particles have faded; the final pose stays
    setTimeout(() => { if (!stopped) { stopped = true; cleanup(); } }, 3000);
  }

  // ---------- run
  (async () => {
    await Promise.resolve();
    for (let i = 0; i < N; i++) {
      if (!alive()) return;
      try { await handle(events[i], i); } catch (e) { console.error('[playback] event error', events[i], e); }
      if (!alive()) return;
      fire(i);
    }
    if (!alive()) return;
    finish();
  })();

  const cleanup = () => {
    removeTicker();
    flushScheduling();
    for (const e of [...fx]) kill(e);
    scene.remove(points);
    pGeo.dispose(); pMat.dispose();
    sharedGeos.forEach(g => g.dispose());
    if (stage.arena) stage.arena.timeScale = 1;
  };

  return {
    pause() { paused = true; },
    resume() { paused = false; },
    setSpeed(x) { speed = clampSpeed(x); },
    get speed() { return speed; },
    get paused() { return paused; },
    get done() { return done; },
    skipToEnd() {
      if (done || stopped || skipping) return;
      skipping = true;
      flushScheduling();
      for (let i = 0; i < N; i++) fire(i);
      applyFinal();
      // keep idle animation running (ticker stays for particles) but stop scheduling
      finish();
    },
    stop() {
      if (stopped) return;
      stopped = true;
      cleanup();
    },
  };
}
