// Rare Wild enemy body plans: hero, heroteam, army, magicalgirl.
//
// Imported once after the stage loads (`import('../render/wild-models.js')`). At module load it registers four
// plan builders on kaiju-model.js via registerPlanBuilder, so buildKaijuModel() returns the usual rig
// (root/motion/body/head/mouth/limbs/tail/height/attach/update/hit/setStatus/...) for these plans too.
//
// Each builder fills the BuildCtx like the core plans do, plus:
//  - rig.extra.wild: per-plan hooks the playback uses (reset, setTarget, onHit, onStrike, origin, strafe,
//    kneel/flare, beginSwap/swapK, reinforce/reinK, tf). Playback treats every hook as optional.
//  - a per-frame tick that runs after the stock rig.update (installed on a microtask, once the model's rig
//    exists), used for capes, turrets, jets, formation moves, sparkles and transformation glow.
//  - "late" decorative objects (aura shells, sparkles, smoke, jets, reserves, cape) that are added after the
//    model is measured, so they don't inflate height/radius and aren't used as surfaces for injury scars.
// Stat tiers get small per-plan flourishes; the generic monster upgrades (horns, crystals, fins, eyes) and
// ability cosmetics are switched off for these humanoid / vehicle plans.

import * as THREE from 'three';
import { registerPlanBuilder, HP_SIZE } from './kaiju-model.js';
import { ABILITY_BY_ID } from '../data/abilities.js';

const V3 = THREE.Vector3;
const v = (x = 0, y = 0, z = 0) => new V3(x, y, z);
const PI = Math.PI;
const lerp = (a, b, t) => a + (b - a) * t;
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const smooth = (k) => { k = clamp01(k); return k * k * (3 - 2 * k); };
const std = (o) => new THREE.MeshStandardMaterial(o);
function color(c, fb) { const k = new THREE.Color(); try { k.set(c || fb); } catch { k.set(fb); } return k; }
function glowMat(col, intensity = 1, extra = {}) {
  const m = std({ color: col, emissive: col, emissiveIntensity: intensity, roughness: 0.35, ...extra });
  m.userData.glow = true;
  return m;
}
function addMat(col, opacity = 1) {
  return new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false, fog: false });
}

// ---------------------------------------------------------------- textures (one set per model, disposed with it)
function canvasTex(draw, size = 64) {
  const cv = document.createElement('canvas'); cv.width = cv.height = size;
  draw(cv.getContext('2d'), size);
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
const glowTex = () => canvasTex((g, s) => {
  const gr = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.3, 'rgba(255,255,255,0.6)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = gr; g.fillRect(0, 0, s, s);
});
const haloTex = () => canvasTex((g, s) => {
  const gr = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  gr.addColorStop(0, 'rgba(255,255,255,0.05)'); gr.addColorStop(0.5, 'rgba(255,255,255,0.22)'); gr.addColorStop(0.78, 'rgba(255,255,255,0.5)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = gr; g.fillRect(0, 0, s, s);
}, 128);
const starTex = () => canvasTex((g, s) => {
  const c = s / 2;
  const gr = g.createRadialGradient(c, c, 0, c, c, c * 0.5);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = gr; g.fillRect(0, 0, s, s);
  g.fillStyle = 'rgba(255,255,255,0.95)';
  g.beginPath();
  for (let i = 0; i < 8; i++) {
    const a = i * PI / 4, r = i % 2 ? c * 0.16 : c * 0.98;
    g.lineTo(c + Math.cos(a) * r, c + Math.sin(a) * r);
  }
  g.closePath(); g.fill();
}, 64);
const smokeTex = () => canvasTex((g, s) => {
  for (let i = 0; i < 7; i++) {
    const x = s * (0.3 + 0.4 * ((i * 37) % 10) / 10), y = s * (0.3 + 0.4 * ((i * 53) % 10) / 10), r = s * (0.22 + 0.1 * (i % 3));
    const gr = g.createRadialGradient(x, y, 0, x, y, r);
    gr.addColorStop(0, 'rgba(255,255,255,0.55)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = gr; g.fillRect(0, 0, s, s);
  }
});
function sprite(tex, col, size, { additive = true, opacity = 1, fog = false } = {}) {
  const m = new THREE.SpriteMaterial({ map: tex, color: col, transparent: true, opacity, depthWrite: false, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending, fog });
  m.userData.ownMap = true;
  const s = new THREE.Sprite(m); s.scale.setScalar(size); s.userData.fx = true;
  return s;
}
// ---------------------------------------------------------------- shared shapes
function starShape(r, ri, n = 5) {
  const s = new THREE.Shape();
  for (let i = 0; i < n * 2; i++) {
    const a = i / (n * 2) * PI * 2 + PI / 2, rr = i % 2 ? ri : r;
    if (i) s.lineTo(Math.cos(a) * rr, Math.sin(a) * rr); else s.moveTo(Math.cos(a) * rr, Math.sin(a) * rr);
  }
  s.closePath();
  return s;
}
function extrude(shape, depth, bevel = 0.03) {
  const g = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel, bevelSegments: 1 });
  g.translate(0, 0, -depth / 2);
  return g;
}
function chevronShape(w = 1, h = 0.55, t = 0.28) {
  const s = new THREE.Shape();
  s.moveTo(-w, -h * 0.4); s.lineTo(0, h * 0.6); s.lineTo(w, -h * 0.4); s.lineTo(w, -h * 0.4 - t); s.lineTo(0, h * 0.6 - t * 1.25); s.lineTo(-w, -h * 0.4 - t);
  s.closePath();
  return s;
}

// ---------------------------------------------------------------- build plumbing
function prep(c) {
  const T = { ...c.T };
  // generic monster upgrades (horns, crystals, fins, extra eyes) don't suit these plans: keep hp size only
  c.T = { hp: T.hp, pow: 0, enr: 0, arm: 0, spd: 0, fcs: 0 };
  c.abilities = (c.abilities || []).filter(id => !ABILITY_BY_ID[id]?.vis);
  return { T, late: [] };
}

function setHead(c, hp, mouths) {
  hp.userData.rest = hp.rotation.clone();
  hp.userData.ph = c.rig.heads.length * 1.7;
  c.rig.heads.push(hp);
  c.rig.headPivot = hp; c.rig.jaw = null;
  for (const m of mouths) c.rig.mouths.push(m);
  c.rig.mouth = mouths[0];
}
function obj3(parent, pos) { const o = new THREE.Object3D(); if (pos) o.position.copy(pos); parent.add(o); return o; }

// Installs the per-frame tick + late objects once buildKaijuModel has finished and attached userData.rig.
function runtime(c, late, api, tick) {
  c.rig.extra.wild = api;
  const root = c.root;
  queueMicrotask(() => {
    const rig = root.parent?.userData?.rig;
    if (!rig || rig.motion !== root || rig.fallen) return;
    api.rig = rig;
    for (const [o, parent] of late) { o.traverse(x => { x.userData.fx = true; }); (parent === 'model' ? root.parent : (parent || root)).add(o); }
    const base = rig.update;
    let warned = false;
    rig.update = (dt, t) => {
      base(dt, t);
      try { tick(rig, dt, t); } catch (e) { if (!warned) { warned = true; console.warn('[wild-models] tick failed', rig.plan, e); } }
    };
  });
}

// Generic humanoid (≈9.1 units tall at scale 1, facing +Z). Returns its groups for per-plan dressing.
function person(c, parent, P = {}) {
  const M = P.mats;
  const tl = P.torsoLen ?? 1, ll = P.legLen ?? 1;
  const cw = P.chestW ?? 1.3, ww = P.waistW ?? 0.78, hw = P.hipW ?? 0.86;
  const ar = P.armR ?? 0.34, lr = P.legR ?? 0.42, hr = P.headR ?? 0.5;
  const hipY = 4.85 * ll;
  const trunk = c.g(parent, v(0, hipY, 0), P.lean ? [P.lean, 0, 0] : null);
  const torso = c.g(trunk);
  const E = [[0, 0.12, 0, hw, 0.6, 0.6], [0, 0.95 * tl, 0.02, ww, 0.9 * tl, 0.55], [0, 2.0 * tl, 0.04, cw, 0.95 * tl, 0.72]];
  c.ell(torso, hw, 0.6, 0.6, M.lower || M.suit, v(0, 0.12, 0));
  c.ell(torso, ww, 0.9 * tl, 0.55, M.suit, v(0, 0.95 * tl, 0.02));
  c.ell(torso, cw, 0.95 * tl, 0.72, M.chest || M.suit, v(0, 2.0 * tl, 0.04));
  if (P.pecs !== false) for (const s of [-1, 1]) c.ell(torso, cw * 0.42, 0.4 * tl, 0.3, M.chest || M.suit, v(s * cw * 0.38, 2.12 * tl, 0.5));
  const neckY = 2.82 * tl;
  const nw = P.neckW ?? 1;
  c.ell(trunk, 0.3 * nw, 0.45, 0.3 * nw, M.skin, v(0, neckY + 0.12, 0));
  const hp = c.g(trunk, v(0, neckY + 0.32, 0.03));
  if (P.head !== false) c.ell(hp, hr * 0.9, hr * 1.1, hr * 0.98, M.skin, v(0, hr * 1.02, 0.03));
  const shY = 2.48 * tl, shX = cw * 0.92 + ar * 0.35;
  const arms = [];
  for (const s of [-1, 1]) {
    const rest = (P.armRest && P.armRest[s < 0 ? 0 : 1]) || {};
    const sh = c.g(trunk, v(s * shX, shY, 0), [rest.x ?? 0.05, 0, s * (rest.z ?? 0.1)]);
    c.ell(sh, ar * 1.3, ar * 1.2, ar * 1.25, M.shoulder || M.suit, v(s * 0.04, 0.02, 0));
    c.ell(sh, ar, 0.92, ar * 1.05, M.arm || M.suit, v(0, -0.85, 0));
    const el = c.g(sh, v(0, -1.72, 0), [rest.el ?? -0.22, 0, 0]);
    c.ell(el, ar * 0.9, 0.84, ar * 0.94, M.glove || M.arm || M.suit, v(0, -0.8, 0));
    const hand = c.g(el, v(0, -1.68, 0));
    c.ell(hand, ar * 0.95, ar * 1.08, ar * 0.95, M.hand || M.glove || M.skin, v(0, 0, 0.03));
    arms.push({ s, sh, el, hand });
    if (P.armLimbs !== false) c.limb({ type: 'arm', pivot: sh, elbow: el, phase: s + (P.ph || 0), reach: rest.reach ?? -1.25, elbowReach: rest.elReach ?? 0.5, vicRaise: rest.vic ?? (s > 0 ? -2.7 : -0.5) });
  }
  const legs = [];
  for (const s of [-1, 1]) {
    const rest = (P.legRest && P.legRest[s < 0 ? 0 : 1]) || {};
    const hip = c.g(trunk, v(s * hw * 0.52, 0, 0), [rest.x ?? 0, 0, s * (rest.z ?? 0.02)]);
    c.ell(hip, lr, 1.3 * ll, lr * 1.1, M.leg || M.suit, v(0, -1.18 * ll, 0));
    const knee = c.g(hip, v(0, -2.38 * ll, 0), [rest.knee ?? 0, 0, 0]);
    c.ell(knee, lr * 0.8, 1.12 * ll, lr * 0.88, M.boot || M.leg || M.suit, v(0, -1.05 * ll, 0));
    const ank = c.g(knee, v(0, -2.14 * ll, 0));
    c.ell(ank, lr * 0.78, 0.27, lr * 1.45, M.boot || M.leg || M.suit, v(0, -0.06, 0.2));
    legs.push({ s, hip, knee, ank });
    if (P.legLimbs !== false) c.limb({ type: 'leg', pivot: hip, knee, phase: (s > 0 ? 0 : PI) + (P.ph || 0) });
  }
  const surfZ = (x, y) => {
    let best = -Infinity;
    for (const [cx, cy, cz, rx, ry, rz] of E) { const q = 1 - ((x - cx) / rx) ** 2 - ((y - cy) / ry) ** 2; best = Math.max(best, cz + rz * Math.sqrt(Math.max(0, q))); }
    return best;
  };
  return { trunk, torso, hp, arms, legs, hipY, neckY, shY, shX, hr, tl, ll, cw, ww, hw, ar, lr, surfZ };
}

function bow(c, parent, pos, size, mat, o = {}) {
  const g = c.g(parent, pos, o.rot);
  for (const s of [-1, 1]) {
    c.ell(g, size * 0.55, size * 0.34, size * 0.2, mat, v(s * size * 0.5, size * 0.08, 0), [0, 0, s * 0.35]);
    if (o.tails !== false) {
      const L = o.tailLen ?? 1;
      c.tube(g, [v(s * size * 0.08, -size * 0.05, 0.02), v(s * size * 0.32, -size * 0.55 * L, 0.06), v(s * size * 0.45, -size * L, 0)], size * 0.14, size * 0.07, mat, { seg: 8, rad: 6 });
    }
  }
  c.ell(g, size * 0.2, size * 0.2, size * 0.18, o.knot || mat, v(0, 0, 0.04));
  return g;
}

// Animated cloth strip (cape / back ribbon). A PlaneGeometry grid re-posed every frame.
function makeCloth(mat, cols, rows) {
  const geo = new THREE.PlaneGeometry(1, 1, cols, rows);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true; mesh.frustumCulled = false;
  return { mesh, geo, cols, rows };
}
function poseCloth(cl, o) {
  const { geo, cols, rows } = cl;
  const p = geo.attributes.position;
  const ds = o.L / rows;
  for (let ix = 0; ix <= cols; ix++) {
    const u = ix / cols * 2 - 1;
    let y = o.top.y, z = o.top.z + (o.wrap ?? 0.25) * u * u;
    for (let iy = 0; iy <= rows; iy++) {
      const vv = iy / rows;
      if (iy) {
        const ang = o.ang * Math.pow(vv, 0.8) + Math.sin(o.t * 3.0 - vv * 5.5 + u * 1.6) * (0.06 + 0.18 * o.flow) * vv;
        y -= ds * Math.cos(ang); z -= ds * Math.sin(ang);
      }
      const x = o.top.x + u * lerp(o.w0, o.w1, vv) * 0.5 + Math.sin(o.t * 1.7 + vv * 3.0) * 0.1 * vv * (1 + o.flow);
      const zc = z - (o.clear ?? 0.25) * (1 - u * u) * (0.35 + 0.65 * Math.sin(Math.min(1, vv * 1.6) * PI * 0.5)) + Math.sin(o.t * 2.3 + u * 2.2 + vv * 4) * 0.12 * vv * o.flow;
      p.setXYZ(iy * (cols + 1) + ix, x, y, zc);
    }
  }
  p.needsUpdate = true;
  geo.computeVertexNormals();
  geo.computeBoundingSphere();
}


// ---------------------------------------------------------------- tiny (human-scale) fighters
// Hero, team and magical girl are ~1 unit tall next to ~10-unit kaiju. These keep them findable on screen:
// a soft ground spotlight + ring, a beacon glow around the body and a motion trail that shows when they dart.
const TINY_H = 0.92;
const tinyScale = (T, baseH) => (TINY_H / baseH) * (1 + 0.04 * T.hp) / HP_SIZE[T.hp]; // cancels the kaiju HP sizing
function tinyFx(c, late, accC, o = {}) {
  c.rig.extra.tiny = true;
  const gTex = glowTex();
  const disc = new THREE.Mesh(new THREE.PlaneGeometry(1, 1).rotateX(-PI / 2), new THREE.MeshBasicMaterial({ map: gTex, color: accC, transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  const dr = o.disc ?? 3.4;
  disc.scale.set(dr, 1, dr); disc.position.y = 0.05; disc.renderOrder = 3;
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 1, 56).rotateX(-PI / 2), addMat(accC, 0.75));
  ring.position.y = 0.07; ring.renderOrder = 3;
  late.push([disc, 'model'], [ring, 'model']);
  const beacon = sprite(gTex, accC, o.beacon ?? 3.4, { opacity: 0.35 });
  beacon.position.y = o.centerY ?? 0.5;
  late.push([beacon, null]);
  const N = o.trail === false ? 0 : 16, hist = [], trail = [];
  for (let i = 0; i < N; i++) { const s = sprite(gTex, accC.clone().lerp(new THREE.Color('#ffffff'), 0.3), 0.55, { opacity: 0 }); trail.push(s); late.push([s, 'model']); }
  const p = v();
  return (rig, dt, t, boost = 0) => {
    const a = rig.anim, up = 1 - a.ko * 0.8, m = rig.motion, near = rig.bankSign ? 1 : 0.45; // dimmer up close in showcase
    const rr = (o.ring ?? 1.3) * (1 + 0.08 * Math.sin(t * 4) + boost * 0.4);
    ring.scale.setScalar(rr); ring.position.x = m.position.x; ring.position.z = m.position.z;
    ring.material.opacity = ((0.55 + 0.25 * Math.sin(t * 4)) * up + boost * 0.3) * near;
    disc.position.x = m.position.x; disc.position.z = m.position.z;
    disc.material.opacity = (0.36 + 0.1 * Math.sin(t * 2.3) + a.charge * 0.4 + boost * 0.4) * up * near;
    beacon.material.opacity = Math.min(1, (0.3 + 0.1 * Math.sin(t * 3) + a.charge * 0.5 + boost * 0.4) * up) * (rig.bankSign ? 1 : 0.35);
    beacon.scale.setScalar((o.beacon ?? 3.4) * (1 + a.charge * 0.4 + boost * 0.5));
    if (!N) return;
    p.set(m.position.x, m.position.y + (o.centerY ?? 0.5), m.position.z);
    hist.unshift(p.clone()); if (hist.length > N + 1) hist.length = N + 1;
    for (let i = 0; i < N; i++) {
      const s = trail[i], q = hist[i], q2 = hist[i + 1];
      if (!q || !q2) { s.material.opacity = 0; continue; }
      s.position.copy(q);
      const spd = q.distanceTo(q2) / Math.max(dt, 1e-3);
      s.material.opacity = clamp01((spd - 1.5) / 8) * (1 - i / N) * 0.85 * up;
      s.scale.setScalar(0.55 * (1 - i / N * 0.6));
    }
  };
}
// Zippy idle: hold a spot, then dart quickly to the next one.
function hash1(n) { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return (s - Math.floor(s)) * 2 - 1; }
function dart(out, t, seed, amp = 1, period = 1.15) {
  const f = t / period + seed, n = Math.floor(f), k = smooth((f - n) * 3.5);
  const ax = lerp(hash1(n - 1), hash1(n), k), ay = lerp(hash1(n + 39), hash1(n + 40), k), az = lerp(hash1(n + 89), hash1(n + 90), k);
  return out.set(ax * 0.8 * amp + Math.sin(t * 5.1) * 0.05, (ay * 0.5 + 0.2) * amp + Math.sin(t * 3.3) * 0.06, az * 0.7 * amp);
}
function rimGlow(mats, accC, k = 0.12) { for (const m of mats) { m.emissive.copy(accC).multiplyScalar(k); m.emissiveIntensity = 1; } }

// ================================================================ HERO
function buildHero(c) {
  const { T, late } = prep(c);
  const body = color(c.colors.body, '#1d3fa8'), belly = color(c.colors.belly, '#c8102e'), acc = color(c.colors.accent, '#ffd23f');
  c.rig.hover = 0.9; c.rig.koMode = 'back';
  const suit = c.M.body; suit.roughness = 0.42; suit.metalness = 0.12;
  const capeMat = std({ color: belly, roughness: 0.62, side: THREE.DoubleSide });
  const bootMat = std({ color: body.clone().multiplyScalar(0.45), roughness: 0.4, metalness: 0.2 });
  const skin = std({ color: '#d6a07c', roughness: 0.62 });
  const hair = std({ color: '#201813', roughness: 0.5 });
  const trim = glowMat(acc, 0.35, { metalness: 0.6, roughness: 0.3 });
  const eyeMat = glowMat(acc, 0.5, { color: '#ffffff' });
  const figS = tinyScale(T, 9.1);
  const fig = c.g(c.root); fig.scale.setScalar(figS);
  const P = person(c, fig, {
    mats: { suit, lower: suit, skin, glove: capeMat, boot: bootMat },
    chestW: 1.45, waistW: 0.8, hipW: 0.84, armR: 0.36, legR: 0.44, headR: 0.5,
    armRest: [{ x: 0.12, z: 0.16, el: -0.3, vic: -0.3 }, { x: 0.05, z: 0.12, el: -0.25, vic: -2.9 }],
    legRest: [{ x: -0.14, knee: 0.28 }, { x: 0.12, knee: 0.42 }],
  });
  c.rig.trunk = P.trunk; c.rig.body = P.torso;
  const { hp, hr, torso } = P;
  // face
  c.ell(hp, hr * 0.64, hr * 0.42, hr * 0.62, skin, v(0, hr * 0.42, hr * 0.22));
  c.ell(hp, hr * 0.11, hr * 0.16, hr * 0.12, skin, v(0, hr * 0.95, hr * 1.0));
  const mouths = [];
  for (const s of [-1, 1]) {
    c.ell(hp, hr * 0.14, hr * 0.08, hr * 0.06, eyeMat, v(s * hr * 0.34, hr * 1.12, hr * 0.92));
    c.ell(hp, hr * 0.21, hr * 0.06, hr * 0.09, hair, v(s * hr * 0.34, hr * 1.28, hr * 0.92), [0, 0, s * -0.18]);
    c.ell(hp, hr * 0.12, hr * 0.2, hr * 0.1, skin, v(s * hr * 0.88, hr * 1.02, 0));
    mouths.push(obj3(hp, v(s * hr * 0.34, hr * 1.12, hr * 1.02)));
  }
  c.ell(hp, hr * 0.96, hr * 0.62, hr * 1.02, hair, v(0, hr * 1.48, -hr * 0.1));
  c.ell(hp, hr * 0.62, hr * 0.26, hr * 0.5, hair, v(0, hr * 1.86, hr * 0.38), [-0.5, 0, 0]);
  setHead(c, hp, mouths);
  // emblem: a gold star in a ring on the chest
  const ey = 2.15 * P.tl, ez = P.surfZ(0, ey);
  const em = c.g(torso, v(0, ey, ez + 0.02), [-0.12, 0, 0]);
  c.m(new THREE.CylinderGeometry(0.62, 0.62, 0.08, 32).rotateX(PI / 2), capeMat, em);
  c.m(new THREE.TorusGeometry(0.62, 0.06, 8, 32), trim, em);
  c.m(extrude(starShape(0.5, 0.22), 0.1, 0.03), c.M.accent, em, v(0, 0, 0.06));
  c.mount('chest', torso, v(0, ey, ez + 0.1), v(0, 0.1, 1));
  // gold piping from the shoulders down to the belt
  for (const s of [-1, 1]) {
    const pts = [];
    for (let i = 0; i <= 8; i++) { const k = i / 8, x = s * lerp(1.0, 0.32, k), y = lerp(2.6, 0.55, k) * P.tl; pts.push(v(x, y, P.surfZ(x, y) + 0.02)); }
    c.tube(torso, pts, 0.05, 0.05, trim, { seg: 16, rad: 5, cap: false });
  }
  const belt = new THREE.TorusGeometry(1, 0.09, 8, 36).rotateX(PI / 2); belt.scale(0.8, 1, 0.58);
  c.m(belt, trim, torso, v(0, 0.55, 0.02));
  c.rbox(torso, 0.42, 0.3, 0.12, 0.05, trim, v(0, 0.55, P.surfZ(0, 0.55) + 0.04));
  // gauntlet cuffs, boot tops, cape clasps
  for (const A of P.arms) c.m(new THREE.CylinderGeometry(P.ar * 1.15, P.ar * 0.95, 0.4, 14), capeMat, A.el, v(0, -0.25, 0));
  for (const L of P.legs) c.m(new THREE.CylinderGeometry(P.lr * 0.98, P.lr * 0.85, 0.32, 14), bootMat, L.knee, v(0, -0.12, 0));
  for (const s of [-1, 1]) c.m(new THREE.CylinderGeometry(0.2, 0.2, 0.08, 16).rotateX(PI / 2), trim, torso, v(s * 0.95, 2.62 * P.tl, P.surfZ(s * 0.95, 2.62 * P.tl) + 0.03));
  // tier flourishes (modest)
  const fistK = 1 + 0.07 * T.pow;
  for (const A of P.arms) A.hand.scale.setScalar(fistK);
  if (T.arm >= 1) for (const A of P.arms) {
    const pd = c.ell(A.sh, P.ar * 1.55, P.ar * 0.55, P.ar * 1.45, T.arm >= 2 ? trim : suit, v(A.s * 0.08, 0.3, 0), [0, 0, A.s * -0.35]);
    pd.scale.multiplyScalar(1 + 0.1 * (T.arm - 1));
  }
  if (T.arm >= 3) for (const L of P.legs) c.ell(L.knee, P.lr * 0.7, P.lr * 0.6, P.lr * 0.45, trim, v(0, 0.05, P.lr * 0.7));
  if (T.enr >= 2) for (const A of P.arms) c.tube(A.el, [v(0, -0.45, P.ar * 0.95), v(0, -1.25, P.ar * 0.9)], 0.05, 0.05, c.M.accent, { seg: 4, rad: 5 });
  c.M.accent.emissiveIntensity *= 1 + 0.12 * T.enr;
  if (T.spd >= 2) for (const L of P.legs) {
    const s = new THREE.Shape(); s.moveTo(0, 0); s.quadraticCurveTo(-0.4, 0.4, -1.0, 0.55); s.quadraticCurveTo(-0.55, 0.15, -0.6, -0.1); s.lineTo(0, 0);
    const wg = new THREE.ShapeGeometry(s); wg.rotateY(PI / 2);
    c.m(wg, std({ color: acc, emissive: acc, emissiveIntensity: 0.2, side: THREE.DoubleSide }), L.knee, v(L.s * P.lr * 0.85, -1.4, -0.1));
  }
  if (T.fcs >= 2) {
    const band = new THREE.TorusGeometry(hr * 0.93, 0.045, 6, 32).rotateX(PI / 2);
    c.m(band, trim, hp, v(0, hr * 1.42, 0.02), [0.12, 0, 0]);
    if (T.fcs >= 3) c.ell(hp, 0.09, 0.11, 0.06, c.M.accent, v(0, hr * 1.45, hr * 0.98));
  }
  // mounts / attach points
  c.mount('back', torso, v(0, 2.2 * P.tl, -0.75), v(0, 0.1, -1));
  c.mount('crest', hp, v(0, hr * 2.05, 0), v(0, 1, 0), v(0, 0, -1));
  for (const A of P.arms) c.mount('shoulders', A.sh, v(A.s * 0.1, P.ar * 1.1, 0), v(A.s * 0.3, 1, 0));
  c.rig.extra.handL = P.arms[0].hand; c.rig.extra.handR = P.arms[1].hand;

  // late: cape, aura shell, hover glow, motes
  const cape = makeCloth(capeMat, 10, 16);
  late.push([cape.mesh, P.trunk]);
  const capeL = 6.0 + 0.35 * T.spd;
  const aura = sprite(haloTex(), acc, 1, { opacity: 0.5 });
  aura.position.set(0, 4.9, 0); aura.renderOrder = 4;
  late.push([aura, fig]);
  const tex = glowTex();
  const footGlow = sprite(tex, acc, 3.2, { opacity: 0.55 });
  footGlow.position.set(0, -0.2, 0.1);
  late.push([footGlow, fig]);
  const motes = [];
  for (let i = 0; i < 10; i++) { const s = sprite(tex, acc, 0.35, { opacity: 0.8 }); motes.push(s); late.push([s, fig]); }
  const gold = new THREE.Color('#fff0b0');
  rimGlow([suit, capeMat, bootMat], acc, 0.1);
  const tfx = tinyFx(c, late, acc, { centerY: 0.5, beacon: 3.2 });
  const dv = v();

  const api = {
    kneel: 0, flare: 0, twinBeam: true,
    reset() { this.kneel = 0; this.flare = 0; },
  };
  const tmpC = new THREE.Color();
  runtime(c, late, api, (rig, dt, t) => {
    const a = rig.anim, up = 1 - a.ko;
    const kn = api.kneel, fl = api.flare;
    const calm = up * (1 - kn) * (1 - Math.min(1, Math.abs(a.lunge) * 3)) * (1 - a.victory * 0.6) * (1 - a.charge * 0.7);
    dart(dv, t, 0.3, rig.bankSign ? 0.9 : 0.25);
    rig.motion.position.x += (dv.x - (rig.bankSign || 0) * 0.5) * calm; rig.motion.position.y += dv.y * calm; rig.motion.position.z += dv.z * calm;
    rig.motion.position.y += Math.max(0, a.lunge) * 3.4 * (1 - kn); // fly up to strike the kaiju's chest
    const flow = clamp01(a.lunge * 0.9 + a.walk * 0.6 + a.victory * 0.35 + Math.abs(a.dodge) * 0.2 + dv.length() * 0.3 * calm) * (1 - kn * 0.8);
    poseCloth(cape, { top: v(0, P.shY + 0.12, -0.7), wrap: 0.38, w0: 2.6, w1: 3.8, L: capeL, ang: 0.32 + flow * 1.05 + 0.05 * Math.sin(t * 1.2) - a.ko * 0.2, flow, t, clear: 0.3 });
    // last-stand kneel: right knee up, left knee down, body sinks and bows
    if (kn > 0.001) {
      const [Ll, Lr] = P.legs;
      Lr.hip.rotation.x -= 1.45 * kn; Lr.knee.rotation.x += 1.5 * kn;
      Ll.hip.rotation.x += 0.25 * kn; Ll.knee.rotation.x += 1.85 * kn;
      rig.motion.position.y -= (rig.hover + 2.25 * figS) * kn;
      P.trunk.rotation.x = 0.32 * kn;
      hp.rotation.x += 0.35 * kn;
      P.arms[1].sh.rotation.x -= 0.7 * kn; P.arms[1].el.rotation.x -= 0.6 * kn;
      P.arms[0].sh.rotation.x += 0.2 * kn;
    } else P.trunk.rotation.x = 0;
    // aura
    aura.material.opacity = Math.min(1, (0.42 + 0.1 * T.enr) * (0.85 + 0.15 * Math.sin(t * 2.4)) * (1 + a.charge * 0.8 + a.roar * 0.4 + fl * 1.4 + a.victory * 0.3)) * up * (1 - kn * 0.5);
    aura.material.color.copy(acc).lerp(gold, fl);
    aura.scale.set(6.4 * (1 + fl * 0.5), 12.5 * (1 + fl * 0.25), 1);
    tfx(rig, dt, t, fl);
    footGlow.material.opacity = (0.5 + 0.15 * Math.sin(t * 3)) * up * (1 - kn) + fl * 0.6;
    footGlow.material.color.copy(tmpC.copy(acc).lerp(gold, fl));
    for (let i = 0; i < motes.length; i++) {
      const s = motes[i], ph = (t * 0.35 + i / motes.length) % 1, an = i * 2.4 + t * 0.8;
      const rr = 2.0 + 0.8 * Math.sin(i * 1.7);
      s.position.set(Math.cos(an) * rr, ph * 10, Math.sin(an) * rr * 0.8);
      s.material.opacity = Math.sin(ph * PI) * 0.8 * up * (0.6 + fl);
      s.material.color.copy(tmpC.copy(acc).lerp(gold, fl));
    }
  });
}

// ================================================================ HERO TEAM
function buildHeroTeam(c) {
  const { T, late } = prep(c);
  c.rig.koMode = 'team';
  const suitC = color(c.colors.body, '#2b2d42'), whiteC = color(c.colors.belly, '#edf2f4'), redC = color(c.colors.accent, '#ef233c');
  const teamSuit = c.M.body; teamSuit.roughness = 0.5;
  const white = c.M.belly; white.color.multiplyScalar(0.78); white.roughness = 0.6;
  const skin = std({ color: '#c8956f', roughness: 0.62 });
  const skin2 = std({ color: '#8a5a3c', roughness: 0.62 });
  const hairD = std({ color: '#1e1814', roughness: 0.5 });
  const teamS = tinyScale(T, 9.1);
  const team = c.g(c.root); team.scale.setScalar(teamS); c.rig.trunk = team;
  const members = [];
  const tint = (hex) => suitC.clone().lerp(color(hex, '#888'), 0.35);
  const mk = (name, accHex, scale, slot, ph) => {
    const g = c.g(team, slot.clone());
    const inner = c.g(g); inner.scale.setScalar(scale);
    const accC = color(accHex, '#fff');
    const m = { name, g, inner, scale, accC, acc: glowMat(accC, 0.55), panel: std({ color: accC.clone().multiplyScalar(0.8), roughness: 0.45 }), ph, off: v(), offV: v(), lean: 0, arms: [] };
    members.push(m);
    return m;
  };
  const S = [v(0, 0, 3.2), v(-4.3, 0, 0.6), v(4.1, 0, 1.3), v(-2.4, 0, -3.3), v(2.7, 0, -3.5)].map(p => p.multiplyScalar(3.8));

  // 1. leader with a shield (team accent)
  const L = mk('leader', '#' + redC.getHexString(), 0.8, S[0], 0);
  {
    const P = person(c, L.inner, { mats: { suit: teamSuit, chest: white, skin, glove: L.panel, boot: L.panel }, chestW: 1.38, armLimbs: false, ph: 0,
      armRest: [{ x: -0.35, z: 0.12, el: -1.0 }, { x: 0.05, z: 0.12, el: -0.25 }] });
    L.P = P; L.arms = P.arms;
    const { hp, hr } = P;
    c.ell(hp, hr * 0.62, hr * 0.4, hr * 0.6, skin, v(0, hr * 0.42, hr * 0.22));
    c.ell(hp, hr * 0.98, hr * 0.6, hr * 1.02, hairD, v(0, hr * 1.5, -hr * 0.1));
    c.ell(hp, hr * 0.94, hr * 0.17, hr * 1.0, L.acc, v(0, hr * 1.13, 0.03));
    for (const s of [-1, 1]) c.ell(hp, hr * 0.12, hr * 0.06, hr * 0.05, white, v(s * hr * 0.32, hr * 1.13, hr * 1.0));
    c.m(extrude(chevronShape(0.55, 0.6, 0.22), 0.08, 0.02), L.acc, P.torso, v(0, 2.1, P.surfZ(0, 2.1) + 0.04), [-0.1, 0, 0]);
    // heater shield on the left forearm
    const sh = new THREE.Shape();
    sh.moveTo(-0.95, 0.85); sh.lineTo(0.95, 0.85); sh.lineTo(0.95, 0.05); sh.quadraticCurveTo(0.85, -0.9, 0, -1.45); sh.quadraticCurveTo(-0.85, -0.9, -0.95, 0.05); sh.closePath();
    const shield = c.g(P.arms[0].el, v(0, -0.85, 0.55), [1.0, 0, 0]);
    const sk = 1 + 0.08 * T.arm;
    shield.scale.setScalar(sk);
    c.m(extrude(sh, 0.18, 0.05), white, shield);
    const rim = new THREE.Shape(); rim.moveTo(-0.7, 0.62); rim.lineTo(0.7, 0.62); rim.lineTo(0.7, 0.02); rim.quadraticCurveTo(0.62, -0.68, 0, -1.08); rim.quadraticCurveTo(-0.62, -0.68, -0.7, 0.02); rim.closePath();
    c.m(extrude(rim, 0.05, 0.02), L.panel, shield, v(0, 0, 0.12));
    c.m(extrude(chevronShape(0.45, 0.55, 0.2), 0.05, 0.02), L.acc, shield, v(0, -0.05, 0.17));
    L.shield = shield;
    L.focus = P.arms[1].hand;
  }
  // 2. bruiser: huge frame, giant fists, small head
  const B = mk('bruiser', '#ff8c2a', 0.98, S[1], 1.3);
  {
    const P = person(c, B.inner, { mats: { suit: std({ color: tint('#ff8c2a'), roughness: 0.55 }), chest: B.panel, skin: skin2, glove: B.panel, hand: B.panel, boot: teamSuit },
      chestW: 2.0, waistW: 1.25, hipW: 1.1, armR: 0.62, legR: 0.62, headR: 0.42, torsoLen: 1.05, legLen: 0.82, neckW: 1.8, lean: 0.16, armLimbs: false, ph: 1.3,
      armRest: [{ x: 0.1, z: 0.28, el: -0.4 }, { x: 0.1, z: 0.28, el: -0.4 }] });
    B.P = P; B.arms = P.arms;
    const { hp, hr } = P;
    c.ell(hp, hr * 0.8, hr * 0.5, hr * 0.7, skin2, v(0, hr * 0.45, hr * 0.25));
    c.ell(hp, hr * 0.92, hr * 0.32, hr * 0.98, hairD, v(0, hr * 1.85, 0), null);
    for (const s of [-1, 1]) {
      c.ell(hp, hr * 0.13, hr * 0.06, hr * 0.05, B.acc, v(s * hr * 0.34, hr * 1.1, hr * 0.96));
      c.ell(hp, hr * 0.26, hr * 0.08, hr * 0.12, hairD, v(s * hr * 0.34, hr * 1.25, hr * 0.92), [0, 0, s * -0.25]);
    }
    for (const A of P.arms) { A.hand.scale.setScalar(1.75 + 0.08 * T.pow); c.ell(A.el, P.ar * 1.05, 0.5, P.ar * 1.05, B.panel, v(0, -1.25, 0)); }
    c.rbox(P.torso, 1.6, 0.35, 0.3, 0.1, B.acc, v(0, 0.55, P.surfZ(0, 0.55) - 0.05));
    B.focus = P.arms[1].hand;
  }
  // 3. speedster: slim, finned helmet, leaning forward
  const Sp = mk('speedster', '#ffd23f', 0.72, S[2], 2.1);
  {
    const P = person(c, Sp.inner, { mats: { suit: teamSuit, chest: Sp.panel, skin, glove: Sp.panel, boot: Sp.panel, leg: teamSuit },
      chestW: 1.05, waistW: 0.62, hipW: 0.7, armR: 0.25, legR: 0.34, legLen: 1.08, lean: 0.28, armLimbs: false, ph: 2.1,
      armRest: [{ x: 0.5, z: 0.1, el: -0.9 }, { x: -0.45, z: 0.1, el: -0.6 }], legRest: [{ x: -0.35, knee: 0.5 }, { x: 0.3, knee: 0.35 }] });
    Sp.P = P; Sp.arms = P.arms;
    const { hp, hr } = P;
    c.ell(hp, hr * 1.0, hr * 1.02, hr * 1.06, Sp.panel, v(0, hr * 1.22, -0.02));
    c.ell(hp, hr * 0.62, hr * 0.38, hr * 0.6, skin, v(0, hr * 0.48, hr * 0.32));
    c.ell(hp, hr * 0.78, hr * 0.17, hr * 0.5, std({ color: '#111', roughness: 0.15, metalness: 0.6 }), v(0, hr * 1.12, hr * 0.62));
    for (const s of [-1, 1]) {
      const f = new THREE.Shape(); f.moveTo(0, 0); f.lineTo(-1.1, 0.25); f.lineTo(-0.2, 0.32); f.closePath();
      const fg = extrude(f, 0.06, 0.02); fg.rotateY(PI / 2);
      c.m(fg, Sp.acc, hp, v(s * hr * 0.95, hr * 1.25, 0.05), [0, 0, s * 0.15]);
    }
    for (const Lg of P.legs) c.tube(Lg.hip, [v(Lg.s * P.lr * 0.95, -0.2, 0), v(Lg.s * P.lr * 0.92, -2.2, 0.05)], 0.05, 0.05, Sp.acc, { seg: 4, rad: 4 });
    c.m(extrude(starShape(0.32, 0.12, 4), 0.06, 0.02), Sp.acc, P.torso, v(0, 2.15, P.surfZ(0, 2.15) + 0.03));
    Sp.focus = P.arms[1].hand;
    if (T.spd >= 1) Sp.trail = true;
  }
  // 4. armoured tech hero: blocky plates, glowing visor, shoulder cannon
  const Tc = mk('tech', '#3ad0ff', 0.84, S[3], 3.4);
  {
    const plate = std({ color: suitC.clone().lerp(new THREE.Color('#8a96a8'), 0.6), roughness: 0.32, metalness: 0.75 });
    const P = person(c, Tc.inner, { mats: { suit: plate, skin: plate, glove: plate, boot: plate }, chestW: 1.4, armR: 0.42, legR: 0.48, armLimbs: false, ph: 3.4, head: false,
      armRest: [{ x: 0.1, z: 0.14, el: -0.3 }, { x: -0.2, z: 0.12, el: -0.9 }] });
    Tc.P = P; Tc.arms = P.arms;
    const { hp, hr, torso } = P;
    c.rbox(hp, hr * 1.7, hr * 2.0, hr * 1.9, 0.18, plate, v(0, hr * 1.0, 0));
    c.rbox(hp, hr * 1.4, hr * 0.28, 0.1, 0.05, Tc.acc, v(0, hr * 1.15, hr * 0.95));
    c.rbox(torso, 2.5, 1.45, 1.55, 0.3, plate, v(0, 2.05, 0.05));
    c.m(new THREE.CylinderGeometry(0.32, 0.32, 0.12, 20).rotateX(PI / 2), Tc.acc, torso, v(0, 2.15, 0.85));
    c.rbox(torso, 1.6, 0.9, 1.1, 0.2, Tc.panel, v(0, 0.95, 0.04));
    c.rbox(torso, 1.6, 1.5, 0.6, 0.15, plate, v(0, 2.0, -0.95));
    for (const s of [-1, 1]) c.m(new THREE.CylinderGeometry(0.22, 0.28, 0.4, 12), Tc.acc, torso, v(s * 0.45, 1.15, -1.3));
    for (const A of P.arms) { c.rbox(A.sh, 1.1, 0.8, 1.1, 0.2, plate, v(A.s * 0.1, 0.15, 0)); c.rbox(A.el, 0.75, 1.2, 0.8, 0.15, Tc.panel, v(0, -0.85, 0)); }
    for (const Lg of P.legs) c.rbox(Lg.knee, 0.8, 1.6, 0.85, 0.15, plate, v(0, -1.0, 0.05));
    const can = c.g(P.arms[1].sh, v(0.15, 0.85, 0), [-PI / 2 + 0.1, 0, 0]);
    c.rbox(can, 0.5, 0.5, 0.6, 0.1, plate, v(0, 0, 0));
    c.m(new THREE.CylinderGeometry(0.13, 0.15, 1.3, 10), plate, can, v(0, 0.75, 0));
    Tc.focus = obj3(can, v(0, 1.45, 0));
    if (T.fcs >= 2) c.rbox(hp, 0.08, 0.6, 0.08, 0.02, Tc.acc, v(hr * 0.75, hr * 2.0, 0));
  }
  // 5. energy caster: hooded robe, hovering, orbs in both hands
  const Ca = mk('caster', '#b56aff', 0.76, S[4], 4.2);
  {
    const robe = std({ color: tint('#6a3aaa'), roughness: 0.7, side: THREE.DoubleSide });
    const P = person(c, Ca.inner, { mats: { suit: robe, skin, glove: robe, boot: robe }, chestW: 1.05, waistW: 0.68, hipW: 0.8, armR: 0.27, legR: 0.34, armLimbs: false, ph: 4.2,
      armRest: [{ x: -0.6, z: 0.35, el: -0.7 }, { x: -0.6, z: 0.35, el: -0.7 }], legRest: [{ x: 0.1, knee: 0.3 }, { x: -0.1, knee: 0.45 }] });
    Ca.P = P; Ca.arms = P.arms; Ca.hover = 1.2;
    const { hp, hr, torso } = P;
    const pts = []; for (let i = 0; i <= 10; i++) { const u = i / 10; pts.push(new THREE.Vector2(lerp(0.72, 1.55, Math.pow(u, 0.8)) + Math.sin(u * PI) * 0.15, 0.6 - u * 4.4)); }
    c.m(new THREE.LatheGeometry(pts, 32), robe, torso);
    c.ell(hp, hr * 0.6, hr * 0.38, hr * 0.58, skin, v(0, hr * 0.44, hr * 0.24));
    const hood = new THREE.SphereGeometry(hr * 1.25, 20, 14, 0, PI * 2, 0, PI * 0.62); hood.rotateX(-0.55);
    c.m(hood, robe, hp, v(0, hr * 1.0, -hr * 0.15));
    for (const s of [-1, 1]) c.ell(hp, hr * 0.12, hr * 0.07, hr * 0.05, Ca.acc, v(s * hr * 0.32, hr * 1.1, hr * 0.97));
    const orbK = 1 + 0.15 * T.enr;
    for (const A of P.arms) c.ell(A.hand, 0.42 * orbK, 0.42 * orbK, 0.42 * orbK, Ca.acc, v(0, -0.35, 0.1));
    c.m(extrude(starShape(0.32, 0.15, 6), 0.06, 0.02), Ca.acc, torso, v(0, 2.1, P.surfZ(0, 2.1) + 0.03));
    Ca.focus = P.arms[1].hand;
  }
  // rig wiring: leader = head/body, beam converges in front of the team
  const lp = L.P;
  const conv = obj3(team, v(0, 9, S[0].z + 5));
  setHead(c, lp.hp, [conv]);
  c.rig.body = lp.torso;
  c.rig.extra.handL = lp.arms[0].hand; c.rig.extra.handR = lp.arms[1].hand;
  c.mount('chest', team, v(0, 4.0, 0.6), v(0, 0, 1));
  c.mount('back', team, v(0, 4.0, S[4].z), v(0, 0.2, -1));
  c.mount('crest', lp.hp, v(0, lp.hr * 2.1, 0), v(0, 1, 0), v(0, 0, -1));
  for (const A of lp.arms) c.mount('shoulders', A.sh, v(A.s * 0.1, lp.ar * 1.1, 0), v(A.s * 0.3, 1, 0));
  if (T.arm >= 2) for (const m of members) for (const A of m.arms) c.ell(A.sh, m.P.ar * 1.5, m.P.ar * 0.5, m.P.ar * 1.4, m.panel, v(A.s * 0.08, m.P.ar * 0.9, 0), [0, 0, A.s * -0.35]);
  for (const m of members) for (const A of m.arms) { A.sh.userData.rest = A.sh.rotation.clone(); A.el.userData.rest = A.el.rotation.clone(); }

  // late: speedster trails, caster rune ring
  const tex = glowTex();
  const trails = [];
  if (Sp.trail) for (let i = 0; i < 3; i++) {
    const tr = new THREE.Mesh(new THREE.PlaneGeometry(0.25, 4.5), addMat(Sp.accC, 0.35));
    tr.position.set(0, 1.5 + i * 1.6, -2.2); tr.rotation.set(PI / 2, 0, 0); trails.push(tr); late.push([tr, Sp.inner]);
  }
  const rune = new THREE.Mesh(new THREE.TorusGeometry(2.3, 0.06, 6, 48), addMat(Ca.accC, 0.8));
  rune.rotation.x = PI / 2; rune.position.y = 0.4;
  late.push([rune, Ca.inner]);
  rimGlow([teamSuit], redC, 0.1);
  const tfx = tinyFx(c, late, redC, { centerY: 0.45, beacon: 5.5, disc: 6.0, ring: 2.7, trail: false });
  const beacons = members.map(m => { const b = sprite(tex, m.accC, 22, { opacity: 0.4 }); b.position.y = 4.5; late.push([b, m.inner]); return b; });
  const casterGlow = sprite(tex, Ca.accC, 3.5, { opacity: 0.5 });
  casterGlow.position.set(0, -0.6, 0);
  late.push([casterGlow, Ca.inner]);

  const N = members.length;
  const order = [0, 2, 1, 3, 4];
  const api = {
    active: 0, seq: 0, swapK: 1, slotOf: members.map((_, i) => i), fromSlot: members.map((_, i) => i), incoming: -1, outgoing: -1,
    members,
    reset() { this.active = 0; this.seq = 0; this.swapK = 1; this.slotOf = members.map((_, i) => i); this.fromSlot = this.slotOf.slice(); this.incoming = this.outgoing = -1; members.forEach(m => { m.off.set(0, 0, 0); m.offV.set(0, 0, 0); m.lean = 0; }); },
    onStrike(hitNo, nHits, kind) {
      if (kind === 'melee' || kind === 'multi' || kind === 'grab') this.active = order[(this.seq++) % N];
      else this.active = this.slotOf.indexOf(0);
    },
    onHit(big) {
      const front = this.slotOf.indexOf(0);
      members.forEach((m, i) => {
        const k = (i === front ? 1.6 : 0.6) * (big ? 1.5 : 1);
        m.offV.add(v((Math.sin(i * 12.9 + this.seq * 3.1)) * 2.2 * k, 0, -2.6 * k));
        m.lean = Math.min(1, m.lean + (i === front ? 0.8 : 0.3) * (big ? 1.4 : 1));
      });
      return null;
    },
    // tag in: the member at the back swaps places with the front member, the rest re-form
    beginSwap() {
      this.fromSlot = this.slotOf.slice();
      const outI = this.slotOf.indexOf(0), inI = this.slotOf.indexOf(N - 1);
      const next = this.slotOf.slice();
      next[outI] = N - 1; next[inI] = 0;
      for (let i = 0; i < N; i++) if (i !== outI && i !== inI) next[i] = this.slotOf[i];
      this.slotOf = next; this.incoming = inI; this.outgoing = outI; this.swapK = 0;
    },
    frontLocal(out) { return out.copy(S[0]); },
    frontWorld(out) { team.updateWorldMatrix(true, false); return team.localToWorld(out.copy(S[0]).setY(4)); },
    beamSources() { return members.map(m => m.focus).filter(Boolean); },
    memberColors() { return members.map(m => m.accC); },
  };
  const frontZ = S[0].z + 1.4;
  runtime(c, late, api, (rig, dt, t) => {
    const a = rig.anim, m0 = rig.motion;
    // whole-group lunge becomes one member dashing in; group KO tilt becomes members falling one by one
    m0.position.z -= a.lunge * a.reach;
    m0.rotation.x = 0;
    const sk = smooth(api.swapK);
    for (let i = 0; i < N; i++) {
      const m = members[i];
      const from = S[api.fromSlot[i]], to = S[api.slotOf[i]];
      const p = m.g.position.lerpVectors(from, to, sk);
      if (i === api.incoming && api.swapK < 1) p.y += Math.sin(api.swapK * PI) * 5;
      if (i === api.outgoing && api.swapK < 1) p.y += Math.sin(api.swapK * PI) * 1.5;
      // hit scatter (spring back)
      m.off.addScaledVector(m.offV, dt); m.offV.multiplyScalar(Math.exp(-dt * 6)); m.off.multiplyScalar(Math.exp(-dt * 3));
      m.lean *= Math.exp(-dt * 3);
      p.add(m.off);
      const act = i === api.active ? a.lunge : 0;
      p.z += act * (a.reach / teamS + frontZ - to.z - 0.6);
      if (m.hover) p.y += (m.hover + Math.sin(t * 1.9 + m.ph) * 0.3) * (1 - a.ko);
      p.y += Math.max(0, act) * 3.0 / teamS; // leap up to the kaiju's chest
      p.y += Math.abs(Math.sin(t * 3.0 + m.ph)) * 0.35 * a.victory;
      const fallK = clamp01(a.ko * 1.5 - i * 0.12);
      m.g.rotation.set(-fallK * 1.45 + act * 0.25 - m.lean * 0.25 - a.recoil * 0.1, 0, Math.sin(t * 0.5 + m.ph) * 0.03 + (i === api.incoming ? (1 - sk) * PI * 2 * 0 : 0));
      if (i === api.incoming && api.swapK < 1) m.g.rotation.y = (1 - sk) * PI * 2;
      // arms: idle sway, punch for the active member, raised in victory
      for (const A of m.arms) {
        const r = A.sh.userData.rest, re = A.el.userData.rest;
        const punch = A.s > 0 ? act : act * 0.3;
        const cast = (m.name === 'caster' || m.name === 'tech') ? a.charge : 0;
        A.sh.rotation.x = r.x + Math.sin(t * 1.4 + m.ph + A.s) * 0.06 - punch * 1.45 - cast * 0.9 - a.victory * (A.s > 0 ? 2.6 : 0.6) - a.walk * Math.sin(t * 6.5 + m.ph + (A.s > 0 ? 0 : PI)) * 0.35;
        A.el.rotation.x = re.x + punch * 0.7 + Math.sin(t * 1.4 + m.ph) * 0.04;
      }
    }
    tfx(rig, dt, t, api.swapK < 1 ? Math.sin(api.swapK * PI) : 0);
    beacons.forEach((b, i) => { b.material.opacity = (0.32 + 0.12 * Math.sin(t * 3 + i) + (i === api.active ? a.lunge * 0.6 : 0)) * (1 - a.ko * 0.8); });
    rune.rotation.z = t * 0.8; rune.material.opacity = (0.55 + 0.25 * Math.sin(t * 3) + a.charge * 0.5) * (1 - a.ko);
    casterGlow.material.opacity = (0.4 + a.charge * 0.6) * (1 - a.ko);
    for (let i = 0; i < trails.length; i++) trails[i].material.opacity = (0.15 + 0.25 * Math.abs(Math.sin(t * 5 + i))) * (1 - a.ko);
  });
}

// ================================================================ ARMY
function tank(c, parent, pos, M, big = 1) {
  const g = c.g(parent, pos);
  const hull = c.g(g);
  c.rbox(hull, 2.3, 0.75, 3.7, 0.15, M.hull, v(0, 0.95, 0));
  c.rbox(hull, 2.1, 0.35, 1.0, 0.1, M.hull, v(0, 0.9, 2.0), [0.45, 0, 0]);
  const wheel = new THREE.CylinderGeometry(0.34, 0.34, 0.22, 12);
  for (const s of [-1, 1]) {
    c.rbox(hull, 0.6, 0.78, 4.0, 0.28, M.dark, v(s * 1.22, 0.42, 0));
    for (let k = 0; k < 5; k++) c.m(wheel, M.metal, hull, v(s * 1.4, 0.4, -1.5 + k * 0.75), [0, 0, PI / 2]);
    c.rbox(hull, 0.72, 0.08, 4.1, 0.03, M.hull, v(s * 1.22, 0.86, 0));
  }
  const turret = c.g(hull, v(0, 1.32, -0.25));
  c.rbox(turret, 1.65, 0.62, 2.0, 0.2, M.hull, v(0, 0.3, 0));
  c.rbox(turret, 1.67, 0.1, 0.32, 0.03, M.mark, v(0, 0.36, -0.55));
  c.m(new THREE.CylinderGeometry(0.28, 0.3, 0.2, 12), M.dark, turret, v(0.4, 0.68, -0.3));
  c.m(new THREE.CylinderGeometry(0.02, 0.02, 1.6, 4), M.dark, turret, v(-0.6, 1.3, -0.75));
  const barrel = c.g(turret, v(0, 0.3, 0.95));
  c.rbox(barrel, 0.55, 0.42, 0.42, 0.08, M.hull, v(0, 0, 0.1));
  const tube = c.g(barrel);
  const bg = new THREE.CylinderGeometry(0.11 * big, 0.13 * big, 2.6 * big, 10); bg.rotateX(PI / 2); bg.translate(0, 0, 1.4 * big);
  c.m(bg, M.metal, tube);
  const mb = new THREE.CylinderGeometry(0.17 * big, 0.17 * big, 0.3, 10); mb.rotateX(PI / 2);
  c.m(mb, M.dark, tube, v(0, 0, 2.7 * big));
  const muzzle = obj3(tube, v(0, 0, 2.95 * big));
  return { g, hull, turret, barrel, tube, muzzle, kind: 'tank', top: v(0, 2.3, -0.2), home: pos.clone(), rec: 0, smoke: 0, floor: 0, jolt: 0, yaw: 0, pitch: 0, pitch0: 0, ph: c.rng() * 6 };
}
function howitzer(c, parent, pos, M, big = 1) {
  const g = c.g(parent, pos);
  const hull = c.g(g);
  for (const s of [-1, 1]) {
    c.m(new THREE.CylinderGeometry(0.62, 0.62, 0.3, 16), M.dark, hull, v(s * 0.95, 0.62, 0.1), [0, 0, PI / 2]);
    c.m(new THREE.CylinderGeometry(0.2, 0.2, 0.34, 10), M.metal, hull, v(s * 0.95, 0.62, 0.1), [0, 0, PI / 2]);
    c.tube(hull, [v(s * 0.35, 0.62, -0.3), v(s * 0.85, 0.35, -1.8), v(s * 1.05, 0.14, -2.7)], 0.12, 0.1, M.hull, { seg: 6, rad: 6 });
  }
  const turret = c.g(hull, v(0, 0.95, 0.1));
  c.rbox(turret, 1.3, 0.42, 1.6, 0.1, M.hull, v(0, 0, 0));
  c.rbox(turret, 1.9, 1.0, 0.1, 0.04, M.hull, v(0, 0.6, 0.55), [-0.18, 0, 0]);
  c.rbox(turret, 1.92, 0.12, 0.12, 0.03, M.mark, v(0, 1.05, 0.48), [-0.18, 0, 0]);
  const barrel = c.g(turret, v(0, 0.42, 0.15), [-0.62, 0, 0]);
  const tube = c.g(barrel);
  const bg = new THREE.CylinderGeometry(0.14 * big, 0.17 * big, 3.9 * big, 12); bg.rotateX(PI / 2); bg.translate(0, 0, 1.6 * big);
  c.m(bg, M.metal, tube);
  const rc = new THREE.CylinderGeometry(0.1, 0.1, 1.6, 8); rc.rotateX(PI / 2); rc.translate(0, 0.24, 0.5);
  c.m(rc, M.hull, tube);
  const muzzle = obj3(tube, v(0, 0, 3.6 * big));
  return { g, hull, turret, barrel, tube, muzzle, kind: 'art', top: v(0, 1.8, 0), home: pos.clone(), rec: 0, smoke: 0, floor: 0, jolt: 0, yaw: 0, pitch: -0.62, pitch0: -0.62, ph: c.rng() * 6 };
}
function jetModel(c, M) {
  const g = new THREE.Group();
  const body = new THREE.Group(); g.add(body);
  const fus = new THREE.CylinderGeometry(0.3, 0.42, 4.2, 10); fus.rotateX(PI / 2);
  c.m(fus, M.jet, body);
  const nose = new THREE.ConeGeometry(0.3, 1.4, 10); nose.rotateX(PI / 2);
  c.m(nose, M.jet, body, v(0, 0, 2.8));
  c.ell(body, 0.24, 0.22, 0.75, M.canopy, v(0, 0.3, 1.3));
  const w = new THREE.Shape();
  w.moveTo(0, 1.3); w.lineTo(2.5, -1.0); w.lineTo(2.5, -1.4); w.lineTo(0.3, -1.25); w.lineTo(-0.3, -1.25); w.lineTo(-2.5, -1.4); w.lineTo(-2.5, -1.0); w.closePath();
  const wg = extrude(w, 0.08, 0.02); wg.rotateX(PI / 2);
  c.m(wg, M.jet, body, v(0, -0.05, -0.3));
  for (const s of [-1, 1]) {
    const f = new THREE.Shape(); f.moveTo(0, 0); f.lineTo(-1.0, 0); f.lineTo(-1.25, 1.1); f.lineTo(-0.8, 1.1); f.closePath();
    const fg = extrude(f, 0.06, 0.02); fg.rotateY(PI / 2);
    c.m(fg, M.jet, body, v(s * 0.3, 0.15, -1.0), [0, 0, s * -0.35]);
  }
  c.rbox(body, 0.5, 0.08, 0.4, 0.02, M.mark, v(0, 0.02, -2.0));
  const flame = new THREE.Mesh(new THREE.ConeGeometry(0.32, 1.4, 10).rotateX(-PI / 2), addMat(M.flameC, 0.85));
  flame.position.z = -2.75; body.add(flame);
  g.userData.flame = flame;
  return g;
}

function buildArmy(c) {
  const { T, late } = prep(c);
  c.rig.koMode = 'army';
  const accC = color(c.colors.accent, '#ff8c1a'), khakiC = color(c.colors.belly, '#8a8d6b');
  const hullM = c.M.body; hullM.roughness = 0.72; hullM.metalness = 0.22;
  const M = {
    hull: hullM,
    dark: std({ color: '#23251c', roughness: 0.85, metalness: 0.2 }),
    metal: std({ color: '#3b3e36', roughness: 0.45, metalness: 0.65 }),
    mark: glowMat(accC, 0.2, { roughness: 0.5 }),
    jet: std({ color: khakiC.clone().lerp(new THREE.Color('#9aa0a6'), 0.6), roughness: 0.4, metalness: 0.55 }),
    canopy: std({ color: '#1a2430', roughness: 0.1, metalness: 0.8 }),
    flameC: accC.clone().lerp(new THREE.Color('#ffe0a0'), 0.4),
  };
  const fig = c.g(c.root); c.rig.trunk = fig;
  const tankN = 4;
  const tankPos = [v(-5.6, 0, 2.2), v(-1.9, 0, 3.0), v(1.9, 0, 3.0), v(5.6, 0, 2.2)];
  const big = 1 + 0.08 * T.pow;
  const tanks = tankPos.slice(0, tankN).map(p => tank(c, fig, p, M, big));
  if (T.arm >= 2) for (const tk of tanks) for (const s of [-1, 1]) for (let k = 0; k < 4; k++) c.rbox(tk.hull, 0.12, 0.3, 0.7, 0.04, M.hull, v(s * 1.6, 0.85, -1.3 + k * 0.85));
  const artPos = [v(-4.6, 0, -4.4), v(0, 0, -4.9), v(4.6, 0, -4.4)];
  const arts = artPos.map(p => howitzer(c, fig, p, M, big));
  // command truck with a radar dish (the rig "head")
  const truck = c.g(fig, v(0, 0, -1.4));
  c.rbox(truck, 1.9, 1.3, 1.5, 0.15, hullM, v(0, 1.2, 1.6));
  c.rbox(truck, 1.6, 0.5, 0.08, 0.05, M.canopy, v(0, 1.5, 2.36));
  const box = c.g(truck); c.rbox(box, 2.1, 1.8, 3.0, 0.12, hullM, v(0, 1.5, -0.8));
  c.rbox(box, 2.12, 0.14, 3.02, 0.04, M.mark, v(0, 2.0, -0.8));
  for (const s of [-1, 1]) for (const z of [1.6, 0, -1.6]) c.m(new THREE.CylinderGeometry(0.42, 0.42, 0.3, 12), M.dark, truck, v(s * 1.0, 0.42, z), [0, 0, PI / 2]);
  c.m(new THREE.CylinderGeometry(0.08, 0.1, 1.4, 8), M.metal, truck, v(0, 3.0, -1.2));
  const hp = c.g(truck, v(0, 3.7, -1.2));
  const spin = c.g(hp);
  const dishK = 1 + 0.15 * T.fcs;
  const dpts = []; for (let i = 0; i <= 6; i++) { const r = i / 6 * 1.0 * dishK; dpts.push(new THREE.Vector2(r, r * r * 0.35)); }
  const dish = new THREE.LatheGeometry(dpts, 20); dish.rotateX(PI / 2 - 0.35);
  c.m(dish, std({ color: khakiC, roughness: 0.5, metalness: 0.3, side: THREE.DoubleSide }), spin, v(0, 0.3, -0.2));
  c.m(new THREE.CylinderGeometry(0.03, 0.03, 0.9, 4).rotateX(PI / 2 - 0.35), M.metal, spin, v(0, 0.45, 0.3));
  if (T.fcs >= 2) c.m(new THREE.CylinderGeometry(0.025, 0.025, 2.4, 4), M.dark, truck, v(0.7, 3.6, 1.2));
  setHead(c, hp, [tanks[1].muzzle, tanks[2].muzzle]);
  c.rig.body = box;
  // sandbag walls
  for (const [x, z, r] of [[-3.7, 0.9, 0.2], [3.7, 0.9, -0.2], [0, -3.2, 0]]) {
    const wall = c.g(fig, v(x, 0, z), [0, r, 0]);
    for (let i = 0; i < 5; i++) for (let j = 0; j < 2; j++) c.ell(wall, 0.36, 0.2, 0.22, c.M.belly, v(-0.8 + i * 0.4 + j * 0.2, 0.2 + j * 0.34, 0), null, 10);
  }
  // lattice comms mast at the rear (gives the formation some height on its own)
  const mast = c.g(fig, v(-6.6, 0, -6.2));
  const mH = 7.5 + 0.5 * T.fcs;
  for (const [x, z] of [[-0.45, -0.45], [0.45, -0.45], [0.45, 0.45], [-0.45, 0.45]]) c.tube(mast, [v(x, 0, z), v(x * 0.35, mH, z * 0.35)], 0.06, 0.04, M.metal, { seg: 2, rad: 5, cap: false });
  for (let k = 1; k < 7; k++) { const y = k * mH / 7, w = lerp(0.9, 0.32, y / mH); c.rbox(mast, w, 0.06, w, 0.02, M.metal, v(0, y, 0)); }
  c.ell(mast, 0.16, 0.16, 0.16, M.mark, v(0, mH + 0.1, 0));
  // infantry (instanced)
  const nInf = 30 + 6 * T.hp;
  const infGeo = new THREE.CapsuleGeometry(0.17, 0.5, 3, 6); infGeo.translate(0, 0.42, 0);
  const helmGeo = new THREE.SphereGeometry(0.19, 8, 6); helmGeo.scale(1, 0.7, 1); helmGeo.translate(0, 0.95, 0);
  const infM = std({ color: hullM.color.clone().multiplyScalar(0.8).lerp(khakiC, 0.25), roughness: 0.9 });
  const inf = new THREE.InstancedMesh(infGeo, infM, nInf); inf.frustumCulled = false; inf.castShadow = true;
  const helm = new THREE.InstancedMesh(helmGeo, hullM, nInf); helm.frustumCulled = false; helm.castShadow = true;
  fig.add(inf, helm);
  const clusters = [[-3.7, 0.0], [0, 0.9], [3.7, 0.0], [-2.4, -2.6], [2.4, -2.6], [-7.0, -1.2], [7.0, -1.2], [0, -6.6]];
  const soldiers = [];
  for (let i = 0; i < nInf; i++) {
    const cl = clusters[i % clusters.length], an = c.rng() * PI * 2, rr = 0.4 + c.rng() * 1.2;
    soldiers.push({ x: cl[0] + Math.cos(an) * rr, z: cl[1] + Math.sin(an) * rr * 0.7, ph: c.rng() * 6, hop: 0, flee: v(Math.cos(an), 0, Math.sin(an) - 0.2).normalize(), yaw: (c.rng() - 0.5) * 0.6 });
  }
  const units = [...tanks, ...arts];
  // mounts / attach points
  c.mount('chest', fig, v(0, 2.0, 0.6), v(0, 0, 1));
  c.mount('back', fig, v(0, 2.0, -4.0), v(0, 0.3, -1));
  c.mount('crest', hp, v(0, 0.9, 0), v(0, 1, 0), v(0, 0, -1));
  c.mount('shoulders', tanks[0].turret, v(0, 0.7, 0), v(0, 1, 0));
  c.mount('shoulders', tanks[tanks.length - 1].turret, v(0, 0.7, 0), v(0, 1, 0));
  c.rig.extra.handL = tanks[0].muzzle; c.rig.extra.handR = tanks[tanks.length - 1].muzzle;

  // late: jets, reserves, smoke + fire
  const sky = new THREE.Group();
  const nJets = 2 + (T.spd >= 3 ? 1 : 0);
  const jets = [];
  for (let j = 0; j < nJets; j++) { const jt = jetModel(c, M); jt.scale.setScalar(0.95); sky.add(jt); jets.push({ g: jt, prev: v(), ang: j * PI * 2 / nJets }); }
  late.push([sky, fig]);
  const reserveG = new THREE.Group();
  const reserves = [v(-8.6, 0, 0.0), v(8.6, 0, 0.0)].map(p => { const r = tank(c, reserveG, p, M, big); r.reserve = true; return r; });
  reserveG.visible = false;
  late.push([reserveG, fig]);
  const sTex = smokeTex(), gTex = glowTex();
  for (const u of [...units, ...reserves]) {
    u.smokes = []; u.fires = [];
    for (let k = 0; k < 4; k++) { const s = sprite(sTex, '#3a3632', 1.5, { additive: false, opacity: 0, fog: true }); s.visible = false; u.smokes.push(s); late.push([s, u.hull]); }
    for (let k = 0; k < 2; k++) { const s = sprite(gTex, '#ff7a20', 1.4, { opacity: 0 }); s.visible = false; u.fires.push(s); late.push([s, u.hull]); }
  }

  let targetFn = null, jt = 0, rr = 0;
  const _p = v(), _q = v(), _m4 = new THREE.Matrix4(), _q4 = new THREE.Quaternion(), _s = v(1, 1, 1), _e = new THREE.Euler();
  const kick = (u) => { u.rec = 1; };
  const allTanks = () => [...tanks, ...(api.wave > 0 ? reserves : [])];
  const api = {
    noLunge: true, noLift: true, strafeK: -1, strafeTgt: v(0, 0, 14), reinK: 0, wave: 0, jetCount: nJets,
    reset() { this.strafeK = -1; this.reinK = 0; this.wave = 0; reserveG.visible = false; for (const u of [...units, ...reserves]) { u.smoke = 0; u.floor = 0; u.jolt = 0; u.rec = 0; } soldiers.forEach(s => { s.hop = 0; }); },
    setTarget(fn) { targetFn = fn; },
    onHit(big, out) {
      const pool = units.concat(api.wave > 0 ? reserves : []);
      const u = pool[(rr++ * 5 + (big ? 2 : 0)) % pool.length];
      u.smoke = 1; u.floor = Math.min(0.45, u.floor + 0.2); u.jolt = big ? 1.4 : 1;
      for (const s of soldiers) if (Math.hypot(s.x - u.home.x, s.z - u.home.z) < 4) s.hop = Math.max(s.hop, 0.6 + Math.random() * 0.6);
      if (!out) return null;
      u.hull.updateWorldMatrix(true, false);
      return u.hull.localToWorld(out.copy(u.top));
    },
    // projectile origin per ability: artillery lobs, tank salvos, infantry rifle fire
    origin(ab, hitNo, nHits, out) {
      const id = String(ab || '');
      const wp = (o) => { o.updateWorldMatrix(true, false); return o.getWorldPosition(new V3()); };
      if (id.includes('artillery')) {
        const u = arts[(hitNo - 1) % arts.length]; kick(u);
        out.copy(wp(u.muzzle));
        return { arcK: 0.5, ms: 760, flash: 1.5, scale: 0.6 };
      }
      if (id.includes('tank') || id.includes('column')) {
        const list = allTanks(); list.forEach(kick);
        out.copy(wp(list[0].muzzle));
        return { arcK: 0.03, ms: 420, flash: 1.3, scale: 0.6, extras: list.slice(1).map(u => wp(u.muzzle)) };
      }
      if (id.includes('rifle') || id.includes('volley')) {
        const pick = (k) => { const s = soldiers[(hitNo * 7 + k * 11) % soldiers.length]; fig.updateWorldMatrix(true, false); return fig.localToWorld(v(s.x, 0.85, s.z)); };
        out.copy(pick(0));
        return { arcK: 0.02, ms: 340, flash: 0.35, scale: 0.32, extras: [pick(1), pick(2), pick(3)] };
      }
      const u = tanks[(hitNo - 1) % tanks.length]; kick(u);
      out.copy(wp(u.muzzle));
      return { arcK: 0.05, ms: 460, flash: 1.1, scale: 0.6 };
    },
    strafe(worldTarget) {
      if (worldTarget) { fig.updateWorldMatrix(true, false); this.strafeTgt.copy(fig.worldToLocal(worldTarget.clone())); }
      this.strafeK = 0;
    },
    jetWorld(i, out) { const j = jets[i % jets.length]; j.g.updateWorldMatrix(true, false); return j.g.getWorldPosition(out); },
    reinforce(instant) { this.wave++; reserveG.visible = true; if (instant) this.reinK = 1; },
  };

  runtime(c, late, api, (rig, dt, t) => {
    const a = rig.anim, ko = a.ko, m0 = rig.motion;
    m0.rotation.x = 0; // a formation doesn't tip over
    jt += dt;
    fig.updateWorldMatrix(true, false);
    let tl = null;
    if (targetFn) { try { tl = fig.worldToLocal(targetFn(_q)); } catch { tl = null; } }
    // ground units: aim, recoil, jolt, smoke/fire
    const allU = units.concat(reserveG.visible ? reserves : []);
    for (let i = 0; i < allU.length; i++) {
      const u = allU[i];
      if (u.reserve) {
        const k = smooth(api.reinK), z = lerp(-26, u.home.z, k);
        u.g.position.set(u.home.x, 0, z);
        u.hull.position.y = k < 1 ? Math.abs(Math.sin(t * 9 + u.ph)) * 0.08 : 0;
      }
      let yaw, pitch;
      if (tl) {
        const dx = tl.x - u.g.position.x, dz = tl.z - u.g.position.z;
        yaw = Math.atan2(dx, dz);
        pitch = u.kind === 'tank' ? -Math.atan2(tl.y * 0.5 - 1.6, Math.hypot(dx, dz)) * 0.6 : u.pitch0;
      } else {
        yaw = Math.sin(t * 0.35 + u.ph) * 0.5; pitch = u.kind === 'tank' ? Math.sin(t * 0.5 + u.ph) * 0.05 : u.pitch0;
      }
      if (ko > 0) { yaw = lerp(yaw, u.ph - 3, ko * 0.4); pitch = lerp(pitch, u.kind === 'tank' ? 0.18 : -0.15, ko); }
      const kk = 1 - Math.exp(-dt * 4);
      u.yaw += (yaw - u.yaw) * kk; u.pitch += (pitch - u.pitch) * kk;
      u.turret.rotation.y = u.yaw;
      u.barrel.rotation.x = u.pitch;
      u.rec *= Math.exp(-dt * 7);
      u.tube.position.z = -0.55 * u.rec;
      u.jolt *= Math.exp(-dt * 5);
      u.hull.rotation.set(-u.jolt * 0.12 * Math.sin(t * 20) + ko * 0.05 * Math.sin(u.ph * 3), 0, u.jolt * 0.1 * Math.sin(t * 17 + u.ph) + ko * 0.07 * Math.cos(u.ph * 2));
      if (!u.reserve) u.hull.position.y = u.jolt * 0.25 * Math.abs(Math.sin(t * 14));
      u.smoke = Math.max(u.floor, u.smoke - dt * 0.35, ko);
      const fire = Math.max(ko, u.smoke > 0.9 ? (u.smoke - 0.9) * 10 : 0);
      for (let k = 0; k < u.smokes.length; k++) {
        const s = u.smokes[k], ph = (jt * 0.45 + k / u.smokes.length + u.ph) % 1;
        s.visible = u.smoke > 0.02;
        if (!s.visible) continue;
        s.position.set(u.top.x + Math.sin(ph * 4 + u.ph) * 0.4 + ph * 1.2, u.top.y + ph * 5 * (0.6 + u.smoke * 0.6), u.top.z - ph * 0.8);
        s.scale.setScalar(1.0 + ph * 3.2 * (0.6 + u.smoke));
        s.material.opacity = Math.sin(ph * PI) * 0.75 * u.smoke;
      }
      for (let k = 0; k < u.fires.length; k++) {
        const s = u.fires[k]; s.visible = fire > 0.02;
        if (!s.visible) continue;
        s.position.set(u.top.x + (k ? 0.4 : -0.3), u.top.y - 0.4 + Math.sin(t * 13 + k) * 0.12, u.top.z + (k ? -0.3 : 0.4));
        s.scale.setScalar((1.2 + 0.4 * Math.sin(t * 17 + k * 2)) * (0.6 + fire * 0.6));
        s.material.opacity = fire * (0.7 + 0.3 * Math.sin(t * 23 + k));
      }
    }
    spin.rotation.y = t * 1.3 * (1 - ko);
    // infantry: idle shuffle, hop when shelled, flee on KO (abandoned)
    for (let i = 0; i < soldiers.length; i++) {
      const s = soldiers[i];
      s.hop = Math.max(0, s.hop - dt * 1.6);
      const fl = smooth(ko * 1.3 - (i % 7) * 0.04);
      const x = s.x + s.flee.x * fl * 14 + Math.sin(t * 0.7 + s.ph) * 0.08;
      const z = s.z + s.flee.z * fl * 14 + a.victory * Math.sin(t * 2 + s.ph) * 0.1;
      const y = Math.abs(Math.sin(s.hop * PI * 2)) * s.hop * 1.2 + Math.abs(Math.sin(t * 4 + s.ph)) * 0.18 * a.victory + (fl > 0 && fl < 1 ? Math.abs(Math.sin(t * 12 + s.ph)) * 0.12 : 0);
      const sc = 1 - smooth((fl - 0.6) / 0.4);
      _e.set(0, fl > 0 ? Math.atan2(s.flee.x, s.flee.z) : s.yaw, 0);
      _q4.setFromEuler(_e); _s.setScalar(Math.max(0.0001, sc));
      _m4.compose(_p.set(x, y, z), _q4, _s);
      inf.setMatrixAt(i, _m4); helm.setMatrixAt(i, _m4);
    }
    inf.instanceMatrix.needsUpdate = true; helm.instanceMatrix.needsUpdate = true;
    // jets: orbit overhead; strafing pass over the target when strafeK in [0,1]; climb away on KO
    for (let j = 0; j < jets.length; j++) {
      const J = jets[j];
      const th = jt * 0.42 + J.ang;
      const R = 10 + ko * 30, h = 12 + Math.sin(jt * 0.7 + j) * 0.8 + ko * 18;
      _p.set(Math.cos(th) * R, h, Math.sin(th) * R - 1);
      if (api.strafeK >= 0) {
        const k = clamp01(api.strafeK - j * 0.07);
        const w = smooth(k / 0.18) * (1 - smooth((k - 0.82) / 0.18));
        const tg = api.strafeTgt;
        _q.set(tg.x + (j - (jets.length - 1) / 2) * 3.0, 15 - 10 * Math.sin(k * PI), lerp(-30, tg.z + 30, k));
        _p.lerp(_q, w);
      }
      J.g.position.copy(_p);
      const d = _q.subVectors(_p, J.prev);
      if (d.lengthSq() > 1e-6) {
        const yaw = Math.atan2(d.x, d.z), pitch = -Math.atan2(d.y, Math.hypot(d.x, d.z));
        const bank = api.strafeK >= 0 ? 0.1 : 0.55;
        J.g.rotation.set(pitch, yaw, bank, 'YXZ');
      }
      J.prev.copy(_p);
      J.g.userData.flame.scale.set(1, 1, 0.8 + 0.3 * Math.sin(jt * 30 + j));
    }
  });
}

// ================================================================ MAGICAL GIRL
function skirtLayer(r0, r1, h, puff, scallops) {
  const pts = [];
  for (let i = 0; i <= 12; i++) { const u = i / 12; pts.push(new THREE.Vector2(lerp(r0, r1, Math.pow(u, 0.75)) + Math.sin(u * PI) * puff, -u * h)); }
  const geo = new THREE.LatheGeometry(pts, 56);
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i), u = -y / h, ang = Math.atan2(x, z);
    const wave = Math.sin(ang * scallops);
    const k = 1 + 0.045 * wave * u * u;
    p.setXYZ(i, x * k, y + 0.12 * Math.max(0, wave) * u * u * u, z * k);
  }
  geo.computeVertexNormals();
  return geo;
}
function buildMagicalGirl(c) {
  const { T, late } = prep(c);
  c.rig.hover = 0.8; c.rig.koMode = 'back';
  const pinkC = color(c.colors.body, '#ff8fc7'), whiteC = color(c.colors.belly, '#ffffff'), accC = color(c.colors.accent, '#8ef6ff');
  const dress = c.M.body; dress.roughness = 0.55;
  const white = c.M.belly; white.roughness = 0.65; white.color.multiplyScalar(0.86);
  const skirtP = std({ color: pinkC, roughness: 0.55, side: THREE.DoubleSide });
  const skirtW = std({ color: whiteC.clone().multiplyScalar(0.86), roughness: 0.7, side: THREE.DoubleSide });
  const skirtA = std({ color: accC.clone().lerp(whiteC, 0.55), roughness: 0.6, side: THREE.DoubleSide });
  const ribbon = std({ color: pinkC.clone().lerp(new THREE.Color('#ff2f8a'), 0.45), roughness: 0.45, side: THREE.DoubleSide });
  const ribbon2 = std({ color: accC.clone().lerp(new THREE.Color('#2a9fd0'), 0.25), roughness: 0.45 });
  const skin = std({ color: '#efc3aa', roughness: 0.65 });
  const hairM = std({ color: '#c9b4ff', roughness: 0.45 });
  const gem = c.M.accent;
  const eyeW = std({ color: '#dcd8e4', roughness: 0.4 });
  const iris = std({ color: accC.clone().multiplyScalar(0.25), emissive: accC.clone().multiplyScalar(0.5), emissiveIntensity: 0.6, roughness: 0.2 }); iris.userData.eye = true;
  const figS = tinyScale(T, 9.6);
  const fig = c.g(c.root); fig.scale.setScalar(figS);
  const P = person(c, fig, {
    mats: { suit: dress, lower: dress, skin, arm: skin, glove: white, hand: white, leg: white, boot: dress, shoulder: white },
    chestW: 0.86, waistW: 0.55, hipW: 0.74, armR: 0.22, legR: 0.31, headR: 0.74, torsoLen: 0.8, legLen: 1.0, neckW: 0.8, pecs: false,
    armRest: [{ x: 0.1, z: 0.25, el: -0.35, reach: -1.0, vic: -0.4 }, { x: -0.85, z: 0.2, el: -1.45, reach: -0.6, elReach: 0.5, vic: -1.6 }],
    legRest: [{ x: -0.08, knee: 0.18 }, { x: 0.18, knee: 0.55 }],
  });
  c.rig.trunk = P.trunk; c.rig.body = P.torso;
  const { hp, hr, torso, tl } = P;
  // big anime head: cheeks, eyes, hair cap, bangs
  c.ell(hp, hr * 0.78, hr * 0.62, hr * 0.74, skin, v(0, hr * 0.62, hr * 0.18));
  for (const s of [-1, 1]) {
    const eg = c.g(hp, v(s * hr * 0.36, hr * 0.92, hr * 0.86), [0, s * 0.3, 0]);
    c.ell(eg, hr * 0.2, hr * 0.26, hr * 0.07, eyeW, v());
    c.ell(eg, hr * 0.16, hr * 0.22, hr * 0.06, iris, v(0, -hr * 0.03, hr * 0.03));
    c.ell(eg, hr * 0.08, hr * 0.11, hr * 0.04, std({ color: '#141020', roughness: 0.3 }), v(0, -hr * 0.04, hr * 0.065));
    c.ell(eg, hr * 0.05, hr * 0.06, hr * 0.03, eyeW, v(s * hr * 0.04, hr * 0.06, hr * 0.07));
    c.ell(eg, hr * 0.2, hr * 0.035, hr * 0.04, std({ color: '#3a2a4a' }), v(0, hr * 0.25, hr * 0.03), [0, 0, s * -0.12]);
    c.ell(hp, hr * 0.12, hr * 0.06, hr * 0.03, std({ color: '#ffaac0', roughness: 0.8 }), v(s * hr * 0.48, hr * 0.62, hr * 0.85));
  }
  c.ell(hp, hr * 0.08, hr * 0.03, hr * 0.03, std({ color: '#c0607a' }), v(0, hr * 0.45, hr * 0.93));
  c.ell(hp, hr * 0.97, hr * 0.9, hr * 1.02, hairM, v(0, hr * 1.25, -hr * 0.1));
  for (let i = -3; i <= 3; i++) c.ell(hp, hr * 0.2, hr * 0.42, hr * 0.14, hairM, v(i * hr * 0.24, hr * 1.42 - Math.abs(i) * hr * 0.05, hr * 0.8 - Math.abs(i) * hr * 0.08), [-0.3, 0, -i * 0.12]);
  for (const s of [-1, 1]) c.ell(hp, hr * 0.18, hr * 0.7, hr * 0.2, hairM, v(s * hr * 0.82, hr * 0.7, hr * 0.35), [0, 0, s * 0.1]);
  // twin tails with big ribbons (animated as tail limbs)
  for (const s of [-1, 1]) {
    const root = v(s * hr * 0.88, hr * 1.7, -hr * 0.35);
    const tail = c.chain(hp, { n: 7, len: 0.95, r0: 0.34, r1: 0.1, mat: hairM, pos: root, rot: [0.3, 0, PI + 0.42 * s], curlZ: -s * 0.07, curlX: 0.03, radial: 10 });
    c.limb({ type: 'tail', joints: tail, ampX: 0.05, ampZ: 0.09, speed: 1.5, phase: s * 1.1 });
    bow(c, hp, root.clone().add(v(s * 0.08, 0.15, 0)), 0.75, ribbon, { rot: [0, s * 1.2, 0], tailLen: 1.1 });
  }
  const mouth = obj3(hp, v(0, hr * 0.6, hr));
  // bodice details: collar, chest bow with gem, puffy sleeves
  const col = new THREE.TorusGeometry(0.42, 0.12, 8, 24).rotateX(PI / 2);
  c.m(col, white, P.trunk, v(0, P.neckY - 0.05, 0.02));
  const by = 2.0 * tl, bz = P.surfZ(0, by);
  bow(c, torso, v(0, by, bz + 0.08), 0.9, ribbon2, { tailLen: 1.2, knot: gem });
  c.mount('chest', torso, v(0, by, bz + 0.2), v(0, 0, 1));
  for (const A of P.arms) c.ell(A.sh, 0.5, 0.44, 0.48, white, v(A.s * 0.06, -0.12, 0));
  for (const A of P.arms) c.m(new THREE.TorusGeometry(P.ar * 1.05, 0.07, 6, 16).rotateX(PI / 2), ribbon, A.el, v(0, -0.05, 0));
  // skirt: three puffy scalloped layers + waist sash and back bow
  const skLayers = T.arm >= 2 ? 4 : 3;
  c.m(skirtLayer(0.6, 1.95, 1.85, 0.32, 14), skirtA, torso, v(0, 0.5, 0));
  c.m(skirtLayer(0.62, 1.85, 1.55, 0.35, 12), skirtW, torso, v(0, 0.55, 0));
  c.m(skirtLayer(0.64, 1.72, 1.25, 0.38, 10), skirtP, torso, v(0, 0.6, 0));
  if (skLayers > 3) c.m(skirtLayer(0.66, 1.4, 0.75, 0.3, 8), skirtW, torso, v(0, 0.65, 0));
  const sash = new THREE.TorusGeometry(1, 0.13, 8, 32).rotateX(PI / 2); sash.scale(0.6, 1, 0.5);
  c.m(sash, ribbon, torso, v(0, 0.66, 0.02));
  bow(c, torso, v(0, 0.75, -0.55), 1.15, ribbon, { rot: [0, PI, 0], tails: false });
  // boots: cuffs + little hearts
  for (const L of P.legs) {
    c.m(new THREE.CylinderGeometry(P.lr * 1.1, P.lr * 0.95, 0.3, 14), white, L.knee, v(0, -0.15, 0));
    const hs = new THREE.Shape(); hs.moveTo(0, -0.2); hs.bezierCurveTo(0.3, 0.05, 0.18, 0.28, 0, 0.12); hs.bezierCurveTo(-0.18, 0.28, -0.3, 0.05, 0, -0.2);
    c.m(extrude(hs, 0.05, 0.015), gem, L.knee, v(0, -0.35, P.lr * 0.95));
  }
  // star wand in the right hand: the beam / bolt origin
  const wandK = 1 + 0.08 * T.pow;
  const wand = c.g(P.arms[1].hand, v(0, -0.1, 0.1), [-0.45, 0, 0]);
  wand.scale.setScalar(wandK);
  c.m(new THREE.CylinderGeometry(0.06, 0.08, 1.7, 8).translate(0, -0.85, 0), white, wand);
  c.m(new THREE.TorusGeometry(0.12, 0.04, 6, 14).rotateX(PI / 2), ribbon, wand, v(0, -0.3, 0));
  const starG = c.g(wand, v(0, -2.05, 0));
  c.m(extrude(starShape(0.62 * (1 + 0.1 * T.enr), 0.27, 5), 0.16, 0.05), gem, starG, null, [PI / 2, 0, 0]);
  c.m(new THREE.TorusGeometry(0.82, 0.04, 6, 32), std({ color: '#fff4c0', roughness: 0.3, metalness: 0.6 }), starG, null, [PI / 2, 0, 0]);
  const wandTip = obj3(starG, v(0, -0.2, 0));
  setHead(c, hp, [wandTip, mouth]);
  gem.emissiveIntensity *= 1 + 0.12 * T.enr;
  if (T.spd >= 2) for (const s of [-1, 1]) {
    const w = new THREE.Shape(); w.moveTo(0, 0); w.bezierCurveTo(0.4, 0.9, 1.4, 1.3, 1.9, 0.8); w.bezierCurveTo(1.3, 0.5, 1.0, 0.1, 0, 0);
    const wg = new THREE.ShapeGeometry(w); wg.scale(s, 1, 1);
    c.m(wg, std({ color: whiteC, emissive: accC, emissiveIntensity: 0.25, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }), torso, v(s * 0.15, 2.0, -0.62), [0.2, 0, 0]);
  }
  if (T.fcs >= 2) {
    const tia = c.g(hp, v(0, hr * 1.9, hr * 0.55), [-0.4, 0, 0]);
    c.m(new THREE.TorusGeometry(hr * 0.5, 0.04, 6, 24, PI).rotateZ(0), std({ color: '#fff4c0', metalness: 0.7, roughness: 0.3 }), tia);
    c.m(extrude(starShape(0.16, 0.07), 0.05, 0.015), gem, tia, v(0, hr * 0.5, 0.02));
  }
  c.mount('back', torso, v(0, 1.8, -0.7), v(0, 0.1, -1));
  c.mount('crest', hp, v(0, hr * 2.2, 0), v(0, 1, 0), v(0, 0, -1));
  for (const A of P.arms) c.mount('shoulders', A.sh, v(A.s * 0.08, 0.35, 0), v(A.s * 0.3, 1, 0));
  c.rig.extra.handL = P.arms[0].hand; c.rig.extra.handR = P.arms[1].hand;

  // late: aura, sparkles, transformation flourishes (light wings, halo, glowing hem)
  const aura = sprite(haloTex(), accC, 1, { opacity: 0.4 });
  aura.position.set(0, 5.2, 0); aura.renderOrder = 4;
  late.push([aura, fig]);
  const sTex = starTex(), gTex = glowTex();
  const sparkles = [];
  for (let i = 0; i < 18; i++) { const s = sprite(sTex, i % 3 ? accC : new THREE.Color('#ffd6f0'), 0.6, { opacity: 0.9 }); sparkles.push(s); late.push([s, fig]); }
  const tipGlow = sprite(gTex, accC, 1.6, { opacity: 0.6 });
  late.push([tipGlow, starG]);
  const wings = new THREE.Group();
  for (const s of [-1, 1]) {
    for (let k = 0; k < 2; k++) {
      const w = new THREE.Shape();
      w.moveTo(0, 0); w.bezierCurveTo(0.6, 1.6 - k, 2.6, 2.6 - k * 1.8, 3.6 - k * 0.6, 1.8 - k * 2.2); w.bezierCurveTo(2.4, 1.0 - k, 1.2, 0.2 - k * 0.3, 0, 0);
      const wg = new THREE.ShapeGeometry(w, 16); wg.scale(s, 1, 1);
      const wm = new THREE.Mesh(wg, addMat(k ? pinkC.clone().lerp(whiteC, 0.3) : accC.clone().lerp(whiteC, 0.3), 0.45));
      wm.material.side = THREE.DoubleSide;
      wm.position.set(s * 0.1, 0, -0.05 * k); wm.rotation.y = -s * 0.8;
      wings.add(wm);
    }
  }
  wings.position.set(0, P.hipY + 2.0 * tl, -0.75);
  wings.scale.setScalar(0.001);
  late.push([wings, fig]);
  const halo = new THREE.Mesh(new THREE.TorusGeometry(1.2, 0.07, 8, 40), addMat(accC.clone().lerp(new THREE.Color('#fff4c0'), 0.5), 0.9));
  halo.rotation.x = PI / 2 - 0.25; halo.position.set(0, hr * 2.6, -0.1); halo.scale.setScalar(0.001);
  late.push([halo, hp]);
  const hem = new THREE.Mesh(new THREE.TorusGeometry(1.0, 0.06, 6, 56).rotateX(PI / 2), addMat(accC, 0.8));
  hem.scale.set(2.0, 1, 2.0); hem.position.y = 0.5 - 1.85 + 0.05; hem.visible = false;
  late.push([hem, torso]);

  rimGlow([dress, skirtP, ribbon], accC, 0.08);
  const tfx = tinyFx(c, late, accC, { centerY: 0.5, beacon: 3.2 });
  const dv = v();
  const api = {
    tf: 0,
    reset() { this.tf = 0; },
    setTransformed(k) { this.tf = clamp01(k); },
  };
  let baseE0 = null, dressE0 = null;
  runtime(c, late, api, (rig, dt, t) => {
    const a = rig.anim, up = 1 - a.ko, tf = api.tf;
    const calm = up * (1 - Math.min(1, Math.abs(a.lunge) * 3)) * (1 - a.victory * 0.6) * (1 - a.charge * 0.7);
    dart(dv, t, 2.7, (rig.bankSign ? 0.85 : 0.25) + 0.25 * tf, 1.3);
    rig.motion.position.x += (dv.x - (rig.bankSign || 0) * 0.5) * calm; rig.motion.position.y += dv.y * calm; rig.motion.position.z += dv.z * calm;
    rig.motion.position.y += Math.max(0, a.lunge) * 3.0;
    tfx(rig, dt, t, tf * 0.8);
    if (!baseE0) {
      baseE0 = rig.mats.filter(m => m.userData.glow && m.userData.e0).map(m => [m, m.userData.e0.clone()]);
      dressE0 = [dress, skirtP, white, skirtW, skirtA, ribbon].filter(m => m.userData.e0).map(m => [m, m.userData.e0.clone()]);
    }
    for (const [m, e] of baseE0) m.userData.e0.copy(e).multiplyScalar(1 + 1.2 * tf);
    for (const [m, e] of dressE0) m.userData.e0.copy(e).add(_tmp.copy(m.color).multiplyScalar(0.18 * tf));
    aura.material.opacity = Math.min(1, (0.24 + 0.07 * T.enr + 0.5 * tf) * (0.85 + 0.15 * Math.sin(t * 2.1)) * (1 + a.charge * 0.8 + a.roar * 0.4)) * up;
    aura.material.color.copy(accC).lerp(_tmp.set('#ffffff'), 0.25 * tf);
    const as = 1 + 0.5 * tf;
    aura.scale.set(6.6 * as, 12.5 * (1 + 0.22 * tf), 1);
    tipGlow.material.opacity = (0.5 + 0.2 * Math.sin(t * 5) + a.charge * 0.8 + tf * 0.4) * up;
    tipGlow.scale.setScalar(1.6 * (1 + a.charge * 0.8 + tf * 0.5));
    const ws = Math.max(0.001, smooth(tf) * (1 + 0.04 * Math.sin(t * 2)));
    wings.scale.setScalar(ws);
    wings.rotation.y = 0;
    wings.children.forEach((w, i) => { const s = i < 2 ? -1 : 1; w.rotation.y = s * (0.75 + 0.2 * Math.sin(t * 3 + i * 0.3)); w.material.opacity = 0.55 * up; });
    halo.scale.setScalar(Math.max(0.001, smooth(tf)));
    halo.rotation.z = t * 0.6;
    hem.visible = tf > 0.01; hem.material.opacity = 0.7 * tf * up;
    const n = sparkles.length;
    for (let i = 0; i < n; i++) {
      const s = sparkles[i], spd = 0.18 + 0.12 * tf, ph = (t * spd + i / n) % 1, an = i * 2.39 + t * (0.5 + tf * 0.6);
      const rr = (2.2 + 1.2 * Math.sin(i * 1.3)) * (1 + 0.4 * tf);
      s.position.set(Math.cos(an) * rr, 1.0 + ph * 10.5, Math.sin(an) * rr * 0.85);
      const tw = 0.5 + 0.5 * Math.sin(t * 7 + i * 1.7);
      s.material.opacity = Math.sin(ph * PI) * (0.55 + 0.45 * tf) * up * (0.4 + 0.6 * tw);
      s.scale.setScalar((0.45 + 0.35 * tw) * (1 + 0.6 * tf));
    }
  });
}
const _tmp = new THREE.Color();

registerPlanBuilder('hero', buildHero);
registerPlanBuilder('heroteam', buildHeroTeam);
registerPlanBuilder('army', buildArmy);
registerPlanBuilder('magicalgirl', buildMagicalGirl);

export const WILD_PLANS = ['hero', 'heroteam', 'army', 'magicalgirl'];
