// Procedural Kaiju models.
//
// buildKaijuModel(snapshot) -> THREE.Group
//   group.userData.rig = {
//     root, motion, trunk, body, head, mouth, limbs, tail, height, radius, front, hover,
//     heads, jaws, mouths, attach: { head, jaw, mouth, chest, back, tailTip, handL, handR, shoulderL, shoulderR, crown },
//     anim: { lunge, reach, recoil, roar, bite, charge, walk, lift, ko, victory, flash, flashColor },
//     update(dt, t), setStatus(id, on), clearStatuses(), hit(color), getMouthWorld(i, out), getCenterWorld(out),
//     getHeadWorld(out), dispose()
//   }
//
// Structure: one builder per body plan (PLAN_BUILDERS) fills a BuildCtx with meshes, limbs and *mounts*
// (named attach points with an outward normal). Stat tiers and ability cosmetics are then added generically
// on those mounts, so any plan (or a GLB loaded later via registerPlanBuilder) gets upgrades for free as long
// as it registers mounts.

import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { FRAME_BY_ID, visualTiers } from '../data/frames.js';
import { ABILITY_BY_ID } from '../data/abilities.js';

const V3 = THREE.Vector3;
const v = (x = 0, y = 0, z = 0) => new V3(x, y, z);
const lerp = (a, b, t) => a + (b - a) * t;
const PI = Math.PI;

export const HP_SIZE = [1, 1.1, 1.22, 1.36];
const HP_BULK = [1, 1.06, 1.13, 1.22];
const SPD_LEAN = [1, 0.95, 0.9, 0.85];
const POW_SCALE = [1, 1.3, 1.62, 1.95];
const DEFAULT_COLORS = { body: '#4a4a52', belly: '#9a9a8a', accent: '#5fd6ff' };
const EYE_ACCENT_PLANS = new Set(['cephalopod', 'moth', 'mantis', 'mech', 'golem', 'spider', 'ray']);

// ---------------------------------------------------------------- utils
function hashStr(s) {
  let h = 2166136261 >>> 0;
  s = String(s ?? '');
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
  return h >>> 0;
}
function mkRng(seed) {
  let a = seed >>> 0 || 1;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function safeColor(c, fallback) {
  try { const col = new THREE.Color(); col.set(c || fallback); return col; } catch { return new THREE.Color(fallback); }
}

const _m4 = new THREE.Matrix4();
export function orientTo(obj, nrm, tan) {
  const y = nrm.clone().normalize();
  let z = tan ? tan.clone() : v(0, 0, 1);
  z.addScaledVector(y, -z.dot(y));
  if (z.lengthSq() < 1e-4) {
    z = Math.abs(y.x) < 0.9 ? v(1, 0, 0) : v(0, 0, 1);
    z.addScaledVector(y, -z.dot(y));
  }
  z.normalize();
  const x = new V3().crossVectors(y, z);
  obj.quaternion.setFromRotationMatrix(_m4.makeBasis(x, y, z));
}

// Tapered tube along a Catmull-Rom curve through `points`; rf(u) = radius at u in [0,1].
export function taperTubeGeo(points, rf, tubular = 20, radial = 10) {
  const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.5);
  const frames = curve.computeFrenetFrames(tubular, false);
  const pos = [], nor = [], uv = [], idx = [];
  const P = new V3(), N = new V3();
  for (let i = 0; i <= tubular; i++) {
    const u = i / tubular;
    curve.getPointAt(u, P);
    const n0 = frames.normals[i], b0 = frames.binormals[i];
    const r = rf(u);
    for (let j = 0; j <= radial; j++) {
      const a = (j / radial) * PI * 2;
      const s = Math.sin(a), c = -Math.cos(a);
      N.set(c * n0.x + s * b0.x, c * n0.y + s * b0.y, c * n0.z + s * b0.z).normalize();
      pos.push(P.x + r * N.x, P.y + r * N.y, P.z + r * N.z);
      nor.push(N.x, N.y, N.z);
      uv.push(u, j / radial);
    }
  }
  for (let i = 0; i < tubular; i++) for (let j = 0; j < radial; j++) {
    const a = i * (radial + 1) + j, b = (i + 1) * (radial + 1) + j;
    const c = b + 1, d = a + 1;
    idx.push(a, b, d, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  return g;
}

// Point + outward normal on an ellipsoid surface. u: azimuth from +Z toward +X, phi: elevation.
function ellSurf(center, rx, ry, rz, rot, u, phi, out = 1) {
  const cp = Math.cos(phi);
  const p = v(rx * cp * Math.sin(u), ry * Math.sin(phi), rz * cp * Math.cos(u));
  const n = v(p.x / (rx * rx), p.y / (ry * ry), p.z / (rz * rz)).normalize();
  p.multiplyScalar(out);
  if (rot) { const e = new THREE.Euler(rot[0] || 0, rot[1] || 0, rot[2] || 0); p.applyEuler(e); n.applyEuler(e); }
  p.add(center);
  return { pos: p, n };
}

function stdMat(o) { return new THREE.MeshStandardMaterial(o); }

function makeMats(colors, plan) {
  const body = safeColor(colors.body, DEFAULT_COLORS.body);
  const belly = safeColor(colors.belly, DEFAULT_COLORS.belly);
  const accent = safeColor(colors.accent, DEFAULT_COLORS.accent);
  const metal = plan === 'mech';
  const hornCol = new THREE.Color('#e6dcc4').lerp(belly, 0.25);
  const plateCol = body.clone().multiplyScalar(0.6).lerp(belly, 0.12);
  const eyeCol = EYE_ACCENT_PLANS.has(plan) ? accent.clone() : new THREE.Color('#ffc83a');
  const M = {
    body: stdMat({ color: body, roughness: metal ? 0.36 : 0.72, metalness: metal ? 0.6 : 0.04 }),
    belly: stdMat({ color: belly, roughness: metal ? 0.42 : 0.85, metalness: metal ? 0.45 : 0 }),
    accent: stdMat({ color: accent, emissive: accent, emissiveIntensity: 1.7, roughness: 0.3 }),
    plate: stdMat({ color: plateCol, roughness: metal ? 0.35 : 0.5, metalness: metal ? 0.7 : 0.12 }),
    horn: stdMat({ color: hornCol, roughness: 0.42 }),
    dark: stdMat({ color: '#16130f', roughness: 0.5, metalness: metal ? 0.6 : 0 }),
    eye: stdMat({ color: '#0c0c0c', emissive: eyeCol, emissiveIntensity: 2.4, roughness: 0.2 }),
    eyeBig: stdMat({ color: eyeCol.clone().multiplyScalar(0.35), emissive: eyeCol, emissiveIntensity: 0.45, roughness: 0.12, metalness: 0.2 }),
    membrane: stdMat({ color: belly.clone().lerp(accent, 0.3), roughness: 0.6, side: THREE.DoubleSide }),
  };
  M.accent.userData.glow = true;
  M.eye.userData.eye = true;
  M.eyeBig.userData.eye = true;
  return M;
}

// ---------------------------------------------------------------- build context
class BuildCtx {
  constructor(snap, plan, tiers, colors, abilities, seed) {
    this.snap = snap; this.plan = plan; this.T = tiers; this.colors = colors; this.abilities = abilities;
    this.rng = mkRng(seed);
    this.bulk = HP_BULK[tiers.hp] * SPD_LEAN[tiers.spd];
    this.M = makeMats(colors, plan);
    this.root = new THREE.Group(); this.root.name = 'motion';
    this.rig = { limbs: [], tail: [], heads: [], jaws: [], mouths: [], powParts: [], hover: 0, koMode: 'back', extra: {} };
    this.mounts = {
      dorsal: [], plates: [], fins: [], eyes: [], horns: [], shoulders: [], tusks: [],
      crest: null, tailTip: null, chest: null, ram: null, back: null,
    };
  }
  g(parent, pos, rot) {
    const o = new THREE.Group();
    if (pos) o.position.copy(pos);
    if (rot) o.rotation.set(rot[0] || 0, rot[1] || 0, rot[2] || 0, rot[3] || 'XYZ');
    o.userData.rest = o.rotation.clone();
    parent.add(o);
    return o;
  }
  m(geo, mat, parent, pos, rot, scl) {
    const me = new THREE.Mesh(geo, mat);
    me.castShadow = true; me.receiveShadow = true;
    if (pos) me.position.copy(pos);
    if (rot) me.rotation.set(rot[0] || 0, rot[1] || 0, rot[2] || 0, rot[3] || 'XYZ');
    if (scl) me.scale.copy(scl);
    parent.add(me);
    return me;
  }
  ell(parent, rx, ry, rz, mat, pos, rot, seg = 22) {
    const geo = new THREE.SphereGeometry(1, seg, Math.max(8, Math.round(seg * 0.7)));
    geo.scale(rx, ry, rz);
    return this.m(geo, mat, parent, pos, rot);
  }
  tube(parent, pts, r0, r1, mat, opt = {}) {
    const rf = opt.rf || ((u) => lerp(r0, r1, u));
    const me = this.m(taperTubeGeo(pts, rf, opt.seg || 18, opt.rad || 10), mat, parent);
    if (opt.cap !== false && r0 > 0.02) this.ell(parent, r0, r0, r0, mat, pts[0], null, 12);
    if (opt.capEnd && r1 > 0.02) this.ell(parent, r1, r1, r1, mat, pts[pts.length - 1], null, 12);
    return me;
  }
  rbox(parent, w, h, d, r, mat, pos, rot) {
    const geo = new RoundedBoxGeometry(w, h, d, 3, Math.min(r, w / 2, h / 2, d / 2) * 0.999);
    return this.m(geo, mat, parent, pos, rot);
  }
  rock(parent, r, mat, pos, rot, sx = 1, sy = 1, sz = 1) {
    const geo = new THREE.IcosahedronGeometry(1, 1);
    const p = geo.attributes.position;
    const seed = Math.floor(this.rng() * 1e6);
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
      const k = hashStr(`${seed}:${Math.round(x * 100)}:${Math.round(y * 100)}:${Math.round(z * 100)}`);
      const f = 0.82 + (k % 1000) / 1000 * 0.36;
      p.setXYZ(i, x * f * r * sx, y * f * r * sy, z * f * r * sz);
    }
    geo.computeVertexNormals();
    return this.m(geo, mat, parent, pos, rot || [this.rng() * 3, this.rng() * 3, this.rng() * 3]);
  }
  // Animatable chain of tapered segments extending along each joint's local +Y.
  chain(parent, o) {
    const n = o.n || 6, len = o.len || 1, r0 = o.r0 ?? 0.5, r1 = o.r1 ?? 0.1, mat = o.mat || this.M.body;
    const curl = (key, i) => (typeof o[key] === 'function' ? o[key](i) : (Array.isArray(o[key]) ? (o[key][i - 1] ?? 0) : (o[key] || 0)));
    const joints = [];
    let p = parent;
    for (let i = 0; i < n; i++) {
      const j = new THREE.Group();
      if (i === 0) {
        if (o.pos) j.position.copy(o.pos);
        if (o.rot) j.rotation.set(o.rot[0] || 0, o.rot[1] || 0, o.rot[2] || 0, o.rot[3] || 'XYZ');
      } else {
        j.position.set(0, len, 0);
        j.rotation.set(curl('curlX', i), curl('curlY', i), curl('curlZ', i));
      }
      j.userData.rest = j.rotation.clone();
      const ra = lerp(r0, r1, i / n), rb = lerp(r0, r1, (i + 1) / n);
      const geo = new THREE.CylinderGeometry(rb, ra, len * 1.02, o.radial || 14, 1, true);
      geo.translate(0, len / 2, 0);
      this.m(geo, mat, j);
      this.ell(j, ra, ra, ra, mat, null, null, o.radial || 14);
      if (o.belly) {
        const bz = (o.bellyDir || 1) * ra * 0.62;
        this.ell(j, ra * 0.66, len * 0.62, ra * 0.42, o.belly, v(0, len * 0.5, bz));
      }
      if (o.dorsal) {
        const dz = -(o.bellyDir || 1);
        this.mount('dorsal', j, v(0, len * 0.5, dz * rb * 0.92), v(0, 0, dz), v(0, 1, 0), lerp(1, 0.5, i / n) * (o.dorsalScale || 1));
      }
      p.add(j); joints.push(j); p = j;
    }
    const tip = new THREE.Group();
    tip.position.set(0, len, 0);
    p.add(tip);
    joints.tip = tip;
    joints.tipRadius = r1;
    return joints;
  }
  mount(list, parent, pos, n, t, s = 1) {
    const mt = { p: parent, pos: pos.clone(), n: n.clone().normalize(), t: t ? t.clone() : null, s };
    if (Array.isArray(this.mounts[list])) this.mounts[list].push(mt); else this.mounts[list] = mt;
    return mt;
  }
  pow(obj) { this.rig.powParts.push(obj); return obj; }
  claw(parent, pos, dir, len = 0.6, r = 0.14, tan) {
    const o = this.g(parent, pos);
    orientTo(o, dir, tan || v(0, 0, 1));
    this.tube(o, [v(0, 0, 0), v(0, len * 0.55, len * 0.12), v(0, len, len * 0.45)], r, 0.012, this.M.horn, { seg: 8, rad: 7 });
    return this.pow(o);
  }
  eye(parent, pos, r, mat) {
    const g = this.g(parent, pos);
    this.ell(g, r, r, r, mat || this.M.eye, null, null, 14);
    return g;
  }
  limb(rec) { this.rig.limbs.push(rec); return rec; }
}

// Attach `obj` at a mount (obj's +Y = mount normal, +Z = mount tangent).
function attach(mt, obj, scale = 1) {
  if (!mt) return obj;
  obj.position.copy(mt.pos);
  orientTo(obj, mt.n, mt.t || v(0, 0, -1));
  obj.scale.multiplyScalar(scale * (mt.s || 1));
  mt.p.add(obj);
  return obj;
}

// ---------------------------------------------------------------- reusable part factories (+Y = outward)
const PART = {
  crystal(c, mat) {
    const g = new THREE.Group();
    const mk = (r, h, x, z, tilt) => {
      const geo = new THREE.OctahedronGeometry(1, 0); geo.scale(r, h, r); geo.translate(0, h * 0.75, 0);
      c.m(geo, mat || c.M.accent, g, v(x, 0, z), [tilt, 0, -tilt * 0.7]);
    };
    mk(0.3, 0.95, 0, 0, 0);
    mk(0.2, 0.6, 0.22, 0.12, 0.45);
    mk(0.18, 0.5, -0.2, -0.1, -0.5);
    return g;
  },
  plate(c, mat) {
    const g = new THREE.Group();
    const geo = new THREE.SphereGeometry(1, 16, 10); geo.scale(0.75, 0.2, 0.95);
    c.m(geo, mat || c.M.plate, g, v(0, 0.04, 0));
    return g;
  },
  spike(c, mat, h = 1) {
    const g = new THREE.Group();
    const geo = new THREE.ConeGeometry(0.22, h, 8); geo.translate(0, h / 2, 0);
    c.m(geo, mat || c.M.horn, g);
    return g;
  },
  fin(c, mat, h = 1.2) {
    const s = new THREE.Shape();
    s.moveTo(0.55, 0);
    s.bezierCurveTo(0.45, 0.5 * h, 0.1, 0.9 * h, -0.45, 1.05 * h);
    s.bezierCurveTo(-0.3, 0.6 * h, -0.55, 0.25 * h, -0.85, 0);
    s.lineTo(0.55, 0);
    const geo = new THREE.ShapeGeometry(s, 10); geo.rotateY(-PI / 2);
    const g = new THREE.Group();
    c.m(geo, mat || c.M.membrane, g);
    return g;
  },
  horn(c, mat, len = 1.4, back = 1) {
    const g = new THREE.Group();
    c.tube(g, [v(0, 0, 0), v(0, len * 0.45, -0.1 * back), v(0, len * 0.8, -0.35 * back), v(0, len, -0.75 * back)], 0.26, 0.015, mat || c.M.horn, { seg: 10, rad: 8 });
    return g;
  },
  eye(c, r = 0.18) {
    const g = new THREE.Group();
    c.ell(g, r * 1.25, r * 0.7, r * 1.25, c.M.dark, v(0, -r * 0.2, 0), null, 10);
    c.ell(g, r, r, r, c.M.eye, v(0, 0, 0), null, 12);
    return g;
  },
};

// Reptilian head used by biped, hydra, serpent. Origin at back of skull, facing +Z.
function reptileHead(c, parent, { len = 1.4, w = 1.0, mat, teeth = true } = {}) {
  const M = c.M; mat = mat || M.body;
  c.ell(parent, w, w * 0.82, len * 0.85, mat, v(0, 0.12, len * 0.45));
  c.ell(parent, w * 0.72, w * 0.5, len * 0.82, mat, v(0, 0.02, len * 1.2));
  for (const s of [-1, 1]) {
    c.ell(parent, w * 0.32, w * 0.17, len * 0.42, M.plate, v(s * w * 0.5, w * 0.56, len * 0.78), [0, s * 0.2, 0]);
    c.eye(parent, v(s * w * 0.62, w * 0.4, len * 0.84), w * 0.15);
    c.ell(parent, w * 0.07, w * 0.05, w * 0.07, M.dark, v(s * w * 0.24, w * 0.28, len * 1.95), null, 8);
  }
  const jaw = c.g(parent, v(0, -w * 0.28, len * 0.32), [0.05, 0, 0]);
  c.ell(jaw, w * 0.64, w * 0.24, len * 0.98, M.belly, v(0, -0.06, len * 0.76));
  if (teeth) {
    const tg = new THREE.ConeGeometry(w * 0.07, w * 0.26, 6);
    tg.rotateX(PI);
    for (const s of [-1, 1]) for (let i = 0; i < 5; i++) {
      const z = len * (0.85 + i * 0.22);
      c.m(tg, M.horn, parent, v(s * w * (0.52 - i * 0.05), -w * 0.33, z));
    }
  }
  const mouth = new THREE.Object3D(); mouth.position.set(0, -w * 0.18, len * 1.95); parent.add(mouth);
  for (const s of [-1, 1]) {
    c.mount('eyes', parent, v(s * w * 0.48, w * 0.55, len * 1.2), v(s * 0.7, 0.7, 0.3));
    c.mount('eyes', parent, v(s * w * 0.72, w * 0.25, len * 0.5), v(s, 0.2, 0.1));
    c.mount('horns', parent, v(s * w * 0.5, w * 0.7, len * 0.32), v(s * 0.35, 1, -0.35));
    c.mount('horns', parent, v(s * w * 0.3, w * 0.55, len * 1.5), v(s * 0.2, 1, 0.3));
    c.mount('tusks', jaw, v(s * w * 0.45, 0.05, len * 1.3), v(s * 0.25, 1, 0.35));
  }
  c.mount('crest', parent, v(0, w * 0.88, len * 0.3), v(0, 1, -0.15), v(0, 0, -1));
  c.mount('ram', parent, v(0, w * 0.45, len * 1.55), v(0, 0.7, 0.75));
  return { jaw, mouth };
}

function registerHead(c, hp, jaw, mouth, main = true) {
  hp.userData.rest = hp.rotation.clone();
  hp.userData.ph = c.rig.heads.length * 1.7;
  c.rig.heads.push(hp);
  if (jaw) { jaw.userData.rest = jaw.rotation.clone(); c.rig.jaws.push(jaw); }
  if (mouth) c.rig.mouths.push(mouth);
  if (main) { c.rig.headPivot = hp; c.rig.jaw = jaw; c.rig.mouth = mouth; }
}

// ---------------------------------------------------------------- body plans
function buildBiped(c) {
  const { M } = c, B = c.bulk;
  const trunk = c.g(c.root, v(0, 4.3, -0.3)); c.rig.trunk = trunk;
  const torso = c.g(trunk); c.rig.body = torso;
  c.ell(torso, 2.1, 3.0, 1.9, M.body, v(0, 2.0, 0.45), [0.32, 0, 0]);
  c.ell(torso, 1.5, 2.6, 1.05, M.belly, v(0, 1.75, 1.45), [0.3, 0, 0]);
  c.ell(torso, 2.05, 1.6, 2.0, M.body, v(0, 0.25, -0.45));
  c.mount('chest', torso, v(0, 2.5, 2.05), v(0, 0.25, 1));
  // dorsal row on the back
  for (let k = 0; k < 6; k++) {
    const { pos, n } = ellSurf(v(0, 2.0, 0.45), 2.1, 3.0, 1.9, [0.32, 0, 0], PI, -0.55 + k * 0.32, 0.96);
    c.mount('dorsal', torso, pos, n, v(0, -0.6, -1), 1.15 - k * 0.06);
  }
  for (const s of [-1, 1]) {
    const p = ellSurf(v(0, 2.0, 0.45), 2.1, 3.0, 1.9, [0.32, 0, 0], s * 1.9, 0.25, 0.98);
    c.mount('plates', torso, p.pos, p.n, v(0, 1, 0), 1.3);
    const p2 = ellSurf(v(0, 2.0, 0.45), 2.1, 3.0, 1.9, [0.32, 0, 0], s * 2.3, -0.4, 0.98);
    c.mount('plates', torso, p2.pos, p2.n, v(0, 1, 0), 1.2);
    const f = ellSurf(v(0, 2.0, 0.45), 2.1, 3.0, 1.9, [0.32, 0, 0], s * 2.6, 0.5, 0.98);
    c.mount('fins', torso, f.pos, f.n, v(0, 0.3, -1), 1.0);
  }
  // base dorsal spikes (classic silhouette)
  for (const mt of c.mounts.dorsal) attach(mt, PART.spike(c, M.plate, 0.9), 0.9);
  // neck + head
  c.tube(trunk, [v(0, 3.5, 0.95), v(0, 4.3, 1.65), v(0, 4.85, 2.25)], 1.05, 0.72, M.body);
  const hp = c.g(trunk, v(0, 4.85, 2.25), [0.08, 0, 0]);
  const H = reptileHead(c, hp, { len: 1.4, w: 1.0 });
  registerHead(c, hp, H.jaw, H.mouth);
  // legs
  for (const s of [-1, 1]) {
    const hip = c.g(trunk, v(s * 1.45 * B, 0.2, -0.3));
    c.ell(hip, 0.95, 1.75, 1.2, M.body, v(0, -1.0, 0.25), [-0.25, 0, 0]);
    const knee = c.g(hip, v(0, -2.3, 0.55));
    c.tube(knee, [v(0, 0, 0), v(0, -0.9, -0.35), v(0, -1.75, -0.55)], 0.72, 0.46, M.body);
    const ankle = c.g(knee, v(0, -1.75, -0.55));
    c.ell(ankle, 0.72, 0.36, 1.15, M.body, v(0, -0.12, 0.45));
    for (const x of [-0.4, 0, 0.4]) c.claw(ankle, v(x, -0.15, 1.4 - Math.abs(x) * 0.3), v(0, 0.1, 1), 0.55, 0.13, v(0, -1, 0));
    c.mount('plates', hip, v(s * 0.85, -0.9, 0.3), v(s, 0.1, 0.2), null, 1.1);
    c.limb({ type: 'leg', pivot: hip, knee, phase: s > 0 ? 0 : PI });
  }
  // arms
  for (const s of [-1, 1]) {
    const sh = c.g(trunk, v(s * 1.75 * B, 3.3, 1.45), [0.1, 0, s * 0.12]);
    c.ell(sh, 0.46, 0.95, 0.46, M.body, v(0, -0.7, 0.15), [0.3, 0, 0]);
    const el = c.g(sh, v(0, -1.5, 0.45), [-0.95, 0, 0]);
    c.ell(el, 0.38, 0.8, 0.38, M.body, v(0, -0.65, 0));
    const hand = c.g(el, v(0, -1.35, 0));
    for (const x of [-0.18, 0, 0.18]) c.claw(hand, v(x, 0, 0.1), v(0, -1, 0.5), 0.45, 0.09);
    c.mount('shoulders', sh, v(0, 0.2, -0.1), v(s * 0.3, 1, -0.2));
    c.limb({ type: 'arm', pivot: sh, elbow: el, phase: s, reach: -1.2, elbowReach: 0.7 });
    c.rig.extra[s < 0 ? 'handL' : 'handR'] = hand;
  }
  // tail
  const tail = c.chain(trunk, { n: 8, len: 1.25, r0: 1.25, r1: 0.12, pos: v(0, 0.5, -1.7), rot: [-(PI / 2 + 0.35), 0, 0], curlX: 0.06, belly: M.belly, bellyDir: -1, dorsal: true, dorsalScale: 0.9 });
  for (let i = 0; i < 6; i++) attach(c.mounts.dorsal[6 + i], PART.spike(c, M.plate, 0.8), 0.7);
  c.rig.tail = tail;
  c.limb({ type: 'tail', joints: tail, ampX: 0.03, ampZ: 0.09, speed: 1.3 });
  c.mount('tailTip', tail.tip, v(0, 0, 0), v(0, 1, 0), v(0, 0, 1));
  c.mount('back', torso, v(0, 2.6, -1.2), v(0, 0.4, -1));
}

function buildApe(c) {
  const { M } = c, B = c.bulk;
  M.body.roughness = 0.95; M.belly.roughness = 0.9;
  const trunk = c.g(c.root, v(0, 4.6, -0.6)); c.rig.trunk = trunk;
  const torso = c.g(trunk); c.rig.body = torso;
  c.ell(torso, 2.6, 2.9, 2.15, M.body, v(0, 2.4, 0.55), [0.5, 0, 0]);
  c.ell(torso, 1.85, 1.9, 1.15, M.belly, v(0, 2.3, 1.95), [0.42, 0, 0]);
  c.ell(torso, 2.0, 1.7, 1.8, M.body, v(0, 0.3, -0.25));
  for (const s of [-1, 1]) c.ell(torso, 1.5, 1.35, 1.5, M.body, v(s * 2.25, 3.9, 1.15));
  c.mount('chest', torso, v(0, 2.6, 2.9), v(0, 0.2, 1));
  for (let k = 0; k < 5; k++) {
    const { pos, n } = ellSurf(v(0, 2.4, 0.55), 2.6, 2.9, 2.15, [0.5, 0, 0], PI, -0.2 + k * 0.28, 0.97);
    c.mount('dorsal', torso, pos, n, v(0, -0.5, -1), 1.1);
  }
  for (const s of [-1, 1]) {
    c.mount('plates', torso, v(s * 2.6, 4.6, 1.2), v(s * 0.5, 1, 0), null, 1.5);
    c.mount('shoulders', torso, v(s * 2.2, 5.1, 0.9), v(s * 0.2, 1, -0.1));
    const f = ellSurf(v(0, 2.4, 0.55), 2.6, 2.9, 2.15, [0.5, 0, 0], s * 2.4, 0.3, 0.98);
    c.mount('fins', torso, f.pos, f.n, v(0, 0.4, -1));
    const p2 = ellSurf(v(0, 2.4, 0.55), 2.6, 2.9, 2.15, [0.5, 0, 0], s * 2.0, -0.35, 0.98);
    c.mount('plates', torso, p2.pos, p2.n, null, 1.3);
  }
  // head, low between the shoulders
  const hp = c.g(trunk, v(0, 4.4, 2.75));
  c.ell(hp, 1.05, 1.0, 1.0, M.body, v(0, 0.2, 0.15));
  c.ell(hp, 0.82, 0.6, 0.62, M.belly, v(0, -0.28, 0.8));
  c.ell(hp, 1.02, 0.24, 0.42, M.dark, v(0, 0.52, 0.72));
  for (const s of [-1, 1]) {
    c.eye(hp, v(s * 0.38, 0.3, 0.98), 0.13);
    c.ell(hp, 0.22, 0.3, 0.12, M.body, v(s * 1.0, 0.25, 0.1));
    c.mount('eyes', hp, v(s * 0.62, 0.42, 0.85), v(s * 0.5, 0.5, 1));
    c.mount('horns', hp, v(s * 0.6, 0.85, 0.3), v(s * 0.5, 1, 0));
    c.mount('horns', hp, v(s * 0.35, 1.0, -0.2), v(s * 0.3, 1, -0.4));
  }
  const jaw = c.g(hp, v(0, -0.45, 0.35));
  c.ell(jaw, 0.7, 0.3, 0.62, M.belly, v(0, -0.1, 0.45));
  const fang = new THREE.ConeGeometry(0.07, 0.3, 6);
  for (const s of [-1, 1]) c.m(fang, M.horn, jaw, v(s * 0.38, 0.12, 0.85));
  const mouth = new THREE.Object3D(); mouth.position.set(0, -0.35, 1.45); hp.add(mouth);
  registerHead(c, hp, jaw, mouth);
  c.mount('crest', hp, v(0, 1.15, 0.0), v(0, 1, -0.2), v(0, 0, -1));
  c.mount('ram', hp, v(0, 0.7, 0.9), v(0, 0.6, 0.8));
  for (const s of [-1, 1]) c.mount('tusks', jaw, v(s * 0.42, 0.05, 0.85), v(s * 0.3, 1, 0.3));
  // long arms, knuckle-walking
  for (const s of [-1, 1]) {
    const sh = c.g(trunk, v(s * 2.9 * B, 3.7, 1.3), [-0.15, 0, s * 0.18]);
    c.ell(sh, 0.88, 1.9, 0.88, M.body, v(0, -1.55, 0));
    const el = c.g(sh, v(0, -3.25, 0), [-0.22, 0, -s * 0.08]);
    c.ell(el, 0.92, 1.65, 0.92, M.body, v(0, -1.4, 0));
    const hand = c.g(el, v(0, -2.95, 0.05));
    c.pow(hand);
    c.ell(hand, 0.98, 0.78, 1.05, M.body, v(0, -0.25, 0.15));
    for (let k = 0; k < 4; k++) c.ell(hand, 0.24, 0.22, 0.26, M.dark, v(-0.5 + k * 0.33, -0.7, 0.55), null, 10);
    c.mount('plates', el, v(s * 0.75, -1.2, 0), v(s, 0, 0.3), null, 1.2);
    c.limb({ type: 'arm', pivot: sh, elbow: el, phase: s, reach: -1.25, elbowReach: 0.25, vicRaise: -2.6 });
    c.rig.extra[s < 0 ? 'handL' : 'handR'] = hand;
  }
  for (const s of [-1, 1]) {
    const hip = c.g(trunk, v(s * 1.5 * B, 0, -0.3));
    c.ell(hip, 0.95, 1.4, 1.0, M.body, v(0, -1.0, 0.3), [-0.3, 0, 0]);
    const knee = c.g(hip, v(0, -2.2, 0.5));
    c.ell(knee, 0.72, 1.1, 0.78, M.body, v(0, -0.9, -0.1));
    c.ell(knee, 0.8, 0.34, 1.2, M.dark, v(0, -2.05, 0.3));
    c.limb({ type: 'leg', pivot: hip, knee, phase: s > 0 ? 0 : PI });
  }
  c.mount('tailTip', torso, v(0, 0.5, -2.0), v(0, 0.2, -1), v(0, 1, 0));
  c.mount('back', torso, v(0, 3.2, -1.2), v(0, 0.6, -1));
}

function buildCephalopod(c) {
  const { M } = c, B = c.bulk;
  M.body.roughness = 0.45; M.belly.roughness = 0.55;
  c.rig.koMode = 'flop';
  const trunk = c.g(c.root, v(0, 0, 0)); c.rig.trunk = trunk;
  const torso = c.g(trunk, v(0, 3.0, 0)); c.rig.body = torso;
  const hp = c.g(torso, v(0, 0.6, 0));
  // mantle (tilted back) + head
  const prof = [[0, 0], [1.5, 0.2], [2.2, 1.2], [2.45, 2.6], [2.25, 4.0], [1.6, 5.0], [0.7, 5.6], [0, 5.8]].map(p => new THREE.Vector2(p[0], p[1]));
  const lath = new THREE.LatheGeometry(prof, 30);
  c.m(lath, M.body, hp, v(0, 0.8, -0.7), [-0.5, 0, 0]);
  c.ell(hp, 2.0, 1.7, 1.9, M.body, v(0, 0, 0.5));
  c.ell(hp, 1.65, 1.0, 1.55, M.belly, v(0, -0.75, 0.75));
  for (const s of [-1, 1]) {
    c.ell(hp, 0.62, 0.62, 0.62, M.belly, v(s * 1.5, 0.35, 1.25), null, 16);
    c.ell(hp, 0.4, 0.4, 0.4, M.eyeBig, v(s * 1.78, 0.38, 1.5), null, 14);
    c.ell(hp, 0.14, 0.3, 0.1, M.dark, v(s * 1.95, 0.38, 1.72), [0, s * 0.6, 0], 8);
    c.mount('eyes', hp, v(s * 1.1, 1.1, 1.5), v(s * 0.4, 0.6, 1));
    c.mount('eyes', hp, v(s * 1.9, -0.1, 0.9), v(s, 0, 0.4));
    c.mount('horns', hp, v(s * 1.2, 1.5, 0.6), v(s * 0.5, 1, 0.2));
    c.mount('fins', hp, v(s * 1.6, 4.4, -2.6), v(s, 0.2, -0.3), v(0, 1, -0.5), 1.6);
  }
  // mantle top mounts
  for (let k = 0; k < 6; k++) {
    const t = k / 5;
    const y = 1.6 + t * 4.0, zb = -0.5 - t * 2.4;
    c.mount('dorsal', hp, v(0, y, zb - 1.9 + t * 1.2), v(0, 0.3 + t * 0.7, -1), v(0, 1, 0), 1.1);
  }
  for (const s of [-1, 1]) for (let k = 0; k < 3; k++) {
    c.mount('plates', hp, v(s * 2.15, 1.8 + k * 1.2, -0.9 - k * 0.7), v(s, 0.3, -0.2), null, 1.4);
  }
  const beak = c.g(hp, v(0, -1.0, 1.8), [0.15, 0, 0]);
  const bg = new THREE.ConeGeometry(0.35, 0.8, 8); bg.rotateX(PI / 2);
  c.m(bg, M.dark, beak, v(0, 0, 0.3));
  const mouth = new THREE.Object3D(); mouth.position.set(0, -0.9, 2.4); hp.add(mouth);
  registerHead(c, hp, beak, mouth);
  c.mount('crest', hp, v(0, 6.2, -3.3), v(0, 0.6, -1), v(0, 1, 0));
  c.mount('ram', hp, v(0, 0.5, 2.3), v(0, 0.3, 1));
  c.mount('chest', hp, v(0, -0.3, 2.2), v(0, 0, 1));
  for (const s of [-1, 1]) c.mount('tusks', hp, v(s * 0.5, -1.2, 2.0), v(s * 0.2, -1, 0.4));
  // 8 tentacles
  for (let k = 0; k < 8; k++) {
    const a = (k / 8) * PI * 2 + PI / 8;
    const front = Math.cos(a) > 0.5;
    const ch = c.chain(torso, {
      n: 9, len: front ? 1.05 : 0.95, r0: 0.6, r1: 0.06,
      pos: v(Math.sin(a) * 1.25 * B, -0.7, Math.cos(a) * 1.25 + 0.45),
      rot: front ? [0.85, a, 0, 'YXZ'] : [PI / 2 + 0.55, a, 0, 'YXZ'],
      curlX: front ? [0.1, 0.12, 0.18, 0.25, 0.35, 0.45, 0.5, 0.5] : [0, 0.05, -0.15, -0.25, -0.35, -0.45, -0.5, -0.55],
      belly: M.belly, bellyDir: 1, radial: 12,
    });
    c.limb({ type: 'tentacle', joints: ch, phase: k * 0.9, ampX: 0.09, ampZ: 0.08, speed: 1.3, reach: front });
    if (k === 0) c.mount('tailTip', ch.tip, v(0, 0, 0), v(0, 1, 0));
    if (front) c.rig.extra[Math.sin(a) < 0 ? 'handL' : 'handR'] = ch.tip;
  }
  c.mount('back', hp, v(0, 3.2, -2.3), v(0, 0.5, -1));
  c.mounts.tailTip = c.mounts.tailTip || c.mount('tailTip', hp, v(0, 6.0, -3.4), v(0, 0.3, -1));
  // the crest of a mantle reads better as the "tail tip" for clubs/stingers
  c.mounts.tailTip = { p: hp, pos: v(0, 5.9, -3.5), n: v(0, 0.4, -1).normalize(), t: v(0, 1, 0), s: 1 };
  for (const s of [-1, 1]) c.mount('shoulders', hp, v(s * 1.6, 1.6, -0.4), v(s * 0.5, 1, 0));
}

function wingTexture(colors, rng, hind) {
  const cv = document.createElement('canvas'); cv.width = cv.height = 256;
  const g = cv.getContext('2d');
  const body = safeColor(colors.body, '#444').getStyle();
  const belly = safeColor(colors.belly, '#aaa').getStyle();
  const accent = safeColor(colors.accent, '#f0f').getStyle();
  const grd = g.createRadialGradient(0, 128, 10, 0, 128, 300);
  grd.addColorStop(0, body); grd.addColorStop(0.55, belly); grd.addColorStop(1, body);
  g.fillStyle = grd; g.fillRect(0, 0, 256, 256);
  g.strokeStyle = 'rgba(0,0,0,0.35)'; g.lineWidth = 2;
  for (let i = 0; i < 9; i++) {
    const a = -1.1 + i * 0.27;
    g.beginPath(); g.moveTo(0, 128);
    g.quadraticCurveTo(110 * Math.cos(a), 128 + 110 * Math.sin(a), 270 * Math.cos(a), 128 + 270 * Math.sin(a)); g.stroke();
  }
  const ex = hind ? 140 : 160, ey = hind ? 150 : 115, er = hind ? 34 : 44;
  const ring = (r, col) => { g.fillStyle = col; g.beginPath(); g.arc(ex, ey, r, 0, PI * 2); g.fill(); };
  ring(er, belly); ring(er * 0.78, accent); ring(er * 0.48, '#111'); ring(er * 0.18, '#fff');
  g.fillStyle = 'rgba(0,0,0,0.55)';
  g.lineWidth = 16; g.strokeStyle = 'rgba(20,10,5,0.55)';
  g.beginPath(); g.arc(0, 128, 238, -1.2, 1.2); g.stroke();
  g.fillStyle = belly;
  for (let i = 0; i < 9; i++) { const a = -1.0 + i * 0.25; g.beginPath(); g.arc(238 * Math.cos(a), 128 + 238 * Math.sin(a), 4, 0, PI * 2); g.fill(); }
  const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace;
  // emissive: just the eyespot
  const ce = document.createElement('canvas'); ce.width = ce.height = 128;
  const e = ce.getContext('2d'); e.fillStyle = '#000'; e.fillRect(0, 0, 128, 128);
  e.fillStyle = accent; e.beginPath(); e.arc(ex / 2, ey / 2, er * 0.39, 0, PI * 2); e.fill();
  e.fillStyle = '#000'; e.beginPath(); e.arc(ex / 2, ey / 2, er * 0.24, 0, PI * 2); e.fill();
  const etex = new THREE.CanvasTexture(ce); etex.colorSpace = THREE.SRGBColorSpace;
  return { tex, etex };
}

function wingGeo(shapeFn) {
  const s = new THREE.Shape(); shapeFn(s);
  const geo = new THREE.ShapeGeometry(s, 20);
  geo.computeBoundingBox();
  const bb = geo.boundingBox, p = geo.attributes.position, uv = geo.attributes.uv;
  for (let i = 0; i < p.count; i++) {
    // u along span (root -> tip), v along chord; texture radial origin at (0, 0.5)
    uv.setXY(i, (p.getX(i) - bb.min.x) / (bb.max.x - bb.min.x), 1 - (p.getY(i) - bb.min.y) / (bb.max.y - bb.min.y));
  }
  geo.rotateX(-PI / 2); // shape y (backward) -> -z
  return geo;
}

function buildMoth(c) {
  const { M } = c, B = c.bulk;
  M.body.roughness = 1; M.belly.roughness = 1;
  c.rig.hover = 3.0; c.rig.koMode = 'side';
  const trunk = c.g(c.root, v(0, 5.2, 0)); c.rig.trunk = trunk;
  const torso = c.g(trunk); c.rig.body = torso;
  c.ell(torso, 1.25, 1.3, 1.45, M.body, v(0, 0.3, 0.4));
  c.ell(torso, 1.15, 0.85, 0.75, M.belly, v(0, 0.75, 1.5), [0.4, 0, 0]);
  const abd = c.g(torso, v(0, -0.2, -0.8), [-0.38, 0, 0]);
  c.ell(abd, 1.05, 1.0, 2.7, M.body, v(0, 0, -1.9));
  const tor = new THREE.TorusGeometry(1, 0.06, 6, 24);
  for (let k = 0; k < 5; k++) {
    const z = -0.5 - k * 0.65, zz = (z + 1.9) / 2.7, r = Math.sqrt(Math.max(0.05, 1 - zz * zz));
    const me = c.m(tor, M.belly, abd, v(0, 0, z)); me.scale.set(1.0 * r, 0.96 * r, 1);
  }
  c.mount('tailTip', abd, v(0, 0, -4.55), v(0, -0.2, -1), v(0, 1, 0));
  c.mount('chest', torso, v(0, 0.0, 1.8), v(0, -0.2, 1));
  for (let k = 0; k < 5; k++) c.mount('dorsal', k < 2 ? torso : abd, k < 2 ? v(0, 1.45, 0.6 - k * 0.8) : v(0, 0.95, -0.6 - (k - 2) * 0.9), v(0, 1, -0.1), v(0, 0, -1), 0.9);
  for (const s of [-1, 1]) {
    c.mount('plates', torso, v(s * 1.1, 0.6, 0.6), v(s, 0.4, 0.1), null, 1.1);
    c.mount('plates', abd, v(s * 0.95, 0.2, -1.6), v(s, 0.4, 0), null, 1.1);
    c.mount('shoulders', torso, v(s * 0.8, 1.35, 0.9), v(s * 0.3, 1, 0.1));
    c.mount('fins', abd, v(s * 0.9, 0, -3.2), v(s, 0.1, 0), v(0, 0, -1), 0.9);
  }
  // head
  const hp = c.g(trunk, v(0, 0.55, 2.0));
  c.ell(hp, 0.8, 0.75, 0.72, M.body, v(0, 0, 0.3));
  for (const s of [-1, 1]) {
    c.ell(hp, 0.56, 0.62, 0.56, M.eyeBig, v(s * 0.62, 0.1, 0.5), null, 16);
    const ant = [v(s * 0.25, 0.55, 0.6), v(s * 0.8, 1.6, 1.0), v(s * 1.4, 2.6, 0.75), v(s * 1.9, 3.2, 0.15)];
    c.tube(hp, ant, 0.08, 0.03, M.body, { seg: 14, rad: 6 });
    const curve = new THREE.CatmullRomCurve3(ant);
    const barb = new THREE.ConeGeometry(0.05, 0.75, 4); barb.translate(0, 0.37, 0);
    for (let i = 2; i < 13; i++) {
      const pnt = curve.getPointAt(i / 13);
      for (const side of [-1, 1]) c.m(barb, M.body, hp, pnt, [0, 0, side * 1.1 - s * 0.2]).scale.setScalar(1 - i / 18);
    }
    c.mount('eyes', hp, v(s * 0.35, 0.6, 0.75), v(s * 0.3, 1, 0.4));
    c.mount('eyes', hp, v(s * 0.3, -0.3, 0.9), v(s * 0.3, -0.3, 1));
    c.mount('horns', hp, v(s * 0.4, 0.65, 0.2), v(s * 0.3, 1, -0.2));
  }
  const jaw = c.g(hp, v(0, -0.45, 0.6));
  c.tube(jaw, [v(0, 0, 0), v(0, -0.5, 0.35), v(0, -0.65, 0.15), v(0, -0.5, 0.0)], 0.1, 0.04, M.dark, { seg: 10, rad: 6 });
  const mouth = new THREE.Object3D(); mouth.position.set(0, -0.3, 1.05); hp.add(mouth);
  registerHead(c, hp, jaw, mouth);
  c.mount('crest', hp, v(0, 0.75, 0.2), v(0, 1, -0.2), v(0, 0, -1));
  c.mount('ram', hp, v(0, 0.5, 0.85), v(0, 0.5, 1));
  for (const s of [-1, 1]) c.mount('tusks', hp, v(s * 0.3, -0.45, 0.85), v(s * 0.3, -1, 0.5));
  // wings
  const fore = wingGeo(s => { s.moveTo(0, -0.3); s.bezierCurveTo(2, -1.2, 5, -1.5, 6.8, -0.6); s.bezierCurveTo(7.1, 0.7, 5.6, 2.3, 3.8, 2.7); s.bezierCurveTo(2.5, 2.5, 1.0, 1.4, 0, 0.8); s.lineTo(0, -0.3); });
  const hind = wingGeo(s => { s.moveTo(0, 0.6); s.bezierCurveTo(1.5, 0.8, 4.7, 1.4, 4.9, 3.1); s.bezierCurveTo(5.0, 4.7, 3.3, 5.8, 2.0, 5.4); s.bezierCurveTo(1.0, 4.3, 0.3, 2.4, 0, 1.2); s.lineTo(0, 0.6); });
  const wtF = wingTexture(c.colors, c.rng, false), wtH = wingTexture(c.colors, c.rng, true);
  const mk = (w) => stdMat({ map: w.tex, emissiveMap: w.etex, emissive: new THREE.Color('#ffffff'), emissiveIntensity: 1.2, roughness: 0.8, side: THREE.DoubleSide });
  const mF = mk(wtF), mH = mk(wtH);
  for (const s of [-1, 1]) {
    const wf = c.g(trunk, v(s * 0.9 * B, 0.8, 0.7), [0, s * -0.15, s * 0.8]);
    const meF = c.m(fore, mF, wf); meF.scale.x = s;
    const wh = c.g(trunk, v(s * 0.85 * B, 0.55, 0.0), [0, s * 0.1, s * 0.6]);
    const meH = c.m(hind, mH, wh); meH.scale.x = s;
    c.limb({ type: 'wing', pivot: wf, side: s, amp: 0.42, speed: 5.5, phase: 0 });
    c.limb({ type: 'wing', pivot: wh, side: s, amp: 0.36, speed: 5.5, phase: -0.4 });
    c.rig.extra[s < 0 ? 'handL' : 'handR'] = wf;
  }
  // six dangling legs
  for (const s of [-1, 1]) for (let k = 0; k < 3; k++) {
    const hip = c.g(torso, v(s * 0.7, -0.6, 1.0 - k * 0.6), [0.2 - k * 0.25, 0, s * 0.5]);
    c.tube(hip, [v(0, 0, 0), v(0, -1.1, 0.3), v(0, -2.0, 0.0), v(0, -2.8, -0.4)], 0.13, 0.05, M.body, { seg: 10, rad: 6 });
    c.limb({ type: 'leg', pivot: hip, phase: k + s });
  }
  c.mount('back', torso, v(0, 1.4, -0.2), v(0, 1, -0.3));
}

function buildTurtle(c) {
  const { M } = c, B = c.bulk;
  c.rig.koMode = 'side';
  const trunk = c.g(c.root, v(0, 0, 0)); c.rig.trunk = trunk;
  const torso = c.g(trunk, v(0, 3.6, 0)); c.rig.body = torso;
  const dome = new THREE.SphereGeometry(1, 36, 18, 0, PI * 2, 0, PI / 2); dome.scale(3.7, 3.0, 4.4);
  c.m(dome, M.body, torso);
  const rim = new THREE.TorusGeometry(1, 0.09, 8, 48); rim.rotateX(PI / 2);
  c.m(rim, M.belly, torso).scale.set(3.72, 1.6, 4.42);
  c.ell(torso, 3.4, 1.0, 4.1, M.belly, v(0, -0.25, 0));
  // hex scutes
  const rings = [[1.35, 1, 0], [0.9, 6, 0.3], [0.45, 10, 0.1]];
  const hex = new THREE.CylinderGeometry(0.95, 1.08, 0.26, 6);
  for (const [phi, cnt, off] of rings) for (let k = 0; k < cnt; k++) {
    const u = (k / cnt) * PI * 2 + off;
    const { pos, n } = ellSurf(v(0, 0, 0), 3.7, 3.0, 4.4, null, u, phi, 0.98);
    const o = c.g(torso, pos); orientTo(o, n, v(0, 0, 1));
    c.m(hex, M.plate, o).scale.setScalar(cnt === 1 ? 1.25 : cnt === 6 ? 1.12 : 0.95);
    c.mount('plates', torso, pos.clone().addScaledVector(n, 0.15), n, null, 1.1);
  }
  for (let k = 0; k < 5; k++) {
    const { pos, n } = ellSurf(v(0, 0, 0), 3.7, 3.0, 4.4, null, PI, 1.45 - k * 0.24, 1.03);
    c.mount('dorsal', torso, pos, n, v(0, 0, -1), 1.2);
    if (k > 0) { const f = ellSurf(v(0, 0, 0), 3.7, 3.0, 4.4, null, 0, 1.45 - k * 0.24, 1.03); c.mount('dorsal', torso, f.pos, f.n, v(0, 0, 1), 1.1); }
  }
  for (const s of [-1, 1]) {
    const f = ellSurf(v(0, 0, 0), 3.7, 3.0, 4.4, null, s * PI / 2, 0.08, 1.0);
    c.mount('fins', torso, f.pos, v(s, 0.15, 0), v(0, 0, -1), 1.5);
    const sh = ellSurf(v(0, 0, 0), 3.7, 3.0, 4.4, null, s * 0.9, 0.55, 1.03);
    c.mount('shoulders', torso, sh.pos, sh.n);
  }
  c.mount('chest', torso, v(0, 0.2, 4.35), v(0, 0.1, 1));
  c.mount('back', torso, v(0, 2.2, -2.8), v(0, 0.7, -1));
  // legs
  for (const [x, z, ph] of [[1, 1, 0], [-1, 1, PI], [1, -1, PI], [-1, -1, 0]]) {
    const hip = c.g(trunk, v(x * 2.55 * B, 3.0, z * 2.5));
    c.ell(hip, 1.05, 1.5, 1.05, M.body, v(x * 0.2, -1.0, 0));
    const knee = c.g(hip, v(x * 0.2, -2.0, 0));
    c.ell(knee, 0.95, 1.0, 0.95, M.body, v(0, -0.4, 0));
    c.ell(knee, 1.1, 0.4, 1.2, M.belly, v(0, -0.85, 0.1));
    for (const dx of [-0.45, 0, 0.45]) c.claw(knee, v(dx, -0.9, 1.05), v(0, 0.1, 1), 0.4, 0.14, v(0, -1, 0));
    c.mount('plates', hip, v(x * 1.05, -0.9, 0), v(x, 0.2, 0), null, 1.1);
    c.limb({ type: 'leg', pivot: hip, knee, phase: ph });
  }
  // neck + head
  c.tube(trunk, [v(0, 3.7, 3.4), v(0, 4.3, 4.6), v(0, 4.8, 5.3)], 0.85, 0.7, M.body);
  const hp = c.g(trunk, v(0, 4.85, 5.3));
  c.ell(hp, 0.85, 0.75, 1.1, M.body, v(0, 0.12, 0.45));
  const beakTop = new THREE.ConeGeometry(0.42, 0.75, 10); beakTop.rotateX(PI / 2 + 0.5);
  c.m(beakTop, M.horn, hp, v(0, 0.0, 1.45));
  for (const s of [-1, 1]) {
    c.eye(hp, v(s * 0.62, 0.35, 0.95), 0.14);
    c.mount('eyes', hp, v(s * 0.55, 0.6, 0.6), v(s * 0.5, 0.8, 0.3));
    c.mount('eyes', hp, v(s * 0.8, 0.1, 0.4), v(s, 0, 0.2));
    c.mount('horns', hp, v(s * 0.5, 0.75, 0.1), v(s * 0.4, 1, -0.3));
  }
  const jaw = c.g(hp, v(0, -0.3, 0.4));
  c.ell(jaw, 0.6, 0.25, 0.9, M.belly, v(0, -0.08, 0.55));
  const mouth = new THREE.Object3D(); mouth.position.set(0, -0.2, 1.85); hp.add(mouth);
  registerHead(c, hp, jaw, mouth);
  c.mount('crest', hp, v(0, 0.85, 0.3), v(0, 1, -0.2), v(0, 0, -1));
  c.mount('ram', hp, v(0, 0.55, 1.1), v(0, 0.6, 0.8));
  for (const s of [-1, 1]) c.mount('tusks', jaw, v(s * 0.38, 0.0, 1.0), v(s * 0.3, 1, 0.4));
  const tail = c.chain(trunk, { n: 4, len: 0.9, r0: 0.6, r1: 0.1, pos: v(0, 3.2, -4.1), rot: [-(PI / 2 + 0.5), 0, 0], curlX: 0.12 });
  c.rig.tail = tail;
  c.limb({ type: 'tail', joints: tail, ampX: 0.03, ampZ: 0.1 });
  c.mount('tailTip', tail.tip, v(0, 0, 0), v(0, 1, 0), v(0, 0, 1));
}

function buildMantis(c) {
  const { M } = c, B = c.bulk;
  M.body.roughness = 0.45; M.belly.roughness = 0.5;
  c.rig.koMode = 'side';
  const trunk = c.g(c.root, v(0, 4.4, -0.4)); c.rig.trunk = trunk;
  const torso = c.g(trunk); c.rig.body = torso;
  const abd = c.g(torso, v(0, -0.1, -0.7), [-0.28, 0, 0]);
  c.ell(abd, 1.25, 1.0, 3.2, M.body, v(0, 0, -2.5));
  c.ell(abd, 1.05, 0.7, 2.9, M.belly, v(0, -0.35, -2.4));
  const tor = new THREE.TorusGeometry(1, 0.07, 6, 24);
  for (let k = 0; k < 5; k++) { const z = -0.9 - k * 0.75, zz = (z + 2.5) / 3.2, r = Math.sqrt(Math.max(0.05, 1 - zz * zz)); c.m(tor, M.plate, abd, v(0, 0, z)).scale.set(1.25 * r, 1.0 * r, 1); }
  for (const s of [-1, 1]) {
    const wg = new THREE.SphereGeometry(1, 16, 10); wg.scale(0.85, 0.06, 3.0);
    c.m(wg, M.membrane, abd, v(s * 0.45, 1.0, -2.2), [0.05, s * 0.08, s * 0.12]);
  }
  c.ell(torso, 0.8, 0.8, 1.25, M.body, v(0, 0.1, 0));
  c.tube(torso, [v(0, 0.3, 0.8), v(0, 2.4, 1.5), v(0, 4.2, 1.9)], 0.6, 0.4, M.body);
  c.mount('chest', torso, v(0, 2.4, 2.0), v(0, 0.1, 1));
  c.mount('tailTip', abd, v(0, 0, -5.7), v(0, -0.1, -1), v(0, 1, 0));
  for (let k = 0; k < 6; k++) c.mount('dorsal', k < 2 ? torso : abd, k < 2 ? v(0, 1.0 + k * 1.4, 0.8 + k * 0.4) : v(0, 1.0, -0.8 - (k - 2) * 0.9), v(0, 1, -0.2), v(0, 0, -1), 0.8);
  for (const s of [-1, 1]) {
    c.mount('plates', torso, v(s * 0.5, 2.6, 1.55), v(s, 0.2, 0.3), null, 0.9);
    c.mount('plates', abd, v(s * 1.15, 0.2, -2.0), v(s, 0.3, 0), null, 1.3);
    c.mount('plates', abd, v(s * 1.05, 0.2, -3.4), v(s, 0.3, 0), null, 1.1);
    c.mount('fins', abd, v(s * 1.0, 0.4, -4.4), v(s, 0.5, 0), v(0, 0, -1), 1.0);
    c.mount('shoulders', torso, v(s * 0.55, 3.8, 1.75), v(s * 0.5, 1, 0));
  }
  // triangular head
  const hp = c.g(trunk, v(0, 4.75, 2.0));
  const hg = new THREE.SphereGeometry(1, 20, 14); hg.scale(0.95, 0.72, 0.55);
  const hpos = hg.attributes.position;
  for (let i = 0; i < hpos.count; i++) { const y = hpos.getY(i); hpos.setX(i, hpos.getX(i) * (0.55 + 0.45 * (y / 0.72 + 1) / 2 + 0.2)); }
  hg.computeVertexNormals();
  c.m(hg, M.body, hp, v(0, 0, 0.25));
  for (const s of [-1, 1]) {
    c.ell(hp, 0.5, 0.58, 0.5, M.eyeBig, v(s * 1.0, 0.35, 0.25), null, 16);
    c.tube(hp, [v(s * 0.2, 0.55, 0.6), v(s * 0.5, 1.8, 0.5), v(s * 0.9, 3.0, -0.4), v(s * 1.2, 3.6, -1.6)], 0.05, 0.02, M.body, { seg: 14, rad: 5 });
    c.mount('eyes', hp, v(s * 0.3, 0.55, 0.6), v(s * 0.2, 1, 0.5));
    c.mount('eyes', hp, v(s * 0.55, 0.1, 0.72), v(s * 0.3, 0.2, 1));
    c.mount('horns', hp, v(s * 0.55, 0.65, 0.15), v(s * 0.4, 1, -0.2));
  }
  const jaw = c.g(hp, v(0, -0.55, 0.45));
  for (const s of [-1, 1]) c.tube(jaw, [v(s * 0.2, 0, 0), v(s * 0.25, -0.3, 0.25), v(s * 0.05, -0.45, 0.35)], 0.12, 0.03, M.horn, { seg: 8, rad: 6 });
  const mouth = new THREE.Object3D(); mouth.position.set(0, -0.6, 0.85); hp.add(mouth);
  registerHead(c, hp, jaw, mouth);
  c.mount('crest', hp, v(0, 0.7, 0.1), v(0, 1, -0.2), v(0, 0, -1));
  c.mount('ram', hp, v(0, 0.4, 0.75), v(0, 0.5, 1));
  for (const s of [-1, 1]) c.mount('tusks', jaw, v(s * 0.2, -0.3, 0.35), v(s * 0.3, -0.6, 1));
  // raptorial scythe arms
  for (const s of [-1, 1]) {
    const sh = c.g(trunk, v(s * 0.6 * B, 3.4, 1.85), [0.3, 0, s * 0.15]);
    c.tube(sh, [v(0, 0, 0), v(0, -0.7, 0.5), v(0, -1.2, 0.9)], 0.32, 0.25, M.body, { capEnd: true });
    const el = c.g(sh, v(0, -1.2, 0.9), [-0.5, 0, 0]);
    c.tube(el, [v(0, 0, 0), v(0, 1.1, 1.2), v(0, 1.8, 2.0)], 0.3, 0.2, M.body, { capEnd: true });
    const spine = new THREE.ConeGeometry(0.06, 0.35, 5);
    for (let i = 1; i < 6; i++) c.m(spine, M.horn, el, v(0, i * 0.33 - 0.08, i * 0.36 + 0.12), [2.3, 0, 0]);
    const kn = c.g(el, v(0, 1.8, 2.0), [0, 0, 0]);
    const blade = c.g(kn);
    c.pow(blade);
    const bt = c.tube(blade, [v(0, 0, 0), v(0, -0.9, -0.45), v(0, -1.8, -0.5), v(0, -2.4, -0.05)], 0.2, 0.02, M.belly, { seg: 14, rad: 8 });
    bt.scale.x = 0.45;
    const edge = c.tube(blade, [v(0, -0.2, -0.25), v(0, -1.0, -0.62), v(0, -1.85, -0.66), v(0, -2.35, -0.15)], 0.05, 0.01, M.accent, { seg: 12, rad: 5, cap: false });
    edge.scale.x = 0.5;
    c.limb({ type: 'arm', pivot: sh, elbow: kn, phase: s, reach: -0.9, elbowReach: -1.6, vicRaise: -1.8 });
    c.rig.extra[s < 0 ? 'handL' : 'handR'] = kn;
  }
  // four walking legs
  for (const s of [-1, 1]) for (const [z, yaw, ph] of [[0.3, 0.75, 0], [-0.6, 2.2, PI]]) {
    const hip = c.g(torso, v(s * 0.55, -0.15, z), [1.25, s * yaw, 0, 'YXZ']);
    c.tube(hip, [v(0, 0, 0), v(0, 1.2, 0), v(0, 2.4, 0)], 0.2, 0.14, M.body, { capEnd: true });
    const knee = c.g(hip, v(0, 2.4, 0), [1.75, 0, 0]);
    c.tube(knee, [v(0, 0, 0), v(0, 2.5, 0), v(0, 5.0, 0)], 0.14, 0.05, M.body, { cap: false });
    c.limb({ type: 'leg', axis: 'y', pivot: hip, knee, phase: ph + (s > 0 ? 0 : PI / 2) });
  }
  c.mount('back', torso, v(0, 1.0, -0.6), v(0, 1, -0.3));
}

function buildSerpent(c) {
  const { M } = c, B = c.bulk;
  M.body.roughness = 0.35; M.body.metalness = 0.15;
  c.rig.koMode = 'flop';
  const trunk = c.g(c.root, v(0, 0, 0)); c.rig.trunk = trunk;
  const torso = c.g(trunk); c.rig.body = torso;
  // coils resting on the ground
  const pts = [], N = 40;
  for (let k = 0; k <= N; k++) {
    const t = k / N, ang = t * 2.1 * PI * 2 + 0.6, rad = 3.3 - t * 1.4;
    pts.push(v(Math.cos(ang) * rad, 1.0 + t * 1.7, Math.sin(ang) * rad - 0.9));
  }
  c.tube(torso, pts, 1.0, 1.12, M.body, { seg: 120, rad: 14, rf: (u) => 1.0 + 0.15 * u });
  for (let k = 4; k < N; k += 3) {
    const p = pts[k], q = pts[Math.min(N, k + 1)];
    c.mount('dorsal', torso, p.clone().add(v(0, 1.0, 0)), v(0, 1, 0), q.clone().sub(p).normalize().negate(), 0.95);
  }
  const p0 = pts[0], dir0 = pts[0].clone().sub(pts[1]).normalize();
  const tailPts = [p0, p0.clone().addScaledVector(dir0, 1.6).setY(0.7), p0.clone().addScaledVector(dir0, 3.2).add(v(0.6, -0.3, 0.4)), p0.clone().addScaledVector(dir0, 4.6).add(v(1.3, -0.4, 1.2))];
  c.tube(torso, tailPts, 1.0, 0.08, M.body, { seg: 24, rad: 12 });
  const tt = tailPts[3];
  c.mount('tailTip', torso, tt, v(0, 1, 0), dir0);
  attach({ p: torso, pos: tt.clone().add(v(0, 0.05, 0)), n: v(0, 1, 0), t: dir0.clone().negate(), s: 1 }, PART.fin(c, M.membrane, 1.6), 1.2);
  for (const s of [-1, 1]) {
    c.mount('plates', torso, pts[12].clone().add(v(0, 0.9, 0)), v(s * 0.3, 1, 0), null, 1.4);
    c.mount('plates', torso, pts[24].clone().add(v(0, 0.95, 0)), v(s * 0.3, 1, 0), null, 1.4);
  }
  // upright neck (S-curve)
  const curl = [-0.12, -0.12, -0.08, 0.2, 0.25, 0.28, 0.3, 0.25];
  const neck = c.chain(trunk, { n: 9, len: 1.15, r0: 1.1, r1: 0.62, pos: v(0, 2.4, -0.6), rot: [0.15, 0, 0], curlX: curl, belly: M.belly, bellyDir: 1, dorsal: false });
  c.limb({ type: 'neck', joints: neck, ampX: 0.03, ampZ: 0.05, speed: 1.1 });
  let total = 0.15; for (const x of curl) total += x;
  neck.forEach((j, i) => {
    if (i > 0 && i < 8) {
      c.mount('fins', j, v(0, 0.5, -0.95 * lerp(1.1, 0.62, i / 9)), v(0, 0, -1), v(0, 1, 0), 0.9);
      attach({ p: j, pos: v(0, 0.55, -lerp(1.1, 0.62, i / 9) * 0.9), n: v(0, 0, -1), t: v(0, -1, 0), s: 1 }, PART.fin(c, M.membrane, 0.8), lerp(1.0, 0.6, i / 9));
    }
    if (i % 2 === 1) c.mount('dorsal', j, v(0, 0.5, -lerp(1.1, 0.62, i / 9) * 0.95), v(0, 0, -1), v(0, -1, 0), 0.8);
  });
  const hp = c.g(neck.tip, v(0, 0.2, 0), [-total + 0.12, 0, 0]);
  const H = reptileHead(c, hp, { len: 1.55, w: 0.95 });
  registerHead(c, hp, H.jaw, H.mouth);
  for (const s of [-1, 1]) {
    attach({ p: hp, pos: v(s * 0.75, 0.3, 0.2), n: v(s, 0.5, -0.4), t: v(0, 0, -1), s: 1 }, PART.fin(c, c.M.membrane, 1.4), 1.0);
    c.tube(hp, [v(s * 0.4, -0.05, 2.6), v(s * 0.9, -0.4, 2.7), v(s * 1.4, -1.0, 2.3), v(s * 1.6, -1.8, 2.0)], 0.06, 0.02, M.belly, { seg: 10, rad: 5 });
  }
  c.mount('chest', neck[3], v(0, 0.5, 1.0), v(0, 0, 1));
  c.mount('back', torso, pts[30].clone().add(v(0, 1.0, 0)), v(0, 1, -0.2));
  for (const s of [-1, 1]) c.mount('shoulders', neck[2], v(s * 0.9, 0.5, -0.2), v(s, 0.4, -0.3));
}

function buildMech(c) {
  const { M } = c, B = c.bulk;
  c.M.membrane = c.M.plate;
  const joint = stdMat({ color: '#2a2d33', roughness: 0.5, metalness: 0.7 });
  const trunk = c.g(c.root, v(0, 5.3, 0)); c.rig.trunk = trunk;
  const torso = c.g(trunk); c.rig.body = torso;
  c.rbox(torso, 2.6, 1.0, 1.7, 0.2, M.body, v(0, 0, 0));
  c.m(new THREE.CylinderGeometry(0.75, 0.85, 0.9, 12), joint, torso, v(0, 0.75, 0));
  c.rbox(torso, 2.0, 1.0, 1.5, 0.2, M.belly, v(0, 1.4, 0.1));
  c.rbox(torso, 3.8, 2.6, 2.4, 0.35, M.body, v(0, 3.0, 0.1));
  c.rbox(torso, 2.7, 1.5, 0.45, 0.15, M.belly, v(0, 3.25, 1.3), [-0.12, 0, 0]);
  const reactor = new THREE.CylinderGeometry(0.42, 0.42, 0.25, 20); reactor.rotateX(PI / 2);
  c.m(reactor, M.accent, torso, v(0, 3.05, 1.6));
  c.rbox(torso, 1.6, 0.5, 1.4, 0.15, M.plate, v(0, 4.4, -0.1));
  c.rbox(torso, 2.6, 2.3, 1.2, 0.25, M.plate, v(0, 3.0, -1.65));
  for (const s of [-1, 1]) {
    c.m(new THREE.CylinderGeometry(0.45, 0.62, 1.2, 14), joint, torso, v(s * 0.75, 2.0, -2.25), [0.25, 0, 0]);
    c.m(new THREE.CylinderGeometry(0.42, 0.42, 0.1, 14), M.accent, torso, v(s * 0.75, 1.35, -2.4), [0.25, 0, 0]);
    c.rbox(torso, 0.18, 0.5, 0.06, 0.03, M.accent, v(s * 1.2, 3.5, 1.33));
  }
  c.mount('chest', torso, v(0, 3.05, 1.65), v(0, 0, 1));
  c.mount('back', torso, v(0, 3.4, -2.3), v(0, 0.3, -1));
  for (let k = 0; k < 4; k++) c.mount('dorsal', torso, v((k % 2 ? 1 : -1) * 0.7, 3.9 - Math.floor(k / 2) * 1.0, -2.3), v(0, 0.6, -1), v(0, 1, 0), 0.9);
  for (const s of [-1, 1]) {
    c.mount('plates', torso, v(s * 1.95, 3.0, 0.4), v(s, 0, 0.1), null, 1.4);
    c.mount('plates', torso, v(s * 1.3, 0.1, 0.8), v(s * 0.4, 0, 1), null, 1.0);
    c.mount('fins', torso, v(s * 1.1, 4.0, -2.2), v(s * 0.8, 0.6, -0.4), v(0, 0.3, -1), 1.3);
  }
  // head
  const hp = c.g(trunk, v(0, 4.85, 0.2));
  c.rbox(hp, 1.15, 0.95, 1.25, 0.22, M.body, v(0, 0.35, 0));
  c.rbox(hp, 0.95, 0.22, 0.12, 0.05, M.eye, v(0, 0.45, 0.62));
  const jaw = c.g(hp, v(0, 0.1, 0.25));
  c.rbox(jaw, 0.75, 0.32, 0.5, 0.08, M.belly, v(0, -0.05, 0.25));
  for (const s of [-1, 1]) {
    c.rbox(hp, 0.1, 0.9, 0.12, 0.04, M.belly, v(s * 0.32, 1.1, 0.5), [0.3, 0, s * -0.7]);
    c.rbox(hp, 0.25, 0.6, 0.7, 0.08, M.plate, v(s * 0.66, 0.35, -0.05));
    c.mount('eyes', hp, v(s * 0.3, 0.75, 0.62), v(0, 0.3, 1));
    c.mount('eyes', hp, v(s * 0.62, 0.55, 0.45), v(s * 0.5, 0.2, 1));
    c.mount('horns', hp, v(s * 0.45, 0.85, 0.0), v(s * 0.5, 1, -0.3));
  }
  const mouth = new THREE.Object3D(); mouth.position.set(0, 0.42, 0.8); hp.add(mouth);
  registerHead(c, hp, jaw, mouth);
  c.mount('crest', hp, v(0, 0.85, -0.1), v(0, 1, -0.1), v(0, 0, -1));
  c.mount('ram', hp, v(0, 0.65, 0.65), v(0, 0.5, 1));
  for (const s of [-1, 1]) c.mount('tusks', jaw, v(s * 0.3, -0.1, 0.45), v(s * 0.3, 0.6, 1));
  // arms
  for (const s of [-1, 1]) {
    const sh = c.g(trunk, v(s * 2.35 * B, 3.6, 0), [0, 0, s * 0.1]);
    c.ell(sh, 0.55, 0.55, 0.55, joint, null, null, 14);
    c.rbox(sh, 1.6, 1.2, 1.8, 0.3, M.body, v(s * 0.3, 0.4, 0), [0, 0, s * -0.25]);
    c.rbox(sh, 1.62, 0.18, 1.82, 0.05, M.belly, v(s * 0.32, 0.05, 0), [0, 0, s * -0.25]);
    c.rbox(sh, 0.8, 2.0, 0.85, 0.15, M.body, v(0, -1.3, 0));
    const el = c.g(sh, v(0, -2.4, 0), [-0.35, 0, 0]);
    c.m(new THREE.CylinderGeometry(0.45, 0.45, 1.0, 14).rotateZ(PI / 2), joint, el);
    c.rbox(el, 1.1, 2.1, 1.15, 0.2, M.belly, v(0, -1.2, 0));
    const fist = c.g(el, v(0, -2.45, 0));
    c.pow(fist);
    c.rbox(fist, 1.15, 1.0, 1.2, 0.22, M.body, v(0, -0.35, 0));
    c.rbox(fist, 1.2, 0.28, 0.4, 0.08, joint, v(0, -0.62, 0.45));
    c.mount('shoulders', sh, v(s * 0.35, 1.05, 0), v(s * 0.1, 1, 0));
    c.mount('plates', el, v(s * 0.6, -1.1, 0), v(s, 0, 0.2), null, 1.1);
    c.limb({ type: 'arm', pivot: sh, elbow: el, phase: s, reach: -1.3, elbowReach: 0.4 });
    c.rig.extra[s < 0 ? 'handL' : 'handR'] = fist;
  }
  // legs
  for (const s of [-1, 1]) {
    const hip = c.g(trunk, v(s * 0.95 * B, -0.4, 0));
    c.ell(hip, 0.5, 0.5, 0.5, joint, null, null, 12);
    c.rbox(hip, 1.0, 2.2, 1.1, 0.2, M.body, v(0, -1.3, 0));
    const knee = c.g(hip, v(0, -2.5, 0.05));
    c.m(new THREE.CylinderGeometry(0.42, 0.42, 0.9, 12).rotateZ(PI / 2), joint, knee);
    c.rbox(knee, 0.9, 0.85, 0.5, 0.15, M.belly, v(0, 0.05, 0.5));
    c.rbox(knee, 1.2, 2.3, 1.3, 0.25, M.body, v(0, -1.3, -0.1));
    const foot = c.g(knee, v(0, -2.45, 0));
    c.rbox(foot, 1.35, 0.5, 2.1, 0.15, M.plate, v(0, -0.12, 0.3));
    c.mount('plates', hip, v(s * 0.55, -1.2, 0), v(s, 0, 0.2), null, 1.1);
    c.limb({ type: 'leg', pivot: hip, knee, phase: s > 0 ? 0 : PI });
  }
  c.mount('tailTip', torso, v(0, 0.0, -0.9), v(0, -0.2, -1), v(0, 1, 0));
}

function buildGolem(c) {
  const { M } = c, B = c.bulk;
  const rockM = stdMat({ color: M.body.color.clone(), roughness: 0.95, flatShading: true });
  const rock2 = stdMat({ color: M.belly.color.clone(), roughness: 0.95, flatShading: true });
  const lava = M.accent; lava.emissiveIntensity = 2.4;
  c.M.plate = rock2; c.M.horn = rockM; c.M.membrane = rock2;
  const trunk = c.g(c.root, v(0, 4.2, 0)); c.rig.trunk = trunk;
  const torso = c.g(trunk); c.rig.body = torso;
  const ctr = v(0, 2.6, 0.2);
  c.ell(torso, 1.75, 2.4, 1.5, lava, ctr, [0.2, 0, 0]);
  const n = 18, ga = PI * (3 - Math.sqrt(5));
  for (let k = 0; k < n; k++) {
    const y = 1 - (k + 0.5) / n * 2, r = Math.sqrt(1 - y * y), th = ga * k;
    const dir = v(Math.cos(th) * r, y, Math.sin(th) * r);
    const pos = v(dir.x * 1.7, dir.y * 2.3, dir.z * 1.45).applyEuler(new THREE.Euler(0.2, 0, 0)).add(ctr);
    c.rock(torso, 0.82 + c.rng() * 0.22, k % 3 ? rockM : rock2, pos, null, 1, 0.8, 1);
  }
  c.rock(torso, 1.7, rockM, v(0, 0.3, 0), null, 1, 0.65, 0.85);
  for (const s of [-1, 1]) c.rock(torso, 1.45, rockM, v(s * 2.75, 4.5, 0.1), null, 1, 0.8, 1);
  c.mount('chest', torso, v(0, 2.8, 1.9), v(0, 0.2, 1));
  c.mount('back', torso, v(0, 3.2, -1.6), v(0, 0.5, -1));
  for (let k = 0; k < 5; k++) {
    const { pos, n } = ellSurf(ctr, 2.1, 2.5, 1.6, [0.2, 0, 0], PI, -0.5 + k * 0.33, 1.25);
    c.mount('dorsal', torso, pos, n, v(0, -0.4, -1), 1.15);
  }
  for (const s of [-1, 1]) {
    c.mount('shoulders', torso, v(s * 2.8, 5.6, 0.1), v(s * 0.2, 1, 0));
    const p = ellSurf(ctr, 2.1, 2.5, 1.6, [0.2, 0, 0], s * 1.6, 0.0, 1.3);
    c.mount('plates', torso, p.pos, p.n, null, 1.5);
    const f = ellSurf(ctr, 2.1, 2.5, 1.6, [0.2, 0, 0], s * 2.5, 0.4, 1.3);
    c.mount('fins', torso, f.pos, f.n, v(0, 0.3, -1), 1.2);
  }
  const hp = c.g(trunk, v(0, 5.5, 1.45));
  c.rock(hp, 0.95, rockM, v(0, 0.2, 0), [0.3, 0.6, 0.1], 1.1, 0.85, 1);
  c.ell(hp, 0.75, 0.6, 0.6, lava, v(0, 0.1, 0.1));
  for (const s of [-1, 1]) {
    c.rbox(hp, 0.32, 0.1, 0.12, 0.03, M.eye, v(s * 0.34, 0.25, 0.82), [0, 0, s * -0.2]);
    c.mount('eyes', hp, v(s * 0.45, 0.6, 0.6), v(s * 0.3, 0.6, 1));
    c.mount('eyes', hp, v(s * 0.75, 0.1, 0.4), v(s, 0, 0.5));
    c.mount('horns', hp, v(s * 0.5, 0.8, 0.0), v(s * 0.5, 1, -0.2));
  }
  const jaw = c.g(hp, v(0, -0.35, 0.3));
  c.rock(jaw, 0.6, rock2, v(0, -0.15, 0.35), [0, 0, 0], 1.2, 0.5, 1);
  c.ell(hp, 0.5, 0.15, 0.4, lava, v(0, -0.25, 0.5));
  const mouth = new THREE.Object3D(); mouth.position.set(0, -0.3, 1.1); hp.add(mouth);
  registerHead(c, hp, jaw, mouth);
  c.mount('crest', hp, v(0, 0.95, 0.0), v(0, 1, -0.2), v(0, 0, -1));
  c.mount('ram', hp, v(0, 0.55, 0.85), v(0, 0.5, 1));
  for (const s of [-1, 1]) c.mount('tusks', jaw, v(s * 0.45, 0, 0.6), v(s * 0.3, 1, 0.4));
  for (const s of [-1, 1]) {
    const sh = c.g(trunk, v(s * 3.4 * B, 4.3, 0.3), [0.1, 0, s * 0.42]);
    c.m(new THREE.CylinderGeometry(0.55, 0.55, 2.8, 10), lava, sh, v(0, -1.3, 0));
    c.rock(sh, 0.95, rockM, v(0, -0.6, 0));
    c.rock(sh, 0.9, rock2, v(0, -1.9, 0.1));
    const el = c.g(sh, v(0, -2.9, 0), [-0.35, 0, -s * 0.3]);
    c.m(new THREE.CylinderGeometry(0.6, 0.6, 2.2, 10), lava, el, v(0, -1.0, 0));
    c.rock(el, 1.15, rockM, v(0, -0.9, 0), null, 1, 1.2, 1);
    const fist = c.g(el, v(0, -2.4, 0.1));
    c.pow(fist);
    c.ell(fist, 1.15, 1.0, 1.15, lava, v(0, -0.5, 0));
    c.rock(fist, 1.45, rockM, v(0, -0.5, 0), null, 1, 0.9, 1);
    c.mount('plates', el, v(s * 0.95, -0.9, 0), v(s, 0, 0.2), null, 1.3);
    c.limb({ type: 'arm', pivot: sh, elbow: el, phase: s, reach: -1.3, elbowReach: 0.35 });
    c.rig.extra[s < 0 ? 'handL' : 'handR'] = fist;
  }
  for (const s of [-1, 1]) {
    const hip = c.g(trunk, v(s * 1.55 * B, 0, 0));
    c.m(new THREE.CylinderGeometry(0.7, 0.7, 3.2, 10), lava, hip, v(0, -1.7, 0));
    c.rock(hip, 1.15, rockM, v(0, -1.1, 0));
    const knee = c.g(hip, v(0, -2.2, 0));
    c.rock(knee, 1.25, rock2, v(0, -1.2, 0.2), null, 1.1, 0.6, 1.3);
    c.limb({ type: 'leg', pivot: hip, knee, phase: s > 0 ? 0 : PI });
  }
  c.mount('tailTip', torso, v(0, 0.4, -1.6), v(0, 0, -1), v(0, 1, 0));
}

function buildHydra(c) {
  const { M } = c, B = c.bulk;
  c.rig.koMode = 'side';
  const trunk = c.g(c.root, v(0, 4.4, -0.8)); c.rig.trunk = trunk;
  const torso = c.g(trunk); c.rig.body = torso;
  c.ell(torso, 2.5, 2.2, 3.5, M.body, v(0, 0.5, 0));
  c.ell(torso, 2.0, 1.4, 3.0, M.belly, v(0, -0.6, 0.2));
  c.ell(torso, 2.2, 2.0, 1.8, M.body, v(0, 0.9, 2.3));
  c.mount('chest', torso, v(0, 0.3, 3.9), v(0, 0, 1));
  c.mount('back', torso, v(0, 2.5, -1.0), v(0, 1, -0.2));
  for (let k = 0; k < 6; k++) {
    const { pos, n } = ellSurf(v(0, 0.5, 0), 2.5, 2.2, 3.5, null, PI, 1.4 - k * 0.22, 0.98);
    c.mount('dorsal', torso, k < 3 ? ellSurf(v(0, 0.5, 0), 2.5, 2.2, 3.5, null, 0, 1.0 + k * 0.2, 0.98).pos : pos, n, v(0, 0, -1), 1.0);
  }
  for (const mt of c.mounts.dorsal) attach(mt, PART.spike(c, M.plate, 0.7), 0.8);
  for (const s of [-1, 1]) {
    for (const u of [0.8, 1.6, 2.4]) { const p = ellSurf(v(0, 0.5, 0), 2.5, 2.2, 3.5, null, s * u, 0.35, 0.98); c.mount('plates', torso, p.pos, p.n, null, 1.3); }
    const f = ellSurf(v(0, 0.5, 0), 2.5, 2.2, 3.5, null, s * 2.2, 0.8, 0.98);
    c.mount('fins', torso, f.pos, f.n, v(0, 0, -1), 1.1);
    const sh = ellSurf(v(0, 0.5, 0), 2.5, 2.2, 3.5, null, s * 0.5, 0.9, 0.98);
    c.mount('shoulders', torso, sh.pos, sh.n);
  }
  for (const [x, z, ph] of [[1, 2.0, 0], [-1, 2.0, PI], [1, -2.2, PI], [-1, -2.2, 0]]) {
    const hip = c.g(trunk, v(x * 2.1 * B, -0.2, z), [0, 0, x * 0.5]);
    c.ell(hip, 0.85, 1.3, 0.95, M.body, v(0, -1.0, 0));
    const knee = c.g(hip, v(0, -2.2, 0), [0, 0, -x * 0.6]);
    c.ell(knee, 0.62, 1.1, 0.68, M.body, v(0, -0.95, 0));
    c.ell(knee, 0.8, 0.35, 1.1, M.body, v(0, -2.0, 0.35));
    for (const dx of [-0.35, 0, 0.35]) c.claw(knee, v(dx, -2.05, 1.35), v(0, 0.1, 1), 0.45, 0.11, v(0, -1, 0));
    c.limb({ type: 'leg', pivot: hip, knee, phase: ph });
  }
  const tail = c.chain(trunk, { n: 8, len: 1.15, r0: 1.05, r1: 0.12, pos: v(0, 0.4, -3.3), rot: [-(PI / 2 + 0.3), 0, 0], curlX: 0.05, belly: M.belly, bellyDir: -1 });
  c.rig.tail = tail;
  c.limb({ type: 'tail', joints: tail, ampX: 0.03, ampZ: 0.1 });
  c.mount('tailTip', tail.tip, v(0, 0, 0), v(0, 1, 0), v(0, 0, 1));
  // three necks
  const neckDefs = [[-1.5, 1.4, 2.5, -1.0, 0.95], [0, 2.0, 3.1, 0, 1.1], [1.5, 1.4, 2.5, 1.0, 0.95]];
  const order = [1, 0, 2];
  for (const idx of order) {
    const [x, y, z, yaw, sc] = neckDefs[idx];
    const curl = [-0.15, -0.1, 0.15, 0.3, 0.35, 0.3];
    const neck = c.chain(trunk, { n: 7, len: 1.05 * sc, r0: 0.78 * sc, r1: 0.48 * sc, pos: v(x * B, y, z), rot: [0.35, yaw, 0, 'YXZ'], curlX: curl, belly: M.belly, bellyDir: 1 });
    c.limb({ type: 'neck', joints: neck, ampX: 0.05, ampZ: 0.06, speed: 1.2, phase: idx * 2.1 });
    let total = 0.35; for (const k of curl) total += k;
    const hp = c.g(neck.tip, v(0, 0.1, 0), [-total + 0.1, 0, 0]);
    const H = reptileHead(c, hp, { len: 1.15 * sc, w: 0.72 * sc });
    registerHead(c, hp, H.jaw, H.mouth, idx === 1);
    for (let i = 1; i < 6; i += 2) attach({ p: neck[i], pos: v(0, 0.5, -0.7 * sc), n: v(0, 0, -1), t: v(0, -1, 0), s: 1 }, PART.spike(c, M.plate, 0.6), 0.7);
  }
  // re-order mounts list so the middle head's eye/horn mounts come first is unnecessary; all three heads get them
}

function buildSpider(c) {
  const { M } = c, B = c.bulk;
  M.body.roughness = 0.6;
  c.rig.koMode = 'side';
  const trunk = c.g(c.root, v(0, 3.6, 0)); c.rig.trunk = trunk;
  const torso = c.g(trunk); c.rig.body = torso;
  c.ell(torso, 1.7, 1.1, 1.9, M.body, v(0, 0, 1.0));
  c.ell(torso, 0.6, 0.55, 0.7, M.body, v(0, 0.2, -0.9));
  const aC = v(0, 1.0, -3.0), aR = [0.3, 0, 0];
  c.ell(torso, 2.5, 2.2, 2.9, M.body, aC, aR);
  c.ell(torso, 2.0, 1.4, 2.3, M.belly, v(0, 0.35, -2.9), aR);
  // markings
  for (let k = 0; k < 4; k++) {
    const { pos, n } = ellSurf(aC, 2.5, 2.2, 2.9, aR, PI, 0.95 - k * 0.32, 1.0);
    const o = c.g(torso, pos); orientTo(o, n, v(0, 0, 1));
    c.ell(o, 0.75 - k * 0.12, 0.1, 0.32, M.accent, null, null, 12);
  }
  for (let k = 0; k < 5; k++) {
    const { pos, n } = ellSurf(aC, 2.5, 2.2, 2.9, aR, PI, 1.15 - k * 0.3, 1.0);
    c.mount('dorsal', torso, pos, n, v(0, 0, -1), 1.0);
  }
  for (const s of [-1, 1]) {
    for (const u of [1.4, 2.2]) { const p = ellSurf(aC, 2.5, 2.2, 2.9, aR, s * u, 0.35, 1.0); c.mount('plates', torso, p.pos, p.n, null, 1.4); }
    const f = ellSurf(aC, 2.5, 2.2, 2.9, aR, s * 2.6, 0.6, 1.0);
    c.mount('fins', torso, f.pos, f.n, v(0, 0, -1), 1.1);
    c.mount('shoulders', torso, v(s * 1.1, 0.95, 0.6), v(s * 0.3, 1, 0));
  }
  c.mount('tailTip', torso, ellSurf(aC, 2.5, 2.2, 2.9, aR, PI, -0.2, 1.0).pos, v(0, -0.2, -1), v(0, 1, 0));
  c.mount('chest', torso, v(0, 0.4, 2.7), v(0, 0.2, 1));
  c.mount('back', torso, v(0, 3.0, -2.8), v(0, 1, -0.3));
  // head / eyes / fangs
  const hp = c.g(trunk, v(0, 0.3, 2.3));
  const eyeDefs = [[0.32, 0.45, 0.55, 0.26], [-0.32, 0.45, 0.55, 0.26], [0.75, 0.35, 0.3, 0.17], [-0.75, 0.35, 0.3, 0.17], [0.2, 0.78, 0.3, 0.12], [-0.2, 0.78, 0.3, 0.12], [0.55, 0.72, 0.1, 0.1], [-0.55, 0.72, 0.1, 0.1]];
  for (const [x, y, z, r] of eyeDefs) c.ell(hp, r, r, r, M.eye, v(x, y, z), null, 12);
  for (const s of [-1, 1]) {
    c.mount('eyes', hp, v(s * 0.95, 0.15, 0.1), v(s, 0.2, 0.3));
    c.mount('eyes', hp, v(s * 0.4, 0.95, 0.0), v(s * 0.2, 1, 0.2));
    c.mount('horns', hp, v(s * 0.6, 0.85, -0.3), v(s * 0.4, 1, -0.3));
  }
  const jaw = c.g(hp, v(0, -0.15, 0.45));
  for (const s of [-1, 1]) {
    c.ell(jaw, 0.36, 0.62, 0.38, M.body, v(s * 0.36, -0.35, 0.2), [0.3, 0, 0]);
    c.claw(jaw, v(s * 0.36, -0.95, 0.35), v(-s * 0.3, -1, 0.3), 0.7, 0.13, v(0, 0, -1));
  }
  const mouth = new THREE.Object3D(); mouth.position.set(0, -0.6, 1.0); hp.add(mouth);
  registerHead(c, hp, jaw, mouth);
  c.mount('crest', hp, v(0, 0.95, -0.4), v(0, 1, -0.2), v(0, 0, -1));
  c.mount('ram', hp, v(0, 0.55, 0.6), v(0, 0.5, 1));
  for (const s of [-1, 1]) c.mount('tusks', jaw, v(s * 0.55, -0.3, 0.4), v(s * 0.4, -0.5, 1));
  // pedipalps
  for (const s of [-1, 1]) {
    const pp = c.g(hp, v(s * 0.75, -0.2, 0.3), [0.9, s * 0.4, 0, 'YXZ']);
    c.tube(pp, [v(0, 0, 0), v(0, 0.6, 0.2), v(0, 1.0, -0.2), v(0, 1.2, -0.7)], 0.16, 0.1, M.body, { seg: 10, rad: 7, capEnd: true });
  }
  // eight legs
  const yaws = [0.55, 1.15, 1.85, 2.45];
  for (const s of [-1, 1]) yaws.forEach((ang, k) => {
    const hip = c.g(torso, v(s * 1.1 * Math.sin(ang) * B, 0.1, 1.0 + Math.cos(ang) * 1.3), [0.75, s * ang, 0, 'YXZ']);
    c.tube(hip, [v(0, 0, 0), v(0, 1.6, 0.18), v(0, 3.2, 0)], 0.27, 0.19, M.body, { capEnd: true, seg: 10 });
    const knee = c.g(hip, v(0, 3.2, 0), [2.0, 0, 0]);
    c.ell(knee, 0.22, 0.22, 0.22, M.belly, null, null, 10);
    c.tube(knee, [v(0, 0, 0), v(0, 2.2, 0.35), v(0, 4.6, 0.25), v(0, 6.5, 0)], 0.19, 0.035, M.body, { cap: false, seg: 14 });
    c.limb({ type: 'leg', axis: 'y', pivot: hip, knee, phase: (k % 2 ? PI : 0) + (s > 0 ? 0 : PI) });
  });
}

function rayWingGeo(colBody, colBelly) {
  const s = new THREE.Shape();
  s.moveTo(0, -2.6);
  s.bezierCurveTo(2.5, -2.4, 5.6, -0.7, 7.6, 1.6);
  s.bezierCurveTo(5.8, 1.9, 3.0, 2.3, 0, 2.9);
  s.lineTo(0, -2.6);
  const geo = new THREE.ExtrudeGeometry(s, { depth: 0.05, bevelEnabled: true, bevelThickness: 0.38, bevelSize: 0.32, bevelSegments: 3, curveSegments: 18 });
  geo.rotateX(-PI / 2);
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i);
    const t = Math.min(1, Math.max(0, x / 7.8));
    p.setY(i, (p.getY(i) - 0.025) * (1 - 0.8 * t));
  }
  geo.computeVertexNormals();
  const nrm = geo.attributes.normal, cols = [];
  const tmp = new THREE.Color();
  for (let i = 0; i < p.count; i++) {
    const k = THREE.MathUtils.smoothstep(nrm.getY(i), -0.35, 0.35);
    tmp.copy(colBelly).lerp(colBody, k);
    cols.push(tmp.r, tmp.g, tmp.b);
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  return geo;
}

function buildRay(c) {
  const { M } = c, B = c.bulk;
  c.rig.hover = 4.2; c.rig.koMode = 'tilt'; c.rig.extra.bank = 0.3;
  M.body.roughness = 0.4;
  const trunk = c.g(c.root, v(0, 3.4, 0), [-0.08, 0, 0]); c.rig.trunk = trunk;
  const torso = c.g(trunk); c.rig.body = torso;
  c.ell(torso, 1.5, 0.75, 3.2, M.body, v(0, 0, 0));
  c.ell(torso, 1.3, 0.42, 2.9, M.belly, v(0, -0.35, 0));
  const wingMat = stdMat({ color: '#ffffff', vertexColors: true, roughness: 0.42, side: THREE.DoubleSide });
  const geo = rayWingGeo(M.body.color, M.belly.color);
  const edge = new THREE.CubicBezierCurve3(v(0, 0, 2.6), v(2.5, 0, 2.4), v(5.6, 0, 0.7), v(7.6, 0, -1.6));
  const edgePts = edge.getPoints(12);
  for (const s of [-1, 1]) {
    const wp = c.g(trunk, v(s * 0.95 * B, 0, 0.2), [0, 0, s * 0.12]);
    const me = c.m(geo, wingMat, wp); me.scale.x = s;
    const ep = edgePts.map(p => v(s * p.x, 0.18 * (1 - p.x / 8), p.z));
    c.tube(wp, ep, 0.09, 0.04, M.accent, { seg: 24, rad: 5, cap: false });
    const back = [v(s * 1.2, 0.3, -0.4), v(s * 3.5, 0.2, 0.0), v(s * 6.4, 0.08, -0.9)];
    c.tube(wp, back, 0.06, 0.03, M.accent, { seg: 16, rad: 5, cap: false });
    c.limb({ type: 'wing', pivot: wp, side: s, amp: 0.38, speed: 2.1, phase: 0 });
    c.mount('dorsal', wp, v(s * 2.6, 0.35, 0.2), v(0, 1, 0), v(0, 0, -1), 0.9);
    c.mount('plates', wp, v(s * 1.8, 0.35, -0.6), v(s * 0.1, 1, 0), null, 1.3);
    c.mount('plates', wp, v(s * 3.6, 0.25, -0.2), v(s * 0.1, 1, 0), null, 1.1);
    c.mount('fins', wp, v(s * 6.2, 0.1, -0.6), v(s * 0.3, 1, 0), v(0, 0, -1), 1.0);
    c.rig.extra[s < 0 ? 'handL' : 'handR'] = wp;
    // cephalic lobes
    c.tube(torso, [v(s * 0.9, 0.0, 2.6), v(s * 1.15, -0.1, 3.6), v(s * 0.85, -0.45, 4.2), v(s * 0.55, -0.7, 4.0)], 0.36, 0.1, M.belly, { seg: 12, rad: 8 });
    c.mount('shoulders', torso, v(s * 0.9, 0.6, 0.3), v(s * 0.2, 1, 0));
  }
  for (let k = 0; k < 4; k++) c.mount('dorsal', torso, v(0, 0.72 - k * 0.04, 1.2 - k * 1.1), v(0, 1, -0.1), v(0, 0, -1), 1.0);
  c.mount('chest', torso, v(0, -0.6, 1.4), v(0, -0.3, 1));
  c.mount('back', torso, v(0, 0.75, -1.2), v(0, 1, -0.2));
  const hp = c.g(trunk, v(0, 0.1, 2.6));
  for (const s of [-1, 1]) {
    c.eye(hp, v(s * 1.05, 0.15, 0.1), 0.17);
    c.mount('eyes', hp, v(s * 0.6, 0.5, 0.1), v(s * 0.3, 1, 0.2));
    c.mount('eyes', hp, v(s * 0.3, 0.55, -0.3), v(0, 1, 0.1));
    c.mount('horns', hp, v(s * 0.7, 0.55, -0.4), v(s * 0.3, 1, -0.3));
  }
  c.ell(hp, 0.85, 0.12, 0.22, M.dark, v(0, -0.32, 0.55));
  const jaw = c.g(hp, v(0, -0.35, 0.3));
  c.ell(jaw, 0.8, 0.12, 0.35, M.belly, v(0, -0.05, 0.3));
  const mouth = new THREE.Object3D(); mouth.position.set(0, -0.3, 1.0); hp.add(mouth);
  registerHead(c, hp, jaw, mouth);
  c.mount('crest', hp, v(0, 0.6, -0.4), v(0, 1, -0.2), v(0, 0, -1));
  c.mount('ram', hp, v(0, 0.35, 0.7), v(0, 0.4, 1));
  for (const s of [-1, 1]) c.mount('tusks', jaw, v(s * 0.55, 0, 0.5), v(s * 0.3, -0.5, 1));
  const tail = c.chain(trunk, { n: 11, len: 0.9, r0: 0.38, r1: 0.05, pos: v(0, 0.05, -3.0), rot: [-(PI / 2) - 0.08, 0, 0], curlX: 0.02, radial: 8 });
  c.rig.tail = tail;
  c.limb({ type: 'tail', joints: tail, ampX: 0.05, ampZ: 0.12, speed: 1.6 });
  c.mount('tailTip', tail.tip, v(0, 0, 0), v(0, 1, 0), v(0, 0, 1));
  // default barb
  const barb = c.g(tail.tip); barb.rotation.x = 0;
  const bg = new THREE.ConeGeometry(0.16, 0.9, 6); bg.translate(0, 0.4, 0);
  c.m(bg, M.accent, barb);
}

export const PLAN_BUILDERS = {
  biped: buildBiped, ape: buildApe, cephalopod: buildCephalopod, moth: buildMoth, turtle: buildTurtle,
  mantis: buildMantis, serpent: buildSerpent, mech: buildMech, golem: buildGolem, hydra: buildHydra,
  spider: buildSpider, ray: buildRay,
};
// Hook for swapping a plan for e.g. a GLB-backed builder later. fn(ctx) must fill ctx.rig + ctx.mounts.
export function registerPlanBuilder(plan, fn) { PLAN_BUILDERS[plan] = fn; }

// ---------------------------------------------------------------- upgrades + cosmetics
function pick(list, n) {
  if (!list.length || n <= 0) return [];
  if (n >= list.length) return list.slice();
  const out = [], step = list.length / n;
  for (let i = 0; i < n; i++) out.push(list[Math.floor(i * step)]);
  return out;
}

function applyUpgrades(c) {
  const T = c.T, M = c.M, mt = c.mounts;
  // pow: claws / fists / blades grow; horns appear
  for (const p of c.rig.powParts) p.scale.multiplyScalar(POW_SCALE[T.pow]);
  if (T.pow >= 1 && mt.horns.length) {
    const n = T.pow === 3 ? mt.horns.length : 2;
    const size = [0, 0.6, 0.85, 1.05][T.pow];
    for (const h of pick(mt.horns, n)) attach(h, PART.horn(c, M.horn, 1.4, 1), size);
  }
  // enr: glowing crystals along the back
  if (T.enr >= 1) {
    const list = T.enr === 1 ? mt.dorsal.filter((_, i) => i % 2 === 0) : mt.dorsal;
    const size = [0, 0.6, 0.8, 1.05][T.enr];
    for (const d of list) attach(d, PART.crystal(c), size);
    if (T.enr >= 3) for (const s of mt.shoulders) attach(s, PART.crystal(c), 0.8);
    M.accent.emissiveIntensity *= [1, 1.05, 1.12, 1.22][T.enr];
  }
  // arm: plates, then spikes
  if (T.arm >= 1) {
    const list = T.arm === 1 ? pick(mt.plates, Math.ceil(mt.plates.length / 2)) : mt.plates;
    for (const p of list) {
      const pl = attach(p, PART.plate(c), 1.0);
      if (T.arm >= 3) {
        const sp = PART.spike(c, M.horn, 0.8);
        sp.position.y = 0.18; pl.add(sp);
      }
    }
    if (T.arm >= 3 && c.plan !== 'biped') for (const d of pick(mt.dorsal, 4)) attach(d, PART.spike(c, M.horn, 0.9), 0.8);
  }
  // spd: fins / frills
  if (T.spd >= 1) {
    const list = T.spd === 1 ? mt.fins.slice(0, 2) : mt.fins;
    for (const f of list) attach(f, PART.fin(c, M.membrane, 1.3), [0, 0.9, 1.1, 1.3][T.spd]);
    if (T.spd >= 3 && mt.crest) {
      const fr = new THREE.Group();
      for (let i = -2; i <= 2; i++) {
        const f = PART.fin(c, M.membrane, 1.1);
        f.rotation.z = i * 0.35; f.position.z = -0.2;
        fr.add(f);
      }
      const back = { ...mt.crest, pos: mt.crest.pos.clone().add(v(0, -0.1, -0.35)) };
      attach(back, fr, 0.8);
    }
  }
  // fcs: extra eyes, crest, third eye
  if (T.fcs >= 1 && mt.eyes.length) {
    const n = T.fcs >= 3 ? mt.eyes.length : 2;
    for (const e of pick(mt.eyes, n)) attach(e, PART.eye(c, 0.17), 1);
  }
  if (T.fcs >= 2 && mt.crest) {
    const crest = new THREE.Group();
    const s = new THREE.Shape();
    s.moveTo(0.6, 0); s.quadraticCurveTo(0.3, 1.3, -1.2, 1.6); s.quadraticCurveTo(-0.9, 0.6, -1.4, 0); s.lineTo(0.6, 0);
    const geo = new THREE.ExtrudeGeometry(s, { depth: 0.08, bevelEnabled: true, bevelThickness: 0.04, bevelSize: 0.04, bevelSegments: 1 });
    geo.translate(0, 0, -0.04); geo.rotateY(-PI / 2);
    c.m(geo, M.accent, crest);
    attach(mt.crest, crest, T.fcs >= 3 ? 1.0 : 0.75);
    if (T.fcs >= 3) {
      const third = { ...mt.crest, pos: mt.crest.pos.clone().add(v(0, -0.05, 0.4)), n: v(0, 0.4, 1).normalize(), t: v(0, 1, 0) };
      attach(third, PART.eye(c, 0.26), 1);
    }
  }
}

const VIS = {
  dorsal_glow(c) {
    for (const d of c.mounts.dorsal) {
      const f = PART.fin(c, c.M.accent, 1.2);
      attach(d, f, 0.85);
    }
    c.rig.extra.glowPulse = true;
  },
  missile_pods(c) {
    const list = c.mounts.shoulders.length ? c.mounts.shoulders : pick(c.mounts.dorsal, 2);
    for (const s of list) {
      const pod = new THREE.Group();
      c.rbox(pod, 1.0, 0.75, 1.3, 0.12, c.M.plate, v(0, 0.38, 0));
      const tip = new THREE.CylinderGeometry(0.1, 0.12, 0.3, 8); tip.rotateX(PI / 2);
      const cone = new THREE.ConeGeometry(0.1, 0.18, 8); cone.rotateX(PI / 2);
      for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) {
        c.m(tip, c.M.dark, pod, v(-0.3 + i * 0.3, 0.22 + j * 0.32, 0.6));
        c.m(cone, c.M.accent, pod, v(-0.3 + i * 0.3, 0.22 + j * 0.32, 0.82));
      }
      // keep pods pointing forward regardless of mount normal
      attach({ ...s, t: v(0, 0, 1) }, pod, 1);
    }
  },
  tail_club(c) {
    const club = new THREE.Group();
    c.ell(club, 0.85, 0.85, 1.0, c.M.plate, v(0, 0.6, 0));
    for (let i = 0; i < 8; i++) {
      const sp = PART.spike(c, c.M.horn, 0.7);
      const n = v(Math.cos(i * 0.785), Math.sin(i * 1.3) * 0.6, Math.sin(i * 0.785)).normalize();
      sp.position.set(0, 0.6, 0).addScaledVector(n, 0.75);
      orientTo(sp, n, v(0, 1, 0));
      club.add(sp);
    }
    attach(c.mounts.tailTip || c.mounts.back, club, 1);
  },
  stinger(c) {
    const st = new THREE.Group();
    c.ell(st, 0.45, 0.6, 0.45, c.M.accent, v(0, 0.35, 0));
    c.tube(st, [v(0, 0.6, 0), v(0, 1.2, 0.1), v(0, 1.6, 0.5), v(0, 1.7, 1.0)], 0.28, 0.01, c.M.horn, { seg: 12, rad: 8 });
    attach(c.mounts.tailTip || c.mounts.back, st, 1);
  },
  tusks(c) {
    for (const t of c.mounts.tusks) {
      const tk = new THREE.Group();
      c.tube(tk, [v(0, 0, 0), v(0, 0.5, 0.1), v(0, 0.9, 0.45), v(0, 1.05, 0.95)], 0.16, 0.015, c.M.horn, { seg: 10, rad: 8 });
      attach({ ...t, t: v(0, 0, 1) }, tk, 1);
    }
  },
  ram_horn(c) {
    const mt = c.mounts.ram; if (!mt) return;
    const h = new THREE.Group();
    c.ell(h, 0.55, 0.25, 0.65, c.M.plate, v(0, 0.05, 0));
    c.tube(h, [v(0, 0, 0), v(0, 0.7, 0.25), v(0, 1.3, 0.1), v(0, 1.7, -0.4)], 0.42, 0.02, c.M.horn, { seg: 12, rad: 10 });
    attach({ ...mt, t: v(0, 1, 0) }, h, 1);
  },
  vents(c) {
    const list = pick(c.mounts.dorsal.length ? c.mounts.dorsal : [c.mounts.back].filter(Boolean), 3);
    for (const d of list) {
      const vt = new THREE.Group();
      c.m(new THREE.CylinderGeometry(0.32, 0.45, 0.9, 10, 1, true), c.M.dark, vt, v(0, 0.45, 0));
      c.m(new THREE.CylinderGeometry(0.3, 0.3, 0.05, 10), c.M.accent, vt, v(0, 0.75, 0));
      attach(d, vt, 0.9);
    }
  },
  core(c) {
    const mt = c.mounts.chest; if (!mt) return;
    const g = new THREE.Group();
    c.ell(g, 0.6, 0.6, 0.6, c.M.accent, v(0, 0.15, 0), null, 18);
    const ring = new THREE.TorusGeometry(0.72, 0.13, 8, 24); ring.rotateX(PI / 2);
    c.m(ring, c.M.plate, g, v(0, 0.1, 0));
    attach(mt, g, 1);
  },
  jaw(c) {
    if (c.rig.jaws.length) {
      for (const j of c.rig.jaws) {
        j.scale.multiplyScalar(1.3);
        const tg = new THREE.ConeGeometry(0.07, 0.3, 6);
        for (let i = 0; i < 4; i++) for (const s of [-1, 1]) c.m(tg, c.M.horn, j, v(s * (0.35 - i * 0.04), 0.12, 0.6 + i * 0.25));
      }
    } else if (c.rig.mouth) {
      for (const s of [-1, 1]) c.claw(c.rig.mouth.parent, c.rig.mouth.position.clone().add(v(s * 0.3, 0, -0.2)), v(s * 0.3, -1, 0.6), 0.6, 0.12);
    }
  },
};

function applyVis(c) {
  const seen = new Set();
  for (const id of c.abilities) {
    const ab = ABILITY_BY_ID[id];
    const key = ab?.vis;
    if (!key || seen.has(key)) continue;
    seen.add(key);
    try { VIS[key]?.(c); } catch (e) { console.warn('[kaiju-model] cosmetic failed', key, e); }
  }
  c.rig.vis = [...seen];
}

// ---------------------------------------------------------------- animation
const STATUS_TINT = {
  burn: ['#ff5a10', 0.38, 'flicker'], poison: ['#3aff3a', 0.28, 'pulse'], bleed: ['#b00010', 0.28, 'pulse'],
  slow: ['#3aa8ff', 0.3, 'steady'], stun: ['#ffee55', 0.12, 'pulse'], bind: ['#d8d0b8', 0.06, 'steady'],
  blind: ['#000000', 0, 'steady'], weaken: ['#8a3aff', 0.16, 'pulse'], sunder: ['#ff9a6a', 0.08, 'pulse'],
  enrage: ['#ff1a1a', 0.22, 'pulse'], charged: ['accent', 0.16, 'pulse'], fortify: ['#9aa8b8', 0.1, 'steady'],
  haste: ['#6affff', 0.1, 'pulse'], evasive: ['#c8c8ff', 0.08, 'pulse'], regen: ['#5aff8a', 0.14, 'pulse'],
  shield: ['accent', 0.04, 'steady'],
};

function shieldMaterial(color) {
  return new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Color(color) }, uTime: { value: 0 }, uAlpha: { value: 1 } },
    vertexShader: `varying vec3 vN; varying vec3 vV; varying vec3 vP;
      void main(){ vec4 mv = modelViewMatrix*vec4(position,1.0); vN = normalize(normalMatrix*normal); vV = normalize(-mv.xyz); vP = position; gl_Position = projectionMatrix*mv; }`,
    fragmentShader: `uniform vec3 uColor; uniform float uTime; uniform float uAlpha; varying vec3 vN; varying vec3 vV; varying vec3 vP;
      void main(){ float f = pow(1.0 - abs(dot(vN, vV)), 2.2);
        float hex = 0.5 + 0.5*sin(vP.x*6.0 + uTime*2.0)*sin(vP.y*6.0 - uTime*1.5)*sin(vP.z*6.0);
        float a = (0.06 + f*0.85 + hex*0.06) * uAlpha;
        gl_FragColor = vec4(uColor*(0.6+f*1.6), a); }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
  });
}

function animLimb(L, a, tt, roar) {
  const ph = L.phase || 0;
  switch (L.type) {
    case 'leg': {
      const w = 6.5, sw = Math.sin(tt * w + ph), amp = 0.42 * a.walk;
      const r = L.pivot.userData.rest;
      if (L.axis === 'y') {
        L.pivot.rotation.y = r.y + sw * amp * 0.5;
        if (L.knee) L.knee.rotation.x = L.knee.userData.rest.x - Math.max(0, Math.cos(tt * w + ph)) * amp * 0.5 + Math.sin(tt * 1.3 + ph) * 0.015;
      } else {
        L.pivot.rotation.x = r.x + sw * amp + Math.sin(tt * 1.2 + ph) * 0.01;
        if (L.knee) L.knee.rotation.x = L.knee.userData.rest.x + Math.max(0, -Math.cos(tt * w + ph)) * amp * 1.1;
      }
      break;
    }
    case 'arm': {
      const r = L.pivot.userData.rest;
      const vic = a.victory * (L.vicRaise ?? -2.2) * (0.85 + 0.15 * Math.sin(tt * 3 + ph));
      L.pivot.rotation.x = r.x + Math.sin(tt * 1.4 + ph) * 0.05 + a.lunge * (L.reach ?? -1.1) + vic
        + Math.sin(tt * 6.5 + ph * 1.6) * 0.3 * a.walk + a.roar * -0.3;
      if (L.elbow) L.elbow.rotation.x = L.elbow.userData.rest.x + a.lunge * (L.elbowReach ?? 0.5) + Math.sin(tt * 1.4 + ph) * 0.03;
      break;
    }
    case 'tentacle': case 'neck': case 'tail': {
      const J = L.joints; if (!J) break;
      const ax = L.ampX ?? 0.08, az = L.ampZ ?? 0.08, spd = (L.speed ?? 1.4) * (1 + a.walk);
      const straighten = L.reach ? a.lunge : (L.type === 'neck' ? a.lunge * 0.6 : 0);
      for (let i = 0; i < J.length; i++) {
        const j = J[i], r = j.userData.rest, k = (i + 1) / J.length;
        const rx = i === 0 ? r.x : r.x * (1 - straighten * 0.85);
        j.rotation.x = rx + Math.sin(tt * spd + i * 0.55 + ph) * ax * (0.4 + k) - (L.type === 'neck' && i > 0 ? roar * 0.06 : 0);
        j.rotation.z = r.z + Math.cos(tt * spd * 0.8 + i * 0.5 + ph) * az * (0.4 + k) * (1 + a.walk);
      }
      break;
    }
    case 'wing': {
      const r = L.pivot.userData.rest;
      const boost = 1 + a.lunge * 0.8 + a.walk * 0.4 + a.victory * 0.5;
      const f = Math.sin(tt * (L.speed || 6) * (1 + a.lunge * 0.6) + ph);
      L.pivot.rotation.z = r.z + L.side * f * (L.amp || 0.4) * boost * (1 - a.ko * 0.9);
      break;
    }
  }
}

function makeRigRuntime(rig, model, colors) {
  const accentCol = safeColor(colors.accent, DEFAULT_COLORS.accent);
  const mats = new Set();
  model.traverse(o => {
    if (o.isMesh && o.material && o.material.isMeshStandardMaterial && !o.userData.fx) mats.add(o.material);
  });
  rig.mats = [...mats];
  for (const m of rig.mats) {
    m.userData.e0 = m.emissive.clone().multiplyScalar(m.emissiveIntensity);
    m.emissiveIntensity = 1;
  }
  rig.anim = { lunge: 0, reach: 4, recoil: 0, roar: 0, bite: 0, charge: 0, walk: 0, lift: 0, ko: 0, victory: 0, flash: 0, flashColor: new THREE.Color('#ff2020'), spin: 0, dodge: 0 };
  rig.status = new Set();
  rig.seed = (hashStr(rig.name || '') % 1000) / 100;
  const tint = new THREE.Color(), tmp = new THREE.Color();
  const bounds = rig.localBounds;
  const ctr = bounds.getCenter(new V3());
  const size = bounds.getSize(new V3());
  let stars = null, bubble = null, rings = null;

  const ensureStars = () => {
    if (stars) return stars;
    stars = new THREE.Group(); stars.userData.fx = true;
    const g = new THREE.OctahedronGeometry(0.35, 0);
    const m = new THREE.MeshBasicMaterial({ color: '#ffe84a' });
    for (let i = 0; i < 4; i++) { const s = new THREE.Mesh(g, m); s.userData.fx = true; stars.add(s); }
    stars.position.set(0, bounds.max.y + 0.8, ctr.z * 0.3 + (rig.headLocalZ || 0) * 0.7);
    rig.motion.add(stars);
    return stars;
  };
  const ensureBubble = () => {
    if (bubble) return bubble;
    const r = Math.max(size.x, size.y, Math.min(size.z, size.y)) * 0.6;
    bubble = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 20), shieldMaterial(accentCol));
    bubble.userData.fx = true; bubble.scale.setScalar(r); bubble.position.set(ctr.x, ctr.y, ctr.z * 0.3);
    bubble.renderOrder = 5;
    rig.motion.add(bubble);
    return bubble;
  };
  const ensureRings = () => {
    if (rings) return rings;
    rings = new THREE.Group(); rings.userData.fx = true;
    const m = new THREE.MeshStandardMaterial({ color: '#e8e2d0', roughness: 0.8 });
    const rr = Math.max(size.x, size.z) * 0.36;
    for (let i = 0; i < 3; i++) {
      const t = new THREE.Mesh(new THREE.TorusGeometry(rr, 0.12, 6, 32), m);
      t.userData.fx = true; t.rotation.x = PI / 2 + (i - 1) * 0.25; t.position.y = ctr.y + (i - 1) * size.y * 0.15;
      t.position.z = ctr.z * 0.5;
      rings.add(t);
    }
    rig.motion.add(rings);
    return rings;
  };

  rig.setStatus = (id, on = true) => { if (on) rig.status.add(id); else rig.status.delete(id); };
  rig.clearStatuses = () => rig.status.clear();
  rig.hit = (color = '#ff2020', amt = 1) => { rig.anim.flash = amt; rig.anim.flashColor.set(color); };

  rig.update = (dt, t) => {
    const a = rig.anim, tt = t + rig.seed;
    a.flash = Math.max(0, a.flash - dt * 3.2);
    // breathing
    const br = Math.sin(tt * 1.7) * 0.5 + 0.5;
    const bs = rig.bodyScale;
    rig.body.scale.set(bs.x * (1 + 0.035 * br * (1 - a.ko)), bs.y * (1 + 0.02 * br * (1 - a.ko)), bs.z * (1 + 0.03 * br * (1 - a.ko)));
    // whole-body motion
    const m = rig.motion;
    const roar = Math.max(a.roar, a.victory * (0.5 + 0.5 * Math.sin(tt * 2.2)));
    const hover = rig.hover ? (rig.hover + Math.sin(tt * 1.9) * 0.35) * (1 - a.ko) : 0;
    const walkBob = Math.abs(Math.sin(tt * 6.5)) * 0.22 * a.walk;
    const hop = Math.abs(Math.sin(tt * 3.2)) * 0.35 * a.victory;
    m.position.set(a.dodge, hover + walkBob + hop + a.lift, a.lunge * a.reach - a.recoil * 1.6);
    m.rotation.set(a.lunge * 0.16 - a.recoil * 0.22 - roar * 0.12, 0, 0);
    m.scale.set(1, 1, 1);
    if (rig.extra.bank && rig.bankSign) m.rotation.z += rig.extra.bank * rig.bankSign * (1 - a.ko) + Math.sin(tt * 0.9) * 0.06;
    const stunned = rig.status.has('stun');
    if (stunned) m.rotation.z += Math.sin(tt * 8) * 0.07;
    switch (rig.koMode) {
      case 'side': m.rotation.z += a.ko * 1.45; m.position.y -= a.ko * 0.3; break;
      case 'flop': m.scale.y = 1 - 0.5 * a.ko; m.rotation.x -= a.ko * 0.25; break;
      case 'tilt': m.rotation.z += a.ko * 0.35; m.rotation.x -= a.ko * 0.15; break;
      default: m.rotation.x -= a.ko * 1.42; break;
    }
    // heads + jaws
    for (const h of rig.heads) {
      const r = h.userData.rest;
      h.rotation.x = r.x - roar * 0.55 + Math.sin(tt * 1.1 + h.userData.ph) * 0.05 + a.lunge * 0.12 + a.ko * 0.3;
      h.rotation.y = r.y + Math.sin(tt * 0.7 + h.userData.ph) * 0.08 * (1 - a.ko);
    }
    const open = Math.min(0.8, roar * 0.6 + a.bite * 0.5 + a.charge * 0.45);
    for (const j of rig.jaws) j.rotation.x = j.userData.rest.x + open;
    for (const L of rig.limbs) animLimb(L, a, tt, roar);
    // tints
    tint.setRGB(0, 0, 0);
    for (const st of rig.status) {
      const d = STATUS_TINT[st]; if (!d) continue;
      let k = d[1];
      if (d[2] === 'flicker') k *= 0.6 + 0.4 * Math.sin(tt * 23) * Math.sin(tt * 7.3);
      else if (d[2] === 'pulse') k *= 0.55 + 0.45 * Math.sin(tt * 4);
      tmp.copy(d[0] === 'accent' ? accentCol : tmp.set(d[0])).multiplyScalar(k);
      tint.add(tmp);
    }
    const tmax = Math.max(tint.r, tint.g, tint.b);
    if (tmax > 0.4) tint.multiplyScalar(0.4 / tmax);
    if (a.flash > 0) tint.lerp(tmp.copy(a.flashColor).multiplyScalar(1.4), Math.min(1, a.flash));
    const glow = 1 + a.charge * 1.6 + (rig.extra.glowPulse ? 0.3 * Math.sin(tt * 3) : 0);
    for (const mat of rig.mats) {
      const e0 = mat.userData.e0;
      let mul = 1;
      if (mat.userData.glow) mul = glow;
      if (mat.userData.eye) mul = 1 - a.ko * 0.95;
      mat.emissive.copy(e0).multiplyScalar(mul).add(tint);
    }
    // status props
    if (stunned || stars) {
      const s = ensureStars(); s.visible = stunned;
      if (stunned) s.children.forEach((c, i) => { const an = tt * 3 + i * PI / 2; c.position.set(Math.cos(an) * 1.6, Math.sin(tt * 5 + i) * 0.2, Math.sin(an) * 1.6); c.rotation.y = tt * 4; });
    }
    const shielded = rig.status.has('shield') || a.shieldPulse > 0;
    if (shielded || bubble) {
      const b = ensureBubble(); b.visible = shielded && a.ko < 0.5;
      b.material.uniforms.uTime.value = tt;
      b.material.uniforms.uAlpha.value = Math.max(rig.status.has('shield') ? 1 : 0, a.shieldPulse || 0) * (1 + a.flash * 0.8);
    }
    const bound = rig.status.has('bind');
    if (bound || rings) { const r = ensureRings(); r.visible = bound; if (bound) r.children.forEach((c, i) => { c.rotation.z = tt * (0.4 + i * 0.2); }); }
  };
}

// ---------------------------------------------------------------- public
export function buildKaijuModel(snapshot = {}) {
  snapshot = snapshot || {};
  const frame = FRAME_BY_ID[snapshot.frame] || null;
  let plan = snapshot.plan || frame?.plan || 'biped';
  if (!PLAN_BUILDERS[plan]) plan = 'biped';
  const colors = { ...DEFAULT_COLORS, ...(frame?.colors || {}), ...(snapshot.colors || {}) };
  let tiers;
  try { tiers = visualTiers({ levels: snapshot.levels || {} }); } catch { tiers = null; }
  tiers = tiers || { hp: 0, pow: 0, enr: 0, arm: 0, spd: 0, fcs: 0 };
  for (const k of ['hp', 'pow', 'enr', 'arm', 'spd', 'fcs']) tiers[k] = Math.max(0, Math.min(3, tiers[k] | 0));
  const abilities = Array.isArray(snapshot.abilities) ? snapshot.abilities.slice() : [];
  if (frame?.basic) abilities.push(frame.basic);
  const seed = hashStr(snapshot.id || snapshot.name || snapshot.frame || 'kaiju');

  const model = new THREE.Group();
  model.name = 'kaiju:' + (snapshot.name || snapshot.frame || plan);
  let c = new BuildCtx(snapshot, plan, tiers, colors, abilities, seed);
  try {
    PLAN_BUILDERS[plan](c);
  } catch (e) {
    console.warn('[kaiju-model] plan builder failed, falling back to biped', plan, e);
    disposeObject(c.root);
    c = new BuildCtx(snapshot, 'biped', tiers, colors, abilities, seed);
    buildBiped(c);
    plan = 'biped';
  }
  model.add(c.root);
  if (!c.rig.body) c.rig.body = c.root;
  if (!c.rig.headPivot) { const hp = c.g(c.root, v(0, 8, 1)); registerHead(c, hp, null, null); }
  if (!c.rig.mouth) { const m = new THREE.Object3D(); c.rig.headPivot.add(m); c.rig.mouth = m; c.rig.mouths.push(m); }
  c.rig.body.scale.set(c.bulk, 1, (c.bulk + 1) / 2);
  try { applyUpgrades(c); } catch (e) { console.warn('[kaiju-model] upgrades failed', e); }
  try { applyVis(c); } catch (e) { console.warn('[kaiju-model] cosmetics failed', e); }

  // measure at unit scale
  model.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(c.root);
  const S = HP_SIZE[tiers.hp];
  model.scale.setScalar(S);

  const mk = (obj, name) => { if (!obj) return null; return obj; };
  const marker = (mt, name) => {
    if (!mt) return null;
    const o = new THREE.Object3D(); o.name = name; o.position.copy(mt.pos); mt.p.add(o); return o;
  };
  const rig = {
    name: snapshot.name || snapshot.frame || plan,
    plan, tiers, scale: S,
    root: model, motion: c.root, trunk: c.rig.trunk || c.root, body: c.rig.body,
    head: c.rig.headPivot, headPivot: c.rig.headPivot, heads: c.rig.heads,
    jaw: c.rig.jaw || null, jaws: c.rig.jaws,
    mouth: c.rig.mouth, mouths: c.rig.mouths.length ? c.rig.mouths : [c.rig.mouth],
    limbs: c.rig.limbs, tail: c.rig.tail || [],
    hover: c.rig.hover, koMode: c.rig.koMode, extra: c.rig.extra, vis: c.rig.vis || [],
    bodyScale: c.rig.body.scale.clone(),
    localBounds: bounds,
    height: (bounds.max.y + (c.rig.hover || 0)) * S,
    radius: Math.max(-bounds.min.x, bounds.max.x, -bounds.min.z, bounds.max.z) * S,
    front: Math.max(1, bounds.max.z) * S,
    back: Math.max(1, -bounds.min.z) * S,
    width: (bounds.max.x - bounds.min.x) * S,
    mounts: c.mounts,
  };
  rig.attach = {
    head: rig.headPivot, jaw: rig.jaw, mouth: rig.mouth,
    chest: marker(c.mounts.chest, 'chest') || rig.body,
    back: marker(c.mounts.back, 'back') || rig.body,
    tailTip: marker(c.mounts.tailTip, 'tailTip'),
    crown: marker(c.mounts.crest, 'crown'),
    handL: mk(c.rig.extra.handL) || rig.mouth, handR: mk(c.rig.extra.handR) || rig.mouth,
    shoulderL: marker(c.mounts.shoulders[0], 'shoulderL'), shoulderR: marker(c.mounts.shoulders[1], 'shoulderR'),
  };
  rig.headLocalZ = (() => { const p = new V3(); rig.headPivot.getWorldPosition(p); model.worldToLocal(p); return p.z; })();
  makeRigRuntime(rig, model, colors);
  const _p = new V3();
  rig.getMouthWorld = (i = 0, out = new V3()) => { const m = rig.mouths[i % rig.mouths.length] || rig.mouth; m.updateWorldMatrix(true, false); return m.getWorldPosition(out); };
  rig.getCenterWorld = (out = new V3()) => { const o = rig.attach.chest; o.updateWorldMatrix(true, false); o.getWorldPosition(out); return out; };
  rig.getHeadWorld = (out = new V3()) => { rig.headPivot.updateWorldMatrix(true, false); return rig.headPivot.getWorldPosition(out); };
  rig.getTopWorld = (out = new V3()) => { model.getWorldPosition(out); out.y += rig.height * (1 - rig.anim.ko * 0.6) + 1; return out; };
  rig.dispose = () => disposeObject(model);
  model.userData.rig = rig;
  model.userData.snapshot = snapshot;
  model.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return model;
}

export function disposeObject(obj) {
  if (!obj) return;
  obj.traverse(o => {
    if (o.geometry) o.geometry.dispose();
    const mats = Array.isArray(o.material) ? o.material : (o.material ? [o.material] : []);
    for (const m of mats) {
      for (const k of ['map', 'emissiveMap', 'alphaMap', 'normalMap', 'roughnessMap']) if (m[k]) m[k].dispose();
      m.dispose();
    }
  });
  if (obj.parent) obj.parent.remove(obj);
}
export const disposeKaiju = disposeObject;
