// GLB-backed Kaiju models (art/blender -> assets/models/<frame>.glb).
//
// Loading is lazy and per frame: preloadFrames([frameId...]) fetches + parses only what is about to be shown and
// caches the parsed scene. buildKaijuModel() (kaiju-model.js) stays synchronous: it calls buildGlbPlan() when the
// frame's GLB is cached and otherwise falls back to the procedural plan builder.
//
// buildGlbPlan(c, frameId) behaves like a procedural plan builder: it fills the BuildCtx (c.root, c.rig, c.mounts)
// so the stock rig runtime, playback, ability cosmetics, injuries and statues keep working. The differences:
//  - Animated parts are *proxies* (Object3D that are not in the scene graph). The stock animation code writes
//    Euler rotations to them exactly as it does to procedural pivots; applyPose() then converts every proxy into
//    its bone's quaternion. A proxy's Euler is interpreted in a model-space basis (identity = model axes, so
//    "rotation.x" is always pitch no matter how the Blender bone is rolled); chains get a per-joint basis whose
//    +Y runs along the bone and whose +X rotation droops the tip.
//  - Injury scars need raycasts. Skinned raycasts are far too slow, so each bone carries an invisible static
//    "collider" mesh holding the bind-pose triangles it dominates. Scars land on those (in bind pose) and are
//    parented to the bone, so they follow the animation.
//  - Stat tiers are the GLB's own pre-placed parts (pow_tN exclusive, arm/enr/spd/fcs additive).
//  - Player colours are written into per-instance material clones by material NAME; geometry + textures are
//    shared between all clones of a frame (userData.shared, skipped by disposeObject).

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';
import { FRAME_BY_ID } from '../data/frames.js';

const V3 = THREE.Vector3, Q4 = THREE.Quaternion;
const PI = Math.PI;
const UP = new V3(0, 1, 0);
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const smooth = (x) => { x = clamp01(x); return x * x * (3 - 2 * x); };
const BASE_URL = new URL('../../assets/models/', import.meta.url);
const HP_SIZE = [1, 1.1, 1.22, 1.36]; // mirrors kaiju-model.js (tiny frames cancel the kaiju HP sizing)
const GLOW_K = 0.5;
const ASSET_V = '1';                // bump when re-exporting GLBs (GitHub Pages caches by URL)                   // GLB emission strengths are authored for Blender; tame them for bloom

// Per-frame rig hints. Everything else is derived from bone names (see buildRig).
export const GLB_FRAMES = {
  saurian: {},
  ape: { arm: { reach: -1.25, elbowReach: 0.25, vicRaise: -2.6 } },
  kraken: { koMode: 'flop' },
  lepidra: { koMode: 'side', hover: 2.6 },
  carapace: { koMode: 'side', quad: true },
  mantid: { koMode: 'side' },
  serpent: { koMode: 'flop' },
  mecha: { arm: { reach: -0.9, elbowReach: 0.4 } },
  golem: { arm: { reach: -1.2, elbowReach: 0.35 } },
  hydra: { koMode: 'side', quad: true },
  arachnid: { koMode: 'side' },
  stormray: { koMode: 'tilt', hover: 3.7, bank: 0.3 },
  // Rare Wild frames stay on the procedural wild-models.js for now: the GLB runtimes don't yet port their
  // special behaviour (cape sim, beam eyes, tag-team swap, air strikes, reinforcements). Re-enable once ported.
  // superhero: { tiny: true, hover: 0.9, wild: 'hero' },
  // magicalgirl: { tiny: true, hover: 0.8, wild: 'hero' },
  // heroteam: { tiny: true, koMode: 'team', wild: 'team' },
  // army: { koMode: 'army', wild: 'army' },
};

let enabled = true;
try { if (new URLSearchParams(globalThis.location?.search || '').get('glb') === '0') enabled = false; } catch { /* no location */ }
export function setGlbEnabled(on) { enabled = !!on; }
export function glbEnabled() { return enabled; }

// ---------------------------------------------------------------- cache + loading
const cache = new Map(); // frameId -> { state: 'loading' | 'ready' | 'failed', promise, scene, colliders, eyes, defaults }
const MAX_CACHED = 8;
let loader = null;

export function glbReady(frameId) { return enabled && cache.get(frameId)?.state === 'ready'; }
// frames that still need a network round-trip before they can be built from their GLB
export function glbMissing(ids) {
  if (!enabled) return [];
  return [...new Set(ids)].filter(id => id && GLB_FRAMES[id] && !cache.get(id));
}

export function preloadFrames(ids) {
  if (!enabled) return Promise.resolve();
  const jobs = [];
  for (const id of new Set(ids || [])) {
    if (!id || !GLB_FRAMES[id]) continue;
    let e = cache.get(id);
    if (!e) {
      e = { state: 'loading', used: 0 };
      cache.set(id, e);
      loader = loader || new GLTFLoader();
      e.promise = loader.loadAsync(new URL(id + '.glb?v=' + ASSET_V, BASE_URL).href)
        .then((gltf) => { prepare(id, e, gltf); e.state = 'ready'; })
        .catch((err) => { e.state = 'failed'; console.warn('[glb-models] could not load', id, '- using the procedural model', err); });
    }
    if (e.state === 'loading') jobs.push(e.promise);
  }
  return Promise.all(jobs).then(() => { evict(new Set(ids)); });
}

function evict(keep) {
  const ready = [...cache].filter(([id, e]) => e.state === 'ready' && !keep.has(id));
  let n = [...cache.values()].filter(e => e.state === 'ready').length;
  ready.sort((a, b) => a[1].used - b[1].used);
  for (const [id, e] of ready) {
    if (n <= MAX_CACHED) break;
    // live clones keep working: three re-uploads a disposed geometry / texture the next time it is drawn
    for (const g of e.geos) g.dispose();
    for (const t of e.textures) t.dispose();
    cache.delete(id); n--;
  }
}

const COLLIDER_MATS = new Set(['body', 'belly', 'plate', 'hide', 'mane', 'beak', 'skin', 'gun', 'hair']);
const nodeName = (o) => o?.userData?.name || '';
const TIER_RE = /^(pow|arm|enr|spd|fcs)_t(\d)$/;

function prepare(id, e, gltf) {
  const scene = gltf.scene;
  scene.updateMatrixWorld(true);
  e.scene = scene;
  e.geos = new Set(); e.textures = new Set();
  e.srcMats = new Map(); // name -> first source material (colour reference)
  let skeleton = null;
  const bodyMeshes = [];
  scene.traverse((o) => {
    if (!o.isMesh) return;
    const g = o.geometry;
    g.userData.shared = true; e.geos.add(g);
    if (!g.boundingBox) g.computeBoundingBox();
    if (!g.boundingSphere) g.computeBoundingSphere();
    if (o.isSkinnedMesh) {
      // bind pose == geometry space: skip three's per-vertex skinned bounds
      o.boundingBox = g.boundingBox.clone(); o.boundingSphere = g.boundingSphere.clone();
      skeleton = skeleton || o.skeleton;
    }
    const m = o.material;
    if (m && !e.srcMats.has(m.name)) e.srcMats.set(m.name, m);
    for (const k of ['map', 'normalMap', 'emissiveMap', 'roughnessMap', 'metalnessMap', 'aoMap']) if (m?.[k]) { m[k].userData.shared = true; e.textures.add(m[k]); }
    const owner = nodeName(o) || nodeName(o.parent);
    if (owner === 'body' && o.isSkinnedMesh) bodyMeshes.push(o);
  });
  e.colliders = []; e.eyes = [];
  if (!skeleton) return;
  // ---- colliders: bind-pose triangles grouped by their dominant bone, stored in that bone's local space
  const bones = skeleton.bones, inv = skeleton.boneInverses;
  const soup = new Map(); // bone index -> number[]
  const P = [new V3(), new V3(), new V3()];
  const eyePts = [];
  for (const me of bodyMeshes) {
    const g = me.geometry, pos = g.attributes.position, si = g.attributes.skinIndex, sw = g.attributes.skinWeight, idx = g.index;
    if (!pos || !si || !sw) continue;
    const dom = (vi) => {
      let b = si.getX(vi), w = sw.getX(vi);
      if (sw.getY(vi) > w) { w = sw.getY(vi); b = si.getY(vi); }
      if (sw.getZ(vi) > w) { w = sw.getZ(vi); b = si.getZ(vi); }
      if (sw.getW(vi) > w) { b = si.getW(vi); }
      return b;
    };
    const name = me.material?.name || '';
    if (name === 'eye') {
      for (let i = 0; i < pos.count; i++) eyePts.push({ p: new V3().fromBufferAttribute(pos, i), b: dom(i) });
      continue;
    }
    if (!COLLIDER_MATS.has(name)) continue;
    const nTri = (idx ? idx.count : pos.count) / 3;
    for (let t = 0; t < nTri; t++) {
      const a = idx ? idx.getX(t * 3) : t * 3, b = idx ? idx.getX(t * 3 + 1) : t * 3 + 1, c = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
      const bi = dom(a);
      let arr = soup.get(bi); if (!arr) { arr = []; soup.set(bi, arr); }
      const M = inv[bi];
      P[0].fromBufferAttribute(pos, a).applyMatrix4(M); P[1].fromBufferAttribute(pos, b).applyMatrix4(M); P[2].fromBufferAttribute(pos, c).applyMatrix4(M);
      arr.push(P[0].x, P[0].y, P[0].z, P[1].x, P[1].y, P[1].z, P[2].x, P[2].y, P[2].z);
    }
  }
  for (const [bi, arr] of soup) {
    if (arr.length < 9 * 4) continue;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(arr, 3));
    g.computeBoundingBox(); g.computeBoundingSphere();
    g.userData.shared = true; e.geos.add(g);
    e.colliders.push({ bone: nodeName(bones[bi]), geo: g });
  }
  // ---- eyes (for the eye injury): one cluster per side
  for (const side of [1, -1, 0]) {
    const pts = eyePts.filter(q => side ? q.p.x * side > 0.03 : Math.abs(q.p.x) <= 0.03);
    if (pts.length < 3) continue;
    const ctr = new V3(); for (const q of pts) ctr.add(q.p); ctr.multiplyScalar(1 / pts.length);
    let r2 = 0; for (const q of pts) r2 += q.p.distanceToSquared(ctr);
    e.eyes.push({ center: ctr, r: Math.sqrt(r2 / pts.length) * 1.35, bone: nodeName(bones[pts[0].b]) });
  }
}

// ---------------------------------------------------------------- colours
const hexSrgb = (h) => { const c = new THREE.Color(); try { c.set(h); } catch { c.set('#808080'); } const o = {}; c.getRGB(o, THREE.SRGBColorSpace); return new V3(o.r, o.g, o.b); };
const mix = (a, b, t) => a.clone().lerp(b, t);
// the same derivations as art/blender/kaiju/materials.py palette() (in sRGB, like the exporter)
const DERIVE = {
  body: (p) => p.body.clone(),
  belly: (p) => p.belly.clone(),
  plate: (p) => mix(p.body.clone().multiplyScalar(0.7), p.belly, 0.42),
  horn: (p) => mix(hexSrgb('#e6dcc4'), p.belly, 0.25),
  membrane: (p) => mix(p.belly, p.accent, 0.3),
  accent: (p) => p.accent.clone().multiplyScalar(0.6),
  cape: (p) => p.belly.clone(),
  ribbon: (p) => p.belly.clone(),
  ribbon2: (p) => p.body.clone(),
};
const _o = {};
const getSrgb = (col) => { col.getRGB(_o, THREE.SRGBColorSpace); return new V3(_o.r, _o.g, _o.b); };
const setSrgb = (col, v) => col.setRGB(clamp01(v.x), clamp01(v.y), clamp01(v.z), THREE.SRGBColorSpace);

// Slot colour = the artist's exported colour + (derived(player) - derived(frame default)): identical to the GLB
// when the player keeps the frame's default palette, and shifts with the player's colours otherwise.
function recolour(mat, name, pal, def, glowK = 1) {
  const fn = DERIVE[name];
  if (fn && mat.color) setSrgb(mat.color, getSrgb(mat.color).add(fn(pal)).sub(fn(def)));
  const em = mat.emissive;
  if (em && em.r + em.g + em.b > 0.02) {
    const es = getSrgb(em);
    if (name === 'accent' || es.distanceTo(def.accent) < 0.14) {
      if (name !== 'accent' && mat.color) {
        // accent-coloured eyes etc.: keep the brightness ratio, swap the hue
        const cs = getSrgb(mat.color), dl = def.accent.length() || 1;
        if (cs.length() > 0.05 && cs.clone().normalize().dot(def.accent.clone().normalize()) > 0.96) setSrgb(mat.color, pal.accent.clone().multiplyScalar(cs.length() / dl));
      }
      setSrgb(em, es.add(pal.accent).sub(def.accent));
    }
    mat.emissiveIntensity = (mat.emissiveIntensity ?? 1) * GLOW_K * glowK;
    if (name === 'accent') mat.userData.glow = true;
    if (name === 'eye' || name === 'iris') mat.userData.eye = true;
  }
  mat.userData.slot = name;
}

// ---------------------------------------------------------------- build
const _q = new Q4(), _q2 = new Q4(), _m4 = new THREE.Matrix4(), _ray = new THREE.Raycaster();
const basisQ = (x, y, z) => new Q4().setFromRotationMatrix(_m4.makeBasis(x, y, z));

export function buildGlbPlan(c, frameId) {
  const e = cache.get(frameId), cfg = GLB_FRAMES[frameId];
  if (!e || e.state !== 'ready' || !cfg) throw new Error('GLB not loaded: ' + frameId);
  e.used = performance.now();
  const T = c.T, statue = c.snap?.fallen === true;
  const inst = SkeletonUtils.clone(e.scene);
  inst.name = 'glb:' + frameId;

  // ---- tiers: pow is exclusive (highest part <= tier), the other families are additive
  const rigNode = inst.getObjectByName('rig') || inst;
  const powHave = [];
  for (const o of rigNode.children) { const m = TIER_RE.exec(nodeName(o)); if (m && m[1] === 'pow') powHave.push(+m[2]); }
  const powShow = Math.max(-1, ...powHave.filter(n => n <= T.pow));
  for (const o of rigNode.children.slice()) {
    const m = TIER_RE.exec(nodeName(o));
    if (!m) continue;
    const fam = m[1], n = +m[2];
    if (fam === 'pow' ? n !== powShow : n > (T[fam] | 0)) o.parent.remove(o);
  }

  // ---- one skeleton for every part, per-instance materials (recoloured by slot name)
  const frame = FRAME_BY_ID[frameId];
  const defC = { body: '#4a4a52', belly: '#9a9a8a', accent: '#5fd6ff', ...(frame?.colors || {}) };
  const def = { body: hexSrgb(defC.body), belly: hexSrgb(defC.belly), accent: hexSrgb(defC.accent) };
  const pal = { body: hexSrgb(c.colors.body), belly: hexSrgb(c.colors.belly), accent: hexSrgb(c.colors.accent) };
  const matMap = new Map();
  let skeleton = null;
  const skinned = [];
  inst.traverse((o) => {
    if (!o.isMesh) return;
    let m = matMap.get(o.material);
    if (!m) {
      m = o.material.clone();
      m.userData = { ...o.material.userData };
      recolour(m, o.material.name, pal, def, cfg.tiny ? 0.5 : 1); // tiny frames are framed close: bloom swamps them
      if (m.userData.glow) m.emissiveIntensity *= [1, 1.05, 1.12, 1.22][T.enr | 0];
      matMap.set(o.material, m);
    }
    o.material = m;
    o.castShadow = true; o.receiveShadow = true;
    if (o.isSkinnedMesh) {
      if (!skeleton) skeleton = o.skeleton; else o.skeleton = skeleton;
      o.frustumCulled = false;
      if (statue) { o.geometry = o.geometry.clone(); o.geometry.userData = {}; } // statues bake weathering into vertex colours
      skinned.push(o);
    }
  });

  // ---- wrappers: motion (c.root) > body (bulk / breathing scale) > glb
  const body = new THREE.Group(); body.name = 'body';
  c.root.add(body); body.add(inst);
  if (cfg.tiny) inst.scale.setScalar((1 + 0.04 * T.hp) / HP_SIZE[T.hp]);
  c.rig.body = body; c.rig.trunk = body;
  c.root.updateMatrixWorld(true);

  const bones = new Map(), empties = new Map();
  inst.traverse((o) => {
    const n = nodeName(o);
    if (!n) return;
    if (o.isBone) bones.set(n, o);
    else if (!o.isMesh && o.parent?.isBone && !empties.has(n)) empties.set(n, o);
  });
  const S = inst.scale.x; // model units per GLB unit

  // ---- proxies
  const poses = [], proxyOf = new Map();
  const proxy = (bone, B) => {
    if (!bone) return null;
    let p = proxyOf.get(bone);
    if (p) return p;
    p = new THREE.Object3D();
    p.userData.rest = new THREE.Euler(0, 0, 0);
    p.userData.bone = bone;
    const Pq = bone.parent.getWorldQuaternion(new Q4());
    const A = Pq.clone().invert(), C = Pq.clone().multiply(bone.quaternion);
    if (B) { A.multiply(B); C.premultiply(B.clone().invert()); }
    poses.push({ p, bone, A, C, Pinv: Pq.clone().invert(), s0: bone.scale.clone(), p0: bone.position.clone(), x: NaN, y: NaN, z: NaN });
    proxyOf.set(bone, p);
    return p;
  };
  const _off = new V3();
  const applyPose = (bounds = false) => {
    for (const k of poses) {
      const r = k.p.rotation;
      if (r.x !== k.x || r.y !== k.y || r.z !== k.z) {
        k.x = r.x; k.y = r.y; k.z = r.z;
        k.bone.quaternion.copy(k.A).multiply(_q.setFromEuler(r)).multiply(k.C);
      }
      k.bone.scale.copy(k.s0).multiply(k.p.scale);
      // proxy.position = model-space offset from the bone's rest position
      k.bone.position.copy(k.p0).add(_off.copy(k.p.position).applyQuaternion(k.Pinv).divideScalar(S));
    }
    if (bounds) {
      c.root.updateMatrixWorld(true);
      for (const o of skinned) { o.computeBoundingBox(); o.computeBoundingSphere(); }
    }
  };
  const wpos = (o) => o.getWorldPosition(new V3());
  const wdir = (bone) => new V3(0, 1, 0).applyQuaternion(bone.getWorldQuaternion(_q2)).normalize();
  // chain joint basis: +Y along the bone, +X rotation droops the tip, +Z rotation swings it sideways
  let prevX = new V3(1, 0, 0);
  const chainBasis = (bone) => {
    const y = wdir(bone);
    let x = new V3().crossVectors(UP, y);
    if (x.length() < 0.35) x = prevX.clone().addScaledVector(y, -prevX.dot(y));
    x.normalize(); prevX = x;
    return basisQ(x, y, new V3().crossVectors(x, y));
  };
  const chain = (names) => {
    prevX = new V3(1, 0, 0);
    const out = names.map(n => bones.get(n)).filter(Boolean).map(b => proxy(b, chainBasis(b)));
    return out.length ? out : null;
  };
  const seq = (prefix, suffix = '') => { const out = []; for (let i = 1; bones.has(prefix + i + suffix); i++) out.push(prefix + i + suffix); return out; };

  // ---- aligned anchors + points: model-axis frames that ride a bone
  const anchors = new Map();
  const anchor = (bone) => {
    let a = anchors.get(bone);
    if (!a) {
      a = new THREE.Object3D(); a.name = 'anchor:' + nodeName(bone);
      a.userData.bone = bone;
      bone.add(a);
      a.quaternion.copy(bone.getWorldQuaternion(new Q4()).invert());
      a.scale.setScalar(1 / (bone.getWorldScale(new V3()).x || 1)); // children are authored in model units
      a.updateMatrixWorld(true);
      anchors.set(bone, a);
    }
    return a;
  };
  const rootBone = bones.get('root') || skeleton?.bones[0] || null;
  const pointAt = (bone, world, name) => {
    const o = new THREE.Object3D(); o.name = name || '';
    const a = anchor(bone || rootBone);
    o.position.copy(a.worldToLocal(world.clone()));
    o.userData.bone = bone || rootBone;
    a.add(o);
    return o;
  };
  const point = (name) => { const em = empties.get(name); return em ? pointAt(em.parent, wpos(em), name) : null; };
  const mountOn = (bone, world, n, t, s = 1) => {
    const a = anchor(bone || rootBone);
    return { p: a, pos: a.worldToLocal(world.clone()), n: n.clone().normalize(), t: t ? t.clone() : null, s };
  };

  // ---- colliders
  const colliders = [];
  for (const k of e.colliders) {
    const b = bones.get(k.bone); if (!b) continue;
    const me = new THREE.Mesh(k.geo, c.M.body);
    me.visible = false; me.userData.collider = true; me.name = 'collider:' + k.bone;
    b.add(me); colliders.push(me);
  }
  c.root.updateMatrixWorld(true);
  const boneOf = (o) => { for (let p = o; p; p = p.parent) if (p.isBone) return p; return rootBone; };
  const cast = (from, dir, list = colliders, far = 60) => {
    _ray.set(from, dir.clone().normalize()); _ray.near = 0; _ray.far = far;
    for (const h of _ray.intersectObjects(list, false)) {
      if (!h.face) continue;
      const n = h.face.normal.clone().transformDirection(h.object.matrixWorld);
      return { point: h.point.clone(), normal: n, bone: boneOf(h.object) };
    }
    return null;
  };
  const under = (bone) => colliders.filter(o => { for (let p = o; p; p = p.parent) if (p === bone) return true; return false; });

  const I = { c, cfg, frameId, bones, empties, proxy, chain, seq, point, pointAt, mountOn, wpos, wdir, cast, under, colliders, anchor, rootBone, S, T };
  buildRig(I);
  if (!cfg.wild) { try { buildMounts(I); } catch (err) { console.warn('[glb-models] mounts failed', frameId, err); } }
  else {
    const ch = point('chest'), bk = point('back');
    if (ch) c.mounts.chest = { p: ch, pos: new V3(), n: new V3(0, 0, 1), t: null, s: 1 };
    if (bk) c.mounts.back = { p: bk, pos: new V3(), n: new V3(0, 0.3, -1).normalize(), t: null, s: 1 };
    for (const n of ['shoulderL', 'shoulderR']) { const o = point(n); if (o) c.mounts.shoulders.push({ p: o, pos: new V3(), n: UP.clone(), t: null, s: 1 }); }
    const cr = point('crown'); if (cr) c.mounts.crest = { p: cr, pos: new V3(), n: UP.clone(), t: null, s: 1 };
    c.abilities = []; // humanoid / vehicle frames never wore the monster ability cosmetics
  }

  const glb = {
    frameId, cfg, applyPose, colliders, skinned, inst,
    eyes: e.eyes.map(q => ({ center: q.center.clone().multiplyScalar(S), r: q.r * S, bone: bones.get(q.bone) || rootBone })),
    tick: null, late: [],
    // called by buildKaijuModel once the rig runtime exists
    finish(rig, model) {
      if (rig.fallen) return;
      for (const [o, parent] of glb.late) { o.traverse(x => { x.userData.fx = true; }); (parent === 'model' ? model : parent || rig.motion).add(o); }
      const base = rig.update;
      let warned = false;
      rig.update = (dt, t) => {
        base(dt, t);
        if (glb.tick) { try { glb.tick(rig, dt, t); } catch (err) { if (!warned) { warned = true; console.warn('[glb-models] tick failed', frameId, err); } } }
        applyPose();
      };
    },
  };
  c.glb = glb;
  if (cfg.wild && !statue) { try { WILD[cfg.wild]?.(I, glb); } catch (err) { console.warn('[glb-models] wild runtime failed', frameId, err); } }
  else if (!statue) glb.tick = stockTick(I);
  return true;
}

// ---------------------------------------------------------------- rig from bone names
function buildRig(I) {
  const { c, cfg, bones, proxy, chain, seq, point, pointAt, wpos, wdir } = I;
  const rig = c.rig;
  rig.koMode = cfg.koMode || 'back';
  rig.hover = cfg.hover || 0;
  if (cfg.bank) rig.extra.bank = cfg.bank;
  if (cfg.tiny) rig.extra.tiny = true;
  const has = (n) => bones.has(n);
  const sideOf = (b) => (wpos(b).x >= 0 ? 1 : -1);
  const limb = (rec) => { if (rec.pivot || rec.joints) rig.limbs.push(rec); return rec; };

  // heads, jaws, mouths
  const headNames = ['head', 'headA', 'headB', 'headC'].filter(has);
  const mainMouth = point('mouth');
  const mouthEm = I.empties.get('mouth');
  headNames.forEach((hn, i) => {
    const hb = bones.get(hn), suffix = hn.slice(4);
    const hp = proxy(hb, null);
    hp.userData.ph = i * 1.7;
    rig.heads.push(hp);
    const jb = bones.get('jaw' + suffix);
    let jaw = null;
    if (jb) { jaw = proxy(jb, null); rig.jaws.push(jaw); }
    let mouth = null;
    if (i === 0) mouth = mainMouth;
    else if (mouthEm) {
      // secondary heads have no mouth empty: reuse the main head's bone-local mouth offset
      const mb = mouthEm.parent, tb = (nodeNameOf(mb).startsWith('jaw') ? jb : hb) || hb;
      mouth = pointAt(tb, tb.localToWorld(mouthEm.position.clone()), 'mouth' + suffix);
    }
    if (mouth) rig.mouths.push(mouth);
    if (i === 0) {
      const headEm = I.empties.get('head');
      const pivot = pointAt(hb, headEm ? wpos(headEm) : wpos(hb), 'headPivot');
      rig.headPivot = pivot; rig.jaw = jaw; rig.mouth = mouth;
    }
  });
  if (!rig.headPivot) {
    // vehicles etc.: no head bone, just the mount empties
    const hp = point('head') || pointAt(I.rootBone, new V3(0, 6, 1), 'headPivot');
    rig.headPivot = hp; rig.jaw = null;
    rig.mouth = mainMouth || hp;
    rig.mouths.push(rig.mouth);
  }
  // arachnid chelicerae act as jaws (they lift instead of dropping)
  for (const n of ['chelL', 'chelR']) if (has(n)) rig.jaws.push(proxy(bones.get(n), new Q4().setFromAxisAngle(UP, PI)));
  if (has('neck')) { const nk = chain(['neck']); if (nk) limb({ type: 'neck', joints: nk, ampX: 0.025, ampZ: 0.03, speed: 1.1 }); }

  // hands (attach points for punches / grabs)
  rig.extra.handL = point('handL'); rig.extra.handR = point('handR');

  // biped / quadruped limbs
  for (const sd of ['L', 'R']) {
    const s = has('thigh' + sd) ? sideOf(bones.get('thigh' + sd)) : (sd === 'L' ? -1 : 1);
    if (has('thigh' + sd)) limb({ type: 'leg', pivot: proxy(bones.get('thigh' + sd)), knee: proxy(bones.get('shin' + sd)), phase: (s > 0 ? 0 : PI) + (cfg.quad ? PI : 0) });
    if (has('upperarm' + sd)) {
      const pv = proxy(bones.get('upperarm' + sd)), el = proxy(bones.get('forearm' + sd));
      if (cfg.quad) limb({ type: 'leg', pivot: pv, knee: el, phase: s > 0 ? 0 : PI });
      else limb({ type: 'arm', pivot: pv, elbow: el, phase: s, reach: -1.2, elbowReach: 0.7, ...(cfg.arm || {}) });
    }
    // mantid raptorial arms + walking legs
    if (has('coxa' + sd)) limb({ type: 'arm', pivot: proxy(bones.get('coxa' + sd)), elbow: proxy(bones.get('tibia' + sd)), phase: s, reach: -0.75, elbowReach: -1.0, vicRaise: -1.2 });
  }
  // radial (insect / spider) legs: swing about the vertical, knee lifts the foot outward
  const radial = (pivotN, kneeN, footN, phase) => {
    const pb = bones.get(pivotN), kb = bones.get(kneeN); if (!pb) return;
    const d = wpos(bones.get(footN) || kb || pb).sub(wpos(pb)); d.y = 0;
    if (d.lengthSq() < 1e-4) d.set(sideOf(pb), 0, 0);
    d.normalize();
    const B = basisQ(new V3().crossVectors(UP, d), UP.clone(), d);
    limb({ type: 'leg', axis: 'y', pivot: proxy(pb, B), knee: kb ? proxy(kb, B) : null, phase });
  };
  for (let k = 1; k <= 4; k++) for (const sd of ['L', 'R']) if (has(`leg${k}${sd}4`)) radial(`leg${k}${sd}1`, `leg${k}${sd}2`, `leg${k}${sd}4`, (k % 2 ? PI : 0) + (sd === 'R' ? 0 : PI));
  ['m', 'h'].forEach((pre, k) => { for (const sd of ['L', 'R']) radial(`${pre}femur${sd}`, `${pre}tibia${sd}`, `${pre}tarsus${sd}`, k * 2.1 + (sd === 'R' ? 0 : PI / 2)); });
  // lepidra: dangling legs + four wings
  for (let k = 0; k <= 2; k++) for (const sd of ['L', 'R']) if (has(`leg${k}${sd}1`) && !has(`leg${k}${sd}4`)) limb({ type: 'leg', pivot: proxy(bones.get(`leg${k}${sd}1`)), phase: k + (sd === 'R' ? 1 : -1) });
  for (const [n, amp, ph] of [['wingF', 0.42, 0], ['wingH', 0.36, -0.4]]) for (const sd of ['L', 'R']) {
    const b = bones.get(n + sd); if (b) limb({ type: 'wing', pivot: proxy(b), side: sideOf(b), amp, speed: 5.5, phase: ph });
  }
  // stormray: three-bone wings flap as a travelling wave
  for (const sd of ['L', 'R']) seq('wing' + sd).forEach((n, i) => { const b = bones.get(n); limb({ type: 'wing', pivot: proxy(b), side: sideOf(b), amp: [0.2, 0.2, 0.16][i] ?? 0.15, speed: 2.1, phase: -0.7 * i }); });

  // chains
  const tail = chain(seq('tail'));
  if (tail) { rig.tail = tail; limb({ type: 'tail', joints: tail, ampX: 0.03, ampZ: 0.09, speed: I.frameId === 'stormray' ? 1.6 : 1.3 }); }
  for (const sfx of ['A', 'B', 'C']) { const nk = chain(seq('neck' + sfx)); if (nk) limb({ type: 'neck', joints: nk, ampX: 0.05, ampZ: 0.06, speed: 1.2, phase: 'ABC'.indexOf(sfx) * 2.1 }); }
  const nk = chain(seq('neck')); if (nk) limb({ type: 'neck', joints: nk, ampX: 0.03, ampZ: 0.04, speed: 1.1 });
  if (I.frameId === 'serpent') { const sp = chain(seq('spine').slice(3)); if (sp) limb({ type: 'neck', joints: sp, ampX: 0.02, ampZ: 0.035, speed: 1.1 }); }
  const abd = chain(seq('abd')); if (abd) limb({ type: 'tail', joints: abd, ampX: 0.02, ampZ: 0.03, speed: 1.2 });
  for (let k = 1; k <= 8; k++) { const t = chain(seq(`tent${k}_`)); if (t) limb({ type: 'tentacle', joints: t, phase: k * 0.9, ampX: 0.07, ampZ: 0.07, speed: 1.3 }); }
  for (const sd of ['L', 'R']) {
    const t = chain(seq('tent' + sd)); if (t) limb({ type: 'tentacle', joints: t, phase: sd === 'L' ? 0 : 1.4, ampX: 0.07, ampZ: 0.06, speed: 1.3, reach: true, whip: true });
    const an = chain(seq('ant' + sd)); if (an) limb({ type: 'tail', joints: an, ampX: 0.06, ampZ: 0.05, speed: 2.2, phase: sd === 'L' ? 0 : 1 });
    const pl = chain(seq('palp' + sd)); if (pl) limb({ type: 'tail', joints: pl, ampX: 0.08, ampZ: 0.03, speed: 2.6, phase: sd === 'L' ? 0 : 2 });
    const hn = chain(seq('horn' + sd)); if (hn) limb({ type: 'tail', joints: hn, ampX: 0.03, ampZ: 0.03, speed: 1.7, phase: sd === 'L' ? 0 : 1 });
  }
  // torso bones the stock tick leans on (breathing lives on the wrapper; these add spine follow-through)
  I.spine = ['spine', 'chest'].filter(has).map(n => proxy(bones.get(n)));
  void wdir;
}
function nodeNameOf(o) { return nodeName(o); }

// Extra bone motion on top of the stock rig.update (which only knows heads / jaws / limbs).
function stockTick(I) {
  const spine = I.spine || [], whips = I.c.rig.limbs.filter(L => L.whip);
  if (!spine.length && !whips.length) return null;
  return (rig, dt, t) => {
    const a = rig.anim, tt = t + (rig.seed || 0), up = 1 - a.ko;
    const roar = Math.max(a.roar, a.victory * (0.5 + 0.5 * Math.sin(tt * 2.2)));
    const n = spine.length || 1;
    for (const s of spine) {
      // only bones that are not already limb pivots: counter-sway on the walk, rear back on a roar, hunch into a lunge
      s.rotation.set((a.lunge * 0.16 - roar * 0.14 + a.charge * 0.06) / n, Math.sin(tt * 6.5) * 0.07 * a.walk * up / n, Math.sin(tt * 6.5 + 1.2) * 0.03 * a.walk * up / n);
    }
    for (const L of whips) {
      const J = L.joints, k = Math.max(0, a.lunge);
      for (let i = 0; i < J.length; i++) J[i].rotation.x -= k * 0.16 * (i === 0 ? 2.2 : 1);
    }
  };
}

// ---------------------------------------------------------------- mounts for ability cosmetics / injuries
function buildMounts(I) {
  const { c, bones, empties, mountOn, wpos, wdir, cast, under, colliders, pointAt } = I;
  const mt = c.mounts, rig = c.rig;
  const em = (n) => empties.get(n) || null;
  const at = (n) => (em(n) ? wpos(em(n)) : null);
  const chestP = at('chest'), backP = at('back');
  // snap a guess onto the real surface (ray from outside, along -n)
  const snap = (bone, guess, n, t, s = 1, list = colliders, reach = 5) => {
    const h = cast(guess.clone().addScaledVector(n, reach), n.clone().negate(), list, reach * 2);
    return h ? mountOn(h.bone, h.point, n, t, s) : mountOn(bone, guess, n, t, s);
  };
  if (chestP) mt.chest = snap(em('chest').parent, chestP, new V3(0, 0.1, 1).normalize(), null, 1, colliders, 2.5);
  let backN = new V3(0, 0.5, -1).normalize();
  if (backP) {
    if (chestP) { backN = backP.clone().sub(chestP.clone().lerp(backP, 0.5)); backN.x = 0; if (backN.lengthSq() < 0.01) backN.set(0, 1, 0); backN.normalize(); }
    mt.back = snap(em('back').parent, backP, backN, null, 1, colliders, 2.5);
    // dorsal row: step along the spine line through the back mount
    const along = new V3().crossVectors(new V3(1, 0, 0), backN).normalize(); // sagittal tangent
    const span = Math.max(1.2, (c.glbSpan || 2.2));
    for (const k of [1.1, 0.35, -0.4, -1.15]) {
      const g = backP.clone().addScaledVector(along, k * span);
      const h = cast(g.clone().addScaledVector(backN, 6), backN.clone().negate(), colliders, 9);
      if (h && h.point.distanceTo(g) < span * 1.6) mt.dorsal.push(mountOn(h.bone, h.point, backN, along, 1));
    }
    if (!mt.dorsal.length) mt.dorsal.push(mt.back);
  }
  for (const n of ['shoulderL', 'shoulderR']) {
    const p = at(n); if (!p) continue;
    mt.shoulders.push(mountOn(em(n).parent, p, new V3(Math.sign(p.x) * 0.25, 1, 0).normalize(), null, 1));
  }
  const tt = em('tailTip');
  if (tt) {
    const tb = tt.parent, isTail = /^(tail|abd|abdomen|tent|spine)/.test(nodeName(tb));
    const n = isTail ? wdir(tb) : new V3(0, -0.2, -1).normalize();
    mt.tailTip = mountOn(tb, wpos(tt), n, UP, 1);
  }
  const crownP = at('crown'), headP = at('head'), mouthP = at('mouth');
  const headBone = rig.headPivot?.userData?.bone;
  const headCols = headBone ? under(headBone) : [];
  if (crownP && headBone) {
    mt.crest = mountOn(headBone, crownP, new V3(0, 1, -0.15).normalize(), new V3(0, 0, -1), 1);
    // horn stump (head injury): on the skull to the right of the crown
    const hw = headCols.length ? new THREE.Box3().setFromObject(headCols[0]).getSize(new V3()).x * 0.22 : 0.4;
    let h = null;
    for (const k of [0.3, 0.6, 0, 0.85]) { h = cast(crownP.clone().lerp(headP || crownP, k).add(new V3(Math.max(0.25, Math.min(0.8, hw)), 3, 0)), new V3(0, -1, 0), headCols, 6); if (h) break; }
    if (h) mt.horns.push({ p: rig.headPivot, pos: rig.headPivot.worldToLocal(h.point.clone()), n: new V3(0.25, 1, -0.2).normalize(), t: new V3(0, 0, -1), s: 1 });
  }
  if (headP && mouthP && headBone) {
    const fwd = mouthP.clone().sub(headP);
    // ram horn on the snout, tusks on the sides of the jaw
    const rg = headP.clone().addScaledVector(fwd, 0.55);
    const rh = cast(rg.clone().add(new V3(0, 3, 0)), new V3(0, -1, 0), headCols, 6);
    if (rh) mt.ram = mountOn(headBone, rh.point, new V3(0, 0.7, 0.75).normalize(), null, 1);
    const jb = bones.get('jaw') || bones.get('jawA') || headBone;
    for (const s of [-1, 1]) {
      const g = headP.clone().addScaledVector(fwd, 0.7); g.y = Math.min(g.y, mouthP.y);
      let th = null;
      for (const dy of [-0.12, -0.25, 0, -0.4, 0.15]) { th = cast(g.clone().add(new V3(s * 3, dy * fwd.length(), 0)), new V3(-s, 0, 0), headCols, 6); if (th) break; }
      if (th) mt.tusks.push(mountOn(jb, th.point.clone().add(new V3(-s * 0.08, 0, 0)), new V3(s * 0.3, 1, 0.35).normalize(), null, 1));
    }
  }
  void pointAt;
}

// ---------------------------------------------------------------- wild (human-scale / vehicle) runtimes
function glowTex() {
  const cv = document.createElement('canvas'); cv.width = cv.height = 64;
  const g = cv.getContext('2d'), gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.35, 'rgba(255,255,255,0.45)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
function sprite(tex, col, size, opacity = 1) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, color: col, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  s.scale.setScalar(size);
  return s;
}
// ground spotlight + ring + beacon + motion trail so ~1-unit fighters stay findable next to ~10-unit kaiju
function tinyFx(glb, accC, o = {}) {
  const tex = glowTex(), late = glb.late;
  const disc = new THREE.Mesh(new THREE.PlaneGeometry(1, 1).rotateX(-PI / 2), new THREE.MeshBasicMaterial({ map: tex, color: accC, transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  const dr = o.disc ?? 3.4;
  disc.scale.set(dr, 1, dr); disc.position.y = 0.05; disc.renderOrder = 3;
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 1, 56).rotateX(-PI / 2), new THREE.MeshBasicMaterial({ color: accC, transparent: true, opacity: 0.75, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide, fog: false }));
  ring.position.y = 0.07; ring.renderOrder = 3;
  late.push([disc, 'model'], [ring, 'model']);
  const beacon = sprite(tex, accC, o.beacon ?? 3.4, 0.35);
  beacon.position.y = o.centerY ?? 0.5;
  late.push([beacon, null]);
  const N = o.trail === false ? 0 : 16, hist = [], trail = [];
  for (let i = 0; i < N; i++) { const s = sprite(tex, accC.clone().lerp(new THREE.Color('#ffffff'), 0.3), 0.55, 0); trail.push(s); late.push([s, 'model']); }
  const p = new V3();
  return (rig, dt, t, boost = 0) => {
    const a = rig.anim, up = 1 - a.ko * 0.8, m = rig.motion, near = rig.bankSign ? 1 : 0.45;
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
const hash1 = (n) => { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return (s - Math.floor(s)) * 2 - 1; };
const lerp = (a, b, t) => a + (b - a) * t;
function dart(out, t, seed, amp = 1, period = 1.15) {
  const f = t / period + seed, n = Math.floor(f), k = smooth((f - n) * 3.5);
  const ax = lerp(hash1(n - 1), hash1(n), k), ay = lerp(hash1(n + 39), hash1(n + 40), k), az = lerp(hash1(n + 89), hash1(n + 90), k);
  return out.set(ax * 0.8 * amp + Math.sin(t * 5.1) * 0.05, (ay * 0.5 + 0.2) * amp + Math.sin(t * 3.3) * 0.06, az * 0.7 * amp);
}
const accentOf = (c) => { const k = new THREE.Color(); try { k.set(c.colors.accent); } catch { k.set('#ffd23f'); } return k; };

const WILD = {
  // superhero / magical girl: zippy hover, fly up to strike the kaiju's chest
  hero(I, glb) {
    const { c } = I, acc = accentOf(c);
    const tfx = tinyFx(glb, acc, { centerY: 0.5, beacon: 3.2 });
    const spineTick = stockTick(I);
    const dv = new V3();
    const api = { reset() {} };
    c.rig.extra.wild = api;
    glb.tick = (rig, dt, t) => {
      const a = rig.anim, up = 1 - a.ko;
      const calm = up * (1 - Math.min(1, Math.abs(a.lunge) * 3)) * (1 - a.victory * 0.6) * (1 - a.charge * 0.7);
      dart(dv, t, 0.3, rig.bankSign ? 0.9 : 0.25);
      const m = rig.motion;
      m.position.x += (dv.x - (rig.bankSign || 0) * 0.5) * calm; m.position.y += dv.y * calm; m.position.z += dv.z * calm;
      m.position.y += Math.max(0, a.lunge) * 3.4;
      spineTick?.(rig, dt, t);
      tfx(rig, dt, t, 0);
    };
  },
  // five heroes: one skinned leader + four single-bone members that bob, dash in turn and fall one by one
  team(I, glb) {
    const { c, bones, proxy } = I, acc = accentOf(c);
    const tfx = tinyFx(glb, acc, { centerY: 0.45, beacon: 5.5, disc: 6.0, ring: 2.7, trail: false });
    const members = [1, 2, 3, 4, 5].map(i => bones.get('hero' + i)).filter(Boolean).map((b, i) => ({ p: proxy(b), ph: i * 1.9, z: I.wpos(b).z }));
    const order = [0, 2, 1, 3, 4];
    const api = {
      active: 0, seq: 0,
      reset() { this.active = 0; this.seq = 0; },
      onStrike(hitNo, nHits, kind) { this.active = (kind === 'melee' || kind === 'multi' || kind === 'grab') ? order[(this.seq++) % Math.max(1, members.length)] : 0; },
    };
    c.rig.extra.wild = api;
    const frontZ = Math.max(0, ...members.map(m => m.z));
    glb.tick = (rig, dt, t) => {
      const a = rig.anim, m0 = rig.motion, sc = rig.scale || 1;
      // the group lunge becomes one member dashing in; the group KO tilt becomes members falling one by one
      m0.position.z -= a.lunge * a.reach;
      m0.rotation.x = 0;
      members.forEach((m, i) => {
        const act = i === api.active ? a.lunge : 0;
        const fallK = clamp01(a.ko * 1.5 - i * 0.12);
        m.p.position.set(0, Math.max(0, act) * 3.0 / sc + Math.abs(Math.sin(t * 3 + m.ph)) * 0.05 * a.victory + Math.abs(Math.sin(t * 1.7 + m.ph)) * 0.012 * (1 - a.ko),
          act * (a.reach / sc + frontZ - m.z - 0.1));
        m.p.rotation.set(-fallK * 1.45 + act * 0.25 - a.recoil * 0.1, 0, Math.sin(t * 0.5 + m.ph) * 0.03);
      });
      tfx(rig, dt, t, 0);
    };
  },
  // vehicles: rotors spin, turrets track lazily, wrecks tilt and the helicopters come down on KO
  army(I, glb) {
    const { c, bones, proxy, wpos } = I;
    const api = { noLunge: true, noLift: true, reset() {} };
    c.rig.extra.wild = api;
    const rotors = ['rotor', 'rotor2'].map(n => bones.get(n)).filter(Boolean).map(b => proxy(b));
    const turrets = [...bones.keys()].filter(n => /_tur$/.test(n)).map((n, i) => ({ p: proxy(bones.get(n)), ph: i * 1.3 }));
    const radar = bones.get('radar') ? proxy(bones.get('radar')) : null;
    const hulls = [...bones.keys()].filter(n => /^(tank|art|truck|wall)\d*$/.test(n)).map((n, i) => ({ p: proxy(bones.get(n)), ph: i * 2.3 }));
    const fliers = ['heli', 'heli2', 'jet', 'drone'].map(n => bones.get(n)).filter(Boolean).map((b, i) => ({ p: proxy(b), ph: i * 1.7, h: wpos(b).y, jet: nodeName(b) === 'jet' }));
    glb.tick = (rig, dt, t) => {
      const a = rig.anim, ko = a.ko, m0 = rig.motion;
      m0.rotation.x = 0; m0.position.z = -a.recoil * 0.4; // an army holds its line: no group lunge / topple
      for (const r of rotors) r.rotation.y = t * 22 * (1 - ko * 0.9);
      if (radar) radar.rotation.y = t * 1.3 * (1 - ko);
      for (const u of turrets) u.p.rotation.set(-a.lunge * 0.05 + ko * 0.12, Math.sin(t * 0.35 + u.ph) * 0.3 * (1 - ko) + ko * (u.ph - 2) * 0.3, 0);
      for (const u of hulls) {
        const jolt = a.recoil * 0.5 + Math.max(0, a.lunge) * 0.15;
        u.p.rotation.set(-jolt * 0.1 * Math.sin(t * 20 + u.ph) + ko * 0.16 * Math.sin(u.ph * 3), 0, jolt * 0.08 * Math.sin(t * 17 + u.ph) + ko * 0.2 * Math.cos(u.ph * 2));
        u.p.position.set(0, jolt * 0.2 * Math.abs(Math.sin(t * 14 + u.ph)) - ko * 0.12, 0);
      }
      for (const f of fliers) {
        const drop = smooth(ko * 1.3 - f.ph * 0.08);
        f.p.position.set(Math.sin(t * 0.6 + f.ph) * 0.5 * (1 - drop), Math.sin(t * 1.4 + f.ph) * 0.25 * (1 - drop) - drop * Math.max(0, f.h - 0.5), Math.cos(t * 0.45 + f.ph) * 0.4 * (1 - drop));
        f.p.rotation.set(0.08 * Math.sin(t * 0.9 + f.ph) + drop * 0.5, 0, 0.1 * Math.sin(t * 0.6 + f.ph) + drop * (f.ph % 2 ? 0.9 : -0.9));
      }
    };
  },
};
