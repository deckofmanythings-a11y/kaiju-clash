// Rare Wild runtimes for the GLB models (superhero, magicalgirl, heroteam, army).
//
// makeWildRuntimes(kit) -> { hero, girl, team, army }: each fn(I, glb) dresses a GLB rig built by glb-models.js
// with the same special behaviour wild-models.js gives the procedural plans, and exposes the same optional
// playback hooks on rig.extra.wild (reset, setTarget, onHit, onStrike, origin, strafe / strafeK / jetWorld,
// kneel / flare, beginSwap / swapK, reinforce / reinK, setTransformed, beamSources, memberColors, twinBeam ...).
//
// `kit` is the set of effect helpers wild-models.js already owns (textures, sprites, cloth, tiny-fighter fx,
// jets). wild-models.js builds the runtimes and registers them with glb-models.js, so this module imports
// neither of them (no import cycles) and the GLB wild frames only switch on once wild-models.js has loaded.
//
// Conventions: I = the build helpers from buildGlbPlan (bones, proxy, anchor, pointAt, cast ...). Proxies take
// model-space Euler rotations and model-space position offsets (see glb-models.js). "late" objects are added
// after the model is measured so fx never inflate its bounds.

import * as THREE from 'three';

const V3 = THREE.Vector3;
const v = (x = 0, y = 0, z = 0) => new V3(x, y, z);
const PI = Math.PI;
const lerp = (a, b, t) => a + (b - a) * t;
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const smooth = (k) => { k = clamp01(k); return k * k * (3 - 2 * k); };
const nodeName = (o) => o?.userData?.name || '';
const colorOf = (h, fb) => { const k = new THREE.Color(); try { k.set(h || fb); } catch { k.set(fb); } return k; };

// Visit every vertex of the instance's skinned meshes with its dominant bone (bind pose, model space).
function scanVerts(I, fn) {
  const P = new V3();
  for (const me of I.skinned) {
    const g = me.geometry, pos = g.attributes.position, si = g.attributes.skinIndex, sw = g.attributes.skinWeight;
    if (!pos || !si || !sw) continue;
    const bones = me.skeleton.bones;
    for (let i = 0; i < pos.count; i++) {
      let b = si.getX(i), w = sw.getX(i);
      if (sw.getY(i) > w) { w = sw.getY(i); b = si.getY(i); }
      if (sw.getZ(i) > w) { w = sw.getZ(i); b = si.getZ(i); }
      if (sw.getW(i) > w) b = si.getW(i);
      fn(nodeName(bones[b]), P.fromBufferAttribute(pos, i).multiplyScalar(I.S), me, i);
    }
  }
}
// Rigid copy of everything skinned to `names` (bind pose), as plain meshes sharing the instance's materials.
// The group's origin sits at `origin` (model space).
function rigidCopy(I, names, origin) {
  const set = new Set(names), grp = new THREE.Group();
  for (const me of I.skinned) {
    const g = me.geometry, pos = g.attributes.position, si = g.attributes.skinIndex, sw = g.attributes.skinWeight, idx = g.index;
    if (!pos || !si || !sw) continue;
    const bones = me.skeleton.bones;
    const dom = (i) => {
      let b = si.getX(i), w = sw.getX(i);
      if (sw.getY(i) > w) { w = sw.getY(i); b = si.getY(i); }
      if (sw.getZ(i) > w) { w = sw.getZ(i); b = si.getZ(i); }
      if (sw.getW(i) > w) b = si.getW(i);
      return set.has(nodeName(bones[b]));
    };
    const attrs = ['position', 'normal', 'color', 'uv'].filter(n => g.attributes[n]);
    const out = Object.fromEntries(attrs.map(n => [n, []]));
    const n = (idx ? idx.count : pos.count) / 3;
    for (let t = 0; t < n; t++) {
      const tri = [0, 1, 2].map(k => (idx ? idx.getX(t * 3 + k) : t * 3 + k));
      if (!dom(tri[0])) continue;
      for (const i of tri) for (const nm of attrs) {
        const at = g.attributes[nm];
        for (let k = 0; k < at.itemSize; k++) out[nm].push(at.getComponent(i, k));
      }
    }
    if (!out.position.length) continue;
    const ng = new THREE.BufferGeometry();
    for (const nm of attrs) {
      const at = g.attributes[nm];
      ng.setAttribute(nm, new THREE.BufferAttribute(new Float32Array(out[nm]), at.itemSize));
    }
    ng.scale(I.S, I.S, I.S); ng.translate(-origin.x, -origin.y, -origin.z);
    const m = new THREE.Mesh(ng, me.material);
    m.castShadow = true; m.receiveShadow = true;
    grp.add(m);
  }
  return grp;
}

export function makeWildRuntimes(kit) {
  const { glowTex, haloTex, starTex, smokeTex, sprite, addMat, makeCloth, poseCloth, tinyFx, dart, jetModel } = kit;
  const std = (o) => new THREE.MeshStandardMaterial(o);
  const _tmp = new THREE.Color();

  // shared: materials by slot, soft accent rim so tiny suits read against dark arenas
  const slotMats = (I, names) => { const out = []; for (const me of I.skinned) if (names.includes(me.material.userData.slot) && !out.includes(me.material)) out.push(me.material); return out; };
  const rim = (mats, acc, k) => { for (const m of mats) { m.emissive.copy(acc).multiplyScalar(k); m.emissiveIntensity = 1; } };
  const legsOf = (I) => {
    const L = I.c.rig.limbs.filter(l => l.type === 'leg' && l.knee).sort((a, b) => I.wpos(a.pivot.userData.bone).x - I.wpos(b.pivot.userData.bone).x);
    return L.length >= 2 ? [L[0], L[L.length - 1]] : null; // [x < 0, x > 0]
  };
  const armsOf = (I) => I.c.rig.limbs.filter(l => l.type === 'arm').sort((a, b) => I.wpos(a.pivot.userData.bone).x - I.wpos(b.pivot.userData.bone).x);
  // a group in the procedural "figure" units (the procedural heroes are ~9 units tall before scaling)
  const figure = (I, baseH) => {
    const crown = I.empties.get('crown'), k = (crown ? I.wpos(crown).y : 0.9) / baseH;
    const fig = new THREE.Group(); fig.scale.setScalar(k);
    return { fig, k };
  };
  // farthest vertex along each bone's +Y (barrel tips, wand tip)
  const boneTips = (I, want) => {
    const tips = new Map(), q = new THREE.Quaternion(), d = new V3(), o = new V3();
    const dirs = new Map();
    for (const n of want) { const b = I.bones.get(n); if (b) dirs.set(n, { o: I.wpos(b), d: I.wdir(b), best: -1e9, p: null }); }
    scanVerts(I, (bn, P) => {
      const k = dirs.get(bn); if (!k) return;
      const s = d.copy(P).sub(k.o).dot(k.d);
      if (s > k.best) { k.best = s; k.p = k.o.clone().addScaledVector(k.d, s); }
    });
    for (const [n, k] of dirs) if (k.p) tips.set(n, k.p);
    void q; void o;
    return tips;
  };

  // ================================================================ SUPERHERO
  function hero(I, glb) {
    const { c, T } = I, rig = c.rig;
    const acc = colorOf(c.colors.accent, '#ffd23f'), belly = colorOf(c.colors.belly, '#c8102e');
    const late = glb.late;
    const { fig, k } = figure(I, 9.1);
    late.push([fig, null]);
    // heat vision: the beam leaves from both eyes
    const eyes = glb.eyes.filter(e => Math.abs(e.center.x) > 0.005).sort((a, b) => a.center.x - b.center.x);
    if (eyes.length >= 2) {
      const pts = [eyes[0], eyes[eyes.length - 1]].map((e, i) => I.pointAt(e.bone, glb.inst.localToWorld(e.center.clone().divideScalar(I.S)).add(v(0, 0, 0.012)), 'eye' + i));
      rig.mouths.length = 0; rig.mouths.push(...pts); rig.mouth = pts[0];
    }
    // cape: the GLB's static cape is swapped for the same cloth strip the procedural hero wears
    for (const me of I.skinned) if (me.material.userData.slot === 'cape') me.visible = false;
    const capeMat = std({ color: belly, roughness: 0.62, side: THREE.DoubleSide });
    const chest = I.bones.get('chest') || I.rootBone;
    const shL = I.empties.get('shoulderL'), shR = I.empties.get('shoulderR');
    const mid = shL && shR ? I.wpos(shL).lerp(I.wpos(shR), 0.5) : I.wpos(chest).add(v(0, 0.1, 0));
    const backHit = I.cast(mid.clone().add(v(0, -0.02, -1)), v(0, 0, 1), I.colliders, 2);
    const capeTop = v(mid.x, mid.y + 0.012, backHit ? backHit.point.z - 0.004 : mid.z - 0.07);
    const capeG = new THREE.Group(); capeG.scale.setScalar(k);
    const capeAnchor = I.anchor(chest);
    capeG.position.copy(capeAnchor.worldToLocal(capeTop.clone()));
    const cape = makeCloth(capeMat, 10, 16);
    capeG.add(cape.mesh);
    late.push([capeG, capeAnchor]);
    const capeL = 6.0 + 0.35 * T.spd;
    // aura shell, hover glow, motes
    const aura = sprite(haloTex(), acc, 1, { opacity: 0.5 });
    aura.position.set(0, 4.9, 0); aura.renderOrder = 4; fig.add(aura);
    const tex = glowTex();
    const footGlow = sprite(tex, acc, 3.2, { opacity: 0.55 });
    footGlow.position.set(0, -0.2, 0.1); fig.add(footGlow);
    const motes = [];
    for (let i = 0; i < 10; i++) { const s = sprite(tex, acc, 0.35, { opacity: 0.8 }); motes.push(s); fig.add(s); }
    const gold = new THREE.Color('#fff0b0');
    rim([...slotMats(I, ['body', 'belly', 'sole', 'gaunt']), capeMat], acc, 0.1);
    const tfx = tinyFx(c, late, acc, { centerY: 0.5, beacon: 3.2 });
    const spineTick = I.stockTick(I), spine = I.spine || [];
    const legs = legsOf(I), arms = armsOf(I), head = rig.heads[0];
    const dv = v(), tmpC = new THREE.Color();
    // last-stand kneel, added on top of whatever the stock animation posed: one knee up, one knee down, the body
    // bows and the raised (flying) fist comes down. The GLB rests in a flying pose, so how far the body has to
    // sink for the knee to reach the ground is measured once on the posed colliders.
    const handY = (L) => { const e = L.elbow?.userData.bone; const h = e?.children.find(o => o.isBone); return h ? I.wpos(h).y : 0; };
    const raised = arms.length ? arms.slice().sort((p, q) => handY(q) - handY(p))[0] : null;
    const shoulderY = raised ? I.wpos(raised.pivot.userData.bone).y : 0;
    const armUp = raised && handY(raised) > shoulderY + 0.05 ? raised : null;
    const kneelPose = (kn) => {
      if (legs) {
        const [Ll, Lr] = legs;
        Lr.pivot.rotation.x -= 1.45 * kn; Lr.knee.rotation.x += 1.5 * kn;
        Ll.pivot.rotation.x += 0.25 * kn; Ll.knee.rotation.x += 1.85 * kn;
      }
      for (const s of spine) s.rotation.x += 0.32 * kn / (spine.length || 1);
      if (head) head.rotation.x += 0.35 * kn;
      for (const A of arms) {
        if (A === armUp) { A.pivot.rotation.x += 2.1 * kn; if (A.elbow) A.elbow.rotation.x -= 0.5 * kn; }
        else A.pivot.rotation.x += 0.2 * kn;
      }
    };
    let kneelDrop = 2.25 * k;
    try {
      kneelPose(1); glb.applyPose(); c.root.updateMatrixWorld(true);
      const box = new THREE.Box3(); for (const o of I.colliders) box.expandByObject(o);
      if (Number.isFinite(box.min.y)) kneelDrop = box.min.y - 0.01;
      kneelPose(-1); glb.applyPose(); c.root.updateMatrixWorld(true);
    } catch { /* keep the estimate */ }
    const api = { kneel: 0, flare: 0, twinBeam: eyes.length >= 2, reset() { this.kneel = 0; this.flare = 0; } };
    rig.extra.wild = api;
    glb.tick = (r, dt, t) => {
      const a = r.anim, up = 1 - a.ko, m = r.motion;
      const kn = api.kneel, fl = api.flare;
      const calm = up * (1 - kn) * (1 - Math.min(1, Math.abs(a.lunge) * 3)) * (1 - a.victory * 0.6) * (1 - a.charge * 0.7);
      dart(dv, t, 0.3, r.bankSign ? 0.9 : 0.25);
      m.position.x += (dv.x - (r.bankSign || 0) * 0.5) * calm; m.position.y += dv.y * calm; m.position.z += dv.z * calm;
      m.position.y += Math.max(0, a.lunge) * 3.4 * (1 - kn); // fly up to strike the kaiju's chest
      spineTick?.(r, dt, t);
      const flow = clamp01(a.lunge * 0.9 + a.walk * 0.6 + a.victory * 0.35 + Math.abs(a.dodge) * 0.2 + dv.length() * 0.3 * calm) * (1 - kn * 0.8);
      poseCloth(cape, { top: v(0, 0, 0), wrap: 0.38, w0: 2.6, w1: 3.8, L: capeL, ang: 0.32 + flow * 1.05 + 0.05 * Math.sin(t * 1.2) - a.ko * 0.2, flow, t, clear: 0.3 });
      // last-stand kneel: one knee up, one knee down, body sinks and bows
      if (kn > 0.001) {
        kneelPose(kn);
        // cancel the stock hover (and its bob) so the knee actually rests on the ground
        m.position.y -= ((r.hover + Math.sin((t + (r.seed || 0)) * 1.9) * 0.35) * (1 - a.ko) + kneelDrop) * kn;
      }
      aura.material.opacity = Math.min(1, (0.42 + 0.1 * T.enr) * (0.85 + 0.15 * Math.sin(t * 2.4)) * (1 + a.charge * 0.8 + a.roar * 0.4 + fl * 1.4 + a.victory * 0.3)) * up * (1 - kn * 0.5);
      aura.material.color.copy(acc).lerp(gold, fl);
      aura.scale.set(6.4 * (1 + fl * 0.5), 12.5 * (1 + fl * 0.25), 1);
      tfx(r, dt, t, fl);
      footGlow.material.opacity = (0.5 + 0.15 * Math.sin(t * 3)) * up * (1 - kn) + fl * 0.6;
      footGlow.material.color.copy(tmpC.copy(acc).lerp(gold, fl));
      for (let i = 0; i < motes.length; i++) {
        const s = motes[i], ph = (t * 0.35 + i / motes.length) % 1, an = i * 2.4 + t * 0.8;
        const rr = 2.0 + 0.8 * Math.sin(i * 1.7);
        s.position.set(Math.cos(an) * rr, ph * 10, Math.sin(an) * rr * 0.8);
        s.material.opacity = Math.sin(ph * PI) * 0.8 * up * (0.6 + fl);
        s.material.color.copy(tmpC.copy(acc).lerp(gold, fl));
      }
    };
  }

  // ================================================================ MAGICAL GIRL
  function girl(I, glb) {
    const { c, T } = I, rig = c.rig;
    const accC = colorOf(c.colors.accent, '#8ef6ff'), pinkC = colorOf(c.colors.body, '#ff8fc7'), whiteC = colorOf(c.colors.belly, '#ffffff');
    const late = glb.late;
    const { fig, k } = figure(I, 9.6);
    late.push([fig, null]);
    // wand: beams and bolts leave from the star at its tip
    const wandBone = I.bones.get('wand');
    let tip = boneTips(I, ['wand']).get('wand') || null;
    if (wandBone) {
      // the shaft may be weighted to the hand: fall back to the far end of the wand-material geometry
      const o = I.wpos(wandBone), d = I.wdir(wandBone), q = v();
      let best = tip ? q.copy(tip).sub(o).dot(d) : 0.02;
      scanVerts(I, (bn, P, me) => {
        if (me.material.userData.slot !== 'wand') return;
        const sd = q.copy(P).sub(o).dot(d);
        if (sd > best && q.addScaledVector(d, -sd).length() < 0.05) { best = sd; tip = o.clone().addScaledVector(d, sd); }
      });
    }
    let wandTip = null;
    if (tip && wandBone) {
      wandTip = I.pointAt(wandBone, tip.clone().addScaledVector(I.wdir(wandBone), -0.03), 'wandTip');
      const old = rig.mouth;
      rig.mouths.length = 0; rig.mouths.push(wandTip); if (old) rig.mouths.push(old);
      rig.mouth = wandTip;
    }
    // aura + sparkles
    const aura = sprite(haloTex(), accC, 1, { opacity: 0.4 });
    aura.position.set(0, 5.2, 0); aura.renderOrder = 4; fig.add(aura);
    const sTex = starTex(), gTex = glowTex();
    const sparkles = [];
    for (let i = 0; i < 18; i++) { const s = sprite(sTex, i % 3 ? accC : new THREE.Color('#ffd6f0'), 0.6, { opacity: 0.9 }); sparkles.push(s); fig.add(s); }
    const tipGlow = sprite(gTex, accC, 1.6 * k, { opacity: 0.6 });
    if (wandTip) late.push([tipGlow, wandTip]);
    // transformation flourishes: light wings, halo, glowing hem
    const chest = I.bones.get('chest') || I.rootBone, pelvis = I.bones.get('pelvis') || I.rootBone, headB = I.bones.get('head') || I.rootBone;
    const wings = new THREE.Group();
    for (const s of [-1, 1]) for (let j = 0; j < 2; j++) {
      const w = new THREE.Shape();
      w.moveTo(0, 0); w.bezierCurveTo(0.6, 1.6 - j, 2.6, 2.6 - j * 1.8, 3.6 - j * 0.6, 1.8 - j * 2.2); w.bezierCurveTo(2.4, 1.0 - j, 1.2, 0.2 - j * 0.3, 0, 0);
      const wg = new THREE.ShapeGeometry(w, 16); wg.scale(s, 1, 1);
      const wm = new THREE.Mesh(wg, addMat(j ? pinkC.clone().lerp(whiteC, 0.3) : accC.clone().lerp(whiteC, 0.3), 0.45));
      wm.material.side = THREE.DoubleSide;
      wm.position.set(s * 0.1, 0, -0.05 * j); wm.rotation.y = -s * 0.8;
      wings.add(wm);
    }
    const backEm = I.empties.get('back');
    const wingG = new THREE.Group(); wingG.scale.setScalar(k); wingG.add(wings);
    const chestA = I.anchor(chest);
    wingG.position.copy(chestA.worldToLocal((backEm ? I.wpos(backEm) : I.wpos(chest)).add(v(0, 0.06, -0.02))));
    wings.scale.setScalar(0.001);
    late.push([wingG, chestA]);
    const halo = new THREE.Mesh(new THREE.TorusGeometry(1.2, 0.07, 8, 40), addMat(accC.clone().lerp(new THREE.Color('#fff4c0'), 0.5), 0.9));
    halo.rotation.x = PI / 2 - 0.25;
    const haloG = new THREE.Group(); haloG.scale.setScalar(k); haloG.add(halo);
    const headA = I.anchor(headB), crown = I.empties.get('crown');
    haloG.position.copy(headA.worldToLocal((crown ? I.wpos(crown) : I.wpos(headB)).add(v(0, 0.09, -0.01))));
    halo.scale.setScalar(0.001);
    late.push([haloG, headA]);
    const hem = new THREE.Mesh(new THREE.TorusGeometry(1.0, 0.06, 6, 56).rotateX(PI / 2), addMat(accC, 0.8));
    hem.scale.set(2.0, 1, 2.0); hem.visible = false;
    const hemG = new THREE.Group(); hemG.scale.setScalar(k); hemG.add(hem);
    const pelA = I.anchor(pelvis);
    hemG.position.copy(pelA.worldToLocal(I.wpos(pelvis).add(v(0, -0.1, 0))));
    late.push([hemG, pelA]);
    // ribbon streamers from the waist bow (cloth strips, like the hero's cape)
    const ribMat = std({ color: pinkC.clone().lerp(new THREE.Color('#ff2f8a'), 0.45), roughness: 0.45, side: THREE.DoubleSide });
    const ribbons = [];
    const waist = I.cast(I.wpos(pelvis).add(v(0, 0.03, -1)), v(0, 0, 1), I.colliders, 2);
    const ribTop = waist ? waist.point.clone().add(v(0, 0, -0.012)) : I.wpos(pelvis).add(v(0, 0.03, -0.09));
    for (const s of [-1, 1]) {
      const cl = makeCloth(ribMat, 2, 12);
      const g = new THREE.Group(); g.scale.setScalar(k); g.add(cl.mesh);
      g.position.copy(pelA.worldToLocal(ribTop.clone().add(v(s * 0.035, 0, 0))));
      g.rotation.y = s * 0.35;
      late.push([g, pelA]);
      ribbons.push({ cl, s });
    }
    const dressMats = slotMats(I, ['body', 'belly', 'ribbon', 'ribbon2', 'plate']);
    rim([...dressMats, ribMat], accC, 0.08);
    const tfx = tinyFx(c, late, accC, { centerY: 0.5, beacon: 3.2 });
    const spineTick = I.stockTick(I);
    const dv = v();
    const api = { tf: 0, reset() { this.tf = 0; }, setTransformed(x) { this.tf = clamp01(x); } };
    rig.extra.wild = api;
    let baseE0 = null, dressE0 = null;
    glb.tick = (r, dt, t) => {
      const a = r.anim, up = 1 - a.ko, tf = api.tf, m = r.motion;
      const calm = up * (1 - Math.min(1, Math.abs(a.lunge) * 3)) * (1 - a.victory * 0.6) * (1 - a.charge * 0.7);
      dart(dv, t, 2.7, (r.bankSign ? 0.85 : 0.25) + 0.25 * tf, 1.3);
      m.position.x += (dv.x - (r.bankSign || 0) * 0.5) * calm; m.position.y += dv.y * calm; m.position.z += dv.z * calm;
      m.position.y += Math.max(0, a.lunge) * 3.0;
      spineTick?.(r, dt, t);
      tfx(r, dt, t, tf * 0.8);
      if (!baseE0) {
        baseE0 = r.mats.filter(x => x.userData.glow && x.userData.e0).map(x => [x, x.userData.e0.clone()]);
        dressE0 = [...dressMats, ribMat].filter(x => x.userData.e0).map(x => [x, x.userData.e0.clone()]);
      }
      for (const [x, e] of baseE0) x.userData.e0.copy(e).multiplyScalar(1 + 1.2 * tf);
      for (const [x, e] of dressE0) x.userData.e0.copy(e).add(_tmp.copy(x.color).multiplyScalar(0.18 * tf));
      aura.material.opacity = Math.min(1, (0.24 + 0.07 * T.enr + 0.5 * tf) * (0.85 + 0.15 * Math.sin(t * 2.1)) * (1 + a.charge * 0.8 + a.roar * 0.4)) * up;
      aura.material.color.copy(accC).lerp(_tmp.set('#ffffff'), 0.25 * tf);
      const as = 1 + 0.5 * tf;
      aura.scale.set(6.6 * as, 12.5 * (1 + 0.22 * tf), 1);
      tipGlow.material.opacity = (0.5 + 0.2 * Math.sin(t * 5) + a.charge * 0.8 + tf * 0.4) * up;
      tipGlow.scale.setScalar(1.6 * k * (1 + a.charge * 0.8 + tf * 0.5));
      wings.scale.setScalar(Math.max(0.001, smooth(tf) * (1 + 0.04 * Math.sin(t * 2))));
      wings.children.forEach((w, i) => { const s = i < 2 ? -1 : 1; w.rotation.y = s * (0.75 + 0.2 * Math.sin(t * 3 + i * 0.3)); w.material.opacity = 0.55 * up; });
      halo.scale.setScalar(Math.max(0.001, smooth(tf)));
      halo.rotation.z = t * 0.6;
      hem.visible = tf > 0.01; hem.material.opacity = 0.7 * tf * up;
      const flow = clamp01(a.lunge * 0.9 + a.walk * 0.6 + a.victory * 0.4 + dv.length() * 0.35 * calm + tf * 0.25 + a.roar * 0.5);
      for (const rb of ribbons) poseCloth(rb.cl, { top: v(0, 0, 0), wrap: 0.02, w0: 0.5, w1: 0.34, L: 3.6 + 0.3 * T.spd, ang: 0.45 + flow * 0.95 + 0.08 * Math.sin(t * 1.4 + rb.s) - a.ko * 0.3, flow: flow + 0.25, t: t + rb.s * 0.7, clear: 0.05 });
      const n = sparkles.length;
      for (let i = 0; i < n; i++) {
        const s = sparkles[i], spd = 0.18 + 0.12 * tf, ph = (t * spd + i / n) % 1, an = i * 2.39 + t * (0.5 + tf * 0.6);
        const rr = (2.2 + 1.2 * Math.sin(i * 1.3)) * (1 + 0.4 * tf);
        s.position.set(Math.cos(an) * rr, 1.0 + ph * 10.5, Math.sin(an) * rr * 0.85);
        const tw = 0.5 + 0.5 * Math.sin(t * 7 + i * 1.7);
        s.material.opacity = Math.sin(ph * PI) * (0.55 + 0.45 * tf) * up * (0.4 + 0.6 * tw);
        s.scale.setScalar((0.45 + 0.35 * tw) * (1 + 0.6 * tf));
      }
    };
  }

  // ================================================================ HERO TEAM
  function team(I, glb) {
    const { c } = I, rig = c.rig;
    const redC = colorOf(c.colors.accent, '#ef233c');
    const late = glb.late;
    const COLS = ['#' + redC.getHexString(), '#ff8c2a', '#ffd23f', '#3ad0ff', '#b56aff'];
    const raw = [1, 2, 3, 4, 5].map(i => I.bones.get('hero' + i)).filter(Boolean);
    // member 0 = the front (leader); slots are ordered front to back like the procedural formation
    const sorted = raw.map(b => ({ b, w: I.wpos(b) })).sort((p, q) => q.w.z - p.w.z);
    const N = sorted.length;
    const S = sorted.map(m => v(m.w.x, 0, m.w.z));
    const tex = glowTex();
    const members = sorted.map((m, i) => {
      const headB = I.bones.get(i === 0 && nodeName(m.b) === 'hero1' ? 'head' : nodeName(m.b) + '_head') || I.bones.get('head');
      const accC = colorOf(COLS[raw.indexOf(m.b)] || COLS[i], '#ffffff');
      // attack / beam source: right hand for the rigged leader, chest height in front for the rest
      const hand = nodeName(m.b) === 'hero1' ? I.empties.get('handR') : null;
      const focus = hand ? I.pointAt(hand.parent, I.wpos(hand), 'focus' + i) : I.pointAt(m.b, m.w.clone().add(v(0, 0.22, 0.12)), 'focus' + i);
      const beacon = sprite(tex, accC, 2.2, { opacity: 0.4 });
      const a = I.anchor(m.b);
      beacon.position.copy(a.worldToLocal(v(m.w.x, 0.45, m.w.z)));
      late.push([beacon, a]);
      return { p: I.proxy(m.b), bone: m.b, headB, accC, focus, beacon, ph: i * 1.3 + 0.2 * i * i, h: m.w.y, hover: m.w.y > 0.6 ? 0.06 : 0, off: v(), offV: v(), lean: 0 };
    });
    // the combined beam converges in front of the team
    const frontZ = S[0].z;
    const conv = I.pointAt(I.rootBone, v(0, 0.9, frontZ + 0.5), 'conv');
    rig.mouths.length = 0; rig.mouths.push(conv); rig.mouth = conv;
    const tfx = tinyFx(c, late, redC, { centerY: 0.45, beacon: 5.5, disc: 6.0, ring: 2.7, trail: false });
    rim(slotMats(I, ['body']), redC, 0.1);
    const order = [0, 2, 1, 3, 4].filter(i => i < N);
    const ids = members.map((_, i) => i);
    const api = {
      active: 0, seq: 0, swapK: 1, slotOf: ids.slice(), fromSlot: ids.slice(), incoming: -1, outgoing: -1,
      members,
      reset() { this.active = 0; this.seq = 0; this.swapK = 1; this.slotOf = ids.slice(); this.fromSlot = ids.slice(); this.incoming = this.outgoing = -1; members.forEach(m => { m.off.set(0, 0, 0); m.offV.set(0, 0, 0); m.lean = 0; }); },
      onStrike(hitNo, nHits, kind) {
        if (kind === 'melee' || kind === 'multi' || kind === 'grab') this.active = order[(this.seq++) % order.length];
        else this.active = this.slotOf.indexOf(0);
      },
      onHit(big) {
        const front = this.slotOf.indexOf(0);
        members.forEach((m, i) => {
          const kk = (i === front ? 1.6 : 0.6) * (big ? 1.5 : 1);
          m.offV.add(v(Math.sin(i * 12.9 + this.seq * 3.1) * 0.22 * kk, 0, -0.26 * kk));
          m.lean = Math.min(1, m.lean + (i === front ? 0.8 : 0.3) * (big ? 1.4 : 1));
        });
        return null;
      },
      // tag in: the member at the back swaps places with the front member
      beginSwap() {
        this.fromSlot = this.slotOf.slice();
        const outI = this.slotOf.indexOf(0), inI = this.slotOf.indexOf(N - 1);
        const next = this.slotOf.slice();
        next[outI] = N - 1; next[inI] = 0;
        this.slotOf = next; this.incoming = inI; this.outgoing = outI; this.swapK = 0;
      },
      frontLocal(out) { return out.copy(S[0]); },
      frontWorld(out) { rig.body.updateWorldMatrix(true, false); return rig.body.localToWorld(out.copy(S[0]).setY(0.4)); },
      beamSources() { return members.map(m => m.focus).filter(Boolean); },
      memberColors() { return members.map(m => m.accC); },
    };
    rig.extra.wild = api;
    const p = v();
    glb.tick = (r, dt, t) => {
      const a = r.anim, m0 = r.motion, sc = r.scale || 1;
      // the group lunge becomes one member dashing in; the group KO tilt becomes members falling one by one
      m0.position.z -= a.lunge * a.reach;
      m0.rotation.x = 0;
      const sk = smooth(api.swapK);
      for (let i = 0; i < N; i++) {
        const m = members[i];
        const from = S[api.fromSlot[i]], to = S[api.slotOf[i]];
        p.lerpVectors(from, to, sk).sub(S[i]);
        if (i === api.incoming && api.swapK < 1) p.y += Math.sin(api.swapK * PI) * 0.5;
        if (i === api.outgoing && api.swapK < 1) p.y += Math.sin(api.swapK * PI) * 0.15;
        m.off.addScaledVector(m.offV, dt); m.offV.multiplyScalar(Math.exp(-dt * 6)); m.off.multiplyScalar(Math.exp(-dt * 3));
        m.lean *= Math.exp(-dt * 3);
        p.add(m.off);
        const act = i === api.active ? a.lunge : 0;
        p.z += act * (a.reach / sc + frontZ + 0.14 - to.z - 0.06);
        if (m.hover) p.y += (m.hover + Math.sin(t * 1.9 + m.ph) * 0.03) * (1 - a.ko) - m.hover;
        p.y += Math.max(0, act) * 3.0 / sc; // leap up to the kaiju's chest
        p.y += Math.abs(Math.sin(t * 3.0 + m.ph)) * 0.035 * a.victory;
        const fallK = clamp01(a.ko * 1.5 - i * 0.12);
        p.y -= m.h * 0.82 * (1 - Math.cos(fallK * 1.45)); // the bone sits at the hips: keep the body on the ground as it falls
        p.z -= m.h * 0.5 * Math.sin(fallK * 1.45);
        m.p.position.copy(p);
        m.p.rotation.set(-fallK * 1.45 + act * 0.25 - m.lean * 0.25 - a.recoil * 0.1, i === api.incoming && api.swapK < 1 ? (1 - sk) * PI * 2 : 0, Math.sin(t * 0.5 + m.ph) * 0.03);
        m.beacon.material.opacity = (0.32 + 0.12 * Math.sin(t * 3 + i) + (i === api.active ? a.lunge * 0.6 : 0)) * (1 - a.ko * 0.8);
      }
      tfx(r, dt, t, api.swapK < 1 ? Math.sin(api.swapK * PI) : 0);
    };
  }

  // ================================================================ ARMY
  function army(I, glb) {
    const { c, T, bones } = I, rig = c.rig;
    const accC = colorOf(c.colors.accent, '#ff8c1a'), khakiC = colorOf(c.colors.belly, '#8a8d6b'), hullC = colorOf(c.colors.body, '#4b5320');
    const late = glb.late;
    const root = c.root; // motion group: model-local space
    const M = {
      jet: std({ color: khakiC.clone().lerp(new THREE.Color('#9aa0a6'), 0.6), roughness: 0.4, metalness: 0.55 }),
      canopy: std({ color: '#1a2430', roughness: 0.1, metalness: 0.8 }),
      mark: std({ color: accC, emissive: accC, emissiveIntensity: 0.2, roughness: 0.5 }),
      flameC: accC.clone().lerp(new THREE.Color('#ffe0a0'), 0.4),
    };
    M.mark.userData.glow = true;
    const turNames = [...bones.keys()].filter(n => /_tur$/.test(n));
    const tips = boneTips(I, turNames);
    const sTex = smokeTex(), gTex = glowTex();
    const mkUnit = (name, kind, i) => {
      const hb = bones.get(name), tb = bones.get(name + '_tur'); if (!hb || !tb) return null;
      const d = I.wdir(tb), w = I.wpos(hb);
      const tip = tips.get(name + '_tur') || I.wpos(tb).addScaledVector(d, 3);
      const u = {
        kind, hull: I.proxy(hb), tur: I.proxy(tb), yaw0: Math.atan2(d.x, d.z), pitch0: Math.asin(clamp01(d.y)),
        home: v(w.x, 0, w.z), muzzle: I.pointAt(tb, tip, 'muzzle:' + name), top: v(0, kind === 'tank' ? 2.1 : 1.9, 0),
        anchor: I.anchor(hb), ph: i * 2.3 + (kind === 'tank' ? 0 : 1.1), yaw: 0, pitch: 0, rec: 0, jolt: 0, smoke: 0, floor: 0, smokes: [], fires: [],
        setTur(yaw, pitch, rec) { this.tur.rotation.set(-pitch, yaw, 0, 'YXZ'); this.tur.position.set(-Math.sin(yaw + this.yaw0) * 0.45 * rec, 0, -Math.cos(yaw + this.yaw0) * 0.45 * rec); },
        setHull(rx, rz, y) { this.hull.rotation.set(rx, 0, rz); this.hull.position.set(0, y, 0); },
        worldTop(out) { this.anchor.updateWorldMatrix(true, false); return this.anchor.localToWorld(out.copy(this.top)); },
      };
      return u;
    };
    const tanks = [1, 2, 3, 4].map((n, i) => mkUnit('tank' + n, 'tank', i)).filter(Boolean);
    const arts = [1, 2, 3].map((n, i) => mkUnit('art' + n, 'art', i)).filter(Boolean);
    const units = [...tanks, ...arts];
    // the two centre tanks fire the default salvo (like the procedural army's twin mouths)
    if (tanks.length >= 3) { rig.mouths.length = 0; rig.mouths.push(tanks[1].muzzle, tanks[2].muzzle); rig.mouth = tanks[1].muzzle; }
    // reinforcements: two more tanks (rigid copies of the GLB tank) that roll in from behind
    const reserveG = new THREE.Group(); reserveG.visible = false;
    const reserves = [];
    if (tanks.length) {
      const hb = bones.get('tank2') || bones.get('tank1'), tb = bones.get(nodeName(hb) + '_tur');
      const hw = I.wpos(hb), tw = I.wpos(tb), d = I.wdir(tb), hullYaw = Math.atan2(d.x, d.z);
      const tip = tips.get(nodeName(tb)) || tw.clone().addScaledVector(d, 3);
      const xs = units.map(u => Math.abs(u.home.x)), edge = Math.max(8.6, ...xs.map(x => x + 3.2));
      [v(-edge, 0, 0.4), v(edge, 0, 0.4)].forEach((pos, i) => {
        const g = new THREE.Group();
        const hull = rigidCopy(I, [nodeName(hb)], hw), tur = rigidCopy(I, [nodeName(tb)], tw);
        // un-yaw the copy so it faces straight ahead, then put the turret back on its ring
        const inner = new THREE.Group(); inner.rotation.y = -hullYaw; inner.position.y = hw.y;
        tur.position.copy(tw).sub(hw);
        const muzzle = new THREE.Object3D(); muzzle.position.copy(tip).sub(tw); tur.add(muzzle);
        hull.add(tur); inner.add(hull); g.add(inner); reserveG.add(g);
        const top = new THREE.Object3D(); top.position.set(0, hw.y + 2.1, 0); g.add(top);
        reserves.push({
          kind: 'tank', reserve: true, g, hullG: hull, turG: tur, home: pos.clone(), muzzle, topO: top, yaw0: 0, pitch0: 0,
          ph: 7.1 + i * 2.9, yaw: 0, pitch: 0, rec: 0, jolt: 0, smoke: 0, floor: 0, smokes: [], fires: [], top: v(0, hw.y + 2.1, 0), anchor: g,
          setTur(yaw, pitch, rec) { this.turG.rotation.set(-pitch, yaw, 0, 'YXZ'); this.turG.position.copy(tw).sub(hw); this.turG.position.x -= Math.sin(yaw) * 0.45 * rec; this.turG.position.z -= Math.cos(yaw) * 0.45 * rec; },
          setHull(rx, rz, y) { this.hullG.rotation.set(rx, 0, rz); this.hullG.position.y = y; },
          worldTop(out) { this.topO.updateWorldMatrix(true, false); return this.topO.getWorldPosition(out); },
        });
        g.position.copy(pos);
      });
    }
    late.push([reserveG, null]);
    for (const u of [...units, ...reserves]) {
      for (let j = 0; j < 4; j++) { const s = sprite(sTex, '#3a3632', 1.5, { additive: false, opacity: 0, fog: true }); s.visible = false; u.smokes.push(s); late.push([s, u.anchor]); }
      for (let j = 0; j < 2; j++) { const s = sprite(gTex, '#ff7a20', 1.4, { opacity: 0 }); s.visible = false; u.fires.push(s); late.push([s, u.anchor]); }
    }
    // infantry: the GLB squads are one bone. They stand their ground as modelled; when the army breaks they are
    // swapped for individual runners that scatter (like the procedural army's soldiers).
    const infB = bones.get('infantry'), infP = infB ? I.proxy(infB) : null;
    const cells = new Map();
    if (infB) scanVerts(I, (bn, P) => { if (bn !== 'infantry') return; const key = Math.round(P.x / 0.55) + ':' + Math.round(P.z / 0.55); const q = cells.get(key); if (!q) cells.set(key, { x: P.x, z: P.z, n: 1 }); else { q.x += P.x; q.z += P.z; q.n++; } });
    const soldiers = [...cells.values()].filter(q => q.n > 12).slice(0, 80).map((q, i) => {
      const x = q.x / q.n, z = q.z / q.n, an = Math.atan2(z + 0.5, x) + Math.sin(i * 7.3) * 0.6;
      return { x, z, ph: i * 1.7, flee: v(Math.cos(an), 0, Math.sin(an) - 0.4).normalize(), yaw: 0 };
    });
    let inf = null, helm = null;
    if (soldiers.length) {
      const infGeo = new THREE.CapsuleGeometry(0.17, 0.5, 3, 6); infGeo.translate(0, 0.42, 0);
      const helmGeo = new THREE.SphereGeometry(0.19, 8, 6); helmGeo.scale(1, 0.7, 1); helmGeo.translate(0, 0.95, 0);
      inf = new THREE.InstancedMesh(infGeo, std({ color: hullC.clone().multiplyScalar(0.8).lerp(khakiC, 0.25), roughness: 0.9 }), soldiers.length);
      helm = new THREE.InstancedMesh(helmGeo, std({ color: hullC, roughness: 0.72, metalness: 0.22 }), soldiers.length);
      for (const im of [inf, helm]) { im.frustumCulled = false; im.visible = false; late.push([im, null]); }
    }
    // jets: orbit overhead, strafe on an air strike, climb away on KO. The GLB jet (when its tier part is
    // present) leads the flight; procedural wingmen fill the rest.
    const nJets = 2 + (T.spd >= 3 ? 1 : 0);
    const sky = new THREE.Group();
    late.push([sky, null]);
    const jets = [];
    const jetB = bones.get('jet');
    let jetHasMesh = false;
    if (jetB) scanVerts(I, (bn) => { if (bn === 'jet') jetHasMesh = true; });
    for (let j = 0; j < nJets; j++) {
      const J = { prev: v(), ang: j * PI * 2 / nJets, pos: v() };
      if (j === 0 && jetHasMesh) { J.p = I.proxy(jetB); J.bone = jetB; J.rest = I.wpos(jetB); }
      else { J.g = jetModel(c, M); J.g.scale.setScalar(0.95); sky.add(J.g); }
      jets.push(J);
    }
    if (jetB && !jetHasMesh) I.proxy(jetB);
    const rotors = ['rotor', 'rotor2'].map(n => bones.get(n)).filter(Boolean).map(b => I.proxy(b));
    const radar = bones.get('radar') ? I.proxy(bones.get('radar')) : null;
    const walls = [...bones.keys()].filter(n => /^(truck|wall\d|mast)$/.test(n)).map((n, i) => ({ p: I.proxy(bones.get(n)), ph: i * 2.3 }));
    const fliers = ['heli', 'heli2', 'drone'].map(n => bones.get(n)).filter(Boolean).map((b, i) => ({ p: I.proxy(b), ph: i * 1.7, h: I.wpos(b).y }));

    let targetFn = null, jt = 0, rr = 0, infHop = 0;
    const _p = v(), _q = v(), _m4 = new THREE.Matrix4(), _q4 = new THREE.Quaternion(), _s = v(1, 1, 1), _e = new THREE.Euler();
    const kick = (u) => { u.rec = 1; };
    const wp = (o) => { o.updateWorldMatrix(true, false); return o.getWorldPosition(new V3()); };
    const allTanks = () => [...tanks, ...(api.wave > 0 ? reserves : [])];
    const api = {
      noLunge: true, noLift: true, strafeK: -1, strafeTgt: v(0, 0, 14), reinK: 0, wave: 0, jetCount: nJets,
      reset() { this.strafeK = -1; this.reinK = 0; this.wave = 0; reserveG.visible = false; infHop = 0; for (const u of [...units, ...reserves]) { u.smoke = 0; u.floor = 0; u.jolt = 0; u.rec = 0; } },
      setTarget(fn) { targetFn = fn; },
      onHit(big, out) {
        const pool = units.concat(api.wave > 0 ? reserves : []);
        if (!pool.length) return null;
        const u = pool[(rr++ * 5 + (big ? 2 : 0)) % pool.length];
        u.smoke = 1; u.floor = Math.min(0.45, u.floor + 0.2); u.jolt = big ? 1.4 : 1;
        infHop = Math.max(infHop, big ? 1 : 0.6);
        return out ? u.worldTop(out) : null;
      },
      // projectile origin per ability: artillery lobs, tank salvos, infantry rifle fire
      origin(ab, hitNo, nHits, out) {
        const id = String(ab || '');
        if (id.includes('artillery') && arts.length) {
          const u = arts[(hitNo - 1) % arts.length]; kick(u);
          out.copy(wp(u.muzzle));
          return { arcK: 0.5, ms: 760, flash: 1.5, scale: 0.6 };
        }
        if (id.includes('tank') || id.includes('column')) {
          const list = allTanks(); list.forEach(kick);
          out.copy(wp(list[0].muzzle));
          return { arcK: 0.03, ms: 420, flash: 1.3, scale: 0.6, extras: list.slice(1).map(u => wp(u.muzzle)) };
        }
        if ((id.includes('rifle') || id.includes('volley')) && soldiers.length) {
          const pick = (j) => { const s = soldiers[(hitNo * 7 + j * 11) % soldiers.length]; root.updateWorldMatrix(true, false); return root.localToWorld(v(s.x, 0.85, s.z)); };
          out.copy(pick(0));
          return { arcK: 0.02, ms: 340, flash: 0.35, scale: 0.32, extras: [pick(1), pick(2), pick(3)] };
        }
        const u = tanks[(hitNo - 1) % tanks.length]; kick(u);
        out.copy(wp(u.muzzle));
        return { arcK: 0.05, ms: 460, flash: 1.1, scale: 0.6 };
      },
      strafe(worldTarget) {
        if (worldTarget) { root.updateWorldMatrix(true, false); this.strafeTgt.copy(root.worldToLocal(worldTarget.clone())); }
        this.strafeK = 0;
      },
      jetWorld(i, out) { const J = jets[i % jets.length], o = J.bone || J.g; o.updateWorldMatrix(true, false); return o.getWorldPosition(out); },
      reinforce(instant) { this.wave++; reserveG.visible = true; if (instant) this.reinK = 1; },
    };
    rig.extra.wild = api;

    glb.tick = (r, dt, t) => {
      const a = r.anim, ko = a.ko, m0 = r.motion;
      m0.rotation.x = 0; // a formation doesn't tip over
      jt += dt;
      root.updateWorldMatrix(true, false);
      let tl = null;
      if (targetFn) { try { tl = root.worldToLocal(targetFn(_q)); } catch { tl = null; } }
      // ground units: aim, recoil, jolt, smoke / fire
      const allU = units.concat(reserveG.visible ? reserves : []);
      for (let i = 0; i < allU.length; i++) {
        const u = allU[i];
        if (u.reserve) {
          const kk = smooth(api.reinK), z = lerp(-26, u.home.z, kk);
          u.g.position.set(u.home.x, kk < 1 ? Math.abs(Math.sin(t * 9 + u.ph)) * 0.08 : 0, z);
        }
        const px = u.reserve ? u.g.position.x : u.home.x, pz = u.reserve ? u.g.position.z : u.home.z;
        let yaw, pitch;
        if (tl) {
          const dx = tl.x - px, dz = tl.z - pz;
          yaw = Math.atan2(dx, dz) - u.yaw0;
          yaw = Math.atan2(Math.sin(yaw), Math.cos(yaw));
          pitch = u.kind === 'tank' ? Math.max(-0.05, Math.min(0.14, Math.atan2(tl.y * 0.5 - 1.6, Math.hypot(dx, dz)) * 0.6)) : 0;
        } else {
          yaw = Math.sin(t * 0.35 + u.ph) * 0.5; pitch = u.kind === 'tank' ? -Math.sin(t * 0.5 + u.ph) * 0.05 : 0;
        }
        if (ko > 0) { yaw = lerp(yaw, u.ph - 3, ko * 0.4); pitch = lerp(pitch, u.kind === 'tank' ? -0.18 : 0.15, ko); }
        const kk = 1 - Math.exp(-dt * 4);
        u.yaw += (yaw - u.yaw) * kk; u.pitch += (pitch - u.pitch) * kk;
        u.rec *= Math.exp(-dt * 7);
        u.setTur(u.yaw, u.pitch, u.rec);
        u.jolt *= Math.exp(-dt * 5);
        u.setHull(-u.jolt * 0.12 * Math.sin(t * 20) + ko * 0.16 * Math.sin(u.ph * 3), u.jolt * 0.1 * Math.sin(t * 17 + u.ph) + ko * 0.2 * Math.cos(u.ph * 2), u.jolt * 0.25 * Math.abs(Math.sin(t * 14)) - ko * 0.12);
        u.smoke = Math.max(u.floor, u.smoke - dt * 0.35, ko);
        const fire = Math.max(ko, u.smoke > 0.9 ? (u.smoke - 0.9) * 10 : 0);
        for (let j = 0; j < u.smokes.length; j++) {
          const s = u.smokes[j], ph = (jt * 0.45 + j / u.smokes.length + u.ph) % 1;
          s.visible = u.smoke > 0.02;
          if (!s.visible) continue;
          s.position.set(u.top.x + Math.sin(ph * 4 + u.ph) * 0.4 + ph * 1.2, u.top.y + ph * 5 * (0.6 + u.smoke * 0.6), u.top.z - ph * 0.8);
          s.scale.setScalar(1.0 + ph * 3.2 * (0.6 + u.smoke));
          s.material.opacity = Math.sin(ph * PI) * 0.75 * u.smoke;
        }
        for (let j = 0; j < u.fires.length; j++) {
          const s = u.fires[j]; s.visible = fire > 0.02;
          if (!s.visible) continue;
          s.position.set(u.top.x + (j ? 0.4 : -0.3), u.top.y - 0.4 + Math.sin(t * 13 + j) * 0.12, u.top.z + (j ? -0.3 : 0.4));
          s.scale.setScalar((1.2 + 0.4 * Math.sin(t * 17 + j * 2)) * (0.6 + fire * 0.6));
          s.material.opacity = fire * (0.7 + 0.3 * Math.sin(t * 23 + j));
        }
      }
      if (radar) radar.rotation.y = t * 1.3 * (1 - ko);
      for (const rt of rotors) rt.rotation.y = t * 22 * (1 - ko * 0.9);
      for (const w of walls) w.p.rotation.set(ko * 0.1 * Math.sin(w.ph * 3), 0, ko * 0.14 * Math.cos(w.ph * 2));
      for (const f of fliers) {
        const drop = smooth(ko * 1.3 - f.ph * 0.08);
        f.p.position.set(Math.sin(t * 0.6 + f.ph) * 0.5 * (1 - drop), Math.sin(t * 1.4 + f.ph) * 0.25 * (1 - drop) - drop * Math.max(0, f.h - 0.5), Math.cos(t * 0.45 + f.ph) * 0.4 * (1 - drop));
        f.p.rotation.set(0.08 * Math.sin(t * 0.9 + f.ph) + drop * 0.5, 0, 0.1 * Math.sin(t * 0.6 + f.ph) + drop * (f.ph % 2 ? 0.9 : -0.9));
      }
      // infantry: hop when shelled; when the army breaks, swap the squads for runners that flee
      infHop = Math.max(0, infHop - dt * 1.6);
      const broke = ko > 0.02;
      if (infP) {
        infP.scale.setScalar(broke ? 0.0001 : 1);
        infP.position.set(0, Math.abs(Math.sin(infHop * PI * 2)) * infHop * 0.3 + Math.abs(Math.sin(t * 4)) * 0.1 * a.victory, 0);
      }
      if (inf) {
        inf.visible = helm.visible = broke;
        if (broke) {
          for (let i = 0; i < soldiers.length; i++) {
            const s = soldiers[i];
            const fl = smooth(ko * 1.3 - (i % 7) * 0.04);
            const sc = 1 - smooth((fl - 0.6) / 0.4);
            _e.set(0, Math.atan2(s.flee.x, s.flee.z), 0);
            _q4.setFromEuler(_e); _s.setScalar(Math.max(0.0001, sc));
            _m4.compose(_p.set(s.x + s.flee.x * fl * 14, fl > 0 && fl < 1 ? Math.abs(Math.sin(t * 12 + s.ph)) * 0.12 : 0, s.z + s.flee.z * fl * 14), _q4, _s);
            inf.setMatrixAt(i, _m4); helm.setMatrixAt(i, _m4);
          }
          inf.instanceMatrix.needsUpdate = true; helm.instanceMatrix.needsUpdate = true;
        }
      }
      // jets
      for (let j = 0; j < jets.length; j++) {
        const J = jets[j];
        const th = jt * 0.42 + J.ang;
        const R = 10 + ko * 30, h = 12 + Math.sin(jt * 0.7 + j) * 0.8 + ko * 18;
        _p.set(Math.cos(th) * R, h, Math.sin(th) * R - 1);
        if (api.strafeK >= 0) {
          const kk = clamp01(api.strafeK - j * 0.07);
          const w = smooth(kk / 0.18) * (1 - smooth((kk - 0.82) / 0.18));
          const tg = api.strafeTgt;
          _q.set(tg.x + (j - (jets.length - 1) / 2) * 3.0, 15 - 10 * Math.sin(kk * PI), lerp(-30, tg.z + 30, kk));
          _p.lerp(_q, w);
        }
        const d = _q.subVectors(_p, J.prev);
        let pitch = 0, yaw = J.yaw || 0;
        if (d.lengthSq() > 1e-6) { yaw = Math.atan2(d.x, d.z); pitch = -Math.atan2(d.y, Math.hypot(d.x, d.z)); }
        J.yaw = yaw;
        const bank = api.strafeK >= 0 ? 0.1 : 0.55;
        if (J.p) { J.p.position.copy(_p).sub(J.rest); J.p.rotation.set(pitch, yaw, bank, 'YXZ'); }
        else {
          J.g.position.copy(_p); J.g.rotation.set(pitch, yaw, bank, 'YXZ');
          J.g.userData.flame.scale.set(1, 1, 0.8 + 0.3 * Math.sin(jt * 30 + j));
        }
        J.prev.copy(_p);
      }
    };
  }

  return { hero, girl, team, army };
}
