// Three.js stage: renderer, camera director, environment and fighters.
//
// createStage(canvas, opts?) -> {
//   scene, camera, renderer, fighters: [modelA, modelB], arena, envId, director,
//   setEnvironment(envId), showcase(snapshot), setFighters(snapA, snapB), resize(), dispose(),
//   preload(snapshots) -> Promise (lazy GLB models; await it before setFighters so a battle never starts on the
//   procedural fallback), whenReady() -> Promise (pending showcase load),
//   addTicker(fn(dt, t)) -> remove, homes: [Vector3, Vector3]
// }

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { buildKaijuModel, disposeObject, preloadKaijuModels, kaijuModelsReady } from './kaiju-model.js';
import { buildArena } from './arena.js';

const V3 = THREE.Vector3;

function makeDirector(camera) {
  const d = {
    mode: 'free',
    target: new V3(0, 5, 0), radius: 30, height: 10, angle: 0,
    cur: { target: new V3(0, 5, 0), radius: 30, height: 10, angle: 0 },
    orbitAmp: 0.45, orbitSpeed: 0.11, orbitT: 0, baseAngle: 0,
    focusPos: null, focusScale: 1, focusLeft: 0, focusAngle: null,
    bias: new V3(), shakeAmp: 0, snapNext: true,
    battle: null,
  };
  d.setShowcase = (model) => {
    const rig = model.userData.rig;
    const h = rig.height, r = Math.max(rig.radius, h * 0.6);
    d.mode = 'showcase';
    d.target.set(0, model.position.y + h * 0.48, 0);
    const tiny = !!rig.extra?.tiny; // human-scale wild fighters: zoom to fit instead of the kaiju minimum
    d.radius = Math.max(tiny ? 3 : 16, r * 1.8 + h * 1.0);
    d.height = tiny ? h * 0.35 + 0.6 : rig.hover ? h * 0.9 + 6 : h * 0.45 + 4;
    if (tiny) d.target.y = model.position.y + h * 0.62;
    d.baseAngle = 0.55; d.orbitAmp = 0.12; d.orbitSpeed = 0.2;
    d.focusLeft = 0; d.snapNext = true;
  };
  d.setBattle = (homes, rigs) => {
    const maxH = Math.max(rigs[0].height, rigs[1].height);
    const span = homes[1].x - homes[0].x;
    d.mode = 'battle';
    const maxR = Math.max(rigs[0].back, rigs[1].back, rigs[0].width / 2, rigs[1].width / 2);
    d.battle = { center: homes[0].clone().add(homes[1]).multiplyScalar(0.5), maxH, span, halfW: span / 2 + maxR * 0.9 };
    d.target.copy(d.battle.center); d.target.y = homes[0].y * 0.5 + maxH * 0.42;
    d.radius = Math.max(30, span * 1.05 + maxH * 1.5);
    d.height = maxH * 0.6 + 7;
    d.baseAngle = 0; d.orbitAmp = 0.5; d.orbitSpeed = 0.09;
    d.focusLeft = 0; d.bias.set(0, 0, 0); d.snapNext = true;
  };
  d.free = (pos, target) => { d.mode = 'free'; camera.position.copy(pos); d.target.copy(target); };
  // push in on a point for `sec` real seconds; scale < 1 = closer
  d.focus = (pos, scale = 0.6, sec = 1, angle = null) => {
    d.focusPos = pos.clone(); d.focusScale = scale; d.focusLeft = sec; d.focusAngle = angle;
  };
  d.clearFocus = () => { d.focusLeft = 0; };
  d.shake = (amp = 0.5) => { d.shakeAmp = Math.max(d.shakeAmp, amp); };
  d.setBias = (x) => { d.bias.set(x, 0, 0); };
  d.update = (dt) => {
    if (d.mode === 'free') { camera.lookAt(d.target); return; }
    d.orbitT += dt;
    let tgt = d.target.clone().add(d.bias), rad = d.radius, h = d.height;
    if (d.mode === 'battle' && d.battle) {
      // keep both fighters in frame on narrow / square canvases
      const hf = Math.atan(Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.aspect);
      rad = Math.max(rad, d.battle.halfW * 1.08 / Math.tan(hf));
    }
    let ang = d.baseAngle + Math.sin(d.orbitT * d.orbitSpeed) * d.orbitAmp;
    if (d.focusLeft > 0) {
      d.focusLeft -= dt;
      tgt = tgt.lerp(d.focusPos, 0.75);
      rad *= d.focusScale; h = h * (0.5 + d.focusScale * 0.5);
      if (d.focusAngle != null) ang = d.focusAngle;
    }
    const c = d.cur;
    const k = d.snapNext ? 1 : 1 - Math.exp(-dt * 2.6);
    d.snapNext = false;
    c.target.lerp(tgt, k); c.radius += (rad - c.radius) * k; c.height += (h - c.height) * k;
    c.angle += (ang - c.angle) * k;
    camera.position.set(c.target.x + Math.sin(c.angle) * c.radius, c.target.y + c.height, c.target.z + Math.cos(c.angle) * c.radius);
    const look = c.target.clone();
    if (d.shakeAmp > 0.001) {
      const s = d.shakeAmp;
      camera.position.x += (Math.random() - 0.5) * s * 2; camera.position.y += (Math.random() - 0.5) * s * 2;
      look.x += (Math.random() - 0.5) * s; look.y += (Math.random() - 0.5) * s;
      d.shakeAmp *= Math.exp(-dt * 4.5);
    }
    camera.lookAt(look);
  };
  return d;
}

export function createStage(canvas, opts = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, opts.maxPixelRatio || 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  // soft image-based lighting so metal / glossy parts read in every arena
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();
  scene.environment = envTex;
  scene.environmentIntensity = 0.35;
  const camera = new THREE.PerspectiveCamera(40, 1, 0.5, 1400);
  camera.position.set(0, 12, 40);

  const useBloom = opts.bloom !== false;
  let composer = null, bloom = null;
  if (useBloom) {
    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.6, 0.45, 1.0);
    composer.addPass(bloom);
    composer.addPass(new OutputPass());
  }

  const director = makeDirector(camera);
  const tickers = new Set();
  const clock = new THREE.Clock();
  let t = 0, raf = 0, disposed = false;

  const stage = {
    scene, camera, renderer, director,
    fighters: [null, null], homes: [new V3(-10, 0, 0), new V3(10, 0, 0)],
    arena: null, envId: null, showcaseModel: null, mode: 'idle',
    _snaps: [null, null],
  };

  stage.addTicker = (fn) => { tickers.add(fn); return () => tickers.delete(fn); };

  stage.resize = () => {
    const w = Math.max(1, canvas.clientWidth || canvas.width || 1);
    const h = Math.max(1, canvas.clientHeight || canvas.height || 1);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (composer) { composer.setSize(w, h); bloom.resolution.set(w / 2, h / 2); }
  };

  const groundY = () => (stage.arena ? stage.arena.fighterY : 0);

  stage.setEnvironment = (envId) => {
    if (stage.arena) { stage.arena.dispose(); stage.arena = null; }
    try { stage.arena = buildArena(envId, { renderer }); }
    catch (e) { console.warn('[stage] arena failed, using studio', envId, e); stage.arena = buildArena('studio', { renderer }); }
    stage.arena.apply(scene);
    stage.envId = envId;
    // reseat models on the new ground
    if (stage.showcaseModel) { stage.showcaseModel.position.y = stage.arena.showcaseY; director.setShowcase(stage.showcaseModel); }
    if (stage.fighters[0] && stage.fighters[1]) layoutFighters();
    return stage.arena;
  };

  const clearModels = () => {
    if (stage.showcaseModel) { disposeObject(stage.showcaseModel); stage.showcaseModel = null; }
    for (let i = 0; i < 2; i++) if (stage.fighters[i]) { disposeObject(stage.fighters[i]); stage.fighters[i] = null; }
    stage._snaps = [null, null];
  };

  // GLB models load lazily, per frame. preload() resolves once the given snapshots' models can be built (it never
  // rejects: a frame that fails to load uses its procedural plan).
  stage.preload = (snaps) => preloadKaijuModels(snaps);
  let showReq = 0, showPending = Promise.resolve();
  stage.whenReady = () => showPending;

  // Synchronous when the frame's model is already cached. Otherwise the current model stays up while the GLB
  // loads and is swapped once it is in (a newer showcase / setFighters call cancels the swap); returns null then.
  stage.showcase = (snapshot) => {
    const req = ++showReq;
    if (!kaijuModelsReady([snapshot])) {
      showPending = preloadKaijuModels([snapshot]).then(() => { if (req === showReq && !disposed) showNow(snapshot); });
      return null;
    }
    return showNow(snapshot);
  };
  const showNow = (snapshot) => {
    clearModels();
    const m = buildKaijuModel(snapshot);
    m.position.set(0, stage.arena ? stage.arena.showcaseY : 0, 0);
    m.rotation.y = 0.5;
    scene.add(m);
    stage.showcaseModel = m;
    stage.mode = 'showcase';
    director.setShowcase(m);
    return m;
  };

  function layoutFighters() {
    const [a, b] = stage.fighters;
    const ra = a.userData.rig, rb = b.userData.rig;
    const gap = 5;
    const y = groundY();
    stage.homes[0].set(-(ra.front + gap / 2), y, 0);
    stage.homes[1].set(rb.front + gap / 2, y, 0);
    // keep the pair roughly centred
    const shift = (stage.homes[0].x + stage.homes[1].x) / 2;
    stage.homes[0].x -= shift; stage.homes[1].x -= shift;
    ra.bankSign = 1; rb.bankSign = -1;
    a.position.copy(stage.homes[0]); a.rotation.set(0, Math.PI / 2, 0);
    b.position.copy(stage.homes[1]); b.rotation.set(0, -Math.PI / 2, 0);
    director.setBattle(stage.homes, [ra, rb]);
  }

  stage.setFighters = (snapA, snapB) => {
    showReq++; // a pending showcase swap must not replace the fighters
    clearModels();
    stage.fighters[0] = buildKaijuModel(snapA);
    stage.fighters[1] = buildKaijuModel(snapB);
    scene.add(stage.fighters[0], stage.fighters[1]);
    stage._snaps = [snapA, snapB];
    stage.mode = 'battle';
    layoutFighters();
    return stage.fighters;
  };

  function frame() {
    if (disposed) return;
    raf = requestAnimationFrame(frame);
    const dt = Math.min(0.1, clock.getDelta());
    t += dt;
    if (stage.arena) stage.arena.update(dt, t);
    if (stage.showcaseModel) {
      // memorial statues turn very slowly
      stage.showcaseModel.rotation.y += dt * (stage.showcaseModel.userData.rig?.fallen ? 0.06 : 0.35);
      stage.showcaseModel.userData.rig.update(dt, t);
    }
    for (const f of stage.fighters) if (f) f.userData.rig.update(dt, t);
    for (const fn of tickers) { try { fn(dt, t); } catch (e) { console.error('[stage] ticker error', e); } }
    director.update(dt);
    if (composer) composer.render(dt); else renderer.render(scene, camera);
  }

  let ro = null;
  if (typeof ResizeObserver !== 'undefined') { ro = new ResizeObserver(() => stage.resize()); ro.observe(canvas); }

  stage.dispose = () => {
    disposed = true;
    cancelAnimationFrame(raf);
    ro?.disconnect();
    tickers.clear();
    clearModels();
    if (stage.arena) { stage.arena.dispose(); stage.arena = null; }
    composer?.dispose?.();
    envTex.dispose();
    renderer.dispose();
  };

  stage.resize();
  stage.setEnvironment(opts.env || 'studio');
  frame();
  return stage;
}
