// Environment scenery, lighting, ambient particles and hazard visuals.
//
// buildArena(envId, { renderer }) -> arena {
//   id, env, look, group, fighterY, showcaseY,
//   apply(scene), update(dt, t), dispose(),
//   timeScale,                              // set by playback (speed, 0 when paused) for transient effects
//   flash(color, strength, ms), groundImpact(pos, power), crumbleNear(pos, radius) -> count,
//   lightning(targetPos), hazard(name, type, targetPositions) -> ms until impact (at speed 1)
// }

import * as THREE from 'three';
import { ENV_BY_ID } from '../data/environments.js?v=6';

const V3 = THREE.Vector3;
const v = (x = 0, y = 0, z = 0) => new V3(x, y, z);
const PI = Math.PI;

const STUDIO = {
  id: 'studio', name: 'Studio',
  look: { sky: '#151a26', fog: '#1d2333', ground: '#22262e', light: '#ffffff', props: 'studio', time: 'day' },
};

function mkRng(seed) {
  let a = seed >>> 0 || 1;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function col(c, f = '#888') { try { return new THREE.Color(c || f); } catch { return new THREE.Color(f); } }

let _softTex = null;
function softTexture() {
  if (_softTex) return _softTex;
  const cv = document.createElement('canvas'); cv.width = cv.height = 64;
  const g = cv.getContext('2d');
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, 'rgba(255,255,255,1)'); grd.addColorStop(0.4, 'rgba(255,255,255,0.55)'); grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd; g.fillRect(0, 0, 64, 64);
  _softTex = new THREE.CanvasTexture(cv);
  _softTex.userData.shared = true;
  return _softTex;
}
export { softTexture };

function noiseTexture(base, vary = 0.12, size = 256, seed = 7, streaks = false) {
  const cv = document.createElement('canvas'); cv.width = cv.height = size;
  const g = cv.getContext('2d');
  const c = col(base);
  const img = g.createImageData(size, size);
  const rng = mkRng(seed);
  // value noise at two octaves
  const grid = (n) => { const a = []; for (let i = 0; i < n * n; i++) a.push(rng()); return a; };
  const g1 = grid(16), g2 = grid(64);
  const samp = (gr, n, x, y) => {
    const fx = x * n / size, fy = y * n / size;
    const x0 = Math.floor(fx) % n, y0 = Math.floor(fy) % n, x1 = (x0 + 1) % n, y1 = (y0 + 1) % n;
    const tx = fx - Math.floor(fx), ty = fy - Math.floor(fy);
    const sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty);
    const a = gr[y0 * n + x0], b = gr[y0 * n + x1], cc = gr[y1 * n + x0], d = gr[y1 * n + x1];
    return (a + (b - a) * sx) * (1 - sy) + (cc + (d - cc) * sx) * sy;
  };
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    let n = samp(g1, 16, x, y) * 0.6 + samp(g2, 64, x, y) * 0.4;
    if (streaks) n = n * 0.6 + 0.4 * (0.5 + 0.5 * Math.sin((x + n * 40) * 0.12));
    const k = 1 + (n - 0.5) * 2 * vary;
    const i = (y * size + x) * 4;
    img.data[i] = Math.min(255, c.r * 255 * k); img.data[i + 1] = Math.min(255, c.g * 255 * k); img.data[i + 2] = Math.min(255, c.b * 255 * k); img.data[i + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(cv);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function skyDome(top, horizon, bottom, stars) {
  const mat = new THREE.ShaderMaterial({
    uniforms: { top: { value: col(top) }, hor: { value: col(horizon) }, bot: { value: col(bottom) }, uFlash: { value: 0 } },
    vertexShader: `varying vec3 vW; void main(){ vW = normalize(position); gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
    fragmentShader: `uniform vec3 top; uniform vec3 hor; uniform vec3 bot; uniform float uFlash; varying vec3 vW;
      void main(){ float h = vW.y; vec3 c = h > 0.0 ? mix(hor, top, pow(clamp(h*1.6,0.0,1.0), 0.7)) : mix(hor, bot, clamp(-h*4.0,0.0,1.0));
        gl_FragColor = vec4(c + vec3(0.75,0.8,1.0)*uFlash, 1.0); }`,
    side: THREE.BackSide, depthWrite: false, fog: false,
  });
  const g = new THREE.Group();
  const dome = new THREE.Mesh(new THREE.SphereGeometry(500, 32, 16), mat);
  dome.renderOrder = -10;
  g.add(dome);
  g.userData.mat = mat;
  g.userData.base = { top: col(top), hor: col(horizon), bot: col(bottom) };
  if (stars) {
    const rng = mkRng(99), pos = [];
    for (let i = 0; i < 900; i++) {
      const u = rng() * PI * 2, y = 0.08 + rng() * 0.92, r = Math.sqrt(1 - y * y);
      pos.push(Math.cos(u) * r * 480, y * 480, Math.sin(u) * r * 480);
    }
    const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({ color: '#ffffff', size: 1.6, sizeAttenuation: false, fog: false, transparent: true, opacity: 0.8 }));
    g.add(pts);
  }
  return g;
}

// Soft particles living in a box around the arena (snow, sand, rain-ish, embers, smoke)
function particleField({ count, area = 120, height = 60, color, size = 0.5, opacity = 0.8, additive = false, seed = 3, zMax = null }) {
  const rng = mkRng(seed);
  const pos = new Float32Array(count * 3), vel = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    pos[i * 3] = (rng() - 0.5) * area; pos[i * 3 + 1] = rng() * height;
    pos[i * 3 + 2] = zMax == null ? (rng() - 0.5) * area : -area / 2 + rng() * (zMax + area / 2);
    vel[i * 3] = rng(); vel[i * 3 + 1] = rng(); vel[i * 3 + 2] = rng();
  }
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({ color, size, map: softTexture(), transparent: true, opacity, depthWrite: false, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending });
  const pts = new THREE.Points(geo, mat);
  pts.frustumCulled = false;
  pts.userData = { area, height, vel };
  return pts;
}

// Water with vertex-shader waves on a MeshStandardMaterial.
function waterMesh(color, amp, size = 700, seg = 160, opacity = 0.9) {
  const geo = new THREE.PlaneGeometry(size, size, seg, seg); geo.rotateX(-PI / 2);
  const uTime = { value: 0 }, uAmp = { value: amp };
  const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.12, metalness: 0.15, transparent: opacity < 1, opacity });
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uTime = uTime; sh.uniforms.uAmp = uAmp;
    sh.vertexShader = 'uniform float uTime; uniform float uAmp;\n' + sh.vertexShader
      .replace('#include <beginnormal_vertex>', `
        vec2 wp = position.xz;
        float w1 = sin(dot(wp, vec2(0.12, 0.05)) + uTime*1.1);
        float w2 = sin(dot(wp, vec2(-0.07, 0.11)) + uTime*0.8);
        float w3 = sin(dot(wp, vec2(0.21, -0.17)) + uTime*1.9);
        float kbH = uAmp*(0.55*w1 + 0.35*w2 + 0.15*w3);
        float c1 = cos(dot(wp, vec2(0.12, 0.05)) + uTime*1.1);
        float c2 = cos(dot(wp, vec2(-0.07, 0.11)) + uTime*0.8);
        float c3 = cos(dot(wp, vec2(0.21, -0.17)) + uTime*1.9);
        float dx = uAmp*(0.55*c1*0.12 + 0.35*c2*-0.07 + 0.15*c3*0.21);
        float dz = uAmp*(0.55*c1*0.05 + 0.35*c2*0.11 + 0.15*c3*-0.17);
        vec3 objectNormal = normalize(vec3(-dx, 1.0, -dz));
        #ifdef USE_TANGENT
          vec3 objectTangent = vec3(1.0, 0.0, 0.0);
        #endif`)
      .replace('#include <begin_vertex>', 'vec3 transformed = vec3(position.x, position.y + kbH, position.z);');
  };
  const mesh = new THREE.Mesh(geo, mat);
  mesh.receiveShadow = true;
  mesh.userData.uTime = uTime; mesh.userData.uAmp = uAmp;
  return mesh;
}

function windowTexture(seed) {
  const cv = document.createElement('canvas'); cv.width = 64; cv.height = 128;
  const g = cv.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, 64, 128);
  const rng = mkRng(seed);
  for (let y = 4; y < 124; y += 8) for (let x = 4; x < 60; x += 8) {
    if (rng() < 0.42) { const w = rng(); g.fillStyle = w < 0.7 ? '#ffd890' : w < 0.9 ? '#bfe6ff' : '#ff9ad0'; g.fillRect(x, y, 5, 4); }
  }
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping; t.magFilter = THREE.NearestFilter;
  return t;
}

function lavaTexture() {
  const cv = document.createElement('canvas'); cv.width = cv.height = 256;
  const g = cv.getContext('2d');
  g.fillStyle = '#ff5a00'; g.fillRect(0, 0, 256, 256);
  const rng = mkRng(11);
  for (let i = 0; i < 70; i++) {
    const x = rng() * 256, y = rng() * 256, r = 6 + rng() * 26;
    const grd = g.createRadialGradient(x, y, 0, x, y, r);
    const hot = rng() < 0.5;
    grd.addColorStop(0, hot ? 'rgba(255,230,120,0.95)' : 'rgba(60,10,0,0.9)'); grd.addColorStop(1, hot ? 'rgba(255,160,40,0)' : 'rgba(80,10,0,0)');
    g.fillStyle = grd;
    for (const ox of [-256, 0, 256]) for (const oy of [-256, 0, 256]) { g.save(); g.translate(ox, oy); g.beginPath(); g.arc(x, y, r, 0, PI * 2); g.fill(); g.restore(); }
  }
  const t = new THREE.CanvasTexture(cv); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// ribbon strip along a curve lying on the ground
function ribbonGeo(curve, width, segs = 80) {
  const pos = [], uv = [], idx = [];
  const P = new V3(), T = new V3(), Nn = new V3();
  const L = curve.getLength();
  for (let i = 0; i <= segs; i++) {
    const u = i / segs;
    curve.getPointAt(u, P); curve.getTangentAt(u, T);
    Nn.set(-T.z, 0, T.x).normalize();
    const w = width * (0.75 + 0.25 * Math.sin(u * 17));
    pos.push(P.x + Nn.x * w, P.y, P.z + Nn.z * w, P.x - Nn.x * w, P.y, P.z - Nn.z * w);
    uv.push(0, u * L / 10, 1, u * L / 10);
    if (i < segs) { const a = i * 2; idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx); g.computeVertexNormals();
  return g;
}

function displacedRock(r, seed, sx = 1, sy = 1, sz = 1, detail = 1) {
  const geo = new THREE.IcosahedronGeometry(1, detail);
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    const k = Math.sin(x * 12.9898 * seed + y * 78.233 + z * 37.719) * 43758.5453;
    const f = 0.8 + (k - Math.floor(k)) * 0.4;
    p.setXYZ(i, x * f * r * sx, y * f * r * sy, z * f * r * sz);
  }
  geo.computeVertexNormals();
  return geo;
}

// ---------------------------------------------------------------------------------------------
export function buildArena(envId, opts = {}) {
  const env = ENV_BY_ID[envId] || (envId === 'studio' || !envId ? STUDIO : { ...STUDIO, id: envId });
  const look = { ...STUDIO.look, ...(env.look || {}) };
  const group = new THREE.Group(); group.name = 'arena:' + env.id;
  const rng = mkRng(1234);
  const night = look.time === 'night', dusk = look.time === 'dusk';
  const skyC = col(look.sky), fogC = col(look.fog), groundC = col(look.ground), lightC = col(look.light, '#ffffff');
  const disposables = [];
  const updaters = [];
  const transients = []; // {update(dtScaled) -> bool alive, obj}
  const arena = {
    id: env.id, env, look, group, fighterY: 0, showcaseY: 0, timeScale: 1,
    fog: null, background: fogC.clone(),
  };

  // ---- lights
  const hemi = new THREE.HemisphereLight(skyC.clone().lerp(new THREE.Color('#ffffff'), night ? 0.25 : 0.55), groundC, night ? 0.9 : 1.25);
  const sun = new THREE.DirectionalLight(lightC, night ? 1.5 : dusk ? 2.2 : 2.8);
  sun.position.set(night ? -40 : 45, 70, night ? 30 : 40);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const sc = sun.shadow.camera; sc.left = -48; sc.right = 48; sc.top = 48; sc.bottom = -48; sc.near = 5; sc.far = 220;
  sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.04;
  sun.target.position.set(0, 0, 0);
  group.add(hemi, sun, sun.target);
  if (look.props === 'ice') { hemi.intensity *= 0.55; sun.intensity *= 0.6; }
  if (look.props === 'dunes') { hemi.intensity *= 0.85; }
  if (look.props === 'lunar') { hemi.intensity *= 0.45; sun.intensity *= 1.35; }   // airless: hard light, dark shadows
  if (look.props === 'jungle') { hemi.intensity *= 0.9; sun.intensity *= 0.75; }
  const baseHemi = hemi.intensity, baseSun = sun.intensity;

  // ---- sky
  const skyTop = skyC.clone().multiplyScalar(night ? 0.55 : 0.85);
  const sky = skyDome(skyTop, fogC, groundC.clone().lerp(fogC, 0.5), night);
  group.add(sky);
  arena.fog = new THREE.Fog(fogC, look.props === 'dunes' ? 50 : look.props === 'jungle' ? 45 : look.props === 'lunar' ? 140 : 80, look.props === 'jungle' ? 230 : look.props === 'lunar' ? 520 : 300);
  const baseFog = { near: arena.fog.near, far: arena.fog.far, color: fogC.clone() };

  // ---- ground
  const groundTex = noiseTexture(look.ground, look.props === 'ice' ? 0.06 : look.props === 'lunar' ? 0.1 : 0.16, 256, 5, look.props === 'dunes');
  groundTex.repeat.set(40, 40);
  disposables.push(groundTex);
  const groundMat = new THREE.MeshStandardMaterial({ color: look.props === 'ice' ? '#b8c4d0' : '#ffffff', map: groundTex, roughness: look.props === 'ice' ? 0.35 : 0.95, metalness: 0 });
  let ground;
  if (look.props === 'dunes') {
    const geo = new THREE.PlaneGeometry(700, 700, 140, 140); geo.rotateX(-PI / 2);
    const p = geo.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), z = p.getZ(i);
      const d = Math.sqrt(x * x + z * z);
      const fade = THREE.MathUtils.smoothstep(d, 28, 70);
      const h = (Math.sin(x * 0.045 + Math.sin(z * 0.02) * 2) * 4 + Math.sin(z * 0.07 + x * 0.01) * 2.5 + 3) * fade;
      p.setY(i, h);
    }
    geo.computeVertexNormals();
    ground = new THREE.Mesh(geo, groundMat);
  } else if (look.props === 'lunar') {
    // cratered regolith: bowl-shaped dents with raised rims, kept flat where the fighters stand
    const geo = new THREE.PlaneGeometry(700, 700, 180, 180); geo.rotateX(-PI / 2);
    const p = geo.attributes.position, cr = mkRng(99), craters = [];
    for (let i = 0; i < 46; i++) { const a = cr() * PI * 2, r = 34 + cr() * 260; craters.push([Math.cos(a) * r, Math.sin(a) * r, 5 + cr() * (r > 120 ? 34 : 14)]); }
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), z = p.getZ(i); let h = 0;
      for (const [cx, cz, cr2] of craters) {
        const d = Math.hypot(x - cx, z - cz) / cr2;
        if (d < 1) h -= (1 - d * d) * cr2 * 0.22; else if (d < 1.45) h += Math.sin((d - 1) / 0.45 * PI) * cr2 * 0.07;
      }
      p.setY(i, h * THREE.MathUtils.smoothstep(Math.hypot(x, z), 24, 46));
    }
    geo.computeVertexNormals();
    ground = new THREE.Mesh(geo, groundMat);
  } else {
    ground = new THREE.Mesh(new THREE.PlaneGeometry(700, 700).rotateX(-PI / 2), groundMat);
  }
  ground.receiveShadow = true;
  group.add(ground);

  // ---- shared particle burst system (impacts, dust, splashes)
  const BURST_MAX = 600;
  const bPos = new Float32Array(BURST_MAX * 3), bVel = new Float32Array(BURST_MAX * 3), bLife = new Float32Array(BURST_MAX);
  const bCol = new Float32Array(BURST_MAX * 3);
  for (let i = 0; i < BURST_MAX; i++) bPos[i * 3 + 1] = -999;
  const bGeo = new THREE.BufferGeometry();
  bGeo.setAttribute('position', new THREE.BufferAttribute(bPos, 3));
  bGeo.setAttribute('color', new THREE.BufferAttribute(bCol, 3));
  const bMat = new THREE.PointsMaterial({ size: 2.2, map: softTexture(), vertexColors: true, transparent: true, opacity: 0.85, depthWrite: false });
  const bursts = new THREE.Points(bGeo, bMat); bursts.frustumCulled = false;
  group.add(bursts);
  let bNext = 0;
  const impactColor = { buildings: '#9a9488', water: '#e8f4ff', storm: '#c8d8e0', lava: '#ff7a2a', ice: '#ffffff', dunes: '#d8b078', studio: '#9aa0aa', jungle: '#5a6a30', lunar: '#b8b8be' }[look.props] || '#aaaaaa';
  arena.groundImpact = (pos, power = 1, color) => {
    const c = col(color || impactColor);
    const n = Math.floor(40 * power);
    for (let k = 0; k < n; k++) {
      const i = bNext; bNext = (bNext + 1) % BURST_MAX;
      const a = Math.random() * PI * 2, sp = (4 + Math.random() * 10) * power;
      bPos[i * 3] = pos.x + Math.cos(a) * 2; bPos[i * 3 + 1] = Math.max(0.3, arena.fighterY + 0.4); bPos[i * 3 + 2] = pos.z + Math.sin(a) * 2;
      bVel[i * 3] = Math.cos(a) * sp; bVel[i * 3 + 1] = 4 + Math.random() * 10 * power; bVel[i * 3 + 2] = Math.sin(a) * sp;
      bLife[i] = 0.9 + Math.random() * 0.8;
      const sh = 0.8 + Math.random() * 0.4;
      bCol[i * 3] = c.r * sh; bCol[i * 3 + 1] = c.g * sh; bCol[i * 3 + 2] = c.b * sh;
    }
    bGeo.attributes.color.needsUpdate = true;
  };
  updaters.push((dt, t, sdt) => {
    let any = false;
    for (let i = 0; i < BURST_MAX; i++) {
      if (bLife[i] <= 0) continue;
      any = true;
      bLife[i] -= sdt;
      if (bLife[i] <= 0) { bPos[i * 3 + 1] = -999; continue; }
      bVel[i * 3 + 1] -= (look.props === 'lunar' ? 5 : 16) * sdt;
      bVel[i * 3] *= 1 - sdt * 1.5; bVel[i * 3 + 2] *= 1 - sdt * 1.5;
      bPos[i * 3] += bVel[i * 3] * sdt; bPos[i * 3 + 1] += bVel[i * 3 + 1] * sdt; bPos[i * 3 + 2] += bVel[i * 3 + 2] * sdt;
      if (bPos[i * 3 + 1] < arena.fighterY) { bPos[i * 3 + 1] = arena.fighterY; bVel[i * 3 + 1] *= -0.2; }
    }
    if (any) bGeo.attributes.position.needsUpdate = true;
  });

  // ---- flash (light + sky)
  let flashAmt = 0, flashCol = new THREE.Color('#ffffff'), flashDecay = 3;
  arena.flash = (color = '#ffffff', strength = 1, ms = 350) => { flashAmt = Math.max(flashAmt, strength); flashCol.set(color); flashDecay = 1000 / Math.max(60, ms); };
  updaters.push((dt, t, sdt) => {
    flashAmt = Math.max(0, flashAmt - sdt * flashDecay);
    hemi.intensity = baseHemi * (1 + flashAmt * 1.6);
    hemi.color.copy(skyC).lerp(new THREE.Color('#ffffff'), night ? 0.25 : 0.55).lerp(flashCol, Math.min(1, flashAmt));
    sky.userData.mat.uniforms.uFlash.value = flashAmt * 0.35;
  });

  // ---- weather surge (whiteout / sandstorm): fog closes in
  let surge = 0, surgeTarget = 0, surgeCol = fogC.clone();
  const surgeTo = (amt, color) => { surgeTarget = amt; if (color) surgeCol.set(color); };
  updaters.push((dt, t, sdt) => {
    surge += (surgeTarget - surge) * Math.min(1, sdt * 2.5);
    if (surgeTarget > 0) surgeTarget = Math.max(0, surgeTarget - sdt * 0.35);
    arena.fog.near = baseFog.near * (1 - surge * 0.95);
    arena.fog.far = baseFog.far * (1 - surge * 0.8);
    arena.fog.color.copy(baseFog.color).lerp(surgeCol, surge);
    const su = sky.userData.mat.uniforms, sb = sky.userData.base;
    su.top.value.copy(sb.top).lerp(surgeCol, surge); su.hor.value.copy(sb.hor).lerp(surgeCol, surge); su.bot.value.copy(sb.bot).lerp(surgeCol, surge);
    if (arena._scene && arena._scene.background && arena._scene.background.isColor) arena._scene.background.copy(arena.fog.color);
  });
  arena.surge = surgeTo;

  // ---- transient effects runner
  const addTransient = (obj, fn) => { if (obj) group.add(obj); transients.push({ obj, fn, t: 0 }); };
  updaters.push((dt, t, sdt) => {
    for (let i = transients.length - 1; i >= 0; i--) {
      const tr = transients[i];
      tr.t += sdt;
      if (tr.fn(tr.t, sdt) === false) {
        if (tr.obj) { group.remove(tr.obj); tr.obj.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material && !o.material.userData?.shared) o.material.dispose(); }); }
        transients.splice(i, 1);
      }
    }
  });

  // ---- lightning bolt
  const boltMat = new THREE.MeshBasicMaterial({ color: '#e6efff', transparent: true, opacity: 1, blending: THREE.AdditiveBlending, depthWrite: false, fog: false });
  const boltGlowMat = new THREE.MeshBasicMaterial({ color: '#7f9cff', transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false, fog: false });
  boltMat.userData.shared = boltGlowMat.userData.shared = true;
  disposables.push(boltMat, boltGlowMat);
  arena.lightning = (to, from) => {
    to = to.clone();
    from = from || to.clone().add(v((Math.random() - 0.5) * 30, 90, (Math.random() - 0.5) * 30 - 20));
    const path = new THREE.CurvePath();
    let prev = from.clone();
    const N = 12;
    for (let i = 1; i <= N; i++) {
      const p = from.clone().lerp(to, i / N);
      if (i < N) p.add(v((Math.random() - 0.5) * 7, 0, (Math.random() - 0.5) * 7));
      path.add(new THREE.LineCurve3(prev, p)); prev = p;
    }
    const g = new THREE.Group();
    g.add(new THREE.Mesh(new THREE.TubeGeometry(path, 60, 0.35, 5, false), boltMat));
    g.add(new THREE.Mesh(new THREE.TubeGeometry(path, 60, 1.4, 6, false), boltGlowMat));
    arena.flash('#cfe0ff', 1.3, 450);
    addTransient(g, (tt) => { g.visible = tt < 0.08 || (tt > 0.13 && tt < 0.3) || tt > 0.36; g.children[0].scale.setScalar(1); g.traverse(o => { if (o.material) o.material.opacity = Math.max(0, 1 - tt / 0.55) * (o.material === boltGlowMat ? 0.35 : 1); }); return tt < 0.55; });
    arena.groundImpact(to, 0.8, '#cfe0ff');
  };

  // ---- buildings (city): instanced, crumble-able
  let buildings = null;
  const crumbling = [];
  const debris = [];
  if (look.props === 'buildings') {
    const list = [];
    const tryAdd = (x, z, w, d, h) => list.push({ x, z, w, d, h, alive: true });
    // near buildings behind each fighter (knockback targets) + background skyline
    for (const s of [-1, 1]) for (let k = 0; k < 4; k++) tryAdd(s * (22 + k * 7 + rng() * 2), -10 + k * 4 - rng() * 6, 5 + rng() * 3, 5 + rng() * 3, 10 + rng() * 14);
    for (let i = 0; i < 260; i++) {
      const a = rng() * PI * 2, r = 30 + Math.pow(rng(), 0.7) * 180;
      const x = Math.cos(a) * r, z = Math.sin(a) * r;
      if (z > -14 && Math.abs(x) < 26 + Math.max(0, z) * 1.5) continue; // keep the camera side clear
      if (z > 10 && Math.abs(x) < 90) continue;
      tryAdd(x, z, 6 + rng() * 8, 6 + rng() * 8, 8 + Math.pow(rng(), 2) * (r > 80 ? 70 : 35));
    }
    const geo = new THREE.BoxGeometry(1, 1, 1); geo.translate(0, 0.5, 0);
    const wtex = windowTexture(5); wtex.repeat.set(1, 2);
    const mat = new THREE.MeshStandardMaterial({ color: '#56607a', roughness: 0.7, metalness: 0.25, emissive: '#ffffff', emissiveMap: wtex, emissiveIntensity: night ? 1.1 : 0.15, map: null });
    disposables.push(wtex);
    const im = new THREE.InstancedMesh(geo, mat, list.length);
    im.castShadow = true; im.receiveShadow = true;
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s3 = new V3(), p3 = new V3();
    const cc = new THREE.Color();
    list.forEach((b, i) => {
      m4.compose(p3.set(b.x, 0, b.z), q.identity(), s3.set(b.w, b.h, b.d));
      im.setMatrixAt(i, m4);
      cc.set('#4a5470').lerp(new THREE.Color(rng() < 0.5 ? '#6a6e78' : '#3a4560'), rng());
      im.setColorAt(i, cc);
    });
    group.add(im);
    buildings = { im, list };
    // roads
    const roadMat = new THREE.MeshStandardMaterial({ color: '#15161a', roughness: 0.9 });
    const lineMat = new THREE.MeshStandardMaterial({ color: '#d8c060', emissive: '#d8c060', emissiveIntensity: 0.3 });
    const road = new THREE.Mesh(new THREE.PlaneGeometry(700, 14).rotateX(-PI / 2), roadMat); road.position.set(0, 0.02, -4); road.receiveShadow = true; group.add(road);
    for (let x = -340; x < 340; x += 10) { const l = new THREE.Mesh(new THREE.PlaneGeometry(4, 0.4).rotateX(-PI / 2), lineMat); l.position.set(x, 0.04, -4); group.add(l); }
    // debris pool
    const dGeo = new THREE.BoxGeometry(1, 1, 1);
    const dIm = new THREE.InstancedMesh(dGeo, new THREE.MeshStandardMaterial({ color: '#5a6070', roughness: 0.9 }), 160);
    dIm.castShadow = true; dIm.count = 160;
    for (let i = 0; i < 160; i++) { m4.compose(p3.set(0, -999, 0), q.identity(), s3.set(1, 1, 1)); dIm.setMatrixAt(i, m4); debris.push({ p: v(0, -999, 0), vel: v(), rot: v(), life: 0, s: 1 }); }
    group.add(dIm);
    buildings.debris = dIm;
    let dNext = 0;
    arena.crumbleNear = (pos, radius = 10) => {
      let n = 0;
      list.forEach((b, i) => {
        if (!b.alive) return;
        const dx = b.x - pos.x, dz = b.z - pos.z;
        if (dx * dx + dz * dz > radius * radius) return;
        b.alive = false; n++;
        crumbling.push({ i, b, t: 0, tilt: (Math.random() - 0.5) * 0.6, tiltZ: (Math.random() - 0.5) * 0.6 });
        for (let k = 0; k < 18; k++) {
          const d = debris[dNext]; dNext = (dNext + 1) % debris.length;
          d.p.set(b.x + (Math.random() - 0.5) * b.w, b.h * (0.3 + Math.random() * 0.7), b.z + (Math.random() - 0.5) * b.d);
          d.vel.set((Math.random() - 0.5) * 14, 4 + Math.random() * 10, (Math.random() - 0.5) * 14);
          d.rot.set(Math.random() * 3, Math.random() * 3, Math.random() * 3);
          d.life = 2.5; d.s = 0.6 + Math.random() * 1.6;
        }
        arena.groundImpact(v(b.x, 0, b.z), 1.2, '#8a8478');
      });
      return n;
    };
    updaters.push((dt, t, sdt) => {
      for (let k = crumbling.length - 1; k >= 0; k--) {
        const c = crumbling[k]; c.t += sdt;
        const f = Math.min(1, c.t / 1.6), e = f * f;
        m4.compose(p3.set(c.b.x, -c.b.h * 0.9 * e, c.b.z), q.setFromEuler(new THREE.Euler(c.tilt * e, 0, c.tiltZ * e)), s3.set(c.b.w, c.b.h * (1 - 0.35 * e), c.b.d));
        im.setMatrixAt(c.i, m4);
        im.instanceMatrix.needsUpdate = true;
        if (f >= 1) crumbling.splice(k, 1);
      }
      let any = false;
      debris.forEach((d, i) => {
        if (d.life <= 0) return;
        any = true; d.life -= sdt;
        d.vel.y -= 22 * sdt; d.p.addScaledVector(d.vel, sdt);
        if (d.p.y < d.s * 0.4) { d.p.y = d.s * 0.4; d.vel.multiplyScalar(0.4); d.vel.y = Math.abs(d.vel.y) * 0.3; }
        d.rot.x += sdt * 3; d.rot.z += sdt * 2;
        m4.compose(d.life > 0 ? d.p : p3.set(0, -999, 0), q.setFromEuler(new THREE.Euler(d.rot.x, d.rot.y, d.rot.z)), s3.setScalar(d.s));
        dIm.setMatrixAt(i, m4);
      });
      if (any) dIm.instanceMatrix.needsUpdate = true;
    });
    // moon
    const moon = new THREE.Mesh(new THREE.SphereGeometry(9, 20, 12), new THREE.MeshBasicMaterial({ color: '#fff4d8', fog: false }));
    moon.position.set(-160, 170, -380); group.add(moon);
  } else {
    arena.crumbleNear = () => 0;
  }

  // ---- water (ocean) / dark sea (storm)
  let water = null;
  if (look.props === 'water' || look.props === 'storm') {
    const storm = look.props === 'storm';
    ground.position.y = -6;
    water = waterMesh(storm ? col('#1c2e34') : col(look.ground).lerp(new THREE.Color('#2a7a9a'), 0.35), storm ? 1.1 : 0.55, 700, 160, storm ? 0.95 : 0.86);
    group.add(water);
    arena.fighterY = -2.4; arena.showcaseY = -1.2;
    updaters.push((dt, t) => { water.userData.uTime.value = t * (storm ? 1.5 : 1); });
    // sea stacks / distant rocks
    const rockMat = new THREE.MeshStandardMaterial({ color: storm ? '#2a3030' : '#4a5a50', roughness: 0.9, flatShading: true });
    for (let i = 0; i < 14; i++) {
      const a = rng() * PI * 2, r = 90 + rng() * 160;
      const x = Math.cos(a) * r, z = Math.sin(a) * r;
      if (z > 0 && Math.abs(x) < 120) continue;
      const h = 15 + rng() * 40;
      const m = new THREE.Mesh(displacedRock(1, i + 1, 8 + rng() * 10, h / 1.0, 8 + rng() * 10), rockMat);
      m.position.set(x, -4, z); m.castShadow = true; group.add(m);
    }
    if (!storm) {
      // sun disc + foam specks
      const sunDisc = new THREE.Mesh(new THREE.SphereGeometry(14, 16, 10), new THREE.MeshBasicMaterial({ color: '#fffbe8', fog: false }));
      sunDisc.position.set(180, 220, -360); group.add(sunDisc);
      const foam = particleField({ count: 500, area: 220, height: 0.1, color: '#ffffff', size: 0.9, opacity: 0.6, seed: 8 });
      foam.position.y = 0.4; group.add(foam);
      updaters.push((dt, t) => { foam.position.y = 0.35 + Math.sin(t * 1.1) * 0.25; });
    }
  }

  // ---- storm: rain + random lightning
  if (look.props === 'storm') {
    const N = 2400, area = 160, H = 80;
    const pos = new Float32Array(N * 6);
    const r2 = mkRng(42);
    for (let i = 0; i < N; i++) {
      const x = (r2() - 0.5) * area, y = r2() * H, z = -area / 2 + r2() * (area / 2 + 14);
      pos.set([x, y, z, x - 0.4, y - 2.2, z], i * 6);
    }
    const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const rain = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: '#9ab0c8', transparent: true, opacity: 0.45, depthWrite: false }));
    rain.frustumCulled = false;
    group.add(rain);
    updaters.push((dt) => {
      const p = geo.attributes.position.array;
      for (let i = 0; i < N; i++) {
        const o = i * 6;
        let y = p[o + 1] - dt * 70;
        let x = p[o] - dt * 12;
        if (y < -3) { y += H; }
        if (x < -area / 2) x += area;
        p[o] = x; p[o + 1] = y; p[o + 3] = x - 0.4; p[o + 4] = y - 2.2;
      }
      geo.attributes.position.needsUpdate = true;
    });
    let nextBolt = 2 + rng() * 4;
    updaters.push((dt, t) => {
      nextBolt -= dt;
      if (nextBolt <= 0) {
        nextBolt = 4 + Math.random() * 7;
        const a = Math.random() * PI * 2, r = 80 + Math.random() * 120;
        const far = v(Math.cos(a) * r, -2, -Math.abs(Math.sin(a) * r) - 40);
        arena.lightning(far);
      }
    });
  }

  // ---- lava
  let volcanoes = [];
  if (look.props === 'lava') {
    const lt = lavaTexture(); disposables.push(lt);
    const lavaMat = new THREE.MeshStandardMaterial({ color: '#200800', emissive: '#ffffff', emissiveMap: lt, emissiveIntensity: 2.2, roughness: 0.6 });
    const rivers = [];
    const mkRiver = (pts, w) => {
      const curve = new THREE.CatmullRomCurve3(pts.map(p => v(p[0], 0.06, p[1])));
      const m = new THREE.Mesh(ribbonGeo(curve, w, 120), lavaMat); m.receiveShadow = false;
      group.add(m); rivers.push(curve);
    };
    mkRiver([[-200, -60], [-90, -40], [-40, -22], [0, -26], [50, -18], [110, -50], [220, -70]], 3.5);
    mkRiver([[-60, 200], [-42, 90], [-38, 40], [-30, 10], [-40, -24]], 2.2);
    mkRiver([[150, 180], [70, 60], [55, 20], [48, -18]], 2.0);
    updaters.push((dt, t) => { lt.offset.y = -t * 0.08; lt.offset.x = Math.sin(t * 0.3) * 0.05; lavaMat.emissiveIntensity = 2.0 + Math.sin(t * 2.1) * 0.25; });
    // rocks scattered
    const rockMat = new THREE.MeshStandardMaterial({ color: '#241a16', roughness: 0.95, flatShading: true });
    for (let i = 0; i < 40; i++) {
      const a = rng() * PI * 2, r = 24 + rng() * 120;
      const x = Math.cos(a) * r, z = Math.sin(a) * r;
      if (z > 8 && Math.abs(x) < 60) continue;
      const m = new THREE.Mesh(displacedRock(1 + rng() * 3, i + 3, 1, 0.7, 1), rockMat);
      m.position.set(x, 0.3, z); m.castShadow = true; m.receiveShadow = true; group.add(m);
    }
    // volcano cones
    const coneMat = new THREE.MeshStandardMaterial({ color: '#2a1c18', roughness: 0.95, flatShading: true });
    const craterMat = new THREE.MeshStandardMaterial({ color: '#300800', emissive: '#ff5a10', emissiveIntensity: 2.5 });
    for (const [x, z, h, r] of [[-70, -170, 90, 70], [90, -220, 120, 90], [200, -120, 60, 55], [-200, -90, 70, 60]]) {
      const geo = new THREE.CylinderGeometry(r * 0.16, r, h, 28, 6, true);
      const p = geo.attributes.position;
      for (let i = 0; i < p.count; i++) { const k = Math.sin(p.getX(i) * 0.3 + p.getY(i) * 0.2) * Math.cos(p.getZ(i) * 0.25); p.setX(i, p.getX(i) * (1 + k * 0.08)); p.setZ(i, p.getZ(i) * (1 + k * 0.08)); }
      geo.computeVertexNormals();
      const cone = new THREE.Mesh(geo, coneMat); cone.position.set(x, h / 2 - 1, z); group.add(cone);
      const crater = new THREE.Mesh(new THREE.CircleGeometry(r * 0.16, 20).rotateX(-PI / 2), craterMat); crater.position.set(x, h - 1.2, z); group.add(crater);
      volcanoes.push({ top: v(x, h, z), r, h });
    }
    // smoke + embers
    const smoke = particleField({ count: 260, area: 260, height: 110, color: '#3a2a26', size: 26, opacity: 0.38, seed: 12 });
    smoke.position.set(0, 0, -150); group.add(smoke);
    const embers = particleField({ count: 500, area: 120, height: 50, color: '#ff8a3a', size: 0.55, opacity: 0.95, additive: true, seed: 13, zMax: 12 });
    group.add(embers);
    updaters.push((dt, t) => {
      for (const [pts, rise, drift, H] of [[smoke, 5, 3, 110], [embers, 3.5, 1.5, 50]]) {
        const p = pts.geometry.attributes.position.array, vel = pts.userData.vel, A = pts.userData.area;
        for (let i = 0; i < p.length / 3; i++) {
          p[i * 3 + 1] += (rise * (0.5 + vel[i * 3 + 1])) * dt;
          p[i * 3] += Math.sin(t * 0.5 + i) * drift * dt;
          if (p[i * 3 + 1] > H) { p[i * 3 + 1] = 0; }
        }
        pts.geometry.attributes.position.needsUpdate = true;
      }
    });
  }

  // ---- ice
  let snow = null;
  if (look.props === 'ice') {
    const iceMat = new THREE.MeshStandardMaterial({ color: '#cfe8f6', roughness: 0.18, metalness: 0.05, flatShading: true });
    const floeGeo = new THREE.CylinderGeometry(1, 1.08, 1, 7, 1); floeGeo.translate(0, 0.5, 0);
    const N = 70;
    const floes = new THREE.InstancedMesh(floeGeo, iceMat, N);
    floes.castShadow = true; floes.receiveShadow = true;
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s3 = new V3(), p3 = new V3();
    let k = 0;
    for (let i = 0; i < 400 && k < N; i++) {
      const a = rng() * PI * 2, r = 26 + rng() * 150;
      const x = Math.cos(a) * r, z = Math.sin(a) * r;
      if (z > 6 && Math.abs(x) < 50) continue;
      const w = 2 + rng() * 6;
      m4.compose(p3.set(x, -0.3, z), q.setFromEuler(new THREE.Euler((rng() - 0.5) * 0.15, rng() * 6, (rng() - 0.5) * 0.15)), s3.set(w, 0.6 + rng() * 1.6, w * (0.6 + rng() * 0.6)));
      floes.setMatrixAt(k++, m4);
    }
    floes.count = k;
    group.add(floes);
    const bergMat = new THREE.MeshStandardMaterial({ color: '#e6f4ff', roughness: 0.3, flatShading: true });
    for (let i = 0; i < 9; i++) {
      const a = PI + (rng() - 0.5) * PI * 0.9, r = 120 + rng() * 120;
      const m = new THREE.Mesh(displacedRock(1, i + 9, 14 + rng() * 14, 18 + rng() * 34, 12 + rng() * 10), bergMat);
      m.position.set(Math.cos(a) * r * 1.2, 0, Math.sin(a) * r); m.castShadow = true; group.add(m);
    }
    // ice cracks: thin dark lines on the ground
    const crackMat = new THREE.MeshBasicMaterial({ color: '#8fb6cc' });
    for (let i = 0; i < 14; i++) {
      const pts = []; let x = (rng() - 0.5) * 100, z = -rng() * 60;
      for (let j = 0; j < 6; j++) { pts.push(v(x, 0.03, z)); x += (rng() - 0.5) * 14; z += (rng() - 0.5) * 14; }
      const m = new THREE.Mesh(ribbonGeo(new THREE.CatmullRomCurve3(pts), 0.12, 30), crackMat); group.add(m);
    }
    snow = particleField({ count: 1800, area: 140, height: 60, color: '#ffffff', size: 0.45, opacity: 0.85, seed: 21, zMax: 12 });
    group.add(snow);
    updaters.push((dt, t) => {
      const p = snow.geometry.attributes.position.array, vel = snow.userData.vel;
      const wind = 2 + surge * 40, fall = 3 + surge * 8;
      for (let i = 0; i < p.length / 3; i++) {
        p[i * 3 + 1] -= fall * (0.6 + vel[i * 3 + 1] * 0.6) * dt;
        p[i * 3] += (wind + Math.sin(t + i) * 1.5) * dt;
        if (p[i * 3 + 1] < 0) p[i * 3 + 1] += 60;
        if (p[i * 3] > 70) p[i * 3] -= 140;
      }
      snow.material.size = 0.5 + surge * 0.5;
      snow.geometry.attributes.position.needsUpdate = true;
    });
  }

  // ---- dunes
  let sand = null;
  if (look.props === 'dunes') {
    const mesaMat = new THREE.MeshStandardMaterial({ color: '#a8643e', roughness: 0.95, flatShading: true });
    for (let i = 0; i < 8; i++) {
      const a = PI + (rng() - 0.5) * PI * 1.1, r = 150 + rng() * 140;
      const h = 20 + rng() * 35, w = 20 + rng() * 30;
      const geo = new THREE.CylinderGeometry(w * 0.85, w, h, 9, 3);
      const p = geo.attributes.position;
      for (let j = 0; j < p.count; j++) { const k = Math.sin(j * 1.7 + i) * 0.08; p.setX(j, p.getX(j) * (1 + k)); p.setZ(j, p.getZ(j) * (1 - k)); }
      geo.computeVertexNormals();
      const m = new THREE.Mesh(geo, mesaMat); m.position.set(Math.cos(a) * r, h / 2 - 2, Math.sin(a) * r); m.castShadow = true; group.add(m);
    }
    const boneMat = new THREE.MeshStandardMaterial({ color: '#e8dcc0', roughness: 0.6 });
    for (let i = 0; i < 6; i++) {
      const rib = new THREE.Mesh(new THREE.TorusGeometry(5, 0.45, 6, 18, PI * 0.9), boneMat);
      rib.position.set(-48 + i * 3.2, 0, -36); rib.rotation.set(0, PI / 2, 0.1); rib.castShadow = true; group.add(rib);
    }
    sand = particleField({ count: 1600, area: 160, height: 20, color: '#e0b878', size: 0.45, opacity: 0.6, seed: 31, zMax: 12 });
    group.add(sand);
    updaters.push((dt, t) => {
      const p = sand.geometry.attributes.position.array, vel = sand.userData.vel;
      const wind = 10 + surge * 60;
      for (let i = 0; i < p.length / 3; i++) {
        p[i * 3] += wind * (0.5 + vel[i * 3]) * dt;
        p[i * 3 + 1] += Math.sin(t * 2 + i) * dt * (1 + surge * 3);
        if (p[i * 3] > 80) p[i * 3] -= 160;
        if (p[i * 3 + 1] < 0 || p[i * 3 + 1] > 20 + surge * 30) p[i * 3 + 1] = Math.random() * (5 + surge * 25);
      }
      sand.material.size = 0.45 + surge * 1.4; sand.material.opacity = 0.6 + surge * 0.3;
      sand.geometry.attributes.position.needsUpdate = true;
    });
  }

  // ---- jungle: giant trees, hanging vines, ferns, drifting spores
  if (look.props === 'jungle') {
    const barkMat = new THREE.MeshStandardMaterial({ color: '#4a3826', roughness: 0.95, flatShading: true });
    const leafMat = new THREE.MeshStandardMaterial({ color: '#2f5a26', roughness: 0.9, flatShading: true });
    const leafMat2 = new THREE.MeshStandardMaterial({ color: '#4a7a2c', roughness: 0.9, flatShading: true });
    const vineMat = new THREE.MeshStandardMaterial({ color: '#3d5f22', roughness: 0.9 });
    const trees = [];
    for (let i = 0; i < 80 && trees.length < 34; i++) {
      const a = rng() * PI * 2, r = 36 + rng() * 170, x = Math.cos(a) * r, z = Math.sin(a) * r;
      if (z > 4 && Math.abs(x) < 60) continue;          // keep the camera side open
      trees.push([x, z, 0.7 + rng() * 0.8]);
    }
    for (const [x, z, k] of trees) {
      const h = 55 * k, tr = 2.6 * k;
      const trunk = new THREE.Mesh(new THREE.CylinderGeometry(tr * 0.7, tr * 1.25, h, 8, 4), barkMat);
      trunk.position.set(x, h / 2, z); trunk.rotation.z = (rng() - 0.5) * 0.08; trunk.castShadow = true; group.add(trunk);
      for (let j = 0; j < 4; j++) {   // buttress roots
        const root = new THREE.Mesh(new THREE.ConeGeometry(tr * 0.7, tr * 3.2, 5), barkMat);
        const ra = j * PI / 2 + rng(); root.position.set(x + Math.cos(ra) * tr * 1.2, tr * 0.9, z + Math.sin(ra) * tr * 1.2);
        root.rotation.set(Math.sin(ra) * 0.9, 0, -Math.cos(ra) * 0.9); group.add(root);
      }
      for (let j = 0; j < 3; j++) {
        const c = new THREE.Mesh(displacedRock(1, (Math.abs(x * 7) + j) | 0, 14 * k + rng() * 6, 7 * k + rng() * 3, 14 * k + rng() * 6), j % 2 ? leafMat2 : leafMat);
        c.position.set(x + (rng() - 0.5) * 10 * k, h - 2 + j * 4 * k, z + (rng() - 0.5) * 10 * k); c.castShadow = true; group.add(c);
      }
      if (rng() < 0.7) for (let j = 0; j < 3; j++) {   // hanging vines
        const top = v(x + (rng() - 0.5) * 14, h - 4, z + (rng() - 0.5) * 14), len = 14 + rng() * 24;
        const curve = new THREE.CatmullRomCurve3([top, top.clone().add(v(rng() - 0.5, -len * 0.5, rng() - 0.5)), top.clone().add(v((rng() - 0.5) * 3, -len, (rng() - 0.5) * 3))]);
        group.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 10, 0.3, 4, false), vineMat));
      }
    }
    // ferns: open cones flipped so they splay upward
    // fern: 7 arching leaf blades fanned around the centre, built into one geometry
    const fp = [], fi = [];
    for (let b = 0; b < 7; b++) {
      const ang = b / 7 * PI * 2, ca = Math.cos(ang), sa = Math.sin(ang), base = fp.length / 3;
      for (let j = 0; j <= 4; j++) {
        const t = j / 4, r = t * 1.0, y = Math.sin(t * PI * 0.8) * 0.9, w = Math.sin(t * PI) * 0.18 + 0.02;
        fp.push(ca * r - sa * w, y, sa * r + ca * w, ca * r + sa * w, y, sa * r - ca * w);
        if (j) { const k = base + (j - 1) * 2; fi.push(k, k + 1, k + 2, k + 1, k + 3, k + 2); }
      }
    }
    const fernGeo = new THREE.BufferGeometry(); fernGeo.setAttribute('position', new THREE.Float32BufferAttribute(fp, 3)); fernGeo.setIndex(fi); fernGeo.computeVertexNormals();
    const ferns = new THREE.InstancedMesh(fernGeo, new THREE.MeshStandardMaterial({ color: '#4a7a2c', roughness: 0.9, flatShading: true, side: THREE.DoubleSide }), 160);
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s3 = new V3(), p3 = new V3();
    let n = 0;
    for (let i = 0; i < 600 && n < 160; i++) {
      const a = rng() * PI * 2, r = 24 + rng() * 120, x = Math.cos(a) * r, z = Math.sin(a) * r;
      if (z > 6 && Math.abs(x) < 40) continue;
      const w = 2 + rng() * 3.5;
      ferns.setMatrixAt(n++, m4.compose(p3.set(x, 0, z), q.setFromEuler(new THREE.Euler(0, rng() * 6, 0)), s3.set(w, w * (0.6 + rng() * 0.5), w)));
    }
    ferns.count = n; ferns.castShadow = true; group.add(ferns);
    const spores = particleField({ count: 700, area: 120, height: 40, color: '#d8ff9a', size: 0.35, opacity: 0.7, additive: true, seed: 41, zMax: 12 });
    group.add(spores);
    updaters.push((dt, t) => {
      const p = spores.geometry.attributes.position.array;
      for (let i = 0; i < p.length / 3; i++) { p[i * 3] += Math.sin(t * 0.3 + i) * dt * 0.8; p[i * 3 + 1] += Math.sin(t * 0.5 + i * 1.3) * dt * 0.5; }
      spores.geometry.attributes.position.needsUpdate = true;
    });
  }

  // ---- lunar: the Earth overhead, stars, boulders, an old lander
  if (look.props === 'lunar') {
    const earth = new THREE.Group();
    // one sphere, vertex-coloured: ocean, continents, polar caps, with a darker night side
    const globeGeo = new THREE.SphereGeometry(26, 64, 40), gp = globeGeo.attributes.position, gc = new Float32Array(gp.count * 3);
    for (let i = 0; i < gp.count; i++) {
      const x = gp.getX(i), y = gp.getY(i), z = gp.getZ(i);
      const n2 = Math.sin(x * 0.21 + Math.sin(z * 0.13) * 2) * Math.cos(y * 0.27 + z * 0.09) + Math.sin(z * 0.31 - y * 0.12) * 0.4;
      let c = Math.abs(y) > 22 ? [0.92, 0.95, 1] : n2 > 0.25 ? [0.3, 0.52, 0.22] : n2 > 0.18 ? [0.72, 0.64, 0.42] : [0.12, 0.34, 0.72];
      const lit = THREE.MathUtils.clamp(0.25 + (x * 0.7 + y * 0.35 + z * 0.6) / 26 * 0.9, 0.12, 1);
      gc[i * 3] = c[0] * lit; gc[i * 3 + 1] = c[1] * lit; gc[i * 3 + 2] = c[2] * lit;
    }
    globeGeo.setAttribute('color', new THREE.BufferAttribute(gc, 3));
    const globe = new THREE.Mesh(globeGeo, new THREE.MeshBasicMaterial({ vertexColors: true, fog: false }));
    const halo = new THREE.Mesh(new THREE.SphereGeometry(29, 32, 24), new THREE.MeshBasicMaterial({ color: '#7fb6ff', fog: false, transparent: true, opacity: 0.18, blending: THREE.AdditiveBlending, side: THREE.BackSide, depthWrite: false }));
    earth.add(globe, halo);
    earth.position.set(-70, 150, -330); earth.rotation.z = 0.4;
    group.add(earth);
    const stars = particleField({ count: 1400, area: 900, height: 400, color: '#ffffff', size: 1.1, opacity: 0.9, seed: 51 });
    stars.material.fog = false; stars.position.set(0, 40, -260); group.add(stars);
    const rockMat = new THREE.MeshStandardMaterial({ color: '#8a8a90', roughness: 1, flatShading: true });
    for (let i = 0; i < 26; i++) {
      const a = rng() * PI * 2, r = 30 + rng() * 140, x = Math.cos(a) * r, z = Math.sin(a) * r;
      if (z > 6 && Math.abs(x) < 50) continue;
      const sz = 1.5 + rng() * (r > 80 ? 9 : 4);
      const m = new THREE.Mesh(displacedRock(1, i + 30, sz, sz * 0.7, sz), rockMat); m.position.set(x, sz * 0.2, z); m.castShadow = true; group.add(m);
    }
    const lander = new THREE.Group(), gold = new THREE.MeshStandardMaterial({ color: '#c9a640', metalness: 0.7, roughness: 0.35 }), grey = new THREE.MeshStandardMaterial({ color: '#cfd0d4', roughness: 0.6 });
    const body = new THREE.Mesh(new THREE.BoxGeometry(4, 2.6, 4), gold); body.position.y = 3; lander.add(body);
    const cab = new THREE.Mesh(new THREE.CylinderGeometry(1.6, 2, 2.4, 6), grey); cab.position.y = 5.5; lander.add(cab);
    for (let j = 0; j < 4; j++) { const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 4.2, 5), grey); const a = j * PI / 2 + PI / 4; leg.position.set(Math.cos(a) * 2.8, 1.6, Math.sin(a) * 2.8); leg.rotation.set(Math.sin(a) * 0.5, 0, -Math.cos(a) * 0.5); lander.add(leg); }
    const flag = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 1.5), new THREE.MeshStandardMaterial({ color: '#e8e8ea', side: THREE.DoubleSide })); flag.position.set(5, 4.6, 0); lander.add(flag);
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 5.5, 4), grey); pole.position.set(3.8, 2.75, 0); lander.add(pole);
    lander.position.set(58, 0, -52); lander.rotation.y = 0.6; lander.traverse(o => { if (o.isMesh) o.castShadow = true; }); group.add(lander);
  }

  // ---- studio: platform + rim
  if (look.props === 'studio') {
    const plat = new THREE.Mesh(new THREE.CylinderGeometry(16, 17, 0.6, 64), new THREE.MeshStandardMaterial({ color: '#2a2f3a', roughness: 0.55, metalness: 0.3 }));
    plat.position.y = -0.3; plat.receiveShadow = true; group.add(plat);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(16.5, 0.12, 8, 96).rotateX(PI / 2), new THREE.MeshStandardMaterial({ color: '#5fd6ff', emissive: '#5fd6ff', emissiveIntensity: 1.5 }));
    ring.position.y = 0.02; group.add(ring);
    const grid = new THREE.GridHelper(400, 80, 0x3a4458, 0x262c38); grid.position.y = -0.58; group.add(grid);
    ground.position.y = -0.6;
    hemi.intensity = 1.4;
  }

  // ---- hazards
  arena.hazard = (name, type, targets = []) => {
    const key = `${name || ''} ${type || ''}`.toLowerCase();
    if (key.includes('erupt') || (type === 'fire' && look.props === 'lava')) {
      const vol = volcanoes[0] ? volcanoes[Math.floor(Math.random() * volcanoes.length)] : { top: v(0, 80, -150) };
      arena.flash('#ff6a20', 1.0, 900);
      // lava bombs arcing onto targets (+ some strays)
      const bombMat = new THREE.MeshStandardMaterial({ color: '#300800', emissive: '#ff6a10', emissiveIntensity: 3 });
      const dests = targets.length ? targets.map(t => t.clone()) : [v(-8, 0, 0), v(8, 0, 0)];
      for (let i = 0; i < 4; i++) dests.push(v((Math.random() - 0.5) * 70, 0, -10 - Math.random() * 40));
      dests.forEach((d, i) => {
        const bomb = new THREE.Mesh(new THREE.IcosahedronGeometry(1.4 + Math.random(), 1), bombMat);
        const from = vol.top.clone(), delay = i < targets.length ? 0 : Math.random() * 0.4;
        const dur = 1.25;
        addTransient(bomb, (tt) => {
          const k = Math.min(1, Math.max(0, (tt - delay) / dur));
          bomb.position.lerpVectors(from, d, k); bomb.position.y += Math.sin(k * PI) * 60;
          bomb.rotation.x += 0.1;
          bomb.visible = tt > delay;
          if (k >= 1) { arena.groundImpact(d, 1.2, '#ff7a2a'); return false; }
          return true;
        });
      });
      const eruptionCloud = particleField({ count: 80, area: 10, height: 10, color: '#ff8a3a', size: 14, opacity: 0.8, additive: true, seed: 77 });
      eruptionCloud.position.copy(vol.top);
      addTransient(eruptionCloud, (tt) => { eruptionCloud.scale.setScalar(1 + tt * 3); eruptionCloud.material.opacity = Math.max(0, 0.8 - tt * 0.5); eruptionCloud.position.y = vol.top.y + tt * 20; return tt < 1.6; });
      return 1250;
    }
    if (key.includes('white') || type === 'cold') {
      surgeTo(1, '#f4f8ff');
      arena.flash('#ffffff', 0.6, 900);
      return 900;
    }
    if (key.includes('sand') || (type === 'physical' && look.props === 'dunes')) {
      surgeTo(1, '#d8a868');
      arena.flash('#ffd8a0', 0.35, 900);
      return 1000;
    }
    if (key.includes('meteor')) {
      const rockMat = new THREE.MeshStandardMaterial({ color: '#3a2a20', emissive: '#ff8a3a', emissiveIntensity: 2.2 });
      const dests = targets.length ? targets.map(t => t.clone()) : [v(0, 0, 0)];
      for (let i = 0; i < 5; i++) dests.push(v((Math.random() - 0.5) * 80, 0, -10 - Math.random() * 50));
      dests.forEach((d, i) => {
        const m = new THREE.Mesh(new THREE.IcosahedronGeometry(1.1 + Math.random() * 0.8, 0), rockMat);
        const from = d.clone().add(v(60 + Math.random() * 20, 110, -40)), delay = i < targets.length ? 0 : Math.random() * 0.6, dur = 0.9;
        addTransient(m, (tt) => {
          const k = Math.min(1, Math.max(0, (tt - delay) / dur));
          m.position.lerpVectors(from, d, k); m.visible = tt > delay; m.rotation.x += 0.2;
          if (k >= 1) { arena.groundImpact(d, 1.3, '#d8d0c8'); arena.flash('#ffd0a0', 0.4, 300); return false; }
          return true;
        });
      });
      return 1200;
    }
    if (key.includes('vine')) {
      const vineMat = new THREE.MeshStandardMaterial({ color: '#4a7a26', roughness: 0.8 });
      for (const t of targets) for (let j = 0; j < 4; j++) {
        const a = j * PI / 2 + Math.random(), base = t.clone().setY(0).add(v(Math.cos(a) * 5, 0, Math.sin(a) * 5)), c = t.clone().setY(0);
        const curve = new THREE.CatmullRomCurve3([base, base.clone().lerp(c, 0.5).add(v(0, 7, 0)), c.clone().add(v(Math.cos(a + 1.5) * 2.5, 12, Math.sin(a + 1.5) * 2.5)), c.clone().add(v(-Math.cos(a) * 1.5, 18, -Math.sin(a) * 1.5))]);
        const m = new THREE.Mesh(new THREE.TubeGeometry(curve, 24, 0.55, 5, false), vineMat);
        m.scale.y = 0.01;
        addTransient(m, (tt) => { m.scale.y = Math.max(0.01, Math.min(1, tt / 0.35) * (tt > 1.2 ? 1 - (tt - 1.2) / 0.4 : 1)); return tt < 1.6; });
      }
      for (const t of targets) arena.groundImpact(t, 0.7, '#5a6a30');
      return 900;
    }
    if (key.includes('lightning') || type === 'electric') {
      const list = targets.length ? targets : [v(0, 0, 0)];
      list.forEach((t, i) => setTimeoutScaled(() => arena.lightning(t), i * 160));
      return 220 + list.length * 160;
    }
    arena.flash('#ffffff', 0.7, 500);
    for (const t of targets) arena.groundImpact(t, 1);
    return 500;
  };
  // timers that respect timeScale
  const timers = [];
  function setTimeoutScaled(fn, ms) { if (ms <= 0) return fn(); timers.push({ left: ms / 1000, fn }); }
  updaters.push((dt, t, sdt) => {
    for (let i = timers.length - 1; i >= 0; i--) { timers[i].left -= sdt; if (timers[i].left <= 0) { const f = timers[i].fn; timers.splice(i, 1); f(); } }
  });

  // ---- public
  arena.apply = (scene) => {
    arena._scene = scene;
    scene.fog = arena.fog;
    scene.background = arena.fog.color.clone();
    scene.add(group);
  };
  arena.update = (dt, t) => {
    const sdt = dt * (arena.timeScale ?? 1);
    for (const u of updaters) u(dt, t, sdt);
  };
  arena.dispose = () => {
    if (group.parent) group.parent.remove(group);
    if (arena._scene && arena._scene.fog === arena.fog) arena._scene.fog = null;
    group.traverse(o => {
      if (o.geometry) o.geometry.dispose();
      const mats = Array.isArray(o.material) ? o.material : (o.material ? [o.material] : []);
      for (const m of mats) { for (const k of ['map', 'emissiveMap']) if (m[k] && !m[k].userData?.shared) m[k].dispose(); m.dispose(); }
    });
    for (const d of disposables) if (!d.userData?.shared) d.dispose?.();
    sun.shadow?.dispose?.();
    transients.length = 0;
  };
  return arena;
}
