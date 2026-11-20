// Deterministic PRNG. sfc32 seeded via splitmix32. Pure integer ops, identical across JS engines.

function splitmix32(a) {
  return function () {
    a |= 0; a = (a + 0x9e3779b9) | 0;
    let t = a ^ (a >>> 16); t = Math.imul(t, 0x21f0aaad);
    t = t ^ (t >>> 15); t = Math.imul(t, 0x735a2d97);
    return (t ^ (t >>> 15)) >>> 0;
  };
}

export function createRng(seed) {
  const sm = splitmix32(seed >>> 0);
  let a = sm(), b = sm(), c = sm(), d = sm();
  const next = () => {
    a |= 0; b |= 0; c |= 0; d |= 0;
    const t = (((a + b) | 0) + d) | 0;
    d = (d + 1) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    c = (c + t) | 0;
    return t >>> 0;
  };
  for (let i = 0; i < 12; i++) next();
  return {
    next,
    int(n) { return n <= 1 ? 0 : next() % n; },          // 0..n-1
    range(lo, hi) { return lo + this.int(hi - lo + 1); },  // inclusive
    roll(pct) { return this.int(100) < pct; },
    pick(arr) { return arr[this.int(arr.length)]; },
  };
}

export function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

export function hashString(str) { return parseInt(fnv1a(str), 16) >>> 0; }

export function randomSeed() {
  // Only used OUTSIDE the engine to pick a new seed for a new fight.
  const u = new Uint32Array(1);
  globalThis.crypto.getRandomValues(u);
  return u[0];
}
