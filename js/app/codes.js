// Share codes: KJ1.<base64url json> for a Kaiju, KR1.<base64url json> for a replay.
// ---- share codes: PREFIX.base64url(json) ----
function b64e(obj) {
  const bytes = new TextEncoder().encode(JSON.stringify(obj));
  let bin = ''; for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
function b64d(str) {
  str = str.replace(/-/g, '+').replace(/_/g, '/');
  const bin = atob(str + '==='.slice((str.length + 3) % 4));
  return JSON.parse(new TextDecoder().decode(Uint8Array.from(bin, c => c.charCodeAt(0))));
}
export function kaijuCode(k) {
  const { name, frame, levels, abilities, colors } = k;
  return 'KJ1.' + b64e({ name, frame, levels, abilities, colors, owner: k.owner || 'A challenger' });
}
export function replayCode(r) {
  return 'KR1.' + b64e({ v: r.v, seed: r.seed, env: r.env, fighters: r.fighters, hash: r.hash });
}
export function parseCode(code) {
  code = (code || '').trim();
  const m = /^(KJ1|KR1)\.([A-Za-z0-9_-]+)$/.exec(code);
  if (!m) throw new Error('Not a Kaiju or replay code.');
  return { kind: m[1] === 'KJ1' ? 'kaiju' : 'replay', data: b64d(m[2]) };
}
