// Local save + share codes. Phase 2 swaps this for Supabase.
import { STARTING_BIO } from '../data/economy.js';
import { randomSeed } from '../engine/rng.js';

const KEY = 'kb_save_v1';
const MAX_HISTORY = 25;

function fresh() {
  return { bio: STARTING_BIO, kaiju: [], activeId: null, rivalSeed: randomSeed(), history: [] };
}

let state;
export function load() {
  try { state = JSON.parse(localStorage.getItem(KEY)) || fresh(); } catch { state = fresh(); }
  if (!Array.isArray(state.kaiju)) state = fresh();
  return state;
}
export function save() {
  try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { console.warn('save failed', e); }
}
export const S = () => state;
export function active() { return state.kaiju.find(k => k.id === state.activeId) || state.kaiju[0] || null; }

export function addHistory(record, extra) {
  // Store only what's needed to re-simulate; events are rebuilt on replay.
  state.history.unshift({
    v: record.v, seed: record.seed, env: record.env, fighters: record.fighters,
    winner: record.winner, reason: record.reason, hash: record.hash, at: Date.now(), ...extra,
  });
  state.history.length = Math.min(state.history.length, MAX_HISTORY);
}

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
