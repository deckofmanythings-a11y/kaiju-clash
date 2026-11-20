// Supabase auth + kaiju-api calls. The publishable key is public by design; all game data sits
// behind RLS with no policies and is only reachable through the kaiju-api Edge Function.
import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.45.4/+esm';

export const SUPABASE_URL = 'https://uaoxvhihwiygrajicend.supabase.co';
const KEY = 'sb_publishable_91Zn_uJ9anW46vAClx6-bQ__wLL4fXl';

// Read the auth redirect BEFORE supabase-js consumes and clears the URL hash. A password-reset
// link lands here as #access_token=…&type=recovery, or as #error=…&error_description=… when the
// link is expired or already used.
const hashParams = new URLSearchParams(location.hash.replace(/^#/, ''));
export const isRecoveryLink = hashParams.get('type') === 'recovery';
export const authLinkError = hashParams.get('error_description')?.replace(/\+/g, ' ') || null;
if (authLinkError) history.replaceState(null, '', location.pathname + location.search);

export const sb = createClient(SUPABASE_URL, KEY, { auth: { storageKey: 'kb-auth' } });

export async function session() {
  try { const { data } = await sb.auth.getSession(); return data.session; } catch { return null; }
}

export async function call(action, payload = {}) {
  const s = await session();
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 20000);
  try {
    const res = await fetch(`${SUPABASE_URL}/functions/v1/kaiju-api`, {
      method: 'POST', signal: ctrl.signal,
      headers: { 'Content-Type': 'application/json', apikey: KEY, Authorization: 'Bearer ' + (s?.access_token || KEY) },
      body: JSON.stringify({ action, ...payload }),
    });
    const body = await res.json().catch(() => ({ ok: false, error: `Server error (${res.status})` }));
    if (!body.ok) { const e = new Error(body.error || 'Request failed'); e.status = res.status; throw e; }
    return body;
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('Server took too long, try again.');
    throw e;
  } finally { clearTimeout(timer); }
}

export const signIn = (email, password) => sb.auth.signInWithPassword({ email, password });
export const signUp = (email, password) => sb.auth.signUp({ email, password });
export const signOut = () => sb.auth.signOut();
export const resetPassword = (email) => sb.auth.resetPasswordForEmail(email, { redirectTo: location.origin + location.pathname });
