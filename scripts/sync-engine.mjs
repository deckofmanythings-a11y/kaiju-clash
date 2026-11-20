// Copies the browser's engine + data modules into the Edge Function's _shared folder so the
// server simulates with byte-identical code. Run before every `supabase functions deploy`.
//   node scripts/sync-engine.mjs          copy
//   node scripts/sync-engine.mjs --check  exit 1 if the copies are out of date
import { readdirSync, readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dest = join(root, 'supabase', 'functions', '_shared', 'kaiju');
const check = process.argv.includes('--check');
let stale = 0;
for (const dir of ['engine', 'data']) {
  const src = join(root, 'js', dir);
  mkdirSync(join(dest, dir), { recursive: true });
  for (const f of readdirSync(src).filter(f => f.endsWith('.js'))) {
    const a = readFileSync(join(src, f), 'utf8');
    const out = join(dest, dir, f);
    const same = existsSync(out) && readFileSync(out, 'utf8') === a;
    if (same) continue;
    stale++;
    if (check) console.log('out of date:', `${dir}/${f}`);
    else { writeFileSync(out, a); console.log('synced', `${dir}/${f}`); }
  }
}
if (check && stale) process.exit(1);
if (!stale) console.log('engine copies up to date');
