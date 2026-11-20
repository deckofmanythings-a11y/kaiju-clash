# Kaiju Battle — Design

Players build a Kaiju on one of 12 **frames**, spend **Biomass (BIO)** to level its stats and unlock abilities, and send it into automated battles against other players' Kaiju across 6 environments. Fights take no input. They're random but **deterministic**: a fight is fully described by `(engine version, seed, environment, both Kaiju snapshots)`, so any battle can be replayed with the exact same result.

## Stack
- Static site, ES modules, no build step. Three.js `0.170.0` via importmap from jsdelivr; supabase-js `2.45.4` (ESM) for auth.
- **Online ladder (live):** Supabase project raided-hex (`uaoxvhihwiygrajicend`), `kb_` tables, email/password Supabase Auth. See "Online ladder" below.
- Friendly sparring (KJ1 Kaiju codes) and replay codes (KR1) still run fully client-side and are unrated.

## Layout
```
index.html            app shell
css/style.css
js/data/frames.js       12 frames, stats, visualTiers()
js/data/abilities.js    ability catalogue + statuses + availability rules
js/data/environments.js 6 environments (engine mods + renderer look)
js/data/economy.js      BIO costs, rewards, power rating
js/engine/rng.js        seeded PRNG (sfc32), integer helpers
js/engine/battle.js     simulateBattle() -> BattleRecord   (pure, deterministic)
js/render/stage.js      Three.js stage (renderer, camera, lights, env)
js/render/kaiju-model.js procedural Kaiju model from a snapshot
js/render/arena.js      environment scenery + hazard visuals
js/render/playback.js   plays a BattleRecord as 3D animation
js/app/main.js          UI: auth, hangar, creator, lab, arena, ladder, replays, battle HUD
js/app/api.js           supabase-js auth + kaiju-api calls
js/app/codes.js         KJ1 / KR1 share codes
supabase/migrations/    kb_ schema + kb_spend / kb_settle RPCs
supabase/functions/kaiju-api/  the whole backend (one function, action-dispatched)
scripts/sync-engine.mjs copies js/engine + js/data into the function (gitignored copy)
test/determinism.mjs    node test: same inputs => identical hash, many seeds
```

## Ability availability
- **Basic** — each frame has one free basic attack, always equipped.
- **Exclusive** (`frames: [id]`) — 3 per frame. The first one is free on creation.
- **Shared** (`tags: [...]`) — any frame with any matching tag (tail, aquatic, flyer, insect, beast, armored, elemental, grappler, …).
- **Universal** — every frame.
- Up to 4 equipped + basic.

## Upgrades → looks
Each stat has levels 0–10. `visualTiers(kaiju)` maps each stat to tier 0–3 (levels 3/6/9). The renderer grows/adds parts per tier:
- hp → overall size and bulk
- pow → bigger claws/fists/horns
- enr → glowing crystals/dorsal plates (accent color, emissive)
- arm → armor plates, then spikes
- spd → fins/frills, leaner proportions
- fcs → extra eyes, crest
Equipped abilities with a `vis` key add a cosmetic too (e.g. `dorsal_glow`, `missile_pods`, `tail_club`, `stinger`, `tusks`, `ram_horn`, `vents`, `core`, `jaw`).

## Determinism rules (engine)
- All randomness from `sfc32` seeded with a uint32. No `Math.random`, `Date`, or `Math.sin` etc. in the engine.
- All combat math is integer (`Math.floor` after every multiply/divide). Iteration is over arrays only, in fixed order.
- The record stores `ENGINE_VERSION`. Replays re-simulate. If the version differs, the stored events are played as-is rather than re-simulated.
- `record.hash` = FNV-1a over the JSON of the events, used to verify a re-sim matches.

## Snapshot (what a Kaiju is)
```js
{ id, name, frame, levels: {hp,pow,enr,arm,spd,fcs}, abilities: [abilityId...max 4], colors: {body, belly, accent}, owner? }
```

## BattleRecord
```js
{
  v: ENGINE_VERSION, seed, env,
  fighters: [snapshotA, snapshotB],
  meta: [{ maxHp, stats }, { maxHp, stats }],   // effective stats after env mods
  events: [ ... ],
  winner: 0 | 1 | null, reason: 'ko' | 'decision' | 'draw', turns, hash
}
```
### Events (in order; `a` = actor index 0|1, `tg` = target index)
| t | fields | meaning |
|---|---|---|
| `start` | | fight begins |
| `turn` | `n, a, en` | fighter `a` acts; `en` = its energy after regen |
| `skip` | `a, reason` ('stun'\|'bind') | lost the turn |
| `use` | `a, tg, ab, fx, name` | ability used (`fx` = renderer hint, see abilities.js) |
| `hit` | `a, tg, dmg, crit, type, hp, absorbed, hitNo, hits` | one hit landed; `hp` = target HP after |
| `miss` | `a, tg, hitNo, hits` | one hit missed / dodged |
| `status` | `tg, st, turns` | status applied |
| `expire` | `tg, st` | status ended |
| `dot` | `tg, st, dmg, hp` | damage-over-time tick |
| `heal` | `tg, amt, hp` | healed |
| `selfdmg` | `a, dmg, hp` | recoil |
| `hazard` | `name, type, hits: [{tg, dmg, hp}]` | environment hazard |
| `ko` | `tg` | fighter knocked out |
| `end` | `winner, reason` | fight over |

Max 40 rounds; then decision by HP %.

## Renderer API (contract for js/render/*)
```js
// stage.js
createStage(canvas) -> {
  scene, camera, renderer,
  setEnvironment(envId),             // builds arena via arena.js, swaps the old one out
  showcase(snapshot),                // single Kaiju on a turntable (creator / upgrade lab)
  setFighters(snapA, snapB),         // two Kaiju facing each other for battle
  fighters: [modelA, modelB],
  resize(), dispose()
}
// kaiju-model.js
buildKaijuModel(snapshot) -> THREE.Group  // userData.rig = { root, body, head, mouth(Object3D beam origin), limbs: [...], tail: [...], height }
// playback.js
playBattle(stage, record, { speed=1, onEvent(ev, index), onDone() }) -> { pause(), resume(), setSpeed(x), skipToEnd(), stop() }
```
`playBattle` animates each event in order (lunges, beams, projectiles, AoE rings, grabs, roars, shield bubbles, heal glow, hit flash/knockback, damage numbers, status tints, hazards, KO collapse) and calls `onEvent` at the moment each event's visual lands so the UI can update HP/energy bars and the log.

## Online ladder
Server-authoritative. Clients never touch the tables (RLS on, zero policies, privileges revoked from anon/authenticated); everything goes through the `kaiju-api` Edge Function using the service role.

- **Identity:** Supabase Auth JWT -> `kb_players` row (display name, BIO, active Kaiju, daily fight counter).
- **Spending:** `kb_spend` RPC deducts BIO and writes the Kaiju in one transaction, guarded by `kb_kaiju.version` (optimistic concurrency), so parallel requests can't double-spend.
- **Matchmaking:** the server stores 3 offers per player in `kb_offers` (opponent + environment). At most 2 are real players' active Kaiju near your Elo (window widens 150 -> 350 -> 800 -> any); at least one slot is always a seeded Wild rival, at a random position, and any Wild slot has a 6% chance to be a rare Wild-only enemy. A fight must consume an offer, so clients can't pick opponents or environments. Rerolling costs 15 BIO.
- **Fights:** the server picks the seed, runs `simulateBattle` (same file as the browser), and settles atomically in `kb_settle` (consumes the offer, inserts `kb_battles`, pays BIO, moves both Elos). The response carries no events: the client re-simulates from `(v, seed, env, fighters)` and shows a ✓ when the hash matches.
- **Elo:** K=32 vs players, K=16 vs wild rivals (attacker only). Defenders' Elo moves the opposite way, and they get 25 BIO for each defense win plus a "while you were away" report on next login.
- **Limits:** 40 rated fights per UTC day; after that fights are unrated (no BIO, no Elo). Max 6 Kaiju per player.
- **Public actions:** `leaderboard` and `battle` (fetch any fight for replay, also via `?battle=<id>` links). Everything else requires a JWT.

### Deploy
```bash
node scripts/sync-engine.mjs           # REQUIRED: server must run the same engine as the client
supabase functions deploy kaiju-api --project-ref uaoxvhihwiygrajicend --no-verify-jwt --use-api
```
`--no-verify-jwt` is deliberate (public actions); the function verifies JWTs itself. Migrations are applied with `supabase db query --linked -f`, then recorded in `supabase_migrations.schema_migrations` by hand (`db push` is unsafe on this shared project).

**Changing engine/data:** bump `ENGINE_VERSION`, sync, and deploy the function at the same time as the client. Stored battles keep their `engine_v`; replays from older versions are flagged as possibly different.

## Injuries and death (engine v2)
- **Data:** `js/data/injuries.js` defines 10 injuries (6 minor, severity 1; 4 major, severity 2). Each one gives percent stat penalties, and penalties on the same stat stack, capped at -60%. Injuries are part of the snapshot (`injuries: [id]`), so they really weaken the Kaiju in battle.
- **Condition:** the sum of severity. Healthy 0, Wounded 1-2, Battered 3-4, **Critical 5+**.
- **Aftermath (`js/engine/aftermath.js`):** after a fight, each real Kaiju may get one new injury. The odds depend on how the fight went: knocked out 70% (35% of those major), lost on time 40/15, draw 35/15, won under 25% HP 30/10, won under 50% HP 12/5, clean win 0. The roll is seeded from the battle seed and hash, so the same record always gives the same aftermath, and the client recomputes it to match the server.
- **Death:** only the **attacker** can die, only if it is **knocked out**, and only if it was **Critical going into the fight**. Chance = 20% + 10% per severity point above 5, capped at 50%. The chance is shown before a fight, and attacking while Critical needs a confirmation. Defenders are never injured or killed, because they aren't there to choose the fight.
- **On death:** the Kaiju is retired, with `died_at` and `death = { killer, killer_owner, env, battle, salvage }` set. 20% of the Biomass invested in it is paid back. It moves to the Hangar's "Hall of the Fallen" and is shown as a stone statue, with a link to its final battle.
- **Healing:** each injury heals on its own in minutes (`healMin`: minor 10-20, major 45-60, never over 1h), or instantly for BIO (minor 25, major 75, pro-rated by time left). They're stored as `{ id, at, heals_at }`. The server appends new injuries in `kb_settle`, so concurrent fights can't overwrite each other's injuries. Healed entries are pruned on the next `kb_spend`.
