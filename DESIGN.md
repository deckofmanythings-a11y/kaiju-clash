# Kaiju Battle — Design

Players build a Kaiju on one of 12 **frames**, spend **Biomass (BIO)** to level its stats and unlock abilities, and send it into automated battles against other players' Kaiju across 6 environments. Fights take no input. They're random but **deterministic**: a fight is fully described by `(engine version, seed, environment, both Kaiju snapshots)`, so any battle can be replayed with the exact same result.

## Stack
- Static site, ES modules, no build step. Three.js `0.170.0` via importmap from jsdelivr.
- Phase 1 (this build): runs locally. Player Kaiju saved in `localStorage`. Opponents come from a deterministic rival generator (seeded) **plus** import/export of Kaiju codes (base64 JSON) so you can fight a friend's Kaiju. Replay codes are shareable too.
- Phase 2 (planned): Supabase (raided-hex project, `kb_` tables) for a real async ladder. The battle engine is pure JS with no DOM, so the same file runs in a Supabase Edge Function (Deno) to make results and BIO payouts server-authoritative.

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
js/app/*.js             UI: creator, upgrade lab, battle screen, storage, rivals
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
