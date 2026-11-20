# Kaiju Clash

Build a Kaiju, upgrade it with Biomass, and send it into automated, deterministic 3D battles.

- **12 frames** (Saurian, Titan Ape, Kraken, Lepidra, Carapace, Mantid, Sea Serpent, Mecha, Magma Golem, Hydra, Arachnid, Storm Ray), each with a basic attack, 3 exclusive abilities, and tags that unlock shared abilities.
- **64 abilities**: 11 basic, 36 exclusive, 11 tag-shared, 6 universal. Equip 4.
- **6 environments** that change combat (stat mods, damage-type multipliers, periodic hazards).
- **Upgrades change the model**: stat levels 3/6/9 add or grow parts, and some abilities add cosmetics.
- **Online ladder**: sign in, fight other players' Kaiju (server-simulated, Elo-ranked), defend while offline, leaderboard.
- **Deterministic**: a fight = engine version + seed + environment + two snapshots. Replays re-simulate and are checked by hash.

## Run
```bash
python -m http.server 4391 --directory kaiju-battle
```
Open http://localhost:4391. The renderer dev page is at `/js/render/render-test.html`.

## Deploy the backend
```bash
node kaiju-battle/scripts/sync-engine.mjs
```
```bash
supabase functions deploy kaiju-api --project-ref uaoxvhihwiygrajicend --no-verify-jwt --use-api
```

## Test
```bash
node kaiju-battle/test/determinism.mjs
```
This checks determinism over 500 random fights and prints a frame win-rate balance report.

See [DESIGN.md](DESIGN.md) for the data model, event format, and renderer contract.
