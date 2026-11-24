// Battle environments. Each one changes combat (engine) and the arena look (renderer).
//
// Engine v6 arenas are FAIR BY CONSTRUCTION (test/arena-balance.mjs checks it):
//   - every arena favors exactly ARENA_SIDE playable frames and hinders exactly ARENA_SIDE playable frames;
//   - across all arenas every playable frame is favored exactly as often as it is hindered (net zero), and no frame
//     is favored (or hindered) more than one arena more than any other frame.
// favored: frame ids that get `mods.favored` (stat % multipliers) and are immune to the hazard.
// hindered: frame ids that get `mods.hindered`. Everyone else is neutral.
// hazard: every N rounds, deals % max HP of `type` to every non-favored fighter (or one random one if `random`).
// Wild-only frames are neutral everywhere. A NEW PLAYABLE FRAME must be added to arenas so the test still passes.
// look: renderer hints.

export const ARENA_SIDE = 5;
// The same perk everywhere, so being favored (or hindered) is worth the same to every frame in every arena.
export const ARENA_MODS = { favored: { pow: 110, enr: 110, arm: 110 }, hindered: { pow: 90, enr: 90, arm: 90 } };

export const ENVIRONMENTS = [
  {
    id: 'city', name: 'Neo-Tokyo Bay', blurb: 'Skyscrapers to smash and hurl.',
    favored: ['colossalwoman', 'giantbaby', 'mecha', 'ape', 'mechasaurian'], hindered: ['lepidra', 'wasp', 'stormray', 'demon', 'kraken'],
    favWhy: 'built for streets: hurl whole buildings', hindWhy: 'flyers clip towers, the Kraken is beached',
    mods: ARENA_MODS,
    hazard: { every: 4, type: 'physical', pct: 3, name: 'Falling Tower', status: { status: 'stun', chance: 20, turns: 1 } },
    look: { sky: '#1a1f3a', fog: '#2a2f4a', ground: '#2a2a2e', light: '#ffd9a8', props: 'buildings', time: 'night' },
  },
  {
    id: 'ocean', name: 'Pacific Trench', blurb: 'Waist-deep surf and riptides.',
    favored: ['kraken', 'serpent', 'carapace', 'stormray', 'hydra'], hindered: ['golem', 'mecha', 'scorpion', 'arachnid', 'mechasaurian'],
    favWhy: 'at home in the water', hindWhy: 'heavy, sinking or water-logged',
    mods: ARENA_MODS,
    hazard: { every: 4, type: 'physical', pct: 3, name: 'Riptide', status: { status: 'slow', chance: 50, turns: 2 } },
    look: { sky: '#6a8fb0', fog: '#7a9fc0', ground: '#1a4a6a', light: '#ffffff', props: 'water', time: 'day' },
  },
  {
    id: 'volcano', name: 'Caldera Rim', blurb: 'Lava everywhere and regular eruptions.',
    favored: ['golem', 'demon', 'saurian', 'scorpion', 'mechasaurian'], hindered: ['lepidra', 'mantid', 'carapace', 'arachnid', 'werewolf'],
    favWhy: 'heat-forged and fireproof', hindWhy: 'brittle shells, wings and fur scorch',
    mods: ARENA_MODS,
    hazard: { every: 4, type: 'fire', pct: 3, name: 'Eruption', status: { status: 'burn', chance: 50, turns: 2, power: 30 } },
    look: { sky: '#3a120a', fog: '#5a1a0a', ground: '#1a0e0a', light: '#ff8a4a', props: 'lava', time: 'dusk' },
  },
  {
    id: 'arctic', name: 'Glacier Shelf', blurb: 'Bitter cold and whiteouts.',
    favored: ['ape', 'werewolf', 'mecha', 'giantbaby', 'colossalwoman'], hindered: ['saurian', 'hydra', 'serpent', 'mantid', 'wasp'],
    favWhy: 'warm-blooded or heated', hindWhy: 'cold-blooded: sluggish in the ice',
    mods: ARENA_MODS,
    hazard: { every: 4, type: 'cold', pct: 3, name: 'Whiteout', status: { status: 'slow', chance: 50, turns: 2 } },
    look: { sky: '#b8d0e8', fog: '#dde8f2', ground: '#e8f0f8', light: '#e8f0ff', props: 'ice', time: 'day' },
  },
  {
    id: 'desert', name: 'Dune Sea', blurb: 'Endless dunes and sandstorms.',
    favored: ['scorpion', 'arachnid', 'mantid', 'carapace', 'golem'], hindered: ['kraken', 'stormray', 'serpent', 'werewolf', 'ape'],
    favWhy: 'desert-hardy shells and stone', hindWhy: 'drying out or overheating',
    mods: ARENA_MODS,
    hazard: { every: 4, type: 'physical', pct: 3, name: 'Sandstorm', status: { status: 'blind', chance: 40, turns: 2 } },
    look: { sky: '#e0b070', fog: '#d8a868', ground: '#c89858', light: '#fff0c8', props: 'dunes', time: 'day' },
  },
  {
    id: 'storm', name: 'Typhoon Strait', blurb: 'A raging lightning storm.',
    favored: ['stormray', 'lepidra', 'wasp', 'demon', 'kraken'], hindered: ['mecha', 'mechasaurian', 'golem', 'colossalwoman', 'scorpion'],
    favWhy: 'ride the winds and waves', hindWhy: 'tall, metal or armored: lightning rods',
    mods: ARENA_MODS,
    hazard: { every: 4, type: 'electric', pct: 3, name: 'Lightning Strike', status: { status: 'stun', chance: 15, turns: 1 } },
    look: { sky: '#202530', fog: '#303845', ground: '#2a3a3a', light: '#a8b8ff', props: 'storm', time: 'night' },
  },
  {
    id: 'jungle', name: 'Verdant Hollow', blurb: 'A primeval jungle choked with giant vines.',
    favored: ['hydra', 'saurian', 'werewolf', 'serpent', 'arachnid'], hindered: ['mecha', 'mechasaurian', 'colossalwoman', 'giantbaby', 'demon'],
    favWhy: 'ambush hunters of the undergrowth', hindWhy: 'tangled, snagged and grounded by the canopy',
    mods: ARENA_MODS,
    hazard: { every: 4, type: 'physical', pct: 3, name: 'Strangler Vines', status: { status: 'bind', chance: 35, turns: 1 } },
    look: { sky: '#3f5a3a', fog: '#4d6a45', ground: '#4a5a32', light: '#e8ffc8', props: 'jungle', time: 'day' },
  },
  {
    id: 'lunar', name: 'Sea of Tranquility', blurb: 'A crater on the Moon. Low gravity, no air.',
    favored: ['lepidra', 'wasp', 'mantid', 'mecha', 'mechasaurian'], hindered: ['carapace', 'saurian', 'hydra', 'ape', 'giantbaby'],
    favWhy: 'light leapers and space-rated machines', hindWhy: 'heavy bruisers lose their footing',
    mods: ARENA_MODS,
    hazard: { every: 4, type: 'physical', pct: 3, name: 'Meteor Shower', status: { status: 'sunder', chance: 30, turns: 2 } },
    look: { sky: '#05060c', fog: '#0a0c16', ground: '#7a7a80', light: '#f4f6ff', props: 'lunar', time: 'night' },
  },
];

export const ENV_BY_ID = Object.fromEntries(ENVIRONMENTS.map(e => [e.id, e]));

// How an arena treats a frame: 1 favored, -1 hindered, 0 neutral.
export const arenaSide = (env, frameId) => (env?.favored?.includes(frameId) ? 1 : env?.hindered?.includes(frameId) ? -1 : 0);

// ---------- Legacy (engine v5 and older): frozen. Records from those engines replay with these. ----------
// tagMods: stat % modifiers by frame tag (_other = no tag matched). typeMult: damage % by type. hazard.immuneTags.
export const LEGACY_ENVIRONMENTS = [
  {
    id: 'city', name: 'Neo-Tokyo Bay', blurb: 'Skyscrapers to smash. Grapplers hurl buildings (+25% grab damage).',
    tagMods: {}, typeMult: {}, fxMult: { grab: 125 },
    hazard: null,
  },
  {
    id: 'ocean', name: 'Pacific Trench', blurb: 'Waist-deep surf. Aquatic kaiju +25% Speed and Energy; others -15% Speed.',
    tagMods: { aquatic: { spd: 125, enr: 125 }, _other: { spd: 85 } }, typeMult: { electric: 115, fire: 85 },
    hazard: null,
  },
  {
    id: 'volcano', name: 'Caldera Rim', blurb: 'Lava everywhere. Fire +25%. Eruptions burn non-elemental kaiju every 4 turns.',
    tagMods: { elemental: { arm: 115 } }, typeMult: { fire: 125, cold: 75 },
    hazard: { every: 4, type: 'fire', pct: 5, immuneTags: ['elemental', 'stone'], name: 'Eruption', status: { status: 'burn', chance: 50, turns: 2, power: 30 } },
  },
  {
    id: 'arctic', name: 'Glacier Shelf', blurb: 'Bitter cold. Cold +25%. Reptiles and insects -20% Speed.',
    tagMods: { reptile: { spd: 80 }, insect: { spd: 80 }, mammal: { arm: 115 } }, typeMult: { cold: 125, fire: 90 },
    hazard: { every: 5, type: 'cold', pct: 3, immuneTags: ['mammal', 'mech', 'aquatic'], name: 'Whiteout', status: { status: 'slow', chance: 50, turns: 2 } },
  },
  {
    id: 'desert', name: 'Dune Sea', blurb: 'Sandstorms. Everyone -10 accuracy. Insects and stone thrive (+15% Armor).',
    tagMods: { insect: { arm: 115 }, stone: { arm: 115 } }, typeMult: { acid: 115 }, accMod: -10,
    hazard: { every: 6, type: 'physical', pct: 3, immuneTags: ['stone', 'insect'], name: 'Sandstorm', status: { status: 'blind', chance: 40, turns: 2 } },
  },
  {
    id: 'storm', name: 'Typhoon Strait', blurb: 'Lightning storm. Electric +30%. Flyers +15% Speed. Lightning strikes every 3 turns.',
    tagMods: { flyer: { spd: 115 } }, typeMult: { electric: 130 },
    hazard: { every: 3, type: 'electric', pct: 4, immuneTags: ['elemental'], name: 'Lightning Strike', random: true, status: { status: 'stun', chance: 15, turns: 1 } },
  },
];
export const LEGACY_ENV_BY_ID = Object.fromEntries(LEGACY_ENVIRONMENTS.map(e => [e.id, e]));
