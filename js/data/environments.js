// Battle environments. Each one changes combat (engine) and the arena look (renderer).
// tagMods: stat percentage modifiers for fighters with a given frame tag (e.g. spd: 130 = +30%).
// typeMult: damage multiplier % by damage type, for everyone.
// hazard: every N turns (global turn counter), deals % max HP of `type` to fighters lacking `immuneTags`.
// look: renderer hints.

export const ENVIRONMENTS = [
  {
    id: 'city', name: 'Neo-Tokyo Bay', blurb: 'Skyscrapers to smash. Grapplers hurl buildings (+25% grab damage).',
    tagMods: {}, typeMult: {}, fxMult: { grab: 125 },
    hazard: null,
    look: { sky: '#1a1f3a', fog: '#2a2f4a', ground: '#2a2a2e', light: '#ffd9a8', props: 'buildings', time: 'night' },
  },
  {
    id: 'ocean', name: 'Pacific Trench', blurb: 'Waist-deep surf. Aquatic kaiju +25% Speed and Energy; others -15% Speed.',
    tagMods: { aquatic: { spd: 125, enr: 125 }, _other: { spd: 85 } }, typeMult: { electric: 115, fire: 85 },
    hazard: null,
    look: { sky: '#6a8fb0', fog: '#7a9fc0', ground: '#1a4a6a', light: '#ffffff', props: 'water', time: 'day' },
  },
  {
    id: 'volcano', name: 'Caldera Rim', blurb: 'Lava everywhere. Fire +25%. Eruptions burn non-elemental kaiju every 4 turns.',
    tagMods: { elemental: { arm: 115 } }, typeMult: { fire: 125, cold: 75 },
    hazard: { every: 4, type: 'fire', pct: 5, immuneTags: ['elemental', 'stone'], name: 'Eruption', status: { status: 'burn', chance: 50, turns: 2, power: 30 } },
    look: { sky: '#3a120a', fog: '#5a1a0a', ground: '#1a0e0a', light: '#ff8a4a', props: 'lava', time: 'dusk' },
  },
  {
    id: 'arctic', name: 'Glacier Shelf', blurb: 'Bitter cold. Cold +25%. Reptiles and insects -20% Speed.',
    tagMods: { reptile: { spd: 80 }, insect: { spd: 80 }, mammal: { arm: 115 } }, typeMult: { cold: 125, fire: 90 },
    hazard: { every: 5, type: 'cold', pct: 3, immuneTags: ['mammal', 'mech', 'aquatic'], name: 'Whiteout', status: { status: 'slow', chance: 50, turns: 2 } },
    look: { sky: '#b8d0e8', fog: '#dde8f2', ground: '#e8f0f8', light: '#e8f0ff', props: 'ice', time: 'day' },
  },
  {
    id: 'desert', name: 'Dune Sea', blurb: 'Sandstorms. Everyone -10 accuracy. Insects and stone thrive (+15% Armor).',
    tagMods: { insect: { arm: 115 }, stone: { arm: 115 } }, typeMult: { acid: 115 }, accMod: -10,
    hazard: { every: 6, type: 'physical', pct: 3, immuneTags: ['stone', 'insect'], name: 'Sandstorm', status: { status: 'blind', chance: 40, turns: 2 } },
    look: { sky: '#e0b070', fog: '#d8a868', ground: '#c89858', light: '#fff0c8', props: 'dunes', time: 'day' },
  },
  {
    id: 'storm', name: 'Typhoon Strait', blurb: 'Lightning storm. Electric +30%. Flyers +15% Speed. Lightning strikes every 3 turns.',
    tagMods: { flyer: { spd: 115 } }, typeMult: { electric: 130 },
    hazard: { every: 3, type: 'electric', pct: 4, immuneTags: ['elemental'], name: 'Lightning Strike', random: true, status: { status: 'stun', chance: 15, turns: 1 } },
    look: { sky: '#202530', fog: '#303845', ground: '#2a3a3a', light: '#a8b8ff', props: 'storm', time: 'night' },
  },
];

export const ENV_BY_ID = Object.fromEntries(ENVIRONMENTS.map(e => [e.id, e]));
