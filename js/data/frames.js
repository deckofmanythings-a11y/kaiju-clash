import { applyTune } from './finetune.js';
// 12 core Kaiju frames. Stats are integers. Effective stat = base + growth * level (level 0..10).
// `plan` is the 3D body plan the renderer builds. `tags` gate shared abilities.
// `resist` is a percentage damage multiplier per damage type (100 = normal, 50 = half, 150 = weak).

export const STATS = ['hp', 'pow', 'enr', 'arm', 'spd', 'fcs'];
export const STAT_NAMES = {
  hp: 'Vitality', pow: 'Might', enr: 'Energy', arm: 'Armor', spd: 'Speed', fcs: 'Focus',
};
export const STAT_BLURBS = {
  hp: 'Max health. Visual: bulk and size.',
  pow: 'Physical attack power. Visual: claws, fists, horns.',
  enr: 'Energy attack power + energy regen. Visual: glowing crystals.',
  arm: 'Damage reduction. Visual: plating, then spikes.',
  spd: 'Turn frequency + dodge. Visual: fins and a leaner frame.',
  fcs: 'Accuracy + crit chance. Visual: extra eyes, crest.',
};
export const MAX_STAT_LEVEL = 10;

export const FRAMES = [
  {
    id: 'saurian', name: 'Saurian', plan: 'biped',
    blurb: 'The classic city-flattener. Balanced, tough, breathes fire.',
    tags: ['reptile', 'beast', 'tail', 'elemental'],
    base: { hp: 1100, pow: 18, enr: 18, arm: 14, spd: 10, fcs: 12 },
    growth: { hp: 90, pow: 2, enr: 2, arm: 2, spd: 1, fcs: 1 },
    resist: { fire: 60, cold: 130 },
    basic: 'claw', exclusive: ['atomic_breath', 'radiant_pulse', 'thermal_hide'],
    colors: { body: '#3f5a3a', belly: '#9aa36b', accent: '#5fd6ff' },
  },
  {
    id: 'ape', name: 'Titan Ape', plan: 'ape',
    blurb: 'A mountain of muscle. Huge physical damage, throws things.',
    tags: ['beast', 'grappler', 'mammal'],
    base: { hp: 1080, pow: 21, enr: 8, arm: 11, spd: 12, fcs: 11 },
    growth: { hp: 95, pow: 3, enr: 1, arm: 1, spd: 1, fcs: 1 },
    resist: { cold: 80, electric: 120 },
    basic: 'punch', exclusive: ['hammer_fists', 'chest_pound', 'mountain_toss'],
    colors: { body: '#3b2f2a', belly: '#7a6455', accent: '#ffb347' },
  },
  {
    id: 'kraken', name: 'Kraken', plan: 'cephalopod',
    blurb: 'Tentacled horror from the deep. Binds and drowns.',
    tags: ['aquatic', 'grappler', 'abyssal'],
    base: { hp: 1180, pow: 18, enr: 18, arm: 11, spd: 11, fcs: 14 },
    growth: { hp: 85, pow: 2, enr: 2, arm: 1, spd: 1, fcs: 2 },
    resist: { cold: 60, electric: 115, fire: 80 },
    basic: 'tentacle_slap', exclusive: ['ink_cloud', 'crushing_embrace', 'abyssal_pull'],
    colors: { body: '#5a2a4a', belly: '#c98ab0', accent: '#7cffc4' },
  },
  {
    id: 'lepidra', name: 'Lepidra', plan: 'moth',
    blurb: 'Colossal winged insect. Fragile, evasive, toxic.',
    tags: ['insect', 'flyer'],
    base: { hp: 960, pow: 11, enr: 22, arm: 9, spd: 16, fcs: 15 },
    growth: { hp: 70, pow: 1, enr: 3, arm: 1, spd: 2, fcs: 2 },
    resist: { acid: 60, fire: 140, sonic: 120 },
    basic: 'wing_buffet', exclusive: ['prismatic_scales', 'toxic_dust', 'lumen_beam'],
    colors: { body: '#4a3a1a', belly: '#e8c86a', accent: '#ff7af5' },
  },
  {
    id: 'carapace', name: 'Carapace', plan: 'turtle',
    blurb: 'Walking fortress. Outlasts everything.',
    tags: ['armored', 'aquatic', 'reptile', 'tail'],
    base: { hp: 1200, pow: 14, enr: 12, arm: 19, spd: 6, fcs: 10 },
    growth: { hp: 100, pow: 2, enr: 1, arm: 3, spd: 1, fcs: 1 },
    resist: { physical: 85, fire: 80, acid: 120 },
    basic: 'shell_bash', exclusive: ['shell_spin', 'turtle_up', 'magma_vent'],
    colors: { body: '#4b4a3a', belly: '#b5a77a', accent: '#ff6a2a' },
  },
  {
    id: 'mantid', name: 'Mantid', plan: 'mantis',
    blurb: 'A blur of blades. Glass cannon with brutal crits.',
    tags: ['insect', 'blade'],
    base: { hp: 900, pow: 19, enr: 9, arm: 9, spd: 15, fcs: 14 },
    growth: { hp: 70, pow: 3, enr: 1, arm: 1, spd: 2, fcs: 2 },
    resist: { acid: 70, cold: 140 },
    basic: 'scythe', exclusive: ['scythe_flurry', 'phantom_step', 'decapitate'],
    colors: { body: '#2f5a2a', belly: '#a6d66a', accent: '#ff3a3a' },
  },
  {
    id: 'serpent', name: 'Sea Serpent', plan: 'serpent',
    blurb: 'Coiling sea dragon. Constricts and blasts water.',
    tags: ['aquatic', 'reptile', 'tail', 'grappler'],
    base: { hp: 1080, pow: 17, enr: 18, arm: 12, spd: 13, fcs: 12 },
    growth: { hp: 85, pow: 2, enr: 2, arm: 1, spd: 2, fcs: 1 },
    resist: { cold: 50, fire: 70, electric: 140 },
    basic: 'bite', exclusive: ['constrict', 'hydro_cannon', 'shed_skin'],
    colors: { body: '#1f4a5a', belly: '#8ac9c9', accent: '#ffe066' },
  },
  {
    id: 'mecha', name: 'Mecha', plan: 'mech',
    blurb: 'Humanity built one too. Missiles, plasma, no mercy.',
    tags: ['armored', 'mech'],
    base: { hp: 1100, pow: 17, enr: 21, arm: 17, spd: 10, fcs: 16 },
    growth: { hp: 80, pow: 2, enr: 3, arm: 2, spd: 1, fcs: 2 },
    resist: { electric: 125, acid: 120, sonic: 70, cold: 80 },
    basic: 'rocket_punch', exclusive: ['missile_barrage', 'plasma_lance', 'overclock'],
    colors: { body: '#5a6270', belly: '#c8ccd2', accent: '#3aa0ff' },
  },
  {
    id: 'golem', name: 'Magma Golem', plan: 'golem',
    blurb: 'Living volcano. Slow, crushing, burns everything.',
    tags: ['armored', 'elemental', 'stone'],
    base: { hp: 1080, pow: 17, enr: 14, arm: 16, spd: 5, fcs: 9 },
    growth: { hp: 95, pow: 2, enr: 2, arm: 2, spd: 1, fcs: 1 },
    resist: { fire: 30, physical: 90, cold: 150 },
    basic: 'stone_fist', exclusive: ['magma_fist', 'stone_skin', 'seismic_quake'],
    colors: { body: '#2e2a28', belly: '#4a403a', accent: '#ff5a1a' },
  },
  {
    id: 'hydra', name: 'Hydra', plan: 'hydra',
    blurb: 'Three heads, three breaths, regrows what it loses.',
    tags: ['reptile', 'beast', 'tail', 'multihead'],
    base: { hp: 1220, pow: 18, enr: 18, arm: 12, spd: 9, fcs: 13 },
    growth: { hp: 90, pow: 2, enr: 2, arm: 1, spd: 1, fcs: 2 },
    resist: { acid: 70 },
    basic: 'bite', exclusive: ['triple_bite', 'regrowth', 'breath_barrage'],
    colors: { body: '#3a2a5a', belly: '#9a8ac9', accent: '#9aff3a' },
  },
  {
    id: 'arachnid', name: 'Arachnid', plan: 'spider',
    blurb: 'Eight legs, endless venom. Webs, poisons, wears you down.',
    tags: ['insect', 'grappler', 'venomous'],
    base: { hp: 1080, pow: 17, enr: 16, arm: 11, spd: 14, fcs: 15 },
    growth: { hp: 80, pow: 2, enr: 2, arm: 1, spd: 2, fcs: 2 },
    resist: { acid: 40, fire: 130 },
    basic: 'fang', exclusive: ['web_snare', 'venom_fang', 'brood_swarm'],
    colors: { body: '#1e1a1a', belly: '#5a1a1a', accent: '#ff2a2a' },
  },
  {
    id: 'stormray', name: 'Storm Ray', plan: 'ray',
    blurb: 'A manta the size of a thunderhead. Fast, electric, aerial.',
    tags: ['aquatic', 'flyer', 'elemental'],
    base: { hp: 880, pow: 12, enr: 19, arm: 9, spd: 15, fcs: 13 },
    growth: { hp: 75, pow: 1, enr: 3, arm: 1, spd: 2, fcs: 2 },
    resist: { electric: 30, cold: 80, acid: 130 },
    basic: 'barb', exclusive: ['thunder_lash', 'storm_call', 'static_field'],
    colors: { body: '#1a2a4a', belly: '#d8e4f0', accent: '#fff23a' },
  },
];

// ---------- Rare Wild-only enemies ----------
// Never playable. They only show up as rare Wild encounters in the Arena. Each has a unique
// passive trait that the battle engine implements (see PASSIVES).
export const PASSIVES = {
  last_stand:     { name: 'Heroic Resolve',       desc: 'The first blow that would defeat it leaves it at 1 HP instead, and it becomes Enraged.' },
  tag_team:       { name: 'Tag Team',             desc: 'The first time it drops below 50% HP, a fresh member tags in: heals 30% and clears debuffs.' },
  swarm:          { name: 'Overwhelming Numbers', desc: 'Too many units to pin down: immune to Stun and Bound.' },
  transformation: { name: 'Transformation',       desc: 'The first time it drops below 50% HP, it transforms: clears debuffs, full energy, Shielded, Charged and Hasted.' },
};

export const WILD_FRAMES = [
  {
    id: 'superhero', name: 'Superhero', plan: 'hero', wildOnly: true, rarity: 30,
    blurb: 'A caped champion of humanity, tiny next to a Kaiju but hits like a freight train. Refuses to stay down.',
    tags: ['human', 'hero', 'flyer'], passive: 'last_stand',
    base: { hp: 880, pow: 16, enr: 14, arm: 11, spd: 14, fcs: 12 },
    growth: { hp: 85, pow: 3, enr: 2, arm: 1, spd: 2, fcs: 1 },
    resist: { physical: 85, acid: 120 },
    basic: 'hero_punch', exclusive: ['heat_vision', 'hurricane_spin', 'heroic_charge'],
    colors: { body: '#1d3fa8', belly: '#c8102e', accent: '#ffd23f' },
  },
  {
    id: 'heroteam', name: 'Superhero Team', plan: 'heroteam', wildOnly: true, rarity: 10,
    blurb: 'Five human-sized heroes fighting as one. Coordinated strikes, and someone always tags in.',
    tags: ['human', 'hero'], passive: 'tag_team',
    base: { hp: 1000, pow: 16, enr: 16, arm: 11, spd: 13, fcs: 14 },
    growth: { hp: 90, pow: 2, enr: 2, arm: 1, spd: 1, fcs: 2 },
    resist: {},
    basic: 'team_strike', exclusive: ['combo_formation', 'shield_wall', 'combined_beam'],
    colors: { body: '#2b2d42', belly: '#edf2f4', accent: '#ef233c' },
  },
  {
    id: 'army', name: 'Army', plan: 'army', wildOnly: true, rarity: 35,
    blurb: 'Tanks, artillery and jets. Endless firepower, impossible to pin down.',
    tags: ['human', 'mech', 'armored'], passive: 'swarm',
    base: { hp: 1300, pow: 17, enr: 16, arm: 18, spd: 8, fcs: 14 },
    growth: { hp: 95, pow: 2, enr: 2, arm: 2, spd: 1, fcs: 1 },
    resist: { physical: 90, acid: 110, electric: 120 },
    basic: 'rifle_volley', exclusive: ['artillery_barrage', 'air_strike', 'tank_column'],
    colors: { body: '#4b5320', belly: '#8a8d6b', accent: '#ff8c1a' },
  },
  {
    id: 'magicalgirl', name: 'Magical Girl', plan: 'magicalgirl', wildOnly: true, rarity: 25,
    blurb: "Human-sized and sparkly, she barely reaches a Kaiju's ankle. Fragile at first, terrifying once she transforms.",
    tags: ['human', 'magical', 'flyer'], passive: 'transformation',
    base: { hp: 840, pow: 11, enr: 17, arm: 9, spd: 14, fcs: 13 },
    growth: { hp: 75, pow: 1, enr: 3, arm: 1, spd: 2, fcs: 2 },
    resist: { electric: 70, sonic: 70, acid: 130 },
    basic: 'wand_bolt', exclusive: ['starlight_beam', 'ribbon_bind', 'heart_barrier'],
    colors: { body: '#ff8fc7', belly: '#ffffff', accent: '#8ef6ff' },
  },
];
export const RARE_FRAME_IDS = WILD_FRAMES.map(f => f.id);

export const FRAME_BY_ID = Object.fromEntries([...FRAMES, ...WILD_FRAMES].map(f => [f.id, f]));

// Includes Fine Tune steps (kaiju.tune), before injuries.
export function effectiveStats(kaiju) {
  const f = FRAME_BY_ID[kaiju.frame];
  const out = {};
  for (const s of STATS) out[s] = f.base[s] + f.growth[s] * (kaiju.levels?.[s] || 0);
  return applyTune(out, kaiju.tune);
}

// 0..3 visual tier per stat, used by the renderer to add/grow parts.
export function visualTiers(kaiju) {
  const out = {};
  for (const s of STATS) {
    const l = kaiju.levels?.[s] || 0;
    out[s] = l >= 9 ? 3 : l >= 6 ? 2 : l >= 3 ? 1 : 0;
  }
  return out;
}
