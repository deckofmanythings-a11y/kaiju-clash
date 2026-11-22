// Ability catalogue.
// Availability: `basic` (frame's free attack), `frames: [...]` (exclusive), `tags: [...]` (any frame with ANY of the tags),
// or `universal: true`.
//
// kind: 'attack' | 'buff' | 'heal' | 'debuff'
// stat: which attacker stat scales damage ('pow' physical, 'enr' energy, 'arm' for armor-scaling rams)
// power: percent multiplier (100 = baseline). hits: number of hits (each rolls accuracy + crit separately).
// cost: energy cost (0-100 meter). cd: cooldown in turns after use. acc: base accuracy %.
// effects: applied on hit (target) or on use (self). chance in %. Status ids are defined in STATUSES below.
// fx: renderer hint — 'melee' | 'beam' | 'projectile' | 'aoe' | 'grab' | 'roar' | 'heal' | 'shield' | 'buff' | 'multi'
// price: Biomass (BIO) to unlock. Basics are free.
// vis: optional cosmetic the renderer adds when this ability is equipped.

export const DMG_TYPES = ['physical', 'fire', 'electric', 'acid', 'cold', 'sonic'];

export const STATUSES = {
  burn:     { name: 'Burning',     dot: 'fire',     desc: 'Takes fire damage each turn.' },
  poison:   { name: 'Poisoned',    dot: 'acid',     desc: 'Takes acid damage each turn.' },
  bleed:    { name: 'Bleeding',    dot: 'physical', desc: 'Takes physical damage each turn.' },
  stun:     { name: 'Stunned',     desc: 'Loses its next turn.' },
  bind:     { name: 'Bound',       desc: 'Held fast: 50% chance to lose each turn, takes extra hits.' },
  slow:     { name: 'Slowed',      desc: 'Speed reduced 35%.' },
  blind:    { name: 'Blinded',     desc: 'Accuracy reduced 30.' },
  weaken:   { name: 'Weakened',    desc: 'Might and Energy reduced 25%.' },
  sunder:   { name: 'Sundered',    desc: 'Armor reduced 40%.' },
  enrage:   { name: 'Enraged',     desc: 'Might +30%.' },
  charged:  { name: 'Charged',     desc: 'Energy +30%.' },
  fortify:  { name: 'Fortified',   desc: 'Armor +50%.' },
  haste:    { name: 'Hasted',      desc: 'Speed +35%.' },
  evasive:  { name: 'Evasive',     desc: 'Dodge +25.' },
  regen:    { name: 'Regenerating',desc: 'Heals each turn.' },
  shield:   { name: 'Shielded',    desc: 'Absorbs damage.' },
};

const A = [
  // ---------- Basic attacks (free, always equipped) ----------
  { id: 'claw',          name: 'Claw Rake',     basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 100, acc: 95, fx: 'melee' },
  { id: 'punch',         name: 'Haymaker',      basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 105, acc: 92, fx: 'melee' },
  { id: 'tentacle_slap', name: 'Tentacle Lash', basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 110,  acc: 98, fx: 'melee' },
  { id: 'wing_buffet',   name: 'Wing Buffet',   basic: true, kind: 'attack', type: 'sonic',    stat: 'enr', power: 90,  acc: 98, fx: 'aoe' },
  { id: 'shell_bash',    name: 'Shell Bash',    basic: true, kind: 'attack', type: 'physical', stat: 'arm', power: 95,  acc: 92, fx: 'melee' },
  { id: 'scythe',        name: 'Scythe Slash',  basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 100, acc: 95, fx: 'melee' },
  { id: 'bite',          name: 'Bite',          basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 100, acc: 94, fx: 'melee' },
  { id: 'rocket_punch',  name: 'Rocket Punch',  basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 95,  acc: 99, fx: 'projectile' },
  { id: 'stone_fist',    name: 'Stone Fist',    basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 105, acc: 90, fx: 'melee' },
  { id: 'fang',          name: 'Fang Strike',   basic: true, kind: 'attack', type: 'acid',     stat: 'pow', power: 90,  acc: 96, fx: 'melee', effects: [{ status: 'poison', chance: 15, turns: 2, power: 30 }] },
  { id: 'barb',          name: 'Tail Barb',     basic: true, kind: 'attack', type: 'electric', stat: 'enr', power: 90,  acc: 97, fx: 'projectile' },

  // ---------- Universal ----------
  { id: 'roar',        name: 'Titanic Roar',  universal: true, kind: 'debuff', cost: 25, cd: 4, acc: 100, fx: 'roar', price: 150,
    effects: [{ status: 'weaken', chance: 100, turns: 2 }], vis: 'jaw' },
  { id: 'regenerate',  name: 'Regenerate',    universal: true, kind: 'heal', cost: 35, cd: 5, heal: 12, fx: 'heal', price: 200 },
  { id: 'rampage',     name: 'Rampage',       universal: true, kind: 'attack', type: 'physical', stat: 'pow', power: 170, acc: 80, cost: 40, cd: 3, fx: 'melee', price: 250,
    selfEffects: [{ status: 'sunder', chance: 100, turns: 2 }] },
  { id: 'brace',       name: 'Brace',         universal: true, kind: 'buff', cost: 20, cd: 4, fx: 'shield', price: 150,
    selfEffects: [{ status: 'shield', chance: 100, turns: 3, power: 12 }] },
  { id: 'adrenal',     name: 'Adrenal Surge', universal: true, kind: 'buff', cost: 0, cd: 6, fx: 'buff', price: 200, energyGain: 40,
    selfEffects: [{ status: 'enrage', chance: 100, turns: 2 }] },
  { id: 'stomp',       name: 'Quake Stomp',   universal: true, kind: 'attack', type: 'physical', stat: 'pow', power: 120, acc: 88, cost: 30, cd: 3, fx: 'aoe', price: 200,
    effects: [{ status: 'stun', chance: 15, turns: 1 }] },

  // ---------- Tag-shared ----------
  { id: 'tail_sweep',   name: 'Tail Sweep',     tags: ['tail'], kind: 'attack', type: 'physical', stat: 'pow', power: 130, acc: 90, cost: 25, cd: 2, fx: 'melee', price: 200,
    effects: [{ status: 'slow', chance: 50, turns: 2 }], vis: 'tail_club' },
  { id: 'tidal_surge',  name: 'Tidal Surge',    tags: ['aquatic'], kind: 'attack', type: 'cold', stat: 'enr', power: 140, acc: 92, cost: 35, cd: 3, fx: 'aoe', price: 250,
    effects: [{ status: 'slow', chance: 60, turns: 2 }] },
  { id: 'dive_bomb',    name: 'Dive Bomb',      tags: ['flyer'], kind: 'attack', type: 'physical', stat: 'spd', power: 190, acc: 85, cost: 40, cd: 3, fx: 'melee', price: 300 },
  { id: 'gale_wings',   name: 'Gale Wings',     tags: ['flyer'], kind: 'debuff', cost: 20, cd: 4, acc: 95, fx: 'aoe', price: 150,
    effects: [{ status: 'blind', chance: 100, turns: 2 }] },
  { id: 'chitin',       name: 'Chitin Harden',  tags: ['insect'], kind: 'buff', cost: 25, cd: 4, fx: 'shield', price: 200,
    selfEffects: [{ status: 'fortify', chance: 100, turns: 3 }] },
  { id: 'venom_sting',  name: 'Venom Sting',    tags: ['insect', 'venomous'], kind: 'attack', type: 'acid', stat: 'pow', power: 80, acc: 95, cost: 20, cd: 2, fx: 'melee', price: 200,
    effects: [{ status: 'poison', chance: 80, turns: 3, power: 45 }], vis: 'stinger' },
  { id: 'savage_bite',  name: 'Savage Bite',    tags: ['beast'], kind: 'attack', type: 'physical', stat: 'pow', power: 120, acc: 92, cost: 25, cd: 2, fx: 'melee', price: 200,
    effects: [{ status: 'bleed', chance: 60, turns: 3, power: 40 }], vis: 'tusks' },
  { id: 'fortify',      name: 'Bulwark',        tags: ['armored', 'stone'], kind: 'buff', cost: 20, cd: 4, fx: 'shield', price: 200,
    selfEffects: [{ status: 'fortify', chance: 100, turns: 3 }] },
  { id: 'ram',          name: 'Battering Ram',  tags: ['armored'], kind: 'attack', type: 'physical', stat: 'arm', power: 150, acc: 90, cost: 30, cd: 3, fx: 'melee', price: 250,
    effects: [{ status: 'stun', chance: 20, turns: 1 }], vis: 'ram_horn' },
  { id: 'overload',     name: 'Elemental Overload', tags: ['elemental'], kind: 'attack', type: 'fire', stat: 'enr', power: 230, acc: 88, cost: 60, cd: 5, fx: 'beam', price: 400,
    selfDamage: 6, vis: 'core' },
  { id: 'grapple',      name: 'Grapple Throw',  tags: ['grappler'], kind: 'attack', type: 'physical', stat: 'pow', power: 150, acc: 85, cost: 35, cd: 3, fx: 'grab', price: 250,
    effects: [{ status: 'stun', chance: 30, turns: 1 }] },

  // ---------- Exclusive: Saurian ----------
  { id: 'atomic_breath', name: 'Atomic Breath', frames: ['saurian'], kind: 'attack', type: 'fire', stat: 'enr', power: 210, acc: 92, cost: 50, cd: 3, fx: 'beam', price: 0,
    effects: [{ status: 'burn', chance: 40, turns: 2, power: 50 }], vis: 'dorsal_glow' },
  { id: 'radiant_pulse', name: 'Radiant Pulse', frames: ['saurian'], kind: 'attack', type: 'fire', stat: 'enr', power: 120, acc: 100, cost: 30, cd: 3, fx: 'aoe', price: 300,
    effects: [{ status: 'burn', chance: 70, turns: 3, power: 40 }] },
  { id: 'thermal_hide',  name: 'Thermal Hide',  frames: ['saurian'], kind: 'buff', cost: 25, cd: 5, fx: 'buff', price: 300,
    selfEffects: [{ status: 'regen', chance: 100, turns: 3, power: 5 }, { status: 'fortify', chance: 100, turns: 2 }] },

  // ---------- Exclusive: Titan Ape ----------
  { id: 'hammer_fists',  name: 'Hammer Fists',  frames: ['ape'], kind: 'attack', type: 'physical', stat: 'pow', power: 85, hits: 2, acc: 92, cost: 25, cd: 2, fx: 'multi', price: 0 },
  { id: 'chest_pound',   name: 'Chest Pound',   frames: ['ape'], kind: 'buff', cost: 0, cd: 5, energyGain: 30, fx: 'roar', price: 250,
    selfEffects: [{ status: 'enrage', chance: 100, turns: 3 }] },
  { id: 'mountain_toss', name: 'Mountain Toss', frames: ['ape'], kind: 'attack', type: 'physical', stat: 'pow', power: 240, acc: 80, cost: 55, cd: 4, fx: 'projectile', price: 400,
    effects: [{ status: 'stun', chance: 25, turns: 1 }] },

  // ---------- Exclusive: Kraken ----------
  { id: 'ink_cloud',        name: 'Ink Cloud',        frames: ['kraken'], kind: 'debuff', cost: 20, cd: 4, acc: 100, fx: 'aoe', price: 0,
    effects: [{ status: 'blind', chance: 100, turns: 3 }], selfEffects: [{ status: 'evasive', chance: 100, turns: 2 }] },
  { id: 'crushing_embrace', name: 'Crushing Embrace', frames: ['kraken'], kind: 'attack', type: 'physical', stat: 'pow', power: 110, acc: 90, cost: 35, cd: 3, fx: 'grab', price: 300,
    effects: [{ status: 'bind', chance: 80, turns: 2 }, { status: 'bleed', chance: 50, turns: 2, power: 40 }] },
  { id: 'abyssal_pull',     name: 'Abyssal Pull',     frames: ['kraken'], kind: 'attack', type: 'cold', stat: 'enr', power: 170, acc: 90, cost: 45, cd: 3, fx: 'beam', price: 350,
    effects: [{ status: 'slow', chance: 70, turns: 2 }] },

  // ---------- Exclusive: Lepidra ----------
  { id: 'prismatic_scales', name: 'Prismatic Scales', frames: ['lepidra'], kind: 'buff', cost: 25, cd: 4, fx: 'shield', price: 0,
    selfEffects: [{ status: 'shield', chance: 100, turns: 3, power: 10 }, { status: 'evasive', chance: 100, turns: 3 }] },
  { id: 'toxic_dust',       name: 'Toxic Dust',       frames: ['lepidra'], kind: 'attack', type: 'acid', stat: 'enr', power: 70, acc: 100, cost: 25, cd: 3, fx: 'aoe', price: 300,
    effects: [{ status: 'poison', chance: 100, turns: 4, power: 45 }] },
  { id: 'lumen_beam',       name: 'Lumen Beam',       frames: ['lepidra'], kind: 'attack', type: 'electric', stat: 'enr', power: 200, acc: 94, cost: 50, cd: 3, fx: 'beam', price: 350,
    effects: [{ status: 'blind', chance: 30, turns: 2 }] },

  // ---------- Exclusive: Carapace ----------
  { id: 'shell_spin', name: 'Shell Spin', frames: ['carapace'], kind: 'attack', type: 'physical', stat: 'arm', power: 60, hits: 3, acc: 90, cost: 30, cd: 3, fx: 'multi', price: 0 },
  { id: 'turtle_up',  name: 'Turtle Up',  frames: ['carapace'], kind: 'buff', cost: 30, cd: 5, fx: 'shield', price: 300,
    selfEffects: [{ status: 'shield', chance: 100, turns: 3, power: 20 }, { status: 'regen', chance: 100, turns: 3, power: 4 }] },
  { id: 'magma_vent', name: 'Magma Vent', frames: ['carapace'], kind: 'attack', type: 'fire', stat: 'enr', power: 160, acc: 92, cost: 40, cd: 3, fx: 'aoe', price: 350,
    effects: [{ status: 'burn', chance: 60, turns: 3, power: 45 }], vis: 'vents' },

  // ---------- Exclusive: Mantid ----------
  { id: 'scythe_flurry', name: 'Scythe Flurry', frames: ['mantid'], kind: 'attack', type: 'physical', stat: 'pow', power: 55, hits: 4, acc: 88, cost: 30, cd: 2, critBonus: 10, fx: 'multi', price: 0 },
  { id: 'phantom_step',  name: 'Phantom Step',  frames: ['mantid'], kind: 'buff', cost: 20, cd: 4, fx: 'buff', price: 250,
    selfEffects: [{ status: 'evasive', chance: 100, turns: 2 }, { status: 'haste', chance: 100, turns: 2 }] },
  { id: 'decapitate',    name: 'Decapitate',    frames: ['mantid'], kind: 'attack', type: 'physical', stat: 'pow', power: 160, acc: 85, cost: 45, cd: 4, critBonus: 25, execute: 30, fx: 'melee', price: 400 },

  // ---------- Exclusive: Sea Serpent ----------
  { id: 'constrict',    name: 'Constrict',    frames: ['serpent'], kind: 'attack', type: 'physical', stat: 'pow', power: 100, acc: 90, cost: 30, cd: 3, fx: 'grab', price: 0,
    effects: [{ status: 'bind', chance: 90, turns: 2 }, { status: 'sunder', chance: 50, turns: 2 }] },
  { id: 'hydro_cannon', name: 'Hydro Cannon', frames: ['serpent'], kind: 'attack', type: 'cold', stat: 'enr', power: 200, acc: 92, cost: 50, cd: 3, fx: 'beam', price: 350 },
  { id: 'shed_skin',    name: 'Shed Skin',    frames: ['serpent'], kind: 'heal', cost: 30, cd: 5, heal: 10, cleanse: true, fx: 'heal', price: 300 },

  // ---------- Exclusive: Mecha ----------
  { id: 'missile_barrage', name: 'Missile Barrage', frames: ['mecha'], kind: 'attack', type: 'fire', stat: 'enr', power: 58, hits: 5, acc: 88, cost: 40, cd: 3, fx: 'projectile', price: 0, vis: 'missile_pods' },
  { id: 'plasma_lance',    name: 'Plasma Lance',    frames: ['mecha'], kind: 'attack', type: 'electric', stat: 'enr', power: 190, acc: 95, cost: 50, cd: 3, fx: 'beam', price: 350,
    effects: [{ status: 'sunder', chance: 70, turns: 2 }] },
  { id: 'overclock',       name: 'Overclock',       frames: ['mecha'], kind: 'buff', cost: 0, cd: 6, energyGain: 50, selfDamage: 5, fx: 'buff', price: 300,
    selfEffects: [{ status: 'haste', chance: 100, turns: 3 }, { status: 'charged', chance: 100, turns: 3 }] },

  // ---------- Exclusive: Magma Golem ----------
  { id: 'magma_fist',    name: 'Magma Fist',    frames: ['golem'], kind: 'attack', type: 'fire', stat: 'pow', power: 150, acc: 90, cost: 30, cd: 2, fx: 'melee', price: 0,
    effects: [{ status: 'burn', chance: 60, turns: 3, power: 45 }] },
  { id: 'stone_skin',    name: 'Stone Skin',    frames: ['golem'], kind: 'buff', cost: 25, cd: 5, fx: 'shield', price: 250,
    selfEffects: [{ status: 'fortify', chance: 100, turns: 3 }, { status: 'shield', chance: 100, turns: 3, power: 10 }] },
  { id: 'seismic_quake', name: 'Seismic Quake', frames: ['golem'], kind: 'attack', type: 'physical', stat: 'pow', power: 170, acc: 88, cost: 50, cd: 4, fx: 'aoe', price: 400,
    effects: [{ status: 'stun', chance: 35, turns: 1 }] },

  // ---------- Exclusive: Hydra ----------
  { id: 'triple_bite',    name: 'Triple Bite',    frames: ['hydra'], kind: 'attack', type: 'physical', stat: 'pow', power: 65, hits: 3, acc: 92, cost: 30, cd: 2, fx: 'multi', price: 0,
    effects: [{ status: 'bleed', chance: 25, turns: 2, power: 30 }] },
  { id: 'regrowth',       name: 'Regrowth',       frames: ['hydra'], kind: 'heal', cost: 40, cd: 5, heal: 18, fx: 'heal', price: 300,
    selfEffects: [{ status: 'regen', chance: 100, turns: 2, power: 4 }] },
  { id: 'breath_barrage', name: 'Breath Barrage', frames: ['hydra'], kind: 'attack', type: 'random', randomTypes: ['fire', 'acid', 'cold'], stat: 'enr', power: 75, hits: 3, acc: 92, cost: 45, cd: 3, fx: 'beam', price: 400 },

  // ---------- Exclusive: Arachnid ----------
  { id: 'web_snare',   name: 'Web Snare',   frames: ['arachnid'], kind: 'debuff', cost: 25, cd: 4, acc: 92, fx: 'projectile', price: 0,
    effects: [{ status: 'bind', chance: 80, turns: 2 }, { status: 'slow', chance: 100, turns: 3 }] },
  { id: 'venom_fang',  name: 'Venom Fang',  frames: ['arachnid'], kind: 'attack', type: 'acid', stat: 'pow', power: 140, acc: 92, cost: 35, cd: 3, fx: 'melee', price: 300,
    effects: [{ status: 'poison', chance: 100, turns: 4, power: 55 }] },
  { id: 'brood_swarm', name: 'Brood Swarm', frames: ['arachnid'], kind: 'attack', type: 'acid', stat: 'enr', power: 35, hits: 6, acc: 80, cost: 40, cd: 3, fx: 'multi', price: 350 },

  // ---------- Exclusive: Storm Ray ----------
  { id: 'thunder_lash',  name: 'Thunder Lash',  frames: ['stormray'], kind: 'attack', type: 'electric', stat: 'enr', power: 125, acc: 92, cost: 35, cd: 2, fx: 'beam', price: 0,
    effects: [{ status: 'stun', chance: 15, turns: 1 }] },
  { id: 'storm_call',    name: 'Storm Call',    frames: ['stormray'], kind: 'attack', type: 'electric', stat: 'enr', power: 190, acc: 85, cost: 55, cd: 4, fx: 'aoe', price: 400 },
  { id: 'static_field',  name: 'Static Field',  frames: ['stormray'], kind: 'buff', cost: 25, cd: 4, fx: 'shield', price: 250,
    selfEffects: [{ status: 'shield', chance: 100, turns: 3, power: 12 }, { status: 'charged', chance: 100, turns: 3 }] },

  // ---------- Exclusive: Giant Baby ----------
  { id: 'tiny_fist',    name: 'Tiny Fist',     basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 100, acc: 93, fx: 'melee' },
  { id: 'tantrum',      name: 'Tantrum',       frames: ['giantbaby'], kind: 'attack', type: 'physical', stat: 'pow', power: 50, hits: 4, acc: 85, cost: 30, cd: 2, fx: 'multi', price: 0 },
  { id: 'wail',         name: 'Ear-Splitting Wail', frames: ['giantbaby'], kind: 'attack', type: 'sonic', stat: 'enr', power: 115, acc: 100, cost: 30, cd: 3, fx: 'roar', price: 300,
    effects: [{ status: 'stun', chance: 25, turns: 1 }, { status: 'weaken', chance: 50, turns: 2 }] },
  { id: 'nap_time',     name: 'Nap Time',      frames: ['giantbaby'], kind: 'heal', cost: 35, cd: 5, heal: 14, fx: 'heal', price: 300,
    selfEffects: [{ status: 'regen', chance: 100, turns: 3, power: 4 }] },

  // ---------- Exclusive: Colossal Woman ----------
  { id: 'backhand',     name: 'Backhand',      basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 100, acc: 95, fx: 'melee' },
  { id: 'car_toss',     name: 'Car Toss',      frames: ['colossalwoman'], kind: 'attack', type: 'physical', stat: 'pow', power: 185, acc: 88, cost: 40, cd: 3, fx: 'projectile', price: 0,
    effects: [{ status: 'stun', chance: 20, turns: 1 }] },
  { id: 'heel_stomp',   name: 'Heel Stomp',    frames: ['colossalwoman'], kind: 'attack', type: 'physical', stat: 'pow', power: 135, acc: 92, cost: 30, cd: 2, fx: 'melee', price: 300,
    effects: [{ status: 'sunder', chance: 60, turns: 2 }] },
  { id: 'scorned_fury', name: 'Scorned Fury',  frames: ['colossalwoman'], kind: 'buff', cost: 0, cd: 5, energyGain: 30, fx: 'roar', price: 350,
    selfEffects: [{ status: 'enrage', chance: 100, turns: 3 }, { status: 'haste', chance: 100, turns: 2 }] },

  // ---------- Exclusive: Mecha Saurian ----------
  { id: 'drill_claw',      name: 'Drill Claw',      basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 100, acc: 95, fx: 'melee' },
  { id: 'maser_breath',    name: 'Maser Breath',    frames: ['mechasaurian'], kind: 'attack', type: 'electric', stat: 'enr', power: 200, acc: 93, cost: 50, cd: 3, fx: 'beam', price: 0,
    effects: [{ status: 'stun', chance: 15, turns: 1 }] },
  { id: 'finger_missiles', name: 'Finger Missiles', frames: ['mechasaurian'], kind: 'attack', type: 'fire', stat: 'enr', power: 62, hits: 4, acc: 88, cost: 35, cd: 3, fx: 'projectile', price: 300 },
  { id: 'reactor_shield',  name: 'Reactor Shield',  frames: ['mechasaurian'], kind: 'buff', cost: 30, cd: 5, fx: 'shield', price: 300,
    selfEffects: [{ status: 'shield', chance: 100, turns: 3, power: 14 }, { status: 'charged', chance: 100, turns: 2 }] },

  // ---------- Exclusive: Vespa Rex ----------
  { id: 'mandible',        name: 'Mandible Snap',   basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 95, acc: 96, fx: 'melee' },
  { id: 'stinger_lance',   name: 'Stinger Lance',   frames: ['wasp'], kind: 'attack', type: 'acid', stat: 'pow', power: 150, acc: 90, cost: 35, cd: 3, critBonus: 10, fx: 'melee', price: 0,
    effects: [{ status: 'poison', chance: 100, turns: 3, power: 50 }] },
  { id: 'hive_swarm',      name: 'Hive Swarm',      frames: ['wasp'], kind: 'attack', type: 'acid', stat: 'enr', power: 32, hits: 6, acc: 85, cost: 40, cd: 3, fx: 'multi', price: 300 },
  { id: 'alarm_pheromone', name: 'Alarm Pheromone', frames: ['wasp'], kind: 'buff', cost: 20, cd: 4, fx: 'buff', price: 300,
    selfEffects: [{ status: 'enrage', chance: 100, turns: 2 }, { status: 'haste', chance: 100, turns: 2 }] },

  // ---------- Exclusive: Sandreaper ----------
  { id: 'pincer',      name: 'Pincer Snip',  basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 100, acc: 94, fx: 'melee' },
  { id: 'tail_strike', name: 'Tail Strike',  frames: ['scorpion'], kind: 'attack', type: 'acid', stat: 'pow', power: 140, acc: 90, cost: 40, cd: 3, fx: 'melee', price: 0,
    effects: [{ status: 'poison', chance: 80, turns: 3, power: 50 }, { status: 'stun', chance: 15, turns: 1 }] },
  { id: 'pincer_lock', name: 'Pincer Lock',  frames: ['scorpion'], kind: 'attack', type: 'physical', stat: 'pow', power: 100, acc: 90, cost: 30, cd: 3, fx: 'grab', price: 300,
    effects: [{ status: 'bind', chance: 80, turns: 2 }, { status: 'bleed', chance: 40, turns: 2, power: 40 }] },
  { id: 'burrow',      name: 'Burrow',       frames: ['scorpion'], kind: 'buff', cost: 25, cd: 4, fx: 'shield', price: 250,
    selfEffects: [{ status: 'evasive', chance: 100, turns: 2 }, { status: 'fortify', chance: 100, turns: 2 }] },

  // ---------- Exclusive: Archfiend ----------
  { id: 'infernal_rend', name: 'Infernal Rend', basic: true, kind: 'attack', type: 'fire', stat: 'pow', power: 95, acc: 95, fx: 'melee',
    effects: [{ status: 'burn', chance: 15, turns: 2, power: 30 }] },
  { id: 'hellfire',      name: 'Hellfire',      frames: ['demon'], kind: 'attack', type: 'fire', stat: 'enr', power: 135, acc: 92, cost: 40, cd: 3, fx: 'aoe', price: 0,
    effects: [{ status: 'burn', chance: 70, turns: 3, power: 45 }] },
  { id: 'soul_drain',    name: 'Soul Drain',    frames: ['demon'], kind: 'attack', type: 'sonic', stat: 'enr', power: 140, acc: 94, cost: 40, cd: 3, drain: 50, fx: 'beam', price: 350 },
  { id: 'blood_pact',    name: 'Blood Pact',    frames: ['demon'], kind: 'buff', cost: 0, cd: 6, energyGain: 40, selfDamage: 8, fx: 'buff', price: 300,
    selfEffects: [{ status: 'enrage', chance: 100, turns: 3 }, { status: 'charged', chance: 100, turns: 3 }] },

  // ---------- Exclusive: Dire Lycan ----------
  { id: 'maul',       name: 'Maul',       basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 100, acc: 94, fx: 'melee',
    effects: [{ status: 'bleed', chance: 20, turns: 2, power: 30 }] },
  { id: 'frenzy',     name: 'Frenzy',     frames: ['werewolf'], kind: 'attack', type: 'physical', stat: 'pow', power: 50, hits: 4, acc: 88, cost: 30, cd: 2, fx: 'multi', price: 0,
    effects: [{ status: 'bleed', chance: 30, turns: 2, power: 30 }] },
  { id: 'moon_howl',  name: 'Moon Howl',  frames: ['werewolf'], kind: 'buff', cost: 25, cd: 5, fx: 'roar', price: 300,
    selfEffects: [{ status: 'enrage', chance: 100, turns: 3 }, { status: 'regen', chance: 100, turns: 3, power: 4 }] },
  { id: 'throat_rip', name: 'Throat Rip', frames: ['werewolf'], kind: 'attack', type: 'physical', stat: 'pow', power: 150, acc: 88, cost: 40, cd: 4, critBonus: 15, execute: 25, fx: 'melee', price: 400,
    effects: [{ status: 'bleed', chance: 60, turns: 3, power: 40 }] },

  // ---------- Rare Wild-only enemies ----------
  { id: 'hero_punch',     name: 'Hero Punch',       basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 105, acc: 96, fx: 'melee' },
  { id: 'heat_vision',    name: 'Heat Vision',      frames: ['superhero'], kind: 'attack', type: 'fire', stat: 'enr', power: 190, acc: 96, cost: 45, cd: 3, fx: 'beam', price: 300,
    effects: [{ status: 'burn', chance: 40, turns: 2, power: 40 }] },
  { id: 'hurricane_spin', name: 'Hurricane Spin',   frames: ['superhero'], kind: 'attack', type: 'physical', stat: 'spd', power: 60, hits: 4, acc: 90, cost: 35, cd: 3, fx: 'multi', price: 300,
    effects: [{ status: 'blind', chance: 40, turns: 2 }] },
  { id: 'heroic_charge',  name: 'Heroic Charge',    frames: ['superhero'], kind: 'attack', type: 'physical', stat: 'pow', power: 210, acc: 85, cost: 50, cd: 4, fx: 'melee', price: 300,
    effects: [{ status: 'stun', chance: 30, turns: 1 }] },

  { id: 'team_strike',     name: 'Team Strike',     basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 55, hits: 2, acc: 95, fx: 'multi' },
  { id: 'combo_formation', name: 'Combo Formation', frames: ['heroteam'], kind: 'attack', type: 'physical', stat: 'pow', power: 50, hits: 5, acc: 90, cost: 40, cd: 3, fx: 'multi', price: 300 },
  { id: 'shield_wall',     name: 'Shield Wall',     frames: ['heroteam'], kind: 'buff', cost: 25, cd: 4, fx: 'shield', price: 300,
    selfEffects: [{ status: 'shield', chance: 100, turns: 3, power: 14 }, { status: 'fortify', chance: 100, turns: 2 }] },
  { id: 'combined_beam',   name: 'Combined Beam',   frames: ['heroteam'], kind: 'attack', type: 'electric', stat: 'enr', power: 230, acc: 90, cost: 60, cd: 4, fx: 'beam', price: 300 },

  { id: 'rifle_volley',      name: 'Rifle Volley',      basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 36, hits: 3, acc: 92, fx: 'projectile' },
  { id: 'artillery_barrage', name: 'Artillery Barrage', frames: ['army'], kind: 'attack', type: 'fire', stat: 'enr', power: 48, hits: 5, acc: 85, cost: 40, cd: 3, fx: 'projectile', price: 300,
    effects: [{ status: 'burn', chance: 50, turns: 2, power: 35 }] },
  { id: 'air_strike',        name: 'Air Strike',        frames: ['army'], kind: 'attack', type: 'fire', stat: 'enr', power: 200, acc: 88, cost: 50, cd: 4, fx: 'aoe', price: 300,
    effects: [{ status: 'sunder', chance: 50, turns: 2 }] },
  { id: 'tank_column',       name: 'Tank Column',       frames: ['army'], kind: 'attack', type: 'physical', stat: 'arm', power: 150, acc: 90, cost: 30, cd: 2, fx: 'projectile', price: 300,
    effects: [{ status: 'slow', chance: 50, turns: 2 }] },

  { id: 'wand_bolt',      name: 'Sparkle Bolt',     basic: true, kind: 'attack', type: 'electric', stat: 'enr', power: 95, acc: 97, fx: 'projectile' },
  { id: 'starlight_beam', name: 'Starlight Beam',   frames: ['magicalgirl'], kind: 'attack', type: 'electric', stat: 'enr', power: 220, acc: 94, cost: 55, cd: 3, fx: 'beam', price: 300 },
  { id: 'ribbon_bind',    name: 'Ribbon Bind',      frames: ['magicalgirl'], kind: 'debuff', cost: 25, cd: 4, acc: 95, fx: 'grab', price: 300,
    effects: [{ status: 'bind', chance: 85, turns: 2 }, { status: 'weaken', chance: 60, turns: 2 }] },
  { id: 'heart_barrier',  name: 'Heart Barrier',    frames: ['magicalgirl'], kind: 'buff', cost: 25, cd: 4, fx: 'shield', price: 300,
    selfEffects: [{ status: 'shield', chance: 100, turns: 3, power: 12 }, { status: 'regen', chance: 100, turns: 3, power: 4 }] },

  { id: 'gaze',          name: 'Unblinking Gaze', basic: true, kind: 'attack', type: 'fire', stat: 'enr', power: 95, acc: 100, fx: 'beam' },
  { id: 'wheel_of_fire', name: 'Wheel of Fire',   frames: ['seraph'], kind: 'attack', type: 'fire', stat: 'enr', power: 55, hits: 4, acc: 95, cost: 40, cd: 3, fx: 'multi', price: 300,
    effects: [{ status: 'burn', chance: 40, turns: 2, power: 40 }] },
  { id: 'be_not_afraid', name: 'Be Not Afraid',   frames: ['seraph'], kind: 'debuff', cost: 25, cd: 4, acc: 100, fx: 'roar', price: 300,
    effects: [{ status: 'weaken', chance: 100, turns: 2 }, { status: 'blind', chance: 100, turns: 2 }] },
  { id: 'judgment',      name: 'Judgment',        frames: ['seraph'], kind: 'attack', type: 'fire', stat: 'enr', power: 230, acc: 95, cost: 60, cd: 4, fx: 'beam', price: 300,
    effects: [{ status: 'sunder', chance: 50, turns: 2 }] },

  { id: 'radiant_blade', name: 'Radiant Blade',   basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 105, acc: 96, fx: 'melee' },
  { id: 'flaming_sword', name: 'Flaming Sword',   frames: ['angel'], kind: 'attack', type: 'fire', stat: 'pow', power: 190, acc: 92, cost: 45, cd: 3, fx: 'melee', price: 300,
    effects: [{ status: 'burn', chance: 50, turns: 2, power: 40 }] },
  { id: 'wing_shield',   name: 'Wing Shield',     frames: ['angel'], kind: 'buff', cost: 25, cd: 4, fx: 'shield', price: 300,
    selfEffects: [{ status: 'shield', chance: 100, turns: 3, power: 12 }, { status: 'evasive', chance: 100, turns: 2 }] },
  { id: 'smite',         name: 'Smite',           frames: ['angel'], kind: 'attack', type: 'electric', stat: 'enr', power: 200, acc: 92, cost: 50, cd: 3, fx: 'beam', price: 300,
    effects: [{ status: 'stun', chance: 20, turns: 1 }] },

  { id: 'tendril_lash',      name: 'Tendril Lash',      basic: true, kind: 'attack', type: 'physical', stat: 'pow', power: 105, acc: 96, fx: 'melee' },
  { id: 'eldritch_whispers', name: 'Eldritch Whispers', frames: ['elderspawn'], kind: 'attack', type: 'sonic', stat: 'enr', power: 120, acc: 100, cost: 30, cd: 3, fx: 'roar', price: 300,
    effects: [{ status: 'weaken', chance: 70, turns: 2 }, { status: 'blind', chance: 40, turns: 2 }] },
  { id: 'tentacle_maw',      name: 'Tentacle Maw',      frames: ['elderspawn'], kind: 'attack', type: 'physical', stat: 'pow', power: 120, acc: 90, cost: 35, cd: 3, fx: 'grab', price: 300,
    effects: [{ status: 'bind', chance: 80, turns: 2 }, { status: 'bleed', chance: 50, turns: 2, power: 40 }] },
  { id: 'abyssal_rebirth',   name: 'Abyssal Rebirth',   frames: ['elderspawn'], kind: 'heal', cost: 40, cd: 5, heal: 15, cleanse: true, fx: 'heal', price: 300 },

  { id: 'shard_lance',  name: 'Shard Lance',  basic: true, kind: 'attack', type: 'electric', stat: 'enr', power: 95, acc: 97, fx: 'projectile' },
  { id: 'prism_beam',   name: 'Prism Beam',   frames: ['polyhedron'], kind: 'attack', type: 'electric', stat: 'enr', power: 235, acc: 92, cost: 65, cd: 4, fx: 'beam', price: 300 },
  { id: 'shard_storm',  name: 'Shard Storm',  frames: ['polyhedron'], kind: 'attack', type: 'physical', stat: 'enr', power: 35, hits: 6, acc: 88, cost: 40, cd: 3, fx: 'multi', price: 300 },
  { id: 'reconfigure',  name: 'Reconfigure',  frames: ['polyhedron'], kind: 'buff', cost: 30, cd: 5, fx: 'shield', price: 300,
    selfEffects: [{ status: 'fortify', chance: 100, turns: 3 }, { status: 'shield', chance: 100, turns: 3, power: 15 }] },
];

export const ABILITIES = A;
export const ABILITY_BY_ID = Object.fromEntries(A.map(a => [a.id, a]));
export const MAX_EQUIPPED = 4; // plus the frame's basic attack

export function abilityAvailableTo(ability, frame) {
  if (ability.basic) return frame.basic === ability.id;
  if (ability.universal) return true;
  if (ability.frames) return ability.frames.includes(frame.id);
  if (ability.tags) return ability.tags.some(t => frame.tags.includes(t));
  return false;
}

export function abilitiesForFrame(frame) {
  return A.filter(a => !a.basic && abilityAvailableTo(a, frame));
}

export function abilityScope(ability) {
  if (ability.basic) return 'Basic';
  if (ability.universal) return 'Universal';
  if (ability.frames) return 'Exclusive';
  return 'Shared: ' + ability.tags.join('/');
}
