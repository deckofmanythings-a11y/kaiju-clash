// Player-facing changelog, newest first. Add an entry with every version bump (APP_VERSION in main.js).
// Sections: added, changed, fixed, removed (omit empty ones). Keep lines short and in players' terms.
// Rare Wild encounters are easter eggs: NEVER name them (or their signature moves) in player-facing text.
export const CHANGELOG = [
  {
    v: '1.9.0', date: '2026-11-20',
    added: [
      'Replay filters: show only wins, losses or draws, close battles, underdog wins, or fights involving a particular monster type.',
      'Replays are now tagged: 🔥 Close battle (the winner finished under 5% health), ⚡ Underdog win (the winner had at least 10% less Power), plus the monster types that fought.',
    ],
  },
  {
    v: '1.8.1', date: '2026-11-20',
    changed: ['Tidied up the changelog wording.'],
  },
  {
    v: '1.8.0', date: '2026-11-20',
    added: ["A changelog: click the version under the logo to see what's new in every update."],
    changed: ['The daily rated-fight limit now resets at midnight Pacific time instead of midnight UTC.'],
  },
  {
    v: '1.7.0', date: '2026-11-20',
    changed: [
      'Rare Wild encounters use new detailed models too, with all of their special moves intact.',
    ],
  },
  {
    v: '1.6.0', date: '2026-11-20',
    changed: [
      'All 12 Kaiju frames have brand-new detailed 3D models, with sculpted bodies, skin textures and real skeletons.',
      'Upgrades show as new armour, spikes, glowing glands and bigger claws on the new models.',
      'Models load only when they are on screen, so the game stays quick to start.',
    ],
  },
  {
    v: '1.5.0', date: '2026-11-20',
    added: [
      'Big, clear warnings when you run low on (or run out of) rated fights for the day, including a top-bar badge and a countdown to the reset.',
      'A confirmation before your first unrated fight of the day, since unrated fights pay no Biomass or Elo.',
      'Opponent cards now show when a weaker foe will pay reduced Biomass.',
      'A clear explanation when your roster is full (6 Kaiju).',
    ],
  },
  {
    v: '1.4.0', date: '2026-11-19',
    changed: [
      'Injuries now heal in minutes instead of hours (never more than 1 hour, even for the worst wounds).',
      'Instant healing costs less.',
    ],
    removed: ['Defenders no longer get injured. Your Kaiju only gets hurt in fights you start.'],
  },
  {
    v: '1.3.1', date: '2026-11-19',
    fixed: ["Hovering over an undiscovered trophy no longer gives away what it is."],
  },
  {
    v: '1.3.0', date: '2026-11-19',
    added: [
      'Rare Wild encounters: something unusual sometimes shows up in the Arena. Beat one for double Biomass and a trophy.',
      'Every set of opponents now includes at least one Wild Kaiju.',
      'Version number shown under the logo.',
    ],
    changed: [
      'Battle rewards are hidden until the fight finishes playing, so the top bar no longer spoils the result.',
      'Injuries now show on your Kaiju as scars, a limp and dimmed glow.',
    ],
    fixed: ['Password-reset links from the email now work.'],
  },
  {
    v: '1.2.0', date: '2026-11-19',
    added: [
      'Injuries: losing badly can leave your Kaiju wounded, with stat penalties until it heals.',
      'Permadeath: a Critical Kaiju that gets knocked out can die for good. The fallen are honoured in the Hangar.',
    ],
  },
  {
    v: '1.1.0', date: '2026-11-19',
    added: ['Online ladder: accounts, ranked fights against other players, Elo, the leaderboard and defense reports.'],
  },
  {
    v: '1.0.0', date: '2026-11-19',
    added: ['Kaiju Clash launches: 12 Kaiju frames, upgrades, 80 abilities, 6 arenas, automatic 3D battles and replays.'],
  },
];
