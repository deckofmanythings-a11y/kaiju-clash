// node test/rare-balance.mjs [powerPct]
// Win rate of each rare Wild enemy vs random player-frame rivals at `powerPct`% of their rating.
// Target: rares win ~60-65% at RARE_POWER_PCT (tough, but worth the double reward).
import { simulateBattle } from '../js/engine/battle.js';
import { generateRival, RARE_POWER_PCT } from '../js/data/economy.js';
import { WILD_FRAMES } from '../js/data/frames.js';
import { ENVIRONMENTS } from '../js/data/environments.js';

const powerPct = Number(process.argv[2]) || RARE_POWER_PCT;
const n = 1200;
console.log(`rare enemies at ${powerPct}% of the player's power rating, ${n} fights each`);
for (const w of WILD_FRAMES) {
  let wins = 0, passives = 0;
  for (let i = 0; i < n; i++) {
    const rating = 120 + (i % 25) * 15;
    const me = generateRival(70000 + i, rating);
    const rare = generateRival(90000 + i, Math.floor((rating * powerPct) / 100), w.id);
    const r = simulateBattle({ seed: i * 977 + 3, env: ENVIRONMENTS[i % ENVIRONMENTS.length].id, a: me, b: rare });
    if (r.winner === 1) wins++;
    if (r.events.some(e => e.t === 'passive')) passives++;
  }
  console.log(`  ${w.name.padEnd(15)} wins ${String(Math.round((100 * wins) / n)).padStart(3)}%   passive fired ${Math.round((100 * passives) / n)}%`);
}
