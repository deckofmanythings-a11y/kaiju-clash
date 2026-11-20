// Post-fight consequences: new injuries and death. Deterministic: the same battle record and
// options always give the same aftermath, so replays can show exactly what happened.
//
// rollAftermath(record, { injurable: [bool, bool], canDie: [bool, bool] })
//   -> [{ injury: id|null, died: bool, deathChance: % }, { ... }]
//
// Existing injuries are read from record.fighters[i].injuries (the snapshot as it fought).
import { createRng, hashString } from './rng.js';
import { MINOR, MAJOR, MAX_INJURIES, deathChance } from '../data/injuries.js';

// [injury chance %, of which major %] by how the fight went for that side.
export const INJURY_ODDS = {
  knockedOut:  [70, 35],
  lostOnTime:  [40, 15],
  draw:        [35, 15],
  wonBarely:   [30, 10],   // won with < 25% HP left
  wonHurt:     [12, 5],    // won with < 50% HP left
  wonClean:    [0, 0],
};

export function outcomeFor(record, i) {
  const lost = record.winner === 1 - i;
  const hpPct = Math.floor(((record.finalHp?.[i] ?? 0) * 100) / record.meta[i].maxHp);
  if (record.winner === null) return (record.finalHp?.[i] ?? 1) <= 0 ? 'knockedOut' : 'draw';
  if (lost) return record.reason === 'ko' ? 'knockedOut' : 'lostOnTime';
  return hpPct < 25 ? 'wonBarely' : hpPct < 50 ? 'wonHurt' : 'wonClean';
}

export function rollAftermath(record, { injurable = [true, true], canDie = [false, false] } = {}) {
  const rng = createRng((record.seed ^ hashString('aftermath:' + record.hash)) >>> 0);
  return [0, 1].map(i => {
    const existing = record.fighters[i].injuries || [];
    const outcome = outcomeFor(record, i);
    const [chance, majorChance] = INJURY_ODDS[outcome];
    // Always consume the same number of rolls per side so one side's result never shifts the other's.
    const rInjury = rng.int(100), rMajor = rng.int(100), rPick = rng.next(), rDeath = rng.int(100);
    const res = { outcome, injury: null, died: false, deathChance: 0 };
    if (!injurable[i]) return res;

    if (rInjury < chance && existing.length < MAX_INJURIES) {
      let pool = (rMajor < majorChance ? MAJOR : MINOR).filter(id => !existing.includes(id));
      if (!pool.length) pool = [...MINOR, ...MAJOR].filter(id => !existing.includes(id));
      if (pool.length) res.injury = pool[rPick % pool.length];
    }
    if (canDie[i] && outcome === 'knockedOut') {
      res.deathChance = deathChance(existing);
      res.died = rDeath < res.deathChance;
    }
    return res;
  });
}
