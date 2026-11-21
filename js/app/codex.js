// Codex: every rule, formula and number the battle engine uses, in players' terms.
// Mirrors js/engine/battle.js, js/engine/aftermath.js and js/data/*. If you change a formula there, change it here.
// Rare Wild encounters are easter eggs: their frames, abilities and passives are NEVER listed here.
import { FRAMES, STATS, STAT_NAMES, MAX_STAT_LEVEL } from '../data/frames.js';
import { ABILITIES, STATUSES, DMG_TYPES, MAX_EQUIPPED, abilityAvailableTo, abilityScope } from '../data/abilities.js';
import { ENVIRONMENTS } from '../data/environments.js';
import { INJURIES, CONDITIONS, CRITICAL_AT, MAX_INJURIES, HEAL_COST, MAX_HEAL_MIN } from '../data/injuries.js';
import { INJURY_ODDS } from '../engine/aftermath.js';
import { REWARD, STARTING_BIO, statUpgradeCost } from '../data/economy.js';

const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const pct = (v, p) => Math.floor((v * p) / 100);
const cap = (s) => s[0].toUpperCase() + s.slice(1);

// Only abilities a player frame can use (rare-enemy kits stay secret).
export const PUBLIC_ABILITIES = ABILITIES.filter(a => FRAMES.some(f => abilityAvailableTo(a, f)));
const usedBy = (a) => FRAMES.filter(f => abilityAvailableTo(a, f)).map(f => f.name);

// ---------- Derived numbers for one Kaiju's stats ----------
export const crit = (fcs, bonus = 0) => Math.max(0, Math.min(60, 5 + Math.floor(fcs / 2) + bonus));
export const reduction = (arm) => 100 - Math.floor(10000 / (100 + arm * 2));     // % of each hit blocked
export const energyGain = (enr) => 12 + Math.floor(enr / 3);
export const tick = (spd) => spd * 5 + 100;

export function statMath(st, f, lv) {
  const next = (s) => (lv[s] < MAX_STAT_LEVEL ? st[s] + f.growth[s] : null);
  const n = (s, fmt) => { const v = next(s); return v === null ? '' : ` <span class="nx">→ next level: ${fmt(v)}</span>`; };
  return {
    hp: `<b>${st.hp}</b> max HP${n('hp', v => v + ' HP')}`,
    pow: `Might-scaled attacks deal <b>power × ${st.pow} ÷ 20</b> per hit (a power-100 attack: <b>${Math.floor(100 * st.pow / 20)}</b> before armor)${n('pow', v => Math.floor(100 * v / 20))}. Damage-over-time uses the higher of Might/Energy.`,
    enr: `Energy-scaled attacks deal <b>power × ${st.enr} ÷ 20</b> per hit (<b>${Math.floor(100 * st.enr / 20)}</b> at power 100). Regains <b>${energyGain(st.enr)}</b> energy each turn (12 + Energy ÷ 3)${n('enr', v => energyGain(v) + ' energy/turn')}.`,
    arm: `Every hit taken is multiplied by 100 ÷ (100 + 2 × ${st.arm}) = <b>${reduction(st.arm)}% blocked</b>${n('arm', v => reduction(v) + '% blocked')}. Armor-scaled rams use it as their attack stat.`,
    spd: `Initiative gauge fills <b>${tick(st.spd)}</b> per tick (Speed × 5 + 100); you act each time it passes 1000, so you act <b>${(tick(st.spd) / tick(10)).toFixed(2)}×</b> as often as a Speed-10 Kaiju. Every enemy attack loses Speed ÷ 2 = <b>${Math.floor(st.spd / 2)}</b> accuracy against you${n('spd', v => tick(v) + '/tick')}.`,
    fcs: `Crit chance <b>${crit(st.fcs)}%</b> (5 + Focus ÷ 2, max 60%; crits deal ×1.5). Adds Focus ÷ 2 = <b>+${Math.floor(st.fcs / 2)}</b> accuracy to your attacks${n('fcs', v => crit(v) + '% crit')}.`,
  };
}

// One line of "your numbers" for an ability, given the user's current stats.
export function abilityMath(ab, st) {
  if (!st) return '';
  const bits = [];
  if (ab.kind === 'attack') {
    const s = st[ab.stat || 'pow'], per = Math.floor((ab.power * s) / 20), hits = ab.hits || 1;
    bits.push(`<b>${per}</b>${hits > 1 ? ` × ${hits} = <b>${per * hits}</b>` : ''} before armor`);
    bits.push(`vs Armor 15: ~${Math.floor((per * 100) / 130) * hits}`);
    bits.push(`crit ${crit(st.fcs, ab.critBonus || 0)}% for ×1.5`);
  }
  if (ab.heal) bits.push(`heals <b>${pct(st.hp, ab.heal)}</b> HP`);
  for (const e of [...(ab.effects || []), ...(ab.selfEffects || [])]) {
    const d = STATUSES[e.status];
    if (d?.dot) bits.push(`${d.name}: <b>${Math.floor(((e.power || 30) * Math.max(st.pow, st.enr)) / 10)}</b>/turn before resist`);
    if (e.status === 'shield') bits.push(`shield absorbs <b>${pct(st.hp, e.power || 10)}</b>`);
    if (e.status === 'regen') bits.push(`regen <b>${pct(st.hp, e.power || 4)}</b>/turn`);
  }
  return bits.length ? `<div class="mathline">Your numbers: ${bits.join(' · ')}</div>` : '';
}

// Full numeric breakdown of an ability (no player stats needed).
function abilityRow(a) {
  const k = [];
  if (a.kind === 'attack') {
    k.push(`<b>${a.type === 'random' ? a.randomTypes.join('/') : a.type}</b> damage`, `power <b>${a.power}</b>${a.hits ? ` × <b>${a.hits}</b> hits` : ''}`, `scales with <b>${STAT_NAMES[a.stat || 'pow']}</b>`);
  } else k.push(`<b>${cap(a.kind)}</b>`);
  if (a.acc != null && a.kind !== 'buff' && a.kind !== 'heal') k.push(`accuracy <b>${a.acc}</b>`);
  k.push(a.cost ? `<b>${a.cost}</b> energy` : 'no energy cost');
  if (a.cd) k.push(`cooldown <b>${a.cd}</b> turns`);
  if (a.heal) k.push(`heals <b>${a.heal}%</b> max HP${a.cleanse ? ' and removes all debuffs' : ''}`);
  for (const e of a.effects || []) k.push(`<b>${e.chance ?? 100}%</b> chance: ${STATUSES[e.status].name} for ${e.turns} turn${e.turns > 1 ? 's' : ''}${e.power ? ` (power ${e.power})` : ''}`);
  for (const e of a.selfEffects || []) k.push(`self: ${STATUSES[e.status].name} for ${e.turns} turn${e.turns > 1 ? 's' : ''}${e.power ? ` (power ${e.power})` : ''}`);
  if (a.energyGain) k.push(`+<b>${a.energyGain}</b> energy`);
  if (a.selfDamage) k.push(`costs <b>${a.selfDamage}%</b> of your max HP`);
  if (a.execute) k.push(`<b>×2</b> damage if target is below ${a.execute}% HP`);
  if (a.critBonus) k.push(`+<b>${a.critBonus}</b> crit chance`);
  return k.join(' · ');
}

export function codexHtml({ stats = null, frameId = null, kaijuName = '' } = {}) {
  const frameOpts = FRAMES.map(f => `<option value="${f.id}" ${f.id === frameId ? 'selected' : ''}>${f.name}</option>`).join('');
  const resistCell = (f, t) => { const v = f.resist?.[t] ?? 100; return `<td class="${v < 100 ? 'w' : v > 100 ? 'l' : 'rec'}">${v}%</td>`; };
  return `
    <h2>Codex</h2>
    <p class="sub">Every rule and number the battle engine uses. Fights are fully automatic and deterministic: the same two Kaiju, arena and seed always play out identically. All math uses whole numbers and rounds down at every step.</p>
    <div class="codex-toc">${[['fight', 'How a fight runs'], ['stats', 'Stats'], ['dmg', 'Damage formula'], ['acc', 'Accuracy'], ['status', 'Status effects'], ['resist', 'Resistances'], ['env', 'Arenas'], ['ai', 'How Kaiju choose moves'], ['abil', 'Ability catalogue'], ['econ', 'Biomass & upgrades'], ['inj', 'Injuries & death']].map(([id, t]) => `<a href="#cx-${id}" data-cx="${id}">${t}</a>`).join('')}</div>

    <details class="cx" id="cx-fight" open><summary>How a fight runs</summary>
      <ul>
        <li>Each Kaiju has an <b>initiative gauge</b>. Every tick it fills by <code>Speed × 5 + 100</code>. Whoever passes <b>1000</b> takes a turn and loses 1000 from the gauge (ties go to the fuller gauge, then a coin flip).</li>
        <li>At the start of each of its turns a Kaiju gains <code>12 + Energy ÷ 3</code> energy (it starts the fight with <b>20</b>, max <b>100</b>), then damage-over-time and regeneration tick.</li>
        <li>Stunned Kaiju lose the turn (the stun is then removed). Bound Kaiju have a 50% chance to lose the turn.</li>
        <li>Otherwise it uses one ability (see <a href="#cx-ai" data-cx="ai">How Kaiju choose moves</a>). Then its cooldowns and its status durations tick down by 1.</li>
        <li>The fight ends on a knockout, or after <b>80 turns</b> (40 each). On time-out, whoever has the higher <b>percentage</b> of HP left wins (to 0.1%); equal is a draw.</li>
        <li>Each Kaiju has a free <b>basic attack</b> plus up to <b>${MAX_EQUIPPED}</b> equipped abilities.</li>
      </ul>
    </details>

    <details class="cx" id="cx-stats" open><summary>Stats</summary>
      <p>Each stat is <code>frame base + frame growth × level</code> (levels 0 to ${MAX_STAT_LEVEL}). Injuries, arenas and status effects then apply percentage changes; a stat never drops below 1.</p>
      ${stats ? `<p class="rec">Showing live numbers for <b>${esc(kaijuName)}</b>.</p>` : ''}
      <table class="cxt"><tr><th>Stat</th><th>What it does</th>${stats ? '<th>Yours</th>' : ''}</tr>
        ${[
          ['hp', 'Max HP.'],
          ['pow', 'Attack stat for physical (Might-scaled) abilities. Also feeds damage-over-time (higher of Might/Energy).'],
          ['enr', 'Attack stat for Energy-scaled abilities, and energy regained per turn: <code>12 + Energy ÷ 3</code>.'],
          ['arm', 'Damage taken × <code>100 ÷ (100 + 2 × Armor)</code>. Armor 10 blocks 17%, 25 blocks 33%, 50 blocks 50%. Doesn\'t reduce DoT or arena hazards. Some rams scale with Armor.'],
          ['spd', 'Initiative: gauge fills <code>Speed × 5 + 100</code> per tick. Also lowers enemy accuracy by <code>Speed ÷ 2</code>.'],
          ['fcs', 'Accuracy +<code>Focus ÷ 2</code>; crit chance <code>5 + Focus ÷ 2</code>% (max 60%), crits deal ×1.5.'],
        ].map(([s, t]) => `<tr><td><b>${STAT_NAMES[s]}</b></td><td>${t}</td>${stats ? `<td class="num">${stats[s]}</td>` : ''}</tr>`).join('')}
      </table>
      <p>Frame base stats and growth per level:</p>
      <div class="scrollx"><table class="cxt small"><tr><th>Frame</th>${STATS.map(s => `<th>${STAT_NAMES[s]}</th>`).join('')}</tr>
        ${FRAMES.map(f => `<tr><td>${f.name}</td>${STATS.map(s => `<td class="num">${f.base[s]} <span class="rec">+${f.growth[s]}</span></td>`).join('')}</tr>`).join('')}
      </table></div>
    </details>

    <details class="cx" id="cx-dmg"><summary>Damage formula</summary>
      <p>Each hit of an attack is worked out in this order, rounding down after every step:</p>
      <ol>
        <li><b>Base</b> = <code>power × attack stat ÷ 20</code> (attack stat is Might, Energy or Armor, as the ability says).</li>
        <li><b>Armor</b>: × <code>100 ÷ (100 + 2 × target Armor)</code>.</li>
        <li><b>Variance</b>: × a random 90–110%.</li>
        <li><b>Critical hit</b>: chance <code>5 + Focus ÷ 2 + ability crit bonus</code>% (0–60%). A crit is ×150%.</li>
        <li><b>Arena</b> damage-type modifier (e.g. Fire +25% at Caldera Rim).</li>
        <li><b>Target resistance</b> for that damage type (see <a href="#cx-resist" data-cx="resist">Resistances</a>).</li>
        <li><b>Arena move bonus</b> (Neo-Tokyo Bay: grabs ×125%).</li>
        <li><b>Bound</b> targets take ×120%.</li>
        <li><b>Execute</b> abilities deal ×2 if the target is below the listed HP%.</li>
        <li>Minimum 1 damage. A <b>shield</b> absorbs it first.</li>
      </ol>
      <p>Multi-hit attacks roll accuracy, variance and crit separately for every hit; on-hit effects roll once if at least one hit landed.</p>
    </details>

    <details class="cx" id="cx-acc"><summary>Accuracy</summary>
      <p>Hit chance = <code>ability accuracy + (your Focus − their Speed) ÷ 2 + arena modifier − 30 if you're Blinded − 25 if they're Evasive</code>, clamped to <b>15–100%</b>. Debuff abilities use the same roll; if it misses, none of their effects apply. Buffs and heals always work.</p>
    </details>

    <details class="cx" id="cx-status"><summary>Status effects</summary>
      <p>A status effect lasts the listed number of the affected Kaiju's own turns. Re-applying one keeps the longer duration and the stronger value; it never stacks twice.</p>
      <table class="cxt">
        ${Object.entries(STATUSES).map(([id, s]) => `<tr><td><b>${s.name}</b></td><td>${{
          burn: 'Fire damage at the start of each turn.', poison: 'Acid damage at the start of each turn.', bleed: 'Physical damage at the start of each turn.',
          stun: 'Loses its next turn.', bind: '50% chance to lose each turn; takes ×120% damage from hits.',
          slow: 'Speed ×65%.', haste: 'Speed ×135%.', weaken: 'Might and Energy ×75%.', enrage: 'Might ×130%.', charged: 'Energy ×130%.',
          fortify: 'Armor ×150%.', sunder: 'Armor ×60%.', blind: '−30 accuracy.', evasive: 'Attackers get −25 accuracy.',
          regen: 'Heals <code>power</code>% of max HP at the start of each turn (default 4%).',
          shield: 'Absorbs damage equal to <code>power</code>% of max HP (default 10%) until used up or expired. DoT goes around it.',
        }[id] || esc(s.desc)}</td></tr>`).join('')}
      </table>
      <p><b>Damage-over-time</b> per turn is locked in when applied: <code>effect power × (higher of the attacker's Might or Energy) ÷ 10</code>, then × the target's resistance to that type (Burn = fire, Poison = acid, Bleed = physical), then × the arena's type modifier when it ticks. Armor and shields don't reduce it.</p>
    </details>

    <details class="cx" id="cx-resist"><summary>Resistances</summary>
      <p>Damage of each type is multiplied by these percentages. <span class="w">Green</span> = resists, <span class="l">red</span> = weak.</p>
      <div class="scrollx"><table class="cxt small"><tr><th>Frame</th>${DMG_TYPES.map(t => `<th>${cap(t)}</th>`).join('')}</tr>
        ${FRAMES.map(f => `<tr><td>${f.name}</td>${DMG_TYPES.map(t => resistCell(f, t)).join('')}</tr>`).join('')}
      </table></div>
    </details>

    <details class="cx" id="cx-env"><summary>Arenas</summary>
      ${ENVIRONMENTS.map(e => {
        const mods = Object.entries(e.tagMods || {}).map(([tag, m]) => `${tag === '_other' ? 'everyone else' : cap(tag) + ' frames'}: ${Object.entries(m).map(([s, v]) => `${STAT_NAMES[s]} ×${v}%`).join(', ')}`);
        const types = Object.entries(e.typeMult || {}).map(([t, v]) => `${cap(t)} ×${v}%`);
        const fx = Object.entries(e.fxMult || {}).map(([t, v]) => `${t === 'grab' ? 'Grab' : cap(t)} moves ×${v}%`);
        const hz = e.hazard;
        return `<div class="card"><b>${e.name}</b>
          <div class="rec">${[...mods, ...types, ...fx, e.accMod ? `Accuracy ${e.accMod > 0 ? '+' : ''}${e.accMod} for everyone` : ''].filter(Boolean).join(' · ') || 'No modifiers.'}</div>
          ${hz ? `<div class="rec">Hazard <b>${hz.name}</b>: every ${hz.every} rounds, ${hz.random ? 'one random Kaiju takes' : 'every Kaiju takes'} ${hz.pct}% of max HP as ${hz.type} damage (× resistance and arena modifier; ignores armor)${hz.status ? `, with a ${hz.status.chance}% chance of ${STATUSES[hz.status.status].name} for ${hz.status.turns} turns` : ''}. Immune: ${hz.immuneTags.map(cap).join(', ')} frames.</div>` : ''}
        </div>`;
      }).join('')}
      <p class="rec">Frame tags: ${FRAMES.map(f => `${f.name} (${f.tags.join(', ')})`).join(' · ')}</p>
    </details>

    <details class="cx" id="cx-ai"><summary>How Kaiju choose moves</summary>
      <p>Every turn the Kaiju scores each ability that is off cooldown and affordable:</p>
      <ul>
        <li><b>Attacks</b>: <code>power × hits × attack stat ÷ 20</code>, × target resistance, × arena modifier, × accuracy %; doubled if an execute would trigger; +40 × chance for each on-hit effect the target doesn't already have.</li>
        <li><b>Heals</b>: 0 if missing less than 30% HP, otherwise <code>3 × % HP missing</code>.</li>
        <li><b>Buffs / debuffs</b>: 70 for each effect not already active, +60 if it gives energy while you're below 40 energy.</li>
      </ul>
      <p>It then takes the <b>top 3</b> and picks one at random, weighted by <code>score² ÷ 100 + 1</code>, so the best move is usually, but not always, chosen.</p>
    </details>

    <details class="cx" id="cx-abil"><summary>Ability catalogue</summary>
      <div class="row" style="gap:6px;flex-wrap:wrap;margin-bottom:8px">
        <select id="cxframe"><option value="">All frames</option>${frameOpts}</select>
        <select id="cxkind"><option value="">All kinds</option><option value="attack">Attacks</option><option value="buff">Buffs</option><option value="debuff">Debuffs</option><option value="heal">Heals</option></select>
      </div>
      <div id="cxabil"></div>
    </details>

    <details class="cx" id="cx-econ"><summary>Biomass &amp; upgrades</summary>
      <ul>
        <li>New players start with <b>${STARTING_BIO}</b> Biomass.</li>
        <li>Upgrading a stat from level L costs <code>60 + 40 × L</code> (${Array.from({ length: MAX_STAT_LEVEL }, (_, l) => statUpgradeCost(l)).join(', ')}).</li>
        <li><b>Power</b> = <code>100 + 10 per stat level + (ability price ÷ 20) per equipped ability</code>. It's a matchmaking guide, not a combat stat.</li>
        <li>Rated fight rewards: Win <b>${REWARD.win}</b>, Draw <b>${REWARD.draw}</b>, Loss <b>${REWARD.loss}</b>, × <code>opponent Power ÷ your Power</code>, clamped to 60–160%.</li>
        <li>Unrated fights (after the daily limit) pay nothing.</li>
      </ul>
    </details>

    <details class="cx" id="cx-inj"><summary>Injuries &amp; death</summary>
      <p>Only the Kaiju that <b>started</b> the fight can be injured or die; defenders are never hurt. After each fight, the attacker rolls for an injury based on how it went:</p>
      <table class="cxt"><tr><th>Outcome</th><th>Injury chance</th><th>…of which Major</th></tr>
        ${[['knockedOut', 'Knocked out'], ['lostOnTime', 'Lost on time'], ['draw', 'Draw'], ['wonBarely', 'Won with < 25% HP'], ['wonHurt', 'Won with < 50% HP'], ['wonClean', 'Won with 50%+ HP']].map(([k, t]) => `<tr><td>${t}</td><td class="num">${INJURY_ODDS[k][0]}%</td><td class="num">${INJURY_ODDS[k][1]}%</td></tr>`).join('')}
      </table>
      <p>An injury lowers stats by a percentage. Penalties to the same stat add up (capped at −60%). A Kaiju can carry at most ${MAX_INJURIES}. Injuries heal on their own (at most ${MAX_HEAL_MIN} minutes) or instantly for Biomass: up to ${HEAL_COST[1]} (Minor) / ${HEAL_COST[2]} (Major), pro-rated by time left, minimum 5.</p>
      <table class="cxt"><tr><th>Injury</th><th>Type</th><th>Effect</th><th>Heals in</th></tr>
        ${INJURIES.map(i => `<tr><td>${i.name}</td><td>${i.severity === 2 ? 'Major (2)' : 'Minor (1)'}</td><td>${Object.entries(i.mods).map(([s, v]) => `${STAT_NAMES[s]} ${v}%`).join(', ')}</td><td class="num">${i.healMin} min</td></tr>`).join('')}
      </table>
      <p>Condition = total severity: ${CONDITIONS.map(c => `${c.name} ${c.min}+`).join(', ')}. A <b>Critical</b> Kaiju (${CRITICAL_AT}+) that gets knocked out in a fight it started can die for good: <code>20% + 10% per severity above ${CRITICAL_AT}</code>, max 50%.</p>
    </details>`;
}

export function renderAbilityList(el, { frame = '', kind = '', stats = null } = {}) {
  const list = PUBLIC_ABILITIES.filter(a => (!kind || a.kind === kind) && (!frame || abilityAvailableTo(a, FRAMES.find(f => f.id === frame))));
  el.innerHTML = list.length ? list.map(a => `<div class="card ab"><div><b>${esc(a.name)}</b> <span class="tag ${a.frames ? 'ex' : a.universal ? 'un' : ''}">${abilityScope(a)}</span>
      <div class="meta">${a.basic ? 'Free basic attack' : a.price + ' BIO'} · ${usedBy(a).join(', ')}</div></div>
      <div class="desc">${abilityRow(a)}</div>${abilityMath(a, stats)}</div>`).join('') : '<p class="rec">No abilities match.</p>';
}
