-- Injuries + death. Engine v2.
-- kb_kaiju.injuries: [{ id, at, heals_at }] (ISO timestamps). Healed entries are pruned by the
-- server on the next write; readers filter by heals_at.
-- A dead Kaiju: retired = true, died_at set, death = { killer, killer_owner, env, battle, salvage }.

alter table kb_kaiju
  add column if not exists injuries jsonb not null default '[]'::jsonb,
  add column if not exists died_at timestamptz,
  add column if not exists death jsonb;
alter table kb_battles add column if not exists aftermath jsonb;
create index if not exists kb_kaiju_fallen on kb_kaiju (owner, died_at desc) where died_at is not null;

drop function if exists kb_spend(uuid, uuid, int, int, jsonb, text[], text[], jsonb, text);
drop function if exists kb_settle(uuid, smallint, uuid, uuid, uuid, int, bigint, text, jsonb, smallint, text, int, text, boolean, int, int, int, int, bigint);

create or replace function kb_spend(
  p_user uuid, p_kaiju uuid, p_version int, p_cost int,
  p_levels jsonb, p_abilities text[], p_unlocked text[], p_colors jsonb, p_name text, p_injuries jsonb
) returns kb_kaiju
language plpgsql security definer set search_path = public as $$
declare k kb_kaiju;
begin
  if p_cost < 0 then raise exception 'bad cost'; end if;
  update kb_players set bio = bio - p_cost where user_id = p_user and bio >= p_cost;
  if not found then raise exception 'not enough biomass'; end if;
  update kb_kaiju set levels = p_levels, abilities = p_abilities, unlocked = p_unlocked,
         colors = p_colors, name = p_name, injuries = p_injuries, version = version + 1, updated_at = now()
   where id = p_kaiju and owner = p_user and version = p_version and not retired
   returning * into k;
  if not found then raise exception 'stale kaiju state, reload'; end if;
  return k;
end $$;

create or replace function kb_settle(
  p_user uuid, p_slot smallint, p_attacker uuid, p_defender uuid, p_defender_owner uuid,
  p_engine_v int, p_seed bigint, p_env text, p_fighters jsonb, p_winner smallint, p_reason text,
  p_turns int, p_hash text, p_rated boolean, p_reward int, p_def_reward int,
  p_delta_a int, p_delta_d int, p_new_rival_seed bigint,
  p_aftermath jsonb, p_new_inj_a jsonb, p_new_inj_d jsonb, p_died_a boolean, p_death jsonb, p_salvage int
) returns uuid
language plpgsql security definer set search_path = public as $$
declare bid uuid; res text;
begin
  delete from kb_offers where user_id = p_user and slot = p_slot and kaiju_id = p_attacker;
  if not found then raise exception 'offer expired, refresh opponents'; end if;
  delete from kb_offers where user_id = p_user;

  insert into kb_battles (attacker_owner, defender_owner, attacker_kaiju, defender_kaiju, engine_v, seed, env,
    fighters, winner, reason, turns, hash, rated, reward, defense_reward, elo_delta_a, elo_delta_d, aftermath)
  values (p_user, p_defender_owner, p_attacker, p_defender, p_engine_v, p_seed, p_env,
    p_fighters, p_winner, p_reason, p_turns, p_hash, p_rated, p_reward, p_def_reward, p_delta_a, p_delta_d, p_aftermath)
  returning id into bid;

  res := case when p_winner = 0 then 'w' when p_winner = 1 then 'l' else 'd' end;
  update kb_players set bio = bio + p_reward + p_salvage, rival_seed = p_new_rival_seed,
         fights_today = case when fights_day = (now() at time zone 'utc')::date then fights_today + 1 else 1 end,
         fights_day = (now() at time zone 'utc')::date,
         active_kaiju = case when p_died_a and active_kaiju = p_attacker then null else active_kaiju end
   where user_id = p_user;
  -- Injuries are APPENDED (never overwritten) so concurrent fights/heals can't lose each other's writes.
  update kb_kaiju set elo = greatest(100, elo + p_delta_a),
         wins = wins + (res = 'w')::int, losses = losses + (res = 'l')::int, draws = draws + (res = 'd')::int,
         injuries = injuries || coalesce(p_new_inj_a, '[]'::jsonb),
         retired = retired or p_died_a,
         died_at = case when p_died_a then now() else died_at end,
         death = case when p_died_a then coalesce(p_death, '{}'::jsonb) || jsonb_build_object('battle', bid) else death end,
         version = version + 1
   where id = p_attacker and owner = p_user;
  if p_defender is not null then
    update kb_kaiju set elo = greatest(100, elo + p_delta_d),
           def_wins = def_wins + (res = 'l')::int, def_losses = def_losses + (res = 'w')::int,
           injuries = injuries || coalesce(p_new_inj_d, '[]'::jsonb),
           version = version + 1
     where id = p_defender;
    if p_def_reward > 0 and p_defender_owner is not null then
      update kb_players set bio = bio + p_def_reward where user_id = p_defender_owner;
    end if;
  end if;
  return bid;
end $$;

revoke all on function kb_spend(uuid, uuid, int, int, jsonb, text[], text[], jsonb, text, jsonb) from public, anon, authenticated;
revoke all on function kb_settle(uuid, smallint, uuid, uuid, uuid, int, bigint, text, jsonb, smallint, text, int, text, boolean, int, int, int, int, bigint, jsonb, jsonb, jsonb, boolean, jsonb, int) from public, anon, authenticated;
grant execute on function kb_spend(uuid, uuid, int, int, jsonb, text[], text[], jsonb, text, jsonb) to service_role;
grant execute on function kb_settle(uuid, smallint, uuid, uuid, uuid, int, bigint, text, jsonb, smallint, text, int, text, boolean, int, int, int, int, bigint, jsonb, jsonb, jsonb, boolean, jsonb, int) to service_role;
