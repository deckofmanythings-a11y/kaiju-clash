-- Ver. 1.11.0 / engine v4: Fine Tune.
-- kb_kaiju.tune: null (not bought) or { on: true, pts: int, alloc: { stat: -3..3 }, mods: { abilityId: modId } }.
-- Validation and prices live in kaiju-api (js/data/finetune.js); kb_spend just stores it atomically.
alter table kb_kaiju add column if not exists tune jsonb;

-- The old 10-arg kb_spend is kept so the currently deployed kaiju-api keeps working until the new one ships;
-- PostgREST picks the overload by parameter names. Drop it in a later migration.

create or replace function kb_spend(
  p_user uuid, p_kaiju uuid, p_version int, p_cost int,
  p_levels jsonb, p_abilities text[], p_unlocked text[], p_colors jsonb, p_name text, p_injuries jsonb,
  p_tune jsonb
) returns kb_kaiju
language plpgsql security definer set search_path = public as $$
declare k kb_kaiju;
begin
  if p_cost < 0 then raise exception 'bad cost'; end if;
  update kb_players set bio = bio - p_cost where user_id = p_user and bio >= p_cost;
  if not found then raise exception 'not enough biomass'; end if;
  update kb_kaiju set levels = p_levels, abilities = p_abilities, unlocked = p_unlocked,
         colors = p_colors, name = p_name, injuries = p_injuries, tune = p_tune,
         version = version + 1, updated_at = now()
   where id = p_kaiju and owner = p_user and version = p_version and not retired
   returning * into k;
  if not found then raise exception 'stale kaiju state, reload'; end if;
  return k;
end $$;

revoke all on function kb_spend(uuid, uuid, int, int, jsonb, text[], text[], jsonb, text, jsonb, jsonb) from public, anon, authenticated;
grant execute on function kb_spend(uuid, uuid, int, int, jsonb, text[], text[], jsonb, text, jsonb, jsonb) to service_role;
