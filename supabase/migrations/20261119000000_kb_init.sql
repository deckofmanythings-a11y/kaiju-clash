-- Kaiju Clash online ladder. Lives in the shared raided-hex project, kb_ prefix.
-- RLS on with NO policies: clients never touch these tables. All reads/writes go through the
-- kaiju-api Edge Function (service role), which runs the deterministic battle engine itself.

create table if not exists kb_players (
  user_id        uuid primary key references auth.users(id) on delete cascade,
  display_name   text not null check (char_length(display_name) between 2 and 20),
  bio            int  not null default 300 check (bio >= 0),
  active_kaiju   uuid,
  rival_seed     bigint not null default 0,
  fights_today   int  not null default 0,
  fights_day     date not null default (now() at time zone 'utc')::date,
  defense_seen_at timestamptz not null default now(),
  created_at     timestamptz not null default now()
);
create unique index if not exists kb_players_name_uq on kb_players (lower(display_name));

create table if not exists kb_kaiju (
  id          uuid primary key default gen_random_uuid(),
  owner       uuid not null references kb_players(user_id) on delete cascade,
  name        text not null check (char_length(name) between 1 and 24),
  frame       text not null,
  levels      jsonb not null,
  abilities   text[] not null default '{}',
  unlocked    text[] not null default '{}',
  colors      jsonb not null,
  elo         int not null default 1000,
  wins        int not null default 0,
  losses      int not null default 0,
  draws       int not null default 0,
  def_wins    int not null default 0,
  def_losses  int not null default 0,
  version     int not null default 0,
  retired     boolean not null default false,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);
create index if not exists kb_kaiju_owner on kb_kaiju (owner);
create index if not exists kb_kaiju_elo on kb_kaiju (elo desc) where not retired;

-- The 3 opponents currently offered to a player. A fight must consume one of these, so the
-- client can't pick its own opponent or environment.
create table if not exists kb_offers (
  user_id      uuid not null references kb_players(user_id) on delete cascade,
  slot         smallint not null check (slot between 0 and 2),
  kaiju_id     uuid not null,             -- the attacker these were generated for
  opponent     uuid,                      -- real player's kaiju, or null for a wild rival
  rival_seed   bigint,
  rival_rating int,
  env          text not null,
  created_at   timestamptz not null default now(),
  primary key (user_id, slot)
);

create table if not exists kb_battles (
  id              uuid primary key default gen_random_uuid(),
  created_at      timestamptz not null default now(),
  attacker_owner  uuid not null references kb_players(user_id) on delete cascade,
  defender_owner  uuid references kb_players(user_id) on delete set null,
  attacker_kaiju  uuid not null,
  defender_kaiju  uuid,
  engine_v        int not null,
  seed            bigint not null,
  env             text not null,
  fighters        jsonb not null,          -- [snapshotA, snapshotB] exactly as simulated
  winner          smallint,
  reason          text not null,
  turns           int not null,
  hash            text not null,
  rated           boolean not null,
  reward          int not null default 0,
  defense_reward  int not null default 0,
  elo_delta_a     int not null default 0,
  elo_delta_d     int not null default 0
);
create index if not exists kb_battles_att on kb_battles (attacker_owner, created_at desc);
create index if not exists kb_battles_def on kb_battles (defender_owner, created_at desc);

alter table kb_players enable row level security;
alter table kb_kaiju   enable row level security;
alter table kb_offers  enable row level security;
alter table kb_battles enable row level security;
revoke all on kb_players, kb_kaiju, kb_offers, kb_battles from anon, authenticated;

-- Atomic spend: deduct BIO and write the new kaiju state in one transaction.
-- Optimistic concurrency on kaiju.version so two tabs can't double-buy against stale state.
create or replace function kb_spend(
  p_user uuid, p_kaiju uuid, p_version int, p_cost int,
  p_levels jsonb, p_abilities text[], p_unlocked text[], p_colors jsonb, p_name text
) returns kb_kaiju
language plpgsql security definer set search_path = public as $$
declare k kb_kaiju;
begin
  if p_cost < 0 then raise exception 'bad cost'; end if;
  update kb_players set bio = bio - p_cost where user_id = p_user and bio >= p_cost;
  if not found then raise exception 'not enough biomass'; end if;
  update kb_kaiju set levels = p_levels, abilities = p_abilities, unlocked = p_unlocked,
         colors = p_colors, name = p_name, version = version + 1, updated_at = now()
   where id = p_kaiju and owner = p_user and version = p_version and not retired
   returning * into k;
  if not found then raise exception 'stale kaiju state, reload'; end if;
  return k;
end $$;

-- Atomic fight settlement. Consuming the offer row makes each offer single-use.
create or replace function kb_settle(
  p_user uuid, p_slot smallint, p_attacker uuid, p_defender uuid, p_defender_owner uuid,
  p_engine_v int, p_seed bigint, p_env text, p_fighters jsonb, p_winner smallint, p_reason text,
  p_turns int, p_hash text, p_rated boolean, p_reward int, p_def_reward int,
  p_delta_a int, p_delta_d int, p_new_rival_seed bigint
) returns uuid
language plpgsql security definer set search_path = public as $$
declare bid uuid; res text;
begin
  delete from kb_offers where user_id = p_user and slot = p_slot and kaiju_id = p_attacker;
  if not found then raise exception 'offer expired, refresh opponents'; end if;
  delete from kb_offers where user_id = p_user;  -- new set is generated after every fight

  insert into kb_battles (attacker_owner, defender_owner, attacker_kaiju, defender_kaiju, engine_v, seed, env,
    fighters, winner, reason, turns, hash, rated, reward, defense_reward, elo_delta_a, elo_delta_d)
  values (p_user, p_defender_owner, p_attacker, p_defender, p_engine_v, p_seed, p_env,
    p_fighters, p_winner, p_reason, p_turns, p_hash, p_rated, p_reward, p_def_reward, p_delta_a, p_delta_d)
  returning id into bid;

  res := case when p_winner = 0 then 'w' when p_winner = 1 then 'l' else 'd' end;
  update kb_players set bio = bio + p_reward, rival_seed = p_new_rival_seed,
         fights_today = case when fights_day = (now() at time zone 'utc')::date then fights_today + 1 else 1 end,
         fights_day = (now() at time zone 'utc')::date
   where user_id = p_user;
  update kb_kaiju set elo = greatest(100, elo + p_delta_a),
         wins = wins + (res = 'w')::int, losses = losses + (res = 'l')::int, draws = draws + (res = 'd')::int
   where id = p_attacker and owner = p_user;
  if p_defender is not null then
    update kb_kaiju set elo = greatest(100, elo + p_delta_d),
           def_wins = def_wins + (res = 'l')::int, def_losses = def_losses + (res = 'w')::int
     where id = p_defender;
    if p_def_reward > 0 and p_defender_owner is not null then
      update kb_players set bio = bio + p_def_reward where user_id = p_defender_owner;
    end if;
  end if;
  return bid;
end $$;

revoke all on function kb_spend(uuid, uuid, int, int, jsonb, text[], text[], jsonb, text) from public, anon, authenticated;
revoke all on function kb_settle(uuid, smallint, uuid, uuid, uuid, int, bigint, text, jsonb, smallint, text, int, text, boolean, int, int, int, int, bigint) from public, anon, authenticated;
grant execute on function kb_spend(uuid, uuid, int, int, jsonb, text[], text[], jsonb, text) to service_role;
grant execute on function kb_settle(uuid, smallint, uuid, uuid, uuid, int, bigint, text, jsonb, smallint, text, int, text, boolean, int, int, int, int, bigint) to service_role;
