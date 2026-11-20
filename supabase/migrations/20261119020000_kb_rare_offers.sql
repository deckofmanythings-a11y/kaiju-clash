-- Rare Wild-only encounters (Superhero, Superhero Team, Army, Magical Girl).
-- A Wild offer with rival_frame set is a rare enemy of that frame. Engine v3 adds their passives.
alter table kb_offers add column if not exists rival_frame text;
