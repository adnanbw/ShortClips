-- ============================================================================
-- OpenShorts -> Instagram auto-poster bridge
-- Run ONCE in the Supabase SQL editor of the poster project.
--
-- Everything here is ADDITIVE. Manual uploads from the poster's own dashboard
-- keep writing `storage_path` and keep being signed by Supabase Storage; the
-- worker only takes the new path when `b2_key` is set. Nothing that works
-- today changes behaviour.
--
-- Why B2 at all: the poster's bucket is capped at 50 MB a file ("Free Supabase
-- projects have a 50 MB global max upload limit" — frontend-setup.sql), and
-- rendered OpenShorts clips measured 28.8 / 30.8 / 33.9 / 49.2 / 51.2 / 55.2 MB
-- across two real jobs. Two of six already do not fit. Backblaze B2 has no
-- such cap and its egress is what Instagram's fetch consumes.
-- ============================================================================


-- ---------------------------------------------------------------------------
-- 1. Media source
-- ---------------------------------------------------------------------------

alter table public.instagram_posts
    add column if not exists b2_key text;

comment on column public.instagram_posts.b2_key is
    'Backblaze B2 object key for media pushed by OpenShorts. Mutually '
    'exclusive with storage_path — the publish worker signs whichever of the '
    'two is set, so manual dashboard uploads are untouched.';

-- storage_path MUST become nullable or no B2 row can ever be inserted. This is
-- the one statement in this file that changes an existing column, and it only
-- widens what is allowed.
alter table public.instagram_posts
    alter column storage_path drop not null;

-- Exactly one media source per row. NOT VALID skips re-checking existing rows:
-- they all have storage_path set and would pass anyway, but a scan is pointless
-- and it keeps this migration instant on a table with history.
alter table public.instagram_posts
    drop constraint if exists instagram_posts_one_media_source;

alter table public.instagram_posts
    add constraint instagram_posts_one_media_source
    check (
        (storage_path is not null and b2_key is null)
        or (storage_path is null and b2_key is not null)
    ) not valid;


-- ---------------------------------------------------------------------------
-- 2. Provenance
-- ---------------------------------------------------------------------------
-- Which system queued this, and which clip it came from. `source_ref` is
-- "<job_id>:<clip_index>", so a post that went out wrong can be traced back to
-- the exact render without guessing from the caption.

alter table public.instagram_posts
    add column if not exists source text not null default 'dashboard';

alter table public.instagram_posts
    add column if not exists source_ref text;

-- Set by the sweeper once the object has been removed from B2, so a published
-- post keeps its row (and its media id) after its video is gone.
alter table public.instagram_posts
    add column if not exists media_deleted_at timestamptz;


-- ---------------------------------------------------------------------------
-- 3. Slot allocation
-- ---------------------------------------------------------------------------
-- The poster's dashboard computes free slots in the BROWSER'S local timezone
-- (candidateSlotDates in Dashboard.js builds `new Date(y, m, d, h, mm)`).
-- `slot_time` is a bare `time` with no zone attached, so the same 20:00 row
-- means 20:00 IST to the dashboard and would mean 20:00 UTC to an Edge
-- Function — a 5.5-hour error that nothing downstream could detect.
--
-- So the zone is an explicit ARGUMENT here and the arithmetic is done in
-- Postgres, which has real timezone support. OpenShorts passes the timezone
-- already chosen in its scheduling modal.
--
-- Returns at most p_count future slot instants that no active post occupies,
-- earliest first. Callers that book several should request them ONE AT A TIME
-- and insert between calls, so each booking is visible to the next.

-- "20:00 in Asia/Kolkata" -> an absolute instant. Explicit times go through
-- here for the same reason slots do: `slot_time` and a picked time are both
-- WALL CLOCKS, and of the four runtimes in this pipeline (browser, Python,
-- Deno, Postgres) only Postgres is guaranteed to carry a timezone database.
-- `new Date("2026-09-24T20:00:00")` in Deno reads that as UTC; the same string
-- in the browser reads as the viewer's zone; Python needs the `tzdata` package
-- on any host without a system zoneinfo, which Windows and slim containers are.
-- One implementation, in the one place that cannot be wrong.
create or replace function public.local_to_utc(
    p_local timestamp,
    p_tz    text default 'UTC'
)
returns timestamptz
language sql
immutable
as $$
    select p_local at time zone p_tz;
$$;

grant execute on function public.local_to_utc(timestamp, text)
    to service_role, authenticated;


create or replace function public.next_free_slots(
    p_account uuid,
    p_count   integer default 1,
    p_tz      text    default 'UTC'
)
returns setof timestamptz
language plpgsql
stable
as $$
declare
    v_slots     time[];
    v_slot      time;
    v_day       date;
    v_candidate timestamptz;
    v_found     integer := 0;
    v_offset    integer := 0;
    v_now       timestamptz := now();
begin
    select array_agg(slot_time order by slot_time)
      into v_slots
      from public.instagram_posting_slots
     where account_id = p_account
       and is_active;

    if v_slots is null then
        raise exception 'No active posting slots for account %', p_account
            using errcode = 'no_data_found';
    end if;

    -- 370 days is the same horizon the dashboard uses: enough for a year of
    -- slots even at one a day, and a hard stop so a fully booked calendar
    -- cannot spin forever.
    while v_found < p_count and v_offset < 370 loop
        v_day := (v_now at time zone p_tz)::date + v_offset;

        foreach v_slot in array v_slots loop
            exit when v_found >= p_count;

            v_candidate := (v_day + v_slot) at time zone p_tz;
            continue when v_candidate <= v_now;

            if not exists (
                select 1
                  from public.instagram_posts
                 where status in ('scheduled', 'creating',
                                  'processing', 'publishing')
                   and scheduled_at = v_candidate
            ) then
                v_found := v_found + 1;
                return next v_candidate;
            end if;
        end loop;

        v_offset := v_offset + 1;
    end loop;
end;
$$;

grant execute on function public.next_free_slots(uuid, integer, text)
    to service_role, authenticated;

-- The occupancy test above runs once per candidate slot.
create index if not exists idx_instagram_posts_scheduled_active
    on public.instagram_posts (scheduled_at)
    where status in ('scheduled', 'creating', 'processing', 'publishing');


-- ---------------------------------------------------------------------------
-- 4. Sanity check (safe to re-run; returns rows, changes nothing)
-- ---------------------------------------------------------------------------

select
    (select count(*) from public.instagram_posting_slots where is_active)
        as active_slots,
    (select count(*) from public.instagram_accounts where is_active)
        as active_accounts,
    (select count(*) from public.instagram_posts)
        as existing_posts;
