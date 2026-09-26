# OpenShorts -> self-hosted Instagram poster

Everything in this folder is applied to the **poster's** Supabase project, not
to OpenShorts. It lives here so the two halves of the bridge are versioned
together — a schema change and the code that depends on it should never be a
guess about which was deployed first.

The OpenShorts half is `instagram_publish.py`, `POST /api/instagram/schedule`
and two controls in `ScheduleWeekModal.jsx`.

## Why it exists

Upload-Post caps posts on its free tier. The poster is a separate project
(Next.js on Netlify + Supabase + a Meta app) that publishes to one Instagram
account through the owner's own Graph API credentials, with `pg_cron` calling an
Edge Function every minute.

## The shape

```
OpenShorts renders a clip
      |
      |  "Schedule to Instagram" in the week modal
      v
upload the mp4 to Backblaze B2            <- instagram_publish.upload_clip
      |
      v
POST the object KEY to ingest-post        <- instagram_publish.queue_posts
      |
      v
row in instagram_posts, status=scheduled
      |
      |  ... nothing happens until the scheduled time ...
      v
pg_cron -> instagram-publish-worker
      |
      +-- mints a FRESH presigned GET for the B2 object
      +-- POST /media          (Instagram fetches the video itself)
      +-- poll status_code until FINISHED
      +-- POST /media_publish
```

Only the key ever crosses the wire, never a URL. A presigned URL minted at
schedule time would have to outlive a schedule measured in days, and the
machine that made it may be switched off long before. Signing at publish time
also means an expired or failed container is self-healing: the row goes back to
`scheduled` and the next pass mints a new URL.

Traffic is **outbound only**, which is not a preference — OpenShorts runs
behind NAT and Supabase cannot reach it.

## Apply, in order

1. **`001_openshorts_bridge.sql`** — paste into the Supabase SQL editor and run.
   Adds `b2_key`, makes `storage_path` nullable, adds `local_to_utc` and
   `next_free_slots`. Additive: manual dashboard uploads are untouched.

2. **`WORKER_PATCH.md`** — apply to the existing
   `supabase/functions/instagram-publish-worker/index.ts`. About eight lines:
   sign B2 when `b2_key` is set, Supabase Storage otherwise.

3. **`ingest-post/index.ts`** — copy to
   `supabase/functions/ingest-post/index.ts` in the poster repo and deploy with
   `--no-verify-jwt` (the `x-ingest-secret` header is the auth; OpenShorts has
   no way to mint a Supabase JWT).

   **This flag is easy to lose.** A dashboard deploy leaves JWT verification
   ON, and Supabase's gateway then refuses the call before the function runs
   with `{"code":"UNAUTHORIZED_NO_AUTH_HEADER"}` — a 401 that looks exactly
   like a wrong shared secret but comes from the platform, not from any code
   here. Either redeploy with the flag, or set `IG_INGEST_ANON_KEY` in
   OpenShorts, which satisfies the gate without a redeploy. The anon key is
   public by design (it ships in the poster's own frontend) and is not what
   authorises the call.

4. **Secrets** — see `WORKER_PATCH.md` §3.

5. **OpenShorts `.env`** — the six `B2_*` / `IG_INGEST_*` variables from
   `.env.example`. No rebuild: `boto3` and `httpx` are already in the image and
   the new files are bind-mounted.

## Decisions worth not re-litigating

**Timezone arithmetic happens in Postgres.** `slot_time` and a picked time are
both wall clocks, and of the four runtimes involved only Postgres is guaranteed
to have a timezone database. `new Date("2026-09-24T20:00:00")` reads as UTC in
Deno and as the viewer's zone in a browser; Python needs the `tzdata` package
on any host without a system zoneinfo, which Windows and slim containers both
are. The first version converted in Python and the tests failed on the dev box
with `ZoneInfoNotFoundError: No time zone found with key UTC` — which would
have shipped as a silent multi-hour scheduling error if the dev box had
happened to have the data.

**"Post now" is a booking, not a bypass.** It inserts with
`scheduled_at = now()`, and `createDue` already selects `scheduled_at <= now`,
so the next cron pass (within a minute) picks it up and it travels the same
container / poll / publish route as everything else, retries included. A second
"publish immediately" path would be a second thing to keep correct. The cost is
that it is not instant — it is *within a minute* — and that N clips at once
means N reels a few minutes apart, which the modal warns about.

**The upload runs in the BACKGROUND and the modal polls it.** A clip is tens
of megabytes and a home connection's upstream is what carries it — measured at
roughly two minutes each for 28-43 MB files. Holding the HTTP request open for
that long is what caused the worst failure so far: the modal looked hung, the
button got pressed again, and each press started ANOTHER full upload of the
same clips. Six overlapping uploads then competed for the same upstream, so
every one got slower, and 870 MB of duplicates landed in the bucket with no
rows pointing at any of it. `POST /api/instagram/schedule` now returns a
`task_id` immediately and `GET /api/instagram/schedule/{task_id}` reports
`stage` and `done`/`total`. A second request for a job that is already
uploading gets the FIRST task's id back rather than starting a race.

**Slots are booked one at a time.** `next_free_slots` is asked for one slot per
insert, because two clips asking for "the next free slot" in the same
transaction would both be handed the same instant.

**Object keys carry a random component.** Re-queueing a clip after a caption fix
must not overwrite the object an earlier, still-scheduled post points at — the
key is resolved at publish time, so an overwrite would change what that post
publishes.

**A refused row's object is deleted again.** An orphan in the bucket is
invisible, unpublishable and billed forever. The poster's own dashboard already
does this after a failed insert; OpenShorts matches it.

**No inline delete after publishing.** The original sketch ended `publish ->
delete from B2`. If `media_publish` succeeds but the row update fails, the retry
finds no media and burns its attempts on a reel that is already live.
`media_deleted_at` plus a sweeper is the safe shape.

## Not done yet

- **Token refresh.** Long-lived Instagram tokens last 60 days and the worker
  reads a static secret. Owner is handling it.
- **The sweeper.** `media_deleted_at` exists; nothing sets it yet. On B2's free
  tier there is no hurry, but objects accumulate.
- **Dashboard preview for B2 rows.** `Dashboard.js` signs Supabase Storage
  only, so OpenShorts rows show caption and time but no thumbnail.
- **A daily publish cap.** Instagram allows 50 published posts per account per
  24 h; nothing counts them.
