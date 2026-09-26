// Queue a post that OpenShorts has already uploaded to Backblaze B2.
//
// WHY THIS EXISTS AT ALL, rather than OpenShorts writing the row itself:
// inserting into instagram_posts needs the service-role key, which bypasses
// every RLS policy in the project. OpenShorts is deployed to a public host as
// well as run locally, so that key must not live in its environment. This
// function is the narrow hole in the wall: it accepts one shape, validates it,
// and writes one row.
//
// It is authorised exactly like the publish worker — a shared secret header,
// held in Supabase Vault and passed by the caller. Same pattern, same blast
// radius, nothing new to reason about.
//
// Deploy:  supabase functions deploy ingest-post --no-verify-jwt
// Secrets: INGEST_SECRET, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
//
// --no-verify-jwt is deliberate and safe here: the INGEST_SECRET check below is
// the auth. Without the flag the platform demands a Supabase JWT, which
// OpenShorts has no way to mint.

import { createClient } from "npm:@supabase/supabase-js@2";

const ACTIVE_STATUSES = ["scheduled", "creating", "processing", "publishing"];

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

function adminClient() {
  const url = Deno.env.get("SUPABASE_URL");
  const key = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  if (!url || !key) throw new Error("Missing Supabase server environment variables");
  return createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

function isAuthorized(req: Request) {
  // Both names are accepted on purpose. OpenShorts calls its copy
  // IG_INGEST_SECRET (it sits beside IG_INGEST_URL in that .env), and the
  // first version of the setup notes called the Supabase copy INGEST_SECRET.
  // One secret with two names is a trap: the mismatch surfaces as a plain
  // "Unauthorized", indistinguishable from a genuinely wrong value.
  const expected = Deno.env.get("IG_INGEST_SECRET") || Deno.env.get("INGEST_SECRET");
  return Boolean(expected && req.headers.get("x-ingest-secret") === expected);
}

/** The single active Instagram account. This app is deliberately single-account. */
async function activeAccount(supabase: any) {
  const { data, error } = await supabase
    .from("instagram_accounts")
    .select("id")
    .eq("is_active", true)
    .limit(1)
    .maybeSingle();
  if (error) throw error;
  if (!data) throw new Error("No active Instagram account configured");
  return data.id as string;
}

/**
 * Book `count` scheduled_at values.
 *
 * Slot mode asks Postgres one slot at a time and inserts between calls: two
 * clips asking for "the next free slot" in the same request would otherwise
 * both be handed the same instant, and Instagram would get two reels at 20:00
 * with one silently displacing the other in the queue.
 */
async function resolveSchedule(
  supabase: any,
  accountId: string,
  body: any,
  index: number,
): Promise<string> {
  // "Post now" is not a special publish path — it is a booking for this
  // instant. createDue selects `scheduled_at <= now`, so the next cron pass
  // (within a minute) picks it up and it travels the identical route as a
  // scheduled post: container, poll, publish, retry on failure. Nothing has to
  // bypass the queue, which is what keeps one code path instead of two.
  if (body.post_now) {
    return new Date().toISOString();
  }

  // An explicit time arrives as a WALL CLOCK plus a zone, never as an instant.
  // `new Date("2026-09-24T20:00:00")` here would read it as UTC, because that
  // is this runtime's local zone — a silent multi-hour error. Postgres holds
  // the timezone database, so Postgres does the arithmetic.
  if (body.scheduled_local) {
    const { data, error } = await supabase.rpc("local_to_utc", {
      p_local: body.scheduled_local,
      p_tz: body.timezone || "UTC",
    });
    if (error) throw error;
    if (!data) throw new Error(`Could not resolve ${body.scheduled_local}`);
    return new Date(data).toISOString();
  }

  const { data, error } = await supabase.rpc("next_free_slots", {
    p_account: accountId,
    p_count: 1,
    p_tz: body.timezone || "UTC",
  });
  if (error) throw error;

  const slot = Array.isArray(data) ? data[0] : data;
  if (!slot) {
    throw new Error(
      `No free posting slot available for clip ${index} — add slots in the ` +
        `poster dashboard, or schedule with an explicit time.`,
    );
  }
  return new Date(slot).toISOString();
}

Deno.serve(async (req: Request) => {
  try {
    if (req.method !== "POST") return jsonResponse({ success: false, error: "POST only" }, 405);
    if (!isAuthorized(req)) return jsonResponse({ success: false, error: "Unauthorized" }, 401);

    const body = await req.json();
    const items = Array.isArray(body.items) ? body.items : [body];
    if (!items.length) return jsonResponse({ success: false, error: "No items" }, 400);

    const supabase = adminClient();
    const accountId = await activeAccount(supabase);

    const results: unknown[] = [];
    for (const [index, item] of items.entries()) {
      try {
        if (!item.b2_key) throw new Error("b2_key is required");

        const scheduledAt = await resolveSchedule(supabase, accountId, {
          ...body,
          ...item,
        }, index);

        const { data, error } = await supabase
          .from("instagram_posts")
          .insert({
            account_id: accountId,
            media_type: item.media_type || "REEL",
            // storage_path stays NULL: the one-media-source constraint and the
            // worker both branch on which of the two columns is populated.
            b2_key: item.b2_key,
            caption: (item.caption || "").slice(0, 2200),
            scheduled_at: scheduledAt,
            status: "scheduled",
            source: "openshorts",
            source_ref: item.source_ref || null,
          })
          .select("id, scheduled_at")
          .single();
        if (error) throw error;

        results.push({
          ok: true,
          index,
          post_id: data.id,
          scheduled_at: data.scheduled_at,
          b2_key: item.b2_key,
        });
      } catch (error) {
        // Reported per item, not thrown: OpenShorts deletes the B2 object for
        // every item that failed here, and keeps the ones that landed. A whole
        // -request abort would orphan the objects of items that were fine.
        results.push({ ok: false, index, b2_key: item.b2_key, error: errorMessage(error) });
      }
    }

    const queued = results.filter((r: any) => r.ok).length;
    return jsonResponse({
      success: true,
      queued,
      failed: results.length - queued,
      results,
    });
  } catch (error) {
    console.error(errorMessage(error));
    return jsonResponse({ success: false, error: errorMessage(error) }, 500);
  }
});
