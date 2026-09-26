import { createClient } from "npm:@supabase/supabase-js@2";
import { AwsClient } from "npm:aws4fetch@1.0.20";

const GRAPH_VERSION = "v26.0";
const STORAGE_BUCKET = "instagram-media";
const RETRY_DELAYS_MINUTES = [2, 5, 15];

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
  return createClient(url, key, { auth: { persistSession: false, autoRefreshToken: false } });
}

function isAuthorized(req: Request) {
  const expected = Deno.env.get("CRON_SECRET");
  return Boolean(expected && req.headers.get("x-cron-secret") === expected);
}

/**
 * Presign a GET for a Backblaze B2 object.
 *
 * Only rows pushed by OpenShorts have a b2_key. They live in B2 rather than
 * Supabase Storage because that bucket caps a file at 50 MB and rendered clips
 * run past it; manual uploads from the dashboard are unaffected and still take
 * the createSignedUrl path below.
 *
 * B2's S3-compatible API is plain SigV4, so aws4fetch signs it exactly as it
 * would S3 — only the endpoint host and the region differ (us-east-005, which
 * Backblaze bakes into the hostname).
 *
 * Instagram fetches the video itself, asynchronously, some time after the
 * container is created, so the URL has to outlive the request that makes it.
 * An hour matches what the Supabase Storage branch already allows for REELs.
 * A container that errors or expires sends its row back to `scheduled` with
 * container_id = null, and the next pass mints a brand new URL — an expired
 * link is self-healing rather than fatal.
 */
async function presignB2(key: string, expiresIn: number) {
  const endpoint = Deno.env.get("B2_ENDPOINT");
  const bucket = Deno.env.get("B2_BUCKET");
  const accessKeyId = Deno.env.get("B2_KEY_ID");
  const secretAccessKey = Deno.env.get("B2_APPLICATION_KEY");
  const region = Deno.env.get("B2_REGION") || "us-east-005";
  if (!endpoint || !bucket || !accessKeyId || !secretAccessKey) {
    throw new Error("Missing Backblaze B2 environment variables");
  }

  const client = new AwsClient({ accessKeyId, secretAccessKey, service: "s3", region });

  // Each path segment is encoded separately: the "/" separators in the key
  // must stay literal while everything else in a segment must not. Clip names
  // come from video titles, so spaces and punctuation are the normal case.
  const path = key.split("/").map(encodeURIComponent).join("/");
  const url = new URL(`${endpoint.replace(/\/$/, "")}/${bucket}/${path}`);
  url.searchParams.set("X-Amz-Expires", String(expiresIn));

  const signed = await client.sign(url.toString(), {
    method: "GET",
    aws: { signQuery: true },
  });
  return signed.url;
}

async function instagramCreateContainer(post: any, mediaUrl: string, igUserId: string, token: string) {
  const body = new URLSearchParams();
  body.set("caption", post.caption ?? "");
  body.set("access_token", token);

  if (post.media_type === "REEL") {
    body.set("media_type", "REELS");
    body.set("video_url", mediaUrl);
    body.set("share_to_feed", "true");
  } else if (post.media_type === "IMAGE") {
    body.set("image_url", mediaUrl);
  } else {
    throw new Error(`Unsupported media type: ${post.media_type}`);
  }

  const response = await fetch(`https://graph.instagram.com/${GRAPH_VERSION}/${igUserId}/media`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  const data = await response.json();
  if (!response.ok || !data.id) throw new Error(`Instagram create container failed: ${JSON.stringify(data)}`);
  return data.id as string;
}

async function containerStatus(containerId: string, token: string) {
  const url = new URL(`https://graph.instagram.com/${GRAPH_VERSION}/${containerId}`);
  url.searchParams.set("fields", "status_code");
  url.searchParams.set("access_token", token);
  const response = await fetch(url.toString());
  const data = await response.json();
  if (!response.ok) throw new Error(`Instagram status check failed: ${JSON.stringify(data)}`);
  return data.status_code as string;
}

async function instagramPublish(containerId: string, igUserId: string, token: string) {
  const body = new URLSearchParams();
  body.set("creation_id", containerId);
  body.set("access_token", token);
  const response = await fetch(`https://graph.instagram.com/${GRAPH_VERSION}/${igUserId}/media_publish`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  const data = await response.json();
  if (!response.ok || !data.id) throw new Error(`Instagram publish failed: ${JSON.stringify(data)}`);
  return data.id as string;
}

async function failOrRetry(supabase: any, post: any, error: unknown, resetContainer = true) {
  const attempts = (post.attempt_count ?? 0) + 1;
  const message = errorMessage(error).slice(0, 5000);

  if (attempts > RETRY_DELAYS_MINUTES.length) {
    await supabase.from("instagram_posts").update({
      status: "failed",
      attempt_count: attempts,
      next_attempt_at: null,
      last_error: message,
      processing_at: null,
    }).eq("id", post.id);
    return;
  }

  const next = new Date(Date.now() + RETRY_DELAYS_MINUTES[attempts - 1] * 60_000).toISOString();
  const values: Record<string, unknown> = {
    status: resetContainer ? "scheduled" : "processing",
    attempt_count: attempts,
    next_attempt_at: next,
    last_error: message,
    processing_at: resetContainer ? null : post.processing_at,
  };
  if (resetContainer) values.container_id = null;

  await supabase.from("instagram_posts").update(values).eq("id", post.id);
}

async function createDue(supabase: any, igUserId: string, token: string) {
  const now = new Date().toISOString();
  const { data: posts, error } = await supabase
    .from("instagram_posts")
    .select("*")
    .eq("status", "scheduled")
    .lte("scheduled_at", now)
    .order("scheduled_at", { ascending: true })
    .limit(10);
  if (error) throw error;

  let created = 0;
  for (const post of posts ?? []) {
    if (post.next_attempt_at && new Date(post.next_attempt_at) > new Date()) continue;
    try {
      const { data: claimed, error: claimError } = await supabase
        .from("instagram_posts")
        .update({ status: "creating", processing_at: new Date().toISOString(), last_error: null })
        .eq("id", post.id)
        .eq("status", "scheduled")
        .select("*")
        .maybeSingle();
      if (claimError) throw claimError;
      if (!claimed) continue;
      if (!claimed.storage_path && !claimed.b2_key) {
        throw new Error("row has neither storage_path nor b2_key");
      }

      // Give Meta more time for video fetches than images.
      const expiresIn = claimed.media_type === "REEL" ? 3600 : 900;

      let mediaUrl: string;
      if (claimed.b2_key) {
        // Pushed by OpenShorts: clips run past the 50 MB Supabase Storage cap,
        // so they live in B2 and are signed here.
        mediaUrl = await presignB2(claimed.b2_key, expiresIn);
      } else {
        const { data: signed, error: signedError } = await supabase.storage
          .from(STORAGE_BUCKET)
          .createSignedUrl(claimed.storage_path, expiresIn);
        if (signedError || !signed?.signedUrl) {
          throw signedError || new Error("Could not create signed URL");
        }
        mediaUrl = signed.signedUrl;
      }

      const containerId = await instagramCreateContainer(claimed, mediaUrl, igUserId, token);
      await supabase.from("instagram_posts").update({
        container_id: containerId,
        status: "processing",
        processing_at: new Date().toISOString(),
        next_attempt_at: new Date().toISOString(),
        last_error: null,
      }).eq("id", claimed.id);
      created++;
    } catch (error) {
      await failOrRetry(supabase, post, error, true);
    }
  }
  return created;
}

async function processDue(supabase: any, igUserId: string, token: string) {
  const now = new Date().toISOString();
  const { data: posts, error } = await supabase
    .from("instagram_posts")
    .select("*")
    .eq("status", "processing")
    .order("processing_at", { ascending: true })
    .limit(10);
  if (error) throw error;

  let published = 0;
  for (const post of posts ?? []) {
    if (post.next_attempt_at && post.next_attempt_at > now) continue;
    try {
      if (!post.container_id) throw new Error("container_id is missing");
      const status = await containerStatus(post.container_id, token);

      if (status === "IN_PROGRESS") {
        await supabase.from("instagram_posts").update({
          next_attempt_at: new Date(Date.now() + 60_000).toISOString(),
        }).eq("id", post.id);
        continue;
      }

      if (status === "ERROR" || status === "EXPIRED") {
        await failOrRetry(supabase, post, new Error(`Instagram container status: ${status}`), true);
        continue;
      }

      if (status !== "FINISHED") {
        await supabase.from("instagram_posts").update({
          next_attempt_at: new Date(Date.now() + 60_000).toISOString(),
        }).eq("id", post.id);
        continue;
      }

      const { data: claimed, error: claimError } = await supabase
        .from("instagram_posts")
        .update({ status: "publishing" })
        .eq("id", post.id)
        .eq("status", "processing")
        .select("*")
        .maybeSingle();
      if (claimError) throw claimError;
      if (!claimed) continue;

      try {
        const mediaId = await instagramPublish(post.container_id, igUserId, token);
        await supabase.from("instagram_posts").update({
          status: "published",
          instagram_media_id: mediaId,
          published_at: new Date().toISOString(),
          processing_at: null,
          next_attempt_at: null,
          last_error: null,
        }).eq("id", post.id);
        published++;
      } catch (publishError) {
        await failOrRetry(supabase, post, publishError, false);
      }
    } catch (error) {
      await failOrRetry(supabase, post, error, true);
    }
  }
  return published;
}

Deno.serve(async (req: Request) => {
  try {
    if (!isAuthorized(req)) return jsonResponse({ success: false, error: "Unauthorized" }, 401);

    const igUserId = Deno.env.get("INSTAGRAM_USER_ID");
    const token = Deno.env.get("INSTAGRAM_ACCESS_TOKEN");
    if (!igUserId || !token) throw new Error("Missing Instagram Edge Function secrets");

    const supabase = adminClient();
    const created = await createDue(supabase, igUserId, token);
    const published = await processDue(supabase, igUserId, token);

    return jsonResponse({ success: true, created, published, executed_at: new Date().toISOString() });
  } catch (error) {
    console.error(errorMessage(error));
    return jsonResponse({ success: false, error: errorMessage(error) }, 500);
  }
});
