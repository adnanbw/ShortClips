# Patch: `instagram-publish-worker` signs Backblaze B2 as well as Supabase Storage

This is the **only** change to the existing publish worker. Container creation,
status polling, the retry ladder, `media_publish`, the atomic status claiming —
all untouched. A row with `storage_path` takes the identical code path it takes
today, so manual uploads from the poster dashboard cannot regress.

## 1. Add the signer

At the top of `supabase/functions/instagram-publish-worker/index.ts`, beside the
existing `createClient` import:

```ts
import { AwsClient } from "npm:aws4fetch@1.0.20";
```

Then, next to the other helpers:

```ts
/**
 * Presign a GET for a Backblaze B2 object.
 *
 * B2's S3-compatible API is plain SigV4, so aws4fetch signs it exactly as it
 * would S3 — the only things that differ are the endpoint host and the region
 * baked into the endpoint (`us-east-005`).
 *
 * Instagram fetches the video itself, asynchronously, some time after the
 * container is created. The URL therefore has to outlive the request that
 * makes it; an hour is what the Supabase Storage branch already allows for
 * REELs. A container that fails or expires sends the row back to `scheduled`
 * with `container_id = null`, and the next pass mints a brand new URL, so an
 * expired link is self-healing rather than fatal.
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

  const client = new AwsClient({
    accessKeyId,
    secretAccessKey,
    service: "s3",
    region,
  });

  // Each path segment is encoded separately: the key contains "/" separators
  // that must stay literal, while everything else in a segment must not.
  const path = key.split("/").map(encodeURIComponent).join("/");
  const url = new URL(`${endpoint.replace(/\/$/, "")}/${bucket}/${path}`);
  url.searchParams.set("X-Amz-Expires", String(expiresIn));

  const signed = await client.sign(url.toString(), {
    method: "GET",
    aws: { signQuery: true },
  });
  return signed.url;
}
```

## 2. Branch on which column is set

In `createDue`, replace this block:

```ts
      // Give Meta more time for video fetches than images.
      const expiresIn = claimed.media_type === "REEL" ? 3600 : 900;
      const { data: signed, error: signedError } = await supabase.storage
        .from(STORAGE_BUCKET)
        .createSignedUrl(claimed.storage_path, expiresIn);
      if (signedError || !signed?.signedUrl) throw signedError || new Error("Could not create signed URL");

      const containerId = await instagramCreateContainer(claimed, signed.signedUrl, igUserId, token);
```

with:

```ts
      // Give Meta more time for video fetches than images.
      const expiresIn = claimed.media_type === "REEL" ? 3600 : 900;

      let mediaUrl: string;
      if (claimed.b2_key) {
        // Pushed by OpenShorts. Clips run past the 50 MB Supabase Storage cap,
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
```

And delete the now-dead guard two lines above it:

```ts
      if (!claimed.storage_path) throw new Error("storage_path is empty");
```

replacing it with:

```ts
      if (!claimed.storage_path && !claimed.b2_key) {
        throw new Error("row has neither storage_path nor b2_key");
      }
```

## 3. Secrets

```
supabase secrets set \
  B2_ENDPOINT=https://s3.us-east-005.backblazeb2.com \
  B2_BUCKET=Memefreak \
  B2_REGION=us-east-005 \
  B2_KEY_ID=<read-only key id> \
  B2_APPLICATION_KEY=<read-only application key> \
  IG_INGEST_SECRET=<a long random string>
```

The worker only ever reads from B2, so give it a **read-only** application key
scoped to the bucket. OpenShorts gets a separate read-write key. If a key leaks
from Supabase, it cannot overwrite or delete your media.

`IG_INGEST_SECRET` is consumed by the `ingest-post` function, not the worker,
but both are set on the same project. It MUST be byte-identical to the
`IG_INGEST_SECRET` in OpenShorts' `.env` — a mismatch comes back as a plain
`{"success":false,"error":"Unauthorized"}`, which looks the same as a missing
header. (The function also still accepts the older name `INGEST_SECRET`.)

## 4. Deploy

```
supabase functions deploy instagram-publish-worker
supabase functions deploy ingest-post --no-verify-jwt
```

Nothing about the pg_cron job changes — it keeps calling the worker every
minute with `x-cron-secret` from Vault.

## What is deliberately NOT in this patch

- **No inline delete after publish.** The original plan ended `publish → delete
  from B2`. If `media_publish` succeeds but the row update fails, the retry
  would find no media and burn its remaining attempts on a reel that is already
  live. `media_deleted_at` plus a sweeper is the safe shape, and on B2's free
  tier there is no hurry.
- **No token refresh.** You are handling that.
- **No dashboard preview for B2 rows.** `Dashboard.js` signs Supabase Storage
  only, so OpenShorts rows will show caption and time but no thumbnail until
  the same branch is added there. Cosmetic, and separate.
