"""Queue rendered clips onto the self-hosted Instagram poster.

The poster is a separate Supabase project (pg_cron every minute -> an Edge
Function -> the Instagram Graph API). This module is the bridge, and it only
ever talks OUTBOUND: OpenShorts runs on a machine behind NAT that Supabase
cannot reach, so the flow is push-only by necessity as well as by design.

    clip on disk
        -> upload to Backblaze B2            (this module)
        -> POST the object key to Supabase   (this module)
        -> ... nothing happens until the scheduled time ...
        -> the worker mints a fresh presigned GET
        -> Instagram fetches the video from B2 itself
        -> publish

Only the KEY crosses the wire, never a URL. A presigned URL minted here would
have to survive until the post goes out — days, in a normal schedule — and this
process may be switched off long before then. The worker signs at publish time,
which also means a failed container simply gets a new URL on the next pass.

WHY B2 AND NOT SUPABASE STORAGE. The poster's own bucket is capped at 50 MB a
file ("Free Supabase projects have a 50 MB global max upload limit", its
frontend-setup.sql). Clips out of this pipeline measured 28.8 / 30.8 / 33.9 /
49.2 / 51.2 / 55.2 MB across two real jobs — two of six already do not fit.

It is strictly an accelerator over doing it by hand, and every failure returns a
message rather than raising: a clip that cannot be queued is still on disk and
still downloadable.
"""
import os
import uuid
from typing import Any, Dict, List, Optional

#: Instagram's caption limit. The poster's own dashboard shows the same 2200.
CAPTION_LIMIT = 2200

#: Refuse to upload something that obviously is not a rendered clip. Instagram
#: allows up to 1 GB for a Reel; anything past that is a bug upstream, not a
#: video worth spending upload bandwidth on.
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def storage_configured() -> bool:
    """True when B2 is wired up well enough to upload."""
    return all(_env(n) for n in
               ("B2_ENDPOINT", "B2_BUCKET", "B2_KEY_ID", "B2_APPLICATION_KEY"))


def ingest_configured() -> bool:
    """True when the Supabase ingest endpoint is wired up."""
    return all(_env(n) for n in ("IG_INGEST_URL", "IG_INGEST_SECRET"))


def configured() -> bool:
    return storage_configured() and ingest_configured()


def missing_settings() -> List[str]:
    """Which env vars are absent — so the API can say what to fix."""
    return [n for n in ("B2_ENDPOINT", "B2_BUCKET", "B2_KEY_ID",
                        "B2_APPLICATION_KEY", "IG_INGEST_URL",
                        "IG_INGEST_SECRET") if not _env(n)]


def _region() -> str:
    """The SigV4 region for the B2 endpoint.

    Backblaze bakes it into the hostname (s3.us-east-005.backblazeb2.com), and
    a mismatched region makes every signature fail with an opaque 403, so it is
    derived from the endpoint rather than asked for twice. B2_REGION overrides.
    """
    explicit = _env("B2_REGION")
    if explicit:
        return explicit
    host = _env("B2_ENDPOINT").split("//")[-1]
    parts = host.split(".")
    # s3.<region>.backblazeb2.com
    if len(parts) >= 3 and parts[0] == "s3":
        return parts[1]
    return "us-east-005"


_client = None


def client():
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config
        _client = boto3.client(
            "s3",
            endpoint_url=_env("B2_ENDPOINT"),
            aws_access_key_id=_env("B2_KEY_ID"),
            aws_secret_access_key=_env("B2_APPLICATION_KEY"),
            region_name=_region(),
            config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
        )
    return _client


def reset_client():
    """Drop the cached client (tests, and an env change without a restart)."""
    global _client
    _client = None


def object_key(job_id: str, filename: str) -> str:
    """Where a clip lives in the bucket.

    The random component is not decoration. Re-queueing a clip after a caption
    fix must not overwrite an object that an earlier, still-scheduled post is
    pointing at — the worker resolves the key at publish time, so an overwrite
    would silently change what that post publishes.
    """
    return f"openshorts/{job_id}/{uuid.uuid4().hex[:8]}_{filename}"


def upload_clip(local_path: str, key: str) -> int:
    """Upload one rendered clip. Returns its size in bytes."""
    size = os.path.getsize(local_path)
    if size > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"{os.path.basename(local_path)} is {size / 1e6:.0f} MB — past "
            f"Instagram's 1 GB limit for a Reel.")
    client().upload_file(local_path, _env("B2_BUCKET"), key,
                         ExtraArgs={"ContentType": "video/mp4"})
    return size


def delete_object(key: str) -> bool:
    """Remove an object. Never raises: this is cleanup, not the task."""
    try:
        client().delete_object(Bucket=_env("B2_BUCKET"), Key=key)
        return True
    except Exception as e:
        print(f"⚠️ Could not delete {key} from B2 ({type(e).__name__}: {e}) — "
              f"it will be swept later.")
        return False


def queue_posts(items: List[Dict[str, Any]],
                timezone: Optional[str] = None,
                post_now: bool = False,
                timeout: float = 30.0) -> Dict[str, Any]:
    """Hand the uploaded keys to the poster's ingest function.

    `items` are dicts of {b2_key, caption, source_ref, scheduled_at?}. A missing
    `scheduled_at` means "book the next free posting slot", which the poster
    resolves in Postgres against `timezone` — its `slot_time` column has no zone
    of its own, so the zone has to travel with the request or a 20:00 slot fires
    at the wrong hour.
    """
    import httpx

    payload: Dict[str, Any] = {"items": items}
    if timezone:
        payload["timezone"] = timezone
    if post_now:
        # Books every item at the current instant. The poster's cron selects
        # scheduled_at <= now, so they go out on its next pass.
        payload["post_now"] = True

    headers = {"x-ingest-secret": _env("IG_INGEST_SECRET"),
               "Content-Type": "application/json"}

    # Supabase's gateway checks for a JWT before a request ever reaches the
    # function, unless that function was deployed with --no-verify-jwt. A
    # dashboard deploy leaves it ON, and the refusal is indistinguishable from
    # a wrong secret at a glance: {"code":"UNAUTHORIZED_NO_AUTH_HEADER"} comes
    # back with a 401 from the platform, not from any code in this repo.
    #
    # Setting IG_INGEST_ANON_KEY satisfies that check without a redeploy. The
    # anon key is designed to be public — it ships in the poster's own frontend
    # — and it is NOT the authorisation here: x-ingest-secret still is, and the
    # function refuses anything without it.
    anon = _env("IG_INGEST_ANON_KEY")
    if anon:
        headers["Authorization"] = f"Bearer {anon}"
        headers["apikey"] = anon

    with httpx.Client(timeout=timeout) as http:
        response = http.post(_env("IG_INGEST_URL"), json=payload, headers=headers)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Poster rejected the batch ({response.status_code}): "
            f"{response.text[:500]}")
    return response.json()


def schedule_clips(clips: List[Dict[str, Any]], job_id: str,
                   timezone: Optional[str] = None,
                   post_now: bool = False,
                   on_progress=None) -> Dict[str, Any]:
    """Upload each clip to B2 and queue it on the poster.

    `clips` are dicts of {path, caption, clip_index, scheduled_at?}.

    Uploads happen first and the queue call is made ONCE for the batch, because
    slot allocation has to see the whole batch: booking them one request at a
    time would be correct but slower, and booking them in one request lets the
    poster hand out consecutive slots without a round trip each.

    Any object whose row the poster refused is deleted again. An orphan in the
    bucket is invisible, costs storage forever and would never be published —
    the poster's own dashboard does the same thing after a failed insert.
    """
    if not configured():
        raise RuntimeError(
            "Instagram publishing is not configured. Missing: "
            + ", ".join(missing_settings()))

    uploaded: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    # Uploading is the slow part by a wide margin — a clip is tens of megabytes
    # and a home connection's UPSTREAM is what carries it. Measured on a real
    # batch: 28-43 MB apiece at roughly two minutes each. Without this callback
    # the caller has nothing to report for the whole run, the UI looks hung, and
    # the user starts the same upload again alongside the first.
    def report(stage, done, **extra):
        if on_progress:
            try:
                on_progress({"stage": stage, "done": done,
                             "total": len(clips), **extra})
            except Exception:
                pass   # progress is telemetry, never the task

    report("uploading", 0)
    for clip in clips:
        path = clip["path"]
        index = clip.get("clip_index")
        if not os.path.exists(path):
            failures.append({"clip_index": index, "error": f"missing file: {path}"})
            continue
        key = object_key(job_id, os.path.basename(path))
        try:
            size = upload_clip(path, key)
        except Exception as e:
            failures.append({"clip_index": index,
                             "error": f"{type(e).__name__}: {e}"})
            continue
        item: Dict[str, Any] = {
            "b2_key": key,
            "media_type": "REEL",
            "caption": (clip.get("caption") or "")[:CAPTION_LIMIT],
            "source_ref": f"{job_id}:{index}",
        }
        if clip.get("scheduled_local"):
            # A WALL CLOCK, deliberately, not an instant: the poster resolves it
            # against `timezone` in Postgres, which is the only runtime in this
            # pipeline guaranteed to have a timezone database. Converting here
            # would need `tzdata` on any host without a system zoneinfo — every
            # Windows box and most slim containers.
            item["scheduled_local"] = clip["scheduled_local"]
        uploaded.append({"item": item, "clip_index": index, "size": size})
        print(f"   ☁️ Uploaded clip {index} to B2 ({size / 1e6:.1f} MB): {key}")
        report("uploading", len(uploaded) + len(failures), last_clip=index)

    if not uploaded:
        return {"queued": 0, "failed": len(failures), "results": failures}

    # Everything that could be uploaded has been, so the count here is clips
    # PROCESSED, not clips uploaded. Using len(uploaded) made the bar run
    # backwards whenever a clip was skipped — 2 of 2, then 1 of 2.
    report("queueing", len(uploaded) + len(failures))
    try:
        answer = queue_posts([u["item"] for u in uploaded], timezone=timezone,
                             post_now=post_now)
    except Exception as e:
        # The whole call failed, so none of these objects has a row pointing at
        # it. Take them back out rather than leaving the bucket littered.
        for u in uploaded:
            delete_object(u["item"]["b2_key"])
        raise

    results: List[Dict[str, Any]] = list(failures)
    for position, outcome in enumerate(answer.get("results") or []):
        if position >= len(uploaded):
            break
        record = uploaded[position]
        entry = {"clip_index": record["clip_index"],
                 "b2_key": record["item"]["b2_key"]}
        if outcome.get("ok"):
            entry.update(ok=True, post_id=outcome.get("post_id"),
                         scheduled_at=outcome.get("scheduled_at"))
        else:
            entry.update(ok=False, error=outcome.get("error"))
            delete_object(record["item"]["b2_key"])
        results.append(entry)

    queued = sum(1 for r in results if r.get("ok"))
    return {"queued": queued, "failed": len(results) - queued, "results": results}
