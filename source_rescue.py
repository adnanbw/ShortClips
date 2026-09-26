"""Take the copy Kaggle already has when YouTube refuses the server.

THE PROBLEM THIS SOLVES
-----------------------
The download is the one stage with no fallback. yt-dlp runs on the server, and
when YouTube challenges the server's datacenter IP the only proof of life is
the account cookies — which expire. Measured on the Oracle box, 26-sep-2026:
the cookies rotated overnight, every attempt came back LOGIN_REQUIRED / "Sign
in to confirm you're not a bot", and the job died before it had a video. The
pipeline could do nothing about it, because the pipeline never had the file.

Meanwhile the Kaggle kernel had already downloaded that exact video, from an
egress YouTube does not challenge, with no cookies at all. Same yt-dlp, same
BgUtils PO token, same mweb client — the technique is identical and the IP is
the entire difference. The kernel then threw the file away, because until now
it only needed the audio.

So: the kernel keeps the video and PUTs it to B2, and the server pulls it from
there when — and only when — its own download was refused.

WHAT THIS IS NOT
----------------
It is not a fix for YouTube blocking. It is a SECOND ROUTE. Kaggle's shared
egress gets blocked too, its GPU quota is weekly and its queue is shared; the
value here is that both routes have to fail on the same day before a job dies,
which is a much rarer event than either one failing alone.

Nor does it make Kaggle load-bearing. The local download still runs first and
still wins; this module is reached only from the `except` around it, and when
it cannot help it re-raises the ORIGINAL download error so the operator sees
the real reason rather than a second, derived one.

CREDENTIALS NEVER LEAVE THIS MACHINE
------------------------------------
The kernel is handed a PRESIGNED PUT URL, not the B2 keys. `worker.py` is
rendered with this job's parameters baked into its source and pushed to
Kaggle, so anything in it is readable by anyone with that account and stays in
the kernel's version history. A presigned URL is scoped to one object key and
one verb and expires; an application key is scoped to the whole bucket and
does not. Getting this backwards would put write access to the bucket the
Instagram poster publishes from into a notebook's history.
"""
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

#: How long the kernel has to use its upload URL. Generous on purpose: the
#: clock starts when the job is DISPATCHED and Kaggle's queue wait is not
#: something this side can predict, so a URL that expires mid-queue would turn
#: a busy hour into a silent loss of the rescue copy.
DEFAULT_TTL_SECONDS = 3 * 60 * 60

#: Object keys already deleted, so cleanup can be called from more than one
#: place without a second pointless API call or a confusing second warning.
_discarded = set()


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def enabled() -> bool:
    """Whether the kernel should be asked to keep a copy of the video.

    Needs B2, because that is where the copy goes. `KAGGLE_SOURCE_RESCUE=0`
    turns it off for a deployment that would rather not spend the ~50s per job
    (a video download instead of an audio one, plus the upload) on a route it
    expects never to need.
    """
    if _env("KAGGLE_SOURCE_RESCUE") == "0":
        return False
    try:
        import instagram_publish
    except Exception:
        return False
    return instagram_publish.storage_configured()


def _bucket() -> str:
    return _env("B2_BUCKET")


def _client():
    # instagram_publish owns the B2 client because it was the first thing to
    # need one, and it is worth importing rather than rebuilding: its
    # `_region()` derives the SigV4 region from the endpoint hostname, and a
    # mismatched region fails every signature with an opaque 403.
    import instagram_publish
    return instagram_publish.client()


def object_key(job_id: str) -> str:
    """Where this job's rescue copy lives.

    Under its own prefix, never beside the published clips: these are
    short-lived working files, and a bucket lifecycle rule or a sweeper should
    be able to find them without touching anything a scheduled post points at.
    """
    import uuid
    safe = "".join(c for c in str(job_id) if c.isalnum() or c in "-_") or "job"
    return f"kaggle-source/{safe}/{uuid.uuid4().hex[:8]}.mp4"


def prepare(job_id: str) -> Optional[Dict[str, Any]]:
    """Mint the upload the kernel should PUT to, or None.

    Returns the dict that travels into the worker's JOB block. It carries a
    URL and a content type and nothing else — see the module docstring.
    """
    if not enabled():
        return None
    key = object_key(job_id)
    try:
        ttl = int(float(_env("KAGGLE_SOURCE_URL_TTL") or DEFAULT_TTL_SECONDS))
    except (TypeError, ValueError):
        ttl = DEFAULT_TTL_SECONDS
    try:
        url = _client().generate_presigned_url(
            "put_object",
            Params={"Bucket": _bucket(), "Key": key,
                    # Signed, so the kernel MUST send exactly this header back
                    # or the signature will not match. Pinning it here and in
                    # the worker is what keeps the two in step.
                    "ContentType": "video/mp4"},
            ExpiresIn=ttl,
        )
    except Exception as exc:
        print(f"⚠️ Could not prepare the Kaggle source rescue "
              f"({type(exc).__name__}: {exc}) — no remote copy will be kept, "
              f"so a refused local download will fail the job as before.")
        return None
    return {"url": url, "key": key, "content_type": "video/mp4"}


def fetch(key: str, destination: str) -> str:
    """Download the rescue copy to `destination`. Raises on failure."""
    started = time.time()
    _client().download_file(_bucket(), key, destination)
    size = os.path.getsize(destination)
    print(f"✅ Recovered {size / 1e6:.1f} MB from B2 in "
          f"{time.time() - started:.1f}s: {destination}")
    return destination


def discard(key: Optional[str]) -> None:
    """Delete the rescue copy. Never raises — this is cleanup, not the task.

    Called whether or not the copy was used. An object nobody will ever read
    is billed for exactly as long as one that gets published, and this one is
    a whole source video rather than a clip.
    """
    if not key or key in _discarded:
        return
    _discarded.add(key)
    try:
        _client().delete_object(Bucket=_bucket(), Key=key)
    except Exception as exc:
        print(f"⚠️ Could not delete the Kaggle source copy {key} "
              f"({type(exc).__name__}: {exc}).")


def source_video(remote) -> Optional[Dict[str, Any]]:
    """The rescue copy a remote job left behind, or None if there is none."""
    if remote is None:
        return None
    getter = getattr(remote, "source_video", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:
        return None


def discard_pending(remote) -> None:
    """Delete whatever copy `remote` uploaded, if any. Safe to call always."""
    info = source_video(remote)
    if info:
        discard(info.get("key"))


def recover(remote, output_dir: str, local_error: BaseException,
            sanitize: Callable[[str], str]) -> Tuple[str, str]:
    """The video from Kaggle, after the local download was refused.

    Raises `local_error` — deliberately the ORIGINAL exception and not a new
    one — whenever this route cannot produce a file. The operator needs to see
    why YouTube refused the server; "the rescue copy was missing" is a
    consequence of that and would bury it.

    `sanitize` is passed in rather than imported: main.py runs as `__main__`,
    so importing it back here would execute that whole module a second time.
    """
    info = source_video(remote)
    # A key alone means "an upload was OFFERED to this kernel" — it is written
    # at dispatch so the object can be deleted even if the run never reports
    # back. Only `uploaded` means a file is actually there.
    if info is not None and not info.get("uploaded"):
        info = None
    if info is None and remote is not None:
        # Normal: the local download fails in seconds and the kernel takes
        # minutes, so there is nothing to find until it has been collected.
        print("⏳ Waiting for the Kaggle kernel — it may already have the "
              "video that YouTube would not give this server...")
        try:
            remote.collect()
        except Exception:
            pass
        info = source_video(remote)
        if info is not None and not info.get("uploaded"):
            info = None
    elif remote is None:
        print("⛔ No Kaggle job was dispatched for this URL, so there is no "
              "second copy of the video to fall back on.")

    if not info or not info.get("key"):
        print("⛔ The Kaggle kernel has no copy of this video either — it may "
              "have been refused as well, or the rescue upload is switched "
              "off (KAGGLE_SOURCE_RESCUE / B2 settings).")
        raise local_error

    title = sanitize(info.get("title") or "youtube_video")
    destination = os.path.join(output_dir, f"{title}.mp4")
    print(f"♻️ The local download was refused; taking the copy Kaggle already "
          f"downloaded ({(info.get('bytes') or 0) / 1e6:.1f} MB)...")
    try:
        fetch(info["key"], destination)
    except Exception as exc:
        print(f"⛔ Could not download the Kaggle copy "
              f"({type(exc).__name__}: {exc}).")
        raise local_error
    finally:
        # Whether or not the download worked. A retried job dispatches a fresh
        # kernel and gets a fresh copy, and leaving whole source videos in the
        # bucket to be swept "later" is how a bucket fills up.
        discard(info.get("key"))

    # The attribution sidecar is normally written by the local download, from
    # the yt-dlp info it had in hand. This path never made that call, so the
    # kernel's copy of the info is the only thing keeping a rescued job from
    # publishing uncredited.
    raw_info = info.get("info")
    if isinstance(raw_info, dict):
        try:
            import attribution
            attribution.save(output_dir, attribution.from_info(raw_info))
        except Exception as exc:
            print(f"⚠️ Could not record source attribution from the Kaggle "
                  f"copy ({type(exc).__name__}: {exc}).")

    return destination, title
