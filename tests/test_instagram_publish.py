"""The bridge from a rendered clip to the self-hosted Instagram poster.

Two properties matter more than the happy path:

1. **No orphans in the bucket.** An object whose row the poster refused will
   never be published and never be seen again, but it is billed forever. Every
   failure path has to take its upload back out.
2. **A wall clock is not an instant.** "20:00" means nothing until a timezone
   is applied, and the three runtimes involved (browser, Python, Deno) would
   each apply a different one.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import instagram_publish as ig


ENV = {
    "B2_ENDPOINT": "https://s3.us-east-005.backblazeb2.com",
    "B2_BUCKET": "Memefreak",
    "B2_KEY_ID": "key-id",
    "B2_APPLICATION_KEY": "application-key",
    "IG_INGEST_URL": "https://example.supabase.co/functions/v1/ingest-post",
    "IG_INGEST_SECRET": "ingest-secret",
}


@pytest.fixture
def configured(monkeypatch):
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("B2_REGION", raising=False)
    ig.reset_client()
    yield
    ig.reset_client()


class FakeB2:
    """Stands in for the boto3 S3 client."""

    def __init__(self, fail_on=()):
        self.uploaded = []
        self.deleted = []
        self.fail_on = set(fail_on)

    def upload_file(self, local_path, bucket, key, ExtraArgs=None):
        if os.path.basename(local_path) in self.fail_on:
            raise RuntimeError("B2 refused the upload")
        self.uploaded.append({"path": local_path, "bucket": bucket, "key": key,
                              "args": ExtraArgs})

    def delete_object(self, Bucket, Key):
        self.deleted.append(Key)


@pytest.fixture
def b2(monkeypatch):
    fake = FakeB2()
    monkeypatch.setattr(ig, "client", lambda: fake)
    return fake


def _clip(tmp_path, name="subtitled_1_Job_clip_1.mp4", size=1024):
    path = tmp_path / name
    path.write_bytes(b"\0" * size)
    return str(path)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_configured_needs_both_halves(monkeypatch, configured):
    assert ig.configured()
    monkeypatch.delenv("IG_INGEST_SECRET")
    assert ig.storage_configured()      # B2 is still fine on its own
    assert not ig.configured()
    assert ig.missing_settings() == ["IG_INGEST_SECRET"]


def test_unconfigured_is_the_normal_self_host_state(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    assert not ig.configured()
    assert len(ig.missing_settings()) == len(ENV)


def test_region_is_derived_from_the_endpoint(configured):
    """Backblaze bakes the region into the hostname, and a mismatch fails
    every signature with an opaque 403. Asking for it twice invites a typo."""
    assert ig._region() == "us-east-005"


def test_region_env_wins(monkeypatch, configured):
    monkeypatch.setenv("B2_REGION", "eu-central-003")
    assert ig._region() == "eu-central-003"


def test_region_falls_back_on_an_unfamiliar_endpoint(monkeypatch, configured):
    monkeypatch.setenv("B2_ENDPOINT", "https://minio.internal:9000")
    assert ig._region() == "us-east-005"


# ---------------------------------------------------------------------------
# Object keys
# ---------------------------------------------------------------------------

def test_object_keys_are_unique_per_upload():
    """Re-queueing a clip after a caption fix must not overwrite the object an
    earlier, still-scheduled post points at — the poster resolves the key at
    publish time, so an overwrite would change what that post publishes."""
    a = ig.object_key("job-1", "clip_1.mp4")
    b = ig.object_key("job-1", "clip_1.mp4")
    assert a != b
    assert a.startswith("openshorts/job-1/") and a.endswith("_clip_1.mp4")


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def _answer(*oks):
    return {"results": [{"ok": ok, "post_id": i, "scheduled_at": "2026-09-24T14:30:00+00:00"}
                        if ok else {"ok": False, "error": "no free slot"}
                        for i, ok in enumerate(oks)]}


def test_clips_are_uploaded_and_queued(tmp_path, configured, b2, monkeypatch):
    sent = {}

    def fake_queue(items, timezone=None, **kwargs):
        sent["items"] = items
        sent["timezone"] = timezone
        return _answer(True, True)

    monkeypatch.setattr(ig, "queue_posts", fake_queue)

    result = ig.schedule_clips([
        {"path": _clip(tmp_path, "a.mp4"), "clip_index": 0, "caption": "one"},
        {"path": _clip(tmp_path, "b.mp4"), "clip_index": 1, "caption": "two"},
    ], "job-1", timezone="Asia/Kolkata")

    assert result["queued"] == 2 and result["failed"] == 0
    assert len(b2.uploaded) == 2
    assert b2.deleted == []
    assert sent["timezone"] == "Asia/Kolkata"
    assert [i["caption"] for i in sent["items"]] == ["one", "two"]
    assert [i["source_ref"] for i in sent["items"]] == ["job-1:0", "job-1:1"]
    assert all(i["media_type"] == "REEL" for i in sent["items"])


def test_uploads_are_tagged_as_mp4(tmp_path, configured, b2, monkeypatch):
    """Instagram fetches the URL itself; a wrong Content-Type is the kind of
    thing that fails inside Meta with an unhelpful container ERROR."""
    monkeypatch.setattr(ig, "queue_posts", lambda *a, **k: _answer(True))
    ig.schedule_clips([{"path": _clip(tmp_path), "clip_index": 0}], "job-1")
    assert b2.uploaded[0]["args"] == {"ContentType": "video/mp4"}


def test_a_rejected_item_has_its_object_deleted(tmp_path, configured, b2, monkeypatch):
    """The poster took one row and refused the other. The refused clip's object
    would otherwise sit in the bucket forever, unreferenced and unpublishable."""
    monkeypatch.setattr(ig, "queue_posts", lambda *a, **k: _answer(True, False))

    result = ig.schedule_clips([
        {"path": _clip(tmp_path, "a.mp4"), "clip_index": 0},
        {"path": _clip(tmp_path, "b.mp4"), "clip_index": 1},
    ], "job-1")

    assert result["queued"] == 1 and result["failed"] == 1
    assert len(b2.deleted) == 1
    kept = [r for r in result["results"] if r.get("ok")][0]
    assert kept["b2_key"] not in b2.deleted


def test_a_failed_ingest_call_deletes_everything(tmp_path, configured, b2, monkeypatch):
    """Nothing got a row, so nothing may keep an object."""
    def boom(*a, **k):
        raise RuntimeError("Poster rejected the batch (503)")
    monkeypatch.setattr(ig, "queue_posts", boom)

    with pytest.raises(RuntimeError):
        ig.schedule_clips([
            {"path": _clip(tmp_path, "a.mp4"), "clip_index": 0},
            {"path": _clip(tmp_path, "b.mp4"), "clip_index": 1},
        ], "job-1")

    assert len(b2.deleted) == 2
    assert sorted(b2.deleted) == sorted(u["key"] for u in b2.uploaded)


def test_a_missing_file_is_reported_not_raised(tmp_path, configured, b2, monkeypatch):
    monkeypatch.setattr(ig, "queue_posts", lambda *a, **k: _answer(True))

    result = ig.schedule_clips([
        {"path": str(tmp_path / "gone.mp4"), "clip_index": 0},
        {"path": _clip(tmp_path, "b.mp4"), "clip_index": 1},
    ], "job-1")

    assert result["queued"] == 1 and result["failed"] == 1
    assert len(b2.uploaded) == 1


def test_nothing_uploadable_never_calls_the_poster(tmp_path, configured, b2, monkeypatch):
    monkeypatch.setattr(ig, "queue_posts",
                        lambda *a, **k: pytest.fail("should not be called"))
    result = ig.schedule_clips(
        [{"path": str(tmp_path / "gone.mp4"), "clip_index": 0}], "job-1")
    assert result["queued"] == 0 and result["failed"] == 1


def test_an_oversize_clip_is_refused_before_it_is_uploaded(tmp_path, configured,
                                                           b2, monkeypatch):
    monkeypatch.setattr(ig, "MAX_UPLOAD_BYTES", 100)
    monkeypatch.setattr(ig, "queue_posts", lambda *a, **k: _answer(True))

    result = ig.schedule_clips(
        [{"path": _clip(tmp_path, "big.mp4", size=500), "clip_index": 0}], "job-1")

    assert result["queued"] == 0
    assert b2.uploaded == []
    assert "1 GB" in result["results"][0]["error"] or "past" in result["results"][0]["error"]


def test_captions_are_truncated_to_instagrams_limit(tmp_path, configured, b2,
                                                    monkeypatch):
    sent = {}
    monkeypatch.setattr(ig, "queue_posts",
                        lambda items, **k: (sent.update(items=items), _answer(True))[1])

    ig.schedule_clips([{"path": _clip(tmp_path), "clip_index": 0,
                        "caption": "x" * 5000}], "job-1")

    assert len(sent["items"][0]["caption"]) == ig.CAPTION_LIMIT == 2200


def test_no_time_means_let_the_poster_pick_a_slot(tmp_path, configured,
                                                          b2, monkeypatch):
    """Absence is the signal. Sending a computed time would duplicate the slot
    logic that already lives in Postgres, where the timezone data is."""
    sent = {}
    monkeypatch.setattr(ig, "queue_posts",
                        lambda items, **k: (sent.update(items=items), _answer(True))[1])

    ig.schedule_clips([{"path": _clip(tmp_path), "clip_index": 0}], "job-1")

    assert "scheduled_local" not in sent["items"][0]


def test_unconfigured_refuses_before_touching_the_network(tmp_path, monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        ig.schedule_clips([{"path": _clip(tmp_path), "clip_index": 0}], "job-1")


# ---------------------------------------------------------------------------
# Wall clock -> instant
# ---------------------------------------------------------------------------
# The conversion itself lives in Postgres (public.local_to_utc), because a
# wall clock read by the wrong runtime is a silent multi-hour error and only
# Postgres is guaranteed to carry a timezone database — Windows and slim
# containers have no system zoneinfo, and `new Date("...T20:00:00")` inside
# Deno reads it as UTC. What this module owes is to forward the pair intact.

def test_an_explicit_time_travels_as_a_wall_clock_plus_zone(tmp_path, configured,
                                                            b2, monkeypatch):
    sent = {}
    monkeypatch.setattr(
        ig, "queue_posts",
        lambda items, timezone=None, **k: (
            sent.update(items=items, timezone=timezone), _answer(True))[1])

    ig.schedule_clips([{"path": _clip(tmp_path), "clip_index": 0,
                        "scheduled_local": "2026-09-24T20:00:00"}],
                      "job-1", timezone="Asia/Kolkata")

    item = sent["items"][0]
    assert item["scheduled_local"] == "2026-09-24T20:00:00"
    assert sent["timezone"] == "Asia/Kolkata"
    # Never pre-converted here: an instant would be this host's guess.
    assert "scheduled_at" not in item


# ---------------------------------------------------------------------------
# Talking to the deployed function
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload or {"results": []}
        self.text = text

    def json(self):
        return self._payload


class FakeHttp:
    def __init__(self, box, status=200):
        self.box = box
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        self.box.update(url=url, json=json, headers=headers)
        return FakeResponse(self.status)


def _patch_http(monkeypatch, box, status=200):
    import httpx
    monkeypatch.setattr(httpx, "Client", lambda **k: FakeHttp(box, status))


def test_the_shared_secret_is_always_sent(monkeypatch, configured):
    box = {}
    _patch_http(monkeypatch, box)
    ig.queue_posts([{"b2_key": "k"}], timezone="UTC")
    assert box["headers"]["x-ingest-secret"] == ENV["IG_INGEST_SECRET"]
    assert box["url"] == ENV["IG_INGEST_URL"]
    assert box["json"] == {"items": [{"b2_key": "k"}], "timezone": "UTC"}


def test_no_anon_key_means_no_authorization_header(monkeypatch, configured):
    """The default. A function deployed with --no-verify-jwt needs nothing
    beyond the shared secret."""
    monkeypatch.delenv("IG_INGEST_ANON_KEY", raising=False)
    box = {}
    _patch_http(monkeypatch, box)
    ig.queue_posts([{"b2_key": "k"}])
    assert "Authorization" not in box["headers"]


def test_an_anon_key_satisfies_the_platform_jwt_gate(monkeypatch, configured):
    """Supabase refuses the request BEFORE the function runs when the function
    was deployed with JWT verification on — a dashboard deploy leaves it on,
    and the 401 says UNAUTHORIZED_NO_AUTH_HEADER, which looks like a wrong
    secret. The anon key is public by design and is not the authorisation."""
    monkeypatch.setenv("IG_INGEST_ANON_KEY", "anon-key")
    box = {}
    _patch_http(monkeypatch, box)
    ig.queue_posts([{"b2_key": "k"}])
    assert box["headers"]["Authorization"] == "Bearer anon-key"
    assert box["headers"]["apikey"] == "anon-key"
    # Still not what authorises the call.
    assert box["headers"]["x-ingest-secret"] == ENV["IG_INGEST_SECRET"]


def test_an_http_error_carries_the_body_into_the_message(monkeypatch, configured):
    box = {}
    _patch_http(monkeypatch, box, status=401)
    with pytest.raises(RuntimeError, match="401"):
        ig.queue_posts([{"b2_key": "k"}])


# ---------------------------------------------------------------------------
# Post now
# ---------------------------------------------------------------------------
# Not a separate publish path. The poster's createDue selects
# `scheduled_at <= now`, so booking at the current instant means the next cron
# pass picks it up and it travels the identical container/poll/publish route,
# retries included. One code path, not two.

def test_post_now_sets_the_flag_and_sends_no_time(tmp_path, configured, b2,
                                                   monkeypatch):
    sent = {}
    monkeypatch.setattr(
        ig, "queue_posts",
        lambda items, **k: (sent.update(items=items, kw=k), _answer(True))[1])

    ig.schedule_clips([{"path": _clip(tmp_path), "clip_index": 0}], "job-1",
                      timezone="Asia/Kolkata", post_now=True)

    assert sent["kw"]["post_now"] is True
    # The poster stamps the instant itself; a time computed here would be this
    # host's clock, which is not the one the queue is ordered by.
    assert "scheduled_local" not in sent["items"][0]


def test_post_now_is_off_by_default(tmp_path, configured, b2, monkeypatch):
    sent = {}
    monkeypatch.setattr(
        ig, "queue_posts",
        lambda items, **k: (sent.update(kw=k), _answer(True))[1])
    ig.schedule_clips([{"path": _clip(tmp_path), "clip_index": 0}], "job-1")
    assert sent["kw"]["post_now"] is False


def test_the_flag_reaches_the_request_body(monkeypatch, configured):
    box = {}
    _patch_http(monkeypatch, box)
    ig.queue_posts([{"b2_key": "k"}], timezone="UTC", post_now=True)
    assert box["json"]["post_now"] is True


def test_scheduling_normally_omits_the_flag_entirely(monkeypatch, configured):
    """Absent, not false: the poster reads it as truthy and an explicit false
    would be one more thing for the two sides to disagree about."""
    box = {}
    _patch_http(monkeypatch, box)
    ig.queue_posts([{"b2_key": "k"}], timezone="UTC")
    assert "post_now" not in box["json"]


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------
# A clip is tens of megabytes over a home upstream — roughly two minutes each,
# measured. Without progress the caller has nothing to show for the whole run,
# the UI looks hung, and the user starts the same upload again beside the
# first. That really happened: six overlapping uploads of the same 7 clips put
# 870 MB of duplicates in the bucket with no rows pointing at any of them.

def test_progress_is_reported_per_clip(tmp_path, configured, b2, monkeypatch):
    monkeypatch.setattr(ig, "queue_posts", lambda *a, **k: _answer(True, True, True))
    seen = []

    ig.schedule_clips([
        {"path": _clip(tmp_path, "a.mp4"), "clip_index": 0},
        {"path": _clip(tmp_path, "b.mp4"), "clip_index": 1},
        {"path": _clip(tmp_path, "c.mp4"), "clip_index": 2},
    ], "job-1", on_progress=seen.append)

    stages = [(p["stage"], p["done"]) for p in seen]
    assert stages == [("uploading", 0), ("uploading", 1), ("uploading", 2),
                      ("uploading", 3), ("queueing", 3)]
    assert all(p["total"] == 3 for p in seen)


def test_a_skipped_clip_still_advances_the_count(tmp_path, configured, b2,
                                                 monkeypatch):
    """Otherwise the bar stalls on a missing file and looks like the hang this
    was built to remove."""
    monkeypatch.setattr(ig, "queue_posts", lambda *a, **k: _answer(True))
    seen = []

    ig.schedule_clips([
        {"path": str(tmp_path / "gone.mp4"), "clip_index": 0},
        {"path": _clip(tmp_path, "b.mp4"), "clip_index": 1},
    ], "job-1", on_progress=seen.append)

    assert [p["done"] for p in seen] == [0, 2, 2]


def test_a_broken_progress_callback_never_fails_the_upload(tmp_path, configured,
                                                           b2, monkeypatch):
    """Progress is telemetry. Losing it must not lose the clips."""
    monkeypatch.setattr(ig, "queue_posts", lambda *a, **k: _answer(True))

    def boom(_):
        raise RuntimeError("the UI went away")

    result = ig.schedule_clips(
        [{"path": _clip(tmp_path), "clip_index": 0}], "job-1", on_progress=boom)
    assert result["queued"] == 1


def test_no_callback_is_fine(tmp_path, configured, b2, monkeypatch):
    monkeypatch.setattr(ig, "queue_posts", lambda *a, **k: _answer(True))
    assert ig.schedule_clips(
        [{"path": _clip(tmp_path), "clip_index": 0}], "job-1")["queued"] == 1
