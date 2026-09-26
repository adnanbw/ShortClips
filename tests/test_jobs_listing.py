"""`GET /api/jobs` — the enumeration self-host never had.

Self-host has no library: /api/history and /api/projects live in
cloud/videos.py and read the R2 archive, so with BILLING_ENABLED off the
History tab had nothing to call and was hidden. The jobs themselves were
never lost — _recover_jobs_from_disk rebuilds them from output/ at startup
and /api/status/{job_id} serves them. Only the list was missing.

These tests pin the listing rules against a real OUTPUT_DIR on disk, because
that directory IS the database here and its shape is the contract.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("BILLING_ENABLED", raising=False)
    for mod in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    import app as app_module
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path))
    return TestClient(app_module.app), tmp_path


def _job(root, job_id, title, clips=2, language="en"):
    d = root / job_id
    d.mkdir()
    (d / f"{title}_metadata.json").write_text(json.dumps({
        "selector": "meaningful_v1",
        "transcript": {"language": language},
        "shorts": [{"start": 0, "end": 30} for _ in range(clips)],
    }))
    for i in range(1, clips + 1):
        (d / f"{title}_clip_{i}.mp4").write_bytes(b"\0")
    return d


def test_lists_finished_jobs(client):
    c, root = client
    _job(root, "job-a", "Trevor_Noah_Set", clips=4)
    body = c.get("/api/jobs").json()
    assert len(body["jobs"]) == 1
    job = body["jobs"][0]
    assert job["job_id"] == "job-a"
    assert job["clip_count"] == 4
    assert job["title"] == "Trevor Noah Set"
    assert job["language"] == "en"
    assert job["preview_url"].startswith("/videos/job-a/")


def test_a_job_with_no_metadata_is_not_listed(client):
    """Analysis never finished — there is nothing to reopen, and offering it
    would put a dead row in the list."""
    c, root = client
    (root / "half-done").mkdir()
    (root / "half-done" / "source.mp4").write_bytes(b"\0")
    assert c.get("/api/jobs").json()["jobs"] == []


def test_the_thumbnails_directory_is_not_a_job(client):
    """OUTPUT_DIR holds one directory that is not a job. It has no metadata so
    it would be filtered anyway, but naming it keeps the intent explicit."""
    c, root = client
    (root / "thumbnails").mkdir()
    assert c.get("/api/jobs").json()["jobs"] == []


def test_newest_first(client):
    c, root = client
    a = _job(root, "old", "First")
    b = _job(root, "new", "Second")
    meta_a = next(a.glob("*_metadata.json"))
    meta_b = next(b.glob("*_metadata.json"))
    os.utime(meta_a, (1_700_000_000, 1_700_000_000))
    os.utime(meta_b, (1_800_000_000, 1_800_000_000))
    ids = [j["job_id"] for j in c.get("/api/jobs").json()["jobs"]]
    assert ids == ["new", "old"]


def test_a_job_with_zero_clips_still_lists_with_no_preview(client):
    """The selector can legitimately return nothing. The job still happened and
    the user should see that it did, rather than wondering where it went."""
    c, root = client
    d = root / "empty"
    d.mkdir()
    (d / "Nothing_Found_metadata.json").write_text(
        json.dumps({"shorts": [], "transcript": {"language": "hi"}}))
    job = c.get("/api/jobs").json()["jobs"][0]
    assert job["clip_count"] == 0
    assert job["preview_url"] is None


def test_a_missing_output_dir_is_an_empty_list_not_an_error(client, tmp_path):
    c, root = client
    for child in root.iterdir():
        pass
    import shutil
    shutil.rmtree(root)
    assert c.get("/api/jobs").json()["jobs"] == []


def test_unreadable_metadata_is_skipped_not_fatal(client):
    """One corrupt job must not take the whole history page down."""
    c, root = client
    _job(root, "good", "Good_One")
    bad = root / "bad"
    bad.mkdir()
    (bad / "Broken_metadata.json").write_text("{not json")
    ids = [j["job_id"] for j in c.get("/api/jobs").json()["jobs"]]
    assert ids == ["good"]
