"""The download had no fallback, and the thing that breaks it is not local.

When YouTube challenges this server's datacenter IP and the account cookies
have expired, every attempt fails and the job dies before it has a video. The
Kaggle kernel fetches the SAME url from an egress that is not challenged and
already has the file. So it keeps it.

Two properties matter more than the feature itself:

1. Kaggle must stay OPTIONAL. The local download still runs first and still
   wins; this path is reached only from the `except` around it, and when it
   cannot help it re-raises the ORIGINAL download error, because "the rescue
   copy was missing" would bury the reason YouTube refused the server.
2. The B2 KEYS must never reach Kaggle. worker.py is rendered with the job
   baked into its source and pushed to a Kaggle account, where it stays in
   the kernel's version history. A presigned PUT is scoped to one object and
   one verb and expires; an application key is neither.
"""
import ast
import io
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kaggle_worker
import source_rescue

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = io.open(os.path.join(ROOT, "kaggle-worker", "worker.py"),
                 encoding="utf-8").read()


def job_of(source):
    """The worker's JOB dict, parsed out of its source.

    Via the AST rather than by slicing on "\\n}": render_worker substitutes a
    `pprint.pformat` literal, which closes on the last line rather than in
    column zero, so a text search finds the end of some later function.
    """
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", None) == "JOB"):
            return ast.literal_eval(node.value)
    raise AssertionError("worker.py has no JOB assignment")


def func_of(source, name):
    """One function's source text, by name. Same reason: no text slicing."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"worker.py has no {name}()")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("KAGGLE_SOURCE_RESCUE", "KAGGLE_SOURCE_URL_TTL", "B2_ENDPOINT",
                 "B2_BUCKET", "B2_KEY_ID", "B2_APPLICATION_KEY", "B2_REGION"):
        monkeypatch.delenv(name, raising=False)
    source_rescue._discarded.clear()


def _b2(monkeypatch):
    monkeypatch.setenv("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
    monkeypatch.setenv("B2_BUCKET", "Memefreak")
    monkeypatch.setenv("B2_KEY_ID", "keyid")
    monkeypatch.setenv("B2_APPLICATION_KEY", "appkey")


class _FakeS3:
    """Records what the rescue asks of B2 without talking to it."""

    def __init__(self, body=b"video bytes"):
        self.body = body
        self.signed = []
        self.deleted = []
        self.downloaded = []
        self.download_error = None

    def generate_presigned_url(self, operation, Params=None, ExpiresIn=None):
        self.signed.append((operation, Params, ExpiresIn))
        return ("https://s3.us-east-005.backblazeb2.com/"
                f"{Params['Bucket']}/{Params['Key']}?X-Amz-Signature=deadbeef")

    def download_file(self, bucket, key, path):
        if self.download_error:
            raise self.download_error
        self.downloaded.append(key)
        with open(path, "wb") as handle:
            handle.write(self.body)

    def delete_object(self, Bucket=None, Key=None):
        self.deleted.append(Key)


def _s3(monkeypatch, client=None):
    client = client or _FakeS3()
    monkeypatch.setattr(source_rescue, "_client", lambda: client)
    return client


# --- when it is on at all ---------------------------------------------------

class TestEnabled:
    def test_off_without_b2(self):
        """The copy has nowhere to go, so the kernel is not asked for one."""
        assert source_rescue.enabled() is False

    def test_on_once_b2_is_configured(self, monkeypatch):
        _b2(monkeypatch)
        assert source_rescue.enabled() is True

    def test_it_can_be_switched_off(self, monkeypatch):
        """The kernel then downloads audio only, as it did before."""
        _b2(monkeypatch)
        monkeypatch.setenv("KAGGLE_SOURCE_RESCUE", "0")
        assert source_rescue.enabled() is False

    def test_prepare_returns_nothing_when_off(self):
        assert source_rescue.prepare("job") is None


# --- the credential boundary ------------------------------------------------

class TestCredentialsStayHere:
    def test_the_kernel_gets_a_presigned_put_not_a_key(self, monkeypatch):
        _b2(monkeypatch)
        client = _s3(monkeypatch)
        upload = source_rescue.prepare("job-1")
        operation, params, _ttl = client.signed[0]
        assert operation == "put_object"
        assert params["Bucket"] == "Memefreak"
        assert "appkey" not in upload["url"]
        assert "keyid" not in upload["url"]

    def test_the_signed_content_type_is_what_the_worker_sends(self, monkeypatch):
        """A mismatch here is a 403 from B2 with nothing saying why."""
        _b2(monkeypatch)
        client = _s3(monkeypatch)
        upload = source_rescue.prepare("job-1")
        assert client.signed[0][1]["ContentType"] == "video/mp4"
        assert upload["content_type"] == "video/mp4"
        assert 'f"Content-Type: {content_type}"' in WORKER

    def test_no_b2_credential_reaches_the_pushed_worker(self, monkeypatch):
        """The rendered worker is pushed to Kaggle and kept forever."""
        _b2(monkeypatch)
        _s3(monkeypatch)
        rendered = kaggle_worker.render_worker(WORKER, {
            "job_id": "j", "url": "https://youtu.be/x",
            "source_upload": {"url": source_rescue.prepare("j")["url"],
                              "content_type": "video/mp4"},
        })
        assert "appkey" not in rendered
        assert "keyid" not in rendered

    def test_the_upload_url_is_never_logged(self):
        """The kernel log is downloaded to the backend and can reach a job's
        own log, and this URL is a writable handle on the bucket until it
        expires. It has to be IN the kernel — the job block is baked into the
        pushed source — but it must not be echoed anywhere.
        """
        block = func_of(WORKER, "upload_source")
        assert "run([" not in block, "run() echoes the command it is given"
        for line in block.splitlines():
            if not line.strip().startswith("log("):
                continue
            # The variable itself, interpolated or passed. "curl" contains the
            # letters, so match the identifier rather than the substring.
            assert "{url" not in line
            assert "(url" not in line
            assert ", url" not in line

    def test_the_key_is_unguessable_per_job(self, monkeypatch):
        """Re-running a job must not overwrite an object still being read."""
        _b2(monkeypatch)
        assert source_rescue.object_key("j") != source_rescue.object_key("j")

    def test_rescue_copies_live_apart_from_published_clips(self):
        """A sweeper on this prefix must not be able to touch a clip that a
        scheduled Instagram post resolves at publish time."""
        assert source_rescue.object_key("j").startswith("kaggle-source/")


# --- recovery ---------------------------------------------------------------

class _Remote:
    def __init__(self, source=None, collect_fills=None):
        self._source = source or {}
        self._collect_fills = collect_fills
        self.collected = 0

    def source_video(self):
        return dict(self._source) if self._source else None

    def collect(self, timeout=None):
        self.collected += 1
        if self._collect_fills is not None:
            self._source = self._collect_fills
        return None


def _boom():
    return RuntimeError("Sign in to confirm you're not a bot")


class TestRecovery:
    def test_the_rescued_file_lands_on_disk(self, monkeypatch, tmp_path):
        _b2(monkeypatch)
        client = _s3(monkeypatch)
        remote = _Remote({"key": "kaggle-source/j/ab.mp4", "uploaded": True,
                          "bytes": 1234, "title": "A Talk"})
        path, title = source_rescue.recover(
            remote, str(tmp_path), _boom(), lambda t: t.replace(" ", "_"))
        assert os.path.exists(path)
        assert title == "A_Talk"
        assert client.downloaded == ["kaggle-source/j/ab.mp4"]

    def test_it_waits_for_a_kernel_that_has_not_finished(self, monkeypatch,
                                                         tmp_path):
        """The local download fails in seconds; the kernel takes minutes."""
        _b2(monkeypatch)
        _s3(monkeypatch)
        remote = _Remote(collect_fills={"key": "k", "uploaded": True,
                                        "title": "t"})
        source_rescue.recover(remote, str(tmp_path), _boom(), lambda t: t)
        assert remote.collected == 1

    def test_the_copy_is_deleted_after_it_is_taken(self, monkeypatch, tmp_path):
        _b2(monkeypatch)
        client = _s3(monkeypatch)
        remote = _Remote({"key": "k", "uploaded": True, "title": "t"})
        source_rescue.recover(remote, str(tmp_path), _boom(), lambda t: t)
        assert client.deleted == ["k"]

    def test_the_copy_is_deleted_even_when_the_download_fails(
            self, monkeypatch, tmp_path):
        """Otherwise a whole source video is billed for forever."""
        _b2(monkeypatch)
        client = _s3(monkeypatch)
        client.download_error = OSError("connection reset")
        remote = _Remote({"key": "k", "uploaded": True, "title": "t"})
        with pytest.raises(RuntimeError):
            source_rescue.recover(remote, str(tmp_path), _boom(), lambda t: t)
        assert client.deleted == ["k"]

    def test_attribution_survives_the_rescue(self, monkeypatch, tmp_path):
        """The sidecar is normally written by the local yt-dlp call, which
        never happened here — so a rescued job would publish uncredited."""
        _b2(monkeypatch)
        _s3(monkeypatch)
        remote = _Remote({
            "key": "k", "uploaded": True, "title": "t",
            "info": {"uploader": "Sharon Verma", "uploader_id": "@sharonv",
                     "description": "Follow me instagram.com/sharonvermacomedy"},
        })
        source_rescue.recover(remote, str(tmp_path), _boom(), lambda t: t)
        import attribution
        saved = attribution.load(str(tmp_path))
        assert saved["uploader"] == "Sharon Verma"
        assert saved["socials"][0]["handle"] == "sharonvermacomedy"


class TestItNeverInventsAFailure:
    """Every dead end re-raises the ORIGINAL download error.

    The operator has to see that YouTube refused the server. A second,
    derived error ("no rescue copy") describes a consequence and hides the
    cause — which is the one thing they can actually act on.
    """

    def _expect_original(self, remote, tmp_path):
        original = _boom()
        with pytest.raises(RuntimeError) as caught:
            source_rescue.recover(remote, str(tmp_path), original, lambda t: t)
        assert caught.value is original

    def test_no_kaggle_job_at_all(self, tmp_path):
        self._expect_original(None, tmp_path)

    def test_the_kernel_was_blocked_too(self, monkeypatch, tmp_path):
        _b2(monkeypatch)
        _s3(monkeypatch)
        self._expect_original(_Remote({}), tmp_path)

    def test_an_offered_upload_that_never_happened(self, monkeypatch, tmp_path):
        """A key alone is written at dispatch; only `uploaded` means a file."""
        _b2(monkeypatch)
        _s3(monkeypatch)
        self._expect_original(_Remote({"key": "k"}), tmp_path)

    def test_b2_refusing_the_download(self, monkeypatch, tmp_path):
        _b2(monkeypatch)
        client = _s3(monkeypatch)
        client.download_error = OSError("403")
        self._expect_original(
            _Remote({"key": "k", "uploaded": True, "title": "t"}), tmp_path)


class TestCleanup:
    def test_an_unused_copy_is_deleted(self, monkeypatch):
        """The common case: the local download won and nobody read this."""
        _b2(monkeypatch)
        client = _s3(monkeypatch)
        source_rescue.discard_pending(
            _Remote({"key": "k", "uploaded": True}))
        assert client.deleted == ["k"]

    def test_an_offered_but_unconfirmed_upload_is_still_deleted(
            self, monkeypatch):
        """The kernel may have uploaded and then died before reporting."""
        _b2(monkeypatch)
        client = _s3(monkeypatch)
        source_rescue.discard_pending(_Remote({"key": "k"}))
        assert client.deleted == ["k"]

    def test_deleting_twice_costs_one_call(self, monkeypatch):
        _b2(monkeypatch)
        client = _s3(monkeypatch)
        source_rescue.discard("k")
        source_rescue.discard("k")
        assert client.deleted == ["k"]

    def test_nothing_to_clean_up_is_not_an_error(self, monkeypatch):
        _b2(monkeypatch)
        _s3(monkeypatch)
        source_rescue.discard_pending(None)
        source_rescue.discard(None)

    def test_a_b2_outage_does_not_propagate(self, monkeypatch):
        """Cleanup is not the task; it must not be able to fail a job."""
        _b2(monkeypatch)

        class _Broken:
            def delete_object(self, **kwargs):
                raise OSError("b2 down")
        monkeypatch.setattr(source_rescue, "_client", lambda: _Broken())
        source_rescue.discard("k")


# --- the dispatcher end -----------------------------------------------------

class TestTheKernelIsToldWhatToDo:
    def test_the_job_carries_the_upload_when_rescue_is_on(self, monkeypatch):
        _b2(monkeypatch)
        _s3(monkeypatch)
        upload = source_rescue.prepare("j")
        rendered = kaggle_worker.render_worker(WORKER, {
            "job_id": "j", "url": "u",
            "source_upload": {"url": upload["url"],
                              "content_type": upload["content_type"]}})
        assert job_of(rendered)["source_upload"]["content_type"] == "video/mp4"

    def test_the_default_job_block_asks_for_nothing(self):
        """A worker run by hand must not try to PUT anywhere."""
        assert job_of(WORKER)["source_upload"] is None

    def test_no_upload_means_audio_only(self):
        """Fetching the video costs download time and upload bandwidth; a
        deployment that does not want the rescue should pay neither."""
        assert ('fmt = VIDEO_FORMAT if want_video else "bestaudio/best"'
                in func_of(WORKER, "download_media"))

    def test_the_rescue_copy_is_1080p_like_a_local_download(self):
        """The reframe inherits the source height, so a 720p rescue copy
        would silently ship narrower clips than a normal run."""
        assert "height<=1080" in WORKER
        assert "height<=720" not in WORKER

    def test_the_upload_happens_before_transcription(self):
        """A transcription that OOMs or runs out of kernel time would
        otherwise take the rescue copy down with it — the one thing that
        kernel could still have delivered."""
        body = func_of(WORKER, "main")
        assert (body.index("upload_source(")
                < body.index("transcript = transcribe("))

    def test_a_failed_run_still_reports_its_upload(self):
        body = func_of(WORKER, "main")
        error_block = body[body.index("except Exception as exc:"):
                           body.index("finally:")]
        assert '"source": source' in error_block


class TestTheDispatcherSurfacesTheCopy:
    def _api(self, monkeypatch, result, extra=None):
        output = {"result.json": result}
        output.update(extra or {})

        class _Api:
            def kernels_push(self, folder):
                pass

            def kernels_status(self, slug):
                return types.SimpleNamespace(
                    status=types.SimpleNamespace(name="COMPLETE"),
                    failure_message="")

            def kernels_output(self, slug, path, **kwargs):
                for name, payload in output.items():
                    with io.open(os.path.join(path, name), "w",
                                 encoding="utf-8") as handle:
                        json.dump(payload, handle)
                return list(output), None

        api = _Api()
        monkeypatch.setattr(kaggle_worker, "_api", lambda: api)
        monkeypatch.setenv("KAGGLE_POLL_SECONDS", "0")
        monkeypatch.setenv("KAGGLE_USERNAME", "adnanbw")
        monkeypatch.setenv("KAGGLE_KEY", "k" * 37)
        return api

    def test_a_kernel_that_died_transcribing_still_hands_over_its_video(
            self, monkeypatch):
        """This is the case the sink exists for: transcribe_url returns None
        and the rescue copy is exactly what the caller needs."""
        _b2(monkeypatch)
        _s3(monkeypatch)
        self._api(monkeypatch,
                  {"ok": False, "error": "CUDA out of memory",
                   "source": {"uploaded": True, "bytes": 99, "title": "T"}})
        sink = {}
        assert kaggle_worker.transcribe_url(
            "https://youtu.be/x", "j", source_sink=sink) is None
        assert sink["uploaded"] is True
        assert sink["bytes"] == 99
        assert sink["key"].startswith("kaggle-source/")

    def test_the_info_json_rides_back_for_attribution(self, monkeypatch):
        _b2(monkeypatch)
        _s3(monkeypatch)
        self._api(monkeypatch,
                  {"ok": False, "source": {"uploaded": True, "bytes": 1}},
                  {"source_info.json": {"title": "A Talk",
                                        "uploader": "Sharon Verma"}})
        sink = {}
        kaggle_worker.transcribe_url("https://youtu.be/x", "j",
                                     source_sink=sink)
        assert sink["info"]["uploader"] == "Sharon Verma"
        assert sink["title"] == "A Talk"

    def test_a_kernel_that_uploaded_nothing_sets_no_flag(self, monkeypatch):
        _b2(monkeypatch)
        _s3(monkeypatch)
        self._api(monkeypatch, {"ok": False, "error": "blocked"})
        sink = {}
        kaggle_worker.transcribe_url("https://youtu.be/x", "j",
                                     source_sink=sink)
        assert not sink.get("uploaded")
        # The key is still there, so the object can be swept if the kernel
        # managed an upload it never got to report.
        assert sink["key"]

    def test_no_sink_means_no_upload_is_even_offered(self, monkeypatch):
        """transcribe_url is also called directly (tools/kaggle_setup.py);
        it must not start minting URLs for callers that cannot clean up."""
        _b2(monkeypatch)
        _s3(monkeypatch)
        api = self._api(monkeypatch, {"ok": False})
        pushed = {}
        original = api.kernels_push

        def _capture(folder):
            with io.open(os.path.join(folder, "worker.py"),
                         encoding="utf-8") as handle:
                pushed["worker"] = handle.read()
            return original(folder)
        api.kernels_push = _capture
        kaggle_worker.transcribe_url("https://youtu.be/x", "j")
        assert "source_upload" not in job_of(pushed["worker"])
