"""Kaggle is an ACCELERATOR. It must never be able to fail a job.

Kaggle runs batch kernels: there is no endpoint, its GPU quota is weekly, its
queue is shared, and its egress IP is sometimes blocked by YouTube. So every
failure path in kaggle_worker returns None and main.transcribe_video then
transcribes locally exactly as it did before the module existed.

The other half is the contract: a transcript that came back from a kernel has
to be indistinguishable from a local one by the time the pipeline sees it, and
it has to clear the same quality gate — "it came from the GPU box" is not a
quality argument.
"""
import json
import types

import pytest

import kaggle_worker


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("KAGGLE_ASR", "KAGGLE_USERNAME", "KAGGLE_USER", "KAGGLE_KEY",
                 "KAGGLE_API_TOKEN", "KAGGLE_KERNEL", "KAGGLE_CACHE_KERNEL",
                 "KAGGLE_GPU", "KAGGLE_WHISPER_MODEL", "KAGGLE_ASR_LANGUAGE",
                 "KAGGLE_TIMEOUT_SECONDS", "KAGGLE_POLL_SECONDS"):
        monkeypatch.delenv(name, raising=False)


def _configured(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "adnanbw")
    monkeypatch.setenv("KAGGLE_KEY", "k" * 37)


class _Status:
    def __init__(self, name, failure_message=""):
        self.status = types.SimpleNamespace(name=name)
        self.failure_message = failure_message


class _FakeApi:
    """Stands in for KaggleApi: records pushes, replays a status sequence."""

    def __init__(self, states=("COMPLETE",), output=None, push_error=None):
        self.states = list(states)
        self.output = output
        self.push_error = push_error
        self.pushed = []
        self.status_calls = 0

    def kernels_push(self, folder):
        if self.push_error:
            raise self.push_error
        from pathlib import Path
        folder = Path(folder)
        self.pushed.append({
            "worker": (folder / "worker.py").read_text(encoding="utf-8"),
            "metadata": json.loads(
                (folder / "kernel-metadata.json").read_text(encoding="utf-8")),
        })

    def kernels_status(self, slug):
        self.status_calls += 1
        name = self.states[min(self.status_calls - 1, len(self.states) - 1)]
        return _Status(name, "boom" if name == "ERROR" else "")

    def kernels_output(self, slug, path, **kwargs):
        """Real signature returns (files, next_page_token)."""
        from pathlib import Path
        if self.output is None:
            return [], None
        for name, payload in self.output.items():
            target = Path(path) / name
            target.write_text(json.dumps(payload, ensure_ascii=False),
                              encoding="utf-8")
        return list(self.output), None

    def kernels_logs(self, slug):
        return "log tail"


def _transcript(segments=3, language="hi"):
    out = []
    for i in range(segments):
        start = i * 5.0
        out.append({
            "start": start, "end": start + 5.0,
            "text": f"sentence number {i + 1}.",
            "words": [{"word": " sentence", "start": start, "end": start + 1},
                      {"word": " number", "start": start + 1, "end": start + 2}],
            "avg_logprob": -0.3, "no_speech_prob": 0.01,
            "compression_ratio": 1.5, "temperature": 0.0,
        })
    return {
        "text": " ".join(s["text"] for s in out),
        "language": language,
        "language_probability": 0.98,
        "segments": out,
        "asr": {"backend": "whisper", "model": "large-v3-turbo",
                "device": "cuda", "compute_type": "float16",
                "task": "transcribe", "decode_seconds": 31.4},
    }


def _install(monkeypatch, api):
    monkeypatch.setattr(kaggle_worker, "_api", lambda: api)
    monkeypatch.setenv("KAGGLE_POLL_SECONDS", "0")
    return api


# --- configuration ----------------------------------------------------------

class TestConfiguration:
    def test_no_credentials_means_disabled(self):
        assert kaggle_worker.enabled() is False

    def test_a_key_is_enough_to_be_enabled(self, monkeypatch):
        _configured(monkeypatch)
        assert kaggle_worker.enabled() is True

    def test_the_website_token_spelling_is_accepted(self, monkeypatch):
        """The key pasted from kaggle.com arrives as a bare token, not as the
        KAGGLE_KEY the CLI documents. Accepting only one spelling meant a
        configured key silently did nothing."""
        monkeypatch.setenv("KAGGLE_USERNAME", "adnanbw")
        monkeypatch.setenv("KAGGLE_API_TOKEN", "t" * 37)
        assert kaggle_worker.enabled() is True

    def test_it_can_be_switched_off_with_a_key_present(self, monkeypatch):
        _configured(monkeypatch)
        monkeypatch.setenv("KAGGLE_ASR", "0")
        assert kaggle_worker.enabled() is False

    def test_the_username_falls_back_to_the_kernel_owner(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_KEY", "k" * 37)
        monkeypatch.setenv("KAGGLE_KERNEL", "someone/openshorts-asr-worker")
        assert kaggle_worker._credentials()[0] == "someone"

    def test_the_default_slugs_follow_the_account(self, monkeypatch):
        _configured(monkeypatch)
        assert kaggle_worker.kernel_slug() == "adnanbw/openshorts-asr-worker"
        assert kaggle_worker._cache_kernel() == "adnanbw/openshorts-asr-cache"


# --- what gets pushed -------------------------------------------------------

class TestWhatGetsPushed:
    def test_the_job_is_substituted_into_the_worker(self):
        source = kaggle_worker.WORKER_SOURCE.read_text(encoding="utf-8")
        rendered = kaggle_worker.render_worker(
            source, {"job_id": "abc", "url": "https://youtu.be/X"})
        assert "'job_id': 'abc'" in rendered
        assert "https://youtu.be/X" in rendered
        # The placeholder URL from the checked-in default must be gone, or a
        # job would transcribe the wrong video and look like it worked.
        assert rendered.count("Qd4jGgu06dw") == 0

    def test_the_rendered_worker_is_still_valid_python(self):
        import ast
        source = kaggle_worker.WORKER_SOURCE.read_text(encoding="utf-8")
        rendered = kaggle_worker.render_worker(
            source, {"job_id": "abc", "url": "https://youtu.be/X",
                     "model": "large-v3-turbo", "language": None})
        ast.parse(rendered)

    def test_the_job_round_trips_as_a_python_literal(self):
        """ast.parse is NOT enough, and this test exists because the weaker
        version of it passed while the first real dispatch died 26 seconds into
        a Kaggle kernel.

        The destination is a Python source file, and json.dumps(None) is
        `null`. `{"language": null}` is syntactically VALID Python — it parses
        cleanly and raises NameError at import. Only evaluating the literal
        catches that, and the same hole covers True/False vs true/false.
        """
        import ast
        source = kaggle_worker.WORKER_SOURCE.read_text(encoding="utf-8")
        job = {"job_id": "abc", "url": "https://youtu.be/X",
               "model": "large-v3-turbo", "language": None,
               "language_detection_segments": 4, "beam_size": 5,
               "some_flag": True, "other_flag": False}
        rendered = kaggle_worker.render_worker(source, job)

        assigned = None
        for node in ast.parse(rendered).body:
            if (isinstance(node, ast.Assign)
                    and getattr(node.targets[0], "id", "") == "JOB"):
                assigned = ast.literal_eval(node.value)
        assert assigned == job, "the worker would not see the job we sent"

    def test_none_never_becomes_the_json_spelling(self):
        source = kaggle_worker.WORKER_SOURCE.read_text(encoding="utf-8")
        rendered = kaggle_worker.render_worker(
            source, {"url": "u", "language": None, "flag": True})
        block = rendered.split("JOB = ", 1)[1].split("# ===", 1)[0]
        assert "null" not in block
        assert "true" not in block

    def test_a_worker_without_a_job_block_is_an_error(self):
        with pytest.raises(ValueError):
            kaggle_worker.render_worker("print('hi')\n", {"url": "u"})

    def test_the_url_is_json_escaped_not_interpolated(self):
        """A URL is attacker-adjacent input: it reaches this as a string from
        the dashboard and is written into a Python source file."""
        source = kaggle_worker.WORKER_SOURCE.read_text(encoding="utf-8")
        nasty = 'https://x/?a="\n import os; os.system("rm -rf /")  #'
        rendered = kaggle_worker.render_worker(source, {"url": nasty})
        import ast
        ast.parse(rendered)
        assert "os.system" not in rendered.split("# ====")[1]

    def test_gpu_is_on_by_default(self, monkeypatch):
        _configured(monkeypatch)
        api = _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        kaggle_worker.transcribe_url("https://youtu.be/X")
        assert api.pushed[0]["metadata"]["enable_gpu"] is True

    def test_gpu_can_be_turned_off(self, monkeypatch):
        _configured(monkeypatch)
        monkeypatch.setenv("KAGGLE_GPU", "0")
        api = _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        kaggle_worker.transcribe_url("https://youtu.be/X")
        assert api.pushed[0]["metadata"]["enable_gpu"] is False

    def test_internet_is_always_on(self, monkeypatch):
        """Without it the worker cannot reach YouTube at all."""
        _configured(monkeypatch)
        api = _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        kaggle_worker.transcribe_url("https://youtu.be/X")
        assert api.pushed[0]["metadata"]["enable_internet"] is True

    def test_the_cache_kernel_is_mounted(self, monkeypatch):
        _configured(monkeypatch)
        api = _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        kaggle_worker.transcribe_url("https://youtu.be/X")
        assert api.pushed[0]["metadata"]["kernel_sources"] == [
            "adnanbw/openshorts-asr-cache"]


# --- every failure falls through --------------------------------------------

class TestFailureIsNeverFatal:
    def test_no_credentials(self):
        assert kaggle_worker.transcribe_url("https://youtu.be/X") is None

    def test_unauthenticated_client(self, monkeypatch):
        _configured(monkeypatch)
        monkeypatch.setattr(kaggle_worker, "_api", lambda: None)
        assert kaggle_worker.transcribe_url("https://youtu.be/X") is None

    def test_push_rejected(self, monkeypatch):
        _configured(monkeypatch)
        _install(monkeypatch, _FakeApi(push_error=RuntimeError("quota")))
        assert kaggle_worker.transcribe_url("https://youtu.be/X") is None

    def test_kernel_error(self, monkeypatch):
        _configured(monkeypatch)
        _install(monkeypatch, _FakeApi(states=("QUEUED", "RUNNING", "ERROR")))
        assert kaggle_worker.transcribe_url("https://youtu.be/X") is None

    def test_timeout(self, monkeypatch):
        _configured(monkeypatch)
        monkeypatch.setenv("KAGGLE_TIMEOUT_SECONDS", "0")
        _install(monkeypatch, _FakeApi(states=("RUNNING",)))
        assert kaggle_worker.transcribe_url("https://youtu.be/X") is None

    def test_missing_transcript_in_the_output(self, monkeypatch):
        _configured(monkeypatch)
        _install(monkeypatch, _FakeApi(output={}))
        assert kaggle_worker.transcribe_url("https://youtu.be/X") is None

    def test_the_worker_reported_its_own_failure(self, monkeypatch):
        """result.json is written even when the worker crashes, so the backend
        learns WHY instead of waiting out the whole timeout."""
        _configured(monkeypatch)
        _install(monkeypatch, _FakeApi(output={
            "result.json": {"ok": False, "error": "RuntimeError: LOGIN_REQUIRED"},
            "transcript.json": _transcript(),
        }))
        assert kaggle_worker.transcribe_url("https://youtu.be/X") is None

    def test_an_empty_transcript(self, monkeypatch):
        _configured(monkeypatch)
        empty = _transcript()
        empty["segments"] = []
        _install(monkeypatch, _FakeApi(output={"transcript.json": empty}))
        assert kaggle_worker.transcribe_url("https://youtu.be/X") is None

    def test_a_client_crash_after_the_transcript_landed_is_survived(
            self, monkeypatch):
        """The kaggle client writes the kernel LOG with open(path, "w") and no
        encoding, so on a Windows host (cp1252) a Devanagari transcript raises
        UnicodeEncodeError. The DATA files are written first and in binary, so
        the transcript is already on disk when it throws — and treating the
        exception as failure threw away a finished GPU transcription.
        """
        _configured(monkeypatch)

        class CrashesOnLog(_FakeApi):
            def kernels_output(self, slug, path, **kwargs):
                from pathlib import Path
                (Path(path) / "transcript.json").write_text(
                    json.dumps(_transcript()), encoding="utf-8")
                raise UnicodeEncodeError(
                    "charmap", "ह", 0, 1, "character maps to <undefined>")

        _install(monkeypatch, CrashesOnLog())
        result = kaggle_worker.transcribe_url("https://youtu.be/X")
        assert result is not None
        assert len(result["segments"]) == 3

    def test_a_client_crash_with_no_transcript_is_still_a_failure(
            self, monkeypatch):
        _configured(monkeypatch)

        class CrashesEarly(_FakeApi):
            def kernels_output(self, slug, path, **kwargs):
                raise RuntimeError("connection reset")

        _install(monkeypatch, CrashesEarly())
        assert kaggle_worker.transcribe_url("https://youtu.be/X") is None

    def test_the_clients_own_pagination_is_not_disabled(self, monkeypatch):
        """It follows output pages itself, but only while page_token is None.
        Passing one explicitly turns that off and truncates the download."""
        _configured(monkeypatch)
        seen = {}

        class Recording(_FakeApi):
            def kernels_output(self, slug, path, **kwargs):
                seen.update(kwargs)
                from pathlib import Path
                (Path(path) / "transcript.json").write_text(
                    json.dumps(_transcript()), encoding="utf-8")
                return ["transcript.json"], None

        _install(monkeypatch, Recording())
        kaggle_worker.transcribe_url("https://youtu.be/X")
        assert seen.get("page_token") is None

    def test_a_queued_kernel_is_abandoned_on_the_SHORT_budget(self, monkeypatch):
        """QUEUED means no GPU was allocated and nothing is happening.

        Measured: a real job polled a queued kernel for the full 1800s and
        then ran the local transcription that could have started at once.
        Kaggle publishes no queue position, so eight minutes of QUEUED is
        indistinguishable from never.
        """
        _configured(monkeypatch)
        monkeypatch.setenv("KAGGLE_QUEUE_TIMEOUT_SECONDS", "0")
        api = _install(monkeypatch, _FakeApi(
            states=("QUEUED",), output={"transcript.json": _transcript()}))
        assert kaggle_worker.transcribe_url("https://youtu.be/X") is None
        # Abandoned on the first look, not polled out to the long timeout.
        assert api.status_calls == 1

    def test_a_running_kernel_keeps_the_LONG_budget(self, monkeypatch):
        """Once the GPU is decoding, giving up throws away nearly-done work."""
        _configured(monkeypatch)
        monkeypatch.setenv("KAGGLE_QUEUE_TIMEOUT_SECONDS", "0")
        _install(monkeypatch, _FakeApi(
            states=("RUNNING", "RUNNING", "COMPLETE"),
            output={"transcript.json": _transcript()}))
        assert kaggle_worker.transcribe_url("https://youtu.be/X") is not None

    def test_queuing_then_running_keeps_the_long_budget(self, monkeypatch):
        """The normal path: a short queue, then work. The short budget must
        stop applying the moment it starts, not stay armed for the run."""
        _configured(monkeypatch)
        monkeypatch.setenv("KAGGLE_QUEUE_TIMEOUT_SECONDS", "0.05")
        monkeypatch.setenv("KAGGLE_POLL_SECONDS", "0.02")
        _install(monkeypatch, _FakeApi(
            states=("QUEUED", "RUNNING", "RUNNING", "RUNNING", "COMPLETE"),
            output={"transcript.json": _transcript()}))
        assert kaggle_worker.transcribe_url("https://youtu.be/X") is not None

    def test_the_queue_budget_never_outlives_the_overall_one(self, monkeypatch):
        """A deployment that lowers only KAGGLE_TIMEOUT_SECONDS must not get
        a LONGER wait from the queue default it never touched."""
        _configured(monkeypatch)
        monkeypatch.setenv("KAGGLE_TIMEOUT_SECONDS", "0")
        _install(monkeypatch, _FakeApi(states=("QUEUED",)))
        assert kaggle_worker.transcribe_url("https://youtu.be/X") is None

    def test_a_status_call_that_throws_does_not_end_the_wait(self, monkeypatch):
        _configured(monkeypatch)

        class Flaky(_FakeApi):
            def kernels_status(self, slug):
                self.status_calls += 1
                if self.status_calls == 1:
                    raise RuntimeError("transient 503")
                return _Status("COMPLETE")

        _install(monkeypatch, Flaky(output={"transcript.json": _transcript()}))
        assert kaggle_worker.transcribe_url("https://youtu.be/X") is not None


# --- the contract -----------------------------------------------------------

class TestTheTranscriptContract:
    def test_a_successful_run_returns_the_transcript(self, monkeypatch):
        _configured(monkeypatch)
        _install(monkeypatch, _FakeApi(
            states=("QUEUED", "RUNNING", "COMPLETE"),
            output={"transcript.json": _transcript()}))
        result = kaggle_worker.transcribe_url("https://youtu.be/X")
        assert result["language"] == "hi"
        assert len(result["segments"]) == 3

    def test_it_has_every_key_a_local_transcript_has(self, monkeypatch):
        _configured(monkeypatch)
        _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        result = kaggle_worker.transcribe_url("https://youtu.be/X")
        assert set(result) >= {"text", "language", "language_probability",
                               "segments", "asr"}
        segment = result["segments"][0]
        assert set(segment) >= {"start", "end", "text", "words", "avg_logprob",
                                "no_speech_prob", "compression_ratio"}

    def test_continuation_words_are_merged_on_this_side(self, monkeypatch):
        """The worker returns RAW whisper words so there is exactly one
        implementation of the merge, and a remote transcript cannot drift from
        a local one in how words are joined."""
        _configured(monkeypatch)
        payload = _transcript()
        payload["segments"][0]["words"] = [
            {"word": " Rhein", "start": 0.0, "end": 0.5},
            {"word": "-Kanal.", "start": 0.5, "end": 1.0},
        ]
        _install(monkeypatch, _FakeApi(output={"transcript.json": payload}))
        result = kaggle_worker.transcribe_url("https://youtu.be/X")
        words = [w["word"] for w in result["segments"][0]["words"]]
        assert words == [" Rhein-Kanal."]


# --- the same quality gate --------------------------------------------------

class TestARemoteTranscriptFacesTheSameGate:
    def test_a_good_remote_transcript_is_accepted(self):
        from transcribe_backends import judge_remote_transcript
        judged = judge_remote_transcript(_transcript(segments=24), duration=120.0)
        assert judged is not None
        assert judged["asr"]["quality"]["status"] in ("GOOD", "PARTIAL", "RETRY")

    def test_an_unusable_remote_transcript_falls_back(self):
        from transcribe_backends import judge_remote_transcript
        assert judge_remote_transcript({"segments": []}, duration=120.0) is None

    def test_it_never_raises_where_the_local_path_would(self):
        """Locally a BAD transcript raises TranscriptQualityError. Here it must
        return None instead: the caller still has a working local path, and a
        remote BAD is a reason to fall back, not to fail the job."""
        from transcribe_backends import judge_remote_transcript
        garbage = _transcript(segments=24)
        for segment in garbage["segments"]:
            segment["text"] = "मुझे ですが the пример 니다"
            segment["words"] = []
            segment["avg_logprob"] = -1.9
            segment["compression_ratio"] = 4.2
            segment["no_speech_prob"] = 0.85
        assert judge_remote_transcript(garbage, duration=120.0) is None


class TestTheLanguageIsPinned:
    """The single most important thing about the remote path.

    Measured on a real Kaggle run: left to detect for itself, large-v3-turbo
    called the Hindi stand-up English at 90% and returned a fluent English
    TRANSLATION — "Yesterday, I went to a lift and I went to a couple" for
    "कल में एक लिफ्ट में गुसी..." — although the task is always "transcribe".
    Nothing downstream catches it: it scores ~99/100 because it is good
    English, and the clips, captions and metadata all change language silently.
    """

    def test_a_pinned_language_reaches_the_worker(self, monkeypatch):
        _configured(monkeypatch)
        api = _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        kaggle_worker.transcribe_url("https://youtu.be/X", language="hi")
        assert "'language': 'hi'" in api.pushed[0]["worker"]

    def test_no_pin_means_the_model_detects(self, monkeypatch):
        _configured(monkeypatch)
        api = _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        kaggle_worker.transcribe_url("https://youtu.be/X")
        assert "'language': None" in api.pushed[0]["worker"]

    def test_an_explicit_pin_beats_the_env_default(self, monkeypatch):
        _configured(monkeypatch)
        monkeypatch.setenv("KAGGLE_ASR_LANGUAGE", "en")
        api = _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        kaggle_worker.transcribe_url("https://youtu.be/X", language="hi")
        assert "'language': 'hi'" in api.pushed[0]["worker"]


class TestTheLanguageIsDetectedOnTheWorker:
    """Detection moved from the backend onto the GPU, and that is what lets the
    dispatch happen at t=0 instead of waiting for the local download."""

    def test_the_worker_is_told_to_detect(self, monkeypatch):
        _configured(monkeypatch)
        api = _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        kaggle_worker.transcribe_url("https://youtu.be/X")
        assert "'detect_language': True" in api.pushed[0]["worker"]

    def test_the_pin_threshold_is_passed_through(self, monkeypatch):
        _configured(monkeypatch)
        monkeypatch.setenv("ASR_PIN_LANGUAGE_ABOVE", "0.7")
        api = _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        kaggle_worker.transcribe_url("https://youtu.be/X")
        assert "'pin_language_above': 0.7" in api.pushed[0]["worker"]

    def test_detection_can_be_switched_off(self, monkeypatch):
        _configured(monkeypatch)
        monkeypatch.setenv("KAGGLE_DETECT_LANGUAGE", "0")
        api = _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        kaggle_worker.transcribe_url("https://youtu.be/X")
        assert "'detect_language': False" in api.pushed[0]["worker"]

    def test_auto_language_is_the_default(self, monkeypatch):
        """large-v3's language ID is much better than turbo's, which is why
        auto is the default now. The probe still runs as a cross-check."""
        _configured(monkeypatch)
        api = _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        kaggle_worker.transcribe_url("https://youtu.be/X")
        pushed = api.pushed[0]["worker"]
        assert "'pin_language': False" in pushed
        assert "'detect_language': True" in pushed

    def test_the_probe_can_be_made_binding_again(self, monkeypatch):
        _configured(monkeypatch)
        monkeypatch.setenv("KAGGLE_PIN_LANGUAGE", "1")
        api = _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        kaggle_worker.transcribe_url("https://youtu.be/X")
        assert "'pin_language': True" in api.pushed[0]["worker"]

    def test_large_v3_is_the_default_model(self, monkeypatch):
        """Excluded locally for 1.8x CPU decode and an OOM on load; the GPU
        erases both, and it measurably beats turbo on code-switched audio."""
        _configured(monkeypatch)
        api = _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        kaggle_worker.transcribe_url("https://youtu.be/X")
        assert "'model': 'large-v3'" in api.pushed[0]["worker"]

    def test_the_model_is_overridable(self, monkeypatch):
        _configured(monkeypatch)
        monkeypatch.setenv("KAGGLE_WHISPER_MODEL", "large-v3-turbo")
        api = _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        kaggle_worker.transcribe_url("https://youtu.be/X")
        assert "'model': 'large-v3-turbo'" in api.pushed[0]["worker"]

    def test_an_explicit_pin_still_wins(self, monkeypatch):
        _configured(monkeypatch)
        api = _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        kaggle_worker.transcribe_url("https://youtu.be/X", language="hi")
        assert "'language': 'hi'" in api.pushed[0]["worker"]


class TestDispatchDoesNotBlock:
    """start() fires the job and returns immediately so the download can run
    at the same time; collect() is the only thing that waits."""

    def test_start_returns_a_handle(self, monkeypatch):
        _configured(monkeypatch)
        _install(monkeypatch, _FakeApi(
            output={"transcript.json": _transcript()}))
        handle = kaggle_worker.start("https://youtu.be/X")
        assert handle is not None
        assert handle.collect(timeout=30) is not None

    def test_start_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_ASR", "0")
        assert kaggle_worker.start("https://youtu.be/X") is None

    def test_collect_never_raises(self, monkeypatch):
        _configured(monkeypatch)

        class Boom(_FakeApi):
            def kernels_push(self, folder):
                raise RuntimeError("quota exceeded")

        _install(monkeypatch, Boom())
        handle = kaggle_worker.start("https://youtu.be/X")
        assert handle.collect(timeout=30) is None
