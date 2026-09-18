"""Adaptive ASR: transcribe once normally, twice only when quality demands it.

The performance rule this file exists to protect: a video that transcribes
cleanly — English or any other language — must be transcribed EXACTLY ONCE.
The retry is an exception, not the workflow.
"""
import pytest

import transcribe_backends as tb
from asr_quality import TranscriptQualityError


def _clean(language="en", model="small", n=24):
    """A transcript the quality gate accepts."""
    vocabulary = ("the thing nobody tells you about starting a business is "
                  "that the first year is mostly learning what does not "
                  "work").split()
    segments = []
    for i in range(n):
        text = " ".join(vocabulary[(i + j) % len(vocabulary)] for j in range(14)) + "."
        start = i * 5.0
        step = 5.0 / 14
        segments.append({
            "start": start, "end": start + 5.0, "text": text,
            "words": [{"word": " " + w, "start": start + k * step,
                       "end": start + (k + 1) * step}
                      for k, w in enumerate(text.split())],
            "avg_logprob": -0.3, "no_speech_prob": 0.02,
            "compression_ratio": 1.6,
        })
    return {"text": " ".join(s["text"] for s in segments),
            "language": language, "language_probability": 0.98,
            "segments": segments,
            "asr": {"backend": "whisper", "model": model, "device": "cpu"}}


def _garbled(model="small"):
    """The observed failure: low confidence, looping, unrelated scripts."""
    transcript = _clean(language="hi", model=model, n=24)
    for i, segment in enumerate(transcript["segments"]):
        segment["text"] = "मुझे ですが the пример 니다"
        segment["words"] = []
        segment["avg_logprob"] = -1.4
        segment["compression_ratio"] = 3.6
        segment["no_speech_prob"] = 0.7
    transcript["text"] = " ".join(s["text"] for s in transcript["segments"])
    transcript["language_probability"] = 0.31
    return transcript


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("ASR_QUALITY_GATE", "ASR_AUTO_RETRY", "ASR_QUALITY_GEMINI",
                 "ASR_QUALITY_LOGPROB_SUSPECT", "WHISPER_RETRY_MODEL",
                 "WHISPER_MODEL", "GEMINI_API_KEY", "ASR_PIN_LANGUAGE_ABOVE"):
        monkeypatch.delenv(name, raising=False)
    # Most tests here describe the two-model ladder. The last-resort rung is
    # switched off so they stay about what they are about; TestThirdPass turns
    # it back on deliberately.
    monkeypatch.setenv("WHISPER_FINAL_MODEL", "")


class _Recorder:
    """Stands in for the two transcription entry points.

    ``probe`` is what _probe_best_model would have returned — the
    (model, language, probability) triple it decides from ~75 seconds of audio.
    The default is "the probe found no reason to change anything", so a test
    that is not about the probe reads exactly as it did before it existed.
    """

    def __init__(self, first, retry=None, probe=(None, None, None)):
        self.first = first
        self.retry = retry
        self.probe = probe
        self.calls = []

    def probed(self, media_path, duration):
        self.calls.append(("probe", self.probe[0], None))
        return self.probe

    def initial(self, path, model_size=None, language=None):
        self.calls.append(("initial", model_size, language))
        return self.first

    def stronger(self, path, model_size=None, language=None):
        self.calls.append(("retry", model_size, language))
        if isinstance(self.retry, Exception):
            raise self.retry
        return self.retry

    @property
    def passes(self):
        """The calls that actually transcribed the whole video."""
        return [c for c in self.calls if c[0] != "probe"]


def _install(monkeypatch, recorder):
    monkeypatch.setattr(tb, "_probe_best_model", recorder.probed)
    monkeypatch.setattr(tb, "transcribe_media", recorder.initial)
    monkeypatch.setattr(tb, "_transcribe_with_whisper", recorder.stronger)
    return recorder


# --- 1, 2: the fast path ----------------------------------------------------

class TestCleanAudioIsTranscribedOnce:
    def test_good_english_does_not_retry(self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(_clean()))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.passes] == ["initial"]
        assert result["asr"]["quality"]["status"] == "GOOD"

    @pytest.mark.parametrize("language", ["en", "es", "hi", "ja", "ar", "pt"])
    def test_no_language_triggers_a_retry_on_its_own(self, monkeypatch, language):
        recorder = _install(monkeypatch, _Recorder(_clean(language=language)))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.passes] == ["initial"]


# --- 7, 8: the retry --------------------------------------------------------

class TestRetry:
    def test_bad_initial_then_good_retry_uses_the_retry(self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(
            _garbled(), _clean(language="hi", model="large-v3-turbo")))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)

        assert [c[0] for c in recorder.passes] == ["initial", "retry"]
        assert recorder.passes[1][1] == "large-v3-turbo"
        # Attempt 1 was only 31% sure of the language, so the stronger model
        # gets to detect it again (see TestLanguageIsPreserved).
        assert recorder.passes[1][2] is None
        assert result["asr"]["quality"]["status"] == "GOOD"
        assert result["asr"]["model"] == "large-v3-turbo"
        assert [a["attempt"] for a in result["asr"]["attempts"]] == [1, 2]

    def test_bad_initial_and_bad_retry_raises_a_transcription_error(self, monkeypatch):
        _install(monkeypatch, _Recorder(_garbled(),
                                        _garbled(model="large-v3-turbo")))
        with pytest.raises(TranscriptQualityError) as excinfo:
            tb.transcribe_media_checked("video.mp4", duration=120.0)

        message = str(excinfo.value)
        assert "Transcription quality was too low" in message
        # Must not read as a clip-selection failure — that was the whole bug.
        assert "clip" in message.lower() and "detection" not in message.lower()

    def test_empty_transcript_fails_with_the_same_clear_error(self, monkeypatch):
        empty = {"text": "", "language": "en", "segments": [],
                 "asr": {"backend": "whisper", "model": "small"}}
        _install(monkeypatch, _Recorder(empty, empty))
        with pytest.raises(TranscriptQualityError):
            tb.transcribe_media_checked("video.mp4", duration=120.0)

    def test_a_worse_retry_does_not_replace_a_better_first_attempt(self, monkeypatch):
        suspicious = _clean(language="hi")
        for segment in suspicious["segments"]:
            segment["avg_logprob"] = -1.06         # RETRY, but usable text
        recorder = _install(monkeypatch, _Recorder(
            suspicious, _garbled(model="large-v3-turbo")))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)

        assert [c[0] for c in recorder.passes] == ["initial", "retry"]
        assert result["asr"]["model"] == "small"

    def test_a_failing_retry_keeps_the_first_transcript(self, monkeypatch):
        suspicious = _clean(language="hi")
        for segment in suspicious["segments"]:
            segment["avg_logprob"] = -1.06
        _install(monkeypatch, _Recorder(suspicious, RuntimeError("out of memory")))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert result["asr"]["model"] == "small"

    def test_suspicious_but_readable_transcript_is_not_fatal(self, monkeypatch):
        """RETRY means "worth a stronger model", not "unusable". With nothing
        stronger left the job continues rather than failing on a transcript a
        human would call fine."""
        suspicious = _clean(language="hi")
        for segment in suspicious["segments"]:
            segment["avg_logprob"] = -1.06
        _install(monkeypatch, _Recorder(suspicious, suspicious))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert result["asr"]["quality"]["status"] == "RETRY"


# --- switches ---------------------------------------------------------------

class TestSwitches:
    def test_auto_retry_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("ASR_AUTO_RETRY", "0")
        recorder = _install(monkeypatch, _Recorder(_garbled(), _clean()))
        with pytest.raises(TranscriptQualityError):
            tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.passes] == ["initial"]

    def test_no_retry_when_every_rung_has_already_been_tried(self, monkeypatch):
        monkeypatch.setenv("WHISPER_RETRY_MODEL", "large-v3-turbo")
        recorder = _install(monkeypatch, _Recorder(
            _garbled(model="large-v3-turbo"), _clean()))
        with pytest.raises(TranscriptQualityError):
            tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.passes] == ["initial"]

    def test_gate_off_means_never_retry_and_never_fail(self, monkeypatch):
        monkeypatch.setenv("ASR_QUALITY_GATE", "0")
        recorder = _install(monkeypatch, _Recorder(_garbled(), _clean()))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.passes] == ["initial"]
        assert result["asr"]["quality"]["skipped"] is True

    def test_raise_on_bad_false_returns_the_transcript(self, monkeypatch):
        _install(monkeypatch, _Recorder(_garbled(), _garbled(model="large-v3-turbo")))
        result = tb.transcribe_media_checked(
            "video.mp4", duration=120.0, raise_on_bad=False)
        assert result["asr"]["quality"]["status"] == "BAD"


# --- debug artefacts --------------------------------------------------------

def test_debug_files_are_written(monkeypatch, tmp_path):
    _install(monkeypatch, _Recorder(_garbled(),
                                    _clean(language="hi", model="large-v3-turbo")))
    tb.transcribe_media_checked("video.mp4", duration=120.0,
                                debug_dir=str(tmp_path))
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"asr_initial.json", "asr_quality_initial.json",
                     "asr_retry.json", "asr_quality_retry.json"}


def test_a_debug_directory_that_cannot_be_written_does_not_fail_the_job(monkeypatch):
    _install(monkeypatch, _Recorder(_clean()))
    monkeypatch.setattr(tb.os, "makedirs",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    assert tb.transcribe_media_checked(
        "video.mp4", duration=120.0, debug_dir="/nope")["segments"]


# --- the grey band: structurally fine, but are the words real words? --------

def _semantic(status, good=1.0, partial=0.0, bad=0.0, regions=None):
    """A canned asr_semantic verdict."""
    return {"status": status, "score": 90 if status == "GOOD" else 40,
            "shares": {"GOOD": good, "PARTIAL": partial, "BAD": bad},
            "regions": regions if regions is not None else [
                {"id": "R01", "start": 0.0, "end": 30.0, "status": status,
                 "score": 90 if status == "GOOD" else 20, "reason": "",
                 "code_switching": False}],
            "dense": True, "sampled_regions": 1}


def _reader(monkeypatch, *verdicts):
    """Install asr_semantic.review_transcript, returning each verdict in turn."""
    calls = []

    def review(transcript, api_key=None, model_name=None, dense=False):
        calls.append({"dense": dense})
        index = min(len(calls) - 1, len(verdicts) - 1)
        return verdicts[index] if verdicts else None

    monkeypatch.setattr(tb.asr_semantic, "review_transcript", review)
    return calls


class TestLanguageCheck:
    """whisper-small transcribed a Hindi stand-up as fluent Devanagari-shaped
    nonsense: no repetition, valid timestamps, 173 words/min, score 87/100.
    Structure cannot catch that; only reading the words can."""

    def _uncertain(self):
        transcript = _clean(language="hi")
        for segment in transcript["segments"]:
            segment["avg_logprob"] = -0.75     # what the real failure measured
        return transcript

    def test_a_confident_transcript_is_never_sent_to_the_reader(self, monkeypatch):
        calls = _reader(monkeypatch, _semantic("GOOD"))
        _install(monkeypatch, _Recorder(_clean()))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert calls == []

    def test_uncertain_is_flagged_but_still_good_on_its_own(self):
        from asr_quality import evaluate_transcript
        quality = evaluate_transcript(self._uncertain(), 120.0)
        assert quality["status"] == "GOOD"
        assert quality["uncertain"] is True

    def test_invented_words_downgrade_and_force_a_retry(self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(
            self._uncertain(), _clean(language="hi", model="large-v3-turbo")))
        _reader(monkeypatch, _semantic("BAD", good=0.0, bad=1.0),
                _semantic("GOOD"))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.passes] == ["initial", "retry"]
        assert result["asr"]["model"] == "large-v3-turbo"

    def test_real_language_keeps_the_first_transcript(self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(self._uncertain(), _clean()))
        _reader(monkeypatch, _semantic("GOOD"))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.passes] == ["initial"]
        assert result["asr"]["quality"]["status"] == "GOOD"

    def test_an_unavailable_check_never_fails_a_job_on_suspicion(self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(self._uncertain(), _clean()))
        _reader(monkeypatch, None)
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.passes] == ["initial"]
        assert result["asr"]["quality"]["status"] == "GOOD"

    def test_a_real_retry_beats_a_downgraded_first_attempt(self, monkeypatch):
        """The downgraded transcript keeps a high STRUCTURAL score - structure
        was never its problem - so comparing scores alone would have kept the
        gibberish over a genuinely good retry."""
        good_retry = _clean(language="hi", model="large-v3-turbo")
        for segment in good_retry["segments"]:
            segment["avg_logprob"] = -0.45      # real speech, lower score
        _install(monkeypatch, _Recorder(self._uncertain(), good_retry))
        _reader(monkeypatch, _semantic("BAD", good=0.0, bad=1.0),
                _semantic("GOOD"))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert result["asr"]["model"] == "large-v3-turbo"

    def test_a_condemned_transcript_with_no_retry_left_is_refused(self, monkeypatch):
        _install(monkeypatch, _Recorder(self._uncertain(), RuntimeError("no model")))
        _reader(monkeypatch, _semantic("BAD", good=0.0, bad=1.0))
        with pytest.raises(TranscriptQualityError):
            tb.transcribe_media_checked("video.mp4", duration=120.0)


# --- TASK 2: the retry must prove it fixed what it was run for --------------

class TestRetryReasonPropagates:
    """The exact bug the real job hit: attempt 1 was retried BECAUSE a reader
    found gibberish, attempt 2 came back at avg_logprob -0.29 and structural
    score 100 - so ``uncertain`` was False, the reader was never called again,
    and a transcript that was still a third invented was accepted as GOOD."""

    def _uncertain(self):
        transcript = _clean(language="hi")
        for segment in transcript["segments"]:
            segment["avg_logprob"] = -0.75
        return transcript

    def _confident_retry(self):
        # avg_logprob -0.29: nothing structural will ever question this.
        retry = _clean(language="hi", model="large-v3-turbo")
        for segment in retry["segments"]:
            segment["avg_logprob"] = -0.29
        return retry

    def test_a_confident_retry_is_still_read_when_attempt_1_was_gibberish(
            self, monkeypatch):
        _install(monkeypatch, _Recorder(self._uncertain(), self._confident_retry()))
        calls = _reader(monkeypatch, _semantic("BAD", good=0.0, bad=1.0),
                        _semantic("GOOD"))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert len(calls) == 2, "the retry skipped the check it existed for"

    def test_a_still_broken_confident_retry_is_not_accepted_as_good(
            self, monkeypatch):
        _install(monkeypatch, _Recorder(self._uncertain(), self._confident_retry()))
        _reader(monkeypatch, _semantic("BAD", good=0.0, bad=1.0),
                _semantic("BAD", good=0.0, bad=1.0))
        with pytest.raises(TranscriptQualityError):
            tb.transcribe_media_checked("video.mp4", duration=120.0)

    def test_the_reason_is_recorded_on_the_attempt(self, monkeypatch):
        _install(monkeypatch, _Recorder(self._uncertain(), self._confident_retry()))
        _reader(monkeypatch, _semantic("BAD", good=0.0, bad=1.0),
                _semantic("GOOD"))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        attempts = result["asr"]["attempts"]
        assert attempts[1]["reason"] == "semantic_bad"
        assert attempts[0]["semantic_status"] == "BAD"
        assert attempts[1]["semantic_status"] == "GOOD"

    def test_a_structural_retry_does_not_force_a_semantic_check(self, monkeypatch):
        """Only a SEMANTIC failure makes the retry prove itself semantically.
        A merely low structural score is what the stronger model itself fixes."""
        weak = _clean(language="hi")
        for segment in weak["segments"]:
            segment["avg_logprob"] = -1.06     # structural RETRY, not uncertain
        _install(monkeypatch, _Recorder(weak, _clean(model="large-v3-turbo")))
        calls = _reader(monkeypatch, _semantic("GOOD"))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert calls == []


# --- TASK 3: the ASR_QUALITY_GEMINI contract --------------------------------

class TestCheckModeContract:
    def test_zero_means_never(self, monkeypatch):
        monkeypatch.setenv("ASR_QUALITY_GEMINI", "0")
        suspicious = _clean(language="hi")
        for segment in suspicious["segments"]:
            segment["avg_logprob"] = -0.75
        _install(monkeypatch, _Recorder(suspicious, _clean(model="large-v3-turbo")))
        calls = _reader(monkeypatch, _semantic("BAD", good=0.0, bad=1.0))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert calls == []

    def test_zero_means_never_even_for_an_exhausted_suspicious_transcript(
            self, monkeypatch):
        """The last-reprieve read in _settle must obey the switch too."""
        monkeypatch.setenv("ASR_QUALITY_GEMINI", "0")
        weak = _clean(language="hi")
        for segment in weak["segments"]:
            segment["avg_logprob"] = -1.06     # structural RETRY
        _install(monkeypatch, _Recorder(weak, weak))
        calls = _reader(monkeypatch, _semantic("BAD", good=0.0, bad=1.0))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert calls == []
        assert result["asr"]["quality"]["status"] == "RETRY"

    def test_one_means_always_even_for_a_clean_transcript(self, monkeypatch):
        monkeypatch.setenv("ASR_QUALITY_GEMINI", "1")
        _install(monkeypatch, _Recorder(_clean()))
        calls = _reader(monkeypatch, _semantic("GOOD"))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert len(calls) == 1, "a clean video must cost exactly one read"

    def test_auto_is_the_default_and_reads_nothing_clean(self, monkeypatch):
        monkeypatch.delenv("ASR_QUALITY_GEMINI", raising=False)
        assert tb.asr_semantic.check_mode() == "auto"
        _install(monkeypatch, _Recorder(_clean()))
        calls = _reader(monkeypatch, _semantic("GOOD"))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert calls == []

    def test_an_unknown_value_falls_back_to_auto(self, monkeypatch):
        monkeypatch.setenv("ASR_QUALITY_GEMINI", "yes-please")
        assert tb.asr_semantic.check_mode() == "auto"


# --- TASK 4/5: PARTIAL is its own outcome -----------------------------------

class TestPartialTranscripts:
    """One ten-minute video really can be half accurate. Throwing the good half
    away helps nobody, and silently treating it as GOOD is how nonsense Shorts
    get published."""

    def _mixed(self):
        transcript = _clean(language="hi", model="large-v3-turbo")
        for segment in transcript["segments"]:
            segment["avg_logprob"] = -0.75
        return transcript

    def _regions(self):
        return [
            {"id": "R01", "start": 0.0, "end": 45.0, "status": "GOOD",
             "score": 95, "reason": "", "code_switching": True},
            {"id": "R02", "start": 45.0, "end": 90.0, "status": "BAD",
             "score": 10, "reason": "invented words", "code_switching": False},
            {"id": "R03", "start": 90.0, "end": 120.0, "status": "GOOD",
             "score": 90, "reason": "", "code_switching": False},
        ]

    def _partial(self):
        return _semantic("PARTIAL", good=0.6, bad=0.4, regions=self._regions())

    def test_partial_is_not_reported_as_good(self, monkeypatch):
        _install(monkeypatch, _Recorder(self._mixed(), self._mixed()))
        _reader(monkeypatch, self._partial())
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert result["asr"]["quality"]["status"] == "PARTIAL"

    def test_partial_still_returns_a_usable_transcript(self, monkeypatch):
        _install(monkeypatch, _Recorder(self._mixed(), self._mixed()))
        _reader(monkeypatch, self._partial())
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert result["segments"], "the good half must survive"

    def test_partial_carries_the_regions_clip_selection_needs(self, monkeypatch):
        _install(monkeypatch, _Recorder(self._mixed(), self._mixed()))
        _reader(monkeypatch, self._partial())
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        regions = result["asr"]["quality"]["semantic"]["regions"]
        assert [r["status"] for r in regions] == ["GOOD", "BAD", "GOOD"]

    def test_partial_is_read_once_and_used(self, monkeypatch):
        """There used to be a second, DENSE reader pass over the whole timeline
        whenever the first came back mixed, to map exactly which stretches were
        bad. Its only consumer was a filter that deleted candidate clips, and
        that filter is gone — so the second call is pure cost."""
        sparse = self._partial()
        sparse["dense"] = False
        recorder = _install(monkeypatch, _Recorder(self._mixed(), self._mixed()))
        calls = _reader(monkeypatch, sparse, self._partial())
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert len(calls) == 1, "the transcript was read twice"
        assert [c[0] for c in recorder.passes] == ["initial"]

    def test_partial_is_never_escalated_to_another_pass(self, monkeypatch):
        """The whole reason this job failed. A mixed transcript is usable, and
        on genuinely code-switched speech a GOOD verdict is close to
        unreachable — the reader marks a 25-second window PARTIAL for one
        oddly-spelled word — so escalating on PARTIAL meant every Hinglish
        video walked the entire ladder whatever the models produced."""
        recorder = _install(monkeypatch, _Recorder(self._mixed(), self._mixed()))
        _reader(monkeypatch, self._partial())
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.passes] == ["initial"]
        assert result["asr"]["quality"]["status"] == "PARTIAL"


# --- an exhausted suspicious transcript is USED, not refused ----------------

class TestSuspiciousIsStillUsable:
    """A structural RETRY means the decoder was unsure, not that the words are
    wrong. It earns the one repair pass this module pays for; if that does not
    settle it, the transcript is used. Refusing here would fail jobs that the
    upstream tool — which has no quality gate at all — cuts clips from."""

    def _suspicious(self):
        transcript = _clean(language="hi")
        for segment in transcript["segments"]:
            segment["avg_logprob"] = -1.06     # structural RETRY
        return transcript

    def test_it_earns_exactly_one_repair_pass(self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(self._suspicious(),
                                                   self._suspicious()))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.passes] == ["initial", "retry"]

    def test_and_is_then_used_rather_than_refused(self, monkeypatch):
        _install(monkeypatch, _Recorder(self._suspicious(), self._suspicious()))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert result["asr"]["quality"]["status"] == "RETRY"
        assert result["segments"], "the words were thrown away"

    def test_it_is_not_read_again_just_to_find_a_reason_to_refuse(
            self, monkeypatch):
        _install(monkeypatch, _Recorder(self._suspicious(), self._suspicious()))
        calls = _reader(monkeypatch, _semantic("BAD", good=0.0, bad=1.0))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert calls == []
        assert result["segments"]

    def test_a_reader_that_condemns_attempt_1_still_forces_the_repair(
            self, monkeypatch):
        """The reader keeps its teeth where it has evidence: a transcript the
        decoder was confident about, that a reader says is invented, is exactly
        the failure nothing structural can see."""
        uncertain = _clean(language="hi")
        for segment in uncertain["segments"]:
            segment["avg_logprob"] = -0.75
        recorder = _install(monkeypatch, _Recorder(
            uncertain, _clean(language="hi", model="large-v3-turbo")))
        _reader(monkeypatch, _semantic("BAD", good=0.0, bad=1.0),
                _semantic("GOOD"))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.passes] == ["initial", "retry"]
        assert result["asr"]["model"] == "large-v3-turbo"


# --- TASK 7: the diagnostics must be serializable ---------------------------

class TestDiagnosticsAreSerializable:
    def test_no_circular_reference_in_the_returned_transcript(self, monkeypatch):
        """The real log said "Could not save transcript checkpoint: Circular
        reference detected" on every job: the attempt records held the very
        dict the attempts list was then hung from."""
        import json
        _install(monkeypatch, _Recorder(_clean()))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        json.dumps(result)          # no default=, no fallback: must be clean

    def test_attempt_records_are_flat_values_not_references(self, monkeypatch):
        _install(monkeypatch, _Recorder(_clean()))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        asr = result["asr"]
        for attempt in asr["attempts"]:
            for value in attempt.values():
                assert not isinstance(value, (dict, list)), attempt
            assert attempt["model"] == asr["model"]

    def test_a_retried_transcript_is_also_serializable(self, monkeypatch):
        import json
        _install(monkeypatch, _Recorder(
            _garbled(), _clean(language="hi", model="large-v3-turbo")))
        _reader(monkeypatch, _semantic("GOOD"))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        json.dumps(result)


# --- the retry must not change the LANGUAGE ---------------------------------

class TestLanguageIsPreserved:
    """Measured: whisper-small detected 'hi' at 89%, large-v3-turbo re-detected
    'en' at 96% and emitted an English TRANSLATION of the Hindi audio, which
    scored 99/100 because it is fluent English. Clip timing, captions and
    metadata would all have switched language behind the user's back."""

    def _suspicious(self, language, probability):
        transcript = _clean(language=language)
        transcript["language_probability"] = probability
        for segment in transcript["segments"]:
            segment["avg_logprob"] = -1.06
        return transcript

    def test_a_confident_language_is_pinned_on_the_retry(self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(
            self._suspicious("hi", 0.89), _clean(language="hi",
                                                 model="large-v3-turbo")))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert recorder.passes[1][2] == "hi"

    def test_an_unsure_detection_lets_the_stronger_model_decide(self, monkeypatch):
        # If the detection itself was the problem, pinning it would just
        # repeat the mistake with a bigger model.
        recorder = _install(monkeypatch, _Recorder(
            self._suspicious("hi", 0.31), _clean(language="en",
                                                 model="large-v3-turbo")))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert recorder.passes[1][2] is None

    def test_a_transcript_without_a_probability_is_not_pinned(self, monkeypatch):
        # Parakeet and old checkpoints carry no detector confidence.
        transcript = self._suspicious("es", 0.9)
        transcript.pop("language_probability")
        recorder = _install(monkeypatch, _Recorder(
            transcript, _clean(language="es", model="large-v3-turbo")))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert recorder.passes[1][2] is None

    def test_the_pin_threshold_is_configurable(self, monkeypatch):
        monkeypatch.setenv("ASR_PIN_LANGUAGE_ABOVE", "0.95")
        recorder = _install(monkeypatch, _Recorder(
            self._suspicious("hi", 0.89), _clean(language="hi",
                                                 model="large-v3-turbo")))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert recorder.passes[1][2] is None


# --- one probe, one pass, one repair ----------------------------------------

class TestTranscriptionNeverLoops:
    """The video this module was rebuilt for went small -> large-v3-turbo ->
    large-v3 and the third load was OOM-killed with the second still resident
    (exit -9), after ~50 minutes of CPU, having produced nothing. Each rung
    re-transcribes the WHOLE video, so the count is the cost."""

    def test_a_clean_video_is_transcribed_exactly_once(self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(_clean()))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.passes] == ["initial"]

    def test_the_worst_case_is_two_passes(self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(
            _garbled(), _garbled(model="large-v3-turbo")))
        with pytest.raises(TranscriptQualityError):
            tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[1] for c in recorder.passes] == [None, "large-v3-turbo"]

    def test_the_repair_never_repeats_a_model(self, monkeypatch):
        # A deployment already running turbo must not transcribe with it twice.
        monkeypatch.setenv("WHISPER_RETRY_MODEL", "large-v3-turbo")
        recorder = _install(monkeypatch, _Recorder(
            _garbled(model="large-v3-turbo"), _garbled()))
        with pytest.raises(TranscriptQualityError):
            tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.passes] == ["initial"]


class TestTheRepairModel:
    def test_the_default_repair_is_turbo(self, monkeypatch):
        monkeypatch.delenv("WHISPER_RETRY_MODEL", raising=False)
        from subtitles import whisper_model_ladder
        assert whisper_model_ladder(["small"]) == ["large-v3-turbo"]

    def test_a_model_already_used_is_dropped(self, monkeypatch):
        monkeypatch.delenv("WHISPER_RETRY_MODEL", raising=False)
        from subtitles import whisper_model_ladder
        assert whisper_model_ladder(["large-v3-turbo"]) == []

    def test_the_repair_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("WHISPER_RETRY_MODEL", "")
        from subtitles import whisper_model_ladder
        assert whisper_model_ladder([]) == []

    def test_it_is_configurable(self, monkeypatch):
        monkeypatch.setenv("WHISPER_RETRY_MODEL", "large-v3")
        from subtitles import whisper_model_ladder
        assert whisper_model_ladder(["small"]) == ["large-v3"]


# --- the probe: choose the model instead of transcribing twice --------------

class TestTheProbe:
    """~75 seconds decide which model transcribes the whole video, so the
    strong model runs FIRST instead of third."""

    def test_a_struggling_probe_transcribes_with_the_stronger_model(
            self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(
            _clean(language="hi", model="large-v3-turbo"),
            probe=("large-v3-turbo", "hi", 0.89)))
        tb.transcribe_media_checked("video.mp4", duration=600.0)
        assert [c[0] for c in recorder.passes] == ["initial"]
        assert recorder.passes[0][1] == "large-v3-turbo"

    def test_the_probes_language_is_pinned_onto_that_pass(self, monkeypatch):
        """whisper-small detected hi at 89%; turbo, re-detecting for itself,
        decided en at 96% and emitted an English TRANSLATION of Hindi audio
        that scored 99/100. The pin is the only thing that stops that."""
        recorder = _install(monkeypatch, _Recorder(
            _clean(language="hi", model="large-v3-turbo"),
            probe=("large-v3-turbo", "hi", 0.89)))
        tb.transcribe_media_checked("video.mp4", duration=600.0)
        assert recorder.passes[0][2] == "hi"

    def test_an_unsure_probe_lets_the_stronger_model_detect(self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(
            _clean(language="hi", model="large-v3-turbo"),
            probe=("large-v3-turbo", "hi", 0.31)))
        tb.transcribe_media_checked("video.mp4", duration=600.0)
        assert recorder.passes[0][2] is None

    def test_a_clean_probe_changes_nothing(self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(
            _clean(), probe=(None, "en", 0.98)))
        tb.transcribe_media_checked("video.mp4", duration=600.0)
        assert recorder.passes[0][1] is None
        assert recorder.passes[0][2] is None


class TestProbeMechanics:
    def test_slices_are_spread_across_the_whole_timeline(self):
        slices = tb._probe_slices(600.0, 3, 25.0)
        assert len(slices) == 3
        starts = [start for start, _ in slices]
        assert starts == sorted(starts)
        # Not all bunched at the front: the first 30 seconds of a video are the
        # least representative part of it.
        assert starts[-1] > 300.0

    def test_slices_stay_inside_the_media(self):
        for start, end in tb._probe_slices(90.0, 3, 25.0):
            assert 0.0 <= start < end <= 90.0

    def test_a_video_too_short_to_sample_is_not_probed(self, monkeypatch):
        assert tb._probe_best_model("video.mp4", 20.0) == (None, None, None)

    def test_an_unknown_duration_is_not_probed(self, monkeypatch):
        assert tb._probe_best_model("video.mp4", None) == (None, None, None)

    def test_no_probe_when_the_configured_model_is_already_the_strong_one(
            self, monkeypatch):
        monkeypatch.setenv("WHISPER_MODEL", "large-v3-turbo")
        monkeypatch.setenv("WHISPER_RETRY_MODEL", "large-v3-turbo")
        assert tb._probe_best_model("video.mp4", 600.0) == (None, None, None)

    def test_it_can_be_switched_off(self, monkeypatch):
        monkeypatch.setenv("ASR_PROBE", "0")
        assert tb._probe_best_model("video.mp4", 600.0) == (None, None, None)

    def test_a_probe_that_explodes_never_fails_the_job(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("model download failed")
        monkeypatch.setattr(tb, "_probe_transcribe", boom)
        assert tb._probe_best_model("video.mp4", 600.0) == (None, None, None)


class TestWhatTheProbeDecidesFrom:
    """It is a QUALITY decision, not a language one: there is no list of
    languages that get the big model."""

    def _sampled(self, avg_logprob, language="en", probability=0.98):
        transcript = _clean(language=language, n=6)
        for segment in transcript["segments"]:
            segment["avg_logprob"] = avg_logprob
        transcript["language_probability"] = probability
        transcript.pop("asr")
        return transcript

    def _probe(self, monkeypatch, sampled):
        monkeypatch.setattr(tb, "_probe_transcribe",
                            lambda path, model, slices: sampled)
        return tb._probe_best_model("video.mp4", 600.0)

    def test_clean_audio_keeps_the_configured_model(self, monkeypatch):
        model, language, _ = self._probe(monkeypatch, self._sampled(-0.3))
        assert model is None
        assert language == "en"

    def test_a_guessing_decoder_upgrades_even_when_the_score_is_good(
            self, monkeypatch):
        """whisper-small's full transcript of the real video scored 77.8/GOOD
        while being a fifth invented. Status alone would have kept it; the
        decoder's own confidence (-0.89) is what gives it away."""
        model, language, probability = self._probe(
            monkeypatch, self._sampled(-0.89, language="hi", probability=0.99))
        assert model == "large-v3-turbo"
        assert (language, probability) == ("hi", 0.99)

    def test_clean_non_english_audio_is_not_upgraded(self, monkeypatch):
        model, _, _ = self._probe(
            monkeypatch, self._sampled(-0.3, language="hi"))
        assert model is None, "upgraded on the language rather than the quality"

    def test_noisy_english_audio_is_upgraded(self, monkeypatch):
        model, _, _ = self._probe(monkeypatch, self._sampled(-0.89))
        assert model == "large-v3-turbo"

    def test_the_probe_keeps_word_timestamps(self, monkeypatch):
        """evaluate_transcript docks a segment with text but no words by up to
        55 points, so a probe that dropped them would score badly on every
        video and upgrade all of them."""
        class Word:
            def __init__(self, word, start, end):
                self.word, self.start, self.end = word, start, end

        class Segment:
            start, end, text = 0.0, 2.0, " hello there"
            words = [Word(" hello", 0.0, 1.0), Word(" there", 1.0, 2.0)]
            avg_logprob, no_speech_prob = -0.3, 0.01
            compression_ratio, temperature = 1.5, 0.0

        class Info:
            language, language_probability, duration = "en", 0.97, 600.0

        monkeypatch.setattr(tb, "run_whisper_transcription",
                            lambda path, model_size=None, **p: ([Segment()], Info()))
        sampled = tb._probe_transcribe("video.mp4", "small", [(0.0, 25.0)])
        assert [w["word"] for w in sampled["segments"][0]["words"]] ==             [" hello", " there"]
