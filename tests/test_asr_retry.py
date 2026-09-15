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


class _Recorder:
    """Stands in for the two transcription entry points."""

    def __init__(self, first, retry=None):
        self.first = first
        self.retry = retry
        self.calls = []

    def initial(self, path):
        self.calls.append(("initial", None, None))
        return self.first

    def stronger(self, path, model_size=None, language=None):
        self.calls.append(("retry", model_size, language))
        if isinstance(self.retry, Exception):
            raise self.retry
        return self.retry


def _install(monkeypatch, recorder):
    monkeypatch.setattr(tb, "transcribe_media", recorder.initial)
    monkeypatch.setattr(tb, "_transcribe_with_whisper", recorder.stronger)
    return recorder


# --- 1, 2: the fast path ----------------------------------------------------

class TestCleanAudioIsTranscribedOnce:
    def test_good_english_does_not_retry(self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(_clean()))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.calls] == ["initial"]
        assert result["asr"]["quality"]["status"] == "GOOD"

    @pytest.mark.parametrize("language", ["en", "es", "hi", "ja", "ar", "pt"])
    def test_no_language_triggers_a_retry_on_its_own(self, monkeypatch, language):
        recorder = _install(monkeypatch, _Recorder(_clean(language=language)))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.calls] == ["initial"]


# --- 7, 8: the retry --------------------------------------------------------

class TestRetry:
    def test_bad_initial_then_good_retry_uses_the_retry(self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(
            _garbled(), _clean(language="hi", model="large-v3-turbo")))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)

        assert [c[0] for c in recorder.calls] == ["initial", "retry"]
        assert recorder.calls[1][1] == "large-v3-turbo"
        # Attempt 1 was only 31% sure of the language, so the stronger model
        # gets to detect it again (see TestLanguageIsPreserved).
        assert recorder.calls[1][2] is None
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

        assert [c[0] for c in recorder.calls] == ["initial", "retry"]
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
        assert [c[0] for c in recorder.calls] == ["initial"]

    def test_no_retry_when_the_initial_model_already_is_the_retry_model(self, monkeypatch):
        monkeypatch.setenv("WHISPER_RETRY_MODEL", "large-v3-turbo")
        recorder = _install(monkeypatch, _Recorder(
            _garbled(model="large-v3-turbo"), _clean()))
        with pytest.raises(TranscriptQualityError):
            tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.calls] == ["initial"]

    def test_gate_off_means_never_retry_and_never_fail(self, monkeypatch):
        monkeypatch.setenv("ASR_QUALITY_GATE", "0")
        recorder = _install(monkeypatch, _Recorder(_garbled(), _clean()))
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.calls] == ["initial"]
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
        called = []
        monkeypatch.setattr(tb, "gemini_second_opinion",
                            lambda *a, **k: called.append(1))
        _install(monkeypatch, _Recorder(_clean()))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert called == []

    def test_uncertain_is_flagged_but_still_good_on_its_own(self):
        from asr_quality import evaluate_transcript
        quality = evaluate_transcript(self._uncertain(), 120.0)
        assert quality["status"] == "GOOD"
        assert quality["uncertain"] is True

    def test_invented_words_downgrade_to_retry(self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(
            self._uncertain(), _clean(language="hi", model="large-v3-turbo")))
        monkeypatch.setattr(tb, "gemini_second_opinion", lambda *a, **k: False)
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.calls] == ["initial", "retry"]
        assert result["asr"]["model"] == "large-v3-turbo"

    def test_real_language_keeps_the_first_transcript(self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(self._uncertain(), _clean()))
        monkeypatch.setattr(tb, "gemini_second_opinion", lambda *a, **k: True)
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.calls] == ["initial"]
        assert result["asr"]["quality"]["status"] == "GOOD"

    def test_an_unavailable_check_never_fails_a_job_on_suspicion(self, monkeypatch):
        recorder = _install(monkeypatch, _Recorder(self._uncertain(), _clean()))
        monkeypatch.setattr(tb, "gemini_second_opinion", lambda *a, **k: None)
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert [c[0] for c in recorder.calls] == ["initial"]
        assert result["asr"]["quality"]["status"] == "GOOD"

    def test_the_check_can_be_switched_off(self, monkeypatch):
        from asr_quality import language_check_mode, gemini_second_opinion
        monkeypatch.setenv("ASR_QUALITY_GEMINI", "0")
        assert language_check_mode() == "0"
        assert gemini_second_opinion({"text": "x" * 200}) is None

    def test_a_real_retry_beats_a_downgraded_first_attempt(self, monkeypatch):
        """The downgraded transcript keeps a high STRUCTURAL score — structure
        was never its problem — so comparing scores alone would have kept the
        gibberish over a genuinely good retry."""
        good_retry = _clean(language="hi", model="large-v3-turbo")
        for segment in good_retry["segments"]:
            segment["avg_logprob"] = -0.45      # real speech, lower score
        _install(monkeypatch, _Recorder(self._uncertain(), good_retry))
        # Only the first attempt is uncertain enough to be read at all.
        monkeypatch.setattr(tb, "gemini_second_opinion", lambda *a, **k: False)
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert result["asr"]["model"] == "large-v3-turbo"

    def test_a_downgraded_score_no_longer_reads_as_passing(self, monkeypatch):
        _install(monkeypatch, _Recorder(self._uncertain(), RuntimeError("no model")))
        monkeypatch.setattr(tb, "gemini_second_opinion", lambda *a, **k: False)
        result = tb.transcribe_media_checked("video.mp4", duration=120.0)
        quality = result["asr"]["quality"]
        from asr_quality import good_score_threshold
        assert quality["status"] == "RETRY"
        assert quality["score"] < good_score_threshold()
        assert quality["downgraded_by"] == "language_check"


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
        assert recorder.calls[1][2] == "hi"

    def test_an_unsure_detection_lets_the_stronger_model_decide(self, monkeypatch):
        # If the detection itself was the problem, pinning it would just
        # repeat the mistake with a bigger model.
        recorder = _install(monkeypatch, _Recorder(
            self._suspicious("hi", 0.31), _clean(language="en",
                                                 model="large-v3-turbo")))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert recorder.calls[1][2] is None

    def test_a_transcript_without_a_probability_is_not_pinned(self, monkeypatch):
        # Parakeet and old checkpoints carry no detector confidence.
        transcript = self._suspicious("es", 0.9)
        transcript.pop("language_probability")
        recorder = _install(monkeypatch, _Recorder(
            transcript, _clean(language="es", model="large-v3-turbo")))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert recorder.calls[1][2] is None

    def test_the_pin_threshold_is_configurable(self, monkeypatch):
        monkeypatch.setenv("ASR_PIN_LANGUAGE_ABOVE", "0.95")
        recorder = _install(monkeypatch, _Recorder(
            self._suspicious("hi", 0.89), _clean(language="hi",
                                                 model="large-v3-turbo")))
        tb.transcribe_media_checked("video.mp4", duration=120.0)
        assert recorder.calls[1][2] is None
