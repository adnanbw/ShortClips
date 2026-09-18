"""Transcript checkpoint: a job re-run after a container restart must not
transcribe (the slow, paid stage) a second time, and must never reuse the
transcript of a DIFFERENT video.

app.py re-enqueues an interrupted job with the same command and the same
output directory (resume manifest). main.py leaves the finished transcript
there, tied to the source's name and duration, picks it up on the re-run, and
removes it once the job completes.
"""
import json
import os

import pytest

main = pytest.importorskip("main")  # needs cv2/mediapipe, absent in minimal CI

TRANSCRIPT = {"text": "hola", "language": "es",
              "segments": [{"start": 0.0, "end": 1.0, "text": "hola", "words": []}]}
SRC = "/app/output/job1/video.mp4"
DUR = 58.64


def _save(d, transcript=TRANSCRIPT, src=SRC, dur=DUR):
    main.save_transcript_checkpoint(str(d), transcript, src, dur)


def _load(d, src=SRC, dur=DUR):
    return main.load_transcript_checkpoint(str(d), src, dur)


class TestRoundTrip:
    def test_same_source_is_reused(self, tmp_path):
        _save(tmp_path)
        assert os.path.isfile(tmp_path / main.TRANSCRIPT_CHECKPOINT)
        assert _load(tmp_path) == TRANSCRIPT

    def test_resumed_cloud_job_redownloads_to_the_same_name(self, tmp_path):
        # The re-run's file lives at the same path; a slightly different
        # duration reading (container decoders differ by a frame) still matches.
        _save(tmp_path)
        assert _load(tmp_path, dur=DUR + 0.3) == TRANSCRIPT

    def test_missing_checkpoint_is_none(self, tmp_path):
        assert _load(tmp_path) is None


class TestNeverTheWrongTranscript:
    def test_another_file_in_the_same_directory_is_ignored(self, tmp_path):
        # THE RISK: CLI run on A dies after transcribing, then B is processed
        # in the same directory without -o. B must get its own transcript.
        _save(tmp_path, src="/videos/A.mp4")
        assert _load(tmp_path, src="/videos/B.mp4") is None

    def test_same_name_but_different_length_is_ignored(self, tmp_path):
        _save(tmp_path, dur=600.0)
        assert _load(tmp_path, dur=58.6) is None

    def test_legacy_shape_without_source_is_ignored(self, tmp_path):
        (tmp_path / main.TRANSCRIPT_CHECKPOINT).write_text(json.dumps(TRANSCRIPT))
        assert _load(tmp_path) is None


class TestRobustness:
    def test_empty_transcript_is_ignored(self, tmp_path):
        _save(tmp_path, transcript={"segments": []})
        assert _load(tmp_path) is None

    def test_truncated_file_is_ignored(self, tmp_path):
        (tmp_path / main.TRANSCRIPT_CHECKPOINT).write_text('{"source": {"name": "vi')
        assert _load(tmp_path) is None

    def test_unwritable_directory_does_not_raise(self, tmp_path):
        main.save_transcript_checkpoint(str(tmp_path / "missing" / "dir"), TRANSCRIPT, SRC, DUR)

    def test_clear_is_idempotent(self, tmp_path):
        _save(tmp_path)
        main.clear_transcript_checkpoint(str(tmp_path))
        main.clear_transcript_checkpoint(str(tmp_path))
        assert _load(tmp_path) is None

    def test_checkpoint_is_hidden_from_job_listings(self):
        # app.py globs *_metadata.json and *.mp4 in the job dir; a dotfile
        # named unlike either can never be mistaken for a deliverable.
        assert main.TRANSCRIPT_CHECKPOINT.startswith(".")
        assert not main.TRANSCRIPT_CHECKPOINT.endswith("_metadata.json")
        assert not main.TRANSCRIPT_CHECKPOINT.endswith(".mp4")


class TestRealAdaptiveDiagnostics:
    """A checkpoint must survive the diagnostics the adaptive ASR path attaches.

    Every real job logged "Could not save transcript checkpoint: Circular
    reference detected" and silently lost its checkpoint — the thing that
    exists so a container restart does not pay for transcription twice. The
    cause was structural, not incidental: each attempt record held a live
    reference to ``transcript["asr"]``, and the attempts list was then stored
    INSIDE that same dict, so the object graph pointed at itself.

    These tests use a transcript shaped exactly like the one
    ``transcribe_media_checked`` returns, so the regression cannot come back
    unnoticed.
    """

    def _adaptive_transcript(self):
        quality = {
            "status": "PARTIAL",
            "score": 69.9,
            "reasons": ["parts of the transcript are not real words"],
            "uncertain": True,
            "downgraded_by": "language_check",
            "metrics": {"segments": 2, "words": 4, "avg_logprob": -0.29,
                        "script_profile": {"devanagari": 0.9, "latin": 0.1}},
            "semantic": {
                "status": "PARTIAL",
                "score": 61.0,
                "shares": {"GOOD": 0.6, "PARTIAL": 0.0, "BAD": 0.4},
                "dense": True,
                "sampled_regions": 2,
                "regions": [
                    {"id": "R01", "start": 0.0, "end": 45.0, "status": "GOOD",
                     "score": 95, "reason": "", "code_switching": True},
                    {"id": "R02", "start": 45.0, "end": 90.0, "status": "BAD",
                     "score": 10, "reason": "invented words",
                     "code_switching": False},
                ],
            },
        }
        asr = {
            "backend": "whisper",
            "model": "large-v3-turbo",
            "device": "cpu",
            "compute_type": "int8",
            "task": "transcribe",
            "language_pinned": "hi",
            "language_detection_segments": 4,
            "quality": quality,
            "attempts": [
                {"attempt": 1, "backend": "whisper", "model": "small",
                 "device": "cpu", "compute_type": "int8", "language": "hi",
                 "language_probability": 0.89, "language_pinned": None,
                 "status": "RETRY", "score": 69.9, "semantic_status": "BAD",
                 "semantic_score": 12.0, "reason": None},
                {"attempt": 2, "backend": "whisper", "model": "large-v3-turbo",
                 "device": "cpu", "compute_type": "int8", "language": "hi",
                 "language_probability": 1.0, "language_pinned": "hi",
                 "status": "PARTIAL", "score": 69.9,
                 "semantic_status": "PARTIAL", "semantic_score": 61.0,
                 "reason": "semantic_bad"},
            ],
        }
        return {
            "text": "मुझे actually ये approach better लगती है",
            "language": "hi",
            "language_probability": 1.0,
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "मुझे actually ये",
                 "words": [{"word": " मुझे", "start": 0.0, "end": 0.7}],
                 "avg_logprob": -0.29, "no_speech_prob": 0.01,
                 "compression_ratio": 1.4, "temperature": 0.0},
                {"start": 2.0, "end": 4.0, "text": "approach better लगती है",
                 "words": [{"word": " approach", "start": 2.0, "end": 2.6}],
                 "avg_logprob": -0.31, "no_speech_prob": 0.02,
                 "compression_ratio": 1.5, "temperature": 0.0},
            ],
            "asr": asr,
        }

    def test_the_real_shape_is_plain_json(self):
        # json.dumps with no default= and no fallback: exactly what
        # save_transcript_checkpoint does.
        json.dumps(self._adaptive_transcript())

    def test_a_circular_reference_would_be_caught_here(self):
        # Proof the test is actually load-bearing: recreate the old bug and
        # confirm this assertion fails on it.
        broken = self._adaptive_transcript()
        broken["asr"]["attempts"][0]["asr"] = broken["asr"]
        with pytest.raises(ValueError):
            json.dumps(broken)

    def test_it_round_trips_through_the_checkpoint(self, tmp_path):
        transcript = self._adaptive_transcript()
        _save(tmp_path, transcript=transcript)
        assert _load(tmp_path) == transcript

    def test_the_diagnostics_survive_the_round_trip(self, tmp_path):
        transcript = self._adaptive_transcript()
        _save(tmp_path, transcript=transcript)
        loaded = _load(tmp_path)
        asr = loaded["asr"]
        assert asr["quality"]["status"] == "PARTIAL"
        assert [a["attempt"] for a in asr["attempts"]] == [1, 2]
        assert asr["attempts"][1]["reason"] == "semantic_bad"
        # The region map is what clip selection needs to avoid bad speech; a
        # checkpoint that dropped it would resume into unfiltered selection.
        assert [r["status"] for r in asr["quality"]["semantic"]["regions"]] \
            == ["GOOD", "BAD"]

    def test_word_timestamps_survive_the_round_trip(self, tmp_path):
        transcript = self._adaptive_transcript()
        _save(tmp_path, transcript=transcript)
        words = _load(tmp_path)["segments"][0]["words"]
        assert words[0]["word"] == " मुझे"
        assert words[0]["start"] == 0.0

    def test_saving_reports_no_error(self, tmp_path, capsys):
        _save(tmp_path, transcript=self._adaptive_transcript())
        assert "Circular reference" not in capsys.readouterr().out
