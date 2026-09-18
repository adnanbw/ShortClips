"""Wordless footage must be clipped by vision, not fail on an empty transcript.

The vision path used to fire only on a missing audio track. Three prod jobs
on 25-aug-2026 had audio but no speech (a nursery rhyme transcribed as
"Uh uh", a dashcam drive as "Yeah."), went through the transcript path and
died with "Gemini did not return usable clips".
"""
import pytest

main = pytest.importorskip("main")  # needs cv2/mediapipe, absent in minimal CI


def _t(*texts):
    return {"language": "en",
            "segments": [{"start": i, "end": i + 1, "text": t} for i, t in enumerate(texts)]}


class TestRealFailures:
    def test_nursery_rhyme_uh_uh_is_sparse(self):
        assert main.speech_is_sparse(_t("Uh uh"), 169)

    def test_dashcam_yeah_is_sparse(self):
        assert main.speech_is_sparse(_t("Yeah."), 621)


class TestRealSpeechIsKept:
    def test_a_talk_is_not_sparse(self):
        words = ["so today we are going to look at how this works"] * 60   # ~10 w/segment
        assert not main.speech_is_sparse(_t(*words), 600)

    def test_a_short_clip_with_a_sentence_is_not_sparse(self):
        # 20s with one real sentence: nothing to gain from vision.
        assert not main.speech_is_sparse(_t("welcome back everyone to another episode of the show"), 20)

    def test_a_song_with_lyrics_is_not_sparse(self):
        # "Wheels on the Bus" style: repetitive but plenty of words.
        assert not main.speech_is_sparse(_t(*["the wheels on the bus go round and round"] * 30), 180)


class TestEdges:
    def test_empty_segments_is_sparse(self):
        assert main.speech_is_sparse({"segments": []}, 100)

    def test_none_transcript_is_sparse(self):
        assert main.speech_is_sparse(None, 100)

    def test_zero_duration_does_not_divide_by_zero(self):
        assert main.speech_is_sparse(_t("hi"), 0)
        assert not main.speech_is_sparse(_t(*["a b c d e f g h i j"] * 3), 0)


class TestFailureMessageNamesTheRightStage:
    """A job that produces no clips must say WHY.

    "Clip detection failed — the AI model did not return usable clips" was
    printed for a video whose transcript was a third invented: the selector had
    behaved correctly on input it could not use, and the message blamed it.
    """

    def _transcript(self, status=None):
        transcript = _t("hello there everyone welcome back to the show")
        if status:
            transcript["asr"] = {"quality": {"status": status}}
        return transcript

    def test_a_partial_transcript_is_reported_as_a_transcription_problem(self):
        message = main.clip_failure_message(self._transcript("PARTIAL"))
        assert "Transcription quality was too low" in message
        assert "did not return usable clips" not in message

    def test_a_bad_transcript_is_reported_the_same_way(self):
        assert "Transcription quality was too low" in             main.clip_failure_message(self._transcript("BAD"))

    def test_a_good_transcript_still_reports_a_clip_detection_failure(self):
        # Genuinely no clip-shaped material: the old message is the right one.
        message = main.clip_failure_message(self._transcript("GOOD"))
        assert message == main.NO_CLIPS_MESSAGE

    def test_a_transcript_with_no_diagnostics_keeps_the_old_message(self):
        assert main.clip_failure_message(self._transcript()) == main.NO_CLIPS_MESSAGE
        assert main.clip_failure_message(None) == main.NO_CLIPS_MESSAGE
        assert main.clip_failure_message({}) == main.NO_CLIPS_MESSAGE

    def test_the_transcription_message_is_the_shared_one(self):
        from asr_quality import TranscriptQualityError
        assert main.clip_failure_message(self._transcript("BAD")) ==             TranscriptQualityError.USER_MESSAGE
