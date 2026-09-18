"""Representative semantic validation of a transcript.

The failure this file exists for: ``large-v3-turbo`` returned the real Hinglish
stand-up at ``avg_logprob`` -0.29 and a structural score of 100/100 while a
third of it was invented phonetic Hindi. Structure cannot read, so the only
check that can see this must actually look — and must look in the MIDDLE, which
is where that transcript's corruption lived. The previous attempt sampled the
first and last 1500 characters and saw none of it.
"""
import pytest

import asr_semantic
from asr_semantic import (aggregate, build_regions, build_samples,
                          overlap_seconds, unreliable_ranges)


def _transcript(texts, seconds=5.0):
    segments = []
    for i, text in enumerate(texts):
        segments.append({"start": i * seconds, "end": (i + 1) * seconds,
                         "text": text,
                         "words": [{"word": " " + w, "start": i * seconds,
                                    "end": i * seconds + 0.3}
                                   for w in text.split()]})
    return {"language": "hi", "text": " ".join(texts), "segments": segments}


# --- sampling ---------------------------------------------------------------

class TestSampling:
    def test_samples_are_spread_across_the_whole_timeline(self):
        # 120 segments x 5s = 10 minutes, exactly the real case.
        transcript = _transcript([f"line {i}" for i in range(120)])
        samples = build_samples(transcript, count=6, seconds=25.0)

        assert len(samples) == 6
        starts = [s["start"] for s in samples]
        assert starts == sorted(starts)
        # The last sample must come from the END, not from the first minute:
        # this is the property the character-offset version did not have.
        assert starts[0] < 30
        assert starts[-1] > 0.75 * 600

    def test_samples_do_not_overlap(self):
        transcript = _transcript([f"line {i}" for i in range(120)])
        samples = build_samples(transcript, count=6, seconds=25.0)
        for earlier, later in zip(samples, samples[1:]):
            assert earlier["end"] <= later["start"]

    def test_a_sample_carries_neighbouring_speech_not_one_fragment(self):
        transcript = _transcript([f"line {i}" for i in range(120)])
        samples = build_samples(transcript, count=6, seconds=25.0)
        for sample in samples:
            assert sample["end"] - sample["start"] >= 15.0
            assert len(sample["text"].split()) > 5

    def test_short_transcripts_do_not_produce_empty_samples(self):
        samples = build_samples(_transcript(["one two three"]), count=6)
        assert len(samples) == 1
        assert samples[0]["text"] == "one two three"

    def test_empty_transcript_samples_nothing(self):
        assert build_samples({"segments": []}) == []
        assert build_samples(None) == []
        assert build_samples({"segments": [{"start": 0, "end": 1, "text": "  "}]}) == []

    def test_segments_without_usable_times_are_skipped(self):
        transcript = _transcript(["a b c", "d e f"])
        transcript["segments"][0]["start"] = None
        samples = build_samples(transcript, count=2)
        assert all(s["start"] is not None for s in samples)


class TestDenseRegions:
    def test_regions_cover_the_whole_transcript(self):
        transcript = _transcript([f"line {i}" for i in range(120)])
        regions = build_regions(transcript, seconds=45.0)

        assert regions[0]["start"] == 0.0
        assert regions[-1]["end"] == pytest.approx(600.0)
        # Contiguous: the point of the dense pass is that nothing is unexamined,
        # because only examined speech may veto a clip.
        for earlier, later in zip(regions, regions[1:]):
            assert later["start"] == earlier["end"]

    def test_regions_are_roughly_the_requested_length(self):
        transcript = _transcript([f"line {i}" for i in range(120)])
        for region in build_regions(transcript, seconds=45.0)[:-1]:
            assert 30.0 <= region["end"] - region["start"] <= 50.0


# --- aggregation ------------------------------------------------------------

def _samples(*spans):
    return [{"id": f"R{i + 1:02d}", "start": a, "end": b, "text": "x"}
            for i, (a, b) in enumerate(spans)]


def _verdicts(*pairs):
    return [{"id": f"R{i + 1:02d}", "status": status, "usability": score,
             "code_switching": False, "reason": ""}
            for i, (status, score) in enumerate(pairs)]


class TestAggregation:
    def test_all_good_is_good(self):
        result = aggregate(_samples((0, 30), (100, 130), (200, 230)),
                           _verdicts(("GOOD", 95), ("GOOD", 90), ("GOOD", 92)))
        assert result["status"] == "GOOD"
        assert result["score"] == pytest.approx(92.3, abs=0.5)

    def test_all_bad_is_bad(self):
        result = aggregate(_samples((0, 30), (100, 130)),
                           _verdicts(("BAD", 5), ("BAD", 10)))
        assert result["status"] == "BAD"

    def test_a_mix_is_partial(self):
        """The real transcript: understandable regions AND invented ones."""
        result = aggregate(_samples((0, 30), (100, 130), (200, 230), (300, 330)),
                           _verdicts(("GOOD", 95), ("BAD", 10),
                                     ("GOOD", 90), ("BAD", 15)))
        assert result["status"] == "PARTIAL"

    def test_one_bad_region_does_not_condemn_a_good_transcript(self):
        result = aggregate(
            _samples((0, 30), (100, 130), (200, 230), (300, 330), (400, 430)),
            _verdicts(("GOOD", 95), ("GOOD", 93), ("GOOD", 90),
                      ("GOOD", 94), ("BAD", 20)))
        assert result["status"] == "PARTIAL"

    def test_regions_are_weighted_by_seconds_not_by_count(self):
        # One long bad stretch outweighs two short good ones.
        long_bad = aggregate(_samples((0, 5), (10, 15), (20, 200)),
                             _verdicts(("GOOD", 95), ("GOOD", 95), ("BAD", 5)))
        assert long_bad["status"] == "BAD"

    def test_unknown_region_ids_are_ignored(self):
        result = aggregate(_samples((0, 30)),
                           [{"id": "NOPE", "status": "BAD", "usability": 0}])
        assert result["status"] is None

    def test_a_malformed_status_is_treated_as_partial_not_trusted(self):
        result = aggregate(_samples((0, 30)),
                           [{"id": "R01", "status": "banana", "usability": 50}])
        assert result["regions"][0]["status"] == "PARTIAL"

    def test_no_verdicts_returns_no_status(self):
        assert aggregate(_samples((0, 30)), [])["status"] is None

    def test_thresholds_are_configurable(self, monkeypatch):
        samples = _samples((0, 30), (100, 130), (200, 230), (300, 330))
        verdicts = _verdicts(("GOOD", 95), ("GOOD", 93),
                             ("GOOD", 90), ("BAD", 20))
        assert aggregate(samples, verdicts)["status"] == "PARTIAL"
        monkeypatch.setenv("ASR_SEMANTIC_GOOD_SHARE", "0.70")
        assert aggregate(samples, verdicts)["status"] == "PARTIAL"


# --- using the regions ------------------------------------------------------

class TestUnreliableRanges:
    def _mixed(self):
        return {"regions": [
            {"id": "R01", "start": 0.0, "end": 45.0, "status": "GOOD", "score": 95},
            {"id": "R02", "start": 45.0, "end": 90.0, "status": "BAD", "score": 10},
            {"id": "R03", "start": 90.0, "end": 135.0, "status": "PARTIAL", "score": 55},
        ]}

    def test_only_bad_regions_are_off_limits_by_default(self):
        assert unreliable_ranges(self._mixed()) == [(45.0, 90.0)]

    def test_partial_can_be_included_when_asked(self):
        assert unreliable_ranges(self._mixed(), include_partial=True) == [
            (45.0, 90.0), (90.0, 135.0)]

    def test_no_semantic_data_means_nothing_is_off_limits(self):
        # Silence is "not examined", never "bad". A transcript nobody read must
        # not have every clip vetoed.
        assert unreliable_ranges(None) == []
        assert unreliable_ranges({}) == []
        assert unreliable_ranges({"regions": []}) == []

    def test_overlap_is_measured_in_seconds(self):
        ranges = [(45.0, 90.0)]
        assert overlap_seconds(0.0, 45.0, ranges) == 0.0
        assert overlap_seconds(40.0, 50.0, ranges) == pytest.approx(5.0)
        assert overlap_seconds(50.0, 60.0, ranges) == pytest.approx(10.0)
        assert overlap_seconds(100.0, 120.0, ranges) == 0.0

    def test_overlap_handles_several_ranges(self):
        ranges = [(0.0, 10.0), (20.0, 30.0)]
        assert overlap_seconds(5.0, 25.0, ranges) == pytest.approx(10.0)


# --- the mode contract ------------------------------------------------------

class TestCheckMode:
    @pytest.mark.parametrize("value,expected", [
        ("0", "0"), ("1", "1"), ("auto", "auto"), ("AUTO", "auto"),
        ("", "auto"), ("nonsense", "auto"),
    ])
    def test_mode_parsing(self, monkeypatch, value, expected):
        monkeypatch.setenv("ASR_QUALITY_GEMINI", value)
        assert asr_semantic.check_mode() == expected

    def test_default_is_auto(self, monkeypatch):
        monkeypatch.delenv("ASR_QUALITY_GEMINI", raising=False)
        assert asr_semantic.check_mode() == "auto"

    def test_disabled_never_calls_out(self, monkeypatch):
        monkeypatch.setenv("ASR_QUALITY_GEMINI", "0")
        monkeypatch.setattr(asr_semantic, "_call_gemini", _must_not_run)
        assert asr_semantic.review_transcript(_transcript(["a b c"])) is None

    def test_no_api_key_never_calls_out(self, monkeypatch):
        monkeypatch.delenv("ASR_QUALITY_GEMINI", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(asr_semantic, "_call_gemini", _must_not_run)
        assert asr_semantic.review_transcript(_transcript(["a b c"])) is None


def _must_not_run(*args, **kwargs):
    raise AssertionError("the language check ran when it must not have")


# --- end to end, with the model stubbed -------------------------------------

class TestReviewTranscript:
    def _run(self, monkeypatch, statuses, dense=False):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.delenv("ASR_QUALITY_GEMINI", raising=False)
        seen = {}

        def fake_call(samples, api_key, model_name):
            seen["samples"] = samples
            return [{"id": s["id"], "status": statuses[i % len(statuses)],
                     "usability": 90 if statuses[i % len(statuses)] == "GOOD" else 10,
                     "code_switching": False, "reason": "x"}
                    for i, s in enumerate(samples)]

        monkeypatch.setattr(asr_semantic, "_call_gemini", fake_call)
        transcript = _transcript([f"line {i}" for i in range(120)])
        result = asr_semantic.review_transcript(transcript, dense=dense)
        return result, seen

    def test_corruption_in_the_middle_is_detected(self, monkeypatch):
        """The exact blind spot of the first/last-1500-characters version."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.delenv("ASR_QUALITY_GEMINI", raising=False)

        def fake_call(samples, api_key, model_name):
            out = []
            for sample in samples:
                middle = 150.0 <= sample["start"] <= 450.0
                out.append({"id": sample["id"],
                            "status": "BAD" if middle else "GOOD",
                            "usability": 10 if middle else 92,
                            "code_switching": False, "reason": "invented words"})
            return out

        monkeypatch.setattr(asr_semantic, "_call_gemini", fake_call)
        result = asr_semantic.review_transcript(
            _transcript([f"line {i}" for i in range(120)]))

        assert result["status"] == "PARTIAL"
        bad = [r for r in result["regions"] if r["status"] == "BAD"]
        assert bad, "the middle of the transcript was never looked at"
        assert all(150.0 <= r["start"] <= 450.0 for r in bad)

    def test_a_good_transcript_reports_good(self, monkeypatch):
        result, _ = self._run(monkeypatch, ["GOOD"])
        assert result["status"] == "GOOD"

    def test_dense_mode_examines_far_more_of_the_video(self, monkeypatch):
        sparse, sparse_seen = self._run(monkeypatch, ["GOOD"])
        dense, dense_seen = self._run(monkeypatch, ["GOOD"], dense=True)
        assert len(dense_seen["samples"]) > len(sparse_seen["samples"])
        assert dense["dense"] is True and sparse["dense"] is False

    def test_an_api_failure_is_not_a_guilty_verdict(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.delenv("ASR_QUALITY_GEMINI", raising=False)

        def boom(*args, **kwargs):
            raise RuntimeError("503 unavailable")

        monkeypatch.setattr(asr_semantic, "_call_gemini", boom)
        assert asr_semantic.review_transcript(_transcript(["a b c d e"])) is None

    def test_the_prompt_carries_every_sample_labelled(self, monkeypatch):
        _, seen = self._run(monkeypatch, ["GOOD"])
        prompt = asr_semantic.REVIEW_PROMPT.format(
            samples=asr_semantic._format_samples(seen["samples"]))
        for sample in seen["samples"]:
            assert sample["id"] in prompt
        # One request for all of them: the cost of reading a video must not
        # scale with how carefully we read it.
        assert prompt.count("###") == len(seen["samples"])


# --- code-switching must never be called damage -----------------------------

class TestCodeSwitchingIsNormal:
    def test_the_prompt_says_code_switching_is_normal_speech(self):
        prompt = asr_semantic.REVIEW_PROMPT
        assert "मुझे actually ये approach better लगती है" in prompt
        assert "normal bilingual speech" in prompt

    def test_the_prompt_forbids_penalising_an_unexpected_language(self):
        assert "never mark a region down for being in a language you did not" \
            in asr_semantic.REVIEW_PROMPT.lower()

    def test_a_code_switched_region_can_be_good(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.delenv("ASR_QUALITY_GEMINI", raising=False)
        monkeypatch.setattr(asr_semantic, "_call_gemini",
                            lambda samples, api_key, model_name: [
                                {"id": s["id"], "status": "GOOD", "usability": 95,
                                 "code_switching": True, "reason": "Hinglish"}
                                for s in samples])
        result = asr_semantic.review_transcript(
            _transcript(["मुझे actually ये approach better लगती है"] * 20))
        assert result["status"] == "GOOD"
        assert all(r["code_switching"] for r in result["regions"])


class TestLongVideosAreFullyExamined:
    """A two-hour source produces ~160 dense regions. Putting them in one
    request would hit MAX_PROMPT_CHARS and quietly drop the back half, which
    would then read as "not examined" and veto nothing — a silent blind spot
    of exactly the kind this module was written to remove."""

    def test_batches_stay_inside_the_prompt_budget(self):
        samples = [{"id": f"R{i:02d}", "start": i * 45.0, "end": (i + 1) * 45.0,
                    "text": "x" * 5000} for i in range(40)]
        batches = asr_semantic._batched(samples)
        assert len(batches) > 1
        for batch in batches:
            assert sum(len(s["text"]) for s in batch) <= asr_semantic.MAX_PROMPT_CHARS

    def test_no_sample_is_lost_or_duplicated(self):
        samples = [{"id": f"R{i:02d}", "start": i * 45.0, "end": (i + 1) * 45.0,
                    "text": "x" * 5000} for i in range(40)]
        flat = [s["id"] for batch in asr_semantic._batched(samples) for s in batch]
        assert flat == [s["id"] for s in samples]

    def test_a_small_transcript_is_one_request(self):
        samples = [{"id": "R01", "start": 0.0, "end": 45.0, "text": "short"}]
        assert len(asr_semantic._batched(samples)) == 1

    def test_every_region_of_a_long_video_gets_a_verdict(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.delenv("ASR_QUALITY_GEMINI", raising=False)
        seen = []

        def fake_call(samples, api_key, model_name):
            seen.append(len(samples))
            return [{"id": s["id"], "status": "GOOD", "usability": 90,
                     "code_switching": False, "reason": ""} for s in samples]

        monkeypatch.setattr(asr_semantic, "_call_gemini", fake_call)
        # ~40 minutes of speech, verbose segments.
        transcript = _transcript(["word " * 200] * 480, seconds=5.0)
        result = asr_semantic.review_transcript(transcript, dense=True)

        assert len(seen) > 1, "one request could not have held it all"
        regions = asr_semantic.build_regions(transcript)
        assert len(result["regions"]) == len(regions)
