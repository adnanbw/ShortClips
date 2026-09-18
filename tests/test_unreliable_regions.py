"""Poorly-read speech is a note on a candidate, never a veto.

This file used to assert the opposite: that a candidate with more than 20% of
its runtime inside an examined-and-BAD stretch was deleted before the critic
ran. That filter was removed because it fired on exactly the videos it was
supposed to help. A reader sampling 25-second windows produces fair evidence
about a TRANSCRIPT and weak evidence about any one clip, and the Blind Context
Critic already reads every candidate in full — so a rough Hinglish transcript
lost most of its proposals to a judge that had never read them, and the
pipeline reported "no usable semantic candidates" instead of the real problem.

What must still hold:
  * a candidate's timestamps are never edited, and the caller's dict is never
    mutated;
  * silence about a stretch means "not examined", never "bad".
"""
import pytest

from meaningful_pipeline import _flag_unreliable_candidates
from asr_semantic import unreliable_ranges


def _candidate(cid, start, end):
    return {"candidate_id": cid, "start": start, "end": end,
            "duration": end - start, "topic": "t"}


BAD = [(45.0, 90.0), (300.0, 345.0)]


class TestNothingIsDropped:
    def test_a_candidate_inside_good_speech_is_kept_unannotated(self):
        kept = _flag_unreliable_candidates([_candidate("C1", 100.0, 160.0)], BAD)
        assert [c["candidate_id"] for c in kept] == ["C1"]
        assert "unreliable_overlap_seconds" not in kept[0]

    def test_a_candidate_entirely_inside_bad_speech_survives(self):
        kept = _flag_unreliable_candidates([_candidate("C1", 50.0, 85.0)], BAD)
        assert [c["candidate_id"] for c in kept] == ["C1"]
        assert kept[0]["unreliable_overlap_seconds"] == pytest.approx(35.0)

    def test_the_overlap_is_recorded_for_the_debug_json(self):
        kept = _flag_unreliable_candidates([_candidate("C1", 85.0, 145.0)], BAD)
        assert kept[0]["unreliable_overlap_seconds"] == pytest.approx(5.0)

    def test_every_candidate_survives_however_bad_the_transcript(self):
        candidates = [
            _candidate("C1", 0.0, 40.0),      # clean
            _candidate("C2", 50.0, 85.0),     # inside a bad region
            _candidate("C3", 305.0, 340.0),   # inside the other bad region
        ]
        kept = _flag_unreliable_candidates(candidates, BAD)
        assert [c["candidate_id"] for c in kept] == ["C1", "C2", "C3"]


class TestTimestampsAndCallerState:
    def test_timestamps_are_never_edited(self):
        kept = _flag_unreliable_candidates([_candidate("C1", 85.0, 145.0)], BAD)
        assert kept[0]["start"] == 85.0 and kept[0]["end"] == 145.0

    def test_the_callers_dict_is_not_mutated(self):
        original = _candidate("C1", 85.0, 145.0)
        _flag_unreliable_candidates([original], BAD)
        assert "unreliable_overlap_seconds" not in original


class TestSilenceIsNotGuilt:
    def test_no_regions_means_no_annotation(self):
        candidates = [_candidate("C1", 50.0, 85.0)]
        assert _flag_unreliable_candidates(candidates, []) == candidates

    def test_an_unexamined_transcript_keeps_every_candidate(self):
        candidates = [_candidate(f"C{i}", i * 60.0, i * 60.0 + 40.0)
                      for i in range(5)]
        assert _flag_unreliable_candidates(candidates, unreliable_ranges(None)) \
            == candidates

    def test_only_examined_bad_regions_come_from_a_verdict(self):
        semantic = {"regions": [
            {"id": "R01", "start": 0.0, "end": 45.0, "status": "GOOD", "score": 95},
            {"id": "R02", "start": 45.0, "end": 90.0, "status": "BAD", "score": 10},
            {"id": "R03", "start": 90.0, "end": 135.0, "status": "PARTIAL", "score": 55},
        ]}
        assert unreliable_ranges(semantic) == [(45.0, 90.0)]


class TestRobustness:
    def test_a_candidate_without_usable_times_is_kept_not_crashed_on(self):
        broken = {"candidate_id": "C1", "start": None, "end": None}
        assert _flag_unreliable_candidates([broken], BAD) == [broken]

    def test_a_zero_length_candidate_does_not_divide_by_zero(self):
        _flag_unreliable_candidates([_candidate("C1", 50.0, 50.0)], BAD)

    def test_an_empty_candidate_list_is_fine(self):
        assert _flag_unreliable_candidates([], BAD) == []
