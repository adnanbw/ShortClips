"""Judges rank clips; they do not get to end the job.

Two real runs sit behind this file. An English stand-up set scored 95-100 on
six of seven candidates and produced six clips. A Hinglish one scored 80/40,
30/40, 30/20 and 60/30 — four candidates that each contained a real joke — and
the job died with "Clip detection failed - the AI model did not return usable
clips for this video", which is not true: the model returned four.

Four judges run in series (finder, duration gate, critic, opening guard) and
every one of them could return an empty list. What is enforced here:

  * a merely-good clip is published, not only a flawless one;
  * a repair that made a clip worse is discarded, not kept;
  * when nothing clears the bar at all, the best of what was found is
    published and flagged, instead of the job failing;
  * one duration band, shared, so raising it cannot half-apply.
"""
import pytest

import meaningful_critic as critic
import meaningful_pipeline as pipeline
import meaningful_opening_guard as guard
from meaningful_selector import (
    CANDIDATE_OVERLAP_SECONDS,
    WHAT_YOU_ARE_JUDGING,
    MAX_CLIP_SECONDS,
    MIN_CLIP_SECONDS,
    duration_rule_text,
)


def _review(cid, standalone, completeness, decision="REJECT"):
    return {
        "candidate_id": cid,
        "decision": decision,
        "start": 10.0,
        "end": 50.0,
        "final_verification": {
            "standalone_score": standalone,
            "completeness_score": completeness,
        },
    }


class TestTheFloor:
    """The exact four candidates from the failing Hinglish run."""

    HINGLISH = [
        _review("C001", 80, 40),
        _review("C002", 30, 40),
        _review("C003", 30, 20),
        _review("C004", 60, 30),
    ]

    def test_the_job_no_longer_produces_nothing(self):
        assert pipeline._best_effort(self.HINGLISH)

    def test_the_best_candidate_comes_first(self):
        best = pipeline._best_effort(self.HINGLISH)
        assert best[0]["candidate_id"] == "C001"

    def test_everything_it_returns_is_flagged(self):
        for item in pipeline._best_effort(self.HINGLISH):
            assert item["low_confidence"] is True

    def test_it_is_capped(self, monkeypatch):
        monkeypatch.setenv("MEANINGFUL_FALLBACK_MAX", "2")
        assert len(pipeline._best_effort(self.HINGLISH)) == 2

    def test_it_ranks_by_the_critics_own_scores(self):
        order = [c["candidate_id"] for c in pipeline._best_effort(self.HINGLISH)]
        assert order == ["C001", "C004", "C002"]

    def test_the_callers_reviews_are_not_mutated(self):
        pipeline._best_effort(self.HINGLISH)
        assert "low_confidence" not in self.HINGLISH[0]

    def test_it_can_be_switched_off(self, monkeypatch):
        monkeypatch.setenv("MEANINGFUL_ALWAYS_PRODUCE", "0")
        assert pipeline._always_produce() is False

    def test_it_is_on_by_default(self, monkeypatch):
        monkeypatch.delenv("MEANINGFUL_ALWAYS_PRODUCE", raising=False)
        assert pipeline._always_produce() is True

    def test_nothing_reviewed_still_means_nothing_produced(self):
        assert pipeline._best_effort([]) == []

    def test_a_review_without_scores_does_not_crash(self):
        assert pipeline._best_effort([{"candidate_id": "C1"}])


class TestTheShippingTier:
    """85 on BOTH axes, on a scale whose own prompt says 90+ means genuinely
    excellent, is a preference. The floor is the quality question."""

    def test_the_default_floor_is_below_the_flawless_bar(self, monkeypatch):
        monkeypatch.delenv("MEANINGFUL_ACCEPT_FLOOR", raising=False)
        assert critic.accept_floor() < 85

    def test_it_is_configurable(self, monkeypatch):
        monkeypatch.setenv("MEANINGFUL_ACCEPT_FLOOR", "55")
        assert critic.accept_floor() == 55.0

    def test_a_junk_value_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("MEANINGFUL_ACCEPT_FLOOR", "banana")
        assert critic.accept_floor() == 70.0


class TestARepairMustNotMakeThingsWorse:
    """Measured: a 60/30 candidate was 'repaired' to 30/20 and the repaired
    version replaced it unconditionally."""

    def test_a_worse_repair_is_recognised(self):
        assert critic._is_worse(
            {"standalone_score": 30, "completeness_score": 20},
            {"standalone_score": 60, "completeness_score": 30},
        )

    def test_a_better_repair_is_not(self):
        assert not critic._is_worse(
            {"standalone_score": 30, "completeness_score": 80},
            {"standalone_score": 30, "completeness_score": 40},
        )

    def test_an_identical_repair_is_not_worse(self):
        same = {"standalone_score": 50, "completeness_score": 50}
        assert not critic._is_worse(same, dict(same))

    def test_missing_scores_do_not_crash(self):
        assert not critic._is_worse({}, {})
        assert not critic._is_worse({"standalone_score": "x"}, {})


class TestOneDurationBand:
    def test_every_stage_uses_the_same_maximum(self):
        assert guard.MAX_CLIP_SECONDS == MAX_CLIP_SECONDS
        assert critic.MAX_CLIP_SECONDS == MAX_CLIP_SECONDS
        assert pipeline.MAX_CLIP_SECONDS == MAX_CLIP_SECONDS

    def test_a_complete_clip_past_the_old_cap_is_now_allowed(self):
        # The lift bit's punchline landed at 107s and the only repair that
        # could include it was deleted by a bare `> 90.0`.
        assert MAX_CLIP_SECONDS >= 107.0

    def test_windows_can_contain_the_longest_allowed_clip(self):
        # An overlap smaller than the longest clip makes long moments straddle
        # two windows and be proposable in neither.
        assert CANDIDATE_OVERLAP_SECONDS >= MAX_CLIP_SECONDS

    def test_the_prompts_still_ask_for_a_short_clip(self):
        rule = duration_rule_text()
        assert "25-60" in rule
        assert "preferred" in rule

    def test_the_prompts_state_the_real_band(self):
        assert f"{MIN_CLIP_SECONDS:.0f}-{MAX_CLIP_SECONDS:.0f}" in duration_rule_text()


class TestTheJudgesKnowWhatTheyAreLookingAt:
    """The critic prompt opened with "You are seeing EXACTLY AND ONLY what the
    eventual viewer will hear", which is false twice: the viewer hears SPEECH,
    not a transcript of it, and the viewer also SEES.

    Measured on the Hinglish job, replaying the same saved transcript through
    the same models: before this block, 5 candidates -> 0 accepted, three of
    the five rejections citing ASR noise ("contains significant transcription
    errors ('पुगली उलूश शुष')", "fragmented, repetitive, and nonsensical").
    After it, 5 -> 4 accepted at 95/95, the critic writing "the narrative is
    easy to follow despite some transcription noise". The English job was
    unchanged at 6 clips.
    """

    def test_it_says_the_transcript_is_machine_made(self):
        assert "automatic speech recognition" in WHAT_YOU_ARE_JUDGING

    def test_it_forbids_scoring_down_for_transcription_noise(self):
        assert "TRANSCRIPTION NOISE" in WHAT_YOU_ARE_JUDGING
        assert "Never lower standalone_score" in WHAT_YOU_ARE_JUDGING

    def test_it_says_the_viewer_can_see_the_speaker(self):
        assert "THE VIEWER CAN SEE" in WHAT_YOU_ARE_JUDGING
        assert "NEVER a missing-context failure" in WHAT_YOU_ARE_JUDGING

    def test_it_still_fails_a_pronoun_with_no_antecedent(self):
        """The distinction that keeps the opening guard honest: the SPEAKER is
        visible, everyone they refer to is not. Without this the block would
        excuse "she told me..." openings, which no picture explains."""
        assert "no antecedent" in WHAT_YOU_ARE_JUDGING
        assert "still a failure" in WHAT_YOU_ARE_JUDGING

    def test_the_blind_critic_is_told(self):
        assert "{what_you_are_judging}" in critic.BLIND_VERIFY_PROMPT

    def test_the_boundary_repairer_is_told(self):
        # It moves boundaries from the critic's failure reason, so it would
        # otherwise try to "repair" its way around a garbled stretch.
        assert "{what_you_are_judging}" in critic.REPAIR_PROMPT

    def test_the_opening_guard_is_told(self):
        import inspect
        source = inspect.getsource(guard)
        assert source.count("{WHAT_YOU_ARE_JUDGING}") == 2

    def test_the_false_premise_is_gone(self):
        assert "what the eventual viewer will hear" not in critic.BLIND_VERIFY_PROMPT
