"""Completeness must not be an educational-content preference.

The blind verifier's STORY TEST used to ask what "lesson/result came from"
an anecdote. Stand-up has no lesson — it has a punchline — so the bar as
written could reject a complete joke for failing to be a lecture. That is a
genre preference wearing a completeness standard's clothes.

The fix is a levelling, NOT a loosening: a joke with its punchline cut off is
still incomplete, and a story that stops before its resolution still fails.
These tests pin the prompt text, because the prompt is the whole mechanism —
there is no code path to assert on.
"""
from meaningful_critic import BLIND_VERIFY_PROMPT, REPAIR_PROMPT


class TestPunchlineCountsAsPayoff:
    def test_a_punchline_is_named_as_a_valid_ending(self):
        assert "PUNCHLINE" in BLIND_VERIFY_PROMPT

    def test_comedy_is_not_required_to_carry_a_lesson(self):
        assert "does not owe anyone a moral, a lesson or a takeaway" \
            in BLIND_VERIFY_PROMPT

    def test_the_old_lesson_only_wording_is_gone(self):
        # "what lesson/result came from it" was the exact line that made a
        # joke look incomplete.
        assert "what lesson/result came from it" not in BLIND_VERIFY_PROMPT

    def test_the_ending_test_says_a_delivered_payoff_is_complete(self):
        assert "delivered its punchline or its result IS complete" \
            in BLIND_VERIFY_PROMPT


class TestExplanatoryContentIsUnchanged:
    def test_explanations_still_end_on_a_conclusion(self):
        assert "ends on the answer, the conclusion, the" in BLIND_VERIFY_PROMPT

    def test_the_incomplete_ending_examples_survive(self):
        for fragment in ("...and the reason why...",
                         "...what that means is...",
                         "...the first thing is..."):
            assert fragment in BLIND_VERIFY_PROMPT

    def test_a_grammatically_complete_sentence_can_still_fail(self):
        assert "grammatically complete sentence can still be semantically " \
            "incomplete" in BLIND_VERIFY_PROMPT


class TestTheBarIsNotLowered:
    def test_no_genre_gets_an_exemption(self):
        assert "Do not lower the bar for any" in BLIND_VERIFY_PROMPT

    def test_a_cut_off_joke_is_still_incomplete(self):
        assert "a joke with its punchline cut off is just as incomplete" \
            in BLIND_VERIFY_PROMPT

    def test_a_story_without_its_resolution_still_fails(self):
        assert "stops before the" in BLIND_VERIFY_PROMPT \
            and "resolution still fails" in BLIND_VERIFY_PROMPT

    def test_there_is_no_comedy_specific_bypass(self):
        # A rule keyed on genre would be a loophole, not a standard.
        lowered = BLIND_VERIFY_PROMPT.lower()
        for bypass in ("if it is comedy, accept", "skip the ending test",
                       "comedy is exempt"):
            assert bypass not in lowered


class TestOpeningStandardUntouched:
    def test_the_opening_test_still_demands_setup(self):
        assert "OPENING TEST" in BLIND_VERIFY_PROMPT
        assert "The first one or two sentences must establish enough context" \
            in BLIND_VERIFY_PROMPT

    def test_the_repair_prompt_still_refuses_topic_bleed(self):
        assert "CRITICAL TOPIC-BOUNDARY RULE" in REPAIR_PROMPT
        assert "Never append a different idea" in REPAIR_PROMPT


class TestMultilingualRulesReachBothPrompts:
    def test_the_blind_verifier_carries_them(self):
        assert "{multilingual_rules}" in BLIND_VERIFY_PROMPT

    def test_the_repairer_carries_them(self):
        assert "{multilingual_rules}" in REPAIR_PROMPT
