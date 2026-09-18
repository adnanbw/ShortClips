"""An over-long candidate must be narrowed, not binned.

On the real Hinglish run the Candidate Finder returned four proposals and three
were thrown away for one reason only — 110.9s, 95.3s and 117.2s. A single
candidate reached the critic, the critic correctly rejected it, and the job
produced nothing. But a 110-second idea very often CONTAINS a complete
60-second one, and the model that proposed it is the cheapest thing that can
find it.

The repair is strictly a narrowing, and everything the model returns is
re-validated here against the real sentence list. There is no clamping and no
arbitrary word trimming: boundaries stay on sentence units exactly like every
other boundary in this pipeline.
"""
import sys
import types
from types import SimpleNamespace

import pytest

from meaningful_selector import (
    MAX_CLIP_SECONDS,
    MIN_CLIP_SECONDS,
    repair_overlong_candidate,
)

#: Every fixture sentence is this long, so a span of N sentences is N * this.
SENTENCE_SECONDS = 6.0

#: How many fixture sentences it takes to be genuinely over-long. DERIVED from
#: the real limit rather than written out: these tests used to hardcode the
#: arithmetic of a 90-second cap, so raising the cap to 110 silently turned
#: "S0001->S0018 is 108s, past the limit" into a span that was inside it and
#: three tests started asserting the opposite of what they were written to say.
OVERLONG_SENTENCES = int(MAX_CLIP_SECONDS // SENTENCE_SECONDS) + 2


def _sid(number):
    return f"S{number:04d}"


def _stub_genai_types():
    """Minimal google.genai stand-in for environments without the SDK.

    The repair function needs GenerateContentConfig to ask for structured
    output; the fake client below ignores it. Stubbed only when the real SDK is
    absent, so in the container these tests exercise the real import path.
    """
    try:
        import google.genai.types  # noqa: F401
        return None
    except Exception:
        pass

    google = sys.modules.setdefault("google", types.ModuleType("google"))
    genai = types.ModuleType("google.genai")
    genai_types = types.ModuleType("google.genai.types")
    genai_types.GenerateContentConfig = lambda **kwargs: kwargs
    genai.types = genai_types
    google.genai = genai
    sys.modules["google.genai"] = genai
    sys.modules["google.genai.types"] = genai_types
    return genai


_stub_genai_types()


def _sentences(count=None, seconds=SENTENCE_SECONDS):
    """S0001..S000N, each ``seconds`` long, back to back."""
    return [{"id": f"S{i + 1:04d}",
             "start": round(i * seconds, 3),
             "end": round((i + 1) * seconds, 3),
             "duration": seconds,
             "text": f"sentence number {i + 1}.",
             "word_count": 3}
            for i in range(count or OVERLONG_SENTENCES * 2)]


class _FakeModels:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def generate_content(self, model=None, contents=None, config=None):
        self.prompts.append(contents)
        return SimpleNamespace(parsed=self.payload)


class _FakeClient:
    def __init__(self, payload):
        self.models = _FakeModels(payload)


def _decision(action="REPAIR", start=None, end=None, reason="ok"):
    return SimpleNamespace(action=action, new_start_sentence=start,
                           new_end_sentence=end, reason=reason)


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.delenv("MEANINGFUL_REPAIR_OVERLONG", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def _repair(client, sentences=None, start="S0001", end=None, **kwargs):
    return repair_overlong_candidate(
        sentences=sentences if sentences is not None else _sentences(),
        start_id=start, end_id=end or _sid(OVERLONG_SENTENCES),
        topic="a flight story",
        client=client, **kwargs)


# --- the successful case ----------------------------------------------------

class TestSuccessfulRepair:
    def test_an_overlong_proposal_becomes_a_valid_clip(self):
        # The default proposal is over the cap; S0003..S0012 is 60s.
        client = _FakeClient(_decision(start="S0003", end="S0012"))
        result = _repair(client)

        assert result is not None
        assert result["start_sentence"] == "S0003"
        assert result["end_sentence"] == "S0012"
        assert result["duration"] == pytest.approx(60.0)

    def test_the_repaired_duration_is_inside_the_allowed_band(self):
        client = _FakeClient(_decision(start="S0003", end="S0012"))
        result = _repair(client)
        assert MIN_CLIP_SECONDS <= result["duration"] <= MAX_CLIP_SECONDS

    def test_the_repaired_ids_stay_inside_the_original_proposal(self):
        first, last = 2, OVERLONG_SENTENCES + 1
        client = _FakeClient(_decision(start="S0005", end="S0014"))
        result = _repair(client, start=_sid(first), end=_sid(last))
        sentences = _sentences()
        index = {s["id"]: i for i, s in enumerate(sentences)}
        assert result is not None
        assert index[_sid(first)] <= index[result["start_sentence"]]
        assert index[result["end_sentence"]] <= index[_sid(last)]

    def test_the_original_proposal_is_recorded(self):
        client = _FakeClient(_decision(start="S0003", end="S0012"))
        result = _repair(client)
        assert result["original_start"] == "S0001"
        assert result["original_end"] == "S0020"
        assert result["original_duration"] == pytest.approx(120.0)

    def test_timestamps_come_from_the_sentences_never_from_the_model(self):
        sentences = _sentences()
        client = _FakeClient(_decision(start="S0003", end="S0012"))
        result = _repair(client, sentences=sentences)
        # Exactly the stored sentence boundaries — no padding, no clamping.
        assert result["duration"] == pytest.approx(
            sentences[11]["end"] - sentences[2]["start"])

    def test_only_the_proposed_range_is_shown_to_the_model(self):
        first, last = 5, OVERLONG_SENTENCES + 4
        client = _FakeClient(_decision(start="S0006", end="S0015"))
        _repair(client, start=_sid(first), end=_sid(last))
        prompt = client.models.prompts[0]
        assert _sid(first) in prompt and _sid(last) in prompt
        # Nothing outside the proposal may be offered as a boundary.
        assert _sid(first - 1) not in prompt
        assert _sid(last + 1) not in prompt


# --- rejection --------------------------------------------------------------

class TestRejection:
    def test_an_explicit_reject_returns_none(self):
        assert _repair(_FakeClient(_decision(action="REJECT"))) is None

    def test_a_still_overlong_answer_is_refused(self):
        # One sentence short of the original: still past the cap, so it is
        # not a repair at all.
        assert _repair(_FakeClient(_decision(
            start="S0001", end=_sid(OVERLONG_SENTENCES - 1)))) is None

    def test_a_too_short_answer_is_refused(self):
        # A model that trims to fit rather than to finish the idea.
        assert _repair(_FakeClient(
            _decision(start="S0003", end="S0004"))) is None

    def test_an_answer_outside_the_original_range_is_refused(self):
        assert _repair(_FakeClient(
            _decision(start="S0025", end="S0030"))) is None

    def test_an_answer_that_only_widens_is_refused(self):
        assert _repair(_FakeClient(
            _decision(start="S0001", end="S0030"))) is None

    def test_returning_the_original_range_unchanged_is_refused(self):
        assert _repair(_FakeClient(
            _decision(start="S0001", end="S0020"))) is None

    def test_reversed_boundaries_are_refused(self):
        assert _repair(_FakeClient(
            _decision(start="S0012", end="S0003"))) is None

    def test_an_invented_sentence_id_is_refused(self):
        assert _repair(_FakeClient(
            _decision(start="S9999", end="S0012"))) is None

    def test_a_missing_id_is_refused(self):
        assert _repair(_FakeClient(_decision(start=None, end=None))) is None

    def test_an_api_failure_is_refused_quietly(self):
        class Boom:
            class models:
                @staticmethod
                def generate_content(**kwargs):
                    raise RuntimeError("503")
        assert _repair(Boom()) is None

    def test_an_unparsed_response_is_refused(self):
        class NoParse:
            class models:
                @staticmethod
                def generate_content(**kwargs):
                    return SimpleNamespace(parsed=None)
        assert _repair(NoParse()) is None


# --- guards -----------------------------------------------------------------

class TestGuards:
    def test_a_candidate_already_short_enough_is_not_repaired(self):
        # Nothing to narrow: the caller should never have asked.
        assert _repair(_FakeClient(_decision(start="S0002", end="S0005")),
                       start="S0001", end="S0010") is None

    def test_unknown_boundary_ids_are_refused(self):
        assert _repair(_FakeClient(_decision(start="S0003", end="S0012")),
                       start="NOPE", end="S0020") is None

    def test_the_repair_can_be_switched_off(self, monkeypatch):
        monkeypatch.setenv("MEANINGFUL_REPAIR_OVERLONG", "0")
        client = _FakeClient(_decision(start="S0003", end="S0012"))
        assert _repair(client) is None
        assert client.models.prompts == [], "paid for a disabled feature"

    def test_no_api_key_and_no_client_is_refused(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert repair_overlong_candidate(
            sentences=_sentences(), start_id="S0001", end_id="S0020") is None


# --- the prompt must not invite a bad trim ----------------------------------

class TestPromptIntent:
    def test_it_asks_for_a_complete_section_not_a_truncation(self):
        client = _FakeClient(_decision(start="S0003", end="S0012"))
        _repair(client)
        prompt = client.models.prompts[0].lower()
        assert "complete" in prompt
        assert "stops mid-thought" in prompt
        assert "reject" in prompt

    def test_it_carries_the_shared_multilingual_rules(self):
        client = _FakeClient(_decision(start="S0003", end="S0012"))
        _repair(client)
        assert "LANGUAGE RULES" in client.models.prompts[0]

    def test_it_states_the_real_duration_band(self):
        client = _FakeClient(_decision(start="S0003", end="S0012"))
        _repair(client, minimum_seconds=15.0, maximum_seconds=90.0)
        prompt = client.models.prompts[0]
        assert "between 15 and 90 seconds" in prompt
