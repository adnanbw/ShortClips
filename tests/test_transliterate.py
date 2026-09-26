"""Burned text is Latin; the words are still the speaker's.

The contract these tests defend is ONE WORD IN, ONE WORD OUT. Captions are
karaoke — every word carries its own start/end and is highlighted on its own —
so a model that merges two words or splits one shifts the timing of everything
after it in the block. Every failure mode here therefore ends the same way:
that word keeps its original text, and nothing moves.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transliterate as tr


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "मैंने वह किया।",       # devanagari
    "こんにちは",            # cjk
    "مرحبا",                # arabic
    "привет",               # cyrillic
    "mixed मुझे actually",   # code-switched: one non-Latin word is enough
])
def test_non_latin_text_needs_transliteration(text):
    assert tr.needs_transliteration(text)


@pytest.mark.parametrize("text", [
    "maine wah kiya.",
    "Bali trip with three boys",
    "2026 was a disaster",
    "¿Qué tal? Señor Muñoz",   # Latin with diacritics is still Latin
    "😂 🌴",                    # emoji are script-less
    "",
])
def test_latin_text_is_left_alone(text):
    assert not tr.needs_transliteration(text)


def test_emoji_alone_never_triggers_a_gemini_call():
    """A caption of "2026 😂" is already renderable; paying for it would mean
    every English job carries a transliteration bill."""
    assert tr.transliterate_words(["😂", "2026", "flight"]) == {}


# ---------------------------------------------------------------------------
# Per-word acceptance
# ---------------------------------------------------------------------------

def test_accepts_a_plain_romanisation():
    assert tr._accept("किया", "kiya")


def test_accepts_a_loanword_spelled_back_in_english():
    """This is the whole reason the romaniser is a model and not a library:
    unidecode turns फ्लाइट into "phlaaitt", which is not a word."""
    assert tr._accept("फ्लाइट", "flight")
    assert tr._accept("एक्सपीरियंस", "experience")


def test_rejects_output_still_in_the_source_script():
    assert not tr._accept("किया", "किया")


def test_rejects_an_empty_result():
    assert not tr._accept("किया", "   ")


def test_rejects_one_word_becoming_two():
    """A merge or a gloss ("धन्यवाद" -> "thank you") would put two words under
    one word's timing and leave the next word with none."""
    assert not tr._accept("धन्यवाद", "thank you")


def test_allows_two_words_in_when_the_input_had_a_space():
    assert tr._accept("क्या हाल", "kya haal")


# ---------------------------------------------------------------------------
# Chunk-level validation
# ---------------------------------------------------------------------------

def test_a_mismatched_chunk_is_never_zipped(monkeypatch):
    """zip() truncates to the shorter side without complaining, so pairing a
    short answer with its input would silently mis-align every word after the
    one the model dropped."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    # Always one short, at every size, so no split ever lines up and the
    # truncating zip is the only way a pair could be produced.
    monkeypatch.setattr(tr, "_gemini_chunk",
                        lambda words, *a, **k: ["x"] * (len(words) - 1))
    assert tr.transliterate_words(["कल", "में", "एक"]) == {}


def test_a_mismatched_chunk_is_split_and_retried(monkeypatch):
    """Measured on a real 12-minute job: one chunk came back 301 words for 300
    and took all 300 with it, which put Devanagari into the last seconds of a
    published clip. Splitting turns "300 words lost" into at most one."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    sizes = []

    def fake(words, *a, **k):
        sizes.append(len(words))
        if len(words) == 4:
            return ["a", "b", "c", "d", "e"]        # 5 back for 4 sent
        return [f"w{i}" for i in range(len(words))]  # halves answer correctly

    monkeypatch.setattr(tr, "_gemini_chunk", fake)
    words = ["कल", "में", "एक", "लिफ्ट"]
    mapping = tr.transliterate_words(words)

    assert sizes == [4, 2, 2]          # one failure, then both halves
    assert len(mapping) == 4           # every word recovered


def test_a_single_word_that_keeps_failing_loses_only_itself(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake(words, *a, **k):
        if "में" in words:
            raise RuntimeError("503")
        return ["kal" if len(words) == 1 else "x" for _ in words]

    monkeypatch.setattr(tr, "_gemini_chunk", fake)
    mapping = tr.transliterate_words(["कल", "में"])
    assert mapping == {"कल": "kal"}


def test_the_split_recursion_terminates_on_total_failure(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []

    def fake(words, *a, **k):
        calls.append(len(words))
        raise RuntimeError("503")

    monkeypatch.setattr(tr, "_gemini_chunk", fake)
    assert tr.transliterate_words(["कल"] * 8) == {}
    # 8 unique-by-position words -> at most 2n-1 calls, never unbounded.
    assert len(calls) <= 15


def test_a_failing_chunk_does_not_fail_the_others(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(tr, "CHUNK_WORDS", 2)
    calls = []

    def fake(words, *a, **k):
        calls.append(list(words))
        if len(calls) == 1:
            raise RuntimeError("503 from the API")
        return ["ek", "lift"]

    monkeypatch.setattr(tr, "_gemini_chunk", fake)
    mapping = tr.transliterate_words(["कल", "में", "एक", "लिफ्ट"])
    assert mapping == {"एक": "ek", "लिफ्ट": "lift"}


def test_the_first_spelling_of_a_word_wins(monkeypatch):
    """The same word must read the same way in every caption of the clip, even
    when two chunks disagree about how to spell it."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(tr, "CHUNK_WORDS", 1)
    answers = iter([["kiya"], ["kia"]])
    monkeypatch.setattr(tr, "_gemini_chunk", lambda *a, **k: next(answers))
    assert tr.transliterate_words(["किया", "किया"]) == {"किया": "kiya"}


# ---------------------------------------------------------------------------
# Transcript annotation
# ---------------------------------------------------------------------------

def _transcript():
    return {
        "language": "hi",
        "segments": [{
            "text": "कल में एक lift में",
            "words": [
                {"word": " कल", "start": 0.0, "end": 0.4},
                {"word": " में", "start": 0.4, "end": 0.7},
                {"word": " lift", "start": 0.7, "end": 1.1},
            ],
        }],
    }


def test_annotate_adds_latin_and_keeps_the_original(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(tr, "_gemini_chunk", lambda *a, **k: ["kal", "mein"])
    transcript = _transcript()

    assert tr.annotate_transcript(transcript) == 2
    words = transcript["segments"][0]["words"]
    assert words[0]["latin"] == " kal"
    assert words[0]["word"] == " कल"      # the real word is never overwritten
    assert "latin" not in words[2]        # already Latin: not sent, not stored


def test_annotate_preserves_the_leading_space_word_boundary(monkeypatch):
    """faster-whisper marks a word boundary with a LEADING SPACE, and
    subtitles.merge_continuation_words reads exactly that. A romanisation that
    dropped it would make every word look like a continuation fragment and glue
    the whole caption into one token."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    # The model is asked for bare words and answers with bare words; the space
    # is re-attached by annotate_transcript, not by the prompt.
    monkeypatch.setattr(tr, "_gemini_chunk", lambda *a, **k: ["kal", "mein"])
    transcript = _transcript()

    tr.annotate_transcript(transcript)

    latins = [w["latin"] for w in transcript["segments"][0]["words"]
              if "latin" in w]
    assert latins == [" kal", " mein"]


def test_segment_text_is_never_touched(monkeypatch):
    """The selector, the critic and the metadata writer read segment text. They
    have to reason about what was really said, in the script it was said in."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(tr, "_gemini_chunk", lambda *a, **k: ["kal", "mein"])
    transcript = _transcript()
    before = transcript["segments"][0]["text"]

    tr.annotate_transcript(transcript)

    assert transcript["segments"][0]["text"] == before


def test_annotate_is_idempotent_so_a_resumed_job_pays_nothing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []

    def fake(words, *a, **k):
        calls.append(list(words))
        return ["kal", "mein"]

    monkeypatch.setattr(tr, "_gemini_chunk", fake)
    transcript = _transcript()
    tr.annotate_transcript(transcript)
    tr.annotate_transcript(transcript)
    assert len(calls) == 1


def test_caption_script_original_disables_the_whole_thing(monkeypatch):
    monkeypatch.setenv("CAPTION_SCRIPT", "original")
    monkeypatch.setattr(tr, "_gemini_chunk",
                        lambda *a, **k: pytest.fail("should not be called"))
    transcript = _transcript()
    assert tr.annotate_transcript(transcript) == 0
    assert "latin" not in transcript["segments"][0]["words"][0]


def test_a_silent_transcript_costs_nothing():
    assert tr.annotate_transcript({"segments": []}) == 0
    assert tr.annotate_transcript({}) == 0


# ---------------------------------------------------------------------------
# What actually gets burned
# ---------------------------------------------------------------------------

def test_caption_text_prefers_latin():
    assert tr.caption_text({"word": " कल", "latin": " kal"}) == " kal"


def test_caption_text_falls_back_to_the_spoken_word():
    """A word the model failed on keeps its script. That costs the caption its
    alphabet; dropping it would cost the caption its timing."""
    assert tr.caption_text({"word": " कल"}) == " कल"
    assert tr.caption_text({"word": " कल", "latin": "  "}) == " कल"


def test_to_latin_text_rewrites_a_hook_word_for_word(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(tr, "_gemini_chunk",
                        lambda *a, **k: ["scuba", "diving", "ka"])
    out = tr.to_latin_text("स्कूबा डाइविंग का experience 😂")
    assert out == "scuba diving ka experience 😂"


def test_to_latin_text_leaves_english_alone():
    assert tr.to_latin_text("Bali trip with three guy friends") == \
        "Bali trip with three guy friends"
