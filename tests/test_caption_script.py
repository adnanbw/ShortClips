"""The burn path reads the Latin spelling; everything upstream reads the words.

These cover the seam between transliterate.py and subtitles.py — the place a
romanised transcript actually turns into pixels.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subtitles


def _transcript(words):
    return {"language": "hi", "segments": [{"text": "…", "words": words}]}


def test_blocks_are_built_from_the_latin_spelling():
    transcript = _transcript([
        {"word": " कल", "latin": " kal", "start": 0.0, "end": 0.4},
        {"word": " में", "latin": " mein", "start": 0.4, "end": 0.8},
    ])
    blocks = subtitles._collect_word_blocks(transcript, 0.0, 10.0)
    assert [w["word"] for w in blocks[0]] == ["kal", "mein"]


def test_a_word_without_a_latin_spelling_keeps_its_own():
    """Per-WORD fallback, not per-transcript: one word the model failed on must
    not drag the rest of the line back into the source script."""
    transcript = _transcript([
        {"word": " कल", "latin": " kal", "start": 0.0, "end": 0.4},
        {"word": " में", "start": 0.4, "end": 0.8},
    ])
    blocks = subtitles._collect_word_blocks(transcript, 0.0, 10.0)
    assert [w["word"] for w in blocks[0]] == ["kal", "में"]


def test_timings_are_untouched_by_romanisation():
    transcript = _transcript([
        {"word": " एक्सपीरियंस", "latin": " experience", "start": 3.0, "end": 3.9},
    ])
    block = subtitles._collect_word_blocks(transcript, 2.0, 10.0)[0]
    assert (block[0]["start"], block[0]["end"]) == (1.0, 1.9)


def test_english_transcripts_are_completely_unaffected():
    transcript = _transcript([
        {"word": " Bali", "start": 0.0, "end": 0.4},
        {"word": " trip", "start": 0.4, "end": 0.8},
    ])
    blocks = subtitles._collect_word_blocks(transcript, 0.0, 10.0)
    assert [w["word"] for w in blocks[0]] == ["Bali", "trip"]


# ---------------------------------------------------------------------------
# merge_continuation_words
# ---------------------------------------------------------------------------

def test_a_continuation_fragment_merges_its_latin_too():
    """faster-whisper splits compound words into a base plus a fragment with no
    leading space. Merging only the `word` field would leave `latin` holding
    the base alone, and the fragment's script would reappear in the caption."""
    merged = subtitles.merge_continuation_words([
        {"word": " लिफ्ट", "latin": " lift", "start": 0.0, "end": 0.4},
        {"word": "में", "latin": "mein", "start": 0.4, "end": 0.6},
    ])
    assert len(merged) == 1
    assert merged[0]["latin"] == " liftmein"
    assert merged[0]["word"] == " लिफ्टमें"
    assert merged[0]["end"] == 0.6


def test_an_already_latin_fragment_contributes_its_own_text():
    merged = subtitles.merge_continuation_words([
        {"word": " यूट्यूब", "latin": " youtube", "start": 0.0, "end": 0.4},
        {"word": "-Kanal.", "start": 0.4, "end": 0.6},
    ])
    assert merged[0]["latin"] == " youtube-Kanal."


def test_a_latin_fragment_on_an_unannotated_base_uses_the_base_word():
    merged = subtitles.merge_continuation_words([
        {"word": " YouTube", "start": 0.0, "end": 0.4},
        {"word": "में", "latin": "mein", "start": 0.4, "end": 0.6},
    ])
    assert merged[0]["latin"] == " YouTubemein"


def test_merging_english_words_adds_no_latin_key():
    merged = subtitles.merge_continuation_words([
        {"word": " YouTube", "start": 0.0, "end": 0.4},
        {"word": "-Kanal.", "start": 0.4, "end": 0.6},
    ])
    assert "latin" not in merged[0]
    assert merged[0]["word"] == " YouTube-Kanal."


def test_merge_still_does_not_mutate_its_input():
    words = [
        {"word": " लिफ्ट", "latin": " lift", "start": 0.0, "end": 0.4},
        {"word": "में", "latin": "mein", "start": 0.4, "end": 0.6},
    ]
    subtitles.merge_continuation_words(words)
    assert words[0]["latin"] == " lift"


# ---------------------------------------------------------------------------
# The rendered file
# ---------------------------------------------------------------------------

def test_the_burned_ass_file_carries_no_devanagari(tmp_path):
    transcript = _transcript([
        {"word": " कल", "latin": " kal", "start": 0.0, "end": 0.4},
        {"word": " में", "latin": " mein", "start": 0.4, "end": 0.8},
        {"word": " एक", "latin": " ek", "start": 0.8, "end": 1.2},
    ])
    out = tmp_path / "subs.ass"
    # The style every generated clip actually gets, so the uppercase pass is
    # exercised too: str.upper() is a no-op on Devanagari and a real change on
    # the romanised text that replaced it.
    assert subtitles.generate_ass(
        transcript, 0.0, 10.0, str(out),
        uppercase=subtitles.AUTO_CAPTION_STYLE["uppercase"])
    body = out.read_text(encoding="utf-8-sig")
    assert "KAL" in body
    assert not any("ऀ" <= ch <= "ॿ" for ch in body)


def test_an_rtl_block_becomes_ltr_once_romanised(tmp_path):
    """_is_rtl_block reads the DISPLAY text, so a romanised Arabic line is laid
    out left to right — the three-run visual-order flip that libass needs for
    real Arabic would read backwards if it fired on Latin text."""
    transcript = _transcript([
        {"word": " مرحبا", "latin": " marhaba", "start": 0.0, "end": 0.4},
        {"word": " بك", "latin": " bik", "start": 0.4, "end": 0.8},
    ])
    block = subtitles._collect_word_blocks(transcript, 0.0, 10.0)[0]
    assert not subtitles._is_rtl_block(block)
