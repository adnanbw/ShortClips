"""build_sentence_units() must find sentence boundaries in any script.

The unit builder is the first thing the meaningful pipeline does, and its only
punctuation rule used to be ". ? !". A Hindi transcript ends sentences with ।,
a Japanese one with 。, an Arabic question with ؟ — none of which matched, so
the whole video collapsed into a handful of pause-split blobs and the candidate
finder had nothing sentence-shaped to choose boundaries from.

Word timestamps are the product, so every test here also checks that units
still start and end on real word boundaries.
"""
import pytest

from meaningful_selector import build_sentence_units, _ends_sentence


def _transcript(words, start=0.0, step=0.4, gaps=None):
    """One segment, evenly spaced words, optional {index: pause_after} gaps."""
    gaps = gaps or {}
    items = []
    t = start
    for i, word in enumerate(words):
        items.append({"word": " " + word, "start": round(t, 3),
                      "end": round(t + step, 3)})
        t += step + gaps.get(i, 0.0)
    return {"language": "xx", "segments": [
        {"start": start, "end": round(t, 3), "text": " ".join(words),
         "words": items}]}


def _texts(units):
    return [u["text"] for u in units]


# --- 9: punctuation across scripts ------------------------------------------

class TestSentenceTerminators:
    @pytest.mark.parametrize("word,expected", [
        ("end.", True), ("end?", True), ("end!", True),
        ("बात।", True), ("बात॥", True),          # Devanagari danda
        ("です。", True), ("ですか？", True), ("すごい！", True),  # CJK
        ("الجملة؟", True),                        # Arabic question mark
        ("جملہ۔", True),                          # Urdu full stop
        ("end⁉", True),                      # ⁉
        ("ends", False), ("and", False), ("बात", False), ("です", False),
        # Closing punctuation after the terminator must not cancel it.
        ('end."', True), ("end.)", True), ("です。」", True),
        # Initials and abbreviations are still not sentence ends.
        ("Dr.", False), ("e.", False), ("etc.", False),
    ])
    def test_terminator_recognition(self, word, expected):
        assert _ends_sentence(word) is expected

    def test_english_splits_on_full_stops(self):
        units = build_sentence_units(_transcript(
            "this is one. and this is two. and a third one here.".split()))
        assert len(units) == 3
        assert _texts(units)[0] == "this is one."

    def test_devanagari_splits_on_danda(self):
        units = build_sentence_units(_transcript(
            "मुझे ये approach अच्छी लगती है। इसमें planning करनी पड़ती है। "
            "और वो हमेशा help करता है।".split()))
        assert len(units) == 3

    def test_cjk_splits_on_ideographic_stop(self):
        units = build_sentence_units(_transcript(
            ["実はこの方法のほうがいいと思います。", "最初に計画を立てる必要があるからです。",
             "それはいつも役に立ちます。"]))
        assert len(units) == 3

    def test_arabic_question_mark_ends_a_sentence(self):
        units = build_sentence_units(_transcript(
            "ما الذي لا ينجح في السوق؟ الجواب بسيط جدا. وهذا ما نتعلمه.".split()))
        assert len(units) == 3

    def test_code_switched_line_is_one_unit(self):
        # Latin words inside a Devanagari sentence must not split it.
        units = build_sentence_units(_transcript(
            "मुझे actually ये approach better लगती है क्योंकि planning "
            "हमेशा help करती है।".split()))
        assert len(units) == 1


# --- pauses remain a first-class boundary -----------------------------------

class TestPauseBoundaries:
    def test_unpunctuated_speech_splits_on_pauses(self):
        words = ["word%d" % i for i in range(30)]
        units = build_sentence_units(_transcript(words, gaps={9: 1.5, 19: 1.5}))
        assert len(units) >= 3

    def test_barely_punctuated_transcript_uses_a_shorter_pause(self):
        # 60 words, no terminators, only 0.8s breaths: the old 1.20s rule
        # produced one unit for the entire video.
        words = ["शब्द%d" % i for i in range(60)]
        gaps = {i: 0.8 for i in (9, 19, 29, 39, 49)}
        units = build_sentence_units(_transcript(words, gaps=gaps))
        assert len(units) >= 5

    def test_continuous_speech_cannot_exceed_the_hard_ceiling(self):
        # No punctuation and no gap at all — a live set over audience noise.
        # One unit longer than the 90s clip limit is useless to every stage
        # downstream, so the builder must still cut it.
        words = ["word%d" % i for i in range(400)]
        units = build_sentence_units(_transcript(words, step=0.4))
        assert len(units) > 1
        assert max(u["duration"] for u in units) <= 90.0


# --- invariants -------------------------------------------------------------

class TestInvariants:
    @pytest.mark.parametrize("words", [
        "this is one. and this is two. a third one here.".split(),
        "मुझे ये अच्छी लगती है। इसमें planning करनी पड़ती है।".split(),
        ["実はこの方法がいいと思います。", "最初に計画を立てる必要があります。"],
    ])
    def test_units_land_on_word_boundaries_and_cover_the_transcript(self, words):
        transcript = _transcript(words)
        flat = transcript["segments"][0]["words"]
        starts = {w["start"] for w in flat}
        ends = {w["end"] for w in flat}
        units = build_sentence_units(transcript)

        assert units
        for unit in units:
            assert unit["start"] in starts
            assert unit["end"] in ends
        # Chronological, non-overlapping, and nothing invented.
        assert units[0]["start"] == flat[0]["start"]
        assert units[-1]["end"] == flat[-1]["end"]
        assert sum(u["word_count"] for u in units) == len(flat)

    def test_ids_are_stable_and_sequential(self):
        units = build_sentence_units(_transcript(
            "one. two. three. four.".split()))
        assert [u["id"] for u in units] == ["S0001", "S0002", "S0003", "S0004"]

    def test_empty_transcript_gives_no_units(self):
        assert build_sentence_units({"segments": []}) == []
        assert build_sentence_units({}) == []
