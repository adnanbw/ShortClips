"""The transcript quality gate: what must pass, what must retry, what must fail.

Written against the real failure it exists for — a Hindi/Hinglish stand-up
upload that whisper-small turned into an incoherent mixture of Devanagari,
English and unrelated scripts, which the clip selector then (correctly)
rejected, producing "Clip detection failed" and blaming the wrong stage.

The two rules these tests defend above all others:
  * quality drives the retry, never the language — a clean Spanish transcript
    must behave exactly like a clean English one;
  * mixing scripts is not a defect — Hinglish is normal speech.
"""
import random

import pytest

import asr_quality
from asr_quality import count_speech_units, evaluate_transcript, script_profile


# --- transcript builders ----------------------------------------------------

def _segment(index, text, seconds=5.0, words=None, **diagnostics):
    start = index * seconds
    tokens = words if words is not None else text.split()
    step = seconds / max(len(tokens), 1)
    payload = {
        "start": start,
        "end": start + seconds,
        "text": text,
        "words": [
            {"word": " " + token,
             "start": start + i * step,
             "end": start + (i + 1) * step}
            for i, token in enumerate(tokens)
        ],
    }
    defaults = {"avg_logprob": -0.28, "no_speech_prob": 0.02,
                "compression_ratio": 1.6}
    defaults.update(diagnostics)
    payload.update(defaults)
    return payload


def _transcript(lines, language="en", language_probability=0.99,
                seconds=5.0, **diagnostics):
    segments = [_segment(i, line, seconds, **diagnostics)
                for i, line in enumerate(lines)]
    return {
        "text": " ".join(lines),
        "language": language,
        "language_probability": language_probability,
        "segments": segments,
    }


def _repeat(line, times):
    return [line] * times


def _lines(vocabulary, count=20, per_line=14, joiner=" ", end=".", seed=7):
    """Distinct, natural-looking sentences.

    Every segment must differ: a fixture that repeats one sentence twenty
    times is exactly what the repetition signal is built to catch, so reusing
    one line would make the clean-transcript tests test the wrong thing.
    """
    rng = random.Random(seed)
    return [
        joiner.join(rng.choice(vocabulary) for _ in range(per_line)) + end
        for _ in range(count)
    ]


ENGLISH = _lines(
    "the thing nobody tells you about starting a business is that first "
    "year mostly learning what does not work in your market".split())

SPANISH = _lines(
    "lo que nadie te cuenta sobre montar una empresa es el primer año "
    "consiste en aprender qué no funciona en tu mercado".split())

# Real Hinglish: Devanagari and Latin inside the same sentence.
HINGLISH = _lines(
    "मुझे actually ये approach better लगती है क्योंकि इसमें आपको पहले से "
    "planning करनी पड़ती और वो हमेशा help करता".split())

# Written without spaces between words — str.split() sees one "word" a line.
JAPANESE = _lines(
    ["実は", "この", "方法", "のほうが", "いいと", "思います", "最初に",
     "計画を", "立てる", "必要が", "あるから", "です"],
    joiner="", end="。")

ARABIC = _lines(
    "ما لا يخبرك به أحد عن بدء مشروع تجاري هو أن السنة الأولى تدور حول "
    "تعلم الذي ينجح في السوق".split())


# --- 1-3: clean transcripts pass, whatever the language ---------------------

class TestCleanTranscriptsPass:
    def test_good_english_is_good(self):
        result = evaluate_transcript(_transcript(ENGLISH), duration=100.0)
        assert result["status"] == "GOOD", result["reasons"]

    def test_good_spanish_is_good_and_not_penalised_for_being_spanish(self):
        spanish = evaluate_transcript(_transcript(SPANISH, language="es"), 100.0)
        english = evaluate_transcript(_transcript(ENGLISH), 100.0)
        assert spanish["status"] == "GOOD", spanish["reasons"]
        assert spanish["score"] == pytest.approx(english["score"], abs=5)

    def test_hindi_english_code_switching_is_not_a_failure(self):
        result = evaluate_transcript(_transcript(HINGLISH, language="hi"), 100.0)
        assert result["status"] == "GOOD", result["reasons"]

    def test_arabic_is_good(self):
        result = evaluate_transcript(_transcript(ARABIC, language="ar"), 100.0)
        assert result["status"] == "GOOD", result["reasons"]

    def test_japanese_without_spaces_is_good(self):
        # One whitespace token per segment: str.split() would call this
        # 20 "words" in 100 seconds and condemn it.
        result = evaluate_transcript(_transcript(JAPANESE, language="ja"), 100.0)
        assert result["status"] == "GOOD", result["reasons"]

    def test_transcript_without_decoder_diagnostics_still_passes(self):
        # Parakeet, saved jobs and --transcript inputs carry no avg_logprob.
        transcript = _transcript(ENGLISH)
        for segment in transcript["segments"]:
            for key in ("avg_logprob", "no_speech_prob", "compression_ratio"):
                segment.pop(key)
        transcript.pop("language_probability")
        result = evaluate_transcript(transcript, 100.0)
        assert result["status"] == "GOOD", result["reasons"]
        assert result["metrics"]["diagnostics_available"] is False


# --- 4-5: garbage must not reach the selector -------------------------------

class TestGarbageIsCaught:
    def test_the_hinglish_failure_signature_is_not_good(self):
        """Low confidence + unrelated scripts + junk fragments, as observed."""
        lines = _lines(
            "मुझे बहुत ですが the пример 니다 и что это الشيء no сказать "
            "pressure क a".split(), count=24, per_line=6, seed=3)
        result = evaluate_transcript(
            _transcript(lines, language="hi", language_probability=0.38,
                        avg_logprob=-1.15, compression_ratio=2.9,
                        no_speech_prob=0.55),
            duration=120.0)
        assert result["status"] in ("RETRY", "BAD"), result
        assert result["score"] < 70

    def test_hallucination_loop_is_caught(self):
        result = evaluate_transcript(
            _transcript(_repeat("thanks for watching", 30),
                        avg_logprob=-0.95, compression_ratio=3.4),
            duration=150.0)
        assert result["status"] in ("RETRY", "BAD"), result
        assert any("duplicate" in r or "repeat" in r or "phrase" in r
                   for r in result["reasons"]), result["reasons"]

    def test_low_decoder_confidence_alone_triggers_a_retry(self):
        result = evaluate_transcript(
            _transcript(ENGLISH, avg_logprob=-1.25), duration=100.0)
        assert result["status"] in ("RETRY", "BAD"), result

    def test_empty_transcript_is_bad(self):
        assert evaluate_transcript({"segments": []})["status"] == "BAD"
        assert evaluate_transcript(None)["status"] == "BAD"
        assert evaluate_transcript({})["status"] == "BAD"

    def test_transcript_with_no_recognized_speech_is_bad(self):
        result = evaluate_transcript(
            {"text": "", "language": "en",
             "segments": [{"start": 0, "end": 60, "text": "", "words": []}]})
        assert result["status"] == "BAD"

    def test_broken_timestamps_are_caught(self):
        transcript = _transcript(ENGLISH)
        for segment in transcript["segments"][:8]:
            segment["end"] = segment["start"] - 1.0
        result = evaluate_transcript(transcript, 100.0)
        assert result["status"] in ("RETRY", "BAD"), result

    def test_missing_word_timestamps_are_caught(self):
        # Clip cutting and karaoke captions are built from word times.
        transcript = _transcript(ENGLISH)
        for segment in transcript["segments"]:
            segment["words"] = []
        result = evaluate_transcript(transcript, 100.0)
        assert result["status"] in ("RETRY", "BAD"), result


# --- the anti-blacklist rules -----------------------------------------------

class TestNoLanguagePenalty:
    @pytest.mark.parametrize("language", ["en", "hi", "es", "ja", "ar", "ko",
                                          "zh", "ru", "de", "pt", "fr"])
    def test_language_code_never_changes_the_verdict(self, language):
        assert evaluate_transcript(
            _transcript(ENGLISH, language=language), 100.0)["status"] == "GOOD"

    def test_two_scripts_alone_never_reach_bad(self):
        """Devanagari + Latin is Hinglish, not damage."""
        result = evaluate_transcript(_transcript(HINGLISH, language="hi"), 100.0)
        assert result["status"] == "GOOD"

    def test_script_churn_alone_can_only_reach_retry(self):
        """Even four unrelated scripts, with every other signal clean, must
        not be BAD: the gate suspects, the stronger model decides."""
        lines = _lines(
            "мне ですが actually इस approach में 니다 planning करनी पड़ती है "
            "और वो हमेशा help करता क्योंकि".split(), seed=11)
        result = evaluate_transcript(_transcript(lines, language="hi"), 100.0)
        assert result["status"] != "BAD", result
        assert result["score"] >= asr_quality.SOFT_SCORE_FLOOR


# --- switches ---------------------------------------------------------------

class TestSwitches:
    def test_gate_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("ASR_QUALITY_GATE", "0")
        result = evaluate_transcript({"segments": []})
        assert result["status"] == "GOOD" and result["skipped"] is True

    def test_thresholds_are_configurable(self, monkeypatch):
        suspicious = _transcript(ENGLISH, avg_logprob=-1.05)
        assert evaluate_transcript(suspicious, 100.0)["status"] == "RETRY"
        monkeypatch.setenv("ASR_QUALITY_GOOD_SCORE", "10")
        assert evaluate_transcript(suspicious, 100.0)["status"] == "GOOD"


# --- script-aware counting --------------------------------------------------

class TestSpeechUnits:
    def test_latin_counts_words(self):
        assert count_speech_units("one two three four") == 4

    def test_space_less_scripts_count_characters(self):
        # 8 kanji/kana ≈ 4 words, not 1.
        assert count_speech_units("実はこの方法がいい") == pytest.approx(4.5, abs=0.5)

    def test_devanagari_is_word_spaced_like_latin(self):
        assert count_speech_units("मुझे ये approach better लगती है") == 6

    def test_empty_is_zero(self):
        assert count_speech_units("") == 0
        assert count_speech_units(None) == 0


class TestScriptProfile:
    def test_hinglish_reports_both_scripts(self):
        profile = script_profile("मुझे actually ये approach better लगती है")
        assert profile["devanagari"] > 0.2 and profile["latin"] > 0.2

    def test_japanese_is_one_family(self):
        # Kana + kanji must not look like two unrelated scripts.
        profile = script_profile("実はこの方法がいいと思います")
        assert set(profile) == {"cjk"}

    def test_digits_and_punctuation_are_not_scripts(self):
        assert script_profile("123 ... !?") == {}
