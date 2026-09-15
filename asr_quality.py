"""Deterministic transcript quality gate for the ASR stage.

Why this exists
---------------
A Hindi/Hinglish stand-up upload transcribed with whisper-``small`` on CPU came
back as an incoherent mixture of Devanagari, English, Cyrillic-looking and
Hangul-looking fragments — the classic small-model hallucination signature. The
meaningful selector then rejected every candidate (correctly: the input was
unusable) and the job died with "Clip detection failed — the AI model did not
return usable clips", which blames the wrong stage and tells the user nothing.

Nothing between the decoder and the selector ever looked at the transcript. This
module is that missing check: it scores a transcript from signals faster-whisper
already computes and throws away (``avg_logprob``, ``no_speech_prob``,
``compression_ratio``, ``language_probability``) plus structural signals that
work for any language, and returns

    {"status": "GOOD" | "RETRY" | "BAD", "score": 0-100,
     "reasons": [...], "metrics": {...}}

Design rules (these are requirements, not preferences)
------------------------------------------------------
* **Quality drives the fallback, never the language.** There is no language
  blacklist anywhere in here. A clean Spanish transcript scores exactly like a
  clean English one; a noisy English transcript retries like any other.
* **Mixing scripts is not a defect.** "मुझे actually ये approach better लगती है"
  is valid Hinglish. Only *unrelated* non-Latin script families piling up (three
  or more, each with real mass) is treated as a hallucination signal, and that
  signal is capped so it can never on its own push a transcript to BAD.
* **Every signal degrades to "absent".** Parakeet, old saved jobs and
  ``--transcript`` inputs carry no decoder diagnostics; those checks simply do
  not fire instead of failing the transcript.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


class TranscriptQualityError(Exception):
    """The transcript is too unreliable to build meaningful clips from."""

    #: What the user is shown. Deliberately about the audio, not the selector.
    USER_MESSAGE = (
        "Transcription quality was too low to reliably create meaningful clips. "
        "The audio may be noisy, heavily code-switched, unsupported, or "
        "difficult to recognize."
    )

    def __init__(self, message: str = "", quality: Optional[Dict[str, Any]] = None):
        super().__init__(message or self.USER_MESSAGE)
        self.quality = quality or {}


# --- script classification --------------------------------------------------
# Grouped into FAMILIES, not Unicode script names: Japanese legitimately mixes
# Han + Hiragana + Katakana, so those are one family. Hangul is its own.

_SCRIPT_RANGES: List[Tuple[int, int, str]] = [
    (0x0041, 0x005A, "latin"), (0x0061, 0x007A, "latin"),
    (0x00C0, 0x024F, "latin"), (0x1E00, 0x1EFF, "latin"),
    (0x0370, 0x03FF, "greek"), (0x1F00, 0x1FFF, "greek"),
    (0x0400, 0x052F, "cyrillic"), (0x2DE0, 0x2DFF, "cyrillic"),
    (0x0530, 0x058F, "armenian"),
    (0x0590, 0x05FF, "hebrew"), (0xFB1D, 0xFB4F, "hebrew"),
    (0x0600, 0x06FF, "arabic"), (0x0750, 0x077F, "arabic"),
    (0x08A0, 0x08FF, "arabic"), (0xFB50, 0xFDFF, "arabic"),
    (0xFE70, 0xFEFF, "arabic"),
    (0x0900, 0x097F, "devanagari"), (0xA8E0, 0xA8FF, "devanagari"),
    (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
    (0x0D80, 0x0DFF, "sinhala"),
    (0x0E00, 0x0E7F, "thai"),
    (0x0E80, 0x0EFF, "lao"),
    (0x0F00, 0x0FFF, "tibetan"),
    (0x1000, 0x109F, "myanmar"),
    (0x10A0, 0x10FF, "georgian"), (0x1C90, 0x1CBF, "georgian"),
    (0x1200, 0x137F, "ethiopic"),
    (0x1780, 0x17FF, "khmer"),
    (0x3040, 0x30FF, "cjk"),          # hiragana + katakana
    (0x3400, 0x4DBF, "cjk"), (0x4E00, 0x9FFF, "cjk"), (0xF900, 0xFAFF, "cjk"),
    (0x1100, 0x11FF, "hangul"), (0x3130, 0x318F, "hangul"),
    (0xA960, 0xA97F, "hangul"), (0xAC00, 0xD7AF, "hangul"),
]

# Scripts written without spaces between words: one "token" there is a whole
# clause, so counting whitespace tokens under-counts speech by ~10x.
_DENSE_SCRIPTS = {"cjk", "thai", "lao", "khmer", "myanmar", "tibetan"}


def script_family(ch: str) -> Optional[str]:
    """The script family of one character, or None for digits/punctuation."""
    code = ord(ch)
    for lo, hi, name in _SCRIPT_RANGES:
        if lo <= code <= hi:
            return name
    return None


def script_profile(text: str) -> Dict[str, float]:
    """Share of letters belonging to each script family (sums to <= 1.0)."""
    counts: Counter = Counter()
    total = 0
    for ch in text or "":
        family = script_family(ch)
        if family:
            counts[family] += 1
            total += 1
    if not total:
        return {}
    return {name: count / total for name, count in counts.items()}


def count_speech_units(text: str) -> float:
    """Approximate spoken-word count that also works for space-less scripts.

    Whitespace tokens for Latin/Devanagari/Cyrillic/… ; for CJK, Thai, Khmer,
    Lao and Burmese a token is a whole clause, so its dense characters are
    counted at ~2 characters per word instead. Without this a perfectly good
    60-second Japanese transcript looks like six "words" and gets routed to the
    silent-video path.
    """
    units = 0.0
    for token in (text or "").split():
        dense = sum(1 for ch in token if script_family(ch) in _DENSE_SCRIPTS)
        if dense >= 2:
            units += dense / 2.0
        else:
            units += 1
    return units


# --- small helpers ----------------------------------------------------------

_PUNCT_STRIP = " \t\r\n.,!?;:\"'“”‘’()[]{}<>«»—–-…·।॥、。！？；：؟٬٫"


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _ratio(part: float, whole: float) -> float:
    return (part / whole) if whole else 0.0


def _scaled(value: float, ok: float, bad: float, max_penalty: float) -> float:
    """Linear penalty: 0 at ``ok``, ``max_penalty`` at ``bad`` and beyond."""
    if bad == ok:
        return 0.0
    position = (value - ok) / (bad - ok)
    return max(0.0, min(1.0, position)) * max_penalty


# --- metric collection ------------------------------------------------------

def _collect_metrics(transcript: Dict[str, Any],
                     duration: Optional[float]) -> Dict[str, Any]:
    segments = [s for s in (transcript.get("segments") or []) if isinstance(s, dict)]

    texts: List[str] = []
    logprobs: List[float] = []
    no_speech: List[float] = []
    compressions: List[float] = []

    total_words = 0
    empty_word_segments = 0
    spoken_seconds = 0.0
    bad_times = 0
    out_of_order = 0
    word_time_errors = 0
    previous_start = None
    last_end = 0.0

    tokens: List[str] = []

    for segment in segments:
        text = str(segment.get("text") or "")
        texts.append(text)

        start = _finite(segment.get("start"))
        end = _finite(segment.get("end"))
        if start is None or end is None or end <= start:
            bad_times += 1
        else:
            spoken_seconds += end - start
            last_end = max(last_end, end)
            if previous_start is not None and start < previous_start - 0.01:
                out_of_order += 1
            previous_start = start

        words = segment.get("words") or []
        if text.strip() and not words:
            empty_word_segments += 1
        total_words += len(words)
        for word in words:
            if not isinstance(word, dict):
                word_time_errors += 1
                continue
            w_start = _finite(word.get("start"))
            w_end = _finite(word.get("end"))
            if w_start is None or w_end is None or w_end < w_start:
                word_time_errors += 1

        for key, bucket in (("avg_logprob", logprobs),
                            ("no_speech_prob", no_speech),
                            ("compression_ratio", compressions)):
            value = _finite(segment.get(key))
            if value is not None:
                bucket.append(value)

        tokens.extend(_norm(text).split())

    full_text = transcript.get("text") or " ".join(t.strip() for t in texts)

    # --- repetition ---------------------------------------------------------
    normalized = [_norm(t) for t in texts if _norm(t)]
    duplicate_ratio = 0.0
    longest_repeat_run = 0
    if len(normalized) >= 6:
        duplicate_ratio = 1.0 - _ratio(len(set(normalized)), len(normalized))
        run = 1
        for i in range(1, len(normalized)):
            run = run + 1 if normalized[i] == normalized[i - 1] else 1
            longest_repeat_run = max(longest_repeat_run, run)

    loop_ratio = 0.0
    sample = tokens[:20000]
    if len(sample) >= 40:
        grams = Counter(tuple(sample[i:i + 4]) for i in range(len(sample) - 3))
        _gram, hits = grams.most_common(1)[0]
        loop_ratio = _ratio(hits * 4, len(sample))

    # --- fragments ----------------------------------------------------------
    stripped = [t.strip(_PUNCT_STRIP) for t in tokens]
    real_tokens = [t for t in stripped if t]
    symbol_ratio = _ratio(len(tokens) - len(real_tokens), len(tokens))
    single_char = sum(
        1 for t in real_tokens
        if len(t) == 1 and script_family(t) not in (None, *_DENSE_SCRIPTS)
    )
    single_char_ratio = _ratio(single_char, len(real_tokens))

    # --- speech rate --------------------------------------------------------
    units = count_speech_units(full_text)
    spoken_minutes = max(spoken_seconds / 60.0, 1e-6)
    media_seconds = _finite(duration) or last_end or spoken_seconds
    media_minutes = max((media_seconds or 0.0) / 60.0, 1e-6)

    profile = script_profile(full_text)
    non_latin = sorted(
        (name for name, share in profile.items()
         if name != "latin" and share >= 0.02),
        key=lambda name: -profile[name],
    )

    return {
        "segments": len(segments),
        "words": total_words,
        "characters": len(full_text),
        "speech_units": round(units, 1),
        "media_seconds": round(media_seconds or 0.0, 2),
        "spoken_seconds": round(spoken_seconds, 2),
        "speech_coverage": round(_ratio(spoken_seconds, media_seconds or 0.0), 3),
        "units_per_spoken_minute": round(units / spoken_minutes, 1),
        "units_per_media_minute": round(units / media_minutes, 1),
        "avg_logprob": None if _mean(logprobs) is None else round(_mean(logprobs), 3),
        "low_logprob_ratio": round(
            _ratio(sum(1 for v in logprobs if v < -1.0), len(logprobs)), 3),
        "avg_no_speech_prob": None if _mean(no_speech) is None else round(_mean(no_speech), 3),
        "high_no_speech_ratio": round(
            _ratio(sum(1 for v in no_speech if v > 0.6), len(no_speech)), 3),
        "max_compression_ratio": round(max(compressions), 3) if compressions else None,
        "high_compression_ratio": round(
            _ratio(sum(1 for v in compressions if v > 2.4), len(compressions)), 3),
        "language": transcript.get("language"),
        "language_probability": _finite(transcript.get("language_probability")),
        "duplicate_segment_ratio": round(duplicate_ratio, 3),
        "longest_repeat_run": longest_repeat_run,
        "loop_ratio": round(loop_ratio, 3),
        "symbol_token_ratio": round(symbol_ratio, 3),
        "single_char_token_ratio": round(single_char_ratio, 3),
        "empty_word_segment_ratio": round(_ratio(empty_word_segments, len(segments)), 3),
        "invalid_timestamp_ratio": round(_ratio(bad_times, len(segments)), 3),
        "out_of_order_segments": out_of_order,
        "word_time_errors": word_time_errors,
        "script_profile": {k: round(v, 3) for k, v in profile.items()},
        "non_latin_scripts": non_latin,
        "diagnostics_available": bool(logprobs or compressions),
    }


# --- scoring ----------------------------------------------------------------

def _score(metrics: Dict[str, Any]) -> Tuple[float, List[str], float]:
    """Return (penalty, reasons, capped_penalty_floor).

    ``capped_penalty_floor`` is the part of the penalty that must never on its
    own drive a BAD verdict (currently only the script-churn signal), expressed
    as the score floor those signals are allowed to pull the transcript down to.
    """
    penalty = 0.0
    reasons: List[str] = []
    soft_penalty = 0.0

    def add(amount: float, reason: str) -> None:
        nonlocal penalty
        if amount <= 0:
            return
        penalty += amount
        reasons.append(reason)

    # 1. Decoder confidence. Whisper's own usable/garbage line is -1.0.
    avg_logprob = metrics.get("avg_logprob")
    if avg_logprob is not None:
        add(_scaled(-avg_logprob, 0.65, 1.15, 42.0),
            f"low decoder confidence (avg_logprob {avg_logprob:.2f})")
    low_ratio = metrics.get("low_logprob_ratio") or 0.0
    if low_ratio > 0.3:
        add(_scaled(low_ratio, 0.3, 0.8, 22.0),
            f"{low_ratio:.0%} of segments below the usable confidence threshold")

    # 2. The decoder itself thought there was no speech.
    high_no_speech = metrics.get("high_no_speech_ratio") or 0.0
    if high_no_speech > 0.35:
        add(_scaled(high_no_speech, 0.35, 0.8, 16.0),
            f"{high_no_speech:.0%} of segments flagged as non-speech")

    # 3. Compression ratio — whisper's own hallucination-loop detector.
    high_compression = metrics.get("high_compression_ratio") or 0.0
    if high_compression > 0.15:
        add(_scaled(high_compression, 0.15, 0.6, 20.0),
            f"{high_compression:.0%} of segments abnormally repetitive")
    max_compression = metrics.get("max_compression_ratio")
    if max_compression is not None and max_compression > 3.0:
        add(_scaled(max_compression, 3.0, 5.0, 12.0),
            f"a segment repeats itself (compression ratio {max_compression:.1f})")

    # 4. Language detection confidence. NOT which language — how sure.
    language_probability = metrics.get("language_probability")
    if language_probability is not None:
        add(_scaled(-language_probability, -0.6, -0.25, 26.0),
            f"language detected with only {language_probability:.0%} confidence")

    # 5. Repetition / hallucination loops.
    duplicate_ratio = metrics.get("duplicate_segment_ratio") or 0.0
    if duplicate_ratio > 0.3:
        add(_scaled(duplicate_ratio, 0.3, 0.8, 25.0),
            f"{duplicate_ratio:.0%} of segments are verbatim duplicates")
    repeat_run = metrics.get("longest_repeat_run") or 0
    if repeat_run >= 4:
        add(_scaled(repeat_run, 4, 12, 18.0),
            f"{repeat_run} identical segments in a row")
    loop_ratio = metrics.get("loop_ratio") or 0.0
    if loop_ratio > 0.2:
        add(_scaled(loop_ratio, 0.2, 0.6, 22.0),
            f"one phrase fills {loop_ratio:.0%} of the transcript")

    # 6. Speech rate against the time actually marked as speech. Real speech
    # runs 100-250 words/min in every language once space-less scripts are
    # counted by character; far below that means the audio was not recognized.
    units_per_minute = metrics.get("units_per_spoken_minute") or 0.0
    if metrics.get("spoken_seconds", 0) >= 30:
        add(_scaled(-units_per_minute, -70.0, -25.0, 24.0),
            f"only {units_per_minute:.0f} words per minute of detected speech")

    # 7. Structural damage: word timestamps are what clip cutting and karaoke
    # captions are built from, so missing/broken ones matter more than text.
    # Weighted heavily on purpose: word timestamps ARE the product here. Clip
    # cutting, karaoke captions and the sentence-unit builder are all built
    # from them, so a transcript missing them is unusable however good its
    # text reads — build_sentence_units returns nothing and the job surfaces
    # as "clip detection failed" again.
    empty_words = metrics.get("empty_word_segment_ratio") or 0.0
    if empty_words > 0.1:
        add(_scaled(empty_words, 0.1, 0.7, 55.0),
            f"{empty_words:.0%} of segments have no word timestamps")
    invalid_times = metrics.get("invalid_timestamp_ratio") or 0.0
    if invalid_times > 0.02:
        add(_scaled(invalid_times, 0.02, 0.25, 45.0),
            f"{invalid_times:.0%} of segments have invalid timestamps")
    if metrics.get("out_of_order_segments"):
        add(12.0, f"{metrics['out_of_order_segments']} segments are out of order")
    if metrics.get("word_time_errors"):
        add(_scaled(_ratio(metrics["word_time_errors"], max(metrics.get("words", 0), 1)),
                    0.01, 0.2, 18.0),
            f"{metrics['word_time_errors']} words have unusable timestamps")

    # 8. Junk fragments.
    single_char = metrics.get("single_char_token_ratio") or 0.0
    if single_char > 0.15:
        add(_scaled(single_char, 0.15, 0.45, 14.0),
            f"{single_char:.0%} of tokens are isolated single characters")
    symbols = metrics.get("symbol_token_ratio") or 0.0
    if symbols > 0.15:
        add(_scaled(symbols, 0.15, 0.5, 10.0),
            f"{symbols:.0%} of tokens are punctuation-only fragments")

    # 9. Script churn — CAPPED, and never fired by code-switching.
    # One non-Latin family beside Latin is exactly what Hinglish, Arabizi,
    # Spanglish and every real bilingual video look like: zero penalty. Three
    # unrelated non-Latin families with real mass in one transcript is not a
    # language, it is a decoder wandering, but even then this signal alone may
    # only reach RETRY (see SOFT_SCORE_FLOOR).
    families = metrics.get("non_latin_scripts") or []
    if len(families) >= 3:
        amount = 14.0 + 6.0 * min(len(families) - 3, 3)
        soft_penalty += amount
        add(amount, "unrelated scripts mixed together: " + ", ".join(families[:5]))
    elif len(families) == 2:
        amount = 6.0
        soft_penalty += amount
        add(amount, f"two unrelated non-Latin scripts present: {', '.join(families)}")

    return penalty, reasons, soft_penalty


#: A transcript whose only complaint is script churn may be told to RETRY but
#: must never be declared BAD — mixed scripts can be perfectly valid speech.
SOFT_SCORE_FLOOR = 45.0

#: avg_logprob below which the deterministic verdict is not trustworthy on its
#: own. Measured, not guessed: whisper-small on the Hinglish stand-up scored
#: -0.75 while producing Devanagari-shaped NONSENSE ("दिसकनेक्टिक फ्लाइट ती आर
#: श्पाइश जेट") that every structural signal likes — no repetition, valid
#: timestamps, 173 words/min, one clean script pair. Structure cannot tell
#: invented words from real ones; only a reader can. Clean transcripts in any
#: language sit at -0.25..-0.45, so this band is narrow by construction.
DEFAULT_LOGPROB_SUSPECT = -0.60


def good_score_threshold() -> float:
    """Score at or above which a transcript is accepted without a retry."""
    return _env_float("ASR_QUALITY_GOOD_SCORE", 70.0)


#: Ordering of the verdicts, best first. Used to pick between two attempts:
#: the STATUS decides before the score does, because a status can be changed
#: by the language check while the structural score stays high.
STATUS_RANK = {"GOOD": 2, "RETRY": 1, "BAD": 0}


def rank_quality(quality: Optional[Dict[str, Any]]) -> Tuple[int, float]:
    quality = quality or {}
    return (STATUS_RANK.get(str(quality.get("status")), 0),
            float(quality.get("score") or 0.0))


def evaluate_transcript(transcript: Optional[Dict[str, Any]],
                        duration: Optional[float] = None) -> Dict[str, Any]:
    """Score a transcript. Returns status/score/reasons/metrics.

    ``duration`` is the media length in seconds when the caller knows it; it is
    only used for coverage reporting, never to fail a transcript on its own.
    """
    transcript = transcript or {}
    segments = [s for s in (transcript.get("segments") or []) if isinstance(s, dict)]
    good_at = good_score_threshold()
    bad_below = _env_float("ASR_QUALITY_BAD_SCORE", 40.0)

    if os.environ.get("ASR_QUALITY_GATE", "1").strip() == "0":
        return {"status": "GOOD", "score": 100.0, "reasons": [],
                "metrics": {"skipped": True}, "skipped": True}

    if not segments:
        return {"status": "BAD", "score": 0.0,
                "reasons": ["the transcript is empty"],
                "metrics": {"segments": 0, "words": 0}}

    metrics = _collect_metrics(transcript, duration)

    if metrics["words"] == 0 and metrics["characters"] == 0:
        return {"status": "BAD", "score": 0.0,
                "reasons": ["no speech was recognized"], "metrics": metrics}

    penalty, reasons, soft_penalty = _score(metrics)
    score = max(0.0, 100.0 - penalty)

    # The soft (script-churn) penalty may never on its own drag a transcript
    # below the floor: the hard signals decide whether something is BAD, mixed
    # scripts only ever add suspicion. ``hard_score`` is the score the same
    # transcript would get with the script signal switched off.
    hard_score = max(0.0, 100.0 - (penalty - soft_penalty))
    score = max(score, min(hard_score, SOFT_SCORE_FLOOR))

    if score >= good_at:
        status = "GOOD"
    elif score >= bad_below:
        status = "RETRY"
    else:
        status = "BAD"

    return {"status": status, "score": round(score, 1),
            "reasons": reasons, "metrics": metrics,
            "uncertain": _is_uncertain(status, metrics)}


def _is_uncertain(status: str, metrics: Dict[str, Any]) -> bool:
    """True when a GOOD verdict rests on structure alone and should be read.

    See DEFAULT_LOGPROB_SUSPECT: a weak model on a language it cannot really
    handle emits fluent-looking invented words. Nothing measurable here can
    separate those from real ones, but the decoder's own confidence says it was
    guessing, which is enough to ask someone who can read.
    """
    if status != "GOOD":
        return False
    suspect_at = _env_float("ASR_QUALITY_LOGPROB_SUSPECT", DEFAULT_LOGPROB_SUSPECT)
    avg_logprob = metrics.get("avg_logprob")
    return avg_logprob is not None and avg_logprob < suspect_at


def format_quality_line(quality: Dict[str, Any], prefix: str = "Transcript quality") -> str:
    quality = quality or {}
    return (f"🧪 {prefix}: {quality.get('status', '?')} "
            f"score={quality.get('score', 0)}")


def quality_reason_lines(quality: Dict[str, Any], limit: int = 4) -> List[str]:
    reasons = (quality or {}).get("reasons") or []
    return [f"   Reasons: {reason}" if index == 0 else f"            {reason}"
            for index, reason in enumerate(reasons[:limit])]


# --- optional second opinion ------------------------------------------------

_GEMINI_CHECK_PROMPT = """You are checking whether a machine transcript is
usable, NOT whether the speech is interesting.

The transcript may be in ANY language, and may freely mix languages and scripts
within a sentence (for example Hindi written in Devanagari mixed with English
words). Code-switching like that is NORMAL SPEECH and must be judged USABLE.
Never mark a transcript unusable for being in a language you did not expect,
for switching language mid-sentence, or for informal or regional wording.

Mark it UNUSABLE when the text is not real language: words that do not exist in
the language they are written in, phonetic gibberish that merely LOOKS like the
script, fragments of unrelated languages strung together, endlessly repeated
phrases, or strings no human would have said.

The important case: a weak speech model transcribing a language it cannot
handle produces fluent-looking text made of INVENTED words. Read the sample as
a native speaker would and say whether these are actual words forming actual
sentences.

TRANSCRIPT SAMPLE:

{sample}
"""


def language_check_mode() -> str:
    """'auto' (default) | '1' (always) | '0' (never).

    'auto' asks a reader ONLY when the deterministic signals cannot decide —
    the grey band of DEFAULT_LOGPROB_SUSPECT, plus the last reprieve before a
    job is refused. A clean transcript never reaches it, so the normal job pays
    nothing; both cases that do reach it were going to be wrong otherwise.
    """
    raw = os.environ.get("ASR_QUALITY_GEMINI", "auto").strip().lower()
    return raw if raw in ("0", "1", "auto") else "auto"


def gemini_second_opinion(transcript: Dict[str, Any],
                          api_key: Optional[str] = None,
                          model_name: Optional[str] = None) -> Optional[bool]:
    """Is this text real language? True/False, or None when unavailable.

    Deterministic signals measure structure: repetition, timing, speech rate,
    the decoder's confidence. None of them can tell an invented Devanagari word
    from a real one, which is exactly how the Hinglish job produced a
    structurally perfect transcript of nothing. This is the only check that can,
    and it is called only when structure has already run out of answers.
    """
    if language_check_mode() == "0":
        return None

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    text = str(transcript.get("text") or "")
    if len(text) < 80:
        return None
    sample = text[:1500] + ("\n…\n" + text[-1500:] if len(text) > 3500 else "")

    try:
        from google import genai
        from google.genai import types as genai_types
        from pydantic import BaseModel

        class _Verdict(BaseModel):
            usable: bool
            explanation: str

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name or os.environ.get("GEMINI_MODEL") or "gemini-3.1-flash-lite",
            contents=_GEMINI_CHECK_PROMPT.format(sample=sample),
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=_Verdict,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if parsed is None:
            return None
        print(f"🧪 Gemini transcript check: "
              f"{'usable' if parsed.usable else 'unusable'} — {parsed.explanation}")
        return bool(parsed.usable)
    except Exception as exc:  # never let the optional check fail a job
        print(f"⚠️ Transcript sanity check unavailable ({type(exc).__name__}: {exc})")
        return None
