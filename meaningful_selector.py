import json
import os
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Sentence building
# ---------------------------------------------------------------------------

_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "st.", "vs.", "etc.", "e.g.", "i.e.", "a.m.", "p.m.",
    "u.s.", "u.k.", "no.", "fig.", "inc.", "ltd.", "co.",
    # Common outside English too; an abbreviation is not a sentence end in any
    # language. Deliberately short — the pause fallback catches the rest.
    "sra.", "srta.", "ud.", "uds.", "núm.", "pág.", "ej.",
    "m.", "mme.", "mlle.", "hr.", "fr.", "bzw.", "z.b.", "u.a.", "ecc.",
}

#: Sentence terminators across the scripts the product supports. ASR emits the
#: punctuation of the language it transcribed, so a Latin-only list silently
#: turned a whole Hindi or Japanese transcript into one giant "sentence".
SENTENCE_TERMINATORS = (
    ".", "?", "!",              # Latin / Cyrillic / Greek / Devanagari-in-Latin
    "।", "॥",         # । ॥  danda, double danda (Indic)
    "。", "？", "！",  # 。？！ CJK fullwidth
    "．", "｡",         # ．｡ fullwidth / halfwidth ideographic stop
    "؟", "۔",         # ؟ ۔  Arabic question mark, Urdu full stop
    "‼", "⁇", "⁈", "⁉",  # ‼ ⁇ ⁈ ⁉
    "։", "՜", "՞",  # ։ ՜ ՞ Armenian
    "።",                   # ። Ethiopic full stop
    "។",                   # ។ Khmer
    "ฯ",                   # ฯ Thai paiyannoi
    "ۖ",                   # Quranic stop (appears in Arabic ASR output)
)

#: Trailing characters that can sit AFTER the terminator without cancelling it
#: (closing quotes and brackets, including the CJK and Arabic shapes).
_CLOSERS = "\"'”’»›)]}、」』）］｝”» "


# ---------------------------------------------------------------------------
# Clip duration: a PREFERENCE that is asked for, and a HARD band that is
# enforced. Those used to be the same number, and that is what threw away the
# best moment of a real job: the candidate's punchline landed at 107s, so the
# only repair that could include it measured 107s, and a bare `> 90.0` check
# deleted it — leaving the version cut off mid-sentence, which the critic then
# correctly rejected as incomplete. A complete 100-second clip beats a
# truncated 75-second one, every time.
#
# 90s was never a platform limit: YouTube Shorts allows 3 minutes, TikTok 10,
# Reels 3. It was an editorial preference enforced as a gate. It stays a
# preference — every prompt still asks for 25-60s and says so — and stops
# being a gate.
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


MIN_CLIP_SECONDS = _env_float("MEANINGFUL_MIN_CLIP_SECONDS", 15.0)
MAX_CLIP_SECONDS = _env_float("MEANINGFUL_MAX_CLIP_SECONDS", 110.0)

#: What the prompts ask for. Never enforced in code.
PREFERRED_MIN_SECONDS = 25.0
PREFERRED_MAX_SECONDS = 60.0

#: Windows must be able to CONTAIN a maximum-length candidate plus surrounding
#: material, or a long moment straddles two windows and is invisible in both.
#: Tied to MAX_CLIP_SECONDS so raising one cannot silently break the other.
CANDIDATE_WINDOW_SECONDS = max(210.0, MAX_CLIP_SECONDS * 2.2)
CANDIDATE_OVERLAP_SECONDS = MAX_CLIP_SECONDS


def duration_rule_text() -> str:
    """The one sentence every prompt uses to state the duration rule."""
    return (
        f"{MIN_CLIP_SECONDS:.0f}-{MAX_CLIP_SECONDS:.0f} seconds is allowed.\n"
        f"{PREFERRED_MIN_SECONDS:.0f}-{PREFERRED_MAX_SECONDS:.0f} seconds is "
        f"strongly preferred — pick the shorter version whenever it is just as "
        f"complete. But NEVER cut off a payoff to hit that range: a complete "
        f"clip at {PREFERRED_MAX_SECONDS + 40:.0f} seconds is better than a "
        f"truncated one at {PREFERRED_MAX_SECONDS:.0f}."
    )


def _clean_word(value: Any) -> str:
    return str(value or "").strip()


def _is_abbreviation(word: str) -> bool:
    w = word.lower().strip("\"'()[]{}")

    if w in _ABBREVIATIONS:
        return True

    # A single Latin letter plus a period ("J.", "e.") is an initial, not an
    # end of sentence. Restricted to Latin on purpose: a single Devanagari or
    # Cyrillic letter followed by a stop is not the same convention.
    if re.fullmatch(r"[a-z]\.", w):
        return True

    return False


def _ends_sentence(word: str) -> bool:
    if not word:
        return False

    if _is_abbreviation(word):
        return False

    cleaned = word.rstrip(_CLOSERS)

    return cleaned.endswith(SENTENCE_TERMINATORS)


def _punctuation_is_scarce(words: List[Dict[str, Any]]) -> bool:
    """True when the ASR barely punctuated this transcript.

    Punctuation quality varies enormously by language and model: English from
    whisper is fully punctuated, while Hindi, Thai or a small model on noisy
    audio can return almost none. When that happens the punctuation rule can
    never fire and everything has to come from pauses, so the pause rule is
    loosened rather than letting the whole video become one sentence unit.
    """
    if len(words) < 40:
        return False

    terminators = sum(1 for w in words if _ends_sentence(w["word"]))

    # Natural speech ends a sentence roughly every 10-20 words in every
    # language; fewer than one per 25 means the punctuation is not there.
    return terminators < len(words) / 25.0


def _words_to_text(words: List[Dict[str, Any]]) -> str:
    return " ".join(
        _clean_word(w.get("word"))
        for w in words
        if _clean_word(w.get("word"))
    ).strip()


def build_sentence_units(
    transcript: Dict[str, Any],
    pause_split_seconds: float = 1.20,
    emergency_max_seconds: float = 25.0,
) -> List[Dict[str, Any]]:

    flat_words: List[Dict[str, Any]] = []

    for segment_index, segment in enumerate(
        transcript.get("segments", [])
    ):
        for word in segment.get("words", []):
            text = _clean_word(word.get("word"))

            if not text:
                continue

            try:
                start = float(word.get("start"))
                end = float(word.get("end"))
            except (TypeError, ValueError):
                continue

            flat_words.append({
                "word": text,
                "start": start,
                "end": end,
                "segment_index": segment_index,
            })

    if not flat_words:
        return []

    # Language-independent adaptation: with almost no punctuation to go on,
    # pauses become the primary boundary signal instead of the fallback.
    scarce_punctuation = _punctuation_is_scarce(flat_words)

    if scarce_punctuation:
        pause_split_seconds = min(pause_split_seconds, 0.70)
        min_words_for_pause_split = 5
        min_seconds_for_pause_split = 2.0
        print(
            "Sentence units: transcript is barely punctuated — "
            "splitting on pauses "
            f"(>= {pause_split_seconds:.2f}s)."
        )
    else:
        min_words_for_pause_split = 8
        min_seconds_for_pause_split = 3.0

    sentences: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []

    def flush(reason: str) -> None:
        nonlocal current

        if not current:
            return

        start = float(current[0]["start"])
        end = float(current[-1]["end"])
        text = _words_to_text(current)

        if not text:
            current = []
            return

        sentence_id = f"S{len(sentences) + 1:04d}"

        sentences.append({
            "id": sentence_id,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "text": text,
            "word_count": len(current),
            "boundary_reason": reason,
            "first_segment": current[0]["segment_index"],
            "last_segment": current[-1]["segment_index"],
        })

        current = []

    for index, word in enumerate(flat_words):

        current.append(word)

        next_word: Optional[Dict[str, Any]] = (
            flat_words[index + 1]
            if index + 1 < len(flat_words)
            else None
        )

        current_start = float(current[0]["start"])
        current_end = float(word["end"])
        duration = current_end - current_start

        pause_after = 0.0

        if next_word is not None:
            pause_after = max(
                0.0,
                float(next_word["start"]) - current_end
            )

        if _ends_sentence(word["word"]):
            flush("punctuation")
            continue

        if (
            pause_after >= pause_split_seconds
            and len(current) >= min_words_for_pause_split
            and duration >= min_seconds_for_pause_split
        ):
            flush(f"pause_{pause_after:.2f}s")
            continue

        if (
            duration >= emergency_max_seconds
            and pause_after >= 0.40
        ):
            flush("emergency_long_utterance")
            continue

        # Hard ceiling. Without it, continuous speech that the ASR neither
        # punctuated nor left a 0.40s gap in (dense Hindi/Japanese delivery,
        # a live stand-up set over audience noise) produced one sentence unit
        # longer than the 90s clip limit, which no candidate could ever use.
        # Always flushed on a word boundary, so word timestamps stay intact.
        if duration >= emergency_max_seconds * 1.6:
            flush("emergency_hard_ceiling")
            continue

    flush("end_of_transcript")

    return sentences


# ---------------------------------------------------------------------------
# Sentence utilities
# ---------------------------------------------------------------------------

def sentence_by_id(
    sentences: List[Dict[str, Any]],
    sentence_id: str
) -> Optional[Dict[str, Any]]:

    for sentence in sentences:
        if sentence.get("id") == sentence_id:
            return sentence

    return None


def sentence_at_time(
    sentences: List[Dict[str, Any]],
    timestamp: float
) -> Optional[Dict[str, Any]]:

    if not sentences:
        return None

    timestamp = float(timestamp)

    for sentence in sentences:
        if sentence["start"] <= timestamp <= sentence["end"]:
            return sentence

    return min(
        sentences,
        key=lambda s: min(
            abs(timestamp - s["start"]),
            abs(timestamp - s["end"])
        )
    )


def get_sentence_range(
    sentences: List[Dict[str, Any]],
    start_id: str,
    end_id: str,
) -> List[Dict[str, Any]]:

    id_to_index = {
        sentence["id"]: index
        for index, sentence in enumerate(sentences)
    }

    if start_id not in id_to_index:
        raise ValueError(
            f"Unknown start sentence: {start_id}"
        )

    if end_id not in id_to_index:
        raise ValueError(
            f"Unknown end sentence: {end_id}"
        )

    start_index = id_to_index[start_id]
    end_index = id_to_index[end_id]

    if end_index < start_index:
        raise ValueError(
            f"End sentence {end_id} occurs before {start_id}"
        )

    return sentences[start_index:end_index + 1]


def sentence_range_timestamps(
    sentences: List[Dict[str, Any]],
    start_id: str,
    end_id: str,
    lead_seconds: float = 0.15,
    tail_seconds: float = 0.25,
) -> Dict[str, float]:

    selected = get_sentence_range(
        sentences,
        start_id,
        end_id
    )

    start = max(
        0.0,
        float(selected[0]["start"]) - lead_seconds
    )

    end = float(selected[-1]["end"]) + tail_seconds

    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(end - start, 3),
    }


def format_sentences_for_ai(
    sentences: List[Dict[str, Any]]
) -> str:

    lines = []

    for sentence in sentences:
        lines.append(
            f'{sentence["id"]} '
            f'[{sentence["start"]:.2f}-{sentence["end"]:.2f}] '
            f'{sentence["text"]}'
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Candidate windows
# ---------------------------------------------------------------------------

def build_candidate_windows(
    sentences: List[Dict[str, Any]],
    window_seconds: float = CANDIDATE_WINDOW_SECONDS,
    overlap_seconds: float = CANDIDATE_OVERLAP_SECONDS,
) -> List[Dict[str, Any]]:
    """
    Split a long transcript into overlapping semantic windows.

    The overlap is MAX_CLIP_SECONDS on purpose: a maximum-length candidate has
    to fit entirely inside at least one window, with surrounding material,
    even when the idea happens near a window edge. An overlap smaller than the
    longest allowed clip makes long moments straddle two windows and be
    proposable in neither.
    """

    if not sentences:
        return []

    windows = []

    i = 0
    window_number = 1
    n = len(sentences)

    while i < n:

        start_time = float(sentences[i]["start"])

        j = i

        while (
            j + 1 < n
            and float(sentences[j + 1]["end"]) - start_time
            <= window_seconds
        ):
            j += 1

        selected = sentences[i:j + 1]

        windows.append({
            "id": f"W{window_number:03d}",
            "start_sentence": selected[0]["id"],
            "end_sentence": selected[-1]["id"],
            "start": selected[0]["start"],
            "end": selected[-1]["end"],
            "sentences": selected,
        })

        window_number += 1

        if j >= n - 1:
            break

        target_start_time = (
            float(sentences[j]["end"])
            - overlap_seconds
        )

        k = i + 1

        while (
            k <= j
            and float(sentences[k]["start"])
            < target_start_time
        ):
            k += 1

        i = max(i + 1, k)

    return windows


# ---------------------------------------------------------------------------
# Shared multilingual instructions
# ---------------------------------------------------------------------------
#
# Every stage of the meaningful pipeline reads the SAME block, so the language
# policy cannot drift between the finder, the critic, the opening guard and the
# metadata writer. It deliberately says nothing about WHICH languages are good:
# the semantic quality rules are identical in every language, and the only new
# instruction is "do not mistake a language you did not expect for a defect".

# ---------------------------------------------------------------------------
# What every judge in this pipeline is ACTUALLY looking at.
#
# The critic prompt used to open with "You are seeing EXACTLY AND ONLY what the
# eventual viewer will hear", which is false in two ways that cost real clips:
# the viewer hears SPEECH, not an imperfect transcript of it, and the viewer
# also SEES. Measured on one Hinglish job, three of five rejections were the
# critic marking a clip down for ASR noise -
#   "contains significant transcription errors ('पुगली उलूश शुष')"
#   "fragmented, repetitive, and nonsensical"
# - and a fourth was it demanding to be told who the speaker was, while she
# stood on a stage holding a microphone in front of a laughing audience.
#
# Shared so the rule cannot drift between the critic, the repairer and the
# opening guard, which all judge the same clip from the same transcript.
# ---------------------------------------------------------------------------

WHAT_YOU_ARE_JUDGING = """
WHAT YOU ARE ACTUALLY LOOKING AT:

This transcript was produced by automatic speech recognition. The viewer will
never read it — they will WATCH and HEAR the clip. Two consequences, and
neither is optional:

1. MISRECOGNISED WORDS ARE NOT THE SPEAKER'S WORDS.
   ASR mishears names, slang, fast speech and code-switched words, and it is
   worst on exactly the audio that is hardest to transcribe. Words that look
   invented, misspelled or grammatically impossible are TRANSCRIPTION NOISE.
   The audience hears the real word and never notices.
   Judge what the speaker plainly MEANT. Never lower standalone_score or
   completeness_score because the text is garbled, and never call a clip
   incoherent when a listener would have followed it perfectly well. If a
   stretch is so damaged that you cannot tell even roughly what was meant,
   say so in your reason — but a handful of mangled words inside an otherwise
   followable passage is a defect in the TRANSCRIPT, not in the clip.

2. THE VIEWER CAN SEE.
   They see the speaker, their face and their delivery, the setting, and — for
   a performance — the stage and the audience reacting. Nothing visible needs
   to be said out loud. Do NOT require the clip to state who the speaker is,
   what they do for a living, their name, where they are, that an audience is
   present, or that this is a comedy set, a lecture or an interview.
   "Who is this person?" is NEVER a missing-context failure.

   This does not relax anything for what the viewer CANNOT see. A pronoun with
   no antecedent — "she told me...", "that's why he did it", "this is the
   thing I mentioned" — is still a failure, because no picture tells the
   viewer who or what is meant. The distinction is simple: the SPEAKER and
   their surroundings are visible; everyone and everything they refer to is
   not.
"""


MULTILINGUAL_RULES = """
LANGUAGE RULES (identical standards in every language):

- The transcript may be in ANY language, not only English.
- Read and judge it in its ORIGINAL language. Do not translate it, and never
  reject, downgrade or doubt a clip merely because it is not in English.
- Speakers frequently switch language mid-sentence (for example Hindi and
  English inside one line: "मुझे actually ये approach better लगती है"). That is
  normal speech, not a transcription error, and never by itself a reason to
  reject a clip.
- Sentences may end with punctuation other than . ? ! (for example । 。 ？ ！ ؟),
  or with none at all, and some languages are written without spaces between
  words. Judge the meaning, not the typography.
- Apply exactly the same standalone-meaning and completeness bar you would
  apply in English. The bar does not move with the language.
""".strip()


# ---------------------------------------------------------------------------
# Candidate finder prompt
# ---------------------------------------------------------------------------

CANDIDATE_FINDER_PROMPT = """
You are selecting meaningful standalone moments from a long spoken video for
Instagram Reels and YouTube Shorts.

This is NOT a generic "find viral quotes" task.

Your first priority is SEMANTIC COMPLETENESS.

A clip must feel like a self-contained mini-piece of content to a viewer who
has NEVER watched the source video.

You receive transcript sentences with immutable sentence IDs.

You MUST choose boundaries using those sentence IDs only.

DO NOT invent timestamps.
DO NOT cut inside a sentence.

You may return ZERO candidates if this window contains nothing genuinely worth
publishing.

You may return up to 4 candidates from this window.

IDEAL CLIP:
- understandable without earlier video context
- one clear idea, story, lesson, argument, explanation, or payoff
- begins where the necessary setup begins
- ends after the actual conclusion/payoff
- contains useful, interesting, surprising, emotional, educational, funny,
  insightful, or strongly opinionated material
- uses the shortest range that preserves the complete idea
{duration_rule}

VERY IMPORTANT BOUNDARY RULES:

BAD START:
"And so one example is smoking..."

That is bad if the viewer does not know what smoking is an example OF.
Include the earlier setup or do not select it.

BAD START:
"The second reason is..."

That is bad unless the previous structure is unnecessary to understand it.

BAD START:
"And she said to me..."

That is bad if the viewer needs the earlier story to know who "she" is and why
the conversation matters.

BAD END:
"...and the reason why..."

Obviously unfinished.

BAD END:
"...and what this means is..."

Obviously unfinished.

BAD END:
"...turn these ideas into..."

Obviously unfinished.

The ending must contain the answer, resolution, conclusion, punchline, result,
or completed thought.

Do NOT prefer a dramatic opening if obtaining that opening destroys context.

Do NOT select:
- generic channel intros
- subscribe/promotional sections
- sponsor filler
- housekeeping
- repetitive filler
- incomplete list fragments
- examples whose parent idea is missing
- moments that only become meaningful because of text before or after them

SCORING:

standalone_score:
0-100. Could a stranger understand this with zero prior context?

completeness_score:
0-100. Does the selected range include the required setup AND final payoff?

content_score:
0-100. Is the actual idea worth watching?

hook_score:
0-100. After context is preserved, how strong is the opening?

The scores must be honest. Do not inflate them just to produce candidates.

{multilingual_rules}

TRANSCRIPT LANGUAGE:
{language}

WINDOW:
{window_id}

TRANSCRIPT:

{transcript}

Return candidates only from the supplied sentence IDs.
"""


# ---------------------------------------------------------------------------
# Over-long candidate repair
# ---------------------------------------------------------------------------
#
# The Candidate Finder is asked for complete ideas, and a complete idea is
# sometimes longer than a Short. The old code simply dropped those proposals:
# on the Hinglish run that discarded 3 of 4 (110.9s, 95.3s, 117.2s) and left
# the critic a single candidate to judge. But a 110-second idea very often
# CONTAINS a complete 60-second one, and the model that proposed it is the
# cheapest thing that can find it.
#
# The repair is strictly a narrowing: it may only return sentence IDs from
# inside the original proposal, and the result is re-validated locally against
# the real sentence timestamps. No clamping, no arbitrary word trimming, no
# invented times — the boundaries stay on sentence units exactly like every
# other boundary in this pipeline.

OVERLONG_REPAIR_PROMPT = """
A proposed short-form clip is TOO LONG to publish. Your job is to find the best
COMPLETE, SELF-CONTAINED SHORTER SECTION inside it — or to say that none exists.

You may ONLY choose from the sentence IDs listed below.
You may NOT choose anything outside this range.
You may NOT cut inside a sentence.
You may NOT invent timestamps.

The section you return MUST:
- last between {minimum_seconds:.0f} and {maximum_seconds:.0f} seconds
- make complete sense to someone who never watched the source video
- begin where the setup that section needs actually begins
- end after its own payoff, punchline, conclusion, answer or result
- contain ONE coherent idea, story or bit
- be the SHORTEST version that still preserves the whole meaning

Do NOT return a section that merely starts well and stops mid-thought just to
fit the time limit. An incomplete section is worse than no section.

If the material only works as a whole, or every shorter section would be
missing its setup or its ending, return action REJECT.

{multilingual_rules}

TOPIC OF THE ORIGINAL PROPOSAL:
{topic}

ORIGINAL PROPOSAL: {original_start} -> {original_end} ({original_duration:.1f} seconds)

SENTENCES YOU MAY CHOOSE FROM:

{transcript}
"""


def _repair_schema():
    from pydantic import BaseModel
    from typing import Literal, Optional as TypingOptional

    class OverlongRepair(BaseModel):
        action: Literal["REPAIR", "REJECT"]
        new_start_sentence: TypingOptional[str] = None
        new_end_sentence: TypingOptional[str] = None
        reason: str = ""

    return OverlongRepair


def repair_overlong_candidate(
    sentences: List[Dict[str, Any]],
    start_id: str,
    end_id: str,
    topic: str = "",
    client: Any = None,
    model_name: Optional[str] = None,
    minimum_seconds: float = MIN_CLIP_SECONDS,
    maximum_seconds: float = MAX_CLIP_SECONDS,
) -> Optional[Dict[str, Any]]:
    """Narrow an over-long proposal to a complete section, or return None.

    Everything the model returns is verified here against the real sentence
    list: the IDs must exist, must lie INSIDE the original proposal, must be in
    order, and the resulting duration must land in the allowed band. A model
    that ignores any of that is treated as a rejection, never trusted.
    """
    if os.environ.get("MEANINGFUL_REPAIR_OVERLONG", "1").strip() == "0":
        return None

    index_by_id = {str(s.get("id")): i for i, s in enumerate(sentences)}
    if start_id not in index_by_id or end_id not in index_by_id:
        return None

    first = index_by_id[start_id]
    last = index_by_id[end_id]
    if last <= first:
        return None

    window = sentences[first:last + 1]
    original_duration = float(window[-1]["end"]) - float(window[0]["start"])

    # Nothing to narrow into: even the whole range is shorter than the floor.
    if original_duration <= maximum_seconds:
        return None

    from google.genai import types as genai_types

    if client is None:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        client = genai.Client(api_key=api_key)

    schema = _repair_schema()
    prompt = OVERLONG_REPAIR_PROMPT.format(
        minimum_seconds=minimum_seconds,
        maximum_seconds=maximum_seconds,
        multilingual_rules=MULTILINGUAL_RULES,
        topic=topic or "(not given)",
        original_start=start_id,
        original_end=end_id,
        original_duration=original_duration,
        transcript=format_sentences_for_ai(window),
    )

    import gemini_calls
    try:
        response = gemini_calls.call(
            client,
            model=model_name or os.getenv("GEMINI_MODEL") or "gemini-3.1-flash-lite",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
    except Exception as exc:
        print(f"  Overlong repair unavailable ({type(exc).__name__}: {exc})")
        return None

    parsed = getattr(response, "parsed", None)
    if parsed is None:
        return None

    if str(getattr(parsed, "action", "")).upper() != "REPAIR":
        return None

    new_start = str(getattr(parsed, "new_start_sentence", "") or "")
    new_end = str(getattr(parsed, "new_end_sentence", "") or "")

    if new_start not in index_by_id or new_end not in index_by_id:
        return None

    new_first = index_by_id[new_start]
    new_last = index_by_id[new_end]

    # Must be a NARROWING of the original range, in order.
    if not (first <= new_first <= new_last <= last):
        return None
    if new_first == first and new_last == last:
        return None

    selected = sentences[new_first:new_last + 1]
    duration = float(selected[-1]["end"]) - float(selected[0]["start"])

    if duration < minimum_seconds or duration > maximum_seconds:
        return None

    return {
        "start_sentence": new_start,
        "end_sentence": new_end,
        "original_start": start_id,
        "original_end": end_id,
        "original_duration": round(original_duration, 3),
        "duration": round(duration, 3),
        "reason": str(getattr(parsed, "reason", "") or "")[:300],
    }


# ---------------------------------------------------------------------------
# Candidate validation / scoring
# ---------------------------------------------------------------------------

def _candidate_combined_score(candidate: Dict[str, Any]) -> float:

    standalone = float(
        candidate.get("standalone_score") or 0
    )

    completeness = float(
        candidate.get("completeness_score") or 0
    )

    content = float(
        candidate.get("content_score") or 0
    )

    hook = float(
        candidate.get("hook_score") or 0
    )

    # Our priorities:
    # standalone meaning > completeness > content > hook
    return round(
        standalone * 0.45
        + completeness * 0.25
        + content * 0.20
        + hook * 0.10,
        2
    )


def _candidate_overlap_ratio(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> float:

    a_start = int(a["_start_index"])
    a_end = int(a["_end_index"])

    b_start = int(b["_start_index"])
    b_end = int(b["_end_index"])

    intersection = max(
        0,
        min(a_end, b_end)
        - max(a_start, b_start)
        + 1
    )

    if intersection <= 0:
        return 0.0

    a_size = a_end - a_start + 1
    b_size = b_end - b_start + 1

    return intersection / min(a_size, b_size)


def _deduplicate_candidates(
    candidates: List[Dict[str, Any]],
    overlap_threshold: float = 0.70,
) -> List[Dict[str, Any]]:

    ranked = sorted(
        candidates,
        key=lambda x: x.get("combined_score", 0),
        reverse=True
    )

    kept = []

    for candidate in ranked:

        duplicate = False

        for existing in kept:

            overlap = _candidate_overlap_ratio(
                candidate,
                existing
            )

            if overlap >= overlap_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return kept


# ---------------------------------------------------------------------------
# Gemini Candidate Finder
# ---------------------------------------------------------------------------

def find_candidates_with_gemini(
    sentences: List[Dict[str, Any]],
    language: str = "unknown",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    minimum_seconds: float = MIN_CLIP_SECONDS,
    maximum_seconds: float = MAX_CLIP_SECONDS,
) -> List[Dict[str, Any]]:
    """
    AI PASS #1.

    Find potentially valuable semantic moments.

    This does NOT render anything.
    This does NOT generate marketing titles.
    This does NOT make final clip decisions.

    The Context Critic will review/repair these candidates later.
    """

    # Lazy imports so sentence-only testing works even on a host machine
    # that does not have OpenShorts' AI dependencies installed.
    from google import genai
    from google.genai import types as genai_types
    from pydantic import BaseModel
    from typing import List as TypingList

    class CandidateModel(BaseModel):
        start_sentence: str
        end_sentence: str
        topic: str
        reason: str
        standalone_score: int
        completeness_score: int
        content_score: int
        hook_score: int

    class CandidateResponse(BaseModel):
        candidates: TypingList[CandidateModel]

    api_key = (
        api_key
        or os.getenv("GEMINI_API_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY."
        )

    model_name = (
        model_name
        or os.getenv("GEMINI_MODEL")
        or "gemini-3.1-flash-lite"
    )

    client = genai.Client(
        api_key=api_key
    )

    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CandidateResponse,
        candidate_count=1,
        temperature=0.2,
    )

    sentence_index = {
        sentence["id"]: index
        for index, sentence in enumerate(sentences)
    }

    windows = build_candidate_windows(sentences)

    print()
    print(
        f"Candidate Finder: "
        f"{len(windows)} transcript windows"
    )

    print(
        f"Gemini model: {model_name}"
    )

    raw_candidates = []
    skipped_windows: List[str] = []

    import gemini_calls as _gemini_calls

    for window_number, window in enumerate(
        windows,
        start=1
    ):

        print(
            f"Analyzing {window['id']} "
            f"({window_number}/{len(windows)}) "
            f"{window['start']:.1f}s -> "
            f"{window['end']:.1f}s..."
        )

        transcript_text = format_sentences_for_ai(
            window["sentences"]
        )

        prompt = CANDIDATE_FINDER_PROMPT.format(
            language=language,
            multilingual_rules=MULTILINGUAL_RULES,
            duration_rule=duration_rule_text(),
            window_id=window["id"],
            transcript=transcript_text,
        )

        # A window that cannot be analysed is SKIPPED, not fatal. This loop
        # used to call Gemini bare: the 14th window of a 50-minute video hit
        # the free tier's 15-per-minute limit, the exception escaped, and
        # meaningful_pipeline restarted the whole stage — discarding the 13
        # windows that had already succeeded and spending more quota to redo
        # them. 21 of 22 windows is a usable result; zero is not.
        try:
            response = _gemini_calls.call(
                client,
                model=model_name,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            skipped_windows.append(window["id"])
            print(f"  {window['id']}: skipped "
                  f"({type(exc).__name__}: {str(exc)[:160]})")
            continue

        parsed_obj = getattr(
            response,
            "parsed",
            None
        )

        if parsed_obj is not None:

            if hasattr(parsed_obj, "model_dump"):
                parsed = parsed_obj.model_dump()
            else:
                parsed = parsed_obj

        else:

            raw_text = (
                getattr(response, "text", "")
                or ""
            ).strip()

            if not raw_text:
                print(
                    f"  {window['id']}: "
                    f"empty Gemini response"
                )
                continue

            parsed = json.loads(raw_text)

        found = parsed.get("candidates") or []

        print(
            f"  Proposed: {len(found)}"
        )

        window_sentence_ids = {
            sentence["id"]
            for sentence in window["sentences"]
        }

        for candidate in found:

            start_id = str(
                candidate.get("start_sentence") or ""
            )

            end_id = str(
                candidate.get("end_sentence") or ""
            )

            # IDs must be real.
            if (
                start_id not in sentence_index
                or end_id not in sentence_index
            ):
                continue

            # Candidate must stay inside the window supplied
            # to that model call.
            if (
                start_id not in window_sentence_ids
                or end_id not in window_sentence_ids
            ):
                continue

            start_index = sentence_index[start_id]
            end_index = sentence_index[end_id]

            if end_index < start_index:
                continue

            selected = sentences[
                start_index:end_index + 1
            ]

            start_time = float(
                selected[0]["start"]
            )

            end_time = float(
                selected[-1]["end"]
            )

            duration = end_time - start_time

            if duration < minimum_seconds:
                print(
                    f"  Rejected {start_id}->{end_id}: "
                    f"{duration:.1f}s is too short"
                )
                continue

            if duration > maximum_seconds:
                # An over-long proposal is not noise: the model found a real
                # idea and drew the boundaries too wide. Throwing it away threw
                # away the window's only candidate three times out of four on
                # the Hinglish run, so ask for the shortest complete
                # SUBSECTION of that same range before giving up.
                repaired = repair_overlong_candidate(
                    sentences=sentences,
                    start_id=start_id,
                    end_id=end_id,
                    topic=str(candidate.get("topic") or ""),
                    client=client,
                    model_name=model_name,
                    minimum_seconds=minimum_seconds,
                    maximum_seconds=maximum_seconds,
                )
                if repaired is None:
                    print(
                        f"  Rejected {start_id}->{end_id}: "
                        f"{duration:.1f}s exceeds {maximum_seconds:.0f}s and no "
                        f"complete shorter section exists inside it"
                    )
                    continue

                start_id, end_id = repaired["start_sentence"], repaired["end_sentence"]
                start_index = sentence_index[start_id]
                end_index = sentence_index[end_id]
                selected = sentences[start_index:end_index + 1]
                start_time = float(selected[0]["start"])
                end_time = float(selected[-1]["end"])
                repaired_duration = end_time - start_time
                print(
                    f"  Repaired {repaired['original_start']}->"
                    f"{repaired['original_end']} ({duration:.1f}s) into "
                    f"{start_id}->{end_id} ({repaired_duration:.1f}s)"
                )
                duration = repaired_duration
                candidate = dict(candidate)
                candidate["start_sentence"] = start_id
                candidate["end_sentence"] = end_id
                candidate["overlong_repair"] = repaired

            candidate = dict(candidate)

            candidate["source_window_id"] = (
                window["id"]
            )

            candidate["start"] = round(
                start_time,
                3
            )

            candidate["end"] = round(
                end_time,
                3
            )

            candidate["duration"] = round(
                duration,
                3
            )

            candidate["combined_score"] = (
                _candidate_combined_score(candidate)
            )

            candidate["transcript"] = " ".join(
                sentence["text"]
                for sentence in selected
            )

            # Internal fields only used for deduplication.
            candidate["_start_index"] = start_index
            candidate["_end_index"] = end_index

            raw_candidates.append(candidate)

    deduplicated = _deduplicate_candidates(
        raw_candidates
    )

    # Remove internal fields and assign stable candidate IDs.
    final = []

    for number, candidate in enumerate(
        deduplicated,
        start=1
    ):

        candidate = dict(candidate)

        candidate.pop(
            "_start_index",
            None
        )

        candidate.pop(
            "_end_index",
            None
        )

        candidate["candidate_id"] = (
            f"C{number:03d}"
        )

        final.append(candidate)

    print()
    print(
        f"Raw valid candidates: "
        f"{len(raw_candidates)}"
    )

    print(
        f"After deduplication: "
        f"{len(final)}"
    )

    # Said plainly, because a thin result from a partly-analysed transcript
    # looks exactly like a video with few good moments in it.
    if skipped_windows:
        print(
            f"⚠️ {len(skipped_windows)} of {len(windows)} window(s) could not "
            f"be analysed and were skipped: "
            f"{', '.join(skipped_windows)}. Fewer candidates than usual is "
            f"expected; raise GEMINI_RPM only if the key's quota allows it."
        )

    return final