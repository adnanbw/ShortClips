"""Semantic validation of a transcript: are these actually words?

Why a second module beside ``asr_quality``
------------------------------------------
``asr_quality`` measures STRUCTURE — decoder confidence, repetition, speech
rate, timestamp sanity. It cannot read. That limit is not theoretical: on the
real Hinglish stand-up, ``large-v3-turbo`` produced ``avg_logprob`` -0.29 and a
structural score of **100/100** while a third of the transcript was invented
phonetic Hindi. Structure was never the problem, so no structural signal could
ever have caught it.

What was wrong with the first attempt at this
---------------------------------------------
The original check sampled the first 1500 and last 1500 characters of the whole
transcript. On a ten-minute video that is roughly the first and last 40 seconds,
and the corruption in that job lived in the MIDDLE. It also only ran when the
decoder said it was unsure, so a confident-but-wrong model bypassed it entirely.

So this module:
  * samples across the WHOLE timeline, by timestamp, not by character offset;
  * sends the samples labelled, with structured output, in as few requests
    as fit the prompt budget - one for a sparse pass, and only as many as
    a long video's dense pass actually needs, so no region is silently
    truncated away;
  * returns a per-region verdict as well as a transcript-level one, so a video
    that is half good is not thrown away whole.

It never judges a language, and never treats code-switching as damage:
"मुझे actually ये approach better लगती है" is normal speech and must pass.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

#: Transcript-level verdicts, best first.
STATUS_ORDER = ("GOOD", "PARTIAL", "BAD")

#: How many windows the representative pass samples, and how long each is.
#: Six ~25s windows over a ten-minute video is ~15% coverage spread evenly —
#: enough to notice that a transcript is mixed, which is all this pass decides.
DEFAULT_SAMPLE_COUNT = 6
DEFAULT_SAMPLE_SECONDS = 25.0

#: The dense pass (only for a transcript already found mixed) labels the whole
#: timeline in windows of this length, so its regions are real coverage rather
#: than an extrapolation from samples.
DEFAULT_REGION_SECONDS = 45.0

#: Longest transcript text we will put in one request.
MAX_PROMPT_CHARS = 60000


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, "").strip() or default))
    except (TypeError, ValueError):
        return default


def check_mode() -> str:
    """``ASR_QUALITY_GEMINI``: '0' never | 'auto' when justified | '1' always."""
    raw = os.environ.get("ASR_QUALITY_GEMINI", "auto").strip().lower()
    return raw if raw in ("0", "1", "auto") else "auto"


# --- sampling ---------------------------------------------------------------

def _segments(transcript: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for segment in (transcript or {}).get("segments") or []:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(segment.get("start"))
            end = float(segment.get("end"))
        except (TypeError, ValueError):
            continue
        out.append({"start": start, "end": max(end, start), "text": text})
    out.sort(key=lambda s: s["start"])
    return out


def _window_from(segments: List[Dict[str, Any]], index: int,
                 seconds: float) -> Tuple[List[Dict[str, Any]], int]:
    """Consecutive segments starting at ``index`` covering ~``seconds``."""
    picked = [segments[index]]
    i = index + 1
    while i < len(segments) and \
            segments[i]["end"] - picked[0]["start"] <= seconds:
        picked.append(segments[i])
        i += 1
    return picked, i


def build_samples(transcript: Optional[Dict[str, Any]],
                  count: int = DEFAULT_SAMPLE_COUNT,
                  seconds: float = DEFAULT_SAMPLE_SECONDS
                  ) -> List[Dict[str, Any]]:
    """``count`` windows spread evenly across the transcript's own timeline.

    Anchored on SEGMENT INDEX rather than on wall-clock time: whisper's
    segments follow the speech, so even spacing by index gives even spacing by
    speech, and a long musical gap does not swallow a sample. Each window keeps
    whole segments, so a sample always contains neighbouring speech to judge
    coherence against rather than an isolated fragment.
    """
    segments = _segments(transcript)
    if not segments:
        return []

    count = max(1, count)
    if count >= len(segments):
        anchors = list(range(len(segments)))
    else:
        # Spread the anchors so the first starts at the beginning and the last
        # window still fits inside the transcript.
        step = (len(segments) - 1) / float(count)
        anchors = sorted({int(round(i * step)) for i in range(count)})

    samples: List[Dict[str, Any]] = []
    used_until = -1
    for anchor in anchors:
        anchor = max(anchor, used_until + 1)
        if anchor >= len(segments):
            break
        picked, used_until = _window_from(segments, anchor, seconds)
        used_until -= 1
        samples.append({
            "id": f"R{len(samples) + 1:02d}",
            "start": round(picked[0]["start"], 3),
            "end": round(picked[-1]["end"], 3),
            "text": " ".join(s["text"] for s in picked),
        })
    return samples


def build_regions(transcript: Optional[Dict[str, Any]],
                  seconds: float = DEFAULT_REGION_SECONDS
                  ) -> List[Dict[str, Any]]:
    """Cover the WHOLE transcript in consecutive windows of ~``seconds``.

    Used only once a transcript is already known to be mixed. Full coverage is
    the difference between "we looked here and it was bad" and guessing, and
    only the former may be allowed to veto a clip.
    """
    segments = _segments(transcript)
    regions: List[Dict[str, Any]] = []
    index = 0
    while index < len(segments):
        picked, index = _window_from(segments, index, seconds)
        regions.append({
            "id": f"R{len(regions) + 1:02d}",
            "start": round(picked[0]["start"], 3),
            "end": round(picked[-1]["end"], 3),
            "text": " ".join(s["text"] for s in picked),
        })
    return regions


def _format_samples(samples: List[Dict[str, Any]]) -> str:
    blocks = []
    for sample in samples:
        blocks.append(f"### {sample['id']} "
                      f"[{sample['start']:.1f}s - {sample['end']:.1f}s]\n"
                      f"{sample['text']}")
    text = "\n\n".join(blocks)
    return text[:MAX_PROMPT_CHARS]


# --- the prompt -------------------------------------------------------------

REVIEW_PROMPT = """You are judging whether an automatic speech transcript
contains REAL LANGUAGE. You are not judging whether the speech is interesting,
well-structured, polite or grammatical.

You are shown several labelled regions sampled from across one video. Judge
EACH region on its own and return one verdict per region.

WHAT IS NORMAL AND MUST PASS
- Any language. Never mark a region down for being in a language you did not
  expect.
- Code-switching, including inside one sentence. This line:
      मुझे actually ये approach better लगती है
  is completely normal bilingual speech, not an error.
- Informal speech, slang, regional wording, filler words, false starts,
  repetition a real speaker would produce, missing punctuation, and words
  transliterated into another script.
- Speech that is fragmentary because it is conversational, as long as the words
  themselves are real words.

WHAT IS CORRUPTION
- Words that do not exist in the language they are written in.
- Phonetic gibberish that merely LOOKS like the script — a weak speech model
  transcribing a language it cannot handle invents fluent-looking words.
- Strings no human would have said, or text that cannot be read aloud as the
  language it claims to be.
- Endlessly repeated phrases the speaker plainly did not say.

For each region return:
- status: GOOD when essentially all of it is real, understandable speech;
  PARTIAL when it is mixed — some of it reads correctly and some is invented;
  BAD when most of it is not real language.
- usability: 0-100, how much of this region a person could actually rely on.
- code_switching: true if the region legitimately mixes languages.
- reason: one short sentence, naming an actual example from the region.

Be strict about invented words and generous about everything else.

REGIONS:

{samples}
"""


def _schema():
    from pydantic import BaseModel
    from typing import List as TypingList, Literal

    class RegionVerdict(BaseModel):
        id: str
        status: Literal["GOOD", "PARTIAL", "BAD"]
        usability: int
        code_switching: bool = False
        reason: str = ""

    class ReviewResponse(BaseModel):
        regions: TypingList[RegionVerdict]

    return ReviewResponse


def _call_gemini(samples: List[Dict[str, Any]], api_key: str,
                 model_name: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    from google import genai
    from google.genai import types as genai_types

    schema = _schema()
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name or os.environ.get("GEMINI_MODEL")
        or "gemini-3.1-flash-lite",
        contents=REVIEW_PROMPT.format(samples=_format_samples(samples)),
        config=genai_types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    parsed = getattr(response, "parsed", None)
    if parsed is None:
        return None
    return [r.model_dump() for r in parsed.regions]


# --- aggregation ------------------------------------------------------------

def aggregate(samples: List[Dict[str, Any]],
              verdicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-region verdicts into one transcript-level result.

    Weighted by how many SECONDS each region covers, not by region count: a
    45-second region and a 5-second one are not equal evidence.
    """
    by_id = {str(s["id"]): s for s in samples}
    regions: List[Dict[str, Any]] = []
    for verdict in verdicts or []:
        sample = by_id.get(str(verdict.get("id")))
        if not sample:
            continue
        status = str(verdict.get("status") or "").upper()
        if status not in STATUS_ORDER:
            status = "PARTIAL"
        try:
            usability = max(0, min(100, int(verdict.get("usability", 0))))
        except (TypeError, ValueError):
            usability = 0
        regions.append({
            "id": sample["id"],
            "start": sample["start"],
            "end": sample["end"],
            "status": status,
            "score": usability,
            "code_switching": bool(verdict.get("code_switching")),
            "reason": str(verdict.get("reason") or "")[:300],
        })

    if not regions:
        return {"status": None, "score": None, "regions": [],
                "reason": "no region verdicts returned"}

    def seconds(region):
        return max(0.001, float(region["end"]) - float(region["start"]))

    total = sum(seconds(r) for r in regions)
    share = {name: sum(seconds(r) for r in regions if r["status"] == name) / total
             for name in STATUS_ORDER}
    score = sum(r["score"] * seconds(r) for r in regions) / total

    good_at = _env_float("ASR_SEMANTIC_GOOD_SHARE", 0.80)
    bad_below = _env_float("ASR_SEMANTIC_BAD_SHARE", 0.25)

    if share["GOOD"] >= good_at and share["BAD"] <= 0.10:
        status = "GOOD"
    elif share["GOOD"] + 0.5 * share["PARTIAL"] < bad_below:
        status = "BAD"
    else:
        status = "PARTIAL"

    return {
        "status": status,
        "score": round(score, 1),
        "regions": regions,
        "shares": {name: round(value, 3) for name, value in share.items()},
        "reason": "; ".join(r["reason"] for r in regions
                            if r["status"] != "GOOD" and r["reason"])[:500],
    }


# --- public entry point -----------------------------------------------------

def _batched(samples: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Split samples into requests that each fit inside MAX_PROMPT_CHARS."""
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    size = 0
    for sample in samples:
        cost = len(sample["text"]) + 60          # + the region header
        if current and size + cost > MAX_PROMPT_CHARS:
            batches.append(current)
            current, size = [], 0
        current.append(sample)
        size += cost
    if current:
        batches.append(current)
    return batches


def review_transcript(transcript: Optional[Dict[str, Any]],
                      api_key: Optional[str] = None,
                      model_name: Optional[str] = None,
                      dense: bool = False) -> Optional[Dict[str, Any]]:
    """Read the transcript and report whether its words are real words.

    ``dense=False`` samples ~6 windows spread across the timeline (one call).
    ``dense=True`` labels the WHOLE timeline in consecutive windows, which is
    what makes the region list usable as a veto rather than a hint; it is only
    worth paying for once a transcript is already known to be mixed.

    Returns None when the check cannot run (no key, disabled, API error) — an
    unavailable reader must never be read as a guilty verdict.
    """
    if check_mode() == "0":
        return None

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    if dense:
        samples = build_regions(
            transcript,
            seconds=_env_float("ASR_SEMANTIC_REGION_SECONDS",
                               DEFAULT_REGION_SECONDS))
    else:
        samples = build_samples(
            transcript,
            count=_env_int("ASR_SEMANTIC_SAMPLES", DEFAULT_SAMPLE_COUNT),
            seconds=_env_float("ASR_SEMANTIC_SAMPLE_SECONDS",
                               DEFAULT_SAMPLE_SECONDS))
    if not samples:
        return None

    # Batch so a long video's later regions cannot be silently truncated out of
    # the prompt. A two-hour source produces ~160 dense regions; putting them
    # in one request would hit MAX_PROMPT_CHARS and quietly drop the back half,
    # which would then read as "not examined" and veto nothing. Sparse passes
    # are always one call.
    verdicts = []
    for batch in _batched(samples):
        try:
            answer = _call_gemini(batch, api_key, model_name)
        except Exception as exc:  # the reader is optional; never fail a job
            print(f"⚠️ Transcript language check unavailable "
                  f"({type(exc).__name__}: {exc})")
            return None
        if answer is None:
            return None
        verdicts.extend(answer)

    if not verdicts:
        return None

    result = aggregate(samples, verdicts)
    result["dense"] = bool(dense)
    result["sampled_regions"] = len(samples)
    return result if result.get("status") else None


# --- using the regions ------------------------------------------------------

def unreliable_ranges(semantic: Optional[Dict[str, Any]],
                      include_partial: bool = False
                      ) -> List[Tuple[float, float]]:
    """Time ranges a clip must not be built through.

    Only regions we ACTUALLY looked at are returned. Silence about a stretch
    means "not examined", never "fine" — and never "bad" either, which is why a
    sparse sampling pass must not be used to veto clips on its own.
    """
    wanted = {"BAD"} | ({"PARTIAL"} if include_partial else set())
    return [(float(r["start"]), float(r["end"]))
            for r in (semantic or {}).get("regions") or []
            if str(r.get("status")).upper() in wanted]


def overlap_seconds(start: float, end: float,
                    ranges: List[Tuple[float, float]]) -> float:
    """Total seconds of [start, end) covered by ``ranges``."""
    total = 0.0
    for low, high in ranges:
        total += max(0.0, min(float(end), high) - max(float(start), low))
    return total


def region_at(semantic: Optional[Dict[str, Any]],
              timestamp: float) -> Optional[Dict[str, Any]]:
    for region in (semantic or {}).get("regions") or []:
        if float(region["start"]) <= timestamp <= float(region["end"]):
            return region
    return None


def format_summary(semantic: Optional[Dict[str, Any]]) -> str:
    if not semantic:
        return "🧪 Language check: unavailable"
    shares = semantic.get("shares") or {}
    return (f"🧪 Language check: {semantic.get('status')} "
            f"score={semantic.get('score')} "
            f"(good {shares.get('GOOD', 0):.0%}, "
            f"partial {shares.get('PARTIAL', 0):.0%}, "
            f"bad {shares.get('BAD', 0):.0%}) "
            f"over {semantic.get('sampled_regions', 0)} region(s)")
