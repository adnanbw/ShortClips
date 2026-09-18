"""
Semantic opening-independence guard for meaningful clip selection.

Purpose
-------
Catch clips whose *opening* depends on information that only existed before the
clip, without using a brittle word/phrase blacklist.

Flow
----
1. Only inspect candidates already accepted/repaired by the main critic.
2. Judge the opening semantically, using only the first few sentence units.
3. If the opening requires prior context, ask a repairer to move ONLY the start
   boundary backward to the minimum earlier sentence needed.
4. Re-run the opening judge on the repaired version.
5. Keep the clip only if the repaired opening is independently understandable.

This module intentionally does not change the clip end boundary.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from meaningful_selector import (
    MULTILINGUAL_RULES,
    WHAT_YOU_ARE_JUDGING,
    MAX_CLIP_SECONDS,
    MIN_CLIP_SECONDS,
)


DEFAULT_MODEL = os.environ.get("MEANINGFUL_OPENING_MODEL", "gemini-3.1-flash-lite")
MAX_CONTEXT_SENTENCES = 8
OPENING_SENTENCE_COUNT = 3


class OpeningAssessment(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    requires_prior_context: bool
    missing_context: str = ""
    explanation: str
    confidence: int = Field(ge=0, le=100)


class OpeningRepairDecision(BaseModel):
    action: Literal["REPAIR", "REJECT"]
    new_start_sentence: Optional[str] = None
    reason: str


def _sentence_map(sentences: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(s.get("id")): s for s in sentences if s.get("id")}


def _sentence_index(sentences: List[Dict[str, Any]]) -> Dict[str, int]:
    return {str(s.get("id")): i for i, s in enumerate(sentences) if s.get("id")}


def _extract_bounds(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Be tolerant of slightly different critic-result schemas.

    The current pipeline normally exposes start_sentence/end_sentence at the
    top level, but older debug runs may have final/repaired candidates nested.
    """
    containers = [
        item,
        item.get("final_candidate") or {},
        item.get("repaired_candidate") or {},
        item.get("candidate") or {},
    ]

    start_keys = (
        "final_start_sentence",
        "repaired_start_sentence",
        "start_sentence",
    )
    end_keys = (
        "final_end_sentence",
        "repaired_end_sentence",
        "end_sentence",
    )

    start_id = None
    end_id = None

    for container in containers:
        if not isinstance(container, dict):
            continue
        if start_id is None:
            for key in start_keys:
                value = container.get(key)
                if value:
                    start_id = str(value)
                    break
        if end_id is None:
            for key in end_keys:
                value = container.get(key)
                if value:
                    end_id = str(value)
                    break
        if start_id and end_id:
            break

    return start_id, end_id


def _apply_bounds(
    item: Dict[str, Any],
    start_id: str,
    end_id: str,
) -> None:
    """
    Keep the top-level schema explicit because meaningful_metadata consumes the
    approved critic result downstream.
    """
    item["start_sentence"] = start_id
    item["end_sentence"] = end_id
    item["final_start_sentence"] = start_id
    item["final_end_sentence"] = end_id

    final_candidate = item.get("final_candidate")
    if isinstance(final_candidate, dict):
        final_candidate["start_sentence"] = start_id
        final_candidate["end_sentence"] = end_id


def _range_duration(
    sentence_by_id: Dict[str, Dict[str, Any]],
    start_id: str,
    end_id: str,
) -> Optional[float]:
    start = sentence_by_id.get(start_id)
    end = sentence_by_id.get(end_id)
    if not start or not end:
        return None
    try:
        return float(end["end"]) - float(start["start"])
    except (KeyError, TypeError, ValueError):
        return None


def _opening_units(
    sentences: List[Dict[str, Any]],
    start_id: str,
    end_id: str,
    count: int = OPENING_SENTENCE_COUNT,
) -> List[Dict[str, Any]]:
    index_by_id = _sentence_index(sentences)
    if start_id not in index_by_id or end_id not in index_by_id:
        return []

    start_i = index_by_id[start_id]
    end_i = index_by_id[end_id]
    if end_i < start_i:
        return []

    return sentences[start_i : min(end_i + 1, start_i + count)]


def _format_units(units: List[Dict[str, Any]]) -> str:
    lines = []
    for s in units:
        sid = s.get("id", "")
        text = str(s.get("text") or "").strip()
        start = float(s.get("start") or 0.0)
        end = float(s.get("end") or 0.0)
        lines.append(f"{sid} [{start:.2f}-{end:.2f}] {text}")
    return "\n".join(lines)


def _response_parsed(response: Any, model_cls: Any) -> Any:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return parsed

    # Defensive fallback for SDK versions where parsed is unavailable.
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini returned no structured response")
    return model_cls.model_validate_json(text)


def _call_structured(
    client: genai.Client,
    *,
    model_name: str,
    prompt: str,
    schema: Any,
    temperature: float = 0.0,
    max_attempts: int = 5,
) -> Any:
    delays = [5, 10, 20, 30, 45]

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            return _response_parsed(response, schema)
        except Exception as exc:
            msg = str(exc).lower()
            transient = any(
                marker in msg
                for marker in (
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "resource_exhausted",
                    "unavailable",
                    "timeout",
                    "temporarily",
                    "rate limit",
                )
            )
            if not transient or attempt == max_attempts - 1:
                raise

            delay = delays[min(attempt, len(delays) - 1)]
            print(
                f"      Opening guard Gemini retry {attempt + 1}/{max_attempts} "
                f"after {delay}s: {type(exc).__name__}"
            )
            time.sleep(delay)

    raise RuntimeError("Opening guard Gemini retry loop exhausted")


def judge_opening_independence(
    *,
    client: genai.Client,
    model_name: str,
    opening_units: List[Dict[str, Any]],
) -> OpeningAssessment:
    """
    Focused semantic judge.

    Important: this is NOT a lexical blacklist. Words such as "this", "and",
    "first", etc. are allowed when the opening itself supplies enough meaning.
    """
    opening_text = _format_units(opening_units)

    prompt = f"""
You are an OPENING-INDEPENDENCE JUDGE for short-form video clips.

You are shown ONLY the first few sentence units of a proposed clip.
Judge one thing only:

CAN A NEW VIEWER UNDERSTAND THE OPENING WITHOUT HEARING ANYTHING BEFORE IT?

Do not judge virality, style, grammar, topic quality, or the ending.

A clip opening FAILS when understanding it requires missing information from
before the clip, for example:
- an action is referenced but the action itself has not been established;
- a sequence begins in the middle of earlier steps;
- a pronoun/demonstrative/reference points backward to an idea that the opening
  never defines;
- a comparison, consequence, continuation, example, or transition only makes
  sense if a viewer already knows the previous discussion.

A clip opening PASSES when the opening itself supplies enough information,
even if it contains transition words or references.

Important distinctions:
- "And so the trap here ... is that they're aiming for confidence" can PASS
  because the same sentence defines what the trap is.
- "The first thing is learning how to do what's called a dependency map" can
  PASS because the actual subject/action is introduced immediately.
- "This ties into a concept..." should FAIL when "this" refers to an idea that
  exists only before the clip.
- "And then you start connecting..." should FAIL when the prior step is absent.
- "The more you try, your clarity goes down" should FAIL when the viewer cannot
  know what "try" refers to from the opening itself.

Do NOT fail merely because a particular word appears.
Evaluate semantic dependency, not vocabulary.

If the meaning becomes clear only much later in the clip, that is still a bad
opening. The viewer should not need to wait for missing setup to be reconstructed.

Return PASS only when the opening is independently understandable.

{WHAT_YOU_ARE_JUDGING}
{MULTILINGUAL_RULES}

OPENING:
{opening_text}
""".strip()

    return _call_structured(
        client,
        model_name=model_name,
        prompt=prompt,
        schema=OpeningAssessment,
        temperature=0.0,
    )


def _eligible_repair_starts(
    *,
    sentences: List[Dict[str, Any]],
    start_id: str,
    end_id: str,
    max_context_sentences: int = MAX_CONTEXT_SENTENCES,
) -> List[Dict[str, Any]]:
    index_by_id = _sentence_index(sentences)
    by_id = _sentence_map(sentences)

    if start_id not in index_by_id or end_id not in index_by_id:
        return []

    start_i = index_by_id[start_id]
    earliest_i = max(0, start_i - max_context_sentences)

    eligible = []
    for i in range(earliest_i, start_i + 1):
        sid = str(sentences[i].get("id"))
        duration = _range_duration(by_id, sid, end_id)
        if duration is None:
            continue
        if MIN_CLIP_SECONDS <= duration <= MAX_CLIP_SECONDS:
            eligible.append(sentences[i])

    return eligible


def repair_opening_start(
    *,
    client: genai.Client,
    model_name: str,
    sentences: List[Dict[str, Any]],
    start_id: str,
    end_id: str,
    failed_assessment: OpeningAssessment,
) -> OpeningRepairDecision:
    """
    Ask Gemini to choose the MINIMUM earlier sentence required to fix the
    opening. End boundary is intentionally locked.
    """
    index_by_id = _sentence_index(sentences)

    eligible = _eligible_repair_starts(
        sentences=sentences,
        start_id=start_id,
        end_id=end_id,
    )

    earlier_only = [
        s for s in eligible
        if index_by_id.get(str(s.get("id")), 10**9) < index_by_id[start_id]
    ]

    if not earlier_only:
        return OpeningRepairDecision(
            action="REJECT",
            new_start_sentence=None,
            reason=(
                "No earlier sentence can repair the opening while "
                f"keeping the clip within {MIN_CLIP_SECONDS:.0f}-"
                f"{MAX_CLIP_SECONDS:.0f} seconds."
            ),
        )

    # Show all allowed earlier starts plus enough current material to understand
    # what must be connected. The model may ONLY pick one listed START ID.
    candidate_starts_text = _format_units(eligible)

    current_opening = _format_units(
        _opening_units(sentences, start_id, end_id, count=OPENING_SENTENCE_COUNT)
    )

    allowed_ids = [str(s.get("id")) for s in earlier_only]

    prompt = f"""
You are repairing ONLY THE START BOUNDARY of a short-form video clip.

The clip has already passed a broader content/completeness review, but a focused
opening judge found that its first lines depend on missing prior context.

FAILED OPENING ASSESSMENT:
Missing context: {failed_assessment.missing_context}
Explanation: {failed_assessment.explanation}

CURRENT OPENING:
{current_opening}

POSSIBLE EARLIER SENTENCE STARTS:
{candidate_starts_text}

Your task:
Choose the SHORTEST / LATEST earlier sentence that makes the clip's opening
fully understandable to a viewer who has seen nothing before the clip.

Rules:
1. You may change ONLY the start boundary.
2. The end boundary is locked at {end_id}.
3. The new start MUST be one of these earlier IDs:
   {", ".join(allowed_ids)}
4. Add only the minimum context needed.
5. Do not add unrelated setup just to make the clip longer.
6. Do not choose an earlier sentence that itself starts in the middle of an
   unresolved thought or sequence.
7. If none of the allowed starts can make the opening independently
   understandable, return REJECT.
8. This is semantic repair. Do not decide based on any blacklist of words.

{WHAT_YOU_ARE_JUDGING}
{MULTILINGUAL_RULES}

Return REPAIR with the chosen sentence ID, or REJECT.
""".strip()

    result: OpeningRepairDecision = _call_structured(
        client,
        model_name=model_name,
        prompt=prompt,
        schema=OpeningRepairDecision,
        temperature=0.0,
    )

    if result.action == "REPAIR":
        if not result.new_start_sentence:
            return OpeningRepairDecision(
                action="REJECT",
                reason="Repairer returned REPAIR without a start sentence.",
            )
        if result.new_start_sentence not in allowed_ids:
            return OpeningRepairDecision(
                action="REJECT",
                reason=(
                    f"Repairer chose invalid start {result.new_start_sentence}; "
                    "it was not in the allowed earlier sentence IDs."
                ),
            )

    return result


def enforce_opening_independence(
    *,
    sentences: List[Dict[str, Any]],
    reviews: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run the focused opening gate over critic-approved clips.

    Input/output is the critic-result list so this can be inserted directly
    between review_candidates_with_gemini(...) and _rank_approved(...).
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for opening guard")

    model_name = model_name or DEFAULT_MODEL
    client = genai.Client(api_key=api_key)

    by_id = _sentence_map(sentences)
    updated: List[Dict[str, Any]] = []

    active = [
        r for r in reviews
        if str(r.get("decision", "")).upper() in {"ACCEPT", "REPAIR"}
    ]

    print(f"Opening Independence Guard: {len(active)} approved candidate(s)")
    print(f"Gemini model: {model_name}")

    active_counter = 0

    for original in reviews:
        review = dict(original)
        decision = str(review.get("decision", "")).upper()

        # Preserve already-rejected candidates untouched.
        if decision not in {"ACCEPT", "REPAIR"}:
            updated.append(review)
            continue

        active_counter += 1
        candidate_id = review.get("candidate_id") or f"C{active_counter:03d}"
        start_id, end_id = _extract_bounds(review)

        print(f"  Opening check {candidate_id} ({active_counter}/{len(active)})...")

        if not start_id or not end_id or start_id not in by_id or end_id not in by_id:
            print("    FAIL: could not resolve sentence boundaries")
            review["decision"] = "REJECT"
            review["opening_guard"] = {
                "status": "REJECT",
                "reason": "Could not resolve sentence boundaries.",
            }
            updated.append(review)
            continue

        opening = _opening_units(sentences, start_id, end_id)
        assessment = judge_opening_independence(
            client=client,
            model_name=model_name,
            opening_units=opening,
        )

        print(
            f"    Opening gate: {assessment.verdict} "
            f"(confidence {assessment.confidence})"
        )

        if assessment.verdict == "PASS" and not assessment.requires_prior_context:
            review["opening_guard"] = {
                "status": "PASS",
                "original_start_sentence": start_id,
                "final_start_sentence": start_id,
                "assessment": assessment.model_dump(),
            }
            updated.append(review)
            continue

        print(f"    Missing context: {assessment.missing_context or assessment.explanation}")
        print("    Forced start-boundary repair...")

        repair = repair_opening_start(
            client=client,
            model_name=model_name,
            sentences=sentences,
            start_id=start_id,
            end_id=end_id,
            failed_assessment=assessment,
        )

        if repair.action != "REPAIR" or not repair.new_start_sentence:
            print(f"    Opening repair: REJECT - {repair.reason}")
            review["decision"] = "REJECT"
            review["opening_guard"] = {
                "status": "REJECT",
                "original_start_sentence": start_id,
                "assessment": assessment.model_dump(),
                "repair": repair.model_dump(),
            }
            updated.append(review)
            continue

        repaired_start = repair.new_start_sentence
        repaired_duration = _range_duration(by_id, repaired_start, end_id)

        if (
            repaired_duration is None
            or repaired_duration < MIN_CLIP_SECONDS
            or repaired_duration > MAX_CLIP_SECONDS
        ):
            print(f"    Opening repair: REJECT - repaired duration outside "
                  f"{MIN_CLIP_SECONDS:.0f}-{MAX_CLIP_SECONDS:.0f}s")
            review["decision"] = "REJECT"
            review["opening_guard"] = {
                "status": "REJECT",
                "original_start_sentence": start_id,
                "assessment": assessment.model_dump(),
                "repair": repair.model_dump(),
                "repaired_duration": repaired_duration,
            }
            updated.append(review)
            continue

        # Re-check the repaired opening BLINDLY. The judge sees only the new
        # opening, not the reason we repaired it.
        repaired_opening = _opening_units(
            sentences,
            repaired_start,
            end_id,
            count=OPENING_SENTENCE_COUNT,
        )
        recheck = judge_opening_independence(
            client=client,
            model_name=model_name,
            opening_units=repaired_opening,
        )

        print(
            f"    Repaired opening gate: {recheck.verdict} "
            f"(confidence {recheck.confidence})"
        )

        if recheck.verdict != "PASS" or recheck.requires_prior_context:
            print("    FINAL: REJECT - repaired opening still depends on prior context")
            review["decision"] = "REJECT"
            review["opening_guard"] = {
                "status": "REJECT",
                "original_start_sentence": start_id,
                "attempted_start_sentence": repaired_start,
                "assessment": assessment.model_dump(),
                "repair": repair.model_dump(),
                "recheck": recheck.model_dump(),
            }
            updated.append(review)
            continue

        _apply_bounds(review, repaired_start, end_id)
        review["decision"] = "REPAIR"
        review["opening_guard"] = {
            "status": "REPAIRED",
            "original_start_sentence": start_id,
            "final_start_sentence": repaired_start,
            "assessment": assessment.model_dump(),
            "repair": repair.model_dump(),
            "recheck": recheck.model_dump(),
        }

        print(
            f"    FINAL: REPAIR {start_id} -> {repaired_start} "
            f"({repaired_duration:.1f}s)"
        )
        updated.append(review)

    return updated
