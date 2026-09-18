import json
import os
import time
from typing import Any, Dict, List, Optional

from meaningful_selector import (
    MULTILINGUAL_RULES,
    WHAT_YOU_ARE_JUDGING,
    MAX_CLIP_SECONDS,
    MIN_CLIP_SECONDS,
    duration_rule_text,
)


def _pair(verification):
    """(standalone, completeness) as plain numbers, for comparing two passes."""
    verification = verification or {}
    try:
        return (
            float(verification.get("standalone_score") or 0),
            float(verification.get("completeness_score") or 0),
        )
    except (TypeError, ValueError):
        return (0.0, 0.0)


def _is_worse(candidate_verification, reference_verification):
    """True when a repair came back scoring lower than what it replaced."""
    return sum(_pair(candidate_verification)) < sum(_pair(reference_verification))


def accept_floor() -> float:
    """Scores at or above this are good enough to SHIP, not just to rank.

    The critic used to have exactly one way through: verdict PASS, four
    booleans true, and 85+ on BOTH standalone and completeness. Measured on
    two real jobs, that bar is not a quality control — it is a coin flip on
    input the critic happens to like. An English stand-up set scored 95-100 on
    six of seven candidates and sailed through; a Hinglish one scored 80/40,
    30/40, 30/20, 60/30 and the job died with "Clip detection failed", which
    tells the user nothing true.

    85 on both axes, on a scale whose own prompt says "90+ should mean
    genuinely excellent", is asking for near-excellence twice over from every
    single clip. That is a preference. The floor below is the actual quality
    question — is this worth publishing at all — and everything between the
    two is decided by RANK, which is what a score is for.
    """
    try:
        return float(os.environ.get("MEANINGFUL_ACCEPT_FLOOR", "").strip() or 70.0)
    except (TypeError, ValueError):
        return 70.0


# ============================================================================
# BLIND VERIFICATION PROMPT
# ============================================================================

BLIND_VERIFY_PROMPT = """
You are the FINAL standalone-quality verifier for a short-form video clip.

IMPORTANT:

This transcript is EXACTLY AND ONLY the stretch of the video the clip covers.

You have NO access to the rest of the source video.

Do not imagine or infer missing context.
Do not assume you know what was discussed earlier.
Judge only the supplied clip.

{what_you_are_judging}

The product's #1 requirement is SEMANTIC COMPLETENESS.

The clip should feel as though it could have been recorded as an independent
piece of content.

A stranger who has never watched the source should understand:
- what the speaker is talking about
- why examples are being given
- what pronouns/references refer to
- where the argument/story begins
- what conclusion/payoff is reached

OPENING TEST:

The first one or two sentences must establish enough context.

FAIL examples:

"And so one example is quitting smoking."

If the viewer does not know what smoking is an example OF, this is NOT a
self-contained opening.

"And what I recommend doing here is..."

If "here" depends on something outside the clip, FAIL.

"The second reason is..."

Second reason for what? FAIL unless the subject is independently established.

"And so I've discovered that hobbies can be the same thing..."

Same thing as WHAT? FAIL.

"She told me..."

If who "she" is or why the conversation matters is missing, FAIL. The viewer
sees the speaker, but nothing on screen tells them who "she" is.

NOT a fail:

"I went to a Muslim school and I told them I wanted to do comedy."

The viewer can see this is a comedian on a stage. The clip does not have to
introduce them, name them, or explain that they perform for a living.

A sentence beginning with "and", "but", or "so" is NOT automatically wrong.
Only fail it when the meaning relies on information outside the supplied clip.

ENDING TEST:

The final sentence must actually finish the idea.

FAIL examples:

"...and the reason why..."
"...what that means is..."
"...and that's because..."
"...turn these ideas into..."
"...the first thing is..."

A grammatically complete sentence can still be semantically incomplete.

Equally, a clip that has delivered its punchline or its result IS complete,
even though nothing was summarised or explained afterwards.

For example:

"This happened for three reasons."

is grammatically complete but is an incomplete ending if the reasons are not
given.

LIST TEST:

If the clip promises "three reasons", "four questions", "five steps", etc.,
the clip must either complete that promised structure or cleanly focus on one
item without falsely promising the others.

STORY TEST:

For anecdotes, the viewer needs enough information to understand:
- what happened
- who matters
- why the event is worth telling
- how it resolves

WHAT COUNTS AS AN ENDING DEPENDS ON THE CONTENT, NOT ON YOUR PREFERENCE:

- Comedy, stand-up and personal anecdotes end on a PUNCHLINE, a payoff, an
  absurd escalation, a reveal, or the outcome of the situation. That IS the
  ending. A joke does not owe anyone a moral, a lesson or a takeaway, and
  demanding one is not a completeness standard — it is a genre preference.
- Explanatory and educational content ends on the answer, the conclusion, the
  result or the lesson.

Judge whether THIS piece finishes what IT started. Do not lower the bar for any
genre: a joke with its punchline cut off is just as incomplete as an
explanation with its conclusion cut off, and a story that stops before the
resolution still fails.

Do not demand unnecessary background.

LENGTH:

The timestamp duration is provided for reference.

{duration_rule}

{multilingual_rules}

SCORING:

standalone_score:
0-100 — Can a cold viewer understand the clip with zero outside context?

completeness_score:
0-100 — Does it contain the necessary setup AND final payoff?

90+ should mean genuinely excellent.

CLIP ID:
{candidate_id}

DURATION:
{duration:.2f} seconds

TRANSCRIPT:

{transcript}
"""


# ============================================================================
# CONTEXT REPAIR PROMPT
# ============================================================================

REPAIR_PROMPT = """
You are repairing the boundaries of a potentially good short-form video clip.

A BLIND verifier saw only the candidate and determined that it did NOT fully
work as an independent clip.

Your task is to use the surrounding transcript to fix the boundaries.

{what_you_are_judging}

So a stretch of garbled text is NOT a reason to move a boundary: the audience
hears the real words. Move a boundary only for MEANING — missing setup, or a
payoff that has not landed yet.

You receive:

BEFORE
CANDIDATE
AFTER

Every sentence has an immutable sentence ID.

You may ONLY choose from the provided sentence IDs.
Never invent timestamps.
Never cut inside sentences.

The repaired clip must:
- make complete sense to someone who never watched the source video
- begin where the necessary setup actually begins
- end after the conclusion/payoff actually finishes
- contain ONE coherent idea/story/lesson
- be the SHORTEST version that preserves the full meaning
- be between {min_clip:.0f} and {max_clip:.0f} seconds

CRITICAL TOPIC-BOUNDARY RULE:

Do NOT extend into the next distinct topic merely to create a stronger ending.

Adjacent transcript does NOT automatically belong to the same clip.

For example, if a speaker finishes:
- explaining a framework,
- listing all promised steps/questions,
- completing a story,
- answering the original question,

and then begins a new topic, principle, anecdote, category, warning, or opinion,
STOP before that new topic.

Never append a different idea just because it sounds like a conclusion.

If the original idea is useful but naturally ends without a verbal summary,
that can still be a complete clip.

A complete list/framework does NOT require an extra motivational conclusion
after it.

Prefer:
one complete focused idea

over:
one idea + the beginning of the next idea.

Do not add background simply because it exists.

Examples:

If candidate starts:
"And so one example is quitting smoking."

look backward and include enough context to explain what smoking is an example
of.

If candidate starts:
"And so hobbies can be the same thing..."

look backward until "the same thing" has an understandable referent OR choose
a later clean independent opening if that is better.

If the opening can simply be trimmed to a later sentence that is independently
understandable, prefer trimming rather than adding unnecessary context.

If the candidate ends:
"...and the reason why..."

extend forward until that explanation finishes.

If no clean standalone clip can be produced inside {max_clip:.0f} seconds, REJECT it.

{multilingual_rules}

BLIND VERIFIER FAILURE:

{failure_reason}

ORIGINAL CANDIDATE:

{original_start} -> {original_end}

TOPIC:

{topic}

CONTEXT:

{context}
"""


# ============================================================================
# HELPERS
# ============================================================================

def _index_map(
    sentences: List[Dict[str, Any]]
) -> Dict[str, int]:

    return {
        sentence["id"]: index
        for index, sentence in enumerate(sentences)
    }


def _range_data(
    sentences: List[Dict[str, Any]],
    start_id: str,
    end_id: str,
) -> Dict[str, Any]:

    indexes = _index_map(sentences)

    if start_id not in indexes:
        raise ValueError(
            f"Unknown sentence ID: {start_id}"
        )

    if end_id not in indexes:
        raise ValueError(
            f"Unknown sentence ID: {end_id}"
        )

    start_index = indexes[start_id]
    end_index = indexes[end_id]

    if end_index < start_index:
        raise ValueError(
            "End sentence occurs before start sentence."
        )

    selected = sentences[
        start_index:end_index + 1
    ]

    start = float(
        selected[0]["start"]
    )

    end = float(
        selected[-1]["end"]
    )

    return {
        "start_index": start_index,
        "end_index": end_index,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(end - start, 3),
        "transcript": " ".join(
            sentence["text"]
            for sentence in selected
        ),
    }


def build_critic_context(
    sentences: List[Dict[str, Any]],
    start_id: str,
    end_id: str,
    before_seconds: float = 40.0,
    after_seconds: float = 40.0,
) -> Dict[str, Any]:

    indexes = _index_map(sentences)

    if (
        start_id not in indexes
        or end_id not in indexes
    ):
        raise ValueError(
            "Candidate contains unknown sentence IDs."
        )

    start_index = indexes[start_id]
    end_index = indexes[end_id]

    candidate_start = float(
        sentences[start_index]["start"]
    )

    candidate_end = float(
        sentences[end_index]["end"]
    )

    # ---------------------------------------------------------
    # Find BEFORE context
    # ---------------------------------------------------------

    context_start = start_index

    target_before = (
        candidate_start
        - before_seconds
    )

    while (
        context_start > 0
        and float(
            sentences[
                context_start - 1
            ]["end"]
        ) >= target_before
    ):
        context_start -= 1

    # ---------------------------------------------------------
    # Find AFTER context
    # ---------------------------------------------------------

    context_end = end_index

    target_after = (
        candidate_end
        + after_seconds
    )

    while (
        context_end + 1 < len(sentences)
        and float(
            sentences[
                context_end + 1
            ]["start"]
        ) <= target_after
    ):
        context_end += 1

    # ---------------------------------------------------------
    # Build prompt text
    # ---------------------------------------------------------

    lines = []

    for index in range(
        context_start,
        context_end + 1
    ):

        sentence = sentences[index]

        if index < start_index:
            zone = "BEFORE"

        elif index > end_index:
            zone = "AFTER"

        else:
            zone = "CANDIDATE"

        lines.append(
            f'{zone:9} '
            f'{sentence["id"]} '
            f'[{sentence["start"]:.2f}-'
            f'{sentence["end"]:.2f}] '
            f'{sentence["text"]}'
        )

    return {
        "context_start_index": context_start,
        "context_end_index": context_end,
        "text": "\n".join(lines),
    }


# ============================================================================
# MAIN CRITIC
# ============================================================================

def review_candidates_with_gemini(
    sentences: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:

    # Lazy imports so the file can still be imported without AI dependencies.
    from google import genai
    from google.genai import types as genai_types
    from pydantic import BaseModel
    from typing import List as TypingList

    # ------------------------------------------------------------------------
    # Gemini response schemas
    # ------------------------------------------------------------------------

    class BlindResponse(BaseModel):
        verdict: str
        opening_is_self_contained: bool
        ending_is_complete: bool
        unresolved_references: TypingList[str]
        standalone_score: int
        completeness_score: int
        reason: str

    class RepairResponse(BaseModel):
        decision: str
        start_sentence: str
        end_sentence: str
        reason: str

    # ------------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------------

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
        or os.getenv(
            "GEMINI_CRITIC_MODEL"
        )
        or os.getenv("GEMINI_MODEL")
        or "gemini-3.1-flash-lite"
    )

    client = genai.Client(
        api_key=api_key
    )

    # ------------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------------

    def generate_with_retry(
        prompt,
        config,
        stage,
        max_attempts=6,
    ):
        """
        Gemini occasionally returns 429/500/503 during temporary demand spikes.

        The Google SDK already performs some internal retries, but this gives
        Gemini a longer recovery window instead of killing the whole selector.
        """

        delays = [
            5,
            10,
            20,
            30,
            45,
            60,
        ]

        last_error = None

        for attempt in range(
            1,
            max_attempts + 1
        ):

            try:

                return (
                    client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config,
                    )
                )

            except Exception as exc:

                last_error = exc

                error_text = str(
                    exc
                ).lower()

                retryable = any(
                    marker in error_text
                    for marker in (
                        "503",
                        "unavailable",
                        "high demand",
                        "429",
                        "resource_exhausted",
                        "500",
                        "internal",
                    )
                )

                if (
                    not retryable
                    or attempt >= max_attempts
                ):
                    raise

                delay = delays[
                    min(
                        attempt - 1,
                        len(delays) - 1
                    )
                ]

                print(
                    f"  Gemini {stage} temporarily unavailable. "
                    f"Retry {attempt}/{max_attempts} "
                    f"in {delay}s..."
                )

                time.sleep(
                    delay
                )

        if last_error:
            raise last_error

        raise RuntimeError(
            "Gemini request failed."
        )

    # ------------------------------------------------------------------------
    # Gemini configs
    # ------------------------------------------------------------------------

    blind_config = (
        genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BlindResponse,
            candidate_count=1,
            temperature=0.05,
        )
    )

    repair_config = (
        genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RepairResponse,
            candidate_count=1,
            temperature=0.1,
        )
    )

    indexes = _index_map(
        sentences
    )

    # ------------------------------------------------------------------------
    # Parse Gemini structured response
    # ------------------------------------------------------------------------

    def parse_response(
        response
    ) -> Dict[str, Any]:

        parsed_obj = getattr(
            response,
            "parsed",
            None
        )

        if parsed_obj is not None:

            if hasattr(
                parsed_obj,
                "model_dump"
            ):
                return (
                    parsed_obj.model_dump()
                )

            return parsed_obj

        raw = (
            getattr(
                response,
                "text",
                ""
            )
            or ""
        ).strip()

        if not raw:
            raise RuntimeError(
                "Gemini returned empty response."
            )

        return json.loads(
            raw
        )

    # ------------------------------------------------------------------------
    # Blind verifier
    # ------------------------------------------------------------------------

    def blind_verify(
        candidate_id: str,
        range_data: Dict[str, Any],
    ) -> Dict[str, Any]:

        prompt = (
            BLIND_VERIFY_PROMPT.format(
                multilingual_rules=MULTILINGUAL_RULES,
                what_you_are_judging=WHAT_YOU_ARE_JUDGING,
                duration_rule=duration_rule_text(),
                candidate_id=candidate_id,
                duration=range_data[
                    "duration"
                ],
                transcript=range_data[
                    "transcript"
                ],
            )
        )

        response = (
            generate_with_retry(
                prompt=prompt,
                config=blind_config,
                stage="blind verification",
            )
        )

        result = parse_response(
            response
        )

        verdict = str(
            result.get(
                "verdict",
                "FAIL"
            )
        ).upper()

        unresolved = (
            result.get(
                "unresolved_references"
            )
            or []
        )

        try:
            standalone = int(
                result.get(
                    "standalone_score",
                    0
                )
            )
        except (
            TypeError,
            ValueError
        ):
            standalone = 0

        try:
            completeness = int(
                result.get(
                    "completeness_score",
                    0
                )
            )
        except (
            TypeError,
            ValueError
        ):
            completeness = 0

        strict_pass = (
            verdict == "PASS"
            and bool(
                result.get(
                    "opening_is_self_contained"
                )
            )
            and bool(
                result.get(
                    "ending_is_complete"
                )
            )
            and len(
                unresolved
            ) == 0
            and standalone >= 85
            and completeness >= 85
        )

        result[
            "strict_pass"
        ] = strict_pass

        # Good enough to publish, even if not flawless. Kept separate from
        # strict_pass so the ranking can still prefer the flawless ones.
        floor = accept_floor()
        result[
            "usable"
        ] = bool(
            standalone >= floor
            and completeness >= floor
        )

        result[
            "standalone_score"
        ] = standalone

        result[
            "completeness_score"
        ] = completeness

        result[
            "verdict"
        ] = verdict

        return result

    # ------------------------------------------------------------------------
    # Review every candidate
    # ------------------------------------------------------------------------

    reviews = []

    print()
    print(
        f"Blind Context Critic: "
        f"{len(candidates)} candidates"
    )

    print(
        f"Gemini model: "
        f"{model_name}"
    )

    for number, candidate in enumerate(
        candidates,
        start=1
    ):

        candidate_id = (
            candidate[
                "candidate_id"
            ]
        )

        original_start = (
            candidate[
                "start_sentence"
            ]
        )

        original_end = (
            candidate[
                "end_sentence"
            ]
        )

        print()
        print(
            f"Reviewing {candidate_id} "
            f"({number}/{len(candidates)})..."
        )

        original_range = (
            _range_data(
                sentences,
                original_start,
                original_end,
            )
        )

        # ====================================================================
        # PASS A
        # BLIND VERIFY ORIGINAL CANDIDATE
        # ====================================================================

        initial_verification = (
            blind_verify(
                candidate_id,
                original_range,
            )
        )

        print(
            "  Blind test: "
            f'{initial_verification.get("verdict")}'
        )

        print(
            "  Standalone: "
            f'{initial_verification.get("standalone_score")}'
        )

        print(
            "  Completeness: "
            f'{initial_verification.get("completeness_score")}'
        )

        print(
            "  Reason: "
            f'{initial_verification.get("reason")}'
        )

        final_start = (
            original_start
        )

        final_end = (
            original_end
        )

        final_range = (
            original_range
        )

        final_verification = (
            initial_verification
        )

        repair_reason = ""

        # ====================================================================
        # ALREADY GOOD
        # ====================================================================

        if (
            initial_verification[
                "strict_pass"
            ]
        ):

            decision = "ACCEPT"

            print(
                "  Result: ACCEPT "
                "(no repair needed)"
            )

        else:

            print(
                "  Result: needs repair"
            )

            # ================================================================
            # PASS B
            # SHOW CONTEXT AND REPAIR BOUNDARIES
            # ================================================================

            context = (
                build_critic_context(
                    sentences,
                    original_start,
                    original_end,
                )
            )

            failure_reason = str(
                initial_verification.get(
                    "reason",
                    ""
                )
            )

            unresolved = (
                initial_verification.get(
                    "unresolved_references",
                    []
                )
                or []
            )

            if unresolved:

                failure_reason += (
                    "\nUnresolved references: "
                    + ", ".join(
                        str(item)
                        for item
                        in unresolved
                    )
                )

            repair_prompt = (
                REPAIR_PROMPT.format(
                    multilingual_rules=MULTILINGUAL_RULES,
                    what_you_are_judging=WHAT_YOU_ARE_JUDGING,
                    min_clip=MIN_CLIP_SECONDS,
                    max_clip=MAX_CLIP_SECONDS,
                    failure_reason=(
                        failure_reason
                    ),
                    original_start=(
                        original_start
                    ),
                    original_end=(
                        original_end
                    ),
                    topic=candidate.get(
                        "topic",
                        ""
                    ),
                    context=context[
                        "text"
                    ],
                )
            )

            response = (
                generate_with_retry(
                    prompt=repair_prompt,
                    config=repair_config,
                    stage="boundary repair",
                )
            )

            repair = parse_response(
                response
            )

            repair_decision = str(
                repair.get(
                    "decision",
                    "REJECT"
                )
            ).upper()

            repair_reason = str(
                repair.get(
                    "reason",
                    ""
                )
            )

            # ================================================================
            # REPAIRER REJECTED
            # ================================================================

            if (
                repair_decision
                == "REJECT"
            ):

                decision = "REJECT"

                print(
                    "  Repairer: REJECT"
                )

            else:

                proposed_start = str(
                    repair.get(
                        "start_sentence",
                        ""
                    )
                )

                proposed_end = str(
                    repair.get(
                        "end_sentence",
                        ""
                    )
                )

                allowed_ids = {
                    sentences[i]["id"]
                    for i in range(
                        context[
                            "context_start_index"
                        ],
                        context[
                            "context_end_index"
                        ] + 1
                    )
                }

                valid_boundaries = (
                    proposed_start
                    in indexes
                    and proposed_end
                    in indexes
                    and proposed_start
                    in allowed_ids
                    and proposed_end
                    in allowed_ids
                    and indexes[
                        proposed_end
                    ]
                    >= indexes[
                        proposed_start
                    ]
                )

                # ============================================================
                # INVALID SENTENCE IDS
                # ============================================================

                if not valid_boundaries:

                    decision = (
                        "REJECT"
                    )

                    repair_reason = (
                        "Repairer returned "
                        "invalid sentence boundaries."
                    )

                    print(
                        "  Repairer: "
                        "invalid boundaries"
                    )

                else:

                    repaired_range = (
                        _range_data(
                            sentences,
                            proposed_start,
                            proposed_end,
                        )
                    )

                    # ========================================================
                    # HARD DURATION GATE
                    # ========================================================

                    if (
                        repaired_range[
                            "duration"
                        ] < MIN_CLIP_SECONDS
                        or repaired_range[
                            "duration"
                        ] > MAX_CLIP_SECONDS
                    ):

                        decision = (
                            "REJECT"
                        )

                        repair_reason = (
                            "Repaired clip falls outside "
                            f"{MIN_CLIP_SECONDS:.0f}-"
                            f"{MAX_CLIP_SECONDS:.0f} seconds."
                        )

                        print(
                            "  Repairer: "
                            "invalid duration"
                        )

                    else:

                        final_start = (
                            proposed_start
                        )

                        final_end = (
                            proposed_end
                        )

                        final_range = (
                            repaired_range
                        )

                        print(
                            "  Repair proposed: "
                            f"{original_start}"
                            f"->{original_end} "
                            f"=> "
                            f"{final_start}"
                            f"->{final_end}"
                        )

                        # ====================================================
                        # PASS C
                        # BLIND VERIFY THE REPAIRED VERSION
                        # ====================================================

                        final_verification = (
                            blind_verify(
                                candidate_id,
                                repaired_range,
                            )
                        )

                        print(
                            "  Repaired blind test: "
                            f'{final_verification.get("verdict")}'
                        )

                        print(
                            "  Repaired standalone: "
                            f'{final_verification.get("standalone_score")}'
                        )

                        print(
                            "  Repaired completeness: "
                            f'{final_verification.get("completeness_score")}'
                        )

                        print(
                            "  Repaired reason: "
                            f'{final_verification.get("reason")}'
                        )

                        # A repair that made the clip WORSE must not be kept.
                        # Measured on a real job: a candidate scoring 60/30 was
                        # "repaired" to 30/20 and the repaired version replaced
                        # it unconditionally, so the repair stage turned a
                        # borderline clip into a certain rejection.
                        if _is_worse(
                            final_verification,
                            initial_verification,
                        ):

                            print(
                                "  Repair made it worse - "
                                "keeping the original boundaries"
                            )

                            final_start = original_start
                            final_end = original_end
                            final_range = original_range
                            final_verification = (
                                initial_verification
                            )

                        if (
                            final_verification[
                                "strict_pass"
                            ]
                        ):

                            decision = (
                                "REPAIR"
                            )

                        elif (
                            final_verification[
                                "usable"
                            ]
                        ):

                            decision = (
                                "REPAIR"
                            )

                            repair_reason += (
                                " Below the flawless bar but "
                                "above the publish floor."
                            )

                        else:

                            decision = (
                                "REJECT"
                            )

                            repair_reason += (
                                " Final blind "
                                "verification failed."
                            )

        # ====================================================================
        # BUILD FINAL REVIEW RECORD
        # ====================================================================

        review = {

            "candidate_id": (
                candidate_id
            ),

            "topic": candidate.get(
                "topic",
                ""
            ),

            "decision": (
                decision
            ),

            "original_start_sentence": (
                original_start
            ),

            "original_end_sentence": (
                original_end
            ),

            "final_start_sentence": (
                final_start
            ),

            "final_end_sentence": (
                final_end
            ),

            "start": (
                final_range[
                    "start"
                ]
            ),

            "end": (
                final_range[
                    "end"
                ]
            ),

            "duration": (
                final_range[
                    "duration"
                ]
            ),

            "candidate_score": (
                candidate.get(
                    "combined_score"
                )
            ),

            "initial_verification": (
                initial_verification
            ),

            "repair_reason": (
                repair_reason
            ),

            "final_verification": (
                final_verification
            ),

            "original_transcript": (
                original_range[
                    "transcript"
                ]
            ),

            "final_transcript": (
                final_range[
                    "transcript"
                ]
            ),
        }

        reviews.append(
            review
        )

        print(
            f"  FINAL: {decision}"
        )

    return reviews