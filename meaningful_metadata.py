import json
import os
import time
from typing import Any, Dict, List, Optional

from meaningful_selector import MULTILINGUAL_RULES


# ============================================================================
# METADATA LANGUAGE
# ============================================================================
#
# The hook is BURNED INTO THE VIDEO, and the renderer that burns it
# (hooks.create_hook_image) draws with PIL and one font file. PIL has no
# fontconfig fallback, so a glyph that font lacks is a tofu box — which is
# exactly what a Hindi hook came out as on job b975769f, while the emoji beside
# it rendered fine because emoji get their own font by hand. Captions escape
# this by being transliterated (transliterate.py); the hook and title escape it
# by being written in English in the first place.
#
# That is a product decision as much as a rendering one: the title and
# descriptions are read on YouTube, TikTok and Instagram search, where the
# creator asked for English.
#
# ENGLISH is the default. METADATA_LANGUAGE=speaker restores the original
# behaviour — metadata in whatever language the speaker used — which is the
# right setting for a deployment whose creators publish in Spanish, Portuguese
# or any other Latin-script language, where English metadata over
# native-language audio is a regression rather than a fix.

_LANGUAGE_ENGLISH = """
METADATA LANGUAGE:

Write EVERY field in ENGLISH, whatever language the transcript is in.

This is the one place a translation is wanted: the hook is burned onto the
video in a Latin-only font and the title is read in platform search. The words
of the clip itself are untouched by this — you are naming the moment, not
re-voicing it.

Keep names, places and brands as they are. Never claim the clip is in English,
and never invent detail that the transcript does not contain just because it
reads better in English.
""".strip()

_LANGUAGE_SPEAKER = """
METADATA LANGUAGE:

Write EVERY field in the same language the speaker actually uses in this
transcript. Do not translate the clip into English.

If the speaker code-switches (for example Hindi with English words mixed in),
write the metadata the same way the speaker talks — that is what the audience
for this clip reads. Hashtags may stay in their usual Latin form.
""".strip()


def metadata_in_english() -> bool:
    return (os.environ.get("METADATA_LANGUAGE", "").strip().lower()
            or "english") != "speaker"


def metadata_language_rule() -> str:
    """The METADATA LANGUAGE block the prompt carries for this deployment."""
    return _LANGUAGE_ENGLISH if metadata_in_english() else _LANGUAGE_SPEAKER


def metadata_language_target(detected_language: str) -> str:
    """The language a prompt should be TOLD to write hooks and titles in.

    hook_grounding rewrites the same two burned fields from the clip's frames,
    so it has to obey the same policy or it would put the source script back
    onto the video after this module took it off.
    """
    if metadata_in_english():
        return "English"
    return detected_language or "unknown"


# ============================================================================
# METADATA PROMPT
# ============================================================================

METADATA_PROMPT = """
You are writing publishing metadata for ONE short-form video.

CRITICAL RULE:

You are seeing EXACTLY the transcript that the viewer will hear.

You have NO access to:
- the rest of the source video
- earlier transcript
- later transcript
- the original video's title
- any external context

Therefore every title, hook and description MUST be grounded entirely in the
supplied transcript.

Never mention, promise, imply or tease information that does not actually exist
inside this clip.

Do NOT exaggerate the clip into a different topic just because that might sound
more viral.

For example, if the clip mainly discusses quitting smoking as an analogy, do
not title it "How social media destroys dopamine" unless that exact idea is
actually explained inside the supplied transcript.

GOAL:

Create compelling short-form metadata without changing the meaning.

viral_hook_text:
- maximum 10 words
- designed as an on-screen opening text overlay
- curiosity-driven but truthful
- concrete rather than generic
- written in the language required by METADATA LANGUAGE below
- do not use fake statistics
- do not invent claims
- emojis are optional, maximum 1

video_title_for_youtube_short:
- maximum 100 characters
- clear and curiosity-driven
- accurately represents the actual clip
- written in the language required by METADATA LANGUAGE below
- no fake claims

video_description_for_tiktok:
- 1-2 short sentences
- describe/tease what this exact clip contains
- 3-5 relevant hashtags
- no generic spam hashtags
- no fake CTA such as "comment X and I'll send you..."
- written in the language required by METADATA LANGUAGE below

video_description_for_instagram:
- 1-2 short sentences
- useful and natural
- 3-5 relevant hashtags
- no claims outside the transcript
- written in the language required by METADATA LANGUAGE below

QUALITY RULE:

A boring but accurate title is better than a viral title that misrepresents
the clip.

{multilingual_rules}

{metadata_language}

TRANSCRIPT LANGUAGE (detected, may be approximate for code-switched speech):
{language}

FINAL APPROVED CLIP TRANSCRIPT:

{transcript}
"""


# ============================================================================
# HELPERS
# ============================================================================

def _safe_int(
    value: Any,
    default: int = 0
) -> int:

    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _clamp_score(
    value: Any
) -> int:

    return max(
        0,
        min(
            100,
            _safe_int(value)
        )
    )


def calculate_final_rank_score(
    candidate: Dict[str, Any]
) -> float:
    """
    Ranking after semantic verification.

    Our priority remains:
    standalone meaning > completeness > original candidate quality.
    """

    verification = (
        candidate.get("final_verification")
        or {}
    )

    standalone = _clamp_score(
        verification.get(
            "standalone_score"
        )
    )

    completeness = _clamp_score(
        verification.get(
            "completeness_score"
        )
    )

    candidate_score = _clamp_score(
        candidate.get(
            "candidate_score"
        )
    )

    score = (
        standalone * 0.45
        + completeness * 0.30
        + candidate_score * 0.25
    )

    return round(
        score,
        2
    )


def _trim_title(
    value: Any,
    max_chars: int = 100,
) -> str:

    text = str(
        value or ""
    ).strip()

    if len(text) <= max_chars:
        return text

    shortened = text[
        :max_chars
    ].rstrip()

    # Avoid leaving half a word.
    if " " in shortened:
        shortened = shortened.rsplit(
            " ",
            1
        )[0]

    return shortened.rstrip(
        " ,.;:-"
    )


def _trim_hook_words(
    value: Any,
    max_words: int = 10,
) -> str:

    text = str(
        value or ""
    ).strip()

    words = text.split()

    if len(words) <= max_words:
        return text

    return " ".join(
        words[:max_words]
    ).rstrip(
        " ,.;:-"
    )


# ============================================================================
# GEMINI METADATA GENERATOR
# ============================================================================

def generate_metadata_with_gemini(
    candidates: List[Dict[str, Any]],
    language: str = "unknown",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:

    from google import genai
    from google.genai import types as genai_types
    from pydantic import BaseModel

    # ------------------------------------------------------------------------
    # Structured response
    # ------------------------------------------------------------------------

    class MetadataResponse(BaseModel):
        viral_hook_text: str
        video_title_for_youtube_short: str
        video_description_for_tiktok: str
        video_description_for_instagram: str

    # ------------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------------

    api_key = (
        api_key
        or os.getenv(
            "GEMINI_API_KEY"
        )
    )

    if not api_key:

        raise RuntimeError(
            "Missing GEMINI_API_KEY."
        )

    model_name = (
        model_name
        or os.getenv(
            "GEMINI_METADATA_MODEL"
        )
        or os.getenv(
            "GEMINI_MODEL"
        )
        or "gemini-3.1-flash-lite"
    )

    client = genai.Client(
        api_key=api_key
    )

    config = (
        genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MetadataResponse,
            candidate_count=1,
            temperature=0.55,
        )
    )

    # ------------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------------

    def generate_with_retry(
        prompt: str,
        max_attempts: int = 6,
    ):

        # Shared policy (gemini_calls): paced under the per-minute quota and
        # retried PER CALL, so one rate limit never discards finished work.
        import gemini_calls
        try:
            return gemini_calls.call(
                client,
                model=model_name,
                contents=prompt,
                config=config,
            )
        except Exception:
            print(f"  Gemini metadata failed after "
                  f"{gemini_calls.attempts()} attempts.")
            raise

    # ------------------------------------------------------------------------
    # Response parser
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
                "Gemini returned empty metadata response."
            )

        return json.loads(
            raw
        )

    # ------------------------------------------------------------------------
    # Generate one metadata package for each approved clip
    # ------------------------------------------------------------------------

    results = []

    print()
    print(
        f"Metadata Generator: "
        f"{len(candidates)} clips"
    )

    print(
        f"Gemini model: "
        f"{model_name}"
    )

    for number, candidate in enumerate(
        candidates,
        start=1
    ):

        candidate_id = str(
            candidate.get(
                "candidate_id",
                f"C{number:03d}"
            )
        )

        transcript = str(
            candidate.get(
                "final_transcript",
                ""
            )
        ).strip()

        if not transcript:

            print(
                f"Skipping {candidate_id}: "
                "empty final transcript"
            )

            continue

        print()
        print(
            f"Generating metadata for "
            f"{candidate_id} "
            f"({number}/{len(candidates)})..."
        )

        # IMPORTANT:
        # We deliberately do NOT provide candidate topic,
        # surrounding context or source title here.
        prompt = (
            METADATA_PROMPT.format(
                multilingual_rules=MULTILINGUAL_RULES,
                metadata_language=metadata_language_rule(),
                language=language,
                transcript=transcript,
            )
        )

        response = (
            generate_with_retry(
                prompt
            )
        )

        metadata = (
            parse_response(
                response
            )
        )

        final_rank_score = (
            calculate_final_rank_score(
                candidate
            )
        )

        title = _trim_title(
            metadata.get(
                "video_title_for_youtube_short"
            )
        )

        hook = _trim_hook_words(
            metadata.get(
                "viral_hook_text"
            )
        )

        short = {
            # ------------------------------------------------------------
            # OpenShorts-compatible fields
            # ------------------------------------------------------------

            "start": float(
                candidate[
                    "start"
                ]
            ),

            "end": float(
                candidate[
                    "end"
                ]
            ),

            "predicted_score": (
                _clamp_score(
                    final_rank_score
                )
            ),

            "video_description_for_tiktok": str(
                metadata.get(
                    "video_description_for_tiktok",
                    ""
                )
            ).strip(),

            "video_description_for_instagram": str(
                metadata.get(
                    "video_description_for_instagram",
                    ""
                )
            ).strip(),

            "video_title_for_youtube_short": (
                title
            ),

            "viral_hook_text": (
                hook
            ),

            # ------------------------------------------------------------
            # Our debug / traceability fields
            # ------------------------------------------------------------

            "candidate_id": (
                candidate_id
            ),

            "final_start_sentence": (
                candidate.get(
                    "final_start_sentence"
                )
            ),

            "final_end_sentence": (
                candidate.get(
                    "final_end_sentence"
                )
            ),

            "duration": float(
                candidate.get(
                    "duration",
                    float(
                        candidate["end"]
                    )
                    - float(
                        candidate["start"]
                    )
                )
            ),

            "semantic_rank_score": (
                final_rank_score
            ),

            "final_transcript": (
                transcript
            ),
        }

        results.append(
            short
        )

        print(
            f'  Hook: {hook}'
        )

        print(
            f'  Title: {title}'
        )

        print(
            f'  Semantic score: '
            f'{final_rank_score}'
        )

    return results