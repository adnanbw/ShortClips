"""Burned-in text is written in the LATIN alphabet, whatever the speaker used.

This is transliteration, never translation. "मैंने वह किया" becomes
"maine wah kiya" — the words the speaker actually said, spelled the way their
own audience types them. It never becomes "I did that": the clip's timing, its
jokes and its metadata are all anchored to the real words, and a caption that
says something different from the audio is worse than one in the wrong script.

WHY THIS IS A GEMINI CALL AND NOT A LIBRARY. Deterministic transliterators
(unidecode, ITRANS, Harvard-Kyoto) romanise CODEPOINTS, and the speech this
product actually gets is code-switched: a Hinglish transcript writes English
loanwords in Devanagari. Measured on one line of the real stand-up,
"डिसकनेक्टिंग फ्लाइट" comes back as ``ddisknekttiNg phlaaitt`` from unidecode
and ``DisakanekTiMga phlAiTa`` from ITRANS, where the correct answer is
"disconnecting flight" — the model is the only thing that knows the word was
English before someone spelled it in another script. That is also why the
fallback here is the ORIGINAL TEXT and not a library: Devanagari a viewer can
read beats Latin nobody can.

THE OUTPUT IS ALIGNED TO WORD TIMINGS, so the contract is one word in, one
word out. Captions are karaoke — every word has its own start and end and gets
highlighted on its own — so a model that merges two words or splits one
silently destroys the timing of everything after it in the block. Every chunk
is validated for length and every word for script and token count, and anything
that fails keeps its original text rather than shifting the line.
"""
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence

from asr_quality import script_family

#: Words per Gemini call. Words are sent IN ORDER rather than deduplicated:
#: the surrounding words are what let the model tell a Devanagari-spelled
#: English loanword from a Hindi word that happens to look like one.
CHUNK_WORDS = 300

_WHITESPACE_RE = re.compile(r"\s")


def transliteration_enabled() -> bool:
    """CAPTION_SCRIPT=original restores captions in the speaker's own script."""
    return (os.environ.get("CAPTION_SCRIPT", "").strip().lower()
            or "latin") != "original"


def needs_transliteration(text: str) -> bool:
    """True when `text` contains letters outside the Latin alphabet.

    Digits, punctuation and emoji are script-less (`script_family` returns
    None) and never trigger this on their own — a caption of "2026 😂" is
    already Latin-renderable.
    """
    for ch in text or "":
        family = script_family(ch)
        if family is not None and family != "latin":
            return True
    return False


def _is_latin_output(text: str) -> bool:
    """True when nothing in `text` is a non-Latin letter."""
    return not needs_transliteration(text)


PROMPT = """You are TRANSLITERATING a speech transcript into the Latin alphabet.

You are NOT translating. The words must stay the words the speaker said.

INPUT: a JSON array of {count} words, in the order they were spoken, from a
transcript whose detected language is: {language}

OUTPUT: a JSON array of EXACTLY {count} strings, in the SAME order, each one
the same word written in the Latin alphabet.

RULES

1. NEVER translate. "मैंने वह किया" -> "maine wah kiya", never "I did that".
   "क्योंकि" -> "kyunki", never "because".

2. A word that is an English (or other Latin-script) word merely SPELLED in a
   non-Latin script is written back in its normal spelling, because that is
   the word the speaker actually said:
       फ्लाइट        -> flight
       एक्सपीरियंस    -> experience
       प्रॉब्लम       -> problem
       इंस्ट्रक्टर     -> instructor
   Do not write "phlaait" or "eksperiyans". Those are not words.

3. Everything else is spelled the way native speakers type their own language
   in Latin script — Hinglish, romaji, romanised Arabic — not by an academic
   transliteration standard. No diacritics, no capital letters in the middle
   of a word, no ā/ṭ/ñ. Plain everyday letters only.

4. ONE INPUT WORD PRODUCES ONE OUTPUT WORD. Never merge two inputs into one
   output, never split one input across two outputs, never add or remove an
   element. These words carry individual video caption timings; a length
   change corrupts every caption after it.

5. Keep punctuation that is attached to the word ("किया।" -> "kiya.") and
   convert Devanagari danda, CJK and Arabic punctuation to its Latin
   equivalent.

6. A word already written in the Latin alphabet is returned COMPLETELY
   UNCHANGED, including its case and punctuation.

7. Output only Latin letters, digits and ASCII punctuation.

WORDS:

{words}
"""


def _gemini_chunk(words: Sequence[str], language: str, api_key: str,
                  model_name: str) -> Optional[List[str]]:
    """Transliterate one chunk. Returns None when the call is unusable."""
    from google import genai
    from google.genai import types as genai_types
    from pydantic import BaseModel

    class Romanized(BaseModel):
        words: List[str]

    client = genai.Client(api_key=api_key)
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Romanized,
        candidate_count=1,
        # Transliteration has one right answer per word; sampling only invents
        # spellings that disagree with the same word three lines later.
        temperature=0.0,
    )
    prompt = PROMPT.format(
        count=len(words),
        language=language or "unknown",
        words=json.dumps(list(words), ensure_ascii=False, indent=1),
    )
    response = client.models.generate_content(
        model=model_name, contents=prompt, config=config)

    payload = json.loads(response.text)
    result = payload.get("words")
    if not isinstance(result, list):
        return None
    return [str(item) for item in result]


def _accept(original: str, candidate: str) -> bool:
    """Whether one transliterated word may replace its original.

    Rejection keeps the original word, which costs the caption its script but
    never its timing.
    """
    candidate = candidate.strip()
    if not candidate:
        return False
    if not _is_latin_output(candidate):
        return False          # still in the source script: nothing gained
    if _WHITESPACE_RE.search(candidate) and not _WHITESPACE_RE.search(original):
        return False          # one word in, two words out — a merge or a gloss
    return True


def transliterate_words(
    words: Sequence[str],
    language: str = "",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, str]:
    """Map every non-Latin word in `words` to its Latin spelling.

    Words already in the Latin alphabet are not sent and not returned. Any word
    the model fails on is simply absent from the mapping, so callers fall back
    to the original text per word rather than per transcript.
    """
    pending = [w for w in words if needs_transliteration(w)]
    if not pending:
        return {}

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY.")
    model_name = (model_name
                  or os.getenv("GEMINI_TRANSLITERATE_MODEL")
                  or os.getenv("GEMINI_MODEL")
                  or "gemini-3.1-flash-lite")

    mapping: Dict[str, str] = {}
    for start in range(0, len(pending), CHUNK_WORDS):
        _resolve(pending[start:start + CHUNK_WORDS], language, api_key,
                 model_name, mapping)
    return mapping


def _resolve(chunk: Sequence[str], language: str, api_key: str,
             model_name: str, mapping: Dict[str, str]) -> None:
    """Transliterate one chunk into `mapping`, HALVING it on a failure.

    The length check belongs immediately above the zip it protects, and not
    inside the call that produced the list: zip() truncates to the shorter side
    without complaining, which is precisely the silent mis-alignment this module
    exists to prevent.

    A mismatched chunk cannot be zipped at ALL — we do not know which word the
    model added or dropped, so every pairing after it would be a guess — but
    discarding it whole is far too blunt. On a real 12-minute job one chunk came
    back 301 words for 300 and took 300 words with it, which put Devanagari into
    the last seconds of a published clip. Splitting and retrying turns "300
    words lost" into at most one, and the halves are also easier questions: the
    model miscounts long lists, not short ones.
    """
    if not chunk:
        return
    try:
        result = _gemini_chunk(chunk, language, api_key, model_name)
    except Exception as e:
        print(f"   ⚠️ Transliteration call failed "
              f"({type(e).__name__}: {e}) — retrying {len(chunk)} word(s) "
              f"in smaller pieces.")
        result = None

    if result is not None and len(result) == len(chunk):
        for original, candidate in zip(chunk, result):
            # First spelling wins, so the same word reads the same way in
            # every caption even when two chunks disagree about it.
            if original not in mapping and _accept(original, candidate):
                mapping[original] = candidate.strip()
        return

    if len(chunk) == 1:
        # Nothing left to split. One word keeps its script; nothing shifts.
        print(f"   ⚠️ Could not transliterate one word — it keeps its script.")
        return

    got = "no array" if result is None else str(len(result))
    print(f"   ⚠️ Transliteration returned {got} word(s) for {len(chunk)} "
          f"sent — splitting and retrying.")
    middle = len(chunk) // 2
    _resolve(chunk[:middle], language, api_key, model_name, mapping)
    _resolve(chunk[middle:], language, api_key, model_name, mapping)


def annotate_transcript(
    transcript: dict,
    language: str = "",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> int:
    """Add a `latin` spelling to every non-Latin WORD of a transcript, in place.

    Only the word timings are annotated. Segment `text` is deliberately left
    alone: the clip selector, the critic and the metadata writer read that, and
    they must reason about what was really said, in the script it was said in.

    Returns the number of words annotated. Re-running is cheap and safe —
    already-annotated words are skipped, which is what makes a resumed job free.
    """
    if not transliteration_enabled():
        return 0
    segments = (transcript or {}).get("segments") or []

    targets = []
    for segment in segments:
        for word in segment.get("words") or []:
            text = word.get("word")
            if not isinstance(text, str) or word.get("latin"):
                continue
            if needs_transliteration(text):
                targets.append(word)
    if not targets:
        return 0

    language = language or (transcript or {}).get("language") or ""
    mapping = transliterate_words(
        [w["word"] for w in targets], language=language,
        api_key=api_key, model_name=model_name)
    if not mapping:
        return 0

    converted = 0
    for word in targets:
        latin = mapping.get(word["word"])
        if latin:
            # faster-whisper marks word boundaries with a LEADING SPACE and
            # merge_continuation_words reads it, so the annotation has to carry
            # the same signal or every word would glue onto the previous one.
            prefix = " " if word["word"].startswith(" ") else ""
            word["latin"] = prefix + latin
            converted += 1
    return converted


def to_latin_text(
    text: str,
    language: str = "",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """Transliterate a short piece of free text (a hook, a title).

    Word-for-word like the caption path, so a hook cannot quietly become a
    translation of itself either. Returns `text` unchanged when it is already
    Latin or when the model could not do the whole thing.
    """
    if not text or not needs_transliteration(text):
        return text
    tokens = text.split(" ")
    mapping = transliterate_words(
        tokens, language=language, api_key=api_key, model_name=model_name)
    out = [mapping.get(token, token) for token in tokens]
    return " ".join(out)


def caption_text(word: dict) -> str:
    """The text to BURN for one transcript word: its Latin spelling if it has
    one, otherwise exactly what was said."""
    latin = word.get("latin")
    if isinstance(latin, str) and latin.strip():
        return latin
    return word.get("word", "")


def iter_transcript_words(transcript: dict) -> Iterable[dict]:
    for segment in (transcript or {}).get("segments") or []:
        for word in segment.get("words") or []:
            yield word
