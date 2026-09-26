"""The hook is burned into the video by a renderer with one Latin-only font.

hooks.create_hook_image draws with PIL, which has no fontconfig fallback, so a
glyph the font lacks is a tofu box — which is what job b975769f shipped: a
Devanagari hook as ☐☐☐☐ with its emoji rendering perfectly beside it. Captions
escape that by being transliterated; the hook and title escape it by being
written in English before they are ever drawn.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import meaningful_metadata as mm


def test_english_is_the_default(monkeypatch):
    monkeypatch.delenv("METADATA_LANGUAGE", raising=False)
    assert mm.metadata_in_english()
    assert "in ENGLISH" in mm.metadata_language_rule()


def test_speaker_restores_the_original_behaviour(monkeypatch):
    """The escape hatch matters: for a Spanish or Portuguese deployment the
    audio is already Latin-script and English metadata over it is a regression,
    not a fix."""
    monkeypatch.setenv("METADATA_LANGUAGE", "speaker")
    assert not mm.metadata_in_english()
    rule = mm.metadata_language_rule()
    assert "same language the speaker actually uses" in rule
    assert "in ENGLISH" not in rule


def test_an_unrecognised_value_stays_on_the_safe_default(monkeypatch):
    monkeypatch.setenv("METADATA_LANGUAGE", "hinglish")
    assert mm.metadata_in_english()


def test_the_rule_reaches_the_prompt(monkeypatch):
    """A placeholder that is formatted but never filled raises KeyError at the
    worst possible moment — after the clips are rendered."""
    monkeypatch.delenv("METADATA_LANGUAGE", raising=False)
    prompt = mm.METADATA_PROMPT.format(
        multilingual_rules="(rules)",
        metadata_language=mm.metadata_language_rule(),
        language="hi",
        transcript="(transcript)",
    )
    assert "in ENGLISH" in prompt
    assert "{" not in prompt.replace("{{", "").replace("}}", "")


def test_every_field_defers_to_the_language_block():
    """The four per-field bullets used to each say "same language as the
    transcript", which silently outranked anything the block below them said."""
    assert "same language as transcript" not in mm.METADATA_PROMPT
    assert "same language as the transcript" not in mm.METADATA_PROMPT
    assert mm.METADATA_PROMPT.count("METADATA LANGUAGE below") == 4


# ---------------------------------------------------------------------------
# hook_grounding rewrites the same two burned fields
# ---------------------------------------------------------------------------

def test_grounded_hook_is_told_english_too(monkeypatch):
    monkeypatch.delenv("METADATA_LANGUAGE", raising=False)
    assert mm.metadata_language_target("hi") == "English"


def test_grounded_hook_follows_the_speaker_setting(monkeypatch):
    monkeypatch.setenv("METADATA_LANGUAGE", "speaker")
    assert mm.metadata_language_target("hi") == "hi"
    assert mm.metadata_language_target("") == "unknown"


def test_the_grounded_hook_prompt_writes_in_that_language():
    import gemini_worker
    prompt = gemini_worker.GROUNDED_HOOK_PROMPT
    # It is no longer "the transcript's language" — that name is what made this
    # stage capable of putting Devanagari back onto a clip after the metadata
    # writer had taken it off.
    assert "TRANSCRIPT_LANGUAGE" not in prompt
    assert prompt.count("WRITE_IN") == 3
