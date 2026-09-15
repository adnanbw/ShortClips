"""Burned captions must be readable in the language that was actually spoken.

Two separate problems, both invisible in English:

1. Anton and the Liberation faces are Latin-only, so a Devanagari, Arabic or
   CJK caption came out as tofu boxes. Fixed in the image (fonts-noto-core +
   fonts-noto-cjk) and in fonts/openshorts-fontmap.conf, which appends Noto as
   a <default> fallback — Latin text still renders in Anton.
2. libass shapes Arabic correctly but applies bidi PER OVERRIDE-DELIMITED RUN,
   and the karaoke path wraps the active word in {\\c...}...{\\r}. That put the
   first word of an Arabic caption at the far LEFT. generate_ass now emits the
   three runs in visual order for an RTL block, and disables auto-wrap there
   because the flip is only valid for a single line.
"""
import os
import re
import subprocess

import pytest

from subtitles import generate_ass, _is_rtl_block


ARABIC = "ما الذي لا ينجح"
HEBREW = "מה לא עובד בשוק"
ENGLISH = "what does not work"
HINGLISH = "मुझे actually ये approach"


def _transcript(text, seconds=2.0):
    words = text.split()
    step = seconds / len(words)
    return {"language": "xx", "segments": [{
        "start": 0.0, "end": seconds, "text": text,
        "words": [{"word": " " + w, "start": i * step, "end": (i + 1) * step}
                  for i, w in enumerate(words)]}]}


def _events(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    return [l for l in lines if l.startswith("Dialogue:")]


def _words_in_order(event):
    """The words of one Dialogue line, left to right, tags stripped."""
    # ASS Dialogue has 9 comma-separated fields before the text.
    text = event.split(",", 9)[9]
    text = re.sub(r"\{[^}]*\}", " ", text)
    return text.split()


def _ass(tmp_path, text, **kwargs):
    out = tmp_path / "subs.ass"
    assert generate_ass(_transcript(text), 0.0, 2.0, str(out),
                        font_name="Anton", **kwargs)
    return str(out)


class TestDirectionDetection:
    @pytest.mark.parametrize("text,rtl", [
        (ARABIC, True), (HEBREW, True),
        (ENGLISH, False), (HINGLISH, False),
        ("実はこの方法がいい", False), ("это лучше", False),
        # An Arabic sentence with one English word is still an RTL line.
        ("ما الذي لا ينجح في marketing", True),
        # An English sentence quoting one Arabic word is not.
        ("the word is سوق in arabic today", False),
    ])
    def test_block_direction(self, text, rtl):
        block = _transcript(text)["segments"][0]["words"]
        assert _is_rtl_block(block) is rtl

    def test_empty_block_is_not_rtl(self):
        assert _is_rtl_block([]) is False


class TestRunOrder:
    def test_rtl_events_are_written_in_visual_order(self, tmp_path):
        """Runs are flipped; words INSIDE a run keep their logical order.

        libass still bidi-orders the text inside one run, so reversing the
        words as well would cancel itself out and put the line back in the
        wrong order — verified on screen, not just on paper.
        """
        events = _events(_ass(tmp_path, ARABIC, max_chars=40))
        words = ARABIC.split()
        for i, event in enumerate(events):
            expected = words[i + 1:] + [words[i]] + words[:i]
            assert _words_in_order(event) == expected
        # The active word of the first event is rightmost on screen.
        assert _words_in_order(events[0])[-1] == words[0]

    def test_ltr_events_keep_logical_order(self, tmp_path):
        events = _events(_ass(tmp_path, ENGLISH, max_chars=40))
        assert _words_in_order(events[0]) == ENGLISH.split()

    def test_code_switched_devanagari_keeps_logical_order(self, tmp_path):
        # Devanagari is left-to-right; the RTL flip must not touch it.
        events = _events(_ass(tmp_path, HINGLISH, max_chars=40))
        assert _words_in_order(events[0]) == HINGLISH.split()

    def test_every_word_gets_highlighted_exactly_once(self, tmp_path):
        events = _events(_ass(tmp_path, ARABIC, max_chars=40))
        assert len(events) == len(ARABIC.split())
        for event in events:
            assert event.count("{\\r}") == 1

    def test_rtl_events_disable_auto_wrap(self, tmp_path):
        # The run flip is only correct for one visual line.
        for event in _events(_ass(tmp_path, ARABIC, max_chars=40)):
            assert "{\\q2}" in event

    def test_ltr_events_do_not_touch_wrapping(self, tmp_path):
        for event in _events(_ass(tmp_path, ENGLISH, max_chars=40)):
            assert "\\q2" not in event


@pytest.mark.skipif(not os.environ.get("RUN_FFMPEG_TESTS", "1") == "1",
                    reason="needs ffmpeg + the image fonts")
class TestGlyphsActuallyRender:
    """libass silently draws a box for a missing glyph, so assert on pixels."""

    SAMPLES = {
        "latin": "this is the baseline",
        "devanagari": "मुझे actually ये approach",
        "arabic": ARABIC,
        "cyrillic": "это на самом деле",
        "japanese": "実はこの方法がいい",
        "korean": "이 방법이 더 낫다",
        "chinese": "其实我觉得更好",
        "accented": "¿Qué tal? Año café",
    }

    def _ink(self, tmp_path, name, text):
        pytest.importorskip("PIL")
        from PIL import Image

        ass = _ass(tmp_path, text, max_chars=30, fontsize=40)
        blank = tmp_path / f"{name}.mp4"
        shot = tmp_path / f"{name}.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "color=c=black:s=540x960:d=1", "-t", "1",
                        str(blank)], check=True, capture_output=True)
        fonts = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "fonts")
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(blank), "-vf",
             f"ass=filename='{ass}':fontsdir='{fonts}'",
             "-frames:v", "1", str(shot)], capture_output=True)
        if result.returncode != 0:
            pytest.skip("ffmpeg could not burn the subtitle here")
        image = Image.open(shot).convert("L")
        return sum(1 for pixel in image.getdata() if pixel > 40)

    @pytest.mark.parametrize("name", sorted(SAMPLES))
    def test_script_renders_ink(self, tmp_path, name):
        ink = self._ink(tmp_path, name, self.SAMPLES[name])
        if name != "latin" and ink == 0:
            pytest.skip("this image has no font for %s" % name)
        assert ink > 1000, f"{name} rendered almost nothing ({ink} pixels)"
