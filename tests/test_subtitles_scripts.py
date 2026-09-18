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


#: Scripts the backend/renderer images are built to support. The image
#: installs fonts-noto-core + fonts-noto-cjk for exactly these, so in an
#: integration environment a missing glyph is a BUILD REGRESSION, not a
#: platform quirk, and the suite must say so.
REQUIRED_SCRIPTS = ("devanagari", "arabic", "cyrillic",
                    "japanese", "korean", "chinese")


def _integration_mode():
    """True when this run is expected to have the image's fonts.

    Set ASR_FONTS_REQUIRED=1 in the Docker/integration run. Left unset on a
    minimal host CI, where the Noto packages are deliberately not installed and
    failing on them would only punish contributors — but there the non-Latin
    cases skip loudly rather than passing silently.
    """
    return os.environ.get("ASR_FONTS_REQUIRED", "").strip() == "1"


@pytest.mark.skipif(not os.environ.get("RUN_FFMPEG_TESTS", "1") == "1",
                    reason="needs ffmpeg + the image fonts")
class TestGlyphsActuallyRender:
    """libass silently draws a box for a missing glyph, so assert on pixels.

    A skip is not a pass. The previous version skipped whenever a script
    rendered nothing, which meant a green suite proved only that Latin worked —
    the image could have shipped with no Devanagari at all and nothing would
    have complained. In integration mode every script in REQUIRED_SCRIPTS must
    now actually put ink on the frame.
    """

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
            if _integration_mode():
                pytest.fail(f"ffmpeg could not burn a {name} subtitle: "
                            f"{result.stderr.decode(errors='replace')[-500:]}")
            pytest.skip("ffmpeg could not burn the subtitle here")
        image = Image.open(shot).convert("L")
        return sum(1 for pixel in image.getdata() if pixel > 40)

    @pytest.mark.parametrize("name", sorted(SAMPLES))
    def test_script_renders_ink(self, tmp_path, name):
        ink = self._ink(tmp_path, name, self.SAMPLES[name])

        if ink == 0 and name in REQUIRED_SCRIPTS and not _integration_mode():
            pytest.skip(f"this host has no font for {name} "
                        f"(set ASR_FONTS_REQUIRED=1 to make that a failure)")

        assert ink > 1000, (
            f"{name} rendered almost nothing ({ink} pixels) — the image is "
            f"missing a font for it, so captions in this script burn as boxes")

    def test_every_supported_script_is_covered_by_a_case(self):
        # Guards the list itself: adding a script to the image without a test
        # would otherwise leave it unverified forever.
        assert set(REQUIRED_SCRIPTS) <= set(self.SAMPLES)


@pytest.mark.skipif(not _integration_mode(),
                    reason="integration only: set ASR_FONTS_REQUIRED=1 in the "
                           "Docker run, where the image's fonts must exist")
class TestImageReallyShipsTheFonts:
    """The authoritative check, run against the built image.

    Rendering ink proves a glyph came from somewhere; fontconfig proves the
    IMAGE owns a font for the script, which is what the Dockerfile change is
    supposed to guarantee.
    """

    LANGS = {"devanagari": "hi", "arabic": "ar", "cyrillic": "ru",
             "japanese": "ja", "korean": "ko", "chinese": "zh"}

    @pytest.mark.parametrize("script,lang", sorted(LANGS.items()))
    def test_fontconfig_knows_a_font_for_the_script(self, script, lang):
        result = subprocess.run(["fc-list", f":lang={lang}", "family"],
                                capture_output=True, text=True)
        assert result.returncode == 0, "fontconfig is not installed in this image"
        families = [line for line in result.stdout.splitlines() if line.strip()]
        assert families, (
            f"no font in the image covers {script} ({lang}); captions in it "
            f"will render as boxes. Check fonts-noto-core / fonts-noto-cjk "
            f"in the Dockerfile.")

    def test_anton_falls_back_rather_than_dropping_a_glyph(self):
        # The caption default is Anton, which is Latin-only. fontconfig must
        # offer a real fallback after it, or the <default> rules in
        # fonts/openshorts-fontmap.conf are not being loaded.
        result = subprocess.run(["fc-match", "-s", "Anton:lang=hi", "family"],
                                capture_output=True, text=True)
        assert result.returncode == 0
        chain = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        assert any("Noto" in family for family in chain[:6]), chain[:6]
