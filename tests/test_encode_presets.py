"""The preset is a SPEED knob, not a quality knob — and it was set wrong.

At a fixed CRF, x264's preset trades encoding speed against bitrate
efficiency; CRF is what holds perceptual quality. The QUALITY tier shipped at
`-preset medium`, and every "burn a filter over a finished clip" pass uses it:
the hook overlay, the caption burn, editor effects and the vertical
passthrough. A clip gets at least two of them.

Measured on a real 55.7s 1080x1920 clip out of this pipeline, re-encoding at
crf 18:

    preset      time     size      SSIM vs source
    medium     144.3s   28.8 MB    0.99756
    fast       115.2s   30.3 MB    0.99755
    veryfast    55.0s   26.5 MB    0.99650

2.6x faster, a SMALLER file, and 0.001 of SSIM surrendered — less than the
platforms destroy re-encoding the upload.
"""
import importlib
import os

import pytest

import ffmpeg_utils


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("FFMPEG_PRESET_QUALITY", raising=False)
    monkeypatch.setenv("FFMPEG_ENCODER", "x264")
    importlib.reload(ffmpeg_utils)
    yield
    monkeypatch.delenv("FFMPEG_PRESET_QUALITY", raising=False)
    importlib.reload(ffmpeg_utils)


def _args(tier):
    return ffmpeg_utils.video_encode_args(tier)


class TestTheQualityTier:
    def test_quality_is_still_crf_18(self):
        """CRF is the quality setting and it has NOT moved."""
        args = _args(ffmpeg_utils.QUALITY)
        assert args[args.index("-crf") + 1] == "18"

    def test_the_preset_is_no_longer_medium(self):
        args = _args(ffmpeg_utils.QUALITY)
        assert args[args.index("-preset") + 1] != "medium"

    def test_the_preset_is_overridable(self, monkeypatch):
        """An escape hatch that needs no code change, for anyone who wants the
        smaller file back."""
        monkeypatch.setenv("FFMPEG_PRESET_QUALITY", "medium")
        importlib.reload(ffmpeg_utils)
        args = ffmpeg_utils.video_encode_args(ffmpeg_utils.QUALITY)
        assert args[args.index("-preset") + 1] == "medium"

    def test_an_empty_override_does_not_produce_an_empty_preset(self, monkeypatch):
        """`-preset ''` makes ffmpeg fail on every clip."""
        monkeypatch.setenv("FFMPEG_PRESET_QUALITY", "   ")
        importlib.reload(ffmpeg_utils)
        args = ffmpeg_utils.video_encode_args(ffmpeg_utils.QUALITY)
        assert args[args.index("-preset") + 1].strip()


class TestTheOtherTiersAreUntouched:
    def test_quality_fast_still_exists(self):
        args = _args(ffmpeg_utils.QUALITY_FAST)
        assert args[args.index("-crf") + 1] == "18"

    def test_delivery_still_trades_quality_for_size(self):
        args = _args(ffmpeg_utils.DELIVERY)
        assert args[args.index("-crf") + 1] == "22"

    def test_an_unknown_tier_is_still_an_error(self):
        with pytest.raises(ValueError):
            ffmpeg_utils.video_encode_args("nonsense")
