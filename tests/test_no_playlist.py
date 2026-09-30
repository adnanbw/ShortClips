"""A YouTube link can carry a playlist, and yt-dlp's default is to take it.

Click a song while a YouTube Mix or radio is running and the URL you copy has
`&list=RD…&start_radio=1` on the end. yt-dlp then treats that as a request for
the LIST, not the video.

Measured on a real 3-minute song (30-sep-2026): "Downloading 1407 items of
1407", 31 pages of playlist API calls, and the job still going a quarter of an
hour later. Nothing reports an error and nothing looks broken — the log just
scrolls through other people's videos, so it reads as a hang in whatever stage
was expected next. yt-dlp even prints the fix in passing:

    add --no-playlist to download just the video rNXmANj72ps

The Kaggle worker always passed it. The two paths on this side did not.

Read as SOURCE rather than by importing, because main.py needs cv2, mediapipe
and torch — this check has to run in a thin environment too, and the thing it
guards is one key in a dict literal.
"""
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source(*parts):
    return io.open(os.path.join(ROOT, *parts), encoding="utf-8").read()


def test_the_download_takes_one_video():
    """Set in `_base_opts`, so BOTH yt-dlp calls get it — the metadata
    `extract_info` and the download that follows it."""
    text = source("main.py")
    block = text[text.index("def _base_opts("):]
    block = block[:block.index("# Wire bytes actually pulled")]
    assert "'noplaylist': True" in block


def test_the_duration_probe_takes_one_video():
    """`skip_download` stops it fetching 1407 files, but `extract_info` still
    walks the whole list — and `duration` on a PLAYLIST is None, so the video
    is metered as unmeasurable rather than as its real length."""
    assert '"noplaylist": True' in source("cloud", "metering.py")


def test_the_kaggle_worker_still_takes_one_video():
    """It was the only path that ever had this right; keep it that way."""
    assert "--no-playlist" in source("kaggle-worker", "worker.py")
