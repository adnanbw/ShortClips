"""Cut a video into fixed-length pieces. No model, no transcript, no Gemini.

WHY THIS EXISTS ALONGSIDE THE MEANINGFUL SELECTOR
-------------------------------------------------
The meaningful selector answers "which moments are worth publishing", which is
the right question for a podcast or a stand-up set and the wrong one for a
film. There is no viral moment to find in a movie scene — the user has already
decided what they want by choosing a time range, and all that is left is to
chop it into pieces a platform will accept.

So this is a third selector, not a second pipeline. It emits the same
``{"shorts": [...]}`` shape every other selector emits, and everything
downstream — the cut, the framing, the clip grid, Instagram scheduling,
History, downloads — is untouched because none of it knows or cares which
selector produced the ranges.

It costs ZERO model calls. No Gemini, and no transcription either: the one
place a transcript would have helped is choosing where to cut, and
``silence_points()`` gets most of that benefit from the audio waveform alone.

WHERE THE CUTS LAND
-------------------
A flat grid at 90-second intervals lands mid-word roughly whenever there is
speech. The fix is not a model: ffmpeg's ``silencedetect`` is signal
processing, it costs one fast audio pass, and the middle of a pause is exactly
where a human would cut. Each nominal boundary is moved to the nearest such
pause within ``SNAP_TOLERANCE``; boundaries with no pause nearby stay put, so
a continuous soundtrack degrades to the flat grid rather than drifting.

OVERLAP IS A LEAD-IN, NOT A SYMMETRIC PAD
-----------------------------------------
The failure a viewer actually notices is a clip that STARTS mid-sentence — the
first thing they hear is half a word. A clip that ends a beat early reads as an
edit. So the overlap is spent entirely at the start of the following piece:
each clip begins ``overlap`` seconds before its boundary, repeating a moment
that has already been seen, and a line straddling the seam is complete in the
later clip.
"""
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence

#: The default piece length. 90s rather than 60s because the target is film:
#: a scene needs room to land, and every platform this pipeline publishes to
#: accepts it (Reels 3 min, Shorts 3 min, TikTok 10 min).
DEFAULT_CLIP_SECONDS = 90.0

#: Seconds of the previous piece repeated at the start of the next one.
DEFAULT_OVERLAP_SECONDS = 2.0

#: How far a boundary may move to find a pause. Beyond this the pieces stop
#: being the length that was asked for, which is a worse failure than a cut
#: landing on a word.
SNAP_TOLERANCE = 3.0

#: Below this, a trailing remainder is absorbed into the piece before it
#: instead of being published as its own clip. A 4-second tail is not a clip.
MIN_TAIL_SECONDS = 20.0

#: Nothing shorter than this is ever emitted, whatever the arithmetic says.
MIN_CLIP_SECONDS = 5.0

#: silencedetect parameters. -30dB is quiet-but-not-digital-silence, which is
#: what a real room sounds like between lines; 0.35s is long enough to be a
#: pause rather than the gap between two words.
SILENCE_NOISE_DB = "-30dB"
SILENCE_MIN_DURATION = 0.35

_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def parse_timecode(value: Any) -> Optional[float]:
    """Seconds from "90", "1:30" or "00:01:30". None when absent or unusable.

    Accepting all three is not indulgence: a user scrubbing a film reads
    timecodes off a player, and making them convert to seconds by hand is how
    a cut ends up in the wrong scene.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if ":" in text:
            parts = [float(p) for p in text.split(":")]
            seconds = 0.0
            for part in parts:
                seconds = seconds * 60.0 + part
            return max(0.0, seconds)
        return max(0.0, float(text))
    except (TypeError, ValueError):
        return None


def silence_points(video_path: str, start: float = 0.0,
                   end: Optional[float] = None,
                   timeout: float = 600.0) -> List[float]:
    """Midpoints of the pauses in ``[start, end]``, in SOURCE seconds.

    Never raises: a source with no audio track, an ffmpeg that is not there, a
    timeout on a three-hour film — every one of them means "no snap points",
    and the caller falls back to the flat grid, which still works.

    ``-ss``/``-to`` go BEFORE ``-i`` so ffmpeg seeks instead of decoding the
    whole file to reach the range. The consequence is that silencedetect then
    reports timestamps relative to the SEEK POINT, so ``start`` is added back.
    Getting that wrong does not fail loudly — it silently moves every cut in
    the job by the offset of the trim.
    """
    command = ["ffmpeg", "-hide_banner", "-nostats"]
    if start > 0:
        command += ["-ss", f"{start:.3f}"]
    if end is not None and end > start:
        command += ["-to", f"{end:.3f}"]
    command += [
        "-i", video_path,
        "-vn",                      # audio only: no reason to decode pictures
        "-af", f"silencedetect=noise={SILENCE_NOISE_DB}:d={SILENCE_MIN_DURATION}",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout, errors="replace")
    except Exception as exc:
        print(f"   ⚠️ Could not analyse pauses ({type(exc).__name__}: {exc}) — "
              f"cutting on the plain grid instead.")
        return []

    points: List[float] = []
    open_at: Optional[float] = None
    for line in (result.stderr or "").splitlines():
        found = _SILENCE_START.search(line)
        if found:
            open_at = float(found.group(1))
            continue
        found = _SILENCE_END.search(line)
        if found and open_at is not None:
            closed_at = float(found.group(1))
            if closed_at > open_at:
                points.append(start + (open_at + closed_at) / 2.0)
            open_at = None
    return points


def snap(boundary: float, points: Sequence[float],
         tolerance: float = SNAP_TOLERANCE) -> float:
    """The pause nearest ``boundary``, or ``boundary`` when none is near."""
    best = boundary
    best_distance = tolerance
    for point in points:
        distance = abs(point - boundary)
        if distance < best_distance:
            best, best_distance = point, distance
    return best


def plan_boundaries(start: float, end: float, clip_seconds: float,
                    points: Sequence[float] = ()) -> List[float]:
    """Cut points across ``[start, end]``, snapped to pauses where possible.

    The returned list always begins at ``start`` and ends at ``end``: the range
    the user selected is honoured exactly at its edges, and only the INTERIOR
    boundaries move. Snapping the outer two would quietly include footage from
    outside the range they chose.
    """
    span = end - start
    if span <= 0:
        return []
    if span <= clip_seconds:
        return [start, end]

    count = int(span // clip_seconds)
    remainder = span - count * clip_seconds
    # A short tail joins the piece before it rather than shipping as its own
    # clip: at 90s pieces a 7-second remainder is not a clip, it is an offcut.
    if remainder >= MIN_TAIL_SECONDS:
        count += 1

    boundaries = [start]
    for index in range(1, count):
        nominal = start + index * clip_seconds
        moved = snap(nominal, points)
        # Monotonic and never degenerate: two boundaries snapping onto the same
        # long pause would otherwise produce a zero-length clip.
        if moved - boundaries[-1] < MIN_CLIP_SECONDS:
            moved = nominal
        if moved - boundaries[-1] < MIN_CLIP_SECONDS:
            continue
        if end - moved < MIN_CLIP_SECONDS:
            continue
        boundaries.append(moved)
    boundaries.append(end)
    return boundaries


def build_ranges(start: float, end: float,
                 clip_seconds: float = DEFAULT_CLIP_SECONDS,
                 overlap: float = DEFAULT_OVERLAP_SECONDS,
                 points: Sequence[float] = ()) -> List[Dict[str, float]]:
    """The final clip ranges. Overlap is spent as a LEAD-IN (see module docs)."""
    boundaries = plan_boundaries(start, end, clip_seconds, points)
    ranges: List[Dict[str, float]] = []
    for index in range(len(boundaries) - 1):
        piece_start = boundaries[index]
        if index > 0:
            piece_start = max(start, piece_start - max(0.0, overlap))
        ranges.append({"start": round(piece_start, 3),
                       "end": round(boundaries[index + 1], 3)})
    return ranges


#: Vertical and square output would normally run the scene classifier and the
#: face tracker to decide a crop. Split mode never does: it is defined as the
#: no-model path, and on the material it exists for — film — cropping in on a
#: face is actively wrong, because the framing IS the content. WIDE is the
#: GENERAL layout with side-cropping disabled: the whole picture, fitted into
#: the output frame over a blurred copy of itself. It needs no camera
#: trajectory, so it costs one encode and no inference.
FORCE_STRATEGY_BY_FORMAT = {
    "vertical": "WIDE",
    "auto": "WIDE",
    "square": "WIDE",
    # horizontal keeps the source framing; render_clip passes it straight
    # through and never reaches a strategy at all.
}


def build_clips(video_path: str, duration: float,
                clip_seconds: float = DEFAULT_CLIP_SECONDS,
                start: Optional[float] = None,
                end: Optional[float] = None,
                overlap: float = DEFAULT_OVERLAP_SECONDS,
                snap_to_silence: bool = True,
                output_format: str = "vertical",
                title: str = "") -> Optional[Dict[str, Any]]:
    """A ``clips_data`` dict in the shape every other selector returns.

    Returns None when the selected range cannot yield a single clip, which the
    caller reports as a real reason rather than "the AI found nothing" — there
    was no AI, and the user picked the range.
    """
    span_start = max(0.0, start if start is not None else 0.0)
    span_end = min(duration, end if end is not None else duration)
    if span_end - span_start < MIN_CLIP_SECONDS:
        print(f"❌ The selected range is {span_end - span_start:.1f}s — too "
              f"short to cut (minimum {MIN_CLIP_SECONDS:.0f}s).")
        return None

    clip_seconds = max(MIN_CLIP_SECONDS, float(clip_seconds))
    points: List[float] = []
    if snap_to_silence:
        print(f"   🔇 Looking for pauses to cut on...")
        points = silence_points(video_path, span_start, span_end)
        print(f"   🔇 {len(points)} pause(s) found in the selected range.")

    ranges = build_ranges(span_start, span_end, clip_seconds, overlap, points)
    if not ranges:
        print("❌ The selected range produced no clips.")
        return None

    total = len(ranges)
    base = (title or "Clip").strip()
    shorts = []
    for index, piece in enumerate(ranges, start=1):
        shorts.append({
            **piece,
            "candidate_id": f"split-{index}",
            "video_title_for_youtube_short": f"{base} — Part {index} of {total}",
            # Deliberately empty. The hook is a BURNED overlay and writing one
            # needs a model; split mode has none, and an invented hook over
            # someone's film is worse than no hook. auto_hook_clip skips an
            # empty string.
            "viral_hook_text": "",
            "video_description_for_instagram": f"Part {index} of {total}",
            "video_description_for_tiktok": f"Part {index} of {total}",
            "split_part": index,
            "split_of": total,
        })
        print(f"   ✂️ Part {index}/{total}: {piece['start']:.2f}s -> "
              f"{piece['end']:.2f}s ({piece['end'] - piece['start']:.1f}s)")

    return {
        "shorts": shorts,
        "selector": "split",
        "force_strategy": FORCE_STRATEGY_BY_FORMAT.get(output_format),
        "split": {
            "clip_seconds": clip_seconds,
            "overlap": overlap,
            "range": [round(span_start, 3), round(span_end, 3)],
            "snapped": bool(points),
        },
    }


def options_from_env() -> Dict[str, Any]:
    """Split parameters as app.py sets them for one job."""
    def number(name, fallback):
        try:
            return float(os.environ.get(name, "").strip() or fallback)
        except (TypeError, ValueError):
            return fallback

    return {
        "clip_seconds": number("SPLIT_CLIP_SECONDS", DEFAULT_CLIP_SECONDS),
        "overlap": number("SPLIT_OVERLAP_SECONDS", DEFAULT_OVERLAP_SECONDS),
        "start": parse_timecode(os.environ.get("SPLIT_START")),
        "end": parse_timecode(os.environ.get("SPLIT_END")),
        "snap_to_silence": os.environ.get("SPLIT_SNAP", "1").strip() != "0",
    }
