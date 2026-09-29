"""Split mode cuts a film into fixed pieces with no model anywhere.

Two things make it worth testing rather than eyeballing.

The cut positions are arithmetic, and arithmetic that is subtly wrong produces
clips that look plausible and are not: a boundary snapped past its neighbour, a
lead-in taken off the first clip so the range the user chose is not the range
they get, a trailing offcut published as a four-second "Part 7 of 7".

And the mode's whole value is what it does NOT do. If it transcribes, or
dispatches to Kaggle, or reaches the Gemini vision selector because the
transcript came back None, then it is just the normal pipeline with worse clip
selection.
"""
import io
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import split_selector as split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = io.open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()


# --- reading what the user typed --------------------------------------------

class TestTimecodes:
    """A user scrubbing a film reads timecodes off a player. Making them
    convert to seconds by hand is how a cut lands in the wrong scene."""

    @pytest.mark.parametrize("text,seconds", [
        ("90", 90.0),
        ("1:30", 90.0),
        ("00:01:30", 90.0),
        ("2:05:00", 7500.0),
        ("0", 0.0),
    ])
    def test_accepted_spellings(self, text, seconds):
        assert split.parse_timecode(text) == seconds

    @pytest.mark.parametrize("text", [None, "", "   ", "abc", "1:2:3:4:x"])
    def test_unusable_input_is_none_not_zero(self, text):
        """None means "not given"; 0 would mean "start at the beginning", and
        silently turning a typo into the latter re-cuts the whole film."""
        assert split.parse_timecode(text) is None


# --- where the cuts land -----------------------------------------------------

class TestRanges:
    def test_a_ten_minute_range_at_90s(self):
        ranges = split.build_ranges(0, 600, 90, overlap=0)
        assert len(ranges) == 7
        assert ranges[0]["start"] == 0
        assert ranges[-1]["end"] == 600

    def test_the_chosen_range_is_honoured_exactly(self):
        """Only INTERIOR boundaries may move. Snapping the outer two would
        include footage from outside the range the user picked."""
        points = [59.0, 61.0, 601.0, 119.0]
        ranges = split.build_ranges(60, 600, 90, overlap=0, points=points)
        assert ranges[0]["start"] == 60
        assert ranges[-1]["end"] == 600

    def test_overlap_is_a_lead_in_only(self):
        """The failure a viewer notices is a clip that STARTS mid-word. So the
        repeat is spent at the start of the LATER piece, and the first piece
        never gets one — it would reach before the chosen range."""
        ranges = split.build_ranges(0, 300, 90, overlap=2)
        assert ranges[0]["start"] == 0
        assert ranges[1]["start"] == 88
        assert ranges[1]["end"] == 180

    def test_a_short_tail_joins_the_piece_before_it(self):
        """195s at 90s is two pieces and 15 seconds. 15 seconds is an offcut,
        not a 'Part 3 of 3'."""
        ranges = split.build_ranges(0, 195, 90, overlap=0)
        assert len(ranges) == 2
        assert ranges[-1]["end"] == 195

    def test_a_tail_worth_keeping_is_kept(self):
        ranges = split.build_ranges(0, 200, 90, overlap=0)
        assert len(ranges) == 3

    def test_a_range_shorter_than_one_piece_is_one_piece(self):
        ranges = split.build_ranges(0, 40, 90, overlap=0)
        assert ranges == [{"start": 0, "end": 40}]

    def test_an_empty_range_yields_nothing(self):
        assert split.build_ranges(100, 100, 90) == []
        assert split.build_ranges(100, 50, 90) == []


class TestSnapping:
    def test_a_boundary_moves_to_a_nearby_pause(self):
        ranges = split.build_ranges(0, 300, 90, overlap=0, points=[92.5])
        assert ranges[0]["end"] == 92.5

    def test_a_distant_pause_is_ignored(self):
        """Past the tolerance the pieces stop being the length that was asked
        for, which is worse than a cut landing on a word."""
        far = 90 + split.SNAP_TOLERANCE + 5
        ranges = split.build_ranges(0, 300, 90, overlap=0, points=[far])
        assert ranges[0]["end"] == 90

    def test_the_nearest_pause_wins(self):
        ranges = split.build_ranges(0, 300, 90, overlap=0, points=[88.0, 91.0])
        assert ranges[0]["end"] == 91.0

    def test_boundaries_stay_ordered_and_never_collapse(self):
        """Two boundaries landing on one long pause would otherwise produce a
        zero-length clip, and ffmpeg would be asked to cut end <= start."""
        points = [90.0] * 5 + [91.0, 89.5, 180.5]
        ranges = split.build_ranges(0, 600, 90, overlap=0, points=points)
        for piece in ranges:
            assert piece["end"] - piece["start"] >= split.MIN_CLIP_SECONDS
        starts = [p["start"] for p in ranges]
        assert starts == sorted(starts)

    def test_no_pauses_at_all_is_the_plain_grid(self):
        """A continuous soundtrack must degrade to fixed cuts, not drift."""
        assert (split.build_ranges(0, 600, 90, overlap=0, points=[])
                == split.build_ranges(0, 600, 90, overlap=0))


# --- finding the pauses ------------------------------------------------------

FFMPEG_OUTPUT = """\
[silencedetect @ 0x1] silence_start: 12.5
[silencedetect @ 0x1] silence_end: 13.5 | silence_duration: 1.0
[silencedetect @ 0x1] silence_start: 40.0
[silencedetect @ 0x1] silence_end: 41.0 | silence_duration: 1.0
"""


class TestSilenceDetection:
    def _run(self, monkeypatch, stderr=FFMPEG_OUTPUT, boom=None):
        captured = {}

        def fake(command, **kwargs):
            captured["command"] = command
            if boom:
                raise boom
            return subprocess.CompletedProcess(command, 0, "", stderr)

        monkeypatch.setattr(split.subprocess, "run", fake)
        return captured

    def test_the_midpoint_of_each_pause_is_the_cut_point(self, monkeypatch):
        """Not the start of the silence: the middle is where a human cuts."""
        self._run(monkeypatch)
        assert split.silence_points("v.mp4") == [13.0, 40.5]

    def test_the_seek_offset_is_added_back(self, monkeypatch):
        """-ss goes BEFORE -i so ffmpeg seeks instead of decoding the whole
        film, and silencedetect then reports times relative to the SEEK POINT.
        Forgetting to re-add it moves every cut in the job by the trim offset,
        with nothing reporting an error."""
        captured = self._run(monkeypatch)
        points = split.silence_points("v.mp4", start=600.0, end=900.0)
        assert points == [613.0, 640.5]
        command = captured["command"]
        assert command.index("-ss") < command.index("-i"), "must be a fast seek"
        assert command.index("-to") < command.index("-i")

    def test_video_is_not_decoded(self, monkeypatch):
        """There is no reason to decode pictures to find a pause, and on a
        two-hour film that is the difference between seconds and minutes."""
        captured = self._run(monkeypatch)
        split.silence_points("v.mp4")
        assert "-vn" in captured["command"]

    def test_an_unpaired_silence_start_is_dropped(self, monkeypatch):
        """ffmpeg reports an open silence at EOF with no matching end."""
        self._run(monkeypatch, stderr="silence_start: 10.0\n")
        assert split.silence_points("v.mp4") == []

    def test_a_source_with_no_audio_is_not_an_error(self, monkeypatch):
        self._run(monkeypatch, stderr="Output file #0 does not contain any stream\n")
        assert split.silence_points("v.mp4") == []

    def test_ffmpeg_missing_or_hanging_falls_back_to_the_grid(self, monkeypatch):
        """Never raises: no pauses just means plain cuts, which still work."""
        self._run(monkeypatch, boom=FileNotFoundError("ffmpeg"))
        assert split.silence_points("v.mp4") == []
        self._run(monkeypatch, boom=subprocess.TimeoutExpired("ffmpeg", 600))
        assert split.silence_points("v.mp4") == []


# --- the clips it hands to the pipeline --------------------------------------

class TestBuildClips:
    def _build(self, monkeypatch, **kwargs):
        monkeypatch.setattr(split, "silence_points", lambda *a, **k: [])
        return split.build_clips("v.mp4", 600, title="The Film", **kwargs)

    def test_it_returns_the_shape_every_selector_returns(self, monkeypatch):
        data = self._build(monkeypatch)
        assert data["selector"] == "split"
        assert len(data["shorts"]) == 7
        assert all("start" in s and "end" in s for s in data["shorts"])

    def test_parts_are_numbered_for_a_human(self, monkeypatch):
        data = self._build(monkeypatch)
        titles = [s["video_title_for_youtube_short"] for s in data["shorts"]]
        assert titles[0] == "The Film — Part 1 of 7"
        assert titles[-1] == "The Film — Part 7 of 7"

    def test_no_hook_text_is_invented(self, monkeypatch):
        """The hook is BURNED onto the video. Writing one needs a model, this
        mode has none, and a made-up hook over someone's film is worse than
        no hook — auto_hook_clip skips an empty string."""
        data = self._build(monkeypatch)
        assert all(s["viral_hook_text"] == "" for s in data["shorts"])

    def test_vertical_output_is_fitted_whole_not_cropped(self, monkeypatch):
        """WIDE fits the whole frame over a blurred copy of itself. Cropping in
        on a face is wrong for film — the framing IS the content — and it is
        also what would drag a detector into the no-model path."""
        assert self._build(monkeypatch, output_format="vertical")["force_strategy"] == "WIDE"
        assert self._build(monkeypatch, output_format="square")["force_strategy"] == "WIDE"

    def test_horizontal_output_keeps_the_source_framing(self, monkeypatch):
        data = self._build(monkeypatch, output_format="horizontal")
        assert data["force_strategy"] is None

    def test_a_range_too_short_to_cut_is_refused_not_fudged(self, monkeypatch):
        monkeypatch.setattr(split, "silence_points", lambda *a, **k: [])
        assert split.build_clips("v.mp4", 600, start=100, end=102) is None

    def test_the_range_is_clamped_to_the_video(self, monkeypatch):
        monkeypatch.setattr(split, "silence_points", lambda *a, **k: [])
        data = split.build_clips("v.mp4", 300, start=0, end=99999, title="X")
        assert data["shorts"][-1]["end"] == 300

    def test_snapping_can_be_switched_off(self, monkeypatch):
        called = []
        monkeypatch.setattr(split, "silence_points",
                            lambda *a, **k: called.append(1) or [])
        split.build_clips("v.mp4", 600, snap_to_silence=False)
        assert not called


class TestEnvOptions:
    def test_defaults_when_nothing_is_set(self, monkeypatch):
        for name in ("SPLIT_CLIP_SECONDS", "SPLIT_OVERLAP_SECONDS",
                     "SPLIT_START", "SPLIT_END", "SPLIT_SNAP"):
            monkeypatch.delenv(name, raising=False)
        options = split.options_from_env()
        assert options["clip_seconds"] == split.DEFAULT_CLIP_SECONDS
        assert options["start"] is None and options["end"] is None
        assert options["snap_to_silence"] is True

    def test_a_malformed_number_does_not_crash_the_job(self, monkeypatch):
        monkeypatch.setenv("SPLIT_CLIP_SECONDS", "not-a-number")
        assert split.options_from_env()["clip_seconds"] == split.DEFAULT_CLIP_SECONDS


# --- what the mode must NOT do ----------------------------------------------

class TestItSkipsTheExpensiveStages:
    """The value of this mode is what it leaves out. Every one of these was a
    real cost on a one-core box: transcription is ~10 minutes, the Kaggle
    dispatch spends weekly GPU quota, and the vision selector is a Gemini call.
    """

    def test_split_mode_never_transcribes(self):
        assert "if transcript is None and not splitting:" in MAIN

    def test_split_mode_never_dispatches_to_kaggle(self):
        assert "None if splitting else kaggle_worker_start" in MAIN

    def test_the_split_branch_runs_before_the_transcript_check(self):
        """Split mode leaves `transcript` as None, so a branch placed after
        `elif transcript is not None` would fall through to get_visual_clips —
        handing a film to the Gemini VISION selector, which is both a model
        call this mode exists to avoid and a wrong answer."""
        assert MAIN.index("elif splitting:") < MAIN.index("elif transcript is not None:")

    def test_the_framing_override_reaches_the_renderer(self):
        assert "force_strategy=clips_data.get('force_strategy')" in MAIN

    def test_the_cli_accepts_the_mode_it_dispatches_on(self):
        """argparse `choices` rejects anything not listed, so a mode reachable
        only through the env var cannot be reproduced by hand — which is how
        every failing job in this repo gets investigated."""
        block = MAIN[MAIN.index("'--selector'"):]
        assert "'split'" in block[:block.index("help=")]


APP = io.open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()


class TestTheApiCarriesIt:
    def test_the_json_path_reads_the_split_fields(self):
        """A URL job posts JSON and a file job posts multipart, and the JSON
        branch re-reads every field by hand. A field added only to the endpoint
        signature is silently dropped for every YouTube link — which presents
        as the feature not working rather than as a missing line."""
        block = APP[APP.index('if "application/json" in content_type:'):]
        block = block[:block.index("# Normalize output format")]
        for field in ("split", "split_seconds", "split_start", "split_end",
                      "split_overlap", "split_snap"):
            assert f'body.get("{field}")' in block

    def test_captions_and_hook_are_turned_off_after_they_are_turned_on(self):
        """Both are set earlier in the same function from their own params.
        Split mode's overrides only win if they come later."""
        assert APP.index('env["CLIP_SELECTOR"] = "split"') > APP.index('env["AUTO_HOOK"] = "1"')
        split_block = APP[APP.index('env["CLIP_SELECTOR"] = "split"'):]
        assert 'env["AUTO_CAPTIONS"] = "0"' in split_block
        assert 'env["AUTO_HOOK"] = "0"' in split_block

    def test_a_bad_range_is_refused_before_the_job_queues(self):
        """Otherwise the user waits for a film to download to learn they typed
        the end time before the start."""
        block = APP[APP.index('if split is not None'):]
        block = block[:block.index('env["CLIP_SELECTOR"] = "split"')]
        assert "HTTPException" in block
        assert "status_code=400" in block
