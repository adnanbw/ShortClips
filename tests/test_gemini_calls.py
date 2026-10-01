"""A per-minute quota spent in a burst, then thrown away.

Measured on a 50-minute video (30-sep-2026). The Candidate Finder needs 22
windows, one Gemini call each; the free tier allows 15 per minute. The job made
13 calls in a few seconds, hit 429 on the 14th, and
`meaningful_pipeline._retry_stage` restarted the WHOLE stage from window 1 —
discarding the 13 that had succeeded. Attempt two redid twelve before failing;
attempt three died on window 2. Twenty-six calls, nothing produced.

The quota was never exhausted for the day. A per-MINUTE allowance was spent in
a burst, and then the results were deleted. So: pace the calls, retry one CALL
at a time, and let a window that still cannot be analysed be skipped rather
than take the stage with it.
"""
import io
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gemini_calls

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source(name):
    return io.open(os.path.join(ROOT, name), encoding="utf-8").read()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("GEMINI_RPM", "GEMINI_CALL_ATTEMPTS"):
        monkeypatch.delenv(name, raising=False)
    gemini_calls.reset()
    yield
    gemini_calls.reset()


class _Client:
    """Stands in for genai.Client: replays a script of results/exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

        class _Models:
            def generate_content(_self, **kwargs):
                self.calls += 1
                item = self.script.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

        self.models = _Models()


def _rate_limit(delay=None):
    body = "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: ..."
    if delay is not None:
        body += " {'retryDelay': '%ss'}" % delay
    return RuntimeError(body)


# --- pacing ------------------------------------------------------------------

class TestPacing:
    def test_calls_are_spaced_to_stay_under_the_limit(self, monkeypatch):
        """15/minute means one call every four seconds, not fifteen at once."""
        monkeypatch.setenv("GEMINI_RPM", "60")  # one per second
        slept = []
        monkeypatch.setattr(gemini_calls.time, "sleep", slept.append)
        client = _Client(["a", "b", "c"])
        for _ in range(3):
            gemini_calls.call(client, model="m", contents="p")
        # The first call goes straight through; the next two wait their turn,
        # so N calls at R per minute cost (N-1)/R minutes of waiting. The
        # slot is claimed before sleeping, so with sleep stubbed out the two
        # waits are 1s and 2s rather than 1s and 1s — the total is the part
        # that has to be right, and it is what keeps the rate under the cap.
        assert len(slept) == 2
        assert sum(slept) == pytest.approx(3.0, abs=0.1)

    def test_the_limiter_spans_stages(self, monkeypatch):
        """A per-stage limiter lets the Finder finish exactly on budget and
        hand a spent quota to the critic — the same failure one stage later."""
        monkeypatch.setenv("GEMINI_RPM", "60")
        slept = []
        monkeypatch.setattr(gemini_calls.time, "sleep", slept.append)
        gemini_calls.call(_Client(["a"]), model="m", contents="p")
        gemini_calls.call(_Client(["b"]), model="m", contents="p")
        assert len(slept) == 1, "second caller ignored the first caller's slot"

    def test_pacing_can_be_switched_off(self, monkeypatch):
        """A billed key has no 15/minute cap to respect."""
        monkeypatch.setenv("GEMINI_RPM", "0")
        slept = []
        monkeypatch.setattr(gemini_calls.time, "sleep", slept.append)
        for _ in range(5):
            gemini_calls.call(_Client(["x"]), model="m", contents="p")
        assert slept == []

    def test_a_malformed_rpm_does_not_crash_a_job(self, monkeypatch):
        monkeypatch.setenv("GEMINI_RPM", "fifteen")
        assert gemini_calls.rpm() == gemini_calls.DEFAULT_RPM

    def test_the_default_matches_the_free_tier(self):
        """15 is the number the real 429 named for gemini-3.1-flash-lite."""
        assert gemini_calls.DEFAULT_RPM == 15.0


# --- retry, one call at a time ----------------------------------------------

class TestRetry:
    def test_a_rate_limited_call_is_retried_not_the_stage(self, monkeypatch):
        monkeypatch.setattr(gemini_calls.time, "sleep", lambda s: None)
        client = _Client([_rate_limit(), "ok"])
        assert gemini_calls.call(client, model="m", contents="p") == "ok"
        assert client.calls == 2

    def test_geminis_own_retry_delay_is_used(self, monkeypatch):
        """It knows when the quota frees; a guessed curve does not."""
        slept = []
        monkeypatch.setenv("GEMINI_RPM", "0")
        monkeypatch.setattr(gemini_calls.time, "sleep", slept.append)
        gemini_calls.call(_Client([_rate_limit(delay=7), "ok"]),
                          model="m", contents="p")
        # Its figure is the instant the quota frees, so landing exactly on it
        # races the server's clock.
        assert slept == [8.0]

    def test_without_a_stated_delay_it_backs_off(self, monkeypatch):
        slept = []
        monkeypatch.setenv("GEMINI_RPM", "0")
        monkeypatch.setattr(gemini_calls.time, "sleep", slept.append)
        gemini_calls.call(_Client([RuntimeError("503 UNAVAILABLE"), "ok"]),
                          model="m", contents="p")
        assert slept == [gemini_calls.FALLBACK_DELAYS[0]]

    def test_a_permanent_error_is_not_retried(self, monkeypatch):
        """A bad API key does not get better by asking six more times."""
        monkeypatch.setattr(gemini_calls.time, "sleep", lambda s: None)
        client = _Client([RuntimeError("403 PERMISSION_DENIED"), "ok"])
        with pytest.raises(RuntimeError, match="403"):
            gemini_calls.call(client, model="m", contents="p")
        assert client.calls == 1

    def test_it_gives_up_and_raises_the_real_error(self, monkeypatch):
        monkeypatch.setattr(gemini_calls.time, "sleep", lambda s: None)
        monkeypatch.setenv("GEMINI_CALL_ATTEMPTS", "3")
        client = _Client([_rate_limit(), _rate_limit(), _rate_limit()])
        with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
            gemini_calls.call(client, model="m", contents="p")
        assert client.calls == 3


# --- what the pipeline now does with all that -------------------------------

class TestThePipelineUsesIt:
    def test_no_stage_calls_gemini_bare(self):
        """One policy, or the Finder goes back to having none at all — it was
        the only stage calling generate_content directly, and the only one
        that died."""
        for name in ("meaningful_selector.py", "meaningful_critic.py",
                     "meaningful_metadata.py", "meaningful_opening_guard.py"):
            assert "client.models.generate_content(" not in source(name), name

    def test_the_whole_stage_is_never_re_run(self):
        """The 13 completed windows must survive the 14th window's rate
        limit. Re-running the stage is what deleted them."""
        text = source("meaningful_pipeline.py")
        assert "Retrying whole stage" not in text
        block = text[text.index("def _retry_stage("):]
        block = block[:block.index("\ndef ", 1)]
        assert "for attempt in range" not in block

    def test_a_window_that_cannot_be_analysed_is_skipped(self):
        """21 of 22 windows is a usable result; zero is not."""
        text = source("meaningful_selector.py")
        block = text[text.index("skipped_windows.append"):]
        assert "continue" in block[:400]

    def test_skipped_windows_are_reported(self):
        """A thin result from a partly-analysed transcript looks exactly like
        a video with few good moments in it."""
        assert "could not " in source("meaningful_selector.py")
        assert "were skipped" in source("meaningful_selector.py")
