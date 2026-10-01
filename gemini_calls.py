"""Pace and retry the pipeline's Gemini calls, in one place.

WHAT WENT WRONG
---------------
A 50-minute video (30-sep-2026) needs 22 Candidate Finder windows, one call
each, and the free tier allows **15 requests per minute**. The job made 13
calls in a few seconds, hit `429 RESOURCE_EXHAUSTED` on the 14th, and
`meaningful_pipeline._retry_stage` restarted the WHOLE stage from window 1 —
discarding all 13 completed windows. Attempt two redid twelve of them before
dying again; attempt three died on window 2. Twenty-six calls spent, nothing
produced, and each retry pushed the quota further under water: the more work
the stage completed, the more certain the next attempt was to fail.

That video could not have succeeded at any length of wait. It was not short
of quota — the daily allowance was barely touched. It spent a per-MINUTE
allowance in a burst and then threw the results away.

SO TWO THINGS, AND THEY BELONG TOGETHER
---------------------------------------
**Pacing** keeps the burst from happening: calls are spaced so the rate stays
under `GEMINI_RPM`. 22 windows at 15/minute is about 90 seconds of waiting,
against a job that spends ten minutes transcribing.

**Retry happens per CALL, not per stage.** A rate limit is a property of the
moment, not of the work already done, so the thirteen windows that succeeded
stay succeeded. Gemini also says how long to wait (`retryDelay` in the error
body); that is used when present, because it is better information than any
backoff curve guessed from outside.

The limiter is process-wide and spans stages on purpose. A per-stage limiter
would let the Candidate Finder finish exactly on budget and then hand a spent
quota to the critic, which is the same failure one stage later.

It is deliberately a MINIMUM INTERVAL rather than a sliding window: simpler to
reason about, and the pipeline's calls are sequential, so a token bucket would
only matter for a burst that cannot happen here.

KNOWN LIMIT: the quota is per KEY, this limiter is per PROCESS, and `app.py`
runs each job as its own `main.py` subprocess. Two jobs at once therefore pace
at `GEMINI_RPM` each and together exceed it. That is fine while jobs run one at
a time, and the right fix if it ever bites is a shared counter on disk rather
than a lower per-process rate, which would slow the single-job case for
nothing.
"""
import os
import re
import threading
import time
from typing import Any, Optional

#: Free-tier requests per minute for gemini-3.1-flash-lite, which is what the
#: 429 in the real failure named. Set GEMINI_RPM higher on a billed key, or 0
#: to disable pacing entirely.
DEFAULT_RPM = 15.0

#: Attempts per CALL. Three is enough for a rate limit that resolves in
#: seconds; past that the problem is the quota itself and waiting longer only
#: delays an error the operator has to act on.
DEFAULT_ATTEMPTS = 3

#: Backoff when the error does not say how long to wait.
FALLBACK_DELAYS = (5.0, 15.0, 30.0)

#: Gemini puts the wait in the error body as `'retryDelay': '2s'`. Its own
#: number beats anything guessed from out here.
_RETRY_DELAY = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")

TRANSIENT_MARKERS = (
    "429", "500", "502", "503", "504",
    "resource_exhausted", "unavailable", "temporarily unavailable",
    "timeout", "timed out", "rate limit", "rate_limit",
    "connection reset", "connection aborted",
)

_lock = threading.Lock()
_last_call_at = 0.0


def rpm() -> float:
    try:
        return float(os.environ.get("GEMINI_RPM", "").strip() or DEFAULT_RPM)
    except (TypeError, ValueError):
        return DEFAULT_RPM


def attempts() -> int:
    try:
        return max(1, int(os.environ.get("GEMINI_CALL_ATTEMPTS", "").strip()
                          or DEFAULT_ATTEMPTS))
    except (TypeError, ValueError):
        return DEFAULT_ATTEMPTS


def is_transient(error: BaseException) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def suggested_delay(error: BaseException) -> Optional[float]:
    """The wait Gemini asked for, if it said."""
    found = _RETRY_DELAY.search(str(error))
    if not found:
        return None
    try:
        return float(found.group(1))
    except (TypeError, ValueError):
        return None


def reset() -> None:
    """Forget the last call time (tests, and a new job in the same process)."""
    global _last_call_at
    with _lock:
        _last_call_at = 0.0


def _wait_turn() -> float:
    """Block until another call is allowed. Returns the seconds slept."""
    global _last_call_at
    limit = rpm()
    if limit <= 0:
        return 0.0
    interval = 60.0 / limit
    with _lock:
        now = time.monotonic()
        earliest = _last_call_at + interval
        delay = max(0.0, earliest - now)
        # The slot is claimed BEFORE sleeping, so two callers cannot both see
        # the same free slot and then both take it.
        _last_call_at = max(now, earliest)
    if delay > 0:
        time.sleep(delay)
    return delay


def call(client: Any, **kwargs: Any) -> Any:
    """`client.models.generate_content(**kwargs)`, paced and retried.

    Raises the last error when every attempt fails. Callers decide whether one
    failed item is fatal — for the Candidate Finder it is not, because 21 of 22
    windows is a usable result and zero is not.
    """
    last_error: Optional[BaseException] = None
    for attempt in range(attempts()):
        _wait_turn()
        try:
            return client.models.generate_content(**kwargs)
        except Exception as exc:
            last_error = exc
            if not is_transient(exc) or attempt >= attempts() - 1:
                raise
            delay = suggested_delay(exc)
            if delay is None:
                delay = FALLBACK_DELAYS[min(attempt, len(FALLBACK_DELAYS) - 1)]
            else:
                # Gemini's own figure is the moment the quota frees, so landing
                # exactly on it races the server's clock.
                delay += 1.0
            print(f"   ⏳ Gemini {type(exc).__name__} — retrying this call in "
                  f"{delay:.0f}s ({attempt + 2}/{attempts()})")
            time.sleep(delay)
    if last_error:
        raise last_error
    raise RuntimeError("gemini_calls.call exhausted its attempts")
