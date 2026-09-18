"""Dispatch transcription to a Kaggle kernel and bring the transcript back.

WHAT KAGGLE IS, AND IS NOT
--------------------------
Kaggle runs BATCH kernels. There is no endpoint to call and nothing can talk to
a kernel while it runs. The only interface is:

    push a version  ->  it queues  ->  it runs  ->  poll status  ->  fetch output

So this module is a dispatcher, not a client. It writes the job parameters INTO
a copy of ``kaggle-worker/worker.py``, pushes that as a new kernel version,
polls ``kernels_status`` until the run reaches a terminal state, and downloads
``transcript.json`` from the kernel's output.

WHY ONLY TRANSCRIPTION
----------------------
It is the long pole (~16 min of CPU for a 9.5-minute video on the dev box) and
the only stage whose input is small (a URL) and whose output is small (a JSON
transcript). The render needs the video file and would have to ship gigabytes
back, so it stays local — the backend downloads the video in parallel with this
call, and the two meet at the clip selector.

FAILURE IS NORMAL AND NEVER FATAL
---------------------------------
Kaggle's GPU quota is weekly, its queue is shared, and its egress IP is
sometimes blocked by YouTube. Every failure path here returns None rather than
raising, and ``main.transcribe_video`` then transcribes locally exactly as it
did before this module existed. A job must never fail because an optional
accelerator was unavailable.
"""
from __future__ import annotations

import json
import os
import pprint
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

WORKER_DIR = Path(__file__).resolve().parent / "kaggle-worker"
WORKER_SOURCE = WORKER_DIR / "worker.py"

#: Terminal states from kagglesdk's KernelWorkerStatus.
TERMINAL_OK = {"COMPLETE"}
TERMINAL_BAD = {"ERROR", "CANCEL_ACKNOWLEDGED"}

DEFAULT_SLUG = "openshorts-asr-worker"
#: large-v3, not turbo: the GPU removes both reasons the local path
#: excludes it (1.8x CPU decode, and an OOM on load), and it is measurably
#: better on code-switched audio.
DEFAULT_MODEL = "large-v3"
DEFAULT_TIMEOUT = 1800.0
DEFAULT_POLL = 15.0


# --- configuration ----------------------------------------------------------

def enabled() -> bool:
    """Whether jobs should try Kaggle at all."""
    if os.environ.get("KAGGLE_ASR", "").strip() == "0":
        return False
    return bool(_credentials()[1])


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _credentials():
    """(username, key) from whichever of the accepted spellings is set.

    The CLI wants KAGGLE_USERNAME + KAGGLE_KEY. A token pasted from the website
    is just the key, so KAGGLE_API_TOKEN is accepted as an alias rather than
    silently doing nothing — that is the shape the key arrived in.
    """
    username = (os.environ.get("KAGGLE_USERNAME")
                or os.environ.get("KAGGLE_USER") or "").strip()
    key = (os.environ.get("KAGGLE_KEY")
           or os.environ.get("KAGGLE_API_TOKEN") or "").strip()

    if not username:
        # Fall back to the owner of the configured kernel, which is the same
        # account the key belongs to in every real setup.
        slug = os.environ.get("KAGGLE_KERNEL", "").strip()
        if "/" in slug:
            username = slug.split("/", 1)[0]
    return username, key


def kernel_slug() -> str:
    """``owner/slug`` of the worker kernel."""
    configured = os.environ.get("KAGGLE_KERNEL", "").strip()
    if configured:
        return configured
    username, _ = _credentials()
    return f"{username}/{DEFAULT_SLUG}" if username else ""


def _cache_kernel() -> str:
    """The kernel whose output carries node_modules and the whisper model."""
    configured = os.environ.get("KAGGLE_CACHE_KERNEL", "").strip()
    if configured:
        return configured
    username, _ = _credentials()
    return f"{username}/openshorts-asr-cache" if username else ""


# --- the Kaggle client ------------------------------------------------------

def _api():
    """An authenticated KaggleApi, or None.

    The credentials go into the environment BEFORE the import: the kaggle
    package authenticates eagerly and raises at import time when it cannot find
    them, which would take down the whole backend instead of disabling one
    optional feature.
    """
    username, key = _credentials()
    if not username or not key:
        return None
    os.environ.setdefault("KAGGLE_USERNAME", username)
    os.environ["KAGGLE_KEY"] = key
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except ImportError:
        # The commonest way this fails, and the least obvious: the package is
        # in requirements.txt but the running CONTAINER was built before that.
        # The dev compose bind-mounts the source, so new .py files appear
        # immediately and new DEPENDENCIES do not.
        print("⚠️ Kaggle unavailable: the 'kaggle' package is not installed "
              "in this environment. Rebuild the image "
              "(docker compose up --build backend), or for a quick test "
              "'docker compose exec backend /opt/venv/bin/pip install "
              "kaggle==2.2.4'.")
        return None
    except Exception as exc:
        print(f"⚠️ Kaggle unavailable ({type(exc).__name__}: {exc})")
        return None


# --- building the pushable folder -------------------------------------------

def render_worker(source: str, job: Dict[str, Any]) -> str:
    """Replace the worker's JOB block with this job's parameters.

    A whole-block replacement rather than a templated string: the worker stays
    a runnable, lintable Python file that can be executed by hand with its
    default JOB, which is how it gets debugged when Kaggle misbehaves.
    """
    # pprint, NOT json.dumps. The destination is a PYTHON SOURCE FILE, and the
    # two literal syntaxes differ exactly where it hurts: json.dumps(None) is
    # `null`, True is `true`, False is `false`. A rendered worker containing
    # `"language": null` is syntactically valid Python that dies at import with
    # NameError: name 'null' is not defined — which is precisely how the first
    # real dispatch failed, 26 seconds into a Kaggle kernel.
    payload = pprint.pformat(job, indent=4, width=79, sort_dicts=False)
    replacement = f"JOB = {payload}"
    # A FUNCTION as the replacement, not the string. re.sub interprets
    # backslashes in a replacement string, so a URL containing \n or \1 —
    # and a URL is user input arriving from the dashboard — would be rewritten
    # into a literal newline or a group reference, producing a worker.py that
    # does not parse. json.dumps escaped it correctly; re.sub would undo that.
    rendered, count = re.subn(
        r"^JOB = \{.*?^\}", lambda _match: replacement, source,
        count=1, flags=re.DOTALL | re.MULTILINE)
    if not count:
        raise ValueError("worker.py has no JOB block to substitute")
    return rendered


def _metadata(slug: str, cache_kernel: str, gpu: bool) -> Dict[str, Any]:
    return {
        "id": slug,
        "title": slug.split("/")[-1],
        "code_file": "worker.py",
        "language": "python",
        # A script, not a notebook: the source is a plain .py that can be
        # rendered, diffed and run locally without touching ipynb JSON.
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": bool(gpu),
        "enable_tpu": False,
        # Without internet the worker cannot reach YouTube at all.
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [cache_kernel] if cache_kernel else [],
        "competition_sources": [],
        "model_sources": [],
    }


def _stage(folder: Path, job: Dict[str, Any], slug: str,
           cache_kernel: str, gpu: bool) -> None:
    source = WORKER_SOURCE.read_text(encoding="utf-8")
    (folder / "worker.py").write_text(
        render_worker(source, job), encoding="utf-8")
    (folder / "kernel-metadata.json").write_text(
        json.dumps(_metadata(slug, cache_kernel, gpu), indent=2),
        encoding="utf-8")


# --- polling ----------------------------------------------------------------

def _status_name(response: Any) -> str:
    status = getattr(response, "status", response)
    name = getattr(status, "name", None)
    if name:
        return str(name).upper()
    # Older clients stringify as "KernelWorkerStatus.RUNNING".
    return str(status).rsplit(".", 1)[-1].upper()


def _wait(api, slug: str, timeout: float, poll: float,
          on_progress=None) -> Optional[str]:
    """Block until the kernel reaches a terminal state. Returns the state."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            response = api.kernels_status(slug)
        except Exception as exc:
            print(f"⚠️ Kaggle status check failed ({type(exc).__name__}: {exc})")
            time.sleep(poll)
            continue

        state = _status_name(response)
        if state != last:
            last = state
            remaining = int(deadline - time.time())
            print(f"☁️ Kaggle {state.lower()} (timeout in {remaining}s)")
            if on_progress:
                on_progress(state)

        if state in TERMINAL_OK:
            return state
        if state in TERMINAL_BAD:
            message = getattr(response, "failure_message", "") or ""
            if message:
                print(f"⚠️ Kaggle kernel failed: {message}")
            return state
        time.sleep(poll)

    print(f"⚠️ Kaggle did not finish within {timeout:.0f}s.")
    return None


def _fetch_transcript(api, slug: str) -> Optional[Dict[str, Any]]:
    """Download the kernel's output and parse the transcript out of it."""
    destination = Path(tempfile.mkdtemp(prefix="kaggle_asr_"))
    try:
        # The client follows its own output pages (`while token and
        # page_token is None`), so this is ONE call — passing page_token
        # explicitly would switch that off.
        #
        # It then writes the kernel LOG with `open(outfile, "w")`, no encoding,
        # which on a host whose default codepage is not UTF-8 (Windows cp1252)
        # raises UnicodeEncodeError on a Devanagari transcript. The DATA files
        # are written first and in BINARY mode, so by then the transcript is
        # already on disk — check for it before believing the exception.
        try:
            api.kernels_output(slug, str(destination), quiet=True)
        except Exception as exc:
            if not (destination / "transcript.json").exists():
                print(f"⚠️ Could not download Kaggle output "
                      f"({type(exc).__name__}: {exc})")
                return None
            print(f"⚠️ The Kaggle client raised writing its own log file "
                  f"({type(exc).__name__}); the transcript arrived intact.")

        result = destination / "result.json"
        if result.exists():
            try:
                payload = json.loads(result.read_text(encoding="utf-8"))
                if not payload.get("ok"):
                    print(f"⚠️ Kaggle worker reported failure: "
                          f"{payload.get('error')}")
                    return None
            except (ValueError, OSError):
                pass

        transcript_path = destination / "transcript.json"
        if not transcript_path.exists():
            print("⚠️ Kaggle run produced no transcript.json.")
            return None
        try:
            return json.loads(transcript_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            print(f"⚠️ Kaggle transcript.json is unreadable: {exc}")
            return None
    finally:
        shutil.rmtree(destination, ignore_errors=True)


# --- the transcript contract ------------------------------------------------

def _finalize(transcript: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Make the remote transcript identical in shape to a local one.

    The worker returns RAW whisper words; the continuation-word merge happens
    here so there is one implementation of it and a Kaggle transcript and a
    local transcript cannot drift apart in how words are joined.
    """
    if not isinstance(transcript, dict):
        return None
    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        print("⚠️ Kaggle transcript has no segments.")
        return None

    from subtitles import merge_continuation_words

    for segment in segments:
        if isinstance(segment, dict) and segment.get("words"):
            segment["words"] = merge_continuation_words(segment["words"])

    transcript.setdefault("asr", {})["backend"] = "whisper"
    return transcript


# --- the public entry point -------------------------------------------------

class RemoteTranscription:
    """A Kaggle job in flight. Dispatched at t=0, collected when needed.

    The whole point of this class is that the dispatch no longer waits for the
    local download. It used to: the language probe ran on the downloaded file,
    so Kaggle could not start until ~36-73s of download plus ~15s of probing
    had finished, and then the job sat idle for Kaggle's ~110s round trip. Now
    the probe happens on the GPU inside the worker and the two run at once.

    ``collect()`` is what blocks, and it can only ever return a transcript or
    None — never raise — because the caller's fallback is a perfectly good
    local transcription.
    """

    def __init__(self, future, slug):
        self._future = future
        self.slug = slug

    def collect(self, timeout=None):
        try:
            return self._future.result(timeout=timeout)
        except Exception as exc:
            print(f"⚠️ Kaggle job failed ({type(exc).__name__}: {exc})")
            return None

    def cancel(self):
        self._future.cancel()


def start(url: str, job_id: str = "job", language: Optional[str] = None,
          on_progress=None) -> Optional["RemoteTranscription"]:
    """Dispatch to Kaggle WITHOUT waiting. Returns a handle, or None.

    None means "not available, transcribe locally" and is returned before any
    work is done, so the caller can take the local path immediately.
    """
    if not enabled():
        return None
    slug = kernel_slug()
    if not slug or "/" not in slug:
        print("⚠️ Kaggle: no kernel configured (set KAGGLE_KERNEL).")
        return None

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1,
                              thread_name_prefix="kaggle-asr")
    future = pool.submit(transcribe_url, url, job_id, language, on_progress)
    # The pool is shut down as soon as the single task finishes; it exists only
    # to keep the poll off the main thread while the video downloads.
    pool.shutdown(wait=False)
    return RemoteTranscription(future, slug)


def transcribe_url(url: str, job_id: str = "job", language: Optional[str] = None,
                   on_progress=None) -> Optional[Dict[str, Any]]:
    """Transcribe ``url`` on Kaggle, or return None to fall back locally.

    Never raises. Every failure — no credentials, push rejected, kernel error,
    timeout, missing output — is a None, because the caller's fallback is a
    perfectly good local transcription and a job must not die because an
    optional accelerator was busy.
    """
    if not enabled():
        return None

    slug = kernel_slug()
    if not slug or "/" not in slug:
        print("⚠️ Kaggle: no kernel configured (set KAGGLE_KERNEL).")
        return None

    api = _api()
    if api is None:
        return None

    job = {
        "job_id": job_id,
        "url": url,
        "model": os.environ.get("KAGGLE_WHISPER_MODEL", DEFAULT_MODEL).strip()
        or DEFAULT_MODEL,
        # PINNING THIS IS NOT OPTIONAL on a non-English video. Left to detect
        # for itself, large-v3-turbo decided the Hindi stand-up was English at
        # 90% and returned a fluent English TRANSLATION of it — "Yesterday, I
        # went to a lift..." for "कल में एक लिफ्ट में गुसी..." — even though the
        # task is always "transcribe". Nothing downstream can catch that: it
        # scores ~99/100 because it is genuinely good English, and the clips,
        # captions and metadata all silently change language. The caller probes
        # the language with the SMALL model first, which gets it right.
        "language": (language
                     or os.environ.get("KAGGLE_ASR_LANGUAGE", "").strip()
                     or None),
        # The worker detects the language itself, with the SMALL model, when
        # the caller did not pin one. That is what lets the dispatch happen at
        # t=0 instead of waiting for the local download to finish so a local
        # probe could run on it.
        # The probe RUNS by default but does not DECIDE by default: auto
        # language means the transcription model picks, and the probe is a
        # cross-check that logs a disagreement. KAGGLE_PIN_LANGUAGE=1 makes
        # the probe binding again; KAGGLE_DETECT_LANGUAGE=0 skips it entirely.
        "detect_language": os.environ.get(
            "KAGGLE_DETECT_LANGUAGE", "1").strip() != "0",
        "pin_language": os.environ.get(
            "KAGGLE_PIN_LANGUAGE", "0").strip() == "1",
        "pin_language_above": _env_float("ASR_PIN_LANGUAGE_ABOVE", 0.5),
        "language_detection_segments": int(
            _env_float("WHISPER_LANG_DETECT_SEGMENTS", 4)),
        "beam_size": int(_env_float("KAGGLE_ASR_BEAM_SIZE", 5)),
    }

    gpu = os.environ.get("KAGGLE_GPU", "1").strip() != "0"
    folder = Path(tempfile.mkdtemp(prefix="kaggle_push_"))
    try:
        _stage(folder, job, slug, _cache_kernel(), gpu)
        print(f"☁️ Dispatching transcription to Kaggle ({slug}, "
              f"{'GPU' if gpu else 'CPU'})...")
        try:
            api.kernels_push(str(folder))
        except Exception as exc:
            print(f"⚠️ Kaggle push failed ({type(exc).__name__}: {exc})")
            return None
    finally:
        shutil.rmtree(folder, ignore_errors=True)

    state = _wait(
        api, slug,
        timeout=_env_float("KAGGLE_TIMEOUT_SECONDS", DEFAULT_TIMEOUT),
        poll=_env_float("KAGGLE_POLL_SECONDS", DEFAULT_POLL),
        on_progress=on_progress,
    )
    if state not in TERMINAL_OK:
        return None

    transcript = _fetch_transcript(api, slug)
    if transcript is None:
        return None

    final = _finalize(transcript)
    if final is None:
        return None

    asr = final.get("asr") or {}
    print(f"☁️ Kaggle transcript: {len(final['segments'])} segments, "
          f"language={final.get('language')}, "
          f"decoded in {asr.get('decode_seconds', '?')}s on "
          f"{asr.get('device', '?')}")
    return final
