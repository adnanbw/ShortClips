"""OpenShorts ASR worker — runs on Kaggle, returns a transcript.

Why this exists
---------------
Transcription is the long pole of a job: ~16 minutes of CPU for a 9.5-minute
video on the dev box, and it is the one stage that is pure compute with a small
input (a URL) and a small output (a JSON transcript). That makes it the only
stage worth shipping off the machine — the render needs the video file and
would have to send gigabytes back.

Kaggle is a BATCH platform, not a server. Nothing can call this script while it
runs. ``kaggle_worker.py`` on the backend pushes a version with JOB baked into
the header below, polls for COMPLETE, and downloads ``transcript.json`` from
the kernel's output. Everything here must therefore be self-contained and must
never prompt, never wait on input, and never leave the output file missing —
a crash with no ``transcript.json`` is indistinguishable from a hang.

The download recipe is NOT improvised: it is the one measured to work from
Kaggle's egress (mweb + a BgUtils PO token, 9.3s for a 10-minute video, no
"Sign in to confirm you're not a bot"). Do not simplify it away.

Output contract — identical to ``transcribe_backends._transcribe_with_whisper``
so the backend can drop it straight into the existing pipeline. Word merging
(``merge_continuation_words``) is deliberately NOT done here: it runs on the
backend, so there is exactly one implementation of it.
"""
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ============================================================================
# JOB — replaced wholesale by kaggle_worker.render_worker() before each push.
# ============================================================================
JOB = {
    "job_id": "local-test",
    "url": "https://youtu.be/Qd4jGgu06dw",
    "model": "large-v3",
    "language": None,
    "detect_language": True,
    "pin_language": False,
    "pin_language_above": 0.5,
    "language_detection_segments": 4,
    "beam_size": 5,
}
# ============================================================================

WORKDIR = Path("/kaggle/working")
CACHE_INPUT = Path("/kaggle/input")
BGUTIL_PORT = 4417
AUDIO_STEM = WORKDIR / "job_audio"
TRANSCRIPT_PATH = WORKDIR / "transcript.json"
RESULT_PATH = WORKDIR / "result.json"

#: Kaggle mounts another kernel's output read-only under /kaggle/input/<slug>.
#: The cache kernel (tools/kaggle_setup.py) puts the faster-whisper model
#: there, which is ~1.6 GB this job does not have to download.
CACHE_NAMES = ("openshorts-asr-cache", "openshorts_asr_cache")


def log(message):
    print(message, flush=True)


def run(cmd, cwd=None, check=True, env=None):
    log("> " + " ".join(map(str, cmd)))
    return subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        check=check,
        env=env,
    )


def find_cache():
    """The mounted cache kernel output, or None when running without it."""
    if not CACHE_INPUT.exists():
        return None
    for name in CACHE_NAMES:
        candidate = CACHE_INPUT / name
        if candidate.exists():
            return candidate
    # Kaggle sometimes mounts under a slug we did not predict; accept any
    # directory that looks like our cache rather than failing over a name.
    for candidate in CACHE_INPUT.iterdir():
        if (candidate / "models").exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# 1. Audio download
# ---------------------------------------------------------------------------

def install_downloader():
    """yt-dlp, Deno and the BgUtils PO-token provider, installed fresh.

    Deliberately NOT restored from the cache kernel. Deno keeps npm packages in
    its own global cache and only symlinks them into ``node_modules``, so a
    copied node_modules is an incomplete tree: the cached version got as far as
    starting the token server and died with "Could not find package 'lru-cache'
    from referrer .../proxy-agent/dist/index.js". Caching it properly means
    shipping DENO_DIR as well and making both builds use identical absolute
    paths — a lot of fragility for the ~1-2 minutes this costs. The 1.6 GB
    model, which IS worth caching, still comes from the cache kernel.
    """
    run([sys.executable, "-m", "pip", "install", "-q", "-U",
         "yt-dlp[default]==2026.8.19", "bgutil-ytdlp-pot-provider==2.0.0"])

    deno = shutil.which("deno")
    if not deno:
        run(["bash", "-c", "curl -fsSL https://deno.land/install.sh | sh"])
        os.environ["PATH"] = f"{Path.home()}/.deno/bin:" + os.environ["PATH"]
        deno = shutil.which("deno")
    if not deno:
        raise RuntimeError("Deno installation failed")

    clone = WORKDIR / "bgutil"
    if clone.exists():
        shutil.rmtree(clone)
    run(["git", "clone", "--depth", "1", "--single-branch",
         "--branch", "2.0.0",
         "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git",
         clone])

    server_dir = clone / "server"
    env = os.environ.copy()
    env["DENO_NO_PROMPT"] = "1"
    env["DENO_NO_UPDATE_CHECK"] = "1"
    run([deno, "install", "--allow-scripts=npm:canvas", "--frozen"],
        cwd=server_dir, env=env)

    node_modules = server_dir / "node_modules"
    if not node_modules.exists():
        raise RuntimeError("BgUtils node_modules missing after install")
    return deno, server_dir, node_modules


def start_token_server(deno, node_modules):
    """The PO-token provider yt-dlp needs for the mweb client."""
    env = os.environ.copy()
    env["DENO_NO_PROMPT"] = "1"
    env["DENO_NO_UPDATE_CHECK"] = "1"
    log_path = WORKDIR / "bgutil-server.log"
    handle = open(log_path, "w")

    server = subprocess.Popen(
        [deno, "run", "--allow-env", "--allow-net", "--allow-ffi=.",
         "--allow-read=.", "../src/main.ts",
         "--host", "127.0.0.1", "--port", str(BGUTIL_PORT)],
        cwd=str(node_modules), stdout=handle, stderr=subprocess.STDOUT, env=env,
    )

    for _ in range(80):
        if server.poll() is not None:
            break
        try:
            with socket.create_connection(("127.0.0.1", BGUTIL_PORT), timeout=0.5):
                log(f"BgUtils token server ready on :{BGUTIL_PORT}")
                return server
        except OSError:
            time.sleep(0.5)

    handle.flush()
    if log_path.exists():
        log(log_path.read_text(errors="ignore")[-4000:])
    raise RuntimeError("BgUtils token server failed to start")


def clear_audio():
    for path in glob.glob(str(AUDIO_STEM) + ".*"):
        try:
            os.remove(path)
        except OSError:
            pass


def find_audio():
    matches = glob.glob(str(AUDIO_STEM) + ".*")
    return matches[0] if matches else None


def download_audio(url, deno):
    """Audio only, through the strategies measured to work from Kaggle.

    mweb + a PO token is the one that actually succeeded (9.3s); the others are
    kept because YouTube's answer varies by video and by egress IP, and a
    second strategy costs seconds while a failed job costs the whole run.
    """
    strategies = [
        ("mweb + BgUtils PO token", "youtube:player_client=mweb"),
        ("default clients", "youtube:player_client=default"),
        ("web_embedded", "youtube:player_client=web_embedded,default"),
    ]

    for attempt in range(1, 3):
        for name, extractor in strategies:
            clear_audio()
            log(f"\n=== download attempt {attempt}: {name} ===")
            started = time.time()
            result = subprocess.run([
                sys.executable, "-m", "yt_dlp",
                "--no-playlist", "--force-ipv4",
                "--js-runtimes", f"deno:{deno}",
                "--extractor-args",
                f"youtubepot-bgutilhttp:base_url=http://127.0.0.1:{BGUTIL_PORT}",
                "--extractor-args", extractor,
                "--retries", "5", "--fragment-retries", "5", "--retry-sleep", "2",
                "-f", "bestaudio/best",
                "-o", str(AUDIO_STEM) + ".%(ext)s",
                url,
            ])
            audio = find_audio()
            if result.returncode == 0 and audio:
                log(f"Downloaded in {time.time() - started:.1f}s: {audio}")
                return audio
            log(f"Failed: {name}")
            time.sleep(3)
        if attempt == 1:
            time.sleep(10)

    raise RuntimeError(
        "Every download strategy failed. If the log says LOGIN_REQUIRED or "
        "'Sign in to confirm you're not a bot', Kaggle's shared egress IP is "
        "currently blocked by YouTube — this is not a packaging problem."
    )


# ---------------------------------------------------------------------------
# 2. Transcription
# ---------------------------------------------------------------------------

def resolve_model(model_name, cache):
    """A local model directory from the cache, or the plain model name.

    faster-whisper accepts a path to a converted CTranslate2 directory, which
    is what the cache kernel stores, so a cached run never touches the network
    and never re-downloads ~1.6GB.
    """
    if not cache:
        return model_name
    local = cache / "models" / model_name
    if (local / "model.bin").exists():
        log(f"Using cached model at {local}")
        return str(local)
    return model_name


def detect_language(audio_path, cache, device, compute_type):
    """Detect the spoken language with the SMALL model, on the GPU.

    This used to happen on the backend, on the downloaded video, and it is the
    only reason the dispatch had to wait for the local download to finish.
    Moving it here is what lets the backend fire the Kaggle job at t=0 and
    download in parallel.

    It must be the SMALL model, not the one that does the transcription. Left
    to detect for itself, large-v3-turbo called a Hindi stand-up **English at
    90%** and returned a fluent English TRANSLATION of it, although the task is
    always "transcribe" — and nothing downstream can catch that, because it
    scores ~99/100 as genuinely good English. whisper-small gets the same audio
    right (hi at 84-89%). Two models, one right answer.

    Returns (language, probability), or (None, None) when it is not confident
    enough to be worth pinning — in which case the big model decides, exactly
    as ``_retry_language`` does locally.
    """
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    threshold = float(JOB.get("pin_language_above") or 0.5)
    model_ref = resolve_model("small", cache)
    log(f"Detecting language with {model_ref} ...")

    model = WhisperModel(model_ref, device=device, compute_type=compute_type)
    try:
        language, probability, _all = model.detect_language(
            audio=decode_audio(audio_path),
            # Sampling ONE window lets applause, music or an English intro pick
            # the language for the whole file; the local path samples 4 for the
            # same reason (WHISPER_LANG_DETECT_SEGMENTS).
            language_detection_segments=int(
                JOB.get("language_detection_segments") or 4),
            vad_filter=True,
        )
    except Exception as exc:
        log(f"Language detection failed ({type(exc).__name__}: {exc}) - "
            f"letting the transcription model decide.")
        return None, None
    finally:
        del model

    log(f"Detected language={language} ({probability:.0%})")
    if probability < threshold:
        log(f"Below the {threshold:.0%} pin threshold - "
            f"letting the transcription model decide.")
        return None, None
    return language, probability


def transcribe(audio_path, cache):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "faster-whisper==1.2.1"], check=True)

    import torch
    from faster_whisper import WhisperModel

    if torch.cuda.is_available():
        device, compute_type = "cuda", "float16"
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        # Not fatal, but worth shouting about: a CPU kernel is the whole
        # reason this job would be no faster than running it at home.
        device, compute_type = "cpu", "int8"
        log("WARNING: no GPU on this kernel — set enable_gpu in "
            "kernel-metadata.json, or this is pointless.")

    # Language, in priority order:
    #   1. an explicit pin from the caller
    #   2. the small-model probe, when pin_language is on
    #   3. AUTO — the transcription model decides (the default)
    #
    # The probe still RUNS in auto mode even though it does not decide. It
    # costs seconds on a GPU and it is the only way to notice the one failure
    # nothing downstream can see: a model that mis-detects the language emits a
    # fluent TRANSLATION, which scores ~99/100 because it is genuinely good
    # English, and the clips, captions and metadata all change language
    # silently. large-v3-turbo did exactly that to a Hindi stand-up ("English,
    # 90%"). large-v3's language ID is much better, which is why auto is now
    # the default — but "much better" is not "verified", so a disagreement
    # between the two models is logged rather than discovered in the output.
    detected, detected_probability = None, None
    if JOB.get("detect_language", True):
        detected, detected_probability = detect_language(
            audio_path, cache, device, compute_type)

    language = JOB.get("language")
    if not language and JOB.get("pin_language") and detected:
        language = detected
        log(f"Pinning language={language} from the probe.")
    elif not language:
        log("Language: auto (the transcription model decides).")

    model_ref = resolve_model(JOB.get("model") or "large-v3-turbo", cache)
    model = WhisperModel(model_ref, device=device, compute_type=compute_type)

    params = {
        "beam_size": JOB.get("beam_size", 5),
        "vad_filter": True,
        "condition_on_previous_text": False,
        "word_timestamps": True,
    }
    if language:
        params["language"] = language
    else:
        # Same reason as the backend: faster-whisper samples ONE window by
        # default, so applause or an English intro picks the language for the
        # whole file and a wrong pick garbles everything with no way back.
        params["language_detection_segments"] = JOB.get(
            "language_detection_segments", 4)

    started = time.time()
    segments, info = model.transcribe(audio_path, **params)

    out_segments = []
    text_parts = []
    for segment in segments:
        out_segments.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text,
            # Raw words on purpose. merge_continuation_words runs on the
            # backend so there is exactly one implementation of it.
            "words": [{"word": w.word, "start": float(w.start),
                       "end": float(w.end)}
                      for w in (segment.words or [])],
            "avg_logprob": _opt_float(getattr(segment, "avg_logprob", None)),
            "no_speech_prob": _opt_float(getattr(segment, "no_speech_prob", None)),
            "compression_ratio": _opt_float(
                getattr(segment, "compression_ratio", None)),
            "temperature": _opt_float(getattr(segment, "temperature", None)),
        })
        if segment.text.strip():
            text_parts.append(segment.text.strip())

    elapsed = time.time() - started
    log(f"Transcribed {len(out_segments)} segments in {elapsed:.1f}s "
        f"({device})")

    # The cross-check. Not fatal, and deliberately loud.
    if detected and info.language and detected != info.language:
        log(f"⚠️ LANGUAGE DISAGREEMENT: the small model heard '{detected}' "
            f"({detected_probability:.0%}) but {JOB.get('model')} transcribed "
            f"as '{info.language}' ({(info.language_probability or 0):.0%}). "
            f"If the output is in the wrong language this is why — set "
            f"KAGGLE_PIN_LANGUAGE=1 to trust the small model instead.")

    return {
        "text": " ".join(text_parts),
        "language": info.language,
        "language_probability": _opt_float(
            getattr(info, "language_probability", None)),
        "segments": out_segments,
        "asr": {
            "backend": "whisper",
            "model": JOB.get("model") or "large-v3-turbo",
            "device": device,
            "compute_type": compute_type,
            "task": "transcribe",
            "language_pinned": language or None,
            "language_pinned_by": (
                "caller" if JOB.get("language")
                else ("worker_probe" if language else None)),
            "language_detected": detected,
            "language_detect_probability": detected_probability,
            "language_disagreement": bool(
                detected and info.language and detected != info.language),
            "language_detection_segments": params.get(
                "language_detection_segments"),
            "source": "kaggle",
            "decode_seconds": round(elapsed, 2),
        },
    }


def _opt_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 3. Entry point
# ---------------------------------------------------------------------------

#: Everything else in /kaggle/working is deleted before the kernel ends.
#: Kaggle publishes the whole directory as the kernel's output, and the backend
#: fetches that output through a PAGINATED api (20 files a page). The BgUtils
#: git clone alone is thousands of files — .git/hooks/*.sample, devcontainer
#: config, node_modules — so transcript.json fell off the end of page one and
#: the backend downloaded twenty useless files and concluded the run produced
#: no transcript, having watched the kernel succeed.
KEEP_IN_OUTPUT = ("transcript.json", "result.json", "bgutil-server.log")


def prune_output():
    """Leave only the few files the backend actually fetches."""
    for entry in WORKDIR.iterdir():
        if entry.name in KEEP_IN_OUTPUT:
            continue
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
        except OSError as exc:
            log(f"Could not remove {entry.name}: {exc}")


def main():
    WORKDIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WORKDIR)

    log("=" * 70)
    log(f"OpenShorts ASR worker — job {JOB.get('job_id')}")
    log("=" * 70)

    cache = find_cache()
    log(f"Cache: {cache or 'NOT MOUNTED (slow path)'}")

    server = None
    try:
        deno, _server_dir, node_modules = install_downloader()
        server = start_token_server(deno, node_modules)
        audio = download_audio(JOB["url"], deno)
        transcript = transcribe(audio, cache)

        TRANSCRIPT_PATH.write_text(
            json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
        _write_result({
            "job_id": JOB.get("job_id"),
            "ok": True,
            "segments": len(transcript["segments"]),
            "language": transcript.get("language"),
        })
        log(f"\nWrote {TRANSCRIPT_PATH} "
            f"({len(transcript['segments'])} segments)")
    except Exception as exc:
        # A crash with no result file is indistinguishable from a hang on the
        # backend, which would then wait out the whole timeout before falling
        # back. Always leave something behind that says what went wrong.
        _write_result({
            "job_id": JOB.get("job_id"),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
        })
        log("\nWORKER FAILED:\n" + traceback.format_exc())
        raise
    finally:
        if server is not None:
            server.terminate()
        # AFTER result.json is written, in both the success and failure paths:
        # a small output is what makes the backend's paginated fetch find it.
        prune_output()


def _write_result(payload):
    try:
        RESULT_PATH.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best effort
        log(f"Could not write result.json: {exc}")


if __name__ == "__main__":
    main()
