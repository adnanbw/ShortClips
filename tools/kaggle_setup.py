"""One-time Kaggle setup: build the warm cache, then smoke-test the worker.

    python tools/kaggle_setup.py cache      # build/refresh the cache kernel
    python tools/kaggle_setup.py status     # what Kaggle thinks is running
    python tools/kaggle_setup.py test <url> # dispatch one real transcription

``cache`` is the one you run first, and it takes a while (it installs BgUtils'
npm dependencies and downloads ~1.6 GB of whisper weights ON KAGGLE — nothing
large is transferred to or from this machine). Every transcription job after it
mounts that output instead of repeating the work.

Requires KAGGLE_USERNAME + KAGGLE_KEY in the environment or .env; the key alone
is also accepted as KAGGLE_API_TOKEN, which is the shape the website hands you.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:
    pass

import kaggle_worker

WORKER_DIR = Path(kaggle_worker.WORKER_DIR)


def _require_api():
    api = kaggle_worker._api()
    if api is None:
        sys.exit("No usable Kaggle credentials. Set KAGGLE_USERNAME and "
                 "KAGGLE_KEY (or KAGGLE_API_TOKEN) in .env.")
    return api


def build_cache(args):
    """Push the cache kernel and wait for it to finish."""
    api = _require_api()
    slug = kaggle_worker._cache_kernel()
    if not slug or "/" not in slug:
        sys.exit("Cannot determine the cache kernel slug. Set "
                 "KAGGLE_CACHE_KERNEL=<user>/openshorts-asr-cache.")

    folder = Path(tempfile.mkdtemp(prefix="kaggle_cache_"))
    try:
        shutil.copy(WORKER_DIR / "cache_builder.py", folder / "cache_builder.py")
        metadata = {
            "id": slug,
            "title": slug.split("/")[-1],
            "code_file": "cache_builder.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            # No GPU: this only downloads files, and GPU quota is weekly and
            # scarce. Spending it on a download would be the wrong trade.
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": True,
            "keywords": [],
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }
        (folder / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")

        print(f"Pushing cache kernel {slug} ...")
        api.kernels_push(str(folder))
    finally:
        shutil.rmtree(folder, ignore_errors=True)

    print("Building. This installs npm deps and downloads ~1.6 GB of model "
          "weights on Kaggle; expect several minutes.")
    state = kaggle_worker._wait(api, slug, timeout=args.timeout, poll=20.0)
    if state in kaggle_worker.TERMINAL_OK:
        print(f"\nCache kernel COMPLETE. The worker will mount it "
              f"automatically via kernel_sources.")
        return 0
    print(f"\nCache build did not complete (state={state}). Logs:")
    _print_logs(api, slug)
    return 1


def status(args):
    api = _require_api()
    for slug in (kaggle_worker._cache_kernel(), kaggle_worker.kernel_slug()):
        if not slug or "/" not in slug:
            continue
        try:
            response = api.kernels_status(slug)
            state = kaggle_worker._status_name(response)
            message = getattr(response, "failure_message", "") or ""
            print(f"{slug}: {state}" + (f" — {message}" if message else ""))
        except Exception as exc:
            print(f"{slug}: unavailable ({type(exc).__name__}: {exc})")
    return 0


def test(args):
    """Dispatch one real job end to end and report what came back."""
    if not kaggle_worker.enabled():
        sys.exit("Kaggle is disabled or unconfigured (KAGGLE_ASR=0, or no key).")

    print(f"Worker kernel : {kaggle_worker.kernel_slug()}")
    print(f"Cache kernel  : {kaggle_worker._cache_kernel() or '(none)'}")
    print(f"GPU           : {os.environ.get('KAGGLE_GPU', '1') != '0'}")

    started = time.time()
    transcript = kaggle_worker.transcribe_url(args.url, job_id="setup-test")
    elapsed = time.time() - started

    if transcript is None:
        print(f"\nFAILED after {elapsed:.0f}s. Kernel logs:")
        api = kaggle_worker._api()
        if api:
            _print_logs(api, kaggle_worker.kernel_slug())
        return 1

    segments = transcript["segments"]
    asr = transcript.get("asr") or {}
    print(f"\nOK in {elapsed:.0f}s wall clock")
    print(f"  language : {transcript.get('language')} "
          f"({transcript.get('language_probability')})")
    print(f"  segments : {len(segments)}")
    print(f"  device   : {asr.get('device')} / {asr.get('compute_type')}")
    print(f"  decode   : {asr.get('decode_seconds')}s")
    print("\nFirst lines:")
    for segment in segments[:5]:
        print(f"  [{segment['start']:.2f}-{segment['end']:.2f}] "
              f"{segment['text'].strip()[:90]}")
    return 0


def _print_logs(api, slug, tail=4000):
    try:
        logs = api.kernels_logs(slug)
    except Exception as exc:
        print(f"  (could not fetch logs: {type(exc).__name__}: {exc})")
        return
    if isinstance(logs, (list, dict)):
        logs = json.dumps(logs, indent=1)[-tail:]
    else:
        logs = str(logs)[-tail:]
    print(logs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cache = sub.add_parser("cache", help="build/refresh the cache kernel")
    cache.add_argument("--timeout", type=float, default=3600.0)
    cache.set_defaults(func=build_cache)

    st = sub.add_parser("status", help="status of both kernels")
    st.set_defaults(func=status)

    tt = sub.add_parser("test", help="dispatch one real transcription")
    tt.add_argument("url")
    tt.set_defaults(func=test)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
