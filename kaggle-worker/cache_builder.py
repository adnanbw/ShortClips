"""Build the warm cache the ASR worker mounts. Run once (and after upgrades).

It caches ONE thing: the faster-whisper weights, ~1.6 GB that would otherwise
be downloaded from HuggingFace on every single transcription job. The worker
lists this kernel in ``kernel_sources``; Kaggle mounts its output read-only at
``/kaggle/input/<slug>/`` and the worker loads the model straight off local
disk.

It deliberately does NOT cache the BgUtils node_modules, and that is a measured
decision rather than an omission. Deno does not keep packages inside
``node_modules`` — it symlinks into its own global cache — so a copied
``node_modules`` is an incomplete tree whose symlinks point at build-time
absolute paths. The first attempt at caching it got as far as starting the
token server and then died with "Could not find package 'lru-cache' from
referrer .../proxy-agent/dist/index.js". Making it work means also shipping
DENO_DIR and arranging for both builds to use identical absolute paths, which
is a lot of fragility to save the ~1-2 minutes ``deno install`` actually costs.
The worker installs it fresh instead.

Deliberately a KERNEL output and not a Kaggle Dataset: a dataset would have to
be built on the dev machine, i.e. download 1.6 GB at home to upload it back to
Google. A kernel's output never leaves Kaggle.

Re-run this when the model changes. It is idempotent.
"""
import os
import subprocess
import sys
from pathlib import Path

WORKDIR = Path("/kaggle/working")

#: Model names as faster-whisper knows them. The HuggingFace repo behind each
#: is DELIBERATELY not written here: faster_whisper.download_model owns that
#: mapping, and a copy of it in this file was already wrong once — it said
#: "Systran/faster-whisper-large-v3-turbo" while faster-whisper 1.2.1 resolves
#: large-v3-turbo to "mobiuslabsgmbh/faster-whisper-large-v3-turbo", and the
#: cache build died on a 401 for a repo that does not exist.
#:
#: large-v3 is FIRST because it is the default on Kaggle. It was excluded from
#: the local path for two reasons that a T4 erases: it costs ~1.8x turbo's
#: decode time on CPU, and loading it there got the container OOM-killed. On
#: the GPU it is ~3 GB of 16 and decodes in seconds, and it is measurably
#: better on exactly the audio this product struggles with — on 36s of the
#: Hinglish stand-up, turbo scored avg_logprob -0.944 (reader: BAD 10,
#: phonetic gibberish) against large-v3's -0.187 (PARTIAL 60, the punchline
#: intelligible), taking three sampled slices from 30% BAD to 0%.
#:
#: "small" stays because the worker cross-checks the detected LANGUAGE with it.
#: turbo stays so switching back costs no download.
BUILD = ["large-v3", "large-v3-turbo", "small"]


def run(cmd, cwd=None, env=None):
    print("\n>", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=str(cwd) if cwd else None,
                   check=True, env=env)


def build_models():
    """Download the CTranslate2 weights faster-whisper loads directly."""
    print("\n[2/2] Whisper models", flush=True)
    run([sys.executable, "-m", "pip", "install", "-q",
         "faster-whisper==1.2.1"])
    from faster_whisper.utils import download_model

    models_dir = WORKDIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for name in BUILD:
        target = models_dir / name
        if (target / "model.bin").exists():
            print(f"{name}: already present", flush=True)
            continue
        print(f"{name}: downloading", flush=True)
        # use_auth_token=False forces an ANONYMOUS fetch. These repos are
        # public, but a Kaggle image can carry an empty or stale HF_TOKEN in
        # its environment; huggingface_hub then sends it and gets back
        # 401 "Invalid username or password" for a repo needing no credentials.
        download_model(name, output_dir=str(target), use_auth_token=False)
        if not (target / "model.bin").exists():
            raise RuntimeError(f"{name} downloaded but model.bin is missing")
        size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        print(f"{name}: cached at {target} ({size / 1e9:.2f} GB)", flush=True)


def main():
    WORKDIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WORKDIR)
    print("=" * 70)
    print("OpenShorts ASR cache builder")
    print("=" * 70)

    build_models()

    print("\nCache contents:")
    for entry in sorted(WORKDIR.iterdir()):
        print("  ", entry.name)
    print("\nDONE. Point the worker at this kernel via kernel_sources.")


if __name__ == "__main__":
    main()
