"""Transcription backends: NVIDIA Parakeet (onnx-asr) with faster-whisper fallback.

Every caller goes through transcribe_media(), which returns the transcript
contract the whole pipeline depends on:

    {
      "text": str,          # full punctuated transcript
      "language": str,      # whisper-style short code ("es", "en", ...)
      "segments": [
        {"start": float, "end": float, "text": str,
         "words": [{"word": str, "start": float, "end": float}, ...]},
      ],
    }

Invariants the consumers rely on (clip cutting, karaoke subtitles, Remotion):
  - word["word"] carries a LEADING SPACE on true word starts; continuation
    fragments are merged into their base word (merge_continuation_words).
  - all numerics are native Python floats (json.dump of the transcript).
  - words sorted by start, segments chronological, absolute file timestamps.

Additive DIAGNOSTIC keys (safe to ignore; nothing above depends on them):
  - "language_probability": how sure the detector was, 0-1.
  - "asr": {"backend", "model", "device", "compute_type", "task",
            "quality": <asr_quality verdict>, "attempts": [...]}.
  - per segment: "avg_logprob", "no_speech_prob", "compression_ratio",
    "temperature" — whisper computes these and the pipeline used to discard
    them, which is how a hallucinated transcript reached the clip selector.

TRANSCRIBE_BACKEND env: "whisper" (default) | "parakeet".
The parakeet path falls back to whisper automatically when the model errors,
produces no usable words, or the detected language is outside its 25
supported European languages (e.g. Japanese/Chinese/Arabic uploads).
GPU whisper in turn falls back to CPU whisper on CUDA errors (VRAM is shared
with other models on the host, so loads can OOM under load).

Two entry points:
  - transcribe_media(path): ONE pass with the configured backend. Unchanged.
  - transcribe_media_checked(path, ...): the adaptive path the clip pipeline
    uses — transcribe, score with asr_quality, and retry ONCE with
    WHISPER_RETRY_MODEL only when the result looks unreliable. A clean video
    (English or otherwise) is still transcribed exactly once; language never
    decides the retry, quality does. Both attempts unusable raises
    TranscriptQualityError so the job names the real cause.

Which entry point a caller should use:
  - main.transcribe_video           -> checked. Feeds the meaningful selector.
  - subtitles.transcribe_audio      -> raw, on purpose: captions for an
                                       ElevenLabs dub of an ALREADY RENDERED
                                       clip. Synthetic audio, and a quality
                                       failure there would fail a finished job.
  - saasshorts.transcribe_audio_for_subs -> raw, on purpose: word timings for
                                       captions over its own TTS narration.
Both raw callers produce SUBTITLE TIMING only; neither feeds clip selection.
Anything that starts reasoning about the words must use the checked path.

The task is always "transcribe", never "translate": clip timing, captions and
metadata all need the words that were actually spoken, in the spoken language.
"""
import gc
import json
import os
import subprocess
import tempfile
import threading
import time

from subtitles import (
    get_whisper_config,
    get_whisper_retry_model,
    whisper_model_ladder,
    WHISPER_TRANSCRIBE_PARAMS,
    whisper_transcribe_params,
    merge_continuation_words,
)
from asr_quality import (
    TranscriptQualityError,
    evaluate_transcript,
    format_quality_line,
    good_score_threshold,
    quality_reason_lines,
    rank_quality,
)
import asr_semantic

PARAKEET_MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"

# The 25 European languages parakeet-tdt-0.6b-v3 supports (ISO 639-1).
PARAKEET_LANGS = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru",
    "uk",
}

# Serializes GPU transcription across concurrent jobs so N jobs can't stack
# N model contexts / decode batches in VRAM. CPU whisper stays ungated
# (CTranslate2 models are thread-safe and that matches the old behavior).
_ASR_GATE = threading.Semaphore(int(os.environ.get("ASR_GPU_CONCURRENCY", "1")))


class _NullGate:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_GATE = _NullGate()


class _TranscribeProgress:
    """Emits '🎙️ Transcribing… NN% (Xs)' lines at 25% steps.

    These are the only transcription lines cloud users see (log_view keeps
    them), so they must stay free of technical detail.
    """

    def __init__(self, total_seconds):
        self.total = max(float(total_seconds or 0), 0.0)
        self.started = time.time()
        self.next_pct = 25

    def update(self, position_seconds):
        if self.total <= 0:
            return
        pct = min(int(position_seconds / self.total * 100), 100)
        while pct >= self.next_pct and self.next_pct <= 100:
            elapsed = int(time.time() - self.started)
            print(f"🎙️ Transcribing… {self.next_pct}% ({elapsed}s)", flush=True)
            self.next_pct += 25

# --- whisper singleton ------------------------------------------------------

_whisper_model = None
_whisper_key = None
_whisper_lock = threading.Lock()
# Bounded cache of resident models. ONE by default: the probe now picks the
# right model up front, so the "keep the retry model warm too" case it was
# sized for is the exception rather than the norm, and two large models
# resident on a CPU box is what the OOM killer ends. WHISPER_MODEL_CACHE
# raises it where the RAM exists.
_whisper_cache = {}
_whisper_cache_order = []
# Set after a CUDA failure (e.g. VRAM exhausted by other models on the GPU)
# so every later transcription goes straight to CPU instead of re-failing.
_whisper_force_cpu = False


def _whisper_cache_max():
    try:
        return max(1, int(os.environ.get("WHISPER_MODEL_CACHE", "1")))
    except ValueError:
        return 1


def _get_whisper_model(model_size=None):
    """Process-wide WhisperModel, rebuilt if the env config changes.

    Keeping the model resident avoids a full reload per transcription (which
    on GPU would also mean re-allocating a couple of GB of VRAM per job).
    ``model_size`` overrides WHISPER_MODEL for the quality retry; the cache
    keeps both models so alternating jobs don't reload on every call.
    """
    global _whisper_model, _whisper_key
    cfg = get_whisper_config(model_size)
    if _whisper_force_cpu:
        cfg["device"] = "cpu"
        cfg["compute_type"] = "int8"
    key = (cfg["model_size"], cfg["device"], cfg["compute_type"])
    with _whisper_lock:
        model = _whisper_cache.get(key)
        if model is None:
            from faster_whisper import WhisperModel
            # Evict BEFORE allocating, not after. The old order held the
            # outgoing model's weights for the whole load of the incoming one,
            # so the peak was two models even with a cache of one — which is
            # how a third whisper load got the container OOM-killed (exit -9)
            # with large-v3-turbo still resident.
            while len(_whisper_cache_order) >= _whisper_cache_max():
                _whisper_cache.pop(_whisper_cache_order.pop(0), None)
            gc.collect()
            model = WhisperModel(key[0], device=key[1], compute_type=key[2])
            _whisper_cache[key] = model
            _whisper_cache_order.append(key)
        # Kept for the existing singleton semantics/tests and for the CUDA
        # fallback, which drops the resident model to release its VRAM.
        _whisper_model = model
        _whisper_key = key
    return model, cfg["device"]


def _drop_whisper_models():
    """Release every cached model (used after a CUDA failure)."""
    global _whisper_model, _whisper_key
    with _whisper_lock:
        _whisper_cache.clear()
        del _whisper_cache_order[:]
        _whisper_model = None
        _whisper_key = None


def _run_whisper_once(media_path, model_size=None, show_progress=True, **params):
    model, device = _get_whisper_model(model_size)
    gate = _ASR_GATE if device != "cpu" else _NULL_GATE
    with gate:
        try:
            segments, info = model.transcribe(media_path, **params)
        except TypeError as e:
            # language_detection_segments landed in faster-whisper 1.0.3; an
            # older wheel must still transcribe rather than crash the job.
            unknown = [k for k in ("language_detection_segments",) if k in params]
            if not unknown or "unexpected keyword" not in str(e):
                raise
            for key in unknown:
                params.pop(key, None)
            segments, info = model.transcribe(media_path, **params)
        progress = _TranscribeProgress(
            getattr(info, "duration", 0) if show_progress else 0)
        materialized = []
        for segment in segments:
            materialized.append(segment)
            progress.update(segment.end)
        # VAD trims trailing silence, so the last segment can end short of the
        # media duration — force the 100% line.
        progress.update(progress.total)
        return materialized, info


def run_whisper_transcription(media_path, model_size=None, show_progress=True,
                              **params):
    """Transcribe and FULLY materialize the segments inside the GPU gate.

    faster-whisper returns a lazy generator — decoding happens while
    iterating, so the gate must wrap list(segments), not just transcribe().
    Returns (segments_list, info).

    A CUDA failure (model load OOM or mid-decode) retries once on CPU and
    pins CPU for the rest of the process — the GPU is shared with other
    models, so a job must degrade instead of dying when VRAM runs out.
    """
    global _whisper_force_cpu
    try:
        return _run_whisper_once(media_path, model_size=model_size,
                                 show_progress=show_progress, **params)
    except RuntimeError as e:
        if _whisper_force_cpu or "cuda" not in str(e).lower():
            raise
        print(f"⚠️ [ASR] whisper GPU failed ({e}) — retrying on CPU", flush=True)
        _whisper_force_cpu = True
        _drop_whisper_models()  # release the GPU models' VRAM
        return _run_whisper_once(media_path, model_size=model_size,
                                 show_progress=show_progress, **params)


def _opt_float(value):
    """Native float or None — numpy scalars and missing attrs both appear."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _transcribe_with_whisper(media_path, model_size=None, language=None):
    """One whisper pass. The extra per-segment/decoder fields are DIAGNOSTICS:
    additive keys on the same transcript contract, read by asr_quality and
    ignored by every existing consumer.

    ``language`` pins the spoken language instead of auto-detecting it. Only
    the quality retry uses it — see _retry_language."""
    params = whisper_transcribe_params()
    if language:
        params["language"] = language
        params.pop("language_detection_segments", None)
    segments, info = run_whisper_transcription(
        media_path, model_size=model_size, **params)

    out_segments = []
    text_parts = []
    for segment in segments:
        words = [
            {"word": w.word, "start": float(w.start), "end": float(w.end)}
            for w in (segment.words or [])
        ]
        out_segments.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text,
            "words": merge_continuation_words(words),
            # Decoder diagnostics — whisper computes these anyway and the
            # pipeline used to throw them away, which is why a hallucinated
            # transcript reached the clip selector unchallenged.
            "avg_logprob": _opt_float(getattr(segment, "avg_logprob", None)),
            "no_speech_prob": _opt_float(getattr(segment, "no_speech_prob", None)),
            "compression_ratio": _opt_float(getattr(segment, "compression_ratio", None)),
            "temperature": _opt_float(getattr(segment, "temperature", None)),
        })
        text_parts.append(segment.text.strip())

    cfg = get_whisper_config(model_size)
    transcript = {
        "text": " ".join(part for part in text_parts if part),
        "language": info.language,
        "segments": out_segments,
        "language_probability": _opt_float(
            getattr(info, "language_probability", None)),
        "asr": {
            "backend": "whisper",
            "model": cfg["model_size"],
            "device": "cpu" if _whisper_force_cpu else cfg["device"],
            "compute_type": "int8" if _whisper_force_cpu else cfg["compute_type"],
            "task": "transcribe",
            "language_pinned": language or None,
            "language_detection_segments": params.get("language_detection_segments"),
        },
    }
    return transcript


# --- parakeet ---------------------------------------------------------------

_parakeet_model = None
_parakeet_lock = threading.Lock()


def _get_parakeet_model():
    global _parakeet_model
    with _parakeet_lock:
        if _parakeet_model is None:
            import onnx_asr
            model = onnx_asr.load_model(
                PARAKEET_MODEL_ID,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            vad = onnx_asr.load_vad("silero")
            _parakeet_model = model.with_vad(vad).with_timestamps()
    return _parakeet_model


def _extract_wav(media_path):
    """Parakeet wants 16kHz mono PCM wav; ffmpeg-extract to a temp file."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="asr_")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", media_path,
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, timeout=1800)
    return wav_path


def _words_from_tokens(tokens, timestamps, seg_start, seg_end):
    """Group parakeet BPE tokens into words with absolute timestamps.

    Verified on the prod model: tokens already carry the leading-space
    word-start convention (" T", "odo", " el", ...) and timestamps are token
    START times in seconds relative to the VAD segment. A token without a
    leading space (subword continuations, punctuation like ",") belongs to
    the previous word — same semantics merge_continuation_words expects.
    Word end is inferred: next word's start, capped near the word's last
    token so a long inter-word silence doesn't stretch the highlight.
    """
    words = []
    last_token_ts = []
    for token, ts in zip(tokens, timestamps):
        if not token:
            continue
        abs_ts = float(ts) + seg_start
        if token.startswith(" ") or not words:
            words.append({
                "word": token if token.startswith(" ") else " " + token,
                "start": abs_ts,
            })
            last_token_ts.append(abs_ts)
        else:
            words[-1]["word"] += token
            last_token_ts[-1] = abs_ts

    for i, word in enumerate(words):
        next_start = words[i + 1]["start"] if i + 1 < len(words) else seg_end
        cap = last_token_ts[i] + 0.6
        word["end"] = float(max(word["start"] + 0.05, min(next_start, cap)))

    return words


def _transcribe_with_parakeet(media_path):
    model = _get_parakeet_model()
    wav_path = _extract_wav(media_path)
    try:
        # 16kHz mono s16le wav -> 32000 bytes per second of audio.
        try:
            duration = os.path.getsize(wav_path) / 32000.0
        except OSError:
            duration = 0.0
        with _ASR_GATE:
            progress = _TranscribeProgress(duration)
            results = []
            for seg in model.recognize(wav_path):
                results.append(seg)
                progress.update(float(seg.end))
            progress.update(progress.total)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    out_segments = []
    text_parts = []
    for seg in results:
        seg_start = float(seg.start)
        seg_end = float(seg.end)
        seg_text = str(seg.text or "").strip()
        if not seg_text:
            continue
        out_segments.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg_text,
            "words": _words_from_tokens(
                list(seg.tokens or []), list(seg.timestamps or []),
                seg_start, seg_end,
            ),
        })
        text_parts.append(seg_text)

    text = " ".join(text_parts)
    return {
        "text": text,
        "language": _detect_language(text),
        "segments": out_segments,
        "asr": {
            "backend": "parakeet",
            "model": PARAKEET_MODEL_ID,
            "task": "transcribe",
        },
    }


def _detect_language(text):
    """Parakeet doesn't report a language; classify the transcribed text.

    py3langid is pure-Python and returns ISO 639-1 codes compatible with the
    whisper codes the pipeline expects (thumbnail titles, Gemini prompts).
    """
    sample = (text or "").strip()
    if len(sample) < 20:
        return "en"
    try:
        import py3langid
        lang, _score = py3langid.classify(sample[:4000])
        return lang
    except Exception:
        return "en"


def _parakeet_fallback_reason(transcript, duration_hint=None):
    """Return why the parakeet result is untrustworthy, or None if it's fine."""
    segments = transcript.get("segments") or []
    total_words = sum(len(s.get("words") or []) for s in segments)
    if total_words == 0:
        return "no words recognized"
    language = transcript.get("language")
    if language not in PARAKEET_LANGS:
        return f"language '{language}' outside parakeet's supported set"
    duration = duration_hint or (segments[-1]["end"] if segments else 0)
    # Real speech averages >100 wpm; under ~12 wpm on a long video means the
    # audio was mostly not recognized (e.g. unsupported language or music).
    if duration > 60 and total_words < duration * 0.2:
        return f"only {total_words} words in {duration:.0f}s of audio"
    return None


# --- public entry point -----------------------------------------------------

class NoAudioError(Exception):
    """The media has no audio track — nothing to transcribe."""


def _has_audio_stream(media_path) -> bool:
    """True if the file has at least one audio stream (ffprobe)."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", media_path],
            capture_output=True, text=True, timeout=60,
        )
        return bool(out.stdout.strip())
    except Exception:
        return True  # probe failed — don't block, let the backend try


def transcribe_media(media_path, model_size=None, language=None):
    """Transcribe with the configured backend, falling back to whisper.

    ``model_size``/``language`` come from the probe (_probe_best_model) and are
    whisper-only: parakeet has one model and detects for itself, so a probe
    that asked for a specific whisper model skips it rather than silently
    ignoring the request.
    """
    # Silent videos (AI-generated clips, muted screen recordings) have no audio
    # stream; every ASR backend then crashes deep inside libav with an opaque
    # "tuple index out of range". Detect it up front and fail with a clear,
    # actionable reason instead.
    if not _has_audio_stream(media_path):
        raise NoAudioError(
            "This video has no audio track. OpenShorts finds viral moments from "
            "speech, so it needs a video with audio.")

    backend = os.environ.get("TRANSCRIBE_BACKEND", "whisper").strip().lower()

    if backend == "parakeet" and not model_size:
        try:
            transcript = _transcribe_with_parakeet(media_path)
            reason = _parakeet_fallback_reason(transcript)
            if reason is None:
                print(f"🎙️ [ASR] parakeet ok: lang={transcript['language']} "
                      f"segments={len(transcript['segments'])}")
                return transcript
            print(f"⚠️ [ASR] parakeet result rejected ({reason}) — "
                  f"falling back to whisper")
        except Exception as e:
            print(f"⚠️ [ASR] parakeet failed ({type(e).__name__}: {e}) — "
                  f"falling back to whisper")

    return _transcribe_with_whisper(media_path, model_size=model_size,
                                    language=language)


# --- adaptive path: transcribe, judge, retry only if needed -----------------

def _save_debug(debug_dir, filename, payload):
    """Best effort — a debug file must never fail a job."""
    if not debug_dir:
        return
    try:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, filename), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        print(f"⚠️ Could not save ASR debug file {filename}: {e}")


def _describe(transcript):
    asr = (transcript or {}).get("asr") or {}
    return (f"backend={asr.get('backend', '?')} model={asr.get('model', '?')} "
            f"device={asr.get('device', 'n/a')}")


def _auto_retry_enabled():
    return os.environ.get("ASR_AUTO_RETRY", "1").strip() != "0"


def _retry_language(initial, quality):
    """The language to PIN on the retry, or None to let it detect again.

    The retry exists to transcribe the same speech better, not to transcribe
    different speech. Measured on the Hinglish stand-up: whisper-small detected
    ``hi`` at 89% and large-v3-turbo, re-detecting on its own, decided ``en`` at
    96% and emitted an ENGLISH TRANSLATION of the Hindi audio ("This was the
    disconnecting flight of Bali, SpiceJet") even though the task is always
    "transcribe". That is whisper's known behaviour once the language token is
    wrong, and it silently breaks the product's promise to keep the original
    spoken language — clip timing, captions and metadata would all switch
    language behind the user's back. The quality gate cannot catch it either:
    that translation scored 99/100 because it is perfectly fluent English.

    So when the first attempt was reasonably sure of the language, the retry is
    told to use it. When it was NOT sure, the detection may well be what is
    wrong, and the stronger model gets to decide for itself.
    """
    language = str(initial.get("language") or "").strip()
    probability = initial.get("language_probability")
    if not language:
        return None
    # A transcript with no probability at all (parakeet, an old checkpoint)
    # carries no evidence either way — let the stronger model detect.
    if probability is None:
        return None
    try:
        floor = float(os.environ.get("ASR_PIN_LANGUAGE_ABOVE", "0.5"))
    except ValueError:
        floor = 0.5
    return language if float(probability) >= floor else None


# --- semantic validation: are these actually words? -------------------------

def _semantic_should_run(quality, forced_reason, mode):
    """Whether to spend one Gemini call reading this transcript.

    ``ASR_QUALITY_GEMINI`` is the contract and it is honoured literally:

    ``0``    never.
    ``1``    always - every ASR result is read, once per attempt.
    ``auto`` only when justified, which is exactly three situations:
             * a GOOD verdict that rests on structure alone (``uncertain``);
             * an attempt made BECAUSE an earlier one was semantic gibberish -
               the retry has to prove it fixed the thing it was run for, and a
               confident decoder must not be allowed to skip that proof. This
               is the case the previous version got wrong: large-v3-turbo came
               back at avg_logprob -0.29 and score 100 on a transcript that was
               still a third invented, and nothing looked;
             * a transcript about to be refused, as a last reprieve.
    """
    if mode == "0":
        return False
    if mode == "1":
        return True
    return bool(forced_reason) or bool(quality.get("uncertain"))


def _apply_semantic(quality, semantic):
    """Fold a semantic verdict into the structural one.

    Semantic evidence outranks the structural score in one direction only: it
    can condemn a structurally perfect transcript (that is the whole point) but
    a clean read never rescues a structurally damaged one.
    """
    if not semantic or not semantic.get("status"):
        return quality

    merged = dict(quality)
    merged["semantic"] = semantic
    verdict = semantic["status"]
    reasons = list(quality.get("reasons") or [])

    if verdict == "GOOD":
        return merged

    if verdict == "PARTIAL":
        good_share = (semantic.get("shares") or {}).get("GOOD", 0)
        reasons.append("parts of the transcript are not real words "
                       f"({good_share:.0%} of the examined speech reads correctly)")
        merged["status"] = "PARTIAL"
    else:
        reasons.append("the transcribed words are not real words in any language")
        merged["status"] = "BAD"

    merged["reasons"] = reasons
    # Structure was never the problem, so the structural number stays
    # misleadingly high; pull it under the accept line so the logs and the
    # attempt comparison both agree with the status.
    merged["score"] = round(min(float(quality.get("score") or 0.0),
                                good_score_threshold() - 0.1), 1)
    merged["downgraded_by"] = "language_check"
    return merged


def _read_the_words(transcript, quality, forced_reason=None, label="Transcript"):
    """Run the semantic check when it is due, and fold the result in.

    ONE call, and its result is advisory. There used to be a second, dense pass
    over the whole timeline whenever the first came back PARTIAL, whose only
    consumer was a filter that deleted candidate clips overlapping a region it
    disliked. Both are gone: a sample of 25-second windows is good evidence
    that a transcript is rough and poor evidence about any particular clip, and
    the Blind Context Critic already reads every candidate in full.
    """
    mode = asr_semantic.check_mode()
    if not _semantic_should_run(quality, forced_reason, mode):
        return quality

    why = (f" (this attempt exists because: {forced_reason})" if forced_reason
           else " (structurally fine, but the decoder was unsure)")
    print(f"[check] Reading the transcript to confirm the words are real{why}...")

    semantic = asr_semantic.review_transcript(transcript)
    if semantic is None:
        # No key, disabled, or the API failed. An unavailable reader is not a
        # guilty verdict: fall through to the structural decision.
        print("[check] Language check unavailable - keeping the structural verdict.")
        return quality

    print(asr_semantic.format_summary(semantic))
    for region in semantic.get("regions") or []:
        if region["status"] != "GOOD":
            print(f"   {region['id']} [{region['start']:.0f}-{region['end']:.0f}s] "
                  f"{region['status']} ({region['score']}): {region['reason']}")

    return _apply_semantic(quality, semantic)


# --- attempt bookkeeping ----------------------------------------------------

def _attempt_summary(number, transcript, quality, reason=None):
    """A FLAT, self-contained record of one attempt.

    Deliberately copies scalars out of the transcript's own ``asr`` dict rather
    than referencing it. The previous version stored ``initial.get("asr")``
    here and then hung the attempt list off that very dict, so the structure
    pointed at itself and every job logged "Could not save transcript
    checkpoint: Circular reference detected" - losing the checkpoint that
    exists so a redeploy does not pay for transcription twice.
    """
    asr = (transcript or {}).get("asr") or {}
    quality = quality or {}
    semantic = quality.get("semantic") or {}
    return {
        "attempt": number,
        "backend": asr.get("backend"),
        "model": asr.get("model"),
        "device": asr.get("device"),
        "compute_type": asr.get("compute_type"),
        "language": (transcript or {}).get("language"),
        "language_probability": (transcript or {}).get("language_probability"),
        "language_pinned": asr.get("language_pinned"),
        "status": quality.get("status"),
        "score": quality.get("score"),
        "semantic_status": semantic.get("status"),
        "semantic_score": semantic.get("score"),
        "reason": reason,
    }


def _attach_quality(transcript, quality, attempts):
    asr = transcript.setdefault("asr", {})
    asr["quality"] = quality
    asr["attempts"] = list(attempts)
    return transcript


def _retry_reason(quality):
    """Why a retry is being run, in words, or None when it is not needed."""
    if quality.get("downgraded_by") == "language_check":
        semantic = quality.get("semantic") or {}
        status = str(semantic.get("status") or "gibberish").lower()
        return f"semantic_{status}"
    if quality.get("status") == "GOOD":
        return None
    return f"structural_{str(quality.get('status', '')).lower()}"


def _print_language(transcript):
    language = transcript.get("language")
    if not language:
        return
    probability = transcript.get("language_probability")
    confidence = f" (confidence {probability:.0%})" if probability else ""
    print(f"[asr] Detected language: {language}{confidence}")


# --- the probe: choose the model BEFORE transcribing the whole video --------

def _env_int(name, default):
    try:
        return int(float(os.environ.get(name, "").strip() or default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


#: How many slices the probe decodes, and how long each is. Three ~25s windows
#: is ~75 seconds of audio: roughly 8% of a ten-minute video, spread across it.
DEFAULT_PROBE_SLICES = 3
DEFAULT_PROBE_SECONDS = 25.0


def _probe_enabled():
    return os.environ.get("ASR_PROBE", "1").strip() != "0"


def _probe_slices(duration, count, seconds):
    """``count`` evenly-spread [start, end] windows inside ``duration``.

    Spread across the WHOLE timeline for the same reason the semantic reader
    samples that way: the first 30 seconds of a video are the least
    representative part of it — applause, music, an intro in another language.
    """
    if not duration or duration <= 0:
        return []
    count = max(1, int(count))
    seconds = max(float(seconds or 0.0), 1.0)
    seconds = min(seconds, max(duration / (count + 1.0), 5.0))
    span = duration / float(count)
    slices = []
    for i in range(count):
        start = i * span + (span - seconds) / 2.0
        start = max(0.0, min(start, max(duration - seconds, 0.0)))
        slices.append((round(start, 2), round(min(start + seconds, duration), 2)))
    return slices


def _probe_transcribe(media_path, model_size, slices):
    """Decode just those windows, as one transcript-shaped dict.

    Uses faster-whisper's own ``clip_timestamps`` so nothing is cut to disk and
    no ffmpeg pass is needed; the returned shape is what evaluate_transcript
    reads, so the probe is scored by exactly the same code as a full pass.
    """
    spec = ",".join(f"{start},{end}" for start, end in slices)
    # faster-whisper ignores vad_filter when clip_timestamps is set, so say so
    # rather than passing a setting that silently does nothing. The progress
    # lines are off because they are the only transcription output cloud users
    # see, and "Transcribing… 100%" twice per job reads like a bug.
    params = whisper_transcribe_params(clip_timestamps=spec, vad_filter=False)
    segments, info = run_whisper_transcription(
        media_path, model_size=model_size, show_progress=False, **params)

    out = []
    for segment in segments:
        # The word timestamps are NOT optional here. evaluate_transcript
        # penalises a segment that has text but no words by up to 55 points
        # ("N% of segments have no word timestamps"), so a probe that dropped
        # them would score badly on every video and upgrade all of them —
        # exactly the always-run-the-big-model behaviour it exists to avoid.
        words = [
            {"word": w.word, "start": float(w.start), "end": float(w.end)}
            for w in (segment.words or [])
        ]
        out.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text,
            "words": merge_continuation_words(words),
            "avg_logprob": _opt_float(getattr(segment, "avg_logprob", None)),
            "no_speech_prob": _opt_float(getattr(segment, "no_speech_prob", None)),
            "compression_ratio": _opt_float(
                getattr(segment, "compression_ratio", None)),
            "temperature": _opt_float(getattr(segment, "temperature", None)),
        })
    return {
        "text": " ".join(s["text"].strip() for s in out if s["text"].strip()),
        "language": info.language,
        "language_probability": _opt_float(
            getattr(info, "language_probability", None)),
        "segments": out,
    }


def _probe_best_model(media_path, duration):
    """Which model should transcribe this video, decided from ~75s of it.

    This exists because the old design was backwards. It transcribed the whole
    video with the base model, judged the result, and — when that model could
    not handle the audio — transcribed the WHOLE video again with a stronger
    one. On the Hinglish stand-up that is 17 minutes of CPU spent producing
    something already known to be unusable, and the escalation then went for a
    third pass that the OOM killer ended.

    Decoding ~75 seconds first costs ~15s and answers the same question. It is
    a QUALITY decision, not a language one: there is no list of languages that
    get the big model. A noisy English video is upgraded by the same rule, and
    a clean Hindi one is not upgraded at all.

    Returns ``(model_size, language, language_probability)``; the model is None
    when the probe did not run or found no reason to change anything.
    """
    stronger = get_whisper_retry_model()
    configured = get_whisper_config()["model_size"]
    if not _probe_enabled() or not stronger or stronger == configured:
        return None, None, None
    if not duration or duration < DEFAULT_PROBE_SECONDS * 2:
        return None, None, None

    slices = _probe_slices(duration, _env_int("ASR_PROBE_SLICES",
                                              DEFAULT_PROBE_SLICES),
                           _env_float("ASR_PROBE_SECONDS",
                                      DEFAULT_PROBE_SECONDS))
    if not slices:
        return None, None, None

    try:
        sampled = _probe_transcribe(media_path, configured, slices)
    except Exception as e:
        # The probe is an optimisation. It must never be the reason a job
        # fails: fall through and transcribe the way we always did.
        print(f"[asr] Probe skipped ({type(e).__name__}: {e}).")
        return None, None, None

    quality = evaluate_transcript(sampled, sum(e - s for s, e in slices))
    language = sampled.get("language")
    probability = sampled.get("language_probability")
    confidence = f" ({probability:.0%})" if probability else ""
    print(f"[asr] Probed {len(slices)} sample(s) with {configured}: "
          f"language={language}{confidence}")
    print(format_quality_line(quality, prefix="   Probe quality"))

    # ``uncertain`` matters as much as the status here, and this is the whole
    # reason the probe beats the old retry. On the real Hinglish video,
    # whisper-small's full transcript scored 77.8/GOOD while being a fifth
    # invented: structure could not see it, but the decoder's own confidence
    # (avg_logprob -0.89) said it was guessing. That flag is available after
    # 75 seconds just as well as after 17 minutes, so it is read there.
    struggling = quality.get("status") != "GOOD" or quality.get("uncertain")
    if quality.get("skipped") or not struggling:
        return None, language, probability

    why = ("the decoder was guessing" if quality.get("uncertain")
           else f"probe {quality.get('status')}, score={quality.get('score')}")
    print(f"[asr] {configured} struggles with this audio ({why}) - "
          f"transcribing with {stronger}, instead of transcribing twice.")
    return stronger, language, probability


# --- the adaptive entry point -----------------------------------------------

def transcribe_media_checked(media_path, duration=None, debug_dir=None,
                             raise_on_bad=True):
    """Transcribe once with the right model, and repair only real failure.

    Returns the transcript contract plus ``asr.quality``.

    The shape of this used to be "transcribe, judge, escalate, judge,
    escalate": up to three full passes over the same video. That is what made
    a Hinglish job take 50 minutes and then die — see _probe_best_model and
    _accepts. Now:

      1. ~75 seconds are decoded to pick the model (_probe_best_model);
      2. the video is transcribed ONCE with it;
      3. only a transcript judged BAD is transcribed again, once.

    Final statuses and what they mean for the caller:
      GOOD     use it.
      PARTIAL  use it. Parts of it read poorly, ``asr.quality.semantic`` says
               which, but a transcript does not have to be CORRECT to locate a
               moment in a video — it has to be good enough to find the words.
      RETRY    structurally suspicious, nothing proven wrong. Use it.
      BAD      raises TranscriptQualityError (unless ``raise_on_bad`` is off).
    """
    model, probed_language, probed_probability = _probe_best_model(
        media_path, duration)
    pinned = _probed_language_pin(model, probed_language, probed_probability)
    initial = transcribe_media(media_path, model_size=model, language=pinned)
    quality = evaluate_transcript(initial, duration)

    print(f"[asr] ASR attempt 1: {_describe(initial)}")
    _print_language(initial)
    print(format_quality_line(quality))
    for line in quality_reason_lines(quality):
        print(line)

    quality = _read_the_words(initial, quality, label="Attempt 1")

    _save_debug(debug_dir, "asr_initial.json", initial)
    _save_debug(debug_dir, "asr_quality_initial.json", quality)

    attempts = [_attempt_summary(1, initial, quality)]

    if _accepts(quality):
        return _settle(initial, quality, attempts, raise_on_bad)

    used = [m for m in [(initial.get("asr") or {}).get("model")] if m]
    ladder = whisper_model_ladder(already_used=used)

    if not _auto_retry_enabled() or quality.get("skipped"):
        why = "disabled" if not _auto_retry_enabled() else "quality gate off"
        print(f"[asr] Not retrying transcription ({why}).")
        return _settle(initial, quality, attempts, raise_on_bad)
    if not ladder:
        print("[asr] Not retrying transcription (no stronger model left to try).")
        return _settle(initial, quality, attempts, raise_on_bad)

    best, best_quality = initial, quality
    model = ladder[0]
    reason = _retry_reason(best_quality)
    pinned = _retry_language(initial, best_quality)
    print(f"[asr] Retrying transcription with model={model}"
          + (f" language={pinned} (kept from attempt 1)" if pinned
             else " language=auto-detect")
          + f" - reason: {reason}")
    try:
        candidate = _transcribe_with_whisper(media_path, model_size=model,
                                             language=pinned)
    except Exception as e:
        print(f"[asr] Attempt 2 failed ({type(e).__name__}: {e}) - "
              f"keeping the best transcript so far.")
        return _settle(best, best_quality, attempts, raise_on_bad)

    candidate_quality = evaluate_transcript(candidate, duration)
    print(f"[asr] ASR attempt 2: {_describe(candidate)}")
    _print_language(candidate)
    print(format_quality_line(candidate_quality,
                              prefix="Attempt 2 transcript quality"))
    for line in quality_reason_lines(candidate_quality):
        print(line)

    # The attempt must prove it fixed the reason it exists for. A strong
    # decoder score is exactly what that failure mode looks like, so passing
    # ``forced_reason`` here is what stops a confident model from skipping the
    # only check that can see the problem.
    candidate_quality = _read_the_words(
        candidate, candidate_quality,
        forced_reason=reason if str(reason).startswith("semantic") else None,
        label="Attempt 2")

    _save_debug(debug_dir, "asr_retry.json", candidate)
    _save_debug(debug_dir, "asr_quality_retry.json", candidate_quality)
    attempts.append(_attempt_summary(2, candidate, candidate_quality,
                                     reason=reason))

    # Keep whichever attempt actually came out better. Status first, then
    # score: the language check can condemn a transcript whose structural
    # score is still high, and that must lose to a merely-decent real one.
    if rank_quality(candidate_quality) >= rank_quality(best_quality):
        best, best_quality = candidate, candidate_quality

    return _settle(best, best_quality, attempts, raise_on_bad)


def judge_remote_transcript(transcript, duration=None, debug_dir=None):
    """Score a transcript produced somewhere else. Returns it, or None.

    A Kaggle transcript must clear exactly the same bar as a local one — the
    whole point of the gate is that clip selection never sees words nobody
    checked, and "it came from the GPU box" is not a quality argument. So it
    goes through ``evaluate_transcript`` and, when that leaves doubt, the same
    reader.

    None means "not usable, transcribe it here instead", which is why this
    returns rather than raising: the caller still has a working local path and
    a remote BAD is a reason to fall back, not a reason to fail the job.
    """
    if not transcript or not (transcript.get("segments") or []):
        return None

    quality = evaluate_transcript(transcript, duration)
    print(f"[asr] Remote transcript: {_describe(transcript)}")
    _print_language(transcript)
    print(format_quality_line(quality))
    for line in quality_reason_lines(quality):
        print(line)

    quality = _read_the_words(transcript, quality, label="Remote")

    _save_debug(debug_dir, "asr_initial.json", transcript)
    _save_debug(debug_dir, "asr_quality_initial.json", quality)

    if quality.get("status") == "BAD":
        print("[asr] Remote transcript is unusable - falling back to local ASR.")
        return None

    attempts = [_attempt_summary(1, transcript, quality, reason="remote")]
    return _settle(transcript, quality, attempts, raise_on_bad=False)


def _probed_language_pin(model, language, probability):
    """Carry the probe's language onto the full pass, when it was sure enough.

    Same reasoning as _retry_language, one stage earlier: the probe already
    detected the language with the base model, and letting the stronger model
    re-decide on its own is how a Hindi video comes back as fluent English
    TRANSLATION that no quality signal can catch. Only meaningful when the
    probe actually changed the model — otherwise the pass detects for itself
    exactly as before.
    """
    if not model or not language:
        return None
    if probability is None:
        return None
    if probability < _env_float("ASR_PIN_LANGUAGE_ABOVE", 0.5):
        return None
    return language


def _accepts(quality):
    """Whether this transcript is good enough to CUT CLIPS from.

    The old rule was ``status == "GOOD"``, and it was the single biggest thing
    wrong with this module. GOOD is a statement about the transcript being
    CORRECT; clip selection does not need correct, it needs locatable. Gemini
    reading a Hinglish transcript with a few mangled words still knows where
    the punchline is — which is exactly why the upstream tool, with no quality
    gate whatsoever, cut this video fine while we escalated past it.

    Worse, on genuinely code-switched speech GOOD is close to unreachable: the
    reader marks a 25-second window PARTIAL for one oddly-spelled word, and
    ``aggregate()`` needs 80% of sampled seconds GOOD. So every Hinglish video
    walked the whole ladder by construction, whatever the models produced.

    So PARTIAL is now an ACCEPT. A structural RETRY still earns the one
    stronger pass this module is willing to pay for — that is a cheap-model
    problem the probe may have missed, and it is worth one attempt — and BAD
    is still a refusal.
    """
    return quality.get("status") in ("GOOD", "PARTIAL")


def _settle(transcript, quality, attempts, raise_on_bad):
    """Log what the caller is getting, and refuse only genuine garbage.

      GOOD/RETRY  continue.
      PARTIAL     continue, and say so. This is the case worth protecting: one
                  ten-minute video really can be half accurate, and throwing
                  the good half away helps nobody. The regions are recorded on
                  the transcript as DIAGNOSTICS — nothing downstream deletes a
                  clip because of them, because a reader sampling 25-second
                  windows is not precise enough to overrule the critic that
                  actually reads the candidate.
      BAD         refuse. A clear failure beats a nonsense Short.
    """
    status = quality.get("status")

    if status == "BAD":
        print("[asr] Transcript remained unreliable after every attempt - "
              f"score={quality.get('score')}, reasons: "
              f"{'; '.join(quality.get('reasons') or []) or 'no usable speech'}")
        _attach_quality(transcript, quality, attempts)
        if not raise_on_bad:
            return transcript
        raise TranscriptQualityError(quality=quality)

    if status == "PARTIAL":
        regions = (quality.get("semantic") or {}).get("regions") or []
        usable = [r for r in regions if r["status"] == "GOOD"]
        print(f"[asr] Transcript is PARTIAL: {len(usable)} of {len(regions)} "
              f"examined stretches read cleanly. Using it anyway - clip "
              f"selection only needs to find the words, not spell them.")
    elif status != "GOOD":
        print(f"[asr] Continuing with a suspicious transcript "
              f"(score={quality.get('score')}).")

    return _attach_quality(transcript, quality, attempts)
