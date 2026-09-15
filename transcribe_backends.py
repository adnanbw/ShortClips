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

The task is always "transcribe", never "translate": clip timing, captions and
metadata all need the words that were actually spoken, in the spoken language.
"""
import json
import os
import subprocess
import tempfile
import threading
import time

from subtitles import (
    get_whisper_config,
    get_whisper_retry_model,
    WHISPER_TRANSCRIBE_PARAMS,
    whisper_transcribe_params,
    merge_continuation_words,
)
from asr_quality import (
    TranscriptQualityError,
    evaluate_transcript,
    format_quality_line,
    gemini_second_opinion,
    good_score_threshold,
    quality_reason_lines,
    rank_quality,
)

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
# Small bounded cache so the quality retry (a different, stronger model) does
# not evict the model every other job would then have to reload. Two entries
# = the initial model + the retry model.
_whisper_cache = {}
_whisper_cache_order = []
# Set after a CUDA failure (e.g. VRAM exhausted by other models on the GPU)
# so every later transcription goes straight to CPU instead of re-failing.
_whisper_force_cpu = False


def _whisper_cache_max():
    try:
        return max(1, int(os.environ.get("WHISPER_MODEL_CACHE", "2")))
    except ValueError:
        return 2


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
            model = WhisperModel(key[0], device=key[1], compute_type=key[2])
            _whisper_cache[key] = model
            _whisper_cache_order.append(key)
            while len(_whisper_cache_order) > _whisper_cache_max():
                _whisper_cache.pop(_whisper_cache_order.pop(0), None)
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


def _run_whisper_once(media_path, model_size=None, **params):
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
        progress = _TranscribeProgress(getattr(info, "duration", 0))
        materialized = []
        for segment in segments:
            materialized.append(segment)
            progress.update(segment.end)
        # VAD trims trailing silence, so the last segment can end short of the
        # media duration — force the 100% line.
        progress.update(progress.total)
        return materialized, info


def run_whisper_transcription(media_path, model_size=None, **params):
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
        return _run_whisper_once(media_path, model_size=model_size, **params)
    except RuntimeError as e:
        if _whisper_force_cpu or "cuda" not in str(e).lower():
            raise
        print(f"⚠️ [ASR] whisper GPU failed ({e}) — retrying on CPU", flush=True)
        _whisper_force_cpu = True
        _drop_whisper_models()  # release the GPU models' VRAM
        return _run_whisper_once(media_path, model_size=model_size, **params)


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


def transcribe_media(media_path):
    """Transcribe with the configured backend, falling back to whisper."""
    # Silent videos (AI-generated clips, muted screen recordings) have no audio
    # stream; every ASR backend then crashes deep inside libav with an opaque
    # "tuple index out of range". Detect it up front and fail with a clear,
    # actionable reason instead.
    if not _has_audio_stream(media_path):
        raise NoAudioError(
            "This video has no audio track. OpenShorts finds viral moments from "
            "speech, so it needs a video with audio.")

    backend = os.environ.get("TRANSCRIBE_BACKEND", "whisper").strip().lower()

    if backend == "parakeet":
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

    return _transcribe_with_whisper(media_path)


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


def _attach_quality(transcript, quality, attempts):
    asr = transcript.setdefault("asr", {})
    asr["quality"] = quality
    asr["attempts"] = attempts
    return transcript


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


def _read_the_words(transcript, quality):
    """Downgrade a structurally-clean transcript whose words are not words.

    The deterministic signals measure structure, and a weak model on a language
    it cannot handle produces structurally PERFECT nonsense: whisper-small
    transcribed the Hinglish stand-up as fluent Devanagari-shaped gibberish
    with no repetition, valid timestamps and 173 words/min, and scored 87/100.
    When the decoder's own confidence says it was guessing (``uncertain``),
    someone who can actually read the language has to look. Costs one small
    call, and only on a job that is about to be wrong either way.
    """
    if not quality.get("uncertain"):
        return quality

    print("🧪 Transcript looks structurally fine but the decoder was unsure — "
          "checking whether the words are real words…")
    verdict = gemini_second_opinion(transcript)
    if verdict is not False:
        # True (real language) or None (no key / check disabled): accept. An
        # unavailable check must never fail a job on suspicion alone.
        return quality

    print("🧪 The transcript is not real language — treating it as unreliable.")
    # The structural score stays high (structure was never the problem), so it
    # is pulled under the accept line too: a RETRY carrying a passing number
    # reads like a bug in the logs, and the attempt comparison below would
    # otherwise prefer this transcript over a genuinely good retry.
    return dict(quality, status="RETRY",
                score=round(min(float(quality.get("score") or 0.0),
                                good_score_threshold() - 0.1), 1),
                reasons=list(quality.get("reasons") or [])
                + ["the transcribed words are not real words in any language"],
                downgraded_by="language_check")


def transcribe_media_checked(media_path, duration=None, debug_dir=None,
                             raise_on_bad=True):
    """Transcribe, score the transcript, and retry once with a stronger model
    only when the first result looks unreliable.

    Returns the same transcript contract as ``transcribe_media`` with an added
    ``asr.quality`` block. Raises ``TranscriptQualityError`` when both attempts
    are unusable and ``raise_on_bad`` — so the job fails naming the real cause
    (the audio could not be transcribed) instead of blaming clip detection.

    A clean video is transcribed EXACTLY ONCE: the retry is the exception.
    """
    initial = transcribe_media(media_path)
    quality = evaluate_transcript(initial, duration)

    print(f"🎙️ ASR attempt 1: {_describe(initial)}")
    language = initial.get("language")
    probability = initial.get("language_probability")
    if language:
        confidence = f" (confidence {probability:.0%})" if probability else ""
        print(f"🎙️ Detected language: {language}{confidence}")
    print(format_quality_line(quality))
    for line in quality_reason_lines(quality):
        print(line)

    _save_debug(debug_dir, "asr_initial.json", initial)
    _save_debug(debug_dir, "asr_quality_initial.json", quality)

    attempts = [{"attempt": 1, "asr": initial.get("asr"),
                 "status": quality.get("status"), "score": quality.get("score")}]

    if quality["status"] == "GOOD":
        quality = _read_the_words(initial, quality)
        if quality["status"] == "GOOD":
            return _attach_quality(initial, quality, attempts)
        # Record the downgrade, score included: an attempt logged as RETRY next
        # to a passing number reads like a bug in the diagnostics.
        attempts[0].update(status=quality["status"], score=quality["score"])

    retry_model = get_whisper_retry_model()
    current_model = (initial.get("asr") or {}).get("model")
    can_retry = (
        _auto_retry_enabled()
        and retry_model
        and retry_model != current_model
        and not quality.get("skipped")
    )

    if not can_retry:
        why = "disabled" if not _auto_retry_enabled() else (
            "no stronger model configured" if not retry_model
            else f"already transcribed with {retry_model}")
        print(f"⏭️ Not retrying transcription ({why}).")
        return _finish_unreliable(initial, quality, attempts, raise_on_bad)

    pinned = _retry_language(initial, quality)
    print(f"🔁 Retrying transcription with model={retry_model}"
          + (f" language={pinned} (kept from attempt 1)" if pinned
             else " language=auto-detect"))
    try:
        retry = _transcribe_with_whisper(media_path, model_size=retry_model,
                                         language=pinned)
    except Exception as e:
        print(f"⚠️ Retry transcription failed ({type(e).__name__}: {e}) — "
              f"keeping the first transcript.")
        return _finish_unreliable(initial, quality, attempts, raise_on_bad)

    retry_quality = evaluate_transcript(retry, duration)
    if retry_quality["status"] == "GOOD":
        retry_quality = _read_the_words(retry, retry_quality)
    print(f"🎙️ ASR attempt 2: {_describe(retry)}")
    retry_language = retry.get("language")
    if retry_language:
        retry_probability = retry.get("language_probability")
        confidence = f" (confidence {retry_probability:.0%})" if retry_probability else ""
        print(f"🎙️ Detected language: {retry_language}{confidence}")
    print(format_quality_line(retry_quality, prefix="Retry transcript quality"))
    for line in quality_reason_lines(retry_quality):
        print(line)

    _save_debug(debug_dir, "asr_retry.json", retry)
    _save_debug(debug_dir, "asr_quality_retry.json", retry_quality)

    attempts.append({"attempt": 2, "asr": retry.get("asr"),
                     "status": retry_quality.get("status"),
                     "score": retry_quality.get("score")})

    # Keep whichever attempt actually came out better; a retry is not
    # automatically an improvement. Status first, then score (rank_quality):
    # the language check can downgrade a transcript whose structural score is
    # still high, and that transcript must lose to a merely-decent real one.
    if rank_quality(retry_quality) >= rank_quality(quality):
        chosen, chosen_quality, label = retry, retry_quality, "retry"
    else:
        chosen, chosen_quality, label = initial, quality, "first"

    if chosen_quality["status"] in ("GOOD", "RETRY"):
        print(f"✅ Using {label} transcript "
              f"({chosen_quality['status']}, score={chosen_quality['score']})")
        return _attach_quality(chosen, chosen_quality, attempts)

    return _finish_unreliable(chosen, chosen_quality, attempts, raise_on_bad)


def _finish_unreliable(transcript, quality, attempts, raise_on_bad):
    """Accept a merely-suspicious transcript; refuse a hopeless one."""
    if quality["status"] != "BAD":
        # RETRY means "worth a stronger model", not "unusable". Nothing
        # stronger is left, so continue with it rather than failing a job a
        # human would consider fine.
        print(f"➡️ Continuing with a suspicious transcript "
              f"(score={quality['score']}); clip quality may suffer.")
        return _attach_quality(transcript, quality, attempts)

    # Last reprieve, opt-in and only for a transcript already condemned:
    # deterministic signals can misjudge a heavily code-switched transcript.
    if gemini_second_opinion(transcript) is True:
        print("➡️ Transcript sanity check says the text is coherent speech — "
              "continuing despite the low score.")
        quality = dict(quality, status="RETRY", overridden_by="gemini")
        return _attach_quality(transcript, quality, attempts)

    print("❌ Transcript remained unreliable after retry — "
          f"score={quality['score']}, reasons: "
          f"{'; '.join(quality.get('reasons') or []) or 'no usable speech'}")

    if not raise_on_bad:
        return _attach_quality(transcript, quality, attempts)

    _attach_quality(transcript, quality, attempts)
    raise TranscriptQualityError(quality=quality)
