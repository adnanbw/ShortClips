"""Is a third, heavier ASR pass actually worth it?

A third pass must be exceptional, never normal, so it may only be added if a
real comparison on real audio shows a MATERIAL improvement. This transcribes a
few short slices of an existing source file with two or more models and has the
same semantic reader that judges the pipeline judge each result.

    python tools/compare_asr_models.py output/<job>/<video>.mp4 \
        --slice 124.1-160.1:bad --slice 354.2-395.6:partial \
        --slice 160.8-203.8:good --language hi \
        --models large-v3-turbo,large-v3

Slices are decoded straight out of the source with faster-whisper's own
``clip_timestamps``, so nothing is cut to disk and no ffmpeg is needed.
"""
import argparse
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env"))
except Exception:
    pass

import asr_semantic


def _parse_slice(raw):
    span, _, label = raw.partition(":")
    start, _, end = span.partition("-")
    return {"start": float(start), "end": float(end), "label": label or "?"}


def _transcribe(media, model_size, language, slices, compute="int8"):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type=compute)
    out = []
    for piece in slices:
        started = time.time()
        segments, info = model.transcribe(
            media,
            language=language,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=True,
            clip_timestamps=f"{piece['start']},{piece['end']}",
        )
        materialized = list(segments)
        text = " ".join(s.text.strip() for s in materialized if s.text.strip())
        logprobs = [s.avg_logprob for s in materialized
                    if s.avg_logprob is not None]
        out.append({
            "label": piece["label"],
            "start": piece["start"],
            "end": piece["end"],
            "text": text,
            "segments": [{"start": float(s.start), "end": float(s.end),
                          "text": s.text} for s in materialized],
            "avg_logprob": round(sum(logprobs) / len(logprobs), 3) if logprobs else None,
            "seconds": round(time.time() - started, 1),
        })
    del model
    return out


def _judge(results, model_size):
    """Have the pipeline's own reader grade each slice, in one call."""
    samples = [{"id": f"R{i + 1:02d}", "start": r["start"], "end": r["end"],
                "text": r["text"]}
               for i, r in enumerate(results) if r["text"].strip()]
    if not samples:
        return None
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        verdicts = asr_semantic._call_gemini(samples, api_key, None)
    except Exception as exc:
        print(f"  (reader unavailable: {type(exc).__name__}: {exc})")
        return None
    if not verdicts:
        return None
    return asr_semantic.aggregate(samples, verdicts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("media")
    parser.add_argument("--slice", action="append", required=True,
                        dest="slices", metavar="START-END[:label]")
    parser.add_argument("--models", default="large-v3-turbo,large-v3")
    parser.add_argument("--language", default=None)
    parser.add_argument("--compute", default="int8")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    slices = [_parse_slice(s) for s in args.slices]
    report = {"media": args.media, "language": args.language, "models": {}}

    for model_size in [m.strip() for m in args.models.split(",") if m.strip()]:
        print()
        print("=" * 72)
        print(f"MODEL: {model_size}")
        print("=" * 72)
        results = _transcribe(args.media, model_size, args.language, slices,
                              args.compute)
        verdict = _judge(results, model_size)

        for i, result in enumerate(results):
            region = (verdict or {}).get("regions") or []
            grade = region[i] if i < len(region) else {}
            print(f"\n--- {result['label']} "
                  f"[{result['start']:.1f}-{result['end']:.1f}s] "
                  f"avg_logprob={result['avg_logprob']} "
                  f"{result['seconds']}s "
                  f"| reader: {grade.get('status', '?')} "
                  f"{grade.get('score', '')}")
            print(result["text"][:400])
            if grade.get("reason"):
                print(f"    reason: {grade['reason'][:160]}")

        if verdict:
            print(f"\n  OVERALL: {verdict['status']} score={verdict['score']} "
                  f"shares={verdict.get('shares')}")
        report["models"][model_size] = {"results": results, "verdict": verdict}

    if args.out:
        with io.open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.out}")

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for model_size, data in report["models"].items():
        verdict = data["verdict"] or {}
        per_slice = ", ".join(
            f"{r['label']}={((verdict.get('regions') or [{}] * 99)[i] or {}).get('score', '?')}"
            for i, r in enumerate(data["results"]))
        print(f"{model_size:18s} overall={verdict.get('status', '?'):8s} "
              f"score={verdict.get('score', '?'):>6} | {per_slice}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
