"""Non-rendering validation of the adaptive-ASR + meaningful-selection chain.

Runs against a job directory that already has its transcripts on disk, so no
audio is re-transcribed and no clip is rendered:

    python tools/validate_asr_semantic.py output/<job-id>

Stages exercised, all with the real models:
    saved transcript -> semantic validation (representative + dense)
                     -> transcript status GOOD/PARTIAL/BAD
                     -> sentence units
                     -> Candidate Finder (incl. over-long repair)
                     -> unreliable-region filtering
                     -> Blind Context Critic
                     -> Opening Independence Guard

Deliberately stops before metadata and rendering: the point is to find out
whether any candidate survives semantic verification at all.
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env"))
except Exception:
    pass

import asr_semantic
from asr_quality import evaluate_transcript
from meaningful_selector import build_sentence_units, find_candidates_with_gemini
from meaningful_critic import review_candidates_with_gemini
from meaningful_opening_guard import enforce_opening_independence
from meaningful_pipeline import _flag_unreliable_candidates


def _load(path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def _dump(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _rule(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir")
    parser.add_argument("--transcript", default="asr_retry.json",
                        help="which saved transcript to validate")
    parser.add_argument("--stop-after", default="guard",
                        choices=["semantic", "sentences", "candidates",
                                 "critic", "guard"])
    args = parser.parse_args()

    debug = os.path.join(args.job_dir, "meaningful_debug")
    transcript = _load(os.path.join(debug, args.transcript))

    _rule(f"TRANSCRIPT: {args.transcript}")
    segments = transcript.get("segments") or []
    print(f"language={transcript.get('language')} "
          f"probability={transcript.get('language_probability')}")
    print(f"segments={len(segments)} "
          f"runtime={segments[-1]['end']:.1f}s" if segments else "empty")

    structural = evaluate_transcript(transcript, duration=segments[-1]["end"])
    print(f"structural: {structural['status']} score={structural['score']} "
          f"uncertain={structural.get('uncertain')}")
    for reason in structural.get("reasons") or []:
        print(f"   - {reason}")

    # --- semantic ----------------------------------------------------------
    _rule("SEMANTIC VALIDATION (representative sample)")
    sparse = asr_semantic.review_transcript(transcript)
    if sparse is None:
        print("unavailable (no key / disabled) — cannot validate")
        return 2
    print(asr_semantic.format_summary(sparse))
    for region in sparse["regions"]:
        print(f"  {region['id']} [{region['start']:7.1f}-{region['end']:7.1f}s] "
              f"{region['status']:7s} {region['score']:3d}  {region['reason'][:90]}")
    _dump(os.path.join(debug, "asr_semantic_sample.json"), sparse)

    semantic = sparse
    if sparse["status"] == "PARTIAL":
        _rule("SEMANTIC VALIDATION (dense: where exactly)")
        dense = asr_semantic.review_transcript(transcript, dense=True)
        if dense:
            print(asr_semantic.format_summary(dense))
            for region in dense["regions"]:
                print(f"  {region['id']} "
                      f"[{region['start']:7.1f}-{region['end']:7.1f}s] "
                      f"{region['status']:7s} {region['score']:3d}  "
                      f"{region['reason'][:80]}")
            semantic = dense
            _dump(os.path.join(debug, "asr_semantic_regions.json"), dense)

    unreliable = asr_semantic.unreliable_ranges(semantic)
    eligible = [r for r in semantic["regions"] if r["status"] == "GOOD"]
    print()
    print(f"FINAL TRANSCRIPT STATUS: {semantic['status']}")
    print(f"eligible (GOOD) regions: {len(eligible)} of {len(semantic['regions'])}")
    print(f"off-limits stretches: {len(unreliable)} "
          f"({sum(b - a for a, b in unreliable):.0f}s)")

    if args.stop_after == "semantic":
        return 0

    # --- sentence units -----------------------------------------------------
    _rule("SENTENCE UNITS")
    sentences = build_sentence_units(transcript)
    print(f"sentence units: {len(sentences)}")
    if args.stop_after == "sentences":
        return 0

    # --- candidates ---------------------------------------------------------
    _rule("CANDIDATE FINDER (with over-long repair)")
    language = str(transcript.get("language") or "unknown")
    candidates = find_candidates_with_gemini(sentences=sentences,
                                             language=language) or []
    repaired = [c for c in candidates if c.get("overlong_repair")]
    print(f"candidates: {len(candidates)}  (repaired from over-long: "
          f"{len(repaired)})")
    for candidate in candidates:
        mark = " [repaired]" if candidate.get("overlong_repair") else ""
        print(f"  {candidate['candidate_id']} "
              f"{candidate['start']:.1f}-{candidate['end']:.1f}s "
              f"({candidate['duration']:.1f}s){mark} "
              f"| {str(candidate.get('topic'))[:60]}")
    _dump(os.path.join(debug, "validate_candidates.json"), candidates)

    _rule("UNRELIABLE-REGION FILTER")
    candidates = _flag_unreliable_candidates(candidates, unreliable)
    print(f"candidates in reliable speech: {len(candidates)}")
    if args.stop_after == "candidates" or not candidates:
        return 0

    # --- critic -------------------------------------------------------------
    _rule("BLIND CONTEXT CRITIC")
    reviews = review_candidates_with_gemini(sentences=sentences,
                                            candidates=candidates) or []
    accepted = [r for r in reviews
                if str(r.get("decision", "")).upper() in ("ACCEPT", "REPAIR")]
    print(f"reviewed {len(reviews)}; accepted {len(accepted)}")
    _dump(os.path.join(debug, "validate_critic.json"), reviews)
    if args.stop_after == "critic":
        return 0

    # --- opening guard ------------------------------------------------------
    _rule("OPENING INDEPENDENCE GUARD")
    guarded = enforce_opening_independence(sentences=sentences,
                                           reviews=reviews) or []
    survivors = [r for r in guarded
                 if str(r.get("decision", "")).upper() in ("ACCEPT", "REPAIR")]
    print(f"survived: {len(survivors)}")
    _dump(os.path.join(debug, "validate_guard.json"), guarded)

    _rule("SURVIVING CANDIDATES — ACTUAL TRANSCRIPT TEXT")
    by_id = {s["id"]: s for s in sentences}
    index = {s["id"]: i for i, s in enumerate(sentences)}
    for item in survivors:
        start_id = item.get("final_start_sentence") or item.get("start_sentence")
        end_id = item.get("final_end_sentence") or item.get("end_sentence")
        if start_id not in index or end_id not in index:
            continue
        span = sentences[index[start_id]:index[end_id] + 1]
        seconds = float(span[-1]["end"]) - float(span[0]["start"])
        print()
        print(f"--- {item.get('candidate_id')} {span[0]['start']:.1f}-"
              f"{span[-1]['end']:.1f}s ({seconds:.1f}s) ---")
        print(" ".join(s["text"] for s in span))
    return 0


if __name__ == "__main__":
    sys.exit(main())
