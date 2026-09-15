"""
Meaningful Clip Selection Pipeline

Flow:
1. Build semantic sentence units.
2. Candidate Finder proposes sentence-ID clip ranges.
3. Blind Context Critic verifies standalone meaning/completeness and may repair.
4. Opening Independence Guard checks only the start boundary and, when needed,
   moves the start backward to the minimum earlier sentence that supplies the
   missing context.
5. Grounded Metadata creates hook/title/descriptions from final approved clips.
6. Return OpenShorts-compatible output.

Rules:
- No minimum clip count.
- Hard duration range: 15-90 seconds.
- Exact sentence timestamps; no arbitrary padding.
- No silent fallback to the legacy selector.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from meaningful_selector import build_sentence_units, find_candidates_with_gemini
from meaningful_critic import review_candidates_with_gemini
from meaningful_opening_guard import enforce_opening_independence
from meaningful_metadata import calculate_final_rank_score, generate_metadata_with_gemini


SELECTOR_NAME = "meaningful_v1"


def _save_json(output_dir: Optional[str], filename: str, data: Any) -> None:
    """Best-effort debug persistence; debug-file errors must not kill a job."""
    if not output_dir:
        return

    try:
        debug_dir = Path(output_dir) / "meaningful_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        print(
            f"⚠️ Could not save meaningful debug file {filename}: "
            f"{type(exc).__name__}: {exc}"
        )


def _retry_stage(stage_name: str, fn, *, max_attempts: int = 3):
    """Retry a whole stage only for transient API/network failures."""
    delays = [10, 20, 40]

    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            text = str(exc).lower()
            transient_markers = (
                "429",
                "500",
                "502",
                "503",
                "504",
                "resource_exhausted",
                "unavailable",
                "temporarily unavailable",
                "timeout",
                "timed out",
                "rate limit",
                "rate_limit",
                "connection reset",
                "connection aborted",
            )
            transient = any(marker in text for marker in transient_markers)

            if not transient or attempt >= max_attempts - 1:
                raise

            delay = delays[min(attempt, len(delays) - 1)]
            print(
                f"⚠️ {stage_name} temporary failure "
                f"({type(exc).__name__}: {exc})"
            )
            print(
                f"   Retrying whole stage in {delay}s "
                f"({attempt + 2}/{max_attempts})..."
            )
            time.sleep(delay)

    raise RuntimeError(f"{stage_name} retry loop exhausted")


def _meaningful_max_clips() -> int:
    """Final clip CAP only. 0 means unlimited; this is never a target/minimum."""
    raw = os.environ.get("MEANINGFUL_MAX_CLIPS", "10").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 10


def _meaningful_min_rank() -> float:
    raw = os.environ.get("MEANINGFUL_MIN_RANK", "0").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _rank_approved(reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep ACCEPT/REPAIR results, compute semantic rank, sort, then cap."""
    approved: List[Dict[str, Any]] = []
    min_rank = _meaningful_min_rank()

    for review in reviews:
        if not isinstance(review, dict):
            continue

        decision = str(review.get("decision") or "").upper()
        if decision not in {"ACCEPT", "REPAIR"}:
            continue

        item = dict(review)

        try:
            score = float(calculate_final_rank_score(item))
        except Exception as exc:
            print(
                f"⚠️ Could not calculate semantic rank for "
                f"{item.get('candidate_id', 'candidate')}: "
                f"{type(exc).__name__}: {exc}"
            )
            score = 0.0

        item["semantic_rank_score"] = round(score, 3)

        if score < min_rank:
            print(
                f"Filtered {item.get('candidate_id', 'candidate')}: "
                f"rank {score:.2f} < minimum {min_rank:.2f}"
            )
            continue

        approved.append(item)

    approved.sort(
        key=lambda x: float(x.get("semantic_rank_score") or 0.0),
        reverse=True,
    )

    max_clips = _meaningful_max_clips()
    if max_clips > 0:
        approved = approved[:max_clips]

    return approved


def _validate_short_times(
    shorts: List[Dict[str, Any]],
    video_duration: float,
) -> List[Dict[str, Any]]:
    """Clamp timestamps and enforce the hard 15-90 second duration rule."""
    valid: List[Dict[str, Any]] = []

    try:
        duration_limit = max(0.0, float(video_duration or 0.0))
    except (TypeError, ValueError):
        duration_limit = 0.0

    for index, raw_short in enumerate(shorts, start=1):
        if not isinstance(raw_short, dict):
            print(f"Rejected metadata item {index}: expected object")
            continue

        short = dict(raw_short)

        try:
            start = float(short.get("start"))
            end = float(short.get("end"))
        except (TypeError, ValueError):
            print(f"Rejected metadata item {index}: start/end are not numeric")
            continue

        start = max(0.0, start)
        if duration_limit > 0:
            end = min(end, duration_limit)

        clip_duration = end - start

        if clip_duration < 15.0:
            print(f"Rejected final clip {index}: {clip_duration:.1f}s is too short")
            continue
        if clip_duration > 90.0:
            print(f"Rejected final clip {index}: {clip_duration:.1f}s exceeds 90s")
            continue

        short["start"] = round(start, 3)
        short["end"] = round(end, 3)
        short["selection_mode"] = SELECTOR_NAME

        if short.get("semantic_rank_score") is None and short.get("final_rank_score") is not None:
            short["semantic_rank_score"] = short.get("final_rank_score")

        valid.append(short)

    valid.sort(
        key=lambda x: float(
            x.get("semantic_rank_score")
            or x.get("final_rank_score")
            or x.get("predicted_score")
            or 0.0
        ),
        reverse=True,
    )

    return valid


def get_meaningful_clips(
    transcript_result: Dict[str, Any],
    video_duration: float,
    output_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    selector_model: Optional[str] = None,
    critic_model: Optional[str] = None,
    metadata_model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run the complete meaningful clip-selection pipeline."""
    transcript_result = transcript_result or {}
    segments = transcript_result.get("segments") or []

    if not segments:
        print("⚠️ Meaningful selector: transcript has no usable segments.")
        return None

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for the meaningful selector.")

    # Detected by the ASR stage; "unknown" is honest for a transcript whose
    # language the detector was not sure about (heavily code-switched speech
    # often is), and the prompts tell the model to read it as it finds it.
    language = str(transcript_result.get("language") or "unknown")

    print("=" * 72)
    print("MEANINGFUL CLIP SELECTOR V1")
    print("=" * 72)
    print(f"Transcript language: {language}")

    # --------------------------------------------------------------
    # Sentence normalization
    # --------------------------------------------------------------
    sentences = build_sentence_units(transcript_result)
    print(f"Sentence units: {len(sentences)}")

    _save_json(output_dir, "pipeline_sentences.json", sentences)

    if not sentences:
        print("⚠️ No semantic sentence units could be built.")
        return None

    # --------------------------------------------------------------
    # 1/4 Candidate Finder
    # --------------------------------------------------------------
    print("[1/4] Candidate Finder")

    candidates = _retry_stage(
        "Candidate Finder",
        lambda: find_candidates_with_gemini(
            sentences=sentences,
            language=language,
            api_key=api_key,
            model_name=selector_model,
        ),
    ) or []

    _save_json(output_dir, "pipeline_candidates.json", candidates)
    print(f"Candidate Finder kept {len(candidates)} candidate(s).")

    if not candidates:
        print("⚠️ Candidate Finder returned no usable semantic candidates.")
        return None

    # --------------------------------------------------------------
    # 2/4 Blind Context Critic
    # --------------------------------------------------------------
    print("[2/4] Blind Context Critic")

    critic_results = _retry_stage(
        "Blind Context Critic",
        lambda: review_candidates_with_gemini(
            sentences=sentences,
            candidates=candidates,
            api_key=api_key,
            model_name=critic_model,
        ),
    ) or []

    # Save pre-opening-guard results separately so we can compare what the
    # broader critic accepted with what the focused start-boundary judge caught.
    _save_json(
        output_dir,
        "pipeline_critic_results_pre_opening_guard.json",
        critic_results,
    )

    # --------------------------------------------------------------
    # 3/4 Opening Independence Guard
    # --------------------------------------------------------------
    print("[3/4] Opening Independence Guard")

    guarded_results = _retry_stage(
        "Opening Independence Guard",
        lambda: enforce_opening_independence(
            sentences=sentences,
            reviews=critic_results,
            api_key=api_key,
            model_name=critic_model,
        ),
    ) or []

    _save_json(output_dir, "pipeline_opening_guard.json", guarded_results)

    # Keep the old filename as the FINAL post-opening-guard critic state.
    _save_json(output_dir, "pipeline_critic_results.json", guarded_results)

    approved = _rank_approved(guarded_results)
    _save_json(output_dir, "pipeline_final_candidates.json", approved)

    print(
        f"Critic/opening guard reviewed {len(candidates)} candidate(s); "
        f"{len(approved)} survived final quality ranking."
    )

    if not approved:
        print("⚠️ No candidate passed standalone/completeness/opening verification.")
        return None

    # --------------------------------------------------------------
    # 4/4 Grounded Metadata
    # --------------------------------------------------------------
    print("[4/4] Grounded Metadata")

    shorts = _retry_stage(
        "Grounded Metadata",
        lambda: generate_metadata_with_gemini(
            candidates=approved,
            language=language,
            api_key=api_key,
            model_name=metadata_model,
        ),
    ) or []

    # Preserve the rank by position if the metadata module did not copy it.
    for idx, short in enumerate(shorts):
        if not isinstance(short, dict):
            continue
        if short.get("semantic_rank_score") is None and idx < len(approved):
            short["semantic_rank_score"] = approved[idx].get(
                "semantic_rank_score",
                0.0,
            )

    shorts = _validate_short_times(shorts, video_duration=video_duration)

    if not shorts:
        print("⚠️ Grounded Metadata produced no valid 15-90s final clips.")
        return None

    result = {
        "selector": SELECTOR_NAME,
        "shorts": shorts,
    }

    _save_json(output_dir, "pipeline_meaningful_shorts.json", result)

    print("=" * 72)
    print(f"MEANINGFUL SELECTOR COMPLETE: {len(shorts)} clip(s)")
    print("=" * 72)

    for index, short in enumerate(shorts, start=1):
        start = short.get("start", 0)
        end = short.get("end", 0)
        title = (
            short.get("video_title_for_youtube_short")
            or short.get("viral_hook_text")
            or short.get("title")
            or ""
        )
        rank = (
            short.get("semantic_rank_score")
            or short.get("final_rank_score")
            or short.get("predicted_score")
            or 0
        )

        try:
            clip_duration = float(end) - float(start)
        except (TypeError, ValueError):
            clip_duration = 0.0

        print(
            f"  {index}. {float(start):.2f}s -> {float(end):.2f}s "
            f"({clip_duration:.1f}s) | rank={rank} | {title}"
        )

    return result
