import argparse
import json
import os

from meaningful_metadata import (
    calculate_final_rank_score,
    generate_metadata_with_gemini,
)


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_json(
    path,
    data
):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate metadata for final "
            "meaningful clip candidates."
        )
    )

    parser.add_argument(
        "metadata",
        help=(
            "Original OpenShorts "
            "*_metadata.json"
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help=(
            "Number of top clips to prepare. "
            "Default: 5"
        )
    )

    parser.add_argument(
        "--ids",
        default=None,
        help=(
            "Optional comma-separated candidate IDs. "
            "Example: C002,C005,C007"
        )
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional Gemini metadata model."
        )
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------------

    metadata_path = os.path.abspath(
        args.metadata
    )

    if not os.path.exists(
        metadata_path
    ):

        raise FileNotFoundError(
            metadata_path
        )

    source_metadata = load_json(
        metadata_path
    )

    language = str(
        (
            source_metadata.get(
                "transcript"
            )
            or {}
        ).get(
            "language",
            "unknown"
        )
    )

    debug_dir = os.path.join(
        os.path.dirname(
            metadata_path
        ),
        "meaningful_debug"
    )

    final_candidates_path = (
        os.path.join(
            debug_dir,
            "final_candidates_v2.json"
        )
    )

    if not os.path.exists(
        final_candidates_path
    ):

        raise RuntimeError(
            "Missing final candidate file: "
            + final_candidates_path
        )

    final_data = load_json(
        final_candidates_path
    )

    candidates = (
        final_data.get(
            "candidates"
        )
        or []
    )

    if not candidates:

        raise RuntimeError(
            "No approved candidates found."
        )

    # ------------------------------------------------------------------------
    # Assign rank scores
    # ------------------------------------------------------------------------

    for candidate in candidates:

        candidate[
            "_rank_score"
        ] = (
            calculate_final_rank_score(
                candidate
            )
        )

    candidates = sorted(
        candidates,
        key=lambda c: c[
            "_rank_score"
        ],
        reverse=True
    )

    # ------------------------------------------------------------------------
    # Optional explicit candidate IDs
    # ------------------------------------------------------------------------

    if args.ids:

        requested_ids = [
            item.strip().upper()
            for item
            in args.ids.split(",")
            if item.strip()
        ]

        candidate_map = {
            str(
                candidate.get(
                    "candidate_id",
                    ""
                )
            ).upper(): candidate
            for candidate
            in candidates
        }

        selected = []

        for candidate_id in requested_ids:

            candidate = (
                candidate_map.get(
                    candidate_id
                )
            )

            if not candidate:

                raise RuntimeError(
                    f"Candidate not found: "
                    f"{candidate_id}"
                )

            selected.append(
                candidate
            )

        candidates = selected

    else:

        limit = max(
            1,
            int(
                args.limit
            )
        )

        candidates = (
            candidates[:limit]
        )

    # Remove our temporary sorting property.
    for candidate in candidates:

        candidate.pop(
            "_rank_score",
            None
        )

    # ------------------------------------------------------------------------
    # Print selection BEFORE spending Gemini calls
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "CLIPS SELECTED FOR METADATA"
    )
    print("=" * 72)

    for candidate in candidates:

        print()

        print(
            f'{candidate["candidate_id"]} | '
            f'{candidate["final_start_sentence"]}'
            f' -> '
            f'{candidate["final_end_sentence"]}'
        )

        print(
            f'Duration: '
            f'{candidate["duration"]:.1f}s'
        )

        print(
            f'Semantic rank: '
            f'{calculate_final_rank_score(candidate)}'
        )

        print(
            f'Transcript: '
            f'{candidate["final_transcript"]}'
        )

    # ------------------------------------------------------------------------
    # Generate metadata
    # ------------------------------------------------------------------------

    shorts = (
        generate_metadata_with_gemini(
            candidates=candidates,
            language=language,
            model_name=args.model,
        )
    )

    # Keep ranked order.
    shorts = sorted(
        shorts,
        key=lambda short: short.get(
            "semantic_rank_score",
            0
        ),
        reverse=True
    )

    # ------------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------------

    json_path = os.path.join(
        debug_dir,
        "meaningful_shorts.json"
    )

    txt_path = os.path.join(
        debug_dir,
        "meaningful_shorts.txt"
    )

    save_json(
        json_path,
        {
            "selector": "meaningful_v1",
            "clip_count": len(
                shorts
            ),
            "shorts": shorts,
        }
    )

    # ------------------------------------------------------------------------
    # Save readable report
    # ------------------------------------------------------------------------

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        for index, short in enumerate(
            shorts,
            start=1
        ):

            f.write(
                f'\n'
                f'CLIP {index} - '
                f'{short["candidate_id"]}\n'
                f'{"=" * 72}\n'
                f'Time: '
                f'{short["start"]:.2f}'
                f' -> '
                f'{short["end"]:.2f}'
                f' '
                f'({short["duration"]:.2f}s)\n'
                f'Sentence range: '
                f'{short["final_start_sentence"]}'
                f' -> '
                f'{short["final_end_sentence"]}\n'
                f'Semantic score: '
                f'{short["semantic_rank_score"]}\n'
                f'\n'
                f'HOOK\n'
                f'{short["viral_hook_text"]}\n'
                f'\n'
                f'YOUTUBE TITLE\n'
                f'{short["video_title_for_youtube_short"]}\n'
                f'\n'
                f'TIKTOK\n'
                f'{short["video_description_for_tiktok"]}\n'
                f'\n'
                f'INSTAGRAM\n'
                f'{short["video_description_for_instagram"]}\n'
                f'\n'
                f'FINAL TRANSCRIPT\n'
                f'{short["final_transcript"]}\n'
            )

    # ------------------------------------------------------------------------
    # Console result
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "MEANINGFUL SHORTS READY"
    )
    print("=" * 72)

    print(
        f"Prepared clips: "
        f"{len(shorts)}"
    )

    for index, short in enumerate(
        shorts,
        start=1
    ):

        print()

        print(
            f'{index}. '
            f'{short["candidate_id"]}'
        )

        print(
            f'   {short["start"]:.2f}'
            f' -> '
            f'{short["end"]:.2f}'
            f' '
            f'({short["duration"]:.1f}s)'
        )

        print(
            f'   Hook: '
            f'{short["viral_hook_text"]}'
        )

        print(
            f'   Title: '
            f'{short["video_title_for_youtube_short"]}'
        )

    print()
    print(
        json_path
    )

    print(
        txt_path
    )


if __name__ == "__main__":
    main()