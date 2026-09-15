import argparse
import json
import os

from meaningful_selector import (
    build_sentence_units
)

from meaningful_critic import (
    review_candidates_with_gemini
)


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def save_json(path, data):
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
            "Run meaningful clip "
            "blind verifier + repairer."
        )
    )

    parser.add_argument(
        "metadata",
        help="OpenShorts metadata JSON"
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Optional Gemini critic model"
    )

    args = parser.parse_args()

    metadata_path = (
        os.path.abspath(
            args.metadata
        )
    )

    metadata = load_json(
        metadata_path
    )

    transcript = metadata[
        "transcript"
    ]

    sentences = (
        build_sentence_units(
            transcript
        )
    )

    debug_dir = os.path.join(
        os.path.dirname(
            metadata_path
        ),
        "meaningful_debug"
    )

    candidate_path = (
        os.path.join(
            debug_dir,
            "candidate_clips.json"
        )
    )

    if not os.path.exists(
        candidate_path
    ):
        raise RuntimeError(
            "Candidate file not found: "
            + candidate_path
        )

    candidate_data = load_json(
        candidate_path
    )

    candidates = (
        candidate_data.get(
            "candidates",
            []
        )
    )

    reviews = (
        review_candidates_with_gemini(
            sentences=sentences,
            candidates=candidates,
            model_name=args.model,
        )
    )

    critic_path = os.path.join(
        debug_dir,
        "critic_results_v2.json"
    )

    final_path = os.path.join(
        debug_dir,
        "final_candidates_v2.json"
    )

    text_path = os.path.join(
        debug_dir,
        "critic_results_v2.txt"
    )

    save_json(
        critic_path,
        {
            "review_count": len(
                reviews
            ),
            "reviews": reviews,
        }
    )

    accepted = [
        review
        for review in reviews
        if review["decision"] in {
            "ACCEPT",
            "REPAIR",
        }
    ]

    rejected = [
        review
        for review in reviews
        if review["decision"]
        == "REJECT"
    ]

    save_json(
        final_path,
        {
            "final_candidate_count": (
                len(accepted)
            ),
            "candidates": accepted,
        }
    )

    with open(
        text_path,
        "w",
        encoding="utf-8"
    ) as f:

        for review in reviews:

            initial = review[
                "initial_verification"
            ]

            final = review[
                "final_verification"
            ]

            f.write(
                f'\n'
                f'{review["candidate_id"]}\n'
                f'{"=" * 72}\n'
                f'Decision: '
                f'{review["decision"]}\n'
                f'Original: '
                f'{review["original_start_sentence"]}'
                f' -> '
                f'{review["original_end_sentence"]}\n'
                f'Final: '
                f'{review["final_start_sentence"]}'
                f' -> '
                f'{review["final_end_sentence"]}\n'
                f'Duration: '
                f'{review["duration"]:.2f}s\n'
                f'\n'
                f'INITIAL BLIND TEST\n'
                f'Verdict: '
                f'{initial.get("verdict")}\n'
                f'Standalone: '
                f'{initial.get("standalone_score")}\n'
                f'Completeness: '
                f'{initial.get("completeness_score")}\n'
                f'Opening self-contained: '
                f'{initial.get("opening_is_self_contained")}\n'
                f'Ending complete: '
                f'{initial.get("ending_is_complete")}\n'
                f'Unresolved: '
                f'{initial.get("unresolved_references")}\n'
                f'Reason: '
                f'{initial.get("reason")}\n'
                f'\n'
                f'REPAIR REASON\n'
                f'{review.get("repair_reason")}\n'
                f'\n'
                f'FINAL BLIND TEST\n'
                f'Verdict: '
                f'{final.get("verdict")}\n'
                f'Standalone: '
                f'{final.get("standalone_score")}\n'
                f'Completeness: '
                f'{final.get("completeness_score")}\n'
                f'Reason: '
                f'{final.get("reason")}\n'
                f'\n'
                f'FINAL TRANSCRIPT\n'
                f'{review["final_transcript"]}\n'
            )

    print()
    print("=" * 72)
    print(
        "BLIND CRITIC COMPLETE"
    )
    print("=" * 72)

    print(
        f"Candidates reviewed: "
        f"{len(reviews)}"
    )

    print(
        f"Accepted/repaired: "
        f"{len(accepted)}"
    )

    print(
        f"Rejected: "
        f"{len(rejected)}"
    )

    print()
    print(critic_path)
    print(final_path)
    print(text_path)


if __name__ == "__main__":
    main()