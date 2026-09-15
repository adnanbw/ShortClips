import argparse
import json
import os
from statistics import mean

from meaningful_selector import (
    build_sentence_units,
    sentence_at_time,
    format_sentences_for_ai,
    find_candidates_with_gemini,
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
            "Test the meaningful clip selector."
        )
    )

    parser.add_argument(
        "metadata",
        help="OpenShorts *_metadata.json file"
    )

    parser.add_argument(
        "--find-candidates",
        action="store_true",
        help=(
            "Run Gemini Candidate Finder "
            "after building sentences."
        )
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Gemini model override. "
            "Defaults to GEMINI_MODEL or "
            "gemini-3.1-flash-lite."
        )
    )

    args = parser.parse_args()

    metadata_path = os.path.abspath(
        args.metadata
    )

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            metadata_path
        )

    metadata = load_json(
        metadata_path
    )

    transcript = metadata.get(
        "transcript"
    )

    if not transcript:
        raise RuntimeError(
            "Metadata JSON does not contain transcript."
        )

    sentences = build_sentence_units(
        transcript
    )

    if not sentences:
        raise RuntimeError(
            "No sentence units were generated."
        )

    output_dir = os.path.join(
        os.path.dirname(metadata_path),
        "meaningful_debug"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    sentences_json_path = os.path.join(
        output_dir,
        "sentences.json"
    )

    sentences_txt_path = os.path.join(
        output_dir,
        "sentences.txt"
    )

    save_json(
        sentences_json_path,
        {
            "language": transcript.get(
                "language"
            ),
            "source_segments": len(
                transcript.get(
                    "segments",
                    []
                )
            ),
            "sentence_count": len(
                sentences
            ),
            "sentences": sentences,
        }
    )

    with open(
        sentences_txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            format_sentences_for_ai(
                sentences
            )
        )

    durations = [
        float(sentence["duration"])
        for sentence in sentences
    ]

    print()
    print("=" * 72)
    print(
        "MEANINGFUL SELECTOR - SENTENCE TEST"
    )
    print("=" * 72)

    print(
        f"Whisper segments : "
        f"{len(transcript.get('segments', []))}"
    )

    print(
        f"Sentence units   : "
        f"{len(sentences)}"
    )

    print(
        f"Average duration : "
        f"{mean(durations):.2f}s"
    )

    print(
        f"Longest unit     : "
        f"{max(durations):.2f}s"
    )

    print()
    print(
        "FIRST 20 SENTENCE UNITS"
    )
    print("-" * 72)

    for sentence in sentences[:20]:

        print(
            f'{sentence["id"]} '
            f'[{sentence["start"]:7.2f} - '
            f'{sentence["end"]:7.2f}] '
            f'{sentence["text"]}'
        )

    # ---------------------------------------------------------
    # Legacy OpenShorts comparison
    # ---------------------------------------------------------

    shorts = metadata.get(
        "shorts"
    ) or []

    if shorts:

        print()
        print("=" * 72)
        print(
            "LEGACY OPENSHORTS BOUNDARIES"
        )
        print("=" * 72)

        report_lines = []

        for index, clip in enumerate(
            shorts,
            start=1
        ):

            try:
                start = float(
                    clip.get("start")
                )

                end = float(
                    clip.get("end")
                )

            except (
                TypeError,
                ValueError
            ):
                continue

            start_sentence = sentence_at_time(
                sentences,
                start
            )

            end_sentence = sentence_at_time(
                sentences,
                end
            )

            print()
            print(
                f"CLIP {index}"
            )

            print(
                f"Legacy timestamps: "
                f"{start:.3f} -> "
                f"{end:.3f} "
                f"({end-start:.2f}s)"
            )

            if start_sentence:

                print(
                    f'START intersects '
                    f'{start_sentence["id"]}: '
                    f'{start_sentence["text"]}'
                )

            if end_sentence:

                print(
                    f'END intersects   '
                    f'{end_sentence["id"]}: '
                    f'{end_sentence["text"]}'
                )

            report_lines.append(
                f"CLIP {index}\n"
                f"Legacy: "
                f"{start:.3f} -> "
                f"{end:.3f}\n"
                f"Start sentence: "
                f"{start_sentence}\n"
                f"End sentence: "
                f"{end_sentence}\n\n"
            )

        report_path = os.path.join(
            output_dir,
            "legacy_boundary_report.txt"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.writelines(
                report_lines
            )

    # ---------------------------------------------------------
    # AI PASS #1
    # ---------------------------------------------------------

    if args.find_candidates:

        print()
        print("=" * 72)
        print(
            "AI PASS #1 - CANDIDATE FINDER"
        )
        print("=" * 72)

        candidates = (
            find_candidates_with_gemini(
                sentences=sentences,
                language=transcript.get(
                    "language",
                    "unknown"
                ),
                model_name=args.model,
            )
        )

        candidate_json_path = os.path.join(
            output_dir,
            "candidate_clips.json"
        )

        candidate_txt_path = os.path.join(
            output_dir,
            "candidate_clips.txt"
        )

        save_json(
            candidate_json_path,
            {
                "candidate_count": len(
                    candidates
                ),
                "candidates": candidates,
            }
        )

        with open(
            candidate_txt_path,
            "w",
            encoding="utf-8"
        ) as f:

            for candidate in candidates:

                block = (
                    f'\n'
                    f'{candidate["candidate_id"]}\n'
                    f'{"=" * 70}\n'
                    f'Sentences: '
                    f'{candidate["start_sentence"]}'
                    f' -> '
                    f'{candidate["end_sentence"]}\n'
                    f'Time: '
                    f'{candidate["start"]:.2f}'
                    f' -> '
                    f'{candidate["end"]:.2f}'
                    f' '
                    f'({candidate["duration"]:.2f}s)\n'
                    f'Topic: '
                    f'{candidate["topic"]}\n'
                    f'Combined score: '
                    f'{candidate["combined_score"]}\n'
                    f'Standalone: '
                    f'{candidate["standalone_score"]}\n'
                    f'Completeness: '
                    f'{candidate["completeness_score"]}\n'
                    f'Content: '
                    f'{candidate["content_score"]}\n'
                    f'Hook: '
                    f'{candidate["hook_score"]}\n'
                    f'Reason: '
                    f'{candidate["reason"]}\n\n'
                    f'{candidate["transcript"]}\n'
                )

                f.write(block)

        print()
        print(
            "=" * 72
        )

        print(
            f"CANDIDATES FOUND: "
            f"{len(candidates)}"
        )

        print(
            "=" * 72
        )

        for candidate in candidates:

            print()

            print(
                f'{candidate["candidate_id"]} | '
                f'{candidate["start_sentence"]}'
                f' -> '
                f'{candidate["end_sentence"]}'
            )

            print(
                f'{candidate["duration"]:.1f}s | '
                f'Score '
                f'{candidate["combined_score"]}'
            )

            print(
                f'Topic: '
                f'{candidate["topic"]}'
            )

            print(
                f'Reason: '
                f'{candidate["reason"]}'
            )

            print(
                f'Transcript: '
                f'{candidate["transcript"]}'
            )

        print()
        print(
            candidate_json_path
        )

        print(
            candidate_txt_path
        )

    print()
    print("=" * 72)
    print("OUTPUT")
    print("=" * 72)

    print(
        sentences_json_path
    )

    print(
        sentences_txt_path
    )

    print()
    print(
        "Selector test complete."
    )


if __name__ == "__main__":
    main()