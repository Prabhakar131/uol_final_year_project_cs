from __future__ import annotations

import argparse
from pathlib import Path

from evaluation_utils import (
    base_metadata,
    candidate_rows,
    render_task_report,
    verify_artifacts,
)
from run_benchmarks import (
    OUTPUT_DIR,
    STT_MODELS,
    filter_model_specs,
    run_stt_benchmark,
    set_hugging_face_download_mode,
    summarise_stt_results,
    write_json,
    write_rows,
)


PREFIX = "stt_model"
REPORT_NAME = "stt_model_evaluation_report"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CloudIR STT models outside Jupyter.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-name", default="stt-baseline")
    parser.add_argument("--models", default="")
    parser.add_argument("--limit-samples", type=int, default=2)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_hugging_face_download_mode(args.allow_downloads)

    selected_models = filter_model_specs(STT_MODELS, args.models)
    candidates = candidate_rows(STT_MODELS, selected_models)
    cases = [
        {
            "task": "stt",
            "case_id": "audio_sample_limit",
            "source": "notebooks/model_benchmarks/stt_audio_samples/audio_samples",
            "expected": f"{args.limit_samples if args.limit_samples else 'all'} samples",
        }
    ]

    results = [] if args.metadata_only else run_stt_for_selection(args.limit_samples, selected_models)
    summary = summarise_stt_results(results)

    write_outputs(args.output_dir, candidates, cases, results, summary)
    metadata = base_metadata(
        batch_name=args.batch_name,
        task="stt",
        metadata_only=args.metadata_only,
        allow_downloads=args.allow_downloads,
        selected_models=selected_models,
        case_count=args.limit_samples,
    )
    metadata["verified_artifacts"] = verify_artifacts(args.output_dir, artifact_names())
    write_json(args.output_dir / f"{PREFIX}_evaluation_metadata.json", metadata)
    report = render_task_report(
        "CloudIR STT Model Evaluation",
        metadata,
        candidates,
        cases,
        summary,
        results,
    )
    (args.output_dir / f"{REPORT_NAME}.md").write_text(report, encoding="utf-8")

    print(f"STT evaluation artifacts written to: {args.output_dir}")
    print(f"Markdown report: {args.output_dir / f'{REPORT_NAME}.md'}")


def run_stt_for_selection(sample_limit: int, selected_models: list[dict]) -> list[dict]:
    original_models = STT_MODELS[:]
    try:
        STT_MODELS[:] = selected_models
        return run_stt_benchmark(sample_limit=sample_limit)
    finally:
        STT_MODELS[:] = original_models


def write_outputs(
    output_dir: Path,
    candidates: list[dict],
    cases: list[dict],
    results: list[dict],
    summary: list[dict],
) -> None:
    write_rows(output_dir / f"{PREFIX}_candidates.csv", candidates)
    write_json(output_dir / f"{PREFIX}_candidates.json", candidates)
    write_rows(output_dir / f"{PREFIX}_cases.csv", cases)
    write_json(output_dir / f"{PREFIX}_cases.json", cases)
    write_rows(output_dir / f"{PREFIX}_results.csv", results)
    write_json(output_dir / f"{PREFIX}_results.json", results)
    write_rows(output_dir / f"{PREFIX}_summary.csv", summary)
    write_json(output_dir / f"{PREFIX}_summary.json", summary)


def artifact_names() -> list[str]:
    return [
        f"{PREFIX}_candidates.csv",
        f"{PREFIX}_cases.csv",
        f"{PREFIX}_results.csv",
        f"{PREFIX}_summary.csv",
        f"{PREFIX}_evaluation_metadata.json",
    ]


if __name__ == "__main__":
    main()
