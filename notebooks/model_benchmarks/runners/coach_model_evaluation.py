from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evaluation_utils import (
    base_metadata,
    candidate_rows,
    render_task_report,
    verify_artifacts,
)
from run_benchmarks import (
    COACH_MODELS,
    OUTPUT_DIR,
    build_security_cases,
    filter_model_specs,
    run_coach_model_benchmarks,
    set_hugging_face_download_mode,
    summarise_coach_results,
    write_json,
    write_rows,
)


PREFIX = "coach_model"
REPORT_NAME = "coach_model_evaluation_report"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CloudIR coach feedback models outside Jupyter.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-name", default="coach-baseline")
    parser.add_argument("--models", default="")
    parser.add_argument("--limit-cases", type=int, default=2)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_hugging_face_download_mode(args.allow_downloads)

    selected_models = filter_model_specs(COACH_MODELS, args.models)
    all_cases = build_security_cases()
    cases = all_cases[: args.limit_cases] if args.limit_cases > 0 else all_cases
    case_rows = build_case_rows(cases)
    candidates = candidate_rows(COACH_MODELS, selected_models)

    results = [] if args.metadata_only else run_coach_for_selection(cases, selected_models)
    summary = summarise_coach_results(results)

    write_outputs(args.output_dir, candidates, case_rows, results, summary)
    metadata = base_metadata(
        batch_name=args.batch_name,
        task="coach",
        metadata_only=args.metadata_only,
        allow_downloads=args.allow_downloads,
        selected_models=selected_models,
        case_count=len(cases),
    )
    metadata["verified_artifacts"] = verify_artifacts(args.output_dir, artifact_names())
    write_json(args.output_dir / f"{PREFIX}_evaluation_metadata.json", metadata)
    report = render_task_report(
        "CloudIR Coach Model Evaluation",
        metadata,
        candidates,
        case_rows,
        summary,
        results,
    )
    (args.output_dir / f"{REPORT_NAME}.md").write_text(report, encoding="utf-8")

    print(f"Coach evaluation artifacts written to: {args.output_dir}")
    print(f"Markdown report: {args.output_dir / f'{REPORT_NAME}.md'}")


def run_coach_for_selection(cases: list[Any], selected_models: list[dict]) -> list[dict]:
    original_models = COACH_MODELS[:]
    try:
        COACH_MODELS[:] = selected_models
        return run_coach_model_benchmarks(cases)
    finally:
        COACH_MODELS[:] = original_models


def build_case_rows(cases: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "task": "coach",
            "case_id": case.case_id,
            "source": case.selected_evidence.get("title", ""),
            "expected": case.expected_verdict,
        }
        for case in cases
    ]


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
