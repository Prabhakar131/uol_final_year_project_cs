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
    VLM_MODELS,
    build_security_cases,
    build_vlm_cases,
    filter_model_specs,
    run_vlm_model_benchmarks,
    set_hugging_face_download_mode,
    set_vlm_generation_limit,
    summarise_vlm_results,
    write_json,
    write_rows,
)


PREFIX = "vlm_model"
REPORT_NAME = "vlm_model_evaluation_report"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CloudIR VLM screenshot models outside Jupyter.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-name", default="vlm-baseline")
    parser.add_argument("--models", default="")
    parser.add_argument("--limit-cases", type=int, default=1)
    # Must stay >= run_benchmarks.VLM_MAX_NEW_TOKENS (220). The prompt asks for JSON ending in
    # "supports_selected_action", so a lower budget truncates output before that field is
    # generated — every model then reads as "unparseable" (see notebook Section 4.3).
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_hugging_face_download_mode(args.allow_downloads)
    set_vlm_generation_limit(args.max_new_tokens)

    selected_models = filter_model_specs(VLM_MODELS, args.models)
    security_cases = build_security_cases()
    vlm_cases = build_vlm_cases(security_cases)
    if args.limit_cases > 0:
        vlm_cases = vlm_cases[: args.limit_cases]
    case_rows = build_case_rows(vlm_cases)
    candidates = candidate_rows(VLM_MODELS, selected_models)

    results = []
    if not args.metadata_only:
        for model in selected_models:
            model_results = run_vlm_model_benchmarks(vlm_cases, [model])
            results.extend(model_results)
            summary = summarise_vlm_results(results)
            write_outputs(args.output_dir, candidates, case_rows, results, summary)
    summary = summarise_vlm_results(results)

    write_outputs(args.output_dir, candidates, case_rows, results, summary)
    metadata = base_metadata(
        batch_name=args.batch_name,
        task="vlm",
        metadata_only=args.metadata_only,
        allow_downloads=args.allow_downloads,
        selected_models=selected_models,
        case_count=len(vlm_cases),
    )
    metadata["max_new_tokens"] = args.max_new_tokens
    metadata["verified_artifacts"] = verify_artifacts(args.output_dir, artifact_names())
    write_json(args.output_dir / f"{PREFIX}_evaluation_metadata.json", metadata)
    report = render_task_report(
        "CloudIR VLM Model Evaluation",
        metadata,
        candidates,
        case_rows,
        summary,
        results,
    )
    (args.output_dir / f"{REPORT_NAME}.md").write_text(report, encoding="utf-8")

    print(f"VLM evaluation artifacts written to: {args.output_dir}")
    print(f"Markdown report: {args.output_dir / f'{REPORT_NAME}.md'}")


def build_case_rows(vlm_cases: list[dict]) -> list[dict]:
    return [
        {
            "task": "vlm",
            "case_id": case["case_id"],
            "source": case["image_path"],
            "expected": case["expected_support_role"],
            "expected_template": case["expected_template"],
            "expected_keywords": ", ".join(case["expected_keywords"]),
        }
        for case in vlm_cases
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
