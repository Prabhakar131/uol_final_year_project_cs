from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from run_benchmarks import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    SECURITY_MODELS,
    build_security_cases,
    filter_model_specs,
    run_security_model_benchmarks,
    set_hugging_face_download_mode,
    summarise_classification_results,
    write_json,
    write_rows,
)


REPORT_NAME = "security_model_evaluation_report"
HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CloudIR security-reasoning models outside Jupyter and "
            "export notebook-ready CSV, JSON, and Markdown artifacts."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated security-evaluation artifacts.",
    )
    parser.add_argument(
        "--models",
        default="",
        help=(
            "Comma-separated security model names to run. Empty means the "
            "current application security model only."
        ),
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=0,
        help="Limit evaluation cases for a quick smoke run. 0 means all cases.",
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow missing Hugging Face model files to download.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write candidate/evaluation plan artifacts without loading models.",
    )
    parser.add_argument(
        "--batch-name",
        default="security-batch",
        help="Human-readable name for this download/evaluation/delete batch.",
    )
    parser.add_argument(
        "--cache-budget-gb",
        type=float,
        default=25.0,
        help="Advisory cache budget for the selected model batch.",
    )
    parser.add_argument(
        "--delete-selected-cache-after-run",
        action="store_true",
        help=(
            "Delete Hugging Face cache directories for selected models after "
            "artifacts are written and verified. Use only for temporary "
            "comparison batches."
        ),
    )
    parser.add_argument(
        "--delete-selected-cache-only",
        action="store_true",
        help=(
            "Verify existing artifacts and delete selected model caches without "
            "rerunning inference or rewriting result CSV files."
        ),
    )
    args = parser.parse_args()

    set_hugging_face_download_mode(args.allow_downloads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_models = select_security_models(args.models)
    if args.delete_selected_cache_only:
        delete_cache_only(args.output_dir, selected_models)
        return

    cases = build_security_cases()
    if args.limit_cases > 0:
        cases = cases[: args.limit_cases]

    candidate_rows = build_candidate_rows(selected_models)
    case_rows = build_case_rows(cases)
    cache_before = build_cache_rows(selected_models)
    skipped_models = build_skipped_model_rows(selected_models)

    write_rows(args.output_dir / "security_model_candidates.csv", candidate_rows)
    write_json(args.output_dir / "security_model_candidates.json", candidate_rows)
    write_rows(args.output_dir / "security_evaluation_cases.csv", case_rows)
    write_json(args.output_dir / "security_evaluation_cases.json", case_rows)
    write_rows(args.output_dir / "security_model_batch_cache_before.csv", cache_before)
    write_json(args.output_dir / "security_model_batch_cache_before.json", cache_before)
    write_rows(args.output_dir / "security_model_batch_skipped.csv", skipped_models)
    write_json(args.output_dir / "security_model_batch_skipped.json", skipped_models)

    if args.metadata_only:
        result_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
    else:
        print(f"Security models selected: {', '.join(model['name'] for model in selected_models)}")
        print(f"Evaluation cases selected: {len(cases)}")
        result_rows = run_security_model_benchmarks_for_selection(cases, selected_models)
        summary_rows = summarise_classification_results(
            result_rows,
            group_key="model_name",
        )

    write_rows(args.output_dir / "security_model_focused_results.csv", result_rows)
    write_json(args.output_dir / "security_model_focused_results.json", result_rows)
    write_rows(args.output_dir / "security_model_focused_summary.csv", summary_rows)
    write_json(args.output_dir / "security_model_focused_summary.json", summary_rows)

    metadata = build_metadata(
        selected_models=selected_models,
        case_count=len(cases),
        metadata_only=args.metadata_only,
        allow_downloads=args.allow_downloads,
        batch_name=args.batch_name,
        cache_budget_gb=args.cache_budget_gb,
        delete_selected_cache_after_run=args.delete_selected_cache_after_run,
        cache_before=cache_before,
        skipped_models=skipped_models,
    )
    write_json(args.output_dir / "security_model_evaluation_metadata.json", metadata)

    verified_artifacts = verify_artifacts(args.output_dir)
    deletion_rows: list[dict[str, Any]] = []
    if args.delete_selected_cache_after_run:
        deletion_rows = delete_selected_model_caches(
            selected_models=selected_models,
            verified_artifacts=verified_artifacts,
        )

    cache_after = build_cache_rows(selected_models)
    write_rows(args.output_dir / "security_model_batch_cache_after.csv", cache_after)
    write_json(args.output_dir / "security_model_batch_cache_after.json", cache_after)
    write_rows(args.output_dir / "security_model_batch_deletions.csv", deletion_rows)
    write_json(args.output_dir / "security_model_batch_deletions.json", deletion_rows)

    metadata["verified_artifacts"] = verified_artifacts
    metadata["cache_after"] = cache_after
    metadata["deletions"] = deletion_rows
    write_json(args.output_dir / "security_model_evaluation_metadata.json", metadata)

    markdown = render_markdown_report(
        metadata=metadata,
        candidates=candidate_rows,
        cases=case_rows,
        summary=summary_rows,
        results=result_rows,
    )
    (args.output_dir / f"{REPORT_NAME}.md").write_text(markdown, encoding="utf-8")

    print(f"Security evaluation artifacts written to: {args.output_dir}")
    print(f"Markdown report: {args.output_dir / f'{REPORT_NAME}.md'}")


def delete_cache_only(output_dir: Path, selected_models: list[dict[str, Any]]) -> None:
    verified_artifacts = verify_artifacts(output_dir)
    deletion_rows = delete_selected_model_caches(
        selected_models=selected_models,
        verified_artifacts=verified_artifacts,
    )
    cache_after = build_cache_rows(selected_models)
    write_rows(output_dir / "security_model_batch_cache_after.csv", cache_after)
    write_json(output_dir / "security_model_batch_cache_after.json", cache_after)
    write_rows(output_dir / "security_model_batch_deletions.csv", deletion_rows)
    write_json(output_dir / "security_model_batch_deletions.json", deletion_rows)

    metadata_path = output_dir / "security_model_evaluation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata["verified_artifacts"] = verified_artifacts
    metadata["cache_after"] = cache_after
    metadata["deletions"] = deletion_rows
    metadata["delete_only_ran_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(metadata_path, metadata)

    print(f"Verified artifacts: {all(row['exists'] for row in verified_artifacts)}")
    print(f"Cache deletion log written to: {output_dir / 'security_model_batch_deletions.csv'}")


def select_security_models(names_csv: str) -> list[dict[str, Any]]:
    if names_csv.strip():
        return filter_model_specs(SECURITY_MODELS, names_csv)

    active_model_id = os.getenv("HF_SECURITY_MODEL_ID")
    active = [
        model
        for model in SECURITY_MODELS
        if model.get("used_in_application") is True
        or (active_model_id and model.get("model_id") == active_model_id)
    ]

    return active or [SECURITY_MODELS[0]]


def run_security_model_benchmarks_for_selection(
    cases: list[Any],
    selected_models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    original_models = SECURITY_MODELS[:]
    try:
        SECURITY_MODELS[:] = selected_models
        return run_security_model_benchmarks(cases)
    finally:
        SECURITY_MODELS[:] = original_models


def build_candidate_rows(selected_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_names = {model["name"] for model in selected_models}
    rows: list[dict[str, Any]] = []

    for model in SECURITY_MODELS:
        rows.append(
            {
                "selected_for_run": model["name"] in selected_names,
                "used_in_application": bool(model.get("used_in_application")),
                "name": model["name"],
                "model_id": model["model_id"],
                "family": model["family"],
                "benchmark_mode": model["benchmark_mode"],
                "metric": model["metric"],
                "notes": model.get("notes", ""),
                "cache_dir": str(hf_cache_dir(model["model_id"])),
                "cache_size_bytes": cache_size_bytes(hf_cache_dir(model["model_id"])),
            }
        )

    return rows


def build_skipped_model_rows(selected_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_names = {model["name"] for model in selected_models}
    return [
        {
            "name": model["name"],
            "model_id": model["model_id"],
            "reason": "not selected for this batch",
            "benchmark_mode": model["benchmark_mode"],
        }
        for model in SECURITY_MODELS
        if model["name"] not in selected_names
    ]


def build_cache_rows(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        cache_dir = hf_cache_dir(model["model_id"])
        size_bytes = cache_size_bytes(cache_dir)
        rows.append(
            {
                "name": model["name"],
                "model_id": model["model_id"],
                "cache_dir": str(cache_dir),
                "cached": cache_dir.exists(),
                "cache_size_bytes": size_bytes,
                "cache_size_gb": round(size_bytes / 1_000_000_000, 3),
            }
        )
    return rows


def hf_cache_dir(model_id: str) -> Path:
    return HF_CACHE_ROOT / ("models--" + model_id.replace("/", "--"))


def cache_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def verify_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    required = [
        "security_model_candidates.csv",
        "security_evaluation_cases.csv",
        "security_model_focused_results.csv",
        "security_model_focused_summary.csv",
        "security_model_evaluation_metadata.json",
    ]
    rows = []
    for name in required:
        path = output_dir / name
        rows.append(
            {
                "artifact": name,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return rows


def delete_selected_model_caches(
    selected_models: list[dict[str, Any]],
    verified_artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    artifacts_ok = all(row["exists"] for row in verified_artifacts)
    production_model_ids = read_production_model_ids()
    rows = []
    for model in selected_models:
        cache_dir = hf_cache_dir(model["model_id"])
        size_before = cache_size_bytes(cache_dir)
        deleted = False
        error = ""
        if not artifacts_ok:
            error = "Skipped deletion because required artifacts were not verified."
        elif model["model_id"] in production_model_ids:
            error = "Skipped deletion because this model is listed in .env production configuration."
        elif cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                deleted = True
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "name": model["name"],
                "model_id": model["model_id"],
                "cache_dir": str(cache_dir),
                "size_before_bytes": size_before,
                "deleted": deleted,
                "error": error,
            }
        )
    return rows


def read_production_model_ids() -> set[str]:
    env_path = PROJECT_ROOT / ".env"
    keys = {
        "HF_SECURITY_MODEL_ID",
        "HF_IMAGE_MODEL_ID",
        "HF_COACH_MODEL_ID",
        "HF_VOICE_MODEL_ID",
    }
    if not env_path.exists():
        return set()
    model_ids = set()
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key.strip() in keys and value.strip():
            model_ids.add(value.strip())
    return model_ids


def build_case_rows(cases: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case.case_id,
            "turn": case.turn,
            "action": case.selected_action.get("title", ""),
            "evidence": case.selected_evidence.get("title", ""),
            "evidence_type": case.selected_evidence.get("template")
            or case.selected_evidence.get("type", ""),
            "expected_support_role": case.expected_support_role,
            "expected_verdict": case.expected_verdict,
        }
        for case in cases
    ]


def build_metadata(
    selected_models: list[dict[str, Any]],
    case_count: int,
    metadata_only: bool,
    allow_downloads: bool,
    batch_name: str,
    cache_budget_gb: float,
    delete_selected_cache_after_run: bool,
    cache_before: list[dict[str, Any]],
    skipped_models: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "batch_name": batch_name,
        "project_root": str(PROJECT_ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "allow_downloads": allow_downloads,
        "metadata_only": metadata_only,
        "cache_budget_gb": cache_budget_gb,
        "delete_selected_cache_after_run": delete_selected_cache_after_run,
        "selected_cache_size_gb_before": round(
            sum(row["cache_size_bytes"] for row in cache_before) / 1_000_000_000,
            3,
        ),
        "case_count": case_count,
        "selected_models": [
            {
                "name": model["name"],
                "model_id": model["model_id"],
                "family": model["family"],
            }
            for model in selected_models
        ],
        "skipped_models": skipped_models,
        "cache_before": cache_before,
    }


def render_markdown_report(
    metadata: dict[str, Any],
    candidates: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> str:
    lines = [
        "# CloudIR Security Model Evaluation",
        "",
        f"Generated: `{metadata['generated_at']}`",
        f"Batch: `{metadata['batch_name']}`",
        f"Cases: `{metadata['case_count']}`",
        f"Mode: `{'metadata only' if metadata['metadata_only'] else 'model inference'}`",
        f"Selected cache before run: `{metadata['selected_cache_size_gb_before']} GB`",
        f"Advisory cache budget: `{metadata['cache_budget_gb']} GB`",
        "",
        "## Selected Models",
        "",
        "| Model | Model ID | Family |",
        "|---|---|---|",
    ]

    for model in metadata["selected_models"]:
        lines.append(
            f"| {model['name']} | `{model['model_id']}` | {model['family']} |"
        )

    if metadata.get("skipped_models"):
        lines.extend(["", "## Skipped Models", ""])
        lines.extend(markdown_table(metadata["skipped_models"]))

    if metadata.get("cache_before"):
        lines.extend(["", "## Cache Before Run", ""])
        lines.extend(markdown_table(metadata["cache_before"]))

    if metadata.get("cache_after"):
        lines.extend(["", "## Cache After Run", ""])
        lines.extend(markdown_table(metadata["cache_after"]))

    if metadata.get("deletions"):
        lines.extend(["", "## Cache Deletion Log", ""])
        lines.extend(markdown_table(metadata["deletions"]))

    lines.extend(["", "## Summary", ""])
    if summary:
        lines.extend(markdown_table(summary))
    else:
        lines.append("No inference summary yet. Run without `--metadata-only` to generate results.")

    lines.extend(["", "## Candidate Matrix", ""])
    lines.extend(
        markdown_table(
            candidates,
            columns=[
                "selected_for_run",
                "used_in_application",
                "name",
                "model_id",
                "family",
                "benchmark_mode",
                "notes",
            ],
        )
    )

    lines.extend(["", "## Evaluation Case Mix", ""])
    case_counts = (
        pd.DataFrame(cases)
        .groupby("expected_support_role")
        .size()
        .reset_index(name="cases")
        .to_dict("records")
        if cases
        else []
    )
    lines.extend(markdown_table(case_counts) if case_counts else ["No cases found."])

    if results:
        lines.extend(["", "## Result Detail Sample", ""])
        lines.extend(markdown_table(results[:12]))

    return "\n".join(lines) + "\n"


def markdown_table(
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
) -> list[str]:
    if not rows:
        return []
    columns = columns or list(rows[0].keys())
    output = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        output.append(
            "| "
            + " | ".join(markdown_cell(row.get(column, "")) for column in columns)
            + " |"
        )
    return output


def markdown_cell(value: Any) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return text[:180] + "..." if len(text) > 180 else text


if __name__ == "__main__":
    main()
