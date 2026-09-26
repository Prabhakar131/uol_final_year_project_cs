from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from run_benchmarks import PROJECT_ROOT


HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"


def base_metadata(
    batch_name: str,
    task: str,
    metadata_only: bool,
    allow_downloads: bool,
    selected_models: list[dict[str, Any]],
    case_count: int,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "batch_name": batch_name,
        "task": task,
        "project_root": str(PROJECT_ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "allow_downloads": allow_downloads,
        "metadata_only": metadata_only,
        "case_count": case_count,
        "selected_models": [
            {
                "name": model["name"],
                "model_id": model["model_id"],
                "family": model["family"],
            }
            for model in selected_models
        ],
    }


def candidate_rows(
    all_models: list[dict[str, Any]],
    selected_models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_names = {model["name"] for model in selected_models}
    rows: list[dict[str, Any]] = []
    for model in all_models:
        cache_dir = hf_cache_dir(model["model_id"])
        rows.append(
            {
                "selected_for_run": model["name"] in selected_names,
                "stage": model["stage"],
                "used_in_application": bool(model.get("used_in_application")),
                "name": model["name"],
                "model_id": model["model_id"],
                "family": model["family"],
                "benchmark_mode": model["benchmark_mode"],
                "metric": model["metric"],
                "cached": cache_dir.exists(),
                "cache_size_gb": round(cache_size_bytes(cache_dir) / 1_000_000_000, 3),
                "notes": model.get("notes", ""),
            }
        )
    return rows


def verify_artifacts(output_dir: Path, names: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "artifact": name,
            "exists": (output_dir / name).exists(),
            "size_bytes": (output_dir / name).stat().st_size
            if (output_dir / name).exists()
            else 0,
        }
        for name in names
    ]


def render_task_report(
    title: str,
    metadata: dict[str, Any],
    candidates: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {title}",
        "",
        f"Generated: `{metadata['generated_at']}`",
        f"Batch: `{metadata['batch_name']}`",
        f"Task: `{metadata['task']}`",
        f"Mode: `{'metadata only' if metadata['metadata_only'] else 'model inference'}`",
        f"Cases: `{metadata['case_count']}`",
        "",
        "## Selected Models",
        "",
    ]
    lines.extend(markdown_table(metadata["selected_models"]) if metadata["selected_models"] else ["No models selected."])
    lines.extend(["", "## Summary", ""])
    lines.extend(markdown_table(summary) if summary else ["No inference summary."])
    lines.extend(["", "## Evaluation Cases", ""])
    lines.extend(markdown_table(cases) if cases else ["No cases found."])
    if results:
        lines.extend(["", "## Result Sample", ""])
        lines.extend(markdown_table(results[:12]))
    lines.extend(["", "## Candidate Matrix", ""])
    lines.extend(markdown_table(candidates) if candidates else ["No candidates found."])
    return "\n".join(lines) + "\n"


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_cell(row.get(column, "")) for column in columns) + " |")
    return lines


def markdown_cell(value: Any) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return text[:180] + "..." if len(text) > 180 else text


def hf_cache_dir(model_id: str) -> Path:
    return HF_CACHE_ROOT / ("models--" + model_id.replace("/", "--"))


def cache_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
