from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from cloudir.paths import GENERATED_EVIDENCE_DIR, PREPARED_DATA_DIR, PROCESSED_DATA_DIR, RUNTIME_DATA_DIR
from cloudir.dataset_preparation.select_case import processed_dir, scenario_slug


GENERATED_ARTIFACT_DIRS = [
    PROCESSED_DATA_DIR,
    PREPARED_DATA_DIR,
    RUNTIME_DATA_DIR,
    GENERATED_EVIDENCE_DIR,
]


def flush_generated_artifacts(
    *,
    scope: str = "all",
    scenario_id: str | None = None,
) -> dict[str, Any]:
    """
    Deletes generated/prepared CloudIR artifacts while keeping source data.

    Preserved:
    - data/acse_eval.jsonl
    - data/source/
    - code, environment files, and reports
    """

    if scope == "scenario":
        return flush_generated_artifacts_for_scenario(scenario_id)

    if scope != "all":
        raise ValueError(f"Unsupported cleanup scope: {scope}")

    deleted: list[str] = []
    already_missing: list[str] = []

    for path in GENERATED_ARTIFACT_DIRS:
        _ensure_safe_generated_path(path)

        if path.exists():
            shutil.rmtree(path)
            deleted.append(str(path))
        else:
            already_missing.append(str(path))

        path.mkdir(parents=True, exist_ok=True)

    return {
        "deleted": deleted,
        "alreadyMissing": already_missing,
        "recreated": [str(path) for path in GENERATED_ARTIFACT_DIRS],
    }


def flush_generated_artifacts_for_scenario(scenario_id: str | None) -> dict[str, Any]:
    if not scenario_id:
        raise ValueError("Scenario id is required for scenario cleanup.")

    slug = scenario_slug(scenario_id)
    scenario_processed_dir = processed_dir(scenario_id)
    _ensure_safe_processed_scenario_path(scenario_processed_dir)

    deleted: list[str] = []
    already_missing: list[str] = []
    recreated: list[str] = []

    if scenario_processed_dir.exists():
        shutil.rmtree(scenario_processed_dir)
        deleted.append(str(scenario_processed_dir))
    else:
        already_missing.append(str(scenario_processed_dir))

    scenario_processed_dir.mkdir(parents=True, exist_ok=True)
    recreated.append(str(scenario_processed_dir))

    # Without this, the scenario would still show as prepared from its saved copy.
    prepared_copy = PREPARED_DATA_DIR / slug
    if prepared_copy.resolve().parent != PREPARED_DATA_DIR.resolve():
        raise ValueError(f"Refusing to delete non-scenario prepared path: {prepared_copy}")
    if prepared_copy.exists():
        shutil.rmtree(prepared_copy)
        deleted.append(str(prepared_copy))

    runtime_scenario = _active_runtime_scenario_slug()
    runtime_cleared = runtime_scenario == slug

    if runtime_cleared:
        for path in [RUNTIME_DATA_DIR, GENERATED_EVIDENCE_DIR]:
            _ensure_safe_generated_path(path)

            if path.exists():
                shutil.rmtree(path)
                deleted.append(str(path))
            else:
                already_missing.append(str(path))

            path.mkdir(parents=True, exist_ok=True)
            recreated.append(str(path))

    return {
        "scope": "scenario",
        "scenarioId": scenario_id,
        "scenarioSlug": slug,
        "activeRuntimeScenarioSlug": runtime_scenario,
        "runtimeCleared": runtime_cleared,
        "deleted": deleted,
        "alreadyMissing": already_missing,
        "recreated": recreated,
    }


def _ensure_safe_generated_path(path: Path) -> None:
    allowed = {item.resolve() for item in GENERATED_ARTIFACT_DIRS}

    if path.resolve() not in allowed:
        raise ValueError(f"Refusing to delete non-generated path: {path}")


def _ensure_safe_processed_scenario_path(path: Path) -> None:
    resolved_path = path.resolve()
    resolved_processed_root = PROCESSED_DATA_DIR.resolve()

    if resolved_path.parent != resolved_processed_root:
        raise ValueError(f"Refusing to delete non-scenario processed path: {path}")


def _active_runtime_scenario_slug() -> str | None:
    config_path = RUNTIME_DATA_DIR / "scenario_config.json"

    if not config_path.exists():
        return None

    try:
        import json

        with config_path.open("r", encoding="utf-8") as file:
            scenario_config = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    scenario_id = scenario_config.get("scenario_id")

    if not isinstance(scenario_id, str) or not scenario_id.strip():
        return None

    try:
        return scenario_slug(scenario_id)
    except ValueError:
        return scenario_id.strip().lower().replace("-", "_")
