from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cloudir.paths import ACSE_DATASET_FILE, PROCESSED_DATA_DIR, WORKSPACE_ROOT as PROJECT_ROOT, SOURCE_DATA_DIR

DATASET_ROOT = PROJECT_ROOT / "data"

TRAINING_SCENARIO_ALLOWLIST = {
    "automated-security-response",
    "cost-management",
    "identity-management",
}


def scenario_slug(scenario_id: str) -> str:
    """
    Converts a learner-facing ACSE scenario id into the local processed folder slug.

    ACSE source folders use hyphenated names, while the original project used
    identity_management locally. Processed files use underscores for stable paths.
    """

    key = scenario_id.strip().lower().replace("_", "-")

    if not key:
        raise ValueError("Scenario id is required.")

    if not acse_case_exists(key):
        raise ValueError(
            f"Unsupported ACSE-Eval scenario: {scenario_id}. "
            "Check data/acse_eval.jsonl and data/source/."
        )

    return key.replace("-", "_")


def processed_dir(scenario_id: str) -> Path:
    return PROCESSED_DATA_DIR / scenario_slug(scenario_id)


def source_assets_dir(scenario_id: str) -> Path:
    key = scenario_id.strip().lower().replace("_", "-")
    slug = scenario_slug(scenario_id)

    candidates = [
        SOURCE_DATA_DIR / key,
        SOURCE_DATA_DIR / slug,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return SOURCE_DATA_DIR / key


def acse_case_exists(scenario_id: str) -> bool:
    data_path = ACSE_DATASET_FILE

    if not data_path.exists():
        return False

    target = _normalise_text(scenario_id)

    try:
        for row in iter_acse_cases(data_path):
            if _row_matches_scenario(row=row, target=target):
                return True
    except ValueError:
        return False

    return False


def iter_acse_cases(data_path: Path = ACSE_DATASET_FILE) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with data_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {data_path}"
                ) from exc

            if isinstance(row, dict):
                rows.append(row)

    return rows


def list_available_acse_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []

    if not ACSE_DATASET_FILE.exists():
        return scenarios

    for row in iter_acse_cases(ACSE_DATASET_FILE):
        scenario_id = str(row.get("name") or "").strip()

        if not scenario_id:
            continue

        if scenario_id not in TRAINING_SCENARIO_ALLOWLIST:
            continue

        slug = scenario_id.lower().replace("_", "-").replace("-", "_")
        source_dir = source_assets_dir(scenario_id)
        architecture_path = source_dir / "architecture.png"
        threat_model_path = source_dir / "threat-model.json"

        scenarios.append(
            {
                "id": scenario_id,
                "slug": slug,
                "title": scenario_id,
                "sourceDir": str(source_dir.relative_to(PROJECT_ROOT)),
                "sourceReady": architecture_path.exists() and threat_model_path.exists(),
            }
        )

    return scenarios


def select_acse_case(scenario_id: str = "identity-management") -> dict[str, Any]:
    """
    Selects the ACSE-Eval case from acse_eval.jsonl and saves it as selected_acse_case.json.

    No AI is used in this stage.
    """

    slug = scenario_slug(scenario_id)
    data_path = ACSE_DATASET_FILE

    if not data_path.exists():
        raise FileNotFoundError(f"ACSE-Eval dataset file not found: {data_path}")

    selected_case = _find_case_in_jsonl(data_path=data_path, scenario_id=scenario_id)

    output = {
        "dataset": "ACSE-Eval",
        "scenario_id": scenario_id,
        "scenario_slug": slug,
        "source_file": "data/acse_eval.jsonl",
        "case": selected_case,
    }

    output_dir = processed_dir(scenario_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "selected_acse_case.json"
    _write_json(output_path, output)

    return output


def _find_case_in_jsonl(data_path: Path, scenario_id: str) -> dict[str, Any]:
    rows = iter_acse_cases(data_path)

    if not rows:
        raise ValueError(f"No JSON rows found in {data_path}")

    target = _normalise_text(scenario_id)

    for row in rows:
        if _row_matches_scenario(row=row, target=target):
            return row

    if len(rows) == 1:
        return rows[0]

    raise ValueError(
        "Could not find an identity-management ACSE case in acse_eval.jsonl. "
        "Check that the row contains identity-management, identity_management, "
        "or Identity Management in one of its fields."
    )


def _row_matches_scenario(row: dict[str, Any], target: str) -> bool:
    priority_fields = [
        "id",
        "case_id",
        "scenario_id",
        "scenario",
        "name",
        "title",
        "category",
        "task",
    ]

    for field in priority_fields:
        value = row.get(field)

        if value is not None and target in _normalise_text(str(value)):
            return True

    full_row_text = _normalise_text(json.dumps(row, ensure_ascii=False))

    return target in full_row_text


def _normalise_text(value: str) -> str:
    return (
        value.lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .strip()
    )


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
