from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from cloudir.paths import (
    ACSE_DATASET_FILE,
    GENERATED_EVIDENCE_DIR,
    PREPARED_DATA_DIR,
    PROCESSED_DATA_DIR,
    RUNTIME_DATA_DIR,
    SOURCE_DATA_DIR,
)
from cloudir.dataset_preparation.select_case import (
    list_available_acse_scenarios,
    scenario_slug as resolve_scenario_slug,
    source_assets_dir,
)


DEFAULT_DATASET_SCENARIO = "identity-management"


def scenario_slug(scenario_id: str) -> str:
    return resolve_scenario_slug(scenario_id)


def get_dataset_status(
    root_dir: Path,
    scenario_id: str = DEFAULT_DATASET_SCENARIO,
) -> dict[str, Any]:
    slug = scenario_slug(scenario_id)

    data_jsonl = ACSE_DATASET_FILE
    source_dir = source_assets_dir(scenario_id)
    architecture_png = source_dir / "architecture.png"
    threat_model_json = source_dir / "threat-model.json"

    processed_dir = PROCESSED_DATA_DIR / slug
    selected_case = processed_dir / "selected_acse_case.json"
    architecture_facts = processed_dir / "architecture_facts.json"
    normalised_threat_model = processed_dir / "normalised_threat_model.json"

    # Starting a scenario restores its saved copy when there is one, so that copy
    # is what is prepared, even after another scenario was built.
    prepared_copy = PREPARED_DATA_DIR / slug
    saved_copy = (prepared_copy / "runtime" / "turns" / "turn_1" / "turn_config.json").exists()
    runtime_dir = prepared_copy / "runtime" if saved_copy else RUNTIME_DATA_DIR

    scenario_seed = runtime_dir / "scenario_seed.json"
    hidden_truth = runtime_dir / "hidden_truth.json"
    scenario_config = runtime_dir / "scenario_config.json"

    turn_1_dir = runtime_dir / "turns" / "turn_1"
    turn_config = turn_1_dir / "turn_config.json"
    actions = turn_1_dir / "actions.json"
    evidence_facts = turn_1_dir / "evidence_facts.json"
    expected_outcomes = turn_1_dir / "expected_outcomes.json"

    evidence_dir = (prepared_copy / "evidence" if saved_copy else GENERATED_EVIDENCE_DIR) / "turn_1"
    screenshots = sorted(evidence_dir.glob("*.png")) if evidence_dir.exists() else []

    required_source_files = {
        "data_jsonl": data_jsonl.exists(),
        "architecture_png": architecture_png.exists(),
        "threat_model_json": threat_model_json.exists(),
    }

    processed_files = {
        "selected_acse_case": selected_case.exists(),
        "architecture_facts": architecture_facts.exists(),
        "normalised_threat_model": normalised_threat_model.exists(),
    }

    runtime_files = {
        "scenario_seed": scenario_seed.exists(),
        "hidden_truth": hidden_truth.exists(),
        "scenario_config": scenario_config.exists(),
        "turn_config": turn_config.exists(),
        "actions": actions.exists(),
        "evidence_facts": evidence_facts.exists(),
        "expected_outcomes": expected_outcomes.exists(),
    }

    source_ready = all(required_source_files.values())
    processed_ready = all(processed_files.values())
    try:
        active_scenario = json.loads(scenario_config.read_text()).get("scenario_id")
    except (OSError, ValueError):
        active_scenario = None
    runtime_matches = active_scenario == scenario_id
    runtime_ready = all(runtime_files.values()) and runtime_matches
    screenshots_ready = len(screenshots) > 0
    prepared = source_ready and processed_ready and runtime_ready and screenshots_ready

    return {
        "dataset": "ACSE-Eval",
        "scenarioId": scenario_id,
        "scenarioSlug": slug,
        "enabledScenarios": list_available_acse_scenarios(),
        "requiredSourceFiles": required_source_files,
        "processedFiles": processed_files,
        "runtimeFiles": runtime_files,
        "sourceReady": source_ready,
        "processedReady": processed_ready,
        "runtimeReady": runtime_ready,
        "activeRuntimeScenarioId": active_scenario,
        "runtimeMatchesScenario": runtime_matches,
        "savedCopy": saved_copy,
        "screenshotsReady": screenshots_ready,
        "prepared": prepared,
        "screenshotCount": len(screenshots),
        "screenshots": [
            str(path.relative_to(root_dir)) for path in screenshots
        ],
        "sourcePaths": {
            "architecture_png": str(architecture_png.relative_to(root_dir)),
            "threat_model_json": str(threat_model_json.relative_to(root_dir)),
        },
        "architectureAssetUrl": f"/dataset-assets/{scenario_id}/architecture.png",
    }
