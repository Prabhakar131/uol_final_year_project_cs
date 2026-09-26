from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cloudir.ai_models.security_model import normalise_acse_threat_model

from .select_case import processed_dir, scenario_slug, source_assets_dir


def normalise_threat(scenario_id: str = "identity-management") -> dict[str, Any]:
    """
    Sends threat-model.json and architecture_facts.json to the real security model,
    then saves normalised_threat_model.json.
    """

    scenario_slug(scenario_id)

    threat_model_path = source_assets_dir(scenario_id) / "threat-model.json"
    architecture_facts_path = processed_dir(scenario_id) / "architecture_facts.json"

    if not threat_model_path.exists():
        raise FileNotFoundError(f"Threat model not found: {threat_model_path}")

    if not architecture_facts_path.exists():
        raise FileNotFoundError(
            f"Architecture facts not found: {architecture_facts_path}"
        )

    threat_model = _read_json(threat_model_path)
    architecture_facts = _read_json(architecture_facts_path)

    normalised_threat_model = normalise_acse_threat_model(
        threat_model=threat_model,
        architecture_facts=architecture_facts,
        scenario_id=scenario_id,
    )

    output_path = processed_dir(scenario_id) / "normalised_threat_model.json"
    _write_json(output_path, normalised_threat_model)

    return normalised_threat_model


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)