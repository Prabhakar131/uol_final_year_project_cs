from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cloudir.ai_models.image_model import analyse_architecture_diagram

from .select_case import processed_dir, scenario_slug, source_assets_dir


def parse_architecture(scenario_id: str = "identity-management") -> dict[str, Any]:
    """
    Sends architecture.png to the real VLM and saves architecture_facts.json.
    """

    scenario_slug(scenario_id)

    architecture_path = source_assets_dir(scenario_id) / "architecture.png"

    if not architecture_path.exists():
        raise FileNotFoundError(f"Architecture diagram not found: {architecture_path}")

    architecture_facts = analyse_architecture_diagram(architecture_path)

    output_path = processed_dir(scenario_id) / "architecture_facts.json"
    _write_json(output_path, architecture_facts)

    return architecture_facts


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)