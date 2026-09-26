from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cloudir.paths import RUNTIME_DATA_DIR
from cloudir.scenario.json_boundary import to_camel_case_keys


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing scenario file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_turn_payload(turn_number: int, root_dir: Path) -> dict[str, Any]:
    turn_dir = RUNTIME_DATA_DIR / "turns" / f"turn_{turn_number}"

    return to_camel_case_keys(
        {
            "turn_config": read_json(turn_dir / "turn_config.json"),
            "actions": read_json(turn_dir / "actions.json"),
            "evidence_facts": read_json(turn_dir / "evidence_facts.json"),
            "expected_outcomes": read_json(turn_dir / "expected_outcomes.json"),
        }
    )
