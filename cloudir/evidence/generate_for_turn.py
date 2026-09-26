from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cloudir.evidence.template_router import render_evidence_image
from cloudir.evidence_core.fact_repair import normalise_evidence_facts
from cloudir.evidence_core.template_inference import force_correct_template, validate_template_alignment
from cloudir.paths import GENERATED_EVIDENCE_DIR, RUNTIME_DATA_DIR
from cloudir.scenario.evidence_support_roles import normalise_evidence_support_roles
from cloudir.scenario.json_boundary import to_camel_case_keys


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing evidence data file: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Evidence data file exists but is empty: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def generate_evidence_for_turn(turn_number: int, root_dir: Path) -> list[dict[str, Any]]:
    turn_dir = RUNTIME_DATA_DIR / "turns" / f"turn_{turn_number}"
    output_dir = GENERATED_EVIDENCE_DIR / f"turn_{turn_number}"
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_path = turn_dir / "evidence_facts.json"
    evidence_items = read_json(evidence_path)

    if not isinstance(evidence_items, list):
        raise ValueError("evidence_facts.json must contain a list of evidence items.")

    normalise_evidence_support_roles(evidence_items)

    generated_items: list[dict[str, Any]] = []

    for evidence in evidence_items:
        if not isinstance(evidence, dict):
            continue

        required_keys = {
            "id",
            "title",
            "type",
            "summary",
            "why_it_may_matter",
            "template",
            "facts",
        }

        missing = required_keys - set(evidence.keys())

        if missing:
            raise ValueError(
                f"Evidence item {evidence.get('id', 'unknown')} is missing keys before rendering: {missing}"
            )

        # Final safety net before PNG rendering.
        # This corrects wrong AI-generated values such as:
        # type=guardduty, template=cloudtrail.
        force_correct_template(evidence)
        normalise_evidence_facts(evidence)
        validate_template_alignment(evidence)

        evidence_id = evidence["id"]
        output_path = output_dir / f"{evidence_id}.png"

        print(
            f"[Evidence Generator] Rendering evidence_id={evidence_id} "
            f"type={evidence.get('type')} template={evidence.get('template')} "
            f"output={output_path}"
        )

        render_evidence_image(evidence=evidence, output_path=output_path)

        generated_items.append(
            to_camel_case_keys(
                {
                    "id": evidence_id,
                    "title": evidence["title"],
                    "type": evidence["type"],
                    "summary": evidence["summary"],
                    "why_it_may_matter": evidence["why_it_may_matter"],
                    "support_role": evidence.get("support_role", ""),
                    "template": evidence["template"],
                    "image_url": f"/generated_evidence/turn_{turn_number}/{evidence_id}.png",
                    "image_path": str(output_path),
                }
            )
        )

    # Save the repaired evidence_facts.json back to disk.
    # This means your runtime JSON will now show the corrected template/type/facts.
    write_json(evidence_path, evidence_items)

    return generated_items
