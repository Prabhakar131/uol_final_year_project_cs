"""
Regenerate the identity-management ACSE-Eval runtime scenario using the real
production dataset-preparation pipeline (cloudir.dataset_preparation.build_acse),
so the report's Chapters 3-4 (which describe identity-management) and the
learner evidence-choice experiment (Issue 7) are testing the same scenario.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cloudir.dataset_preparation.build_acse import build_acse_dataset_scenario  # noqa: E402

if __name__ == "__main__":
    result = build_acse_dataset_scenario(scenario_id="identity-management")
    print("Rebuild complete.")
    print(result)
