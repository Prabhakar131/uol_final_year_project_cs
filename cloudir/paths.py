from __future__ import annotations

from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FRONTEND_DIR = PROJECT_ROOT / "frontend"
WORKSPACE_ROOT = Path(os.environ.get("CLOUDIR_WORKSPACE_DIR", str(PROJECT_ROOT))).resolve()
DATA_DIR = WORKSPACE_ROOT / "data"
SOURCE_DATA_DIR = DATA_DIR / "source"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RUNTIME_DATA_DIR = DATA_DIR / "runtime"
# Each scenario's last successful build, so switching scenarios needs no rebuild.
PREPARED_DATA_DIR = DATA_DIR / "prepared"
OUTPUT_DIR = WORKSPACE_ROOT / "output"
GENERATED_EVIDENCE_DIR = OUTPUT_DIR / "generated_evidence"
ACSE_DATASET_FILE = DATA_DIR / "acse_eval.jsonl"
