"""Durable, versioned snapshots around the single local simulation workspace.

Checkpoints are immutable. Only the active workspace is edited by the AI pipeline;
reset, cleanup and scenario preparation cannot delete saved runs.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid
import zipfile
from io import BytesIO

from cloudir.scenario.scenario_state import ScenarioState


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def replace_directories(pairs: list[tuple[Path, Path]]) -> None:
    """Replace each target with a copy of its source (empty if missing), all or nothing."""
    # Stage every tree first; roll back directory swaps if any replacement fails.
    swaps = []
    try:
        for source, target in pairs:
            stage = target.with_name("." + target.name + "-" + uuid.uuid4().hex)
            old = target.with_name("." + target.name + "-old-" + uuid.uuid4().hex)
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                shutil.copytree(source, stage)
            else:
                stage.mkdir()
            swaps.append((target, stage, old))
        for target, stage, old in swaps:
            if target.exists():
                os.replace(target, old)
            os.replace(stage, target)
    except Exception:
        for target, stage, old in reversed(swaps):
            if old.exists():
                if target.exists():
                    shutil.rmtree(target)
                os.replace(old, target)
        raise
    finally:
        for _, stage, old in swaps:
            for path in (stage, old):
                if path.exists():
                    shutil.rmtree(path)


class RunStore:
    def __init__(self, root: Path):
        self.root = root
        self.directory = root / "data" / "runs"
        self.runtime = root / "data" / "runtime"
        self.evidence = root / "output" / "generated_evidence"
        self.prepared = root / "data" / "prepared"
        self.pointer = self.directory / "active.json"

    def run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"run-[a-f0-9]{32}", run_id or ""):
            raise ValueError("Invalid run ID.")
        return self.directory / run_id

    def get(self, run_id: str) -> dict:
        result = read_json(self.run_dir(run_id) / "run.json")
        if result is None:
            raise ValueError("Saved run was not found.")
        return result

    def active(self) -> dict | None:
        pointer = read_json(self.pointer, {})
        return self.get(pointer["runId"]) if pointer.get("runId") else None

    def list_runs(self) -> list[dict]:
        return sorted(
            [read_json(path) for path in self.directory.glob("run-*/run.json")],
            key=lambda item: item["updatedAt"], reverse=True,
        )

    def checkpoint(self, run_id: str, checkpoint_id: str | None = None) -> tuple[Path, dict]:
        meta = self.get(run_id)
        checkpoint_id = checkpoint_id or meta["latestCheckpoint"]
        if not re.fullmatch(r"cp-[a-f0-9]{32}", checkpoint_id or ""):
            raise ValueError("Invalid checkpoint ID.")
        path = self.run_dir(run_id) / "checkpoints" / checkpoint_id
        record = read_json(path / "checkpoint.json")
        if record is None:
            raise ValueError("Checkpoint was not found.")
        return path, record

    def details(self, run_id: str) -> dict:
        meta = self.get(run_id)
        checkpoints = [read_json(p) for p in (self.run_dir(run_id) / "checkpoints").glob("*/checkpoint.json")]
        checkpoints.sort(key=lambda c: c["createdAt"])
        latest, record = self.checkpoint(run_id)
        traces = [read_json(p) for p in sorted((latest / "runtime" / "evaluations").glob("turn_*_evaluation.json"))]
        completed = bool(record["state"].get("completed"))
        # The timeline marks suspicious events, so the report only sees it for finished runs.
        return {"run": meta, "checkpoints": checkpoints, "evaluations": traces,
                "state": record["state"], "events": read_json(self.run_dir(run_id) / "events.json", []),
                "finalDebrief": read_json(latest / "runtime" / "evaluations" / "final_debrief.json", None),
                "timeline": read_json(latest / "runtime" / "incident_timeline.json", None) if completed else None}

    def record_event(self, event: dict) -> None:
        meta = self.active()
        if meta:
            path = self.run_dir(meta["id"]) / "events.json"
            events = read_json(path, [])
            events.append({"at": now(), **event})
            atomic_json(path, events)

    def create(self, state: ScenarioState, *, label: str = "", parent: dict | None = None,
               legacy: bool = False, test_mode: bool = False) -> dict:
        config = read_json(self.runtime / "scenario_config.json", {})
        if not config.get("scenario_id"):
            raise ValueError("Prepare a scenario before starting a run.")
        run_id = "run-" + uuid.uuid4().hex
        meta = {"schemaVersion": 1, "id": run_id, "scenarioId": config["scenario_id"],
                "label": (label or config.get("title") or config["scenario_id"])[:200],
                "createdAt": now(), "updatedAt": now(), "status": "legacy" if legacy else "active",
                "resumable": not legacy, "testMode": test_mode, "parent": parent,
                "latestCheckpoint": None, "turn": state.current_turn, "completedTurns": 0}
        atomic_json(self.run_dir(run_id) / "run.json", meta)
        atomic_json(self.pointer, {"runId": run_id})
        self.save(state, kind="legacy" if legacy else "initial", status=meta["status"])
        return self.get(run_id)

    def archive_legacy(self, state: ScenarioState) -> None:
        """Never infer a resumable state from old traces belonging to unknown attempts."""
        if read_json(self.pointer, {}).get("lastRunId"):
            return
        if not self.active() and (self.runtime / "scenario_config.json").exists() and any((self.runtime / "evaluations").glob("*.json")):
            self.create(state, label="Existing working data (unverified)", legacy=True)
            self.detach()

    def save(self, state: ScenarioState, *, kind: str = "saved", draft: dict | None = None,
             label: str | None = None, note: str = "", status: str | None = None) -> dict:
        meta = self.active()
        if not meta:
            raise ValueError("No active run. Start or resume a run first.")
        run_dir = self.run_dir(meta["id"])
        checkpoint_id = "cp-" + uuid.uuid4().hex
        staging = run_dir / "checkpoints" / ("." + checkpoint_id)
        destination = staging.with_name(checkpoint_id)
        staging.mkdir(parents=True)
        try:
            for source, name in [(self.runtime, "runtime"), (self.evidence, "evidence")]:
                if source.exists():
                    shutil.copytree(source, staging / name)
            slug = meta["scenarioId"].replace("-", "_")
            processed = self.root / "data" / "processed" / slug
            if processed.exists():
                shutil.copytree(processed, staging / "processed")
            source_dir = self.root / "data" / "source" / meta["scenarioId"]
            if not source_dir.exists():
                source_dir = self.root / "data" / "source" / slug
            if source_dir.exists():
                shutil.copytree(source_dir, staging / "source")
            previous = self.checkpoint(meta["id"])[1] if meta["latestCheckpoint"] else {}
            record = {"id": checkpoint_id, "createdAt": now(), "kind": kind,
                      "state": asdict(state), "draft": draft if draft is not None else previous.get("draft", {}),
                      "note": note[:2000], "models": {key: value for key, value in os.environ.items()
                      if key.startswith("HF_") and (key.endswith("_MODEL_ID") or "MAX_NEW_TOKENS" in key)},
                      "testMode": meta["testMode"]}
            try:
                record["codeVersion"] = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                    stderr=subprocess.DEVNULL, text=True, timeout=2).strip()
                record["codeHasLocalChanges"] = bool(subprocess.check_output(
                    ["git", "diff", "--name-only"], cwd=Path(__file__).resolve().parents[2],
                    stderr=subprocess.DEVNULL, text=True, timeout=2).strip())
            except (OSError, subprocess.SubprocessError):
                record["codeVersion"] = "unknown"
            manifest = {str(p.relative_to(staging)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in staging.rglob("*") if p.is_file()}
            atomic_json(staging / "manifest.json", manifest)
            atomic_json(staging / "checkpoint.json", record)
            os.replace(staging, destination)
            meta.update(latestCheckpoint=checkpoint_id, updatedAt=now(), turn=state.current_turn,
                        completedTurns=len(state.action_history),
                        status=status or ("completed" if state.completed else "active"))
            if label is not None and label.strip():
                meta["label"] = label.strip()[:200]
            atomic_json(run_dir / "run.json", meta)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return {"run": meta, "checkpoint": record}

    def detach(self) -> None:
        current = read_json(self.pointer, {})
        atomic_json(self.pointer, {"runId": None, "lastRunId": current.get("runId") or current.get("lastRunId")})

    @staticmethod
    def apply_state(state: ScenarioState, values: dict) -> None:
        restored = ScenarioState(**{f.name: values[f.name] for f in fields(ScenarioState) if f.name in values})
        state.__dict__.update(restored.__dict__)

    def restore(self, run_id: str, state: ScenarioState, checkpoint_id: str | None = None) -> dict:
        meta = self.get(run_id)
        if not meta["resumable"]:
            raise ValueError("Legacy working data can be exported but cannot be resumed reliably.")
        source, record = self.checkpoint(run_id, checkpoint_id)
        for name, digest in read_json(source / "manifest.json", {}).items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Saved checkpoint failed its integrity check.")
        self.keep_workspace_build()
        targets = [("runtime", self.runtime), ("evidence", self.evidence)]
        if (source / "processed").exists():
            targets.append(("processed", self.root / "data" / "processed" / meta["scenarioId"].replace("-", "_")))
        replace_directories([(source / key, target) for key, target in targets])
        self.apply_state(state, record["state"])
        atomic_json(self.pointer, {"runId": run_id})
        return record

    def prepared_dir(self, scenario_id: str) -> Path:
        slug = str(scenario_id or "").strip().lower().replace("-", "_")
        if not re.fullmatch(r"[a-z0-9_]+", slug):
            raise ValueError("Invalid scenario ID.")
        return self.prepared / slug

    def has_prepared(self, scenario_id: str) -> bool:
        try:
            return (self.prepared_dir(scenario_id) / "runtime" / "turns" / "turn_1" / "turn_config.json").exists()
        except ValueError:
            return False

    def save_prepared(self) -> Path:
        """Keep a copy of a freshly built scenario's Turn 1 starting point.

        The pipeline has one shared workspace, so building or resuming another
        scenario overwrites it. Starting this scenario later restores the copy.
        """
        scenario_id = read_json(self.runtime / "scenario_config.json", {}).get("scenario_id")
        if not scenario_id or not (self.runtime / "turns" / "turn_1" / "turn_config.json").exists():
            raise ValueError("The workspace does not hold a prepared scenario.")
        destination = self.prepared_dir(scenario_id)
        staging = destination.with_name("." + destination.name + "-" + uuid.uuid4().hex)

        def starting_point(folder: str, names: list[str]) -> list[str]:
            return [name for name in names if name == "evaluations" or (Path(folder).name == "turns" and name != "turn_1")]

        try:
            shutil.copytree(self.runtime, staging / "runtime", ignore=starting_point)
            if (self.evidence / "turn_1").exists():
                shutil.copytree(self.evidence / "turn_1", staging / "evidence" / "turn_1")
            replace_directories([(staging, destination)])
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return destination

    def keep_workspace_build(self) -> None:
        """Save the workspace's scenario as its prepared copy if it has none yet.

        Covers builds made before copies were kept, before the workspace is replaced.
        """
        try:
            scenario_id = read_json(self.runtime / "scenario_config.json", {}).get("scenario_id")
        except ValueError:
            return
        turn_1 = self.runtime / "turns" / "turn_1"
        complete = all((turn_1 / name).exists() for name in
                       ("turn_config.json", "actions.json", "evidence_facts.json", "expected_outcomes.json"))
        if scenario_id and complete and not self.has_prepared(scenario_id):
            self.save_prepared()

    def restore_prepared(self, scenario_id: str) -> None:
        source = self.prepared_dir(scenario_id)
        replace_directories([(source / "runtime", self.runtime), (source / "evidence", self.evidence)])

    def recover_active(self, state: ScenarioState) -> None:
        meta = self.active()
        if meta and meta["resumable"]:
            self.restore(meta["id"], state)
        elif meta:
            self.detach()

    def fresh_start(self, state: ScenarioState, *, label: str = "", test_mode: bool = False,
                    scenario_id: str | None = None) -> dict:
        self.archive_legacy(state)
        if self.active():
            self.save(state, kind="before_restart", status="completed" if state.completed else "paused")
        if scenario_id and self.has_prepared(scenario_id):
            # Start from the scenario's latest build, even if another was built or resumed since.
            self.keep_workspace_build()
            self.restore_prepared(scenario_id)
        config = read_json(self.runtime / "scenario_config.json", {})
        if not (self.runtime / "turns" / "turn_1" / "turn_config.json").exists():
            raise ValueError("Prepare a scenario before starting a run.")
        # Only the active workspace is cleared. Saved runs and original Turn 1 survive.
        for base in [self.runtime / "turns", self.evidence]:
            for directory in base.glob("turn_*"):
                if directory.name != "turn_1" and directory.is_dir():
                    shutil.rmtree(directory)
        evaluations = self.runtime / "evaluations"
        if evaluations.exists():
            shutil.rmtree(evaluations)
        state.reset()
        state.apply_max_turns(config.get("total_turns"))
        return self.create(state, label=label, test_mode=test_mode)

    def export(self, run_id: str) -> BytesIO:
        self.get(run_id)
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in self.run_dir(run_id).rglob("*"):
                if file.is_file() and not any(p.startswith(".") for p in file.relative_to(self.run_dir(run_id)).parts):
                    archive.write(file, file.relative_to(self.run_dir(run_id)))
        buffer.seek(0)
        return buffer
