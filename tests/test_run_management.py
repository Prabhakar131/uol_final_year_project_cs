"""Exercise real Flask routes/storage with deterministic AI substitutes.

All mutations happen in an isolated workspace; no model weights are loaded.
Run: .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import atexit
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
from io import BytesIO

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = tempfile.TemporaryDirectory(prefix="cloudir-run-tests-")
atexit.register(WORKSPACE.cleanup)
# Scenario runtime files (turns 1-5 and their evaluations) come from a fixed copy,
# not data/runtime: the app rewrites that folder as runs start and cleanups run,
# and committing the live folder once removed the turns these tests need.
RUNTIME_FIXTURE = Path(__file__).parent / "fixtures" / "runtime"
# Processed scenario files and turn 1 screenshots likewise: the live folders are
# emptied by cleanups and hold only the scenarios built so far.
PROCESSED_FIXTURE = Path(__file__).parent / "fixtures" / "processed"
EVIDENCE_FIXTURE = Path(__file__).parent / "fixtures" / "generated_evidence"
os.environ["CLOUDIR_WORKSPACE_DIR"] = WORKSPACE.name
# Tests control skipping explicitly; ignore any local .env test setting.
os.environ["CLOUDIR_SKIP_NEXT_TURN"] = "0"

# cloudir.paths is read once at import. If another test module imported it
# first, the app would reset and clean the real project data instead.
_early_paths = sys.modules.get("cloudir.paths")
if _early_paths is not None and _early_paths.WORKSPACE_ROOT != Path(WORKSPACE.name).resolve():
    raise RuntimeError(
        f"cloudir.paths was imported before CLOUDIR_WORKSPACE_DIR was set and points at "
        f"{_early_paths.WORKSPACE_ROOT}; refusing to run workspace-mutating tests."
    )

import app as application
from cloudir.services.run_store import RunStore, atomic_json, read_json
from cloudir.scenario.scenario_state import ScenarioState


class RunManagementTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(WORKSPACE.name)
        for child in self.root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        shutil.copytree(RUNTIME_FIXTURE, self.root / "data/runtime")
        shutil.copytree(PROCESSED_FIXTURE, self.root / "data/processed")
        shutil.copytree(REPO / "data/source", self.root / "data/source")
        shutil.copytree(EVIDENCE_FIXTURE, self.root / "output/generated_evidence")
        shutil.copy2(REPO / "data/acse_eval.jsonl", self.root / "data/acse_eval.jsonl")
        self.store = application.app.extensions["run_store"]
        application.scenario_state.reset()
        self.client = application.app.test_client()
        self.patches = [
            patch("app.generate_evidence_for_turn", side_effect=self.fake_evidence),
            patch("cloudir.evidence.generate_for_turn.generate_evidence_for_turn", side_effect=self.fake_evidence),
            patch("app.run_ai_evaluation_turn", return_value=self.ai_output()),
            patch("app.stream_ai_evaluation_turn", side_effect=self.fake_stream),
            patch("cloudir.services.turn_evaluation_service.generate_and_save_next_turn", side_effect=self.fake_next_turn),
        ]
        for mock in self.patches:
            mock.start()
        self.addCleanup(lambda: [mock.stop() for mock in reversed(self.patches)])

    def fake_evidence(self, turn_number, root_dir):
        items = read_json(self.store.runtime / "turns" / f"turn_{turn_number}" / "evidence_facts.json")
        return [{**e, "imagePath": str(self.store.evidence / f"turn_{turn_number}" / f"{e['id']}.png"),
                 "imageUrl": f"/generated_evidence/turn_{turn_number}/{e['id']}.png"} for e in items]

    def ai_output(self):
        return {"vlm_output": {"visible_facts": ["test fact"]},
                "security_output": {"verdict": "Partial Support", "reasoning": "test reason"},
                "coach_output": {"feedback": "test feedback"}}

    def fake_stream(self, **kwargs):
        output = self.ai_output()
        for stage in ["vlm", "security", "coach"]:
            yield {"stage": stage, "output": output[stage + "_output"]}
        yield {"stage": "done", "output": output}

    def fake_next_turn(self, **kwargs):
        number = kwargs["next_turn_number"]
        source = RUNTIME_FIXTURE / "turns" / f"turn_{number}"
        target = self.store.runtime / "turns" / f"turn_{number}"
        shutil.copytree(source, target, dirs_exist_ok=True)
        return {"turn_config": read_json(target / "turn_config.json")}

    def start(self):
        response = self.client.post("/api/runs/start", json={"scenarioId": "identity-management", "testMode": True})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.json

    def evaluate(self, stream=False, skip_next_turn=False):
        data = self.client.get("/api/state").json
        body = {"selectedAction": data["turn"]["actions"][0], "selectedEvidence": data["generatedEvidence"][0],
                "learnerJustification": {"transcript": "The visible event supports further investigation."}}
        if skip_next_turn:
            body["skipNextTurn"] = True
        return self.client.post("/api/evaluate-stream" if stream else "/api/evaluate", json=body, buffered=stream)

    def test_legacy_is_preserved_and_new_run_is_clean(self):
        existing = (self.store.runtime / "evaluations/turn_1_evaluation.json").read_bytes()
        data = self.start()
        runs = self.store.list_runs()
        legacy = next(run for run in runs if run["status"] == "legacy")
        folder, _ = self.store.checkpoint(legacy["id"])
        self.assertEqual((folder / "runtime/evaluations/turn_1_evaluation.json").read_bytes(), existing)
        self.assertFalse(legacy["resumable"])
        self.assertFalse((self.store.runtime / "turns/turn_2").exists())
        self.assertFalse((self.store.runtime / "evaluations").exists())
        self.assertEqual(data["state"]["actionHistory"], [])

    def test_save_leave_resume_restores_exact_state_draft_and_images(self):
        data = self.start()
        run_id = data["run"]["id"]
        state = application.scenario_state
        state.visibility = 63
        draft = {"screen": "justification", "selectedAction": data["turn"]["actions"][0],
                 "justification": {"transcript": "My unfinished explanation"}}
        evidence = next(self.store.evidence.rglob("*.png"))
        original = evidence.read_bytes()
        response = self.client.post("/api/runs/leave", json={"draft": draft, "label": "Manual weak evidence test"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.store.active())
        evidence.write_bytes(b"changed")
        state.reset()
        response = self.client.post(f"/api/runs/{run_id}/resume")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["state"]["visibility"], 63)
        self.assertEqual(response.json["draft"], draft)
        self.assertEqual(evidence.read_bytes(), original)
        self.assertEqual(response.json["run"]["label"], "Manual weak evidence test")

    def test_server_restart_recovers_saved_state(self):
        self.start()
        application.scenario_state.current_turn = 1
        application.scenario_state.visibility = 71
        self.client.post("/api/runs/save", json={})
        fresh_state = ScenarioState()
        RunStore(self.root).recover_active(fresh_state)
        self.assertEqual(fresh_state.visibility, 71)

    def test_evaluation_is_saved_and_resume_does_not_evaluate_twice(self):
        run_id = self.start()["run"]["id"]
        response = self.evaluate(stream=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('"stage": "done"', response.get_data(as_text=True))
        details = self.store.details(run_id)
        self.assertEqual(details["evaluations"][0]["security_output"]["verdict"], "Partial Support")
        self.assertEqual(details["state"]["pending_turn"], 2)
        self.assertTrue(details["events"][0]["stages"])
        self.client.post("/api/runs/leave", json={})
        restored = self.client.post(f"/api/runs/{run_id}/resume").json
        self.assertEqual(restored["state"]["pendingTurn"], 2)
        self.assertIsNotNone(restored["currentEvaluation"])
        self.assertEqual(self.evaluate().status_code, 409)
        self.assertEqual(self.client.post("/api/continue").json["state"]["currentTurn"], 2)

    def test_skipping_next_turn_keeps_verdict_but_generates_nothing(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                run_id = self.start()["run"]["id"]
                with patch("cloudir.services.turn_evaluation_service.generate_and_save_next_turn") as generate:
                    response = self.evaluate(stream=stream, skip_next_turn=True)
                    self.assertEqual(response.status_code, 200)
                    generate.assert_not_called()
                text = response.get_data(as_text=True)
                self.assertNotIn("next_turn_generation", text)
                self.assertRegex(text, r'"skipped":\s*true')
                details = self.store.details(run_id)
                self.assertEqual(details["evaluations"][0]["security_output"]["verdict"], "Partial Support")
                self.assertIsNone(details["state"]["pending_turn"])
                self.assertEqual(self.client.post("/api/continue").status_code, 409)

    def test_env_setting_skips_next_turn_without_browser_flag(self):
        run_id = self.start()["run"]["id"]
        with patch.dict(os.environ, {"CLOUDIR_SKIP_NEXT_TURN": "1"}), \
             patch("cloudir.services.turn_evaluation_service.generate_and_save_next_turn") as generate:
            response = self.evaluate(stream=True)
            self.assertEqual(response.status_code, 200)
            generate.assert_not_called()
        self.assertIsNone(self.store.details(run_id)["state"]["pending_turn"])

    def test_repeating_before_turn_creates_independent_attempt(self):
        original = self.start()["run"]["id"]
        self.assertEqual(self.evaluate().status_code, 200)
        details = self.store.details(original)
        checkpoint = next(cp for cp in details["checkpoints"] if cp["kind"] == "before_turn")
        response = self.client.post(f"/api/runs/{original}/repeat", json={"checkpointId": checkpoint["id"]})
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(response.json["run"]["id"], original)
        self.assertEqual(response.json["state"]["actionHistory"], [])
        self.assertEqual(response.json["run"]["parent"]["runId"], original)
        self.assertEqual(len(self.store.details(original)["evaluations"]), 1)

    def test_reset_archives_instead_of_deleting_previous_run(self):
        original = self.start()["run"]["id"]
        self.evaluate()
        self.assertEqual(self.client.post("/api/reset").status_code, 200)
        self.assertNotEqual(self.store.active()["id"], original)
        self.assertEqual(len(self.store.details(original)["evaluations"]), 1)

    def test_other_scenario_cleanup_does_not_clear_active_run(self):
        original = self.start()["run"]["id"]
        self.evaluate()
        before = asdict(application.scenario_state)
        response = self.client.post("/api/dataset/flush-generated", json={"confirm": "flush-generated", "scope": "scenario", "scenarioId": "cost-management"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.active()["id"], original)
        self.assertEqual(asdict(application.scenario_state), before)
        self.assertTrue((self.store.runtime / "evaluations/turn_1_evaluation.json").exists())

    def test_all_cleanup_preserves_saved_runs_and_resume(self):
        original = self.start()["run"]["id"]
        self.evaluate()
        response = self.client.post("/api/dataset/flush-generated", json={"confirm": "flush-generated", "scope": "all"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.store.active())
        self.assertEqual(len(self.store.details(original)["evaluations"]), 1)
        self.assertEqual(self.client.post(f"/api/runs/{original}/resume").status_code, 200)

    def test_readiness_rejects_other_scenario_runtime(self):
        self.start()
        status = self.client.get("/api/dataset/status?scenario=cost-management").json
        self.assertFalse(status["runtimeMatchesScenario"])
        self.assertFalse(status["prepared"])

    def test_failed_generation_preserves_failure_and_rolls_back_progress(self):
        original = self.start()["run"]["id"]
        with patch("cloudir.services.turn_evaluation_service.generate_and_save_next_turn", side_effect=ValueError("test generation failure")):
            response = self.evaluate(stream=True)
        self.assertIn("test generation failure", response.get_data(as_text=True))
        self.assertEqual(application.scenario_state.action_history, [])
        self.assertIsNone(application.scenario_state.pending_turn)
        details = self.store.details(original)
        self.assertIn("failed_operation", [cp["kind"] for cp in details["checkpoints"]])
        self.assertEqual(details["run"]["status"], "interrupted")
        self.assertIn("test generation failure", details["events"][0]["error"])
        self.assertEqual(self.evaluate().status_code, 200)

    def test_extraction_failure_cannot_award_progress_in_either_route(self):
        from cloudir.ai_models import run_turn
        for stream in (False, True):
            with self.subTest(stream=stream):
                original = self.start()["run"]["id"]
                before = asdict(application.scenario_state)
                with patch("app.stream_ai_evaluation_turn", side_effect=run_turn.stream_ai_evaluation_turn), \
                     patch("app.run_ai_evaluation_turn", side_effect=run_turn.run_ai_evaluation_turn), \
                     patch.object(run_turn, "analyse_evidence_image", return_value={}) as extract, \
                     patch.object(run_turn, "evaluate_action_evidence") as judge, \
                     patch.object(run_turn, "generate_coach_feedback") as coach, \
                     patch.object(run_turn, "unload_image_model"), \
                     patch.object(run_turn, "unload_all_models"):
                    response = self.evaluate(stream=stream)
                self.assertIn("Evidence could not be read reliably", response.get_data(as_text=True))
                self.assertEqual(extract.call_count, 2)
                judge.assert_not_called()
                coach.assert_not_called()
                self.assertEqual(asdict(application.scenario_state), before)
                self.assertIn("failed_operation", [cp["kind"] for cp in self.store.details(original)["checkpoints"]])

    def test_feedback_grounding_failure_cannot_award_progress_in_either_route(self):
        from cloudir.ai_models import run_turn
        for stream in (False, True):
            for failing_stage in ("security", "coach"):
                with self.subTest(stream=stream, stage=failing_stage):
                    original = self.start()["run"]["id"]
                    before = asdict(application.scenario_state)
                    error = ValueError("Feedback failed grounding checks after retry; no progress awarded.")
                    with patch("app.stream_ai_evaluation_turn", side_effect=run_turn.stream_ai_evaluation_turn), \
                         patch("app.run_ai_evaluation_turn", side_effect=run_turn.run_ai_evaluation_turn), \
                         patch.object(run_turn, "analyse_evidence_image", return_value={
                             "visible_facts_extracted": ["An audit event"], "matched_records": 1,
                             "event_rows": [{"message": "An audit event", "truncated": False}]}), \
                         patch.object(run_turn, "evaluate_action_evidence", return_value={"verdict": "Strong Support"},
                                      side_effect=error if failing_stage == "security" else None), \
                         patch.object(run_turn, "generate_coach_feedback", side_effect=error) as coach, \
                         patch.object(run_turn, "unload_image_model"), \
                         patch.object(run_turn, "unload_security_model"), \
                         patch.object(run_turn, "unload_coach_model"), \
                         patch.object(run_turn, "unload_all_models"):
                        response = self.evaluate(stream=stream)
                    self.assertIn("Feedback failed grounding checks", response.get_data(as_text=True))
                    if failing_stage == "security":
                        coach.assert_not_called()
                    else:
                        coach.assert_called_once()
                    self.assertEqual(asdict(application.scenario_state), before)
                    self.assertIn("failed_operation", [cp["kind"] for cp in self.store.details(original)["checkpoints"]])

    def test_busy_lock_blocks_reset_and_save(self):
        self.start()
        lock = application.app.extensions["run_lock"]
        lock.acquire()
        try:
            self.assertEqual(self.client.post("/api/reset").status_code, 409)
            self.assertEqual(self.client.post("/api/runs/save", json={}).status_code, 409)
        finally:
            lock.release()
        self.assertEqual(self.client.post("/api/runs/save", json={}).status_code, 200)

    def test_stale_tab_and_invalid_ids_are_rejected(self):
        self.start()
        self.assertEqual(self.client.post("/api/runs/save", json={}, headers={"X-Run-ID": "old-run"}).status_code, 409)
        self.assertEqual(self.client.get("/api/runs/not-a-run").status_code, 404)
        self.assertEqual(self.client.post("/api/runs/not-a-run/resume").status_code, 400)

    def test_export_contains_state_evidence_and_configuration(self):
        run_id = self.start()["run"]["id"]
        self.evaluate()
        response = self.client.get(f"/api/runs/{run_id}/export")
        self.assertEqual(response.status_code, 200)
        with ZipFile(BytesIO(response.data)) as archive:
            names = archive.namelist()
            self.assertIn("run.json", names)
            self.assertTrue(any(name.endswith("checkpoint.json") for name in names))
            self.assertTrue(any(name.endswith("scenario_config.json") for name in names))
            self.assertTrue(any(name.endswith(".png") for name in names))

    def test_corrupt_snapshot_is_rejected_without_changing_workspace(self):
        run_id = self.start()["run"]["id"]
        folder, _ = self.store.checkpoint(run_id)
        (folder / "runtime/scenario_config.json").write_text("{}")
        original = (self.store.runtime / "scenario_config.json").read_bytes()
        self.store.detach()
        self.assertEqual(self.client.post(f"/api/runs/{run_id}/resume").status_code, 400)
        self.assertEqual((self.store.runtime / "scenario_config.json").read_bytes(), original)

    def test_stream_holds_lock_until_completion_and_recovers_on_disconnect(self):
        run_id = self.start()["run"]["id"]
        data = self.client.get("/api/state").json
        response = self.client.post("/api/evaluate-stream", json={
            "selectedAction": data["turn"]["actions"][0], "selectedEvidence": data["generatedEvidence"][0],
        }, buffered=False)
        self.assertTrue(application.app.extensions["run_lock"].locked())
        self.assertEqual(self.client.post("/api/runs/leave", json={}).status_code, 409)
        response.close()
        self.assertFalse(application.app.extensions["run_lock"].locked())
        self.assertEqual(self.store.get(run_id)["status"], "interrupted")
        self.assertEqual(application.scenario_state.action_history, [])

    def test_prepare_archives_active_run_and_rebuild_removes_later_turns(self):
        run_id = self.start()["run"]["id"]
        self.evaluate()
        from cloudir.dataset_preparation.build_acse import _reset_turn_1_outputs

        def prepare(scenario_id):
            _reset_turn_1_outputs()
            shutil.copytree(RUNTIME_FIXTURE / "turns/turn_1", self.store.runtime / "turns/turn_1", dirs_exist_ok=True)
            yield {"stage": "complete", "status": "complete"}

        with patch("app.stream_build_acse_dataset_scenario", side_effect=prepare):
            response = self.client.post("/api/dataset/prepare-stream", json={"scenarioId": "identity-management"}, buffered=True)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.store.active())
        self.assertFalse((self.store.runtime / "turns/turn_2").exists())
        self.assertEqual(len(self.store.details(run_id)["evaluations"]), 1)

    def build(self, scenario_id, fail=False):
        """Stand-in for preparation: writes the scenario's Turn 1 into the shared workspace."""
        from cloudir.dataset_preparation.build_acse import _reset_turn_1_outputs

        def prepare(_):
            _reset_turn_1_outputs()
            config = read_json(RUNTIME_FIXTURE / "scenario_config.json")
            atomic_json(self.store.runtime / "scenario_config.json", {**config, "scenario_id": scenario_id})
            if fail:
                yield {"stage": "error", "status": "error", "message": "Invalid incident timeline"}
                return
            shutil.copytree(RUNTIME_FIXTURE / "turns/turn_1", self.store.runtime / "turns/turn_1", dirs_exist_ok=True)
            (self.store.evidence / "turn_1" / f"{scenario_id}.png").write_bytes(b"png")
            yield {"stage": "complete", "status": "complete"}

        with patch("app.stream_build_acse_dataset_scenario", side_effect=prepare):
            response = self.client.post("/api/dataset/prepare-stream", json={"scenarioId": scenario_id}, buffered=True)
        self.assertEqual(response.status_code, 200)

    def test_switching_scenarios_does_not_need_a_rebuild(self):
        self.build("identity-management")
        self.build("automated-security-response")  # overwrites the shared workspace
        status = self.client.get("/api/dataset/status?scenario=identity-management").json
        self.assertTrue(status["prepared"] and status["savedCopy"])

        run = self.start()["run"]
        self.assertEqual(run["scenarioId"], "identity-management")
        self.assertTrue((self.store.evidence / "turn_1/identity-management.png").exists())
        self.evaluate()

        response = self.client.post("/api/runs/start", json={"scenarioId": "automated-security-response", "testMode": True})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(response.json["run"]["scenarioId"], "automated-security-response")
        self.assertEqual(read_json(self.store.runtime / "scenario_config.json")["scenario_id"], "automated-security-response")
        self.assertFalse((self.store.runtime / "evaluations/turn_1_evaluation.json").exists())
        # The identity attempt was saved before the workspace switched.
        self.assertEqual(len(self.store.details(run["id"])["evaluations"]), 1)

    def test_build_made_before_copies_existed_survives_another_build(self):
        # The fixture workspace is an identity build from before copies were kept.
        self.build("automated-security-response")
        self.assertTrue(self.client.get("/api/dataset/status?scenario=identity-management").json["prepared"])
        self.assertEqual(self.start()["run"]["scenarioId"], "identity-management")
        self.assertFalse((self.store.runtime / "turns/turn_2").exists())

    def test_failed_rebuild_keeps_the_last_working_build(self):
        self.build("identity-management")
        self.build("identity-management", fail=True)
        self.assertFalse((self.store.runtime / "turns/turn_1/turn_config.json").exists())
        self.assertTrue(self.client.get("/api/dataset/status?scenario=identity-management").json["prepared"])
        self.start()
        self.assertTrue((self.store.runtime / "turns/turn_1/turn_config.json").exists())

    def test_clean_restart_removes_the_saved_build(self):
        self.build("identity-management")
        self.build("automated-security-response")
        response = self.client.post("/api/dataset/flush-generated", json={
            "confirm": "flush-generated", "scope": "scenario", "scenarioId": "identity-management"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["status"]["prepared"])
        self.assertEqual(self.client.post("/api/runs/start", json={"scenarioId": "identity-management"}).status_code, 409)
        self.assertTrue(self.client.get("/api/dataset/status?scenario=automated-security-response").json["prepared"])

        self.client.post("/api/dataset/flush-generated", json={"confirm": "flush-generated", "scope": "all"})
        self.assertFalse(any((self.root / "data/prepared").iterdir()))

    def test_completed_run_restores_as_completed(self):
        run_id = self.start()["run"]["id"]
        application.scenario_state.max_turns = 1
        with patch("cloudir.services.turn_evaluation_service.generate_and_save_final_debrief", return_value={"feedback": "done"}):
            self.assertEqual(self.evaluate().status_code, 200)
        self.assertTrue(application.scenario_state.completed)
        self.client.post("/api/runs/leave", json={})
        response = self.client.post(f"/api/runs/{run_id}/resume")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["state"]["completed"])
        self.assertEqual(response.json["run"]["status"], "completed")

    def test_paused_run_does_not_become_duplicate_legacy_on_next_start(self):
        self.start()
        self.evaluate()
        self.client.post("/api/runs/leave", json={})
        legacy_count = sum(r["status"] == "legacy" for r in self.store.list_runs())
        self.start()
        self.assertEqual(sum(r["status"] == "legacy" for r in self.store.list_runs()), legacy_count)

    def test_run_controls_are_served(self):
        with self.client.get("/") as response:
            self.assertIn(b"Save &amp; Leave", response.data)
        with self.client.get("/run_controls.js") as response:
            self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
