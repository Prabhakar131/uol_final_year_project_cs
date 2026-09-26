"""Run management APIs and exclusive access to the single local AI workspace."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
import time

from flask import g, jsonify, request, send_file

from cloudir.scenario.scenario_state import ScenarioState
from cloudir.services.run_store import RunStore, read_json


def register_run_routes(app, state: ScenarioState, root: Path) -> None:
    store = RunStore(root)
    operation_lock = Lock()
    app.extensions["run_store"] = store
    app.extensions["run_lock"] = operation_lock
    store.recover_active(state)
    try:
        store.keep_workspace_build()
    except (OSError, ValueError) as exc:
        print(f"[runs] could not keep a copy of the workspace scenario: {exc}")

    def payload():
        from cloudir.scenario.turn_payload_loader import load_turn_payload
        from cloudir.evidence.generate_for_turn import generate_evidence_for_turn
        meta = store.active()
        draft = store.checkpoint(meta["id"])[1].get("draft", {}) if meta else {}
        trace = read_json(store.runtime / "evaluations" / f"turn_{state.current_turn}_evaluation.json")
        return {"state": state.to_dict(), "run": meta, "draft": draft, "currentEvaluation": trace,
                "turn": load_turn_payload(state.current_turn, root),
                "generatedEvidence": generate_evidence_for_turn(state.current_turn, root)}

    @app.before_request
    def lock_workspace():
        g.run_operation = None
        path = request.path
        protected = path.startswith("/api/runs") or path in {
            "/api/state", "/api/evaluate", "/api/evaluate-stream", "/api/continue", "/api/reset",
            "/api/dataset/prepare-stream", "/api/dataset/flush-generated", "/api/transcribe-justification",
        }
        if not protected:
            return None
        if not operation_lock.acquire(blocking=False):
            return jsonify(error="An operation is still running. Wait for it to finish before saving, leaving or switching runs.", busy=True), 409
        op = {"released": False, "path": path, "started": time.perf_counter(), "before": None,
              "events": [], "error": None, "success": False}
        g.run_operation = op
        try:
            active = store.active()
            expected = request.headers.get("X-Run-ID")
            if expected and path in {"/api/state", "/api/evaluate", "/api/evaluate-stream", "/api/continue", "/api/reset",
                                     "/api/runs/save", "/api/runs/leave"} and (not active or expected != active["id"]):
                return jsonify(error="The active run changed. Resume the intended run from Saved Runs."), 409
            if path in {"/api/state", "/api/evaluate", "/api/evaluate-stream"} and not active:
                if not (store.runtime / "turns" / "turn_1" / "turn_config.json").exists():
                    return jsonify(error="Prepare a scenario first.", needs_dataset_preparation=True), 409
                store.fresh_start(state, test_mode=request.headers.get("X-Test-Mode") == "1")
                active = store.active()
            if path in {"/api/evaluate", "/api/evaluate-stream"}:
                if state.completed or state.has_recorded_turn(state.current_turn):
                    return jsonify(error="This turn is already evaluated. Continue or create a new attempt from its checkpoint."), 409
                body = request.get_json(silent=True) or {}
                if body.get("selectedAction") is None or body.get("selectedEvidence") is None:
                    return jsonify(error="Select an action and evidence first."), 400
                draft = {"screen": "justification", "selectedAction": body["selectedAction"],
                         "selectedEvidence": body["selectedEvidence"], "justification": body.get("learnerJustification", {})}
                saved = store.save(state, kind="before_turn", draft=draft)
                op["before"] = (active["id"], saved["checkpoint"]["id"])
            elif path in {"/api/continue", "/api/dataset/prepare-stream", "/api/dataset/flush-generated"}:
                if active:
                    saved = store.save(state, kind="before_operation",
                                       status="paused" if "prepare" in path and not state.completed else None)
                    op["before"] = (active["id"], saved["checkpoint"]["id"])
                else:
                    store.archive_legacy(state)
                if path == "/api/dataset/prepare-stream":
                    store.keep_workspace_build()  # a build replaces the workspace
            if path == "/api/continue" and state.pending_turn is None and not state.completed:
                return jsonify(error="Evaluate this turn before continuing."), 409
        except (ValueError, OSError) as exc:
            op["error"] = str(exc)
            return jsonify(error=str(exc)), 400

    def finish_operation(op):
        if op["released"]:
            return
        try:
            path = op["path"]
            mutates = path in {"/api/evaluate", "/api/evaluate-stream", "/api/continue"}
            if op["before"] and op["error"]:
                store.record_event({"operation": path, "error": op["error"], "stages": op["events"],
                                    "elapsedSeconds": round(time.perf_counter() - op["started"], 3)})
                store.save(state, kind="failed_operation", status="failed")
                run_id, checkpoint_id = op["before"]
                record = store.restore(run_id, state, checkpoint_id)
                store.save(state, kind="recovered", draft=record["draft"], status="interrupted")
            elif op["success"]:
                if mutates and store.active():
                    store.record_event({"operation": path, "stages": op["events"],
                                        "elapsedSeconds": round(time.perf_counter() - op["started"], 3)})
                    store.save(state, kind="after_turn" if "evaluate" in path else "turn_ready", draft={})
                elif path == "/api/dataset/prepare-stream":
                    store.detach()
                elif path == "/api/dataset/flush-generated":
                    # A cleanup of another scenario must leave the active run attached.
                    config = read_json(store.runtime / "scenario_config.json", {})
                    active = store.active()
                    if active and config.get("scenario_id") != active["scenarioId"]:
                        store.detach()
        finally:
            op["released"] = True
            operation_lock.release()

    @app.after_request
    def checkpoint_response(response):
        op = getattr(g, "run_operation", None)
        if not op:
            return response
        if response.mimetype == "text/event-stream":
            original = response.response

            def stream():
                ended = False
                try:
                    for chunk in original:
                        text = chunk.decode() if isinstance(chunk, bytes) else chunk
                        for line in text.splitlines():
                            if not line.startswith("data: "):
                                continue
                            event = json.loads(line[6:])
                            op["events"].append({"elapsedSeconds": round(time.perf_counter() - op["started"], 3), **event})
                            if event.get("stage") == "error" or event.get("status") == "error":
                                op["error"] = event.get("message") or event.get("output", {}).get("error") or "Operation failed"
                            if event.get("stage") == "done" or event.get("status") == "complete":
                                op["success"] = True
                        yield chunk
                    ended = True
                except Exception as exc:
                    op["error"] = str(exc)
                    raise
                finally:
                    if not ended and not op["success"]:
                        op["error"] = "Operation interrupted before completion."
                    try:
                        if hasattr(original, "close"):
                            original.close()
                    finally:
                        finish_operation(op)

            response.response = stream()
            response.call_on_close(lambda: finish_operation(op))
        else:
            op["success"] = response.status_code < 400
            if response.status_code >= 500:
                op["error"] = (response.get_json(silent=True) or {}).get("error", "Request failed")
            if op["path"] == "/api/state" and response.status_code == 200:
                data = response.get_json()
                active = store.active()
                data.update(run=active, draft=store.checkpoint(active["id"])[1].get("draft", {}) if active else {},
                            currentEvaluation=read_json(store.runtime / "evaluations" / f"turn_{state.current_turn}_evaluation.json"))
                response.set_data(json.dumps(data))
            finish_operation(op)
        return response

    @app.teardown_request
    def unlock_on_exception(error):
        op = getattr(g, "run_operation", None)
        if error and op and not op["released"]:
            op["error"] = str(error) or type(error).__name__
            finish_operation(op)

    @app.get("/api/runs")
    def list_runs():
        return jsonify(runs=store.list_runs(), activeRun=store.active())

    @app.get("/api/runs/<run_id>")
    def get_run(run_id):
        try:
            return jsonify(store.details(run_id))
        except ValueError as exc:
            return jsonify(error=str(exc)), 404

    @app.post("/api/runs/start")
    def start_run():
        body = request.get_json(silent=True) or {}
        scenario_id = str(body.get("scenarioId") or "")
        config = read_json(store.runtime / "scenario_config.json", {})
        if config.get("scenario_id") != scenario_id and not store.has_prepared(scenario_id):
            return jsonify(error="The selected scenario must be prepared before starting it."), 409
        store.fresh_start(state, label=str(body.get("label") or ""), test_mode=bool(body.get("testMode")),
                          scenario_id=scenario_id)
        return jsonify(payload())

    @app.post("/api/runs/save")
    @app.post("/api/runs/leave")
    def save_run():
        if not store.active():
            return jsonify(error="There is no active run to save."), 409
        body = request.get_json(silent=True) or {}
        leaving = request.path.endswith("/leave")
        result = store.save(state, kind="paused" if leaving else "saved", draft=body.get("draft", {}),
                            label=body.get("label"), note=str(body.get("note") or ""),
                            status="completed" if state.completed else "paused" if leaving else None)
        if leaving:
            store.detach()
        return jsonify(result)

    @app.post("/api/runs/<run_id>/resume")
    def resume_run(run_id):
        try:
            store.get(run_id)
            if store.active():
                store.save(state, kind="before_switch", status="completed" if state.completed else "paused")
            record = store.restore(run_id, state)
            store.save(state, kind="resumed", draft=record["draft"])
            return jsonify(payload())
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.post("/api/runs/<run_id>/repeat")
    def repeat_run(run_id):
        body = request.get_json(silent=True) or {}
        try:
            _, record = store.checkpoint(run_id, body.get("checkpointId"))
            if record["kind"] not in {"initial", "before_turn"}:
                return jsonify(error="Choose a starting-point or before-turn checkpoint."), 400
            if store.active():
                store.save(state, kind="before_switch", status="completed" if state.completed else "paused")
            store.restore(run_id, state, record["id"])
            store.create(state, label=str(body.get("label") or "Repeated test"),
                         parent={"runId": run_id, "checkpointId": record["id"]}, test_mode=True)
            return jsonify(payload())
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.get("/api/runs/<run_id>/export")
    def export_run(run_id):
        try:
            return send_file(store.export(run_id), mimetype="application/zip", as_attachment=True,
                             download_name=f"{run_id}.zip")
        except ValueError as exc:
            return jsonify(error=str(exc)), 404
