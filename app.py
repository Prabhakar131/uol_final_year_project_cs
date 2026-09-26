from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from werkzeug.utils import secure_filename

from cloudir.ai_models.run_turn import run_ai_evaluation_turn, stream_ai_evaluation_turn
from cloudir.ai_models.voice_model import transcribe_learner_justification, unload_voice_model
from cloudir.services.dataset_status import get_dataset_status, scenario_slug
from cloudir.dataset_preparation.select_case import source_assets_dir
from cloudir.services.evaluation_trace_service import clear_evaluation_traces
from cloudir.services.final_debrief_service import get_final_debrief
from cloudir.services.generated_artifact_cleanup import flush_generated_artifacts
from cloudir.services.turn_evaluation_service import finish_evaluation
from cloudir.evidence.generate_for_turn import generate_evidence_for_turn
from cloudir.scenario.turn_orchestrator import turn_files_exist
from cloudir.scenario.scenario_state import ScenarioState
from cloudir.scenario.turn_payload_loader import load_turn_payload

from cloudir.dataset_preparation import stream_build_acse_dataset_scenario
from cloudir.paths import FRONTEND_DIR, GENERATED_EVIDENCE_DIR, RUNTIME_DATA_DIR, WORKSPACE_ROOT, SOURCE_DATA_DIR


ROOT_DIR = WORKSPACE_ROOT
UI_DIR = FRONTEND_DIR
DATASET_SOURCE_ASSETS_DIR = SOURCE_DATA_DIR
DEFAULT_DATASET_SCENARIO = "identity-management"


def should_skip_next_turn(data: dict[str, Any]) -> bool:
    """
    Test-only switch: evaluate a turn without generating the next one.

    CLOUDIR_SKIP_NEXT_TURN=1 in .env enables it on the server regardless of the
    browser; the Test Mode "Skip next turn" button can also request it.
    """

    from_env = os.getenv("CLOUDIR_SKIP_NEXT_TURN", "").strip() == "1"
    from_browser = data.get("skipNextTurn") is True
    skip = from_env or from_browser
    source = "env" if from_env else "browser" if from_browser else "off"
    print(f"[next-turn] skip={'YES' if skip else 'NO'} (source={source})", flush=True)
    return skip

app = Flask(
    __name__,
    static_folder=str(UI_DIR),
    static_url_path="",
)

scenario_state = ScenarioState()

ALLOWED_AUDIO_EXTENSIONS = {
    "wav",
    "mp3",
    "m4a",
    "mp4",
    "mpeg",
    "mpga",
    "webm",
    "ogg",
    "oga",
    "flac",
}


def sync_state_with_runtime_config() -> None:
    config_path = ROOT_DIR / "data" / "runtime" / "scenario_config.json"

    if not config_path.exists():
        return

    try:
        with config_path.open("r", encoding="utf-8") as file:
            scenario_config = json.load(file)
    except (OSError, json.JSONDecodeError):
        return

    scenario_state.apply_max_turns(scenario_config.get("total_turns"))


@app.route("/")
def index():
    return send_from_directory(UI_DIR, "index.html")


@app.route("/style.css")
def serve_css():
    return send_from_directory(UI_DIR, "style.css")


@app.route("/script.js")
def serve_js():
    return send_from_directory(UI_DIR, "script.js")


@app.route("/generated_evidence/<path:filename>")
def serve_generated_evidence(filename: str):
    response = send_from_directory(GENERATED_EVIDENCE_DIR, filename)
    # The same working URL can belong to a different restored run.
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/dataset-assets/<scenario>/<path:filename>")
def serve_dataset_asset(scenario: str, filename: str):
    """
    Serves ACSE source assets to the frontend.

    The architecture diagram is dataset context only.
    It is not learner-selected evidence.
    """

    try:
        slug = scenario_slug(scenario)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404

    allowed_files = {
        "architecture.png",
        "threat-model.json",
    }

    if filename not in allowed_files:
        return jsonify({"error": f"Dataset asset is not allowed: {filename}"}), 404

    asset_dir = source_assets_dir(scenario)

    return send_from_directory(asset_dir, filename)


@app.route("/api/dataset/status", methods=["GET"])
def dataset_status():
    """
    Checks whether the ACSE source files and prepared CloudIR runtime files exist.
    """

    scenario_id = request.args.get("scenario", DEFAULT_DATASET_SCENARIO)

    try:
        status = get_dataset_status(root_dir=ROOT_DIR, scenario_id=scenario_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(status)


@app.route("/api/transcribe-justification", methods=["POST"])
def transcribe_justification():
    """
    Transcribes a learner's spoken justification.

    The frontend should upload multipart/form-data with an "audio" file.
    The transcript is later passed into the AI evaluation pipeline as learner
    reasoning, while action and evidence selection remain explicit UI choices.
    """

    audio_file = request.files.get("audio")

    if audio_file is None:
        return jsonify({"error": "audio file is required"}), 400

    original_name = secure_filename(audio_file.filename or "justification.webm")
    extension = Path(original_name).suffix.lower().lstrip(".")

    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify(
            {
                "error": (
                    "Unsupported audio file type. "
                    f"Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
                )
            }
        ), 400

    with tempfile.TemporaryDirectory(prefix="cloudir_voice_") as temp_dir:
        audio_path = Path(temp_dir) / original_name
        audio_file.save(audio_path)

        try:
            transcript = transcribe_learner_justification(audio_path)
        except Exception as exc:
            return jsonify(
                {
                    "error": str(exc),
                    "errorType": type(exc).__name__,
                }
            ), 500
        finally:
            unload_voice_model()

    return jsonify(transcript)


@app.route("/api/dataset/flush-generated", methods=["POST"])
def flush_generated_dataset_files():
    """
    Deletes prepared/generated artifacts so the dataset can be rebuilt cleanly.

    This intentionally preserves source files:
    - data/acse_eval.jsonl
    - data/source/<scenario>/
    """

    data = request.get_json(silent=True) or {}

    if data.get("confirm") != "flush-generated":
        return jsonify(
            {
                "error": "Confirmation value is required to flush generated files.",
                "requiredConfirm": "flush-generated",
            }
        ), 400

    scope = str(data.get("scope") or "all")
    scenario_id = str(data.get("scenarioId") or DEFAULT_DATASET_SCENARIO)

    result = flush_generated_artifacts(
        scope=scope,
        scenario_id=scenario_id,
    )
    if scope == "all" or result.get("runtimeCleared"):
        scenario_state.reset()
        clear_evaluation_traces()

    status = get_dataset_status(
        root_dir=ROOT_DIR,
        scenario_id=scenario_id,
    )

    return jsonify(
        {
            "message": "Generated CloudIR artifacts were flushed.",
            "result": result,
            "status": status,
        }
    )


@app.route("/api/dataset/prepare-stream", methods=["POST"])
def prepare_dataset_stream():
    """
    Streams the ACSE dataset preparation pipeline stage by stage.

    This route performs:
    1. ACSE case selection
    2. Architecture VLM parsing
    3. Threat model normalisation with security text model
    4. Runtime scenario file generation
    5. Coach AI Turn 1 generation
    6. Evidence screenshot rendering
    """

    data = request.get_json(silent=True) or {}
    scenario_id = data.get("scenarioId", DEFAULT_DATASET_SCENARIO)

    def send_event(event: dict[str, Any]) -> str:
        return f"data: {json.dumps(event)}\n\n"

    def generate_stream():
        try:
            for event in stream_build_acse_dataset_scenario(scenario_id):
                if event.get("status") == "complete":
                    scenario_state.reset()
                    sync_state_with_runtime_config()
                    clear_evaluation_traces()
                    try:
                        app.extensions["run_store"].save_prepared()
                    except (OSError, ValueError) as exc:
                        # The build still succeeded; only switching back later will need a rebuild.
                        print(f"[prepare] could not keep a copy of the prepared scenario: {exc}")

                yield send_event(event)

        except Exception as exc:
            yield send_event(
                {
                    "stage": "error",
                    "label": "Dataset preparation failed",
                    "status": "error",
                    "kind": "error",
                    "message": str(exc),
                    "output": {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                }
            )

    return Response(
        stream_with_context(generate_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/state", methods=["GET"])
def get_state():
    """
    Loads the current turn.

    Important:
    Turn 1 is no longer generated secretly here.
    The dataset preparation phase must create Turn 1 before runtime starts.
    """

    turn_number = scenario_state.current_turn

    if not turn_files_exist(root_dir=ROOT_DIR, turn_number=turn_number):
        return jsonify(
            {
                "error": (
                    f"Turn {turn_number} files do not exist. "
                    "Run dataset preparation before starting the scenario."
                ),
                "needs_dataset_preparation": turn_number == 1,
                "state": scenario_state.to_dict(),
            }
        ), 409

    turn_payload = load_turn_payload(
        turn_number=turn_number,
        root_dir=ROOT_DIR,
    )

    generated_evidence = generate_evidence_for_turn(
        turn_number=turn_number,
        root_dir=ROOT_DIR,
    )

    return jsonify(
        {
            "state": scenario_state.to_dict(),
            "turn": turn_payload,
            "generatedEvidence": generated_evidence,
        }
    )


@app.route("/api/final-debrief", methods=["GET"])
def final_debrief():
    """
    Returns the saved Coach AI final debrief for the completed scenario.

    If the debrief has not been generated yet, a fallback summary based on any
    saved evaluation traces is returned so the end screen can still render.
    """

    return jsonify(
        {
            "debrief": get_final_debrief(scenario_state.to_dict()),
        }
    )


@app.route("/api/incident-timeline", methods=["GET"])
def incident_timeline():
    """
    Returns the incident timeline for the final report's indicators and event log.

    The timeline shows which events were suspicious, so it stays hidden until
    the scenario is complete.
    """

    if not scenario_state.completed:
        return jsonify({"error": "The incident timeline is available once the scenario is complete."}), 403

    timeline_path = RUNTIME_DATA_DIR / "incident_timeline.json"

    try:
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return jsonify({"timeline": None})

    return jsonify({"timeline": report_timeline(timeline)})


def report_timeline(timeline: dict[str, Any]) -> dict[str, Any]:
    """Keeps only the timeline fields the final report displays."""

    def pick(item: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
        return {key: item.get(key) for key in keys}

    return {
        "account": timeline.get("account_id"),
        "region": timeline.get("region"),
        "cost": timeline.get("cost"),
        "actors": [
            pick(actor, ("id", "role", "principal", "source_ip", "user_agent", "mfa"))
            for actor in timeline.get("actors", [])
        ],
        "events": [
            pick(event, ("time", "actor", "event_name", "event_source", "result", "source_ip", "suspicious", "baseline", "turn"))
            for event in timeline.get("events", [])
        ],
        "findings": [
            pick(finding, ("finding_type", "severity", "resource", "description", "turn"))
            for finding in timeline.get("findings", [])
        ],
    }


@app.route("/api/evaluate", methods=["POST"])
def evaluate_turn():
    """
    Runs the full local AI pipeline for one completed learner turn:

    1. VLM analyses selected generated evidence image.
    2. Security text AI evaluates action-evidence support.
    3. Coach AI generates learner-facing feedback.
    4. Coach AI generates the next turn JSON in the background.

    This is the original non-streaming route.
    It returns everything together at the end.
    """

    data = request.get_json(force=True)

    selected_action = data.get("selectedAction")
    selected_evidence = data.get("selectedEvidence")
    learner_justification = data.get("learnerJustification") or {}
    skip_next_turn = should_skip_next_turn(data)

    if selected_action is None:
        return jsonify({"error": "selectedAction is required"}), 400

    if selected_evidence is None:
        return jsonify({"error": "selectedEvidence is required"}), 400

    current_turn_number = scenario_state.current_turn

    if not turn_files_exist(root_dir=ROOT_DIR, turn_number=current_turn_number):
        return jsonify(
            {
                "error": (
                    f"Turn {current_turn_number} files do not exist. "
                    "Run dataset preparation before evaluating a turn."
                ),
                "needs_dataset_preparation": current_turn_number == 1,
            }
        ), 409

    completed_turn = load_turn_payload(
        turn_number=current_turn_number,
        root_dir=ROOT_DIR,
    )
    selected_action = refresh_selected_action_from_turn(
        selected_action=selected_action,
        completed_turn=completed_turn,
    )
    selected_evidence = refresh_selected_evidence_from_turn(
        selected_evidence=selected_evidence,
        turn_number=current_turn_number,
    )

    try:
        ai_result = run_ai_evaluation_turn(
            selected_action=selected_action,
            selected_evidence=selected_evidence,
            scenario_state=scenario_state.to_dict(),
            root_dir=ROOT_DIR,
            learner_justification=learner_justification,
        )
    except RuntimeError as exc:
        # Preserve a readable pipeline failure for non-streaming clients too.
        # The run checkpoint middleware records this and restores the attempt.
        return jsonify(error=str(exc)), 500

    response_payload = finish_evaluation(
        root_dir=ROOT_DIR,
        scenario_state=scenario_state,
        current_turn_number=current_turn_number,
        completed_turn=completed_turn,
        selected_action=selected_action,
        selected_evidence=selected_evidence,
        learner_justification=learner_justification,
        vlm_output=ai_result["vlm_output"],
        security_output=ai_result["security_output"],
        coach_output=ai_result["coach_output"],
        skip_next_turn=skip_next_turn,
    )

    return jsonify(response_payload)


@app.route("/api/evaluate-stream", methods=["POST"])
def evaluate_turn_stream():
    """
    Streams the AI pipeline stage by stage:

    1. Sends VLM output as soon as it is ready.
    2. Sends Security model output as soon as it is ready.
    3. Sends Coach AI feedback as soon as it is ready.
    4. Then generates the next turn.
    5. Sends final done event.

    This route requires script.js to read the response stream.
    """

    data = request.get_json(force=True)

    selected_action = data.get("selectedAction")
    selected_evidence = data.get("selectedEvidence")
    learner_justification = data.get("learnerJustification") or {}
    skip_next_turn = should_skip_next_turn(data)

    if selected_action is None:
        return jsonify({"error": "selectedAction is required"}), 400

    if selected_evidence is None:
        return jsonify({"error": "selectedEvidence is required"}), 400

    current_turn_number = scenario_state.current_turn

    if not turn_files_exist(root_dir=ROOT_DIR, turn_number=current_turn_number):
        return jsonify(
            {
                "error": (
                    f"Turn {current_turn_number} files do not exist. "
                    "Run dataset preparation before evaluating a turn."
                ),
                "needs_dataset_preparation": current_turn_number == 1,
            }
        ), 409

    completed_turn = load_turn_payload(
        turn_number=current_turn_number,
        root_dir=ROOT_DIR,
    )
    selected_action = refresh_selected_action_from_turn(
        selected_action=selected_action,
        completed_turn=completed_turn,
    )
    selected_evidence = refresh_selected_evidence_from_turn(
        selected_evidence=selected_evidence,
        turn_number=current_turn_number,
    )

    def send_event(event: dict[str, Any]) -> str:
        return f"data: {json.dumps(event)}\n\n"

    def generate_stream():
        vlm_output: dict[str, Any] | None = None
        security_output: dict[str, Any] | None = None
        coach_output: dict[str, Any] | None = None

        try:
            yield send_event(
                {
                    "stage": "started",
                    "label": "AI pipeline started",
                    "output": {
                        "message": "Starting VLM analysis.",
                    },
                }
            )

            for event in stream_ai_evaluation_turn(
                selected_action=selected_action,
                selected_evidence=selected_evidence,
                learner_justification=learner_justification,
                scenario_state=scenario_state.to_dict(),
                root_dir=ROOT_DIR,
            ):
                stage = event.get("stage")

                if stage == "extraction_retry":
                    yield send_event(event)

                elif stage == "vlm":
                    vlm_output = event.get("output")
                    yield send_event(event)

                elif stage == "security":
                    security_output = event.get("output")
                    yield send_event(event)

                elif stage == "coach":
                    coach_output = event.get("output")
                    yield send_event(event)

                elif stage == "error":
                    yield send_event(event)
                    return

                elif stage == "done":
                    done_output = event.get("output", {})
                    vlm_output = done_output.get("vlm_output", vlm_output)
                    security_output = done_output.get(
                        "security_output",
                        security_output,
                    )
                    coach_output = done_output.get("coach_output", coach_output)

            if vlm_output is None:
                raise RuntimeError("VLM output was not produced.")

            if security_output is None:
                raise RuntimeError("Security model output was not produced.")

            if coach_output is None:
                raise RuntimeError("Coach output was not produced.")

            if current_turn_number >= scenario_state.max_turns:
                yield send_event(
                    {
                        "stage": "scenario_complete_generation",
                        "label": "Scenario complete",
                        "output": {
                            "message": "Final turn complete. Preparing end screen.",
                        },
                    }
                )
            elif not skip_next_turn:
                yield send_event(
                    {
                        "stage": "next_turn_generation",
                        "label": "Generating next turn",
                        "output": {
                            "message": "Coach AI is generating the next turn.",
                        },
                    }
                )

            response_payload = finish_evaluation(
                root_dir=ROOT_DIR,
                scenario_state=scenario_state,
                current_turn_number=current_turn_number,
                completed_turn=completed_turn,
                selected_action=selected_action,
                selected_evidence=selected_evidence,
                learner_justification=learner_justification,
                vlm_output=vlm_output,
                security_output=security_output,
                coach_output=coach_output,
                skip_next_turn=skip_next_turn,
            )

            next_turn_preview = response_payload["nextTurnPreview"]
            next_number = next_turn_preview.get("nextTurn")

            # Let the next-turn board show what the coach actually wrote. Only titles
            # and screenshots are sent; choice roles stay hidden.
            if next_number and not next_turn_preview.get("isComplete") and not next_turn_preview.get("skipped"):
                try:
                    next_payload = load_turn_payload(turn_number=next_number, root_dir=ROOT_DIR)
                    next_evidence = generate_evidence_for_turn(turn_number=next_number, root_dir=ROOT_DIR)
                    next_turn_preview["content"] = {
                        "actions": [action.get("title", "") for action in next_payload.get("actions", [])],
                        "evidence": [
                            {"title": item.get("title", ""), "imageUrl": item.get("imageUrl", "")}
                            for item in next_evidence
                        ],
                    }
                except Exception as exc:  # the board falls back to its outline
                    print(f"[next-turn] preview content unavailable: {exc}", flush=True)

            yield send_event(
                {
                    "stage": "next_turn_preview",
                    "label": "Next turn ready",
                    "output": next_turn_preview,
                }
            )

            yield send_event(
                {
                    "stage": "done",
                    "label": "Evaluation complete",
                    "output": response_payload,
                }
            )

        except Exception as exc:
            yield send_event(
                {
                    "stage": "error",
                    "label": "Evaluation failed",
                    "output": {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                }
            )

    return Response(
        stream_with_context(generate_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def refresh_selected_action_from_turn(
    selected_action: dict[str, Any],
    completed_turn: dict[str, Any],
) -> dict[str, Any]:
    selected_id = selected_action.get("id")

    if not selected_id:
        return selected_action

    for action in completed_turn.get("actions", []):
        if isinstance(action, dict) and action.get("id") == selected_id:
            return {
                **selected_action,
                **action,
            }

    selected_role = (
        selected_action.get("choiceRole")
        or selected_action.get("choice_role")
    )

    for action in completed_turn.get("actions", []):
        action_role = action.get("choiceRole") or action.get("choice_role")

        if (
            isinstance(action, dict)
            and selected_role
            and str(action_role).lower() == str(selected_role).lower()
        ):
            return {
                **selected_action,
                **action,
            }

    return selected_action


def refresh_selected_evidence_from_turn(
    selected_evidence: dict[str, Any],
    turn_number: int,
) -> dict[str, Any]:
    selected_id = selected_evidence.get("id")

    if not selected_id:
        return selected_evidence

    generated_evidence = generate_evidence_for_turn(
        turn_number=turn_number,
        root_dir=ROOT_DIR,
    )

    for evidence in generated_evidence:
        if isinstance(evidence, dict) and evidence.get("id") == selected_id:
            return {
                **selected_evidence,
                **evidence,
            }

    selected_role = (
        selected_evidence.get("supportRole")
        or selected_evidence.get("support_role")
    )

    for evidence in generated_evidence:
        evidence_role = evidence.get("supportRole") or evidence.get("support_role")

        if (
            isinstance(evidence, dict)
            and selected_role
            and str(evidence_role).lower() == str(selected_role).lower()
        ):
            return {
                **selected_evidence,
                **evidence,
            }

    return selected_evidence


@app.route("/api/continue", methods=["POST"])
def continue_to_next_turn():
    """Moves from the evaluated turn into the generated next turn."""

    if scenario_state.completed:
        scenario_state.normalise_action_history()

        return jsonify(
            {
                "isComplete": True,
                "state": scenario_state.to_dict(),
                "message": "Scenario is already complete.",
            }
        )

    scenario_state.commit_pending_turn()

    turn_number = scenario_state.current_turn

    if turn_number >= scenario_state.max_turns and scenario_state.has_recorded_turn(turn_number):
        scenario_state.completed = True
        scenario_state.normalise_action_history()

        return jsonify(
            {
                "isComplete": True,
                "state": scenario_state.to_dict(),
                "message": "Scenario is already complete.",
            }
        )

    if not turn_files_exist(root_dir=ROOT_DIR, turn_number=turn_number):
        return jsonify(
            {
                "error": f"Turn {turn_number} has not been generated yet.",
            }
        ), 500

    turn_payload = load_turn_payload(
        turn_number=turn_number,
        root_dir=ROOT_DIR,
    )

    generated_evidence = generate_evidence_for_turn(
        turn_number=turn_number,
        root_dir=ROOT_DIR,
    )

    return jsonify(
        {
            "isComplete": False,
            "state": scenario_state.to_dict(),
            "turn": turn_payload,
            "generatedEvidence": generated_evidence,
        }
    )


@app.route("/api/reset", methods=["POST"])
def reset():
    """
    Archives the current run and starts a new attempt from the same Turn 1.
    Later turns and evaluations are cleared only from the working directory.
    Existing saved runs are retained.
    """

    # The run manager archives the old attempt and creates a new run first.
    app.extensions["run_store"].fresh_start(scenario_state, test_mode=request.headers.get("X-Test-Mode") == "1")

    if not turn_files_exist(root_dir=ROOT_DIR, turn_number=scenario_state.current_turn):
        return jsonify(
            {
                "error": (
                    "Turn 1 files do not exist. Run dataset preparation before "
                    "resetting into the scenario."
                ),
                "needs_dataset_preparation": True,
                "state": scenario_state.to_dict(),
            }
        ), 409

    turn_payload = load_turn_payload(
        turn_number=scenario_state.current_turn,
        root_dir=ROOT_DIR,
    )

    generated_evidence = generate_evidence_for_turn(
        turn_number=scenario_state.current_turn,
        root_dir=ROOT_DIR,
    )

    return jsonify(
        {
            "state": scenario_state.to_dict(),
            "turn": turn_payload,
            "generatedEvidence": generated_evidence,
        }
    )


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "ok",
            "message": "CloudIR Trainer backend is running.",
        }
    )


from cloudir.services.run_routes import register_run_routes

register_run_routes(app, scenario_state, ROOT_DIR)

if __name__ == "__main__":
    # PyTorch model calls run in worker processes that exit afterwards, so the
    # MPS backend's per-shape graph cache cannot build up in the server
    # (cloudir/ai_models/model_worker.py). CLOUDIR_ISOLATE_MODELS=0 turns it off.
    os.environ.setdefault("CLOUDIR_ISOLATE_MODELS", "1")
    app.run(debug=True, threaded=True, port=int(os.getenv("CLOUDIR_PORT", "5000")))
