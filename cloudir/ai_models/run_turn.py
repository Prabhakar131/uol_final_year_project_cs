from __future__ import annotations

import json
import traceback
from collections.abc import Generator
from pathlib import Path
from typing import Any

from cloudir.ai_models.image_model import analyse_evidence_image, unload_image_model
from cloudir.ai_models.security_model import evaluate_action_evidence, unload_security_model
from cloudir.ai_models.coach_model import generate_coach_feedback, unload_coach_model
from cloudir.ai_models.model_lifecycle import log_memory, unload_all_models
from cloudir.ai_models.extraction_quality import ExtractionQualityError, validate_extraction
from cloudir.scenario.incident_timeline import action_anchor, load_incident_timeline, marking_scheme


PipelineStage = dict[str, Any]


def learner_visible_context(root_dir: Path, scenario_state: dict[str, Any]) -> dict[str, Any]:
    """The current turn's briefing and known context, exactly as the learner sees them.

    Without it the judge cannot tell whether evidence concerns the principal under
    investigation or someone else's routine activity.
    """

    turn = scenario_state.get("currentTurn") or scenario_state.get("current_turn")
    path = root_dir / "data" / "runtime" / "turns" / f"turn_{turn}" / "turn_config.json"

    try:
        turn_config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    return {"briefing": turn_config.get("briefing", ""), "known_context": turn_config.get("known_context", [])}


def turn_marking_scheme(root_dir: Path, scenario_state: dict[str, Any]) -> dict[str, str] | None:
    """The judge's marking scheme for the current turn, rebuilt from the incident timeline.

    Rebuilt rather than read from expected_outcomes.json, so scenarios built before
    it existed (whose outcomes the coach wrote) are graded against their real
    evidence. Scenarios without a timeline get None and the judge's general rules.
    """

    turn = scenario_state.get("currentTurn") or scenario_state.get("current_turn")
    runtime = root_dir / "data" / "runtime"

    try:
        scenario_id = json.loads((runtime / "scenario_config.json").read_text(encoding="utf-8")).get("scenario_id")
        timeline = load_incident_timeline(runtime, scenario_id)
        items = json.loads((runtime / "turns" / f"turn_{turn}" / "evidence_facts.json").read_text(encoding="utf-8"))
        return marking_scheme(timeline, items, action_anchor(timeline, items, int(turn))) if timeline else None
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration):
        return None


def stream_ai_evaluation_turn(
    selected_action: dict[str, Any],
    selected_evidence: dict[str, Any],
    learner_justification: dict[str, Any] | None,
    scenario_state: dict[str, Any],
    root_dir: Path,
) -> Generator[PipelineStage, None, None]:
    """
    Runs the real local AI pipeline as a generator.

    This allows app.py to stream each stage to the browser:

    1. VLM output
    2. Security model output
    3. Coach AI output
    4. Done event

    Important:
    This function only yields stages. Actual browser streaming must be handled
    in app.py using Flask Response + stream_with_context.
    """

    image_path = selected_evidence.get("imagePath")

    if not image_path:
        raise ValueError("selected_evidence is missing imagePath")

    vlm_output: dict[str, Any] | None = None
    security_output: dict[str, Any] | None = None
    coach_output: dict[str, Any] | None = None

    # Start from a clean slate: a model left loaded by an earlier request (next-turn
    # generation, final debrief, voice) would otherwise sit in memory next to the
    # 8B security model and push a 24 GB machine into swap.
    unload_all_models()
    log_memory("pipeline start")

    try:
        try:
            for attempt in range(2):
                try:
                    vlm_output = analyse_evidence_image(
                        image_path=image_path,
                        selected_action=selected_action,
                        selected_evidence=selected_evidence,
                        retry_extraction=bool(attempt),
                    )
                    validate_extraction(vlm_output, selected_evidence)
                    break
                except ValueError as exc:
                    traceback.clear_frames(exc.__traceback__)
                    if attempt:
                        raise ExtractionQualityError(
                            "Evidence could not be read reliably after two attempts. "
                            "Retry evaluation. No verdict or progress was awarded."
                        ) from exc
                    yield {
                        "stage": "extraction_retry", "label": "Re-reading evidence",
                        "output": {"message": "The evidence extraction was incomplete. Re-reading the screenshot once.",
                                   "reason": str(exc), "failed_extraction": vlm_output},
                    }

            yield {
                "stage": "vlm",
                "label": "Vision-language model",
                "output": vlm_output,
            }

        finally:
            unload_image_model()
            log_memory("after VLM unload")

        try:
            security_output = evaluate_action_evidence(
                selected_action=selected_action,
                selected_evidence=selected_evidence,
                learner_justification=learner_justification,
                vlm_output=vlm_output,
                scenario_state=scenario_state,
                incident_context=learner_visible_context(root_dir, scenario_state),
                marking_scheme=turn_marking_scheme(root_dir, scenario_state),
            )

            yield {
                "stage": "security",
                "label": "Security reasoning model",
                "output": security_output,
            }

        finally:
            unload_security_model()
            log_memory("after security unload")

        try:
            coach_output = generate_coach_feedback(
                selected_action=selected_action,
                selected_evidence=selected_evidence,
                learner_justification=learner_justification,
                vlm_output=vlm_output,
                security_output=security_output,
                scenario_state=scenario_state,
            )

            yield {
                "stage": "coach",
                "label": "Coach AI",
                "output": coach_output,
            }

        finally:
            unload_coach_model()
            log_memory("after coach unload")

        yield {
            "stage": "done",
            "label": "Pipeline complete",
            "output": {
                "vlm_output": vlm_output,
                "security_output": security_output,
                "coach_output": coach_output,
            },
        }

    except Exception as exc:
        # The traceback keeps the failing stage's frames (and the model inside
        # them) alive; clear them so the model can actually be released.
        traceback.clear_frames(exc.__traceback__)
        unload_all_models()
        log_memory("after pipeline error")

        yield {
            "stage": "error",
            "label": "Pipeline error",
            "output": {
                "error": str(exc),
                "error_type": type(exc).__name__,
                **({"extraction_details": str(exc.__cause__), "last_extraction": vlm_output}
                   if isinstance(exc, ExtractionQualityError) else {}),
            },
        }


def run_ai_evaluation_turn(
    selected_action: dict[str, Any],
    selected_evidence: dict[str, Any],
    scenario_state: dict[str, Any],
    root_dir: Path,
    learner_justification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Runs the real local AI pipeline in normal non-streaming mode.

    This keeps backward compatibility with your existing app.py.
    It internally consumes stream_ai_evaluation_turn() and returns the final result.
    """

    final_output: dict[str, Any] | None = None

    for event in stream_ai_evaluation_turn(
        selected_action=selected_action,
        selected_evidence=selected_evidence,
        learner_justification=learner_justification,
        scenario_state=scenario_state,
        root_dir=root_dir,
    ):
        if event["stage"] == "done":
            final_output = event["output"]

        if event["stage"] == "error":
            error_output = event["output"]
            raise RuntimeError(
                f"{error_output['error_type']}: {error_output['error']}"
            )

    if final_output is None:
        raise RuntimeError("AI pipeline finished without producing final output.")

    return final_output
