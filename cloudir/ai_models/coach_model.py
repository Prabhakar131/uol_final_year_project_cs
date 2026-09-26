from __future__ import annotations

import gc
import json
import os
import re
from functools import lru_cache
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from cloudir.evidence_core.evidence_text_helpers import clean_value
from cloudir.evidence_core.fact_repair import repair_next_turn_template_facts
from cloudir.evidence_core.schemas import EVIDENCE_TEMPLATE_SCHEMAS
from cloudir.scenario.action_choice_roles import normalise_action_choice_roles
from cloudir.scenario.evidence_support_roles import normalise_evidence_support_roles
from cloudir.scenario.evidence_strength import SUPPORT_ROLE_CONTENT_RULES
from cloudir.ai_models.learner_grounding import validate_evidence_only_prose, keep_evidence_only_prose
from cloudir.ai_models.model_worker import isolated_model_call


load_dotenv()


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


COACH_BACKEND_ENV = "COACH_MODEL_BACKEND"
DEFAULT_COACH_GGUF_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
# 16-bit, the precision the coach ran at under PyTorch.
DEFAULT_COACH_GGUF_FILE = "qwen2.5-1.5b-instruct-fp16.gguf"
_coach_gguf: Any = None


def coach_backend() -> str:
    """llama_cpp by default: the same Qwen2.5-1.5B model as a 16-bit GGUF.

    Under PyTorch on the Apple GPU the coach's memory grew with every generated
    token (a compiled graph per tensor shape): one 800-token next turn took its
    worker from 3.6 to 12.3 GB and a final debrief to 21 GB, for a model that
    needs about 4 GB. llama.cpp has no such cache. COACH_MODEL_BACKEND=transformers
    restores the 16-bit PyTorch model.
    """

    backend = os.getenv(COACH_BACKEND_ENV, "llama_cpp").strip().lower()

    if backend not in {"llama_cpp", "transformers"}:
        raise RuntimeError(f"{COACH_BACKEND_ENV} must be llama_cpp or transformers, not {backend!r}.")

    return backend


@lru_cache(maxsize=1)
def _load_coach_model() -> tuple[Any, AutoTokenizer]:
    model_id = os.getenv("HF_COACH_MODEL_ID")

    if not model_id:
        raise RuntimeError("HF_COACH_MODEL_ID is missing from .env")

    device = _device()

    # Both backends format prompts with the original tokenizer's chat template.
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if coach_backend() == "llama_cpp":
        return _load_gguf_coach_model(), tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device in {"mps", "cuda"} else torch.float32,
        low_cpu_mem_usage=True,
    )

    model.to(device)
    model.eval()

    return model, tokenizer


def clear_torch_memory() -> None:
    """
    Clears Python references and releases cached CUDA/MPS memory.

    Note:
    PyTorch/macOS may still keep some memory reserved temporarily, so Activity Monitor
    may not drop to zero immediately.
    """

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _load_gguf_coach_model() -> Any:
    global _coach_gguf

    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    model_path = os.getenv("HF_COACH_GGUF_PATH") or hf_hub_download(
        os.getenv("HF_COACH_GGUF_REPO", DEFAULT_COACH_GGUF_REPO),
        os.getenv("HF_COACH_GGUF_FILE", DEFAULT_COACH_GGUF_FILE),
    )
    _coach_gguf = Llama(
        model_path=model_path,
        # Next-turn prompts are ~4-5k tokens and may answer with 3200.
        n_ctx=int(os.getenv("HF_COACH_GGUF_CONTEXT", "16384")),
        n_gpu_layers=-1,
        verbose=False,
    )
    return _coach_gguf


@lru_cache(maxsize=1)
def _repetition_penalty() -> float:
    """The model's own default, which transformers applies even to greedy decoding."""

    from transformers import GenerationConfig

    try:
        config = GenerationConfig.from_pretrained(os.environ["HF_COACH_MODEL_ID"])
    except (OSError, KeyError):
        return 1.0
    return float(config.repetition_penalty or 1.0)


def _transformers_repetition_penalty(penalty: float):
    """transformers' repetition penalty for llama.cpp: every token already in the
    prompt or the answer is penalised, as RepetitionPenaltyLogitsProcessor does.

    llama.cpp's own repeat_penalty looks at the last 64 tokens only. Without the
    full-context penalty the coach copied prompt text such as the schema's
    "short phase name" into its turns, at 8-bit and at 16-bit alike.
    """

    import numpy as np

    def process(input_ids, scores):
        seen = np.unique(input_ids)
        picked = scores[seen]
        scores[seen] = np.where(picked > 0, picked / penalty, picked * penalty)
        return scores

    return process


def _generate_coach_text(tokenizer: Any, model: Any, device: str, input_text: str, max_new_tokens: int) -> str:
    """Greedy generation on either backend; returns only the new text."""

    if hasattr(model, "create_completion"):
        from llama_cpp import LogitsProcessorList

        penalty = _repetition_penalty()
        # The chat template already holds every special token Qwen needs.
        completion = model.create_completion(
            tokenizer(input_text, add_special_tokens=False)["input_ids"],
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            min_p=0.0,
            repeat_penalty=1.0,
            logits_processor=LogitsProcessorList([_transformers_repetition_penalty(penalty)]) if penalty != 1.0 else None,
        )
        return completion["choices"][0]["text"]

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def unload_coach_model() -> None:
    """
    Unloads the cached coach model so run_turn.py can control model lifecycle.
    """

    global _coach_gguf

    if _coach_gguf is not None:
        close = getattr(_coach_gguf, "close", None)
        if callable(close):
            close()
        _coach_gguf = None

    _load_coach_model.cache_clear()
    clear_torch_memory()


@isolated_model_call
def generate_coach_feedback(
    selected_action: dict[str, Any],
    selected_evidence: dict[str, Any],
    learner_justification: dict[str, Any] | None,
    vlm_output: dict[str, Any],
    security_output: dict[str, Any],
    scenario_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Uses a local Hugging Face text-generation model to create learner-facing feedback.

    This is a real model call. No mock output is produced.
    """

    model, tokenizer = _load_coach_model()
    device = _device()

    ground_truth_keys = {"choice_role", "choiceRole", "support_role", "supportRole"}
    action_for_prompt = {k: v for k, v in selected_action.items() if k not in ground_truth_keys}
    evidence_for_prompt = {k: v for k, v in selected_evidence.items()
                           if k in {"id", "title", "type", "template"}}
    observed_facts = {k: v for k, v in vlm_output.items()
                      if k in {"evidence_type", "visible_evidence_summary", "visible_facts_extracted",
                               "important_visible_fields", "extraction_warnings", "event_rows",
                               "query_text", "time_range", "log_group", "matched_records"}}
    # Learner statements/assessments are displayed separately with exact quotes.
    # Do not let the coach paraphrase them or propagate old ungrounded praise.
    evidence_judgement = {k: security_output[k] for k in (
        "verdict", "reasoning", "recommended_next_focus", "risk_of_wrong_interpretation",
        "supporting_row_numbers") if k in security_output}
    validate_evidence_only_prose(evidence_judgement, (
        "reasoning", "recommended_next_focus", "risk_of_wrong_interpretation"))

    prompt = f"""
You are the AI Coach in CloudIR Trainer.

Your job is to coach the learner after the system has evaluated their action and evidence.
Sound like a thoughtful incident-response instructor, not a command checklist.

Selected action:
{json.dumps(action_for_prompt, indent=2)}

Selected evidence:
{json.dumps(evidence_for_prompt, indent=2)}

Observed screenshot facts from the vision-language model:
{json.dumps(observed_facts, indent=2)}

Security reasoning AI output:
{json.dumps(evidence_judgement, indent=2)}

Current scenario state:
{json.dumps(scenario_state, indent=2)}

Return ONLY valid JSON with exactly these keys:
{{
  "feedback": "learner-facing feedback paragraph",
  "next_turn_guidance": "short guidance for what the learner should consider next"
}}

Rules:
- Do not mention hidden truth.
- Do not expose scoring logic.
- Do not dump raw technical JSON.
- Explain what the evidence supports, one limitation, and a practical next check. Use impersonal evidence-focused language. Learner reasoning is assessed separately using exact transcript statements; do not assess, praise, or paraphrase it here.
- Explain why the selected evidence was strong, partial, weak, or unsupported without sounding like a verdict machine.
- Follow the security verdict when describing evidence strength. Do not add blanket praise or imply that choosing relevant evidence proves understanding.
- Never treat a clipped value ending in ... as a complete IP address, identifier, or separate entity. Explain the truncation or use a complete value from a specifically identified visible event row.
- Query descriptions express what was searched for, not what happened. Base event claims on the extracted event rows. Do not credit the learner with details that appear only in prompt cues.
- Never attribute observations, acknowledgement, understanding or reasoning to a person. Do not use 'you', 'your', 'learner', 'transcript' or 'justification' in either output field.
- Prefer phrases like "The visible events support", "The evidence does not establish", and "Which additional record would clarify".
- next_turn_guidance should be a coaching question or reflective nudge, not an instruction list.
- Keep the tone warm, calm, and specific.
- Maximum 120 words for feedback.
"""

    messages = [
        {
            "role": "system",
            "content": "You are a calm cybersecurity training coach. Return JSON only.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    for attempt in range(2):
        output_text = _generate_feedback_response(tokenizer, model, device, messages)
        try:
            result = _parse_coach_feedback_output(output_text)
            audit = keep_evidence_only_prose(result, ("feedback", "next_turn_guidance"))
            return {**{key: result[key] for key in ("feedback", "next_turn_guidance")},
                    "feedback_grounding": audit}
        except ValueError as exc:
            if attempt:
                raise ValueError("Coach feedback failed grounding checks after retry; no progress awarded.") from exc
            messages.append({"role": "user", "content": (
                f"Regenerate both JSON fields. Validation failed: {exc}. "
                "Describe only evidence, limitations and next checks. No personal praise, second-person wording or learner attribution."
            )})


def _generate_feedback_response(tokenizer, model, device, messages) -> str:
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    max_new_tokens = int(os.getenv("HF_MAX_NEW_TOKENS", "350"))

    output_text = _generate_coach_text(tokenizer, model, device, input_text, max_new_tokens)

    return output_text


@isolated_model_call
def generate_final_debrief_json(
    *,
    evaluation_traces: list[dict[str, Any]],
    final_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Uses the Coach AI to synthesize a final learner debrief after the scenario.

    The debrief is intended for the end screen and for project evaluation logs.
    It must be concrete: refer to the actual learner choices, evidence facts,
    reasoning gaps, and risk warnings captured across the completed turns.
    """

    model, tokenizer = _load_coach_model()
    device = _device()

    compact_traces = [
        _compact_trace_for_final_debrief(trace)
        for trace in evaluation_traces
    ]

    prompt = f"""
You are the final debrief Coach AI for CloudIR Trainer.

You are reviewing a learner's completed cloud incident response simulation.
Use the actual evaluation traces. Be specific and educational.

FINAL SCENARIO STATE:
{json.dumps(final_state, indent=2)}

TURN EVALUATION TRACES:
{json.dumps(compact_traces, indent=2)}

Return ONLY valid JSON with exactly these keys:
{{
  "overall_assessment": "detailed paragraph evaluating the whole run",
  "performance_level": "Strong | Developing | Needs Work",
  "evidence_quality_score": "for example 5/5 strong",
  "incident_summary": {{
    "what_happened": "short concrete incident summary",
    "what_was_proven": "what the selected evidence actually established",
    "what_remains_uncertain": "what still needs verification"
  }},
  "decision_scorecard": [
    {{
      "label": "Evidence relevance",
      "score": 0,
      "feedback": "specific score reason"
    }}
  ],
  "strengths": [
    "specific strength based on actual evidence/action use"
  ],
  "missed_details": [
    "specific visible or reasoning detail the learner missed"
  ],
  "risky_interpretations": [
    "specific interpretation risk or overclaim to avoid"
  ],
  "recommended_response": "concrete final incident response recommendation",
  "response_checklist": [
    "concrete operational follow-up step"
  ],
  "reflection_prompts": [
    "post-incident learning question"
  ],
  "learning_targets": [
    "specific improvement target for the learner"
  ],
  "turn_debriefs": [
    {{
      "turn": 1,
      "action": "selected action title",
      "evidence": "selected evidence title",
      "verdict": "security verdict",
      "what_went_well": "specific positive feedback",
      "what_was_missing": "specific missing detail or empty string",
      "evidence_facts_used": [
        "specific visible fact used in the reasoning"
      ],
      "risk_warning": "specific risk of wrong interpretation",
      "next_time_improve": "specific improvement suggestion"
    }}
  ]
}}

Rules:
- Do not invent services, users, IPs, or facts not present in the traces.
- Do not reveal hidden truth.
- Do not say the learner did something if the transcript was empty.
- Use the visible facts and security evaluation details as the evidence base.
- Mention concrete fields such as source IP, IAM principal, event name, MFA status, access key status, region, or timestamp when they appear.
- Separate what the screenshot proves from what still needs investigation.
- If all turns were Strong Support, still include at least one missed detail or improvement target.
- decision_scorecard must include Evidence relevance, Field extraction, Avoiding overclaiming, and Response progression with 0-100 scores.
- response_checklist must contain 4-6 concrete operational steps.
- reflection_prompts must contain 2-4 learner reflection questions.
- Keep the final report detailed but concise enough for the UI.
- Generate exactly one turn_debrief for each trace supplied.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise cybersecurity training debrief coach. "
                "Return JSON only and ground every claim in the supplied traces."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    configured_max_new_tokens = int(os.getenv("HF_FINAL_DEBRIEF_MAX_NEW_TOKENS", "1800"))
    max_new_tokens = max(configured_max_new_tokens, 1200)

    output_text = _generate_coach_text(tokenizer, model, device, input_text, max_new_tokens)

    return _parse_final_debrief_output(output_text, compact_traces)


@isolated_model_call
def generate_next_turn_json(
    scenario_config: dict[str, Any],
    hidden_truth: dict[str, Any],
    current_state: dict[str, Any],
    completed_turn: dict[str, Any],
    selected_action: dict[str, Any],
    selected_evidence: dict[str, Any],
    vlm_output: dict[str, Any],
    security_output: dict[str, Any],
    coach_output: dict[str, Any],
    next_turn_number: int,
    retry_instruction: str = "",
    evidence_brief: str = "",
) -> dict[str, Any]:
    """
    Uses the same local Hugging Face generative coach model to generate
    the next turn JSON files. With evidence_brief (fixed timeline evidence) the
    coach writes the turn around it and the caller replaces the evidence facts.

    This is not learner-facing feedback.
    This is structured scenario generation for the next turn.
    """

    model, tokenizer = _load_coach_model()
    device = _device()

    scenario_values = _extract_scenario_values(
        scenario_config=scenario_config,
        hidden_truth=hidden_truth,
        current_state=current_state,
        completed_turn=completed_turn,
        selected_action=selected_action,
        selected_evidence=selected_evidence,
        vlm_output=vlm_output,
        security_output=security_output,
        coach_output=coach_output,
    )

    progression_guidance = _build_scenario_progression_guidance(
        scenario_config=scenario_config,
        current_state=current_state,
        completed_turn=completed_turn,
        selected_action=selected_action,
        selected_evidence=selected_evidence,
        security_output=security_output,
        coach_output=coach_output,
        next_turn_number=next_turn_number,
    )

    evidence_plan = _build_evidence_plan(
        next_turn_number=next_turn_number,
        scenario_values=scenario_values,
        scenario_config=scenario_config,
        current_state=current_state,
        completed_turn=completed_turn,
        selected_action=selected_action,
        selected_evidence=selected_evidence,
        security_output=security_output,
    )

    retry_guidance = ""
    if retry_instruction.strip():
        retry_guidance = f"""
QUALITY REVIEW RETRY INSTRUCTION:
The previous generated turn did not pass the AI quality review.

You MUST address this retry instruction:
{retry_instruction}

When regenerating:
- Do not repeat the same weak turn.
- Change the scenario progression meaningfully.
- Change the evidence focus where appropriate.
- Use concrete AWS-style facts.
- Keep the scenario consistent with hidden_truth.json.
"""

    compact_context = {
        "turn": next_turn_number,
        "scenario_values": scenario_values,
        "current_state": current_state,
        "completed_phase": _get_completed_turn_config(completed_turn).get("phase", ""),
        "previous_action_titles": _extract_titles(completed_turn, "actions"),
        "previous_evidence_titles": _extract_titles(completed_turn, "evidence_facts")
        or _extract_titles(completed_turn, "evidenceFacts"),
        "previous_evidence_templates": _extract_evidence_templates(completed_turn),
        "selected_action": {
            "title": selected_action.get("title", selected_action.get("id", "")),
            "description": selected_action.get("description", ""),
        },
        "selected_evidence": {
            "title": selected_evidence.get("title", selected_evidence.get("id", "")),
            "template": selected_evidence.get("template", selected_evidence.get("type", "")),
            "summary": selected_evidence.get("summary", ""),
        },
        "security_verdict": security_output.get("verdict", ""),
        "security_next_focus": security_output.get("recommended_next_focus", ""),
        "coach_next_turn_guidance": coach_output.get("next_turn_guidance", ""),
    }

    prompt = f"""
You are the scenario generation component of CloudIR Trainer.

Generate Turn {next_turn_number} as COMPLETE, VALID, COMPACT JSON.
The output must fit comfortably within the token limit, but learner-facing
briefing, known_context, action titles, and coach_guidance must still feel
specific to this incident.

COMPACT CONTEXT:
{json.dumps(compact_context, indent=2)}

MANDATORY SCENARIO PROGRESSION GUIDANCE:
{progression_guidance}

ADAPTIVE EVIDENCE PLANNING GUIDANCE FOR THIS TURN:
{evidence_plan}

{retry_guidance}

Return ONLY valid JSON with exactly these top-level keys:
{{
  "turn_config": {{
    "turn": {next_turn_number},
    "phase": "specific incident response phase",
    "briefing": "2 sentence learner-visible incident briefing for the next turn",
    "known_context": [
      "concrete known incident fact with identity, log, source, credential, or monitoring detail",
      "concrete known incident fact with identity, log, source, credential, or monitoring detail",
      "concrete known incident fact with identity, log, source, credential, or monitoring detail"
    ],
    "coach_guidance": "practical learner-facing guidance for choosing the next action"
  }},
  "actions": [
    {{"id": "snake_case_action_id", "title": "action title", "description": "one sentence description", "recommended_next_focus": "short next focus", "choice_role": "best"}},
    {{"id": "snake_case_action_id", "title": "action title", "description": "one sentence description", "recommended_next_focus": "short next focus", "choice_role": "partial"}},
    {{"id": "snake_case_action_id", "title": "action title", "description": "one sentence description", "recommended_next_focus": "short next focus", "choice_role": "weak"}}
  ],
  "evidence_facts": [
    {{"id": "snake_case_evidence_id", "title": "evidence title", "type": "template", "summary": "short summary", "why_it_may_matter": "short reason", "support_role": "strong", "template": "template", "facts": {{"key": "value"}}}},
    {{"id": "snake_case_evidence_id", "title": "evidence title", "type": "template", "summary": "short summary", "why_it_may_matter": "short reason", "support_role": "partial", "template": "template", "facts": {{"key": "value"}}}},
    {{"id": "snake_case_evidence_id", "title": "evidence title", "type": "template", "summary": "short summary", "why_it_may_matter": "short reason", "support_role": "weak", "template": "template", "facts": {{"key": "value"}}}}
  ],
  "expected_outcomes": {{
    "strong_support": "what counts as strong evidence use",
    "partial_support": "what counts as partial evidence use",
    "weak_support": "what counts as weak evidence use",
    "unsupported": "what counts as unsupported evidence use"
  }}
}}

ABSOLUTE OUTPUT RULES:
- Return JSON only.
- No markdown.
- No explanation outside JSON.
- Generate exactly 3 actions.
- Generate exactly 3 evidence items.
- Action choice_role values must be exactly one best, one partial, and one weak.
- The best action should be the most appropriate response for the current turn goal.
- The partial action should be reasonable but less complete.
- The weak action should be plausible but not the preferred investigation or response path.
- Evidence support_role values must be exactly one strong, one partial, and one weak.
- The strong item should directly justify the best action for this turn.
- The partial item should be relevant but incomplete.
- The weak item should be plausible cloud evidence but not enough to justify the best action by itself.
- Keep the entire JSON concise.
- briefing must be 2 sentences and roughly 35-65 words.
- known_context must contain exactly 3 concrete facts, each roughly 10-24 words.
- coach_guidance must be 1 practical sentence and roughly 18-35 words.
- Keep action descriptions, evidence summaries, why_it_may_matter, and expected_outcome values under 24 words.
- The 3 evidence items MUST use 3 different templates.
- Do not use cloudtrail more than once in the same turn.
- Do not use any evidence template more than once in the same turn.
- Use the mandatory scenario progression guidance when deciding the next phase, briefing, actions, and evidence.
- The next turn must reflect the learner's selected action, selected evidence, security verdict, coach feedback, and action history.
- If the learner made a strong evidence-based choice, progress the scenario to a deeper or later-stage investigation.
- If the learner made a partially supported choice, progress the scenario but include uncertainty or missing context.
- If the learner made a weak or unsupported choice, refocus the next turn toward the missing evidence path without revealing the hidden truth directly.
- Do not repeat the same phase, briefing goal, action purpose, or evidence purpose from the completed turn.
- Evidence must progress or refocus the scenario instead of repeating the exact same evidence every turn.
- The 3 evidence items should follow the adaptive evidence planning guidance unless the learner's previous choice makes a different evidence mix more coherent.
- The "type" field and "template" field MUST be identical.
- Allowed templates only: cloudtrail, iam_activity, guardduty, cloudwatch, access_key, billing.
- Never generate type="guardduty" with template="cloudtrail".
- Never generate type="iam_activity" with template="cloudtrail".
- Never generate type="cloudwatch" with template="cloudtrail".
- Never leave facts as {{}}.
- Never generate facts with all values as "Unknown".
- Never put all important information only in summary.
- Every evidence item must contain at least 2 concrete fact keys. The app normalises fact keys but never adds missing events.
- Evidence must be consistent with the scenario values above.
- Do not write bare context such as only "Unauthorized login attempt detected".
- Do not use vague action titles such as "Threat Analysis", "Access Logs Review", or "Inspection of Affected Assets".
- Use AWS incident language: CloudTrail, IAM activity, MFA, source IP, access key, GuardDuty, CloudWatch, logging, containment.
- Turn 2 must not mark "review more CloudTrail logs" as the best action after a strong Turn 1 CloudTrail choice; make the best action correlation, scope, IAM timeline, credential risk, or monitoring correlation.
- Later turns must not mark "review more CloudTrail logs" as the best action after a strong or partial previous choice unless the target stage specifically needs final audit or recovery evidence.
- CloudTrail can still appear after Turn 1, but only when it has a different role such as correlation, logging recovery, final audit, or supporting timeline evidence.

SCENARIO DESIGN RULES:
- Include one best action, one reasonable partial action, and at least one weak or distracting action.
- Include one best evidence item, one partial evidence item, and one weaker evidence item.
- Evidence must progress or refocus the scenario instead of repeating the exact same evidence every turn.
- Keep the scenario consistent with previous turns.
- Do not reveal hidden truth directly to the learner.
- Keep all learner-visible content realistic for AWS cloud incident response.
{SUPPORT_ROLE_CONTENT_RULES}"""

    if evidence_brief.strip():
        prompt += evidence_brief + """
These fixed items override the evidence rules above: return each item's facts as {}.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You generate strict JSON for a cloud incident response training simulator. "
                "Return JSON only. Evidence facts must be concrete, scenario-specific, realistic, and renderable. "
                "Never generate placeholder evidence. Generated turns must evolve based on learner choices."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    configured_max_new_tokens = int(os.getenv("HF_NEXT_TURN_MAX_NEW_TOKENS", "3200"))
    max_new_tokens = max(configured_max_new_tokens, 3200)

    output_text = _generate_coach_text(tokenizer, model, device, input_text, max_new_tokens)

    parsed = _parse_next_turn_output(output_text)

    if evidence_brief.strip():
        # The caller replaces evidence with the timeline items; only text is kept.
        normalise_action_choice_roles(parsed.get("actions"))
        return parsed

    repaired = repair_next_turn_template_facts(parsed)
    normalise_action_choice_roles(repaired.get("actions"))
    normalise_evidence_support_roles(repaired.get("evidence_facts"))
    repaired = _ensure_unique_evidence_templates(
        generated_turn=repaired,
        scenario_values=scenario_values,
        next_turn_number=next_turn_number,
    )

    normalise_evidence_support_roles(repaired.get("evidence_facts"))
    repaired = repair_next_turn_template_facts(repaired)
    normalise_action_choice_roles(repaired.get("actions"))
    normalise_evidence_support_roles(repaired.get("evidence_facts"))

    return repaired


def _ensure_unique_evidence_templates(
    generated_turn: dict[str, Any],
    scenario_values: dict[str, str],
    next_turn_number: int,
) -> dict[str, Any]:
    """
    Ensures each generated evidence item in a turn uses a different template.

    The Coach model can sometimes repeat cloudtrail/cloudtrail/cloudtrail.
    The structural validator rejects that because the learner needs distinct
    evidence sources in each turn.
    """

    evidence_facts = generated_turn.get("evidence_facts")

    if not isinstance(evidence_facts, list):
        return generated_turn

    allowed_templates = [
        "cloudtrail",
        "iam_activity",
        "guardduty",
        "cloudwatch",
        "access_key",
        "billing",
    ]

    used_templates: set[str] = set()

    for index, item in enumerate(evidence_facts):
        if not isinstance(item, dict):
            continue

        template = str(
            item.get("template")
            or item.get("type")
            or ""
        ).strip().lower()

        must_replace = template not in allowed_templates or template in used_templates

        if must_replace:
            raise ValueError(
                f"Duplicate or invalid evidence template: {template}. Regenerate distinct "
                "evidence content with justified support roles; do not replace facts while retaining labels."
            )

        if "why_it_may_matter" not in item and "whyItMayMatter" in item:
            item["why_it_may_matter"] = item.pop("whyItMayMatter")

        item["template"] = template
        item["type"] = template

        used_templates.add(template)

    return generated_turn


def _choose_replacement_template(
    used_templates: set[str],
    allowed_templates: list[str],
    index: int,
    next_turn_number: int,
) -> str:
    """
    Chooses a replacement evidence template that has not already been used.
    """

    preferred_by_turn = {
        1: ["cloudtrail", "iam_activity", "guardduty"],
        2: ["iam_activity", "access_key", "cloudwatch"],
        3: ["guardduty", "cloudwatch", "billing"],
    }

    preferred_templates = preferred_by_turn.get(
        next_turn_number,
        ["cloudtrail", "iam_activity", "guardduty"],
    )

    for template in preferred_templates:
        if template in allowed_templates and template not in used_templates:
            return template

    for template in allowed_templates:
        if template not in used_templates:
            return template

    return allowed_templates[index % len(allowed_templates)]


def _fallback_facts_for_template(
    template: str,
    scenario_values: dict[str, str],
) -> dict[str, Any]:
    """
    Provides renderable fallback facts only when a duplicate template must be repaired.

    The values are based on extracted scenario values, not random placeholders.
    """

    event_name = scenario_values.get("event_name", "StopLogging")
    event_source = scenario_values.get("event_source", "cloudtrail.amazonaws.com")
    user = scenario_values.get("user", "admin-test")
    source_ip = scenario_values.get("source_ip", "185.220.101.42")
    mfa = scenario_values.get("mfa", "false")
    event_time = scenario_values.get("event_time", "10:21:35 UTC")
    region = scenario_values.get("region", "ap-southeast-1")
    user_agent = scenario_values.get("user_agent", "aws-cli/2.13.7 Python/3.11")
    account_id = scenario_values.get("account_id", "123456789012")
    access_key_id = scenario_values.get("access_key_id", "AKIA4Z7EXAMPLE92K")

    if template == "cloudtrail":
        return {
            "event_name": event_name,
            "event_source": event_source,
            "user": user,
            "source_ip": source_ip,
            "mfa": mfa,
            "event_time": event_time,
            "region": region,
            "error_code": "-",
            "risk_signal": (
                f"{event_name} was observed for {user} from {source_ip} "
                f"in {region}."
            ),
            "event_id": "8f42b4ac-9f1b-4d1a-a3c8-5df2c9a71c03",
            "user_agent": user_agent,
            "recipient_account_id": account_id,
            "request_parameters": f"userName={user}, sourceIPAddress={source_ip}",
            "related_events": [
                ["2023-10-01T10:12:10Z", "ConsoleLogin", user, source_ip, f"MFA {mfa}"],
                ["2023-10-01T10:18:44Z", "ListTrails", user, source_ip, "Success"],
                [event_time, event_name, user, source_ip, "Success"],
                ["2023-10-01T10:24:02Z", "CreateAccessKey", user, source_ip, "Success"],
            ],
        }

    if template == "iam_activity":
        return {
            "principal": user,
            "source_ip": source_ip,
            "mfa": mfa,
            "policy_change": "No approved IAM change ticket found",
            "access_key_status": "New active key observed after login",
            "risk_flags": [
                "Console login without MFA",
                "Role and credential actions from same source IP",
            ],
            "activity_rows": [
                ["2023-10-01T10:12:10Z", user, "ConsoleLogin", source_ip],
                ["2023-10-01T10:18:44Z", user, "ListTrails", source_ip],
                [event_time, user, event_name, source_ip],
                ["2023-10-01T10:24:02Z", user, "CreateAccessKey", source_ip],
                ["2023-10-01T10:26:11Z", user, "GetCallerIdentity", source_ip],
            ]
        }

    if template == "guardduty":
        return {
            "finding_type": "Stealth:IAMUser/CloudTrailLoggingDisabled",
            "severity": "Medium",
            "resource": user,
            "principal": user,
            "remote_ip": source_ip,
            "first_seen": event_time,
            "last_seen": "2023-10-01T10:26:11Z",
            "summary": (
                "GuardDuty detected suspicious IAM activity from the same "
                "remote IP shortly after the identity-management signal."
            ),
        }

    if template == "cloudwatch":
        return {
            "query": "fields @timestamp, @logStream, @message | filter @message like /ConsoleLogin|AssumeRole|CreateAccessKey|mfaAuthenticated=false/ | sort @timestamp asc",
            "log_group": "/aws/cloudtrail/organization/prod",
            "matched_records": "5",
            "scanned_bytes": "1400000",
            "time_range": "2023-10-01 10:10-10:30 UTC",
            "alarm_state": "Identity correlation signal",
            "log_rows": [
                [
                    event_time,
                    "security/cloudtrail-monitor",
                    f"{event_name} observed for {user} from {source_ip}",
                ],
                [
                    "2023-10-01T10:22:01Z",
                    "security/correlation",
                    f"Same source IP {source_ip} observed across auth and IAM actions",
                ],
                [
                    "2023-10-01T10:24:02Z",
                    "iam/credential-monitor",
                    f"CreateAccessKey observed for {user}",
                ],
            ]
        }

    if template == "access_key":
        return {
            "access_key_id": access_key_id,
            "owner": user,
            "status": "Active",
            "last_used_service": event_source,
            "last_used_region": region,
            "last_used_time": "2023-10-01T10:24:02Z",
            "source_ip": source_ip,
        }

    if template == "billing":
        return {
            "current_spend": "$184.20",
            "previous_average": "$42.75",
            "largest_service": "EC2",
            "region": region,
            "change": "+331%",
        }

    return {}


def _title_for_template(template: str) -> str:
    titles = {
        "cloudtrail": "CloudTrail Event Review",
        "iam_activity": "IAM Activity Timeline",
        "guardduty": "GuardDuty Finding",
        "cloudwatch": "CloudWatch Security Logs",
        "access_key": "Access Key Usage",
        "billing": "Billing Impact Snapshot",
    }

    return titles.get(template, "Cloud Evidence Review")


def _summary_for_template(template: str) -> str:
    summaries = {
        "cloudtrail": "CloudTrail shows a relevant identity-management API event.",
        "iam_activity": "IAM activity shows a timeline of identity actions around the incident.",
        "guardduty": "GuardDuty shows a suspicious identity-related security finding.",
        "cloudwatch": "CloudWatch logs show monitoring signals related to the investigation.",
        "access_key": "Access key metadata shows credential usage relevant to the investigation.",
        "billing": "Billing data shows possible impact or resource usage changes.",
    }

    return summaries.get(template, "This evidence may help investigate the incident.")


def _why_it_matters_for_template(template: str) -> str:
    reasons = {
        "cloudtrail": "It can confirm who performed an API action, when it happened, and from where.",
        "iam_activity": "It helps correlate identity activity before and after the suspicious event.",
        "guardduty": "It can provide threat context and suspicious-source indicators.",
        "cloudwatch": "It can reveal monitoring gaps, alerts, or related operational signals.",
        "access_key": "It can show whether credentials may have been created, used, or abused.",
        "billing": "It can help assess impact, abnormal usage, or business risk.",
    }

    return reasons.get(template, "It may provide additional context for the learner's response decision.")


@isolated_model_call
def generate_initial_turn_from_dataset(
    scenario_seed: dict[str, Any],
    hidden_truth: dict[str, Any],
    scenario_config: dict[str, Any],
    retry_instruction: str = "",
    evidence_brief: str = "",
) -> dict[str, Any]:
    """
    Uses the local Coach AI to generate the first playable CloudIR turn
    from prepared ACSE dataset files.

    This function is used during dataset preparation, before the normal
    learner runtime starts. Decoding is greedy, so a retry only differs from
    the first attempt through retry_instruction. With evidence_brief (fixed
    timeline evidence), the coach writes the turn around that evidence and the
    caller replaces the evidence facts afterwards.
    """

    model, tokenizer = _load_coach_model()
    device = _device()

    scenario_values = _extract_scenario_values(
        scenario_config=scenario_config,
        hidden_truth=hidden_truth,
        current_state={},
        completed_turn={},
        selected_action={},
        selected_evidence={},
        vlm_output={},
        security_output={},
        coach_output={},
    )

    prompt = f"""
You are the Coach / Turn Preparation AI inside CloudIR Trainer.

You are preparing Turn 1 of a cloud incident response training simulator from a real ACSE-Eval dataset scenario.

The learner has not started yet.
Your job is to create the first playable turn.

SCENARIO SEED:
{json.dumps(scenario_seed, indent=2)}

HIDDEN TRUTH:
{json.dumps(hidden_truth, indent=2)}

SCENARIO CONFIG:
{json.dumps(scenario_config, indent=2)}

Return ONLY valid JSON with exactly these top-level keys:
{{
  "turn_config": {{
    "turn": 1,
    "phase": "short phase name",
    "briefing": "learner-visible incident briefing for Turn 1",
    "known_context": [
      "learner-visible known fact",
      "learner-visible known fact",
      "learner-visible known fact"
    ],
    "coach_guidance": "short learner-facing guidance for choosing the first action"
  }},
  "actions": [
    {{
      "id": "snake_case_action_id",
      "title": "action title",
      "description": "one sentence description",
      "recommended_next_focus": "what this action should lead to",
      "choice_role": "best"
    }},
    {{
      "id": "snake_case_action_id",
      "title": "action title",
      "description": "one sentence description",
      "recommended_next_focus": "what this action should lead to",
      "choice_role": "partial"
    }},
    {{
      "id": "snake_case_action_id",
      "title": "action title",
      "description": "one sentence description",
      "recommended_next_focus": "what this action should lead to",
      "choice_role": "weak"
    }},
    {{
      "id": "snake_case_action_id",
      "title": "action title",
      "description": "one sentence description",
      "recommended_next_focus": "what this action should lead to",
      "choice_role": "weak"
    }}
  ],
  "evidence_facts": [
    {{
      "id": "snake_case_evidence_id",
      "title": "evidence title",
      "type": "one exact template name",
      "summary": "short summary",
      "why_it_may_matter": "why this evidence may matter",
      "support_role": "strong",
      "template": "one exact template name",
      "facts": {{}}
    }},
    {{
      "id": "snake_case_evidence_id",
      "title": "evidence title",
      "type": "one exact template name",
      "summary": "short summary",
      "why_it_may_matter": "why this evidence may matter",
      "support_role": "partial",
      "template": "one exact template name",
      "facts": {{}}
    }},
    {{
      "id": "snake_case_evidence_id",
      "title": "evidence title",
      "type": "one exact template name",
      "summary": "short summary",
      "why_it_may_matter": "why this evidence may matter",
      "support_role": "weak",
      "template": "one exact template name",
      "facts": {{}}
    }}
  ],
  "expected_outcomes": {{
    "strong_support": "what counts as strong evidence use in Turn 1",
    "partial_support": "what counts as partial evidence use in Turn 1",
    "weak_support": "what counts as weak evidence use in Turn 1",
    "unsupported": "what counts as unsupported evidence use in Turn 1"
  }}
}}

ABSOLUTE OUTPUT RULES:
- Return JSON only.
- No markdown.
- No explanation outside JSON.
- Generate exactly 4 actions.
- Generate exactly 3 evidence items.
- Action choice_role values must include one best, one partial, and at least one weak.
- The best action should be the most appropriate Turn 1 investigation choice.
- The partial action should be relevant but incomplete.
- The weak action should be plausible but not ideal for the first decision point.
- Evidence support_role values must be exactly one strong, one partial, and one weak.
- The strong evidence item should directly support the best Turn 1 action.
- The partial evidence item should be relevant but incomplete.
- The weak evidence item should be plausible but not enough to justify the best action.
- The 3 evidence items MUST use 3 different templates.
- Do not use cloudtrail more than once in the same turn.
- Do not use any evidence template more than once in the same turn.
- This is Turn 1, so the scenario should begin with detection, triage, or initial investigation.
- The learner must not see hidden truth directly.
- Use hidden truth only to keep the generated turn internally consistent.
- Keep the scenario realistic for AWS cloud incident response.
- The "type" field and "template" field MUST be identical.
- Allowed templates only: cloudtrail, iam_activity, guardduty, cloudwatch, access_key, billing.
- Never generate type="guardduty" with template="cloudtrail".
- Never generate type="iam_activity" with template="cloudtrail".
- Never generate type="cloudwatch" with template="cloudtrail".
- Never leave facts as {{}}.
- Never generate facts with all values as "Unknown".
- Never use camelCase key "whyItMayMatter".
- Always use snake_case key "why_it_may_matter".
- Every evidence item must contain realistic AWS-style facts that the screenshot template can display directly.
- Include one best evidence item, one partial evidence item, and one weaker or distracting evidence item.
- Include one best action, one reasonable partial action, and at least one weak or distracting action.
- Write Turn 1 briefing and known_context as incident-facing information, not report/meta text.
- Never write phrases like "the learner is aware", "the learner understands", "the learner knows", or "ACSE-Eval" in learner-visible briefing, known_context, actions, or coach_guidance.
- The briefing should introduce the incident situation in-world: what alert/signal appeared, what account/environment is affected, and what the learner should investigate first.
- known_context should list concrete known incident facts, such as suspicious IAM activity, CloudTrail or audit visibility, source IP/authentication clues, access key or role risk, and production account context.
- coach_guidance should give a practical first-step investigation hint, not say "choose the first action".

{SUPPORT_ROLE_CONTENT_RULES}
STRICT IAM ACTIVITY RULES:
- If an evidence item uses template="iam_activity", it MUST have at least 1 activity_rows entry.
- Include only the rows that fit the item's support_role; partial and weak timelines should be short.
- Each row MUST have exactly 4 values:
  [time, principal, action, source_ip]
- Do not generate rows like ["Unknown", "Unknown", "StopLogging", "Unknown"].
- Good IAM actions include:
  ConsoleLogin, GetCallerIdentity, ListTrails, StopLogging, CreateAccessKey,
  ListAccessKeys, AttachUserPolicy, PutUserPolicy, UpdateTrail, DeleteTrail.

STRICT GUARD DUTY RULES:
- If an evidence item uses template="guardduty", finding_type must be realistic.
- Prefer one of:
  UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration
  Recon:IAMUser/MaliciousIPCaller
  Stealth:IAMUser/CloudTrailLoggingDisabled
  UnauthorizedAccess:IAMUser/TorIPCaller
- Do not use vague names like "Potential Threat Detection".

STRICT CLOUDWATCH RULES:
- If an evidence item uses template="cloudwatch", it MUST have log_rows.
- Include only the log rows justified by the intended evidence. Do not pad the table with suspicious events to reach a minimum row count.
- Preserve the intended gap for partial/weak evidence; do not provide a complete identity activity chain and label it partial merely because it uses CloudWatch.
- The query must be a query expression, not a claim that an attack was confirmed. matched_records must be consistent with the supplied rows; retain a normal/OK signal when appropriate.
- Each row must have exactly 3 values:
  [timestamp, log_stream, message]
- Messages must look like logs, not essay explanations.

STRICT ACCESS KEY RULES:
- If an evidence item uses template="access_key", access_key_id must look like an AWS access key ID.
- It should start with AKIA and be followed by realistic uppercase letters or numbers.
- owner, source_ip, last_used_time, last_used_service, and status must not be Unknown if scenario context contains enough information.

STRICT BILLING RULES:
- If an evidence item uses template="billing", use currency-like and percentage-like values.
- Use realistic AWS services such as EC2, S3, Lambda, CloudTrail, GuardDuty, IAM, or VPC.

REFERENCE SCHEMAS:
{EVIDENCE_TEMPLATE_SCHEMAS}
"""

    if evidence_brief.strip():
        prompt += evidence_brief + """
These fixed items override the evidence rules above: return each item's facts as {}.
"""

    if retry_instruction.strip():
        prompt += f"""
RETRY INSTRUCTION:
The previous Turn 1 was rejected. Fix this problem and regenerate the whole turn:
{retry_instruction.strip()}
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You generate strict JSON for the first playable turn of a cloud incident "
                "response training simulator. Return JSON only. Evidence facts must be "
                "concrete, scenario-specific, realistic, and renderable."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    max_new_tokens = int(
        os.getenv(
            "HF_INITIAL_TURN_MAX_NEW_TOKENS",
            os.getenv("HF_NEXT_TURN_MAX_NEW_TOKENS", "2600"),
        )
    )

    output_text = _generate_coach_text(tokenizer, model, device, input_text, max_new_tokens)

    parsed = _parse_next_turn_output(output_text)
    parsed = _normalise_initial_turn_output(parsed, fixed_evidence=bool(evidence_brief.strip()))

    if evidence_brief.strip():
        # The caller replaces evidence with the timeline items; only text is kept.
        return parsed

    repaired = repair_next_turn_template_facts(parsed)
    repaired = _ensure_unique_evidence_templates(
        generated_turn=repaired,
        scenario_values=scenario_values,
        next_turn_number=1,
    )

    return repair_next_turn_template_facts(repaired)


def _normalise_initial_turn_output(parsed: dict[str, Any], fixed_evidence: bool = False) -> dict[str, Any]:
    """
    Applies lightweight schema cleanup to the Coach-generated first turn.

    This does not invent scenario meaning. It only fixes formatting issues that
    would break the runtime or evidence generator.
    """

    turn_config = parsed.get("turn_config")

    if isinstance(turn_config, dict):
        turn_config["turn"] = 1
        _repair_initial_turn_config(turn_config)

    # The hardcoded baseline replaces actions and evidence; timeline evidence
    # needs actions written for it instead.
    if not fixed_evidence:
        _repair_initial_turn_action_evidence_alignment(parsed)

    normalise_action_choice_roles(parsed.get("actions"))

    evidence_facts = parsed.get("evidence_facts")

    if isinstance(evidence_facts, list):
        normalise_evidence_support_roles(evidence_facts)

        for item in evidence_facts:
            if not isinstance(item, dict):
                continue

            if "why_it_may_matter" not in item and "whyItMayMatter" in item:
                item["why_it_may_matter"] = item.pop("whyItMayMatter")

            template = item.get("template") or item.get("type")

            if isinstance(template, str):
                template = template.strip().lower()
                item["template"] = template
                item["type"] = template

    return parsed


def _repair_initial_turn_config(turn_config: dict[str, Any]) -> None:
    """
    Repairs weak Turn 1 learner-facing copy produced during dataset preparation.

    The initial turn must feel like an incident handover, not a report about
    what the learner knows. This keeps the prepared scenario usable even when
    the local model generates generic educational wording.
    """

    # Both coach backends sometimes narrate ("The learner is alerted to...") instead of
    # addressing the reader. Rewriting keeps the scenario's facts; the fallback below
    # used to replace them with identity-management text in every scenario.
    for key in ("briefing", "coach_guidance"):
        if isinstance(turn_config.get(key), str):
            turn_config[key] = address_the_reader(turn_config[key])
    if isinstance(turn_config.get("known_context"), list):
        turn_config["known_context"] = [address_the_reader(str(item)) for item in turn_config["known_context"]]

    briefing = str(turn_config.get("briefing", "")).strip()
    known_context = turn_config.get("known_context")
    coach_guidance = str(turn_config.get("coach_guidance", "")).strip()

    combined_context = " ".join(str(item) for item in known_context) if isinstance(known_context, list) else ""
    weak_text = " ".join([briefing, combined_context, coach_guidance]).lower()

    weak_turn_copy = (
        not briefing
        or "acse-eval" in weak_text
        or "choose the first action" in weak_text
        or "select appropriate initial evidence" in weak_text
    )

    if weak_turn_copy:
        # Scenario-neutral: this runs for every scenario, not only identity management.
        turn_config["phase"] = turn_config.get("phase") or "Detection and Initial Triage"
        turn_config["briefing"] = (
            "A security alert in this AWS account needs investigation. Start by finding which "
            "identity acted, where the activity came from, and what the available evidence shows."
        )
        turn_config["known_context"] = [
            "The alert concerns activity in a production AWS account.",
            "Each evidence item for this turn comes from a different AWS source.",
            "The first response decision should rest on evidence that shows who acted, from where and when.",
        ]
        turn_config["coach_guidance"] = (
            "Start with the evidence that shows the suspicious activity itself before moving "
            "to containment or recovery decisions."
        )


_LEARNER = re.compile(r"\b(the learner's|the learner is|the learner was|the learner has|the learner)\b", re.IGNORECASE)
_YOU_VERB = re.compile(r"\b(You|you) ([a-z]+)\b")
_IRREGULAR = {"is": "are", "was": "were", "has": "have", "does": "do", "goes": "go"}


def address_the_reader(text: str) -> str:
    """'The learner is alerted to X' -> 'You are alerted to X'; 'the learner's' -> 'your'."""

    def replace(match: re.Match) -> str:
        phrase = match.group(1).lower()
        word = {"the learner's": "your", "the learner is": "you are", "the learner was": "you were",
                "the learner has": "you have", "the learner": "you"}[phrase]
        return word.capitalize() if match.group(1)[0].isupper() else word

    rewritten = _LEARNER.sub(replace, text)
    if rewritten == text:
        return text

    def verb(match: re.Match) -> str:
        subject, word = match.groups()
        if word in _IRREGULAR:
            word = _IRREGULAR[word]
        elif word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3 and word not in {"always", "thus"}:
            word = word[:-1]
        return f"{subject} {word}"

    return _YOU_VERB.sub(verb, rewritten)


def _repair_initial_turn_action_evidence_alignment(parsed: dict[str, Any]) -> None:
    """
    Keeps Turn 1 action and evidence wording coherent for the polished
    identity-management scenario.

    Local models sometimes generate an "Inspect Audit Logs" action but then
    label the matching evidence as a vague access attempt or unrelated public
    endpoint issue. For the first turn, a deterministic baseline is better than
    confusing the learner before the adaptive flow has started.
    """

    actions = parsed.get("actions")
    evidence_facts = parsed.get("evidence_facts")

    if not isinstance(actions, list) or not isinstance(evidence_facts, list):
        return

    combined = json.dumps({"actions": actions, "evidence_facts": evidence_facts}).lower()
    weak_alignment = (
        "inspect audit logs" in combined
        or "audit logs" in combined
        or "unencrypted method" in combined
        or "public endpoints" in combined
        or "john doe" in combined
        or "iam identity center" in combined
    )

    if not weak_alignment:
        return

    parsed["actions"] = [
        {
            "id": "inspect_cloudtrail_audit_event",
            "title": "Inspect CloudTrail Audit Event",
            "description": "Review the CloudTrail event record to identify the IAM action, principal, source IP, MFA status, and event time.",
            "recommended_next_focus": "Confirm whether the event indicates suspicious identity activity before choosing containment steps.",
            "choice_role": "best",
        },
        {
            "id": "correlate_iam_activity",
            "title": "Correlate IAM Activity",
            "description": "Compare related IAM actions around the incident window to see whether the same principal and source IP appear repeatedly.",
            "recommended_next_focus": "Look for login, logging, and credential events that suggest account misuse.",
            "choice_role": "partial",
        },
        {
            "id": "review_guardduty_signal",
            "title": "Review GuardDuty Signal",
            "description": "Check whether GuardDuty reports a threat signal that supports the identity-risk investigation.",
            "recommended_next_focus": "Use the finding as supporting context, then verify it against audit events.",
            "choice_role": "weak",
        },
    ]

    parsed["evidence_facts"] = [
        {
            "id": "cloudtrail_audit_event",
            "title": "CloudTrail Audit Event",
            "type": "cloudtrail",
            "summary": "CloudTrail records a ConsoleLogin event without MFA from the suspicious source IP.",
            "why_it_may_matter": "This is the strongest initial evidence because it shows the identity, source IP, time, MFA status, and API event in one audit record.",
            "support_role": "strong",
            "template": "cloudtrail",
            "facts": {
                "event_name": "ConsoleLogin",
                "event_source": "signin.amazonaws.com",
                "user": "arn:aws:iam::123456789012:user/admin-test",
                "source_ip": "185.220.101.42",
                "mfa": "false",
                "event_time": "2023-10-01T10:12:10Z",
                "region": "ap-southeast-1",
                "error_code": "-",
                "risk_signal": "Console login without MFA from a repeated source IP starts the identity investigation.",
                "event_id": "c0a12345-6789-4abc-def0-1234567890ab",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "recipient_account_id": "123456789012",
                "request_parameters": "ConsoleLogin, MFAUsed=false",
                "related_events": [
                    ["2023-10-01T10:12:10Z", "ConsoleLogin", "arn:aws:iam::123456789012:user/admin-test", "185.220.101.42", "MFA false"],
                    ["2023-10-01T10:18:44Z", "ListTrails", "arn:aws:iam::123456789012:user/admin-test", "185.220.101.42", "Success"],
                    ["2023-10-01T10:21:35Z", "StopLogging", "arn:aws:iam::123456789012:user/admin-test", "185.220.101.42", "Success"],
                    ["2023-10-01T10:24:02Z", "CreateAccessKey", "arn:aws:iam::123456789012:user/admin-test", "185.220.101.42", "Success"],
                ],
            },
        },
        {
            "id": "iam_activity_timeline",
            "title": "IAM Activity Timeline",
            "type": "iam_activity",
            "summary": "IAM activity shows read-only identity calls by the principal, without the login record itself.",
            "why_it_may_matter": "This shows the principal was active, but it lacks the audit record needed to judge the login.",
            "support_role": "partial",
            "template": "iam_activity",
            "facts": {
                "principal": "arn:aws:iam::123456789012:user/admin-test",
                "source_ip": "185.220.101.42",
                "mfa": "false",
                "policy_change": "No policy changes in window",
                "access_key_status": "Active key present",
                "risk_flags": ["No MFA device assigned"],
                "activity_rows": [
                    ["2023-10-01T10:13:02Z", "arn:aws:iam::123456789012:user/admin-test", "GetCallerIdentity", "185.220.101.42"],
                    ["2023-10-01T10:14:37Z", "arn:aws:iam::123456789012:user/admin-test", "ListAttachedUserPolicies", "185.220.101.42"],
                ],
            },
        },
        {
            "id": "access_key_metadata",
            "title": "Access Key Metadata",
            "type": "access_key",
            "summary": "Access key metadata shows an active key for the principal, last used days before the alert.",
            "why_it_may_matter": "This is credential inventory, not a record of the activity under investigation.",
            "support_role": "weak",
            "template": "access_key",
            "facts": {
                "access_key_id": "AKIA4Z7EXAMPLE31Q",
                "owner": "arn:aws:iam::123456789012:user/admin-test",
                "status": "Active",
                "last_used_service": "s3.amazonaws.com",
                "last_used_region": "ap-southeast-1",
                "last_used_time": "2023-09-28T08:14:52Z",
                "source_ip": "203.0.113.25",
            },
        },
    ]


@isolated_model_call
def evaluate_generated_turn_quality(
    scenario_config: dict[str, Any],
    hidden_truth: dict[str, Any],
    current_state: dict[str, Any],
    completed_turn: dict[str, Any],
    generated_turn: dict[str, Any],
    selected_action: dict[str, Any],
    selected_evidence: dict[str, Any],
    vlm_output: dict[str, Any],
    security_output: dict[str, Any],
    coach_output: dict[str, Any],
    next_turn_number: int,
    attempt_number: int,
) -> dict[str, Any]:
    """
    Uses the coach model as an AI quality judge for generated scenario turns.

    The judge checks whether the generated turn:
    - progresses or refocuses the scenario
    - reacts to the learner's previous action/evidence
    - avoids repeated turn content
    - contains useful renderable evidence facts

    This function does not save or modify the generated turn.
    It returns a JSON quality review used by scenario_engine/turn_orchestrator.py.
    """

    model, tokenizer = _load_coach_model()
    device = _device()

    prompt = f"""
You are the AI quality judge for CloudIR Trainer.

Your job is to review a generated next turn before it is saved.

You must judge whether the generated turn is good enough for a cloud incident response training simulator.

SCENARIO CONFIG:
{json.dumps(scenario_config, indent=2)}

HIDDEN TRUTH:
{json.dumps(hidden_truth, indent=2)}

CURRENT SCENARIO STATE:
{json.dumps(current_state, indent=2)}

COMPLETED TURN:
{json.dumps(completed_turn, indent=2)}

GENERATED TURN TO REVIEW:
{json.dumps(generated_turn, indent=2)}

LEARNER SELECTED ACTION:
{json.dumps(selected_action, indent=2)}

LEARNER SELECTED EVIDENCE:
{json.dumps(selected_evidence, indent=2)}

VISION-LANGUAGE MODEL OUTPUT:
{json.dumps(vlm_output, indent=2)}

SECURITY REASONING OUTPUT:
{json.dumps(security_output, indent=2)}

COACH FEEDBACK OUTPUT:
{json.dumps(coach_output, indent=2)}

NEXT TURN NUMBER:
{next_turn_number}

ATTEMPT NUMBER:
{attempt_number}

Return ONLY valid JSON with exactly these keys:
{{
  "pass": true,
  "quality_score": 0,
  "issues": [
    "issue 1",
    "issue 2"
  ],
  "retry_instruction": "specific instruction to improve the next generation if this review fails"
}}

QUALITY SCORING RULES:
- quality_score must be an integer from 0 to 100.
- pass should be true only if quality_score is 70 or higher and no critical issue exists.
- Score high if the generated turn clearly progresses or refocuses the scenario.
- Score high if the generated turn reacts to the learner's selected action, selected evidence, security verdict, and coach feedback.
- Score high if the generated turn uses realistic, concrete AWS-style evidence facts.
- Score high if the generated turn is consistent with hidden truth without revealing it directly.
- Score low if the generated turn repeats the completed turn's phase, briefing, action focus, evidence titles, evidence purpose, or expected outcomes.
- Score low if Turn {next_turn_number} feels like a generic continuation instead of a meaningful next scenario step.
- Score low if evidence facts are mostly empty, vague, Unknown, duplicated, or not renderable.
- Score low if the generated evidence would produce screenshots that are visually different but semantically useless.
- Score low if the evidence set does not include one clear strong item, one partial item, and one weak or distracting item.
- Score low if the generated turn ignores the learner's previous action/evidence choice.
- Score low if the generated turn reveals hidden truth directly to the learner.
- Score low if the phase label changes but the actual briefing, known context, evidence purpose, and expected outcomes still describe the same investigation goal as the completed turn.
- Score low if the generated turn only says to continue investigating the same event without adding a specific new purpose such as scope, correlation, credential risk, detection gap, impact, response, recovery, escalation, or final decision-making.
- Score low if the evidence titles are different but the evidence facts still revolve around the same user, same source IP, same timestamp, same API event, and same conclusion without introducing a new analytical purpose.

SCENARIO PROGRESSION STRICTNESS:
- Turn 1 may focus on detection, initial investigation, and confirming the first suspicious signal.
- Turn 2 should not feel like another initial investigation. It should move into scope, correlation, affected identity, credential risk, monitoring gaps, related activity, or investigation refinement.
- Turn 3 should not repeat Turn 1 or Turn 2's investigation goal. In a five-turn run it should usually move toward scope, blast-radius, monitoring, detection, or impact review unless the learner's previous choice was weak or unsupported.
- Turn 4 should usually move toward containment or remediation.
- Turn 5 should usually move toward recovery verification, final reporting, residual risk, or debrief evidence.
- If a later turn uses the same main event, same user, same source IP, and same timestamp as previous turns, it must introduce a clearly different purpose, such as correlation, credential exposure, detection gap, blast radius, containment, recovery, or business impact.
- Do not pass a generated turn just because the evidence titles are slightly different. Judge whether the scenario goal and evidence purpose are meaningfully different.
- If the generated turn says "continue investigating the CloudTrail logging disablement" without a more specific new goal, score it below 70.
- If Turn 2 marks a CloudTrail review action as best after a strong Turn 1 CloudTrail choice, score it below 70.
- If a later turn marks a generic CloudTrail review action as best after a strong or partial previous choice, score it below 70.
- If a later turn's strong evidence repeats the same CloudTrail purpose instead of matching the target stage, score it below 70.
- If the generated turn describes any later turn as if it is still the initial investigation, score it below 70.
- If Turn 3 does not move toward scope/blast-radius or Turn 4/5 do not move toward containment/recovery after a strong or partial previous choice, score it below 70.

EVIDENCE QUALITY STRICTNESS:
- Evidence must be useful for reasoning, not just visually renderable.
- Evidence support_role values should be exactly one strong, one partial, and one weak.
- Check the actual facts against the intended best action: a partial item needs a specific missing link relevant to that action, and a weak item must not already show direct activity. Reject labels assigned merely from template type or list position; request revised content, not just relabelled answers.
- The strong evidence must directly support the best action; the partial evidence must be relevant but incomplete; the weak evidence must be plausible but insufficient.
- Evidence should contain enough concrete fields for the VLM and Security Reasoning AI to evaluate the learner's selected action.
- IAM activity evidence should contain multiple timeline rows, not only one repeated event.
- CloudWatch evidence should contain multiple log rows, not a single generic log message.
- Access key evidence should include concrete owner, status, key ID, last-used time, service, region, and source IP when the scenario context contains those values.
- GuardDuty evidence should include a concrete finding type, severity, principal or resource, remote IP, first seen time, and summary.
- CloudTrail evidence should not simply repeat the same StopLogging event every turn unless it serves a different investigation purpose.
- Billing evidence should include concrete cost, previous average, largest service, region, and change values.
- Score below 70 if the evidence is technically valid JSON but too shallow for the learner to make a meaningful decision.
- Score below 70 if the evidence would lead to screenshots that look different but communicate the same information as the completed turn.
- Score below 70 if all three evidence items would reasonably receive the same support verdict.

CRITICAL FAILURE CONDITIONS:
- The generated turn repeats the same evidence titles and evidence facts from the completed turn.
- The generated turn has facts that are mostly Unknown.
- The generated turn does not contain a meaningful new investigation, response, impact, recovery, escalation, or refocus goal.
- The generated turn is inconsistent with the scenario.
- The generated turn exposes hidden truth directly to the learner.
- Turn 2 or Turn 3 has a phase label that changes, but the briefing, evidence purpose, known context, and expected outcomes still feel like the same initial investigation.
- Turn 3 does not move toward a response, impact, escalation, recovery, containment, or final decision-making step when the previous learner choice was strong or partially supported.
- The generated turn repeats the same main event facts from the completed turn without introducing a new learning purpose or decision point.
- The generated evidence is mostly a shallow restatement of previous evidence, even if the titles or templates are slightly different.
- The generated evidence lacks a clear strong/partial/weak evidence-quality spread.

RETRY INSTRUCTION RULES:
- If pass is false, retry_instruction must be specific and actionable.
- Do not write vague retry instructions like "make it better".
- Mention exactly what should change, such as:
  - change the phase goal
  - use different evidence templates
  - avoid repeated evidence titles
  - add concrete AWS facts
  - react to the learner's weak/partial/strong choice
  - move toward the next scenario_progression stage
  - make Turn 2 focus on scope, correlation, credential risk, monitoring gaps, or related activity
  - make Turn 3 focus on response, containment, recovery, impact review, escalation, or final decision-making
  - make evidence facts richer instead of repeating the same user/IP/timestamp/event
- If pass is true, retry_instruction should be an empty string.

IMPORTANT:
- Do not judge based on perfect cybersecurity correctness.
- Judge whether the generated turn is useful, coherent, non-repetitive, and suitable for the training simulator.
- Do not include markdown.
- Return JSON only.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict AI quality judge for generated cybersecurity training scenarios. "
                "Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    max_new_tokens = int(os.getenv("HF_QUALITY_JUDGE_MAX_NEW_TOKENS", "650"))

    output_text = _generate_coach_text(tokenizer, model, device, input_text, max_new_tokens)

    return _parse_quality_judge_output(output_text)


def _extract_scenario_values(
    scenario_config: dict[str, Any],
    hidden_truth: dict[str, Any],
    current_state: dict[str, Any],
    completed_turn: dict[str, Any],
    selected_action: dict[str, Any],
    selected_evidence: dict[str, Any],
    vlm_output: dict[str, Any],
    security_output: dict[str, Any],
    coach_output: dict[str, Any],
) -> dict[str, str]:
    """
    Extracts concrete values from the current scenario context so the model
    generates scenario-based evidence instead of vague placeholders.
    """

    context_blob = json.dumps(
        {
            "scenario_config": scenario_config,
            "hidden_truth": hidden_truth,
            "current_state": current_state,
            "completed_turn": completed_turn,
            "selected_action": selected_action,
            "selected_evidence": selected_evidence,
            "vlm_output": vlm_output,
            "security_output": security_output,
            "coach_output": coach_output,
        },
        ensure_ascii=False,
    )

    lower_blob = context_blob.lower()

    source_ip = _find_first_ip(context_blob) or "185.220.101.42"
    if source_ip.startswith("192.0.2."):
        source_ip = "185.220.101.42"

    region = _find_first_region(context_blob) or "ap-southeast-1"
    if region == "us-east-1":
        region = "ap-southeast-1"

    event_time = _find_first_time(context_blob) or "2023-10-01T10:21:35Z"

    user = _find_first_user(context_blob) or "admin-test"
    if user.lower() in ["users", "username"] or "username" in user.lower():
        user = "arn:aws:iam::123456789012:user/admin-test"

    event_name = "StopLogging"
    if "createaccesskey" in lower_blob and "stoplogging" not in lower_blob:
        event_name = "CreateAccessKey"
    elif "deletebucket" in lower_blob:
        event_name = "DeleteBucket"
    elif "putuserpolicy" in lower_blob:
        event_name = "PutUserPolicy"
    elif "attachuserpolicy" in lower_blob:
        event_name = "AttachUserPolicy"
    elif "stoplogging" in lower_blob:
        event_name = "StopLogging"

    mfa = "false"
    if '"mfa": "true"' in lower_blob or "mfa true" in lower_blob or "mfa used true" in lower_blob:
        mfa = "true"

    event_source = _event_source_for_event(event_name)

    return {
        "event_name": event_name,
        "event_source": event_source,
        "user": user,
        "source_ip": source_ip,
        "mfa": mfa,
        "event_time": event_time,
        "region": region,
        "user_agent": "signin.amazonaws.com" if event_name == "ConsoleLogin" else "aws-cli/2.13.7 Python/3.11",
        "account_id": "123456789012",
        "access_key_id": "AKIA4Z7EXAMPLE92K",
        "account": "Production",
    }


def _event_source_for_event(event_name: str) -> str:
    event = event_name.lower()

    if event == "consolelogin":
        return "signin.amazonaws.com"

    if event in ["assumerole", "getcalleridentity"]:
        return "sts.amazonaws.com"

    if event in ["createaccesskey", "listaccesskeys", "attachuserpolicy", "putuserpolicy"]:
        return "iam.amazonaws.com"

    if event in ["stoplogging", "listtrails", "updatetrail", "deletetrail"]:
        return "cloudtrail.amazonaws.com"

    return "iam.amazonaws.com"


def _build_scenario_progression_guidance(
    scenario_config: dict[str, Any],
    current_state: dict[str, Any],
    completed_turn: dict[str, Any],
    selected_action: dict[str, Any],
    selected_evidence: dict[str, Any],
    security_output: dict[str, Any],
    coach_output: dict[str, Any],
    next_turn_number: int,
) -> str:
    """
    Builds adaptive scenario progression guidance from scenario_config.json.

    This avoids hardcoding one scenario path in the app while still forcing the
    generated turns to evolve instead of repeating the same investigation.
    """

    scenario_progression = scenario_config.get("scenario_progression", {})
    evidence_template_strategy = scenario_config.get("evidence_template_strategy", {})
    stages = scenario_progression.get("incident_response_stages", [])

    target_stage = {}
    if isinstance(stages, list) and stages:
        stage_index = min(max(next_turn_number - 1, 0), len(stages) - 1)
        target_stage = stages[stage_index]

    completed_turn_config = _get_completed_turn_config(completed_turn)
    completed_phase = completed_turn_config.get("phase", "Unknown previous phase")
    completed_briefing = completed_turn_config.get("briefing", "")

    previous_action_titles = _extract_titles(completed_turn, "actions")
    previous_evidence_titles = _extract_titles(completed_turn, "evidence_facts")
    if not previous_evidence_titles:
        previous_evidence_titles = _extract_titles(completed_turn, "evidenceFacts")

    verdict = str(security_output.get("verdict", "Unknown"))
    selected_action_title = str(selected_action.get("title", selected_action.get("id", "Unknown action")))
    selected_evidence_title = str(selected_evidence.get("title", selected_evidence.get("id", "Unknown evidence")))

    feedback = str(coach_output.get("feedback", ""))
    next_turn_guidance = str(coach_output.get("next_turn_guidance", ""))

    return f"""
Progression mode:
{scenario_progression.get("progression_mode", "ai_guided_adaptive")}

Turn strategy:
{scenario_progression.get("turn_strategy", "Each turn must move the incident forward based on the learner's previous action, evidence choice, and evaluation verdict.")}

Target stage for Turn {next_turn_number}:
{json.dumps(target_stage, indent=2)}

Current scenario state:
{json.dumps(current_state, indent=2)}

Completed phase:
{completed_phase}

Completed briefing:
{completed_briefing}

Previous action titles:
{json.dumps(previous_action_titles, indent=2)}

Previous evidence titles:
{json.dumps(previous_evidence_titles, indent=2)}

Learner selected action:
{selected_action_title}

Learner selected evidence:
{selected_evidence_title}

Security verdict:
{verdict}

Coach feedback:
{feedback}

Coach next-turn guidance:
{next_turn_guidance}

Adaptation rules from scenario_config:
{json.dumps(scenario_progression.get("adaptation_rules", []), indent=2)}

Mandatory progression requirements:
- Use the target stage as the main direction for this generated turn.
- Do not copy the completed phase name unless the learner's previous choice was weak or unsupported and the scenario needs correction.
- Do not repeat the same briefing goal from the completed turn.
- Do not repeat the same action titles from the completed turn.
- Do not repeat the same evidence titles from the completed turn.
- If the verdict is Strong Support, advance the scenario into deeper investigation, response planning, impact assessment, escalation, or recovery depending on the target stage.
- If the verdict is Partial Support, advance the scenario but include uncertainty and missing evidence.
- If the verdict is Weak Support or Unsupported, refocus the learner toward the missing investigation path without exposing hidden truth directly.
- The generated turn must feel like a consequence of the learner's previous choice, not a generic next page.
"""


def _build_evidence_plan(
    next_turn_number: int,
    scenario_values: dict[str, str],
    scenario_config: dict[str, Any],
    current_state: dict[str, Any],
    completed_turn: dict[str, Any],
    selected_action: dict[str, Any],
    selected_evidence: dict[str, Any],
    security_output: dict[str, Any],
) -> str:
    """
    Gives the model adaptive evidence guidance without hardcoding a single
    scenario path such as "Turn 3 must always be containment".

    The model still generates the actual evidence content, but this guidance
    reduces repeated evidence across turns.
    """

    user = scenario_values["user"]
    source_ip = scenario_values["source_ip"]
    event_time = scenario_values["event_time"]
    region = scenario_values["region"]
    event_name = scenario_values["event_name"]

    allowed_templates = scenario_config.get(
        "allowed_evidence_templates",
        ["cloudtrail", "iam_activity", "guardduty", "cloudwatch", "access_key", "billing"],
    )

    scenario_progression = scenario_config.get("scenario_progression", {})
    evidence_template_strategy = scenario_config.get("evidence_template_strategy", {})
    stages = scenario_progression.get("incident_response_stages", [])

    target_stage = {}
    if isinstance(stages, list) and stages:
        stage_index = min(max(next_turn_number - 1, 0), len(stages) - 1)
        target_stage = stages[stage_index]

    previous_evidence_titles = _extract_titles(completed_turn, "evidence_facts")
    if not previous_evidence_titles:
        previous_evidence_titles = _extract_titles(completed_turn, "evidenceFacts")

    previous_evidence_templates = _extract_evidence_templates(completed_turn)
    turn_role_templates = {}

    if isinstance(evidence_template_strategy, dict):
        role_templates_by_turn = evidence_template_strategy.get("role_templates_by_turn", {})

        if isinstance(role_templates_by_turn, dict):
            turn_role_templates = (
                role_templates_by_turn.get(str(next_turn_number))
                or role_templates_by_turn.get(next_turn_number)
                or {}
            )

            if not isinstance(turn_role_templates, dict):
                turn_role_templates = {}

    verdict = str(security_output.get("verdict", "Unknown")).lower()
    selected_action_title = str(selected_action.get("title", selected_action.get("id", "Unknown action")))
    selected_evidence_title = str(selected_evidence.get("title", selected_evidence.get("id", "Unknown evidence")))

    if "strong" in verdict:
        adaptation_direction = (
            "The learner made a strong evidence-based choice. Generate evidence that advances the scenario "
            "to a deeper stage rather than repeating the same investigation."
        )
    elif "partial" in verdict:
        adaptation_direction = (
            "The learner made a partially supported choice. Generate evidence that progresses the scenario "
            "but includes uncertainty, correlation gaps, or missing confirmation."
        )
    elif "weak" in verdict or "unsupported" in verdict:
        adaptation_direction = (
            "The learner made a weak or unsupported choice. Generate evidence that refocuses the learner "
            "toward the missing investigation path without revealing the hidden truth directly."
        )
    else:
        adaptation_direction = (
            "Generate evidence that moves the scenario forward while staying consistent with the learner's previous choice."
        )

    return f"""
Allowed evidence templates:
{json.dumps(allowed_templates, indent=2)}

Target stage for this turn:
{json.dumps(target_stage, indent=2)}

Learner selected action:
{selected_action_title}

Learner selected evidence:
{selected_evidence_title}

Previous evidence titles:
{json.dumps(previous_evidence_titles, indent=2)}

Previous evidence templates:
{json.dumps(previous_evidence_templates, indent=2)}

Scenario evidence strategy for this turn:
{json.dumps(turn_role_templates, indent=2)}

Security verdict:
{security_output.get("verdict", "Unknown")}

Adaptive evidence direction:
{adaptation_direction}

Evidence generation requirements:
- Generate exactly 3 evidence items.
- The 3 evidence items MUST use 3 different templates.
- Do not use any evidence template more than once in the same turn.
- Use templates only from the allowed evidence templates list.
- Prefer using at least 2 templates that were not the main focus of the previous turn.
- Do not repeat the exact same evidence titles from the completed turn.
- Do not repeat the exact same evidence purpose from the completed turn.
- Choose evidence that supports the target stage and reacts to the learner's previous selected action and evidence.
- Evidence should reveal information gradually and must not expose hidden truth directly.
- Evidence should include one strong item, one partial item, and one weaker or distracting item.
- Assign support_role so the evidence set contains exactly one strong, one partial, and one weak item.
- Assign roles by what the concrete facts justify for the intended action, not by template or position. For partial evidence, leave a specific link needed by that action unknown; for weak evidence, provide adjacent context without direct activity. Do not fill these gaps with invented events or identities.
- For CloudWatch, provide only intended log_rows, a neutral query expression and an accurate matched_records count. One or two rows are allowed. Do not add suspicious events just to fill a table, and preserve normal/OK signals.
- If the scenario evidence strategy names templates for this turn, use those templates for the matching support roles whenever possible.
- Evidence must be screenshot-renderable using the available templates.
- Evidence must reuse scenario values where relevant:
  user={user}
  source_ip={source_ip}
  event_time={event_time}
  region={region}
  event_name={event_name}

Template selection guidance:
- Use cloudtrail when the learner needs direct API event evidence.
- Use iam_activity when the learner needs a timeline of identity activity.
- Use guardduty when the learner needs threat-signal or malicious-source context.
- Use cloudwatch when the learner needs monitoring, logging, or operational signals.
- Use access_key when the learner needs credential investigation or credential-risk evidence.
- Use billing when the learner needs impact, resource-spend, or business-impact evidence.
- Follow the scenario evidence strategy first when it defines strong, partial, and weak templates for this turn.
- For containment stages, prefer access_key plus either cloudtrail or cloudwatch as the strongest evidence path.
- For scope stages, guardduty can be strong when the learner needs threat-signal, severity, principal, source IP, or blast-radius evidence.

The model may choose the best 3 templates for this turn, but the chosen templates must clearly move or refocus the scenario.
"""


def _get_completed_turn_config(completed_turn: dict[str, Any]) -> dict[str, Any]:
    turn_config = completed_turn.get("turn_config")

    if isinstance(turn_config, dict):
        return turn_config

    turn_config = completed_turn.get("turnConfig")

    if isinstance(turn_config, dict):
        return turn_config

    return {}


def _extract_titles(payload: dict[str, Any], key: str) -> list[str]:
    items = payload.get(key, [])

    if not isinstance(items, list):
        return []

    titles: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        title = item.get("title")

        if isinstance(title, str) and title.strip():
            titles.append(title.strip())

    return titles


def _extract_evidence_templates(completed_turn: dict[str, Any]) -> list[str]:
    evidence_items = completed_turn.get("evidence_facts")

    if not isinstance(evidence_items, list):
        evidence_items = completed_turn.get("evidenceFacts")

    if not isinstance(evidence_items, list):
        return []

    templates: list[str] = []

    for item in evidence_items:
        if not isinstance(item, dict):
            continue

        template = item.get("template") or item.get("type")

        if isinstance(template, str) and template.strip():
            templates.append(template.strip())

    return templates


def _find_first_ip(text: str) -> str | None:
    match = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    return match.group(0) if match else None


def _find_first_region(text: str) -> str | None:
    match = re.search(r"\b[a-z]{2}-[a-z]+-\d\b", text)
    return match.group(0) if match else None


def _find_first_time(text: str) -> str | None:
    patterns = [
        r"\b\d{2}:\d{2}:\d{2}\s?UTC\b",
        r"\b\d{2}:\d{2}\s?UTC\b",
        r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?\s?UTC\b",
        r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?Z\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(0)

    return None


def _find_first_user(text: str) -> str | None:
    preferred_patterns = [
        r"\badmin-test\b",
        r"\b[a-zA-Z0-9._-]*admin[a-zA-Z0-9._-]*\b",
        r"\b[a-zA-Z0-9._-]*user[a-zA-Z0-9._-]*\b",
    ]

    for pattern in preferred_patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(0)

    return None


def _extract_json_from_output(output_text: str) -> dict[str, Any]:
    """
    Extracts JSON from a model response.

    Handles:
    - raw JSON
    - ```json fenced JSON
    - extra text before/after JSON by slicing from first { to last }
    """

    cleaned = output_text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").removesuffix("```").strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").removesuffix("```").strip()

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Model did not return valid JSON. Raw output was:\n\n"
            f"{output_text}"
        ) from exc


def _parse_coach_feedback_output(output_text: str) -> dict[str, Any]:
    try:
        parsed = _extract_json_from_output(output_text)

        required_keys = {
            "feedback",
            "next_turn_guidance",
        }

        missing = required_keys - set(parsed.keys())

        if missing:
            raise ValueError(f"Coach feedback output is missing keys: {missing}")

        return parsed

    except ValueError:
        feedback = ""
        next_turn_guidance = ""

        text = output_text.strip()

        if "### Feedback:" in text and "### Next Turn Guidance:" in text:
            feedback_part = text.split("### Feedback:", 1)[1]
            feedback = feedback_part.split("### Next Turn Guidance:", 1)[0].strip()
            next_turn_guidance = text.split("### Next Turn Guidance:", 1)[1].strip()

        elif "Feedback:" in text and "Next Turn Guidance:" in text:
            feedback_part = text.split("Feedback:", 1)[1]
            feedback = feedback_part.split("Next Turn Guidance:", 1)[0].strip()
            next_turn_guidance = text.split("Next Turn Guidance:", 1)[1].strip()

        else:
            feedback = text
            next_turn_guidance = "Continue by reviewing the next most relevant evidence before choosing a containment action."

        return {
            "feedback": feedback,
            "next_turn_guidance": next_turn_guidance,
        }


def _parse_final_debrief_output(
    output_text: str,
    compact_traces: list[dict[str, Any]],
) -> dict[str, Any]:
    parsed = _extract_json_from_output(output_text)

    required_keys = {
        "overall_assessment",
        "performance_level",
        "evidence_quality_score",
        "strengths",
        "missed_details",
        "risky_interpretations",
        "recommended_response",
        "learning_targets",
        "turn_debriefs",
    }

    missing = required_keys - set(parsed.keys())

    if missing:
        raise ValueError(f"Final debrief output is missing keys: {missing}")

    if not isinstance(parsed.get("turn_debriefs"), list):
        raise ValueError("Final debrief turn_debriefs must be a list.")

    if len(parsed["turn_debriefs"]) != len(compact_traces):
        raise ValueError(
            "Final debrief must contain one turn_debrief for each evaluation trace."
        )

    parsed.setdefault(
        "incident_summary",
        _fallback_incident_summary(compact_traces),
    )
    parsed.setdefault(
        "decision_scorecard",
        _fallback_decision_scorecard(compact_traces),
    )
    parsed.setdefault(
        "response_checklist",
        _fallback_response_checklist(compact_traces),
    )
    parsed.setdefault(
        "reflection_prompts",
        _fallback_reflection_prompts(),
    )
    parsed["generated_by"] = "coach_ai"

    return parsed


def build_fallback_final_debrief(
    *,
    evaluation_traces: list[dict[str, Any]],
    final_state: dict[str, Any],
    error: str = "",
) -> dict[str, Any]:
    strong_count = sum(
        1
        for trace in evaluation_traces
        if "strong" in clean_value(
            trace.get("security_output", {}).get("verdict")
        ).lower()
    )
    total_turns = len(evaluation_traces)
    performance_level = "Strong" if total_turns and strong_count == total_turns else "Developing"

    turn_debriefs = [
        _fallback_turn_debrief(trace)
        for trace in evaluation_traces
    ]

    missed_details = [
        item.get("what_was_missing")
        for item in turn_debriefs
        if item.get("what_was_missing")
    ]
    risky_interpretations = [
        item.get("risk_warning")
        for item in turn_debriefs
        if item.get("risk_warning")
    ]

    if not missed_details:
        missed_details = [
            "When giving spoken justification, explicitly separate visible screenshot facts from inferred incident impact.",
        ]

    if not risky_interpretations:
        risky_interpretations = [
            "Avoid treating one strong evidence item as proof that every affected resource has already been identified.",
        ]

    debrief = {
        "overall_assessment": (
            f"The learner completed {total_turns} evaluated turns with {strong_count} strong evidence verdicts. "
            f"The final state reached {final_state.get('containment', '--')}% containment, "
            f"{final_state.get('visibility', '--')}% visibility, and residual risk marked as "
            f"{final_state.get('risk', 'Unknown')}."
        ),
        "performance_level": performance_level,
        "evidence_quality_score": f"{strong_count}/{total_turns} strong",
        "incident_summary": _fallback_incident_summary(evaluation_traces),
        "decision_scorecard": _fallback_decision_scorecard(evaluation_traces),
        "strengths": _fallback_strengths(evaluation_traces),
        "missed_details": missed_details[:4],
        "risky_interpretations": risky_interpretations[:4],
        "recommended_response": _fallback_recommended_response(evaluation_traces),
        "response_checklist": _fallback_response_checklist(evaluation_traces),
        "reflection_prompts": _fallback_reflection_prompts(),
        "learning_targets": [
            "Name the exact visible fields that support each response action.",
            "Explain what the evidence does not prove before deciding containment.",
            "Connect detection, correlation, and response decisions across the full investigation path.",
        ],
        "turn_debriefs": turn_debriefs,
        "generated_by": "fallback",
    }

    if error:
        debrief["generation_error"] = error

    return debrief


def _compact_trace_for_final_debrief(trace: dict[str, Any]) -> dict[str, Any]:
    selected_action = trace.get("selected_action", {})
    selected_evidence = trace.get("selected_evidence", {})
    learner_justification = trace.get("learner_justification", {})
    vlm_output = trace.get("vlm_output", {})
    security_output = trace.get("security_output", {})
    coach_output = trace.get("coach_output", {})

    return {
        "turn": trace.get("turn"),
        "selected_action": {
            "title": selected_action.get("title"),
            "description": selected_action.get("description"),
            "choice_role": selected_action.get("choice_role") or selected_action.get("choiceRole"),
        },
        "selected_evidence": {
            "title": selected_evidence.get("title"),
            "type": selected_evidence.get("template") or selected_evidence.get("type"),
            "summary": selected_evidence.get("summary"),
            "support_role": selected_evidence.get("support_role") or selected_evidence.get("supportRole"),
        },
        "learner_transcript": learner_justification.get("text")
        or learner_justification.get("transcript")
        or learner_justification.get("typed_text")
        or learner_justification.get("typedText")
        or "",
        "vlm_summary": vlm_output.get("summary", ""),
        "visible_facts": vlm_output.get("visible_facts_extracted")
        or vlm_output.get("visible_facts")
        or [],
        "security_verdict": security_output.get("verdict", ""),
        "security_reasoning": security_output.get("reasoning", ""),
        "justification_assessment": security_output.get("justification_assessment", ""),
        "recommended_next_focus": security_output.get("recommended_next_focus", ""),
        "risk_of_wrong_interpretation": security_output.get("risk_of_wrong_interpretation", ""),
        "coach_feedback": coach_output.get("feedback", ""),
        "coach_next_turn_guidance": coach_output.get("next_turn_guidance", ""),
    }


def _fallback_turn_debrief(trace: dict[str, Any]) -> dict[str, Any]:
    compact = _compact_trace_for_final_debrief(trace)
    visible_facts = compact["visible_facts"]

    if not isinstance(visible_facts, list):
        visible_facts = [clean_value(visible_facts)]

    facts = [
        clean_value(fact)
        for fact in visible_facts
        if clean_value(fact)
    ]

    what_was_missing = clean_value(compact["justification_assessment"])

    if not what_was_missing:
        what_was_missing = (
            "The learner should explain which visible fields support the chosen action "
            "and which claims remain unproven."
        )

    return {
        "turn": compact["turn"],
        "action": compact["selected_action"].get("title") or "Unknown action",
        "evidence": compact["selected_evidence"].get("title") or "Unknown evidence",
        "verdict": compact["security_verdict"] or "Unknown",
        "what_went_well": compact["security_reasoning"] or compact["coach_feedback"],
        "what_was_missing": what_was_missing,
        "evidence_facts_used": facts[:6],
        "risk_warning": compact["risk_of_wrong_interpretation"],
        "next_time_improve": compact["recommended_next_focus"] or compact["coach_next_turn_guidance"],
    }


def _fallback_strengths(evaluation_traces: list[dict[str, Any]]) -> list[str]:
    strengths: list[str] = []

    for trace in evaluation_traces:
        compact = _compact_trace_for_final_debrief(trace)
        verdict = clean_value(compact["security_verdict"])
        action = clean_value(compact["selected_action"].get("title"))
        evidence = clean_value(compact["selected_evidence"].get("title"))

        if "strong" in verdict.lower() and action and evidence:
            strengths.append(
                f"Turn {compact['turn']} aligned {action} with {evidence}, producing a {verdict} verdict."
            )

    if not strengths:
        strengths.append("The learner completed the full investigation and produced evaluable evidence choices.")

    return strengths[:4]


def _fallback_recommended_response(evaluation_traces: list[dict[str, Any]]) -> str:
    combined_text = json.dumps(evaluation_traces, ensure_ascii=False).lower()

    if "access key" in combined_text or "createaccesskey" in combined_text:
        return (
            "Rotate or disable the suspicious access key, confirm the affected IAM principal, "
            "restore or verify CloudTrail visibility, and preserve the evidence trail for review."
        )

    if "stoplogging" in combined_text:
        return (
            "Investigate the StopLogging activity, verify audit logging is enabled, "
            "and correlate the same source IP across IAM and monitoring evidence."
        )

    return (
        "Continue correlating identity, monitoring, and impact evidence before closing the incident response path."
    )


def _fallback_incident_summary(evaluation_traces: list[dict[str, Any]]) -> dict[str, str]:
    compact_traces = [
        _compact_trace_for_final_debrief(trace)
        if "selected_action" in trace
        else trace
        for trace in evaluation_traces
    ]
    evidence_titles = [
        clean_value(trace.get("selected_evidence", {}).get("title"))
        for trace in compact_traces
        if clean_value(trace.get("selected_evidence", {}).get("title"))
    ]

    return {
        "what_happened": (
            "The scenario presented suspicious identity-management activity requiring evidence-backed "
            "triage, correlation, and response decision-making."
        ),
        "what_was_proven": (
            f"The learner selected evidence including {', '.join(evidence_titles)}."
            if evidence_titles
            else "The selected evidence was used to support the learner's response path."
        ),
        "what_remains_uncertain": (
            "The remaining uncertainty is whether all affected resources, credential usage paths, "
            "and downstream impact have been fully correlated."
        ),
    }


def _fallback_decision_scorecard(evaluation_traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verdicts = [
        clean_value(
            trace.get("security_output", {}).get("verdict")
            or trace.get("security_verdict")
        )
        for trace in evaluation_traces
    ]
    total_turns = len(verdicts) or 1
    strong_count = sum(1 for verdict in verdicts if "strong" in verdict.lower())
    evidence_score = round((strong_count / total_turns) * 100)

    return [
        {
            "label": "Evidence relevance",
            "score": evidence_score,
            "feedback": "How directly the selected evidence supported the chosen response action.",
        },
        {
            "label": "Field extraction",
            "score": 80 if evidence_score >= 90 else 68,
            "feedback": "How consistently the learner named concrete fields such as identity, source IP, event time, and status.",
        },
        {
            "label": "Avoiding overclaiming",
            "score": 75,
            "feedback": "Whether the reasoning separated visible evidence from impact that still needed correlation.",
        },
        {
            "label": "Response progression",
            "score": 82 if total_turns >= 3 else 65,
            "feedback": "Whether the investigation moved from detection toward correlation, containment, and final response.",
        },
    ]


def _fallback_response_checklist(evaluation_traces: list[dict[str, Any]]) -> list[str]:
    combined_text = json.dumps(evaluation_traces, ensure_ascii=False).lower()

    checklist = [
        "Preserve CloudTrail, IAM, GuardDuty, and CloudWatch evidence before making containment changes.",
        "Confirm the affected principal, source IP, event time, account, and region from the strongest evidence.",
        "Correlate identity activity with monitoring and service evidence to estimate blast radius.",
        "Document residual risk and identify post-containment monitoring checks.",
    ]

    if "access key" in combined_text or "createaccesskey" in combined_text:
        checklist.insert(
            2,
            "Disable or rotate the suspicious access key after recording owner and last-used details.",
        )

    if "stoplogging" in combined_text:
        checklist.insert(
            2,
            "Verify CloudTrail logging status and recover audit visibility if logging was stopped.",
        )

    return checklist[:6]


def _fallback_reflection_prompts() -> list[str]:
    return [
        "Which visible field most strongly justified your final response decision?",
        "What did the selected evidence not prove, even though it looked suspicious?",
        "What should be preserved before disabling credentials or changing IAM configuration?",
    ]


def _parse_next_turn_output(output_text: str) -> dict[str, Any]:
    parsed = _extract_json_from_output(output_text)

    required_keys = {
        "turn_config",
        "actions",
        "evidence_facts",
        "expected_outcomes",
    }

    missing = required_keys - set(parsed.keys())

    if missing:
        raise ValueError(f"Next-turn output is missing keys: {missing}")

    return parsed


def _parse_quality_judge_output(output_text: str) -> dict[str, Any]:
    try:
        parsed = _extract_json_from_output(output_text)
    except ValueError:
        return {
            "pass": False,
            "quality_score": 0,
            "issues": [
                "The AI quality judge did not return valid JSON."
            ],
            "retry_instruction": (
                "Regenerate the turn with a clearly different phase goal, non-repeated evidence, "
                "concrete AWS-style facts, and a stronger connection to the learner's previous choice."
            ),
        }

    required_keys = {
        "pass",
        "quality_score",
        "issues",
        "retry_instruction",
    }

    missing = required_keys - set(parsed.keys())

    if missing:
        return {
            "pass": False,
            "quality_score": 0,
            "issues": [
                f"The AI quality judge output is missing keys: {sorted(missing)}"
            ],
            "retry_instruction": (
                "Regenerate the turn with a clearly different phase goal, non-repeated evidence, "
                "concrete AWS-style facts, and a stronger connection to the learner's previous choice."
            ),
        }

    quality_score = parsed.get("quality_score")

    try:
        quality_score = int(quality_score)
    except (TypeError, ValueError):
        quality_score = 0

    quality_score = max(0, min(100, quality_score))

    issues = parsed.get("issues")

    if not isinstance(issues, list):
        issues = [str(issues)]

    retry_instruction = parsed.get("retry_instruction")

    if not isinstance(retry_instruction, str):
        retry_instruction = ""

    passed = bool(parsed.get("pass"))

    if quality_score < 70:
        passed = False

    return {
        "pass": passed,
        "quality_score": quality_score,
        "issues": issues,
        "retry_instruction": retry_instruction,
    }
