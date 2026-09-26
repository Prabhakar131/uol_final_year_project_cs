from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from functools import lru_cache
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from cloudir.ai_models.extraction_quality import security_observations
from cloudir.scenario.incident_timeline import SUGGESTED_EVENTS
from cloudir.ai_models.learner_grounding import (
    ground_justification, learner_statements, keep_evidence_only_prose,
)


load_dotenv()


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# "llama_cpp" runs a quantized GGUF copy of the security model (8-bit by default,
# about 8.5 GB). "transformers" loads the full 16-bit weights (about 16 GB), which
# does not fit alongside macOS on a 24 GB Mac and makes every call swap.
SECURITY_BACKEND_ENV = "SECURITY_MODEL_BACKEND"
DEFAULT_GGUF_REPO = "fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF"
DEFAULT_GGUF_FILE = "foundation-sec-8b-instruct-q8_0.gguf"

_gguf_model: Any = None


def security_backend() -> str:
    backend = os.getenv(SECURITY_BACKEND_ENV, "llama_cpp").strip().lower()

    if backend not in {"llama_cpp", "transformers"}:
        raise RuntimeError(f"{SECURITY_BACKEND_ENV} must be llama_cpp or transformers, not {backend!r}.")

    return backend


def security_model_label() -> str:
    """Names the weights that actually run, so experiment reports record precision."""

    if security_backend() == "llama_cpp":
        return os.getenv("HF_SECURITY_GGUF_PATH") or (
            f"{os.getenv('HF_SECURITY_GGUF_REPO', DEFAULT_GGUF_REPO)}/"
            f"{os.getenv('HF_SECURITY_GGUF_FILE', DEFAULT_GGUF_FILE)} (llama.cpp)"
        )

    return f"{os.getenv('HF_SECURITY_MODEL_ID')} (16-bit transformers)"


@lru_cache(maxsize=1)
def _load_security_model() -> tuple[Any, AutoTokenizer]:
    model_id = os.getenv("HF_SECURITY_MODEL_ID")

    if not model_id:
        raise RuntimeError("HF_SECURITY_MODEL_ID is missing from .env")

    # Both backends format prompts with the original tokenizer's chat template,
    # so switching backend changes the weights' precision, not the prompt.
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if security_backend() == "llama_cpp":
        return _load_gguf_security_model(), tokenizer

    device = _device()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device in {"mps", "cuda"} else torch.float32,
        low_cpu_mem_usage=True,
    )

    model.to(device)
    model.eval()

    return model, tokenizer


def _load_gguf_security_model() -> Any:
    global _gguf_model

    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    model_path = os.getenv("HF_SECURITY_GGUF_PATH") or hf_hub_download(
        os.getenv("HF_SECURITY_GGUF_REPO", DEFAULT_GGUF_REPO),
        os.getenv("HF_SECURITY_GGUF_FILE", DEFAULT_GGUF_FILE),
    )

    _gguf_model = Llama(
        model_path=model_path,
        # Largest prompt (threat normalisation) is ~4.5k tokens plus ~700 output.
        n_ctx=int(os.getenv("HF_SECURITY_GGUF_CONTEXT", "8192")),
        n_gpu_layers=-1,  # Offload every layer when Metal/CUDA exists; ignored on CPU-only builds.
        verbose=False,
    )

    return _gguf_model


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


def unload_security_model() -> None:
    """
    Unloads the cached security model so run_turn.py can control model lifecycle.
    """

    global _gguf_model

    # llama.cpp memory lives outside PyTorch, so release it explicitly.
    if _gguf_model is not None:
        close = getattr(_gguf_model, "close", None)

        if callable(close):
            close()

        _gguf_model = None

    _load_security_model.cache_clear()
    clear_torch_memory()


def evaluate_action_evidence(
    selected_action: dict[str, Any],
    selected_evidence: dict[str, Any],
    learner_justification: dict[str, Any] | None,
    vlm_output: dict[str, Any],
    scenario_state: dict[str, Any],
    incident_context: dict[str, Any] | None = None,
    marking_scheme: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Uses a local Hugging Face text model to evaluate whether the selected evidence
    supports the selected incident response action.

    incident_context is the learner-visible briefing: it tells the judge which
    principals and resources are under investigation, never what happened.

    marking_scheme is the turn's expected outcomes: what each support level looks
    like this turn, never which screenshot is which. With it the judge copies the
    facts it grades on from the extraction, and the app checks the copies exist,
    so a verdict cannot rest on an owner or event the screenshot does not show.
    """

    # Authoring-only labels are never part of model judgement inputs.
    ground_truth_keys = {"choice_role", "choiceRole", "support_role", "supportRole"}
    action_for_prompt = {k: v for k, v in selected_action.items() if k not in ground_truth_keys}
    # Author-written summaries and relevance hints may describe an intended role
    # rather than what the screenshot actually proves. The VLM's support grade is
    # also a fallible opinion; use its observed facts, not its verdict, here.
    evidence_for_prompt = {
        k: v for k, v in selected_evidence.items()
        if k in {"id", "title", "type", "template"}
    }
    vlm_facts_for_prompt = security_observations(vlm_output, selected_evidence)
    statements = learner_statements(learner_justification)
    # With no statements the example claim below was copied as a claim about
    # statement 1, which does not exist, so the example is empty too.
    claims_example = '[{"statement_id": 1, "assessment": "needs_detail"}]' if statements else "[]"
    structured = "event_rows" in vlm_output
    citation_schema = ',\n  "supporting_row_numbers": []' if structured else ""
    truncated_rows = [i for i, row in enumerate(vlm_output.get("event_rows", []), 1) if row.get("truncated")]
    truncation_rule = (
        f"- Only event rows {truncated_rows} are marked truncated. Never treat a clipped value as a complete address or infer its missing characters. Complete values in other rows may be cited separately. Truncation is not an automatic downgrade: missing text matters only if needed to justify this action."
        if truncated_rows else
        "- No extracted event row is marked truncated. Do not claim that a message or IP is clipped. Absence of truncation does not establish the full scope of the incident."
    ) if structured else "- Do not reconstruct unreadable values or treat clipped fragments as complete identifiers."
    credential_rule = (
        "- A static credential/status page is not a recent activity history. Do not infer past misuse merely because a key exists or is inactive."
        # Under a marking scheme the scheme decides: on some turns the suspect's active
        # key is the strong evidence, and this rule graded it Partial (26 Sep, 3 of 3 turn 4s).
        if (selected_evidence.get("template") or selected_evidence.get("type")) == "access_key"
        and not marking_scheme else ""
    )
    if marking_scheme:
        lead_schema = (
            '  "whose_evidence": "copy word for word the observed fact that shows whose activity or resource this is '
            '(user, principal or owner), or none shown",\n'
            '  "key_fact": "copy word for word the one observed fact that decides the verdict, or none",\n'
            '  "scheme_match": "strong_support, partial_support or weak_support: the marking-scheme description '
            'whose who and what match the copied facts",'
        )
        job = f"""Your job is to grade the learner's selected evidence like an examiner with a marking scheme. The marking scheme below was written for this turn's investigation. The verdict is the level whose description matches what the selected screenshot shows; then explain what the evidence shows and does not show.

Marking scheme for this turn (what each support level looks like; it is NOT evidence and does not say which screenshot was selected):
{json.dumps(marking_scheme, indent=2)}"""
        unrelated_rule = ""
        verdict_steps = """How to choose the verdict (fill whose_evidence and key_fact first, then follow these steps in order):
1. Copy whose_evidence and key_fact only from the observed screenshot facts. Never write a name, key, IP, finding or event that is not in them: the app checks every copied fact. If no observed fact names a user, principal or owner (for example a billing or cost summary), whose_evidence is "none shown"; do not take a name from the incident context or the marking scheme. The incident context does not make a screenshot about someone else into evidence about the principal under investigation.
2. Compare the copied facts with the marking scheme and choose the one level whose description they match: Strong, Partial or Weak Support. Match who and what is shown: a different principal, owner, finding or event from the one in the Strong description is not Strong Support, however related it seems. A screenshot that matches the Partial or Weak description is Partial or Weak Support; a different IP, time or event is what those descriptions expect.
3. Strong Support and Partial Support need a key_fact copied from the observed facts."""
    else:
        lead_schema = (
            '  "what_action_needs": "one short phrase: the activity, condition or outcome this action needs to see",\n'
            '  "event_or_condition_visible": "yes (name the event, finding or condition) or no",'
        )
        job = "Your job is to evaluate whether the learner's selected evidence supports their selected incident response action."
        unrelated_rule = ("- Events by a principal, source IP or resource that the incident context does not concern are unrelated "
                          "activity: they do not show what the action needs, even if they are real events.")
        verdict_steps = """How to choose the verdict (answer what_action_needs and event_or_condition_visible first, then follow these steps in order and stop at the first that applies):
1. Weak Support: nothing visible relates to what the action needs, or the screenshot contradicts it.
2. Definitions. An EVENT is something that happened and was logged or detected: an API call, log row or console/IAM activity entry, or a security finding or alert (for example a CloudTrail call, a CloudWatch log row, a GuardDuty finding, an automation or workflow execution), shown with an actor or resource and context. A CONDITION is a current state or measured figure (for example a key status, a resource configuration, a cost or usage figure). A user, IP address, ID or timestamp printed on a status or inventory page is only an attribute; it is neither an event nor a condition on its own.
3. If the action needs activity or events (for example reviewing, investigating or tracing what happened) and NO event is visible: Weak Support. A status, configuration, billing summary or inventory page is Weak Support here, even if it names a related user, IP or resource.
4. If the action needs a specific condition (for example checking a key status, a cost spike or a remediation setting) and that condition is visible, treat it like a visible event in the next steps.
5. A relevant event or condition is visible, notable or suspicious, and directly relevant to the action: Strong Support. One is enough to justify starting an investigation; a wider pattern, proof of intent or full scope is not required.
6. An event or condition is visible but generic, normal, or missing a link the action specifically needs: Partial Support. Name the missing link."""

    # Under a marking scheme the judge grades the evidence without seeing the action.
    # Told to ignore it, the judge still graded fit to the action when the learner chose
    # a wrong one (26 Sep: 6 of 30 wrong-action turns). The coach still sees the action
    # and the score rewards the choice itself.
    if marking_scheme:
        action_block = ""
        reasoning_hint = "brief explanation of what the evidence shows and does not show against the marking scheme"
    else:
        action_block = f"Learner selected action:\n{json.dumps(action_for_prompt, indent=2)}\n"
        reasoning_hint = "brief explanation of why the evidence does or does not support the action"

    model, tokenizer = _load_security_model()
    device = _device()

    prompt = f"""
You are the Security Reasoning AI in CloudIR Trainer.

{job}

{action_block}
Learner selected evidence:
{json.dumps(evidence_for_prompt, indent=2)}

Incident context shown to the learner this turn (identifies what is under investigation; it is NOT evidence and establishes no events):
{json.dumps(incident_context or {}, indent=2)}

Actual learner statements (only these numbered statements may receive credit):
{json.dumps(statements, indent=2)}

Instructor question (NOT learner speech and NOT a source of learner statements):
{json.dumps((learner_justification or {}).get('prompt', ''), indent=2)}

Observed screenshot facts from the vision-language model (check their limits):
{json.dumps(vlm_facts_for_prompt, indent=2)}

Current scenario state:
{json.dumps(scenario_state, indent=2)}

Return ONLY valid JSON with exactly these keys, filled in this order:
{{
{lead_schema}
  "verdict": "Strong Support | Partial Support | Weak Support",
  "reasoning": "{reasoning_hint}",
  "justification_claims": {claims_example},
  "recommended_next_focus": "the next evidence or relationship to investigate",
  "risk_of_wrong_interpretation": "a specific inference that these events do not establish"{citation_schema}
}}

Rules:
- Do not invent facts outside the VLM output, selected action, selected evidence, and learner transcript.
- Only observed evidence establishes events. The learner transcript, incident context and evidence titles are claims/context, not additional evidence.
{unrelated_rule}
- query_metadata_not_event_evidence describes a query, its scope and record count. Searching for unauthorized activity does not establish unauthorized activity. Use event_rows to establish what actually occurred.
{truncation_rule}
- Distinguish investigation from confirmation: relevant events can justify investigating without proving compromise, malicious intent, cross-region activity, or a complete pattern.
- Judge usable rows together. If a complete relevant row already justifies choosing an investigative action, missing information in other rows does not remove that support. Discuss those limits separately.
- Assess only the numbered learner statements. justification_claims contains at most three distinct existing statement IDs and one assessment each, exactly one of "supported", "overclaim", "needs_detail". Use [] when no statement can be assessed, and always when no statements were supplied. Do not write a justification_assessment or add paraphrases/quotes: the application will display the original statements verbatim.
- supported means the WHOLE statement is consistent with the evidence, including negation and qualifications. A future plan is not a completed observation. needs_detail applies to generic plans without a concrete visible fact; overclaim applies to assertions beyond the visible facts.
- All other prose must be impersonal evidence analysis: describe the selected action, visible events, limits and next checks. Never praise or describe the learner, their transcript or understanding; never use 'you', 'your', 'learner', 'transcript' or 'justification' in those fields. Personal assessment is restricted to justification_claims.
- Keep risk_of_wrong_interpretation specific to this evidence. Do not introduce unrelated credential-status examples.
- When event_rows are provided, cite row numbers and concrete visible details in reasoning. Return supporting_row_numbers as a list of those row numbers, or [] if none support the action. Strong Support must identify at least one relevant event row; metadata or a zero-result query alone cannot prove the activity.
{credential_rule}
- If a numbered statement overclaims something not visible, assess that statement as overclaim. Do not invent an additional claim.
- If there are no learner statements, evaluate action/evidence normally and return justification_claims: [].
- Keep the answer concise.

{verdict_steps}
"""

    messages = [
        {
            "role": "system",
            "content": "You are a careful cloud incident response evaluator.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    required_keys = {
        "verdict",
        "reasoning",
        "justification_claims",
        "recommended_next_focus",
        "risk_of_wrong_interpretation",
    }

    if structured:
        required_keys.add("supporting_row_numbers")
    if marking_scheme:
        required_keys |= {"whose_evidence", "key_fact"}
    for attempt in range(2):
        output_text = _generate_security_response(
            tokenizer=tokenizer, model=model, device=device, messages=messages,
            max_new_tokens=max(int(os.getenv("HF_MAX_NEW_TOKENS", "350")), 600),
        )
        try:
            parsed = _parse_json_output(output_text, required_keys, "Security model")
            # Drop unsolicited fields, including any free-form assessment. Only
            # the server may render what the learner said from the actual input.
            result = {key: parsed[key] for key in required_keys}
            # An empty answer has nothing to assess. Claims written anyway name
            # statements that do not exist; they are dropped rather than failing
            # the turn twice, and no reasoning is credited.
            claims = result["justification_claims"] if statements else []
            result["justification_claims"], result["justification_assessment"] = ground_justification(
                claims, statements,
            )
            result["feedback_grounding"] = keep_evidence_only_prose(result, (
                "reasoning", "recommended_next_focus", "risk_of_wrong_interpretation"))
            _validate_evidence_verdict(result, vlm_output)
            if marking_scheme:
                _validate_copied_facts(result, vlm_facts_for_prompt)
            return result
        except ValueError as exc:
            if attempt:
                # The cause reaches the server log and the run's error record.
                raise ValueError("Security feedback failed grounding checks after retry; no progress awarded. "
                                 f"Last problem: {exc}") from exc
            messages.append({"role": "user", "content": (
                f"Regenerate the complete JSON. Validation failed: {exc}. "
                "Use only existing statement IDs and enum assessments, or []. "
                "Keep every other field impersonal and evidence-focused; no learner attribution."
                + (" Copy whose_evidence and key_fact word for word from the observed screenshot facts."
                   if marking_scheme else "")
            )})


def _validate_evidence_verdict(result: dict[str, Any], vlm_output: dict[str, Any]) -> None:
    # Every turn offers three evidence items built as strong, partial and weak, so
    # the judge has those three levels; Weak and a mismatch update state alike.
    if result.get("verdict") not in {"Strong Support", "Partial Support", "Weak Support"}:
        raise ValueError("Security evaluation returned an invalid verdict; choose Strong Support, "
                         "Partial Support or Weak Support. No progress awarded.")
    for key in ("reasoning", "justification_assessment", "recommended_next_focus", "risk_of_wrong_interpretation"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f"Security evaluation returned an invalid {key}; no progress awarded.")
    if "event_rows" not in vlm_output:
        return
    refs = result.get("supporting_row_numbers")
    rows = vlm_output["event_rows"]
    if not isinstance(refs, list) or any(type(i) is not int or not 1 <= i <= len(rows) for i in refs):
        raise ValueError("Security evaluation cited invalid event rows; no progress awarded.")
    if result["verdict"] == "Strong Support" and not any(rows[i - 1].get("message") for i in refs):
        raise ValueError("Strong Support lacked a cited readable event row; no progress awarded.")


NOTHING_COPIED = {"", "none", "noneshown", "notshown", "na"}
# Identifying details the judge could invent: ARNs, names like d.okafor, IPs, key
# IDs, API names such as PutEvents. Plain words ("the user identity shown is")
# are wording, not facts, and a correct copy wrapped in them must not fail.
_CHUNK = re.compile(r"[^\s'\"`,;()\[\]{}]+")
_IDENTIFIER = re.compile(r"[0-9:/._@-]|[a-z][A-Z]")


def _validate_copied_facts(result: dict[str, Any], observed: dict[str, Any]) -> None:
    """Every identifying detail the judge copied must be in the extraction; the verdict stays the judge's.

    Catches a verdict built on a fact the screenshot does not show, such as
    calling another principal's access key the suspect's. Weak Support rests on
    what is missing, so a copy it cannot ground ("no AttachUserPolicy call shown")
    is cleared instead of failing the evaluation: it cannot raise the grade.
    """

    seen = _squash(json.dumps(observed, ensure_ascii=False))

    for key in ("whose_evidence", "key_fact"):
        copied = result.get(key)

        if not isinstance(copied, str):
            raise ValueError(f"{key} must be text copied from the observed facts.")
        if re.sub(r"[^a-z]", "", copied.lower()) in NOTHING_COPIED:
            if key == "key_fact" and result["verdict"] in {"Strong Support", "Partial Support"}:
                raise ValueError(f"{result['verdict']} needs a key_fact copied from the observed facts.")
            continue

        missing = [chunk.strip(".:") for chunk in _CHUNK.findall(copied)
                   if _IDENTIFIER.search(chunk.strip(".:")) and _squash(chunk) not in seen]

        if missing and result["verdict"] == "Weak Support":
            result[key] = "none"
        elif missing:
            hint = (" If no observed fact names a user, principal or owner, write \"none shown\"."
                    if key == "whose_evidence" else "")
            raise ValueError(f"{key} '{copied}' names details the observed screenshot facts do not show "
                             f"({', '.join(missing[:4])}); copy the fact itself from the observed facts.{hint}")


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def normalise_acse_threat_model(
    threat_model: dict[str, Any],
    architecture_facts: dict[str, Any],
    scenario_id: str,
) -> dict[str, Any]:
    """
    Uses the local security reasoning model to convert an ACSE threat model and
    VLM-generated architecture facts into CloudIR-ready incident facts.

    scenario_type is the scenario being prepared, set by code: the prompt once
    showed "identity-management" and the output was forced to it, so every
    scenario's threat model and hidden truth claimed to be identity management.

    This function is used during dataset preparation before the normal CloudIR
    runtime starts.

    Input:
    - data/source/<scenario>/threat-model.json
    - data/processed/<scenario>/architecture_facts.json

    Output:
    - structured facts for normalised_threat_model.json
    """

    scenario_type = scenario_id.strip().lower().replace("_", "-")
    model, tokenizer = _load_security_model()
    device = _device()

    prompt = f"""
You are the Security Text Normaliser inside CloudIR Trainer.

You are preparing the ACSE-Eval "{scenario_type}" scenario before the learner starts the incident-response simulation.

Your task is to read:
1. The raw ACSE threat model.
2. The architecture facts extracted from the architecture diagram by the VLM.

Raw ACSE threat model:
{json.dumps(threat_model, indent=2)}

Architecture facts from VLM:
{json.dumps(architecture_facts, indent=2)}

Convert these inputs into CloudIR-ready incident facts.

Return ONLY valid JSON with exactly these keys:
{{
  "primary_risk": "main security risk in one sentence",
  "affected_assets": [
    "cloud asset, identity, service, account, role, data store or system that may be affected"
  ],
  "likely_attack_path": [
    "ordered attacker or misuse step inferred from the threat model and architecture facts"
  ],
  "security_signals": [
    "observable cloud security signal that could appear in evidence"
  ],
  "investigation_goals": [
    "learner investigation goal for the incident-response scenario"
  ],
  "recommended_evidence_sources": [
    "CloudTrail | IAM activity | GuardDuty | CloudWatch | access keys | billing | other relevant evidence source"
  ],
  "hidden_truth_candidates": [
    "security truth that can be used later by the simulator but should not be directly shown to the learner"
  ]
}}

Rules:
- Use the threat model and architecture facts as the source of security meaning.
- Do not invent a completely unrelated incident.
- Do not create final evidence screenshots.
- Do not write learner feedback.
- Do not generate actions.
- Do not mention that you are an AI model.
- Keep the output concise.
- Return JSON only.
- If a field has no clear information, return an empty list for that field.
"""

    messages = [
        {
            "role": "system",
            "content": "You are a careful cloud security threat-model normaliser.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    max_new_tokens = int(
        os.getenv(
            "HF_SECURITY_NORMALISER_MAX_NEW_TOKENS",
            os.getenv("HF_MAX_NEW_TOKENS", "700"),
        )
    )

    output_text = _generate_security_response(
        tokenizer=tokenizer,
        model=model,
        device=device,
        messages=messages,
        max_new_tokens=max_new_tokens,
    )

    required_keys = {
        "primary_risk",
        "affected_assets",
        "likely_attack_path",
        "security_signals",
        "investigation_goals",
        "recommended_evidence_sources",
        "hidden_truth_candidates",
    }

    parsed = _parse_json_output(
        output_text=output_text,
        required_keys=required_keys,
        model_task_name="Security threat normaliser",
    )

    return _normalise_acse_threat_output(parsed, scenario_type)


# The model reuses whatever example names the prompt shows, so each supported
# scenario gets different ones from these pools, and always the same ones.
EXAMPLE_PEOPLE = ("d.okafor", "m.tanaka", "s.brennan", "a.kowalski", "r.iyer", "l.fischer", "k.adeyemi", "p.novak")
EXAMPLE_ROLES = ("etl-runner", "ci-deploy", "log-shipper", "backup-agent", "report-builder", "metrics-sync")
EXAMPLE_RESOURCES = ("invoice-archive", "customer-api", "metrics-ingest", "payments-queue", "media-uploads", "orders-fn")


def example_names(scenario_id: str) -> str:
    # Imported here: dataset_preparation imports this module.
    from cloudir.dataset_preparation.select_case import TRAINING_SCENARIO_ALLOWLIST

    key = scenario_id.strip().lower().replace("_", "-")
    supported = sorted(TRAINING_SCENARIO_ALLOWLIST)
    index = supported.index(key) if key in supported else hashlib.sha256(key.encode()).digest()[0]
    return (f"user/{EXAMPLE_PEOPLE[index % len(EXAMPLE_PEOPLE)]}, "
            f"role/{EXAMPLE_ROLES[index % len(EXAMPLE_ROLES)]}, "
            f"{EXAMPLE_RESOURCES[index % len(EXAMPLE_RESOURCES)]}")


def write_incident_timeline(
    normalised_threat_model: dict[str, Any],
    architecture_facts: dict[str, Any],
    turn_goals: list[dict[str, Any]],
    retry_instruction: str = "",
    scenario_services: list[str] | None = None,
    attempt: int = 1,
    rejected_timeline: dict[str, Any] | None = None,
    scenario_id: str = "",
) -> dict[str, Any]:
    """
    Asks the security model for one incident timeline for the scenario.

    Every turn's evidence is built from this timeline by code, so it only has
    to describe what happened; cloudir.scenario.incident_timeline validates it.
    A retry shows the model its rejected timeline so it fixes the listed
    problems instead of writing a new story with new mistakes.
    """

    model, tokenizer = _load_security_model()

    goals = "\n".join(
        f"- Turn {goal.get('turn')} ({goal.get('stage')}): {goal.get('goal')}" for goal in turn_goals
    )
    names = example_names(scenario_id)
    # The model invents plausible names (CreateFinding) for services it knows less well.
    services = "\n".join(
        f"- {service}" + (f": real eventNames include {SUGGESTED_EVENTS[service]}" if service in SUGGESTED_EVENTS else "")
        for service in scenario_services or []
    ) or "- those named in the threat model"

    prompt = f"""
You are the Incident Timeline Author in CloudIR Trainer, an AWS incident-response training simulator.

Write ONE realistic AWS incident timeline for this scenario. All evidence screenshots in the
5-turn exercise are built from it, so it must be internally consistent.

Normalised threat model:
{json.dumps(normalised_threat_model, indent=2)}

Architecture facts:
{json.dumps(architecture_facts, indent=2)}

Turn goals:
{goals}

Services in scope for this scenario:
{services}
The attacker's incident steps must include API calls to these services, following the threat model's attack path.

Return ONLY valid JSON with this shape:
{{
  "account_id": "12 digits",
  "actors": [
    {{"id": "attacker", "role": "attacker", "principal": "arn:aws:iam::<account_id>:user/<name>", "source_ip": "public IPv4", "mfa": false}},
    {{"id": "<short id>", "role": "background", "principal": "arn:aws:iam::<account_id>:role/<name>", "source_ip": "IPv4"}},
    {{"id": "<short id>", "role": "responder", "principal": "arn:aws:iam::<account_id>:role/<name>", "source_ip": "IPv4"}}
  ],
  "events": [
    {{"time": "2023-10-01T10:12:10Z", "actor": "attacker", "event_name": "<exact CloudTrail eventName>", "event_source": "<service>.amazonaws.com", "result": "Success", "error_code": "-", "mfa": "false", "resource": "<short resource name>", "turn": 1, "suspicious": true}}
  ],
  "findings": [
    {{"finding_type": "<GuardDuty or Security Hub finding type>", "severity": "High", "resource": "affected user or resource", "description": "one factual sentence", "turn": 3}}
  ],
  "cost": {{"baseline_daily_usd": 42.75, "incident_daily_usd": 48.10, "service": "EC2", "normal_service": "S3"}}
}}

Rules:
- event_name must be the exact CloudTrail eventName with no spaces, for example ConsoleLogin, AssumeRole, CreateAccessKey, AttachUserPolicy, StopLogging, RunInstances, BatchImportFindings, PutEvents, PutRule, UpdateFunctionCode20150331v2, StartExecution. Never write descriptions such as "Failed Login Attempt".
- event_source is the service endpoint, such as signin.amazonaws.com, iam.amazonaws.com, sts.amazonaws.com.
- Times are ISO UTC on one day, in chronological order.
- Exactly one attacker, at least one background actor doing normal work, and one responder (the security team).
- Give each key event a "turn" from 1 to 5, following the turn goals: turn 1 the first suspicious signal; turn 2 the attacker's follow-up actions; turn 3 actions that show how far it spread; turn 4 the attacker's still-active access that containment must stop; turn 5 the responder's recovery actions, using real APIs such as UpdateAccessKey, DeleteLoginProfile, DetachUserPolicy, DeleteAccessKey or StartLogging (there is no DisableUser API).
- Turns 1 to 3 each need at least one suspicious attacker event. Turn 5 needs at least one responder event.
- Give background actors 3 to 5 routine read-only events with no "turn", such as DescribeInstances, GetObject or ListObjectsV2. Background activity must not touch the attacker's principal or resources.
- Use realistic IPv4 addresses. Never use 192.0.2.x addresses.
- Use realistic names for every principal and resource (for example {names}). Never use words such as malicious, attacker, hacker, compromised or suspicious in names; the evidence must not announce the answer.
- 12 to 20 events in total. Keep every string short.
- findings: at least one Medium or High GuardDuty finding about the attacker's activity, using a real type in the form Purpose:Resource/Family, such as UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration.OutsideAWS, Persistence:IAMUser/AnomalousBehavior, Stealth:IAMUser/CloudTrailLoggingDisabled or Backdoor:Lambda/C&CActivity.B.
- cost: daily USD spend before and during the incident; keep them close if the incident does not affect cost.
- Use the same account_id in every ARN. Follow the threat model; do not invent an unrelated incident.
"""

    if retry_instruction.strip() and rejected_timeline:
        prompt += f"""
Your previous timeline was rejected:
{json.dumps(rejected_timeline)}

Fix only these problems and return the whole corrected timeline, keeping everything else:
{retry_instruction.strip()}
"""
    elif retry_instruction.strip():
        prompt += f"""
The previous timeline was rejected. Fix these problems and return the whole timeline again:
{retry_instruction.strip()}
"""

    messages = [
        {"role": "system", "content": "You write consistent, realistic AWS CloudTrail incident timelines as JSON."},
        {"role": "user", "content": prompt},
    ]

    output_text = _generate_security_response(
        tokenizer=tokenizer,
        model=model,
        device=_device(),
        messages=messages,
        max_new_tokens=int(os.getenv("HF_SECURITY_TIMELINE_MAX_NEW_TOKENS", "2600")),
        # Greedy decoding tends to repeat a rejected answer; retries sample a little.
        sampling_seed=attempt if attempt > 1 else None,
    )

    return _parse_json_output(
        output_text=output_text,
        required_keys={"account_id", "actors", "events", "findings", "cost"},
        model_task_name="Incident timeline",
    )


def _generate_security_response(
    tokenizer: AutoTokenizer,
    model: Any,
    device: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    sampling_seed: int | None = None,
) -> str:
    """Greedy by default. sampling_seed enables mild, reproducible sampling for
    authoring retries, where greedy decoding would repeat a rejected answer;
    judgements always stay greedy."""

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    sample = sampling_seed is not None

    if hasattr(model, "create_completion"):
        # The chat template already starts with <|begin_of_text|>; adding special
        # tokens again would send it twice.
        completion = model.create_completion(
            tokenizer(input_text, add_special_tokens=False)["input_ids"],
            max_tokens=max_new_tokens,
            temperature=0.5 if sample else 0.0,
            top_k=40 if sample else 1,
            top_p=0.95 if sample else 1.0,
            min_p=0.0,
            repeat_penalty=1.0,
            seed=sampling_seed,
        )

        return completion["choices"][0]["text"]

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    if sample:
        torch.manual_seed(sampling_seed)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=0.5 if sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(generated, skip_special_tokens=True)


def _normalise_acse_threat_output(parsed: dict[str, Any], scenario_type: str) -> dict[str, Any]:
    primary_risk = str(
        parsed.get("primary_risk")
        or f"The threat model indicates a {scenario_type} security risk."
    ).strip()

    return {
        "scenario_type": scenario_type,
        "primary_risk": primary_risk,
        "affected_assets": _ensure_list_of_strings(parsed.get("affected_assets")),
        "likely_attack_path": _ensure_list_of_strings(parsed.get("likely_attack_path")),
        "security_signals": _ensure_list_of_strings(parsed.get("security_signals")),
        "investigation_goals": _ensure_list_of_strings(parsed.get("investigation_goals")),
        "recommended_evidence_sources": _ensure_list_of_strings(
            parsed.get("recommended_evidence_sources")
        ),
        "hidden_truth_candidates": _ensure_list_of_strings(
            parsed.get("hidden_truth_candidates")
        ),
    }


def _parse_json_output(
    output_text: str,
    required_keys: set[str],
    model_task_name: str,
) -> dict[str, Any]:
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
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{model_task_name} did not return valid JSON. Raw output was:\n\n"
            f"{output_text}"
        ) from exc

    missing = required_keys - set(parsed.keys())

    if missing:
        raise ValueError(f"{model_task_name} output is missing keys: {missing}")

    return parsed


def _ensure_list_of_strings(value: Any) -> list[str]:
    if value in [None, "", [], {}]:
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    return [str(value).strip()]
