from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    LlavaOnevisionForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    pipeline,
)
from transformers.utils import cached_file


LOCAL_FILES_ONLY = True
VLM_MAX_NEW_TOKENS = 220
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"
EVIDENCE_DIR = PROJECT_ROOT / "output" / "generated_evidence"
BENCHMARK_DIR = PROJECT_ROOT / "notebooks" / "model_benchmarks"
OUTPUT_DIR = BENCHMARK_DIR / "results"
STT_SAMPLE_DIR = BENCHMARK_DIR / "stt_audio_samples" / "audio_samples"
STT_CONVERTED_DIR = STT_SAMPLE_DIR / "converted_wav"


SUPPORT_TO_VERDICT = {
    "strong": "Strong Support",
    "partial": "Partial Support",
    "weak": "Weak Support",
    "unsupported": "Unsupported",
}

VERDICT_LABELS = [
    "Strong Support",
    "Partial Support",
    "Weak Support",
    "Unsupported",
]

VLM_MODELS = [
    {
        "stage": "vlm",
        "name": "qwen2.5-vl-3b-instruct",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "family": "vision-language instruction model",
        "benchmark_mode": "executed_when_included",
        "used_in_application": True,
        "metric": "field extraction and evidence-type alignment",
        "notes": "Current application VLM. Strong screenshot/layout reader for evidence screenshots.",
    },
    {
        "stage": "vlm",
        "name": "qwen2-vl-2b-instruct",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "family": "smaller vision-language instruction model",
        "benchmark_mode": "executed_when_included",
        "used_in_application": False,
        "metric": "field extraction and latency",
        "notes": "Smaller Qwen VLM baseline for comparing speed and evidence extraction.",
    },
    {
        "stage": "vlm",
        "name": "llava-onevision-0.5b",
        "model_id": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "family": "small open VLM baseline",
        "benchmark_mode": "executed_when_included",
        "used_in_application": False,
        "metric": "field extraction and hallucination rate",
        "notes": "Lightweight multimodal baseline for screenshot interpretation.",
    },
    {
        "stage": "vlm",
        "name": "florence-2-base-ft",
        "model_id": "microsoft/Florence-2-base-ft",
        "family": "compact OCR/captioning vision-language model",
        "benchmark_mode": "executed_when_included",
        "used_in_application": False,
        "metric": "OCR keyword recall and latency",
        "notes": "Small Florence-2 candidate for screenshot text extraction; useful CPU-friendly baseline.",
        "trust_remote_code": True,
    },
]

SECURITY_MODELS = [
    {
        "stage": "security",
        "name": "foundation-sec-8b-instruct",
        "model_id": "fdtn-ai/Foundation-Sec-8B-Instruct",
        "family": "cybersecurity-focused instruction-tuned causal LM",
        "kind": "causal_lm",
        "benchmark_mode": "executed",
        "used_in_application": True,
        "metric": "support-verdict accuracy and latency",
        "notes": "Current application security model. Cybersecurity-focused model for SOC, incident response, and security workflow reasoning.",
    },
    {
        "stage": "security",
        "name": "qwen2.5-1.5b-instruct",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "family": "general instruction-tuned causal LM",
        "kind": "causal_lm",
        "benchmark_mode": "executed",
        "used_in_application": False,
        "metric": "support-verdict accuracy and latency",
        "notes": "Previous application security model; retained as a general-purpose baseline.",
    },
    {
        "stage": "security",
        "name": "qwen2.5-0.5b-instruct",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "family": "small general instruction-tuned causal LM",
        "kind": "causal_lm",
        "benchmark_mode": "executed",
        "used_in_application": False,
        "metric": "support-verdict accuracy and latency",
    },
    {
        "stage": "security",
        "name": "flan-t5-small",
        "model_id": "google/flan-t5-small",
        "family": "small sequence-to-sequence instruction model",
        "kind": "seq2seq",
        "benchmark_mode": "executed",
        "used_in_application": False,
        "metric": "support-verdict accuracy and latency",
    },
    {
        "stage": "security",
        "name": "flan-t5-base",
        "model_id": "google/flan-t5-base",
        "family": "sequence-to-sequence instruction model",
        "kind": "seq2seq",
        "benchmark_mode": "executed",
        "used_in_application": False,
        "metric": "support-verdict accuracy and latency",
    },
    {
        "stage": "security",
        "name": "bart-large-mnli",
        "model_id": "facebook/bart-large-mnli",
        "family": "zero-shot NLI classifier",
        "kind": "zero_shot",
        "benchmark_mode": "executed",
        "used_in_application": False,
        "metric": "support-verdict accuracy and latency",
    },
    {
        "stage": "security",
        "name": "foundation-sec-8b-base",
        "model_id": "fdtn-ai/Foundation-Sec-8B",
        "family": "cybersecurity-focused base causal LM",
        "kind": "causal_lm",
        "benchmark_mode": "candidate_not_downloaded",
        "used_in_application": False,
        "metric": "support-verdict accuracy and latency",
        "notes": "Cybersecurity base model from Cisco Foundation AI; less instruction-optimised than Foundation-Sec-8B-Instruct.",
    },
    {
        "stage": "security",
        "name": "cyberpal2-20b",
        "model_id": "cyber-pal-security/CyberPal2.0-20B",
        "family": "cybersecurity-expert instruction-tuned causal LM",
        "kind": "causal_lm",
        "benchmark_mode": "candidate_not_downloaded",
        "used_in_application": False,
        "metric": "support-verdict accuracy and latency",
        "notes": "SOC/IR, CTI, vulnerability, CWE/CVE, and MITRE ATT&CK focused model; likely too large for the local prototype.",
    },
    {
        "stage": "security",
        "name": "cybersecqwen-4b",
        "model_id": "athena129/CyberSecQwen-4B",
        "family": "cybersecurity-specialised Qwen-derived instruction model",
        "kind": "causal_lm",
        "benchmark_mode": "candidate_not_downloaded",
        "used_in_application": False,
        "metric": "support-verdict accuracy and latency",
        "notes": "Defensive cybersecurity model focused on CTI-Bench tasks such as CWE mapping and CTI multiple-choice reasoning.",
    },
    {
        "stage": "security",
        "name": "lily-cybersecurity-7b-v0.2",
        "model_id": "segolilylabs/Lily-Cybersecurity-7B-v0.2",
        "family": "cybersecurity-specialised causal LM",
        "kind": "causal_lm",
        "benchmark_mode": "candidate_not_downloaded",
        "used_in_application": False,
        "metric": "support-verdict accuracy and latency",
        "notes": "Cybersecurity text-generation model candidate; included for evaluation discussion only.",
    },
]

COACH_MODELS = [
    {
        "stage": "coach",
        "name": "qwen2.5-1.5b-instruct",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "family": "instruction-tuned causal LM",
        "kind": "causal_lm",
        "benchmark_mode": "executed",
        "used_in_application": True,
        "metric": "5-point feedback rubric and latency",
    },
    {
        "stage": "coach",
        "name": "qwen2.5-0.5b-instruct",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "family": "small instruction-tuned causal LM",
        "kind": "causal_lm",
        "benchmark_mode": "executed",
        "used_in_application": False,
        "metric": "5-point feedback rubric and latency",
    },
    {
        "stage": "coach",
        "name": "flan-t5-small",
        "model_id": "google/flan-t5-small",
        "family": "small sequence-to-sequence instruction model",
        "kind": "seq2seq",
        "benchmark_mode": "executed",
        "used_in_application": False,
        "metric": "5-point feedback rubric and latency",
    },
    {
        "stage": "coach",
        "name": "flan-t5-base",
        "model_id": "google/flan-t5-base",
        "family": "sequence-to-sequence instruction model",
        "kind": "seq2seq",
        "benchmark_mode": "executed",
        "used_in_application": False,
        "metric": "5-point feedback rubric and latency",
    },
]

STT_MODELS = [
    {
        "stage": "stt",
        "name": "whisper-tiny-en",
        "model_id": "openai/whisper-tiny.en",
        "family": "small English speech recognition model",
        "benchmark_mode": "executed_when_recordings_exist",
        "used_in_application": False,
        "metric": "keyword recall, meaning preservation, latency",
        "notes": "Fast English-only baseline for learner voice recordings.",
    },
    {
        "stage": "stt",
        "name": "whisper-base",
        "model_id": "openai/whisper-base",
        "family": "encoder-decoder speech recognition model",
        "benchmark_mode": "executed_when_recordings_exist",
        "used_in_application": True,
        "metric": "keyword recall and meaning preservation",
        "notes": "Configured locally. The benchmark auto-discovers recorded learner audio samples.",
    },
    {
        "stage": "stt",
        "name": "whisper-base-en",
        "model_id": "openai/whisper-base.en",
        "family": "English speech recognition model",
        "benchmark_mode": "executed_when_recordings_exist",
        "used_in_application": False,
        "metric": "keyword recall, meaning preservation, latency",
        "notes": "English-only base model for comparing against multilingual Whisper Base.",
    },
    {
        "stage": "stt",
        "name": "whisper-small",
        "model_id": "openai/whisper-small",
        "family": "larger speech recognition model",
        "benchmark_mode": "executed_when_recordings_exist",
        "used_in_application": False,
        "metric": "keyword recall, meaning preservation, latency",
        "notes": "Larger Whisper baseline to test whether quality improves enough to justify slower inference.",
    },
]

STT_SAMPLE_DEFINITIONS = {
    "strongcloudtrailstoplogging": {
        "id": "strong_cloudtrail_stop_logging",
        "text": (
            "The CloudTrail log strongly supports stopping the incident because it shows "
            "StopLogging by an IAM user from a suspicious source IP without MFA."
        ),
        "keywords": ["cloudtrail", "stoplogging", "iam user", "source ip", "mfa"],
    },
    "strongaccesskeycontainment": {
        "id": "strong_access_key_containment",
        "text": (
            "The access key evidence strongly supports containment because the same key "
            "is active and should be disabled or rotated immediately."
        ),
        "keywords": ["access key", "containment", "active", "disable", "rotate"],
    },
    "partialiamactivity": {
        "id": "partial_iam_activity",
        "text": (
            "The IAM activity gives partial support because it shows suspicious API calls, "
            "but it does not prove the full attack path on its own."
        ),
        "keywords": ["iam", "activity", "partial", "api calls", "attack path"],
    },
    "partialcloudwatchcorrelation": {
        "id": "partial_cloudwatch_correlation",
        "text": (
            "The CloudWatch correlation gives partial support because it helps confirm "
            "the timeline, but it still needs CloudTrail or IAM evidence."
        ),
        "keywords": ["cloudwatch", "correlation", "partial", "timeline", "cloudtrail"],
    },
    "weakbillingsnapshot": {
        "id": "weak_billing_snapshot",
        "text": (
            "The billing snapshot is weak support because it may show impact, but it does "
            "not directly identify the IAM user or the suspicious source IP."
        ),
        "keywords": ["billing", "weak", "impact", "iam user", "source ip"],
    },
    "weakaccesskeyreview": {
        "id": "weak_access_key_review",
        "text": (
            "The access key review is weak support because it suggests credential risk, "
            "but it does not prove the exact action or incident cause."
        ),
        "keywords": ["access key", "weak", "credential", "risk", "incident"],
    },
    "unsupportedgenericpolicy": {
        "id": "unsupported_generic_policy",
        "text": (
            "The generic policy evidence is unsupported because it does not show the "
            "specific incident, user, source IP, or CloudTrail event."
        ),
        "keywords": ["unsupported", "policy", "incident", "source ip", "cloudtrail"],
    },
    "unsupportedoverclaimingguardduty": {
        "id": "unsupported_overclaiming_guardduty",
        "text": (
            "The GuardDuty finding should not be overclaimed because it is unsupported "
            "without matching IAM, CloudTrail, or access key evidence."
        ),
        "keywords": ["guardduty", "unsupported", "iam", "cloudtrail", "access key"],
    },
    "unsupportedoverclaimingguarduty": {
        "id": "unsupported_overclaiming_guardduty",
        "text": (
            "The GuardDuty finding should not be overclaimed because it is unsupported "
            "without matching IAM, CloudTrail, or access key evidence."
        ),
        "keywords": ["guardduty", "unsupported", "iam", "cloudtrail", "access key"],
    },
}


@dataclass
class BenchmarkCase:
    case_id: str
    turn: int
    selected_action: dict[str, Any]
    selected_evidence: dict[str, Any]
    learner_justification: dict[str, Any]
    vlm_output: dict[str, Any]
    expected_verdict: str
    expected_support_role: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run CloudIR selected-model evaluation and optional benchmark-only "
            "Hugging Face baselines."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for benchmark result files.",
    )
    parser.add_argument(
        "--include-coach",
        action="store_true",
        help="Also compare coach feedback generation models.",
    )
    parser.add_argument(
        "--include-vlm",
        action="store_true",
        help="Also compare VLM candidates on generated evidence screenshots.",
    )
    parser.add_argument(
        "--include-stt",
        action="store_true",
        help=(
            "Benchmark Whisper using real audio samples under "
            "notebooks/model_benchmarks/stt_audio_samples/audio_samples/."
        ),
    )
    parser.add_argument(
        "--limit-security-cases",
        type=int,
        default=0,
        help="Limit security cases for a faster smoke run. 0 means all cases.",
    )
    parser.add_argument(
        "--limit-vlm-cases",
        type=int,
        default=3,
        help="Limit VLM screenshot cases. VLMs are heavy on CPU, so the default is 3. 0 means all image cases.",
    )
    parser.add_argument(
        "--vlm-models",
        default="",
        help="Comma-separated VLM model names to run. Empty means all configured VLM candidates.",
    )
    parser.add_argument(
        "--vlm-max-new-tokens",
        type=int,
        default=220,
        help="Maximum generated tokens for VLM outputs. Lower this for CPU-only benchmark runs.",
    )
    parser.add_argument(
        "--limit-stt-samples",
        type=int,
        default=0,
        help="Limit STT audio samples for a faster smoke run. 0 means all samples.",
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Hugging Face to download missing model files. By default, only cached local models are used.",
    )
    args = parser.parse_args()

    set_hugging_face_download_mode(args.allow_downloads)
    set_vlm_generation_limit(args.vlm_max_new_tokens)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = collect_environment_metadata()
    write_json(args.output_dir / "benchmark_environment.json", metadata)
    candidate_matrix = build_model_candidate_matrix()
    write_rows(args.output_dir / "model_candidate_matrix.csv", candidate_matrix)
    write_json(args.output_dir / "model_candidate_matrix.json", candidate_matrix)

    cases = build_security_cases()
    if args.limit_security_cases > 0:
        cases = cases[: args.limit_security_cases]

    print(f"Security benchmark cases: {len(cases)}")
    security_results = run_security_model_benchmarks(cases)
    write_rows(args.output_dir / "security_model_results.csv", security_results)
    write_json(args.output_dir / "security_model_results.json", security_results)

    security_summary = summarise_classification_results(
        security_results,
        group_key="model_name",
    )
    write_rows(args.output_dir / "security_model_summary.csv", security_summary)
    write_json(args.output_dir / "security_model_summary.json", security_summary)
    print_summary("Security Model Summary", security_summary)

    evidence_results = audit_evidence_images()
    write_rows(args.output_dir / "evidence_image_audit.csv", evidence_results)
    write_json(args.output_dir / "evidence_image_audit.json", evidence_results)

    if args.include_vlm:
        vlm_cases = build_vlm_cases(cases)
        if args.limit_vlm_cases > 0:
            vlm_cases = vlm_cases[: args.limit_vlm_cases]
        vlm_models = filter_model_specs(VLM_MODELS, args.vlm_models)
        vlm_results = run_vlm_model_benchmarks(vlm_cases, vlm_models)
        write_rows(args.output_dir / "vlm_model_results.csv", vlm_results)
        write_json(args.output_dir / "vlm_model_results.json", vlm_results)
        vlm_summary = summarise_vlm_results(vlm_results)
        write_rows(args.output_dir / "vlm_model_summary.csv", vlm_summary)
        write_json(args.output_dir / "vlm_model_summary.json", vlm_summary)
        print_summary("VLM Model Summary", vlm_summary)

    if args.include_coach:
        coach_cases = cases[: min(6, len(cases))]
        coach_results = run_coach_model_benchmarks(coach_cases)
        write_rows(args.output_dir / "coach_model_results.csv", coach_results)
        write_json(args.output_dir / "coach_model_results.json", coach_results)
        coach_summary = summarise_coach_results(coach_results)
        write_rows(args.output_dir / "coach_model_summary.csv", coach_summary)
        write_json(args.output_dir / "coach_model_summary.json", coach_summary)
        print_summary("Coach Model Summary", coach_summary)

    if args.include_stt:
        stt_results = run_stt_benchmark(sample_limit=args.limit_stt_samples)
        write_rows(args.output_dir / "stt_model_results.csv", stt_results)
        write_json(args.output_dir / "stt_model_results.json", stt_results)
        stt_summary = summarise_stt_results(stt_results)
        write_rows(args.output_dir / "stt_model_summary.csv", stt_summary)
        write_json(args.output_dir / "stt_model_summary.json", stt_summary)
        print_summary("STT Model Summary", stt_summary)

    print(f"Results written to: {args.output_dir}")


def set_hugging_face_download_mode(allow_downloads: bool) -> None:
    global LOCAL_FILES_ONLY

    LOCAL_FILES_ONLY = not allow_downloads
    if allow_downloads:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def set_vlm_generation_limit(max_new_tokens: int) -> None:
    global VLM_MAX_NEW_TOKENS
    VLM_MAX_NEW_TOKENS = max(16, max_new_tokens)


def filter_model_specs(
    model_specs: list[dict[str, Any]],
    names_csv: str,
) -> list[dict[str, Any]]:
    if not names_csv.strip():
        return model_specs
    requested = {name.strip() for name in names_csv.split(",") if name.strip()}
    filtered = [model for model in model_specs if model["name"] in requested]
    missing = sorted(requested - {model["name"] for model in filtered})
    if missing:
        print(f"Skipping unknown model names: {', '.join(missing)}")
    return filtered


def collect_environment_metadata() -> dict[str, Any]:
    return {
        "project_root": str(PROJECT_ROOT),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "device_used": device(),
        "local_files_only": LOCAL_FILES_ONLY,
        "models": {
            "security_candidates": SECURITY_MODELS,
            "coach_candidates": COACH_MODELS,
            "vlm_candidates": VLM_MODELS,
            "stt_candidates": STT_MODELS,
            "stt_candidate": os.getenv("HF_VOICE_MODEL_ID", "openai/whisper-base"),
            "configured_image_model": os.getenv(
                "HF_IMAGE_MODEL_ID",
                "Qwen/Qwen2.5-VL-3B-Instruct",
            ),
        },
    }


def build_model_candidate_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_spec in [*VLM_MODELS, *SECURITY_MODELS, *COACH_MODELS, *STT_MODELS]:
        rows.append(
            {
                "stage": model_spec["stage"],
                "name": model_spec["name"],
                "model_id": model_spec["model_id"],
                "family": model_spec["family"],
                "benchmark_mode": model_spec["benchmark_mode"],
                "used_in_application": model_spec["used_in_application"],
                "metric": model_spec["metric"],
                "notes": model_spec.get("notes", ""),
            }
        )
    return rows


def build_security_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []

    for turn_dir in sorted((RUNTIME_DIR / "turns").glob("turn_*")):
        turn = int(turn_dir.name.split("_")[-1])
        actions = read_json(turn_dir / "actions.json")
        evidence_items = read_json(turn_dir / "evidence_facts.json")
        saved_trace_path = RUNTIME_DIR / "evaluations" / f"turn_{turn}_evaluation.json"
        saved_trace = read_json(saved_trace_path) if saved_trace_path.exists() else {}
        saved_vlm = saved_trace.get("vlm_output", {})

        best_action = find_by_role(actions, "choice_role", "best") or actions[0]
        for evidence in evidence_items:
            support_role = normalise_support_role(evidence)
            expected_verdict = SUPPORT_TO_VERDICT.get(support_role)
            if expected_verdict is None:
                continue

            selected_evidence = to_selected_evidence(turn, evidence)
            cases.append(
                BenchmarkCase(
                    case_id=f"turn_{turn}_{evidence['id']}",
                    turn=turn,
                    selected_action=best_action,
                    selected_evidence=selected_evidence,
                    learner_justification={
                        "mode": "typed",
                        "source": "benchmark_generated",
                        "transcript": build_justification(best_action, evidence, support_role),
                    },
                    vlm_output=vlm_output_from_evidence(evidence, saved_vlm),
                    expected_verdict=expected_verdict,
                    expected_support_role=support_role,
                )
            )

        unrelated_evidence = build_unrelated_evidence(turn)
        cases.append(
            BenchmarkCase(
                case_id=f"turn_{turn}_unsupported_unrelated",
                turn=turn,
                selected_action=best_action,
                selected_evidence=unrelated_evidence,
                learner_justification={
                    "mode": "typed",
                    "source": "benchmark_generated",
                    "transcript": (
                        "This item does not show the selected IAM or CloudTrail incident fields. "
                        "It should not be treated as direct support for the chosen response."
                    ),
                },
                vlm_output={
                    "evidence_type": "unknown",
                    "visible_evidence_summary": "Unrelated operational note with no incident fields.",
                    "visible_facts_extracted": [
                        "No CloudTrail event name is visible",
                        "No IAM principal is visible",
                        "No source IP address is visible",
                    ],
                    "important_visible_fields": [],
                    "security_relevance": "unknown",
                    "supports_selected_action": "unknown",
                    "not_proven_by_this_evidence": [
                        "Does not prove identity misuse, credential exposure, or audit-log tampering."
                    ],
                    "reason": "The evidence is unrelated to the selected action.",
                },
                expected_verdict="Unsupported",
                expected_support_role="unsupported",
            )
        )

    return cases


def find_by_role(items: list[dict[str, Any]], key: str, role: str) -> dict[str, Any] | None:
    for item in items:
        if str(item.get(key, "")).lower() == role:
            return item
    return None


def normalise_support_role(evidence: dict[str, Any]) -> str:
    return str(evidence.get("support_role") or evidence.get("supportRole") or "").lower()


def to_selected_evidence(turn: int, evidence: dict[str, Any]) -> dict[str, Any]:
    evidence_id = evidence["id"]
    return {
        "id": evidence_id,
        "title": evidence.get("title"),
        "type": evidence.get("type"),
        "summary": evidence.get("summary"),
        "whyItMayMatter": evidence.get("why_it_may_matter"),
        "supportRole": normalise_support_role(evidence),
        "template": evidence.get("template"),
        "imageUrl": f"/generated_evidence/turn_{turn}/{evidence_id}.png",
        "imagePath": str(EVIDENCE_DIR / f"turn_{turn}" / f"{evidence_id}.png"),
    }


def build_justification(
    action: dict[str, Any],
    evidence: dict[str, Any],
    support_role: str,
) -> str:
    # NOTE (evaluation-validity fix): this used to state the ground-truth
    # support_role directly in plain English ("...should provide {support_role}
    # support..."), which leaked the answer into the security model's prompt via
    # learner_justification.transcript. It now only describes the action/evidence
    # selection and the evidence's own summary, without asserting how strongly it
    # supports the action -- that judgement is left for the model to make.
    return (
        f"I selected {action.get('title')} and chose {evidence.get('title')} as "
        f"supporting evidence. It shows {evidence.get('summary', 'the relevant incident context')}."
    )


def vlm_output_from_evidence(
    evidence: dict[str, Any],
    saved_vlm: dict[str, Any],
) -> dict[str, Any]:
    facts = evidence.get("facts") or {}
    visible_facts = flatten_facts(facts)[:8]

    # NOTE (evaluation-validity fix): security_relevance / supports_selected_action
    # used to be set directly from the case's own ground-truth support_role, which
    # leaked the answer straight into the security model's prompt. They are now
    # fixed to "unknown" -- the model must infer strong/partial/weak/unsupported
    # from the visible facts alone, not read it off this field.
    return {
        "evidence_type": evidence.get("template") or evidence.get("type") or "unknown",
        "visible_evidence_summary": evidence.get("summary", ""),
        "visible_facts_extracted": visible_facts
        or saved_vlm.get("visible_facts_extracted", []),
        "important_visible_fields": visible_facts[:4]
        or saved_vlm.get("important_visible_fields", []),
        "security_relevance": "unknown",
        "supports_selected_action": "unknown",
        "not_proven_by_this_evidence": [
            "Additional corroborating evidence may still be needed."
        ],
        "reason": evidence.get("why_it_may_matter", evidence.get("summary", "")),
    }


def flatten_facts(value: Any, prefix: str = "") -> list[str]:
    facts: list[str] = []
    if isinstance(value, dict):
        for key, inner in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            facts.extend(flatten_facts(inner, next_prefix))
    elif isinstance(value, list):
        for index, inner in enumerate(value):
            next_prefix = f"{prefix}[{index}]"
            facts.extend(flatten_facts(inner, next_prefix))
    elif value is not None and str(value).strip():
        facts.append(f"{prefix}: {value}")
    return facts


def build_unrelated_evidence(turn: int) -> dict[str, Any]:
    return {
        "id": f"unrelated_policy_note_turn_{turn}",
        "title": "Unrelated Policy Note",
        "type": "unknown",
        "summary": "A generic reminder to review policy documentation.",
        "whyItMayMatter": "This is general context only and does not prove the incident.",
        "supportRole": "unsupported",
        "template": "unknown",
        "imageUrl": "",
        "imagePath": "",
    }


def build_vlm_cases(cases: list[BenchmarkCase]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        image_path = Path(case.selected_evidence.get("imagePath") or "")
        # is_file() rather than exists(): an empty imagePath (used by the synthetic
        # "unrelated evidence" case) resolves to Path(""), which is equivalent to
        # Path(".") -- the current directory -- and exists() is True for a directory,
        # incorrectly treating it as a real image and crashing Image.open() later.
        if not image_path.is_file() or str(image_path) in seen:
            continue
        seen.add(str(image_path))
        expected_keywords = evidence_keywords(case)
        rows.append(
            {
                "case_id": case.case_id,
                "turn": case.turn,
                "image_path": str(image_path),
                "expected_template": case.selected_evidence.get("template", ""),
                "expected_support_role": case.expected_support_role,
                "expected_keywords": expected_keywords,
                "selected_action": case.selected_action,
                "selected_evidence": case.selected_evidence,
            }
        )
    return rows


def evidence_keywords(case: BenchmarkCase) -> list[str]:
    text = " ".join(
        [
            str(case.selected_evidence.get("title", "")),
            str(case.selected_evidence.get("summary", "")),
            " ".join(case.vlm_output.get("visible_facts_extracted", [])),
        ]
    ).lower()
    tokens = re.findall(r"[a-z][a-z0-9_/-]{2,}", text)
    stop_words = {
        "this",
        "that",
        "with",
        "from",
        "shows",
        "selected",
        "evidence",
        "support",
        "because",
        "relevant",
        "incident",
        "additional",
        "corroborating",
        "needed",
    }
    keywords: list[str] = []
    for token in tokens:
        token = token.replace("_", " ")
        if token in stop_words or token in keywords:
            continue
        keywords.append(token)
        if len(keywords) >= 8:
            break
    return keywords


def run_vlm_model_benchmarks(
    vlm_cases: list[dict[str, Any]],
    model_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not vlm_cases:
        return rows

    for model_spec in model_specs:
        print(f"Loading VLM candidate: {model_spec['name']}")
        try:
            runner = load_vlm_runner(model_spec)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            rows.extend(vlm_load_error_rows(model_spec, vlm_cases, error))
            continue

        for case in vlm_cases:
            prompt = build_vlm_prompt(case)
            started = time.perf_counter()
            try:
                output_text = runner(case["image_path"], prompt)
                error = ""
            except Exception as exc:
                output_text = ""
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            output_lower = output_text.lower()
            expected_keywords = case["expected_keywords"]
            matched = [keyword for keyword in expected_keywords if keyword.lower() in output_lower]
            keyword_recall = len(matched) / len(expected_keywords) if expected_keywords else 0.0
            predicted_support = extract_support_role(output_text)
            rows.append(
                {
                    "model_name": model_spec["name"],
                    "model_id": model_spec["model_id"],
                    "case_id": case["case_id"],
                    "turn": case["turn"],
                    "expected_template": case["expected_template"],
                    "expected_support_role": case["expected_support_role"],
                    "predicted_support_role": predicted_support,
                    "support_role_correct": predicted_support == case["expected_support_role"],
                    "expected_keywords": ", ".join(expected_keywords),
                    "matched_keywords": ", ".join(matched),
                    "keyword_recall": round(keyword_recall, 3),
                    "latency_seconds": round(elapsed, 3),
                    "output_text": output_text,
                    "error": error,
                }
            )
        release_model_runner(runner)
    return rows


def load_vlm_runner(model_spec: dict[str, Any]):
    model_id = model_spec["model_id"]
    if LOCAL_FILES_ONLY and not model_config_is_cached(model_id):
        raise FileNotFoundError(
            f"{model_id} is not cached locally. Rerun with --allow-downloads to download it."
        )

    if model_id.startswith("Qwen/Qwen2.5-VL"):
        return load_qwen_vlm_runner(
            model_id=model_id,
            model_class=Qwen2_5_VLForConditionalGeneration,
        )

    if model_id.startswith("Qwen/Qwen2-VL"):
        return load_qwen_vlm_runner(
            model_id=model_id,
            model_class=Qwen2VLForConditionalGeneration,
        )

    if model_id.startswith("microsoft/Florence-2"):
        return load_florence_runner(model_id)

    if model_id.startswith("llava-hf/llava-onevision"):
        return load_llava_onevision_runner(model_id)

    image_pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        device=-1,
        torch_dtype=torch.float32,
        trust_remote_code=bool(model_spec.get("trust_remote_code", False)),
        model_kwargs={"local_files_only": LOCAL_FILES_ONLY},
    )

    def run_vlm(image_path: str, prompt: str) -> str:
        message_variants = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        ]
        last_error: Exception | None = None
        for messages in message_variants:
            try:
                result = image_pipe(
                    messages,
                    max_new_tokens=VLM_MAX_NEW_TOKENS,
                    return_full_text=False,
                )
                return normalise_pipeline_text(result)
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return ""

    run_vlm._model = image_pipe  # type: ignore[attr-defined]
    return run_vlm


def load_llava_onevision_runner(model_id: str):
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=LOCAL_FILES_ONLY,
    )
    # CPU + bfloat16 rather than device()/float32: the same MPS OOM / swap-thrashing
    # bug documented in load_text_runner() (see the note there) was found here too
    # during the VLM rerun -- vm.swapusage showed ~10GB/11GB used and the process sat
    # in uninterruptible sleep (STAT "UN") making no progress. Forcing CPU with
    # bfloat16 avoids both the MPS allocation cap and the float32 memory footprint.
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        local_files_only=LOCAL_FILES_ONLY,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to("cpu")
    model.eval()

    def run_llava(image_path: str, prompt: str) -> str:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            input_text = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
            )
            inputs = processor(
                images=image,
                text=input_text,
                return_tensors="pt",
            )
        inputs = {key: value.to("cpu") for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=VLM_MAX_NEW_TOKENS,
                do_sample=False,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return processor.decode(generated_ids, skip_special_tokens=True)

    run_llava._model = (model, processor)  # type: ignore[attr-defined]
    return run_llava


def load_qwen_vlm_runner(model_id: str, model_class):
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=LOCAL_FILES_ONLY,
    )
    # CPU + bfloat16 rather than device()/float32 -- see the note in
    # load_llava_onevision_runner() for why (MPS OOM / swap-thrashing bug found
    # during the VLM rerun, same class as the earlier text-model fix).
    model = model_class.from_pretrained(
        model_id,
        local_files_only=LOCAL_FILES_ONLY,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to("cpu")
    model.eval()

    def run_qwen_vlm(image_path: str, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        input_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[input_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cpu")
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=VLM_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return processor.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    run_qwen_vlm._model = (model, processor)  # type: ignore[attr-defined]
    return run_qwen_vlm


def load_florence_runner(model_id: str):
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=True,
    )
    # CPU rather than device() -- see the note in load_llava_onevision_runner() for
    # why (MPS OOM / swap-thrashing bug found during the VLM rerun, same class as
    # the earlier text-model fix). Unlike the other two VLM loaders this one stays
    # at float32 rather than bfloat16: Florence-2-base-ft is small (~230M params, no
    # memory-pressure reason to shrink it), and switching it to bfloat16 caused a
    # genuine new bug -- its own image processor always returns float32 pixel_values,
    # so the model's bfloat16 weights and the float32 input tensor mismatched and
    # every case failed with "Input type (float) and bias type (c10::BFloat16)
    # should be the same". float32 throughout avoids that without reintroducing the
    # MPS/swap problem, since the fix that mattered was moving off MPS, not the dtype.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.to("cpu")
    model.eval()

    def run_florence(image_path: str, prompt: str) -> str:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            parts: list[str] = []
            for task_prompt in ["<OCR>", "<MORE_DETAILED_CAPTION>"]:
                inputs = processor(
                    text=task_prompt,
                    images=image,
                    return_tensors="pt",
                )
                inputs = {key: value.to("cpu") for key, value in inputs.items()}
                with torch.inference_mode():
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=VLM_MAX_NEW_TOKENS,
                        do_sample=False,
                        num_beams=1,
                    )
                generated_text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=False,
                )[0]
                try:
                    parsed = processor.post_process_generation(
                        generated_text,
                        task=task_prompt,
                        image_size=(image.width, image.height),
                    )
                    parts.append(json.dumps(parsed))
                except Exception:
                    parts.append(generated_text)
        return "\n".join(parts)

    run_florence._model = (model, processor)  # type: ignore[attr-defined]
    return run_florence


def model_config_is_cached(model_id: str) -> bool:
    try:
        cached_file(model_id, "config.json", local_files_only=True)
        return True
    except Exception:
        return False


def build_vlm_prompt(case: dict[str, Any]) -> str:
    # NOTE (evaluation-validity fix): case["selected_action"] carries "choice_role" and
    # case["selected_evidence"] carries "supportRole" -- authoring-only ground-truth labels
    # used only for scoring. Dumping them straight into the prompt leaked the answer to the
    # model (this was the one prompt builder the Aug 16 evaluation-validity fix missed --
    # build_security_prompt already redacted "supportRole" but build_vlm_prompt did not), so
    # the prompt is now built from redacted copies that omit them, mirroring the
    # GROUND_TRUTH_KEYS stripping applied in the production cloudir.ai_models code.
    ground_truth_keys = {"choice_role", "choiceRole", "support_role", "supportRole"}
    prompt_action = {k: v for k, v in case["selected_action"].items() if k not in ground_truth_keys}
    prompt_evidence = {k: v for k, v in case["selected_evidence"].items()
                       if k in {"id", "title", "type", "template"}}
    return f"""
Inspect this AWS-style incident-response evidence screenshot.

Selected action:
{json.dumps(prompt_action, indent=2)}

Selected evidence metadata:
{json.dumps(prompt_evidence, indent=2)}

Return compact JSON with these fields IN THIS ORDER:
- evidence_type
- supports_selected_action: strong, partial, weak, or unsupported
- visible_facts
- missing_or_not_proven

Put supports_selected_action early, right after evidence_type, so it is not cut off if the
response is truncated.

Only use what is visible in the screenshot. Do not invent fields.
""".strip()


def vlm_load_error_rows(
    model_spec: dict[str, Any],
    vlm_cases: list[dict[str, Any]],
    error: str,
) -> list[dict[str, Any]]:
    return [
        {
            "model_name": model_spec["name"],
            "model_id": model_spec["model_id"],
            "case_id": case["case_id"],
            "turn": case["turn"],
            "expected_template": case["expected_template"],
            "expected_support_role": case["expected_support_role"],
            "predicted_support_role": "load_error",
            "support_role_correct": False,
            "expected_keywords": ", ".join(case["expected_keywords"]),
            "matched_keywords": "",
            "keyword_recall": 0.0,
            "latency_seconds": 0.0,
            "output_text": "",
            "error": error,
        }
        for case in vlm_cases
    ]


def normalise_pipeline_text(result: Any) -> str:
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            generated = first.get("generated_text") or first.get("text") or ""
            if isinstance(generated, list):
                return json.dumps(generated)
            return str(generated)
        return str(first)
    if isinstance(result, dict):
        return str(result.get("generated_text") or result.get("text") or result)
    return str(result)


# Word-form variants for the STT keyword vocabulary (see STT_SAMPLE_DEFINITIONS below). A
# spoken paraphrase rarely repeats a keyword's exact inflection (e.g. says "partly" instead
# of "partial"), so plain substring matching under-counts genuine meaning-preserving
# transcripts. This is a small, explicit map rather than a generic stemmer/prefix-truncation
# on purpose: truncating to a fixed-length prefix caused false positives between unrelated
# compound words that share a root (e.g. "cloudtrail" and "cloudwatch" both truncate to
# "clou"), which a fixed vocabulary map avoids entirely.
_STT_WORD_VARIANTS: dict[str, set[str]] = {
    "partial": {"partial", "partly", "part"},
    "support": {"support", "supports", "supporting", "supported"},
    "active": {"active", "activated", "activate"},
    "disable": {"disable", "disabled", "disabling"},
    "rotate": {"rotate", "rotated", "rotating"},
}


def stt_word_variants(word: str) -> set[str]:
    return _STT_WORD_VARIANTS.get(word.lower(), {word.lower()})


def stt_keyword_matches(keyword_phrase: str, transcript_lower: str) -> bool:
    """A (possibly multi-word) keyword matches if every one of its words is present."""
    return all(
        any(variant in transcript_lower for variant in stt_word_variants(word))
        for word in keyword_phrase.lower().split()
    )


def extract_support_role(output_text: str) -> str:
    lowered = output_text.lower()
    for role in ["unsupported", "partial", "strong", "weak"]:
        if role in lowered:
            return role
    return "unparseable"


def run_security_model_benchmarks(cases: list[BenchmarkCase]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for model_spec in SECURITY_MODELS:
        print(f"Loading security candidate: {model_spec['name']}")
        try:
            runner = load_text_runner(model_spec)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            results.extend(text_load_error_rows(model_spec, cases, error))
            continue
        for case in cases:
            prompt = build_security_prompt(case)
            started = time.perf_counter()
            try:
                output_text = runner(prompt)
                error = ""
            except Exception as exc:
                output_text = ""
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            predicted = extract_verdict(output_text)
            results.append(
                {
                    "model_name": model_spec["name"],
                    "model_id": model_spec["model_id"],
                    "case_id": case.case_id,
                    "turn": case.turn,
                    "expected_support_role": case.expected_support_role,
                    "expected_verdict": case.expected_verdict,
                    "predicted_verdict": predicted,
                    "correct": predicted == case.expected_verdict,
                    "latency_seconds": round(elapsed, 3),
                    "output_text": output_text,
                    "error": error,
                }
            )
        release_model_runner(runner)
    return results


def load_text_runner(model_spec: dict[str, str]):
    kind = model_spec["kind"]
    model_id = model_spec["model_id"]
    trust_remote_code = bool(model_spec.get("trust_remote_code", False))

    if kind == "zero_shot":
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=LOCAL_FILES_ONLY,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            local_files_only=LOCAL_FILES_ONLY,
            trust_remote_code=trust_remote_code,
        )
        classifier = pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
        )

        def run_zero_shot(prompt: str) -> str:
            result = classifier(
                prompt,
                candidate_labels=VERDICT_LABELS,
                hypothesis_template="This incident-response evidence provides {}.",
            )
            return str(result["labels"][0])

        run_zero_shot._model = classifier  # type: ignore[attr-defined]
        return run_zero_shot

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=trust_remote_code,
    )

    # NOTE: causal_lm/seq2seq text models are forced onto CPU here rather than
    # device() (which would pick MPS on Apple Silicon). A genuine bug was found
    # during the expanded security benchmark run: fdtn-ai/Foundation-Sec-8B-Instruct
    # in float32 needs ~32GB just for weights, which exceeds this machine's ~30GB
    # MPS allocation cap and raises "RuntimeError: MPS backend out of memory"
    # during generate(). Because clear_torch_memory()/torch.mps.empty_cache() does
    # not fully release memory after a mid-generate crash, every subsequent text
    # model in the same run then also failed the same way (verified in
    # security_model_focused_results.json prior to this fix — all six candidates,
    # including small Qwen/flan-t5 models, showed the identical MPS OOM error).
    # CPU has no such hard allocation cap, so this is a reliability fix, not a
    # metric-optimisation change.
    text_device = "cpu"

    if kind == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            local_files_only=LOCAL_FILES_ONLY,
            trust_remote_code=trust_remote_code,
        )
        model.to(text_device)
        model.eval()

        def run_seq2seq(prompt: str) -> str:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {key: value.to(text_device) for key, value in inputs.items()}
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                )
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

        run_seq2seq._model = model  # type: ignore[attr-defined]
        return run_seq2seq

    # bfloat16 rather than float32: this machine has 24GB RAM, and
    # fdtn-ai/Foundation-Sec-8B-Instruct needs ~32GB in float32, which does not
    # OOM-crash on CPU (unlike MPS) but instead causes catastrophic swap
    # thrashing (observed: RSS ~30GB, ~26GB compressed swap, process effectively
    # stalled with near-0% CPU utilisation). bfloat16 halves the memory
    # footprint (~16GB), which fits in RAM without swapping, and is
    # numerically safe for inference (same exponent range as float32).
    text_dtype = torch.bfloat16 if text_device == "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=trust_remote_code,
        torch_dtype=text_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(text_device)
    model.eval()

    def run_causal(prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a strict cloud incident-response evaluator.",
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            input_text = prompt
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {key: value.to(text_device) for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)

    run_causal._model = model  # type: ignore[attr-defined]
    return run_causal


def release_model_runner(runner) -> None:
    if hasattr(runner, "_model"):
        del runner._model
    del runner
    clear_torch_memory()


def text_load_error_rows(
    model_spec: dict[str, Any],
    cases: list[BenchmarkCase],
    error: str,
) -> list[dict[str, Any]]:
    return [
        {
            "model_name": model_spec["name"],
            "model_id": model_spec["model_id"],
            "case_id": case.case_id,
            "turn": case.turn,
            "expected_support_role": case.expected_support_role,
            "expected_verdict": case.expected_verdict,
            "predicted_verdict": "Load Error",
            "correct": False,
            "latency_seconds": 0.0,
            "output_text": "",
            "error": error,
        }
        for case in cases
    ]


def build_security_prompt(case: BenchmarkCase) -> str:
    # Match production judgement inputs: neither author labels nor the VLM's
    # own support opinion should tell the security model the expected answer.
    ground_truth_keys = {"choice_role", "choiceRole", "support_role", "supportRole"}
    prompt_action = {k: v for k, v in case.selected_action.items() if k not in ground_truth_keys}
    prompt_evidence = {k: v for k, v in case.selected_evidence.items()
                       if k in {"id", "title", "type", "template"}}
    prompt_vlm = {k: v for k, v in case.vlm_output.items()
                  if k in {"evidence_type", "visible_evidence_summary", "visible_facts_extracted",
                           "important_visible_fields", "not_proven_by_this_evidence"}}
    return f"""
Classify whether the selected evidence supports the selected incident-response action.

Return exactly one label from:
- Strong Support
- Partial Support
- Weak Support
- Unsupported

Selected action:
{json.dumps(prompt_action, indent=2)}

Selected evidence:
{json.dumps(prompt_evidence, indent=2)}

Learner justification:
{case.learner_justification.get("transcript", "")}

Visible facts from the evidence:
{json.dumps(prompt_vlm, indent=2)}

Answer with only the label.
""".strip()


def extract_verdict(output_text: str) -> str:
    text = output_text.strip()
    lowered = text.lower()
    for label in VERDICT_LABELS:
        if label.lower() in lowered:
            return label
    if "strong" in lowered:
        return "Strong Support"
    if "partial" in lowered:
        return "Partial Support"
    if "weak" in lowered:
        return "Weak Support"
    if "unsupported" in lowered or "not support" in lowered:
        return "Unsupported"
    return "Unparseable"


def run_coach_model_benchmarks(cases: list[BenchmarkCase]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for model_spec in COACH_MODELS:
        print(f"Loading coach candidate: {model_spec['name']}")
        try:
            runner = load_text_runner(model_spec)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            results.extend(coach_load_error_rows(model_spec, cases, error))
            continue
        for case in cases:
            prompt = build_coach_prompt(case)
            started = time.perf_counter()
            try:
                output_text = runner(prompt)
                error = ""
            except Exception as exc:
                output_text = ""
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            breakdown = coach_rubric_breakdown(output_text, case)
            row = {
                "model_name": model_spec["name"],
                "model_id": model_spec["model_id"],
                "case_id": case.case_id,
                "turn": case.turn,
                "expected_verdict": case.expected_verdict,
                "automated_rubric_score": sum(breakdown.values()),
                "automated_rubric_max": len(COACH_RUBRIC_CRITERIA),
                "latency_seconds": round(elapsed, 3),
                "word_count": len(output_text.split()),
                "output_text": output_text,
                "error": error,
            }
            for criterion, passed in breakdown.items():
                row[f"criterion_{criterion}"] = passed
            results.append(row)
        release_model_runner(runner)
    return results


def coach_load_error_rows(
    model_spec: dict[str, Any],
    cases: list[BenchmarkCase],
    error: str,
) -> list[dict[str, Any]]:
    return [
        {
            "model_name": model_spec["name"],
            "model_id": model_spec["model_id"],
            "case_id": case.case_id,
            "turn": case.turn,
            "expected_verdict": case.expected_verdict,
            "automated_rubric_score": 0,
            "automated_rubric_max": len(COACH_RUBRIC_CRITERIA),
            "latency_seconds": 0.0,
            "word_count": 0,
            "output_text": "",
            "error": error,
            **{f"criterion_{c}": False for c in COACH_RUBRIC_CRITERIA},
        }
        for case in cases
    ]


def build_coach_prompt(case: BenchmarkCase) -> str:
    return f"""
Write concise learner feedback for a cloud incident-response trainee.

Selected action:
{case.selected_action.get("title")}

Selected evidence:
{case.selected_evidence.get("title")} - {case.selected_evidence.get("summary")}

Security verdict:
{case.expected_verdict}

Visible facts:
{json.dumps(case.vlm_output.get("visible_facts_extracted", []), indent=2)}

The feedback should explain why the evidence is or is not sufficient and what to check next.
Limit to 80 words.
""".strip()


COACH_RUBRIC_CRITERIA = [
    "output_nonempty",
    "concise_length_20_140_words",
    "references_selected_evidence_or_action",
    "uses_support_terminology",
    "acknowledges_expected_verdict",
    "proposes_next_investigation_step",
    "avoids_hidden_truth_leak",
]


def _hidden_truth_phrases() -> list[str]:
    """Distinctive phrases from the scenario's hidden ground truth that a coach
    response should never reveal verbatim (it should guide, not spoil)."""
    path = RUNTIME_DIR / "hidden_truth.json"
    if not path.exists():
        return []
    try:
        data = read_json(path)
    except Exception:
        return []
    phrases: list[str] = []
    primary_risk = data.get("primary_risk")
    if isinstance(primary_risk, str) and len(primary_risk) > 20:
        phrases.append(primary_risk.lower())
    for step in data.get("likely_attack_path", []) or []:
        if isinstance(step, str) and len(step) > 8:
            phrases.append(step.lower())
    return phrases


def coach_rubric_breakdown(output_text: str, case: BenchmarkCase) -> dict[str, bool]:
    """Automated Coach Response Rubric Score — each criterion is an objective,
    code-checkable heuristic over the raw model output. This is NOT a measure
    of human-judged coaching quality; it only checks for the presence/absence
    of specific, pre-defined textual signals."""
    text = output_text.strip()
    lower = text.lower()
    word_count = len(text.split())
    leak_phrases = _hidden_truth_phrases()

    return {
        "output_nonempty": bool(text),
        "concise_length_20_140_words": 20 <= word_count <= 140,
        "references_selected_evidence_or_action": (
            case.selected_evidence.get("title", "").lower() in lower
            or case.selected_action.get("title", "").lower() in lower
        ),
        "uses_support_terminology": any(
            token in lower for token in ["evidence", "support", "strong", "partial", "weak", "unsupported"]
        ),
        "acknowledges_expected_verdict": case.expected_verdict.lower() in lower,
        "proposes_next_investigation_step": any(
            token in lower for token in ["next", "check", "investigate", "review", "confirm"]
        ),
        "avoids_hidden_truth_leak": not any(phrase in lower for phrase in leak_phrases),
    }


def coach_score(output_text: str, case: BenchmarkCase) -> int:
    """Automated Coach Response Rubric Score out of 7 criteria. This is an
    automated, code-checkable heuristic score — not a human-rated quality
    score. See COACH_RUBRIC_CRITERIA / coach_rubric_breakdown() for the
    precise per-criterion definitions."""
    return sum(coach_rubric_breakdown(output_text, case).values())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein word-level WER = (S + D + I) / N_reference_words.

    Lower is better; 0.0 = perfect transcription, 1.0 = as many edits as
    reference words (can exceed 1.0 if the hypothesis is much longer/noisier
    than the reference). Implemented locally with dynamic-programming edit
    distance over word tokens — no external WER library is a project
    dependency, so this avoids adding one.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return 0.0 if m == 0 else 1.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return round(dp[n][m] / n, 4)


def run_stt_benchmark(sample_limit: int = 0) -> list[dict[str, Any]]:
    samples = discover_stt_samples()
    if sample_limit > 0:
        samples = samples[:sample_limit]

    if not samples:
        return [
            {
                "model_name": "whisper-base",
                "model_id": os.getenv("HF_VOICE_MODEL_ID", "openai/whisper-base"),
                "sample_id": "no_audio_samples",
                "expected_text": "",
                "transcript": "",
                "expected_keywords": "",
                "matched_keywords": "",
                "keyword_recall": 0.0,
                "meaning_preserved": False,
                "word_error_rate": 1.0,
                "latency_seconds": 0.0,
                "error": (
                    "No audio samples found. Add recordings under "
                    "notebooks/model_benchmarks/stt_audio_samples/audio_samples/."
                ),
            }
        ]

    audio_paths = []
    for sample in samples:
        audio_path = prepare_audio_for_stt(Path(sample["file"]))
        if audio_has_samples(audio_path):
            audio_paths.append((sample, audio_path))

    if not audio_paths:
        return [
            {
                "model_name": "whisper-base",
                "model_id": os.getenv("HF_VOICE_MODEL_ID", "openai/whisper-base"),
                "sample_id": "no_valid_audio_samples",
                "expected_text": "",
                "transcript": "",
                "expected_keywords": "",
                "matched_keywords": "",
                "keyword_recall": 0.0,
                "meaning_preserved": False,
                "word_error_rate": 1.0,
                "latency_seconds": 0.0,
                "error": "Recorded audio files exist but none contains readable samples.",
            }
        ]

    rows: list[dict[str, Any]] = []
    for model_spec in STT_MODELS:
        model_id = model_spec["model_id"]
        print(f"Loading STT candidate: {model_spec['name']}")
        try:
            processor = AutoProcessor.from_pretrained(
                model_id,
                local_files_only=LOCAL_FILES_ONLY,
            )
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                local_files_only=LOCAL_FILES_ONLY,
            )
            recogniser = pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=-1,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            rows.extend(stt_load_error_rows(model_spec, audio_paths, error))
            continue

        for sample, audio_path in audio_paths:
            started = time.perf_counter()
            try:
                result = recogniser(str(audio_path), return_timestamps=True)
                transcript = (
                    str(result.get("text", "")).strip()
                    if isinstance(result, dict)
                    else str(result).strip()
                )
                error = ""
            except Exception as exc:
                transcript = ""
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            transcript_lower = transcript.lower()
            expected_keywords = sample.get("keywords") or sample.get("expected_meaning_keywords") or []
            matched = [
                keyword for keyword in expected_keywords
                if stt_keyword_matches(keyword, transcript_lower)
            ]
            recall = len(matched) / len(expected_keywords) if expected_keywords else 0.0
            rows.append(
                {
                    "model_name": model_spec["name"],
                    "model_id": model_id,
                    "sample_id": sample.get("id") or audio_path.stem,
                    "expected_text": sample.get("text") or sample.get("expected_text", ""),
                    "transcript": transcript,
                    "expected_keywords": ", ".join(expected_keywords),
                    "matched_keywords": ", ".join(matched),
                    "keyword_recall": round(recall, 3),
                    # >= 0.5 = at least half of the expected keyword-concepts recovered. The old
                    # 0.8 threshold effectively demanded near-exact phrase repetition, which
                    # normal spoken paraphrasing rarely produces even when meaning is preserved
                    # (see notebook Section 4.5).
                    "meaning_preserved": recall >= 0.5,
                    "word_error_rate": word_error_rate(sample.get("text") or sample.get("expected_text", ""), transcript),
                    "latency_seconds": round(elapsed, 3),
                    "error": error,
                }
            )
        del recogniser
        del model
        del processor
        clear_torch_memory()
    return rows


def stt_load_error_rows(
    model_spec: dict[str, Any],
    audio_paths: list[tuple[dict[str, Any], Path]],
    error: str,
) -> list[dict[str, Any]]:
    return [
        {
            "model_name": model_spec["name"],
            "model_id": model_spec["model_id"],
            "sample_id": sample.get("id") or audio_path.stem,
            "expected_text": sample.get("text") or sample.get("expected_text", ""),
            "transcript": "",
            "expected_keywords": ", ".join(
                sample.get("keywords") or sample.get("expected_meaning_keywords") or []
            ),
            "matched_keywords": "",
            "keyword_recall": 0.0,
            "meaning_preserved": False,
            "word_error_rate": 1.0,
            "latency_seconds": 0.0,
            "error": error,
        }
        for sample, audio_path in audio_paths
    ]


def discover_stt_samples() -> list[dict[str, Any]]:
    if not STT_SAMPLE_DIR.exists():
        return []

    rows: list[dict[str, Any]] = []
    for path in sorted(STT_SAMPLE_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".wav", ".m4a", ".mp3", ".flac", ".webm"}:
            continue
        definition = STT_SAMPLE_DEFINITIONS.get(normalise_audio_stem(path.stem), {})
        rows.append(
            {
                "id": definition.get("id") or path.stem,
                "file": str(path),
                "text": definition.get("text", ""),
                "keywords": definition.get("keywords", default_stt_keywords(path.stem)),
            }
        )
    return rows


def prepare_audio_for_stt(audio_path: Path) -> Path:
    if audio_path.suffix.lower() == ".wav":
        return audio_path

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return audio_path

    STT_CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = STT_CONVERTED_DIR / f"{audio_path.stem}.wav"
    source_mtime = audio_path.stat().st_mtime
    if wav_path.exists() and wav_path.stat().st_mtime >= source_mtime:
        return wav_path

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return wav_path


def normalise_audio_stem(stem: str) -> str:
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def default_stt_keywords(stem: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", stem.lower())
    useful = [token for token in tokens if token not in {"sample", "audio", "recording"}]
    return useful[:5] or [stem.lower()]


def audio_has_samples(audio_path: Path) -> bool:
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return info.frames > 0 and info.duration > 0
    except Exception:
        return audio_path.exists() and audio_path.stat().st_size > 4096


def audit_evidence_images() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(EVIDENCE_DIR.glob("turn_*/*.png")):
        started = time.perf_counter()
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            ok = True
            error = ""
        except Exception as exc:
            width = height = 0
            ok = False
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "image": str(path.relative_to(PROJECT_ROOT)),
                "readable": ok,
                "width": width,
                "height": height,
                "latency_seconds": round(elapsed, 4),
                "error": error,
            }
        )
    return rows


def summarise_classification_results(
    rows: list[dict[str, Any]],
    group_key: str,
) -> list[dict[str, Any]]:
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return []

    summary: list[dict[str, Any]] = []
    for group, group_df in dataframe.groupby(group_key):
        summary.append(
            {
                group_key: group,
                "cases": int(len(group_df)),
                "accuracy": round(float(group_df["correct"].mean()), 3),
                "avg_latency_seconds": round(float(group_df["latency_seconds"].mean()), 3),
                "total_latency_seconds": round(float(group_df["latency_seconds"].sum()), 3),
                "unparseable": int((group_df["predicted_verdict"] == "Unparseable").sum()),
                "errors": int(group_df["error"].astype(bool).sum()),
            }
        )
    return summary


def summarise_coach_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return []
    summary: list[dict[str, Any]] = []
    criterion_cols = [c for c in dataframe.columns if c.startswith("criterion_")]
    for model_name, group_df in dataframe.groupby("model_name"):
        row = {
            "model_name": model_name,
            "cases": int(len(group_df)),
            "avg_automated_rubric_score": round(float(group_df["automated_rubric_score"].mean()), 3),
            "automated_rubric_max": int(group_df["automated_rubric_max"].iloc[0]) if "automated_rubric_max" in group_df else len(COACH_RUBRIC_CRITERIA),
            "avg_latency_seconds": round(float(group_df["latency_seconds"].mean()), 3),
            "avg_word_count": round(float(group_df["word_count"].mean()), 1),
            "errors": int(group_df["error"].astype(bool).sum()),
        }
        for col in criterion_cols:
            row[f"{col}_pass_rate"] = round(float(group_df[col].mean()), 3)
        summary.append(row)
    return summary


def summarise_vlm_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return []
    summary: list[dict[str, Any]] = []
    for model_name, group_df in dataframe.groupby("model_name"):
        summary.append(
            {
                "model_name": model_name,
                "cases": int(len(group_df)),
                "support_role_accuracy": round(float(group_df["support_role_correct"].mean()), 3),
                "avg_keyword_recall": round(float(group_df["keyword_recall"].mean()), 3),
                "avg_latency_seconds": round(float(group_df["latency_seconds"].mean()), 3),
                "errors": int(group_df["error"].astype(bool).sum()),
            }
        )
    return summary


def summarise_stt_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return []
    skipped_sample_ids = {"no_stt_manifest", "no_audio_samples", "no_valid_audio_samples"}
    real_samples = dataframe[~dataframe["sample_id"].isin(skipped_sample_ids)]
    summary: list[dict[str, Any]] = []
    for model_name, group_df in real_samples.groupby("model_name"):
        wer_series = group_df["word_error_rate"] if "word_error_rate" in group_df else pd.Series(dtype=float)
        summary.append(
            {
                "model_name": model_name,
                "model_id": str(group_df.iloc[0]["model_id"]),
                "samples": int(len(group_df)),
                "mean_wer": round(float(wer_series.mean()), 4) if not wer_series.empty else None,
                "median_wer": round(float(wer_series.median()), 4) if not wer_series.empty else None,
                "min_wer": round(float(wer_series.min()), 4) if not wer_series.empty else None,
                "max_wer": round(float(wer_series.max()), 4) if not wer_series.empty else None,
                "meaning_preservation_rate": round(float(group_df["meaning_preserved"].mean()), 3),
                "avg_keyword_recall": round(float(group_df["keyword_recall"].mean()), 3),
                "avg_latency_seconds": round(float(group_df["latency_seconds"].mean()), 3),
                "errors": int(group_df.get("error", pd.Series(dtype=str)).astype(bool).sum()),
            }
        )
    if summary:
        return summary
    return [
        {
            "model_name": str(dataframe.iloc[0]["model_name"]),
            "model_id": str(dataframe.iloc[0]["model_id"]),
            "samples": 0,
            "meaning_preservation_rate": 0.0,
            "avg_keyword_recall": 0.0,
            "avg_latency_seconds": 0.0,
            "errors": int(dataframe.get("error", pd.Series(dtype=str)).astype(bool).sum()),
        }
    ]


def print_summary(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    if not rows:
        print("No rows.")
        return
    print(pd.DataFrame(rows).to_string(index=False))


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def clear_torch_memory() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
