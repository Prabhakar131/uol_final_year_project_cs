from __future__ import annotations

import gc
import itertools
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from qwen_vl_utils import process_vision_info
from cloudir.ai_models.model_worker import isolated_model_call


load_dotenv()


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _block_diagonal_vision_attention(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Vision attention computed separately for each image/window.

    transformers 4.49 builds one dense (patches x patches) mask across every
    image and window, although attention is only allowed inside each block.
    That dense mask made CloudWatch extraction peak at ~24 GB on MPS. Computing
    each block on its own, and splitting a large image's queries into slices
    that still attend to all of its keys, gives the same result with memory
    that grows linearly with image size.
    """
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    if position_embeddings is None:
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos().float(), emb.sin().float())
    q, k = modeling_qwen2_5_vl.apply_rotary_pos_emb_vision(q, k, *position_embeddings)
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    outputs, start = [], 0
    # Neighbouring windows usually have equal lengths; batch each such run.
    for length, run in itertools.groupby(lengths):
        count = len(list(run))
        end = start + length * count
        q_block, k_block, v_block = (t[start:end].reshape(count, length, self.num_heads, -1).transpose(1, 2)
                                     for t in (q, k, v))
        out = torch.cat([F.scaled_dot_product_attention(q_slice, k_block, v_block)
                         for q_slice in q_block.split(_VISION_QUERY_SLICE, dim=2)], dim=2)
        outputs.append(out.transpose(1, 2).reshape(count * length, -1))
        start = end
    return self.proj(torch.cat(outputs))


# Queries per attention call in full-image vision layers: 16 heads x 512 x
# 11k patches is ~0.2 GB in fp16, versus ~4 GB for the whole 2x table at once.
_VISION_QUERY_SLICE = 512


def _use_block_diagonal_vision_attention(model: Qwen2_5_VLForConditionalGeneration) -> None:
    attention_class = type(model.visual.blocks[0].attn)
    if attention_class is modeling_qwen2_5_vl.Qwen2_5_VLVisionSdpaAttention:
        attention_class.forward = _block_diagonal_vision_attention
    if _device() == "mps":
        # Return the vision encoder's freed buffers before the language model
        # prefill. Otherwise MPS keeps both stages' caches (~19 GB vs ~13 GB peak).
        model.visual.register_forward_hook(lambda *_: torch.mps.empty_cache())


class _ReturnFreedMpsMemory(LogitsProcessor):
    """Hands the MPS allocator's freed blocks back to the system every few tokens.

    Generation grows the key/value cache one token at a time, and each step frees
    the previous, slightly smaller buffers. The allocator kept those blocks until
    generation ended: reading a CloudTrail screenshot reached 32 GB of Metal
    memory, swapping on a 24 GB Mac, while live tensors never passed 9.5 GB.
    """

    def __init__(self, every: int):
        self.every = every
        self.steps = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.steps += 1
        if self.steps % self.every == 0:
            torch.mps.empty_cache()
        return scores


_MPS_RELEASE_EVERY = int(os.getenv("HF_IMAGE_MPS_RELEASE_EVERY", "32"))


def _memory_bounded_generation(device: str) -> dict[str, Any]:
    if device != "mps":
        return {}
    return {"logits_processor": LogitsProcessorList([_ReturnFreedMpsMemory(_MPS_RELEASE_EVERY)])}


def _image_dtype(device: str) -> torch.dtype:
    """bfloat16 on a GPU: Qwen2.5-VL's native precision, with float32's range.

    In float16 some screenshots overflow inside the model: every generated token
    became "!", both extraction attempts ran to the token limit (about three
    minutes each) and the evaluation failed. bfloat16 uses the same memory.
    HF_IMAGE_DTYPE=float16 restores the old behaviour for GPUs without bfloat16.
    """

    if device not in {"mps", "cuda"}:
        return torch.float32
    return torch.float16 if os.getenv("HF_IMAGE_DTYPE", "").strip().lower() == "float16" else torch.bfloat16


@lru_cache(maxsize=1)
def _load_image_model() -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    model_id = os.getenv("HF_IMAGE_MODEL_ID")

    if not model_id:
        raise RuntimeError("HF_IMAGE_MODEL_ID is missing from .env")

    device = _device()

    processor = AutoProcessor.from_pretrained(model_id)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=_image_dtype(device),
        low_cpu_mem_usage=True,
    )

    model.to(device)
    model.eval()
    _use_block_diagonal_vision_attention(model)

    return model, processor


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


def unload_image_model() -> None:
    """
    Unloads the cached image/VLM model so run_turn.py can control model lifecycle.
    """

    _load_image_model.cache_clear()
    clear_torch_memory()


def _evidence_prompt_inputs(
    selected_action: dict[str, Any], selected_evidence: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], str]:
    ground_truth_keys = {"choice_role", "choiceRole", "support_role", "supportRole"}
    action = {key: value for key, value in selected_action.items() if key not in ground_truth_keys}
    # A generated summary or "why it may matter" note can hint at the intended
    # answer. Keep identification metadata, but make the image the source of facts.
    evidence = {key: value for key, value in selected_evidence.items()
                if key in {"id", "title", "type", "template"}}
    evidence_type_hint = evidence.get("template") or evidence.get("type") or evidence.get("id") or "unknown"
    return action, evidence, evidence_type_hint


@isolated_model_call
def analyse_evidence_image(
    image_path: str | Path,
    selected_action: dict[str, Any],
    selected_evidence: dict[str, Any],
    *,
    retry_extraction: bool = False,
) -> dict[str, Any]:
    """
    Uses the local Qwen2.5-VL model to inspect the generated evidence screenshot.

    Returns observed screenshot facts and extraction warnings only.
    The security model evaluates support for the selected action.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Evidence image not found: {image_path}")

    with Image.open(image_path) as img:
        img.verify()

    model, processor = _load_image_model()
    device = _device()

    _, evidence_for_prompt, evidence_type_hint = _evidence_prompt_inputs(
        selected_action, selected_evidence
    )

    prompt = f"""
You are the vision-language model inside CloudIR Trainer.

You are inspecting a generated AWS-style evidence screenshot selected by a learner.

Selected evidence metadata:
{json.dumps(evidence_for_prompt, indent=2)}

Look only at the screenshot and extract the visible evidence.

Return ONLY valid JSON with exactly these keys:
{{
  "evidence_type": "unknown",
  "visible_evidence_summary": "short summary of what the screenshot visibly shows",
  "visible_facts_extracted": [
    "visible fact or field-value pair 1",
    "visible fact or field-value pair 2",
    "visible fact or field-value pair 3"
  ],
  "important_visible_fields": [
    "important visible field or value 1",
    "important visible field or value 2"
  ],
  "extraction_warnings": ["any unreadable, cropped, or ambiguous fields"]
}}

Rules:
- The screenshot is likely evidence type: {evidence_type_hint}.
- Choose exactly one evidence_type: cloudtrail, iam_activity, guardduty, cloudwatch, access_key, billing, or unknown. Never return the list of choices.
- Extract facts only. Do not judge security relevance, action support, or whether activity is unauthorized. A visible claim may be quoted as a label, not established as fact.
- Report unreadable or cropped fields in extraction_warnings; use an empty list when none are observed.
- Do not invent values that are not visible in the screenshot.
- Do not use hidden truth.
- Do not mention that you are an AI model.
- Do not return markdown.
- Return JSON only.
- First transcribe the concrete visible fields and event rows; do not substitute the query heading or panel title for the rows below it.
- For access-key pages, copy the exact key ID, owner, status, last-used time/service, source IP, and rotation status if visible. Do not infer misuse from an inactive key.
- For CloudWatch tables, extract visible timestamps, principals, and actions from the result rows, not just the query status or record count.
- If the screenshot shows an IAM timeline, extract the visible time, principal, action, and source IP rows.
- If the screenshot shows CloudTrail, extract event name, user identity, source IP, MFA, event time, region, and risk signal if visible.
- If the screenshot shows GuardDuty, extract finding type, severity, principal, remote IP, first seen, last seen, and summary if visible.
"""

    image_content = [{"type": "image", "image": str(image_path)}]
    if evidence_type_hint == "cloudwatch":
        image_content, prompt = _cloudwatch_images(image_path)

    if retry_extraction:
        prompt += "\nThe previous extraction was incomplete or malformed. Re-read every visible row carefully. Return the requested JSON schema, not headings alone. Use null for unreadable fields and preserve clipped text; do not fill gaps by guessing.\n"

    messages = [
        {
            "role": "user",
            "content": [
                *image_content,
                {
                    "type": "text",
                    "text": prompt,
                },
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

    inputs = inputs.to(device)

    max_new_tokens = int(os.getenv("HF_IMAGE_MAX_NEW_TOKENS", "650"))
    if evidence_type_hint == "cloudwatch":
        max_new_tokens = max(max_new_tokens, 1400)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            **_memory_bounded_generation(device),
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    output_text = processor.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    result = _parse_vlm_output(
        output_text=output_text,
        selected_evidence=selected_evidence,
    )
    if evidence_type_hint == "cloudwatch" and "event_rows" not in result:
        raise ValueError("CloudWatch extraction omitted the structured event_rows field.")
    return result


# The renderer draws CloudWatch text at 16-18px. At the old ~1.1x effective
# zoom one 14px vision patch covered "rn", so "arn:" was read as "am:".
# The enlarged crop must stay legible, but the language model's attention
# memory grows with the square of its image tokens, so only the results card
# is enlarged and the pixel cap bounds tall tables.
_CLOUDWATCH_CARD_SCALE = float(os.getenv("HF_CLOUDWATCH_CARD_SCALE", "1.75"))
_CLOUDWATCH_CARD_MAX_PIXELS = int(os.getenv("HF_CLOUDWATCH_CARD_MAX_PIXELS", "2200000"))
# Legacy layouts also send the full screenshot, which carries the query editor.
_CLOUDWATCH_IMAGE_MAX_PIXELS = int(os.getenv("HF_CLOUDWATCH_IMAGE_MAX_PIXELS", "700000"))
# The renderer draws the matched-records line at y=226 and the query editor
# from y=260. Images saved before 24 Sep 2026 record only the metadata/table
# box, so their crop is extended up to this line.
_CLOUDWATCH_CARD_TOP = 216
_VISION_PATCH = 28


def _cloudwatch_images(image_path: Path) -> tuple[list[dict[str, Any]], str]:
    """Return enlarged CloudWatch image inputs and the prompt that describes them.

    Uses image pixels only. New renderings carry bounds-only PNG layout
    metadata, and one enlarged crop covers the matched-records line, query
    editor, metadata strip and results table. Legacy 1200x760 images use the
    old template's coordinates, and unknown layouts keep the whole image; both
    also send the original screenshot, which contains the query editor.
    """
    with Image.open(image_path) as source:
        layout = source.info.get("cloudir_cloudwatch_detail_box")
        view = source.convert("RGB")
    box = None
    if layout:
        try:
            candidate = json.loads(layout)
            if (isinstance(candidate, list) and len(candidate) == 4
                    and all(type(v) is int for v in candidate)
                    and 0 <= candidate[0] < candidate[2] <= view.width
                    and 0 <= candidate[1] < candidate[3] <= view.height):
                box = tuple(candidate)
        except (TypeError, ValueError):
            pass
    if box:
        card = view.crop((box[0], min(box[1], _CLOUDWATCH_CARD_TOP), box[2], box[3]))
        return [_enlarged_image(card)], _cloudwatch_extraction_prompt(
            "The image is an enlarged crop of the Logs Insights results card: the matched-records\n"
            "line, the query editor, the metadata strip and the results table.")
    if view.size == (1200, 760):
        # Include the metadata strip so the time range is magnified as well.
        detail = view.crop((78, 406, 1130, 676))
    else:
        detail = view.copy()
        detail.thumbnail((1200, 1200))
    return [
        {"type": "image", "image": str(image_path), "max_pixels": _CLOUDWATCH_IMAGE_MAX_PIXELS},
        _enlarged_image(detail),
    ], _cloudwatch_extraction_prompt(
        "Image 1 is the full screenshot. Image 2 magnifies the metadata strip and table\n"
        "of the SAME screenshot, not additional records.")


def _enlarged_image(view: Image.Image) -> dict[str, Any]:
    scale = min(_CLOUDWATCH_CARD_SCALE,
                (_CLOUDWATCH_CARD_MAX_PIXELS / (view.width * view.height)) ** 0.5)
    # Size to whole vision patches so the processor does not resample again.
    width = max(_VISION_PATCH, int(view.width * scale) // _VISION_PATCH * _VISION_PATCH)
    height = max(_VISION_PATCH, int(view.height * scale) // _VISION_PATCH * _VISION_PATCH)
    return {"type": "image", "image": view.resize((width, height), Image.Resampling.LANCZOS),
            "resized_width": width, "resized_height": height}


def _cloudwatch_extraction_prompt(image_description: str) -> str:
    return f'''Transcribe the CloudWatch screenshot. {image_description}
Read each visible results-table row from top to bottom, once only.
Return JSON only in this order:
{{
  "evidence_type": "cloudwatch",
  "event_rows": [
    {{"timestamp": null, "log_stream": null, "message": null, "truncated": false}}
  ],
  "query_text": null,
  "time_range": null,
  "log_group": null,
  "matched_records": null,
  "extraction_warnings": []
}}
Replace the example row with every visible row, or [] if none are visible.
Copy timestamp, log_stream and message exactly as displayed, including punctuation.
A message may wrap onto several lines inside its cell; it is still one message.
If a message ends in ... or …, preserve that ending and set truncated=true.
If the message is complete and has no ellipsis, use truncated=false. Decide for
each row individually; a neighbouring truncated row does not make this row truncated.
NEVER complete missing characters or reconstruct hidden text from another row.
Use null for unreadable fields and name them in extraction_warnings.
matched_records is the integer printed on the badge. It is NOT the number of rows
visible in the table. Do not invent rows to match that badge.
query_text is the text INSIDE the query editor. It is a query/description, not an event.
time_range is the exact displayed range as one string: preserve both endpoints,
spaces, separators and timezone. Do not turn a range into a single timestamp.
Ignore buttons, headings, and the correlation note. Do not judge authorization,
security relevance or action support. Extract only what can be read from pixels.
'''


@isolated_model_call
def analyse_architecture_diagram(image_path: str | Path) -> dict[str, Any]:
    """
    Uses the local Qwen2.5-VL model to inspect an ACSE architecture diagram.

    This function is used during dataset preparation, before the normal CloudIR
    runtime starts.

    The architecture image is resized before VLM inference to avoid large Apple
    MPS memory allocations from high-resolution architecture diagrams.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Architecture diagram not found: {image_path}")

    with Image.open(image_path) as img:
        img.verify()

    vlm_image_path = _prepare_architecture_image_for_vlm(image_path)

    model, processor = _load_image_model()
    device = _device()

    prompt = """
You are the vision-language model inside CloudIR Trainer.

You are inspecting an ACSE-Eval cloud architecture diagram before an incident-response scenario is generated.

Your task is to read the architecture diagram and extract only visible architecture facts.

Return ONLY valid JSON with exactly these keys:
{
  "cloud_provider": "AWS | Azure | GCP | multi-cloud | unknown",
  "visible_services": [
    "service or component visibly shown in the diagram"
  ],
  "identity_components": [
    "IAM, users, roles, policies, identity providers, authentication or authorisation components visibly shown"
  ],
  "compute_components": [
    "compute services, workloads, functions, instances, containers or application components visibly shown"
  ],
  "storage_components": [
    "storage services, databases, buckets or data stores visibly shown"
  ],
  "network_components": [
    "VPCs, subnets, internet gateways, NAT gateways, load balancers, firewalls or network paths visibly shown"
  ],
  "logging_components": [
    "CloudTrail, CloudWatch, audit logs, monitoring, SIEM or logging components visibly shown"
  ],
  "security_components": [
    "security controls, detection services, guardrails, policies or monitoring components visibly shown"
  ],
  "external_entities": [
    "external users, attackers, third-party services, APIs or internet-facing entities visibly shown"
  ],
  "trust_boundaries": [
    "trust boundary, account boundary, network boundary or external/internal separation visibly shown"
  ],
  "data_flows": [
    "visible data flow, request path or dependency shown in the diagram"
  ],
  "identity_flows": [
    "visible authentication, authorisation, role assumption or permission flow shown in the diagram"
  ],
  "security_observations": [
    "security-relevant observation based only on what is visible in the diagram"
  ],
  "possible_investigation_evidence": [
    "cloud evidence source that appears relevant based on visible architecture, such as CloudTrail, IAM activity, access keys, GuardDuty, CloudWatch, billing or logs"
  ],
  "architecture_summary": "short factual summary of the visible architecture"
}

Rules:
- Use only what is visible in the architecture diagram.
- Do not invent services, risks, attackers or incident details that are not visible.
- Do not assume hidden truth.
- Do not mention that you are an AI model.
- Do not return markdown.
- Return JSON only.
- If a field has no visible information, return an empty list for that field.
- Keep values concise and useful for a cloud incident-response simulator.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(vlm_image_path),
                },
                {
                    "type": "text",
                    "text": prompt,
                },
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

    inputs = inputs.to(device)

    max_new_tokens = int(os.getenv("HF_IMAGE_MAX_NEW_TOKENS", "900"))

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            **_memory_bounded_generation(device),
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    output_text = processor.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return _parse_architecture_output(output_text)


def _prepare_architecture_image_for_vlm(image_path: Path) -> Path:
    """
    Creates a resized RGB copy of the ACSE architecture diagram for VLM parsing.

    This prevents Apple MPS from trying to allocate extremely large buffers when
    the source architecture.png is high resolution.
    """

    max_side = int(os.getenv("ACSE_ARCH_VLM_MAX_SIDE", "1280"))

    try:
        dataset_root = image_path.parents[2]
    except IndexError:
        dataset_root = Path.cwd() / "data"

    prepared_dir = dataset_root / "processed" / "_vlm_inputs"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    prepared_path = prepared_dir / f"{image_path.stem}_vlm_input.png"

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        img.save(prepared_path, format="PNG", optimize=True)

    return prepared_path


def _parse_architecture_output(output_text: str) -> dict[str, Any]:
    try:
        parsed = _extract_json_from_output(output_text)
    except ValueError:
        return {
            "cloud_provider": "unknown",
            "visible_services": [],
            "identity_components": [],
            "compute_components": [],
            "storage_components": [],
            "network_components": [],
            "logging_components": [],
            "security_components": [],
            "external_entities": [],
            "trust_boundaries": [],
            "data_flows": [],
            "identity_flows": [],
            "security_observations": [
                "The architecture diagram could not be parsed as strict JSON."
            ],
            "possible_investigation_evidence": [],
            "architecture_summary": output_text.strip()
            or "The VLM returned an empty architecture analysis response.",
        }

    return _normalise_architecture_output(parsed)


def _normalise_architecture_output(parsed: dict[str, Any]) -> dict[str, Any]:
    cloud_provider = str(parsed.get("cloud_provider") or "unknown").strip()

    allowed_cloud_providers = {
        "aws",
        "azure",
        "gcp",
        "multi-cloud",
        "unknown",
    }

    if cloud_provider.lower() not in allowed_cloud_providers:
        cloud_provider = "unknown"

    architecture_summary = str(
        parsed.get("architecture_summary")
        or "The architecture diagram contains visible cloud components relevant to incident response."
    ).strip()

    return {
        "cloud_provider": cloud_provider,
        "visible_services": _ensure_list_of_strings(parsed.get("visible_services")),
        "identity_components": _ensure_list_of_strings(parsed.get("identity_components")),
        "compute_components": _ensure_list_of_strings(parsed.get("compute_components")),
        "storage_components": _ensure_list_of_strings(parsed.get("storage_components")),
        "network_components": _ensure_list_of_strings(parsed.get("network_components")),
        "logging_components": _ensure_list_of_strings(parsed.get("logging_components")),
        "security_components": _ensure_list_of_strings(parsed.get("security_components")),
        "external_entities": _ensure_list_of_strings(parsed.get("external_entities")),
        "trust_boundaries": _ensure_list_of_strings(parsed.get("trust_boundaries")),
        "data_flows": _ensure_list_of_strings(parsed.get("data_flows")),
        "identity_flows": _ensure_list_of_strings(parsed.get("identity_flows")),
        "security_observations": _ensure_list_of_strings(parsed.get("security_observations")),
        "possible_investigation_evidence": _ensure_list_of_strings(
            parsed.get("possible_investigation_evidence")
        ),
        "architecture_summary": architecture_summary,
    }


def _parse_vlm_output(
    output_text: str,
    selected_evidence: dict[str, Any],
) -> dict[str, Any]:
    try:
        parsed = _extract_json_from_output(output_text)

    except ValueError:
        return _fallback_vlm_output(
            raw_output=output_text,
            selected_evidence=selected_evidence,
        )

    return _normalise_vlm_output(
        parsed=parsed,
        selected_evidence=selected_evidence,
    )


def _normalise_vlm_output(
    parsed: dict[str, Any],
    selected_evidence: dict[str, Any],
) -> dict[str, Any]:
    if "event_rows" in parsed:
        return _normalise_cloudwatch_extraction(parsed)
    evidence_type = (
        parsed.get("evidence_type")
        or selected_evidence.get("template")
        or selected_evidence.get("type")
        or "unknown"
    )

    evidence_type = str(evidence_type).strip().lower()

    allowed_types = {"cloudtrail", "iam_activity", "guardduty", "cloudwatch", "access_key", "billing", "unknown"}
    if evidence_type not in allowed_types:
        evidence_type = "unknown"

    visible_summary = str(
        parsed.get("visible_evidence_summary")
        or parsed.get("summary")
        or "The screenshot contains visible cloud investigation evidence."
    ).strip()

    visible_facts = parsed.get("visible_facts_extracted")

    if visible_facts is None:
        visible_facts = parsed.get("important_visible_fields", [])

    visible_facts = _ensure_list_of_strings(visible_facts)

    important_fields = parsed.get("important_visible_fields")

    if important_fields is None:
        important_fields = visible_facts

    important_fields = _ensure_list_of_strings(important_fields)

    return {
        # UI-facing keys
        "evidence_type": evidence_type,
        "visible_facts_extracted": visible_facts,
        "extraction_warnings": _ensure_list_of_strings(parsed.get("extraction_warnings", [])),

        # Existing/internal keys
        "visible_evidence_summary": visible_summary,
        "important_visible_fields": important_fields,
    }


def _normalise_cloudwatch_extraction(parsed: dict[str, Any]) -> dict[str, Any]:
    """Keep editor text separate; derive compatibility facts from rows, not prose."""
    rows = parsed["event_rows"]
    if not isinstance(rows, list):
        raise ValueError("CloudWatch event_rows must be a list.")
    warnings = _ensure_list_of_strings(parsed.get("extraction_warnings", []))
    normalised = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError("Each CloudWatch event row must be an object.")
        clean = {}
        for key in ("timestamp", "log_stream", "message"):
            value = row.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"CloudWatch row {index}: {key} must be text or null.")
            # Line breaks come from cell wrapping in the screenshot, not the log text.
            clean[key] = " ".join(value.split()) or None if isinstance(value, str) else None
            if clean[key] is None:
                warnings.append(f"Row {index}: {key} was not read.")
        truncated = row.get("truncated")
        if not isinstance(truncated, bool):
            raise ValueError(f"CloudWatch row {index}: truncated must be true or false.")
        clean["truncated"] = truncated or (clean["message"] or "").endswith(("...", "…"))
        if clean["truncated"]:
            warnings.append(f"Row {index}: message is truncated; hidden text was not extracted.")
        normalised.append(clean)
    metadata = {}
    for key in ("query_text", "time_range", "log_group"):
        value = parsed.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"CloudWatch {key} must be text or null.")
        metadata[key] = " ".join(value.split()) or None if isinstance(value, str) else None
    matched = parsed.get("matched_records")
    if matched is not None and (type(matched) is not int or matched < 0):
        raise ValueError("CloudWatch matched_records must be a non-negative integer or null.")
    facts = [f"Event row {i}: " + json.dumps(row, ensure_ascii=False)
             for i, row in enumerate(normalised, 1)]
    fields = [f"{key}: {value}" for key, value in metadata.items()
              if key != "query_text" and value is not None]
    if matched is not None:
        fields.append(f"Matched records badge: {matched} (not the displayed row count)")
    return {
        "evidence_type": "cloudwatch", "event_rows": normalised,
        **metadata, "matched_records": matched,
        "visible_facts_extracted": facts + fields,
        "important_visible_fields": fields,
        "visible_evidence_summary": f"Extracted {len(normalised)} visible CloudWatch table rows.",
        "extraction_warnings": list(dict.fromkeys(warnings)),
    }


def _fallback_vlm_output(
    raw_output: str,
    selected_evidence: dict[str, Any],
) -> dict[str, Any]:
    raise ValueError("VLM extraction failed: response was not valid JSON. Retry the evidence evaluation.")


def _ensure_list_of_strings(value: Any) -> list[str]:
    if value in [None, "", [], {}]:
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    return [str(value).strip()]


def _extract_json_from_output(output_text: str) -> dict[str, Any]:
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
            "VLM did not return valid JSON. Raw output was:\n\n"
            f"{output_text}"
        ) from exc
