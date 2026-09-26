# CloudIR VLM, Coach, and STT Evaluation

Generated: `2026-08-16T07:34:11`
Batch: `multitask-production-baseline`
Mode: `model inference`

## Selected Models

### VLM
| name | model_id | family |
| --- | --- | --- |
| qwen2.5-vl-3b-instruct | Qwen/Qwen2.5-VL-3B-Instruct | vision-language instruction model |

### COACH
| name | model_id | family |
| --- | --- | --- |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | instruction-tuned causal LM |

### STT
| name | model_id | family |
| --- | --- | --- |
| whisper-base | openai/whisper-base | encoder-decoder speech recognition model |

## Evaluation Case Counts

| coach | stt | vlm |
| --- | --- | --- |
| 2 | 1 | 1 |

## VLM Summary

| model_name | cases | support_role_accuracy | avg_keyword_recall | avg_latency_seconds | errors |
| --- | --- | --- | --- | --- | --- |
| qwen2.5-vl-3b-instruct | 1 | 0.0 | 0.625 | 54.658 | 0 |

## Coach Summary

| model_name | cases | avg_score_out_of_5 | avg_latency_seconds | avg_word_count | errors |
| --- | --- | --- | --- | --- | --- |
| qwen2.5-1.5b-instruct | 2 | 4.0 | 6.466 | 62.5 | 0 |

## STT Summary

| model_name | model_id | samples | meaning_preservation_rate | avg_keyword_recall | avg_latency_seconds | errors |
| --- | --- | --- | --- | --- | --- | --- |
| whisper-base | openai/whisper-base | 2 | 0.0 | 0.2 | 0.734 | 0 |

## Candidate Matrix

| selected_for_run | stage | used_in_application | name | model_id | family | benchmark_mode | metric | cached | cache_size_gb | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | vlm | True | qwen2.5-vl-3b-instruct | Qwen/Qwen2.5-VL-3B-Instruct | vision-language instruction model | executed_when_included | field extraction and evidence-type alignment | True | 15.042 | Current application VLM. Strong screenshot/layout reader for evidence screenshots. |
| False | vlm | False | qwen2-vl-2b-instruct | Qwen/Qwen2-VL-2B-Instruct | smaller vision-language instruction model | executed_when_included | field extraction and latency | False | 0.0 | Smaller Qwen VLM baseline for comparing speed and evidence extraction. |
| False | vlm | False | llava-onevision-0.5b | llava-hf/llava-onevision-qwen2-0.5b-ov-hf | small open VLM baseline | executed_when_included | field extraction and hallucination rate | False | 0.0 | Lightweight multimodal baseline for screenshot interpretation. |
| False | vlm | False | florence-2-base-ft | microsoft/Florence-2-base-ft | compact OCR/captioning vision-language model | executed_when_included | OCR keyword recall and latency | False | 0.0 | Small Florence-2 candidate for screenshot text extraction; useful CPU-friendly baseline. |
| True | coach | True | qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | instruction-tuned causal LM | executed | 5-point feedback rubric and latency | True | 6.198 |  |
| False | coach | False | qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct | small instruction-tuned causal LM | executed | 5-point feedback rubric and latency | False | 0.0 |  |
| False | coach | False | flan-t5-small | google/flan-t5-small | small sequence-to-sequence instruction model | executed | 5-point feedback rubric and latency | False | 0.0 |  |
| False | coach | False | flan-t5-base | google/flan-t5-base | sequence-to-sequence instruction model | executed | 5-point feedback rubric and latency | False | 0.0 |  |
| False | stt | False | whisper-tiny-en | openai/whisper-tiny.en | small English speech recognition model | executed_when_recordings_exist | keyword recall, meaning preservation, latency | False | 0.0 | Fast English-only baseline for learner voice recordings. |
| True | stt | True | whisper-base | openai/whisper-base | encoder-decoder speech recognition model | executed_when_recordings_exist | keyword recall and meaning preservation | True | 0.59 | Configured locally. The benchmark auto-discovers recorded learner audio samples. |
| False | stt | False | whisper-base-en | openai/whisper-base.en | English speech recognition model | executed_when_recordings_exist | keyword recall, meaning preservation, latency | False | 0.0 | English-only base model for comparing against multilingual Whisper Base. |
| False | stt | False | whisper-small | openai/whisper-small | larger speech recognition model | executed_when_recordings_exist | keyword recall, meaning preservation, latency | False | 0.0 | Larger Whisper baseline to test whether quality improves enough to justify slower inference. |
