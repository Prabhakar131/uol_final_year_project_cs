# CloudIR Security Model Evaluation

Generated: `2026-08-16T07:25:01`
Batch: `batch-01-security-small-baselines`
Cases: `2`
Mode: `model inference`
Selected cache before run: `6.198 GB`
Advisory cache budget: `25.0 GB`

## Selected Models

| Model | Model ID | Family |
|---|---|---|
| qwen2.5-1.5b-instruct | `Qwen/Qwen2.5-1.5B-Instruct` | general instruction-tuned causal LM |
| qwen2.5-0.5b-instruct | `Qwen/Qwen2.5-0.5B-Instruct` | small general instruction-tuned causal LM |
| flan-t5-small | `google/flan-t5-small` | small sequence-to-sequence instruction model |
| flan-t5-base | `google/flan-t5-base` | sequence-to-sequence instruction model |
| bart-large-mnli | `facebook/bart-large-mnli` | zero-shot NLI classifier |

## Skipped Models

| name | model_id | reason | benchmark_mode |
| --- | --- | --- | --- |
| foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | not selected for this batch | executed |
| foundation-sec-8b-base | fdtn-ai/Foundation-Sec-8B | not selected for this batch | candidate_not_downloaded |
| cyberpal2-20b | cyber-pal-security/CyberPal2.0-20B | not selected for this batch | candidate_not_downloaded |
| cybersecqwen-4b | athena129/CyberSecQwen-4B | not selected for this batch | candidate_not_downloaded |
| lily-cybersecurity-7b-v0.2 | segolilylabs/Lily-Cybersecurity-7B-v0.2 | not selected for this batch | candidate_not_downloaded |

## Cache Before Run

| name | model_id | cache_dir | cached | cache_size_bytes | cache_size_gb |
| --- | --- | --- | --- | --- | --- |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct | True | 6197911376 | 6.198 |
| qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct | False | 0 | 0.0 |
| flan-t5-small | google/flan-t5-small | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--google--flan-t5-small | False | 0 | 0.0 |
| flan-t5-base | google/flan-t5-base | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--google--flan-t5-base | False | 0 | 0.0 |
| bart-large-mnli | facebook/bart-large-mnli | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--facebook--bart-large-mnli | False | 0 | 0.0 |

## Cache After Run

| name | model_id | cache_dir | cached | cache_size_bytes | cache_size_gb |
| --- | --- | --- | --- | --- | --- |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct | True | 6197911376 | 6.198 |
| qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct | True | 1999172734 | 1.999 |
| flan-t5-small | google/flan-t5-small | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--google--flan-t5-small | True | 622178152 | 0.622 |
| flan-t5-base | google/flan-t5-base | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--google--flan-t5-base | True | 1987134180 | 1.987 |
| bart-large-mnli | facebook/bart-large-mnli | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--facebook--bart-large-mnli | True | 3264298700 | 3.264 |

## Summary

| model_name | cases | accuracy | avg_latency_seconds | total_latency_seconds | unparseable | errors |
| --- | --- | --- | --- | --- | --- | --- |
| bart-large-mnli | 2 | 1.0 | 2.527 | 5.055 | 0 | 0 |
| flan-t5-base | 2 | 1.0 | 0.125 | 0.25 | 0 | 0 |
| flan-t5-small | 2 | 0.5 | 0.081 | 0.163 | 0 | 0 |
| qwen2.5-0.5b-instruct | 2 | 1.0 | 0.208 | 0.417 | 0 | 0 |
| qwen2.5-1.5b-instruct | 2 | 1.0 | 0.587 | 1.173 | 0 | 0 |

## Candidate Matrix

| selected_for_run | used_in_application | name | model_id | family | benchmark_mode | notes |
| --- | --- | --- | --- | --- | --- | --- |
| False | True | foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | cybersecurity-focused instruction-tuned causal LM | executed | Current application security model. Cybersecurity-focused model for SOC, incident response, and security workflow reasoning. |
| True | False | qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | general instruction-tuned causal LM | executed | Previous application security model; retained as a general-purpose baseline. |
| True | False | qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct | small general instruction-tuned causal LM | executed |  |
| True | False | flan-t5-small | google/flan-t5-small | small sequence-to-sequence instruction model | executed |  |
| True | False | flan-t5-base | google/flan-t5-base | sequence-to-sequence instruction model | executed |  |
| True | False | bart-large-mnli | facebook/bart-large-mnli | zero-shot NLI classifier | executed |  |
| False | False | foundation-sec-8b-base | fdtn-ai/Foundation-Sec-8B | cybersecurity-focused base causal LM | candidate_not_downloaded | Cybersecurity base model from Cisco Foundation AI; less instruction-optimised than Foundation-Sec-8B-Instruct. |
| False | False | cyberpal2-20b | cyber-pal-security/CyberPal2.0-20B | cybersecurity-expert instruction-tuned causal LM | candidate_not_downloaded | SOC/IR, CTI, vulnerability, CWE/CVE, and MITRE ATT&CK focused model; likely too large for the local prototype. |
| False | False | cybersecqwen-4b | athena129/CyberSecQwen-4B | cybersecurity-specialised Qwen-derived instruction model | candidate_not_downloaded | Defensive cybersecurity model focused on CTI-Bench tasks such as CWE mapping and CTI multiple-choice reasoning. |
| False | False | lily-cybersecurity-7b-v0.2 | segolilylabs/Lily-Cybersecurity-7B-v0.2 | cybersecurity-specialised causal LM | candidate_not_downloaded | Cybersecurity text-generation model candidate; included for evaluation discussion only. |

## Evaluation Case Mix

| expected_support_role | cases |
| --- | --- |
| partial | 1 |
| strong | 1 |

## Result Detail Sample

| model_name | model_id | case_id | turn | expected_support_role | expected_verdict | predicted_verdict | correct | latency_seconds | output_text | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | turn_1_cloudwatch_insights_query | 1 | strong | Strong Support | Strong Support | True | 0.71 | Strong Support |  |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | turn_1_access_key_status | 1 | partial | Partial Support | Partial Support | True | 0.463 | Partial Support |  |
| qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct | turn_1_cloudwatch_insights_query | 1 | strong | Strong Support | Strong Support | True | 0.244 | Strong Support |  |
| qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct | turn_1_access_key_status | 1 | partial | Partial Support | Partial Support | True | 0.173 | Partial Support |  |
| flan-t5-small | google/flan-t5-small | turn_1_cloudwatch_insights_query | 1 | strong | Strong Support | Strong Support | True | 0.119 | Strong Support |  |
| flan-t5-small | google/flan-t5-small | turn_1_access_key_status | 1 | partial | Partial Support | Unsupported | False | 0.044 | Unsupported |  |
| flan-t5-base | google/flan-t5-base | turn_1_cloudwatch_insights_query | 1 | strong | Strong Support | Strong Support | True | 0.143 | Strong Support |  |
| flan-t5-base | google/flan-t5-base | turn_1_access_key_status | 1 | partial | Partial Support | Partial Support | True | 0.107 | Partial Support |  |
| bart-large-mnli | facebook/bart-large-mnli | turn_1_cloudwatch_insights_query | 1 | strong | Strong Support | Strong Support | True | 2.572 | Strong Support |  |
| bart-large-mnli | facebook/bart-large-mnli | turn_1_access_key_status | 1 | partial | Partial Support | Partial Support | True | 2.483 | Partial Support |  |
