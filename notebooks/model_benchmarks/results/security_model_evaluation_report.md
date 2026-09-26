# CloudIR Security Model Evaluation

Generated: `2026-08-16T13:32:05`
Batch: `security-full-20case-run-leakfree`
Cases: `20`
Mode: `model inference`
Selected cache before run: `46.234 GB`
Advisory cache budget: `25.0 GB`

## Selected Models

| Model | Model ID | Family |
|---|---|---|
| foundation-sec-8b-instruct | `fdtn-ai/Foundation-Sec-8B-Instruct` | cybersecurity-focused instruction-tuned causal LM |
| qwen2.5-1.5b-instruct | `Qwen/Qwen2.5-1.5B-Instruct` | general instruction-tuned causal LM |
| qwen2.5-0.5b-instruct | `Qwen/Qwen2.5-0.5B-Instruct` | small general instruction-tuned causal LM |
| flan-t5-small | `google/flan-t5-small` | small sequence-to-sequence instruction model |
| flan-t5-base | `google/flan-t5-base` | sequence-to-sequence instruction model |
| bart-large-mnli | `facebook/bart-large-mnli` | zero-shot NLI classifier |

## Skipped Models

| name | model_id | reason | benchmark_mode |
| --- | --- | --- | --- |
| foundation-sec-8b-base | fdtn-ai/Foundation-Sec-8B | not selected for this batch | candidate_not_downloaded |
| cyberpal2-20b | cyber-pal-security/CyberPal2.0-20B | not selected for this batch | candidate_not_downloaded |
| cybersecqwen-4b | athena129/CyberSecQwen-4B | not selected for this batch | candidate_not_downloaded |
| lily-cybersecurity-7b-v0.2 | segolilylabs/Lily-Cybersecurity-7B-v0.2 | not selected for this batch | candidate_not_downloaded |

## Cache Before Run

| name | model_id | cache_dir | cached | cache_size_bytes | cache_size_gb |
| --- | --- | --- | --- | --- | --- |
| foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--fdtn-ai--Foundation-Sec-8B-Instruct | True | 32162981114 | 32.163 |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct | True | 6197911376 | 6.198 |
| qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct | True | 1999172734 | 1.999 |
| flan-t5-small | google/flan-t5-small | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--google--flan-t5-small | True | 622178152 | 0.622 |
| flan-t5-base | google/flan-t5-base | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--google--flan-t5-base | True | 1987134180 | 1.987 |
| bart-large-mnli | facebook/bart-large-mnli | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--facebook--bart-large-mnli | True | 3264298700 | 3.264 |

## Cache After Run

| name | model_id | cache_dir | cached | cache_size_bytes | cache_size_gb |
| --- | --- | --- | --- | --- | --- |
| foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--fdtn-ai--Foundation-Sec-8B-Instruct | True | 32162981114 | 32.163 |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct | True | 6197911376 | 6.198 |
| qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct | True | 1999172734 | 1.999 |
| flan-t5-small | google/flan-t5-small | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--google--flan-t5-small | True | 622178152 | 0.622 |
| flan-t5-base | google/flan-t5-base | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--google--flan-t5-base | True | 1987134180 | 1.987 |
| bart-large-mnli | facebook/bart-large-mnli | /Users/karunanidhiprabhakar/.cache/huggingface/hub/models--facebook--bart-large-mnli | True | 3264298700 | 3.264 |

## Summary

| model_name | cases | accuracy | avg_latency_seconds | total_latency_seconds | unparseable | errors |
| --- | --- | --- | --- | --- | --- | --- |
| bart-large-mnli | 20 | 0.2 | 3.254 | 65.074 | 0 | 0 |
| flan-t5-base | 20 | 0.5 | 0.413 | 8.255 | 0 | 0 |
| flan-t5-small | 20 | 0.4 | 0.103 | 2.064 | 0 | 0 |
| foundation-sec-8b-instruct | 20 | 0.75 | 49.346 | 986.91 | 0 | 0 |
| qwen2.5-0.5b-instruct | 20 | 0.6 | 2.895 | 57.898 | 0 | 0 |
| qwen2.5-1.5b-instruct | 20 | 0.75 | 9.998 | 199.968 | 0 | 0 |

## Candidate Matrix

| selected_for_run | used_in_application | name | model_id | family | benchmark_mode | notes |
| --- | --- | --- | --- | --- | --- | --- |
| True | True | foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | cybersecurity-focused instruction-tuned causal LM | executed | Current application security model. Cybersecurity-focused model for SOC, incident response, and security workflow reasoning. |
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
| partial | 5 |
| strong | 5 |
| unsupported | 5 |
| weak | 5 |

## Result Detail Sample

| model_name | model_id | case_id | turn | expected_support_role | expected_verdict | predicted_verdict | correct | latency_seconds | output_text | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | turn_1_cloudwatch_insights_query | 1 | strong | Strong Support | Partial Support | False | 89.331 | Partial Support |  |
| foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | turn_1_access_key_status | 1 | partial | Partial Support | Partial Support | True | 64.984 | Partial Support |  |
| foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | turn_1_cloudtrail_event | 1 | weak | Weak Support | Partial Support | False | 50.683 | Partial Support |  |
| foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | turn_1_unsupported_unrelated | 1 | unsupported | Unsupported | Unsupported | True | 33.742 | Unsupported |  |
| foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | turn_2_access_key_status_check | 2 | partial | Partial Support | Weak Support | False | 49.014 | Weak Support |  |
| foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | turn_2_cloudtrail_event_1 | 2 | strong | Strong Support | Partial Support | False | 50.132 | Partial Support |  |
| foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | turn_2_cloudwatch_log_insight_query_result | 2 | weak | Weak Support | Partial Support | False | 61.794 | Partial Support |  |
| foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | turn_2_unsupported_unrelated | 2 | unsupported | Unsupported | Unsupported | True | 34.0 | Unsupported |  |
| foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | turn_3_cloudtrail_logging_recovery_check | 3 | partial | Partial Support | Partial Support | True | 52.147 | Partial Support |  |
| foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | turn_3_billing_impact_snapshot | 3 | weak | Weak Support | Weak Support | True | 45.658 | Weak Support |  |
| foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | turn_3_guardduty_scope_finding | 3 | strong | Strong Support | Strong Support | True | 54.481 | Strong Support |  |
| foundation-sec-8b-instruct | fdtn-ai/Foundation-Sec-8B-Instruct | turn_3_unsupported_unrelated | 3 | unsupported | Unsupported | Unsupported | True | 34.345 | Unsupported |  |
