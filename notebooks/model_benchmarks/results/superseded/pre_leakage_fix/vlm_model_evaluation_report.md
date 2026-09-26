# CloudIR VLM Model Evaluation

Generated: `2026-08-16T23:17:07`
Batch: `vlm-candidate-comparison-cpu-tokenfix-v3-florencefix`
Task: `vlm`
Mode: `model inference`
Cases: `5`

## Selected Models

| name | model_id | family |
| --- | --- | --- |
| qwen2.5-vl-3b-instruct | Qwen/Qwen2.5-VL-3B-Instruct | vision-language instruction model |
| qwen2-vl-2b-instruct | Qwen/Qwen2-VL-2B-Instruct | smaller vision-language instruction model |
| llava-onevision-0.5b | llava-hf/llava-onevision-qwen2-0.5b-ov-hf | small open VLM baseline |
| florence-2-base-ft | microsoft/Florence-2-base-ft | compact OCR/captioning vision-language model |

## Summary

| model_name | cases | support_role_accuracy | avg_keyword_recall | avg_latency_seconds | errors |
| --- | --- | --- | --- | --- | --- |
| florence-2-base-ft | 5 | 0.0 | 0.575 | 2.879 | 0 |
| llava-onevision-0.5b | 5 | 0.0 | 0.05 | 111.741 | 0 |
| qwen2-vl-2b-instruct | 5 | 0.4 | 0.275 | 96.589 | 0 |
| qwen2.5-vl-3b-instruct | 5 | 0.4 | 0.4 | 127.972 | 0 |

## Evaluation Cases

| task | case_id | source | expected | expected_template | expected_keywords |
| --- | --- | --- | --- | --- | --- |
| vlm | turn_1_cloudwatch_insights_query | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_1/cloudwatch_insights_query.png | strong | cloudwatch | cloudwatch, log, insight, query, execute, logs, insights, analyze |
| vlm | turn_1_access_key_status | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_1/access_key_status.png | partial | access_key | access, key, status, check, review, the, keys, associated |
| vlm | turn_1_cloudtrail_event | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_1/cloudtrail_event.png | weak | cloudtrail | cloudtrail, event, review, logs, for, recent, events, involving |
| vlm | turn_2_access_key_status_check | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_2/access_key_status_check.png | partial | access_key | access, key, status, check, keys, involved, the, iam |
| vlm | turn_2_cloudtrail_event_1 | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_2/cloudtrail_event_1.png | strong | cloudtrail | iam, activity, cost, management, account, logs, show, multiple |

## Result Sample

| model_name | model_id | case_id | turn | expected_template | expected_support_role | predicted_support_role | support_role_correct | expected_keywords | matched_keywords | keyword_recall | latency_seconds | output_text | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-vl-3b-instruct | Qwen/Qwen2.5-VL-3B-Instruct | turn_1_cloudwatch_insights_query | 1 | cloudwatch | strong | strong | True | cloudwatch, log, insight, query, execute, logs, insights, analyze | cloudwatch, log | 0.25 | 134.696 | ```json {   "evidence_type": "cloudwatch",   "supports_selected_action": "strong",   "visible_facts": {     "timestamp": [       "2023-01-01T12:00:02Z",       "2023-01-01T11:59:52Z... |  |
| qwen2.5-vl-3b-instruct | Qwen/Qwen2.5-VL-3B-Instruct | turn_1_access_key_status | 1 | access_key | partial | strong | False | access, key, status, check, review, the, keys, associated | access, key, status | 0.375 | 122.52 | ```json {   "evidence_type": "access_key_status",   "supports_selected_action": "strong",   "visible_facts": {     "Status": "Active",     "Last used service": "iam.amazonaws.com",... |  |
| qwen2.5-vl-3b-instruct | Qwen/Qwen2.5-VL-3B-Instruct | turn_1_cloudtrail_event | 1 | cloudtrail | weak | strong | False | cloudtrail, event, review, logs, for, recent, events, involving | cloudtrail, event | 0.25 | 131.557 | ```json {   "evidence_type": "cloudtrail",   "supports_selected_action": "strong",   "visible_facts": {     "Event record": {       "Event history": [         {           "Event": ... |  |
| qwen2.5-vl-3b-instruct | Qwen/Qwen2.5-VL-3B-Instruct | turn_2_access_key_status_check | 2 | access_key | partial | strong | False | access, key, status, check, keys, involved, the, iam | access, key, status, check, keys, involved, the, iam | 1.0 | 120.684 | ```json {   "evidence_type": "access_key_status_check",   "supports_selected_action": "strong",   "visible_facts": [     "Access keys involved in the IAM activities are active and ... |  |
| qwen2.5-vl-3b-instruct | Qwen/Qwen2.5-VL-3B-Instruct | turn_2_cloudtrail_event_1 | 2 | cloudtrail | strong | strong | True | iam, activity, cost, management, account, logs, show, multiple | account | 0.125 | 130.404 | ```json {   "evidence_type": "cloudtrail",   "supports_selected_action": "strong",   "visible_facts": {     "Event ID": "8f42b4ac-9f1b-4d1a-a3c8-5df2c9a71c03",     "User identity":... |  |
| qwen2-vl-2b-instruct | Qwen/Qwen2-VL-2B-Instruct | turn_1_cloudwatch_insights_query | 1 | cloudwatch | strong | strong | True | cloudwatch, log, insight, query, execute, logs, insights, analyze | cloudwatch, log | 0.25 | 104.671 | {   "evidence_type": "cloudwatch",   "supports_selected_action": "strong",   "visible_facts": [     {       "timestamp": "2023-10-01T12:00Z",       "log_stream": "aws_cost_manageme... |  |
| qwen2-vl-2b-instruct | Qwen/Qwen2-VL-2B-Instruct | turn_1_access_key_status | 1 | access_key | partial | strong | False | access, key, status, check, review, the, keys, associated | access, key, status | 0.375 | 94.866 | {   "evidence_type": "access_key",   "supports_selected_action": "strong",   "visible_facts": [     "Access key ID",     "Status",     "Last used service",     "Region",     "Last ... |  |
| qwen2-vl-2b-instruct | Qwen/Qwen2-VL-2B-Instruct | turn_1_cloudtrail_event | 1 | cloudtrail | weak | strong | False | cloudtrail, event, review, logs, for, recent, events, involving | cloudtrail, event | 0.25 | 93.748 | {   "evidence_type": "cloudtrail",   "supports_selected_action": "strong",   "visible_facts": [     "CloudTrail Event history with selected API record details",     "Management eve... |  |
| qwen2-vl-2b-instruct | Qwen/Qwen2-VL-2B-Instruct | turn_2_access_key_status_check | 2 | access_key | partial | strong | False | access, key, status, check, keys, involved, the, iam | access, key, status | 0.375 | 94.465 | {   "evidence_type": "access_key",   "supports_selected_action": "strong",   "visible_facts": [     "Access key ID",     "Status",     "Last used service",     "Region",     "Last ... |  |
| qwen2-vl-2b-instruct | Qwen/Qwen2-VL-2B-Instruct | turn_2_cloudtrail_event_1 | 2 | cloudtrail | strong | strong | True | iam, activity, cost, management, account, logs, show, multiple | management | 0.125 | 95.195 | {   "evidence_type": "cloudtrail",   "supports_selected_action": "strong",   "visible_facts": [     "CloudTrail Event history with selected API record details",     "Management eve... |  |
| llava-onevision-0.5b | llava-hf/llava-onevision-qwen2-0.5b-ov-hf | turn_1_cloudwatch_insights_query | 1 | cloudwatch | strong | unparseable | False | cloudwatch, log, insight, query, execute, logs, insights, analyze |  | 0.0 | 112.038 | Only use what is visible in the screenshot. Do not invent fields. |  |
| llava-onevision-0.5b | llava-hf/llava-onevision-qwen2-0.5b-ov-hf | turn_1_access_key_status | 1 | access_key | partial | unparseable | False | access, key, status, check, review, the, keys, associated | the | 0.125 | 111.368 | Only use what is visible in the screenshot. Do not invent fields. |  |

## Candidate Matrix

| selected_for_run | stage | used_in_application | name | model_id | family | benchmark_mode | metric | cached | cache_size_gb | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | vlm | True | qwen2.5-vl-3b-instruct | Qwen/Qwen2.5-VL-3B-Instruct | vision-language instruction model | executed_when_included | field extraction and evidence-type alignment | True | 15.042 | Current application VLM. Strong screenshot/layout reader for evidence screenshots. |
| True | vlm | False | qwen2-vl-2b-instruct | Qwen/Qwen2-VL-2B-Instruct | smaller vision-language instruction model | executed_when_included | field extraction and latency | True | 8.859 | Smaller Qwen VLM baseline for comparing speed and evidence extraction. |
| True | vlm | False | llava-onevision-0.5b | llava-hf/llava-onevision-qwen2-0.5b-ov-hf | small open VLM baseline | executed_when_included | field extraction and hallucination rate | True | 3.598 | Lightweight multimodal baseline for screenshot interpretation. |
| True | vlm | False | florence-2-base-ft | microsoft/Florence-2-base-ft | compact OCR/captioning vision-language model | executed_when_included | OCR keyword recall and latency | True | 0.932 | Small Florence-2 candidate for screenshot text extraction; useful CPU-friendly baseline. |
