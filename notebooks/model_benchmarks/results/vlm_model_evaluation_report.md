# CloudIR VLM Model Evaluation

Generated: `2026-08-23T16:38:47`
Batch: `vlm-baseline`
Task: `vlm`
Mode: `metadata only`
Cases: `15`

## Selected Models

| name | model_id | family |
| --- | --- | --- |
| qwen2.5-vl-3b-instruct | Qwen/Qwen2.5-VL-3B-Instruct | vision-language instruction model |
| qwen2-vl-2b-instruct | Qwen/Qwen2-VL-2B-Instruct | smaller vision-language instruction model |
| llava-onevision-0.5b | llava-hf/llava-onevision-qwen2-0.5b-ov-hf | small open VLM baseline |
| florence-2-base-ft | microsoft/Florence-2-base-ft | compact OCR/captioning vision-language model |

## Summary

No inference summary.

## Evaluation Cases

| task | case_id | source | expected | expected_template | expected_keywords |
| --- | --- | --- | --- | --- | --- |
| vlm | turn_1_cloudwatch_insights_query | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_1/cloudwatch_insights_query.png | partial | cloudwatch | cloudwatch, log, insight, query, matches, multiple, entries, related |
| vlm | turn_1_access_key_info | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_1/access_key_info.png | weak | access_key | access, key, information, associated, iam, user, marked, inactive |
| vlm | turn_1_cloudtrail_log_row | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_1/cloudtrail_log_row.png | strong | cloudtrail | unauthorized, access, attempt, cloudtrail, log, entry, stop, logging |
| vlm | turn_2_access_key_review | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_2/access_key_review.png | partial | access_key | access, key, review, active, associated, the, suspicious, principal |
| vlm | turn_2_iam_scope_timeline | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_2/iam_scope_timeline.png | strong | iam_activity | iam, scope, timeline, activity, links, login, logging, and |
| vlm | turn_2_cloudwatch_correlation_logs | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_2/cloudwatch_correlation_logs.png | weak | cloudwatch | cloudwatch, correlation, logs, correlate, authentication, cloudtrail, and, credential-monitoring |
| vlm | turn_3_billing_impact_snapshot | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_3/billing_impact_snapshot.png | weak | billing | billing, impact, snapshot, modest, spend, increase, should, checked |
| vlm | turn_3_cloudtrail_logging_recovery_check | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_3/cloudtrail_logging_recovery_check.png | partial | cloudtrail | cloudtrail, scope, check, the, stoplogging, event, must, addressed |
| vlm | turn_3_guardduty_scope_finding | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_3/guardduty_scope_finding.png | strong | guardduty | guardduty, scope, finding, links, the, suspicious, principal, remote |
| vlm | turn_4_containment_billing_context | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_4/containment_billing_context.png | weak | billing | containment, billing, context, modest, spend, increase, during, the |
| vlm | turn_4_containment_monitoring_check | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_4/containment_monitoring_check.png | partial | cloudwatch | containment, monitoring, check, cloudwatch, logs, show, identity-monitoring, alerts |
| vlm | turn_4_access_key_containment_review | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_4/access_key_containment_review.png | strong | access_key | access, key, containment, review, the, affected, principal, has |
| vlm | turn_5_final_identity_timeline | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_5/final_identity_timeline.png | partial | iam_activity | final, identity, timeline, iam, activity, summarises, suspicious, login |
| vlm | turn_5_final_billing_snapshot | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_5/final_billing_snapshot.png | weak | billing | final, billing, snapshot, residual, cost, impact, after, the |
| vlm | turn_5_final_cloudtrail_recovery_record | /Users/karunanidhiprabhakar/Desktop/FYP/output/generated_evidence/turn_5/final_cloudtrail_recovery_record.png | strong | cloudtrail | final, cloudtrail, recovery, record, confirms, logging, and, credential-control |

## Candidate Matrix

| selected_for_run | stage | used_in_application | name | model_id | family | benchmark_mode | metric | cached | cache_size_gb | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | vlm | True | qwen2.5-vl-3b-instruct | Qwen/Qwen2.5-VL-3B-Instruct | vision-language instruction model | executed_when_included | field extraction and evidence-type alignment | True | 15.042 | Current application VLM. Strong screenshot/layout reader for evidence screenshots. |
| True | vlm | False | qwen2-vl-2b-instruct | Qwen/Qwen2-VL-2B-Instruct | smaller vision-language instruction model | executed_when_included | field extraction and latency | True | 8.859 | Smaller Qwen VLM baseline for comparing speed and evidence extraction. |
| True | vlm | False | llava-onevision-0.5b | llava-hf/llava-onevision-qwen2-0.5b-ov-hf | small open VLM baseline | executed_when_included | field extraction and hallucination rate | True | 3.598 | Lightweight multimodal baseline for screenshot interpretation. |
| True | vlm | False | florence-2-base-ft | microsoft/Florence-2-base-ft | compact OCR/captioning vision-language model | executed_when_included | OCR keyword recall and latency | True | 0.932 | Small Florence-2 candidate for screenshot text extraction; useful CPU-friendly baseline. |
