# CloudIR Coach Model Evaluation

Generated: `2026-08-16T11:37:31`
Batch: `coach-full-16case-run`
Task: `coach`
Mode: `model inference`
Cases: `16`

## Selected Models

| name | model_id | family |
| --- | --- | --- |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | instruction-tuned causal LM |
| qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct | small instruction-tuned causal LM |
| flan-t5-small | google/flan-t5-small | small sequence-to-sequence instruction model |
| flan-t5-base | google/flan-t5-base | sequence-to-sequence instruction model |

## Summary

| model_name | cases | avg_automated_rubric_score | automated_rubric_max | avg_latency_seconds | avg_word_count | errors | criterion_output_nonempty_pass_rate | criterion_concise_length_20_140_words_pass_rate | criterion_references_selected_evidence_or_action_pass_rate | criterion_uses_support_terminology_pass_rate | criterion_acknowledges_expected_verdict_pass_rate | criterion_proposes_next_investigation_step_pass_rate | criterion_avoids_hidden_truth_leak_pass_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flan-t5-base | 16 | 3.25 | 7 | 0.672 | 9.6 | 0 | 1.0 | 0.0 | 0.062 | 0.75 | 0.0 | 0.438 | 1.0 |
| flan-t5-small | 16 | 1.875 | 7 | 2.799 | 12.2 | 0 | 0.625 | 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| qwen2.5-0.5b-instruct | 16 | 5.75 | 7 | 1.863 | 40.3 | 0 | 1.0 | 1.0 | 0.188 | 0.938 | 0.688 | 0.938 | 1.0 |
| qwen2.5-1.5b-instruct | 16 | 5.25 | 7 | 3.451 | 57.9 | 0 | 1.0 | 1.0 | 0.125 | 1.0 | 0.188 | 0.938 | 1.0 |

## Evaluation Cases

| task | case_id | source | expected |
| --- | --- | --- | --- |
| coach | turn_1_cloudwatch_insights_query | CloudWatch Log Insight Query | Strong Support |
| coach | turn_1_access_key_status | Access Key Status Check | Partial Support |
| coach | turn_1_cloudtrail_event | CloudTrail Event Review | Weak Support |
| coach | turn_1_unsupported_unrelated | Unrelated Policy Note | Unsupported |
| coach | turn_2_access_key_status_check | Access Key Status Check | Partial Support |
| coach | turn_2_cloudtrail_event_1 | IAM Activity - Cost Management Account | Strong Support |
| coach | turn_2_cloudwatch_log_insight_query_result | CloudWatch Log Insight Query Result | Weak Support |
| coach | turn_2_unsupported_unrelated | Unrelated Policy Note | Unsupported |
| coach | turn_3_cloudtrail_logging_recovery_check | CloudTrail Scope Check | Partial Support |
| coach | turn_3_billing_impact_snapshot | Billing Impact Snapshot | Weak Support |
| coach | turn_3_guardduty_scope_finding | GuardDuty Scope Finding | Strong Support |
| coach | turn_3_unsupported_unrelated | Unrelated Policy Note | Unsupported |
| coach | turn_4_containment_monitoring_check | Containment Monitoring Check | Partial Support |
| coach | turn_4_containment_billing_context | Containment Billing Context | Weak Support |
| coach | turn_4_access_key_containment_review | Access Key Containment Review | Strong Support |
| coach | turn_4_unsupported_unrelated | Unrelated Policy Note | Unsupported |

## Result Sample

| model_name | model_id | case_id | turn | expected_verdict | automated_rubric_score | automated_rubric_max | latency_seconds | word_count | output_text | error | criterion_output_nonempty | criterion_concise_length_20_140_words | criterion_references_selected_evidence_or_action | criterion_uses_support_terminology | criterion_acknowledges_expected_verdict | criterion_proposes_next_investigation_step | criterion_avoids_hidden_truth_leak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | turn_1_cloudwatch_insights_query | 1 | Strong Support | 5 | 7 | 7.985 | 71 | The selected evidence shows that there were two log entries in the 'aws_cost_management' account on October 1, 2023, with one being an error (INFO) and the other an error (ERROR). ... |  | True | True | False | True | False | True | True |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | turn_1_access_key_status | 1 | Partial Support | 5 | 7 | 3.776 | 54 | The evidence shows that the access key `AKIA4Z7EXAMPLE92K` in the 'cost-management' account is still active, but it lacks details about its last usage (service, region, time), whic... |  | True | True | False | True | False | True | True |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | turn_1_cloudtrail_event | 1 | Weak Support | 5 | 7 | 4.037 | 65 | The selected evidence shows an unauthorized console login attempt from an unknown IP address, which could indicate a security breach. However, the lack of additional context (e.g.,... |  | True | True | False | True | False | True | True |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | turn_1_unsupported_unrelated | 1 | Unsupported | 5 | 7 | 2.387 | 29 | The selected evidence does not provide specific details about recent IAM activities, making it unsupported. To verify recent actions, ensure visibility of CloudTrail events, IAM pr... |  | True | True | False | True | True | False | True |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | turn_2_access_key_status_check | 2 | Partial Support | 5 | 7 | 2.527 | 44 | The evidence shows that access keys are active, but there's no information about their usage or any recent activity. To provide more support, review logs for recent IAM activities ... |  | True | True | False | True | False | True | True |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | turn_2_cloudtrail_event_1 | 2 | Strong Support | 5 | 7 | 5.441 | 73 | The selected evidence shows that an attempt was made to stop logging in the AWS console on October 1st, 2023, by an admin user from IP address 185.220.101.42. This indicates potent... |  | True | True | False | True | False | True | True |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | turn_2_cloudwatch_log_insight_query_result | 2 | Weak Support | 5 | 7 | 5.077 | 77 | The selected action of reviewing CloudTrail logs was weakly supported by the evidence provided, which shows only one log entry indicating an IAM user calling `ListTrails` from an I... |  | True | True | False | True | False | True | True |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | turn_2_unsupported_unrelated | 2 | Unsupported | 5 | 7 | 2.759 | 51 | The selected evidence does not support the security verdict as it lacks specific details about CloudTrail events, IAM principals, or source IPs that would indicate an issue. Next s... |  | True | True | False | True | False | True | True |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | turn_3_cloudtrail_logging_recovery_check | 3 | Partial Support | 5 | 7 | 1.991 | 43 | The evidence confirms the StopLogging event, which needs addressing before recovery can be trusted. However, it lacks details about the impact on services and any remediation steps... |  | True | True | False | True | False | True | True |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | turn_3_billing_impact_snapshot | 3 | Weak Support | 6 | 7 | 3.571 | 66 | The selected evidence indicates a potential issue with resource misuse, but it lacks specific details about which resources were used and their costs. To confirm detection scope ef... |  | True | True | True | True | False | True | True |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | turn_3_guardduty_scope_finding | 3 | Strong Support | 6 | 7 | 2.886 | 73 | The evidence supports strong support as it confirms GuardDuty's detection of unauthorized access by an IAM user with high severity from a specific IP address in a particular region... |  | True | True | False | True | True | True | True |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | turn_3_unsupported_unrelated | 3 | Unsupported | 6 | 7 | 2.124 | 52 | The selected evidence, an unrelated policy note, does not support the need to confirm detection scope as it lacks specific details about CloudTrail events, IAM principals, or sourc... |  | True | True | True | True | False | True | True |

## Candidate Matrix

| selected_for_run | stage | used_in_application | name | model_id | family | benchmark_mode | metric | cached | cache_size_gb | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | coach | True | qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | instruction-tuned causal LM | executed | 5-point feedback rubric and latency | True | 6.198 |  |
| True | coach | False | qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct | small instruction-tuned causal LM | executed | 5-point feedback rubric and latency | False | 0.0 |  |
| True | coach | False | flan-t5-small | google/flan-t5-small | small sequence-to-sequence instruction model | executed | 5-point feedback rubric and latency | False | 0.0 |  |
| True | coach | False | flan-t5-base | google/flan-t5-base | sequence-to-sequence instruction model | executed | 5-point feedback rubric and latency | False | 0.0 |  |
