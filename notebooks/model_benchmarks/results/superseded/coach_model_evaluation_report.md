# CloudIR Coach Model Evaluation

Generated: `2026-08-16T07:56:54`
Batch: `coach-candidate-comparison`
Task: `coach`
Mode: `model inference`
Cases: `2`

## Selected Models

| name | model_id | family |
| --- | --- | --- |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | instruction-tuned causal LM |
| qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct | small instruction-tuned causal LM |
| flan-t5-small | google/flan-t5-small | small sequence-to-sequence instruction model |
| flan-t5-base | google/flan-t5-base | sequence-to-sequence instruction model |

## Summary

| model_name | cases | avg_score_out_of_5 | avg_latency_seconds | avg_word_count | errors |
| --- | --- | --- | --- | --- | --- |
| flan-t5-base | 2 | 2.5 | 0.164 | 6.5 | 0 |
| flan-t5-small | 2 | 0.5 | 0.77 | 7.0 | 0 |
| qwen2.5-0.5b-instruct | 2 | 3.5 | 0.97 | 36.0 | 0 |
| qwen2.5-1.5b-instruct | 2 | 4.0 | 3.274 | 62.5 | 0 |

## Evaluation Cases

| task | case_id | source | expected |
| --- | --- | --- | --- |
| coach | turn_1_cloudwatch_insights_query | CloudWatch Log Insight Query | Strong Support |
| coach | turn_1_access_key_status | Access Key Status Check | Partial Support |

## Result Sample

| model_name | model_id | case_id | turn | expected_verdict | score_out_of_5 | latency_seconds | word_count | output_text | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | turn_1_cloudwatch_insights_query | 1 | Strong Support | 4 | 3.69 | 71 | The selected evidence shows that there were two log entries in the 'aws_cost_management' account on October 1, 2023, with one being an error (INFO) and the other an error (ERROR). ... |  |
| qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | turn_1_access_key_status | 1 | Partial Support | 4 | 2.858 | 54 | The evidence shows that the access key `AKIA4Z7EXAMPLE92K` in the 'cost-management' account is still active, but it lacks details about its last usage (service, region, time), whic... |  |
| qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct | turn_1_cloudwatch_insights_query | 1 | Strong Support | 4 | 1.231 | 39 | The evidence shows that there was no unusual activity related to the 'cost-management' account on October 1, 2023. The strong support indicates that this information is accurate. N... |  |
| qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct | turn_1_access_key_status | 1 | Partial Support | 3 | 0.708 | 33 | The partial support in the security verdict suggests that there might be issues with the access key, but further investigation into the source IP and last used service could provid... |  |
| flan-t5-small | google/flan-t5-small | turn_1_cloudwatch_insights_query | 1 | Strong Support | 0 | 1.183 | 0 |                                         |  |
| flan-t5-small | google/flan-t5-small | turn_1_access_key_status | 1 | Partial Support | 1 | 0.358 | 14 | Using the AKIA4Z7EXAMPLE92K is a good way to learn how to use the AKIA4Z7EXAMPLE92K. |  |
| flan-t5-base | google/flan-t5-base | turn_1_cloudwatch_insights_query | 1 | Strong Support | 3 | 0.225 | 7 | IAM trainees should review recent IAM activities. |  |
| flan-t5-base | google/flan-t5-base | turn_1_access_key_status | 1 | Partial Support | 2 | 0.103 | 6 | IAM activities are not sufficient evidence. |  |

## Candidate Matrix

| selected_for_run | stage | used_in_application | name | model_id | family | benchmark_mode | metric | cached | cache_size_gb | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | coach | True | qwen2.5-1.5b-instruct | Qwen/Qwen2.5-1.5B-Instruct | instruction-tuned causal LM | executed | 5-point feedback rubric and latency | True | 6.198 |  |
| True | coach | False | qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct | small instruction-tuned causal LM | executed | 5-point feedback rubric and latency | False | 0.0 |  |
| True | coach | False | flan-t5-small | google/flan-t5-small | small sequence-to-sequence instruction model | executed | 5-point feedback rubric and latency | False | 0.0 |  |
| True | coach | False | flan-t5-base | google/flan-t5-base | sequence-to-sequence instruction model | executed | 5-point feedback rubric and latency | False | 0.0 |  |
