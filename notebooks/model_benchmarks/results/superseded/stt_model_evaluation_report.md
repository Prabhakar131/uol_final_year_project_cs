# CloudIR STT Model Evaluation

Generated: `2026-08-16T07:54:39`
Batch: `stt-candidate-comparison`
Task: `stt`
Mode: `model inference`
Cases: `2`

## Selected Models

| name | model_id | family |
| --- | --- | --- |
| whisper-tiny-en | openai/whisper-tiny.en | small English speech recognition model |
| whisper-base | openai/whisper-base | encoder-decoder speech recognition model |
| whisper-base-en | openai/whisper-base.en | English speech recognition model |
| whisper-small | openai/whisper-small | larger speech recognition model |

## Summary

| model_name | model_id | samples | meaning_preservation_rate | avg_keyword_recall | avg_latency_seconds | errors |
| --- | --- | --- | --- | --- | --- | --- |
| whisper-base | openai/whisper-base | 2 | 0.0 | 0.2 | 0.686 | 0 |
| whisper-base-en | openai/whisper-base.en | 2 | 0.0 | 0.4 | 0.513 | 0 |
| whisper-small | openai/whisper-small | 2 | 0.0 | 0.4 | 1.32 | 0 |
| whisper-tiny-en | openai/whisper-tiny.en | 2 | 0.0 | 0.4 | 0.271 | 0 |

## Evaluation Cases

| task | case_id | source | expected |
| --- | --- | --- | --- |
| stt | audio_sample_limit | notebooks/model_benchmarks/stt_audio_samples/audio_samples | 2 samples |

## Result Sample

| model_name | model_id | sample_id | expected_text | transcript | expected_keywords | matched_keywords | keyword_recall | meaning_preserved | latency_seconds | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| whisper-tiny-en | openai/whisper-tiny.en | partial_iam_activity | The IAM activity gives partial support because it shows suspicious API calls, but it does not prove the full attack path on its own. | The IAM activity timeline partly supports the action because it shows the principle and actions but I still need cloud trail details to confirm the exact source IP and event time. | iam, activity, partial, api calls, attack path | iam, activity | 0.4 | False | 0.293 |  |
| whisper-tiny-en | openai/whisper-tiny.en | partial_cloudwatch_correlation | The CloudWatch correlation gives partial support because it helps confirm the timeline, but it still needs CloudTrail or IAM evidence. | Cloudwatch correlation logs are useful contacts but they only partly support the containment decision unless I can connect the log entries to the IAM user and access key. | cloudwatch, correlation, partial, timeline, cloudtrail | cloudwatch, correlation | 0.4 | False | 0.249 |  |
| whisper-base | openai/whisper-base | partial_iam_activity | The IAM activity gives partial support because it shows suspicious API calls, but it does not prove the full attack path on its own. | Iaam, aktiviti, timeline, periksaan, kerana berada di tempat, persoan dan periksaan, tapi saya masih perlu mempunyai periksaan yang berada di tempat, untuk menggunakan IP dan event... | iam, activity, partial, api calls, attack path |  | 0.0 | False | 0.84 |  |
| whisper-base | openai/whisper-base | partial_cloudwatch_correlation | The CloudWatch correlation gives partial support because it helps confirm the timeline, but it still needs CloudTrail or IAM evidence. | CloudWatch Correlation logs are useful contacts, but they only partly support the containment decision unless I can connect the log entries to the IAM user and access key. | cloudwatch, correlation, partial, timeline, cloudtrail | cloudwatch, correlation | 0.4 | False | 0.533 |  |
| whisper-base-en | openai/whisper-base.en | partial_iam_activity | The IAM activity gives partial support because it shows suspicious API calls, but it does not prove the full attack path on its own. | The IAM activity timeline partly supports the action because it shows the principle and actions but I still need CloudTrail details to confirm the exact source IP and event time. | iam, activity, partial, api calls, attack path | iam, activity | 0.4 | False | 0.463 |  |
| whisper-base-en | openai/whisper-base.en | partial_cloudwatch_correlation | The CloudWatch correlation gives partial support because it helps confirm the timeline, but it still needs CloudTrail or IAM evidence. | CloudWatch correlation logs are useful contacts but they only partly support the containment the decision unless I can connect the log entries to the IAM user and access key. | cloudwatch, correlation, partial, timeline, cloudtrail | cloudwatch, correlation | 0.4 | False | 0.564 |  |
| whisper-small | openai/whisper-small | partial_iam_activity | The IAM activity gives partial support because it shows suspicious API calls, but it does not prove the full attack path on its own. | The IAM activity timeline partly supports the action because it shows the principal and actions but I still need cloud trail details to confirm the exact source IP and event time. | iam, activity, partial, api calls, attack path | iam, activity | 0.4 | False | 1.274 |  |
| whisper-small | openai/whisper-small | partial_cloudwatch_correlation | The CloudWatch correlation gives partial support because it helps confirm the timeline, but it still needs CloudTrail or IAM evidence. | CloudWatch Correlation logs are useful contacts but they only partly support the containment decision unless I can connect the log entries to the IAM user and access key. | cloudwatch, correlation, partial, timeline, cloudtrail | cloudwatch, correlation | 0.4 | False | 1.366 |  |

## Candidate Matrix

| selected_for_run | stage | used_in_application | name | model_id | family | benchmark_mode | metric | cached | cache_size_gb | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | stt | False | whisper-tiny-en | openai/whisper-tiny.en | small English speech recognition model | executed_when_recordings_exist | keyword recall, meaning preservation, latency | False | 0.0 | Fast English-only baseline for learner voice recordings. |
| True | stt | True | whisper-base | openai/whisper-base | encoder-decoder speech recognition model | executed_when_recordings_exist | keyword recall and meaning preservation | True | 0.59 | Configured locally. The benchmark auto-discovers recorded learner audio samples. |
| True | stt | False | whisper-base-en | openai/whisper-base.en | English speech recognition model | executed_when_recordings_exist | keyword recall, meaning preservation, latency | False | 0.0 | English-only base model for comparing against multilingual Whisper Base. |
| True | stt | False | whisper-small | openai/whisper-small | larger speech recognition model | executed_when_recordings_exist | keyword recall, meaning preservation, latency | False | 0.0 | Larger Whisper baseline to test whether quality improves enough to justify slower inference. |
