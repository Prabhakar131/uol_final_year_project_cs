# CloudIR STT Model Evaluation

Generated: `2026-08-16T11:36:41`
Batch: `stt-full-8sample-run`
Task: `stt`
Mode: `model inference`
Cases: `0`

## Selected Models

| name | model_id | family |
| --- | --- | --- |
| whisper-tiny-en | openai/whisper-tiny.en | small English speech recognition model |
| whisper-base | openai/whisper-base | encoder-decoder speech recognition model |
| whisper-base-en | openai/whisper-base.en | English speech recognition model |
| whisper-small | openai/whisper-small | larger speech recognition model |

## Summary

| model_name | model_id | samples | mean_wer | median_wer | min_wer | max_wer | meaning_preservation_rate | avg_keyword_recall | avg_latency_seconds | errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| whisper-base | openai/whisper-base | 8 | 1.5024 | 1.1666 | 0.52 | 4.913 | 0.5 | 0.425 | 1.131 | 0 |
| whisper-base-en | openai/whisper-base.en | 8 | 0.9556 | 0.9148 | 0.52 | 1.4211 | 0.75 | 0.625 | 0.487 | 0 |
| whisper-small | openai/whisper-small | 8 | 0.9662 | 0.9574 | 0.52 | 1.4211 | 0.75 | 0.625 | 2.325 | 0 |
| whisper-tiny-en | openai/whisper-tiny.en | 8 | 0.9662 | 0.9574 | 0.52 | 1.4211 | 0.75 | 0.625 | 0.283 | 0 |

## Evaluation Cases

| task | case_id | source | expected |
| --- | --- | --- | --- |
| stt | audio_sample_limit | notebooks/model_benchmarks/stt_audio_samples/audio_samples | all samples |

## Result Sample

| model_name | model_id | sample_id | expected_text | transcript | expected_keywords | matched_keywords | keyword_recall | meaning_preserved | word_error_rate | latency_seconds | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| whisper-tiny-en | openai/whisper-tiny.en | partial_iam_activity | The IAM activity gives partial support because it shows suspicious API calls, but it does not prove the full attack path on its own. | The IAM activity timeline partly supports the action because it shows the principle and actions but I still need cloud trail details to confirm the exact source IP and event time. | iam, activity, partial, api calls, attack path | iam, activity, partial | 0.6 | True | 0.9583 | 0.306 |  |
| whisper-tiny-en | openai/whisper-tiny.en | partial_cloudwatch_correlation | The CloudWatch correlation gives partial support because it helps confirm the timeline, but it still needs CloudTrail or IAM evidence. | Cloudwatch correlation logs are useful contacts but they only partly support the containment decision unless I can connect the log entries to the IAM user and access key. | cloudwatch, correlation, partial, timeline, cloudtrail | cloudwatch, correlation, partial | 0.6 | True | 1.25 | 0.261 |  |
| whisper-tiny-en | openai/whisper-tiny.en | strong_access_key_containment | The access key evidence strongly supports containment because the same key is active and should be disabled or rotated immediately. | The Access Key Containment Review supports containment because the key is active linked to the suspicious principle and should be rotated or disabled after checking last used activ... | access key, containment, active, disable, rotate | access key, containment, active, disable, rotate | 1.0 | True | 0.75 | 0.264 |  |
| whisper-tiny-en | openai/whisper-tiny.en | strong_cloudtrail_stop_logging | The CloudTrail log strongly supports stopping the incident because it shows StopLogging by an IAM user from a suspicious source IP without MFA. | I chose to review recent cloud trail logs because the stop logging event IAM user source IP event time region and missing MFA directly support investigating suspicious identity act... | cloudtrail, stoplogging, iam user, source ip, mfa | iam user, source ip, mfa | 0.6 | True | 1.1739 | 0.293 |  |
| whisper-tiny-en | openai/whisper-tiny.en | unsupported_generic_policy | The generic policy evidence is unsupported because it does not show the specific incident, user, source IP, or CloudTrail event. | A generic policy reminder is unsupported evidence because it does not show the event name IAM user access key source IP, MFA status or any incident timestamp. | unsupported, policy, incident, source ip, cloudtrail | unsupported, policy, incident, source ip | 0.8 | True | 0.7 | 0.279 |  |
| whisper-tiny-en | openai/whisper-tiny.en | unsupported_overclaiming_guardduty | The GuardDuty finding should not be overclaimed because it is unsupported without matching IAM, CloudTrail, or access key evidence. | I should not claim God duty proves the access key was abused unless the finding shows the same IAM principle. Remote IP address severity and matching time window. | guardduty, unsupported, iam, cloudtrail, access key | iam, access key | 0.4 | False | 1.4211 | 0.287 |  |
| whisper-tiny-en | openai/whisper-tiny.en | weak_access_key_review | The access key review is weak support because it suggests credential risk, but it does not prove the exact action or incident cause. | The Access Key Review is weak evidence if it only shows key status without proving the key was used from the suspicious source IP during the cloud trail event window. | access key, weak, credential, risk, incident | access key, weak | 0.4 | False | 0.9565 | 0.298 |  |
| whisper-tiny-en | openai/whisper-tiny.en | weak_billing_snapshot | The billing snapshot is weak support because it may show impact, but it does not directly identify the IAM user or the suspicious source IP. | The billing snapshot is weak support for identity containment because it shows cause impact but it does not prove the IAM principle, MFA status or suspicious IP. | billing, weak, impact, iam user, source ip | billing, weak, impact | 0.6 | True | 0.52 | 0.274 |  |
| whisper-base | openai/whisper-base | partial_iam_activity | The IAM activity gives partial support because it shows suspicious API calls, but it does not prove the full attack path on its own. | Iaam, aktiviti, timeline, periksaan, kerana berada di tempat, persoan dan periksaan, tapi saya masih perlu mempunyai periksaan yang berada di tempat, untuk menggunakan IP dan event... | iam, activity, partial, api calls, attack path |  | 0.0 | False | 1.0833 | 1.906 |  |
| whisper-base | openai/whisper-base | partial_cloudwatch_correlation | The CloudWatch correlation gives partial support because it helps confirm the timeline, but it still needs CloudTrail or IAM evidence. | CloudWatch Correlation logs are useful contacts, but they only partly support the containment decision unless I can connect the log entries to the IAM user and access key. | cloudwatch, correlation, partial, timeline, cloudtrail | cloudwatch, correlation, partial | 0.6 | True | 1.25 | 0.618 |  |
| whisper-base | openai/whisper-base | strong_access_key_containment | The access key evidence strongly supports containment because the same key is active and should be disabled or rotated immediately. | The access key containment review supports containment because the key is active linked to the suspicious principle and should be rotated or disabled after checking last used activ... | access key, containment, active, disable, rotate | access key, containment, active, disable, rotate | 1.0 | True | 0.75 | 0.531 |  |
| whisper-base | openai/whisper-base | strong_cloudtrail_stop_logging | The CloudTrail log strongly supports stopping the incident because it shows StopLogging by an IAM user from a suspicious source IP without MFA. | Saya melihat kemungkinan kelihatan kelihatan kelihatan kelihatan kelihatan kelihatan kelihatan kelihatan kelihatan kelihatan kelihatan kelihatan kelihatan kelihatan kelihatan kelih... | cloudtrail, stoplogging, iam user, source ip, mfa |  | 0.0 | False | 4.913 | 3.025 |  |

## Candidate Matrix

| selected_for_run | stage | used_in_application | name | model_id | family | benchmark_mode | metric | cached | cache_size_gb | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | stt | False | whisper-tiny-en | openai/whisper-tiny.en | small English speech recognition model | executed_when_recordings_exist | keyword recall, meaning preservation, latency | False | 0.0 | Fast English-only baseline for learner voice recordings. |
| True | stt | True | whisper-base | openai/whisper-base | encoder-decoder speech recognition model | executed_when_recordings_exist | keyword recall and meaning preservation | True | 0.59 | Configured locally. The benchmark auto-discovers recorded learner audio samples. |
| True | stt | False | whisper-base-en | openai/whisper-base.en | English speech recognition model | executed_when_recordings_exist | keyword recall, meaning preservation, latency | False | 0.0 | English-only base model for comparing against multilingual Whisper Base. |
| True | stt | False | whisper-small | openai/whisper-small | larger speech recognition model | executed_when_recordings_exist | keyword recall, meaning preservation, latency | False | 0.0 | Larger Whisper baseline to test whether quality improves enough to justify slower inference. |
