"""Evidence content must match its support_role before a turn is saved.

The security judge grades only the rendered screenshot. These tests cover the
case where partial or weak evidence visibly shows strong content, which the
judge then (correctly) rewarded as Strong Support.
"""
import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from cloudir.ai_models.coach_model import _repair_initial_turn_action_evidence_alignment
from cloudir.evidence.templates.cloudtrail_template import _normalise_related_events
from cloudir.evidence_core.fact_repair import (
    _repair_access_key_facts,
    _repair_cloudtrail_facts,
    _repair_iam_activity_facts,
    normalise_evidence_facts,
)
from cloudir.scenario import scenario_progression as progression
from cloudir.scenario.evidence_strength import support_role_content_problems
from cloudir.scenario.turn_orchestrator import build_deterministic_next_turn
from cloudir.scenario.turn_validation import validate_evidence_facts

USER = "arn:aws:iam::123456789012:user/admin-test"
IP = "185.220.101.42"


def cloudtrail_stoplogging(role="strong"):
    return {"id": "cloudtrail_log_row", "title": "Unauthorized Access Attempt", "type": "cloudtrail",
            "template": "cloudtrail", "support_role": role, "summary": "s", "why_it_may_matter": "w",
            "facts": {"event_name": "StopLogging", "event_source": "cloudtrail.amazonaws.com", "user": USER,
                      "source_ip": IP, "mfa": "false", "event_time": "2023-10-01T10:21:35Z",
                      "related_events": [["2023-10-01T10:21:35Z", "StopLogging", USER, IP, "Success"]]}}


def cloudwatch(rows, role, alarm_state="OK"):
    return {"id": "cloudwatch_insights_query", "title": "CloudWatch Log Insight Query", "type": "cloudwatch",
            "template": "cloudwatch", "support_role": role, "summary": "s", "why_it_may_matter": "w",
            "facts": {"log_rows": rows, "query": "fields @message", "log_group": "/aws/cloudtrail/organization/prod",
                      "matched_records": str(len(rows)), "time_range": "2023-10-01 10:10-10:30 UTC",
                      "alarm_state": alarm_state}}


def access_key(role="weak", last_used_time="2023-09-28T08:14:52Z", source_ip="203.0.113.25"):
    return {"id": "access_key_info", "title": "Access Key Information", "type": "access_key",
            "template": "access_key", "support_role": role, "summary": "s", "why_it_may_matter": "w",
            "facts": {"access_key_id": "AKIA4Z7EXAMPLE92K", "owner": USER, "status": "Inactive",
                      "last_used_service": "iam.amazonaws.com", "last_used_region": "ap-southeast-1",
                      "last_used_time": last_used_time, "source_ip": source_ip}}


def billing(role="weak"):
    return {"id": "billing", "title": "Billing Snapshot", "type": "billing", "template": "billing",
            "support_role": role, "summary": "s", "why_it_may_matter": "w",
            "facts": {"current_spend": "$72.40", "previous_average": "$68.10", "largest_service": "EC2",
                      "region": "ap-southeast-1", "change": "+6%"}}


# The "partial" item from the saved Turn 1 that the judge graded Strong Support.
PADDED_PARTIAL_ROWS = [
    ["2023-10-01T10:12:10Z", "auth/identity-center", f"ConsoleLogin success for {USER} from {IP}; mfaAuthenticated=false"],
    ["2023-10-01T10:18:44Z", "cloudtrail/audit", f"ListTrails called by {USER} from {IP}"],
    ["2023-10-01T10:21:35Z", "cloudtrail/audit", f"StopLogging called on prod-org-trail by {USER}"],
    ["2023-10-01T10:24:02Z", "iam/credential-monitor", f"CreateAccessKey succeeded for {USER}; key status Active"],
    ["2023-10-01T10:26:11Z", "security/correlation", f"Same source IP {IP} observed across authentication and credential events"],
]


class SupportRoleContentTests(unittest.TestCase):
    def test_saved_turn_1_partial_that_shows_the_whole_attack_is_rejected(self):
        items = [cloudtrail_stoplogging(), cloudwatch(PADDED_PARTIAL_ROWS, "partial", "Identity correlation signal"),
                 access_key()]
        with self.assertRaises(ValueError) as raised:
            validate_evidence_facts(items)
        message = str(raised.exception)
        self.assertIn("stoplogging, the strong item's main event", message)
        self.assertIn("3 suspicious events", message)
        self.assertIn("states a conclusion", message)

    def test_partial_with_one_generic_event_and_weak_without_activity_pass(self):
        partial = cloudwatch([["2023-10-01T10:12:10Z", "auth/identity-center", f"ConsoleLogin success for {USER}"]], "partial")
        validate_evidence_facts([cloudtrail_stoplogging(), partial, billing()])

    def test_weak_evidence_must_not_show_suspicious_activity_or_findings(self):
        finding = {"id": "gd", "title": "GuardDuty", "type": "guardduty", "template": "guardduty",
                   "support_role": "weak", "summary": "s", "why_it_may_matter": "w",
                   "facts": {"finding_type": "Stealth:IAMUser/CloudTrailLoggingDisabled", "severity": "Medium"}}
        partial = cloudwatch([["2023-10-01T10:13:00Z", "app/api", "GET /health returned 200"]], "partial")
        problems = support_role_content_problems([cloudtrail_stoplogging(), partial, finding])
        self.assertTrue(any("Weak evidence gd shows suspicious activity" in p for p in problems))

    def test_weak_evidence_must_not_repeat_strong_events(self):
        weak = cloudwatch([["2023-10-01T10:18:44Z", "cloudtrail/audit", "ListTrails called"]], "weak")
        strong = cloudtrail_stoplogging()
        strong["facts"]["related_events"].append(["2023-10-01T10:18:44Z", "ListTrails", USER, IP, "Success"])
        problems = support_role_content_problems([strong, access_key("partial"), weak])
        self.assertEqual(problems, ["Weak evidence cloudwatch_insights_query repeats the strong item's events (listtrails)."])

    def test_partial_must_not_show_every_strong_event(self):
        strong = cloudwatch([["t1", "app", "Validated target against allowlist"], ["t2", "app", "Workflow requested"]], "strong")
        partial = {"id": "iam", "title": "IAM", "type": "iam_activity", "template": "iam_activity", "support_role": "partial",
                   "facts": {"activity_rows": [["t1", USER, "Validated target against allowlist", IP],
                                               ["t2", USER, "Workflow requested", IP], ["t3", USER, "GetCallerIdentity", IP]]}}
        problems = support_role_content_problems([strong, partial, billing()])
        self.assertIn("Partial evidence iam shows every event the strong item shows.", problems)

    def test_conclusion_wording_in_risk_flags_is_rejected(self):
        partial = {"id": "iam", "title": "IAM", "type": "iam_activity", "template": "iam_activity", "support_role": "partial",
                   "facts": {"activity_rows": [["t", USER, "GetCallerIdentity", IP]],
                             "risk_flags": ["Credential actions linked to suspicious IP"]}}
        problems = support_role_content_problems([cloudtrail_stoplogging(), partial, billing()])
        self.assertIn("Partial evidence iam states a conclusion ('linked to') instead of recording an event.", problems)

    def test_strong_event_evidence_must_show_an_event(self):
        problems = support_role_content_problems([cloudwatch([], "strong"), access_key("partial"), billing()])
        self.assertIn("Strong evidence cloudwatch_insights_query shows no event rows or finding.", problems)

    def test_recovery_turn_partial_may_show_incident_history(self):
        # Suspicious-event caps apply only when the strong item itself is about suspicious activity.
        strong = cloudtrail_stoplogging()
        strong["facts"].update(event_name="StartLogging", mfa="true",
                               related_events=[["t", "StartLogging", "SecurityAdmin", "AWS Internal", "Success"]])
        partial = {"id": "iam", "title": "IAM", "type": "iam_activity", "template": "iam_activity", "support_role": "partial",
                   "facts": {"mfa": "false", "activity_rows": [["t1", USER, "ConsoleLogin", IP], ["t2", USER, "CreateAccessKey", IP],
                                                               ["t3", "SecurityAdmin", "UpdateAccessKey", "AWS Internal"]]}}
        self.assertEqual(support_role_content_problems([strong, partial, billing()]), [])


class HardcodedEvidenceSetTests(unittest.TestCase):
    """Fallback and repair sets bypass the model, so they must satisfy the same checks."""

    @staticmethod
    def repaired(repair, turn):
        generated = {"turn_config": {"turn": turn},
                     "evidence_facts": [{"template": "cloudtrail", "support_role": "strong", "title": "CloudTrail log entry", "facts": {}}]}
        repair(generated)
        return generated["evidence_facts"]

    def test_every_hardcoded_evidence_set_passes_validation(self):
        baseline = {"actions": [{"title": "Inspect Audit Logs"}], "evidence_facts": [{}]}
        _repair_initial_turn_action_evidence_alignment(baseline)
        sets = {
            "turn 1 baseline": baseline["evidence_facts"],
            "fallback turn 2": build_deterministic_next_turn(2, {})["evidence_facts"],
            "fallback turn 3": build_deterministic_next_turn(3, {})["evidence_facts"],
            "identity turn 2": self.repaired(progression.repair_turn2_correlation_evidence, 2),
            "identity turn 3": self.repaired(progression.repair_turn3_response_evidence, 3),
            "identity turn 4": self.repaired(progression.repair_turn4_containment_evidence, 4),
            "identity turn 5": self.repaired(progression.repair_turn5_recovery_evidence, 5),
            "automation turn 2": progression.automated_security_response_turn2_evidence(),
            "automation turn 3": progression.automated_security_response_turn3_evidence(),
            "automation turn 4": progression.automated_security_response_turn4_evidence(),
            "automation turn 5": progression.automated_security_response_turn5_evidence(),
        }
        for name, items in sets.items():
            with self.subTest(name):
                self.assertEqual(len(items), 3)
                validate_evidence_facts(copy.deepcopy(items))


class NoIncidentPaddingTests(unittest.TestCase):
    def test_cloudtrail_without_related_events_shows_only_the_selected_event(self):
        evidence = cloudtrail_stoplogging("partial")
        del evidence["facts"]["related_events"]
        normalise_evidence_facts(evidence)
        repaired = _repair_cloudtrail_facts(copy.deepcopy(cloudtrail_stoplogging()["facts"]) | {"related_events": None}, "", {})
        for rows in (evidence["facts"]["related_events"], repaired["related_events"]):
            self.assertEqual([row[1] for row in rows], ["StopLogging"])
        self.assertEqual([row[1] for row in _normalise_related_events([], evidence["facts"])], ["StopLogging"])

    def test_iam_timeline_keeps_only_authored_rows_and_unknown_defaults(self):
        one_row = [["2023-10-01T10:13:02Z", USER, "GetCallerIdentity", IP]]
        evidence = {"id": "iam", "type": "iam_activity", "template": "iam_activity", "support_role": "weak",
                    "summary": "StopLogging was investigated", "facts": {"activity_rows": copy.deepcopy(one_row)}}
        normalise_evidence_facts(evidence)
        repaired = _repair_iam_activity_facts({"activity_rows": copy.deepcopy(one_row)}, "StopLogging", {})
        for facts in (evidence["facts"], repaired):
            self.assertEqual(facts["activity_rows"], one_row)
            self.assertEqual(facts["risk_flags"], [])
            self.assertEqual(facts["access_key_status"], "Unknown")
        empty = _repair_iam_activity_facts({}, "StopLogging by admin-test", {"title": "StopLogging"})
        self.assertEqual(empty["activity_rows"], [])

    def test_access_key_is_not_tied_to_incident_ip_or_time_by_default(self):
        evidence = access_key(last_used_time="", source_ip="")
        normalise_evidence_facts(evidence)
        repaired = _repair_access_key_facts({"access_key_id": "AKIA4Z7EXAMPLE92K"}, "", {})
        for facts in (evidence["facts"], repaired):
            self.assertEqual(facts["last_used_time"], "Unknown")
            self.assertEqual(facts["source_ip"], "Unknown")


class InitialTurnRetryTests(unittest.TestCase):
    def test_rejected_turn_1_is_regenerated_with_the_error(self):
        import json
        import tempfile
        from pathlib import Path

        from cloudir.dataset_preparation import build_acse

        timeline = json.loads((Path(__file__).parent / "fixtures" / "sample_incident_timeline.json").read_text())
        accepted = {"turn_config": {"turn": 1}, "actions": [], "expected_outcomes": {}, "evidence_facts": []}
        runtime = {"scenario_seed": {}, "hidden_truth": {}, "scenario_config": {"scenario_id": "identity-management"}}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            # No model, file-system or rendering stage runs; only the Turn 1 loop is real.
            for name in ("scenario_slug", "processed_dir", "select_acse_case", "parse_architecture", "normalise_threat", "unload_image_model",
                         "unload_security_model", "unload_coach_model", "_reset_turn_1_outputs", "_write_json",
                         "_render_turn_1_evidence", "_validate_outputs"):
                stack.enter_context(patch.object(build_acse, name))
            stack.enter_context(patch.object(build_acse, "SCENARIO_DATA_DIR", Path(directory)))
            stack.enter_context(patch.object(build_acse, "build_runtime_files", return_value=runtime))
            stack.enter_context(patch.object(build_acse, "write_incident_timeline", return_value=timeline))
            generate = stack.enter_context(patch.object(
                build_acse, "generate_initial_turn_from_dataset",
                side_effect=[ValueError("Coach output is missing keys: {'actions'}"), accepted]))
            events = list(build_acse.stream_build_acse_dataset_scenario("identity-management"))

        self.assertEqual(events[-1]["status"], "complete", events[-1])
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(generate.call_args_list[0].kwargs["retry_instruction"], "")
        self.assertIn("missing keys", generate.call_args_list[1].kwargs["retry_instruction"])
        self.assertTrue(any("Turn 1 was rejected" in event["message"] for event in events))


if __name__ == "__main__":
    unittest.main()
