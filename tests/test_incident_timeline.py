"""Every turn's evidence is built from one validated incident timeline.

No model runs here: timelines come from a fixture, and model calls are stubbed.
"""
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from cloudir.evidence_core.fact_repair import normalise_evidence_facts
from cloudir.scenario.evidence_strength import support_role_content_problems
from cloudir.scenario.incident_timeline import (
    remove_decoy_mentions,
    EVIDENCE_IDS,
    REVEALING_NAME,
    action_anchor,
    action_anchor_problems,
    apply_timeline_evidence,
    build_turn_evidence,
    finalise_incident_timeline,
    load_incident_timeline,
    marking_scheme,
    enforce_action_anchor,
    save_incident_timeline,
    timeline_prompt_block,
    turn_template_plan,
)
from cloudir.scenario.turn_validation import validate_evidence_facts, validate_expected_outcomes

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "sample_incident_timeline.json").read_text())
SCENARIOS = ("identity-management", "automated-security-response", "cost-management")


def config(scenario="identity-management"):
    # Imported lazily: build_runtime binds cloudir.paths, which test_run_management
    # must bind to its temporary workspace first. The slug lookup reads dataset
    # files that a test workspace may not have, so it is replaced.
    from cloudir.dataset_preparation import build_runtime

    with patch.object(build_runtime, "scenario_slug", side_effect=lambda s: s.replace("-", "_")):
        strategy = build_runtime.build_evidence_template_strategy(scenario)

    return {"scenario_id": scenario, "evidence_template_strategy": strategy}


def raw(**changes):
    timeline = copy.deepcopy(FIXTURE)
    timeline.update(changes)
    return timeline


class TimelineEvidenceTests(unittest.TestCase):
    def test_every_turn_of_every_scenario_validates_and_separates_roles(self):
        for scenario in SCENARIOS:
            timeline = finalise_incident_timeline(raw(), config(scenario))
            for turn in range(1, 6):
                with self.subTest(scenario=scenario, turn=turn):
                    items = build_turn_evidence(timeline, config(scenario), turn)
                    self.assertEqual(sorted(i["support_role"] for i in items), ["partial", "strong", "weak"])
                    self.assertEqual(support_role_content_problems(items), [])
                    validate_evidence_facts(copy.deepcopy(items))

    def test_strong_partial_and_weak_come_from_the_right_timeline_events(self):
        timeline = finalise_incident_timeline(raw(), config())
        items = {i["support_role"]: i for i in build_turn_evidence(timeline, config(), 1)}
        attacker = "arn:aws:iam::210987654321:user/dev-priya"

        self.assertEqual(items["strong"]["facts"]["event_name"], "ConsoleLogin")  # turn 1 key event
        partial_rows = items["partial"]["facts"]["activity_rows"]
        self.assertEqual({row[1] for row in partial_rows}, {attacker})  # the same principal...
        self.assertNotIn("185.220.101.42", {row[3] for row in partial_rows})  # ...from its usual IP
        self.assertTrue(all(row[0].startswith("2023-09-30") for row in partial_rows))  # the day before
        self.assertEqual(items["weak"]["template"], "billing")
        self.assertEqual(items["weak"]["facts"]["largest_service"], "S3")

    def test_partial_never_carries_the_attackers_red_flags(self):
        # The attacker actor has mfa=false; the owner's baseline day must not show it.
        timeline = finalise_incident_timeline(raw(), config())
        for template in ("cloudtrail", "iam_activity"):
            plan = {"strong": "guardduty", "partial": template, "weak": "billing"}
            scenario = {"evidence_template_strategy": {"role_templates_by_turn": {"1": plan}}}
            partial = next(i for i in build_turn_evidence(timeline, scenario, 1) if i["support_role"] == "partial")
            self.assertEqual(partial["facts"]["mfa"], "true", template)

    def test_finding_window_covers_only_the_attack_so_far(self):
        timeline = finalise_incident_timeline(raw(), config())
        plan = {"strong": "guardduty", "partial": "iam_activity", "weak": "billing"}
        scenario = {"evidence_template_strategy": {"role_templates_by_turn": {str(t): plan for t in range(1, 6)}}}
        window = {turn: next(i["facts"] for i in build_turn_evidence(timeline, scenario, turn)
                             if i["support_role"] == "strong") for turn in (1, 5)}

        self.assertEqual(window[1]["first_seen"], "2023-10-01T10:11:02Z")  # not the baseline day
        self.assertEqual(window[1]["last_seen"], "2023-10-01T10:14:05Z")  # no later turn's steps
        self.assertEqual(window[5]["last_seen"], "2023-10-01T10:33:15Z")  # recovery shows the whole attack

    def test_identities_stay_consistent_across_turns(self):
        timeline = finalise_incident_timeline(raw(), config())
        text = json.dumps([build_turn_evidence(timeline, config(), turn) for turn in range(1, 6)])

        self.assertIn("185.220.101.42", text)
        self.assertNotIn("admin-test", text)
        self.assertNotIn("123456789012", text)  # the app's old default account never leaks in

    def test_normalisers_leave_timeline_facts_unchanged(self):
        for scenario in SCENARIOS:
            timeline = finalise_incident_timeline(raw(), config(scenario))
            for turn in range(1, 6):
                for item in build_turn_evidence(timeline, config(scenario), turn):
                    after = copy.deepcopy(item)
                    normalise_evidence_facts(after)
                    self.assertEqual(after["facts"], item["facts"], (scenario, turn, item["template"]))

    def test_evidence_ids_never_reveal_the_role(self):
        timeline = finalise_incident_timeline(raw(), config())
        for turn in range(1, 6):
            for item in build_turn_evidence(timeline, config(), turn):
                self.assertEqual(item["id"], EVIDENCE_IDS[item["template"]])
                self.assertNotRegex(item["id"] + item["title"], r"strong|partial|weak")

    def test_the_action_anchor_is_shown_only_by_the_strong_item(self):
        for scenario in SCENARIOS:
            timeline = finalise_incident_timeline(raw(), config(scenario))
            for turn in range(1, 6):
                items = {i["support_role"]: i for i in build_turn_evidence(timeline, config(scenario), turn)}
                anchor = action_anchor(timeline, list(items.values()), turn)
                if items["strong"]["template"] not in {"cloudtrail", "iam_activity", "cloudwatch"}:
                    continue
                with self.subTest(scenario=scenario, turn=turn):
                    self.assertIn(anchor["terms"][0], json.dumps(items["strong"]["facts"]))
                    for role in ("partial", "weak"):
                        self.assertNotIn(anchor["terms"][0], json.dumps(items[role]["facts"]))

    def test_marking_scheme_describes_each_level_without_naming_screenshots(self):
        for scenario in SCENARIOS:
            timeline = finalise_incident_timeline(raw(), config(scenario))
            for turn in range(1, 6):
                items = build_turn_evidence(timeline, config(scenario), turn)
                anchor = action_anchor(timeline, items, turn)
                with self.subTest(scenario=scenario, turn=turn):
                    scheme = marking_scheme(timeline, items, anchor)
                    validate_expected_outcomes(scheme)
                    text = json.dumps(scheme)
                    for evidence_id in EVIDENCE_IDS.values():
                        self.assertNotIn(evidence_id, text)
                    self.assertIn(anchor["shows"], scheme["strong_support"])
                    self.assertIn("dev-priya", scheme["partial_support"])
                    # The lower levels name the strong step only as what they lack.
                    for level in ("partial_support", "weak_support"):
                        shown, _, lacking = scheme[level].partition(" It does not show ")
                        self.assertNotIn(anchor["shows"], shown)
                        self.assertIn(anchor["shows"], lacking)

    def test_the_learner_context_never_names_the_weak_items_principal(self):
        # Live cost-management turn 3: the known context said CI-Deploy's key was active,
        # so the judge treated CI-Deploy's (weak) key page as part of the incident.
        for scenario in SCENARIOS:
            timeline = finalise_incident_timeline(raw(), config(scenario))
            for turn_number in range(1, 6):
                items = build_turn_evidence(timeline, config(scenario), turn_number)
                weak = next(i for i in items if i["support_role"] == "weak")
                decoy = weak["facts"].get("owner") or weak["facts"].get("principal")
                if not decoy:
                    continue
                name = decoy.split("/")[-1]
                turn = {"turn_config": {
                    "briefing": f"Activity by dev-priya needs review. Access key for {name} is active.",
                    "known_context": ["dev-priya signed in without MFA.", f"{name} used S3 recently."]}}
                with self.subTest(scenario=scenario, turn=turn_number):
                    removed = remove_decoy_mentions(turn, timeline, items)
                    self.assertEqual(turn["turn_config"]["briefing"], "Activity by dev-priya needs review.")
                    self.assertEqual(turn["turn_config"]["known_context"], ["dev-priya signed in without MFA."])
                    self.assertEqual(len(removed), 2)

    def test_evaluation_rebuilds_the_marking_scheme_for_the_current_turn(self):
        from cloudir.ai_models.run_turn import turn_marking_scheme

        timeline = finalise_incident_timeline(raw(), config())
        items = build_turn_evidence(timeline, config(), 2)
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "data" / "runtime"
            (runtime / "turns" / "turn_2").mkdir(parents=True)
            (runtime / "scenario_config.json").write_text(json.dumps(config()))
            (runtime / "turns" / "turn_2" / "evidence_facts.json").write_text(json.dumps(items))
            self.assertIsNone(turn_marking_scheme(Path(directory), {"currentTurn": 2}))  # no timeline yet
            save_incident_timeline(runtime, timeline)

            self.assertEqual(turn_marking_scheme(Path(directory), {"currentTurn": 2}),
                             marking_scheme(timeline, items, action_anchor(timeline, items, 2)))
            self.assertIsNone(turn_marking_scheme(Path(directory), {"currentTurn": 3}))

    def test_duplicate_strategy_templates_become_distinct(self):
        plan = turn_template_plan(config("cost-management"), 3)  # strategy lists billing twice
        self.assertEqual(len(set(plan.values())), 3)


class TimelineValidationTests(unittest.TestCase):
    def assertRejected(self, timeline, fragment):
        with self.assertRaises(ValueError) as raised:
            finalise_incident_timeline(timeline, config())
        self.assertIn(fragment, str(raised.exception))

    def test_invented_event_names_are_rejected(self):
        for name, fragment in [("Failed Login Attempt", "not a CloudTrail eventName"),
                               ("ServiceRequest", "not a CloudTrail eventName"),
                               ("DisableIAMUser", "not a real iam.amazonaws.com API"),
                               # Near misses name the real APIs they were probably meant to be.
                               ("UpdateRolePolicy", "closest real ones are UpdateAssumeRolePolicy, UpdateRole, PutRolePolicy")]:
            timeline = raw()
            timeline["events"][-2]["event_name"] = name
            with self.subTest(name=name):
                self.assertRejected(timeline, fragment)

    def test_invalid_background_calls_are_dropped_not_fatal(self):
        events = copy.deepcopy(FIXTURE["events"])
        events[0]["event_name"] = "DescribePolicies"  # not an IAM or EC2 API, but only background noise
        events[0]["event_source"] = "iam.amazonaws.com"
        finalised = finalise_incident_timeline(raw(events=events), config())
        self.assertNotIn("DescribePolicies", {e["event_name"] for e in finalised["events"]})

        events = copy.deepcopy(FIXTURE["events"])
        next(e for e in events if e["event_name"] == "ListUsers")["event_name"] = "DescribePolicies"
        self.assertRejected(raw(events=events), "not a real iam.amazonaws.com API")  # the attack stays strict

    def test_names_that_give_the_answer_away_are_renamed_everywhere(self):
        # The model's favourite giveaways; renaming beats spending an attempt on them.
        timeline = json.loads(json.dumps(raw()).replace("dev-priya", "malicioususer"))
        next(e for e in timeline["events"] if e.get("resource") == "c5.4xlarge")["resource"] = "attackerEC2"
        timeline["findings"][0]["description"] = "API calls by malicioususer came from a known malicious IP address."
        finalised = finalise_incident_timeline(timeline, config())

        names = [actor["principal"] for actor in finalised["actors"]]
        names += [item["resource"] for item in finalised["events"] + finalised["findings"]]
        self.assertFalse([name for name in names if REVEALING_NAME.search(name)])

        user = finalised["actors"][0]["principal"].rsplit("/", 1)[1]
        self.assertRegex(user, r"^[a-z]\.[a-z]+$")  # a person, like the fixture's other users
        # Every mention of the renamed user follows it, so the story still lines up.
        self.assertEqual({e["resource"] for e in finalised["events"] if e["event_name"] == "UpdateAccessKey"}, {user})
        self.assertEqual(finalised["findings"][0]["resource"], user)
        self.assertEqual(finalised["findings"][0]["description"],
                         f"API calls by {user} came from a known malicious IP address.")
        self.assertIn("opsEC2", names)

    def test_placeholder_principals_and_ips_are_rejected(self):
        timeline = raw()
        timeline["actors"][0]["principal"] = "IAM User X"
        self.assertRejected(timeline, "must be an IAM ARN")

        timeline = raw()
        timeline["actors"][2]["source_ip"] = "192.0.2.3"
        self.assertRejected(timeline, "must not use 192.0.2.x")

    def test_incomplete_turn_labels_are_derived_from_time_order(self):
        events = copy.deepcopy(FIXTURE["events"])
        for event in events:
            if event.get("turn") in (3, 4):
                event.pop("turn")  # the model forgot turns 3 and 4
        finalised = finalise_incident_timeline(raw(events=events), config())
        steps = [e for e in finalised["events"] if e["actor"] == "attacker" and e["turn"]]

        self.assertEqual(sorted({e["turn"] for e in steps}), [1, 2, 3, 4])
        self.assertEqual([e["turn"] for e in steps], sorted(e["turn"] for e in steps))  # chronological
        self.assertTrue(all(e["turn"] == 5 for e in finalised["events"] if e["actor"] == "secops"))

    def test_responders_act_in_recovery_and_attackers_never_do(self):
        events = copy.deepcopy(FIXTURE["events"])
        next(e for e in events if e["event_name"] == "RunInstances")["turn"] = 5
        next(e for e in events if e["event_name"] == "UpdateAccessKey")["turn"] = 4
        finalised = finalise_incident_timeline(raw(events=events), config())
        turn_of = {e["event_name"]: e["turn"] for e in finalised["events"]}

        self.assertEqual(turn_of["RunInstances"], 4)
        self.assertEqual(turn_of["UpdateAccessKey"], 5)

    def test_key_status_shows_the_state_before_each_decision(self):
        from cloudir.scenario.incident_timeline import _access_keys

        timeline = finalise_incident_timeline(raw(), config())  # responder disables the key in turn 5
        self.assertEqual(_access_keys(timeline, 4)[0]["status"], "Active")  # containment still to decide
        self.assertEqual(_access_keys(timeline, 5)[0]["status"], "Inactive")  # recovery shows the outcome

    def test_endpoint_aliases_and_finding_types(self):
        timeline = raw()
        timeline["events"][-1] = {**timeline["events"][-1], "event_name": "PutRule", "event_source": "eventbridge.amazonaws.com"}
        finalised = finalise_incident_timeline(timeline, config())
        self.assertIn("events.amazonaws.com", {e["event_source"] for e in finalised["events"]})

        timeline = raw()
        timeline["findings"][0]["finding_type"] = "UnauthorizedAccess:EC2"
        with self.assertRaises(ValueError) as raised:
            finalise_incident_timeline(timeline, config())
        self.assertIn("not a real GuardDuty type", str(raised.exception))

    def test_automated_response_services_use_real_event_names(self):
        def with_attack_step(name, source):
            timeline = raw()
            step = next(e for e in timeline["events"] if e.get("resource") == "c5.4xlarge")
            step.update(event_name=name, event_source=source, resource="orders-remediation")
            return timeline

        # Security Hub is where the model invented names (CreateFinding, PutFindingsFeedback).
        self.assertRejected(with_attack_step("CreateFinding", "securityhub.amazonaws.com"),
                            "not a real securityhub.amazonaws.com API")

        def step_of(timeline):
            finalised = finalise_incident_timeline(timeline, config())
            return next(e for e in finalised["events"] if e["resource"] == "orders-remediation")

        # CloudTrail records the Lambda API under its versioned eventName.
        self.assertEqual(step_of(with_attack_step("UpdateFunctionCode", "lambda.amazonaws.com"))["event_name"],
                         "UpdateFunctionCode20150331v2")
        # GetFindings exists in GuardDuty and Security Hub; the written service is kept.
        self.assertEqual(step_of(with_attack_step("GetFindings", "securityhub.amazonaws.com"))["event_source"],
                         "securityhub.amazonaws.com")
        self.assertEqual(step_of(with_attack_step("BatchImportFindings", "iam.amazonaws.com"))["event_source"],
                         "securityhub.amazonaws.com")

    def test_too_few_attacker_steps_are_rejected(self):
        events = [e for e in FIXTURE["events"] if e.get("turn") in (1, 5) or e["actor"] == "ops"]  # 2 attacker calls
        self.assertRejected(raw(events=events), "at least 4 incident steps")

    def test_background_actor_cannot_do_incident_activity(self):
        timeline = raw()
        timeline["events"][0]["event_name"] = "StopLogging"
        timeline["events"][0]["event_source"] = "cloudtrail.amazonaws.com"
        self.assertRejected(timeline, "looks like incident activity")

    def test_attacker_steps_must_use_the_threat_models_services(self):
        from cloudir.scenario.incident_timeline import threat_model_services

        lambda_threat = {"affected_assets": ["Lambda Functions", "Security Hub", "EventBridge"],
                         "likely_attack_path": ["Crafted Security Hub findings reach Lambda."]}
        self.assertEqual(threat_model_services(lambda_threat),
                         ["events.amazonaws.com", "lambda.amazonaws.com", "securityhub.amazonaws.com"])
        with self.assertRaises(ValueError) as raised:
            finalise_incident_timeline(raw(), config(), threat_model=lambda_threat)  # fixture is an IAM story
        self.assertIn("scenario's own services", str(raised.exception))

        iam_threat = {"affected_assets": ["IAM Roles"], "likely_attack_path": []}
        finalise_incident_timeline(raw(), config(), threat_model=iam_threat)

    def test_a_serious_finding_is_required(self):
        self.assertRejected(raw(findings=[]), "Medium or High finding")

    def test_known_event_sources_and_arn_accounts_are_corrected(self):
        timeline = raw()
        timeline["events"][1]["event_source"] = "iam.amazonaws.com"  # ConsoleLogin
        timeline["actors"][1]["principal"] = "arn:aws:iam::999999999999:role/OpsAutomation"
        finalised = finalise_incident_timeline(timeline, config())

        login = next(e for e in finalised["events"] if e["event_name"] == "ConsoleLogin")
        self.assertEqual(login["event_source"], "signin.amazonaws.com")
        self.assertIn("210987654321", finalised["actors"][1]["principal"])

    def test_every_model_written_attacker_call_is_part_of_the_attack(self):
        finalised = finalise_incident_timeline(raw(), config())
        attack = [e for e in finalised["events"] if e["actor"] == "attacker" and not e.get("baseline")]
        self.assertTrue(all(e["turn"] and e["suspicious"] for e in attack))  # untagged GetCallerIdentity included
        self.assertEqual(finalised["events"], sorted(finalised["events"], key=lambda e: e["time"]))

    def test_background_activity_that_touches_the_suspect_is_replaced(self):
        events = copy.deepcopy(FIXTURE["events"])
        events[0] = {**events[0], "event_name": "DetachUserPolicy", "event_source": "iam.amazonaws.com", "resource": "dev-priya"}
        finalised = finalise_incident_timeline(raw(events=events), config())
        background = [e for e in finalised["events"] if e["actor"] == "ops"]

        self.assertNotIn("DetachUserPolicy", {e["event_name"] for e in background})
        self.assertGreaterEqual(len(background), 3)
        self.assertTrue(all(e["event_name"].startswith(("Describe", "Get", "List", "Head", "Lookup", "Put"))
                            for e in background))


class TimelineMergeTests(unittest.TestCase):
    def setUp(self):
        self.timeline = finalise_incident_timeline(raw(), config())
        self.items = build_turn_evidence(self.timeline, config(), 2)

    def test_facts_come_from_the_timeline_and_neutral_text_from_the_coach(self):
        strong = next(i for i in self.items if i["support_role"] == "strong")
        turn = {"evidence_facts": [
            {"id": strong["id"], "template": strong["template"], "title": "Credential Change Record",
             "summary": "Attacker confirmed the key theft", "why_it_may_matter": "Shows which key was made.",
             "support_role": "weak", "facts": {"event_name": "Failed Login Attempt"}},
        ]}
        apply_timeline_evidence(turn, self.items)
        merged = next(i for i in turn["evidence_facts"] if i["id"] == strong["id"])

        self.assertEqual(merged["facts"], strong["facts"])
        self.assertEqual(merged["support_role"], "strong")
        self.assertEqual(merged["title"], "Credential Change Record")
        self.assertEqual(merged["summary"], strong["summary"])  # caption, never the coach's claim
        self.assertEqual(len(turn["evidence_facts"]), 3)

    def test_text_naming_the_role_is_replaced_with_neutral_defaults(self):
        weak = next(i for i in self.items if i["support_role"] == "weak")
        turn = {"evidence_facts": [{"id": weak["id"], "title": "Weak distractor", "why_it_may_matter": "The best clue."}]}
        apply_timeline_evidence(turn, self.items)
        merged = next(i for i in turn["evidence_facts"] if i["id"] == weak["id"])

        self.assertEqual(merged["title"], weak["title"])
        self.assertEqual(merged["why_it_may_matter"], weak["why_it_may_matter"])

    def test_saved_timeline_loads_only_for_its_scenario(self):
        with tempfile.TemporaryDirectory() as directory:
            save_incident_timeline(Path(directory), self.timeline)
            self.assertIsNotNone(load_incident_timeline(Path(directory), "identity-management"))
            self.assertIsNone(load_incident_timeline(Path(directory), "cost-management"))


class ActionAnchorTests(unittest.TestCase):
    # The identity scenario's turn 1 as the coach wrote it: best and partial swapped.
    ANCHOR = {"terms": ["GuardDuty"], "title": "Investigate GuardDuty Finding",
              "example": "Investigate the High-severity GuardDuty finding for j.moreno"}

    def actions(self):
        return [
            {"title": "Review Access Keys", "description": "Inspect j.moreno's access keys.", "choice_role": "weak"},
            {"title": "Detect IAM Activity", "description": "Look for suspicious IAM activities involving j.moreno.",
             "choice_role": "best"},
            {"title": "Check GuardDuty Finding", "description": "Verify the GuardDuty finding.", "choice_role": "partial"},
        ]

    def roles(self, actions):
        return {action["title"]: action["choice_role"] for action in actions}

    def test_swapped_labels_are_swapped_back_at_once(self):
        # Repaired immediately: each retry cost a full turn generation.
        actions = self.actions()

        self.assertEqual(enforce_action_anchor(actions, self.ANCHOR), "swapped best and partial")
        self.assertEqual(self.roles(actions), {"Review Access Keys": "weak", "Detect IAM Activity": "partial",
                                               "Check GuardDuty Finding": "best"})

    def test_a_best_action_nobody_anchored_is_rewritten(self):
        actions = self.actions()
        actions[2]["title"], actions[2]["description"] = "Check Threat Alerts", "Verify the alert."

        self.assertEqual(enforce_action_anchor(actions, self.ANCHOR), "rewrote the best action")
        self.assertEqual(action_anchor_problems(actions, self.ANCHOR), [])
        self.assertEqual(actions[1]["description"], self.ANCHOR["example"] + ".")

    def test_correct_actions_are_left_alone(self):
        actions = self.actions()
        enforce_action_anchor(actions, self.ANCHOR)
        self.assertIsNone(enforce_action_anchor(actions, self.ANCHOR))

    def test_lambda_versioned_event_names_use_the_plain_api_name(self):
        # The coach writes UpdateFunctionCode; the check demanded UpdateFunctionCode20150331v2.
        timeline = copy.deepcopy(finalise_incident_timeline(raw(), config()))
        turn_1 = [e for e in timeline["events"] if e["turn"] == 1]
        # A failed ConsoleLogin is suspicious, so it would be the headline instead.
        turn_1[0].update(mfa="true", result="Success", error_code="-")
        turn_1[-1]["event_name"], turn_1[-1]["event_source"] = "UpdateFunctionCode20150331v2", "lambda.amazonaws.com"
        plan = {"strong": "cloudtrail", "partial": "iam_activity", "weak": "billing"}
        scenario = {"evidence_template_strategy": {"role_templates_by_turn": {"1": plan}}}
        items = build_turn_evidence(timeline, scenario, 1)
        anchor = action_anchor(timeline, items, 1)

        self.assertEqual(anchor["terms"], ["UpdateFunctionCode"])
        self.assertIn("UpdateFunctionCode20150331v2", anchor["shows"])  # what the screenshot shows
        best = {"title": "Trace the UpdateFunctionCode change", "description": "d", "choice_role": "best"}
        self.assertIsNone(enforce_action_anchor([best], anchor))

    def test_spacing_and_case_do_not_matter(self):
        anchor = {"terms": ["CreateAccessKey"], "title": "t", "example": "e"}
        actions = [{"title": "Trace the create access key call", "description": "d", "choice_role": "Best"},
                   {"title": "Review logins", "description": "d", "choice_role": "partial"}]
        self.assertEqual(action_anchor_problems(actions, anchor), [])

    def test_the_coach_is_told_the_anchor(self):
        timeline = finalise_incident_timeline(raw(), config())
        items = build_turn_evidence(timeline, config(), 1)
        brief = timeline_prompt_block(timeline, items, action_anchor(timeline, items, 1))

        self.assertIn('The best action MUST contain "ConsoleLogin"', brief)


class TimelineIntegrationTests(unittest.TestCase):
    def test_next_turn_uses_timeline_evidence_and_briefs_the_coach(self):
        from cloudir.scenario import turn_orchestrator

        timeline = finalise_incident_timeline(raw(), config())
        coach_turn = {
            "turn_config": {"turn": 2, "phase": "Correlation", "briefing": "Brief.", "known_context": ["a", "b", "c"],
                            "coach_guidance": "Guide."},
            "actions": [{"id": f"a{n}", "title": f"Action {n}", "description": "d", "recommended_next_focus": "f",
                         "choice_role": role} for n, role in enumerate(["best", "partial", "weak"])],
            "evidence_facts": [{"id": "x", "template": "billing", "facts": {"current_spend": "$1"}}],
            "expected_outcomes": {"strong_support": "s", "partial_support": "p", "weak_support": "w", "unsupported": "u"},
        }
        with patch.object(turn_orchestrator, "generate_next_turn_json", return_value=coach_turn) as generate, \
             patch.object(turn_orchestrator, "evaluate_generated_turn_quality",
                          return_value={"pass": False, "quality_score": 60,  # as every live review scored
                                        "retry_instruction": "Repeats the same evidence titles."}), \
             patch.object(turn_orchestrator, "log_memory"):
            turn = turn_orchestrator.generate_turn_with_quality_retry(
                scenario_config=config(), hidden_truth={}, current_state={}, completed_turn={}, selected_action={},
                selected_evidence={}, vlm_output={}, security_output={"verdict": "Strong Support"}, coach_output={},
                next_turn_number=2, timeline=timeline)

        items = build_turn_evidence(timeline, config(), 2)
        expected = {i["id"]: i["facts"] for i in items}
        self.assertEqual({i["id"]: i["facts"] for i in turn["evidence_facts"]}, expected)
        self.assertIn("FIXED EVIDENCE FOR THIS TURN", generate.call_args.kwargs["evidence_brief"])
        # Generic actions are repaired at once, and the advisory review never forces a retry.
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(action_anchor_problems(turn["actions"], action_anchor(timeline, items, 2)), [])
        diagnostics = turn["generation_diagnostics"]
        self.assertEqual(diagnostics["outcome"], "accepted_timeline_turn")
        self.assertEqual(diagnostics["anchor_repairs"], ["attempt 1: rewrote the best action"])
        # The judge's marking scheme replaces the coach's expected outcomes.
        self.assertEqual(turn["expected_outcomes"], marking_scheme(timeline, items, action_anchor(timeline, items, 2)))

    def test_preparation_writes_the_timeline_and_builds_turn_1_from_it(self):
        from cloudir.dataset_preparation import build_acse

        coach_turn = {
            "turn_config": {"turn": 1}, "actions": [], "expected_outcomes": {},
            "evidence_facts": [{"id": "anything", "template": "cloudtrail", "facts": {"event_name": "Failed Login Attempt"}}],
        }
        runtime = {"scenario_seed": {}, "hidden_truth": {}, "scenario_config": {**config(), "scenario_progression": {}}}
        written = []

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            # Nothing touches the real workspace: paths, writers and models are stubbed.
            for name in ("scenario_slug", "processed_dir", "select_acse_case", "parse_architecture", "normalise_threat",
                         "unload_image_model", "unload_security_model", "unload_coach_model", "_reset_turn_1_outputs",
                         "_render_turn_1_evidence", "_validate_outputs"):
                stack.enter_context(patch.object(build_acse, name))
            stack.enter_context(patch.object(build_acse, "SCENARIO_DATA_DIR", Path(directory)))
            stack.enter_context(patch.object(build_acse, "build_runtime_files", return_value=runtime))
            stack.enter_context(patch.object(build_acse, "_write_json", side_effect=lambda path, data: written.append((path.name, data))))
            timeline_writer = stack.enter_context(patch.object(
                build_acse, "write_incident_timeline", side_effect=[raw(findings=[]), raw()]))
            coach = stack.enter_context(patch.object(build_acse, "generate_initial_turn_from_dataset", return_value=coach_turn))
            events = list(build_acse.stream_build_acse_dataset_scenario("identity-management"))
            saved_timeline = load_incident_timeline(Path(directory), "identity-management")

        self.assertEqual(events[-1]["status"], "complete", events[-1])
        self.assertEqual(timeline_writer.call_count, 2)
        first, retry = timeline_writer.call_args_list
        self.assertIn("Medium or High finding", retry.kwargs["retry_instruction"])
        # The retry edits the rejected draft instead of starting a new story.
        self.assertIsNone(first.kwargs["rejected_timeline"])
        self.assertEqual(retry.kwargs["rejected_timeline"], raw(findings=[]))
        self.assertIsNotNone(saved_timeline)
        self.assertIn("FIXED EVIDENCE FOR THIS TURN", coach.call_args.kwargs["evidence_brief"])
        evidence = dict(written)["evidence_facts.json"]
        turn_1_items = build_turn_evidence(saved_timeline, config(), 1)
        expected = {i["id"]: i["facts"] for i in turn_1_items}
        self.assertEqual({i["id"]: i["facts"] for i in evidence}, expected)
        self.assertEqual(dict(written)["expected_outcomes.json"],
                         marking_scheme(saved_timeline, turn_1_items, action_anchor(saved_timeline, turn_1_items, 1)))


if __name__ == "__main__":
    unittest.main()
