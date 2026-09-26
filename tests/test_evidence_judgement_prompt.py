"""Protect the live evaluator from author labels and another model's verdict."""

import json
import unittest
from unittest.mock import patch

from cloudir.ai_models import image_model, security_model
from notebooks.model_benchmarks.runners.run_benchmarks import (
    BenchmarkCase, build_security_prompt, build_vlm_prompt,
)


class EvidenceJudgementPromptTests(unittest.TestCase):
    def test_extraction_discards_verdicts_and_rejects_type_list(self):
        result = image_model._parse_vlm_output(json.dumps({
            "evidence_type": "cloudtrail | cloudwatch | unknown",
            "visible_facts_extracted": ["Matched records: 5"],
            "security_relevance": "strong",
            "supports_selected_action": "strong",
            "reason": "This supports investigating",
            "extraction_warnings": ["Result rows unreadable"],
        }), {"type": "cloudwatch"})
        self.assertEqual(result["evidence_type"], "unknown")
        self.assertEqual(result["visible_facts_extracted"], ["Matched records: 5"])
        self.assertEqual(result["extraction_warnings"], ["Result rows unreadable"])
        for key in ("security_relevance", "supports_selected_action", "reason"):
            self.assertNotIn(key, result)

    def test_failed_extraction_does_not_grade_authored_summary(self):
        with self.assertRaisesRegex(ValueError, "VLM extraction failed"):
            image_model._parse_vlm_output("unparseable", {
                "summary": "Author claims suspicious activity", "supportRole": "strong"
            })

    def test_benchmark_judgement_prompts_exclude_author_answers(self):
        action = {"title": "Review Recent IAM Activities", "choice_role": "best"}
        evidence = {"id": "access_key_info", "title": "Access Key Information",
                    "type": "access_key", "supportRole": "weak", "summary": "Author summary"}
        case = BenchmarkCase("weak_case", 1, action, evidence,
                             {"transcript": "I would check the screenshot"},
                             {"visible_facts_extracted": ["Status: Inactive"],
                              "supports_selected_action": "strong"},
                             "Weak Support", "weak")
        prompts = [build_vlm_prompt({"selected_action": action, "selected_evidence": evidence}),
                   build_security_prompt(case)]
        for prompt in prompts:
            self.assertIn("Access Key Information", prompt)
            for leaked in ("choice_role", "supportRole", "Author summary"):
                self.assertNotIn(leaked, prompt)
        self.assertNotIn('"supports_selected_action": "strong"', prompts[1])

    def test_vision_model_receives_image_identity_without_author_labels(self):
        action = {"title": "Review Recent IAM Activities", "choice_role": "best"}
        evidence = {"id": "access_key_info", "title": "Access Key Information",
                    "type": "access_key", "supportRole": "weak",
                    "summary": "Author summary", "whyItMayMatter": "Expected weak evidence"}
        prompt_action, prompt_evidence, hint = image_model._evidence_prompt_inputs(action, evidence)
        self.assertEqual(hint, "access_key")
        self.assertEqual(prompt_action, {"title": "Review Recent IAM Activities"})
        self.assertEqual(prompt_evidence, {"id": "access_key_info", "title": "Access Key Information",
                                           "type": "access_key"})

    def test_security_model_receives_observations_without_answer_hints(self):
        # Inline case keeps this regression independent of ignored, local run files.
        action = {"id": "review_iam_activities", "title": "Review Recent IAM Activities", "choice_role": "best"}
        evidence = {"id": "access_key_info", "title": "Access Key Information",
                    "type": "access_key", "supportRole": "weak",
                    "summary": "An inactive key", "whyItMayMatter": "Expected weak evidence"}
        vlm = {"visible_facts_extracted": ["Status: Inactive", "Last used: 2023-10-01"],
               "visible_evidence_summary": "Inactive access key details",
               "supports_selected_action": "strong", "security_relevance": "strong",
               "reason": "This is strong support"}
        captured = {}

        def fake_generate(**kwargs):
            captured["prompt"] = kwargs["messages"][-1]["content"]
            return json.dumps({
                "verdict": "Weak Support", "reasoning": "No activity history shown",
                "justification_claims": [],
                "justification_assessment": "Needs event history", "recommended_next_focus": "Check logs",
                "risk_of_wrong_interpretation": "Assuming status shows activity",
            })

        with patch.object(security_model, "_load_security_model", return_value=(object(), object())), \
             patch.object(security_model, "_device", return_value="cpu"), \
             patch.object(security_model, "_generate_security_response", side_effect=fake_generate):
            result = security_model.evaluate_action_evidence(action, evidence, {}, vlm, {})

        prompt = captured["prompt"]
        self.assertEqual(result["verdict"], "Weak Support")
        self.assertIn("Status: Inactive", prompt)
        self.assertIn("static credential/status page", prompt)
        self.assertIn("How to choose the verdict", prompt)
        for leaked in ("choice_role", "supportRole", "Expected weak evidence",
                       "supports_selected_action", "security_relevance", "This is strong support"):
            self.assertNotIn(leaked, prompt)


    def test_judge_sees_learner_visible_context_as_context_only(self):
        import tempfile
        from pathlib import Path
        from cloudir.ai_models.run_turn import learner_visible_context

        with tempfile.TemporaryDirectory() as directory:
            turn_dir = Path(directory) / "data" / "runtime" / "turns" / "turn_2"
            turn_dir.mkdir(parents=True)
            (turn_dir / "turn_config.json").write_text(json.dumps({
                "briefing": "Activity by user/j.moreno from 198.51.100.7 needs review.",
                "known_context": ["j.moreno signed in without MFA."], "coach_guidance": "not passed"}))
            context = learner_visible_context(Path(directory), {"currentTurn": 2})
            self.assertEqual(context["known_context"], ["j.moreno signed in without MFA."])
            self.assertNotIn("coach_guidance", context)
            self.assertEqual(learner_visible_context(Path(directory), {"currentTurn": 3}), {})

        captured = {}

        def fake_generate(**kwargs):
            captured["prompt"] = kwargs["messages"][-1]["content"]
            return json.dumps({"verdict": "Weak Support", "reasoning": "Unrelated principal",
                               "justification_claims": [], "recommended_next_focus": "Check j.moreno",
                               "risk_of_wrong_interpretation": "Treating routine work as incident activity"})

        with patch.object(security_model, "_load_security_model", return_value=(object(), object())), \
             patch.object(security_model, "_device", return_value="cpu"), \
             patch.object(security_model, "_generate_security_response", side_effect=fake_generate):
            security_model.evaluate_action_evidence({"title": "Correlate activity"},
                                                    {"id": "iam_principal_activity", "type": "iam_activity"}, {},
                                                    {"visible_facts_extracted": ["Principal: role/ops", "DescribeInstances"]},
                                                    {}, incident_context=context)

        self.assertIn("Incident context shown to the learner", captured["prompt"])
        self.assertIn("198.51.100.7", captured["prompt"])
        self.assertIn("NOT evidence", captured["prompt"])


class MarkingSchemeJudgementTests(unittest.TestCase):
    """The judge grades against the turn's marking scheme and copies the facts it relies on."""

    SCHEME = {"strong_support": "Shows the GuardDuty finding for j.moreno.",
              "partial_support": "Calls by j.moreno from 203.0.113.41 on 2023-09-30.",
              "weak_support": "Access key owned by billing-sync."}
    VLM = {"visible_facts_extracted": ["Owner: arn:aws:iam::123456789012:role/billing-sync", "Status: Active"]}

    def judge(self, *replies):
        prompts = []
        answers = iter(replies)

        def fake_generate(**kwargs):
            prompts.append(list(kwargs["messages"]))  # a snapshot: retries append to the same list
            return json.dumps({"reasoning": "Key page for another principal.", "justification_claims": [],
                               "recommended_next_focus": "Check the finding", "risk_of_wrong_interpretation": "None",
                               **next(answers)})

        with patch.object(security_model, "_load_security_model", return_value=(object(), object())), \
             patch.object(security_model, "_device", return_value="cpu"), \
             patch.object(security_model, "_generate_security_response", side_effect=fake_generate):
            result = security_model.evaluate_action_evidence(
                {"title": "Check GuardDuty Finding"}, {"id": "access_key_details", "type": "access_key"}, {},
                self.VLM, {}, marking_scheme=self.SCHEME)
        return result, prompts

    def test_the_marking_scheme_reaches_the_judge_and_copied_facts_are_kept(self):
        result, prompts = self.judge({"whose_evidence": "Owner: role/billing-sync", "key_fact": "Status: Active",
                                      "verdict": "Weak Support"})

        prompt = prompts[0][-1]["content"]
        self.assertIn("Marking scheme for this turn", prompt)
        self.assertIn("Access key owned by billing-sync.", prompt)
        self.assertIn('"whose_evidence"', prompt)
        self.assertEqual(result["verdict"], "Weak Support")
        self.assertEqual(result["whose_evidence"], "Owner: role/billing-sync")

    def test_the_judge_grades_the_evidence_without_seeing_the_action(self):
        # With a wrong action the judge graded fit to the action, not the scheme (26 Sep).
        _, prompts = self.judge({"whose_evidence": "Owner: role/billing-sync", "key_fact": "Status: Active",
                                 "verdict": "Weak Support"})

        prompt = prompts[0][-1]["content"]
        self.assertNotIn("Check GuardDuty Finding", prompt)
        self.assertNotIn("Learner selected action", prompt)
        self.assertIn("against the marking scheme", prompt)

    def test_an_owner_the_screenshot_does_not_show_is_sent_back(self):
        # The turn 1 failure: the judge called billing-sync's key j.moreno's and graded it Strong.
        result, prompts = self.judge(
            {"whose_evidence": "Owner: user/j.moreno", "key_fact": "Status: Active", "verdict": "Strong Support"},
            {"whose_evidence": "Owner: role/billing-sync", "key_fact": "Status: Active", "verdict": "Weak Support"})

        self.assertIn("details the observed screenshot facts do not show (user/j.moreno)", prompts[1][-1]["content"])
        self.assertEqual(result["verdict"], "Weak Support")

    def test_the_judge_has_only_the_three_evidence_levels(self):
        result, prompts = self.judge(
            {"whose_evidence": "Owner: role/billing-sync", "key_fact": "Status: Active", "verdict": "Unsupported"},
            {"whose_evidence": "Owner: role/billing-sync", "key_fact": "Status: Active", "verdict": "Weak Support"})

        self.assertIn('"verdict": "Strong Support | Partial Support | Weak Support"', prompts[0][-1]["content"])
        self.assertIn("choose Strong Support, Partial Support or Weak Support", prompts[1][-1]["content"])
        self.assertEqual(result["verdict"], "Weak Support")

    def test_a_correct_copy_wrapped_in_a_sentence_is_accepted(self):
        # Live turn 3: right owner and verdict, but written as a sentence; the check
        # rejected "the" and "shown" twice and the learner got no verdict.
        result, prompts = self.judge({
            "whose_evidence": "The owner shown in the evidence is 'arn:aws:iam::123456789012:role/billing-sync'.",
            "key_fact": "Status: Active", "verdict": "Weak Support"})

        self.assertEqual(len(prompts), 1)
        self.assertIn("billing-sync", result["whose_evidence"])

    def test_grounding_failures_say_why(self):
        with self.assertRaisesRegex(ValueError, "Last problem: .*j.moreno"):
            self.judge({"whose_evidence": "Owner: user/j.moreno", "key_fact": "Status: Active", "verdict": "Strong Support"},
                       {"whose_evidence": "Owner: user/j.moreno", "key_fact": "Status: Active", "verdict": "Strong Support"})

    def test_weak_support_may_state_what_is_missing(self):
        result, prompts = self.judge({"whose_evidence": "Owner: role/billing-sync",
                                      "key_fact": "no GuardDuty finding shown", "verdict": "Weak Support"})

        self.assertEqual(len(prompts), 1)  # accepted first time, not sent back
        self.assertEqual(result["verdict"], "Weak Support")
        self.assertEqual(result["key_fact"], "none")

    def test_strong_or_partial_support_must_copy_a_key_fact(self):
        with self.assertRaisesRegex(ValueError, "failed grounding checks"):
            self.judge({"whose_evidence": "none shown", "key_fact": "none", "verdict": "Strong Support"},
                       {"whose_evidence": "none shown", "key_fact": "none", "verdict": "Partial Support"})
    def test_a_page_with_no_owner_is_told_to_write_none_shown(self):
        # Live cost-management turn 3: a billing summary names nobody; the judge wrote
        # the suspect from the incident context twice and the turn got no verdict.
        result, prompts = self.judge(
            {"whose_evidence": "The billing activity is associated with the customer-api user.",
             "key_fact": "Status: Active", "verdict": "Strong Support"},
            {"whose_evidence": "none shown", "key_fact": "Status: Active", "verdict": "Strong Support"})

        self.assertIn('whose_evidence is "none shown"', prompts[0][-1]["content"])
        self.assertIn('write "none shown"', prompts[1][-1]["content"])
        self.assertEqual(result["verdict"], "Strong Support")

if __name__ == "__main__":
    unittest.main()
