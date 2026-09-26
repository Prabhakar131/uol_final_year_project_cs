import json
import unittest
from unittest.mock import patch

from cloudir.ai_models import coach_model, security_model
from cloudir.ai_models.learner_grounding import (
    LearnerGroundingError, ground_justification, learner_statements,
    validate_evidence_only_prose, keep_evidence_only_prose,
)


TRANSCRIPT = "I will check the visible identity and source IP. I would compare this with another source."
JUSTIFICATION = {"transcript": TRANSCRIPT, "prompt": "Acknowledge truncation and explain clipped fields."}
VLM = {"visible_facts_extracted": ["StopLogging by admin-test"], "evidence_type": "cloudtrail"}
EVIDENCE = {"type": "cloudtrail", "title": "Audit event"}


def response():
    return {"verdict": "Strong Support", "reasoning": "The StopLogging event supports reviewing activity.",
            "justification_claims": [{"statement_id": 1, "assessment": "needs_detail"}],
            "recommended_next_focus": "Check the actor's activity history.",
            "risk_of_wrong_interpretation": "A single event does not prove malicious intent."}


class LearnerGroundingTests(unittest.TestCase):
    def test_prompt_is_never_a_statement_and_empty_canonical_text_wins(self):
        self.assertEqual(len(learner_statements(JUSTIFICATION)), 2)
        self.assertNotIn("truncation", str(learner_statements(JUSTIFICATION)))
        self.assertEqual(learner_statements({"transcript": "", "text": TRANSCRIPT, "prompt": "I noticed clipping"}), [])
        self.assertEqual(learner_statements({"text": TRANSCRIPT})[0]["text"], TRANSCRIPT.split('. ')[0] + '.')

    def test_statements_preserve_negation_and_exact_qualifications(self):
        text = "I did not acknowledge truncation. I cannot identify the full IP 185.220.101.42."
        claims, rendered = ground_justification([{"statement_id": 1, "assessment": "supported"}],
                                                learner_statements({"transcript": text}))
        self.assertEqual(claims[0]["transcript_quote"], "I did not acknowledge truncation.")
        self.assertIn("I did not acknowledge truncation.", rendered)

    def test_fabricated_quotes_ids_assessments_and_extra_prose_rejected(self):
        for claims in ([{"statement_id": 9, "assessment": "supported"}],
                       [{"statement_id": True, "assessment": "supported"}],
                       [{"statement_id": 1, "assessment": "acknowledged truncation"}],
                       [{"statement_id": 1, "assessment": "supported", "quote": "I noticed truncation"}],
                       [{"statement_id": 1, "assessment": "supported"}] * 2,
                       "The learner acknowledges truncation"):
            with self.subTest(claims=claims), self.assertRaises(LearnerGroundingError):
                ground_justification(claims, learner_statements(JUSTIFICATION))

    def test_empty_or_unassessed_transcript_gets_no_invented_credit(self):
        self.assertIn("No learner justification", ground_justification([], [])[1])
        self.assertIn("No specific reasoning is credited", ground_justification([], learner_statements(JUSTIFICATION))[1])
        with self.assertRaises(LearnerGroundingError):
            ground_justification([{"statement_id": 1, "assessment": "supported"}], [])

    def security(self, outputs, justification=JUSTIFICATION):
        with patch.object(security_model, "_load_security_model", return_value=(object(), object())), \
             patch.object(security_model, "_device", return_value="cpu"), \
             patch.object(security_model, "_generate_security_response", side_effect=[json.dumps(x) for x in outputs]) as generate:
            result = security_model.evaluate_action_evidence({}, EVIDENCE, justification, VLM, {})
        return result, generate

    def test_free_form_assessment_is_discarded_not_credited(self):
        original = response()
        original["justification_assessment"] = "The learner acknowledges the truncation."
        result, _ = self.security([original])
        self.assertNotIn("truncation", result["justification_assessment"])
        self.assertEqual(result["justification_claims"][0]["transcript_quote"], learner_statements(JUSTIFICATION)[0]["text"])
        self.assertEqual(result["verdict"], "Strong Support")

    def test_invalid_attribution_retries_then_returns_grounded_feedback(self):
        invalid = {**response(), "reasoning": "The learner correctly acknowledges truncation."}
        result, generate = self.security([invalid, response()])
        self.assertEqual(generate.call_count, 2)
        self.assertNotIn("acknowledges", result["reasoning"])

    def test_persistent_invalid_statement_stops_evaluation(self):
        invalid = {**response(), "justification_claims": [{"statement_id": 99, "assessment": "supported"}]}
        with self.assertRaisesRegex(ValueError, "after retry"):
            self.security([invalid, invalid])

    def test_empty_answer_drops_invented_claims_instead_of_failing(self):
        # 27 Sep: an empty answer failed both attempts because the model copied the
        # schema's example claim about statement 1.
        result, generate = self.security([response()], justification={"transcript": "", "prompt": "Explain why."})
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(result["justification_claims"], [])
        self.assertIn("No learner justification", result["justification_assessment"])
        self.assertEqual(result["verdict"], "Strong Support")
        prompt = generate.call_args.kwargs["messages"][-1]["content"]
        self.assertIn('"justification_claims": [],', prompt)
        self.assertNotIn('"statement_id": 1', prompt)

    def test_coach_receives_no_transcript_prompt_or_old_assessment(self):
        security = {**response(), "justification_assessment": "The learner acknowledges truncation."}
        valid = {"feedback": "The event supports activity review; intent remains unproven.",
                 "next_turn_guidance": "Which audit record would clarify the actor's intent?"}
        with patch.object(coach_model, "_load_coach_model", return_value=(object(), object())), \
             patch.object(coach_model, "_device", return_value="cpu"), \
             patch.object(coach_model, "_generate_feedback_response", return_value=json.dumps(valid)) as generate:
            result = coach_model.generate_coach_feedback({}, EVIDENCE, JUSTIFICATION, VLM, security, {})
        prompt = generate.call_args.args[-1][-1]["content"]
        for forbidden in (TRANSCRIPT, JUSTIFICATION["prompt"], security["justification_assessment"]):
            self.assertNotIn(forbidden, prompt)
        self.assertEqual({key: result[key] for key in valid}, valid)

    def test_coach_praise_retried_and_persistent_praise_blocked(self):
        invalid = {"feedback": "You correctly identified truncation.", "next_turn_guidance": "Check logs."}
        valid = {"feedback": "The event supports review.", "next_turn_guidance": "Which log would clarify scope?"}
        for outputs, raises in (([invalid, valid], False), ([invalid, invalid], True)):
            with patch.object(coach_model, "_load_coach_model", return_value=(object(), object())), \
                 patch.object(coach_model, "_device", return_value="cpu"), \
                 patch.object(coach_model, "_generate_feedback_response", side_effect=[json.dumps(x) for x in outputs]) as generate:
                if raises:
                    with self.assertRaisesRegex(ValueError, "after retry"):
                        coach_model.generate_coach_feedback({}, EVIDENCE, JUSTIFICATION, VLM, response(), {})
                else:
                    result = coach_model.generate_coach_feedback({}, EVIDENCE, JUSTIFICATION, VLM, response(), {})
                    self.assertEqual({key: result[key] for key in valid}, valid)
                self.assertEqual(generate.call_count, 2)

    def test_tripwire_catches_original_false_credit_and_implicit_praise(self):
        for text in ("The learner acknowledges the truncation.", "You were right to notice clipping.",
                     "Correctly identified the truncated IP.", "Great work!"):
            with self.subTest(text=text), self.assertRaises(LearnerGroundingError):
                validate_evidence_only_prose({"feedback": text}, ("feedback",))

    def test_personal_credit_is_removed_without_discarding_evidence_analysis(self):
        result = {"reasoning": "The event supports review. The learner acknowledges truncation.",
                  "recommended_next_focus": "The learner should inspect the actor's audit history.",
                  "risk_of_wrong_interpretation": "The learner might assume malicious intent."}
        audit = keep_evidence_only_prose(result, tuple(result))
        self.assertEqual(result["reasoning"], "The event supports review.")
        self.assertEqual(result["recommended_next_focus"], "Inspect the actor's audit history.")
        self.assertEqual(result["risk_of_wrong_interpretation"], "A possible mistake is to assume malicious intent.")
        self.assertEqual(audit["removed_attribution_fields"], ["reasoning"])

    def test_actual_claim_is_never_rewritten_into_future_advice(self):
        with self.assertRaises(LearnerGroundingError):
            keep_evidence_only_prose({"feedback": "The learner correctly acknowledged truncation."}, ("feedback",))


if __name__ == "__main__":
    unittest.main()
