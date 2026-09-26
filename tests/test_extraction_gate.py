import unittest
import json
from pathlib import Path
from unittest.mock import patch

from cloudir.ai_models import run_turn, security_model
from cloudir.ai_models.extraction_quality import (
    ExtractionQualityError, security_observations, validate_extraction,
)


def observed():
    return {"evidence_type": "cloudwatch", "matched_records": 5,
            "event_rows": [{"timestamp": "2023-10-01T10:21:35Z",
                            "log_stream": "audit", "message": "StopLogging by user alice...",
                            "truncated": True}], "query_text": "search unauthorized access",
            "time_range": "2023-10-01 10:10-10:30 UTC", "log_group": "audit",
            "extraction_warnings": ["Message is clipped"]}


EVIDENCE = {"template": "cloudwatch", "imagePath": "unused.png"}


class ExtractionGateTests(unittest.TestCase):
    def test_visible_rows_need_not_equal_badge_and_clipping_is_allowed(self):
        validate_extraction(observed(), EVIDENCE)

    def test_missing_rows_and_unreadable_messages_are_blocked(self):
        for change in ({"event_rows": []}, {"event_rows": None},
                       {"event_rows": [{"message": None, "truncated": False}]},
                       {"matched_records": 0}):
            with self.subTest(change=change), self.assertRaises(ExtractionQualityError):
                validate_extraction({**observed(), **change}, EVIDENCE)

    def test_readable_zero_results_are_not_an_extraction_failure(self):
        validate_extraction({**observed(), "event_rows": [], "matched_records": 0}, EVIDENCE)

    def test_dropped_timestamp_digit_is_rejected_without_inventing_a_correction(self):
        facts = observed()
        facts["event_rows"][0]["timestamp"] = "2023-10-01T10:12:0Z"
        with self.assertRaisesRegex(ExtractionQualityError, "malformed ISO timestamp"):
            validate_extraction(facts, EVIDENCE)
        self.assertEqual(facts["event_rows"][0]["timestamp"], "2023-10-01T10:12:0Z")

    def test_query_claims_and_summaries_are_separate_from_observations(self):
        result = security_observations({**observed(), "visible_evidence_summary": "confirmed breach",
                                        "visible_facts_extracted": ["confirmed breach"]}, EVIDENCE)
        self.assertNotIn("confirmed breach", str(result))
        self.assertNotIn("search unauthorized", str(result["event_rows"]))
        self.assertIn("query_text", result["query_metadata_not_event_evidence"])

    def test_direct_security_call_is_guarded_before_loading_model(self):
        with patch.object(security_model, "_load_security_model") as loader:
            with self.assertRaises(ExtractionQualityError):
                security_model.evaluate_action_evidence({}, EVIDENCE, {}, {"event_rows": []}, {})
            loader.assert_not_called()

    def pipeline(self, outputs):
        with patch.object(run_turn, "analyse_evidence_image", side_effect=outputs) as extract, \
             patch.object(run_turn, "evaluate_action_evidence", return_value={"verdict": "Strong Support"}) as judge, \
             patch.object(run_turn, "generate_coach_feedback", return_value={}) as coach, \
             patch.object(run_turn, "unload_image_model"), \
             patch.object(run_turn, "unload_security_model"), \
             patch.object(run_turn, "unload_coach_model"), \
             patch.object(run_turn, "unload_all_models"):
            events = list(run_turn.stream_ai_evaluation_turn({}, EVIDENCE, {}, {}, Path('.')))
        return events, extract, judge, coach

    def test_one_retry_then_error_without_judge_or_coach(self):
        events, extract, judge, coach = self.pipeline([{"event_rows": []}, {"event_rows": []}])
        self.assertEqual([e["stage"] for e in events], ["extraction_retry", "error"])
        self.assertEqual(extract.call_count, 2)
        self.assertTrue(extract.call_args.kwargs["retry_extraction"])
        judge.assert_not_called()
        coach.assert_not_called()
        self.assertIn("No verdict or progress", events[-1]["output"]["error"])

    def test_retry_recovers_and_only_valid_output_reaches_security(self):
        events, extract, judge, _ = self.pipeline([ValueError("bad JSON"), observed()])
        self.assertEqual([e["stage"] for e in events], ["extraction_retry", "vlm", "security", "coach", "done"])
        self.assertEqual(judge.call_args.kwargs["vlm_output"], observed())
        judge.assert_called_once()

    def test_security_verdict_requires_valid_citations(self):
        result = {"verdict": "Strong Support", "supporting_row_numbers": [1],
                  **{k: "explanation" for k in ("reasoning", "justification_assessment",
                                               "recommended_next_focus", "risk_of_wrong_interpretation")}}
        security_model._validate_evidence_verdict(result, observed())
        for refs in ([], [2], [True], "1"):
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                security_model._validate_evidence_verdict({**result, "supporting_row_numbers": refs}, observed())
        with self.assertRaises(ValueError):
            security_model._validate_evidence_verdict({**result, "verdict": "strong-ish"}, observed())

    def test_complete_rows_do_not_receive_truncation_or_credential_examples(self):
        facts = observed()
        facts["event_rows"][0].update(message="Complete event message", truncated=False)
        facts["extraction_warnings"] = []
        facts["visible_evidence_summary"] = "Untrusted summary claim"
        response = {"verdict": "Weak Support", "supporting_row_numbers": [],
                    "justification_claims": [],
                    **{k: "explanation" for k in ("reasoning", "justification_assessment",
                                                 "recommended_next_focus", "risk_of_wrong_interpretation")}}
        with patch.object(security_model, "_load_security_model", return_value=(object(), object())), \
             patch.object(security_model, "_device", return_value="cpu"), \
             patch.object(security_model, "_generate_security_response", return_value=json.dumps(response)) as generate:
            security_model.evaluate_action_evidence({}, EVIDENCE, {}, facts, {})
        prompt = generate.call_args.kwargs["messages"][-1]["content"]
        self.assertIn("No extracted event row is marked truncated", prompt)
        self.assertIn("query_metadata_not_event_evidence", prompt)
        self.assertNotIn("Untrusted summary claim", prompt)
        self.assertNotIn("A static credential/status page", prompt)
