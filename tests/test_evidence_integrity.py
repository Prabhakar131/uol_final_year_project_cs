import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw
from cloudir.evidence_core.cloudwatch_facts import normalise_cloudwatch_facts
from cloudir.evidence_core.fact_repair import normalise_evidence_facts, _repair_cloudwatch_facts
from cloudir.evidence.templates.cloudwatch_template import render_cloudwatch, _wrap_complete
from cloudir.evidence.templates.base_template import load_font, text_width
from cloudir.scenario.evidence_support_roles import normalise_evidence_support_roles
from cloudir.scenario.evidence_template_strategy import apply_evidence_template_strategy
from cloudir.scenario.turn_validation import validate_evidence_facts
from cloudir.ai_models.image_model import _CLOUDWATCH_CARD_MAX_PIXELS, _cloudwatch_images
from cloudir.ai_models.coach_model import _ensure_unique_evidence_templates


class EvidenceIntegrityTests(unittest.TestCase):
    def test_both_repair_paths_preserve_weak_rows_and_normal_signal(self):
        facts = {"log_rows": [["2023-10-01T10:00:00Z", "health", "public endpoint returned 200"]],
                 "alarm_state": "OK", "matched_records": "1"}
        evidence = {"id": "cloudwatch", "type": "cloudwatch", "template": "cloudwatch",
                    "support_role": "weak", "summary": "An unrelated health check", "facts": copy.deepcopy(facts)}
        normalise_evidence_facts(evidence)
        repaired = _repair_cloudwatch_facts(facts, "unauthorized IAM activity", evidence)
        for output in (evidence["facts"], repaired):
            self.assertEqual(output["log_rows"], facts["log_rows"])
            self.assertEqual(output["alarm_state"], "OK")
            self.assertNotIn("StopLogging", str(output))

    def test_missing_facts_do_not_become_incident_defaults(self):
        result = _repair_cloudwatch_facts({}, "StopLogging by admin-test", {"summary": "Unauthorized access"})
        self.assertEqual(result["log_rows"], [])
        self.assertEqual(result["time_range"], "Unknown")
        self.assertEqual(result["alarm_state"], "Unknown")

    def test_explicit_empty_rows_and_missing_values_are_preserved(self):
        self.assertEqual(normalise_cloudwatch_facts({"log_rows": [], "events": [["t", "s", "m"]]})["log_rows"], [])
        self.assertEqual(normalise_cloudwatch_facts({"log_rows": [["t", None, "message"]]})["log_rows"], [["t", "Unknown", "message"]])
        with self.assertRaises(ValueError):
            normalise_cloudwatch_facts({"log_rows": [["t", "s", "m"]], "matched_records": "0"})

    def test_roles_are_not_assigned_by_position_or_template(self):
        items = [{"template": "cloudwatch", "support_role": "weak"},
                 {"template": "cloudtrail", "support_role": "strong"},
                 {"template": "billing", "support_role": "partial"}]
        original = copy.deepcopy(items)
        apply_evidence_template_strategy({"evidence_facts": items}, {
            "evidence_template_strategy": {"role_templates_by_turn": {"1": {"strong": "cloudwatch", "weak": "cloudtrail"}}}
        }, 1)
        self.assertEqual(items, original)
        invalid = [{"support_role": "strong"}, {"support_role": "strong"}, {}]
        normalise_evidence_support_roles(invalid)
        self.assertEqual([x["support_role"] for x in invalid], ["strong", "strong", ""])
        with self.assertRaises(ValueError):
            validate_evidence_facts(invalid)

    def test_duplicate_templates_rejected_instead_of_replacing_content(self):
        turn = {"evidence_facts": [{"template": "cloudwatch", "facts": {"log_rows": []}},
                                   {"template": "cloudwatch", "facts": {"log_rows": []}}]}
        with self.assertRaises(ValueError):
            _ensure_unique_evidence_templates(turn, {}, 1)
        self.assertTrue(all(item["facts"] == {"log_rows": []} for item in turn["evidence_facts"]))

    def test_initial_turn_invalid_roles_are_rejected_before_any_files_are_saved(self):
        from cloudir.dataset_preparation import build_acse
        turn = {"turn_config": {}, "actions": [], "expected_outcomes": {},
                "evidence_facts": [{"support_role": "strong"}] * 3}
        with patch.object(build_acse, "_write_json") as write:
            with self.assertRaises(ValueError):
                build_acse._save_turn_1_files(turn)
            write.assert_not_called()

    def test_wrapping_never_elides_long_identifiers(self):
        draw = ImageDraw.Draw(Image.new("RGB", (1200, 760)))
        font = load_font(16)
        text = "event for " + "A" * 200 + " from 185.220.101.42"
        lines = _wrap_complete(draw, text, font, 565)
        self.assertEqual("".join(lines), text)
        self.assertTrue(all(text_width(draw, line, font) <= 565 for line in lines))

    def test_renderer_shows_all_rows_and_saves_only_layout_metadata(self):
        rows = [[f"2023-10-01T10:{i:02}:00Z", "app/health", f"health check number {i} returned 200"] for i in range(7)]
        captured = []
        original_text = ImageDraw.ImageDraw.text
        def capture(draw, xy, text, *args, **kwargs):
            captured.append(str(text))
            return original_text(draw, xy, text, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.png"
            with patch.object(ImageDraw.ImageDraw, "text", new=capture):
                render_cloudwatch({"facts": {"log_rows": rows, "matched_records": "9", "alarm_state": "OK"}}, path)
            joined = " ".join(captured)
            for row in rows:
                self.assertIn(row[2], joined)
            self.assertIn("Displayed rows: 7", joined)
            self.assertIn("Matched records: 9", joined)
            self.assertIn("Status: OK", joined)
            self.assertNotIn("Suspicious auth failures", joined)
            self.assertNotIn("Same principal/source IP", joined)
            with Image.open(path) as image:
                self.assertGreater(image.height, 760)
                self.assertEqual(set(image.info), {"cloudir_cloudwatch_detail_box"})
                x1, y1, x2, y2 = json.loads(image.info["cloudir_cloudwatch_detail_box"])
            self.assertLessEqual(y1, 226)  # Includes the matched-records line and query editor.
            images, _ = _cloudwatch_images(path)
            self.assertEqual(len(images), 1)
            view = images[0]["image"]
            # Enlarged, but capped so the VLM's attention fits in MPS memory.
            self.assertLessEqual(view.width * view.height, _CLOUDWATCH_CARD_MAX_PIXELS)
            self.assertAlmostEqual(view.width / view.height, (x2-x1) / (y2-y1), delta=0.02)
            self.assertGreater(view.width, x2 - x1)


if __name__ == "__main__":
    unittest.main()
