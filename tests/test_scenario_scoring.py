"""Progress rewards the decision as well as the evidence."""

import json
import tempfile
import unittest
from pathlib import Path

from cloudir.scenario.scenario_state import ScenarioState
from cloudir.services.turn_evaluation_service import authored_action_role


class ScoringTests(unittest.TestCase):
    def test_a_wrong_action_with_strong_evidence_gains_visibility_but_not_containment(self):
        # Automated sessions: wrong actions gained 12.4 points a turn against 12.7 for the best.
        best, wrong = ScenarioState(), ScenarioState()
        best.record_action("Check finding", "GuardDuty", "Strong Support", action_role="best")
        wrong.record_action("Rotate keys", "GuardDuty", "Strong Support", action_role="weak")

        self.assertEqual((best.visibility, best.containment), (50, 30))
        self.assertEqual((wrong.visibility, wrong.containment), (50, 20))

    def test_the_partial_action_earns_part_of_the_containment(self):
        state = ScenarioState()
        state.record_action("Review logs", "CloudTrail", "Weak Support", action_role="partial")
        self.assertEqual((state.visibility, state.containment), (35, 24))

    def test_an_unknown_action_role_keeps_the_old_verdict_rule(self):
        state = ScenarioState()
        state.record_action("Old run", "Evidence", "Partial Support")
        self.assertEqual((state.visibility, state.containment), (43, 24))

    def test_the_action_role_comes_from_the_turn_files_not_the_request(self):
        with tempfile.TemporaryDirectory() as directory:
            turn = Path(directory) / "data" / "runtime" / "turns" / "turn_2"
            turn.mkdir(parents=True)
            (turn / "actions.json").write_text(json.dumps([
                {"id": "rotate_keys", "title": "Rotate keys", "choice_role": "weak"},
                {"id": "check_finding", "title": "Check finding", "choice_role": "best"}]))

            claimed = {"id": "rotate_keys", "title": "Rotate keys", "choiceRole": "best"}
            self.assertEqual(authored_action_role(Path(directory), 2, claimed), "weak")
            self.assertEqual(authored_action_role(Path(directory), 2, {"title": "Check finding"}), "best")
            self.assertIsNone(authored_action_role(Path(directory), 3, claimed))


if __name__ == "__main__":
    unittest.main()
