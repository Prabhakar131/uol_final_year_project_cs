from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cloudir.scenario.json_boundary import to_camel_case_keys


VISIBILITY_GAIN = {"Strong Support": 15, "Partial Support": 8}
CONTAINMENT_GAIN = {"best": 10, "partial": 4, "weak": 0}
LEGACY_CONTAINMENT_GAIN = {"Strong Support": 10, "Partial Support": 4}


@dataclass
class ScenarioState:
    current_turn: int = 1
    max_turns: int = 5
    containment: int = 20
    visibility: int = 35
    risk: str = "High"
    phase: str = "Initial Triage"
    action_history: list[dict[str, Any]] = field(default_factory=list)
    pending_turn: int | None = None
    completed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return to_camel_case_keys(
            {
                "current_turn": self.current_turn,
                "max_turns": self.max_turns,
                "containment": self.containment,
                "visibility": self.visibility,
                "risk": self.risk,
                "phase": self.phase,
                "action_history": self.action_history,
                "pending_turn": self.pending_turn,
                "completed": self.completed,
            }
        )

    def has_recorded_turn(self, turn_number: int) -> bool:
        return any(entry.get("turn") == turn_number for entry in self.action_history)

    def normalise_action_history(self) -> None:
        seen_turns: set[int] = set()
        unique_entries: list[dict[str, Any]] = []

        for entry in self.action_history:
            turn = entry.get("turn")

            if not isinstance(turn, int) or turn in seen_turns:
                continue

            seen_turns.add(turn)
            unique_entries.append(entry)

        self.action_history = sorted(unique_entries, key=lambda item: item["turn"])

    def record_action(self, action_title: str, evidence_title: str, verdict: str,
                      action_role: str | None = None) -> None:
        """Visibility follows the evidence verdict; containment follows the action choice.

        Progress once came from the verdict alone, so in automated sessions a learner
        who always picked a wrong action gained 12.4 points a turn against 12.7 for
        the best action. The decision now has to be right to contain the incident,
        and the evidence has to be right to see it. Without a known action role
        (older runs) containment keeps following the verdict.
        """
        self.action_history.append(
            {
                "turn": self.current_turn,
                "action": action_title,
                "evidence": evidence_title,
                "verdict": verdict,
                "action_role": action_role,
            }
        )

        self.visibility = min(100, self.visibility + VISIBILITY_GAIN.get(verdict, 0))
        if action_role in CONTAINMENT_GAIN:
            self.containment = min(100, self.containment + CONTAINMENT_GAIN[action_role])
        else:
            self.containment = min(100, self.containment + LEGACY_CONTAINMENT_GAIN.get(verdict, 0))

        if self.current_turn == 1:
            self.phase = "Initial Triage"
        elif self.current_turn == 2:
            self.phase = "Focused Investigation"
        elif self.current_turn == 3:
            self.phase = "Scope Review"
        elif self.current_turn == 4:
            self.phase = "Containment Decision"
        elif self.current_turn == 5:
            self.phase = "Recovery Review"

        if self.visibility >= 60 and self.containment >= 40:
            self.risk = "Medium"

        if self.visibility >= 80 and self.containment >= 70:
            self.risk = "Reduced"

    def commit_pending_turn(self) -> None:
        if self.pending_turn is not None:
            self.current_turn = self.pending_turn
            self.pending_turn = None

    def reset(self) -> None:
        self.current_turn = 1
        self.max_turns = 5
        self.containment = 20
        self.visibility = 35
        self.risk = "High"
        self.phase = "Initial Triage"
        self.action_history.clear()
        self.pending_turn = None
        self.completed = False

    def apply_max_turns(self, max_turns: int | None) -> None:
        if isinstance(max_turns, int) and max_turns >= 1:
            self.max_turns = max_turns
