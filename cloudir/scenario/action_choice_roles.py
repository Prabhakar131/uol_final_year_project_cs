from __future__ import annotations

from typing import Any


CHOICE_ROLE_SEQUENCE = ("best", "partial", "weak")
CHOICE_ROLE_SET = set(CHOICE_ROLE_SEQUENCE)


def normalise_action_choice_roles(actions: Any) -> None:
    """
    Adds internal testing metadata to action choices.

    The role is hidden in learner mode and only surfaced in Test Mode so the
    project owner can verify the intended best/partial/weak path quickly.
    """

    if not isinstance(actions, list):
        return

    existing_roles = [
        normalise_choice_role(action.get("choice_role"))
        for action in actions
        if isinstance(action, dict)
    ]

    if len(actions) >= 3 and not CHOICE_ROLE_SET.issubset(set(existing_roles)):
        for index, action in enumerate(actions):
            if isinstance(action, dict):
                action["choice_role"] = CHOICE_ROLE_SEQUENCE[min(index, 2)]
        return

    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue

        role = normalise_choice_role(action.get("choice_role"))
        action["choice_role"] = role or CHOICE_ROLE_SEQUENCE[min(index, 2)]


def normalise_choice_role(value: Any) -> str:
    role = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")

    aliases = {
        "strong": "best",
        "correct": "best",
        "best_action": "best",
        "strong_support": "best",
        "reasonable": "partial",
        "partial_support": "partial",
        "distractor": "weak",
        "weak_support": "weak",
        "wrong": "weak",
    }

    role = aliases.get(role, role)

    return role if role in CHOICE_ROLE_SET else ""
