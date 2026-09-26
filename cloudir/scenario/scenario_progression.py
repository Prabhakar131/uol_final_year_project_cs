from __future__ import annotations

from typing import Any

from cloudir.evidence_core.evidence_text_helpers import clean_value
from cloudir.scenario.action_choice_roles import normalise_action_choice_roles
from cloudir.scenario.evidence_template_strategy import apply_evidence_template_strategy
from cloudir.scenario.evidence_support_roles import normalise_evidence_support_roles
from cloudir.scenario.text_similarity import (
    is_placeholder_text,
    join_sentences,
    normalise_comparison_text,
    texts_match,
    texts_too_similar,
)


def apply_scenario_progression_safety(
    generated_turn: dict[str, Any],
    scenario_config: dict[str, Any],
    completed_turn: dict[str, Any],
    security_output: dict[str, Any],
    next_turn_number: int,
    timeline_evidence: bool = False,
) -> None:
    """
    Applies a light, config-driven safety layer after the AI generates a turn.

    With timeline_evidence, the scenario-specific repairs are skipped: they swap
    in hardcoded context, actions and evidence about a fixed storyline, which
    would contradict the scenario's own incident timeline.

    This does not hardcode a CloudTrail-specific path.
    It only uses scenario_progression from scenario_config.json to reduce weak
    repeated turns while preserving AI-generated content.
    """

    scenario_progression = scenario_config.get("scenario_progression", {})
    scenario_id = clean_value(scenario_config.get("scenario_id")).lower().replace("_", "-")

    if not isinstance(scenario_progression, dict):
        return

    turn_config = generated_turn.get("turn_config")

    if not isinstance(turn_config, dict):
        return

    target_stage = get_progression_stage(
        scenario_progression=scenario_progression,
        next_turn_number=next_turn_number,
    )

    if not target_stage:
        return

    completed_turn_config = get_completed_turn_config(completed_turn)
    previous_phase = clean_value(completed_turn_config.get("phase"))
    previous_briefing = clean_value(completed_turn_config.get("briefing"))

    generated_phase = clean_value(turn_config.get("phase"))
    generated_briefing = clean_value(turn_config.get("briefing"))
    generated_guidance = clean_value(turn_config.get("coach_guidance"))

    target_stage_name = format_stage_name(clean_value(target_stage.get("stage")))
    target_stage_purpose = clean_value(
        target_stage.get("purpose")
        or target_stage.get("goal")
    )

    verdict = clean_value(security_output.get("verdict")).lower()

    learner_needs_refocus = (
        "weak" in verdict
        or "unsupported" in verdict
    )

    learner_can_advance = (
        next_turn_number == 1
        or "strong" in verdict
        or "partial" in verdict
        or not verdict
        or verdict == "unknown"
    )

    phase_is_placeholder = is_placeholder_text(generated_phase)
    phase_repeats_previous = texts_match(generated_phase, previous_phase)

    if learner_can_advance and target_stage_name:
        if phase_is_placeholder or phase_repeats_previous:
            turn_config["phase"] = target_stage_name

    elif learner_needs_refocus:
        if phase_is_placeholder:
            turn_config["phase"] = (
                f"Refocused {previous_phase}"
                if previous_phase
                else "Refocused Investigation"
            )

    briefing_repeats_previous = texts_too_similar(
        generated_briefing,
        previous_briefing,
    )

    if target_stage_name and target_stage_purpose:
        progression_sentence = (
            f"This turn should focus on {target_stage_name}: {target_stage_purpose}"
        )

        if learner_can_advance and (
            is_placeholder_text(generated_briefing)
            or briefing_repeats_previous
        ):
            turn_config["briefing"] = join_sentences(
                progression_sentence,
                generated_briefing,
            )

        elif learner_needs_refocus and briefing_repeats_previous:
            refocus_sentence = (
                "The previous evidence choice did not fully resolve the investigation path, "
                "so this turn should refocus the learner on the missing evidence before moving forward."
            )
            turn_config["briefing"] = join_sentences(
                refocus_sentence,
                generated_briefing,
            )

    if is_placeholder_text(generated_guidance):
        if learner_needs_refocus:
            turn_config["coach_guidance"] = (
                "Review the previous uncertainty and choose evidence that directly supports "
                "or corrects the investigation path."
            )
        elif target_stage_purpose:
            turn_config["coach_guidance"] = (
                f"Choose the action and evidence that best support this stage: {target_stage_purpose}"
            )

    if scenario_progression.get("must_not_repeat_previous_action_focus"):
        mark_repeated_titles_as_follow_up(
            generated_items=generated_turn.get("actions"),
            previous_titles=extract_titles(completed_turn, "actions"),
        )

    if scenario_progression.get("must_not_repeat_previous_evidence_focus"):
        previous_evidence_titles = extract_titles(completed_turn, "evidence_facts")

        if not previous_evidence_titles:
            previous_evidence_titles = extract_titles(completed_turn, "evidenceFacts")

        mark_repeated_titles_as_follow_up(
            generated_items=generated_turn.get("evidence_facts"),
            previous_titles=previous_evidence_titles,
        )

    if timeline_evidence:
        normalise_action_choice_roles(generated_turn.get("actions"))
        normalise_evidence_support_roles(generated_turn.get("evidence_facts"))
        return

    repair_bare_turn_context(
        turn_config=turn_config,
        scenario_id=scenario_id,
        security_output=security_output,
        next_turn_number=next_turn_number,
    )

    repair_bare_turn_actions(
        generated_turn=generated_turn,
        scenario_id=scenario_id,
        next_turn_number=next_turn_number,
    )
    normalise_action_choice_roles(generated_turn.get("actions"))

    if scenario_id == "identity-management":
        repair_turn2_correlation_evidence(generated_turn)
        repair_turn3_response_evidence(generated_turn)
        repair_turn4_containment_evidence(generated_turn)
        repair_turn5_recovery_evidence(generated_turn)
    elif scenario_id == "automated-security-response":
        repair_automated_security_response_turn(generated_turn, next_turn_number)

    normalise_evidence_support_roles(generated_turn.get("evidence_facts"))
    apply_evidence_template_strategy(
        generated_turn=generated_turn,
        scenario_config=scenario_config,
        turn_number=next_turn_number,
    )


def get_progression_stage(
    scenario_progression: dict[str, Any],
    next_turn_number: int,
) -> dict[str, Any]:
    stages = scenario_progression.get("incident_response_stages", [])

    if not isinstance(stages, list) or not stages:
        return {}

    stage_index = min(max(next_turn_number - 1, 0), len(stages) - 1)
    stage = stages[stage_index]

    return stage if isinstance(stage, dict) else {}


def format_stage_name(stage: str) -> str:
    if not stage:
        return ""

    return stage.replace("_", " ").title()


def repair_bare_turn_context(
    turn_config: dict[str, Any],
    scenario_id: str,
    security_output: dict[str, Any],
    next_turn_number: int,
) -> None:
    """
    Keeps learner-facing turn context from becoming too thin.

    The local model sometimes returns structurally valid JSON with only one
    generic incident fact. This repairs that presentation layer without changing
    the underlying evidence choices.
    """

    if next_turn_number not in {2, 3, 4, 5}:
        return

    briefing = clean_value(turn_config.get("briefing"))
    known_context = turn_config.get("known_context")
    guidance = clean_value(turn_config.get("coach_guidance"))

    if not turn_context_is_bare(briefing, known_context, guidance):
        return

    verdict = clean_value(security_output.get("verdict")) or "not yet final"

    if scenario_id == "automated-security-response":
        repair_automated_security_response_context(
            turn_config=turn_config,
            next_turn_number=next_turn_number,
            verdict=verdict,
        )
        return

    if scenario_id != "identity-management":
        return

    if next_turn_number == 2:
        turn_config["phase"] = "Correlation and Scope"
        turn_config["briefing"] = (
            "The first audit event confirmed suspicious identity activity, but the scope is still unclear. "
            "Correlate IAM activity, credential changes, and monitoring logs to determine whether the same "
            "principal and source IP appear across multiple risky actions."
        )
        turn_config["known_context"] = [
            "A ConsoleLogin without MFA was observed for arn:aws:iam::123456789012:user/admin-test from 185.220.101.42.",
            "Related activity in the same incident window may include logging, credential, or role-management events.",
            f"The previous evidence was assessed as {verdict}, so this turn should test whether the signal has wider impact.",
        ]
        turn_config["coach_guidance"] = (
            "Choose evidence that links the same identity and source IP across IAM activity, access key changes, "
            "CloudWatch logs, or GuardDuty findings."
        )
        return

    if next_turn_number == 3:
        turn_config["phase"] = "Scope and Blast Radius"
        turn_config["briefing"] = (
            "The investigation has linked identity activity and credential risk. "
            "Now determine whether the same signal appears in threat-detection, logging, "
            "or impact evidence before choosing containment."
        )
        turn_config["known_context"] = [
            "Suspicious identity activity and credential activity have appeared in the same investigation window.",
            "The next step is to confirm whether monitoring or detection systems show wider incident scope.",
            f"The previous evidence was assessed as {verdict}, so this turn should separate confirmed scope from assumptions.",
        ]
        turn_config["coach_guidance"] = (
            "Look for evidence that confirms blast radius, affected principal, source IP, severity, or monitored impact."
        )
        return

    if next_turn_number == 4:
        turn_config["phase"] = "Containment and Remediation"
        turn_config["briefing"] = (
            "The incident scope is now clearer. Choose a containment action that is justified "
            "by direct credential, access, or monitoring evidence."
        )
        turn_config["known_context"] = [
            "Identity, credential, and monitoring evidence should now be considered together.",
            "Containment should focus on the affected principal or credentials rather than broad unrelated controls.",
            f"The previous evidence was assessed as {verdict}, so this turn should justify a proportionate remediation step.",
        ]
        turn_config["coach_guidance"] = (
            "Choose evidence that directly supports containment, such as active credential status, last-used details, or recovery logs."
        )
        return

    turn_config["phase"] = "Recovery and Final Debrief"
    turn_config["briefing"] = (
        "The main response decision has been made. Verify recovery, residual risk, and final reporting evidence "
        "before closing the incident path."
    )
    turn_config["known_context"] = [
        "The learner should now verify whether containment and visibility are restored.",
        "Final reporting should distinguish confirmed evidence from remaining uncertainty.",
        f"The previous evidence was assessed as {verdict}, so the final turn should close the loop with recovery evidence.",
    ]
    turn_config["coach_guidance"] = (
        "Choose the evidence that proves recovery or clearly documents residual risk for the final incident summary."
    )


def turn_context_is_bare(
    briefing: str,
    known_context: Any,
    guidance: str,
) -> bool:
    briefing_words = briefing.split()
    guidance_words = guidance.split()

    if len(briefing_words) < 22:
        return True

    if len(guidance_words) < 12:
        return True

    if not isinstance(known_context, list) or len(known_context) < 3:
        return True

    generic_context = {
        "unauthorized login attempt detected",
        "suspicious activity detected",
        "identity activity detected",
        "security alert detected",
    }

    concrete_items = 0

    for item in known_context:
        text = clean_value(item)
        normalised = normalise_comparison_text(text)

        if not text:
            continue

        if normalised in generic_context:
            continue

        if len(text.split()) >= 8:
            concrete_items += 1

    return concrete_items < 3


def repair_bare_turn_actions(
    generated_turn: dict[str, Any],
    scenario_id: str,
    next_turn_number: int,
) -> None:
    actions = generated_turn.get("actions")

    if not isinstance(actions, list):
        return

    titles = [
        normalise_comparison_text(clean_value(action.get("title")))
        for action in actions
        if isinstance(action, dict)
    ]

    bare_titles = {
        "threat analysis",
        "access logs review",
        "inspection of affected assets",
        "investigate affected assets",
        "review access logs",
        "check audit logs",
        "implement security measures",
    }

    if (
        len(actions) >= 3
        and not any(title in bare_titles for title in titles)
        and not turn_actions_need_stage_repair(actions, next_turn_number)
    ):
        return

    if scenario_id == "automated-security-response":
        repair_automated_security_response_actions(generated_turn, next_turn_number)
        return

    if scenario_id != "identity-management":
        return

    if next_turn_number == 2:
        generated_turn["actions"] = [
            {
                "id": "correlate_identity_timeline",
                "title": "Correlate Identity Timeline",
                "description": "Compare IAM activity around the suspicious login to see whether the same principal and source IP repeat.",
                "recommended_next_focus": "Look for login, logging, credential, and caller-identity events in the same incident window.",
                "choice_role": "best",
            },
            {
                "id": "review_access_key_risk",
                "title": "Review Access Key Risk",
                "description": "Check whether a new or recently used access key appears after the suspicious identity activity.",
                "recommended_next_focus": "Confirm whether credential creation or usage increases the containment priority.",
                "choice_role": "partial",
            },
            {
                "id": "query_security_logs",
                "title": "Query Security Logs",
                "description": "Use CloudWatch or GuardDuty signals to validate whether the identity activity appears across monitoring sources.",
                "recommended_next_focus": "Use monitoring evidence to support or challenge the CloudTrail and IAM timeline.",
                "choice_role": "weak",
            },
        ]
        return

    if next_turn_number == 3:
        generated_turn["actions"] = [
            {
                "id": "confirm_threat_detection_scope",
                "title": "Confirm Detection Scope",
                "description": "Check whether threat-detection evidence confirms the same principal, source IP, and incident window.",
                "recommended_next_focus": "Use detection evidence to decide whether the incident is isolated or broader.",
                "choice_role": "best",
            },
            {
                "id": "review_logging_gap_scope",
                "title": "Review Logging Gap Scope",
                "description": "Check whether audit logging gaps affect confidence in the incident timeline.",
                "recommended_next_focus": "Use CloudTrail context to understand visibility gaps before containment.",
                "choice_role": "partial",
            },
            {
                "id": "review_business_impact",
                "title": "Review Business Impact",
                "description": "Assess whether the incident caused cost or operational impact.",
                "recommended_next_focus": "Keep impact review separate from proof of identity compromise.",
                "choice_role": "weak",
            },
        ]
        return

    if next_turn_number == 4:
        generated_turn["actions"] = [
            {
                "id": "contain_suspicious_credentials",
                "title": "Contain Suspicious Credentials",
                "description": "Disable or rotate credentials tied to the confirmed suspicious principal.",
                "recommended_next_focus": "Use active key status and last-used activity to justify containment.",
                "choice_role": "best",
            },
            {
                "id": "increase_monitoring_controls",
                "title": "Increase Monitoring Controls",
                "description": "Add targeted monitoring while containment is being prepared.",
                "recommended_next_focus": "Use logs to support monitoring, but confirm whether credentials still require action.",
                "choice_role": "partial",
            },
            {
                "id": "review_cost_impact",
                "title": "Review Cost Impact",
                "description": "Check billing impact before changing credentials.",
                "recommended_next_focus": "Use cost evidence only as context unless it links to the confirmed principal.",
                "choice_role": "weak",
            },
        ]
        return

    if next_turn_number == 5:
        generated_turn["actions"] = [
            {
                "id": "verify_recovery_and_reporting",
                "title": "Verify Recovery and Reporting",
                "description": "Confirm that audit visibility and credential containment are reflected in final evidence.",
                "recommended_next_focus": "Use recovery logs or final audit records to close the incident cleanly.",
                "choice_role": "best",
            },
            {
                "id": "summarise_identity_timeline",
                "title": "Summarise Identity Timeline",
                "description": "Prepare the final identity timeline for reporting.",
                "recommended_next_focus": "Use the timeline as support, but include recovery status before closure.",
                "choice_role": "partial",
            },
            {
                "id": "check_remaining_costs",
                "title": "Check Remaining Costs",
                "description": "Review whether any unusual spend remains after containment.",
                "recommended_next_focus": "Treat billing as residual impact context, not proof that recovery is complete.",
                "choice_role": "weak",
            },
        ]


def turn_actions_need_stage_repair(actions: list[Any], next_turn_number: int) -> bool:
    action_text = " ".join(
        " ".join(
            [
                clean_value(action.get("title")),
                clean_value(action.get("description")),
                clean_value(action.get("recommended_next_focus")),
            ]
        )
        for action in actions
        if isinstance(action, dict)
    ).lower()

    best_action_text = ""

    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue

        choice_role = clean_value(action.get("choice_role")).lower()

        if choice_role == "best" or (not choice_role and index == 0):
            best_action_text = " ".join(
                [
                    clean_value(action.get("title")),
                    clean_value(action.get("description")),
                    clean_value(action.get("recommended_next_focus")),
                ]
            ).lower()
            break

    cloudtrail_review_terms = (
        "review additional cloudtrail",
        "review recent cloudtrail",
        "inspect cloudtrail",
        "cloudtrail audit",
        "cloudtrail log",
        "additional cloudtrail logs",
        "recent cloudtrail logs",
        "stoplogging request",
    )

    if next_turn_number == 2:
        correlation_terms = (
            "correlate",
            "timeline",
            "scope",
            "access key",
            "credential",
            "cloudwatch",
            "guardduty",
            "same principal",
            "same source ip",
        )

        cloudtrail_is_best = any(term in best_action_text for term in cloudtrail_review_terms)
        no_correlation_focus = not any(term in action_text for term in correlation_terms)

        return cloudtrail_is_best or no_correlation_focus

    if next_turn_number == 3:
        scope_terms = (
            "scope",
            "blast",
            "guardduty",
            "severity",
            "finding",
            "monitoring",
            "affected",
            "same principal",
            "same source ip",
            "impact",
        )

        cloudtrail_is_best = any(term in best_action_text for term in cloudtrail_review_terms)
        no_scope_focus = not any(term in action_text for term in scope_terms)

        return cloudtrail_is_best or no_scope_focus

    if next_turn_number == 4:
        containment_terms = (
            "contain",
            "disable",
            "rotate",
            "restrict",
            "credential",
            "access key",
            "remediat",
            "active key",
            "last-used",
        )

        cloudtrail_is_best = any(term in best_action_text for term in cloudtrail_review_terms)
        no_containment_focus = not any(term in action_text for term in containment_terms)

        return cloudtrail_is_best or no_containment_focus

    if next_turn_number == 5:
        recovery_terms = (
            "recover",
            "restore",
            "verify",
            "final",
            "report",
            "debrief",
            "closure",
            "residual",
            "post-incident",
        )

        cloudtrail_is_best = any(term in best_action_text for term in cloudtrail_review_terms)
        no_recovery_focus = not any(term in action_text for term in recovery_terms)

        return cloudtrail_is_best or no_recovery_focus

    return False


def repair_automated_security_response_context(
    turn_config: dict[str, Any],
    next_turn_number: int,
    verdict: str,
) -> None:
    if next_turn_number == 2:
        turn_config["phase"] = "Automation Scope and Validation"
        turn_config["briefing"] = (
            "The remediation workflow was triggered, but the trust boundary is not settled. "
            "Correlate the Security Hub finding, EventBridge rule, Lambda validation logs, "
            "and Step Functions execution before deciding whether the automated response is safe."
        )
        turn_config["known_context"] = [
            "Security Hub forwarded a finding into EventBridge and started a remediation workflow.",
            "Validation Lambda logs should show whether finding fields were sanitised before workflow handoff.",
            f"The previous evidence was assessed as {verdict}, so this turn should test whether the automation path is trustworthy.",
        ]
        turn_config["coach_guidance"] = (
            "Choose evidence that connects the finding payload to the workflow execution and validation controls."
        )
        return

    if next_turn_number == 3:
        turn_config["phase"] = "Automation Scope Review"
        turn_config["briefing"] = (
            "The validation path has been checked, but the blast radius of the automated response "
            "still needs review. Compare workflow, logging, and cost or detection evidence before containment."
        )
        turn_config["known_context"] = [
            "Security Hub, EventBridge, Lambda, and Step Functions evidence should now be compared as one workflow.",
            "Scope evidence should show whether the automation affected one finding or a wider set of resources.",
            f"The previous evidence was assessed as {verdict}, so this turn should confirm blast radius before remediation.",
        ]
        turn_config["coach_guidance"] = (
            "Choose evidence that shows whether the automated response stayed within the intended workflow boundary."
        )
        return

    if next_turn_number == 4:
        turn_config["phase"] = "Containment and Remediation"
        turn_config["briefing"] = (
            "The automation scope is now clearer. Decide whether to restrict the remediation role, "
            "pause unsafe automation, or allow a controlled remediation path."
        )
        turn_config["known_context"] = [
            "The remediation role can affect production resources through Systems Manager actions.",
            "Containment should focus on workflow permissions and approval controls.",
            f"The previous evidence was assessed as {verdict}, so this turn should justify the safest remediation boundary.",
        ]
        turn_config["coach_guidance"] = (
            "Choose evidence that proves which workflow role or permission boundary should be changed."
        )
        return

    if next_turn_number == 5:
        turn_config["phase"] = "Recovery and Final Debrief"
        turn_config["briefing"] = (
            "The remediation decision has been made. Verify that workflow monitoring, audit logs, "
            "and final status evidence support closing the incident."
        )
        turn_config["known_context"] = [
            "The final turn should confirm whether the automated response is now constrained and observable.",
            "Recovery evidence should link monitoring status to the same finding or workflow execution.",
            f"The previous evidence was assessed as {verdict}, so this turn should document residual risk clearly.",
        ]
        turn_config["coach_guidance"] = (
            "Choose evidence that verifies recovery and supports the final incident debrief."
        )


def repair_automated_security_response_actions(
    generated_turn: dict[str, Any],
    next_turn_number: int,
) -> None:
    if next_turn_number == 2:
        generated_turn["actions"] = [
            {
                "id": "review_lambda_validation_path",
                "title": "Review Lambda Validation Path",
                "description": "Check the validation Lambda and Step Functions logs before trusting the remediation path.",
                "recommended_next_focus": "Confirm whether the finding payload was validated before workflow handoff.",
                "choice_role": "best",
            },
            {
                "id": "correlate_finding_workflow_execution",
                "title": "Correlate Finding to Workflow",
                "description": "Compare Security Hub, EventBridge, and Step Functions records for the same finding ID.",
                "recommended_next_focus": "Confirm whether one finding caused one expected workflow execution.",
                "choice_role": "partial",
            },
            {
                "id": "inspect_remediation_role_scope",
                "title": "Inspect Remediation Role Scope",
                "description": "Review the remediation role permissions that would be used after workflow approval.",
                "recommended_next_focus": "Estimate blast radius if the automation executes the wrong remediation.",
                "choice_role": "weak",
            },
        ]
        return

    if next_turn_number == 3:
        generated_turn["actions"] = [
            {
                "id": "confirm_workflow_blast_radius",
                "title": "Confirm Workflow Blast Radius",
                "description": "Check whether one finding caused only the expected workflow execution and resource changes.",
                "recommended_next_focus": "Use workflow and monitoring evidence to confirm the automation scope.",
                "choice_role": "best",
            },
            {
                "id": "review_remediation_role_usage",
                "title": "Review Remediation Role Usage",
                "description": "Inspect the remediation role session used by the workflow.",
                "recommended_next_focus": "Use role evidence as context before choosing the containment boundary.",
                "choice_role": "partial",
            },
            {
                "id": "check_billing_side_effects",
                "title": "Check Billing Side Effects",
                "description": "Review billing for possible automation side effects.",
                "recommended_next_focus": "Use billing only as impact context unless it links to the workflow execution.",
                "choice_role": "weak",
            },
        ]
        return

    if next_turn_number == 4:
        generated_turn["actions"] = [
            {
                "id": "restrict_remediation_role_before_resume",
                "title": "Restrict Role Before Resume",
                "description": "Reduce remediation role permissions before allowing automated actions to continue.",
                "recommended_next_focus": "Use role evidence to justify a least-privilege remediation boundary.",
                "choice_role": "best",
            },
            {
                "id": "pause_unsafe_automation",
                "title": "Pause Unsafe Automation",
                "description": "Pause the remediation workflow until validation and role scope are confirmed safe.",
                "recommended_next_focus": "Use workflow and validation evidence to justify a controlled pause.",
                "choice_role": "partial",
            },
            {
                "id": "continue_monitored_remediation",
                "title": "Continue Monitored Remediation",
                "description": "Allow the automation to continue while monitoring CloudWatch and Security Hub outputs.",
                "recommended_next_focus": "Use only if evidence shows the trigger and validation path are trustworthy.",
                "choice_role": "weak",
            },
        ]
        return

    if next_turn_number == 5:
        generated_turn["actions"] = [
            {
                "id": "verify_automation_recovery",
                "title": "Verify Automation Recovery",
                "description": "Confirm the workflow is constrained, observable, and ready for final reporting.",
                "recommended_next_focus": "Use recovery monitoring evidence to close the automation incident.",
                "choice_role": "best",
            },
            {
                "id": "summarise_workflow_audit_trail",
                "title": "Summarise Workflow Audit Trail",
                "description": "Prepare a final timeline of Security Hub, EventBridge, Lambda, and Step Functions events.",
                "recommended_next_focus": "Use the audit trail as support, but verify final monitoring state before closure.",
                "choice_role": "partial",
            },
            {
                "id": "review_remaining_cost_impact",
                "title": "Review Remaining Cost Impact",
                "description": "Check whether any cost anomaly remains after remediation.",
                "recommended_next_focus": "Treat billing as residual impact context unless it proves recovery.",
                "choice_role": "weak",
            },
        ]


def repair_automated_security_response_turn(
    generated_turn: dict[str, Any],
    next_turn_number: int,
) -> None:
    text = json_like_generated_turn(generated_turn)
    drift_terms = (
        "iam activity",
        "admin-test",
        "member-account",
        "consolelogin",
        "access key risk",
        "identity timeline",
        "same source ip",
    )

    evidence_facts = generated_turn.get("evidence_facts")
    evidence_text = " ".join(
        clean_value(item.get("title")) + " " + clean_value(item.get("summary"))
        for item in evidence_facts
        if isinstance(item, dict)
    ).lower() if isinstance(evidence_facts, list) else ""
    strong_template = ""

    if isinstance(evidence_facts, list):
        strong_item = get_strong_evidence_item(evidence_facts)

        if strong_item:
            strong_template = clean_value(
                strong_item.get("template") or strong_item.get("type")
            ).lower()

    needs_repair = any(term in text or term in evidence_text for term in drift_terms)

    if not needs_repair:
        return

    turn_config = generated_turn.get("turn_config")
    if isinstance(turn_config, dict):
        repair_automated_security_response_context(
            turn_config=turn_config,
            next_turn_number=next_turn_number,
            verdict="not yet final",
        )

    repair_automated_security_response_actions(generated_turn, next_turn_number)
    generated_turn["expected_outcomes"] = automated_security_response_expected_outcomes(
        next_turn_number
    )

    if next_turn_number == 2:
        generated_turn["evidence_facts"] = automated_security_response_turn2_evidence()
    elif next_turn_number == 3:
        generated_turn["evidence_facts"] = automated_security_response_turn3_evidence()
    elif next_turn_number == 4:
        generated_turn["evidence_facts"] = automated_security_response_turn4_evidence()
    elif next_turn_number == 5:
        generated_turn["evidence_facts"] = automated_security_response_turn5_evidence()


def automated_security_response_turn2_evidence() -> list[dict[str, Any]]:
    return [
        {
            "id": "lambda_validation_path_logs",
            "title": "Lambda Validation Path Logs",
            "type": "cloudwatch",
            "summary": "CloudWatch logs show payload validation and workflow handoff for the Security Hub finding.",
            "why_it_may_matter": "This is the strongest evidence for deciding whether the automation validated the input before remediation.",
            "support_role": "strong",
            "template": "cloudwatch",
            "facts": {
                "query": "fields @timestamp, @logStream, @message | filter @message like /validation|sanitised|workflow|finding/",
                "log_group": "/aws/lambda/securityhub-check-state",
                "matched_records": "4",
                "scanned_bytes": "2.4 MB",
                "time_range": "2023-10-01 10:18-10:25 UTC",
                "alarm_state": "Validation warning",
                "log_rows": [
                    ["2023-10-01T10:19:15Z", "check-state/[$LATEST]a1", "Received Security Hub finding custom-finding-7421"],
                    ["2023-10-01T10:19:17Z", "check-state/[$LATEST]a1", "Validated remediationTarget against allowlist"],
                    ["2023-10-01T10:20:02Z", "check-state/[$LATEST]a1", "Sanitised optional note field before workflow handoff"],
                    ["2023-10-01T10:21:35Z", "check-state/[$LATEST]a1", "Workflow execution securityhub-remediation-7421 requested"],
                ],
            },
        },
        {
            "id": "finding_workflow_correlation",
            "title": "Finding Workflow Correlation",
            "type": "cloudtrail",
            "summary": "CloudTrail links the Security Hub finding to EventBridge and a Step Functions execution.",
            "why_it_may_matter": "This supports correlation, but the validation logs still decide whether the payload was safe.",
            "support_role": "partial",
            "template": "cloudtrail",
            "facts": {
                "event_name": "StartExecution",
                "event_source": "states.amazonaws.com",
                "user": "events.amazonaws.com/securityhub-remediation-rule",
                "source_ip": "AWS Internal",
                "mfa": "service",
                "event_time": "2023-10-01T10:21:35Z",
                "region": "ap-southeast-1",
                "error_code": "-",
                "risk_signal": "Finding custom-finding-7421 started one remediation workflow execution.",
                "event_id": "evt-securityhub-remediation-002",
                "user_agent": "events.amazonaws.com",
                "recipient_account_id": "123456789012",
                "request_parameters": "executionName=securityhub-remediation-7421; findingId=custom-finding-7421",
                "related_events": [
                    ["10:19:12 UTC", "BatchImportFindings", "securityhub.amazonaws.com", "AWS Internal", "Success"],
                    ["10:20:04 UTC", "PutEvents", "events.amazonaws.com", "AWS Internal", "Success"],
                    ["10:21:35 UTC", "StartExecution", "events.amazonaws.com/securityhub-remediation-rule", "AWS Internal", "Success"],
                    ["10:22:04 UTC", "AssumeRole", "states.amazonaws.com", "AWS Internal", "Success"],
                ],
            },
        },
        {
            "id": "remediation_role_scope_review",
            "title": "Remediation Role Scope Review",
            "type": "access_key",
            "summary": "Credential review shows temporary activity for the remediation role.",
            "why_it_may_matter": "This shows blast-radius risk, but role activity alone does not prove unsafe automation.",
            "support_role": "weak",
            "template": "access_key",
            "facts": {
                "access_key_id": "AKIAREM7421SAFETY999",
                "owner": "RemediationRole/session/securityhub-remediation",
                "status": "Active",
                "last_used_service": "ssm.amazonaws.com",
                "last_used_region": "ap-southeast-1",
                "last_used_time": "2023-10-01T10:22:04Z",
                "source_ip": "AWS Internal",
            },
        },
    ]


def automated_security_response_turn3_evidence() -> list[dict[str, Any]]:
    return [
        {
            "id": "workflow_execution_scope_logs",
            "title": "Workflow Execution Scope Logs",
            "type": "cloudtrail",
            "summary": "CloudTrail shows one Security Hub finding starting one Step Functions remediation workflow.",
            "why_it_may_matter": "This directly supports checking blast radius before changing remediation permissions.",
            "support_role": "strong",
            "template": "cloudtrail",
            "facts": {
                "event_name": "StartExecution",
                "event_source": "states.amazonaws.com",
                "user": "events.amazonaws.com/securityhub-remediation-rule",
                "source_ip": "AWS Internal",
                "mfa": "service",
                "event_time": "2023-10-01T10:21:35Z",
                "region": "ap-southeast-1",
                "error_code": "-",
                "risk_signal": "One finding custom-finding-7421 maps to one remediation workflow execution.",
                "event_id": "evt-securityhub-remediation-003",
                "user_agent": "events.amazonaws.com",
                "recipient_account_id": "123456789012",
                "request_parameters": "executionName=securityhub-remediation-7421; findingId=custom-finding-7421",
                "related_events": [
                    ["2023-10-01T10:19:12Z", "BatchImportFindings", "securityhub.amazonaws.com", "AWS Internal", "Success"],
                    ["2023-10-01T10:20:04Z", "PutEvents", "events.amazonaws.com", "AWS Internal", "Success"],
                    ["2023-10-01T10:21:35Z", "StartExecution", "events.amazonaws.com/securityhub-remediation-rule", "AWS Internal", "Success"],
                ],
            },
        },
        {
            "id": "automation_scope_monitoring",
            "title": "Automation Scope Monitoring",
            "type": "cloudwatch",
            "summary": "CloudWatch logs show the same workflow execution and no repeated remediation loops.",
            "why_it_may_matter": "This supports the scope review, but CloudTrail is stronger for proving the trigger path.",
            "support_role": "partial",
            "template": "cloudwatch",
            "facts": {
                "query": "fields @timestamp, @message | filter @message like /execution|remediation|finding/",
                "log_group": "/aws/states/securityhub-remediation-workflow",
                "matched_records": "4",
                "scanned_bytes": "1.1 MB",
                "time_range": "2023-10-01 10:21-10:35 UTC",
                "alarm_state": "Single execution observed",
                "log_rows": [
                    ["2023-10-01T10:21:35Z", "workflow/execution", "Started execution securityhub-remediation-7421"],
                    ["2023-10-01T10:22:04Z", "workflow/execution", "Assumed remediation role for approved workflow branch"],
                    ["2023-10-01T10:25:48Z", "workflow/control", "No repeated trigger observed for finding custom-finding-7421"],
                ],
            },
        },
        {
            "id": "automation_billing_context",
            "title": "Automation Billing Context",
            "type": "billing",
            "summary": "Billing shows no major cost spike during the automation window.",
            "why_it_may_matter": "This is useful impact context, but it does not prove the automation boundary.",
            "support_role": "weak",
            "template": "billing",
            "facts": {
                "current_spend": "$72.40",
                "previous_average": "$68.10",
                "largest_service": "Lambda",
                "region": "ap-southeast-1",
                "change": "+6%",
            },
        },
    ]


def automated_security_response_turn4_evidence() -> list[dict[str, Any]]:
    return [
        {
            "id": "remediation_role_restriction_check",
            "title": "Remediation Role Restriction Check",
            "type": "access_key",
            "summary": "Role credential review shows the remediation role session, active status, and SSM usage.",
            "why_it_may_matter": "This directly supports restricting the remediation role before automated changes resume.",
            "support_role": "strong",
            "template": "access_key",
            "facts": {
                "access_key_id": "AKIAREM7421SAFETY999",
                "owner": "RemediationRole/session/securityhub-remediation",
                "status": "Active",
                "last_used_service": "ssm.amazonaws.com",
                "last_used_region": "ap-southeast-1",
                "last_used_time": "2023-10-01T10:22:04Z",
                "source_ip": "AWS Internal",
            },
        },
        {
            "id": "remediation_detection_context",
            "title": "Remediation Detection Context",
            "type": "guardduty",
            "summary": "Security monitoring context shows the affected automation workflow and medium severity finding.",
            "why_it_may_matter": "This supports caution, but role evidence is stronger for defining least privilege.",
            "support_role": "partial",
            "template": "guardduty",
            "facts": {
                "finding_id": "custom-finding-7421",
                "severity": "Medium",
                "finding_type": "SecurityHub.CustomFinding/RemediationInput",
                "resource": "securityhub-remediation-workflow",
                "principal": "events.amazonaws.com/securityhub-remediation-rule",
                "remote_ip": "AWS Internal",
                "region": "ap-southeast-1",
                "first_seen": "2023-10-01T10:19:12Z",
                "description": "Finding triggered an automated remediation path that should be constrained before continuation.",
            },
        },
        {
            "id": "remediation_cost_context",
            "title": "Remediation Cost Context",
            "type": "billing",
            "summary": "Billing remains close to normal during the remediation window.",
            "why_it_may_matter": "This is weak support because cost does not define the remediation role boundary.",
            "support_role": "weak",
            "template": "billing",
            "facts": {
                "current_spend": "$72.40",
                "previous_average": "$68.10",
                "largest_service": "Lambda",
                "region": "ap-southeast-1",
                "change": "+6%",
            },
        },
    ]


def automated_security_response_turn5_evidence() -> list[dict[str, Any]]:
    return [
        {
            "id": "automation_recovery_monitoring",
            "title": "Automation Recovery Monitoring",
            "type": "cloudwatch",
            "summary": "CloudWatch logs show the workflow paused, role boundary updated, and monitoring restored.",
            "why_it_may_matter": "This directly supports recovery verification and final incident closure.",
            "support_role": "strong",
            "template": "cloudwatch",
            "facts": {
                "query": "fields @timestamp, @message | filter @message like /boundary|paused|monitoring|recovered/",
                "log_group": "/aws/states/securityhub-remediation-workflow",
                "matched_records": "4",
                "scanned_bytes": "1.3 MB",
                "time_range": "2023-10-01 10:35-11:05 UTC",
                "alarm_state": "Recovered",
                "log_rows": [
                    ["2023-10-01T10:36:10Z", "workflow/control", "Execution remains paused before production remediation"],
                    ["2023-10-01T10:43:18Z", "iam/boundary", "RemediationRole permissions boundary updated"],
                    ["2023-10-01T10:55:42Z", "monitoring/recovery", "Workflow monitoring alarm returned to OK"],
                    ["2023-10-01T11:02:09Z", "incident/report", "Final automation recovery status recorded"],
                ],
            },
        },
        {
            "id": "workflow_final_audit_trail",
            "title": "Workflow Final Audit Trail",
            "type": "cloudtrail",
            "summary": "CloudTrail confirms the final role-boundary change and workflow-control events.",
            "why_it_may_matter": "This supports reporting, but monitoring recovery is stronger for final closure.",
            "support_role": "partial",
            "template": "cloudtrail",
            "facts": {
                "event_name": "PutRolePermissionsBoundary",
                "event_source": "iam.amazonaws.com",
                "user": "arn:aws:iam::123456789012:role/SecurityAdmin",
                "source_ip": "AWS Internal",
                "mfa": "true",
                "event_time": "2023-10-01T10:43:18Z",
                "region": "ap-southeast-1",
                "error_code": "-",
                "risk_signal": "RemediationRole permissions boundary updated before workflow recovery.",
                "event_id": "evt-securityhub-remediation-005",
                "user_agent": "aws-cli/2.13",
                "recipient_account_id": "123456789012",
                "request_parameters": "roleName=RemediationRole; permissionsBoundary=CloudIRRestrictedRemediation",
            },
        },
        {
            "id": "final_cost_snapshot",
            "title": "Final Cost Snapshot",
            "type": "billing",
            "summary": "Billing shows only a small Lambda and Step Functions increase after the incident.",
            "why_it_may_matter": "This is residual impact context rather than proof that automation recovery is complete.",
            "support_role": "weak",
            "template": "billing",
            "facts": {
                "current_spend": "$74.15",
                "previous_average": "$68.10",
                "largest_service": "Lambda",
                "region": "ap-southeast-1",
                "change": "+9%",
            },
        },
    ]


def automated_security_response_expected_outcomes(turn_number: int) -> dict[str, str]:
    if turn_number == 2:
        return {
            "strong_support": "CloudWatch validation logs prove whether the finding payload was checked before workflow handoff.",
            "partial_support": "CloudTrail correlation shows workflow execution but needs validation context to prove safe automation.",
            "weak_support": "Evidence only reviews role scope or credentials without linking the finding to workflow execution.",
            "unsupported": "Evidence does not connect the finding, trigger, validation logs, or remediation workflow.",
        }

    if turn_number == 3:
        return {
            "strong_support": "CloudTrail workflow evidence proves the finding mapped to the expected remediation execution.",
            "partial_support": "CloudWatch scope monitoring supports the workflow review but needs audit context.",
            "weak_support": "Billing context does not prove the automation boundary or blast radius.",
            "unsupported": "Evidence does not address workflow scope, execution, or affected automation resources.",
        }

    if turn_number == 4:
        return {
            "strong_support": "Role and credential evidence supports restricting the remediation role before automation resumes.",
            "partial_support": "Detection context supports caution but does not fully define the least-privilege boundary.",
            "weak_support": "Cost evidence does not justify the remediation permission boundary.",
            "unsupported": "Evidence does not justify the containment or remediation decision.",
        }

    return {
        "strong_support": "Recovery monitoring proves the workflow is constrained, observable, and ready for final closure.",
        "partial_support": "Final audit evidence supports reporting but should be paired with recovery status.",
        "weak_support": "Billing context only shows residual impact and does not prove recovery.",
        "unsupported": "Evidence does not support recovery verification or final incident reporting.",
    }


def repair_turn2_correlation_evidence(generated_turn: dict[str, Any]) -> None:
    turn_config = generated_turn.get("turn_config")

    if not isinstance(turn_config, dict):
        return

    if turn_config.get("turn") != 2:
        return

    evidence_facts = generated_turn.get("evidence_facts")

    if not isinstance(evidence_facts, list):
        return

    templates = [
        clean_value(item.get("template") or item.get("type")).lower()
        for item in evidence_facts
        if isinstance(item, dict)
    ]
    titles = " ".join(
        clean_value(item.get("title"))
        for item in evidence_facts
        if isinstance(item, dict)
    ).lower()

    first_template = templates[0] if templates else ""
    missing_correlation_mix = not {"iam_activity", "access_key"}.issubset(set(templates))
    cloudtrail_still_drives_turn = (
        first_template == "cloudtrail"
        or "cloudtrail log entry" in titles
        or "cloudtrail logs" in titles
    )

    if not (missing_correlation_mix or cloudtrail_still_drives_turn):
        return

    generated_turn["evidence_facts"] = [
        {
            "id": "iam_scope_timeline",
            "title": "IAM Scope Timeline",
            "type": "iam_activity",
            "summary": "IAM activity links login, logging, and credential actions to the same principal and source IP.",
            "why_it_may_matter": "This helps determine whether the suspicious login became a wider identity-risk pattern.",
            "support_role": "strong",
            "template": "iam_activity",
            "facts": {
                "principal": "arn:aws:iam::123456789012:user/admin-test",
                "source_ip": "185.220.101.42",
                "mfa": "false",
                "policy_change": "No approved IAM change ticket found",
                "access_key_status": "New active key observed after login",
                "risk_flags": [
                    "Console login without MFA",
                    "Credential action from same source IP",
                ],
                "activity_rows": [
                    ["2023-10-01T10:12:10Z", "arn:aws:iam::123456789012:user/admin-test", "ConsoleLogin", "185.220.101.42"],
                    ["2023-10-01T10:18:44Z", "arn:aws:iam::123456789012:user/admin-test", "ListTrails", "185.220.101.42"],
                    ["2023-10-01T10:21:35Z", "arn:aws:iam::123456789012:user/admin-test", "StopLogging", "185.220.101.42"],
                    ["2023-10-01T10:24:02Z", "arn:aws:iam::123456789012:user/admin-test", "CreateAccessKey", "185.220.101.42"],
                    ["2023-10-01T10:26:11Z", "arn:aws:iam::123456789012:user/admin-test", "GetCallerIdentity", "185.220.101.42"],
                ],
            },
        },
        {
            "id": "access_key_review",
            "title": "Access Key Review",
            "type": "access_key",
            "summary": "An active access key is associated with the suspicious principal shortly after the login.",
            "why_it_may_matter": "This is relevant credential evidence, but it needs timeline context before containment.",
            "support_role": "partial",
            "template": "access_key",
            "facts": {
                "access_key_id": "AKIA4Z7EXAMPLE92K",
                "owner": "arn:aws:iam::123456789012:user/admin-test",
                "status": "Active",
                "last_used_service": "iam.amazonaws.com",
                "last_used_region": "ap-southeast-1",
                "last_used_time": "2023-10-01T10:24:02Z",
                "source_ip": "185.220.101.42",
            },
        },
        routine_application_logs_evidence(),
    ]


def routine_application_logs_evidence() -> dict[str, Any]:
    """Weak identity-scenario evidence: plausible logs from the window, no identity activity."""

    return {
        "id": "cloudwatch_application_logs",
        "title": "CloudWatch Application Logs",
        "type": "cloudwatch",
        "summary": "CloudWatch shows routine application logs from the incident window.",
        "why_it_may_matter": "These logs show service health during the window but contain no identity activity.",
        "support_role": "weak",
        "template": "cloudwatch",
        "facts": {
            "query": "fields @timestamp, @logStream, @message | sort @timestamp desc | limit 20",
            "log_group": "/aws/lambda/orders-api",
            "matched_records": "2",
            "scanned_bytes": "0.6 MB",
            "time_range": "2023-10-01 10:10-10:30 UTC",
            "alarm_state": "OK",
            "log_rows": [
                ["2023-10-01T10:15:02Z", "orders-api/[$LATEST]7c", "GET /health returned 200 in 41 ms"],
                ["2023-10-01T10:20:11Z", "orders-api/[$LATEST]7c", "Processed 128 order events from queue orders-inbound"],
            ],
        },
    }


def repair_turn3_response_evidence(generated_turn: dict[str, Any]) -> None:
    turn_config = generated_turn.get("turn_config")

    if not isinstance(turn_config, dict):
        return

    if turn_config.get("turn") != 3:
        return

    evidence_facts = generated_turn.get("evidence_facts")

    if not isinstance(evidence_facts, list):
        return

    templates = {
        clean_value(item.get("template") or item.get("type")).lower()
        for item in evidence_facts
        if isinstance(item, dict)
    }
    titles = " ".join(
        clean_value(item.get("title"))
        for item in evidence_facts
        if isinstance(item, dict)
    ).lower()
    strong_item = get_strong_evidence_item(evidence_facts)
    strong_template = ""

    if strong_item:
        strong_template = clean_value(
            strong_item.get("template") or strong_item.get("type")
        ).lower()

    needs_response_repair = (
        "guardduty" not in templates
        or strong_template != "guardduty"
        or "implement security measures" in json_like_generated_turn(generated_turn)
        or "review additional cloudtrail" in json_like_generated_turn(generated_turn)
        or "review recent cloudtrail" in json_like_generated_turn(generated_turn)
        or "inspect cloudtrail" in json_like_generated_turn(generated_turn)
        or "cloudtrail log entry" in titles
        or "access key containment" in titles
    )

    if not needs_response_repair:
        return

    generated_turn["evidence_facts"] = [
        {
            "id": "guardduty_scope_finding",
            "title": "GuardDuty Scope Finding",
            "type": "guardduty",
            "summary": "GuardDuty links the suspicious principal, remote IP, severity, and finding window.",
            "why_it_may_matter": "This directly supports deciding the blast radius before choosing containment.",
            "support_role": "strong",
            "template": "guardduty",
            "facts": {
                "finding_id": "gd-identity-scope-1042",
                "severity": "High",
                "finding_type": "UnauthorizedAccess:IAMUser/ConsoleLogin",
                "resource": "arn:aws:iam::123456789012:user/admin-test",
                "principal": "arn:aws:iam::123456789012:user/admin-test",
                "remote_ip": "185.220.101.42",
                "region": "ap-southeast-1",
                "first_seen": "2023-10-01T10:12:10Z",
                "description": "Suspicious console login and credential activity observed for the same IAM user.",
            },
        },
        {
            "id": "cloudtrail_logging_recovery_check",
            "title": "CloudTrail Scope Check",
            "type": "cloudtrail",
            "summary": "CloudTrail shows the StopLogging event that must be addressed before recovery can be trusted.",
            "why_it_may_matter": "This supports the scope review, but detection evidence is stronger for blast radius.",
            "support_role": "partial",
            "template": "cloudtrail",
            "facts": {
                "event_name": "StopLogging",
                "event_source": "cloudtrail.amazonaws.com",
                "user": "arn:aws:iam::123456789012:user/admin-test",
                "source_ip": "185.220.101.42",
                "mfa": "false",
                "event_time": "2023-10-01T10:21:35Z",
                "region": "ap-southeast-1",
                "error_code": "-",
                "risk_signal": "Trail logging was stopped for prod-org-trail.",
                "event_id": "e7b31d90-8b6a-4d5c-93f1-7b3d4f6a19c2",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "recipient_account_id": "123456789012",
                "request_parameters": "name=prod-org-trail",
                "related_events": [
                    ["2023-10-01T10:21:35Z", "StopLogging", "arn:aws:iam::123456789012:user/admin-test", "185.220.101.42", "Success"],
                ],
            },
        },
        {
            "id": "billing_impact_snapshot",
            "title": "Billing Impact Snapshot",
            "type": "billing",
            "summary": "Billing shows a modest spend increase that should be checked for resource misuse.",
            "why_it_may_matter": "This is useful impact context, but it does not prove the identity blast radius.",
            "support_role": "weak",
            "template": "billing",
            "facts": {
                "current_spend": "$184.20",
                "previous_average": "$42.75",
                "largest_service": "EC2",
                "region": "ap-southeast-1",
                "change": "+331%",
            },
        },
    ]


def repair_turn4_containment_evidence(generated_turn: dict[str, Any]) -> None:
    turn_config = generated_turn.get("turn_config")

    if not isinstance(turn_config, dict) or turn_config.get("turn") != 4:
        return

    evidence_facts = generated_turn.get("evidence_facts")

    if not isinstance(evidence_facts, list):
        return

    templates = {
        clean_value(item.get("template") or item.get("type")).lower()
        for item in evidence_facts
        if isinstance(item, dict)
    }
    strong_item = get_strong_evidence_item(evidence_facts)
    strong_template = ""

    if strong_item:
        strong_template = clean_value(
            strong_item.get("template") or strong_item.get("type")
        ).lower()

    if "access_key" in templates and strong_template == "access_key":
        return

    generated_turn["evidence_facts"] = [
        {
            "id": "access_key_containment_review",
            "title": "Access Key Containment Review",
            "type": "access_key",
            "summary": "The affected principal has an active access key used shortly after the suspicious login sequence.",
            "why_it_may_matter": "This directly supports disabling, rotating, or reviewing credentials during containment.",
            "support_role": "strong",
            "template": "access_key",
            "facts": {
                "access_key_id": "AKIA4Z7EXAMPLE92K",
                "owner": "arn:aws:iam::123456789012:user/admin-test",
                "status": "Active",
                "last_used_service": "iam.amazonaws.com",
                "last_used_region": "ap-southeast-1",
                "last_used_time": "2023-10-01T10:24:02Z",
                "source_ip": "185.220.101.42",
            },
        },
        {
            "id": "containment_monitoring_check",
            "title": "Containment Monitoring Check",
            "type": "cloudwatch",
            "summary": "CloudWatch shows a credential-change alarm firing for admin-test, without the key's current status.",
            "why_it_may_matter": "This supports containment monitoring, but active key status is stronger for the direct response decision.",
            "support_role": "partial",
            "template": "cloudwatch",
            "facts": {
                "query": "fields @timestamp, @logStream, @message | filter @message like /AccessKey|IAMCredentialChange/",
                "log_group": "/aws/cloudtrail/organization/prod",
                "matched_records": "2",
                "scanned_bytes": "1.2 MB",
                "time_range": "2023-10-01 10:24-10:45 UTC",
                "alarm_state": "ALARM",
                "log_rows": [
                    ["2023-10-01T10:24:02Z", "iam/credential-monitor", "CreateAccessKey event received for admin-test"],
                    ["2023-10-01T10:24:40Z", "alarms/iam-credential-change", "Alarm IAMCredentialChange changed state from OK to ALARM"],
                ],
            },
        },
        {
            "id": "containment_billing_context",
            "title": "Containment Billing Context",
            "type": "billing",
            "summary": "Billing shows a modest spend increase during the incident window.",
            "why_it_may_matter": "This is weak support because cost context does not prove which credential to contain.",
            "support_role": "weak",
            "template": "billing",
            "facts": {
                "current_spend": "$184.20",
                "previous_average": "$42.75",
                "largest_service": "EC2",
                "region": "ap-southeast-1",
                "change": "+331%",
            },
        },
    ]


def repair_turn5_recovery_evidence(generated_turn: dict[str, Any]) -> None:
    turn_config = generated_turn.get("turn_config")

    if not isinstance(turn_config, dict) or turn_config.get("turn") != 5:
        return

    evidence_facts = generated_turn.get("evidence_facts")

    if not isinstance(evidence_facts, list):
        return

    templates = {
        clean_value(item.get("template") or item.get("type")).lower()
        for item in evidence_facts
        if isinstance(item, dict)
    }
    strong_item = get_strong_evidence_item(evidence_facts)
    strong_template = ""

    if strong_item:
        strong_template = clean_value(
            strong_item.get("template") or strong_item.get("type")
        ).lower()

    strong_facts = strong_item.get("facts") if isinstance(strong_item, dict) else None
    strong_title = clean_value(strong_item.get("title")) if isinstance(strong_item, dict) else ""
    strong_is_recovery_record = (
        isinstance(strong_facts, dict)
        and bool(strong_facts)
        and (
            "recovery" in strong_title.lower()
            or clean_value(strong_facts.get("event_name")).lower()
            in {"startlogging", "updateaccesskey", "putrolepermissionsboundary"}
        )
    )

    if "cloudtrail" in templates and strong_template == "cloudtrail" and strong_is_recovery_record:
        return

    generated_turn["evidence_facts"] = [
        {
            "id": "final_cloudtrail_recovery_record",
            "title": "Final CloudTrail Recovery Record",
            "type": "cloudtrail",
            "summary": "CloudTrail confirms logging recovery and credential-control actions for the affected principal.",
            "why_it_may_matter": "This directly supports final recovery verification and incident reporting.",
            "support_role": "strong",
            "template": "cloudtrail",
            "facts": {
                "event_name": "StartLogging",
                "event_source": "cloudtrail.amazonaws.com",
                "user": "arn:aws:iam::123456789012:role/SecurityAdmin",
                "source_ip": "AWS Internal",
                "mfa": "true",
                "event_time": "2023-10-01T11:08:17Z",
                "region": "ap-southeast-1",
                "error_code": "-",
                "risk_signal": "Audit logging restored after suspicious credential activity was contained.",
                "event_id": "e7b31d90-8b6a-4d5c-93f1-7b3d4f6a19c5",
                "user_agent": "aws-cli/2.13",
                "recipient_account_id": "123456789012",
                "request_parameters": "name=prod-org-trail",
                "related_events": [
                    ["2023-10-01T10:45:22Z", "UpdateAccessKey", "arn:aws:iam::123456789012:role/SecurityAdmin", "AWS Internal", "Inactive"],
                    ["2023-10-01T11:08:17Z", "StartLogging", "arn:aws:iam::123456789012:role/SecurityAdmin", "AWS Internal", "Success"],
                    ["2023-10-01T11:14:41Z", "GetTrailStatus", "arn:aws:iam::123456789012:role/SecurityAdmin", "AWS Internal", "Logging"],
                ],
            },
        },
        {
            "id": "final_identity_timeline",
            "title": "Final Identity Timeline",
            "type": "iam_activity",
            "summary": "IAM activity shows the suspicious access key was disabled, but not whether audit logging was restored.",
            "why_it_may_matter": "This supports reporting containment, but CloudTrail recovery evidence is needed to prove closure.",
            "support_role": "partial",
            "template": "iam_activity",
            "facts": {
                "principal": "arn:aws:iam::123456789012:user/admin-test",
                "source_ip": "185.220.101.42",
                "mfa": "false",
                "policy_change": "SecurityAdmin disabled suspicious access key",
                "access_key_status": "Inactive after containment",
                "risk_flags": [
                    "Suspicious login without MFA",
                    "Access key disabled by SecurityAdmin",
                ],
                "activity_rows": [
                    ["2023-10-01T10:12:10Z", "arn:aws:iam::123456789012:user/admin-test", "ConsoleLogin", "185.220.101.42"],
                    ["2023-10-01T10:24:02Z", "arn:aws:iam::123456789012:user/admin-test", "CreateAccessKey", "185.220.101.42"],
                    ["2023-10-01T10:45:22Z", "arn:aws:iam::123456789012:role/SecurityAdmin", "UpdateAccessKey", "AWS Internal"],
                ],
            },
        },
        {
            "id": "final_billing_snapshot",
            "title": "Final Billing Snapshot",
            "type": "billing",
            "summary": "Billing shows residual cost impact after the identity incident.",
            "why_it_may_matter": "This is weak support because it shows impact, not recovery verification.",
            "support_role": "weak",
            "template": "billing",
            "facts": {
                "current_spend": "$191.80",
                "previous_average": "$42.75",
                "largest_service": "EC2",
                "region": "ap-southeast-1",
                "change": "+349%",
            },
        },
    ]


def get_strong_evidence_item(evidence_facts: list[Any]) -> dict[str, Any] | None:
    for index, item in enumerate(evidence_facts):
        if not isinstance(item, dict):
            continue

        support_role = clean_value(item.get("support_role")).lower()

        if support_role == "strong" or (not support_role and index == 0):
            return item

    return None


def json_like_generated_turn(generated_turn: dict[str, Any]) -> str:
    return " ".join(
        [
            clean_value(item.get("title"))
            for item in generated_turn.get("actions", [])
            if isinstance(item, dict)
        ]
    ).lower()


def get_completed_turn_config(completed_turn: dict[str, Any]) -> dict[str, Any]:
    turn_config = completed_turn.get("turn_config")

    if isinstance(turn_config, dict):
        return turn_config

    turn_config = completed_turn.get("turnConfig")

    if isinstance(turn_config, dict):
        return turn_config

    return {}


def extract_titles(payload: dict[str, Any], key: str) -> list[str]:
    items = payload.get(key, [])

    if not isinstance(items, list):
        return []

    titles: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        title = clean_value(item.get("title"))

        if title:
            titles.append(title)

    return titles


def mark_repeated_titles_as_follow_up(
    generated_items: Any,
    previous_titles: list[str],
) -> None:
    """
    Avoids identical titles across turns without inventing new actions or evidence.

    This is intentionally light-touch. It only renames exact repeated titles as
    follow-up items so the UI does not look like the same turn was regenerated.
    """

    if not isinstance(generated_items, list):
        return

    previous_title_set = {
        normalise_comparison_text(title)
        for title in previous_titles
        if title
    }

    if not previous_title_set:
        return

    for item in generated_items:
        if not isinstance(item, dict):
            continue

        title = clean_value(item.get("title"))

        if not title:
            continue

        normalised_title = normalise_comparison_text(title)

        if normalised_title in previous_title_set and "follow-up" not in normalised_title:
            item["title"] = f"{title} Follow-up"
