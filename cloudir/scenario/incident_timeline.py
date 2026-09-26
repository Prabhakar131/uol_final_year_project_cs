"""One incident timeline per scenario; every turn's evidence is built from it.

During preparation the security model writes one timeline for the scenario: who
acted and from where, the AWS API events, the security finding and the cost
picture. Code validates it once, then builds each turn's evidence from it:

- strong: the events the timeline marks for that turn (what the decision needs)
- partial: the same principal's normal baseline activity the day before, from its
  usual IP (right principal, but the suspicious step is missing)
- weak: other principals' normal activity (plausible but unrelated)

The model still decides what happened; code guarantees that every screenshot
uses real API names, the same identities, and a strong/partial/weak gradient.
The coach model writes each turn's briefing, actions and titles around it.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from cloudir.evidence_core.evidence_text_helpers import clean_value
from cloudir.scenario.action_choice_roles import normalise_choice_role
from cloudir.scenario.evidence_strength import (
    CONCLUSION_PHRASES,
    event_is_suspicious,
    support_role_content_problems,
)


TIMELINE_FILE = "incident_timeline.json"
TURNS = (1, 2, 3, 4, 5)
ROLES = ("strong", "partial", "weak")
# The training environment's region. Fact normalisers rewrite us-east-1, so a
# model-chosen region could differ between screenshots.
REGION = "ap-southeast-1"

DEFAULT_TEMPLATE_PLAN = {"strong": "cloudtrail", "partial": "iam_activity", "weak": "billing"}
TEMPLATE_PREFERENCE = ("cloudwatch", "iam_activity", "cloudtrail", "access_key", "guardduty", "billing")

# Role-free ids: the evidence id reaches the security judge's prompt.
EVIDENCE_IDS = {
    "cloudtrail": "cloudtrail_event_record",
    "iam_activity": "iam_principal_activity",
    "cloudwatch": "cloudwatch_logs_query",
    "guardduty": "guardduty_finding",
    "access_key": "access_key_details",
    "billing": "billing_daily_spend",
}

WHY_IT_MAY_MATTER = {
    "cloudtrail": "It records who called an API, when, from where and with what result.",
    "iam_activity": "It shows the recent API activity of one principal.",
    "cloudwatch": "It lists matching CloudTrail log lines for the query window.",
    "guardduty": "It shows what threat detection flagged and for which resource.",
    "access_key": "It shows a credential's owner, status and most recent use.",
    "billing": "It compares current daily spend with the normal average.",
}

# Real eventNames for the services incidents here mostly touch. Names under
# these sources must come from the list; other services are checked by pattern.
EVENT_CATALOGUE = {
    "signin.amazonaws.com": "ConsoleLogin CheckMfa SwitchRole ExitRole RenewRole GetSigninToken",
    "sts.amazonaws.com": "AssumeRole AssumeRoleWithSAML AssumeRoleWithWebIdentity GetCallerIdentity "
                         "GetSessionToken GetFederationToken DecodeAuthorizationMessage GetAccessKeyInfo",
    "iam.amazonaws.com": "AddClientIDToOpenIDConnectProvider AddRoleToInstanceProfile AddUserToGroup AttachGroupPolicy "
                         "AttachRolePolicy AttachUserPolicy ChangePassword CreateAccessKey CreateAccountAlias CreateGroup "
                         "CreateInstanceProfile CreateLoginProfile CreateOpenIDConnectProvider CreatePolicy "
                         "CreatePolicyVersion CreateRole CreateSAMLProvider CreateServiceLinkedRole "
                         "CreateServiceSpecificCredential CreateUser CreateVirtualMFADevice DeactivateMFADevice "
                         "DeleteAccessKey DeleteAccountAlias DeleteAccountPasswordPolicy DeleteGroup DeleteGroupPolicy "
                         "DeleteInstanceProfile DeleteLoginProfile DeleteOpenIDConnectProvider DeletePolicy "
                         "DeletePolicyVersion DeleteRole DeleteRolePermissionsBoundary DeleteRolePolicy DeleteSAMLProvider "
                         "DeleteServerCertificate DeleteServiceLinkedRole DeleteServiceSpecificCredential "
                         "DeleteSigningCertificate DeleteSSHPublicKey DeleteUser DeleteUserPermissionsBoundary "
                         "DeleteUserPolicy DeleteVirtualMFADevice DetachGroupPolicy DetachRolePolicy DetachUserPolicy "
                         "EnableMFADevice GenerateCredentialReport GenerateServiceLastAccessedDetails GetAccessKeyLastUsed "
                         "GetAccountAuthorizationDetails GetAccountPasswordPolicy GetAccountSummary "
                         "GetContextKeysForPrincipalPolicy GetCredentialReport GetGroup GetGroupPolicy GetInstanceProfile "
                         "GetLoginProfile GetOpenIDConnectProvider GetPolicy GetPolicyVersion GetRole GetRolePolicy "
                         "GetSAMLProvider GetServerCertificate GetServiceLastAccessedDetails GetSSHPublicKey GetUser "
                         "GetUserPolicy ListAccessKeys ListAccountAliases ListAttachedGroupPolicies ListAttachedRolePolicies "
                         "ListAttachedUserPolicies ListEntitiesForPolicy ListGroupPolicies ListGroups ListGroupsForUser "
                         "ListInstanceProfiles ListInstanceProfilesForRole ListMFADevices ListOpenIDConnectProviders "
                         "ListPolicies ListPolicyVersions ListRolePolicies ListRoles ListRoleTags ListSAMLProviders "
                         "ListServerCertificates ListSigningCertificates ListSSHPublicKeys ListUserPolicies ListUsers "
                         "ListUserTags ListVirtualMFADevices PutGroupPolicy PutRolePermissionsBoundary PutRolePolicy "
                         "PutUserPermissionsBoundary PutUserPolicy RemoveRoleFromInstanceProfile RemoveUserFromGroup "
                         "ResetServiceSpecificCredential ResyncMFADevice SetDefaultPolicyVersion SimulateCustomPolicy "
                         "SimulatePrincipalPolicy TagRole TagUser UntagRole UntagUser UpdateAccessKey "
                         "UpdateAccountPasswordPolicy UpdateAssumeRolePolicy UpdateGroup UpdateLoginProfile UpdateRole "
                         "UpdateRoleDescription UpdateSAMLProvider UpdateSigningCertificate UpdateSSHPublicKey UpdateUser "
                         "UploadServerCertificate UploadSigningCertificate UploadSSHPublicKey",
    "cloudtrail.amazonaws.com": "StopLogging StartLogging DeleteTrail UpdateTrail CreateTrail DescribeTrails "
                                "GetTrailStatus ListTrails PutEventSelectors GetEventSelectors LookupEvents",
    "s3.amazonaws.com": "ListBuckets GetObject PutObject DeleteObject DeleteObjects CopyObject HeadBucket HeadObject "
                         "GetObjectAcl PutObjectAcl GetBucketPolicy PutBucketPolicy DeleteBucketPolicy GetBucketAcl "
                         "PutBucketAcl GetBucketLocation ListObjects ListObjectsV2 ListObjectVersions CreateBucket "
                         "DeleteBucket GetBucketVersioning PutBucketVersioning GetBucketLogging PutBucketLogging "
                         "GetBucketEncryption PutBucketEncryption DeleteBucketEncryption GetBucketTagging PutBucketTagging "
                         "GetBucketCors PutBucketCors GetBucketPublicAccessBlock PutBucketPublicAccessBlock "
                         "DeleteBucketPublicAccessBlock GetPublicAccessBlock PutPublicAccessBlock DeletePublicAccessBlock "
                         "PutBucketLifecycle PutLifecycleConfiguration GetBucketLifecycle GetBucketReplication "
                         "PutBucketReplication",
    "ec2.amazonaws.com": "RunInstances TerminateInstances StopInstances StartInstances RebootInstances DescribeInstances "
                         "DescribeInstanceStatus DescribeInstanceAttribute ModifyInstanceAttribute "
                         "ModifyInstanceMetadataOptions DescribeRegions DescribeAvailabilityZones DescribeAccountAttributes "
                         "DescribeInstanceTypes DescribeSecurityGroups CreateSecurityGroup DeleteSecurityGroup "
                         "AuthorizeSecurityGroupIngress AuthorizeSecurityGroupEgress RevokeSecurityGroupIngress "
                         "RevokeSecurityGroupEgress CreateKeyPair ImportKeyPair DeleteKeyPair DescribeKeyPairs CreateSnapshot "
                         "DeleteSnapshot CopySnapshot DescribeSnapshots ModifySnapshotAttribute CreateVolume AttachVolume "
                         "DetachVolume DeleteVolume DescribeVolumes CreateImage DescribeImages ModifyImageAttribute "
                         "DescribeVpcs CreateVpc DescribeSubnets DescribeNetworkInterfaces CreateTags DeleteTags DescribeTags "
                         "AllocateAddress AssociateAddress DescribeAddresses RequestSpotInstances DescribeSpotInstanceRequests "
                         "CreateLaunchTemplate DescribeLaunchTemplates GetPasswordData GetConsoleOutput CreateFlowLogs "
                         "DeleteFlowLogs DescribeFlowLogs",
    "guardduty.amazonaws.com": "DeleteDetector UpdateDetector ListDetectors GetDetector ListFindings GetFindings "
                               "ArchiveFindings",
    "securityhub.amazonaws.com": "AcceptAdministratorInvitation BatchDeleteAutomationRules BatchDisableStandards "
                                 "BatchEnableStandards BatchGetAutomationRules BatchGetSecurityControls "
                                 "BatchImportFindings BatchUpdateAutomationRules BatchUpdateFindings CreateActionTarget "
                                 "CreateAutomationRule CreateFindingAggregator CreateInsight CreateMembers "
                                 "DeleteActionTarget DeleteFindingAggregator DeleteInsight DeleteMembers "
                                 "DescribeActionTargets DescribeHub DescribeOrganizationConfiguration DescribeProducts "
                                 "DescribeStandards DescribeStandardsControls DisableImportFindingsForProduct "
                                 "DisableSecurityHub DisassociateFromAdministratorAccount DisassociateMembers "
                                 "EnableImportFindingsForProduct EnableSecurityHub GetAdministratorAccount "
                                 "GetEnabledStandards GetFindingHistory GetFindings GetInsightResults GetInsights "
                                 "GetMembers InviteMembers ListAutomationRules ListEnabledProductsForImport "
                                 "ListFindingAggregators ListInvitations ListMembers ListSecurityControlDefinitions "
                                 "ListTagsForResource TagResource UntagResource UpdateActionTarget "
                                 "UpdateFindingAggregator UpdateFindings UpdateInsight UpdateOrganizationConfiguration "
                                 "UpdateSecurityControl UpdateSecurityHubConfiguration UpdateStandardsControl",
    "events.amazonaws.com": "PutRule PutTargets PutEvents DeleteRule EnableRule DisableRule RemoveTargets "
                            "DescribeRule ListRules ListTargetsByRule ListRuleNamesByTarget PutPermission "
                            "RemovePermission CreateEventBus DeleteEventBus DescribeEventBus ListEventBuses "
                            "TestEventPattern TagResource UntagResource",
    "states.amazonaws.com": "StartExecution StartSyncExecution StopExecution CreateStateMachine UpdateStateMachine "
                            "DeleteStateMachine DescribeStateMachine DescribeExecution ListExecutions "
                            "ListStateMachines GetExecutionHistory SendTaskSuccess SendTaskFailure "
                            "SendTaskHeartbeat TagResource UntagResource",
    # CloudTrail records most Lambda calls under a versioned eventName.
    "lambda.amazonaws.com": "Invoke CreateFunction20150331 DeleteFunction20150331 GetFunction20150331v2 "
                            "GetFunctionConfiguration20150331v2 UpdateFunctionCode20150331v2 "
                            "UpdateFunctionConfiguration20150331v2 AddPermission20150331v2 "
                            "RemovePermission20150331v2 GetPolicy20150331v2 ListFunctions20150331 "
                            "PublishVersion20150331 CreateAlias20150331 UpdateAlias20150331 DeleteAlias20150331 "
                            "ListVersionsByFunction20150331 ListAliases20150331 CreateEventSourceMapping20150331 "
                            "UpdateEventSourceMapping20150331 DeleteEventSourceMapping20150331 "
                            "ListEventSourceMappings20150331 PutFunctionConcurrency20171031 "
                            "DeleteFunctionConcurrency20171031",
}
EVENT_CATALOGUE = {source: set(names.split()) for source, names in EVENT_CATALOGUE.items()}
KNOWN_EVENT_SOURCES = {name: source for source, names in EVENT_CATALOGUE.items() for name in names}
KNOWN_EVENT_SOURCES.update({"SendCommand": "ssm.amazonaws.com", "Decrypt": "kms.amazonaws.com",
                            "GetCostAndUsage": "ce.amazonaws.com"})
# Models write the Lambda API name ("UpdateFunctionCode"); CloudTrail adds the version.
LAMBDA_VERSIONED = {re.sub(r"20\d{6}(v\d)?$", "", name): name for name in EVENT_CATALOGUE["lambda.amazonaws.com"]}
# Named in rejection messages so the model switches to a real, relevant API.
SUGGESTED_EVENTS = {
    "iam.amazonaws.com": "UpdateAccessKey, DeleteLoginProfile, DetachUserPolicy, AttachUserPolicy, CreateAccessKey, ListUsers",
    "sts.amazonaws.com": "AssumeRole, GetCallerIdentity, GetSessionToken",
    "signin.amazonaws.com": "ConsoleLogin, CheckMfa, SwitchRole",
    "cloudtrail.amazonaws.com": "StopLogging, StartLogging, UpdateTrail, DeleteTrail, GetTrailStatus",
    "s3.amazonaws.com": "ListBuckets, GetObject, PutBucketPolicy, GetBucketAcl, PutBucketPublicAccessBlock",
    "ec2.amazonaws.com": "RunInstances, DescribeInstances, TerminateInstances, AuthorizeSecurityGroupIngress",
    "guardduty.amazonaws.com": "UpdateDetector, DeleteDetector, ListFindings, ArchiveFindings",
    "securityhub.amazonaws.com": "BatchImportFindings, BatchUpdateFindings, GetFindings, CreateActionTarget, "
                                 "CreateAutomationRule, UpdateSecurityHubConfiguration, DisableSecurityHub",
    "events.amazonaws.com": "PutRule, PutTargets, PutEvents, DisableRule, RemoveTargets, DescribeRule, ListRules",
    "states.amazonaws.com": "StartExecution, StopExecution, UpdateStateMachine, DescribeExecution, ListExecutions",
    "lambda.amazonaws.com": "Invoke, UpdateFunctionCode20150331v2, UpdateFunctionConfiguration20150331v2, "
                            "AddPermission20150331v2, GetFunction20150331v2, ListFunctions20150331",
}
# Threat-model wording -> the service endpoint whose API calls the incident should use.
SERVICE_KEYWORDS = {
    r"\blambda\b": "lambda.amazonaws.com", r"security ?hub": "securityhub.amazonaws.com",
    r"eventbridge|cloudwatch events": "events.amazonaws.com", r"step functions|state machine": "states.amazonaws.com",
    r"identity center|\bsso\b": "sso.amazonaws.com", r"\biam\b": "iam.amazonaws.com",
    r"\bs3\b|bucket": "s3.amazonaws.com", r"\bec2\b": "ec2.amazonaws.com", r"cloudtrail": "cloudtrail.amazonaws.com",
    r"guardduty": "guardduty.amazonaws.com", r"\bkms\b": "kms.amazonaws.com", r"organizations": "organizations.amazonaws.com",
    r"systems manager|\bssm\b": "ssm.amazonaws.com", r"dynamodb": "dynamodb.amazonaws.com", r"\brds\b": "rds.amazonaws.com",
    r"cost explorer|billing|\bcost\b": "ce.amazonaws.com", r"budget": "budgets.amazonaws.com", r"\bsns\b": "sns.amazonaws.com",
    r"\bsqs\b": "sqs.amazonaws.com", r"api gateway": "apigateway.amazonaws.com", r"cognito": "cognito-idp.amazonaws.com",
}
REAL_FINDING_TYPES = ("UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration.OutsideAWS, "
                      "UnauthorizedAccess:IAMUser/TorIPCaller, Discovery:IAMUser/AnomalousBehavior, "
                      "Persistence:IAMUser/AnomalousBehavior, PrivilegeEscalation:IAMUser/AnomalousBehavior, "
                      "Stealth:IAMUser/CloudTrailLoggingDisabled, CryptoCurrency:EC2/BitcoinTool.B!DNS, "
                      "Backdoor:Lambda/C&CActivity.B")
# Endpoints models write that CloudTrail records under another name.
SOURCE_ALIASES = {"eventbridge.amazonaws.com": "events.amazonaws.com",
                  "stepfunctions.amazonaws.com": "states.amazonaws.com",
                  "cloudwatch.amazonaws.com": "monitoring.amazonaws.com",
                  "identitycenter.amazonaws.com": "sso.amazonaws.com"}
# Names that announce the answer ("malicioususer", "attackerEC2") on every screenshot.
# The trailing lowercase run takes "attacker" whole but stops at camelCase ("MaliciousFinding").
REVEALING_NAME = re.compile(r"(?i:malicious|attack|hack|compromis|evil|rogue|suspicious|unauthori[sz]ed|intruder|"
                            r"threat|exploit|backdoor|bad[-_]?actor)[a-z]*")
# Stand-ins code uses to rename them, so one bad name does not cost a whole attempt.
NEUTRAL_WORDS = ("ops", "data", "sync", "batch", "build", "infra", "svc", "app")
NEUTRAL_USERS = ("d.okafor", "m.tanaka", "s.brennan", "a.kowalski", "r.iyer", "l.fischer")
# Everyday calls for the principal's baseline. Reconnaissance-style calls such as
# DescribeRegions or GetAccountSummary look suspicious to a security judge.
BASELINE_CALL_POOL = (("GetObject", "s3.amazonaws.com"), ("ListObjectsV2", "s3.amazonaws.com"),
                      ("DescribeInstances", "ec2.amazonaws.com"), ("GetBucketLocation", "s3.amazonaws.com"),
                      ("DescribeVolumes", "ec2.amazonaws.com"), ("PutObject", "s3.amazonaws.com"))
ROUTINE_PREFIXES = ("Describe", "Get", "List", "Head", "Lookup")
ROUTINE_WRITES = {"PutMetricData", "PutObject", "PutLogEvents"}
# Plausible-looking names models invent instead of real CloudTrail eventNames.
VAGUE_EVENT_NAMES = {"servicerequest", "login", "loginattempt", "failedlogin", "failedloginattempt",
                     "accessrequest", "apicall", "event", "activity", "request", "unknown"}
KEY_CONTROL_EVENTS = {"updateaccesskey", "deleteaccesskey"}

_EVENT_NAME = re.compile(r"^[A-Z][A-Za-z0-9]{2,}$")
_EVENT_SOURCE = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)*\.amazonaws\.com$")
_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_FINDING_TYPE = re.compile(r"^[A-Za-z]+:[A-Za-z0-9]+/[A-Za-z0-9.!&_-]+$")  # Purpose:Resource/Family
_ARN = re.compile(r"^arn:aws:(iam|sts)::(\d{12}):[A-Za-z0-9+=,.@_/-]+$")
_SERVICE_PRINCIPAL = re.compile(r"^[a-z0-9-]+\.amazonaws\.com(/[A-Za-z0-9+=,.@_/-]+)?$")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def threat_model_services(threat_model: dict[str, Any] | None) -> list[str]:
    """Service endpoints named by the threat model's affected assets and attack path."""

    threat_model = threat_model if isinstance(threat_model, dict) else {}
    text = " ".join(str(item) for key in ("affected_assets", "likely_attack_path")
                    for item in (threat_model.get(key) or [])).lower()
    return sorted({service for pattern, service in SERVICE_KEYWORDS.items() if re.search(pattern, text)})


def finalise_incident_timeline(raw: Any, scenario_config: dict[str, Any],
                               threat_model: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalise and validate a model-written timeline, including every turn's evidence.

    Raises ValueError listing the problems so the model can be asked to fix them.
    """

    timeline = normalise_incident_timeline(raw)
    timeline["scenario_id"] = clean_value(scenario_config.get("scenario_id"))

    # A generic account compromise is easy to write; the scenario's own services are not.
    services = threat_model_services(threat_model)
    steps = {event["event_source"] for event in timeline["events"]
             if event["turn"] and _actor(timeline, event["actor"])["role"] == "attacker"}
    if services and not steps & set(services):
        raise ValueError("The attacker's incident steps must use the scenario's own services "
                         f"({', '.join(services)}), following the threat model's attack path.")

    problems = []
    for turn in TURNS:
        try:
            items = build_turn_evidence(timeline, scenario_config, turn)
        except (KeyError, IndexError, StopIteration, ValueError) as exc:
            raise ValueError(f"Turn {turn} evidence could not be built from the timeline: {exc!r}.") from exc
        problems.extend(f"Turn {turn}: {problem}" for problem in support_role_content_problems(items))

    if problems:
        raise ValueError("Timeline evidence does not separate strong, partial and weak: " + " ".join(problems[:6]))

    return timeline


def normalise_incident_timeline(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("The timeline must be a JSON object.")

    raw = _neutralise_names(raw)
    errors: list[str] = []
    account_id = clean_value(raw.get("account_id"))

    if not re.fullmatch(r"\d{12}", account_id):
        errors.append("account_id must be exactly 12 digits.")
        account_id = "123456789012"

    actors = _normalise_actors(raw.get("actors"), account_id, errors)
    events = _normalise_events(raw.get("events"), actors, errors)
    findings = _normalise_findings(raw.get("findings"), errors)
    cost = _normalise_cost(raw.get("cost"), errors)

    # The model writes the story; code does the bookkeeping it found hard to get
    # right in one reply: turn labels, routine calls and background noise.
    if not errors:
        events = _assign_turns(events, actors, errors)
    if not errors:
        events = _keep_routine_background(events, actors)
        events = _add_baseline_activity(events, actors)
        events = _add_background_noise(events, actors)
        _check_event_structure(events, actors, errors)

    if errors:
        raise ValueError("Invalid incident timeline: " + " ".join(errors[:10]))

    return {"account_id": account_id, "region": REGION, "actors": actors,
            "events": events, "findings": findings, "cost": cost}


def _neutralise_names(raw: dict[str, Any]) -> dict[str, Any]:
    """Rename principals and resources whose names announce the answer.

    The model keeps writing names such as "malicious-lambda" however the prompt is
    worded. Each renamed segment is reused for every later mention, so the story
    stays consistent across events, findings and descriptions.
    """

    raw = copy.deepcopy(raw)
    renames: dict[str, str] = {}
    words: dict[str, str] = {}
    taken = json.dumps(raw)
    people = iter(user for user in NEUTRAL_USERS if user not in taken)

    def neutral_word(match: re.Match) -> str:
        word = match.group(0)
        new = words.setdefault(word.lower(), NEUTRAL_WORDS[len(words) % len(NEUTRAL_WORDS)])
        return new.capitalize() if word[0].isupper() else new

    def rename(text: str, person: bool) -> str:
        def segment(match: re.Match) -> str:
            old = match.group(0)
            if old not in renames and REVEALING_NAME.search(old):
                renames[old] = (next(people, None) if person else None) or REVEALING_NAME.sub(neutral_word, old)
            return renames.get(old, old)

        return re.sub(r"[^/:\s]+", segment, text)

    # Actors first, so a user renamed to a person keeps that name in event resources.
    for field, keys in (("actors", ("principal", "user_agent")), ("events", ("resource",)), ("findings", ("resource",))):
        for item in raw.get(field) if isinstance(raw.get(field), list) else []:
            for key in keys:
                if isinstance(item, dict) and isinstance(item.get(key), str):
                    item[key] = rename(item[key], person=key == "principal" and ":user/" in item[key])

    # Descriptions are prose ("a known malicious IP" is fine); only renamed names change.
    for finding in raw.get("findings") if isinstance(raw.get("findings"), list) else []:
        if isinstance(finding, dict) and isinstance(finding.get("description"), str):
            for old in sorted(renames, key=len, reverse=True):
                finding["description"] = finding["description"].replace(old, renames[old])

    return raw


def _normalise_actors(raw: Any, account_id: str, errors: list[str]) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        errors.append("actors must be a non-empty list.")
        return []

    actors = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append(f"actors[{index}] must be an object.")
            continue

        actor_id = clean_value(item.get("id"))
        role = clean_value(item.get("role")).lower()
        principal = _with_account(clean_value(item.get("principal")), account_id)
        source_ip = clean_value(item.get("source_ip"))

        if not actor_id:
            errors.append(f"actors[{index}] needs an id.")
        if role not in {"attacker", "background", "responder"}:
            errors.append(f"Actor {actor_id or index} role must be attacker, background or responder.")
        if not (_ARN.match(principal) or _SERVICE_PRINCIPAL.match(principal)):
            errors.append(f"Actor {actor_id or index} principal '{principal}' must be an IAM ARN or AWS service principal, not a placeholder name.")
        if source_ip.startswith("192.0.2."):
            errors.append(f"Actor {actor_id or index} must not use 192.0.2.x; choose another IPv4 address.")
        elif role == "attacker" and not _is_public_ipv4(source_ip):
            errors.append(f"The attacker source_ip '{source_ip}' must be a public IPv4 address.")
        elif role != "attacker" and source_ip != "AWS Internal" and not _is_ipv4(source_ip):
            errors.append(f"Actor {actor_id or index} source_ip must be an IPv4 address or AWS Internal.")

        actors.append({
            "id": actor_id,
            "role": role,
            "principal": principal,
            "source_ip": source_ip,
            "user_agent": clean_value(item.get("user_agent")) or _default_user_agent(role),
            "mfa": _normalise_mfa(item.get("mfa")),
        })

    ids = [actor["id"] for actor in actors]
    if len(set(ids)) != len(ids):
        errors.append("Actor ids must be unique.")

    roles = [actor["role"] for actor in actors]
    if roles.count("attacker") != 1:
        errors.append("There must be exactly one attacker actor.")
    if "background" not in roles:
        errors.append("Add at least one background actor doing normal work.")
    if "responder" not in roles:
        errors.append("Add one responder actor (the security team).")

    return actors


def _normalise_events(raw: Any, actors: list[dict[str, Any]], errors: list[str]) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        errors.append("events must be a non-empty list.")
        return []

    by_id = {actor["id"]: actor for actor in actors}
    events = []

    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append(f"events[{index}] must be an object.")
            continue

        name = clean_value(item.get("event_name"))
        if "lambda" in clean_value(item.get("event_source")).lower():
            name = LAMBDA_VERSIONED.get(name, name)
        actor = by_id.get(clean_value(item.get("actor")))
        time = clean_value(item.get("time"))
        result = "Failure" if clean_value(item.get("result")).lower() in {"failure", "failed", "fail"} else "Success"
        error_code = clean_value(item.get("error_code"))
        error_code = "" if error_code in {"-", "none", "None"} else error_code

        # Background activity is replaceable noise: an invalid background call is
        # dropped (and topped up later) instead of rejecting the whole timeline.
        noise = actor is not None and actor["role"] == "background"
        event_errors: list[str] = []

        if not _EVENT_NAME.match(name) or name.lower() in VAGUE_EVENT_NAMES:
            event_errors.append(f"Event '{name}' is not a CloudTrail eventName; use the exact API name, e.g. ConsoleLogin or CreateAccessKey.")
        elif actor is None:
            event_errors.append(f"Event {name} names an unknown actor '{item.get('actor')}'.")
        elif not _TIME.match(time):
            event_errors.append(f"Event {name} time '{time}' must look like 2023-10-01T10:12:10Z.")

        source = "" if event_errors else _event_source(name, item.get("event_source"), event_errors)
        if not source:
            if not noise:
                errors.extend(event_errors)
            continue

        if result == "Failure" and not error_code:
            error_code = "Failed authentication" if name == "ConsoleLogin" else "AccessDenied"

        turn = _parse_turn(item.get("turn"))
        if turn not in TURNS or actor["role"] == "background":
            turn = None  # Background activity is never key evidence; bad labels are reassigned later.

        mfa = _normalise_mfa(item.get("mfa")) or (actor["mfa"] if name == "ConsoleLogin" else "")
        context = f"mfa {mfa} {result} {error_code}"
        suspicious = actor["role"] == "attacker" and (
            item.get("suspicious") is True or event_is_suspicious(name, context))

        if actor["role"] == "background" and event_is_suspicious(name, context):
            errors.append(f"Background actor {actor['id']} performs {name}, which looks like incident activity; give it routine calls only.")
            continue

        resource = clean_value(item.get("resource")).strip("-").strip()[:80]
        events.append({
            "time": time,
            "actor": actor["id"],
            "event_name": name,
            "event_source": source,
            "result": result,
            "error_code": error_code or "-",
            "mfa": mfa,
            "resource": resource,
            "turn": turn,
            "suspicious": suspicious,
        })

    return sorted(events, key=lambda event: event["time"])


def _event_source(name: str, raw_source: Any, errors: list[str]) -> str:
    """The event's service endpoint, or "" after recording why the name is not real."""

    source = clean_value(raw_source).lower()
    source = SOURCE_ALIASES.get(source, source)
    # Some names exist under several services (GetFindings, TagResource); keep a
    # source the name really belongs to, otherwise correct it.
    if name not in EVENT_CATALOGUE.get(source, ()):
        source = KNOWN_EVENT_SOURCES.get(name) or source

    if not _EVENT_SOURCE.match(source):
        errors.append(f"Event {name} event_source '{source}' must be an AWS endpoint such as iam.amazonaws.com.")
        return ""
    if source in EVENT_CATALOGUE and name not in EVENT_CATALOGUE[source]:
        closest = difflib.get_close_matches(name, sorted(EVENT_CATALOGUE[source]), n=3, cutoff=0.6)
        hint = (f"the closest real ones are {', '.join(closest)}" if closest
                else f"use an actual eventName such as {SUGGESTED_EVENTS[source]}")
        errors.append(f"{name} is not a real {source} API; {hint}.")
        return ""

    return source


def _add_baseline_activity(events: list[dict[str, Any]], actors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The attacker principal's normal work the day before, from its usual IP.

    This is the partial evidence: the right principal, but ordinary activity that
    does not show the suspicious step. Code generates it so it stays mundane. The
    owner works with MFA; without it the attacker's mfa=false shows on these rows
    and reads as a red flag.
    """

    attacker = next(actor for actor in actors if actor["role"] == "attacker")
    taken = {event["event_name"] for event in events}
    day_before = datetime.strptime(events[0]["time"][:10], "%Y-%m-%d") - timedelta(days=1)
    baseline_ip = _baseline_ip(actors)
    calls = [(name, source) for name, source in BASELINE_CALL_POOL if name not in taken][:3]

    for index, (name, source) in enumerate(calls):
        time = day_before + timedelta(hours=9, minutes=14 + 23 * index, seconds=5 + 11 * index)
        events.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "actor": attacker["id"], "event_name": name,
                       "event_source": source, "result": "Success", "error_code": "-", "mfa": "true", "resource": "",
                       "turn": None, "suspicious": False, "baseline": True, "source_ip": baseline_ip})

    return sorted(events, key=lambda event: event["time"])


def _baseline_ip(actors: list[dict[str, Any]]) -> str:
    """An office address near the background actors' network, never the attacker's."""

    attacker_ip = next(actor["source_ip"] for actor in actors if actor["role"] == "attacker")
    network = next((actor["source_ip"] for actor in actors
                    if actor["role"] == "background" and _is_ipv4(actor["source_ip"])), "203.0.113.20")
    prefix, last = network.rsplit(".", 1)
    candidate = f"{prefix}.{(int(last) + 37) % 250 + 2}"
    return candidate if candidate != attacker_ip else f"{prefix}.{(int(last) + 91) % 250 + 2}"


def _assign_turns(events: list[dict[str, Any]], actors: list[dict[str, Any]],
                  errors: list[str]) -> list[dict[str, Any]]:
    """Keep the model's turn labels when complete; otherwise derive them from time order.

    The attacker's incident steps (events it labelled with a turn or flagged
    suspicious) are spread over turns 1-4 in chronological order and the
    responder's events become turn 5, matching the stage order of the exercise.
    """

    role_of = {actor["id"]: actor["role"] for actor in actors}
    # Every call the model gives the attacker is part of the attack; the harmless
    # baseline is added by code afterwards.
    steps = [e for e in events if role_of[e["actor"]] == "attacker"]
    responses = [e for e in events if role_of[e["actor"]] == "responder"]

    labelled = (
        all(any(e["turn"] == turn for e in steps) for turn in (1, 2, 3))
        and any(e["turn"] in (4, 5) for e in steps)
    )

    if labelled:
        current = min(e["turn"] for e in steps if e["turn"])
        for event in steps:
            current = event["turn"] or current
            event["turn"] = current  # an unlabelled call belongs to the step before it
    else:
        if len(steps) < 4:
            errors.append("Give the attacker at least 4 incident steps (first signal, follow-up, spread, persistence).")
        if errors:
            return events

        bounds = [round(i * len(steps) / 4) for i in range(5)]
        for turn in range(1, 5):
            for event in steps[bounds[turn - 1]:bounds[turn]]:
                event["turn"] = turn
        for event in responses:
            event["turn"] = 5

    if not responses:
        errors.append("Add at least one responder recovery event.")
        return events

    for event in steps:
        event["suspicious"] = True  # the attacker's incident steps are the evidence
        event["turn"] = min(event["turn"], 4)
    for event in responses:
        event["turn"] = 5  # recovery evidence; the containment decision is the learner's

    return events


def _keep_routine_background(events: list[dict[str, Any]], actors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep background calls that are routine and do not touch the suspect.

    Changes such as DetachUserPolicy on the suspect's user are part of the story,
    not unrelated activity; they are dropped and replaced with routine noise.
    """

    role_of = {actor["id"]: actor["role"] for actor in actors}
    attacker = next(actor for actor in actors if actor["role"] == "attacker")
    suspect = _short(attacker["principal"]).lower()
    key_names = {event["event_name"] for event in events if event["turn"]}

    def routine(event: dict[str, Any]) -> bool:
        name = event["event_name"]
        return ((name.startswith(ROUTINE_PREFIXES) or name in ROUTINE_WRITES)
                and name not in key_names and suspect not in event["resource"].lower())

    return [event for event in events if role_of[event["actor"]] != "background" or routine(event)]


BACKGROUND_NOISE_POOL = (("DescribeInstances", "ec2.amazonaws.com"), ("GetObject", "s3.amazonaws.com"),
                         ("PutMetricData", "monitoring.amazonaws.com"), ("ListObjectsV2", "s3.amazonaws.com"),
                         ("DescribeVolumes", "ec2.amazonaws.com"), ("GetBucketLocation", "s3.amazonaws.com"))


def _add_background_noise(events: list[dict[str, Any]], actors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top background activity up to three normal calls, spread across the window."""

    background = next((actor for actor in actors if actor["role"] == "background"), None)
    existing = [event for event in events if background and event["actor"] == background["id"]]
    other_background = [event for event in events if event["actor"] != (background or {}).get("id")
                        and any(a["id"] == event["actor"] and a["role"] == "background" for a in actors)]
    missing = 3 - len(existing) - len(other_background)

    if background is None or missing <= 0:
        return events

    taken = {event["event_name"] for event in events}
    first = datetime.strptime(events[0]["time"], "%Y-%m-%dT%H:%M:%SZ")
    last = datetime.strptime(events[-1]["time"], "%Y-%m-%dT%H:%M:%SZ")
    pool = [(name, source) for name, source in BACKGROUND_NOISE_POOL if name not in taken]

    for index, (name, source) in enumerate(pool[:missing]):
        time = first + (last - first) * (index + 1) / (missing + 1) + timedelta(seconds=13)
        events.append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "actor": background["id"], "event_name": name,
                       "event_source": source, "result": "Success", "error_code": "-", "mfa": "", "resource": "",
                       "turn": None, "suspicious": False})

    return sorted(events, key=lambda event: event["time"])


def _check_event_structure(events: list[dict[str, Any]], actors: list[dict[str, Any]], errors: list[str]) -> None:
    role_of = {actor["id"]: actor["role"] for actor in actors}
    key = [event for event in events if event["turn"]]
    generic = [event for event in events if event.get("baseline")]
    background = [event for event in events if role_of[event["actor"]] == "background"]

    for turn in TURNS:
        if not any(event["turn"] == turn for event in key):
            errors.append(f"No event is marked turn {turn}.")

    if not any(event["turn"] == 5 and role_of[event["actor"]] == "responder" for event in key):
        errors.append("Turn 5 needs at least one responder recovery event.")
    if any(role_of[event["actor"]] == "background" for event in key):
        errors.append("Turn events must come from the attacker or the responder.")
    if len(generic) < 2:
        errors.append("The timeline leaves no everyday API names free for the principal's baseline activity.")
    if len(background) < 3:
        errors.append("Give background actors at least 3 normal events.")



def _normalise_findings(raw: Any, errors: list[str]) -> list[dict[str, Any]]:
    findings = []

    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue

        finding_type = clean_value(item.get("finding_type"))
        if not _FINDING_TYPE.match(finding_type):
            errors.append(f"Finding type '{finding_type}' is not a real GuardDuty type; use one such as {REAL_FINDING_TYPES}.")
            continue

        description = clean_value(item.get("description"))[:220]
        if any(phrase in description.lower() for phrase in CONCLUSION_PHRASES):
            description = ""

        turn = _parse_turn(item.get("turn"))
        findings.append({
            "finding_type": finding_type,
            "severity": _severity_label(item.get("severity")),
            "resource": clean_value(item.get("resource"))[:80],
            "description": description,
            "turn": turn if turn in TURNS else None,
        })

    if not any(finding["severity"] in {"Medium", "High"} for finding in findings):
        errors.append("Add at least one Medium or High finding about the attacker's activity.")

    return findings


def _normalise_cost(raw: Any, errors: list[str]) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}

    try:
        baseline = float(raw.get("baseline_daily_usd"))
        incident = float(raw.get("incident_daily_usd"))
    except (TypeError, ValueError):
        errors.append("cost needs numeric baseline_daily_usd and incident_daily_usd.")
        return {}

    if baseline <= 0 or incident <= 0:
        errors.append("cost values must be positive.")

    return {
        "baseline_daily_usd": baseline,
        "incident_daily_usd": incident,
        "service": clean_value(raw.get("service")) or "EC2",
        "normal_service": clean_value(raw.get("normal_service")) or "S3",
    }


# ---------------------------------------------------------------------------
# Evidence for a turn
# ---------------------------------------------------------------------------

def build_turn_evidence(timeline: dict[str, Any], scenario_config: dict[str, Any], turn: int) -> list[dict[str, Any]]:
    """Strong, partial and weak evidence items for one turn, with role-free ids."""

    items = []

    for role, template in turn_template_plan(scenario_config, turn).items():
        facts = _VIEWS[template](timeline, role, turn)
        items.append({
            "id": EVIDENCE_IDS[template],
            "title": _default_title(template, facts),
            "type": template,
            "template": template,
            "summary": _caption(template, facts),
            "why_it_may_matter": WHY_IT_MAY_MATTER[template],
            "support_role": role,
            "facts": facts,
        })

    return items


def turn_template_plan(scenario_config: dict[str, Any], turn: int) -> dict[str, str]:
    """Templates per role from the scenario strategy, made distinct within the turn."""

    strategy = (scenario_config.get("evidence_template_strategy") or {}).get("role_templates_by_turn") or {}
    planned = strategy.get(str(turn)) or strategy.get(turn) or DEFAULT_TEMPLATE_PLAN
    plan: dict[str, str] = {}

    for role in ROLES:
        template = clean_value(planned.get(role) or DEFAULT_TEMPLATE_PLAN[role]).lower()

        if template not in EVIDENCE_IDS or template in plan.values():
            template = next(t for t in TEMPLATE_PREFERENCE if t not in plan.values())

        plan[role] = template

    return plan


def apply_timeline_evidence(generated_turn: dict[str, Any], items: list[dict[str, Any]]) -> None:
    """Replace generated evidence with timeline evidence, keeping the model's wording.

    Facts, templates and roles always come from the timeline. The model's title and
    why_it_may_matter are kept when present and role-neutral; summaries are captions
    of the rendered facts so the learner never reads a claim the screenshot lacks.
    """

    authored = [item for item in generated_turn.get("evidence_facts") or [] if isinstance(item, dict)]
    by_id = {clean_value(item.get("id")): item for item in authored}
    by_template = {clean_value(item.get("template") or item.get("type")).lower(): item for item in authored}
    final = []

    for item in items:
        source = by_id.get(item["id"]) or by_template.get(item["template"]) or {}
        merged = copy.deepcopy(item)
        merged["title"] = _neutral_text(source.get("title"), 80) or item["title"]
        merged["why_it_may_matter"] = _neutral_text(source.get("why_it_may_matter"), 220) or item["why_it_may_matter"]
        final.append(merged)

    generated_turn["evidence_facts"] = final


def timeline_prompt_block(timeline: dict[str, Any], items: list[dict[str, Any]], anchor: dict[str, Any]) -> str:
    """Tells the coach which evidence is fixed so its turn text matches the screenshots."""

    attacker = _actor_by_role(timeline, "attacker")
    evidence = [{"id": item["id"], "template": item["template"], "support_role": item["support_role"],
                 "shows": item["summary"]} for item in items]
    terms = ", ".join(f'"{term}"' for term in anchor["terms"])

    return f"""
FIXED EVIDENCE FOR THIS TURN (built from the scenario's incident timeline):
{json.dumps(evidence, indent=2)}

Incident identities: the suspicious principal is {attacker['principal']} from {attacker['source_ip']}.
- Return exactly these three evidence items in evidence_facts, with the same id, template and support_role.
- Write only a short title and a neutral why_it_may_matter for each. The facts are filled in by the app.
- Never call an item strong, partial, weak, best or a distractor in learner-visible text.
- Keep the briefing and known_context consistent with these identities and events.

ACTIONS (checked by the app; a turn that breaks these is rejected):
- The best action MUST contain "{anchor['terms'][0]}" in its title or description, for example: "{anchor['example']}".
- The partial and weak actions must NOT contain {terms}. Write the partial action about what the partial item shows, and the weak action as plausible but aimed elsewhere.
"""


def action_anchor(timeline: dict[str, Any], items: list[dict[str, Any]], turn: int) -> dict[str, Any]:
    """The one thing only the strong item shows; the best action must name it.

    The partial item shows the same principal's routine calls, so a vague best
    action ("look for activity by <suspect>") is answered by it too and the judge
    rightly grades that pair Strong. Naming the strong item's key event or
    condition leaves the partial and weak items visibly missing it.
    """

    strong = next(item for item in items if item["support_role"] == "strong")
    facts, template = strong["facts"], strong["template"]

    if template == "guardduty":
        shows = f"the {facts['severity']}-severity GuardDuty finding {facts['finding_type']} for {_short(facts['principal'])}"
        return {"terms": ["GuardDuty"], "title": "Investigate GuardDuty Finding", "shows": shows,
                "example": f"Investigate the {facts['severity']}-severity GuardDuty finding for {_short(facts['principal'])}"}
    if template == "access_key":
        # Identified by owner and last-use IP, not the 20-character key ID: the VLM
        # misread one ID by a few characters and the judge then called the right
        # key the wrong one. The partial key has the same owner but the office IP.
        shows = (f"an access key owned by {_short(facts['owner'])}, {facts['status'].lower()}, "
                 f"last used from {facts['source_ip']}")
        return {"terms": ["access key"], "title": "Review Access Key Use", "shows": shows,
                "example": f"Review the recent use of {_short(facts['owner'])}'s access key "
                           f"from {facts['source_ip']}"}
    if template == "billing":
        shows = f"the {facts['change']} {facts['largest_service']} spend increase ({facts['current_spend']} a day)"
        return {"terms": ["spend", "cost", "billing"], "title": "Investigate Spend Increase", "shows": shows,
                "example": f"Investigate the {facts['change']} {facts['largest_service']} spend increase"}

    # Event views: this turn's headline step, among the rows the screenshot shows.
    shown = _squash(json.dumps(facts))
    events = _role_events(timeline, "strong", turn)
    event = _headline_event([e for e in events if _squash(e["event_name"]) in shown] or events, turn)
    verb = "Verify" if turn == TURNS[-1] else "Investigate"
    # Lambda's versioned eventNames (UpdateFunctionCode20150331v2): the coach
    # writes the plain API name, and the action check rejected that.
    name = re.sub(r"\d{8}(v\d+)?$", "", event["event_name"]) or event["event_name"]
    principal = _short(_actor(timeline, event["actor"])["principal"])
    # The action frames the call as the centre of an investigation rather than
    # the only thing that counts, and leaves the IP to the marking scheme: an
    # action asking for one exact call made the judge mark partial evidence Weak.
    return {"terms": [name], "title": f"{verb} Activity Around {name}",
            "shows": f"the {event['event_name']} call by {principal} from {_event_ip(timeline, event)}",
            "example": f"{verb} the recent activity around the {name} call by {principal}"}


def marking_scheme(timeline: dict[str, Any], items: list[dict[str, Any]], anchor: dict[str, Any]) -> dict[str, str]:
    """The security judge's reference for this turn: what each support level looks like.

    Built from the timeline like the evidence itself, and saved as the turn's
    expected_outcomes. It describes content, never which screenshot is which;
    the judge still has to read the selected screenshot and decide which
    description it matches. Learners never see it.
    """

    suspect = _short(_actor_by_role(timeline, "attacker")["principal"])
    by_role = {item["support_role"]: item for item in items}

    return {
        "strong_support": f"Shows {anchor['shows']}. This is the step the decision this turn depends on.",
        "partial_support": (f"Related to {suspect} or the incident but incomplete: {_describe(by_role['partial'])}. "
                            f"It does not show {anchor['shows']}."),
        # Both lower levels lack the strong step, so both say so: when only Partial did,
        # the judge matched a weak item to Partial on that sentence (26 Sep).
        "weak_support": (f"Plausible cloud evidence that does not show {suspect}'s incident activity: "
                         f"{_describe(by_role['weak'])}. It does not show {suspect} or {anchor['shows']}."),
    }


def _describe(item: dict[str, Any]) -> str:
    """One line naming who and what an evidence item shows, from its facts."""

    facts, template = item["facts"], item["template"]

    if template == "guardduty":
        return (f"a {facts['severity']}-severity {facts['finding_type']} finding on {facts['resource']} "
                f"(principal {_short(facts['principal'])})")
    if template == "access_key":
        return (f"an access key owned by {_short(facts['owner'])}, {facts['status'].lower()}, "
                f"last used from {facts['source_ip']}")
    if template == "billing":
        return (f"{facts['current_spend']} daily spend ({facts['change']} against {facts['previous_average']}), "
                f"largest service {facts['largest_service']}")

    rows = _event_rows(item)
    names = ", ".join(dict.fromkeys(row[1] for row in rows))
    principals = ", ".join(dict.fromkeys(_short(row[2]) for row in rows))
    ips = ", ".join(dict.fromkeys(row[3] for row in rows))
    dates = ", ".join(dict.fromkeys(row[0][:10] for row in rows))
    return f"calls {names} by {principals} from {ips} on {dates}"


def _event_rows(item: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """(time, event name, principal, source IP) for each row an event view shows."""

    facts = item["facts"]

    if item["template"] == "cloudtrail":
        return [(row[0], row[1], row[2], row[3]) for row in facts["related_events"]]
    if item["template"] == "iam_activity":
        return [(row[0], row[2], row[1], row[3]) for row in facts["activity_rows"]]

    rows = []
    for time, _stream, message in facts["log_rows"]:
        match = re.match(r"(\S+) \S+ by (\S+) from ([^;\s]+)", message)
        if match:
            rows.append((time, match.group(1), match.group(2), match.group(3)))
    return rows


def action_anchor_problems(actions: Any, anchor: dict[str, Any]) -> list[str]:
    if not isinstance(actions, list):
        return []

    problems = []
    terms = " / ".join(anchor["terms"])

    for action in actions:
        role = normalise_choice_role(action.get("choice_role")) if isinstance(action, dict) else ""
        if role == "best" and not _names_anchor(action, anchor):
            problems.append(f"The best action must name {anchor['terms'][0]}, for example: \"{anchor['example']}\".")
        elif role in {"partial", "weak"} and _names_anchor(action, anchor):
            problems.append(f"The {role} action mentions {terms}; only the best action may name it.")

    return problems


def remove_decoy_mentions(turn: dict[str, Any], timeline: dict[str, Any], items: list[dict[str, Any]]) -> list[str]:
    """Removes briefing sentences and known-context lines that name the weak item's principal.

    The weak item shows someone uninvolved. When the coach wrote "Access key ...
    for CI-Deploy is active" into the known context, that principal read as part
    of the incident and the judge graded CI-Deploy's key Partial (26 Sep, twice).
    Names come from this turn's weak item, so rebuilds and swapped names work.
    Returns what was removed, for the turn's diagnostics.
    """

    config = turn.get("turn_config") or {}
    weak = next((item for item in items if item.get("support_role") == "weak"), None)
    if not weak:
        return []

    suspect = _actor_by_role(timeline, "attacker")["principal"]
    facts = weak["facts"]
    principals = {facts.get("owner"), facts.get("principal")}
    if weak["template"] in {"cloudtrail", "iam_activity", "cloudwatch"}:
        principals |= {row[2] for row in _event_rows(weak)}
    names = {_short(p) for p in principals if p and _short(p) != _short(suspect)}
    if facts.get("access_key_id"):
        names.add(facts["access_key_id"])
    if not names:
        return []

    pattern = re.compile(r"(?<![\w.-])(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
                         + r")(?![\w-])", re.IGNORECASE)
    removed = []

    context = config.get("known_context")
    if isinstance(context, list):
        removed += [line for line in context if isinstance(line, str) and pattern.search(line)]
        config["known_context"] = [line for line in context if not (isinstance(line, str) and pattern.search(line))]

    briefing = config.get("briefing")
    if isinstance(briefing, str):
        sentences = re.split(r"(?<=[.!?])\s+", briefing.strip())
        kept = [sentence for sentence in sentences if not pattern.search(sentence)]
        if kept:  # a briefing about nothing but the decoy is left for the reviewer
            removed += [sentence for sentence in sentences if pattern.search(sentence)]
            config["briefing"] = " ".join(kept)

    return removed


def enforce_action_anchor(actions: Any, anchor: dict[str, Any]) -> str | None:
    """Makes the best action name the anchor, and says what was changed.

    Repaired at once rather than sent back to the coach: each retry cost a full
    turn generation (about 4 minutes), and in a live session every anchor retry
    ended in the same repair. Returns None when the coach got it right.
    """

    if not action_anchor_problems(actions, anchor):
        return None

    holders = [action for action in actions if isinstance(action, dict) and _names_anchor(action, anchor)]
    best = next((action for action in actions if isinstance(action, dict)
                 and normalise_choice_role(action.get("choice_role")) == "best"), None)

    if best is None:
        return "no best action to repair"
    if len(holders) == 1:
        # The coach wrote the right action under the wrong label (turn 1 of the
        # identity scenario swapped best and partial): swap the labels back.
        holders[0]["choice_role"], best["choice_role"] = "best", holders[0]["choice_role"]
        return "swapped best and " + str(best["choice_role"])
    if not holders:
        best["title"] = anchor["title"]
        best["description"] = anchor["example"] + "."
        return "rewrote the best action"
    return "several actions name the anchor; left unchanged"


def _names_anchor(action: dict[str, Any], anchor: dict[str, Any]) -> bool:
    text = _squash(f"{action.get('title', '')} {action.get('description', '')}")
    return any(_squash(term) in text for term in anchor["terms"])


def _squash(text: str) -> str:
    """'Create Access Key' and 'CreateAccessKey' compare equal."""

    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def load_incident_timeline(runtime_dir: Path, scenario_id: str | None = None) -> dict[str, Any] | None:
    path = runtime_dir / TIMELINE_FILE

    if not path.exists():
        return None

    timeline = json.loads(path.read_text(encoding="utf-8"))

    if scenario_id and clean_value(timeline.get("scenario_id")) != clean_value(scenario_id):
        return None  # Left over from another scenario.

    return timeline


def save_incident_timeline(runtime_dir: Path, timeline: dict[str, Any]) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / TIMELINE_FILE).write_text(json.dumps(timeline, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Views: one function per screenshot template
# ---------------------------------------------------------------------------

def _role_events(timeline: dict[str, Any], role: str, turn: int) -> list[dict[str, Any]]:
    roles = {actor["id"]: actor["role"] for actor in timeline["actors"]}
    events = timeline["events"]

    if role == "strong":
        # Turns 1-4 show the attacker's trail so far; recovery shows the response.
        return [event for event in events if event["turn"] and (
            event["turn"] == turn or (turn < 5 and event["turn"] < turn and roles[event["actor"]] == "attacker"))]
    if role == "partial":
        return [event for event in events if event.get("baseline")]
    return [event for event in events if roles[event["actor"]] == "background"]


def _cloudtrail_view(timeline: dict[str, Any], role: str, turn: int) -> dict[str, Any]:
    events = _role_events(timeline, role, turn)[-4:]
    primary = _headline_event(events, turn) if role == "strong" else events[0]
    actor = _actor(timeline, primary["actor"])

    return {
        "event_name": primary["event_name"],
        "event_source": primary["event_source"],
        "user": actor["principal"],
        "source_ip": _event_ip(timeline, primary),
        "mfa": primary["mfa"] or actor["mfa"] or "Unknown",
        "event_time": primary["time"],
        "region": timeline["region"],
        "error_code": primary["error_code"],
        "risk_signal": primary["result"],
        "event_id": _stable_uuid(timeline, primary),
        "user_agent": actor["user_agent"],
        "recipient_account_id": timeline["account_id"],
        "request_parameters": f"resource={primary['resource']}" if primary["resource"] else "-",
        "resources": primary["resource"] or "None recorded",
        "related_events": [
            [event["time"], event["event_name"], _actor(timeline, event["actor"])["principal"],
             _event_ip(timeline, event), _result_label(event)]
            for event in events
        ],
    }


def _iam_activity_view(timeline: dict[str, Any], role: str, turn: int) -> dict[str, Any]:
    events = _role_events(timeline, role, turn)[-5:]
    actor_id = max({event["actor"] for event in events}, key=lambda a: sum(e["actor"] == a for e in events))
    actor = _actor(timeline, actor_id)
    rows = [event for event in events if event["actor"] == actor_id][:5]
    policy = next((event for event in reversed(rows) if "Policy" in event["event_name"]), None)
    keys = [key for key in _access_keys(timeline, turn) if key["owner"] == actor["principal"]]
    active = sum(key["status"] == "Active" for key in keys)

    return {
        "principal": actor["principal"],
        "source_ip": _event_ip(timeline, rows[-1]),
        # The rows' own MFA first: baseline rows differ from the actor's attack-time value.
        "mfa": next((e["mfa"] for e in reversed(rows) if e["mfa"]), "") or actor["mfa"] or "Unknown",
        "policy_change": f"{policy['event_name']} at {policy['time']}" if policy else "None in this window",
        "access_key_status": f"{active} active" if keys else "No keys",
        "risk_flags": [],
        "activity_rows": [[event["time"], actor["principal"], event["event_name"], _event_ip(timeline, event)]
                          for event in rows],
    }


def _cloudwatch_view(timeline: dict[str, Any], role: str, turn: int) -> dict[str, Any]:
    events = _role_events(timeline, role, turn)[-5:]
    actor_ids = {event["actor"] for event in events}
    stream = f"{timeline['account_id']}_CloudTrail_{timeline['region']}"

    if len(actor_ids) == 1:
        principal = _actor(timeline, next(iter(actor_ids)))["principal"]
        query = f'fields @timestamp, @message | filter userIdentity.arn = "{principal}" | sort @timestamp asc'
    else:
        names = ", ".join(sorted({f'"{event["event_name"]}"' for event in events}))
        query = f"fields @timestamp, @message | filter eventName in [{names}] | sort @timestamp asc"

    return {
        "query": query,
        "log_group": f"aws-cloudtrail-logs-{timeline['account_id']}",
        "matched_records": str(len(events)),
        "scanned_bytes": f"{0.3 + 0.2 * len(events):.1f} MB",
        "time_range": f"{events[0]['time'][:10]} {events[0]['time'][11:16]}-{_minute_after(events[-1]['time'])} UTC",
        "alarm_state": "Complete",
        "log_rows": [[event["time"], stream, _log_message(timeline, event)] for event in events],
    }


def _guardduty_view(timeline: dict[str, Any], role: str, turn: int) -> dict[str, Any]:
    attacker = _actor_by_role(timeline, "attacker")
    background = _actor_by_role(timeline, "background")
    # The attack so far: not the baseline day, and no later turn's steps.
    attack_times = [event["time"] for event in timeline["events"] if event["actor"] == attacker["id"]
                    and event["turn"] and (event["turn"] <= turn or turn == TURNS[-1])]

    if role == "strong":
        serious = [f for f in timeline["findings"] if f["severity"] in {"Medium", "High"}]
        finding = next((f for f in serious if f["turn"] == turn), serious[0])
        return _finding_facts(finding["finding_type"], finding["severity"], finding["resource"] or attacker["principal"],
                              attacker["principal"], attacker["source_ip"], attack_times,
                              finding["description"] or f"{finding['finding_type']} was reported for {_short(attacker['principal'])}.")

    if role == "partial":
        baseline = _role_events(timeline, "partial", turn)
        return _finding_facts("Discovery:IAMUser/AnomalousBehavior", "Low", attacker["principal"], attacker["principal"],
                              _event_ip(timeline, baseline[-1]), [event["time"] for event in baseline],
                              f"Read-only API calls by {_short(attacker['principal'])} differed from its usual pattern.")

    instance = f"i-0{_digest(timeline, 'instance')[:16]}"
    scanner_ip = f"203.0.113.{int(_digest(timeline, 'scanner')[:2], 16) % 200 + 20}"
    times = [event["time"] for event in _role_events(timeline, "weak", turn)]
    return _finding_facts("Recon:EC2/PortProbeUnprotectedPort", "Low", instance, background["principal"], scanner_ip,
                          times, f"Port 22 on {instance} was probed by a known scanner.")


def _access_key_view(timeline: dict[str, Any], role: str, turn: int) -> dict[str, Any]:
    keys = _access_keys(timeline, turn)
    key = {"strong": keys[0], "partial": keys[1], "weak": keys[2]}[role]
    return {field: key[field] for field in ("access_key_id", "owner", "status", "last_used_service",
                                            "last_used_region", "last_used_time", "source_ip")}


def _billing_view(timeline: dict[str, Any], role: str, turn: int) -> dict[str, Any]:
    cost = timeline["cost"]
    baseline = cost["baseline_daily_usd"]
    incident = cost["incident_daily_usd"]
    current = {"strong": incident, "partial": baseline + (incident - baseline) * 0.35, "weak": baseline * 1.03}[role]
    change = (current - baseline) / baseline * 100

    return {
        "current_spend": f"${current:,.2f}",
        "previous_average": f"${baseline:,.2f}",
        "largest_service": cost["normal_service"] if role == "weak" else cost["service"],
        "region": timeline["region"],
        "change": f"{'+' if change >= 0 else ''}{change:.0f}%",
    }


_VIEWS = {
    "cloudtrail": _cloudtrail_view,
    "iam_activity": _iam_activity_view,
    "cloudwatch": _cloudwatch_view,
    "guardduty": _guardduty_view,
    "access_key": _access_key_view,
    "billing": _billing_view,
}


def _access_keys(timeline: dict[str, Any], turn: int) -> list[dict[str, Any]]:
    """Attacker's key, the same principal's older key, and a background key."""

    attacker = _actor_by_role(timeline, "attacker")
    background = _actor_by_role(timeline, "background")
    events = timeline["events"]
    attacker_events = [event for event in events if event["actor"] == attacker["id"] and not event.get("baseline")]
    baseline = [event for event in events if event.get("baseline")]
    background_events = [event for event in events if event["actor"] == background["id"]]
    # Evidence shows the state before this turn's decision; recovery shows the outcome.
    disabled = any(event["event_name"].lower() in KEY_CONTROL_EVENTS and event["turn"]
                   and (event["turn"] < turn or turn == TURNS[-1]) for event in events)
    last = attacker_events[-1]
    background_last = background_events[-1] if background_events else None

    def key_id(owner: str, label: str) -> str:
        prefix = "ASIA" if ":role/" in owner or "assumed-role" in owner else "AKIA"
        return prefix + _digest(timeline, label).upper()[:16].replace("0", "Q")

    return [
        {"access_key_id": key_id(attacker["principal"], "attacker-key"), "owner": attacker["principal"],
         "status": "Inactive" if disabled else "Active", "last_used_service": last["event_source"],
         "last_used_region": timeline["region"], "last_used_time": last["time"], "source_ip": attacker["source_ip"]},
        {"access_key_id": key_id(attacker["principal"], "older-key"), "owner": attacker["principal"],
         "status": "Active", "last_used_service": baseline[-1]["event_source"], "last_used_region": timeline["region"],
         "last_used_time": baseline[-1]["time"], "source_ip": _event_ip(timeline, baseline[-1])},
        {"access_key_id": key_id(background["principal"], "background-key"), "owner": background["principal"],
         "status": "Active",
         "last_used_service": background_last["event_source"] if background_last else "s3.amazonaws.com",
         "last_used_region": timeline["region"],
         "last_used_time": background_last["time"] if background_last else events[0]["time"],
         "source_ip": background["source_ip"]},
    ]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _headline_event(events: list[dict[str, Any]], turn: int) -> dict[str, Any]:
    """This turn's newest step, preferring one that is sensitive on its own."""

    current = [event for event in events if event["turn"] == turn] or events
    sensitive = [event for event in current if event_is_suspicious(event["event_name"], _log_message_context(event))]
    return (sensitive or current)[-1]


def _log_message_context(event: dict[str, Any]) -> str:
    return f"mfa {event['mfa']} {event['result']} {event['error_code']}"


def _finding_facts(finding_type, severity, resource, principal, remote_ip, times, summary) -> dict[str, Any]:
    times = sorted(times) or ["Unknown"]
    return {"finding_type": finding_type, "severity": severity, "resource": resource, "principal": principal,
            "remote_ip": remote_ip, "first_seen": times[0], "last_seen": times[-1], "summary": summary}


def _log_message(timeline: dict[str, Any], event: dict[str, Any]) -> str:
    actor = _actor(timeline, event["actor"])
    message = f"{event['event_name']} {event['result'].lower()} by {actor['principal']} from {_event_ip(timeline, event)}"

    if event["result"] == "Failure":
        message += f" errorCode={event['error_code']}"
    if event["event_name"] == "ConsoleLogin" and event["mfa"]:
        message += f"; mfaAuthenticated={event['mfa']}"
    if event["resource"]:
        message += f"; resource={event['resource']}"

    return message


def _result_label(event: dict[str, Any]) -> str:
    if event["result"] == "Failure":
        return "Failure"
    if event["event_name"] == "ConsoleLogin" and event["mfa"]:
        return f"MFA {event['mfa']}"
    return "Success"


def _caption(template: str, facts: dict[str, Any]) -> str:
    if template == "cloudtrail":
        return (f"CloudTrail record of {facts['event_name']} by {_short(facts['user'])} "
                f"from {facts['source_ip']} at {facts['event_time'][11:16]} UTC.")
    if template == "iam_activity":
        names = ", ".join(dict.fromkeys(row[2] for row in facts["activity_rows"]))
        return f"Recent IAM activity for {_short(facts['principal'])}: {names}."
    if template == "cloudwatch":
        return f"Logs Insights query returning {facts['matched_records']} CloudTrail log lines."
    if template == "guardduty":
        return f"GuardDuty {facts['severity']} finding {facts['finding_type']} for {_short(facts['resource'])}."
    if template == "access_key":
        return (f"Access key {facts['access_key_id']} for {_short(facts['owner'])} is {facts['status']}; "
                f"last used {facts['last_used_time'][:16].replace('T', ' ')} from {facts['source_ip']}.")
    return (f"Daily spend {facts['current_spend']} against a {facts['previous_average']} average; "
            f"largest service {facts['largest_service']}.")


def _default_title(template: str, facts: dict[str, Any]) -> str:
    return {
        "cloudtrail": f"CloudTrail Event: {facts.get('event_name', '')}",
        "iam_activity": f"IAM Activity: {_short(facts.get('principal', ''))}",
        "cloudwatch": "CloudWatch Logs Insights Query",
        "guardduty": "GuardDuty Finding",
        "access_key": "Access Key Details",
        "billing": "Daily Billing Summary",
    }[template]


def _neutral_text(value: Any, limit: int) -> str:
    text = clean_value(value)
    if not text or len(text) > limit:
        return ""
    if re.search(r"\b(strong|partial|weak|best|distractor|unsupported)\b", text, re.IGNORECASE):
        return ""
    return text


def _event_ip(timeline: dict[str, Any], event: dict[str, Any]) -> str:
    return event.get("source_ip") or _actor(timeline, event["actor"])["source_ip"]


def _actor(timeline: dict[str, Any], actor_id: str) -> dict[str, Any]:
    return next(actor for actor in timeline["actors"] if actor["id"] == actor_id)


def _actor_by_role(timeline: dict[str, Any], role: str) -> dict[str, Any]:
    # With several actors in a role, use the one with the most events.
    actors = [actor for actor in timeline["actors"] if actor["role"] == role]
    return max(actors, key=lambda actor: sum(event["actor"] == actor["id"] for event in timeline["events"]))


def _parse_turn(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = clean_value(value)
    return int(text) if text.isdigit() else None


def _short(principal: str) -> str:
    return re.split(r"[/:]", principal)[-1] or principal


def _with_account(principal: str, account_id: str) -> str:
    # Models often mix account ids; one account keeps ARNs consistent.
    return re.sub(r"^(arn:aws:(?:iam|sts)::)\d{12}(:)", rf"\g<1>{account_id}\g<2>", principal)


def _is_ipv4(value: str) -> bool:
    parts = value.split(".")
    return len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)


def _is_public_ipv4(value: str) -> bool:
    if not _is_ipv4(value) or value.startswith("192.0.2."):
        return False
    first, second = (int(part) for part in value.split(".")[:2])
    return not (first in {0, 10, 127} or (first == 172 and 16 <= second <= 31) or (first == 192 and second == 168))


def _normalise_mfa(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = clean_value(value).lower()
    return text if text in {"true", "false"} else ""


def _severity_label(value: Any) -> str:
    text = clean_value(value).lower()
    try:
        number = float(text)
        return "High" if number >= 7 else "Medium" if number >= 4 else "Low"
    except ValueError:
        return {"high": "High", "critical": "High", "medium": "Medium", "low": "Low"}.get(text, "Low")


def _default_user_agent(role: str) -> str:
    return {"attacker": "aws-cli/2.15.30 Python/3.11.8", "background": "Boto3/1.34.69 Python/3.11.8",
            "responder": "console.amazonaws.com"}.get(role, "console.amazonaws.com")


def _digest(timeline: dict[str, Any], label: str) -> str:
    seed = f"{timeline['account_id']}|{_actor_by_role(timeline, 'attacker')['principal']}|{label}"
    return hashlib.sha256(seed.encode()).hexdigest()


def _stable_uuid(timeline: dict[str, Any], event: dict[str, Any]) -> str:
    h = _digest(timeline, f"{event['time']}{event['event_name']}")
    return f"{h[:8]}-{h[8:12]}-4{h[13:16]}-a{h[17:20]}-{h[20:32]}"


def _minute_after(time: str) -> str:
    return (datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ") + timedelta(minutes=1)).strftime("%H:%M")
