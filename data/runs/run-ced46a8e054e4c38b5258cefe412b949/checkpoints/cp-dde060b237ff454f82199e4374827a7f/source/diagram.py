#!/usr/bin/env python3
from diagrams import Diagram, Cluster, Edge
from diagrams.aws.security import SecurityHub, IAM, WAF
from diagrams.aws.management import (
    SystemsManager, CloudwatchEventEventBased, 
    Cloudwatch, OrganizationsAccount
)
from diagrams.aws.integration import SimpleNotificationServiceSns, StepFunctions
from diagrams.aws.storage import S3
from diagrams.aws.analytics import KinesisDataFirehose
from diagrams.aws.compute import Lambda

# Diagram configuration
graph_attr = {
    "fontsize": "45",
    "bgcolor": "white",
    "splines": "spline",
    "pad": "0.5",
    "nodesep": "0.60",
    "ranksep": "0.75"
}

with Diagram(
    "AWS Automated Security Response Architecture",
    show=True,
    direction="TB",
    graph_attr=graph_attr,
    filename="data/automated-security-response/architecture"
):
    # Security Hub and Event Sources
    securityhub = SecurityHub("Security Hub")
    events = CloudwatchEventEventBased("EventBridge")
    
    # Orchestration Components
    with Cluster("Orchestration Layer"):
        stepfunctions = StepFunctions("Step Functions\nOrchestrator")
        ssm = SystemsManager("Systems Manager\nAutomation")
        
    # Monitoring and Logging
    with Cluster("Monitoring & Logging"):
        cloudwatch = Cloudwatch("CloudWatch Logs")
        sns = SimpleNotificationServiceSns("SNS\nNotifications")
        firehose = KinesisDataFirehose("Kinesis Firehose")
        metrics_bucket = S3("Metrics Bucket")

    # Account Management
    with Cluster("Account Structure"):
        admin = OrganizationsAccount("Admin Account")
        member = OrganizationsAccount("Member Account")
        
    # IAM Roles and Permissions
    with Cluster("IAM & Permissions"):
        admin_role = IAM("Admin Role")
        member_role = IAM("Member Role")
        remediation_role = IAM("Remediation Role")

    # Lambda Functions
    with Cluster("Lambda Functions"):
        check_state = Lambda("Check State")
        exec_remediation = Lambda("Execute Remediation")
        send_notification = Lambda("Send Notification")

    # Define relationships
    securityhub >> events >> stepfunctions
    stepfunctions >> check_state
    stepfunctions >> exec_remediation
    stepfunctions >> send_notification
    
    exec_remediation >> ssm
    ssm >> member_role
    ssm >> remediation_role
    
    # Logging and monitoring flows
    exec_remediation >> cloudwatch
    ssm >> cloudwatch
    cloudwatch >> firehose >> metrics_bucket
    
    # Notification flow
    stepfunctions >> sns
    
    # Account relationships
    admin >> admin_role
    member >> member_role
    
    # Role relationships
    admin_role >> stepfunctions
    member_role >> ssm
