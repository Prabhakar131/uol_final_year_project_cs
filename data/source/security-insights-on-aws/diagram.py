#!/usr/bin/env python3
from diagrams import Diagram, Cluster, Edge
from diagrams.aws.analytics import (
    Athena,
    Quicksight,
    GlueCrawlers,
    GlueDataCatalog,
    LakeFormation
)
from diagrams.aws.security import (
    SecurityHub
)
from diagrams.aws.analytics import (
    AmazonOpensearchService
)
from diagrams.aws.compute import (
    Lambda
)
from diagrams.aws.integration import (
    Eventbridge
)
from diagrams.aws.management import (
    SystemsManagerParameterStore,
    CloudwatchLogs,
    Cloudwatch,
    Cloudtrail
)
from diagrams.aws.network import (
    VPC
)
from diagrams.aws.storage import (
    S3
)
from diagrams.aws.database import (
    DatabaseMigrationService
)

# Diagram configuration
graph_attr = {
    "fontsize": "45",
    "bgcolor": "white",
    "splines": "ortho",
    "pad": "0.5",
}

with Diagram("Security Insights on AWS", show=False, filename="data/security-insights-on-aws/architecture", direction="TB", graph_attr=graph_attr):
    
    # Data Sources
    with Cluster("Security Data Sources"):
        opensearch = AmazonOpensearchService("OpenSearch")
        vpc_flow = VPC("VPC Flow Logs")
        cloudtrail = Cloudtrail("CloudTrail")
        sec_hub = SecurityHub("Security Hub")
        app_fabric = DatabaseMigrationService("App Fabric")

        data_sources = [vpc_flow, cloudtrail, sec_hub, app_fabric]
        for source in data_sources:
            source >> opensearch

    # Data Processing Layer
    with Cluster("Data Processing & Catalog"):
        lake_formation = LakeFormation("Lake Formation")
        glue_catalog = GlueDataCatalog("Glue Catalog")
        glue_crawler = GlueCrawlers("Glue Crawler")
        athena = Athena("Athena Workgroup")
        athena_results = S3("Athena Results")

        opensearch >> lake_formation
        lake_formation >> glue_catalog
        glue_crawler >> glue_catalog
        athena >> Edge(color="darkgreen") >> athena_results
        glue_catalog >> athena

    # Lambda Functions Layer
    with Cluster("Lambda Functions"):
        create_permissions = Lambda("Create Lake Formation\nPermissions")
        create_datasets = Lambda("Create QuickSight\nDatasets")
        create_schedules = Lambda("Create Refresh\nSchedules")
        user_groups = Lambda("User Group\nManager")

    # Control & Monitoring
    with Cluster("Control & Monitoring"):
        ssm = SystemsManagerParameterStore("SSM Parameters")
        eventbridge = Eventbridge("EventBridge")
        cloudwatch = Cloudwatch("CloudWatch")
        logs = CloudwatchLogs("CloudWatch Logs")

        ssm >> eventbridge
        eventbridge >> Edge(style="dashed") >> [create_permissions, create_datasets]
        [create_permissions, create_datasets, create_schedules, user_groups] >> logs
        athena >> cloudwatch

    # QuickSight Layer
    with Cluster("QuickSight Visualization"):
        quicksight = Quicksight("QuickSight")
        with Cluster("QuickSight Assets"):
            datasets = Quicksight("Datasets")
            analysis = Quicksight("Analysis")
            dashboard = Quicksight("Dashboard")

        athena >> create_datasets
        create_datasets >> datasets
        datasets >> analysis
        analysis >> dashboard
        create_schedules >> datasets
        user_groups >> Edge(color="blue") >> [analysis, dashboard]

    # Permissions Flow
    lake_formation >> Edge(color="red", style="dashed") >> create_permissions
    create_permissions >> Edge(color="red", style="dashed") >> quicksight
