#!/usr/bin/env python3
from diagrams import Diagram, Cluster, Edge
from diagrams.aws.analytics import (
    KinesisDataFirehose,
    KinesisDataStreams,
    KinesisDataAnalytics,
    Glue,
)
from diagrams.aws.compute import (
    Lambda,
    EC2,
)
from diagrams.aws.database import (
    Dynamodb,
)
from diagrams.aws.integration import (
    SNS,
    SQS,
)
from diagrams.aws.management import (
    Cloudwatch,
    CloudwatchEventTimeBased,
)
from diagrams.aws.security import (
    IAM,
    Inspector,
    Shield,
    WAF
)
from diagrams.aws.storage import (
    S3,
)
from diagrams.aws.analytics import (
    ElasticsearchService,
    AmazonOpensearchService
)

# Diagram configuration
graph_attr = {
    "splines": "spline",  # Curved edges
    "nodesep": "0.8",     # Node separation
    "ranksep": "1.0",     # Rank separation
    "pad": "2.0",         # Padding
}

# Create the diagram
with Diagram(
    "AWS Log Analysis and Monitoring System",
    filename="data/log-analysis-monitoring-system/architecture",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
):
    # Security and IAM
    with Cluster("Security & IAM"):
        security_group = [
            Inspector("Inspector"),
            Shield("Shield"),
            WAF("WAF")
        ]
        iam = IAM("IAM Roles")

    # Log Sources
    with Cluster("Log Sources"):
        cloudwatch = Cloudwatch("CloudWatch Logs")
        ec2_logs = EC2("Application Logs")
        custom_sources = EC2("Custom Sources")
        
    # Data Ingestion Layer
    with Cluster("Data Ingestion"):
        firehose = KinesisDataFirehose("Kinesis Firehose")
        streams = KinesisDataStreams("Kinesis Streams")
        kinesis_agent = KinesisDataStreams("Kinesis Agent")

    # Processing Layer
    with Cluster("Processing & Enrichment"):
        lambda_processor = Lambda("Log Processor")
        lambda_alerting = Lambda("Alert Generator")
        kinesis_analytics = KinesisDataAnalytics("Real-time Analytics")
        glue_etl = Glue("Glue ETL")

    # Storage Layer
    with Cluster("Storage & Search"):
        s3_raw = S3("Raw Logs")
        s3_processed = S3("Processed Logs")
        opensearch = ElasticsearchService("OpenSearch")
        dynamodb = Dynamodb("Metadata Store")

    # Notification Layer
    with Cluster("Alerting & Notification"):
        sns = SNS("SNS")
        sqs = SQS("SQS")
        cloudwatch_alerts = CloudwatchEventTimeBased("CloudWatch Alerts")

    # Define connections with labels
    # Log Sources to Ingestion
    cloudwatch >> Edge(label="Logs") >> firehose
    ec2_logs >> Edge(label="App Logs") >> streams
    custom_sources >> Edge(label="Custom Data") >> kinesis_agent

    # Ingestion to Processing
    firehose >> Edge(label="Raw Logs") >> s3_raw
    streams >> Edge(label="Stream Data") >> lambda_processor
    kinesis_agent >> Edge(label="Agent Data") >> streams

    # Processing Layer Connections
    lambda_processor >> Edge(label="Processed") >> s3_processed
    lambda_processor >> Edge(label="Index") >> opensearch
    lambda_processor >> Edge(label="Metadata") >> dynamodb
    
    # Analytics and Monitoring
    s3_processed >> Edge(label="Batch Process") >> glue_etl
    streams >> Edge(label="Real-time") >> kinesis_analytics
    
    # Security Monitoring
    security_group >> Edge(label="Findings") >> sns
    lambda_processor >> Edge(label="Alerts") >> sns
    
    # Alerting Flow
    sns >> Edge(label="Notifications") >> sqs
    sns >> Edge(label="Alerts") >> cloudwatch_alerts
    
    # IAM Connections (shown as dotted lines)
    iam >> Edge(style="dotted") >> lambda_processor
    iam >> Edge(style="dotted") >> streams
    iam >> Edge(style="dotted") >> firehose
