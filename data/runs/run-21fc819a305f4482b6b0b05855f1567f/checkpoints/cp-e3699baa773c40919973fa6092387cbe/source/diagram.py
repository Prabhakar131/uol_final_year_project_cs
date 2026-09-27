#!/usr/bin/env python3

"""
AWS Serverless Multi-cloud Cost Management System
This diagram illustrates a serverless architecture for managing and analyzing
costs across multiple cloud providers, with secure API access and automated
data processing pipelines.
"""

# --- Analytics related imports ---
from diagrams.aws.analytics import Quicksight
from diagrams.aws.cost import CostExplorer, CostAndUsageReport

# --- Network related imports ---
from diagrams.aws.network import APIGateway

# --- Compute related imports ---
from diagrams.aws.compute import Lambda

# --- Integration related imports ---
from diagrams.aws.integration import Eventbridge

# --- Security related imports ---
from diagrams.aws.security import IAM, SecretsManager

# --- Storage related imports ---
from diagrams.aws.storage import S3

# --- Database related imports ---
from diagrams.aws.database import Dynamodb

# --- Management and Governance related imports ---
from diagrams.aws.management import Cloudwatch, Organizations

# --- General diagram imports ---
from diagrams import Diagram, Cluster, Edge
from diagrams.aws.general import Users

# Graph attributes for better visualization
graph_attr = {
    "fontsize": "45",
    "bgcolor": "white",
    "splines": "polyline",
    "pad": "2.0",
    "nodesep": "1.0",
    "ranksep": "1.2",
    "labelloc": "t",
    "label": "AWS Multi-cloud Cost Management System",
    "fontname": "Helvetica",
}

# Edge attributes for better label placement
edge_attr = {
    "minlen": "1.5",
    "labelangle": "0",
    "labeldistance": "2.0"
}

# Create the diagram
with Diagram(
    name="Serverless Multi-cloud Cost Management",
    filename="cost_management",
    direction="TB",
    show=True,
    graph_attr=graph_attr,
    edge_attr=edge_attr,
    outformat="png"
):
    # External Users
    users = Users("Cloud Admins")
    
    # AWS Account Boundary
    with Cluster("AWS Account Boundary", graph_attr={"style": "dashed", "color": "#FF9900", "bgcolor": "#FFFAF0"}):
        # Frontend Layer
        with Cluster("Frontend Services"):
            dashboard = Quicksight("Cost Dashboard")
        
        # API Layer
        with Cluster("API Layer"):
            api = APIGateway("Cost Management API")
            auth = IAM("Authentication")
        
        # Data Collection Layer
        with Cluster("Data Collection"):
            cost_explorer = CostExplorer("Cost Explorer")
            cost_reports = CostAndUsageReport("Usage Reports")
            org = Organizations("AWS Organizations")
            
            # Data Processing
            with Cluster("Processing Pipeline"):
                collector = Lambda("Cost Collector")
                processor = Lambda("Cost Processor")
                analyzer = Lambda("Cost Analyzer")
        
        # Data Storage Layer
        with Cluster("Data Storage"):
            raw_bucket = S3("Raw Cost Data")
            processed_bucket = S3("Processed Data")
            cost_db = Dynamodb("Cost Database")
        
        # Integration & Monitoring
        scheduler = Eventbridge("Scheduler")
        secrets = SecretsManager("Cloud Credentials")
        monitor = Cloudwatch("Monitoring")

    # Define connections with sequential flow
    
    # User Interactions
    users >> Edge(label="1. Access Dashboard", color="#000080") >> dashboard
    users >> Edge(label="2. API Request", color="#000080") >> api
    
    # Authentication
    api >> Edge(label="3. Verify", color="#000080") >> auth
    
    # Data Collection
    scheduler >> Edge(label="4. Schedule", color="#000080") >> collector
    collector >> Edge(label="5. Get Credentials", color="#000080") >> secrets
    collector >> Edge(label="6. Fetch AWS Costs", color="#000080") >> cost_explorer
    collector >> Edge(label="7. Get AWS Reports", color="#000080") >> cost_reports
    collector >> Edge(label="8. Query Org Data", color="#000080") >> org
    collector >> Edge(label="9. Fetch Other Cloud Costs", style="dashed", color="#000080") >> Edge(label="(Azure, GCP, etc.)")
    
    # Data Processing
    collector >> Edge(label="10. Save", color="#000080", fontsize="14", labelbgcolor="white") >> raw_bucket
    raw_bucket >> Edge(label="11. Process", color="#000080", fontsize="14", labelbgcolor="white") >> processor
    processor >> Edge(label="12. Analyze", color="#000080") >> analyzer
    
    # Data Storage
    analyzer >> Edge(label="13. Store", color="#000080") >> processed_bucket
    analyzer >> Edge(label="14. Update", color="#000080") >> cost_db
    
    # Dashboard Updates
    processed_bucket >> Edge(label="15. Load", color="#000080") >> dashboard
    cost_db >> Edge(label="16. Sync", color="#000080") >> dashboard
    
    # Single monitoring connection
    analyzer >> Edge(label="17. Monitor", style="dotted", color="#800000") >> monitor
    
    # Single IAM connection
    auth >> Edge(label="18. Authorize", style="dashed", color="#006400") >> api
