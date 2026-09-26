import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as opensearch from 'aws-cdk-lib/aws-opensearchservice';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as glue from 'aws-cdk-lib/aws-glue';
import * as lakeformation from 'aws-cdk-lib/aws-lakeformation';
import * as athena from 'aws-cdk-lib/aws-athena';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import * as logs from 'aws-cdk-lib/aws-logs';
// import * as quicksight from 'aws-cdk-lib/aws-quicksight'; // L1 constructs, L2 might not cover all aspects

interface SecurityInsightsStackProps extends cdk.StackProps {
  defaultRegion: string;
  defaultAccount: string;
  envName: string;
  projectName: string;
  vpcCidr: string;
  openSearchDomainName: string;
  quicksightUserArn: string;
}

export class SecurityInsightsStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: SecurityInsightsStackProps) {
    super(scope, id, props);

    const { projectName, envName, openSearchDomainName, vpcCidr, quicksightUserArn } = props;

    // 1. Network Configuration (VPC for Flow Logs and OpenSearch)
    const vpc = new ec2.Vpc(this, `${projectName}-Vpc`, {
      cidr: vpcCidr,
      maxAzs: 2,
      natGateways: 1,
      subnetConfiguration: [
        {
          cidrMask: 24,
          name: 'public',
          subnetType: ec2.SubnetType.PUBLIC,
        },
        {
          cidrMask: 24,
          name: 'private',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
        },
        {
          cidrMask: 28,
          name: 'isolated',
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED, // For OpenSearch if needed
        }
      ],
      enableDnsHostnames: true,
      enableDnsSupport: true,
    });

    // Enable VPC Flow Logs (Source: diagram)
    const flowLogBucket = new s3.Bucket(this, `${projectName}-FlowLogBucket`, {
        bucketName: `${projectName.toLowerCase()}-flow-logs-${envName}-${this.account}-${this.region}`,
        removalPolicy: cdk.RemovalPolicy.DESTROY, // Change for production
        autoDeleteObjects: true, // Change for production
        blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
        encryption: s3.BucketEncryption.S3_MANAGED,
        enforceSSL: true,
    });
    vpc.addFlowLog(`${projectName}FlowLogs`, {
        destination: ec2.FlowLogDestination.toS3(flowLogBucket),
        trafficType: ec2.FlowLogTrafficType.ALL,
    });

    // 2. Data Storage
    // Athena Results Bucket (Source: diagram)
    const athenaResultsBucket = new s3.Bucket(this, `${projectName}-AthenaResultsBucket`, {
      bucketName: `${projectName.toLowerCase()}-athena-results-${envName}-${this.account}-${this.region}`,
      removalPolicy: cdk.RemovalPolicy.DESTROY, // Change for production
      autoDeleteObjects: true, // Change for production
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      lifecycleRules: [{
        expiration: cdk.Duration.days(90), // Configure as needed
      }]
    });

    // Placeholder for other data source S3 buckets (e.g., for AppFabric, CloudTrail if not direct)
    const dataSourcesBucket = new s3.Bucket(this, `${projectName}-DataSourcesBucket`, {
        bucketName: `${projectName.toLowerCase()}-data-sources-${envName}-${this.account}-${this.region}`,
        removalPolicy: cdk.RemovalPolicy.DESTROY, // Change for production
        autoDeleteObjects: true, // Change for production
        blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
        encryption: s3.BucketEncryption.S3_MANAGED,
        enforceSSL: true,
    });
    new cdk.CfnOutput(this, 'DataSourcesBucketName', { value: dataSourcesBucket.bucketName });

    // 3. OpenSearch Service (Source: diagram)
    const osUserMasterRole = new iam.Role(this, `${projectName}-OSMasterUserRole`, {
        assumedBy: new iam.AccountRootPrincipal() // Or a specific IAM user/role
    });

    const openSearchDomain = new opensearch.Domain(this, `${projectName}-OpenSearchDomain`, {
      domainName: openSearchDomainName,
      version: opensearch.EngineVersion.OPENSEARCH_1_3, // Choose appropriate version
      vpc: vpc,
      vpcSubnets: [{ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }],
      securityGroups: [new ec2.SecurityGroup(this, `${projectName}-OpenSearchSG`, {
        vpc,
        allowAllOutbound: true,
        description: 'Security group for OpenSearch Domain',
      })],
      capacity: {
        masterNodes: envName === 'prod' ? 3 : undefined,
        masterNodeInstanceType: envName === 'prod' ? 'm6g.large.search' : 't3.small.search',
        dataNodes: envName === 'prod' ? 3 : 1,
        dataNodeInstanceType: envName === 'prod' ? 'r6g.large.search' : 't3.small.search',
      },
      ebs: {
        volumeSize: 20,
        volumeType: ec2.EbsDeviceVolumeType.GP3,
      },
      zoneAwareness: {
        enabled: envName === 'prod',
        availabilityZoneCount: envName === 'prod' ? 3 : undefined,
      },
      fineGrainedAccessControl: {
        masterUserArn: osUserMasterRole.roleArn, // Replace with actual Master User ARN or manage through internal user DB
      },
      enforceHttps: true,
      nodeToNodeEncryption: true,
      encryptionAtRest: {
        enabled: true,
      },
      logging: {
        slowSearchLogEnabled: true,
        appLogEnabled: true,
        slowIndexLogEnabled: true,
        auditLogEnabled: true, // If required and configured with CW Logs
      },
      removalPolicy: cdk.RemovalPolicy.DESTROY, // Change for production
    });
    // Allow access to OpenSearch from VPC, e.g., from Lambdas or EC2 instances
    // openSearchDomain.connections.allowFrom(...);

    new cdk.CfnOutput(this, 'OpenSearchDomainEndpoint', { value: openSearchDomain.domainEndpoint });

    // 4. Data Processing and Catalog (Lake Formation, Glue, Athena)

    // Lake Formation Admin (Could be the Lambda execution roles or a dedicated role)
    const lakeFormationAdminRole = new iam.Role(this, `${projectName}LakeFormationAdminRole`, {
        assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'), // Or other principals
        managedPolicies: [
            iam.ManagedPolicy.fromAwsManagedPolicyName('AWSLakeFormationDataAdmin'),
            iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSGlueServiceRole'), // For Glue jobs/crawlers if LF manages them
        ],
    });
    dataSourcesBucket.grantReadWrite(lakeFormationAdminRole);
    athenaResultsBucket.grantReadWrite(lakeFormationAdminRole);

    // Register S3 locations with Lake Formation
    new lakeformation.CfnResource(this, `${projectName}DataSourceLocation`, {
        resourceArn: dataSourcesBucket.bucketArn,
        useServiceLinkedRole: true,
        roleArn: lakeFormationAdminRole.roleArn, // Required if not SLP & managing permissions
    });
    new lakeformation.CfnResource(this, `${projectName}AthenaResultsLocation`, {
        resourceArn: athenaResultsBucket.bucketArn,
        useServiceLinkedRole: true,
        roleArn: lakeFormationAdminRole.roleArn,
    });
    // Grant Lake Formation permissions to roles/users as needed
    // Example: grant SELECT on tables to Athena query role or QuickSight role

    // Glue Data Catalog Database (Source: diagram)
    const glueDatabase = new glue.CfnDatabase(this, `${projectName}-GlueDatabase`, {
      catalogId: cdk.Aws.ACCOUNT_ID,
      databaseInput: {
        name: `${projectName.toLowerCase()}_security_data_${envName}`,
        description: 'Database for security insights data',
      },
    });

    // Glue Crawler (Source: diagram)
    const glueCrawlerRole = new iam.Role(this, `${projectName}-GlueCrawlerRole`, {
      assumedBy: new iam.ServicePrincipal('glue.amazonaws.com'),
      managedPolicies: [iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSGlueServiceRole')],
    });
    // Grant crawler role access to data sources (e.g., S3 buckets, OpenSearch through Lake Formation)
    dataSourcesBucket.grantRead(glueCrawlerRole);
    openSearchDomain.grantRead(glueCrawlerRole); // This might need specific LF permissions too

    const glueCrawler = new glue.CfnCrawler(this, `${projectName}-GlueCrawler`, {
      name: `${projectName}-SecurityDataCrawler-${envName}`,
      role: glueCrawlerRole.roleArn,
      databaseName: glueDatabase.databaseInput.name,
      description: 'Crawler for security data sources',
      targets: {
        // Example for S3, OpenSearch targets need specific connection/path setup
        s3Targets: [
          {
            path: `s3://${dataSourcesBucket.bucketName}/`,
            // exclusions: ['internal/', 'temp/'],
          },
        ],
        // For OpenSearch, you might need a JDBC connection or custom classifier
        // catalogTargets: [ { databaseName: string, tables: string[] } ] for other Glue tables
      },
      // schedule: { scheduleExpression: 'cron(0 0 * * ? *)' }, // Daily
      schemaChangePolicy: {
        updateBehavior: 'UPDATE_IN_DATABASE',
        deleteBehavior: 'LOG',
      },
      configuration: JSON.stringify({
        Version: 1.0,
        CrawlerOutput: {
          Partitions: { AddOrUpdateBehavior: 'InheritFromTable' },
        },
      }),
    });
    glueCrawler.node.addDependency(glueDatabase);

    // Athena Workgroup (Source: diagram)
    const athenaWorkgroup = new athena.CfnWorkGroup(this, `${projectName}-AthenaWorkgroup`, {
      name: `${projectName}-SecurityInsightsWorkgroup-${envName}`,
      description: 'Workgroup for security insights queries',
      state: 'ENABLED',
      workGroupConfiguration: {
        resultConfiguration: {
          outputLocation: `s3://${athenaResultsBucket.bucketName}/athena_results/`,
          encryptionConfiguration: {
            encryptionOption: 'SSE_S3',
          },
        },
        enforceWorkGroupConfiguration: true,
        publishCloudWatchMetricsEnabled: true,
        requesterPaysEnabled: false,
      },
    });
    new cdk.CfnOutput(this, 'AthenaWorkgroupName', { value: athenaWorkgroup.name });

    // 5. Lambda Functions Layer (Source: diagram)
    const lambdaExecutionRole = new iam.Role(this, `${projectName}LambdaExecutionRole`, {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AWSLakeFormationDataAdmin'), // For LF permissions Lambda
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonAthenaFullAccess'), // For QS Datasets Lambda
        // Add QuickSight permissions if Lambdas interact with it directly
      ],
      inlinePolicies: {
        QuickSightAccess: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              actions: [
                'quicksight:CreateDataSet',
                'quicksight:UpdateDataSet',
                'quicksight:DeleteDataSet',
                'quicksight:DescribeDataSet',
                'quicksight:PassDataSet',
                'quicksight:CreateDataSource',
                'quicksight:UpdateDataSource',
                'quicksight:DeleteDataSource',
                'quicksight:DescribeDataSource',
                'quicksight:PassDataSource',
                'quicksight:CreateRefreshSchedule',
                'quicksight:UpdateRefreshSchedule',
                'quicksight:DeleteRefreshSchedule',
                'quicksight:DescribeRefreshSchedule',
                // Permissions for managing users/groups if needed by UserGroupManagerLambda
                'quicksight:CreateGroup',
                'quicksight:DeleteGroup',
                'quicksight:UpdateGroup',
                'quicksight:DescribeGroup',
                'quicksight:CreateGroupMembership',
                'quicksight:DeleteGroupMembership',
              ],
              resources: [`arn:aws:quicksight:${this.region}:${this.account}:*`], // Be more specific if possible
            }),
          ],
        }),
      }
    });

    athenaResultsBucket.grantReadWrite(lambdaExecutionRole);
    dataSourcesBucket.grantRead(lambdaExecutionRole);
    openSearchDomain.grantReadWrite(lambdaExecutionRole); // Grant access to OpenSearch from Lambdas

    // Dummy Lambda Code Asset
    const dummyLambdaCode = lambda.Code.fromInline(
      'exports.handler = async (event) => { console.log("Event:", event); return { statusCode: 200, body: "Hello from Lambda!" }; };'
    );

    const createLfPermsLambda = new lambda.Function(this, `${projectName}-CreateLfPermsLambda`, {
      functionName: `${projectName}-CreateLfPerms-${envName}`,
      runtime: lambda.Runtime.NODEJS_18_X,
      handler: 'index.handler',
      code: dummyLambdaCode, // Replace with actual code
      role: lambdaExecutionRole,
      timeout: cdk.Duration.minutes(1),
      environment: {
        PROJECT_NAME: projectName,
        ENV_NAME: envName,
        QUICKSIGHT_USER_ARN: quicksightUserArn,
        GLUE_DATABASE_NAME: glueDatabase.databaseInput.name!,
      },
      logRetention: logs.RetentionDays.ONE_MONTH,
    });

    const createQsDatasetsLambda = new lambda.Function(this, `${projectName}-CreateQsDatasetsLambda`, {
      functionName: `${projectName}-CreateQsDatasets-${envName}`,
      runtime: lambda.Runtime.NODEJS_18_X,
      handler: 'index.handler',
      code: dummyLambdaCode, // Replace with actual code
      role: lambdaExecutionRole,
      timeout: cdk.Duration.minutes(5),
      environment: {
        PROJECT_NAME: projectName,
        ENV_NAME: envName,
        ATHENA_WORKGROUP: athenaWorkgroup.name,
        GLUE_DATABASE_NAME: glueDatabase.databaseInput.name!,
        QUICKSIGHT_USER_ARN: quicksightUserArn,
      },
      logRetention: logs.RetentionDays.ONE_MONTH,
    });

    const createRefreshSchedulesLambda = new lambda.Function(this, `${projectName}-CreateRefreshSchedulesLambda`, {
      functionName: `${projectName}-CreateRefreshSchedules-${envName}`,
      runtime: lambda.Runtime.NODEJS_18_X,
      handler: 'index.handler',
      code: dummyLambdaCode, // Replace with actual code
      role: lambdaExecutionRole,
      timeout: cdk.Duration.minutes(1),
      environment: {
        PROJECT_NAME: projectName,
        ENV_NAME: envName,
        QUICKSIGHT_USER_ARN: quicksightUserArn,
      },
      logRetention: logs.RetentionDays.ONE_MONTH,
    });

    const userGroupManagerLambda = new lambda.Function(this, `${projectName}-UserGroupManagerLambda`, {
      functionName: `${projectName}-UserGroupManager-${envName}`,
      runtime: lambda.Runtime.NODEJS_18_X,
      handler: 'index.handler',
      code: dummyLambdaCode, // Replace with actual code
      role: lambdaExecutionRole,
      timeout: cdk.Duration.minutes(1),
      environment: {
        PROJECT_NAME: projectName,
        ENV_NAME: envName,
        QUICKSIGHT_USER_ARN: quicksightUserArn,
      },
      logRetention: logs.RetentionDays.ONE_MONTH,
    });

    // 6. Control & Monitoring (SSM, EventBridge, CloudWatch)
    // SSM Parameter (Source: diagram - triggers EventBridge)
    const controlParameter = new ssm.StringParameter(this, `${projectName}-ControlParameter`, {
      parameterName: `/security-insights/${envName}/control/someSetting`,
      stringValue: 'initialValue',
      description: 'Control parameter for Security Insights workflow',
    });

    // EventBridge Rule (Source: diagram - triggered by SSM, targets Lambdas)
    const ssmChangeEventRule = new events.Rule(this, `${projectName}-SsmChangeEventRule`, {
      description: 'Rule to detect changes in SSM Parameter Store for security insights',
      eventPattern: {
        source: ['aws.ssm'],
        detailType: ['Parameter Store Change'],
        detail: {
          name: [controlParameter.parameterName],
          operation: ['Update', 'Create', 'Delete', 'LabelParameterVersion'],
        },
      },
    });
    ssmChangeEventRule.addTarget(new targets.LambdaFunction(createLfPermsLambda));
    ssmChangeEventRule.addTarget(new targets.LambdaFunction(createQsDatasetsLambda));

    // CloudWatch Logs already configured for Lambdas via logRetention
    // Additional CloudWatch Alarms or Dashboards can be defined here
    // Example: Alarm for Athena query failures (requires Athena metrics enabled)

    // 7. QuickSight Layer (Source: diagram - primarily through Lambdas and manual setup)
    // QuickSight resources like DataSources, DataSets, Analyses, Dashboards are often managed via SDK/API calls within Lambdas or manually.
    // Here we ensure the necessary IAM permissions are on Lambda roles.
    // We also need a QuickSight user. This must exist or be managed separately.

    // Example of creating a QuickSight Data Source (L1 construct as L2 is limited)
    // This would typically be done by a Lambda or setup script after the stack is deployed.
    /*
    const quicksightDataSource = new quicksight.CfnDataSource(this, `${projectName}QuickSightAthenaSource`, {
      awsAccountId: this.account,
      dataSourceId: `${projectName}-AthenaDataSource-${envName}`,
      name: 'SecurityInsightsAthenaSource',
      type: 'ATHENA',
      dataSourceParameters: {
        athenaParameters: {
          workGroup: athenaWorkgroup.name,
        },
      },
      permissions: [{
        principal: quicksightUserArn, // ARN of the QuickSight user/group
        actions: [
          'quicksight:DescribeDataSource',
          'quicksight:DescribeDataSourcePermissions',
          'quicksight:PassDataSource',
          'quicksight:UpdateDataSource',
          'quicksight:DeleteDataSource',
          'quicksight:UpdateDataSourcePermissions',
        ],
      }],
      sslProperties: { disableSsl: false }, // Athena typically doesn't require SSL for connection itself
    });
    quicksightDataSource.node.addDependency(athenaWorkgroup);
    */

    // Outputs
    new cdk.CfnOutput(this, 'VpcId', { value: vpc.vpcId });
    new cdk.CfnOutput(this, 'AthenaResultsBucketNameOutput', { value: athenaResultsBucket.bucketName });
    new cdk.CfnOutput(this, 'GlueDatabaseNameOutput', { value: glueDatabase.databaseInput.name! });
    new cdk.CfnOutput(this, 'CreateLfPermsLambdaArn', { value: createLfPermsLambda.functionArn });
    new cdk.CfnOutput(this, 'CreateQsDatasetsLambdaArn', { value: createQsDatasetsLambda.functionArn });

    // Tagging (applied from cdk.json or can be added here too)
    cdk.Tags.of(this).add('Application', `${projectName}-${envName}`);
  }
}

// Application entry point
const app = new cdk.App();

const defaultRegion = app.node.tryGetContext('defaultRegion') || 'us-east-1';
const defaultAccount = app.node.tryGetContext('defaultAccount') || process.env.CDK_DEFAULT_ACCOUNT;
const envName = app.node.tryGetContext('envName') || 'dev';
const projectName = app.node.tryGetContext('projectName') || 'SecurityInsights';
const vpcCidr = app.node.tryGetContext('vpcCidr') || '10.0.0.0/16';
const openSearchDomainName = app.node.tryGetContext('openSearchDomainName') || `secinsights-${envName}`.toLowerCase();
const quicksightUserArn = app.node.tryGetContext('quicksightUserArn');

if (!defaultAccount) {
    throw new Error('Default AWS account is not specified. Set in cdk.json context or via CDK_DEFAULT_ACCOUNT env variable.');
}
if (!quicksightUserArn) {
    throw new Error('QuickSight User ARN (quicksightUserArn) must be specified in cdk.json context.');
}

const stackProps: SecurityInsightsStackProps = {
  env: {
    account: defaultAccount,
    region: defaultRegion,
  },
  defaultRegion,
  defaultAccount,
  envName,
  projectName,
  vpcCidr,
  openSearchDomainName,
  quicksightUserArn,
  description: 'Stack for Security Insights on AWS solution',
};

new SecurityInsightsStack(app, `${projectName}-Stack-${envName}`, stackProps);

app.synth(); 