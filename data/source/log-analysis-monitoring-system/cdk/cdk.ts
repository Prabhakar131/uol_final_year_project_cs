import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as kinesis from 'aws-cdk-lib/aws-kinesis';
import * as firehose from 'aws-cdk-lib/aws-kinesisfirehose'; // L1 construct for Kinesis Firehose, will use CfnDeliveryStream
import { CfnDeliveryStream } from 'aws-cdk-lib/aws-kinesisfirehose'; // More direct way to use Kinesis Firehose
import * as logs from 'aws-cdk-lib/aws-logs';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as opensearchservice from 'aws-cdk-lib/aws-opensearchservice';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import * as lambdaEventSources from 'aws-cdk-lib/aws-lambda-event-sources';
import * as glue from 'aws-cdk-lib/aws-glue'; // L1 construct for Glue, will use CfnJob, CfnDatabase, CfnCrawler etc.
import { CfnJob, CfnDatabase, CfnCrawler, CfnTable } from 'aws-cdk-lib/aws-glue';
import * as kinesisanalytics from 'aws-cdk-lib/aws-kinesisanalytics'; // L1, will use CfnApplicationV2
import { CfnApplicationV2, CfnApplicationCloudWatchLoggingOptionV2 } from 'aws-cdk-lib/aws-kinesisanalyticsv2';

import * as wafv2 from 'aws-cdk-lib/aws-wafv2';
// Inspector and Shield are typically enabled at account level or specific resources,
// Shield Advanced requires subscription. Will show WAF example.

export interface LogAnalysisMonitoringStackProps extends cdk.StackProps {
  vpcCidr: string;
  environment: string;
  appName: string;
  logRetentionDays: number;
  openSearchInstanceType: string;
  openSearchInstanceCount: number;
  enableWaf: boolean;
}

export class LogAnalysisMonitoringStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: LogAnalysisMonitoringStackProps) {
    super(scope, id, props);

    const appName = props.appName;
    const environment = props.environment;

    // === Networking ===
    const vpc = new ec2.Vpc(this, `${appName}-Vpc-${environment}`, {
      cidr: props.vpcCidr,
      maxAzs: 2,
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
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED, // For OpenSearch and other internal services
        },
      ],
      natGateways: 1, // Cost optimization for dev/staging
    });

    // Security Group for OpenSearch
    const openSearchSg = new ec2.SecurityGroup(this, `${appName}-OpenSearchSG-${environment}`, {
      vpc,
      description: 'Security group for OpenSearch Domain',
      allowAllOutbound: true,
    });
    // Allow access from within the VPC, e.g. Lambda. Be more specific in production.
    openSearchSg.addIngressRule(ec2.Peer.ipv4(vpc.vpcCidrBlock), ec2.Port.tcp(443), 'Allow HTTPS from within VPC');


    // === IAM Roles ===
    const lambdaLogProcessorRole = new iam.Role(this, `${appName}-LambdaLogProcessorRole-${environment}`, {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaVPCAccessExecutionRole'), // If Lambda needs VPC access
      ],
    });

    const firehoseDeliveryRole = new iam.Role(this, `${appName}-FirehoseDeliveryRole-${environment}`, {
        assumedBy: new iam.ServicePrincipal('firehose.amazonaws.com'),
    });

    const glueJobRole = new iam.Role(this, `${appName}-GlueJobRole-${environment}`, {
        assumedBy: new iam.ServicePrincipal('glue.amazonaws.com'),
        managedPolicies: [
            iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSGlueServiceRole'),
        ],
    });

    const kinesisAnalyticsRole = new iam.Role(this, `${appName}-KinesisAnalyticsRole-${environment}`, {
        assumedBy: new iam.ServicePrincipal('kinesisanalytics.amazonaws.com'),
    });


    // === Storage Layer ===
    const rawLogsBucket = new s3.Bucket(this, `${appName}-RawLogsBucket-${environment}`, {
      bucketName: `${appName.toLowerCase()}-raw-logs-${this.account}-${this.region}-${environment}`,
      removalPolicy: cdk.RemovalPolicy.RETAIN, // Retain raw logs, consider lifecycle policies
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: true,
      lifecycleRules: [{
        expiration: cdk.Duration.days(props.logRetentionDays * 2), // Longer for raw
        enabled: true,
      }],
      serverAccessLogsPrefix: 'access-logs/',
    });
    rawLogsBucket.grantReadWrite(firehoseDeliveryRole); // Firehose needs to write here

    const processedLogsBucket = new s3.Bucket(this, `${appName}-ProcessedLogsBucket-${environment}`, {
      bucketName: `${appName.toLowerCase()}-processed-logs-${this.account}-${this.region}-${environment}`,
      removalPolicy: cdk.RemovalPolicy.RETAIN, // Retain processed logs
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: true,
      lifecycleRules: [{
        expiration: cdk.Duration.days(props.logRetentionDays),
        enabled: true,
      }],
      serverAccessLogsPrefix: 'access-logs/',
    });
    processedLogsBucket.grantReadWrite(lambdaLogProcessorRole); // Lambda processor writes here
    processedLogsBucket.grantRead(glueJobRole); // Glue ETL reads from here

    const metadataTable = new dynamodb.Table(this, `${appName}-MetadataTable-${environment}`, {
      tableName: `${appName}-Metadata-${environment}`,
      partitionKey: { name: 'logId', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'timestamp', type: dynamodb.AttributeType.NUMBER },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.RETAIN, // Retain metadata, consider backup
      pointInTimeRecovery: true,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
    });
    metadataTable.grantReadWriteData(lambdaLogProcessorRole); // Lambda processor reads/writes metadata

    const openSearchDomain = new opensearchservice.Domain(this, `${appName}-OpenSearchDomain-${environment}`, {
      domainName: `${appName.toLowerCase()}-domain-${environment}`,
      version: opensearchservice.EngineVersion.OPENSEARCH_1_3, // Specify a version
      vpc: vpc,
      vpcSubnets: [{ subnetType: ec2.SubnetType.PRIVATE_ISOLATED }],
      securityGroups: [openSearchSg],
      capacity: {
        dataNodes: props.openSearchInstanceCount,
        dataNodeInstanceType: props.openSearchInstanceType,
      },
      ebs: {
        volumeSize: 10, // GB
        volumeType: ec2.EbsDeviceVolumeType.GP3,
      },
      zoneAwareness: {
        enabled: props.openSearchInstanceCount > 1, // Enable if multi-node
      },
      encryptionAtRest: {
        enabled: true,
      },
      nodeToNodeEncryption: true,
      enforceHttps: true,
      fineGrainedAccessControl: { // Enable fine-grained access control
        masterUserArn: new iam.AccountRootPrincipal().arn, // Replace with a specific IAM user/role ARN for master user
      },
      removalPolicy: cdk.RemovalPolicy.RETAIN, // Or Snapshot on delete
      logging: {
        slowSearchLogEnabled: true,
        appLogEnabled: true,
        auditLogEnabled: true, // If fineGrainedAccessControl is enabled
        slowSearchLogGroup: new logs.LogGroup(this, `${appName}-OpenSearchSlowSearchLogs-${environment}`, {
          retention: props.logRetentionDays,
          removalPolicy: cdk.RemovalPolicy.DESTROY,
        }),
        appLogGroup: new logs.LogGroup(this, `${appName}-OpenSearchAppLogs-${environment}`, {
          retention: props.logRetentionDays,
          removalPolicy: cdk.RemovalPolicy.DESTROY,
        }),
        auditLogGroup: new logs.LogGroup(this, `${appName}-OpenSearchAuditLogs-${environment}`, {
          retention: props.logRetentionDays,
          removalPolicy: cdk.RemovalPolicy.DESTROY,
        }),
      },
    });
    openSearchDomain.grantWrite(lambdaLogProcessorRole); // Lambda processor writes to OpenSearch


    // === Data Ingestion Layer ===
    const mainKinesisStream = new kinesis.Stream(this, `${appName}-MainStream-${environment}`, {
      streamName: `${appName}-MainStream-${environment}`,
      shardCount: 1, // Adjust based on expected throughput
      retentionPeriod: cdk.Duration.days(1),
      encryption: kinesis.StreamEncryption.KMS, // Use KMS for encryption
    });
    mainKinesisStream.grantReadWrite(lambdaLogProcessorRole); // Lambda processor reads from this stream
    mainKinesisStream.grantRead(kinesisAnalyticsRole); // Kinesis Analytics reads from this stream
    // IAM principal for EC2 instances (Kinesis Agent) would need write access to this stream
    // Example: ec2InstanceRole.addToPolicy(new iam.PolicyStatement({ actions: ['kinesis:PutRecord', 'kinesis:PutRecords'], resources: [mainKinesisStream.streamArn] }));


    const firehoseToS3 = new CfnDeliveryStream(this, `${appName}-FirehoseToS3-${environment}`, {
        deliveryStreamName: `${appName}-FirehoseToS3-${environment}`,
        deliveryStreamType: 'DirectPut', // Or KinesisStreamAsSource if Firehose consumes from Kinesis Stream
        s3DestinationConfiguration: {
            bucketArn: rawLogsBucket.bucketArn,
            roleArn: firehoseDeliveryRole.roleArn,
            bufferingHints: {
                intervalInSeconds: 300, // Default 300s (5 mins)
                sizeInMBs: 5,          // Default 5MB
            },
            compressionFormat: 'GZIP', // UNCOMPRESSED, GZIP, ZIP, Snappy
            encryptionConfiguration: {
                noEncryptionConfig: 'NoEncryption', // Or KMSEncryptionConfig
            },
            cloudWatchLoggingOptions: {
                enabled: true,
                logGroupName: `/aws/kinesisfirehose/${appName}-FirehoseToS3-${environment}`,
                logStreamName: 'S3Delivery',
            },
            // Optional: prefix for S3 objects, error output prefix
            prefix: 'firehose-raw-logs/year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/hour=!{timestamp:HH}/',
            errorOutputPrefix: 'firehose-errors/year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/hour=!{timestamp:HH}/!{firehose:error-output-type}',
        },
    });
    // Grant CloudWatch Logs permissions to Firehose role
    firehoseDeliveryRole.addToPolicy(new iam.PolicyStatement({
        actions: ['logs:CreateLogGroup', 'logs:CreateLogStream', 'logs:PutLogEvents'],
        resources: [`arn:aws:logs:${this.region}:${this.account}:log-group:/aws/kinesisfirehose/${appName}-FirehoseToS3-${environment}:*`],
    }));


    // === Processing Layer ===
    const logProcessorLambda = new lambda.Function(this, `${appName}-LogProcessorLambda-${environment}`, {
      functionName: `${appName}-LogProcessor-${environment}`,
      runtime: lambda.Runtime.NODEJS_18_X, // Choose appropriate runtime
      handler: 'index.handler', // Placeholder
      code: lambda.Code.fromAsset('lambda/log-processor'), // Requires a 'lambda/log-processor' directory with code
      role: lambdaLogProcessorRole,
      vpc: vpc, // Run Lambda in VPC to access OpenSearch
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [openSearchSg], // Allow access to OpenSearch
      environment: {
        RAW_LOGS_BUCKET_NAME: rawLogsBucket.bucketName,
        PROCESSED_LOGS_BUCKET_NAME: processedLogsBucket.bucketName,
        METADATA_TABLE_NAME: metadataTable.tableName,
        OPENSEARCH_ENDPOINT: `https://${openSearchDomain.domainEndpoint}`,
        MAIN_KINESIS_STREAM_NAME: mainKinesisStream.streamName,
        SNS_TOPIC_ARN_FOR_ALERTS: '', // To be filled by SNS topic
      },
      timeout: cdk.Duration.minutes(1),
      memorySize: 256,
      logRetention: props.logRetentionDays,
    });
    logProcessorLambda.addEventSource(new lambdaEventSources.KinesisEventSource(mainKinesisStream, {
      batchSize: 100,
      startingPosition: lambda.StartingPosition.LATEST,
      maxBatchingWindow: cdk.Duration.seconds(60),
    }));


    // Placeholder for Glue ETL Job
    const glueDatabase = new CfnDatabase(this, `${appName}-GlueDatabase-${environment}`, {
        catalogId: this.account,
        databaseInput: {
            name: `${appName.toLowerCase()}_db_${environment}`,
            description: 'Database for processed logs metadata',
        },
    });

    const glueS3TargetPath = `s3://${processedLogsBucket.bucketName}/curated/`;

    const glueCrawler = new CfnCrawler(this, `${appName}-GlueCrawler-${environment}`, {
        name: `${appName}-ProcessedLogsCrawler-${environment}`,
        role: glueJobRole.roleArn,
        databaseName: glueDatabase.databaseInput.name,
        targets: {
            s3Targets: [{ path: `s3://${processedLogsBucket.bucketName}/` }], // Crawl the root of processed bucket or a specific prefix
        },
        schemaChangePolicy: {
            updateBehavior: 'UPDATE_IN_DATABASE',
            deleteBehavior: 'LOG',
        },
        configuration: JSON.stringify({
            Version: 1.0,
            CrawlerOutput: {
                Partitions: { AddOrUpdateBehavior: 'InheritFromTable' },
            },
            Grouping: {
                TableGroupingPolicy: 'CombineCompatibleSchemas',
            },
        }),
        schedule: { // Optional: run periodically
             scheduleExpression: 'cron(0 0 * * ? *)', // e.g., daily at midnight UTC
        },
        recrawlPolicy: {
            recrawlBehavior: 'CRAWL_EVERYTHING'
        }
    });
    glueCrawler.node.addDependency(glueDatabase);
    // Grant Glue permissions to access the S3 bucket for crawling
    processedLogsBucket.grantRead(glueJobRole);


    const glueJob = new CfnJob(this, `${appName}-GlueEtlJob-${environment}`, {
      name: `${appName}-EtlJob-${environment}`,
      role: glueJobRole.roleArn,
      command: {
        name: 'glueetl', // or 'pythonshell'
        pythonVersion: '3',
        scriptLocation: `s3://${rawLogsBucket.bucketName}/glue-scripts/etl_script.py`, // Store your Glue script in S3
      },
      glueVersion: '3.0', // Specify Glue version
      workerType: 'G.1X',
      numberOfWorkers: 2,
      defaultArguments: {
        '--TempDir': `s3://${rawLogsBucket.bucketName}/glue-temp/`,
        '--job-bookmark-option': 'job-bookmark-enable',
        '--enable-metrics': '',
        '--S3_SOURCE_PATH': `s3://${processedLogsBucket.bucketName}/`, // Example parameter
        '--S3_TARGET_PATH': glueS3TargetPath, // Example parameter
        '--DATABASE_NAME': glueDatabase.databaseInput.name,
        // '--TABLE_NAME': 'processed_logs_table', // Table name created by crawler or manually
      },
      maxRetries: 0,
      timeout: 60, // In minutes
      // SecurityConfiguration: 'your-security-config', // If using encryption for Glue
      // NotificationProperty: { NotifyDelayAfter: ... } // For job run state notifications
    });
    // Ensure Glue script is uploaded to S3 (e.g. via s3deploy.BucketDeployment)
    // For simplicity, assuming script is manually uploaded or via another process.


    // Kinesis Data Analytics V2 (SQL or Flink)
    const kinesisAnalyticsApp = new CfnApplicationV2(this, `${appName}-KinesisAnalyticsApp-${environment}`, {
        applicationName: `${appName}-RealtimeAnalytics-${environment}`,
        runtimeEnvironment: 'SQL-1_0', // Or FLINK-1_15, FLINK-1_13, etc.
        serviceExecutionRole: kinesisAnalyticsRole.roleArn,
        applicationConfiguration: {
            sqlApplicationConfiguration: {
                inputs: [{
                    namePrefix: 'SOURCE_SQL_STREAM',
                    kinesisStreamsInput: {
                        resourceArn: mainKinesisStream.streamArn,
                    },
                    inputSchema: {
                        recordFormat: {
                            recordFormatType: 'JSON',
                            mappingParameters: {
                                jsonMappingParameters: {
                                    recordRowPath: '$', // Assuming root is an array of records or single record
                                },
                            },
                        },
                        recordEncoding: 'UTF-8',
                        recordColumns: [ // Define schema based on your Kinesis stream data
                            { name: 'eventId', sqlType: 'VARCHAR(64)', mapping: '$.eventId' },
                            { name: 'message', sqlType: 'VARCHAR(1024)', mapping: '$.message' },
                            { name: 'timestamp', sqlType: 'TIMESTAMP', mapping: '$.timestamp' },
                        ],
                    },
                }],
            },
            // Example: Output to another Kinesis Stream or Firehose for alerts/dashboarding
            // applicationCodeConfiguration: { codeContentType: 'PLAINTEXT', codeContent: 'SELECT STREAM * FROM SOURCE_SQL_STREAM WHERE message LIKE \'%ERROR%\';'}
        },
        // CloudWatch logging for the Kinesis Analytics application
        // cloudWatchLoggingOptions: [{ // This is an array
        //     logStreamArn: new logs.LogGroup(this, `${appName}-KDAAppLogs-${environment}`, { // Create a log group
        //         retention: props.logRetentionDays,
        //         removalPolicy: cdk.RemovalPolicy.DESTROY,
        //     }).logGroupArn,
        // }],
    });
    // Grant Kinesis Analytics role permissions to read from the source stream
    mainKinesisStream.grantRead(kinesisAnalyticsRole);
    // Grant Kinesis Analytics role permissions to write to CloudWatch Logs
    kinesisAnalyticsRole.addToPolicy(new iam.PolicyStatement({
        actions: ['logs:CreateLogGroup', 'logs:CreateLogStream', 'logs:PutLogEvents', 'logs:DescribeLogGroups', 'logs:DescribeLogStreams'],
        resources: [`arn:aws:logs:${this.region}:${this.account}:log-group:*`], // Be more specific if possible
    }));

    // === Alerting & Notification Layer ===
    const snsAlertsTopic = new sns.Topic(this, `${appName}-AlertsTopic-${environment}`, {
      topicName: `${appName}-AlertsTopic-${environment}`,
      displayName: 'Log Analysis System Alerts',
    });
    // Lambda Log Processor needs to publish to this topic
    snsAlertsTopic.grantPublish(lambdaLogProcessorRole);
    // Update Lambda environment variable with SNS Topic ARN
    logProcessorLambda.addEnvironment('SNS_TOPIC_ARN_FOR_ALERTS', snsAlertsTopic.topicArn);


    const criticalAlertsQueue = new sqs.Queue(this, `${appName}-CriticalAlertsQueue-${environment}`, {
      queueName: `${appName}-CriticalAlertsQueue-${environment}.fifo`, // Example: FIFO queue
      fifo: true,
      contentBasedDeduplication: true, // If FIFO
      encryption: sqs.QueueEncryption.KMS_MANAGED,
      retentionPeriod: cdk.Duration.days(7),
      visibilityTimeout: cdk.Duration.minutes(5),
      deadLetterQueue: { // Best practice: configure a DLQ
        maxReceiveCount: 5,
        queue: new sqs.Queue(this, `${appName}-CriticalAlertsDLQ-${environment}`, {
          queueName: `${appName}-CriticalAlertsDLQ-${environment}.fifo`,
          fifo: true,
          retentionPeriod: cdk.Duration.days(14),
        }),
      },
    });
    snsAlertsTopic.addSubscription(new sns.subscriptions.SqsSubscription(criticalAlertsQueue, {
        rawMessageDelivery: true, // Depending on how subscriber processes message
        filterPolicy: { // Example filter policy: only messages with 'severity' attribute 'critical'
            severity: sns.SubscriptionFilter.stringFilter({
                allowlist: ['critical', 'high'],
            }),
        },
    }));


    // CloudWatch Alarms (Example: based on SQS queue size)
    // This represents the CloudWatchEventTimeBased from diagram.py, often implemented as alarms.
    // Or, EventBridge rules listening to SNS.
    const highQueueDepthAlarm = new cdk.aws_cloudwatch.Alarm(this, `${appName}-HighQueueDepthAlarm-${environment}`, {
        alarmName: `${appName}-CriticalAlertsQueue-HighDepth-${environment}`,
        alarmDescription: 'Alarm when critical alerts SQS queue depth is too high',
        metric: criticalAlertsQueue.metricApproximateNumberOfMessagesVisible({
            period: cdk.Duration.minutes(1),
            statistic: 'Sum',
        }),
        threshold: 100,
        evaluationPeriods: 3,
        comparisonOperator: cdk.aws_cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
        treatMissingData: cdk.aws_cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    // Action for the alarm (e.g., notify another SNS topic for operational alerts)
    const opsAlertsTopic = new sns.Topic(this, `${appName}-OpsAlertsTopic-${environment}`, {
        topicName: `${appName}-OpsAlertsTopic-${environment}`,
    });
    highQueueDepthAlarm.addAlarmAction(new cdk.aws_cloudwatch_actions.SnsAction(opsAlertsTopic));


    // === Security (WAF) ===
    if (props.enableWaf) {
        const webAcl = new wafv2.CfnWebACL(this, `${appName}-WebAcl-${environment}`, {
            name: `${appName}-WebAcl-${environment}`,
            scope: 'REGIONAL', // Or 'CLOUDFRONT' if protecting a CloudFront distribution
            defaultAction: { allow: {} }, // Default action: allow requests
            visibilityConfig: {
                cloudWatchMetricsEnabled: true,
                metricName: `${appName}WebAcl`,
                sampledRequestsEnabled: true,
            },
            rules: [ // Add AWS Managed Rules or custom rules
                {
                    name: 'AWS-AWSManagedRulesCommonRuleSet',
                    priority: 1,
                    statement: {
                        managedRuleGroupStatement: {
                            vendorName: 'AWS',
                            name: 'AWSManagedRulesCommonRuleSet',
                        },
                    },
                    overrideAction: { none: {} },
                    visibilityConfig: {
                        cloudWatchMetricsEnabled: true,
                        metricName: 'AWSManagedRulesCommonRuleSet',
                        sampledRequestsEnabled: true,
                    },
                },
                {
                    name: 'AWS-AWSManagedRulesAmazonIpReputationList',
                    priority: 2,
                    statement: {
                        managedRuleGroupStatement: {
                            vendorName: 'AWS',
                            name: 'AWSManagedRulesAmazonIpReputationList',
                        },
                    },
                    overrideAction: { none: {} },
                    visibilityConfig: {
                        cloudWatchMetricsEnabled: true,
                        metricName: 'AWSManagedRulesAmazonIpReputationList',
                        sampledRequestsEnabled: true,
                    },
                },
                // Add more rules as needed (SQLi, XSS, etc.)
            ],
        });
        // Associate WAF with resources (e.g., Application Load Balancer, API Gateway)
        // This would typically happen in the construct defining those resources.
        // For example, if you have an ALB:
        // new wafv2.CfnWebACLAssociation(this, `${appName}-WebAclAssocAlb-${environment}`, {
        //   resourceArn: alb.loadBalancerArn,
        //   webAclArn: webAcl.attrArn,
        // });
    }
    // Shield Advanced is a subscription, Inspector is often enabled account-wide or for specific resources.

    // === Outputs ===
    new cdk.CfnOutput(this, 'RawLogsBucketName', { value: rawLogsBucket.bucketName });
    new cdk.CfnOutput(this, 'ProcessedLogsBucketName', { value: processedLogsBucket.bucketName });
    new cdk.CfnOutput(this, 'MetadataTableName', { value: metadataTable.tableName });
    new cdk.CfnOutput(this, 'OpenSearchDomainEndpoint', { value: openSearchDomain.domainEndpoint });
    new cdk.CfnOutput(this, 'MainKinesisStreamName', { value: mainKinesisStream.streamName });
    new cdk.CfnOutput(this, 'FirehoseDeliveryStreamName', { value: firehoseToS3.deliveryStreamName || `${appName}-FirehoseToS3-${environment}` });
    new cdk.CfnOutput(this, 'LogProcessorLambdaName', { value: logProcessorLambda.functionName });
    new cdk.CfnOutput(this, 'SnsAlertsTopicArn', { value: snsAlertsTopic.topicArn });
    new cdk.CfnOutput(this, 'CriticalAlertsQueueUrl', { value: criticalAlertsQueue.queueUrl });
    new cdk.CfnOutput(this, 'VpcId', { value: vpc.vpcId });

    // Tag all resources in this stack
    cdk.Tags.of(this).add('AppName', appName);
    cdk.Tags.of(this).add('Environment', environment);
    cdk.Tags.of(this).add('Project', 'DataAnalytics'); // From cdk.json or as needed
    cdk.Tags.of(this).add('CostCenter', 'IT');
  }
}

const app = new cdk.App();

// Retrieve context values
const vpcCidr = app.node.tryGetContext('vpcCidr') || '10.0.0.0/16';
const environment = app.node.tryGetContext('environment') || 'dev';
const appName = app.node.tryGetContext('appName') || 'LogAnalysisSystem';
const logRetentionDays = app.node.tryGetContext('logRetentionDays') || 7;
const openSearchInstanceType = app.node.tryGetContext('openSearchInstanceType') || 't3.small.search';
const openSearchInstanceCount = app.node.tryGetContext('openSearchInstanceCount') || 1;
const enableWaf = app.node.tryGetContext('enableWaf') === 'true' || false; // Context values are strings


new LogAnalysisMonitoringStack(app, `${appName}-Stack-${environment}`, {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
  vpcCidr,
  environment,
  appName,
  logRetentionDays,
  openSearchInstanceType,
  openSearchInstanceCount,
  enableWaf,
  description: `Log Analysis and Monitoring System Stack for ${environment} environment. AppName: ${appName}`,
});

app.synth(); 