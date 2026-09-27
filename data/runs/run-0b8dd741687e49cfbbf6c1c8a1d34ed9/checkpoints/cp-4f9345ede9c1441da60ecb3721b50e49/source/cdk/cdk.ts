import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as subscriptions from 'aws-cdk-lib/aws-sns-subscriptions';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as firehose from 'aws-cdk-lib/aws-kinesisfirehose'; // L1 construct for Kinesis Data Firehose
import { CfnDeliveryStream } from 'aws-cdk-lib/aws-kinesisfirehose'; // L1 for specific props
import * as ssm from 'aws-cdk-lib/aws-ssm';

// Interface for Stack Props
export interface AutomatedSecurityResponseStackProps extends cdk.StackProps {
  adminAccountId: string;
  memberAccountId: string;
  primaryRegion: string;
  snsNotificationEmail: string;
  logRetentionDays: number;
  enableS3BucketEncryption: boolean;
  enableSnsTopicEncryption: boolean;
}

// Main Stack Class
export class AutomatedSecurityResponseStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: AutomatedSecurityResponseStackProps) {
    super(scope, id, props);

    // === Get context variables ===
    const adminAccountId = props.adminAccountId;
    const memberAccountId = props.memberAccountId; // Used for IAM policies, SSM targets
    const snsNotificationEmail = props.snsNotificationEmail;
    const logRetentionDays = props.logRetentionDays;
    const enableS3BucketEncryption = props.enableS3BucketEncryption;
    const enableSnsTopicEncryption = props.enableSnsTopicEncryption;

    // === IAM Roles ===

    // Remediation Role (to be assumed by SSM in Member Account)
    // This role would typically be deployed in the member account(s).
    // For this consolidated stack, we define it here and it would need to be created in the member account
    // or a trust relationship established if SSM in admin account is to manage member accounts.
    const remediationRole = new iam.Role(this, 'RemediationRole', {
      assumedBy: new iam.ServicePrincipal('ssm.amazonaws.com'),
      description: 'IAM Role for SSM to execute remediations in member accounts',
      // Add specific permissions required for the remediations SSM will perform
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AWSSupportAccess'), // Example, tailor to actual needs
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ReadOnlyAccess'), // Example
      ],
    });
    // Output Remediation Role ARN to be configured in Member accounts if not deploying there directly
    new cdk.CfnOutput(this, 'RemediationRoleArn', {
      value: remediationRole.roleArn,
      description: 'ARN of the Remediation Role to be used by SSM in member accounts',
    });

    // Lambda Execution Role
    const lambdaExecutionRole = new iam.Role(this, 'LambdaExecutionRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
      ],
    });
    lambdaExecutionRole.addToPolicy(new iam.PolicyStatement({
      actions: ['ssm:startAutomationExecution', 'ssm:getParameter'],
      resources: [`arn:aws:ssm:${props.primaryRegion}:${adminAccountId}:automation-definition/*`, `arn:aws:ssm:${props.primaryRegion}:${memberAccountId}:document/*`, `arn:aws:ssm:${props.primaryRegion}:${memberAccountId}:automation-execution/*`, `arn:aws:ssm:${props.primaryRegion}:${adminAccountId}:parameter/*`],
    }));
    lambdaExecutionRole.addToPolicy(new iam.PolicyStatement({
        actions: ['logs:CreateLogGroup', 'logs:CreateLogStream', 'logs:PutLogEvents'],
        resources: [`arn:aws:logs:${props.primaryRegion}:${adminAccountId}:log-group:/aws/lambda/*`],
    }));

    // Step Functions Execution Role
    const stepFunctionsExecutionRole = new iam.Role(this, 'StepFunctionsExecutionRole', {
      assumedBy: new iam.ServicePrincipal('states.amazonaws.com'),
      // Policy to allow invoking Lambda functions & other services like SNS
    });
    stepFunctionsExecutionRole.addToPolicy(new iam.PolicyStatement({
      actions: ['lambda:InvokeFunction'],
      resources: ['*'], // Restrict this to specific Lambda ARNs in production
    }));
    stepFunctionsExecutionRole.addToPolicy(new iam.PolicyStatement({
      actions: ['sns:Publish'],
      resources: ['*'], // Restrict this to the specific SNS topic ARN
    }));
    // If Step Functions needs to assume roles in other accounts or start SSM directly (not via Lambda):
    stepFunctionsExecutionRole.addToPolicy(new iam.PolicyStatement({
        actions: ['sts:AssumeRole'], // If assuming roles in other accounts directly
        resources: [`arn:aws:iam::${memberAccountId}:role/*`], // Example, be specific
    }));

    // === Storage (S3) ===
    const metricsBucket = new s3.Bucket(this, 'MetricsBucket', {
      bucketName: `automated-security-metrics-${adminAccountId}-${props.primaryRegion}`,
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN, // Protect data on stack deletion
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: enableS3BucketEncryption ? s3.BucketEncryption.S3_MANAGED : s3.BucketEncryption.UNENCRYPTED,
      lifecycleRules: [
        {
          enabled: true,
          expiration: cdk.Duration.days(365), // Example: delete objects after 1 year
          noncurrentVersionExpiration: cdk.Duration.days(90),
          transitions: [
            {
              storageClass: s3.StorageClass.INFREQUENT_ACCESS,
              transitionAfter: cdk.Duration.days(30),
            },
            {
              storageClass: s3.StorageClass.GLACIER,
              transitionAfter: cdk.Duration.days(90),
            },
          ],
        },
      ],
    });
    // Bucket policy to allow Firehose to write to it
    metricsBucket.addToResourcePolicy(new iam.PolicyStatement({
        actions: ['s3:AbortMultipartUpload', 's3:GetBucketLocation', 's3:GetObject', 's3:ListBucket', 's3:ListBucketMultipartUploads', 's3:PutObject'],
        resources: [metricsBucket.bucketArn, `${metricsBucket.bucketArn}/*`],
        principals: [new iam.ServicePrincipal('firehose.amazonaws.com')],
        // Condition to ensure it's this specific Firehose stream (optional but good practice)
        // conditions: {
        //   StringEquals: { "aws:SourceArn": firehoseDeliveryStream.attrArn } // Placeholder
        // }
    }));

    // === Monitoring and Logging ===

    // CloudWatch Log Group for Lambda functions and SSM
    const centralLogGroup = new logs.LogGroup(this, 'CentralLogGroup', {
      logGroupName: '/aws/automated-security/central-logs',
      retention: logRetentionDays, // From context
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Kinesis Data Firehose to S3
    // Firehose Role
    const firehoseRole = new iam.Role(this, 'FirehoseS3DeliveryRole', {
        assumedBy: new iam.ServicePrincipal('firehose.amazonaws.com'),
    });
    firehoseRole.addToPolicy(new iam.PolicyStatement({
        actions: ['s3:AbortMultipartUpload', 's3:GetBucketLocation', 's3:GetObject', 's3:ListBucket', 's3:ListBucketMultipartUploads', 's3:PutObject'],
        resources: [metricsBucket.bucketArn, `${metricsBucket.bucketArn}/*`],
    }));
    firehoseRole.addToPolicy(new iam.PolicyStatement({ // Policy for CloudWatch Logs
        actions: ['logs:PutLogEvents'],
        resources: [`arn:aws:logs:${props.primaryRegion}:${adminAccountId}:log-group:/aws/kinesisfirehose/*:log-stream:*`], // Firehose's own logs
    }));

    const firehoseDeliveryStream = new CfnDeliveryStream(this, 'MetricsFirehoseStream', {
      deliveryStreamName: 'automated-security-metrics-stream',
      deliveryStreamType: 'DirectPut',
      s3DestinationConfiguration: {
        bucketArn: metricsBucket.bucketArn,
        roleArn: firehoseRole.roleArn,
        bufferingHints: {
          intervalInSeconds: 300,
          sizeInMBs: 5,
        },
        compressionFormat: 'GZIP', // UNCOMPRESSED, GZIP, ZIP, Snappy
        cloudWatchLoggingOptions: {
            enabled: true,
            logGroupName: centralLogGroup.logGroupName,
            logStreamName: 'firehose-s3-delivery-logs'
        }
        // Add encryptionConfiguration if needed
      },
    });

    // Subscription filter for the central log group to Firehose
    new logs.SubscriptionFilter(this, 'LogSubscriptionFilter', {
      logGroup: centralLogGroup,
      destination: new targets.KinesisFirehoseStream(firehoseDeliveryStream as any), // Cast needed as L1 returns CfnResource
      filterPattern: logs.FilterPattern.allEvents(), // Adjust filter pattern as needed
    });

    // === SNS Notification Topic ===
    const notificationTopic = new sns.Topic(this, 'SecurityNotificationTopic', {
      displayName: 'Security Event Notifications',
      topicName: 'automated-security-notifications',
      fifo: false, // Standard topic
    });
    if (enableSnsTopicEncryption) {
        notificationTopic.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY); // Or RETAIN
        // Consider adding KMS key for encryption: notificationTopic.encryptionKey = kmsKey;
    }
    notificationTopic.addSubscription(new subscriptions.EmailSubscription(snsNotificationEmail));

    // === Lambda Functions ===
    // Placeholder for Lambda code - In a real scenario, this would be in separate files or inline
    const commonLambdaProps = {
      runtime: lambda.Runtime.NODEJS_18_X,
      handler: 'index.handler',
      role: lambdaExecutionRole,
      logRetention: logRetentionDays,
      environment: {
        PRIMARY_REGION: props.primaryRegion,
        ADMIN_ACCOUNT_ID: adminAccountId,
        MEMBER_ACCOUNT_ID: memberAccountId,
        NOTIFICATION_TOPIC_ARN: notificationTopic.topicArn,
        CENTRAL_LOG_GROUP_NAME: centralLogGroup.logGroupName,
      },
    };

    const checkStateLambda = new lambda.Function(this, 'CheckStateLambda', {
      ...commonLambdaProps,
      functionName: 'AutomatedSecurity-CheckState',
      code: lambda.Code.fromInline(
        'exports.handler = async (event) => { console.log("CheckStateLambda invoked:", event); return { ...event, stateChecked: true }; };'
      ),
      description: 'Lambda function to check state based on Security Hub finding',
    });

    const executeRemediationLambda = new lambda.Function(this, 'ExecuteRemediationLambda', {
      ...commonLambdaProps,
      functionName: 'AutomatedSecurity-ExecuteRemediation',
      code: lambda.Code.fromInline(
        `exports.handler = async (event) => { 
          console.log("ExecuteRemediationLambda invoked:", event);
          // Example: Trigger SSM Automation Document
          // const ssm = new AWS.SSM(); 
          // await ssm.startAutomationExecution({...}).promise();
          return { ...event, remediationExecuted: true }; 
        };`
      ),
      description: 'Lambda function to execute remediation tasks, e.g., via SSM',
    });
    // Grant permission for this lambda to log to the central log group
    centralLogGroup.grantWrite(executeRemediationLambda);

    const sendNotificationLambda = new lambda.Function(this, 'SendNotificationLambda', {
      ...commonLambdaProps,
      functionName: 'AutomatedSecurity-SendNotification',
      code: lambda.Code.fromInline(
        `exports.handler = async (event) => { 
          console.log("SendNotificationLambda invoked:", event);
          // const sns = new AWS.SNS();
          // await sns.publish({ TopicArn: process.env.NOTIFICATION_TOPIC_ARN, Message: JSON.stringify(event) }).promise();
          return { ...event, notificationSent: true }; 
        };`
      ),
      description: 'Lambda function to send notifications via SNS',
    });
    notificationTopic.grantPublish(sendNotificationLambda); // Grant publish permissions

    // === Systems Manager (SSM) Automation Document (Placeholder) ===
    // This would typically be a separate YAML/JSON file or created via CDK
    // For simplicity, we reference it by name, assuming it exists or will be created in member accounts.
    const ssmAutomationDocumentName = 'AWSConfigRemediation-Example'; // Replace with your actual document name
    // If you were to create one via CDK (more complex for cross-account if document is in member):
    // new ssm.CfnDocument(this, 'MySsmAutomationDoc', { ... });

    // === Step Functions State Machine ===
    const checkStateTask = new tasks.LambdaInvoke(this, 'CheckFindingState', {
      lambdaFunction: checkStateLambda,
      outputPath: '$.Payload',
    });

    const executeRemediationTask = new tasks.LambdaInvoke(this, 'ExecuteRemediationAction', {
      lambdaFunction: executeRemediationLambda,
      outputPath: '$.Payload',
    });

    const sendNotificationTask = new tasks.LambdaInvoke(this, 'SendNotificationOfAction', {
      lambdaFunction: sendNotificationLambda,
      inputPath: '$', // Send the whole state machine input to the notification lambda
      resultPath: '$.NotificationResult', // Optional: store notification result
    });

    const remediationSucceeded = new sfn.Succeed(this, 'RemediationSucceeded');
    const remediationFailed = new sfn.Fail(this, 'RemediationFailed', {
      comment: 'Remediation action failed or an error occurred.',
    });

    const definition = checkStateTask
      .next(executeRemediationTask)
      .next(
        new sfn.Choice(this, 'IsRemediationComplete?')
          .when(sfn.Condition.booleanEquals('$.remediationExecuted', true), sendNotificationTask.next(remediationSucceeded))
          .otherwise(remediationFailed)
      );

    const stateMachine = new sfn.StateMachine(this, 'SecurityResponseOrchestrator', {
      stateMachineName: 'AutomatedSecurityResponseOrchestrator',
      definitionBody: sfn.DefinitionBody.fromChainable(definition),
      role: stepFunctionsExecutionRole,
      timeout: cdk.Duration.minutes(30),
      tracingEnabled: true, // Enable X-Ray tracing
      logs: {
        destination: centralLogGroup,
        level: sfn.LogLevel.ALL,
        includeExecutionData: true,
      },
    });

    // === EventBridge Rule (Security Hub Findings) ===
    const securityHubRule = new events.Rule(this, 'SecurityHubFindingsRule', {
      ruleName: 'SecurityHubFindingsToStepFunctions',
      description: 'Rule to capture Security Hub findings and trigger Step Functions',
      eventPattern: {
        source: ['aws.securityhub'],
        detailType: ['Security Hub Findings - Imported'], // Adjust as per your findings source
        // You can add more filters here, e.g., for specific finding types or severity
        // detail: {
        //   findings: {
        //     Severity: {
        //       Label: ["HIGH", "CRITICAL"]
        //     }
        //   }
        // }
      },
    });
    securityHubRule.addTarget(new targets.SfnStateMachine(stateMachine));

    // === Outputs ===
    new cdk.CfnOutput(this, 'StateMachineArn', {
      value: stateMachine.stateMachineArn,
      description: 'ARN of the Step Functions State Machine',
    });
    new cdk.CfnOutput(this, 'MetricsS3BucketName', {
      value: metricsBucket.bucketName,
      description: 'Name of the S3 bucket for metrics and logs',
    });
    new cdk.CfnOutput(this, 'NotificationTopicArnOutput', {
      value: notificationTopic.topicArn,
      description: 'ARN of the SNS topic for notifications',
    });
    new cdk.CfnOutput(this, 'CentralLogGroupNameOutput', {
        value: centralLogGroup.logGroupName,
        description: 'Name of the Central CloudWatch Log Group'
    });
  }
}

// === CDK App Entry Point ===
const app = new cdk.App();

// Retrieve context values or use defaults
const adminAccountId = app.node.tryGetContext('adminAccountId') || process.env.CDK_DEFAULT_ACCOUNT;
const memberAccountId = app.node.tryGetContext('memberAccountId') || 'MEMBER_ACCOUNT_PLACEHOLDER'; // Requires actual member ID for some policies
const primaryRegion = app.node.tryGetContext('primaryRegion') || process.env.CDK_DEFAULT_REGION;
const snsNotificationEmail = app.node.tryGetContext('snsNotificationEmail') || 'placeholder-email@example.com';
const logRetentionDays = app.node.tryGetContext('logRetentionDays') || 7;
const enableS3BucketEncryption = app.node.tryGetContext('featureFlags')?.ENABLE_S3_BUCKET_ENCRYPTION ?? true;
const enableSnsTopicEncryption = app.node.tryGetContext('featureFlags')?.ENABLE_SNS_TOPIC_ENCRYPTION ?? true;

if (!adminAccountId || !primaryRegion) {
  throw new Error(
    "Required context variables 'adminAccountId' and 'primaryRegion' are missing. " +
    "Please provide them in cdk.json or as command line context (-c variable=value)."
  );
}
if (memberAccountId === 'MEMBER_ACCOUNT_PLACEHOLDER') {
    console.warn("Warning: 'memberAccountId' is not set. Some IAM policies might be too broad or incorrect for cross-account SSM.");
}
if (snsNotificationEmail === 'placeholder-email@example.com') {
    console.warn("Warning: 'snsNotificationEmail' is using a placeholder. Please update in cdk.json.");
}

const stackProps: AutomatedSecurityResponseStackProps = {
  env: {
    account: adminAccountId,
    region: primaryRegion,
  },
  adminAccountId,
  memberAccountId,
  primaryRegion,
  snsNotificationEmail,
  logRetentionDays,
  enableS3BucketEncryption,
  enableSnsTopicEncryption,
  description: 'AWS CDK Stack for Automated Security Response based on Security Hub findings.',
  tags: app.node.tryGetContext('tags') || { Project: 'AutomatedSecurityResponse', Environment: 'Development', ManagedBy: 'CDK' },
};

new AutomatedSecurityResponseStack(app, 'AutomatedSecurityResponseStack', stackProps);

app.synth(); 