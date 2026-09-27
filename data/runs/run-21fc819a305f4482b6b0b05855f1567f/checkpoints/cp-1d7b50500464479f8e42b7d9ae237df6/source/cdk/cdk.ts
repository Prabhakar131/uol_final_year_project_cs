import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import { RemovalPolicy, Duration } from 'aws-cdk-lib';

// Define a props interface for our main stack for type safety and clarity
export interface CostManagementStackProps extends cdk.StackProps {
  readonly environment: string;
  readonly appName: string;
  readonly bucketNamePrefix: string;
  readonly param1: string;
}

// Single Stack Class Definition
export class CostManagementStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: CostManagementStackProps) {
    super(scope, id, props);

    // --- Environment-aware configurations ---
    const isProd = props.environment === 'prod';
    const bucketName = `${props.bucketNamePrefix}-${props.environment}`;

    // --- S3 Bucket for data storage or static assets ---
    // Implements backup strategy through versioning and a lifecycle rule for non-current versions.
    // Uses server-side encryption with AWS-managed keys (SSE-S3).
    const dataBucket = new s3.Bucket(this, 'DataBucket', {
      bucketName: bucketName,
      versioned: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL, // Security best practice
      removalPolicy: isProd ? RemovalPolicy.RETAIN : RemovalPolicy.DESTROY, // Retain in prod, destroy in dev
      autoDeleteObjects: !isProd, // Automatically delete objects when bucket is destroyed in non-prod
      lifecycleRules: [
        {
          id: 'ArchiveNonCurrentVersions',
          enabled: true,
          noncurrentVersionExpiration: Duration.days(isProd ? 365 : 30), // Keep non-current versions longer in prod
          noncurrentVersionTransitions: [
            {
              storageClass: s3.StorageClass.INFREQUENT_ACCESS,
              transitionAfter: Duration.days(isProd ? 90 : 15),
            },
            {
              storageClass: s3.StorageClass.GLACIER,
              transitionAfter: Duration.days(isProd ? 180 : 30),
            },
          ],
        },
        {
            id: 'AbortIncompleteMultipartUploads',
            enabled: true,
            abortIncompleteMultipartUpload: Duration.days(7),
        }
      ],
      serverAccessLogsPrefix: 'access-logs/', // Enable server access logging
    });

    // --- DynamoDB Table for structured data ---
    // Implements backup strategy (PITR) and encryption at rest (AWS-owned key by default).
    const dataTable = new dynamodb.Table(this, 'DataTable', {
      partitionKey: { name: 'id', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST, // Cost-effective for unpredictable workloads
      removalPolicy: isProd ? RemovalPolicy.RETAIN : RemovalPolicy.DESTROY,
      pointInTimeRecovery: isProd, // Enable PITR for production environments
      encryption: dynamodb.TableEncryption.AWS_MANAGED, // Default, can be changed to KMS if specific key needed
      // stream: dynamodb.StreamViewType.NEW_AND_OLD_IMAGES, // Uncomment if you need to capture changes
    });

    // --- IAM Role for Lambda Function ---
    // Follows least privilege principle.
    const lambdaExecutionRole = new iam.Role(this, 'LambdaExecutionRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      description: 'Execution role for the data processing Lambda function',
    });

    // Grant necessary permissions to the Lambda role
    lambdaExecutionRole.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole')
    );
    dataBucket.grantReadWrite(lambdaExecutionRole); // Permission to read/write to the S3 bucket
    dataTable.grantReadWriteData(lambdaExecutionRole); // Permission to read/write data to the DynamoDB table

    // Add a more specific policy for fine-grained control if needed
    lambdaExecutionRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ['s3:GetObjectTagging', 's3:PutObjectTagging'],
        resources: [dataBucket.bucketArn + '/*'],
        effect: iam.Effect.ALLOW,
      })
    );

    // --- Lambda Function for data processing ---
    // Includes environment variables and logging configuration.
    const dataProcessorLambda = new lambda.Function(this, 'DataProcessorLambda', {
      runtime: lambda.Runtime.NODEJS_18_X,
      handler: 'index.handler', // Assumes an 'index.js' file with a 'handler' function
      code: lambda.Code.fromAsset('lambda-handler'), // Placeholder for your Lambda code directory
      role: lambdaExecutionRole,
      environment: {
        BUCKET_NAME: dataBucket.bucketName,
        TABLE_NAME: dataTable.tableName,
        PARAM1_VALUE: props.param1,
        LOG_LEVEL: isProd ? 'INFO' : 'DEBUG',
      },
      timeout: Duration.seconds(30),
      memorySize: 256,
      logRetention: isProd ? logs.RetentionDays.ONE_YEAR : logs.RetentionDays.ONE_WEEK, // Longer retention for prod
      tracing: lambda.Tracing.ACTIVE, // Enable AWS X-Ray tracing
    });

    // --- API Gateway to expose the Lambda function ---
    // Configured with logging and a specific deployment stage.
    const api = new apigateway.LambdaRestApi(this, 'Endpoint', {
      handler: dataProcessorLambda,
      proxy: true, // Proxies all requests to the Lambda function
      description: `API Gateway for ${props.appName}`,
      deployOptions: {
        stageName: props.environment, // Use environment name as stage name
        tracingEnabled: true, // Enable X-Ray tracing for API Gateway
        loggingLevel: apigateway.MethodLoggingLevel.INFO,
        dataTraceEnabled: true, // Log full message data for request/response (can be verbose)
        metricsEnabled: true, // Enable detailed CloudWatch metrics
      },
      defaultMethodOptions: {
        // Define default method options, e.g., authorization (not shown for brevity)
      },
      endpointConfiguration: {
        types: [apigateway.EndpointType.REGIONAL],
      },
      // It's good practice to define a usage plan for APIs
      // const usagePlan = api.addUsagePlan(...);
      // usagePlan.addApiKey(...);
    });

    // --- Security Group (Example, if needed for VPC resources) ---
    // This is a placeholder. If your Lambda or other resources were in a VPC,
    // you would define security groups here.
    // const vpc = ec2.Vpc.fromLookup(this, 'DefaultVPC', { isDefault: true }); // Example: using default VPC
    // const mySecurityGroup = new ec2.SecurityGroup(this, 'MyLambdaSG', {
    //   vpc,
    //   description: 'Security group for Lambda in VPC',
    //   allowAllOutbound: true, // Or restrict as needed
    // });
    // dataProcessorLambda.connections.addSecurityGroup(mySecurityGroup);

    // --- Nested Construct Example (Optional for further organization) ---
    // new MyCustomNestedConstruct(this, 'MyNestedResources', {
    //   mainBucket: dataBucket,
    //   environment: props.environment,
    // });

    // --- CloudFormation Outputs ---
    new cdk.CfnOutput(this, 'BucketNameOutput', {
      value: dataBucket.bucketName,
      description: 'Name of the S3 data bucket',
      exportName: `${props.appName}-DataBucketName-${props.environment}`,
    });

    new cdk.CfnOutput(this, 'TableNameOutput', {
      value: dataTable.tableName,
      description: 'Name of the DynamoDB data table',
      exportName: `${props.appName}-DataTableName-${props.environment}`,
    });

    new cdk.CfnOutput(this, 'LambdaFunctionNameOutput', {
      value: dataProcessorLambda.functionName,
      description: 'Name of the data processor Lambda function',
      exportName: `${props.appName}-DataProcessorLambdaName-${props.environment}`,
    });

    new cdk.CfnOutput(this, 'ApiEndpointOutput', {
      value: api.url,
      description: 'Endpoint URL for the API Gateway',
      exportName: `${props.appName}-ApiEndpoint-${props.environment}`,
    });
  }
}

// --- Main CDK Application ---
const app = new cdk.App();

// Retrieve context variables defined in cdk.json or via CLI
const environment = app.node.tryGetContext('env') || 'dev'; // Default to 'dev' if not set
const appName = app.node.tryGetContext('appName') || 'CostManagementApp';
const region = app.node.tryGetContext('region') || process.env.CDK_DEFAULT_REGION || 'us-east-1';

// Fetch environment-specific configurations
const envConfig = app.node.tryGetContext(environment) || {}; // Empty object if environment block not found
const bucketNamePrefix = envConfig.bucketNamePrefix || 'cost-app'; // Default prefix
const param1 = envConfig.param1 || 'defaultValue1';

// Instantiate the stack
new CostManagementStack(app, `${appName}-Stack-${environment}`, {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: region,
  },
  environment: environment,
  appName: appName,
  bucketNamePrefix: bucketNamePrefix,
  param1: param1,
  description: `Stack for ${appName} in ${environment} environment. Manages resources for cost management solution.`,
  tags: { // Stack-specific tags, will merge with/override global tags from cdk.json
    Environment: environment, // More specific environment tag
    Application: appName,
  },
});

// Apply global tags from cdk.json to the app, which then applies to stacks
const globalTags = app.node.tryGetContext('tags') || {};
for (const [key, value] of Object.entries(globalTags)) {
  cdk.Tags.of(app).add(key, value as string);
}

app.synth();

/*
// --- Example Nested Construct Definition (Optional) ---
interface MyCustomNestedConstructProps {
  readonly mainBucket: s3.IBucket;
  readonly environment: string;
}

class MyCustomNestedConstruct extends Construct {
  constructor(scope: Construct, id: string, props: MyCustomNestedConstructProps) {
    super(scope, id);

    // Example: Create another S3 bucket within this nested construct
    const nestedBucket = new s3.Bucket(this, 'NestedBucket', {
      bucketName: `nested-data-${props.environment}-${cdk.Aws.ACCOUNT_ID}`.toLowerCase(),
      removalPolicy: props.environment === 'prod' ? RemovalPolicy.RETAIN : RemovalPolicy.DESTROY,
      autoDeleteObjects: props.environment !== 'prod',
      versioned: true,
    });

    // Grant the main bucket's user (e.g., Lambda role) access to this nested bucket if needed
    // props.mainBucket.grantRead(new iam.AccountRootPrincipal()); // Example, be specific with principals

    new cdk.CfnOutput(this, 'NestedBucketNameOutput', {
        value: nestedBucket.bucketName,
        description: 'Name of the nested S3 bucket',
    });
  }
}
*/ 