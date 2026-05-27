// =======================================================================
// monitoring.bicep
// Provisions observability infrastructure for the Wine Quality Classifier
// pipeline: Log Analytics Workspace, Application Insights, Action Group,
// and three Alert Rules (latency, error rate, pipeline failure).
// =======================================================================

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Short environment tag appended to resource names (e.g. prod, staging).')
param environmentName string = 'prod'

@description('E-mail address that receives alert notifications.')
param alertEmail string

@description('Slack incoming-webhook URL for alert notifications.')
param slackWebhookUrl string

// -----------------------------------------------------------------------
// Variables
// -----------------------------------------------------------------------
var prefix = 'wine-quality-${environmentName}'
var logAnalyticsName = '${prefix}-law'
var appInsightsName  = '${prefix}-ai'
var actionGroupName  = '${prefix}-ag'

// -----------------------------------------------------------------------
// Log Analytics Workspace
// -----------------------------------------------------------------------
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 90
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
  tags: {
    project: 'wine-quality-classifier'
    environment: environmentName
    managed_by: 'bicep'
  }
}

// -----------------------------------------------------------------------
// Application Insights (linked to Log Analytics)
// -----------------------------------------------------------------------
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
    RetentionInDays: 90
  }
  tags: {
    project: 'wine-quality-classifier'
    environment: environmentName
    managed_by: 'bicep'
  }
}

// -----------------------------------------------------------------------
// Action Group — email + Slack webhook
// -----------------------------------------------------------------------
resource actionGroup 'Microsoft.Insights/actionGroups@2023-01-01' = {
  name: actionGroupName
  location: 'global'
  properties: {
    groupShortName: 'WineQuality'
    enabled: true
    emailReceivers: [
      {
        name: 'MLOps On-Call'
        emailAddress: alertEmail
        useCommonAlertSchema: true
      }
    ]
    webhookReceivers: [
      {
        name: 'Slack'
        serviceUri: slackWebhookUrl
        useCommonAlertSchema: true
        useAadAuth: false
      }
    ]
  }
  tags: {
    project: 'wine-quality-classifier'
    environment: environmentName
  }
}

// -----------------------------------------------------------------------
// Alert Rule 1 — Endpoint P95 Latency > 2000 ms
// -----------------------------------------------------------------------
resource latencyAlert 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: '${prefix}-latency-p95-alert'
  location: 'global'
  properties: {
    description: 'Fires when the P95 response latency of the wine-quality-prod endpoint exceeds 2000 ms.'
    severity: 2
    enabled: true
    evaluationFrequency: 'PT1M'
    windowSize: 'PT5M'
    targetResourceType: 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints'
    scopes: []   // populated at deploy time via CLI --set
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          name: 'LatencyP95'
          metricName: 'RequestLatency_P95'
          metricNamespace: 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints'
          operator: 'GreaterThan'
          threshold: 2000
          timeAggregation: 'Average'
          criterionType: 'StaticThresholdCriterion'
        }
      ]
    }
    actions: [
      {
        actionGroupId: actionGroup.id
        webHookProperties: {}
      }
    ]
    autoMitigate: true
  }
  tags: {
    project: 'wine-quality-classifier'
    alert_type: 'latency'
  }
}

// -----------------------------------------------------------------------
// Alert Rule 2 — Endpoint Error Rate > 5 %
// -----------------------------------------------------------------------
resource errorRateAlert 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: '${prefix}-error-rate-alert'
  location: 'global'
  properties: {
    description: 'Fires when the HTTP 5xx error rate of the wine-quality-prod endpoint exceeds 5%.'
    severity: 1
    enabled: true
    evaluationFrequency: 'PT1M'
    windowSize: 'PT5M'
    targetResourceType: 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints'
    scopes: []
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          name: 'ErrorRate'
          metricName: 'RequestsFailedRate'
          metricNamespace: 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints'
          operator: 'GreaterThan'
          threshold: 5
          timeAggregation: 'Average'
          criterionType: 'StaticThresholdCriterion'
        }
      ]
    }
    actions: [
      {
        actionGroupId: actionGroup.id
        webHookProperties: {}
      }
    ]
    autoMitigate: true
  }
  tags: {
    project: 'wine-quality-classifier'
    alert_type: 'error-rate'
  }
}

// -----------------------------------------------------------------------
// Alert Rule 3 — GitHub Actions Pipeline Failure (custom metric via App Insights)
// A GitHub Actions workflow emits a custom event "PipelineFailure" to App Insights.
// This alert fires when that custom metric count >= 1 in any 5-minute window.
// -----------------------------------------------------------------------
resource pipelineFailureAlert 'Microsoft.Insights/scheduledQueryRules@2022-06-15' = {
  name: '${prefix}-pipeline-failure-alert'
  location: location
  properties: {
    description: 'Fires when a GitHub Actions CI/CD pipeline failure event is received in Application Insights.'
    severity: 1
    enabled: true
    evaluationFrequency: 'PT5M'
    windowSize: 'PT5M'
    scopes: [
      appInsights.id
    ]
    criteria: {
      allOf: [
        {
          query: '''
            customEvents
            | where name == "PipelineFailure"
            | where customDimensions.project == "wine-quality-classifier"
            | summarize FailureCount = count()
          '''
          timeAggregation: 'Count'
          metricMeasureColumn: 'FailureCount'
          operator: 'GreaterThanOrEqual'
          threshold: 1
          failingPeriods: {
            numberOfEvaluationPeriods: 1
            minFailingPeriodsToAlert: 1
          }
        }
      ]
    }
    actions: {
      actionGroups: [
        actionGroup.id
      ]
    }
    autoMitigate: false
  }
  tags: {
    project: 'wine-quality-classifier'
    alert_type: 'pipeline-failure'
  }
}

// -----------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------
output appInsightsConnectionString string = appInsights.properties.ConnectionString
output appInsightsInstrumentationKey string = appInsights.properties.InstrumentationKey
output logAnalyticsWorkspaceId string = logAnalytics.properties.customerId
output logAnalyticsResourceId string = logAnalytics.id
output actionGroupResourceId string = actionGroup.id
