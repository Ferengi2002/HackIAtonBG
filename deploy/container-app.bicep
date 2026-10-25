targetScope = 'resourceGroup'

param location string
@minLength(2)
@maxLength(32)
param containerAppName string
param managedEnvironmentName string
param identityName string
param logAnalyticsName string
param registryName string
param registryResourceGroupName string
@description('Imagen de esta ejecucion, fijada por digest sha256 de ACR.')
param containerImage string
param revisionSuffix string
@minValue(1)
param minReplicas int = 1
@minValue(1)
param maxReplicas int = 3
param azureOpenAiEndpoint string = ''
param azureOpenAiDeployment string = 'gpt-4o'
param azureOpenAiApiVersion string = '2024-10-21'
@description('Opcional. Vault RBAC existente que contiene la clave de Foundry.')
param keyVaultName string = ''
param keyVaultResourceGroupName string = resourceGroup().name
param openAiKeySecretName string = 'azure-openai-api-key'
param tags object = {
  project: 'ia-credit-bg'
  environment: 'dev'
  managedBy: 'AzurePipelines'
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
  tags: tags
}

// El registro puede estar en un grupo distinto al de la aplicacion.
module registryAccess 'modules/acr-pull.bicep' = {
  name: 'acr-pull-${uniqueString(containerAppName, registryName)}'
  scope: resourceGroup(registryResourceGroupName)
  params: {
    registryName: registryName
    principalId: identity.properties.principalId
  }
}

module keyVaultAccess 'modules/key-vault-access.bicep' = if (!empty(keyVaultName)) {
  name: 'key-vault-${uniqueString(containerAppName, keyVaultName)}'
  scope: resourceGroup(keyVaultResourceGroupName)
  params: {
    keyVaultName: keyVaultName
    principalId: identity.properties.principalId
    secretName: openAiKeySecretName
  }
}

resource workspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
}

resource managedEnvironment 'Microsoft.App/managedEnvironments@2025-01-01' = {
  name: managedEnvironmentName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'azure-monitor'
    }
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
  }
}

resource diagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name: 'container-app-logs'
  scope: managedEnvironment
  properties: {
    workspaceId: workspace.id
    logs: [
      {
        category: 'ContainerAppConsoleLogs'
        enabled: true
      }
      {
        category: 'ContainerAppSystemLogs'
        enabled: true
      }
    ]
  }
}

resource containerApp 'Microsoft.App/containerApps@2025-01-01' = {
  name: containerAppName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: managedEnvironment.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      registries: [
        {
          server: registryAccess.outputs.loginServer
          identity: identity.id
        }
      ]
      secrets: !empty(keyVaultName) ? [
        {
          name: 'azure-openai-api-key'
          keyVaultUrl: keyVaultAccess!.outputs.secretUri
          identity: identity.id
        }
      ] : []
    }
    template: {
      revisionSuffix: revisionSuffix
      containers: [
        {
          name: 'copiloto-api'
          image: containerImage
          resources: {
            cpu: 1
            memory: '2Gi'
          }
          env: concat([
            {
              name: 'BG_RISK_MODEL_PATH'
              value: 'models/bg_risk_lightgbm.joblib'
            }
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: azureOpenAiEndpoint
            }
            {
              name: 'AZURE_OPENAI_DEPLOYMENT'
              value: azureOpenAiDeployment
            }
            {
              name: 'AZURE_OPENAI_API_VERSION'
              value: azureOpenAiApiVersion
            }
          ], !empty(keyVaultName) ? [
            {
              name: 'AZURE_OPENAI_API_KEY'
              secretRef: 'azure-openai-api-key'
            }
          ] : [])
          probes: [for probeType in ['Startup', 'Readiness', 'Liveness']: {
            type: probeType
            httpGet: {
              path: '/healthz'
              port: 8000
              scheme: 'HTTP'
            }
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: probeType == 'Startup' ? 30 : 3
          }]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-concurrency'
            http: {
              metadata: {
                concurrentRequests: '20'
              }
            }
          }
        ]
      }
    }
  }
}

output appName string = containerApp.name
output appUrl string = 'https://${containerApp.properties.configuration.ingress!.fqdn}'
output revisionName string = '${containerAppName}--${revisionSuffix}'
output image string = containerImage
output identityId string = identity.id
