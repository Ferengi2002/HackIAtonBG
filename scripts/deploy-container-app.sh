#!/usr/bin/env bash
set -euo pipefail

: "${RESOURCE_GROUP:?Falta RESOURCE_GROUP}"
: "${ACR_NAME:?Falta ACR_NAME}"
: "${IMAGE_REPOSITORY:?Falta IMAGE_REPOSITORY}"
: "${CONTAINER_IMAGE:?Falta la imagen de BuildAndPush}"
: "${REVISION_SUFFIX:?Falta REVISION_SUFFIX}"
: "${DEPLOYMENT_NAME:?Falta DEPLOYMENT_NAME}"
: "${TEMPLATE_FILE:?Falta TEMPLATE_FILE}"
: "${PARAMETERS_FILE:?Falta PARAMETERS_FILE}"
: "${DEPLOY_REPORT_DIR:?Falta DEPLOY_REPORT_DIR}"

mkdir -p "$DEPLOY_REPORT_DIR"
az extension add --name containerapp --upgrade --only-show-errors

# Solo se consume el registro existente; el grupo esta administrado por Terraform.
az group show --name "$RESOURCE_GROUP" --output none
registry_json=$(az acr show --name "$ACR_NAME" --output json)
registry_group=$(jq -r '.resourceGroup' <<< "$registry_json")
registry_server=$(jq -r '.loginServer' <<< "$registry_json")
permission_mode=$(jq -r '.roleAssignmentMode // "LegacyRegistryPermissions"' <<< "$registry_json")
if [[ "$permission_mode" != 'LegacyRegistryPermissions' ]]; then
  echo 'Este despliegue requiere ACR en modo RBAC Registry Permissions para usar AcrPull.' >&2
  exit 1
fi
image_prefix="$registry_server/$IMAGE_REPOSITORY@"
image_digest="${CONTAINER_IMAGE#"$image_prefix"}"
if [[ "$CONTAINER_IMAGE" != "$image_prefix"* || ! "$image_digest" =~ ^sha256:[a-f0-9]{64}$ ]]; then
  echo 'La imagen debe ser el digest de este repositorio producido por BuildAndPush.' >&2
  exit 1
fi
arm_auth=$(az acr config authentication-as-arm show --registry "$ACR_NAME" --query status --output tsv)
if [[ "${arm_auth,,}" != 'enabled' ]]; then
  echo 'ACR debe permitir ARM audience tokens para descargar con identidad administrada.' >&2
  exit 1
fi

min_replicas=$(jq -r '.parameters.minReplicas.value' "$PARAMETERS_FILE")
max_replicas=$(jq -r '.parameters.maxReplicas.value' "$PARAMETERS_FILE")
if (( min_replicas < 1 || max_replicas < min_replicas )); then
  echo 'Configurar 1 <= minReplicas <= maxReplicas.' >&2
  exit 1
fi

parameters=(
  "@$PARAMETERS_FILE"
  "registryName=$ACR_NAME"
  "registryResourceGroupName=$registry_group"
  "containerImage=$CONTAINER_IMAGE"
  "revisionSuffix=$REVISION_SUFFIX"
)

az deployment group validate \
  --resource-group "$RESOURCE_GROUP" \
  --template-file "$TEMPLATE_FILE" \
  --parameters "${parameters[@]}" \
  --output none

az deployment group what-if \
  --resource-group "$RESOURCE_GROUP" \
  --template-file "$TEMPLATE_FILE" \
  --parameters "${parameters[@]}" \
  --no-pretty-print \
  --result-format FullResourcePayloads \
  --output json > "$DEPLOY_REPORT_DIR/what-if.json"

az deployment group create \
  --name "$DEPLOYMENT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --template-file "$TEMPLATE_FILE" \
  --parameters "${parameters[@]}" \
  --mode Incremental \
  --query properties.outputs \
  --output json > "$DEPLOY_REPORT_DIR/outputs.json"

app_name=$(jq -er '.appName.value' "$DEPLOY_REPORT_DIR/outputs.json")
app_url=$(jq -er '.appUrl.value' "$DEPLOY_REPORT_DIR/outputs.json")
revision=$(jq -er '.revisionName.value' "$DEPLOY_REPORT_DIR/outputs.json")

# En modo Single, Azure conserva el trafico en la revision sana hasta que la nueva este lista.
ready=false
for attempt in {1..40}; do
  ready_revision=$(az containerapp show \
    --name "$app_name" --resource-group "$RESOURCE_GROUP" \
    --query properties.latestReadyRevisionName --output tsv)
  if [[ "$ready_revision" == "$revision" ]]; then
    ready=true
    break
  fi
  echo "Esperando revision $revision ($attempt/40)..."
  sleep 15
done

az containerapp revision show \
  --name "$app_name" --resource-group "$RESOURCE_GROUP" --revision "$revision" \
  --output json > "$DEPLOY_REPORT_DIR/revision.json"
if [[ "$ready" != true ]]; then
  echo 'La nueva revision no alcanzo el estado Ready.' >&2
  exit 1
fi
deployed_image=$(jq -er '.properties.template.containers[0].image' "$DEPLOY_REPORT_DIR/revision.json")
if [[ "$deployed_image" != "$CONTAINER_IMAGE" ]]; then
  echo 'La revision activa no corresponde a la imagen de esta ejecucion.' >&2
  exit 1
fi

# Estas comprobaciones se ejecutan en el agente del pipeline, sobre el servicio desplegado.
curl --fail --silent --show-error --retry 12 --retry-all-errors \
  --retry-delay 5 --max-time 15 "$app_url/healthz" > "$DEPLOY_REPORT_DIR/health.json"
jq -e '.status == "UP"' "$DEPLOY_REPORT_DIR/health.json" > /dev/null
curl --fail --silent --show-error --retry 6 --retry-all-errors \
  --retry-delay 5 --max-time 15 "$app_url/openapi.json" > "$DEPLOY_REPORT_DIR/openapi.json"
jq -e '.paths["/api/v1/copiloto/evaluar"].post' "$DEPLOY_REPORT_DIR/openapi.json" > /dev/null

printf '##vso[task.setvariable variable=containerAppUrl;isOutput=true]%s\n' "$app_url"
printf '##vso[task.setvariable variable=containerAppRevision;isOutput=true]%s\n' "$revision"
printf 'Container App disponible: %s\nRevision: %s\nImagen: %s\n' "$app_url" "$revision" "$CONTAINER_IMAGE"
printf '# Container App desplegada\n\n- API publica sin autenticacion: %s\n- Swagger: %s/docs\n- Revision: `%s`\n- Imagen: `%s`\n' \
  "$app_url" "$app_url" "$revision" "$CONTAINER_IMAGE" > "$DEPLOY_REPORT_DIR/summary.md"
printf '##vso[task.uploadsummary]%s\n' "$DEPLOY_REPORT_DIR/summary.md"
