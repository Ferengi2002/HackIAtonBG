#!/usr/bin/env bash
set -euo pipefail

# El pipeline publica una revision; Terraform administra la infraestructura.
exec 3>&1
log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >&3
}

run_step() {
  local label="$1" limit="$2"
  shift 2
  local started=$SECONDS next_message=$((SECONDS + 30)) command_pid status
  log "INICIO: $label (limite ${limit}s)"
  timeout --signal=TERM --kill-after=15s "${limit}s" "$@" &
  command_pid=$!
  while kill -0 "$command_pid" 2>/dev/null; do
    sleep 1
    if (( SECONDS >= next_message )); then
      log "EN CURSO: $label ($((SECONDS - started))s)"
      next_message=$((SECONDS + 30))
    fi
  done
  if wait "$command_pid"; then
    log "FIN: $label ($((SECONDS - started))s)"
  else
    status=$?
    log "ERROR: $label, codigo $status ($((SECONDS - started))s)"
    return "$status"
  fi
}

: "${RESOURCE_GROUP:?Falta RESOURCE_GROUP}"
: "${CONTAINER_APP_NAME:?Falta CONTAINER_APP_NAME}"
: "${CONTAINER_NAME:?Falta CONTAINER_NAME}"
: "${ACR_NAME:?Falta ACR_NAME}"
: "${IMAGE_REPOSITORY:?Falta IMAGE_REPOSITORY}"
: "${CONTAINER_IMAGE:?Falta la imagen de BuildAndPush}"
: "${REVISION_SUFFIX:?Falta REVISION_SUFFIX}"
: "${DEPLOY_REPORT_DIR:?Falta DEPLOY_REPORT_DIR}"
DEPLOYMENT_TIMEOUT_SECONDS="${DEPLOYMENT_TIMEOUT_SECONDS:-1200}"
if [[ ! "$DEPLOYMENT_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  log 'DEPLOYMENT_TIMEOUT_SECONDS debe ser un entero positivo.'
  exit 1
fi
if [[ ! "$REVISION_SUFFIX" =~ ^[a-z][a-z0-9-]{0,63}$ || "$REVISION_SUFFIX" == *--* || "$REVISION_SUFFIX" == *- ]]; then
  log 'REVISION_SUFFIX debe ser un sufijo valido de Container Apps.'
  exit 1
fi

mkdir -p "$DEPLOY_REPORT_DIR"
export AZURE_EXTENSION_USE_DYNAMIC_INSTALL=no
if ! timeout --kill-after=15s 60s az extension show --name containerapp --output none 2>/dev/null; then
  run_step 'Instalar extension containerapp' 180 az extension add --name containerapp --yes --only-show-errors
fi

# Proyectar solo metadatos; las evidencias nunca incluyen valores de env o secretos.
app_json=$(run_step 'Consultar Container App existente de Terraform' 90 \
  az containerapp show --name "$CONTAINER_APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --query '{id:id,name:name,identity:identity,environmentId:properties.environmentId,state:properties.provisioningState,latestRevision:properties.latestRevisionName,latestReadyRevision:properties.latestReadyRevisionName,mode:properties.configuration.activeRevisionsMode,registries:properties.configuration.registries,ingress:properties.configuration.ingress,containers:properties.template.containers[].{name:name,image:image}}' \
  --output json)
printf '%s\n' "$app_json" > "$DEPLOY_REPORT_DIR/before.json"
if ! jq -e --arg container "$CONTAINER_NAME" \
  '([.containers[] | select(.name == $container)] | length) == 1' <<< "$app_json" > /dev/null; then
  log "El contenedor $CONTAINER_NAME debe existir dentro de $CONTAINER_APP_NAME."
  exit 1
fi
if ! jq -e '.mode == "Single" and .ingress.external == true and .ingress.allowInsecure == false and .ingress.targetPort == 8000 and (.ingress.fqdn | type == "string" and length > 0) and any(.ingress.traffic[]?; .latestRevision == true and .weight == 100)' \
  <<< "$app_json" > /dev/null; then
  log 'Terraform debe configurar revision Single, ingress HTTPS publico en 8000 y trafico a la ultima revision.'
  exit 1
fi

registry_json=$(run_step 'Consultar ACR existente' 90 az acr show --name "$ACR_NAME" \
  --query '{id:id,loginServer:loginServer,roleAssignmentMode:roleAssignmentMode}' --output json)
registry_id=$(jq -er '.id' <<< "$registry_json")
registry_server=$(jq -er '.loginServer' <<< "$registry_json")
permission_mode=$(jq -r '.roleAssignmentMode // "LegacyRegistryPermissions"' <<< "$registry_json")
if [[ "$permission_mode" != 'LegacyRegistryPermissions' ]]; then
  log 'Se requiere ACR en modo RBAC Registry Permissions para usar AcrPull.'
  exit 1
fi
image_prefix="$registry_server/$IMAGE_REPOSITORY@"
image_digest="${CONTAINER_IMAGE#"$image_prefix"}"
if [[ "$CONTAINER_IMAGE" != "$image_prefix"* || ! "$image_digest" =~ ^sha256:[a-f0-9]{64}$ ]]; then
  log 'La imagen debe ser el digest de este repositorio producido por BuildAndPush.'
  exit 1
fi
arm_auth=$(run_step 'Comprobar autenticacion ARM de ACR' 90 \
  az acr config authentication-as-arm show --registry "$ACR_NAME" --query status --output tsv)
if [[ "${arm_auth,,}" != 'enabled' ]]; then
  log 'ACR debe permitir ARM audience tokens para descargar con identidad administrada.'
  exit 1
fi

# Reutilizar la identidad configurada en registry por Terraform.
pull_identity=$(jq -er --arg server "$registry_server" \
  '[.registries[]? | select((.server | ascii_downcase) == ($server | ascii_downcase))] | if length == 1 then .[0].identity else empty end' <<< "$app_json")
if [[ "$pull_identity" == system ]]; then
  principal_id=$(jq -er '.identity.principalId' <<< "$app_json")
else
  principal_id=$(jq -er --arg identity "$pull_identity" \
    '.identity.userAssignedIdentities | to_entries[] | select((.key | ascii_downcase) == ($identity | ascii_downcase)) | .value.principalId' <<< "$app_json")
fi
assignments=$(run_step 'Comprobar AcrPull de la identidad de Terraform' 90 \
  az role assignment list --assignee "$principal_id" --fill-principal-name false \
  --scope "$registry_id" --include-inherited --output json)
if ! jq -e 'any(.[]; ((.roleDefinitionId // "") | ascii_downcase | endswith("/7f951dda-4ed3-4680-a7ca-43fe172d538d")))' \
  <<< "$assignments" > /dev/null; then
  printf '##vso[task.logissue type=error]Falta AcrPull para la identidad %s en %s.\n' "$principal_id" "$registry_id" >&2
  exit 1
fi

revision="$CONTAINER_APP_NAME--$REVISION_SUFFIX"
if jq -e --arg revision "$revision" --arg container "$CONTAINER_NAME" --arg image "$CONTAINER_IMAGE" \
  '.latestRevision == $revision and any(.containers[]; .name == $container and .image == $image)' \
  <<< "$app_json" > /dev/null; then
  log "Continuar comprobacion de la revision ya enviada: $revision."
else
  run_step 'Publicar imagen en la Container App existente' 120 \
    az containerapp update --name "$CONTAINER_APP_NAME" --resource-group "$RESOURCE_GROUP" \
    --container-name "$CONTAINER_NAME" --image "$CONTAINER_IMAGE" \
    --revision-suffix "$REVISION_SUFFIX" --no-wait --output none
fi

ready=false
deployment_deadline=$((SECONDS + DEPLOYMENT_TIMEOUT_SECONDS))
while (( SECONDS < deployment_deadline )); do
  status_json=$(run_step 'Consultar estado de la nueva revision' 90 \
    az containerapp show --name "$CONTAINER_APP_NAME" --resource-group "$RESOURCE_GROUP" \
    --query '{state:properties.provisioningState,latestRevision:properties.latestRevisionName,latestReadyRevision:properties.latestReadyRevisionName}' \
    --output json)
  printf '%s\n' "$status_json" > "$DEPLOY_REPORT_DIR/status.json"
  state=$(jq -er '.state' <<< "$status_json")
  log "Estado: $state | Revision lista: $(jq -r '.latestReadyRevision // "ninguna"' <<< "$status_json")."
  if [[ "$state" == Failed || "$state" == Canceled ]]; then
    log 'Azure no pudo publicar la nueva revision; revisar status.json y las revisiones de la app.'
    exit 1
  fi
  if jq -e --arg revision "$revision" '.state == "Succeeded" and .latestReadyRevision == $revision' \
    <<< "$status_json" > /dev/null; then
    ready=true
    break
  fi
  sleep 15
done
if [[ "$ready" != true ]]; then
  log "Se agoto la espera de ${DEPLOYMENT_TIMEOUT_SECONDS}s; revisar la app antes de reintentar."
  exit 1
fi

run_step 'Consultar revision desplegada' 90 az containerapp revision show \
  --name "$CONTAINER_APP_NAME" --resource-group "$RESOURCE_GROUP" --revision "$revision" \
  --query '{name:name,active:properties.active,healthState:properties.healthState,provisioningState:properties.provisioningState,runningState:properties.runningState,containers:properties.template.containers[].{name:name,image:image}}' \
  --output json > "$DEPLOY_REPORT_DIR/revision.json"
if ! jq -e --arg container "$CONTAINER_NAME" --arg image "$CONTAINER_IMAGE" \
  'any(.containers[]; .name == $container and .image == $image)' \
  "$DEPLOY_REPORT_DIR/revision.json" > /dev/null; then
  log 'La revision lista no corresponde al digest de esta ejecucion.'
  exit 1
fi

app_url="https://$(jq -er '.ingress.fqdn' <<< "$app_json")"
run_step 'Comprobar /healthz publico' 200 \
  curl --fail --silent --show-error --retry 12 --retry-all-errors \
  --retry-delay 5 --retry-max-time 180 --max-time 15 "$app_url/healthz" > "$DEPLOY_REPORT_DIR/health.json"
jq -e '.status == "UP"' "$DEPLOY_REPORT_DIR/health.json" > /dev/null
run_step 'Comprobar contrato OpenAPI publico' 200 \
  curl --fail --silent --show-error --retry 6 --retry-all-errors \
  --retry-delay 5 --retry-max-time 180 --max-time 15 "$app_url/openapi.json" > "$DEPLOY_REPORT_DIR/openapi.json"
jq -e '.paths["/api/v1/copiloto/evaluar"].post' "$DEPLOY_REPORT_DIR/openapi.json" > /dev/null

printf '##vso[task.setvariable variable=containerAppUrl;isOutput=true]%s\n' "$app_url"
printf '##vso[task.setvariable variable=containerAppRevision;isOutput=true]%s\n' "$revision"
printf '# API desplegada en Container App existente\n\n- API: %s\n- Swagger: %s/docs\n- Contenedor: `%s`\n- Revision: `%s`\n- Imagen: `%s`\n' \
  "$app_url" "$app_url" "$CONTAINER_NAME" "$revision" "$CONTAINER_IMAGE" > "$DEPLOY_REPORT_DIR/summary.md"
printf '##vso[task.uploadsummary]%s\n' "$DEPLOY_REPORT_DIR/summary.md"
log "Despliegue comprobado: $app_url | Revision: $revision | Imagen: $CONTAINER_IMAGE"
