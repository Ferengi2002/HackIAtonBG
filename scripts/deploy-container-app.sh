#!/usr/bin/env bash
set -euo pipefail

# Separar progreso de stdout para no mezclar mensajes con JSON o valores capturados.
exec 3>&1
log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >&3
}

# Mantener visible una operacion larga sin imprimir comandos ni credenciales.
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
    if [[ "$status" == 124 || "$status" == 137 ]]; then
      log "Se alcanzo el limite de espera de ${limit}s."
    fi
    return "$status"
  fi
}

: "${RESOURCE_GROUP:?Falta RESOURCE_GROUP}"
: "${ACR_NAME:?Falta ACR_NAME}"
: "${IMAGE_REPOSITORY:?Falta IMAGE_REPOSITORY}"
: "${CONTAINER_IMAGE:?Falta la imagen de BuildAndPush}"
: "${REVISION_SUFFIX:?Falta REVISION_SUFFIX}"
: "${DEPLOYMENT_NAME:?Falta DEPLOYMENT_NAME}"
: "${TEMPLATE_FILE:?Falta TEMPLATE_FILE}"
: "${PARAMETERS_FILE:?Falta PARAMETERS_FILE}"
: "${DEPLOY_REPORT_DIR:?Falta DEPLOY_REPORT_DIR}"
DEPLOYMENT_TIMEOUT_SECONDS="${DEPLOYMENT_TIMEOUT_SECONDS:-1800}"
if [[ ! "$DEPLOYMENT_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  log 'DEPLOYMENT_TIMEOUT_SECONDS debe ser un entero positivo.'
  exit 1
fi

mkdir -p "$DEPLOY_REPORT_DIR"
log "Inicio del despliegue $DEPLOYMENT_NAME en $RESOURCE_GROUP."
export AZURE_EXTENSION_USE_DYNAMIC_INSTALL=no
log 'Comprobar extension containerapp instalada.'
if extension_version=$(timeout --kill-after=15s 60s az extension show \
  --name containerapp --query version --output tsv 2>/dev/null); then
  log "Reutilizar containerapp $extension_version; no se fuerza una actualizacion."
else
  run_step 'Instalar extension containerapp' 180 \
    az extension add --name containerapp --yes --only-show-errors
fi

# Solo se consume el registro existente; el grupo esta administrado por Terraform.
run_step 'Comprobar grupo de recursos' 90 az group show --name "$RESOURCE_GROUP" --output none
registry_json=$(run_step 'Consultar ACR' 90 az acr show --name "$ACR_NAME" --output json)
registry_group=$(jq -r '.resourceGroup' <<< "$registry_json")
registry_server=$(jq -r '.loginServer' <<< "$registry_json")
permission_mode=$(jq -r '.roleAssignmentMode // "LegacyRegistryPermissions"' <<< "$registry_json")
if [[ "$permission_mode" != 'LegacyRegistryPermissions' ]]; then
  log 'Este despliegue requiere ACR en modo RBAC Registry Permissions para usar AcrPull.'
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

min_replicas=$(jq -r '.parameters.minReplicas.value' "$PARAMETERS_FILE")
max_replicas=$(jq -r '.parameters.maxReplicas.value' "$PARAMETERS_FILE")
if (( min_replicas < 1 || max_replicas < min_replicas )); then
  log 'Configurar 1 <= minReplicas <= maxReplicas.'
  exit 1
fi

parameters=(
  "@$PARAMETERS_FILE"
  "registryName=$ACR_NAME"
  "registryResourceGroupName=$registry_group"
  "containerImage=$CONTAINER_IMAGE"
  "revisionSuffix=$REVISION_SUFFIX"
)

run_step 'ARM Validate' 300 az deployment group validate \
  --resource-group "$RESOURCE_GROUP" \
  --template-file "$TEMPLATE_FILE" \
  --parameters "${parameters[@]}" \
  --output none

run_step 'ARM What-If' 300 az deployment group what-if \
  --resource-group "$RESOURCE_GROUP" \
  --template-file "$TEMPLATE_FILE" \
  --parameters "${parameters[@]}" \
  --no-pretty-print \
  --result-format FullResourcePayloads \
  --output json > "$DEPLOY_REPORT_DIR/what-if.json"

save_operations() {
  if timeout --kill-after=10s 30s az deployment operation group list \
    --name "$DEPLOYMENT_NAME" --resource-group "$RESOURCE_GROUP" \
    --output json > "$DEPLOY_REPORT_DIR/operations.tmp.json"; then
    mv "$DEPLOY_REPORT_DIR/operations.tmp.json" "$DEPLOY_REPORT_DIR/operations.json"
    jq -r '.[] | select(.properties.targetResource != null) |
      "  Recurso: \(.properties.targetResource.resourceName) | Estado: \(.properties.provisioningState)"' \
      "$DEPLOY_REPORT_DIR/operations.json" >&3
    jq -c '.[] | select(.properties.provisioningState == "Failed") |
      {resource: .properties.targetResource.resourceName, error: .properties.statusMessage}' \
      "$DEPLOY_REPORT_DIR/operations.json" >&2
  else
    log 'No se pudo obtener el detalle de operaciones ARM en esta consulta.'
  fi
}

# Enviar sin una espera opaca del SDK; consultar el estado y los recursos explicitamente.
run_step 'Enviar despliegue ARM' 120 az deployment group create \
  --name "$DEPLOYMENT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --template-file "$TEMPLATE_FILE" \
  --parameters "${parameters[@]}" \
  --mode Incremental \
  --no-wait --output none

deployment_started=$SECONDS
deployment_deadline=$((SECONDS + DEPLOYMENT_TIMEOUT_SECONDS))
next_operations=$SECONDS
deployment_state='Running'
while (( SECONDS < deployment_deadline )); do
  log "Consultar despliegue ARM ($((SECONDS - deployment_started))s transcurridos)."
  if timeout --kill-after=10s 60s az deployment group show \
    --name "$DEPLOYMENT_NAME" --resource-group "$RESOURCE_GROUP" \
    --query '{state:properties.provisioningState,error:properties.error,outputs:properties.outputs}' \
    --output json > "$DEPLOY_REPORT_DIR/deployment-state.tmp.json"; then
    mv "$DEPLOY_REPORT_DIR/deployment-state.tmp.json" "$DEPLOY_REPORT_DIR/deployment-state.json"
  else
    log 'No se pudo consultar el estado ARM dentro de 60s; Azure puede seguir ejecutando el despliegue.'
    save_operations
    exit 1
  fi
  deployment_state=$(jq -er '.state' "$DEPLOY_REPORT_DIR/deployment-state.json")
  log "Estado ARM: $deployment_state ($((SECONDS - deployment_started))s / ${DEPLOYMENT_TIMEOUT_SECONDS}s)."
  case "$deployment_state" in
    Succeeded)
      jq '.outputs' "$DEPLOY_REPORT_DIR/deployment-state.json" > "$DEPLOY_REPORT_DIR/outputs.json"
      break
      ;;
    Failed|Canceled)
      save_operations
      jq '.error' "$DEPLOY_REPORT_DIR/deployment-state.json" >&2
      log "El despliegue ARM termino en $deployment_state."
      exit 1
      ;;
  esac
  if (( SECONDS >= next_operations )); then
    save_operations
    next_operations=$((SECONDS + 60))
  fi
  sleep 20
done
if [[ "$deployment_state" != 'Succeeded' ]]; then
  save_operations
  log "Se agoto la espera de ${DEPLOYMENT_TIMEOUT_SECONDS}s para $DEPLOYMENT_NAME."
  log 'La operacion de Azure puede continuar; revisar sus operaciones antes de reintentar.'
  exit 1
fi
save_operations

app_name=$(jq -er '.appName.value' "$DEPLOY_REPORT_DIR/outputs.json")
app_url=$(jq -er '.appUrl.value' "$DEPLOY_REPORT_DIR/outputs.json")
revision=$(jq -er '.revisionName.value' "$DEPLOY_REPORT_DIR/outputs.json")

# En modo Single, Azure conserva el trafico en la revision sana hasta que la nueva este lista.
log "Esperar revision Ready: $revision (limite 600s)."
ready=false
ready_deadline=$((SECONDS + 600))
while (( SECONDS < ready_deadline )); do
  ready_revision=$(timeout --kill-after=10s 60s az containerapp show \
    --name "$app_name" --resource-group "$RESOURCE_GROUP" \
    --query properties.latestReadyRevisionName --output tsv)
  if [[ "$ready_revision" == "$revision" ]]; then
    ready=true
    break
  fi
  log "Revision lista actual: ${ready_revision:-ninguna}; esperando $revision."
  sleep 15
done

run_step 'Consultar revision desplegada' 90 az containerapp revision show \
  --name "$app_name" --resource-group "$RESOURCE_GROUP" --revision "$revision" \
  --output json > "$DEPLOY_REPORT_DIR/revision.json"
if [[ "$ready" != true ]]; then
  log 'La nueva revision no alcanzo el estado Ready dentro de 600s.'
  exit 1
fi
deployed_image=$(jq -er '.properties.template.containers[0].image' "$DEPLOY_REPORT_DIR/revision.json")
if [[ "$deployed_image" != "$CONTAINER_IMAGE" ]]; then
  log 'La revision activa no corresponde a la imagen de esta ejecucion.'
  exit 1
fi

# Estas comprobaciones se ejecutan en el agente del pipeline, sobre el servicio desplegado.
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
log "Container App disponible: $app_url | Revision: $revision | Imagen: $CONTAINER_IMAGE"
printf '# Container App desplegada\n\n- API publica sin autenticacion: %s\n- Swagger: %s/docs\n- Revision: `%s`\n- Imagen: `%s`\n' \
  "$app_url" "$app_url" "$revision" "$CONTAINER_IMAGE" > "$DEPLOY_REPORT_DIR/summary.md"
printf '##vso[task.uploadsummary]%s\n' "$DEPLOY_REPORT_DIR/summary.md"
