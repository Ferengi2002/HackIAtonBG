# Copiloto de Salud Financiera y Prevención de Morosidad — Banco Guayaquil
**Hackathon Banco Guayaquil 2026 — Reto: AI Banking (Construir la próxima generación de banca digital)**

Microservicio de Inteligencia Artificial de grado de producción que integra un modelo predictivo de Machine Learning (**LightGBM** con ROC-AUC 0.9975), **Azure AI Document Intelligence (`prebuilt-layout`)**, un motor de **Guardrails de Ciberseguridad y Ahorro de Tokens**, **RAG Regulatorio Ecuatoriano** (JPRF, Superintendencia de Bancos, COMFyF y LOPDP), un **Motor de Asesoría Patrimonial por Tiers** y generación explicativa empática con **Azure AI Foundry (`gpt-4o`)**.

---

## 1. Arquitectura General del Pipeline de IA

```
[Cliente: Petición en Lenguaje Natural + Datos de las 8 Tablas + Documento Opcional]
                                      │
                                      ▼
     [PASO 0: Guardrails de Ciberseguridad y Filtro Rápido (src/inference/guardrails.py)]
         ├── Intento Exfiltración (.env, código, PII ajena) ──► Bloqueo Inmediato (< 3ms | 0 Tokens)
         ├── Saludo / Cortesía Simple ────────────────────────► Respuesta Inmediata (< 1ms | 0 Tokens)
         ├── Fuera de Dominio / Trivia ───────────────────────► Rechazo Cortés (< 1ms | 0 Tokens)
         ├── Urgencia / Fraude / Tarjeta Clonada ─────────────► Derivación Humana Call Center 04 3730100
         └── Consulta Financiera Válida ──────────────────────► Pasa al Pipeline de IA
                                      │
                                      ▼
     [AZURE AI DOCUMENT INTELLIGENCE: prebuilt-layout (src/inference/document_intelligence_ia.py)]
         ├── Si adjunta PDF / Imagen (Estado de Cuenta externo, rol de pagos, factura):
         ├── Extrae tablas estructuradas y texto con Azure Cognitive Services
         └── gpt-4o clasifica y suma: GASTOS_OCULTOS_OCR e INGRESOS_EXTRA_OCR
                                      │
                                      ▼
     [SUBPASO A: Extracción NLU Financiero con Azure AI Foundry (gpt-4o)]
         ├── Clasifica intención: ENDEUDAMIENTO | INVERSION | AHORRO | REESTRUCTURACION_DEUDA
         ├── Extrae monto solicitado, plazo en meses y cuota fija (Amortización Francesa al 15.5%)
         └── Detecta banderas de estrés financiero o sobreendeudamiento
                                      │
                                      ▼
     [SUBPASO B: Inferencia ML LightGBM + Reglas de Negocio (models/bg_risk_lightgbm.joblib)]
         ├── Inyecta impacto de la nueva deuda + OCR en el perfil del cliente (19 features)
         ├── Inferencia probabilística de caer en Mora (VENCIDO) con modelo LightGBM
         ├── Calcula capacidad máxima de cuota segura y DTI proyectado
         ├── Determina Semáforo: VERDE (Óptimo) | AMARILLO (Ajustado) | ROJO (Riesgo / No recomendado)
         ├── Asesoría Patrimonial por Tiers: Póliza Plazo Fijo 7.25% (Tier Alto) | Híbrido | Metas
         └── Protocolo Sobreendeudamiento Crítico: Bloqueo de deuda y plan de alivio financiero
                                      │
                                      ▼
     [RAG REGULATORIO ECUADOR (src/inference/rag_regulatorio.py)]
         └── Recupera sustento jurídico vinculante:
             ├── JPRF (Resolución JPRF-F-2023-080: DTI máximo <= 40%)
             ├── Super de Bancos (Codificación SB: Calificación A1, Mora Art. 12 y Reestructuración Cap. VI)
             ├── COSEDE (Seguro de Depósitos en Pólizas garantizado por ley hasta $32,000 USD)
             ├── COMFyF (Art. 209: Principio de Crédito Responsable)
             └── LOPDP (Artículos 19 y 20: Explicabilidad y no discriminación algorítmica)
                                      │
                                      ▼
     [SUBPASO C: Explicación XAI con Azure AI Foundry (gpt-4o)]
         ├── Redacta veredicto transparente y empático con cifras exactas (< 85 palabras, cero emojis)
         ├── Si detectó gastos ocultos en OCR: menciona explícitamente la entidad y cuota externa
         └── Genera botones de acción dinámicos para la UI móvil/web
```

---

## 2. Componentes y Capacidades Principales

### A. Detección de Sobreendeudamiento Crítico y Negociación con Asesor
Si el usuario expresa desesperación (*"ando sobre endeudado y no puedo más"*, *"estoy al límite"*), tiene flujo deficitario ($< \$0$), DTI $\ge 50\%$, o mora activa (`CAPITAL_VENCIDO > 0`):
- **Dictamen:** Prohíbe estrictamente nuevo crédito (anti-bola de nieve).
- **Derivación de urgencia:** Guía a contactar de inmediato a un asesor de Banco Guayaquil vía telefónica (**`04 3730100` opción 3**) o **WhatsApp oficial (`+593 98 373 0100`)**.
- **Soluciones de Alivio:** Plantea reestructuración de pasivos, consolidación en cuota única y ampliación de plazos para reducir cuotas hasta un 40%.

### B. Asesoría Patrimonial Inteligente por Tiers de Liquidez
Para consultas como *"¿Qué puedo hacer con mi dinero?"* o clientes con liquidez excedente:
- **Tier Alto ($\ge \$3,000$ USD en cuenta):** Recomienda Póliza a Plazo Fijo en `MAESTRO_POLIZAS` (180 días al **7.25% anual**) con respaldo estatal de **COSEDE hasta \$32,000 USD**, reservando un colchón operativo en cuenta de ahorros.
- **Tier Medio (\$500 a \$3,000 USD):** Estrategia híbrida: Póliza a corto plazo (90 días al 6.50%) + Ahorro Programado flexible en `MAESTRO_METAS`.
- **Tier Básico ($< \$500$ USD):** Ahorro Flexible en `MAESTRO_METAS` (\$30-\$50/mes) para construir un Fondo de Emergencia de 3 meses.
- **Desendeudamiento Rentable:** Si mantiene saldos diferidos en tarjeta de crédito (~16% anual), le aconseja amortizarlos primero para maximizar su ahorro neto.

### C. Guardrails de Ciberseguridad Defensiva (OWASP LLM02)
- Intercepta intentos de extracción de credenciales, variables del archivo `.env`, llaves de Azure, código fuente del sistema o datos confidenciales de otros clientes (Anti-IDOR / LOPDP) en **$< 3$ ms con 0 tokens de LLM**, protegiendo los secretos de la infraestructura.

### D. Módulo Azure AI Document Intelligence
- Integra el modelo `prebuilt-layout` de Azure AI Document Intelligence para procesar estados de cuenta de otros bancos (ej. Banco Pichincha), roles de pago o facturas subidas en PDF/imagen, extrayendo deudas no registradas para recalcular el riesgo real.

---

## 3. Estructura del Repositorio

```
HackatonBG/
├── app.py                             # Microservicio REST oficial (FastAPI + Swagger UI interactivo)
├── motor_ia_foundry.py                # Orquestador del pipeline hibrido (ML LightGBM + Azure gpt-4o)
├── requirements.txt                   # Dependencias oficiales validadas
├── .env.example                       # Plantilla de variables de entorno
├── models/
│   ├── bg_risk_lightgbm.joblib        # Modelo LightGBM entrenado (ROC-AUC 0.9975)
│   └── model_metadata.json            # Metadatos del modelo, features y metricas
├── src/
│   └── inference/
│       ├── guardrails.py              # Guardrails de seguridad, filtro de saludos y anti-exfiltracion
│       ├── document_intelligence_ia.py# Extraccion con Azure Document Intelligence (prebuilt-layout)
│       ├── rag_regulatorio.py         # RAG Regulatorio Ecuatoriano (JPRF, Super de Bancos, LOPDP)
│       ├── recommendations.py         # Motor de asesoria patrimonial por tiers y ofertas
│       ├── financial_capacity.py      # Formulas financieras de amortizacion francesa y DTI
│       └── feature_extractor.py       # Transformador de las 8 tablas a las 19 features del modelo
├── data/
│   └── samples/
│       ├── sample_input.json          # Contrato de datos de entrada (8 tablas PostgreSQL + Equifax + OCR)
│       └── sample_output.json         # Contrato oficial de salida JSON para Backend y Frontend
├── scripts/
│   └── create_demo_pdf.py             # Generador del PDF de prueba de estado de cuenta bancario
├── probar_document_intelligence.py    # Test E2E: Extraccion de PDF externo y cambio de semaforo
├── probar_asesoria_patrimonial.py      # Test E2E: Evaluacion de los 4 tiers de liquidez patrimonial
├── probar_sobreendeudamiento_asesor.py# Test E2E: Casos de crisis financiera y protocolo de renegociacion
└── probar_seguridad_guardrails.py     # Test E2E: 6 vectores de ataque de ciberseguridad bloqueados
```

---

## 4. Guía de Instalación y Ejecución Local

### 1. Requisitos Previos
- Python 3.10 o superior instalado.
- Conexión a Internet para consumir Azure AI Foundry y Document Intelligence.

### 2. Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### Pruebas unitarias para CI

Las pruebas de `tests/` validan los guardrails sin credenciales de Azure ni cargar
el modelo LightGBM. Los scripts de prueba manuales quedan fuera de la recolección
automática configurada en `pytest.ini`.

```bash
python -m pip install pytest pytest-cov pydantic
python -m pytest tests --junitxml=test-results.xml --cov=. --cov-report=xml
```

Azure Pipelines genera y publica `test-results.xml` en cada ejecución de pruebas,
incluso si hay pruebas fallidas. Los reportes generados no se versionan en Git.

### 3. Configurar Variables de Entorno (`.env`)
El proyecto está configurado para operar con el recurso unificado de Azure Cognitive Services / Foundry:
```ini
AZURE_OPENAI_ENDPOINT=https://jrengifo-6526-resource.services.ai.azure.com/openai/v1
AZURE_OPENAI_API_KEY=tu_clave_de_azure_ai_foundry
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-10-21

# Document Intelligence reutiliza automaticamente la misma llave si no se especifica otra:
AZURE_DOC_INTELLIGENCE_ENDPOINT=https://jrengifo-6526-resource.cognitiveservices.azure.com/
AZURE_DOC_INTELLIGENCE_KEY=tu_clave_de_azure_ai_foundry
```

### 4. Iniciar el Microservicio REST
```bash
python app.py
```
El servicio iniciará en `http://localhost:8000`.

### 5. Documentación Interactiva Swagger UI
Abre en tu navegador:
```
http://localhost:8000/docs
```
La interfaz de Swagger incluye **10 ejemplos preconfigurados listos para probar con un clic (*Try it out*)**:
1. Evaluación con Estado de Cuenta Externo (Azure Document Intelligence).
2. Compra Viable de Moto (\$2,500 a 24 meses).
3. Solicitud Excesiva con Alerta JPRF (\$24,000 a 8 meses).
4. Inversión en Póliza con Respaldo COSEDE.
5. Saludo Simple (Costo 0 Tokens).
6. Pregunta Fuera de Dominio (Costo 0 Tokens).
7. Reporte de Fraude o Tarjeta Clonada (Derivación Humana).
8. Asesoría Patrimonial Tier Alto (*"Tengo mucho dinero en el banco"*).
9. Crisis de Sobreendeudamiento (*"Ando sobre endeudado y no puedo más"*).
10. Ataque de Exfiltración de Credenciales / `.env` (Guardrail Seguridad).

---

## 5. Endpoints de la API REST

### `POST /api/v1/copiloto/evaluar`
Recibe la petición del usuario en lenguaje natural junto con el JSON del cliente (8 tablas) y una ruta opcional de documento:
```bash
curl -X POST "http://localhost:8000/api/v1/copiloto/evaluar" \
  -H "Content-Type: application/json" \
  -d '{
    "peticion_usuario": "Quiero comprarme una moto de $2,500 a 24 meses, ¿me alcanza?",
    "datos_cliente": { ... },
    "ruta_documento_opcional": null
  }'
```

### `POST /api/v1/copiloto/evaluar-con-documento`
Recibe directamente un archivo PDF o imagen mediante `multipart/form-data`:
```bash
curl -X POST "http://localhost:8000/api/v1/copiloto/evaluar-con-documento" \
  -F "archivo=@estado_cuenta_externo_demo.pdf" \
  -F "peticion_usuario=Quiero comprar una moto a 24 meses, ¿me alcanza?" \
  -F "datos_cliente_json={ ... }"
```

### `GET /healthz`
Liveness probe para balanceadores de carga, Kubernetes y Spring Boot:
```json
{
  "status": "UP",
  "service": "banco-guayaquil-copiloto-ai",
  "engine": "LightGBM + Azure Document Intelligence + Azure AI Foundry (gpt-4o)"
}
```

---

## 6. Consumo desde Backend Java Spring Boot

El equipo de Backend en Java puede consumir el servicio mediante `RestClient` de Spring Boot 3:

### DTOs en Java (Records)
```java
public record CopilotoRequestDto(
    String peticion_usuario,
    Map<String, Object> datos_cliente,
    String ruta_documento_opcional
) {}

public record CopilotoResponseDto(
    String transaccion_id,
    Long COD_CLTE,
    String idioma_detectado,
    Map<String, Object> subpaso_a_nlu,
    EvaluacionRiesgoDto evaluacion_riesgo_ml,
    DictamenDto dictamen_endeudamiento,
    Map<String, Object> recomendaciones_productos_bg,
    Map<String, Object> fundamentacion_legal_ecuador,
    AgenteConversacionalDto agente_conversacional_foundry
) {}

public record EvaluacionRiesgoDto(
    int score_salud_interna,
    double probabilidad_caer_en_vencido_mora,
    String semaforo_salud_financiera,
    double flujo_neto_mensual_disponible,
    double capacidad_maxima_nueva_cuota,
    double endeudamiento_actual_pct
) {}

public record DictamenDto(
    boolean se_puede_endeudar,
    String estado_resolucion,
    double monto_maximo_recomendado,
    double cuota_mensual_segura,
    List<String> factores_clave_modelo
) {}

public record AgenteConversacionalDto(
    String titulo_pantalla,
    String explicacion_ia,
    String origen_generacion,
    List<AccionUiDto> acciones_ui
) {}

public record AccionUiDto(
    String tipo,
    String label,
    String payload_tabla
) {}
```

### Cliente HTTP en Spring Boot
```java
@Service
public class CopilotoAiClientService {

    private final RestClient restClient;

    public CopilotoAiClientService(@Value("${copiloto.ai.base-url:http://localhost:8000}") String baseUrl) {
        this.restClient = RestClient.builder()
            .baseUrl(baseUrl)
            .defaultHeader("Content-Type", "application/json")
            .build();
    }

    public CopilotoResponseDto evaluarCliente(String peticion, Map<String, Object> datosCliente) {
        CopilotoRequestDto request = new CopilotoRequestDto(peticion, datosCliente, null);
        return restClient.post()
            .uri("/api/v1/copiloto/evaluar")
            .body(request)
            .retrieve()
            .body(CopilotoResponseDto.class);
    }
}
```

---

## 7. Ejecución de Suites de Pruebas Automatizadas

Se han desarrollado 4 suites de pruebas independientes para validar el comportamiento del sistema en consola:

```bash
# 1. Probar Extraccion OCR con Azure Document Intelligence + Cambio de Semáforo
python probar_document_intelligence.py

# 2. Probar Asesoría Patrimonial por Tiers (Tier Alto, Medio y Básico)
python probar_asesoria_patrimonial.py

# 3. Probar Crisis de Sobreendeudamiento y Derivación a Asesor
python probar_sobreendeudamiento_asesor.py

# 4. Probar Guardrails de Ciberseguridad Defensiva (Anti-Exfiltración de .env y Código)
python probar_seguridad_guardrails.py
```

---

## 8. Cumplimiento Normativo Ecuatoriano

El motor está alineado formalmente con los marcos regulatorios vigentes en la República del Ecuador:
- **Junta de Política y Regulación Financiera (JPRF):** Resolución JPRF-F-2023-080 (Norma Prudencial de Gestión Crediticia y límite de DTI $\le 40\%$).
- **Superintendencia de Bancos del Ecuador (SB):** Codificación de Resoluciones (Libro I, Título IX sobre calificación de cartera A1, cartera vencida Art. 12 y Capítulo VI sobre Mecanismos de Reestructuración y Alivio Financiero).
- **Corporación del Seguro de Depósitos (COSEDE):** Ley de Creación de la Red de Seguridad Financiera (cobertura de depósitos e inversiones en Pólizas hasta \$32,000 USD).
- **Código Orgánico Monetario y Financiero (COMFyF):** Artículos 209 y 211 (Principio de Crédito Responsable y techos BCE).
- **Ley Orgánica de Protección de Datos Personales (LOPDP):** Artículos 19 y 20 (Derecho a la explicabilidad algorítmica y confidencialidad).
