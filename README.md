# Copiloto de Salud Financiera y Prevención de Morosidad — Banco Guayaquil
**Hackathon Banco Guayaquil — Reto: AI Banking (Construir la próxima generación de banca digital)**

Microservicio de Inteligencia Artificial que conecta un modelo tabular de Machine Learning (**LightGBM** con ROC-AUC 0.9975), un motor de **Guardrails de seguridad**, un **RAG Regulatorio Ecuatoriano** (JPRF, Superintendencia de Bancos, COMFyF y LOPDP), y **Azure AI Foundry (`gpt-4o`)** para generar recomendaciones personalizadas y accionables.

---

## 1. Arquitectura del Pipeline de Inferencia

```
[Cliente: Petición en Lenguaje Natural + Datos de las 8 Tablas]
                               │
                               ▼
     [PASO 0: GuardrailsEngine (src/inference/guardrails.py)]
         ├── Saludo / Cortesía ──────────► Respuesta Inmediata (0 Tokens)
         ├── Fuera de Dominio / Trivia ──► Rechazo Seguro (0 Tokens)
         ├── Urgencia / Fraude ──────────► Derivación Call Center 04 3730100
         └── Consulta Financiera Válida ──► Pasa al Pipeline Bancario
                               │
                               ▼
     [SUBPASO A: Extracción NLU con Azure AI Foundry (gpt-4o)]
         └── Extrae: {intención, monto, plazo, cuota (tasa referencial 15.5%)}
                               │
                               ▼
     [SUBPASO B: Inferencia ML LightGBM (models/bg_risk_lightgbm.joblib)]
         ├── Inyecta impacto de nueva deuda en flujo y pasivos
         ├── Inferencia probabilística de caer en mora (VENCIDO)
         └── Determina Semáforo: VERDE / AMARILLO / ROJO
                               │
                               ▼
     [RAG REGULATORIO ECUADOR (src/inference/rag_regulatorio.py)]
         └── Recupera sustento legal: JPRF (DTI <= 40%), Super de Bancos (Cat. A1/Mora), COMFyF y LOPDP
                               │
                               ▼
     [SUBPASO C: Explicación XAI con Azure AI Foundry (gpt-4o)]
         ├── Redacta veredicto empático con cifras exactas
         └── Genera acciones de UI: [CREAR_CREDITO], [CREAR_POLIZA], [AUMENTAR_META]
```

---

## 2. Estructura del Repositorio

- **`app.py`**: Microservicio REST con **FastAPI**, Swagger UI (`/docs`) y health check (`/healthz`).
- **`motor_ia_foundry.py`**: Orquestador principal del pipeline de IA híbrido.
- **`models/`**:
  - `bg_risk_lightgbm.joblib`: Modelo ganador entrenado (ROC-AUC: 0.9975, Recall Mora: 0.9802).
  - `model_metadata.json`: Metadatos y umbrales de decisión.
- **`src/inference/`**:
  - `guardrails.py`: Filtros de seguridad, detección de idiomas y ahorro del 100% de tokens.
  - `rag_regulatorio.py`: RAG con la normativa de la JPRF, Super de Bancos y COMFyF.
  - `financial_capacity.py`: Fórmulas financieras de amortización francesa y cálculo de DTI.
  - `feature_extractor.py`: Extractor de las 19 features desde las 8 tablas de Banco Guayaquil.
- **`notebooks/`**:
  - `BG_Copiloto_Entrenamiento_Colab.ipynb`: Notebook de Google Colab para entrenamiento y benchmarking de 6 algoritmos de Machine Learning.
- **`data/samples/`**:
  - `sample_input.json`: Contrato de datos de entrada (8 tablas de PostgreSQL + Equifax + OCR).
  - `sample_output.json`: Contrato de respuesta oficial para el Backend y Frontend.

---

## 3. Guía de Ejecución Rápida

### Requisitos
- Python 3.10+
- Dependencias: `pip install -r requirements.txt` o `pip install fastapi uvicorn lightgbm pydantic openai python-dotenv`

### Configurar Variables de Entorno
Copia `.env.example` a `.env` y configura tus credenciales de Azure AI Foundry:
```ini
AZURE_OPENAI_ENDPOINT=https://tu-recurso.services.ai.azure.com/openai/v1
AZURE_OPENAI_API_KEY=tu_clave_de_api
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-10-21
```

### Iniciar la API REST
```bash
python app.py
```
Abre en tu navegador: **`http://localhost:8000/docs`** para probar interactivamente los 6 casos de uso en Swagger.

---

## 4. Consumo desde Java Spring Boot

Configura tu cliente HTTP en Spring Boot para invocar `POST http://localhost:8000/api/v1/copiloto/evaluar`:

```java
public record CopilotoRequestDto(
    String peticion_usuario,
    Map<String, Object> datos_cliente
) {}
```
