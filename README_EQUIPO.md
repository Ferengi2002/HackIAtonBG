# Guía Rápida para el Equipo — Copiloto de Salud Financiera Banco Guayaquil
**Reto: AI Banking — Hackathon Banco Guayaquil**

Este paquete contiene el modelo entrenado de **LightGBM**, los metadatos y el motor de inferencia listo para ser integrado en el backend (FastAPI, Flask, Node.js) o consumido por el frontend.

---

## 1. Contenido del Paquete
- `models/bg_risk_lightgbm.joblib`: Modelo LightGBM serializado (1.17 MB, tiempo de inferencia < 5ms).
- `models/model_metadata.json`: Metadatos del modelo (versión, métricas de validación ROC-AUC=0.9975, umbrales de riesgo).
- `src/inference/`:
  - `copilot_engine.py`: Orquestador principal (`copilot.process(payload)`).
  - `feature_extractor.py`: Extrae las 19 variables desde las 8 tablas de Banco Guayaquil + Equifax + OCR.
  - `financial_capacity.py`: Calcula DTI actual vs simulado, semáforo (`VERDE`/`AMARILLO`/`ROJO`), y capacidad de cuota.
  - `recommendations.py`: Genera ofertas para `MAESTRO_CREDITOS`, `MAESTRO_POLIZAS` y `MAESTRO_METAS`.
- `data/samples/sample_input.json`: Contrato de entrada con las 8 tablas oficiales.
- `data/samples/sample_output.json`: Contrato de salida con las evaluaciones, ofertas y texto para la UI.

---

## 2. ¿Cómo usarlo en el Backend en 3 líneas de código?

```python
import json
import joblib
from src.inference.copilot_engine import CopilotEngine

# 1. Cargar el modelo LightGBM
model = joblib.load("models/bg_risk_lightgbm.joblib")
copilot = CopilotEngine(model)

# 2. Recibir la petición del usuario (sample_input.json)
with open("data/samples/sample_input.json", "r", encoding="utf-8") as f:
    input_data = json.load(f)

# 3. Procesar y obtener la respuesta para la App Móvil
resultado = copilot.process(input_data)
print(json.dumps(resultado, indent=2, ensure_ascii=False))
```

---

## 3. Integración con el Agente de Azure AI Foundry (Paso 3)
El resultado en `resultado["evaluacion_riesgo_ml"]` y `resultado["recomendaciones_productos_bg"]` contiene los datos deterministas validados por riesgo crediticio. El Agente de **Azure AI Foundry (`gpt-4o-mini`)** toma este JSON como contexto para generar la respuesta conversacional final y los botones interactivos de la App.

