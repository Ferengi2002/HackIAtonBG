"""
API REST Oficial del Copiloto de Salud Financiera - Banco Guayaquil
Framework: FastAPI (Documentacion interactiva Swagger / OpenAPI lista para probar en navegador)

Endpoints:
- POST /api/v1/copiloto/evaluar : Ejecuta Guardrails -> NLU -> LightGBM -> RAG -> XAI Foundry.
- GET  /healthz                 : Liveness probe para Spring Boot / Kubernetes / ACA.
- GET  /docs                    : Documentacion interactiva Swagger UI con ejemplos listos para 'Try it out'.
"""

from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from typing import Any, Dict
from fastapi import Body, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from motor_ia_foundry import MotorIAFoundry

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("bg_api_copiloto")

# Cargar datos base del cliente ejemplo (8 tablas de Banco Guayaquil + Equifax + OCR)
WORKSPACE_ROOT = Path(__file__).resolve().parent
SAMPLE_INPUT_PATH = WORKSPACE_ROOT / "data" / "samples" / "sample_input.json"
try:
    with open(SAMPLE_INPUT_PATH, "r", encoding="utf-8") as f:
        SAMPLE_CLIENTE_DATA = json.load(f)
except Exception:
    SAMPLE_CLIENTE_DATA = {"COD_CLTE": 10045892}


# ----------------------------------------------------------------------------
# DTO de Entrada para FastAPI
# ----------------------------------------------------------------------------
class EvaluacionCopilotoRequest(BaseModel):
    peticion_usuario: str = Field(
        ...,
        description="Pregunta en lenguaje natural del cliente (ej. 'Quiero comprarme una moto de $2,500 a 24 meses').",
    )
    datos_cliente: Dict[str, Any] = Field(
        ...,
        description="JSON con la informacion financiera del cliente de las 8 tablas de Banco Guayaquil + Equifax + OCR.",
    )


# ----------------------------------------------------------------------------
# Ciclo de Vida: El modelo LightGBM se carga UNA SOLA VEZ en memoria al arrancar
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Inicializando MotorIAFoundry y cargando modelo LightGBM en memoria...")
    app.state.motor = MotorIAFoundry()
    logger.info("Motor IA listo para atender peticiones en Swagger y Spring Boot.")
    yield
    logger.info("Cerrando recursos de la API del Copiloto...")


app = FastAPI(
    title="Banco Guayaquil - API Copiloto de Salud Financiera",
    description=(
        "Microservicio de Inteligencia Artificial para Banco Guayaquil. "
        "Integra Guardrails de seguridad, modelo LightGBM en memoria, RAG Regulatorio Ecuatoriano "
        "(JPRF, Super de Bancos, COMFyF, LOPDP) y generacion explicativa con Azure AI Foundry (gpt-4o)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz", status_code=status.HTTP_200_OK, tags=["Salud"])
def health_check():
    """Liveness probe para verificar que el servicio este arriba con el modelo cargado."""
    return {
        "status": "UP",
        "service": "banco-guayaquil-copiloto-ai",
        "engine": "LightGBM + Azure AI Foundry (gpt-4o)",
    }


@app.post(
    "/api/v1/copiloto/evaluar",
    status_code=status.HTTP_200_OK,
    tags=["Copiloto Financiero"],
    summary="Evalua la peticion del cliente (Guardrails + LightGBM + RAG + Azure Foundry)",
)
def evaluar_solicitud(
    payload: EvaluacionCopilotoRequest = Body(
        openapi_examples={
            "compra_moto": {
                "summary": "1. Compra Moto ($2,500 a 24 meses - Evaluacion Viable)",
                "description": "Calcula cuota de $121.81/mes al 15.5%, evalua liquidez del cliente y genera botones de accion.",
                "value": {
                    "peticion_usuario": "Quiero comprarme una moto de $2,500 a 24 meses para ir a mi trabajo, ¿me alcanza con mi sueldo actual?",
                    "datos_cliente": SAMPLE_CLIENTE_DATA,
                },
            },
            "sobreendeudamiento": {
                "summary": "2. Solicitud Excesiva ($24,000 a 8 meses - Semaforo Rojo + RAG JPRF)",
                "description": "Alerta de sobreendeudamiento: cuota de $3,176.99/mes excede capacidad. Cita Resolucion JPRF-F-2023-080.",
                "value": {
                    "peticion_usuario": "Hola quiero un credito de 24000 dolares a 8 meses plazo, ¿puedo?",
                    "datos_cliente": SAMPLE_CLIENTE_DATA,
                },
            },
            "inversion_poliza": {
                "summary": "3. Inversion en Poliza (Semaforo Verde + RAG COSEDE)",
                "description": "Recomienda rentabilizar ahorros al 6.50% anual respaldada por el seguro de depositos COSEDE hasta $32,000.",
                "value": {
                    "peticion_usuario": "¿Donde me conviene invertir los $600 que tengo ahorrados en mi cuenta para ganar mas intereses?",
                    "datos_cliente": SAMPLE_CLIENTE_DATA,
                },
            },
            "saludo_guardrail": {
                "summary": "4. Saludo Simple (Guardrail - Costo 0 Tokens)",
                "description": "Responde de inmediato en < 1ms sin gastar tokens ni ejecutar inferencia.",
                "value": {
                    "peticion_usuario": "Hola como estas",
                    "datos_cliente": {"COD_CLTE": 10045892},
                },
            },
            "fuera_de_dominio": {
                "summary": "5. Pregunta Fuera de Dominio (Guardrail - Costo 0 Tokens)",
                "description": "Rechaza cortesmente preguntas absurdas o ajenas al ambito financiero.",
                "value": {
                    "peticion_usuario": "¿Cual es la capital de Francia?",
                    "datos_cliente": {"COD_CLTE": 10045892},
                },
            },
            "clonacion_tarjeta": {
                "summary": "6. Urgencia / Fraude (Derivacion a Asesor Humano)",
                "description": "Deriva inmediatamente a canales oficiales de Banco Guayaquil (Call Center 04 3730100 y WhatsApp).",
                "value": {
                    "peticion_usuario": "Ayuda urgente, creo que me clonaron la tarjeta de credito en un cajero",
                    "datos_cliente": {"COD_CLTE": 10045892},
                },
            },
        }
    )
):
    try:
        motor: MotorIAFoundry = app.state.motor
        resultado = motor.evaluar_peticion_cliente(
            peticion_usuario=payload.peticion_usuario,
            datos_cliente=payload.datos_cliente,
        )
        return resultado
    except Exception as exc:
        logger.error(f"Error procesando solicitud: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en el motor de IA de Banco Guayaquil: {str(exc)}",
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
