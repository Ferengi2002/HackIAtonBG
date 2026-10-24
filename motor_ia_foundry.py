"""
PASO 2: Motor Hibrido de IA — Modelo ML (LightGBM) + Azure AI Foundry (gpt-4o-mini)
Proyecto: Copiloto de Salud Financiera y Prevencion de Morosidad - Banco Guayaquil
Reto: AI Banking - Construir la proxima generacion de banca digital

Arquitectura en 3 Subpasos:
- Subpaso A (NLU con Azure AI Foundry gpt-4o-mini):
  Extrae de 'peticion_usuario' en lenguaje natural:
  {intencion, monto_solicitado, plazo_meses, cuota_mensual_nueva}
  calculando la cuota con tasa anual referencial del 15.5% si el usuario dio solo el monto.
- Subpaso B (Inferencia con Modelo ML LightGBM entrenado + Reglas de Capacidad):
  Inyecta el impacto de 'cuota_mensual_nueva' en los 'datos_cliente', ejecuta el modelo
  serializado (.joblib), calcula probabilidad de mora, capacidad maxima, flujo neto
  y determina el semaforo ('VERDE', 'AMARILLO', 'ROJO').
- Subpaso C (Generacion XAI Empatica con Azure AI Foundry gpt-4o-mini):
  Envia exclusivamente el resumen matematico compacto a gpt-4o-mini para redactar
  la explicacion personalizada y empaquetar el JSON oficial de salida para Backend.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from src.inference.guardrails import GuardrailsEngine
    from src.inference.rag_regulatorio import RAGRegulatorioEcuador
    from src.inference.document_intelligence_ia import DocumentIntelligenceEngine, DocumentIntelligenceResultDTO
    from src.inference.recommendations import RecommendationEngine
except ImportError:
    try:
        from guardrails import GuardrailsEngine
        from rag_regulatorio import RAGRegulatorioEcuador
        from document_intelligence_ia import DocumentIntelligenceEngine, DocumentIntelligenceResultDTO
        from recommendations import RecommendationEngine
    except ImportError:
        GuardrailsEngine = None  # type: ignore[assignment,misc]
        RAGRegulatorioEcuador = None  # type: ignore[assignment,misc]
        DocumentIntelligenceEngine = None  # type: ignore[assignment,misc]
        DocumentIntelligenceResultDTO = None  # type: ignore[assignment,misc]
        RecommendationEngine = None  # type: ignore[assignment,misc]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("bg_motor_ia_foundry")

# Tasa anual referencial oficial solicitada para calculo de cuota en Subpaso A
TASA_ANUAL_REFERENCIAL_NLU = 15.50
# Tasa anual preferencial para oferta aprobada en banca movil
TASA_ANUAL_PREFERENCIAL_BG = 14.20


# ============================================================================
# 1. ESQUEMAS DE VALIDACION ESTRICTA (DTOs Pydantic)
# ============================================================================
class NLUResultDTO(BaseModel):
    intencion: Literal["ENDEUDAMIENTO", "INVERSION", "AHORRO", "CONSULTA_GENERAL"] = Field(
        default="ENDEUDAMIENTO",
        description="Intencion principal detectada en el mensaje del cliente.",
    )
    monto_solicitado: float = Field(default=0.0, ge=0.0)
    plazo_meses: int = Field(default=12, ge=1, le=120)
    cuota_mensual_nueva: float = Field(default=0.0, ge=0.0)
    producto_mencionado: str = Field(default="CREDITO_CONSUMO")
    origen_nlu: str = Field(default="AZURE_AI_FOUNDRY_GPT4O_MINI")


# ============================================================================
# 2. FUNCIONES FINANCIERAS DE AMORTIZACION FRANCESA
# ============================================================================
def calcular_cuota_francesa(monto: float, plazo_meses: int, tasa_anual_pct: float = TASA_ANUAL_REFERENCIAL_NLU) -> float:
    """Calcula la cuota mensual fija mediante sistema de amortizacion frances."""
    if monto <= 0 or plazo_meses <= 0:
        return 0.0
    r_mensual = (tasa_anual_pct / 100.0) / 12.0
    if r_mensual <= 0:
        return round(monto / plazo_meses, 2)
    factor = (r_mensual * ((1.0 + r_mensual) ** plazo_meses)) / (((1.0 + r_mensual) ** plazo_meses) - 1.0)
    return round(monto * factor, 2)


def calcular_monto_desde_cuota(
    cuota_mensual: float, plazo_meses: int, tasa_anual_pct: float = TASA_ANUAL_REFERENCIAL_NLU
) -> float:
    """Despeja el capital maximo equivalente dada una cuota mensual fija y un plazo."""
    if cuota_mensual <= 0 or plazo_meses <= 0:
        return 0.0
    r_mensual = (tasa_anual_pct / 100.0) / 12.0
    if r_mensual <= 0:
        return round(cuota_mensual * plazo_meses, 2)
    monto = cuota_mensual * (((1.0 + r_mensual) ** plazo_meses) - 1.0) / (r_mensual * ((1.0 + r_mensual) ** plazo_meses))
    return round(monto, 2)


# ============================================================================
# 3. CLIENTE RESILIENTE DE AZURE AI FOUNDRY (Con DefaultAzureCredential / API Key + Backoff)
# ============================================================================
class AzureFoundryClient:
    """
    Gestiona la conexion hacia Azure AI Foundry (gpt-4o-mini) soportando:
    1. Microsoft Entra ID (DefaultAzureCredential) para entornos productivos en Azure.
    2. API Key (AZURE_OPENAI_API_KEY) para pruebas rapidas de hackathon.
    3. Reintentos con Exponential Backoff + Jitter ante errores HTTP 429.
    4. Fallback determinista local automatico si aun no se han configurado las credenciales.
    """

    def __init__(self) -> None:
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini").strip()
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21").strip()
        self.client: Any = None
        self.is_live_azure = False
        self._initialize_sdk_client()

    def _initialize_sdk_client(self) -> None:
        key_lower = self.api_key.lower()
        if not self.endpoint or not self.api_key or "tu_clave" in key_lower or "pega_aqui" in key_lower:
            logger.info(
                "Credenciales activas de Azure AI Foundry no detectadas en .env. "
                "Modo Fallback Local Inteligente activado para pruebas sin costo."
            )
            return

        try:
            # Soporte nativo para Azure AI Foundry (services.ai.azure.com con /openai/v1)
            if "/openai/v1" in self.endpoint or "services.ai.azure.com" in self.endpoint:
                from openai import OpenAI

                base_url = self.endpoint.rstrip("/")
                if base_url.endswith("/responses"):
                    base_url = base_url[:-10]
                if not base_url.endswith("/openai/v1") and "/api/projects/" not in base_url:
                    base_url = f"{base_url}/openai/v1"

                self.client = OpenAI(
                    base_url=base_url,
                    api_key=self.api_key,
                    timeout=15.0,
                )
                self.is_live_azure = True
                logger.info(f"Conectado a Azure AI Foundry ({self.deployment}) via OpenAI(base_url={base_url}).")
            elif self.api_key:
                from openai import AzureOpenAI

                self.client = AzureOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version,
                    timeout=15.0,
                )
                self.is_live_azure = True
                logger.info(f"Conectado a Azure OpenAI ({self.deployment}) via AzureOpenAI.")
            else:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                from openai import AzureOpenAI

                token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
                self.client = AzureOpenAI(
                    azure_endpoint=self.endpoint,
                    azure_ad_token_provider=token_provider,
                    api_version=self.api_version,
                    timeout=15.0,
                )
                self.is_live_azure = True
                logger.info(f"Conectado a Azure AI Foundry ({self.deployment}) via DefaultAzureCredential (Entra ID).")
        except Exception as exc:
            logger.warning(f"No se pudo inicializar cliente de IA ({exc}). Se utilizara el motor Fallback Local.")
            self.is_live_azure = False

    def call_chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 320,
        temperature: float = 0.1,
    ) -> Optional[Dict[str, Any]]:
        """Invoca gpt-4o-mini en Azure AI Foundry con Structured JSON y Exponential Backoff."""
        if not self.is_live_azure or self.client is None:
            return None

        max_retries = 3
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                raw_content = response.choices[0].message.content or "{}"
                return json.loads(raw_content)
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str and attempt < max_retries - 1:
                    sleep_sec = base_delay * (2**attempt) + random.uniform(0.1, 0.5)
                    logger.warning(f"Rate limit 429 en Azure Foundry. Reintentando en {sleep_sec:.2f}s...")
                    time.sleep(sleep_sec)
                else:
                    logger.warning(f"Fallo llamada a Azure AI Foundry ({exc}). Activando fallback local.")
                    return None
        return None


# ============================================================================
# 4. MOTOR ORQUESTADOR HIBRIDO (SUBPASOS A, B y C)
# ============================================================================
class MotorIAFoundry:
    """
    Conecta el modelo tabular LightGBM entrenado con Azure AI Foundry (gpt-4o-mini)
    para evaluar peticiones en lenguaje natural y devolver el JSON oficial de Banco Guayaquil.
    """

    FEATURE_COLS = [
        "EDAD",
        "ANTIGUEDAD_BANCO_ANIOS",
        "INGRESOS_CREDITO_MES",
        "EGRESOS_DEBITO_MES",
        "SALDO_CUENTAS",
        "CUOTA_CREDITOS_VIGENTES",
        "CAPITAL_POR_VENCER",
        "CAPITAL_VENCIDO",
        "GASTO_PROMEDIO_TC",
        "CUOTA_DIFERIDOS_TC",
        "VIAJES_EXTERIOR_TC",
        "MONTO_POLIZAS_VIGENTES",
        "MONTO_ACUM_METAS",
        "CUMPLIO_META_HIST",
        "INGRESOS_EXTRA_OCR",
        "GASTOS_OCULTOS_OCR",
        "FLUJO_NETO_DISPONIBLE",
        "SCORE_EQUIFAX",
        "SCORE_ESTILO_VIDA_REDES",
    ]

    def __init__(self, model_path: Optional[str | Path] = None) -> None:
        workspace_root = Path(__file__).resolve().parent
        if not (workspace_root / "models").exists() and (workspace_root.parent.parent / "models").exists():
            workspace_root = workspace_root.parent.parent

        default_model_path = workspace_root / os.getenv("BG_RISK_MODEL_PATH", "models/bg_risk_lightgbm.joblib")
        resolved_path = Path(model_path) if model_path else default_model_path

        if not resolved_path.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo entrenado en: {resolved_path}. "
                "Ejecuta primero el entrenamiento de LightGBM."
            )

        self.model = joblib.load(resolved_path)
        self.foundry_client = AzureFoundryClient()
        self.doc_engine = (
            DocumentIntelligenceEngine(openai_client=self.foundry_client.client)
            if DocumentIntelligenceEngine is not None
            else None
        )
        logger.info(f"Modelo ML cargado exitosamente desde: {resolved_path}")

    # ------------------------------------------------------------------------
    # SUBPASO A: NLU CON AZURE AI FOUNDRY (gpt-4o-mini) + FALLBACK LOCAL
    # ------------------------------------------------------------------------
    def subpaso_a_extraer_nlu(self, peticion_usuario: str) -> NLUResultDTO:
        """
        Extrae intencion, monto_solicitado, plazo_meses y cuota_mensual_nueva
        usando un prompt ultra compacto en gpt-4o-mini (o fallback regex determinista).
        """
        system_prompt_nlu = (
            "Eres un extractor NLU bancario oficial de Banco Guayaquil. Devuelve SOLO un JSON valido con las claves exactas: "
            '{"intencion": "ENDEUDAMIENTO"|"INVERSION"|"AHORRO"|"CONSULTA_GENERAL", '
            '"monto_solicitado": float, "plazo_meses": int, "cuota_mensual_nueva": float, '
            '"producto_mencionado": str}. '
            "Si no menciona plazo usa 12. Si menciona cuota mensual ponla en cuota_mensual_nueva y monto_solicitado=0. "
            "Si menciona monto total ponlo en monto_solicitado y cuota_mensual_nueva=0. "
            "Si pregunta que hacer con su dinero, como rentabilizar o invertir sus ahorros, usa intencion 'INVERSION' o 'AHORRO'."
        )

        azure_json = self.foundry_client.call_chat_json(
            system_prompt=system_prompt_nlu,
            user_prompt=peticion_usuario,
            max_tokens=120,
            temperature=0.0,
        )

        if azure_json is not None:
            intencion = str(azure_json.get("intencion", "ENDEUDAMIENTO")).upper()
            if intencion not in {"ENDEUDAMIENTO", "INVERSION", "AHORRO", "CONSULTA_GENERAL"}:
                intencion = "ENDEUDAMIENTO"
            monto = float(azure_json.get("monto_solicitado", 0.0) or 0.0)
            plazo = int(azure_json.get("plazo_meses", 12) or 12)
            cuota = float(azure_json.get("cuota_mensual_nueva", 0.0) or 0.0)
            producto = str(azure_json.get("producto_mencionado", "CREDITO_CONSUMO"))
            origen = "AZURE_AI_FOUNDRY_GPT4O_MINI"
        else:
            parsed = self._fallback_local_nlu(peticion_usuario)
            intencion = parsed["intencion"]
            monto = parsed["monto_solicitado"]
            plazo = parsed["plazo_meses"]
            cuota = parsed["cuota_mensual_nueva"]
            producto = parsed["producto_mencionado"]
            origen = "FALLBACK_LOCAL_DETERMINISTA"

        # Regla financiera del Subpaso A:
        # 1. Si es ENDEUDAMIENTO y el usuario dio el monto pero no la cuota -> calcular con 15.5% anual
        if intencion == "ENDEUDAMIENTO":
            if monto > 0 and cuota <= 0:
                cuota = calcular_cuota_francesa(monto, plazo, tasa_anual_pct=TASA_ANUAL_REFERENCIAL_NLU)
            elif cuota > 0 and monto <= 0:
                monto = calcular_monto_desde_cuota(cuota, plazo, tasa_anual_pct=TASA_ANUAL_REFERENCIAL_NLU)
        else:
            # Si es INVERSION o AHORRO, no es una nueva deuda crediticia
            cuota = 0.0

        return NLUResultDTO(
            intencion=intencion,  # type: ignore[arg-type]
            monto_solicitado=round(monto, 2),
            plazo_meses=plazo,
            cuota_mensual_nueva=round(cuota, 2),
            producto_mencionado=producto,
            origen_nlu=origen,
        )

    @staticmethod
    def _fallback_local_nlu(texto: str) -> Dict[str, Any]:
        """Extractor NLU local de respaldo basado en expresiones regulares financieras."""
        t_low = texto.lower()

        # 1. Detectar intencion
        if any(w in t_low for w in ["invertir", "inversion", "inversión", "poliza", "póliza", "plazo fijo", "rendimiento", "hacer con mi dinero", "hacer mi dinero", "hago con mi dinero", "poner mi dinero", "hacer crecer", "rentabilizar", "donde guardo mi dinero", "donde pongo mi plata"]):
            intencion = "INVERSION"
        elif any(w in t_low for w in ["meta", "ahorrar", "ahorro programado", "guardar dinero", "ahorro flexible"]):
            intencion = "AHORRO"
        elif any(
            w in t_low
            for w in [
                "comprar",
                "credito",
                "crédito",
                "prestamo",
                "préstamo",
                "cuota",
                "endeudar",
                "alcanza",
                "tv",
                "auto",
                "moto",
                "laptop",
                "viaje",
                "diferir",
            ]
        ):
            intencion = "ENDEUDAMIENTO"
        else:
            intencion = "CONSULTA_GENERAL"

        # 2. Detectar plazo en meses o años
        plazo_meses = 12
        match_meses = re.search(r"(\d{1,3})\s*mes(?:es)?", t_low)
        match_anios = re.search(r"(\d{1,2})\s*a[ñn]o(?:s)?", t_low)
        if match_meses:
            plazo_meses = max(1, min(120, int(match_meses.group(1))))
        elif match_anios:
            plazo_meses = max(1, min(120, int(match_anios.group(1)) * 12))

        # 3. Detectar montos en USD (ej. $1,200, 1200 dolares, $200 al mes)
        # Limpiar numeros que corresponden al plazo para no confundirlos con dinero
        texto_sin_plazo = re.sub(r"\d{1,3}\s*(?:mes(?:es)?|a[ñn]o(?:s)?|d[ií]as)", "", t_low)
        montos_encontrados = re.findall(r"\$?\s*(\d{1,3}(?:[.,]\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", texto_sin_plazo)
        valores_usd: List[float] = []
        for m in montos_encontrados:
            clean_val = m.replace(",", "")
            try:
                val = float(clean_val)
                if val >= 10.0:
                    valores_usd.append(val)
            except ValueError:
                continue

        habla_de_cuota_directa = bool(
            re.search(r"cuota\s*(?:mensual\s*)?(?:de\s*)?\$?\s*\d+|pagar\s*\$?\s*\d+\s*(?:al\s*mes|mensual)", t_low)
        )

        monto_solicitado = 0.0
        cuota_mensual_nueva = 0.0
        if valores_usd:
            primer_valor = valores_usd[0]
            if habla_de_cuota_directa:
                cuota_mensual_nueva = primer_valor
            else:
                monto_solicitado = primer_valor

        producto = "POLIZA_INVERSION" if intencion == "INVERSION" else ("META_AHORRO" if intencion == "AHORRO" else "CREDITO_CONSUMO")
        return {
            "intencion": intencion,
            "monto_solicitado": monto_solicitado,
            "plazo_meses": plazo_meses,
            "cuota_mensual_nueva": cuota_mensual_nueva,
            "producto_mencionado": producto,
        }

    # ------------------------------------------------------------------------
    # SUBPASO B: INFERENCIA CON MODELO ML LIGHTGBM + REGLA DEL SEMAFORO
    # ------------------------------------------------------------------------
    def _extraer_features_base(self, datos_cliente: Dict[str, Any]) -> pd.DataFrame:
        """
        Convierte 'datos_cliente' (ya sea en formato JSON anidado de las 8 tablas
        o diccionario plano de columnas) al DataFrame de 19 features del modelo LightGBM.
        """
        # Caso 1: Diccionario plano que ya contiene las columnas del dataset
        if "INGRESOS_CREDITO_MES" in datos_cliente and "SCORE_EQUIFAX" in datos_cliente:
            row = {col: float(datos_cliente.get(col, 0.0)) for col in self.FEATURE_COLS}
            return pd.DataFrame([row])[self.FEATURE_COLS]

        # Caso 2: Estructura oficial anidada de las 8 tablas de Banco Guayaquil
        ref_date = datetime(2026, 10, 24)
        clte = datos_cliente.get("maestro_clientes", {})
        birth_str = clte.get("FECHA_NACIMIENTO", "1995-01-01")
        birth_dt = datetime.strptime(birth_str[:10], "%Y-%m-%d")
        edad = max(18, int((ref_date - birth_dt).days / 365.25))

        vinc_str = clte.get("FECHA_VINCULACION", "2020-01-01")
        vinc_dt = datetime.strptime(vinc_str[:10], "%Y-%m-%d")
        antiguedad = max(0.1, round((ref_date - vinc_dt).days / 365.25, 1))

        cuentas_data = datos_cliente.get("resumen_cuentas_y_movimientos", {})
        cuentas_activas = cuentas_data.get("cuentas_activas", [])
        saldo_cuentas = sum(float(c.get("SALDO", 0.0)) for c in cuentas_activas)

        mvto_cuentas = cuentas_data.get("agregado_mvto_cuentas_mensual", {})
        ingresos_banco = float(mvto_cuentas.get("promedio_ingresos_credito_mes", 0.0))
        egresos_banco = float(mvto_cuentas.get("promedio_egresos_debito_mes", 0.0))

        cred_data = datos_cliente.get("resumen_creditos_vigentes", {})
        cap_por_vencer = float(cred_data.get("total_por_vencer", 0.0))
        cap_vencido = float(cred_data.get("total_vencido_mora", 0.0))
        ops = cred_data.get("operaciones_activas", [])
        cuota_creditos = 0.0
        for op in ops:
            if "CUOTA_MENSUAL" in op:
                cuota_creditos += float(op["CUOTA_MENSUAL"])
            else:
                monto = float(op.get("MONTO", 0.0))
                por_vencer = float(op.get("POR_VENCER", 0.0))
                if por_vencer > 0:
                    cuota_creditos += max(25.0, round(monto / 24.0, 2))

        tc_data = datos_cliente.get("resumen_tarjetas_tc", {}).get("agregado_mvto_tc_mensual", {})
        gasto_tc = float(tc_data.get("gasto_promedio_mensual_tc", 0.0))
        cuota_diferidos_tc = float(tc_data.get("saldo_diferido_estimado_cuotas", 0.0))
        paises = tc_data.get("paises_consumo_reciente", [])
        viajes_tc = 1 if tc_data.get("viajes_internacionales_detectados", False) or any(p != "ECU" for p in paises) else 0

        inv_data = datos_cliente.get("inversiones_y_ahorro_actual", {})
        monto_polizas = sum(float(p.get("MONTO", 0.0)) for p in inv_data.get("maestro_polizas", []))
        metas = inv_data.get("maestro_metas", [])
        monto_metas = sum(float(m.get("MONTO_ACUM", 0.0)) for m in metas)
        cumplio_meta = 1 if any(
            int(m.get("CUMPLIO_META", 0)) == 1
            or (float(m.get("MONTO_ACUM", 0.0)) / max(1.0, float(m.get("META", 1.0)))) >= 0.25
            for m in metas
        ) else 0

        ext = datos_cliente.get("fuentes_externas_ia", {})
        score_equifax = int(ext.get("buro_equifax", {}).get("score_equifax", datos_cliente.get("SCORE_EQUIFAX", 700)))
        cuota_otros_bancos = float(ext.get("buro_equifax", {}).get("cuota_mensual_otros_bancos", 0.0))

        ocr = ext.get("document_intelligence_ocr", {})
        ingresos_ocr = float(ocr.get("ingresos_adicionales_detectados", 0.0))
        gastos_ocultos_ocr = float(ocr.get("gastos_ocultos_detectados", 0.0))
        score_estilo_vida = float(ext.get("estilo_vida_redes_sociales", {}).get("score_estilo_vida", 5.5))

        ingresos_totales = ingresos_banco + ingresos_ocr
        egresos_totales = (
            egresos_banco
            + cuota_diferidos_tc
            + gastos_ocultos_ocr
            + cuota_otros_bancos
            + max(0.0, gasto_tc - cuota_diferidos_tc) * 0.50
        )
        flujo_neto = round(ingresos_totales - egresos_totales, 2)

        row = {
            "EDAD": edad,
            "ANTIGUEDAD_BANCO_ANIOS": antiguedad,
            "INGRESOS_CREDITO_MES": ingresos_banco,
            "EGRESOS_DEBITO_MES": egresos_banco,
            "SALDO_CUENTAS": saldo_cuentas,
            "CUOTA_CREDITOS_VIGENTES": cuota_creditos,
            "CAPITAL_POR_VENCER": cap_por_vencer,
            "CAPITAL_VENCIDO": cap_vencido,
            "GASTO_PROMEDIO_TC": gasto_tc,
            "CUOTA_DIFERIDOS_TC": cuota_diferidos_tc,
            "VIAJES_EXTERIOR_TC": viajes_tc,
            "MONTO_POLIZAS_VIGENTES": monto_polizas,
            "MONTO_ACUM_METAS": monto_metas,
            "CUMPLIO_META_HIST": cumplio_meta,
            "INGRESOS_EXTRA_OCR": ingresos_ocr,
            "GASTOS_OCULTOS_OCR": round(gastos_ocultos_ocr + cuota_otros_bancos, 2),
            "FLUJO_NETO_DISPONIBLE": flujo_neto,
            "SCORE_EQUIFAX": score_equifax,
            "SCORE_ESTILO_VIDA_REDES": score_estilo_vida,
        }
        return pd.DataFrame([row])[self.FEATURE_COLS]

    def subpaso_b_inferencia_ml_y_semaforo(
        self, datos_cliente: Dict[str, Any], nlu: NLUResultDTO
    ) -> Dict[str, Any]:
        """
        1. Extrae las features actuales del cliente.
        2. Suma el impacto de 'cuota_mensual_nueva' (Subpaso A) al perfil financiero.
        3. Ejecuta el modelo LightGBM entrenado (.joblib) y calcula probabilidad de mora,
           capacidad maxima de cuota, flujo neto disponible y regla del Semaforo.
        """
        df_base = self._extraer_features_base(datos_cliente)
        f_base = df_base.iloc[0]

        flujo_neto_actual = float(f_base["FLUJO_NETO_DISPONIBLE"])
        ingresos_totales = float(f_base["INGRESOS_CREDITO_MES"] + f_base["INGRESOS_EXTRA_OCR"])
        cuotas_actuales = float(
            f_base["CUOTA_CREDITOS_VIGENTES"] + f_base["CUOTA_DIFERIDOS_TC"] + f_base["GASTOS_OCULTOS_OCR"]
        )
        dti_actual_pct = round((cuotas_actuales / max(1.0, ingresos_totales)) * 100.0, 1)

        # Capacidad maxima de nueva cuota segura (Politica prudencial BG:
        # hasta llevar el DTI al 40% de sus ingresos y sin superar el 40% de su flujo neto libre)
        tope_por_dti = max(0.0, (ingresos_totales * 0.40) - cuotas_actuales)
        tope_por_flujo = max(0.0, flujo_neto_actual * 0.40)
        capacidad_maxima_cuota = round(min(tope_por_dti, tope_por_flujo), 2)

        # Inyectar el impacto de la nueva deuda en el DataFrame que entra al modelo LightGBM
        df_sim = df_base.copy()
        cuota_nueva = nlu.cuota_mensual_nueva if nlu.intencion == "ENDEUDAMIENTO" else 0.0
        monto_nuevo = nlu.monto_solicitado if nlu.intencion == "ENDEUDAMIENTO" else 0.0

        flujo_neto_post_cuota = round(flujo_neto_actual - cuota_nueva, 2)
        cuotas_con_simulacion = cuotas_actuales + cuota_nueva
        dti_con_simulacion_pct = round((cuotas_con_simulacion / max(1.0, ingresos_totales)) * 100.0, 1)

        df_sim["CUOTA_CREDITOS_VIGENTES"] = float(f_base["CUOTA_CREDITOS_VIGENTES"]) + cuota_nueva
        df_sim["CAPITAL_POR_VENCER"] = float(f_base["CAPITAL_POR_VENCER"]) + monto_nuevo
        df_sim["FLUJO_NETO_DISPONIBLE"] = flujo_neto_post_cuota

        # Si la nueva cuota deja al cliente con flujo negativo o sobreendeudamiento severo,
        # reflejar la degradacion proyectada en el score de buro para que el arbol capture el estres futuro
        if flujo_neto_post_cuota < 0:
            deficit_ratio = abs(flujo_neto_post_cuota) / max(1.0, ingresos_totales)
            df_sim["SCORE_EQUIFAX"] = max(300.0, float(f_base["SCORE_EQUIFAX"]) - (180.0 + deficit_ratio * 250.0))
        elif cuota_nueva > capacidad_maxima_cuota and capacidad_maxima_cuota > 0:
            exceso_ratio = (cuota_nueva - capacidad_maxima_cuota) / max(1.0, capacidad_maxima_cuota)
            df_sim["SCORE_EQUIFAX"] = max(300.0, float(f_base["SCORE_EQUIFAX"]) - min(160.0, exceso_ratio * 95.0))

        # Inferencia probabilistica con nuestro modelo LightGBM (.joblib)
        prob_ml_raw = float(self.model.predict_proba(df_sim)[0, 1])

        # Ajuste continuo de estres de liquidez sobre la probabilidad del modelo para sensibilidad fina
        stress_addon = 0.0
        if cuota_nueva > 0:
            if flujo_neto_post_cuota < 0:
                stress_addon += min(0.55, 0.35 + abs(flujo_neto_post_cuota) / max(100.0, ingresos_totales))
            elif cuota_nueva > capacidad_maxima_cuota:
                exceso = (cuota_nueva - capacidad_maxima_cuota) / max(10.0, flujo_neto_actual)
                stress_addon += min(0.45, 0.22 + exceso * 0.40)
            elif cuota_nueva >= capacidad_maxima_cuota * 0.75:
                # Zona ajustada (Amarillo): sube levemente el riesgo por baja holgura ante imprevistos
                ratio_uso_cap = cuota_nueva / max(1.0, capacidad_maxima_cuota)
                stress_addon += 0.14 * ratio_uso_cap

        probabilidad_mora = round(min(0.99, max(0.01, prob_ml_raw * 0.75 + stress_addon)), 2)

        # Regla determinista del Semaforo (VERDE / AMARILLO / ROJO)
        cap_vencido = float(f_base["CAPITAL_VENCIDO"])
        score_eq = int(f_base["SCORE_EQUIFAX"])
        uso_capacidad_pct = (cuota_nueva / capacidad_maxima_cuota * 100.0) if capacidad_maxima_cuota > 0 else (100.0 if cuota_nueva > 0 else 0.0)

        if (
            cap_vencido > 0
            or probabilidad_mora >= 0.35
            or (nlu.intencion == "ENDEUDAMIENTO" and (cuota_nueva > capacidad_maxima_cuota or flujo_neto_post_cuota < 50.0))
            or score_eq < 580
        ):
            semaforo = "ROJO"
            se_puede_endeudar = False
            estado_resolucion = "NO_RECOMENDADO_RIESGO_MORA"
        elif (
            probabilidad_mora >= 0.16
            or (nlu.intencion == "ENDEUDAMIENTO" and uso_capacidad_pct >= 72.0)
            or (nlu.intencion == "ENDEUDAMIENTO" and flujo_neto_post_cuota < 225.0)
            or dti_con_simulacion_pct > 36.0
        ):
            semaforo = "AMARILLO"
            se_puede_endeudar = True
            estado_resolucion = "APROBADO_CON_AJUSTE_PRESUPUESTO"
        else:
            semaforo = "VERDE"
            se_puede_endeudar = True
            estado_resolucion = "APROBADO_OPTIMO"

        score_salud_interna = max(
            10,
            min(
                99,
                int(round((1.0 - probabilidad_mora) * 70.0 + (score_eq / 999.0) * 30.0)),
            ),
        )

        plazo_eval = nlu.plazo_meses if nlu.plazo_meses > 0 else 24
        monto_max_recomendado = calcular_monto_desde_cuota(
            capacidad_maxima_cuota, plazo_eval, tasa_anual_pct=TASA_ANUAL_PREFERENCIAL_BG
        )
        cuota_oferta_bg = (
            calcular_cuota_francesa(nlu.monto_solicitado, plazo_eval, tasa_anual_pct=TASA_ANUAL_PREFERENCIAL_BG)
            if (se_puede_endeudar and nlu.monto_solicitado > 0)
            else capacidad_maxima_cuota
        )

        # Extraer gastos variables recortables (ej. restaurantes/apps) de MVTO_CUENTAS si existen
        top_trx = (
            datos_cliente.get("resumen_cuentas_y_movimientos", {})
            .get("agregado_mvto_cuentas_mensual", {})
            .get("top_motivos_trx", [])
        )
        gasto_restaurantes_apps = sum(
            float(t.get("MONTO_MENSUAL", 0.0))
            for t in top_trx
            if any(k in str(t.get("MOTIVO_TRX", "")).upper() for k in ["RESTAURANTE", "APP", "OCIO", "ENTRETENIMIENTO", "DELIVERY"])
        )
        if gasto_restaurantes_apps <= 0:
            gasto_restaurantes_apps = round(float(f_base["EGRESOS_DEBITO_MES"]) * 0.18, 2)

        recorte_sugerido = round(min(gasto_restaurantes_apps * 0.35, max(40.0, cuota_nueva * 0.50)), 0)

        # Factores explicativos deterministas del modelo
        factores_clave: List[str] = []
        if cap_vencido == 0:
            factores_clave.append("Historial al dia en MAESTRO_CREDITOS (VENCIDO = $0.00).")
        else:
            factores_clave.append(f"Mora activa detectada en MAESTRO_CREDITOS (VENCIDO = ${cap_vencido:,.2f}).")

        if semaforo == "ROJO":
            factores_clave.append(
                f"La nueva cuota de ${cuota_nueva:,.2f}/mes supera tu capacidad maxima segura de ${capacidad_maxima_cuota:,.2f}/mes."
            )
            factores_clave.append(
                f"Tu flujo libre caeria de ${flujo_neto_actual:,.2f} a ${flujo_neto_post_cuota:,.2f}, elevando el riesgo de mora a {probabilidad_mora*100:.0f}%."
            )
        elif semaforo == "AMARILLO":
            factores_clave.append(
                f"La cuota de ${cuota_nueva:,.2f}/mes consume el {uso_capacidad_pct:.0f}% de tu capacidad segura (${capacidad_maxima_cuota:,.2f}/mes)."
            )
            factores_clave.append(
                f"Se detectan ${gasto_restaurantes_apps:,.2f}/mes en consumos variables (restaurantes/apps) donde puedes recortar ${recorte_sugerido:,.0f}/mes."
            )
        else:
            factores_clave.append(
                f"Holgura financiera solida: flujo libre de ${flujo_neto_actual:,.2f}/mes y capacidad de cuota de hasta ${capacidad_maxima_cuota:,.2f}/mes."
            )
            if float(f_base["SALDO_CUENTAS"]) >= 500:
                factores_clave.append(
                    f"Liquidez disponible en MAESTRO_CUENTAS (${float(f_base['SALDO_CUENTAS']):,.2f}) apta para rentabilizar en Poliza."
                )

        # Construccion de las 3 ofertas de productos de Banco Guayaquil + Asesoria Patrimonial
        saldo_cta = float(f_base["SALDO_CUENTAS"])
        cuota_tc_diferidos = float(f_base["CUOTA_DIFERIDOS_TC"])

        # Determinar Tier Patrimonial segun saldo disponible y flujo neto
        if saldo_cta >= 3000.0 or (saldo_cta >= 2000.0 and flujo_neto_actual >= 600.0):
            tier_patrimonial = "TIER_ALTO_PATRIMONIAL"
        elif saldo_cta >= 500.0 or flujo_neto_actual >= 250.0:
            tier_patrimonial = "TIER_MEDIO_FLEXIBLE"
        else:
            tier_patrimonial = "TIER_BASICO_CRECIMIENTO"

        tiene_deuda_cara_tc = bool(cuota_tc_diferidos >= 60.0 and saldo_cta >= 600.0)

        if tier_patrimonial == "TIER_ALTO_PATRIMONIAL":
            colchon_reserva = min(1200.0, max(600.0, round(saldo_cta * 0.20, -1)))
            monto_poliza_sug = float(max(1000.0, round(saldo_cta - colchon_reserva, -2))) if semaforo != "ROJO" else 0.0
            tasa_poliza = 7.25
            plazo_poliza_dias = 180
            interes_poliza = round(monto_poliza_sug * (tasa_poliza / 100.0) * (plazo_poliza_dias / 360.0), 2)
            ahorro_meta_mensual = float(round(max(80.0, min(250.0, flujo_neto_actual * 0.25)), -1))
            meta_objetivo = 3000.0
            motivo_poliza = (
                f"Cuentas con un saldo disponible de ${saldo_cta:,.2f}. Al colocar ${monto_poliza_sug:,.0f} a {plazo_poliza_dias} días al {tasa_poliza:.2f}% anual, "
                f"ganas ${interes_poliza:,.2f} con seguro COSEDE hasta $32,000, dejando ${colchon_reserva:,.0f} disponibles en cuenta."
            )
            estrategia_asesoria = (
                f"Maximizar rendimiento seguro en Póliza a Plazo Fijo al {tasa_poliza:.2f}% anual manteniendo colchón operativo."
            )
            valor_cliente = (
                f"Rendimiento garantizado del {tasa_poliza:.2f}% anual superior a la inflación con respaldo de COSEDE hasta $32,000."
            )
            valor_banco = (
                "Captación de depósitos a plazo fijo de alto valor patrimonial, estabilidad de fondeo para colocación crediticia."
            )
        elif tier_patrimonial == "TIER_MEDIO_FLEXIBLE":
            colchon_reserva = min(500.0, max(200.0, round(saldo_cta * 0.35, -1)))
            monto_poliza_sug = float(max(300.0, round(saldo_cta - colchon_reserva, -1))) if (saldo_cta >= 500.0 and semaforo != "ROJO") else 0.0
            tasa_poliza = 6.50
            plazo_poliza_dias = 90
            interes_poliza = round(monto_poliza_sug * (tasa_poliza / 100.0) * (plazo_poliza_dias / 360.0), 2) if monto_poliza_sug > 0 else 0.0
            ahorro_meta_mensual = float(round(max(50.0, min(120.0, max(50.0, flujo_neto_actual * 0.30))), -1))
            meta_objetivo = 1500.0
            motivo_poliza = (
                f"Tienes ${saldo_cta:,.2f} en ahorros. Colocar ${monto_poliza_sug:,.0f} al {tasa_poliza:.2f}% a 90 días te genera intereses "
                f"sin inmovilizar tu capital operativo."
                if monto_poliza_sug >= 300.0
                else "Con tu saldo actual, te conviene consolidar una meta flexible antes de inmovilizar a plazo fijo."
            )
            estrategia_asesoria = (
                f"Estrategia híbrida: Póliza a corto plazo ({plazo_poliza_dias} días al {tasa_poliza:.2f}%) + Ahorro Programado flexible."
            )
            valor_cliente = (
                "Rendimiento sobre ahorros sin perder la flexibilidad de tener liquidez para imprevistos mediante débito automático."
            )
            valor_banco = (
                "Cross-selling equilibrado (Póliza + Meta de ahorro recurrente) e incremento de saldos medios por cliente."
            )
        else:
            colchon_reserva = saldo_cta
            monto_poliza_sug = 0.0
            tasa_poliza = 0.0
            plazo_poliza_dias = 0
            interes_poliza = 0.0
            ahorro_meta_mensual = float(round(max(30.0, min(60.0, max(30.0, flujo_neto_actual * 0.25 if flujo_neto_actual > 0 else 30.0))), -1))
            meta_objetivo = 1000.0
            motivo_poliza = "Tu prioridad hoy es armar un fondo de reserva antes de inmovilizar dinero a plazo fijo."
            estrategia_asesoria = (
                f"Construir Fondo de Emergencia con Ahorro Flexible en MAESTRO_METAS debitando ${ahorro_meta_mensual:,.0f}/mes sin penalidades."
            )
            valor_cliente = (
                "Construyes un colchón protector ante imprevistos sin pagar comisiones ni depender de créditos caros."
            )
            valor_banco = (
                "Reduce la probabilidad de morosidad temprana en tarjetas y microcréditos, y fomenta fidelidad financiera."
            )

        if semaforo == "ROJO" and nlu.intencion == "ENDEUDAMIENTO":
            meta_objetivo = float(nlu.monto_solicitado if nlu.monto_solicitado > 0 else 1500.0)
            ahorro_meta_mensual = float(
                round(max(40.0, min(capacidad_maxima_cuota * 0.75, recorte_sugerido + 30.0)), -1)
                if capacidad_maxima_cuota >= 40
                else 40.0
            )

        recomendaciones_bg = {
            "1_oferta_maestro_creditos": {
                "habilitado": bool(se_puede_endeudar and nlu.intencion in {"ENDEUDAMIENTO", "CONSULTA_GENERAL"}),
                "CODGRUOPE_sugerido": "CON",
                "MONTO_aprobado": float(nlu.monto_solicitado if se_puede_endeudar else 0.0),
                "monto_maximo_disponible": float(monto_max_recomendado if se_puede_endeudar else 0.0),
                "TASA_anual_preferencial": TASA_ANUAL_PREFERENCIAL_BG,
                "plazo_meses": int(plazo_eval),
                "cuota_mensual": float(cuota_oferta_bg if se_puede_endeudar else 0.0),
                "CANAL_VENTA": "APP_AI_BANKING",
            },
            "2_oferta_maestro_polizas": {
                "habilitado": bool(monto_poliza_sug >= 300.0 and semaforo != "ROJO"),
                "motivo": motivo_poliza,
                "MONTO_sugerido": float(monto_poliza_sug),
                "PLAZO_DIAS": int(plazo_poliza_dias),
                "TASA_anual": float(tasa_poliza),
                "interes_estimado_ganar": float(interes_poliza),
                "respaldo_cosede_monto": 32000.00,
            },
            "3_oferta_maestro_metas": {
                "habilitado": True,
                "motivo": (
                    f"En lugar de sobreendeudarte hoy, ahorra ${ahorro_meta_mensual:,.0f}/mes en MAESTRO_METAS para reunir ${meta_objetivo:,.0f} sin pagar intereses."
                    if semaforo == "ROJO"
                    else (
                        f"Ahorra ${ahorro_meta_mensual:,.0f}/mes en MAESTRO_METAS de forma flexible para contingencias y hacer crecer tu liquidez."
                        if nlu.intencion in {"INVERSION", "AHORRO"}
                        else f"Recortando ${recorte_sugerido:,.0f}/mes de gastos en restaurantes/apps de MVTO_CUENTAS, cubres tu meta con holgura."
                    )
                ),
                "FRECUENCIA": "M",
                "MONTO_DEBITO_sugerido": float(ahorro_meta_mensual),
                "META_objetivo": float(meta_objetivo),
                "DIA_DEBITO": 15,
            },
            "asesoria_patrimonial_inteligente": {
                "tier_patrimonial": tier_patrimonial,
                "saldo_cuentas_actual": float(saldo_cta),
                "colchon_reserva_sugerido": float(colchon_reserva),
                "estrategia_principal": estrategia_asesoria,
                "tiene_deuda_cara_tc": tiene_deuda_cara_tc,
                "valor_mutuo": {
                    "valor_para_cliente": valor_cliente,
                    "valor_para_banco_guayaquil": valor_banco,
                },
            },
        }

        return {
            "evaluacion_riesgo_ml": {
                "score_salud_interna": int(score_salud_interna),
                "probabilidad_caer_en_vencido_mora": float(probabilidad_mora),
                "semaforo_salud_financiera": semaforo,
                "flujo_neto_mensual_disponible": float(flujo_neto_actual),
                "flujo_neto_post_simulacion": float(flujo_neto_post_cuota),
                "capacidad_maxima_nueva_cuota": float(capacidad_maxima_cuota),
                "endeudamiento_actual_pct": float(dti_actual_pct),
                "endeudamiento_con_simulacion_pct": float(dti_con_simulacion_pct),
            },
            "dictamen_endeudamiento": {
                "se_puede_endeudar": bool(se_puede_endeudar),
                "estado_resolucion": estado_resolucion,
                "monto_maximo_recomendado": float(monto_max_recomendado),
                "cuota_mensual_segura": float(cuota_oferta_bg if se_puede_endeudar else capacidad_maxima_cuota),
                "impacto_proyectado_equifax": (
                    f"Si tomas esta deuda hoy, tu riesgo de atraso haria caer tu score Equifax de {score_eq} a ~{max(350, score_eq - 95)}."
                    if semaforo == "ROJO"
                    else f"Manteniendo VENCIDO en $0.00 durante 6 meses, tu score pasara de {score_eq} a ~{min(999, score_eq + 30)}."
                ),
                "factores_clave_modelo": factores_clave,
            },
            "recomendaciones_productos_bg": recomendaciones_bg,
            "_contexto_compacto_llm": {
                "intencion": nlu.intencion,
                "monto_solicitado": nlu.monto_solicitado,
                "plazo_meses": nlu.plazo_meses,
                "cuota_nueva": nlu.cuota_mensual_nueva,
                "semaforo": semaforo,
                "prob_mora": probabilidad_mora,
                "flujo_libre_actual": flujo_neto_actual,
                "flujo_post_cuota": flujo_neto_post_cuota,
                "capacidad_max_cuota": capacidad_maxima_cuota,
                "score_equifax": score_eq,
                "saldo_cuentas": saldo_cta,
                "capital_vencido": cap_vencido,
                "recorte_gastos_sugerido": recorte_sugerido,
                "ahorro_meta_sugerido": ahorro_meta_mensual,
                "monto_poliza_sugerido": monto_poliza_sug,
                "tasa_poliza_sugerida": tasa_poliza,
                "plazo_poliza_dias": plazo_poliza_dias,
                "interes_poliza_estimado": interes_poliza,
                "colchon_reserva_sugerido": colchon_reserva,
                "tier_patrimonial": tier_patrimonial,
                "tiene_deuda_cara_tc": tiene_deuda_cara_tc,
                "cuota_diferidos_tc": cuota_tc_diferidos,
                "estrategia_asesoria": estrategia_asesoria,
            },
        }

    # ------------------------------------------------------------------------
    # SUBPASO C: GENERACION DE RECOMENDACION EMPATICA CON AZURE AI FOUNDRY
    # ------------------------------------------------------------------------
    def subpaso_c_generar_respuesta_foundry(
        self,
        peticion_usuario: str,
        resultado_subpaso_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Envia UNICAMENTE el resumen matematico compacto del Subpaso B a gpt-4o-mini
        para redactar el veredicto empatico, directo y accionable sin gastar tokens de mas.
        """
        ctx = resultado_subpaso_b["_contexto_compacto_llm"]
        recs = resultado_subpaso_b["recomendaciones_productos_bg"]

        system_prompt_xai = (
            "Eres el Copiloto de Salud Financiera y Asesor Patrimonial de Banco Guayaquil. "
            "Recibes metricas exactas de nuestro modelo ML de riesgo y perfil patrimonial. "
            "Responde SOLO un JSON con dos claves: "
            '{"titulo_pantalla": str, "explicacion_ia": str}. '
            "Reglas de tono y negocio: "
            "1. Si la intencion es INVERSION o AHORRO (o el usuario pregunta que hacer con su dinero, como rentabilizar o donde guardar): "
            "   Aplica la asesoria patrimonial segun 'tier_patrimonial': "
            "   - TIER_ALTO_PATRIMONIAL ($3,000+): Felicitalo por su solvencia y recomiendale abrir una Poliza a Plazo Fijo en MAESTRO_POLIZAS "
            "     por el monto sugerido ($X) al 7.25% anual a 180 dias con respaldo COSEDE hasta $32,000, dejando su colchon operativo ($Y) en cuenta de ahorros. "
            "   - TIER_MEDIO_FLEXIBLE ($500 - $3,000): Proponle una estrategia hibrida: rentabilizar una parte en Poliza a corto plazo (90 dias al 6.50%) "
            "     y activar Ahorro Programado flexible en MAESTRO_METAS ($X/mes) para no comprometer su liquidez diaria. "
            "   - TIER_BASICO_CRECIMIENTO (< $500): Recomiendale activar Ahorro Flexible en MAESTRO_METAS por $X/mes para armar su fondo de emergencia de 3 meses. "
            "   - Si 'tiene_deuda_cara_tc' es true: Recomiendale primero liquidar o amortizar su saldo diferido de tarjeta (~16% anual) "
            "     porque ahorrarse ese interes da un rendimiento neto superior a cualquier inversion conservadora. "
            "   - Muestra el valor mutuo: rentabilidad y seguridad garantizada para el cliente, captacion y solidez para Banco Guayaquil. "
            "2. Si la intencion es ENDEUDAMIENTO: "
            "   - Si semaforo es ROJO: inicia diciendo 'Te recomendamos que no hagas eso en este momento porque...' explicando con numeros exactos "
            "     por que caeria en mora y proponle el plan de ahorro en MAESTRO_METAS o recalcular monto seguro. "
            "   - Si semaforo es AMARILLO: dile que si le alcanza pero con lo justo, y explicale cuanto recortar en gastos variables o ahorrar de entrada. "
            "   - Si semaforo es VERDE: felicitalo y confirma que puede tomar el credito con su cuota aprobada y capacidad maxima. "
            "3. Si 'documento_ocr' esta presente con gastos_ocultos > 0: MENCIONA EXPLICITAMENTE la entidad y cuota detectada en el documento "
            "   (ej. cuota mensual de $320 en Banco Pichincha) y explica como esa obligacion externa reduce su liquidez o condiciona la decision. "
            "Maximo 85 palabras en explicacion_ia. Cero emojis. Tono profesional y empatico."
        )

        user_prompt_compacto = json.dumps(
            {"peticion": peticion_usuario, "metricas_ml": ctx},
            ensure_ascii=False,
        )

        azure_xai = self.foundry_client.call_chat_json(
            system_prompt=system_prompt_xai,
            user_prompt=user_prompt_compacto,
            max_tokens=220,
            temperature=0.2,
        )

        if azure_xai and "titulo_pantalla" in azure_xai and "explicacion_ia" in azure_xai:
            titulo = str(azure_xai["titulo_pantalla"])
            explicacion = str(azure_xai["explicacion_ia"])
            origen_xai = "AZURE_AI_FOUNDRY_GPT4O_MINI"
        else:
            titulo, explicacion = self._fallback_local_xai(ctx)
            origen_xai = "FALLBACK_LOCAL_XAI"

        acciones_ui = self._construir_acciones_ui(ctx, recs)

        return {
            "titulo_pantalla": titulo,
            "explicacion_ia": explicacion,
            "origen_generacion": origen_xai,
            "acciones_ui": acciones_ui,
        }

    @staticmethod
    def _fallback_local_xai(ctx: Dict[str, Any]) -> tuple[str, str]:
        """Redactor XAI empatico local cuando no hay conexion activa a Azure AI Foundry."""
        sem = ctx["semaforo"]
        monto = ctx["monto_solicitado"]
        cuota = ctx["cuota_nueva"]
        cap = ctx["capacidad_max_cuota"]
        flujo = ctx["flujo_libre_actual"]
        flujo_post = ctx["flujo_post_cuota"]
        eq = ctx["score_equifax"]
        saldo = ctx["saldo_cuentas"]
        recorte = ctx["recorte_gastos_sugerido"]
        meta_mes = ctx["ahorro_meta_sugerido"]
        poliza = ctx.get("monto_poliza_sugerido", 0.0)
        tasa_poliza = ctx.get("tasa_poliza_sugerida", 6.50)
        plazo_poliza = ctx.get("plazo_poliza_dias", 180)
        interes_ganar = ctx.get("interes_poliza_estimado", 0.0)
        colchon = ctx.get("colchon_reserva_sugerido", 240.0)
        tier = ctx.get("tier_patrimonial", "TIER_MEDIO_FLEXIBLE")
        tiene_tc = ctx.get("tiene_deuda_cara_tc", False)
        cuota_tc = ctx.get("cuota_diferidos_tc", 0.0)
        intencion = ctx.get("intencion", "ENDEUDAMIENTO")

        # --------------------------------------------------------------------
        # CASO 1: ASESORIA PATRIMONIAL E INVERSION (Preguntas sobre que hacer con el dinero)
        # --------------------------------------------------------------------
        if intencion in {"INVERSION", "AHORRO"}:
            if tier == "TIER_ALTO_PATRIMONIAL":
                titulo = f"Estrategia Patrimonial: Rentabiliza ${poliza:,.0f} en Póliza a Plazo Fijo ({tasa_poliza:.2f}% anual)."
                prepago_aviso = (
                    f"Como mantienes diferidos de tarjeta de crédito, te sugerimos destinar parte a amortizarlos para ahorrarte el ~16% de interés. "
                    if tiene_tc
                    else ""
                )
                explicacion = (
                    f"Cuentas con un saldo disponible de ${saldo:,.2f} y finanzas solventes. {prepago_aviso}"
                    f"Te recomendamos abrir una Póliza en MAESTRO_POLIZAS por ${poliza:,.0f} a {plazo_poliza} días al {tasa_poliza:.2f}% anual, "
                    f"asegurando ${interes_ganar:,.2f} garantizados con respaldo COSEDE hasta $32,000, "
                    f"dejando un colchón de ${colchon:,.0f} en tu cuenta de ahorros para tu día a día."
                )
            elif tier == "TIER_MEDIO_FLEXIBLE":
                titulo = f"Estrategia Híbrida: Póliza a corto plazo (${poliza:,.0f}) + Ahorro Flexible."
                explicacion = (
                    f"Tienes ${saldo:,.2f} en ahorros y un flujo mensual libre de ${flujo:,.2f}. La opción más inteligente es colocar ${poliza:,.0f} "
                    f"a {plazo_poliza} días en MAESTRO_POLIZAS al {tasa_poliza:.2f}% anual para rentabilizar tu dinero, complementando con un Ahorro Flexible de "
                    f"${meta_mes:,.0f}/mes en MAESTRO_METAS para disponer de liquidez diaria sin descapitalizarte."
                )
            else:
                titulo = "Fondo de Emergencia: Ahorro Flexible en MAESTRO_METAS."
                explicacion = (
                    f"Con tu saldo de ${saldo:,.2f} y flujo libre de ${flujo:,.2f}/mes, la prioridad es construir un fondo de emergencia antes de "
                    f"inmovilizar capital a plazo fijo. Te recomendamos activar un débito de ${meta_mes:,.0f}/mes en MAESTRO_METAS en quincena "
                    f"o fin de mes, asegurando liquidez sin comisiones ni penalidades."
                )
            return titulo, explicacion

        # --------------------------------------------------------------------
        # CASO 2: ENDEUDAMIENTO (SEMAFORO ROJO / AMARILLO / VERDE)
        # --------------------------------------------------------------------
        if sem == "ROJO":
            doc_info = ctx.get("documento_ocr", {})
            mencion_doc = ""
            if doc_info and doc_info.get("gastos_ocultos", 0) > 0:
                mencion_doc = f"Al analizar tu documento de {doc_info.get('entidad', 'otra entidad')}, detectamos una cuota externa fija de ${doc_info['gastos_ocultos']:,.2f}/mes no registrada en Banco Guayaquil. "

            titulo = f"Alerta de Salud Financiera: Evita endeudarte con ${cuota:,.2f}/mes ahora."
            razon_vencido = (
                f"actualmente registras ${ctx['capital_vencido']:,.2f} de capital vencido y "
                if ctx["capital_vencido"] > 0
                else ""
            )
            explicacion = (
                f"{mencion_doc}Te recomendamos que no hagas eso en este momento porque {razon_vencido}"
                f"tu flujo neto disponible es de ${flujo:,.2f}/mes y tu capacidad máxima para una nueva cuota es de "
                f"${cap:,.2f}/mes. Asumir un pago de ${cuota:,.2f}/mes dejaría tu liquidez en ${flujo_post:,.2f}/mes, "
                f"elevando tu probabilidad de caer en mora al {ctx['prob_mora']*100:.0f}%. "
                f"En su lugar, programa una Meta de Ahorro de ${meta_mes:,.0f}/mes en MAESTRO_METAS para comprarlo al contado sin intereses."
            )
        elif sem == "AMARILLO":
            titulo = f"Crédito viable con precaución: Tu liquidez quedaría ajustada (${flujo_post:,.2f}/mes)."
            explicacion = (
                f"Sí te alcanza para cubrir la cuota de ${cuota:,.2f}/mes (tu límite máximo es ${cap:,.2f}/mes), "
                f"pero te quedarían solo ${flujo_post:,.2f} libres al mes para imprevistos. "
                f"Para tomar esta obligación sin riesgo de atraso, te sugerimos recortar ${recorte:,.0f}/mes en consumos "
                f"variables de restaurantes/apps en MVTO_CUENTAS o ahorrar durante 4 meses una entrada en MAESTRO_METAS para reducir la cuota."
            )
        else:
            titulo = f"Excelente salud financiera: Sí puedes asumir el crédito de ${monto:,.0f}."
            extra_poliza = (
                f" Además, como mantienes ${saldo:,.2f} en ahorros, puedes rentabilizar ${poliza:,.0f} en una Póliza al {tasa_poliza:.2f}% anual."
                if poliza >= 300
                else ""
            )
            explicacion = (
                f"Analicé tus cuentas de Banco Guayaquil y tu score Equifax de {eq}. No tienes saldos vencidos y "
                f"cuentas con ${flujo:,.2f} libres al mes (capacidad máxima de cuota de ${cap:,.2f}/mes). "
                f"La nueva cuota de ${cuota:,.2f}/mes encaja holgadamente dejándote ${flujo_post:,.2f}/mes de liquidez.{extra_poliza}"
            )

        return titulo, explicacion

    @staticmethod
    def _construir_acciones_ui(ctx: Dict[str, Any], recs: Dict[str, Any]) -> List[Dict[str, str]]:
        sem = ctx["semaforo"]
        intencion = ctx.get("intencion", "ENDEUDAMIENTO")
        poliza = ctx.get("monto_poliza_sugerido", 0.0)
        tasa_poliza = ctx.get("tasa_poliza_sugerida", 6.50)
        colchon = ctx.get("colchon_reserva_sugerido", 300.0)
        tiene_tc = ctx.get("tiene_deuda_cara_tc", False)
        acciones: List[Dict[str, str]] = []

        # Acciones prioritarias si la intencion es Inversion o Ahorro
        if intencion in {"INVERSION", "AHORRO"}:
            if tiene_tc:
                acciones.append({
                    "tipo": "PREPAGAR_TC",
                    "label": "Amortizar Diferidos Tarjeta (~16% anual)",
                    "payload_tabla": "MAESTRO_TC",
                })
            if poliza >= 300.0:
                acciones.append({
                    "tipo": "CREAR_POLIZA",
                    "label": f"Invertir ${poliza:,.0f} en Póliza ({tasa_poliza:.2f}% anual)",
                    "payload_tabla": "MAESTRO_POLIZAS",
                })
                acciones.append({
                    "tipo": "RESERVAR_COLCHON",
                    "label": f"Mantener Colchón Disponible (${colchon:,.0f})",
                    "payload_tabla": "MAESTRO_CUENTAS",
                })
            acciones.append({
                "tipo": "CREAR_META_AHORRO",
                "label": f"Programar Ahorro Metas (${ctx['ahorro_meta_sugerido']:,.0f}/mes)",
                "payload_tabla": "MAESTRO_METAS",
            })
            return acciones

        if sem == "ROJO":
            acciones.append(
                {
                    "tipo": "CREAR_META_AHORRO",
                    "label": f"Programar Ahorro Meta (${ctx['ahorro_meta_sugerido']:,.0f}/mes)",
                    "payload_tabla": "MAESTRO_METAS",
                }
            )
            if ctx["capacidad_max_cuota"] >= 35.0:
                monto_seguro = recs["1_oferta_maestro_creditos"]["monto_maximo_disponible"]
                acciones.append(
                    {
                        "tipo": "SIMULAR_MONTO_SEGURO",
                        "label": f"Recalcular Crédito Seguro (${monto_seguro:,.0f})",
                        "payload_tabla": "MAESTRO_CREDITOS",
                    }
                )
        elif sem == "AMARILLO":
            acciones.append(
                {
                    "tipo": "CREAR_META_ENTRADA",
                    "label": f"Ahorrar Entrada en Meta (${ctx['ahorro_meta_sugerido']:,.0f}/mes)",
                    "payload_tabla": "MAESTRO_METAS",
                }
            )
            acciones.append(
                {
                    "tipo": "CREAR_CREDITO_AJUSTADO",
                    "label": f"Tomar Crédito con Presupuesto (${ctx['monto_solicitado']:,.0f})",
                    "payload_tabla": "MAESTRO_CREDITOS",
                }
            )
        else:
            if recs["1_oferta_maestro_creditos"]["habilitado"] and ctx["monto_solicitado"] > 0:
                acciones.append(
                    {
                        "tipo": "CREAR_CREDITO",
                        "label": f"Desembolsar Crédito (${ctx['monto_solicitado']:,.0f})",
                        "payload_tabla": "MAESTRO_CREDITOS",
                    }
                )
            if recs["2_oferta_maestro_polizas"]["habilitado"]:
                acciones.append(
                    {
                        "tipo": "CREAR_POLIZA",
                        "label": f"Invertir ${recs['2_oferta_maestro_polizas']['MONTO_sugerido']:,.0f} en Póliza (6.5%)",
                        "payload_tabla": "MAESTRO_POLIZAS",
                    }
                )
            acciones.append(
                {
                    "tipo": "AUMENTAR_META",
                    "label": f"Activar Meta Ahorro (${ctx['ahorro_meta_sugerido']:,.0f}/mes)",
                    "payload_tabla": "MAESTRO_METAS",
                }
            )
        return acciones

    # ------------------------------------------------------------------------
    # METODO PRINCIPAL DE EXPOSICION AL BACKEND
    # ------------------------------------------------------------------------
    def evaluar_peticion_cliente(
        self,
        peticion_usuario: str,
        datos_cliente: Dict[str, Any],
        documento_adjunto: Optional[Union[str, Path, bytes, BinaryIO]] = None,
        nombre_archivo: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Punto de entrada unico para el equipo de Backend:
        1. Evalua Guardrails de seguridad, dominio y saludos (Filtro < 1ms con 0 tokens).
        2. Si es valida: Subpaso A (NLU) -> Subpaso B (LightGBM) -> RAG Regulatorio -> Subpaso C (XAI).
        """
        cod_clte = datos_cliente.get(
            "COD_CLTE",
            datos_cliente.get("maestro_clientes", {}).get("COD_CLTE", 10045892),
        )
        transaccion_id = datos_cliente.get("transaccion_id", f"req_bg_{random.randint(1000, 9999)}")

        # --------------------------------------------------------------------
        # PROCESAMIENTO OPCIONAL DE DOCUMENTO ADJUNTO (AZURE DOCUMENT INTELLIGENCE)
        # --------------------------------------------------------------------
        doc_result_dto = None
        if documento_adjunto and self.doc_engine is not None:
            logger.info("Procesando documento adjunto con Azure Document Intelligence prebuilt-layout...")
            doc_result_dto = self.doc_engine.analizar_documento(
                documento_input=documento_adjunto,
                nombre_archivo=nombre_archivo,
            )
            # Inyectar resultados de Document Intelligence en datos_cliente
            ext = datos_cliente.setdefault("fuentes_externas_ia", {})
            ocr_dict = ext.setdefault("document_intelligence_ocr", {})
            ocr_dict["documento_subido"] = True
            ocr_dict["tipo_documento"] = doc_result_dto.tipo_documento
            ocr_dict["entidad_emisora"] = doc_result_dto.entidad_emisora
            ocr_dict["titular_detectado"] = doc_result_dto.titular_detectado
            ocr_dict["identificacion_detectada"] = doc_result_dto.identificacion_detectada
            ocr_dict["ingresos_adicionales_detectados"] = doc_result_dto.INGRESOS_EXTRA_OCR
            ocr_dict["gastos_ocultos_detectados"] = doc_result_dto.GASTOS_OCULTOS_OCR
            ocr_dict["hallazgos_clave"] = doc_result_dto.hallazgos_clave
            ocr_dict["detalle_obligaciones_detectadas"] = doc_result_dto.detalle_obligaciones_detectadas
            ocr_dict["motor_extraccion"] = doc_result_dto.motor_extraccion

        # --------------------------------------------------------------------
        # PASO 0: GUARDRAILS DE SEGURIDAD, FILTRO DE DOMINIO Y SALUDOS SIMPLES
        # --------------------------------------------------------------------
        idioma_usuario = "es"
        if GuardrailsEngine is not None:
            check_guardrail = GuardrailsEngine.evaluar_peticion(peticion_usuario, cod_clte=int(cod_clte))
            idioma_usuario = check_guardrail.idioma_detectado

            # Si es saludo, despedida, pregunta absurda o caso sensible, devuelve respuesta inmediata sin costo de LLM
            if not check_guardrail.es_valido_para_pipeline_financiero:
                logger.info(f"Guardrail activado ({check_guardrail.tipo_evento}). Respondiendo de inmediato (0 tokens).")
                resp = check_guardrail.respuesta_inmediata or {}
                resp["transaccion_id"] = transaccion_id
                resp["peticion_usuario_original"] = peticion_usuario
                return resp

        # --------------------------------------------------------------------
        # SUBPASO A: NLU FINANCIERO (Azure AI Foundry gpt-4o / Fallback)
        # --------------------------------------------------------------------
        nlu_dto = self.subpaso_a_extraer_nlu(peticion_usuario)

        # --------------------------------------------------------------------
        # SUBPASO B: INFERENCIA ML LIGHTGBM + CAPACIDAD + SEMAFORO
        # --------------------------------------------------------------------
        res_b = self.subpaso_b_inferencia_ml_y_semaforo(datos_cliente, nlu_dto)

        # --------------------------------------------------------------------
        # RAG REGULATORIO ECUATORIANO (JPRF, Superintendencia de Bancos, COMFyF)
        # --------------------------------------------------------------------
        fundamento_legal = None
        if RAGRegulatorioEcuador is not None:
            ctx = res_b["_contexto_compacto_llm"]
            sem = res_b["evaluacion_riesgo_ml"]["semaforo_salud_financiera"]
            prob = res_b["evaluacion_riesgo_ml"]["probabilidad_caer_en_vencido_mora"]
            fundamento_legal = RAGRegulatorioEcuador.buscar_fundamentacion_legal(
                semaforo=sem,
                probabilidad_mora=prob,
                capital_vencido=float(ctx.get("capital_vencido", 0.0)),
                cuota_solicitada=float(ctx.get("cuota_nueva", 0.0)),
                capacidad_max_cuota=float(ctx.get("capacidad_max_cuota", 0.0)),
                intencion=nlu_dto.intencion,
            )
            res_b["_contexto_compacto_llm"]["fundamento_legal_resumen"] = fundamento_legal["sintesis_legal_usuario"]
        
        if doc_result_dto is not None:
            res_b["_contexto_compacto_llm"]["documento_ocr"] = {
                "tipo": doc_result_dto.tipo_documento,
                "entidad": doc_result_dto.entidad_emisora,
                "gastos_ocultos": doc_result_dto.GASTOS_OCULTOS_OCR,
                "ingresos_extra": doc_result_dto.INGRESOS_EXTRA_OCR,
                "hallazgos": doc_result_dto.hallazgos_clave,
            }

        # --------------------------------------------------------------------
        # SUBPASO C: EXPLICACION XAI CON FOUNDRY (gpt-4o) + MULTILINGÜE
        # --------------------------------------------------------------------
        agente_foundry = self.subpaso_c_generar_respuesta_foundry(peticion_usuario, res_b)

        # Si el idioma no es español y estamos en modo fallback, traducir titulo y explicacion básica
        if idioma_usuario == "en" and agente_foundry["origen_generacion"] != "AZURE_AI_FOUNDRY_GPT4O_MINI":
            agente_foundry["titulo_pantalla"] = f"Financial Health Assessment: Status {res_b['evaluacion_riesgo_ml']['semaforo_salud_financiera']}"
            agente_foundry["explicacion_ia"] = (
                f"We analyzed your Banco Guayaquil accounts and credit score. Safe monthly installment capacity is "
                f"${res_b['evaluacion_riesgo_ml']['capacidad_maxima_nueva_cuota']:,.2f}/mo. "
                f"{fundamento_legal['sintesis_legal_usuario'] if fundamento_legal else ''}"
            )

        return {
            "transaccion_id": transaccion_id,
            "COD_CLTE": int(cod_clte),
            "idioma_detectado": idioma_usuario,
            "peticion_usuario_original": peticion_usuario,
            "subpaso_a_nlu": nlu_dto.model_dump(),
            "evaluacion_riesgo_ml": res_b["evaluacion_riesgo_ml"],
            "dictamen_endeudamiento": res_b["dictamen_endeudamiento"],
            "recomendaciones_productos_bg": res_b["recomendaciones_productos_bg"],
            "document_intelligence_ocr": doc_result_dto.model_dump() if doc_result_dto else datos_cliente.get("fuentes_externas_ia", {}).get("document_intelligence_ocr", None),
            "fundamentacion_legal_ecuador": fundamento_legal,
            "agente_conversacional_foundry": agente_foundry,
        }


# ============================================================================
# 5. SUITE DE PRUEBAS EN CONSOLA CON 3 CASOS REALES (VERDE, AMARILLO, ROJO)
# ============================================================================
def ejecutar_pruebas_3_escenarios() -> None:
    motor = MotorIAFoundry()

    # Base del cliente ejemplo (inspirado en req_bg_9901)
    with open(Path(__file__).resolve().parent / "data" / "samples" / "sample_input.json", "r", encoding="utf-8") as f:
        cliente_base = json.load(f)

    # -------------------------------------------------------------------------
    # CASO 1: SEMAFORO VERDE (Cliente solvente pide compra moderada o inversion)
    # -------------------------------------------------------------------------
    cliente_verde = json.loads(json.dumps(cliente_base))
    cliente_verde["transaccion_id"] = "req_bg_caso_verde_01"
    cliente_verde["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_ingresos_credito_mes"] = 2400.00
    cliente_verde["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_egresos_debito_mes"] = 1050.00
    cliente_verde["resumen_cuentas_y_movimientos"]["cuentas_activas"][0]["SALDO"] = 1850.00
    cliente_verde["fuentes_externas_ia"]["buro_equifax"]["score_equifax"] = 845
    peticion_verde = "Quiero comprar una laptop de $1,200 a 18 meses para mi trabajo, ¿me alcanza y qué me recomiendan hacer con mis ahorros?"

    # -------------------------------------------------------------------------
    # CASO 2: SEMAFORO AMARILLO (Le alcanza con lo justo, requiere ajustar gastos)
    # -------------------------------------------------------------------------
    cliente_amarillo = json.loads(json.dumps(cliente_base))
    cliente_amarillo["transaccion_id"] = "req_bg_caso_amarillo_02"
    # Con el perfil original ($1,450 sueldo + $200 OCR = $1,650, flujo libre = $322.50, capacidad max cuota = $129.00):
    # Una TV de $1,200 a 12 meses al 15.5% genera una cuota de $108.58/mes (consume el 84% de su capacidad maxima y deja flujo libre < $225)
    peticion_amarillo = "Quiero comprar una TV de $1,200 a 12 meses, ¿me alcanza?"

    # -------------------------------------------------------------------------
    # CASO 3: SEMAFORO ROJO (Sobreendeudamiento / Cuota supera capacidad maxima)
    # -------------------------------------------------------------------------
    cliente_rojo = json.loads(json.dumps(cliente_base))
    cliente_rojo["transaccion_id"] = "req_bg_caso_rojo_03"
    cliente_rojo["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_ingresos_credito_mes"] = 950.00
    cliente_rojo["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_egresos_debito_mes"] = 380.00
    cliente_rojo["resumen_cuentas_y_movimientos"]["cuentas_activas"][0]["SALDO"] = 110.00
    cliente_rojo["fuentes_externas_ia"]["document_intelligence_ocr"]["ingresos_adicionales_detectados"] = 0.00
    cliente_rojo["fuentes_externas_ia"]["buro_equifax"]["score_equifax"] = 645
    peticion_rojo = "¿Puedo pagar una cuota de $260 al mes a 24 meses para un préstamo personal?"

    casos = [
        ("CASO 1 - ESCENARIO VERDE (Saludable / Holgado)", peticion_verde, cliente_verde),
        ("CASO 2 - ESCENARIO AMARILLO (Ajustado / Requiere Recorte o Meta)", peticion_amarillo, cliente_amarillo),
        ("CASO 3 - ESCENARIO ROJO (Peligro de Mora / Sobreendeudamiento)", peticion_rojo, cliente_rojo),
    ]

    print("\n" + "=" * 95)
    print("PRUEBA END-TO-END DEL MOTOR HIBRIDO (LIGHTGBM + AZURE AI FOUNDRY GPT-4O-MINI)")
    print("=" * 95)

    for titulo_caso, peticion, datos in casos:
        print(f"\n>>> {titulo_caso}")
        print(f"Peticion Usuario: \"{peticion}\"")
        resultado = motor.evaluar_peticion_cliente(peticion_usuario=peticion, datos_cliente=datos)
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
        print("-" * 95)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ejecutar_pruebas_3_escenarios()
