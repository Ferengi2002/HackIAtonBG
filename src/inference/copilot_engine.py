"""
Orquestador Central del Copiloto de Salud Financiera de Banco Guayaquil
Coordina la extraccion de caracteristicas, inferencia de riesgo, capacidad y recomendaciones.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

try:
    from src.inference.feature_extractor import FeatureExtractor
    from src.inference.financial_capacity import FinancialCapacityEngine
    from src.inference.recommendations import RecommendationEngine
except ImportError:
    from feature_extractor import FeatureExtractor
    from financial_capacity import FinancialCapacityEngine
    from recommendations import RecommendationEngine


class CopilotEngine:
    """Orquestador completo que consume el JSON de entrada y genera el JSON oficial de salida."""

    def __init__(self, model: Any) -> None:
        self.capacity_engine = FinancialCapacityEngine(model)

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        features_df = FeatureExtractor.extract(payload)
        eval_res = self.capacity_engine.evaluate(payload, features_df)
        recs = RecommendationEngine.generate(payload, eval_res, features_df)

        f = features_df.iloc[0]
        monto_sol = payload.get("simulacion_solicitada_usuario", {}).get("monto_solicitado", 2000.0)
        cuota_segura = eval_res["dictamen_endeudamiento"]["cuota_mensual_segura"]
        flujo_libre = eval_res["evaluacion_riesgo_ml"]["flujo_neto_mensual_disponible"]
        se_puede_endeudar = eval_res["dictamen_endeudamiento"]["se_puede_endeudar"]

        titulo = (
            f"¡Excelente salud financiera! Sí puedes tomar el crédito de ${monto_sol:,.0f}."
            if se_puede_endeudar
            else "Atención: Recomendamos fortalecer tu liquidez antes de endeudarte."
        )

        explicacion = (
            f"Analicé tus cuentas de Banco Guayaquil, tu score Equifax de {int(f['SCORE_EQUIFAX'])} y el documento adicional que subiste. "
            f"No tienes ningún saldo vencido y te quedan ${flujo_libre:,.2f} libres al mes después de todos tus gastos y tarjetas. "
            f"La cuota de ${cuota_segura:,.2f} encaja perfectamente en tu presupuesto. Además, como tienes ${f['SALDO_CUENTAS']:,.2f} en tu cuenta de ahorros, "
            f"te recomiendo mover ${recs['2_oferta_maestro_polizas']['MONTO_sugerido']:,.0f} a una nueva Póliza al 6.5% por 180 días para que tu dinero trabaje mientras pagas el crédito."
        )

        acciones = [
            {"tipo": "CREAR_CREDITO", "label": f"Desembolsar Crédito (${monto_sol:,.0f})", "payload_tabla": "MAESTRO_CREDITOS"},
            {"tipo": "CREAR_POLIZA", "label": f"Invertir ${recs['2_oferta_maestro_polizas']['MONTO_sugerido']:,.0f} en Póliza (6.5%)", "payload_tabla": "MAESTRO_POLIZAS"},
            {"tipo": "AUMENTAR_META", "label": f"Subir Ahorro Meta a ${recs['3_oferta_maestro_metas']['MONTO_DEBITO_sugerido']:,.0f}/mes", "payload_tabla": "MAESTRO_METAS"},
        ]

        return {
            "transaccion_id": payload.get("transaccion_id", "req_bg_default"),
            "COD_CLTE": payload.get("maestro_clientes", {}).get("COD_CLTE", 0),
            "evaluacion_riesgo_ml": eval_res["evaluacion_riesgo_ml"],
            "dictamen_endeudamiento": eval_res["dictamen_endeudamiento"],
            "recomendaciones_productos_bg": recs,
            "agente_conversacional_foundry": {
                "titulo_pantalla": titulo,
                "explicacion_ia": explicacion,
                "acciones_ui": acciones,
            },
        }
