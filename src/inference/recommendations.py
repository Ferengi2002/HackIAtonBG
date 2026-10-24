"""
Motor de Recomendacion de Productos de Banco Guayaquil
Genera ofertas hiper-personalizadas para MAESTRO_CREDITOS, MAESTRO_POLIZAS y MAESTRO_METAS.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


class RecommendationEngine:
    """Genera recomendaciones accionables alineadas a las tablas de Banco Guayaquil."""

    @staticmethod
    def generate(payload: Dict[str, Any], evaluation: Dict[str, Any], features_df: pd.DataFrame) -> Dict[str, Any]:
        f = features_df.iloc[0]
        dictamen = evaluation["dictamen_endeudamiento"]
        sim = payload.get("simulacion_solicitada_usuario", {})

        # 1. Oferta Credito
        monto_sol = float(sim.get("monto_solicitado", 2000.0))
        oferta_credito = {
            "habilitado": bool(dictamen["se_puede_endeudar"]),
            "CODGRUOPE_sugerido": "CON",
            "MONTO_aprobado": float(monto_sol if dictamen["se_puede_endeudar"] else 0.0),
            "TASA_anual_preferencial": 14.20,
            "plazo_meses": int(sim.get("plazo_meses", 24)),
            "cuota_mensual": float(dictamen["cuota_mensual_segura"]),
            "CANAL_VENTA": "APP_AI_BANKING",
        }

        # 2. Oferta Poliza de Inversion
        saldo = float(f["SALDO_CUENTAS"])
        monto_poliza_sug = float(round(max(0.0, saldo - 240.0), -1) if saldo >= 500 else 0.0)
        tasa_poliza = 6.50
        interes_ganar = float(round(monto_poliza_sug * (tasa_poliza / 100.0) * (180.0 / 360.0), 2))
        oferta_poliza = {
            "habilitado": bool(monto_poliza_sug >= 300.0),
            "motivo": f"Tienes un SALDO de ${saldo:,.2f} en tu cuenta de ahorros que no genera alto rendimiento.",
            "MONTO_sugerido": float(monto_poliza_sug),
            "PLAZO_DIAS": 180,
            "TASA_anual": float(tasa_poliza),
            "interes_estimado_ganar": float(interes_ganar),
        }

        # 3. Oferta Metas de Ahorro
        oferta_metas = {
            "habilitado": True,
            "motivo": "Recortando $60 de gastos prescindibles en compras y apps de MVTO_CUENTAS, puedes acelerar tu meta.",
            "FRECUENCIA": "M",
            "MONTO_DEBITO_sugerido": 80.00,
            "META_objetivo": 1500.00,
            "DIA_DEBITO": 15,
        }

        return {
            "1_oferta_maestro_creditos": oferta_credito,
            "2_oferta_maestro_polizas": oferta_poliza,
            "3_oferta_maestro_metas": oferta_metas,
        }
