"""
Extractor de Caracteristicas desde el JSON de entrada de Banco Guayaquil
Mapea estrictamente a las 8 tablas de PostgreSQL + Fuentes Externas (OCR, Equifax, Redes)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import pandas as pd


class FeatureExtractor:
    """Extrae las 19 features tabulares requeridas por el modelo LightGBM."""

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

    @classmethod
    def extract(cls, payload: Dict[str, Any], ref_date: datetime | None = None) -> pd.DataFrame:
        if ref_date is None:
            ref_date = datetime(2026, 10, 24)

        # 1. MAESTRO_CLIENTES
        clte = payload.get("maestro_clientes", {})
        birth_str = clte.get("FECHA_NACIMIENTO", "1995-01-01")
        birth_dt = datetime.strptime(birth_str, "%Y-%m-%d")
        edad = max(18, int((ref_date - birth_dt).days / 365.25))

        vinc_str = clte.get("FECHA_VINCULACION", "2020-01-01")
        vinc_dt = datetime.strptime(vinc_str, "%Y-%m-%d")
        antiguedad = max(0.1, round((ref_date - vinc_dt).days / 365.25, 1))

        # 2. MAESTRO_CUENTAS & MVTO_CUENTAS
        cuentas_data = payload.get("resumen_cuentas_y_movimientos", {})
        cuentas_activas = cuentas_data.get("cuentas_activas", [])
        saldo_cuentas = sum(float(c.get("SALDO", 0.0)) for c in cuentas_activas)

        mvto_cuentas = cuentas_data.get("agregado_mvto_cuentas_mensual", {})
        ingresos_banco = float(mvto_cuentas.get("promedio_ingresos_credito_mes", 0.0))
        egresos_banco = float(mvto_cuentas.get("promedio_egresos_debito_mes", 0.0))

        # 3. MAESTRO_CREDITOS
        cred_data = payload.get("resumen_creditos_vigentes", {})
        cap_por_vencer = float(cred_data.get("total_por_vencer", 0.0))
        cap_vencido = float(cred_data.get("total_vencido_mora", 0.0))
        ops = cred_data.get("operaciones_activas", [])
        cuota_creditos = 0.0
        for op in ops:
            monto = float(op.get("MONTO", 0.0))
            por_vencer = float(op.get("POR_VENCER", 0.0))
            if por_vencer > 0:
                cuota_creditos += max(25.0, round(monto / 24.0, 2))

        # 4. MAESTRO_TC & MVTO_TC
        tc_data = payload.get("resumen_tarjetas_tc", {}).get("agregado_mvto_tc_mensual", {})
        gasto_tc = float(tc_data.get("gasto_promedio_mensual_tc", 0.0))
        cuota_diferidos_tc = float(tc_data.get("saldo_diferido_estimado_cuotas", 0.0))
        paises = tc_data.get("paises_consumo_reciente", [])
        viajes_tc = 1 if tc_data.get("viajes_internacionales_detectados", False) or any(p != "ECU" for p in paises) else 0

        # 5. MAESTRO_POLIZAS & MAESTRO_METAS
        inv_data = payload.get("inversiones_y_ahorro_actual", {})
        monto_polizas = sum(float(p.get("MONTO", 0.0)) for p in inv_data.get("maestro_polizas", []))
        metas = inv_data.get("maestro_metas", [])
        monto_metas = sum(float(m.get("MONTO_ACUM", 0.0)) for m in metas)
        cumplio_meta = 1 if any(
            int(m.get("CUMPLIO_META", 0)) == 1
            or (float(m.get("MONTO_ACUM", 0.0)) / max(1.0, float(m.get("META", 1.0)))) >= 0.25
            for m in metas
        ) else 0

        # 6. FUENTES EXTERNAS IA
        ext = payload.get("fuentes_externas_ia", {})
        score_equifax = int(ext.get("buro_equifax", {}).get("score_equifax", 650))
        cuota_otros_bancos = float(ext.get("buro_equifax", {}).get("cuota_mensual_otros_bancos", 0.0))

        ocr = ext.get("document_intelligence_ocr", {})
        ingresos_ocr = float(ocr.get("ingresos_adicionales_detectados", 0.0))
        gastos_ocultos_ocr = float(ocr.get("gastos_ocultos_detectados", 0.0))
        score_estilo_vida = float(ext.get("estilo_vida_redes_sociales", {}).get("score_estilo_vida", 5.0))

        # 7. FLUJO NETO DISPONIBLE
        ingresos_totales = ingresos_banco + ingresos_ocr
        # Egresos corrientes + cuotas diferidas TC + gastos ocultos OCR + cuota otros bancos + consumos corrientes TC
        egresos_totales = (
            egresos_banco
            + cuota_diferidos_tc
            + gastos_ocultos_ocr
            + cuota_otros_bancos
            + max(0.0, gasto_tc - cuota_diferidos_tc) * 0.50
        )
        flujo_neto = round(ingresos_totales - egresos_totales, 2)

        data = {
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
            "SCORE_EQUIFAX": score_equifax,
            "INGRESOS_EXTRA_OCR": ingresos_ocr,
            "GASTOS_OCULTOS_OCR": round(gastos_ocultos_ocr + cuota_otros_bancos, 2),
            "SCORE_ESTILO_VIDA_REDES": score_estilo_vida,
            "FLUJO_NETO_DISPONIBLE": flujo_neto,
        }
        return pd.DataFrame([data])[cls.FEATURE_COLS]
