"""
Motor de Capacidad Financiera, Riesgo y Semaforo de Salud
Aplica las reglas de negocio y limites de endeudamiento seguro de Banco Guayaquil.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


class FinancialCapacityEngine:
    """Calcula el dictamen de credito, semaforo y limites de endeudamiento seguro."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def evaluate(self, payload: Dict[str, Any], features_df: pd.DataFrame) -> Dict[str, Any]:
        prob_mora = float(self.model.predict_proba(features_df)[0, 1])
        f = features_df.iloc[0]

        ingresos_totales = f["INGRESOS_CREDITO_MES"] + f["INGRESOS_EXTRA_OCR"]
        cuotas_actuales = f["CUOTA_CREDITOS_VIGENTES"] + f["CUOTA_DIFERIDOS_TC"] + f["GASTOS_OCULTOS_OCR"]
        dti_actual_pct = round((cuotas_actuales / max(1.0, ingresos_totales)) * 100.0, 1)

        sim = payload.get("simulacion_solicitada_usuario", {})
        monto_solicitado = float(sim.get("monto_solicitado", 2000.0))
        plazo_meses = int(sim.get("plazo_meses", 24))
        tasa_preferencial = 14.20
        r_m = (tasa_preferencial / 100.0) / 12.0
        cuota_simulada = round(
            monto_solicitado * (r_m * ((1 + r_m) ** plazo_meses)) / (((1 + r_m) ** plazo_meses) - 1), 2
        )

        cuotas_con_sim = cuotas_actuales + cuota_simulada
        dti_con_sim_pct = round((cuotas_con_sim / max(1.0, ingresos_totales)) * 100.0, 1)

        # Politica Banco Guayaquil: DTI maximo seguro 40% y maximo 40% del flujo libre
        capacidad_max_cuota = max(
            0.0,
            round(min((ingresos_totales * 0.40) - cuotas_actuales, f["FLUJO_NETO_DISPONIBLE"] * 0.40), 2),
        )

        score_salud = max(10, min(99, int(round((1.0 - prob_mora) * 70.0 + (f["SCORE_EQUIFAX"] / 999.0) * 30.0))))

        # Semaforo de Riesgo
        if prob_mora < 0.15 and dti_con_sim_pct <= 40.0 and f["CAPITAL_VENCIDO"] == 0:
            semaforo = "VERDE"
            se_puede_endeudar = True
            resolucion = "APROBADO_OPTIMO"
        elif prob_mora < 0.35 and dti_con_sim_pct <= 50.0 and f["CAPITAL_VENCIDO"] == 0:
            semaforo = "AMARILLO"
            se_puede_endeudar = True
            resolucion = "APROBADO_CONDICIONADO"
        else:
            semaforo = "ROJO"
            se_puede_endeudar = False
            resolucion = "NO_RECOMENDADO"

        monto_max_aprobable = float(round(
            capacidad_max_cuota * (((1 + r_m) ** plazo_meses) - 1) / (r_m * ((1 + r_m) ** plazo_meses)), -1
        ))

        factores = []
        if f["CAPITAL_VENCIDO"] == 0:
            factores.append("Historial impecable en MAESTRO_CREDITOS (VENCIDO = $0.00).")
        else:
            factores.append(f"Alerta: Registra CAPITAL_VENCIDO de ${f['CAPITAL_VENCIDO']:,.2f}.")
        if f["MONTO_ACUM_METAS"] > 0:
            factores.append("Buen hábito de ahorro: tienes metas activas en MAESTRO_METAS en progreso.")
        if f["VIAJES_EXTERIOR_TC"] == 1 or f["SALDO_CUENTAS"] > 500:
            factores.append(f"Liquidez positiva en cuenta (${f['SALDO_CUENTAS']:,.2f}) y consumos que respaldan estabilidad.")

        return {
            "evaluacion_riesgo_ml": {
                "score_salud_interna": int(score_salud),
                "probabilidad_caer_en_vencido_mora": float(round(prob_mora, 2)),
                "semaforo_salud_financiera": str(semaforo),
                "flujo_neto_mensual_disponible": float(round(f["FLUJO_NETO_DISPONIBLE"], 2)),
                "capacidad_maxima_nueva_cuota": float(capacidad_max_cuota),
                "endeudamiento_actual_pct": float(dti_actual_pct),
                "endeudamiento_con_simulacion_pct": float(dti_con_sim_pct),
            },
            "dictamen_endeudamiento": {
                "se_puede_endeudar": bool(se_puede_endeudar),
                "estado_resolucion": str(resolucion),
                "monto_maximo_recomendado": float(monto_max_aprobable),
                "cuota_mensual_segura": float(cuota_simulada if se_puede_endeudar else capacidad_max_cuota),
                "impacto_proyectado_equifax": f"Manteniendo VENCIDO en $0.00 durante 6 meses, tu score pasará de {int(f['SCORE_EQUIFAX'])} a ~{min(999, int(f['SCORE_EQUIFAX']) + 30)}.",
                "factores_clave_modelo": factores,
            },
        }
