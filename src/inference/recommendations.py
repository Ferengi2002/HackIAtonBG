"""
Motor de Recomendación de Productos y Asesoría Patrimonial de Banco Guayaquil
Genera ofertas hiper-personalizadas y asesoría patrimonial por tiers de liquidez:
- TIER ALTO (Patrimonial / Alta Liquidez): Póliza a Plazo Fijo (7.00% - 7.50%, COSEDE hasta $32,000).
- TIER MEDIO (Moderado / Balanceado): Estrategia Híbrida (Póliza Corto Plazo + Ahorro Programado en Metas).
- TIER BÁSICO (Crecimiento / Microahorro): Ahorro Flexible Programado (Fondo de Emergencia en Metas de Ahorro).
- DEUDA CARA PREEXISTENTE: Sugerencia de prepago de TC (~16% anual) para maximizar ahorro neto.
Ofrece valor mutuo ganar-ganar: rentabilidad y seguridad para el cliente, captación y menor mora para Banco Guayaquil.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple
import pandas as pd


class RecommendationEngine:
    """Genera recomendaciones accionables alineadas a las tablas de Banco Guayaquil."""

    # Parámetros oficiales de productos Banco Guayaquil
    COSEDE_MAX_PROTECCION_USD = 32000.00
    TASA_PREFERENCIAL_CREDITO = 14.20
    TASA_POLIZA_ALTA_LIQUIDEZ = 7.25  # Pólizas de $3,000+ a 180-360 días
    TASA_POLIZA_MEDIA_LIQUIDEZ = 6.50  # Pólizas de $500 - $3,000 a 90-180 días
    TASA_REFERENCIAL_TC_DIFERIDO = 16.06  # Tasa legal activa para crédito de consumo / diferidos

    @classmethod
    def determinar_tier_patrimonial(
        cls, saldo_cuentas: float, flujo_neto: float
    ) -> Literal["TIER_ALTO_PATRIMONIAL", "TIER_MEDIO_FLEXIBLE", "TIER_BASICO_CRECIMIENTO"]:
        """Clasifica al cliente según su patrimonio líquido disponible en Banco Guayaquil."""
        if saldo_cuentas >= 3000.0 or (saldo_cuentas >= 2000.0 and flujo_neto >= 600.0):
            return "TIER_ALTO_PATRIMONIAL"
        elif saldo_cuentas >= 500.0 or flujo_neto >= 250.0:
            return "TIER_MEDIO_FLEXIBLE"
        else:
            return "TIER_BASICO_CRECIMIENTO"

    @classmethod
    def generar_oferta_patrimonial_inteligente(
        cls,
        saldo_cuentas: float,
        flujo_neto: float,
        cuota_diferidos_tc: float = 0.0,
        semaforo: str = "VERDE",
    ) -> Dict[str, Any]:
        """
        Calcula la propuesta de captación e inversión personalizada según el tier patrimonial.
        Alinea beneficios para el cliente y para Banco Guayaquil.
        """
        tier = cls.determinar_tier_patrimonial(saldo_cuentas, flujo_neto)
        tiene_deuda_cara_tc = bool(cuota_diferidos_tc >= 60.0 and saldo_cuentas >= 600.0)

        # --------------------------------------------------------------------
        # 1. TIER ALTO: PÓLIZA DE INVERSIÓN A PLAZO FIJO (MAESTRO_POLIZAS)
        # --------------------------------------------------------------------
        if tier == "TIER_ALTO_PATRIMONIAL":
            # Reservar un fondo de reserva líquido mínimo de $600 - $1,000 en cuenta
            colchon_reserva = min(1200.0, max(600.0, round(saldo_cuentas * 0.20, -1)))
            monto_poliza = max(1000.0, round(saldo_cuentas - colchon_reserva, -2))
            plazo_dias = 180  # Semestral renovable
            tasa_poliza = cls.TASA_POLIZA_ALTA_LIQUIDEZ
            interes_estimado = round(monto_poliza * (tasa_poliza / 100.0) * (plazo_dias / 360.0), 2)
            meta_ahorro_complemento = round(max(80.0, min(250.0, flujo_neto * 0.25)), -1)

            estrategia = (
                f"Maximizar el rendimiento de tu capital excedente (${monto_poliza:,.0f}) en una Póliza a Plazo Fijo al "
                f"{tasa_poliza:.2f}% anual, manteniendo un fondo de emergencia de ${colchon_reserva:,.0f} en tu cuenta de ahorros."
            )
            motivo_poliza = (
                f"Cuentas con un saldo disponible de ${saldo_cuentas:,.2f}. Al colocar ${monto_poliza:,.0f} a {plazo_dias} días, "
                f"ganas ${interes_estimado:,.2f} garantizados con respaldo estatal de COSEDE hasta ${cls.COSEDE_MAX_PROTECCION_USD:,.0f}."
            )
            valor_cliente = (
                f"Ganas un rendimiento fijo del {tasa_poliza:.2f}% anual sin riesgo de mercado, bates la inflación y mantienes tu "
                f"capital 100% asegurado por COSEDE hasta ${cls.COSEDE_MAX_PROTECCION_USD:,.0f}."
            )
            valor_banco_guayaquil = (
                "Captación de depósitos a plazo fijo institucional/personal, fondeo estable para cartera crediticia, cumplimiento de "
                "ratios de liquidez estructural Basilea III y fidelización de clientes de alto valor."
            )

        # --------------------------------------------------------------------
        # 2. TIER MEDIO: ESTRATEGIA HÍBRIDA (PÓLIZA CORTO PLAZO + AHORRO FLEXIBLE)
        # --------------------------------------------------------------------
        elif tier == "TIER_MEDIO_FLEXIBLE":
            colchon_reserva = min(500.0, max(200.0, round(saldo_cuentas * 0.35, -1)))
            monto_poliza = max(300.0, round(saldo_cuentas - colchon_reserva, -1)) if saldo_cuentas >= 500.0 else 0.0
            plazo_dias = 90  # Trimestral para mayor flexibilidad
            tasa_poliza = cls.TASA_POLIZA_MEDIA_LIQUIDEZ
            interes_estimado = (
                round(monto_poliza * (tasa_poliza / 100.0) * (plazo_dias / 360.0), 2) if monto_poliza > 0 else 0.0
            )
            meta_ahorro_complemento = round(max(50.0, min(120.0, max(50.0, flujo_neto * 0.30))), -1)

            estrategia = (
                f"Estrategia balanceada: Rentabilizar ${monto_poliza:,.0f} a corto plazo ({plazo_dias} días al {tasa_poliza:.2f}%) "
                f"y programar un Ahorro Flexible de ${meta_ahorro_complemento:,.0f}/mes en Metas de Ahorro para cubrir gastos futuros."
                if monto_poliza >= 300.0
                else f"Fortalecer tu liquidez con Ahorro Programado en Metas de Ahorro de ${meta_ahorro_complemento:,.0f}/mes."
            )
            motivo_poliza = (
                f"Tienes ${saldo_cuentas:,.2f} en ahorros. Colocar ${monto_poliza:,.0f} al {tasa_poliza:.2f}% te genera intereses "
                f"sin inmovilizar todo tu dinero."
                if monto_poliza >= 300.0
                else "Con tu saldo actual, es preferible consolidar una meta flexible antes de inmovilizar a plazo fijo."
            )
            valor_cliente = (
                "Equilibrio óptimo: obtienes rendimiento sobre tus ahorros sin perder la flexibilidad de contar con dinero disponible "
                "para imprevistos mediante débito automático controlado."
            )
            valor_banco_guayaquil = (
                "Cross-selling inteligente (Póliza + Metas), retención de saldos a la vista, aumento del ahorro recurrente y "
                "vinculación transaccional periódica."
            )

        # --------------------------------------------------------------------
        # 3. TIER BÁSICO: CRECIMIENTO Y MICROAHORRO (MAESTRO_METAS)
        # --------------------------------------------------------------------
        else:
            colchon_reserva = saldo_cuentas
            monto_poliza = 0.0
            plazo_dias = 0
            tasa_poliza = 0.0
            interes_estimado = 0.0
            meta_ahorro_complemento = round(max(30.0, min(60.0, max(30.0, flujo_neto * 0.25 if flujo_neto > 0 else 30.0))), -1)

            estrategia = (
                f"Construir tu Fondo de Emergencia con Ahorro Flexible en Metas de Ahorro debitando ${meta_ahorro_complemento:,.0f}/mes "
                f"en quincena o fin de mes, sin comisiones ni penalidades."
            )
            motivo_poliza = "Tu prioridad hoy es armar un fondo de reserva antes de inmovilizar dinero a plazo fijo."
            valor_cliente = (
                "Construyes un colchón protector ante cualquier imprevisto sin pagar comisiones ni caer en mora o créditos costosos."
            )
            valor_banco_guayaquil = (
                "Disminuye la probabilidad de morosidad temprana en tarjetas y microcréditos, fomenta educación financiera y crea "
                "un cliente fiel con historial crediticio limpio."
            )

        # Sugerencia de Desendeudamiento Rentable si tiene deudas diferidas
        consejo_prepago_tc = None
        if tiene_deuda_cara_tc:
            prepago_sug = min(saldo_cuentas * 0.50, cuota_diferidos_tc * 2.0)
            consejo_prepago_tc = (
                f"Detectamos diferidos de tarjeta de crédito con tasa de ~{cls.TASA_REFERENCIAL_TC_DIFERIDO:.1f}% anual. "
                f"Te conviene destinar ${prepago_sug:,.0f} de tu saldo a amortizar esta deuda antes de invertir todo: "
                f"ahorrarte un {cls.TASA_REFERENCIAL_TC_DIFERIDO:.1f}% de interés genera un rendimiento superior a cualquier inversión conservadora."
            )

        return {
            "tier_patrimonial": tier,
            "saldo_cuentas_actual": float(saldo_cuentas),
            "flujo_neto_disponible": float(flujo_neto),
            "colchon_reserva_sugerido": float(colchon_reserva),
            "estrategia_principal": estrategia,
            "tiene_deuda_cara_tc": tiene_deuda_cara_tc,
            "consejo_prepago_tc": consejo_prepago_tc,
            "valor_mutuo": {
                "valor_para_cliente": valor_cliente,
                "valor_para_banco_guayaquil": valor_banco_guayaquil,
            },
            "oferta_poliza_optima": {
                "habilitado": bool(monto_poliza >= 300.0 and semaforo != "ROJO"),
                "motivo": motivo_poliza,
                "MONTO_sugerido": float(monto_poliza),
                "PLAZO_DIAS": int(plazo_dias),
                "TASA_anual": float(tasa_poliza),
                "interes_estimado_ganar": float(interes_estimado),
                "respaldo_cosede_monto": cls.COSEDE_MAX_PROTECCION_USD,
            },
            "oferta_meta_optima": {
                "habilitado": True,
                "motivo": f"Ahorra ${meta_ahorro_complemento:,.0f}/mes para asegurar liquidez o cumplir objetivos sin endeudarte.",
                "FRECUENCIA": "M",
                "MONTO_DEBITO_sugerido": float(meta_ahorro_complemento),
                "META_objetivo": float(max(600.0, meta_ahorro_complemento * 12.0)),
                "DIA_DEBITO": 15,
            },
        }

    @classmethod
    def generate(cls, payload: Dict[str, Any], evaluation: Dict[str, Any], features_df: pd.DataFrame) -> Dict[str, Any]:
        """Genera el bloque completo de recomendaciones incluyendo crédito, pólizas y metas."""
        f = features_df.iloc[0]
        dictamen = evaluation.get("dictamen_endeudamiento", {})
        riesgo = evaluation.get("evaluacion_riesgo_ml", {})
        sim = payload.get("simulacion_solicitada_usuario", {})
        semaforo = riesgo.get("semaforo_salud_financiera", "VERDE")

        saldo = float(f.get("SALDO_CUENTAS", 0.0))
        flujo = float(f.get("FLUJO_NETO_DISPONIBLE", 0.0))
        cuota_tc = float(f.get("CUOTA_DIFERIDOS_TC", 0.0))

        # 1. Asesoría Patrimonial Inteligente
        patrimonio = cls.generar_oferta_patrimonial_inteligente(
            saldo_cuentas=saldo,
            flujo_neto=flujo,
            cuota_diferidos_tc=cuota_tc,
            semaforo=semaforo,
        )

        # 2. Oferta Crédito
        monto_sol = float(sim.get("monto_solicitado", 2000.0))
        se_puede_endeudar = bool(dictamen.get("se_puede_endeudar", True))
        cuota_segura = float(dictamen.get("cuota_mensual_segura", 0.0))
        monto_aprobado = float(monto_sol if se_puede_endeudar else 0.0)

        oferta_credito = {
            "habilitado": se_puede_endeudar,
            "CODGRUOPE_sugerido": "CON",
            "MONTO_aprobado": monto_aprobado,
            "monto_maximo_disponible": float(dictamen.get("monto_maximo_recomendado", monto_aprobado)),
            "TASA_anual_preferencial": cls.TASA_PREFERENCIAL_CREDITO,
            "plazo_meses": int(sim.get("plazo_meses", 24)),
            "cuota_mensual": cuota_segura,
            "CANAL_VENTA": "APP_AI_BANKING",
        }

        # 3. Empaquetar ofertas alineadas con tablas de Banco Guayaquil
        protocolo_alivio = {
            "alerta_activa": bool(not se_puede_endeudar or semaforo == "ROJO"),
            "canales_contacto": {
                "telefono": "04 3730100 (opción 3 - Soluciones de Pago)",
                "whatsapp": "+593 98 373 0100",
            },
            "recomendacion": "NO adquirir nueva deuda. Contactar a un asesor de renegociación lo más pronto posible.",
        }

        return {
            "1_oferta_maestro_creditos": oferta_credito,
            "2_oferta_maestro_polizas": patrimonio["oferta_poliza_optima"],
            "3_oferta_maestro_metas": patrimonio["oferta_meta_optima"],
            "asesoria_patrimonial_inteligente": patrimonio,
            "protocolo_alivio_financiero": protocolo_alivio,
        }
