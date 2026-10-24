"""
Prueba de Validación de Asesoría Patrimonial Inteligente y Recomendaciones por Tiers
Banco Guayaquil - Hackathon AI Banking
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from motor_ia_foundry import MotorIAFoundry

def run_tests():
    motor = MotorIAFoundry()
    sample_path = Path("data/samples/sample_input.json")
    with open(sample_path, "r", encoding="utf-8") as f:
        cliente_base = json.load(f)

    # ------------------------------------------------------------------------
    # CASO 1: TIER ALTO (Mucho dinero en cuenta: $8,500)
    # ------------------------------------------------------------------------
    c1 = json.loads(json.dumps(cliente_base))
    c1["transaccion_id"] = "req_tier_alto_01"
    c1["resumen_cuentas_y_movimientos"]["cuentas_activas"][0]["SALDO"] = 8500.00
    c1["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_ingresos_credito_mes"] = 3200.00
    c1["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_egresos_debito_mes"] = 1200.00
    c1["resumen_tarjetas_tc"]["agregado_mvto_tc_mensual"]["saldo_diferido_estimado_cuotas"] = 0.00
    peticion_1 = "Tengo mucho dinero en el banco, ¿qué puedo hacer con mi dinero?"

    # ------------------------------------------------------------------------
    # CASO 2: TIER MEDIO (Dinero moderado: $1,200)
    # ------------------------------------------------------------------------
    c2 = json.loads(json.dumps(cliente_base))
    c2["transaccion_id"] = "req_tier_medio_02"
    c2["resumen_cuentas_y_movimientos"]["cuentas_activas"][0]["SALDO"] = 1200.00
    c2["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_ingresos_credito_mes"] = 1450.00
    c2["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_egresos_debito_mes"] = 920.00
    c2["resumen_tarjetas_tc"]["agregado_mvto_tc_mensual"]["saldo_diferido_estimado_cuotas"] = 0.00
    peticion_2 = "¿Qué me recomiendan hacer con mi dinero? Tengo unos $1,200 ahorrados."

    # ------------------------------------------------------------------------
    # CASO 3: TIER BÁSICO (Poco dinero: $150)
    # ------------------------------------------------------------------------
    c3 = json.loads(json.dumps(cliente_base))
    c3["transaccion_id"] = "req_tier_basico_03"
    c3["resumen_cuentas_y_movimientos"]["cuentas_activas"][0]["SALDO"] = 150.00
    c3["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_ingresos_credito_mes"] = 650.00
    c3["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_egresos_debito_mes"] = 480.00
    c3["resumen_tarjetas_tc"]["agregado_mvto_tc_mensual"]["saldo_diferido_estimado_cuotas"] = 0.00
    peticion_3 = "Tengo poco dinero guardado, ¿dónde lo pongo para ahorrar o qué puedo hacer?"

    # ------------------------------------------------------------------------
    # CASO 4: TIER ALTO CON DEUDA CARA EN TARJETA DE CRÉDITO
    # ------------------------------------------------------------------------
    c4 = json.loads(json.dumps(cliente_base))
    c4["transaccion_id"] = "req_tier_alto_con_tc_04"
    c4["resumen_cuentas_y_movimientos"]["cuentas_activas"][0]["SALDO"] = 5000.00
    c4["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_ingresos_credito_mes"] = 2800.00
    c4["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_egresos_debito_mes"] = 1100.00
    c4["resumen_tarjetas_tc"]["agregado_mvto_tc_mensual"]["saldo_diferido_estimado_cuotas"] = 320.00  # cuota diferidos alta
    peticion_4 = "¿Qué puedo hacer con mi dinero para que me rinda más?"

    casos = [
        ("CASO 1: TIER ALTO (Mucho dinero: $8,500 en cuenta)", peticion_1, c1),
        ("CASO 2: TIER MEDIO (Dinero moderado: $1,200 en cuenta)", peticion_2, c2),
        ("CASO 3: TIER BÁSICO (Poco dinero: $150 en cuenta)", peticion_3, c3),
        ("CASO 4: TIER ALTO CON DEUDA DIFERIDA EN TARJETA", peticion_4, c4),
    ]

    print("=" * 95)
    print("VALIDACION DE MOTOR DE ASESORIA PATRIMONIAL Y TIERS DE LIQUIDEZ")
    print("=" * 95)

    for nombre, peticion, datos in casos:
        print(f"\n>>> {nombre}")
        print(f"Peticion: \"{peticion}\"")
        res = motor.evaluar_peticion_cliente(peticion_usuario=peticion, datos_cliente=datos)
        
        nlu = res["subpaso_a_nlu"]
        riesgo = res["evaluacion_riesgo_ml"]
        asesoria = res["recomendaciones_productos_bg"].get("asesoria_patrimonial_inteligente", {})
        agente = res["agente_conversacional_foundry"]

        print(f"  - NLU Intencion: {nlu['intencion']}")
        print(f"  - Tier Patrimonial: {asesoria.get('tier_patrimonial')}")
        print(f"  - Saldo en Cuenta: ${asesoria.get('saldo_cuentas_actual', 0.0):,.2f}")
        print(f"  - Colchón Reserva Sugerido: ${asesoria.get('colchon_reserva_sugerido', 0.0):,.2f}")
        print(f"  - Póliza Sugerida: ${res['recomendaciones_productos_bg']['2_oferta_maestro_polizas']['MONTO_sugerido']:,.2f} al {res['recomendaciones_productos_bg']['2_oferta_maestro_polizas']['TASA_anual']}%")
        print(f"  - Meta Sugerida: ${res['recomendaciones_productos_bg']['3_oferta_maestro_metas']['MONTO_DEBITO_sugerido']:,.2f}/mes")
        print(f"  - Origen XAI: {agente.get('origen_generacion')}")
        print(f"  - Titulo UI: {agente.get('titulo_pantalla')}")
        print(f"  - Explicacion IA: {agente.get('explicacion_ia')}")
        print(f"  - Acciones UI: {[a['label'] for a in agente.get('acciones_ui', [])]}")
        print("-" * 95)

if __name__ == "__main__":
    run_tests()

