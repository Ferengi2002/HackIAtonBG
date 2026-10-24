"""
Suite de Pruebas: Detección de Sobreendeudamiento Crítico, Alerta Anti-Deuda y Derivación a Asesor
Banco Guayaquil - Hackathon AI Banking
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from motor_ia_foundry import MotorIAFoundry

def run_distress_tests():
    motor = MotorIAFoundry()
    sample_path = Path("data/samples/sample_input.json")
    with open(sample_path, "r", encoding="utf-8") as f:
        cliente_base = json.load(f)

    # ------------------------------------------------------------------------
    # CASO 1: USUARIO DESESPERADO POR DEUDAS ("NO PUEDO MÁS")
    # ------------------------------------------------------------------------
    c1 = json.loads(json.dumps(cliente_base))
    c1["transaccion_id"] = "req_distress_01"
    c1["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_ingresos_credito_mes"] = 1100.00
    c1["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_egresos_debito_mes"] = 980.00
    c1["resumen_tarjetas_tc"]["agregado_mvto_tc_mensual"]["saldo_diferido_estimado_cuotas"] = 180.00
    peticion_1 = "Ando sobre endeudado y no puedo más, ¿qué puedo hacer con mis deudas?"

    # ------------------------------------------------------------------------
    # CASO 2: PEDIR MÁS DEUDA PARA PAGAR OTRAS DEUDAS (BOLA DE NIEVE)
    # ------------------------------------------------------------------------
    c2 = json.loads(json.dumps(cliente_base))
    c2["transaccion_id"] = "req_distress_02"
    c2["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_ingresos_credito_mes"] = 1200.00
    c2["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_egresos_debito_mes"] = 950.00
    c2["resumen_tarjetas_tc"]["agregado_mvto_tc_mensual"]["saldo_diferido_estimado_cuotas"] = 280.00
    peticion_2 = "Necesito un préstamo de $2,000 para pagar mis otras tarjetas porque ya no aguanto las deudas y estoy al límite."

    # ------------------------------------------------------------------------
    # CASO 3: USUARIO CON CAPITAL VENCIDO EN MORA ($340 USD)
    # ------------------------------------------------------------------------
    c3 = json.loads(json.dumps(cliente_base))
    c3["transaccion_id"] = "req_distress_03"
    c3["resumen_creditos_vigentes"]["total_vencido_mora"] = 340.00
    c3["resumen_creditos_vigentes"]["operaciones_activas"][0]["VENCIDO"] = 340.00
    peticion_3 = "¿Me pueden aprobar un préstamo de $1,000 para comprarme una laptop a 12 meses?"

    # ------------------------------------------------------------------------
    # CASO 4: USUARIO CON FLUJO MENSUAL DEFICITARIO (Gasta más de lo que gana)
    # ------------------------------------------------------------------------
    c4 = json.loads(json.dumps(cliente_base))
    c4["transaccion_id"] = "req_distress_04"
    c4["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_ingresos_credito_mes"] = 800.00
    c4["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_egresos_debito_mes"] = 860.00
    peticion_4 = "Quiero comprar una moto a crédito con cuota de $180 al mes a 24 meses."

    casos = [
        ("CASO 1: Desesperación Financiera Expresa ('No puedo más / Sobreendeudado')", peticion_1, c1),
        ("CASO 2: Efecto Bola de Nieve ('Préstamo para pagar otras deudas / Al límite')", peticion_2, c2),
        ("CASO 3: Mora Activa Preexistente (Capital Vencido = $340)", peticion_3, c3),
        ("CASO 4: Flujo Deficitario en Números Rojos (Egresos > Ingresos)", peticion_4, c4),
    ]

    print("=" * 95)
    print("VALIDACION DE ALERTA DE SOBREENDEUDAMIENTO CRÍTICO Y NEGOCIACIÓN CON ASESOR")
    print("=" * 95)

    for nombre, peticion, datos in casos:
        print(f"\n>>> {nombre}")
        print(f"Peticion: \"{peticion}\"")
        res = motor.evaluar_peticion_cliente(peticion_usuario=peticion, datos_cliente=datos)
        
        nlu = res["subpaso_a_nlu"]
        riesgo = res["evaluacion_riesgo_ml"]
        dictamen = res["dictamen_endeudamiento"]
        agente = res["agente_conversacional_foundry"]
        alivio = res["recomendaciones_productos_bg"].get("protocolo_alivio_financiero", {})
        legal = res.get("fundamentacion_legal_ecuador", {})

        print(f"  - NLU Intención: {nlu['intencion']} (Estrés: {nlu.get('en_estres_financiero', False)})")
        print(f"  - Semáforo: {riesgo['semaforo_salud_financiera']} | Mora Prob: {riesgo['probabilidad_caer_en_vencido_mora']*100:.0f}%")
        print(f"  - ¿Se puede endeudar?: {dictamen['se_puede_endeudar']} ({dictamen['estado_resolucion']})")
        print(f"  - Alerta Alivio Activa: {alivio.get('alerta_activa')} | Dictamen: {alivio.get('dictamen_deuda')}")
        print(f"  - Origen XAI: {agente.get('origen_generacion')}")
        print(f"  - Título Pantalla UI: {agente.get('titulo_pantalla')}")
        print(f"  - Explicación IA: {agente.get('explicacion_ia')}")
        print(f"  - Acciones UI Disponibles: {[a['label'] for a in agente.get('acciones_ui', [])]}")
        print(f"  - Respaldo Legal RAG: {legal.get('sintesis_legal_usuario')}")
        print("-" * 95)

if __name__ == "__main__":
    run_distress_tests()

