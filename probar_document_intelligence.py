"""
Demostracion End-to-End: Azure AI Document Intelligence (prebuilt-layout) + GPT-4o + LightGBM
Proyecto: Copiloto de Salud Financiera y Prevencion de Morosidad - Banco Guayaquil
"""

import json
import sys
from pathlib import Path

# Soporte UTF-8 en consola de Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from motor_ia_foundry import MotorIAFoundry


def ejecutar_demostracion_ocr():
    workspace_root = Path(__file__).resolve().parent
    sample_file = workspace_root / "data" / "samples" / "sample_input.json"
    pdf_path = workspace_root / "estado_cuenta_externo_demo.pdf"

    if not pdf_path.exists():
        pdf_path = workspace_root / "data" / "samples" / "estado_cuenta_externo_demo.pdf"

    with open(sample_file, "r", encoding="utf-8") as f:
        datos_cliente_base = json.load(f)

    # Limpiar OCR previo en el cliente base para ver el impacto real de 'antes vs despues'
    datos_cliente_base["fuentes_externas_ia"]["document_intelligence_ocr"] = {
        "documento_subido": False,
        "tipo_documento": "NINGUNO",
        "ingresos_adicionales_detectados": 0.0,
        "gastos_ocultos_detectados": 0.0,
    }

    peticion = "Quiero comprarme una moto de $2,500 a 24 meses para ir a mi trabajo, ¿me alcanza con mi sueldo actual?"

    print("=" * 95)
    print("DEMOSTRACION: IMPACTO DE AZURE DOCUMENT INTELLIGENCE EN EL MODELO ML DE RIESGO")
    print("=" * 95)
    print(f"CLIENTE:  {datos_cliente_base['maestro_clientes']['COD_CLTE']} (Score Equifax: {datos_cliente_base['fuentes_externas_ia']['buro_equifax']['score_equifax']})")
    print(f"PETICION: \"{peticion}\"")
    print("=" * 95)

    motor = MotorIAFoundry()

    # -------------------------------------------------------------------------
    # ESCENARIO 1: EVALUACION BASE (SIN DOCUMENTO ADJUNTO)
    # -------------------------------------------------------------------------
    print("\n>>> ESCENARIO 1: CLIENTE SIN SUBIR DOCUMENTO (Solo datos registrados en Banco Guayaquil)")
    res_sin_doc = motor.evaluar_peticion_cliente(
        peticion_usuario=peticion,
        datos_cliente=json.loads(json.dumps(datos_cliente_base)),
    )
    eval_1 = res_sin_doc["evaluacion_riesgo_ml"]
    nlu_1 = res_sin_doc["subpaso_a_nlu"]
    agente_1 = res_sin_doc["agente_conversacional_foundry"]

    print(f"   -> Cuota calculada moto:       ${nlu_1['cuota_mensual_nueva']:,.2f}/mes")
    print(f"   -> Flujo libre mensual:        ${eval_1['flujo_neto_mensual_disponible']:,.2f}/mes")
    print(f"   -> Capacidad maxima cuota:     ${eval_1['capacidad_maxima_nueva_cuota']:,.2f}/mes")
    print(f"   -> Flujo post-simulacion:      ${eval_1['flujo_neto_post_simulacion']:,.2f}/mes")
    print(f"   -> Semaforo ML resultante:     {eval_1['semaforo_salud_financiera']}")
    print(f"   -> Probabilidad de mora ML:    {eval_1['probabilidad_caer_en_vencido_mora']*100:.1f}%")
    print(f"   -> Dictamen:                   {res_sin_doc['dictamen_endeudamiento']['estado_resolucion']}")

    # -------------------------------------------------------------------------
    # ESCENARIO 2: EVALUACION CON ESTADO DE CUENTA EXTERNO (AZURE DOCUMENT INTELLIGENCE)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 95)
    print(f">>> ESCENARIO 2: CLIENTE ADJUNTA DOCUMENTO ({pdf_path.name})")
    print("    [Procesando con Azure Document Intelligence 'prebuilt-layout' + gpt-4o]...")

    res_con_doc = motor.evaluar_peticion_cliente(
        peticion_usuario=peticion,
        datos_cliente=json.loads(json.dumps(datos_cliente_base)),
        documento_adjunto=pdf_path,
        nombre_archivo=pdf_path.name,
    )

    doc_ocr = res_con_doc.get("document_intelligence_ocr", {})
    eval_2 = res_con_doc["evaluacion_riesgo_ml"]
    agente_2 = res_con_doc["agente_conversacional_foundry"]
    legal_2 = res_con_doc.get("fundamentacion_legal_ecuador", {})

    print("\n1. RESULTADO DEL ANALISIS DE DOCUMENT INTELLIGENCE:")
    print(f"   -> Motor utilizado:            {doc_ocr.get('motor_extraccion')}")
    print(f"   -> Entidad externa detectada:  {doc_ocr.get('entidad_emisora')}")
    print(f"   -> Tipo de documento:          {doc_ocr.get('tipo_documento')}")
    print(f"   -> Titular detectado:          {doc_ocr.get('titular_detectado')}")
    print(f"   -> GASTOS_OCULTOS_OCR:         ${doc_ocr.get('gastos_ocultos_detectados', 0.0):,.2f}/mes (Deuda externa)")
    print(f"   -> INGRESOS_EXTRA_OCR:         ${doc_ocr.get('ingresos_adicionales_detectados', 0.0):,.2f}/mes")
    print("   -> Hallazgos clave en tablas:")
    for h in doc_ocr.get("hallazgos_clave", []):
        print(f"      * {h}")

    print("\n2. IMPACTO MATEMATICO EN EL MODELO ML (LIGHTGBM):")
    print(f"   -> Flujo libre mensual:        ${eval_2['flujo_neto_mensual_disponible']:,.2f}/mes (Cayo por la cuota de $320)")
    print(f"   -> Capacidad maxima cuota:     ${eval_2['capacidad_maxima_nueva_cuota']:,.2f}/mes (Se redujo a casi $0)")
    print(f"   -> Flujo si asume la moto:     ${eval_2['flujo_neto_post_simulacion']:,.2f}/mes (DEFICIT SEVERO)")
    print(f"   -> NUEVO SEMAFORO DE RIESGO:   {eval_2['semaforo_salud_financiera']} (Cambio de AMARILLO a ROJO)")
    print(f"   -> NUEVA PROBABILIDAD DE MORA: {eval_2['probabilidad_caer_en_vencido_mora']*100:.1f}%")
    print(f"   -> Estado de resolucion:       {res_con_doc['dictamen_endeudamiento']['estado_resolucion']}")

    print("\n3. RESPUESTA GENERADA POR GPT-4o (CON MENCION DEL DOCUMENTO):")
    print(f"   -> Titulo:     \"{agente_2.get('titulo_pantalla')}\"")
    print(f"   -> Explicacion:\"{agente_2.get('explicacion_ia')}\"")
    print(f"   -> Acciones:   {[b['label'] for b in agente_2.get('acciones_ui', [])]}")
    if legal_2:
        print(f"   -> Sustento JPRF/SuperBancos: {legal_2.get('sintesis_legal_usuario')}")

    # -------------------------------------------------------------------------
    # TABLA COMPARATIVA ANTES VS DESPUES
    # -------------------------------------------------------------------------
    g_ocultos_str = f"${doc_ocr.get('gastos_ocultos_detectados', 0.0):,.2f}/mes"
    flujo_1_str = f"${eval_1['flujo_neto_mensual_disponible']:,.2f}/mes"
    flujo_2_str = f"${eval_2['flujo_neto_mensual_disponible']:,.2f}/mes"
    cap_1_str = f"${eval_1['capacidad_maxima_nueva_cuota']:,.2f}/mes"
    cap_2_str = f"${eval_2['capacidad_maxima_nueva_cuota']:,.2f}/mes"
    mora_1_str = f"{eval_1['probabilidad_caer_en_vencido_mora']*100:.1f}%"
    mora_2_str = f"{eval_2['probabilidad_caer_en_vencido_mora']*100:.1f}%"

    print("\n" + "=" * 95)
    print("TABLA COMPARATIVA: EFECTO DE AZURE DOCUMENT INTELLIGENCE EN LA TOMA DE DECISION")
    print("=" * 95)
    print(f"{'Metrica Financiera':<35} | {'Sin Documento (Escenario 1)':<25} | {'Con Documento OCR (Escenario 2)':<25}")
    print("-" * 95)
    print(f"{'Gastos Ocultos Detectados':<35} | {'$0.00/mes':<25} | {g_ocultos_str:<25}")
    print(f"{'Flujo Neto Libre del Cliente':<35} | {flujo_1_str:<25} | {flujo_2_str:<25}")
    print(f"{'Capacidad Maxima Nueva Cuota':<35} | {cap_1_str:<25} | {cap_2_str:<25}")
    print(f"{'Semaforo Salud Financiera':<35} | {eval_1['semaforo_salud_financiera']:<25} | {eval_2['semaforo_salud_financiera']:<25}")
    print(f"{'Probabilidad de Caer en Mora':<35} | {mora_1_str:<25} | {mora_2_str:<25}")
    print(f"{'Resolucion del Banco':<35} | {res_sin_doc['dictamen_endeudamiento']['estado_resolucion']:<25} | {res_con_doc['dictamen_endeudamiento']['estado_resolucion']:<25}")
    print("=" * 95)

    print("\n" + "=" * 95)
    print("JSON OFICIAL FINAL DE RESPUESTA PARA EL BACKEND (ESCENARIO 2 CON DOCUMENT INTELLIGENCE):")
    print("=" * 95)
    print(json.dumps(res_con_doc, indent=2, ensure_ascii=False))
    print("=" * 95)


if __name__ == "__main__":
    ejecutar_demostracion_ocr()
