"""
Script simple de verificacion:
Toma el payload oficial de entrada(con datos de las 8 tablas de Banco Guayaquil + Buro + OCR),
ejecuta el modelo LightGBM entrenado y genera el JSON oficial de salida.
"""

import json
import sys
from pathlib import Path

# Soporte de codificacion UTF-8 para consola Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.inference.copilot_engine import CopilotEngine

def main():
    workspace_root = Path(__file__).resolve().parent
    input_file = workspace_root / "data" / "samples" / "sample_input.json"

    print("=" * 85)
    print("DEMOSTRACION DEL COPILOTO DE SALUD FINANCIERA - BANCO GUAYAQUIL")
    print("=" * 85)
    print(f"1. Cargando datos del cliente desde: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        datos_cliente = json.load(f)

    print(f"   -> Cliente ID: {datos_cliente['maestro_clientes']['COD_CLTE']}")
    print(f"   -> Transaccion: {datos_cliente['transaccion_id']}")
    print(f"   -> Simulacion solicitada: {datos_cliente['simulacion_solicitada_usuario']}")
    print(f"   -> Datos OCR (Document Intelligence): {datos_cliente['fuentes_externas_ia']['document_intelligence_ocr']}")

    print("\n2. Ejecutando motor de inferencia (LightGBM + Reglas de Capacidad)...")
    import joblib
    model_path = workspace_root / "models" / "bg_risk_lightgbm.joblib"
    modelo_ml = joblib.load(model_path)
    motor = CopilotEngine(model=modelo_ml)
    resultado = motor.process(datos_cliente)

    print("\n" + "=" * 85)
    print("3. JSON OFICIAL DE RESPUESTA GENERADO EXITOSAMENTE:")
    print("=" * 85)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    print("=" * 85)

    print("\nRESUMEN EJECUTIVO:")
    eval_ml = resultado["evaluacion_riesgo_ml"]
    dictamen = resultado["dictamen_endeudamiento"]
    agente = resultado["agente_conversacional_foundry"]

    print(f"-> Semaforo: {eval_ml['semaforo_salud_financiera']}")
    print(f"-> Probabilidad de Mora: {eval_ml['probabilidad_caer_en_vencido_mora']*100:.1f}%")
    print(f"-> Capacidad Maxima de Cuota Segura: ${eval_ml['capacidad_maxima_nueva_cuota']:,.2f}/mes")
    print(f"-> Resolucion: {dictamen['estado_resolucion']}")
    print(f"-> Mensaje al Usuario: \"{agente['titulo_pantalla']}\"")
    print(f"-> Explicacion IA: \"{agente['explicacion_ia']}\"")
    print("=" * 85)

if __name__ == "__main__":
    main()
