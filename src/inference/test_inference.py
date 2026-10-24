"""
Script de prueba de inferencia local con el JSON de muestra req_bg_9901
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import joblib
from src.inference.copilot_engine import CopilotEngine


def run_test() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    model_path = workspace_root / "models" / "bg_risk_lightgbm.joblib"
    sample_json_path = workspace_root / "data" / "samples" / "sample_input.json"

    if not model_path.exists():
        print(f"Modelo no encontrado en {model_path}. Entrenando primero...")
        from src.models.train_lightgbm import train_and_save_model
        train_and_save_model()

    print(f"Cargando modelo LightGBM desde: {model_path}")
    model = joblib.load(model_path)

    print(f"Cargando JSON de prueba desde: {sample_json_path}")
    with open(sample_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    copilot = CopilotEngine(model)
    output = copilot.process(payload)

    print("\n" + "=" * 80)
    print("RESULTADO DE INFERENCIA OFICIAL (JSON DE SALIDA)")
    print("=" * 80)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_test()
