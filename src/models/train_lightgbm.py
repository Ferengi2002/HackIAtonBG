"""
Modulo de Entrenamiento y Serializacion de LightGBM (Modelo Oficial de Riesgo)
Banco Guayaquil - Copiloto de Salud Financiera y Prevencion de Morosidad
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("bg_train_lightgbm")

RANDOM_SEED = 42


def train_and_save_model() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    dataset_path = workspace_root / "dataset_banco_guayaquil_entrenamiento.csv"
    models_dir = workspace_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "bg_risk_lightgbm.joblib"
    metadata_path = models_dir / "model_metadata.json"

    logger.info(f"Cargando dataset de entrenamiento: {dataset_path}")
    df = pd.read_csv(dataset_path)

    feature_cols = [c for c in df.columns if c not in ["COD_CLTE", "TARGET_RIESGO_MORA"]]
    X = df[feature_cols].copy()
    y = df["TARGET_RIESGO_MORA"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    logger.info(f"Entrenando LightGBM sobre {len(X_train):,} registros...")
    model = LGBMClassifier(
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        min_child_samples=30,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=2.0,
        n_jobs=-1,
        verbose=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    prob_test = model.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= 0.50).astype(int)

    auc = roc_auc_score(y_test, prob_test)
    pr_auc = average_precision_score(y_test, prob_test)
    f1 = f1_score(y_test, pred_test)
    recall = recall_score(y_test, pred_test)
    acc = accuracy_score(y_test, pred_test)
    brier = brier_score_loss(y_test, prob_test)

    idx_redes = feature_cols.index("SCORE_ESTILO_VIDA_REDES")
    imp = model.feature_importances_
    imp_pct = (imp / imp.sum()) * 100.0
    peso_redes = float(round(imp_pct[idx_redes], 2))

    logger.info(f"Metricas en Test: ROC-AUC={auc:.4f} | Recall Mora={recall:.4f} | F1={f1:.4f} | Peso Redes={peso_redes}%")

    joblib.dump(model, model_path)
    logger.info(f"Modelo serializado guardado en: {model_path}")

    metadata = {
        "model_name": "Banco Guayaquil - Credit Risk & Health Copilot",
        "algorithm": "LightGBM (Leaf-wise Gradient Boosting)",
        "version": "1.0.0",
        "feature_names": feature_cols,
        "metrics": {
            "roc_auc_test": round(auc, 4),
            "pr_auc_test": round(pr_auc, 4),
            "f1_score": round(f1, 4),
            "recall_mora": round(recall, 4),
            "accuracy": round(acc, 4),
            "brier_loss": round(brier, 4),
            "peso_redes_pct": peso_redes,
        },
        "risk_thresholds": {
            "green_max_prob": 0.15,
            "yellow_max_prob": 0.35,
            "max_safe_dti_pct": 40.0,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadatos guardados en: {metadata_path}")


if __name__ == "__main__":
    train_and_save_model()

