"""
Pipeline de Preparacion y Adecuacion de Datos (Paso 1 - Version Balanceada 50/50 y Ampliada)
Proyecto: Copiloto de Salud Financiera y Prevencion de Morosidad - Banco Guayaquil
Reto: AI Banking - Construir la proxima generacion de banca digital

Objetivo Cientifico:
Generar un dataset de entrenamiento robusto de 100,000 clientes unicos
('dataset_banco_guayaquil_entrenamiento.csv') con balanceo exacto 50% / 50%
en la variable objetivo (TARGET_RIESGO_MORA) sin duplicar filas (evitando
data leakage y overfitting, y evitando que el modelo sea 'vago' por desbalance de clases).

Glosario Matematico del Pipeline:
- inc_rank (Percentil Salarial u_i en [0, 1]): Posicion relativa del ingreso del cliente.
- engel_base_ratio: Proporcion de gasto corriente segun la Ley de Engel bancaria
  (a mayor ingreso mensual, menor porcentaje se destina a subsistencia basica).
- dti_ratio (Debt-to-Income Ratio): Carga financiera mensual dividida para el ingreso total.
- z1 (Puntaje Financiero Estandarizado): Combinacion lineal de solvencia, flujo neto y buro.
- z2 (Penalizacion por Atrasos y Apalancamiento): Impacto acumulado de mora vigente e historica.
- z3 (Factor Demografico y Resiliencia Salarial): Efecto determinante de EDAD (>=18 sin tope)
  y nivel salarial (desde SBU $460 hasta clientes del exterior de $20,000+ USD/mes).
- z_total (Logit de Riesgo Latente): z_total = z0 + z1 + z2 + z3 + epsilon.
- P(Mora) = 1 / (1 + exp(-z_total)): Funcion sigmoide logistica que transforma z_total en [0, 1].
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.datasets import fetch_openml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("bg_data_pipeline")

RANDOM_SEED = 42

PUBLIC_GITHUB_URLS = [
    "https://raw.githubusercontent.com/ChicagoBoothML/DATA___GiveMeSomeCredit/master/cs-training.csv",
    "https://raw.githubusercontent.com/afaan-nazar/Give-Me-Some-Credit/master/cs-training.csv",
    "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/cs-training.csv",
]


def download_base_credit_dataset(cache_dir: Path) -> pd.DataFrame:
    """
    Descarga el dataset real 'Give Me Some Credit' (cs-training.csv, 150,000 registros)
    desde OpenML o espejos en GitHub y lo almacena en cache local.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_path = cache_dir / "cs_training_raw.csv"

    if raw_cache_path.exists():
        logger.info(f"Cargando dataset base desde cache local: {raw_cache_path}")
        return pd.read_csv(raw_cache_path)

    try:
        logger.info("Cargando dataset oficial 'GiveMeSomeCredit' (150,000 filas) desde OpenML...")
        openml_ds = fetch_openml(
            name="GiveMeSomeCredit",
            version=1,
            as_frame=True,
            parser="auto",
        )
        df_raw = openml_ds.frame.copy()
    except Exception as openml_exc:
        logger.warning(f"OpenML no disponible ({openml_exc}), intentando mirrors en GitHub...")
        df_raw = None
        for url in PUBLIC_GITHUB_URLS:
            try:
                response = requests.get(url, timeout=8)
                if response.status_code == 200 and len(response.content) > 1_000_000:
                    raw_cache_path.write_bytes(response.content)
                    df_raw = pd.read_csv(raw_cache_path)
                    break
            except Exception as exc:
                logger.warning(f"Mirror GitHub no disponible ({url}): {exc}")
        if df_raw is None:
            raise RuntimeError("No se pudo descargar el dataset base de ninguna fuente.") from openml_exc

    rename_map = {}
    for col in df_raw.columns:
        col_lower = col.lower()
        if "serious" in col_lower or "dlq" in col_lower or "distress" in col_lower or col_lower == "class":
            rename_map[col] = "SeriousDlqin2yrs"
        elif "revolving" in col_lower:
            rename_map[col] = "RevolvingUtilizationOfUnsecuredLines"
        elif col_lower == "age":
            rename_map[col] = "age"
        elif "30-59" in col_lower or "30_59" in col_lower:
            rename_map[col] = "NumberOfTime30-59DaysPastDueNotWorse"
        elif "debtratio" in col_lower or "debt_ratio" in col_lower:
            rename_map[col] = "DebtRatio"
        elif "monthlyincome" in col_lower or "monthly_income" in col_lower:
            rename_map[col] = "MonthlyIncome"
        elif "opencredit" in col_lower or "open_credit" in col_lower:
            rename_map[col] = "NumberOfOpenCreditLinesAndLoans"
        elif "90days" in col_lower or "90_days" in col_lower:
            rename_map[col] = "NumberOfTimes90DaysLate"
        elif "realestate" in col_lower or "real_estate" in col_lower:
            rename_map[col] = "NumberRealEstateLoansOrLines"
        elif "60-89" in col_lower or "60_89" in col_lower:
            rename_map[col] = "NumberOfTime60-89DaysPastDueNotWorse"
        elif "dependents" in col_lower:
            rename_map[col] = "NumberOfDependents"

    df_raw = df_raw.rename(columns=rename_map)
    df_raw.to_csv(raw_cache_path, index=False)
    logger.info(f"Dataset base guardado en cache local: {raw_cache_path} ({len(df_raw):,} filas)")
    return df_raw


def clean_and_expand_base_population(
    df_raw: pd.DataFrame,
    pool_size: int = 180_000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Limpia el dataset de 150,000 filas y genera un pool enriquecido de perfiles
    de riesgo y saludables (sin duplicados exactos, mediante interpolacion tipo SMOTE
    empirico sobre la clase minoritaria de impago) para luego extraer 100,000 clientes
    perfectamente balanceados (50% clase 0 y 50% clase 1).
    """
    rng = np.random.default_rng(seed)
    df = df_raw.copy()

    # Mapear 'Yes'/'No' o {1, 2} en SeriousDlqin2yrs antes de convertir a numerico
    raw_target = df["SeriousDlqin2yrs"].astype(str).str.strip().str.lower()
    df["SeriousDlqin2yrs"] = np.where(
        raw_target.isin(["yes", "1", "true", "1.0"]),
        1,
        0,
    ).astype(int)

    numeric_cols = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. Filtrar unicamente mayores de edad (>= 18 años), manteniendo hasta 100+ años
    df = df[df["age"] >= 18].copy().reset_index(drop=True)

    # Inyectar segmento joven (18-22 años) clave en banca digital
    young_mask = rng.random(len(df)) < 0.05
    df.loc[young_mask, "age"] = rng.integers(18, 23, size=young_mask.sum())

    # 2. Corregir codigos de error 96/98 de Kaggle
    late_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    for col in late_cols:
        df[col] = np.where(df[col] >= 96, rng.integers(3, 8, size=len(df)), df[col])
        df[col] = df[col].fillna(0).clip(0, 15)

    # 3. Limpiar valores extremos corruptos por division para cero
    df["RevolvingUtilizationOfUnsecuredLines"] = np.where(
        df["RevolvingUtilizationOfUnsecuredLines"] > 3.0,
        rng.uniform(0.65, 1.45, size=len(df)),
        df["RevolvingUtilizationOfUnsecuredLines"].fillna(0.25),
    ).clip(0.0, 2.2)

    df["DebtRatio"] = np.where(
        df["DebtRatio"] > 4.0,
        rng.uniform(0.32, 0.90, size=len(df)),
        df["DebtRatio"].fillna(0.30),
    ).clip(0.02, 2.50)

    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0).clip(0, 8).astype(int)
    df["NumberOfOpenCreditLinesAndLoans"] = df["NumberOfOpenCreditLinesAndLoans"].fillna(2).clip(0, 25).astype(int)
    df["NumberRealEstateLoansOrLines"] = df["NumberRealEstateLoansOrLines"].fillna(0).clip(0, 8).astype(int)

    # 4. Ampliacion organica de perfiles de estres crediticio (Tipo SMOTE con perturbacion continua)
    # En Kaggle original solo hay ~10,026 casos con SeriousDlqin2yrs == 1.
    # Para lograr 50,000 casos positivos reales y variados (sin clonar filas que causen overfitting),
    # generamos interpolaciones convexas entre pares aleatorios de clientes morosos/estresados.
    df_pos = df[df["SeriousDlqin2yrs"] == 1].reset_index(drop=True)
    n_synthetic_pos = 55_000
    idx_a = rng.integers(0, len(df_pos), size=n_synthetic_pos)
    idx_b = rng.integers(0, len(df_pos), size=n_synthetic_pos)
    alpha = rng.uniform(0.15, 0.85, size=n_synthetic_pos)

    synth_pos = pd.DataFrame()
    synth_pos["SeriousDlqin2yrs"] = np.ones(n_synthetic_pos, dtype=int)
    synth_pos["RevolvingUtilizationOfUnsecuredLines"] = np.clip(
        alpha * df_pos.loc[idx_a, "RevolvingUtilizationOfUnsecuredLines"].values
        + (1 - alpha) * df_pos.loc[idx_b, "RevolvingUtilizationOfUnsecuredLines"].values
        + rng.normal(0.0, 0.06, size=n_synthetic_pos),
        0.05,
        2.2,
    )
    synth_pos["age"] = np.round(
        alpha * df_pos.loc[idx_a, "age"].values
        + (1 - alpha) * df_pos.loc[idx_b, "age"].values
        + rng.normal(0.0, 2.0, size=n_synthetic_pos)
    ).clip(18, 105).astype(int)

    for col in late_cols:
        synth_pos[col] = np.round(
            alpha * df_pos.loc[idx_a, col].values + (1 - alpha) * df_pos.loc[idx_b, col].values
        ).clip(0, 15).astype(int)

    synth_pos["DebtRatio"] = np.clip(
        alpha * df_pos.loc[idx_a, "DebtRatio"].values
        + (1 - alpha) * df_pos.loc[idx_b, "DebtRatio"].values
        + rng.normal(0.0, 0.05, size=n_synthetic_pos),
        0.05,
        2.5,
    )

    inc_a = df_pos.loc[idx_a, "MonthlyIncome"].fillna(df["MonthlyIncome"].median()).values
    inc_b = df_pos.loc[idx_b, "MonthlyIncome"].fillna(df["MonthlyIncome"].median()).values
    synth_pos["MonthlyIncome"] = np.maximum(250.0, alpha * inc_a + (1 - alpha) * inc_b)

    for col in ["NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines", "NumberOfDependents"]:
        synth_pos[col] = np.round(
            alpha * df_pos.loc[idx_a, col].values + (1 - alpha) * df_pos.loc[idx_b, col].values
        ).astype(int)

    df_expanded = pd.concat([df, synth_pos], ignore_index=True)
    if len(df_expanded) > pool_size:
        df_expanded = df_expanded.sample(n=pool_size, random_state=seed).reset_index(drop=True)

    return df_expanded


def build_banco_guayaquil_features(
    df_base: pd.DataFrame,
    target_total_rows: int = 100_000,
    target_positive_ratio: float = 0.50,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Construye las 21 columnas oficiales del dataset de entrenamiento de Banco Guayaquil
    y aplica muestreo estratificado final para obtener exactamente:
    - 100,000 registros unicos (50,000 Clase 0 Saludable + 50,000 Clase 1 Riesgo de Mora).
    """
    rng = np.random.default_rng(seed)
    n = len(df_base)
    bg = pd.DataFrame()

    # 1. MAESTRO_CLIENTES
    bg["COD_CLTE"] = np.arange(10_000_001, 10_000_001 + n, dtype=np.int64)
    bg["EDAD"] = df_base["age"].astype(int).values
    max_tenure = np.maximum(0.2, (bg["EDAD"] - 18) * rng.uniform(0.15, 0.85, size=n))
    bg["ANTIGUEDAD_BANCO_ANIOS"] = np.round(np.clip(max_tenure, 0.1, 45.0), 1)

    # 2. MVTO_CUENTAS & MAESTRO_CUENTAS (Salarios desde $350/SBU $460 hasta $20,000-$48,000+ USD/mes)
    raw_income = df_base["MonthlyIncome"].values.copy()
    missing_inc = np.isnan(raw_income) | (raw_income <= 0)
    raw_income[missing_inc] = rng.lognormal(mean=7.2, sigma=0.75, size=missing_inc.sum())

    inc_rank = pd.Series(raw_income).rank(pct=True).values
    age_income_factor = np.where(
        bg["EDAD"] < 22,
        rng.uniform(0.45, 0.85, size=n),
        np.where(bg["EDAD"] > 78, rng.uniform(0.65, 1.10, size=n), 1.0),
    )

    base_ecuador_income = np.select(
        [
            inc_rank < 0.25,
            inc_rank < 0.70,
            inc_rank < 0.90,
            inc_rank < 0.98,
        ],
        [
            380.0 + (inc_rank / 0.25) * 270.0,
            650.0 + ((inc_rank - 0.25) / 0.45) * 1550.0,
            2200.0 + ((inc_rank - 0.70) / 0.20) * 4300.0,
            6500.0 + ((inc_rank - 0.90) / 0.08) * 11500.0,
        ],
        default=18000.0 + ((inc_rank - 0.98) / 0.02) * 27000.0,
    )
    bg["INGRESOS_CREDITO_MES"] = np.round(
        np.maximum(350.0, base_ecuador_income * age_income_factor * rng.uniform(0.92, 1.08, size=n)),
        2,
    )

    delinq_30_59 = df_base["NumberOfTime30-59DaysPastDueNotWorse"].values
    delinq_60_89 = df_base["NumberOfTime60-89DaysPastDueNotWorse"].values
    delinq_90_plus = df_base["NumberOfTimes90DaysLate"].values
    total_past_due_events = delinq_30_59 + (delinq_60_89 * 2) + (delinq_90_plus * 3)

    rev_util = df_base["RevolvingUtilizationOfUnsecuredLines"].values
    debt_ratio_raw = df_base["DebtRatio"].values

    # Ley de Engel Bancaria para Egresos Corrientes
    engel_base_ratio = 0.74 - 0.36 * (inc_rank ** 0.75)
    expense_ratio = np.clip(
        engel_base_ratio
        + rng.normal(0.0, 0.09, size=n)
        + np.where(total_past_due_events > 0, 0.12, 0.0)
        + np.where(df_base["SeriousDlqin2yrs"].values == 1, 0.08, 0.0),
        0.18,
        0.96,
    )
    bg["EGRESOS_DEBITO_MES"] = np.round(bg["INGRESOS_CREDITO_MES"] * expense_ratio, 2)

    liquidity_months = np.where(
        (total_past_due_events == 0) & (rev_util < 0.45) & (df_base["SeriousDlqin2yrs"].values == 0),
        rng.lognormal(mean=-0.05, sigma=0.72, size=n),
        rng.exponential(scale=0.20, size=n),
    )
    bg["SALDO_CUENTAS"] = np.round(
        np.maximum(5.0, bg["INGRESOS_CREDITO_MES"] * np.clip(liquidity_months, 0.01, 6.0)),
        2,
    )

    # 3. MAESTRO_CREDITOS
    open_loans = df_base["NumberOfOpenCreditLinesAndLoans"].values
    has_active_loan = (open_loans >= 2) | (debt_ratio_raw > 0.20)
    loan_burden_pct = np.where(
        has_active_loan,
        np.clip(debt_ratio_raw * rng.uniform(0.25, 0.55, size=n), 0.04, 0.52),
        0.0,
    )
    bg["CUOTA_CREDITOS_VIGENTES"] = np.round(bg["INGRESOS_CREDITO_MES"] * loan_burden_pct, 2)

    remaining_installments = rng.integers(6, 48, size=n)
    bg["CAPITAL_POR_VENCER"] = np.round(
        bg["CUOTA_CREDITOS_VIGENTES"] * remaining_installments * rng.uniform(0.75, 0.92, size=n),
        2,
    )

    has_current_arrears = (
        (delinq_90_plus > 0)
        | (delinq_60_89 > 0)
        | ((delinq_30_59 > 0) & (rng.random(n) < 0.65))
        | ((df_base["SeriousDlqin2yrs"].values == 1) & (rng.random(n) < 0.58))
    )
    arrears_installments = np.clip(
        delinq_30_59 * 0.8 + delinq_60_89 * 1.8 + delinq_90_plus * 3.2 + rng.uniform(0.5, 2.0, size=n),
        0.5,
        12.0,
    )
    bg["CAPITAL_VENCIDO"] = np.where(
        has_current_arrears,
        np.round(np.maximum(45.0, bg["CUOTA_CREDITOS_VIGENTES"] * arrears_installments), 2),
        0.0,
    )

    # 4. MAESTRO_TC & MVTO_TC
    bg["GASTO_PROMEDIO_TC"] = np.round(
        bg["INGRESOS_CREDITO_MES"] * np.clip(rev_util * rng.uniform(0.12, 0.32, size=n), 0.0, 0.45),
        2,
    )
    bg["CUOTA_DIFERIDOS_TC"] = np.round(
        bg["GASTO_PROMEDIO_TC"] * np.clip(rev_util * rng.uniform(0.25, 0.65, size=n), 0.05, 0.75),
        2,
    )
    travel_prob = np.clip(0.06 + 0.45 * (inc_rank ** 1.6), 0.05, 0.85)
    bg["VIAJES_EXTERIOR_TC"] = (rng.random(n) < travel_prob).astype(int)

    # 5. MAESTRO_POLIZAS & MAESTRO_METAS
    saver_prob = np.where(
        (bg["CAPITAL_VENCIDO"] == 0) & (rev_util < 0.55) & (df_base["SeriousDlqin2yrs"].values == 0),
        np.clip(0.25 + 0.40 * inc_rank, 0.20, 0.75),
        0.06,
    )
    has_poliza = rng.random(n) < saver_prob
    bg["MONTO_POLIZAS_VIGENTES"] = np.where(
        has_poliza,
        np.round(np.maximum(500.0, bg["SALDO_CUENTAS"] * rng.uniform(0.4, 2.5, size=n)), 2),
        0.0,
    )

    has_meta = rng.random(n) < (saver_prob * 1.15).clip(0.08, 0.80)
    bg["MONTO_ACUM_METAS"] = np.where(
        has_meta,
        np.round(np.maximum(50.0, bg["INGRESOS_CREDITO_MES"] * rng.uniform(0.10, 0.65, size=n)), 2),
        0.0,
    )
    bg["CUMPLIO_META_HIST"] = np.where(
        has_meta & (total_past_due_events == 0) & (rng.random(n) < 0.68),
        1,
        0,
    ).astype(int)

    # 6. FUENTES EXTERNAS IA (OCR, Equifax, Redes Sociales)
    has_ocr_income = rng.random(n) < 0.35
    bg["INGRESOS_EXTRA_OCR"] = np.where(
        has_ocr_income,
        np.round(bg["INGRESOS_CREDITO_MES"] * rng.uniform(0.08, 0.35, size=n), 2),
        0.0,
    )

    has_hidden_expenses = (rng.random(n) < 0.30) | (rev_util > 0.80) | (df_base["SeriousDlqin2yrs"].values == 1)
    bg["GASTOS_OCULTOS_OCR"] = np.where(
        has_hidden_expenses,
        np.round(bg["INGRESOS_CREDITO_MES"] * rng.uniform(0.05, 0.30, size=n), 2),
        0.0,
    )

    ingresos_totales = bg["INGRESOS_CREDITO_MES"] + bg["INGRESOS_EXTRA_OCR"]
    egresos_y_cuotas_totales = (
        bg["EGRESOS_DEBITO_MES"]
        + bg["CUOTA_CREDITOS_VIGENTES"]
        + bg["CUOTA_DIFERIDOS_TC"]
        + (bg["GASTO_PROMEDIO_TC"] * 0.35)
        + bg["GASTOS_OCULTOS_OCR"]
    )
    bg["FLUJO_NETO_DISPONIBLE"] = np.round(ingresos_totales - egresos_y_cuotas_totales, 2)

    total_debt_service = bg["CUOTA_CREDITOS_VIGENTES"] + bg["CUOTA_DIFERIDOS_TC"] + bg["GASTOS_OCULTOS_OCR"]
    dti_ratio = total_debt_service / np.maximum(1.0, ingresos_totales)

    equifax_base = (
        825.0
        - (delinq_30_59 * 48.0)
        - (delinq_60_89 * 85.0)
        - (delinq_90_plus * 135.0)
        - np.where(bg["CAPITAL_VENCIDO"] > 0, 115.0, 0.0)
        - (np.clip(rev_util, 0.0, 1.5) * 110.0)
        - (np.clip(dti_ratio - 0.35, 0.0, 1.0) * 165.0)
        + (inc_rank * 55.0)
        + np.where(bg["FLUJO_NETO_DISPONIBLE"] < 0, -65.0, np.clip(bg["FLUJO_NETO_DISPONIBLE"] / 150.0, 0.0, 35.0))
        + np.where(bg["CUMPLIO_META_HIST"] == 1, 28.0, 0.0)
        + np.where(bg["MONTO_POLIZAS_VIGENTES"] > 0, 22.0, 0.0)
        + np.clip(bg["ANTIGUEDAD_BANCO_ANIOS"] * 2.2, 0.0, 32.0)
        + rng.normal(0.0, 20.0, size=n)
    )
    bg["SCORE_EQUIFAX"] = np.round(np.clip(equifax_base, 300, 999)).astype(int)

    lifestyle_raw = 5.0 + (bg["VIAJES_EXTERIOR_TC"] * 1.4) + (inc_rank * 0.9) + rng.normal(0.0, 1.8, size=n)
    bg["SCORE_ESTILO_VIDA_REDES"] = np.round(np.clip(lifestyle_raw, 1.0, 10.0), 1)

    # 7. ECUACION ESTRUCTURAL DE RIESGO LATENTE (z_total = z0 + z1 + z2 + z3 + epsilon)
    # z1: Solvencia de Flujo de Caja, Liquidez y Buro
    liquidity_buffer_ratio = (bg["SALDO_CUENTAS"] + bg["MONTO_POLIZAS_VIGENTES"] + bg["MONTO_ACUM_METAS"]) / np.maximum(
        1.0, ingresos_totales
    )
    z1_solvencia_flujo = (
        np.where(bg["FLUJO_NETO_DISPONIBLE"] < 0, 2.25, np.where(bg["FLUJO_NETO_DISPONIBLE"] < 150.0, 0.95, -0.70))
        + ((690.0 - bg["SCORE_EQUIFAX"]) / 110.0)
        - np.clip(liquidity_buffer_ratio * 0.35, 0.0, 1.10)
    )

    # z2: Historial de Impago, Mora Vigente y Sobreendeudamiento (DTI)
    z2_mora_y_deuda = (
        (df_base["SeriousDlqin2yrs"].values * 2.45)
        + np.where(bg["CAPITAL_VENCIDO"] > 0, 2.65, 0.0)
        + (np.clip(dti_ratio - 0.36, -0.2, 0.8) * 2.85)
    )

    # z3: Factores Determinantes de Edad (>=18 sin tope) y Nivel Salarial (Resiliencia de Ingresos)
    age_risk_penalty = np.where(
        (bg["EDAD"] <= 22) & (bg["SALDO_CUENTAS"] < 400.0),
        0.75,
        np.where(
            (bg["EDAD"] > 75) & (dti_ratio > 0.30) & (bg["MONTO_POLIZAS_VIGENTES"] < bg["CAPITAL_POR_VENCER"] * 0.5),
            0.90,
            0.0,
        ),
    )
    income_determinant_effect = -1.25 * (inc_rank ** 0.85) + np.where(
        (bg["INGRESOS_CREDITO_MES"] < 600.0) & (dti_ratio > 0.32),
        0.85,
        0.0,
    )
    z3_demografia_y_salario = age_risk_penalty + income_determinant_effect - (bg["SCORE_ESTILO_VIDA_REDES"] * 0.015)

    # z_total: Logit total de riesgo
    z_total = -1.65 + z1_solvencia_flujo + z2_mora_y_deuda + z3_demografia_y_salario + rng.normal(0.0, 0.32, size=n)
    prob_mora_true = 1.0 / (1.0 + np.exp(-z_total))
    bg["TARGET_RIESGO_MORA"] = (prob_mora_true >= 0.50).astype(int)

    # 8. BALANCEO EXACTO 50% / 50% SOBRE 100,000 CLIENTES UNICOS
    bg_pos = bg[bg["TARGET_RIESGO_MORA"] == 1]
    bg_neg = bg[bg["TARGET_RIESGO_MORA"] == 0]

    n_pos_target = int(target_total_rows * target_positive_ratio)
    n_neg_target = target_total_rows - n_pos_target

    n_pos_actual = min(len(bg_pos), n_pos_target)
    n_neg_actual = min(len(bg_neg), n_neg_target)
    # Asegurar paridad exacta 50-50
    half_size = min(n_pos_actual, n_neg_actual)

    bg_balanced = (
        pd.concat(
            [
                bg_pos.sample(n=half_size, random_state=seed),
                bg_neg.sample(n=half_size, random_state=seed),
            ],
            ignore_index=True,
        )
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    # Reasignar COD_CLTE secuencial limpio tras el balanceo
    bg_balanced["COD_CLTE"] = np.arange(10_000_001, 10_000_001 + len(bg_balanced), dtype=np.int64)
    return bg_balanced


def validate_and_report(df: pd.DataFrame) -> None:
    """
    Ejecuta validaciones estadisticas y verifica el balanceo 50/50 libre de duplicados.
    """
    print("\n" + "=" * 92)
    print("REPORTE DE VALIDACION DEL DATASET BALANCEADO: dataset_banco_guayaquil_entrenamiento.csv")
    print("=" * 92)
    print(f"Total de registros (Clientes unicos COD_CLTE): {len(df):,}")
    print(f"Filas duplicadas (sin contar COD_CLTE):        {df.drop(columns=['COD_CLTE']).duplicated().sum()}")
    print(f"Total de columnas:                             {df.shape[1]}")

    null_count = int(df.isna().sum().sum())
    print(f"Total de valores nulos (NaN):                  {null_count} (Validacion: {'OK' if null_count == 0 else 'ERROR'})")

    counts = df["TARGET_RIESGO_MORA"].value_counts()
    pcts = df["TARGET_RIESGO_MORA"].value_counts(normalize=True) * 100
    print(
        f"Distribucion TARGET_RIESGO_MORA (Balanceo):    "
        f"0 (Saludable) = {counts.get(0, 0):,} ({pcts.get(0, 0.0):.2f}%) | "
        f"1 (Riesgo Mora) = {counts.get(1, 0):,} ({pcts.get(1, 0.0):.2f}%)"
    )

    print("\n--- 1. ESTADISTICAS DESCRIPTIVAS (REALIDAD ECUADOR + INTERNACIONAL USD) ---")
    desc_cols = [
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
        "MONTO_POLIZAS_VIGENTES",
        "MONTO_ACUM_METAS",
        "SCORE_EQUIFAX",
        "INGRESOS_EXTRA_OCR",
        "GASTOS_OCULTOS_OCR",
        "SCORE_ESTILO_VIDA_REDES",
        "FLUJO_NETO_DISPONIBLE",
    ]
    stats = df[desc_cols].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
    stats_view = stats[["mean", "std", "min", "25%", "50%", "75%", "95%", "99%", "max"]].round(2)
    print(stats_view.to_string())

    print("\n--- 2. CORRELACION CON TARGET_RIESGO_MORA Y SCORE_EQUIFAX ---")
    corr_matrix = df.drop(columns=["COD_CLTE"]).corr()
    corr_summary = pd.DataFrame(
        {
            "Corr_con_TARGET_RIESGO_MORA": corr_matrix["TARGET_RIESGO_MORA"],
            "Corr_con_SCORE_EQUIFAX": corr_matrix["SCORE_EQUIFAX"],
        }
    ).sort_values(by="Corr_con_TARGET_RIESGO_MORA", ascending=False)
    print(corr_summary.round(4).to_string())

    print("\n--- 3. VERIFICACION DE SEGMENTOS DE EDAD Y ALTOS INGRESOS (SIN TOPE ARTIFICIAL) ---")
    print(f"Rango de EDAD:                  Min = {df['EDAD'].min()} años | Max = {df['EDAD'].max()} años")
    print(f"Clientes Adultos Mayores (>75): {(df['EDAD'] > 75).sum():,} clientes")
    print(f"Clientes Jovenes (18-22 años):  {(df['EDAD'] <= 22).sum():,} clientes")
    print(
        f"Rango INGRESOS_CREDITO_MES:     Min = ${df['INGRESOS_CREDITO_MES'].min():,.2f} | "
        f"Max = ${df['INGRESOS_CREDITO_MES'].max():,.2f} USD/mes"
    )
    print(f"Clientes Alta Renta (>$10,000): {(df['INGRESOS_CREDITO_MES'] >= 10000).sum():,} clientes")
    print(f"Clientes Exterior/Wealth(>$20k):{(df['INGRESOS_CREDITO_MES'] >= 20000).sum():,} clientes")
    print("=" * 92 + "\n")


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    cache_dir = workspace_root / "data" / "raw"
    output_csv_path = workspace_root / "dataset_banco_guayaquil_entrenamiento.csv"

    df_raw = download_base_credit_dataset(cache_dir=cache_dir)
    df_pool = clean_and_expand_base_population(df_raw, pool_size=195_000, seed=RANDOM_SEED)
    df_bg = build_banco_guayaquil_features(
        df_pool,
        target_total_rows=100_000,
        target_positive_ratio=0.50,
        seed=RANDOM_SEED,
    )

    df_bg.to_csv(output_csv_path, index=False)
    logger.info(f"Dataset oficial balanceado generado en: {output_csv_path}")

    validate_and_report(df_bg)


if __name__ == "__main__":
    main()
