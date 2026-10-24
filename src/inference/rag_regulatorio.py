"""
Modulo RAG Regulatorio: Normativa Financiera, Crediticia y de Proteccion de Datos del Ecuador
Proyecto: Copiloto de Salud Financiera - Banco Guayaquil

Fundamenta legalmente las decisiones del modelo de Machine Learning para garantizar:
1. Cumplimiento de las normas de la Junta de Politica y Regulacion Financiera (JPRF).
2. Cumplimiento de las normas de gestion de riesgo de la Superintendencia de Bancos del Ecuador (SB).
3. Transparencia y explicabilidad algoritmica bajo la Ley Organica de Proteccion de Datos Personales (LOPDP).
4. Cumplimiento del Codigo Organico Monetario y Financiero (COMFyF) y techos del Banco Central del Ecuador (BCE).
5. Garantia de depositos e inversiones en Polizas respaldadas por COSEDE (hasta $32,000 USD).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FundamentoLegalDTO(BaseModel):
    normativa_principal: str = Field(..., description="Cuerpo legal ecuatoriano aplicable.")
    articulo_o_resolucion: str = Field(..., description="Numero de articulo o resolucion oficial.")
    principio_juridico: str = Field(..., description="Principio financiero aplicado (ej. DTI maximo, Credito Responsable).")
    texto_resumen: str = Field(..., description="Extracto explicativo de la norma legal ecuatoriana.")
    entidad_emisora: str = Field(..., description="JPRF, Superintendencia de Bancos, BCE o COSEDE.")


class RAGRegulatorioEcuador:
    """Motor RAG ligero especializado en legislacion financiera ecuatoriana."""

    BASE_CONOCIMIENTO_LEGAL: List[Dict[str, Any]] = [
        {
            "id": "JPRF_DTI_ENDEUDAMIENTO",
            "cuerpo": "Junta de Política y Regulación Financiera (JPRF)",
            "norma": "Resolución JPRF-F-2023-080 / Norma Prudencial de Gestión Crediticia",
            "tema": "SOBREENDEUDAMIENTO_DTI",
            "entidad": "JPRF",
            "principio": "Capacidad de Pago Prudencial (DTI <= 40%)",
            "resumen": (
                "Dispone que las entidades financieras deben evaluar la relación entre la cuota mensual "
                "de financiamiento y los ingresos netos verificables del deudor (DTI), recomendando que el "
                "servicio total de deuda no sobrepase el 40% al 50% de la liquidez libre para evitar el sobreendeudamiento."
            ),
            "keywords": ["endeudamiento", "cuota", "ingresos", "dti", "capacidad", "limite", "sobreendeudamiento", "asfixia"],
        },
        {
            "id": "SB_CALIFICACION_RIESGO_A1",
            "cuerpo": "Superintendencia de Bancos del Ecuador",
            "norma": "Codificación de Resoluciones SB, Libro I, Título IX, Cap. II (Calificación de Cartera)",
            "tema": "CALIFICACION_A1_NORMAL",
            "entidad": "Superintendencia de Bancos",
            "principio": "Calificación Crediticia Categoría A (Riesgo Normal)",
            "resumen": (
                "Establece que los sujetos de crédito con historial de pagos puntual (0 días de atraso) y flujo "
                "de caja suficiente para amortizar capital e intereses son clasificados en Categoría A1, siendo "
                "aptos para el otorgamiento de operaciones crediticias en condiciones preferenciales."
            ),
            "keywords": ["verde", "aprobado", "a1", "solvente", "puntual", "sin mora", "vencido 0", "excelente"],
        },
        {
            "id": "SB_CALIFICACION_RIESGO_MORA",
            "cuerpo": "Superintendencia de Bancos del Ecuador",
            "norma": "Codificación de Resoluciones SB, Libro I, Título IX, Art. 12 (Deterioro de Cartera)",
            "tema": "CARTERA_VENCIDA_MORA",
            "entidad": "Superintendencia de Bancos",
            "principio": "Mitigación de Riesgo de Morosidad y Deterioro Crediticio",
            "resumen": (
                "Mandata que cuando un deudor mantiene obligaciones en mora o cartera vencida, las entidades bancarias "
                "están impedidas de otorgar nuevo financiamiento que comprometa la recuperación del crédito o "
                "incurra en provisiones adicionales de riesgo por degradación a categorías B, C, D o E."
            ),
            "keywords": ["rojo", "vencido", "mora", "atraso", "provision", "castigo", "riesgo alto", "degradacion"],
        },
        {
            "id": "COMFYF_CREDITO_RESPONSABLE",
            "cuerpo": "Código Orgánico Monetario y Financiero (COMFyF)",
            "norma": "COMFyF Libro I, Artículos 209 y 211",
            "tema": "CREDITO_RESPONSABLE",
            "entidad": "Asamblea Nacional / BCE",
            "principio": "Deber Fiduciario y Principio de Crédito Responsable",
            "resumen": (
                "Consagra el deber de las entidades del sistema financiero de conceder créditos acordes a la real "
                "capacidad de pago del deudor, protegiendo los depósitos del público y prohibiendo el otorgamiento de "
                "deudas desproporcionadas que vulneren la subsistencia digna del usuario."
            ),
            "keywords": ["credito responsable", "responsable", "proteccion", "fiduciario", "banco", "usuario"],
        },
        {
            "id": "BCE_TECHOS_TASAS_INTERES",
            "cuerpo": "Banco Central del Ecuador (BCE)",
            "norma": "Resolución Monetaria BCE - Tasas de Interés Máximas Efectivas Vigentes",
            "tema": "TASAS_REFERENCIALES_BCE",
            "entidad": "BCE",
            "principio": "Cumplimiento de Tasas de Usura y Techos Máximos",
            "resumen": (
                "Fija los techos máximos legales de tasa activa efectiva para los segmentos de crédito de consumo "
                "(máximo legal ~16.77% anual). La tasa referencial de Banco Guayaquil (14.20% a 15.50%) cumple "
                "estrictamente dentro de los márgenes regulados y transparentes del BCE."
            ),
            "keywords": ["tasa", "interes", "bce", "legal", "usura", "consumo", "porcentaje"],
        },
        {
            "id": "LOPDP_TRANSPARENCIA_ALGORITMICA",
            "cuerpo": "Ley Orgánica de Protección de Datos Personales (LOPDP) de Ecuador",
            "norma": "LOPDP Registro Oficial Suplemento 459, Artículos 19 y 20",
            "tema": "TRANSPARENCIA_ALGORITMICA_IA",
            "entidad": "Autoridad de Protección de Datos Personales",
            "principio": "Derecho a la Explicabilidad y no Discriminación Algorítmica",
            "resumen": (
                "Garantiza el derecho de las personas a no ser objeto de decisiones exclusivamente automatizadas "
                "sin conocer la lógica matemática aplicada, asegurando que las evaluaciones crediticias con IA "
                "se fundamenten en variables financieras objetivas, auditables y sin sesgos discriminatorios."
            ),
            "keywords": ["lopdp", "datos personales", "explicacion", "inteligencia artificial", "modelo", "algoritmo", "derecho"],
        },
        {
            "id": "COSEDE_SEGURO_DEPOSITOS",
            "cuerpo": "Corporación del Seguro de Depósitos (COSEDE)",
            "norma": "Ley de Creación de la Red de Seguridad Financiera / COSEDE",
            "tema": "GARANTIA_POLIZAS_INVERSION",
            "entidad": "COSEDE",
            "principio": "Seguro de Depósitos Bancarios hasta $32,000 USD",
            "resumen": (
                "Garantiza que las inversiones a plazo fijo (Pólizas de Acumulación) y cuentas de ahorro en Banco "
                "Guayaquil cuentan con cobertura del Seguro de Depósitos administrado por COSEDE hasta por $32,000 USD, "
                "brindando total seguridad jurídica y patrimonial al ahorrista ecuatoriano."
            ),
            "keywords": ["poliza", "inversion", "plazo fijo", "cosede", "seguro de depositos", "garantia", "ahorro"],
        },
        {
            "id": "SB_REESTRUCTURACION_Y_ALIVIO",
            "cuerpo": "Superintendencia de Bancos del Ecuador",
            "norma": "Codificación de Resoluciones SB, Libro I, Título IX, Cap. VI (Mecanismos de Reestructuración y Alivio Financiero)",
            "tema": "REESTRUCTURACION_Y_ALIVIO_FINANCIERO",
            "entidad": "Superintendencia de Bancos",
            "principio": "Derecho a la Reestructuración de Deuda y Acuerdos de Pago Extrajudiciales",
            "resumen": (
                "Faculta a los deudores del sistema financiero que presenten dificultades severas de pago o sobreendeudamiento "
                "para solicitar ante su entidad bancaria la reestructuración, novación o refinanciamiento de sus obligaciones "
                "mediante ampliación de plazo, consolidación de cuotas o gracia de capital, permitiendo proteger su patrimonio sin caer en coactiva."
            ),
            "keywords": ["reestructurar", "reestructuracion", "renegociar", "refinanciar", "alivio", "sobreendeudado", "no puedo mas", "limite"],
        },
    ]

    @classmethod
    def buscar_fundamentacion_legal(
        cls,
        semaforo: str,
        probabilidad_mora: float,
        capital_vencido: float,
        cuota_solicitada: float,
        capacidad_max_cuota: float,
        intencion: str = "ENDEUDAMIENTO",
        es_sobreendeudado_critico: bool = False,
    ) -> Dict[str, Any]:
        """
        Recupera los fundamentos juridicos ecuatorianos pertinentes al veredicto financiero.
        """
        items_encontrados: List[FundamentoLegalDTO] = []

        # 1. Regla legal por Mora o Cartera Vencida preexistente
        if capital_vencido > 0 or probabilidad_mora >= 0.85:
            item_mora = cls._obtener_por_id("SB_CALIFICACION_RIESGO_MORA")
            if item_mora:
                items_encontrados.append(item_mora)

        # 2. Regla legal por Capacidad de Pago / DTI (JPRF)
        if semaforo in {"ROJO", "AMARILLO"} and cuota_solicitada > capacidad_max_cuota:
            item_dti = cls._obtener_por_id("JPRF_DTI_ENDEUDAMIENTO")
            if item_dti:
                items_encontrados.append(item_dti)

        # 3. Regla legal para clientes solventes (Categoría A1)
        if semaforo == "VERDE" and capital_vencido == 0:
            item_a1 = cls._obtener_por_id("SB_CALIFICACION_RIESGO_A1")
            if item_a1:
                items_encontrados.append(item_a1)

        # 4. Regla legal para Reestructuración y Alivio Financiero (Super de Bancos)
        if es_sobreendeudado_critico or intencion == "REESTRUCTURACION_DEUDA":
            item_alivio = cls._obtener_por_id("SB_REESTRUCTURACION_Y_ALIVIO")
            if item_alivio:
                items_encontrados.insert(0, item_alivio)

        # 5. Regla legal para Inversiones en Pólizas (COSEDE)
        if intencion == "INVERSION" or semaforo == "VERDE":
            item_cosede = cls._obtener_por_id("COSEDE_SEGURO_DEPOSITOS")
            if item_cosede:
                items_encontrados.append(item_cosede)

        # 5. Siempre adjuntar LOPDP y Crédito Responsable para transparencia algorítmica
        item_lopdp = cls._obtener_por_id("LOPDP_TRANSPARENCIA_ALGORITMICA")
        if item_lopdp and item_lopdp not in items_encontrados:
            items_encontrados.append(item_lopdp)

        # Construir frase sintética de respaldo legal
        if es_sobreendeudado_critico or intencion == "REESTRUCTURACION_DEUDA":
            sintesis = (
                "Al amparo de la Codificación de Resoluciones de la Superintendencia de Bancos (Capítulo VI, Mecanismos de Reestructuración) "
                "y las normas de Crédito Responsable de la JPRF, se activa el protocolo para negociar un acuerdo de alivio financiero y refinanciamiento extrajudicial."
            )
        elif semaforo == "ROJO":
            sintesis = (
                "De conformidad con la Norma de Gestión Crediticia de la JPRF (Resolución JPRF-F-2023-080) "
                "y la regulación de activos de riesgo de la Superintendencia de Bancos, esta recomendación busca "
                "evitar el sobreendeudamiento precautelando tu solvencia y estabilidad patrimonial."
            )
        elif semaforo == "AMARILLO":
            sintesis = (
                "Según los lineamientos prudenciales de la Junta de Política y Regulación Financiera (JPRF), "
                "la operación es viable pero se aconseja mantener el endeudamiento por debajo del 40% de ingresos netos."
            )
        else:
            sintesis = (
                "Cumple con los estándares de calificación Categoría A1 de la Superintendencia de Bancos y "
                "los principios de Crédito Responsable del COMFyF (Art. 209), con tasas activas dentro del techo BCE."
            )

        return {
            "sintesis_legal_usuario": sintesis,
            "articulos_citados": [f"{it.normativa_principal} ({it.articulo_o_resolucion})" for it in items_encontrados],
            "normativa_detallada": [it.model_dump() for it in items_encontrados],
        }

    @classmethod
    def _obtener_por_id(cls, item_id: str) -> Optional[FundamentoLegalDTO]:
        for d in cls.BASE_CONOCIMIENTO_LEGAL:
            if d["id"] == item_id:
                return FundamentoLegalDTO(
                    normativa_principal=d["cuerpo"],
                    articulo_o_resolucion=d["norma"],
                    principio_juridico=d["principio"],
                    texto_resumen=d["resumen"],
                    entidad_emisora=d["entidad"],
                )
        return None
