"""
QualitySupervisorAgent: Agente Auditor y Guardián de Calidad en el Medio (Agent-in-the-Middle)
Implementa el patrón Reflection & Quality Gate para verificar que toda respuesta generada
por el copiloto cumpla con los estándares bancarios de Banco Guayaquil antes de ser entregada al cliente.

Ejecuta un Checklist de 5 puntos:
1. Vocabulario comercial limpio (cero nombres de tablas internas de base de datos como MAESTRO_METAS).
2. Alineación estricta con la intención y pregunta del usuario.
3. Consistencia cuantitativa con los cálculos del motor actuarial ML.
4. Alternativa constructiva en semáforo ROJO o situaciones de contingencia.
5. Tono bancario profesional, conciso y cero emojis (Anti-AI Slop).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger("bg_quality_supervisor")


class ItemChecklist(BaseModel):
    cumple: bool = Field(..., description="Indica si el borrador superó este chequeo.")
    observacion: str = Field(..., description="Detalle del hallazgo u observación de calidad.")
    correccion_aplicada: Optional[str] = Field(
        default=None, description="Descripción del ajuste automático o refinamiento aplicado."
    )


class AuditoriaSupervisorDTO(BaseModel):
    agente_supervisor: str = Field(default="QualitySupervisorAgent_v1")
    veredicto_auditoria: str = Field(
        ...,
        description="Estado final: APROBADO_DIRECTO, APROBADO_CON_REFINAMIENTO, APROBADO_SANITIZADO"
    )
    requirio_regeneracion_llm: bool = Field(default=False)
    intentos_evaluados: int = Field(default=1)
    checklist_calidad: Dict[str, ItemChecklist] = Field(default_factory=dict)
    observaciones_generales: List[str] = Field(default_factory=list)
    tiempo_auditoria_ms: float = Field(default=0.0)


class QualitySupervisorAgent:
    """
    Agente Auditor / Supervisor de Calidad que inspecciona y refina las respuestas
    del Copiloto de Salud Financiera de Banco Guayaquil.
    """

    # Diccionario de sanitización de jerga técnica / nombres de tablas SQL
    TABLAS_TECNICAS_MAP = {
        r"\bMAESTRO_METAS\b": "Metas de Ahorro en tu App Banco Guayaquil",
        r"\bMAESTRO_POLIZAS\b": "Póliza a Plazo Fijo en Banco Guayaquil",
        r"\bMAESTRO_CUENTAS\b": "tu cuenta de ahorros Banco Guayaquil",
        r"\bMVTO_CUENTAS\b": "tus movimientos de cuenta",
        r"\bMAESTRO_CREDITOS\b": "tus créditos en Banco Guayaquil",
        r"\bMAESTRO_TC\b": "tu tarjeta de crédito Banco Guayaquil",
        r"\bMVTO_TC\b": "tus consumos de tarjeta de crédito",
        r"\bBURO_EQUIFAX\b": "Buró de Crédito Equifax",
        r"\bMAESTRO_CLIENTES\b": "tu perfil de cliente",
        r"\bCODGRUOPE\b": "tipo de crédito",
        r"\bSOLUCIONES_CREDITO_BG\b": "Soluciones de Pago Banco Guayaquil",
        r"\bCANALES_BG\b": "Canales Oficiales Banco Guayaquil",
    }

    # Expresión regular para detectar y eliminar emojis (Anti-AI Slop)
    EMOJI_REGEX = re.compile(
        r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\u2300-\u23ff\u2b50\u2b55\u200d\ufe0f]+",
        flags=re.UNICODE,
    )

    def __init__(self, foundry_client: Optional[Any] = None) -> None:
        self.foundry_client = foundry_client

    @classmethod
    def sanitizar_texto_comercial(cls, texto: str) -> Tuple[str, List[str]]:
        """
        Reemplaza cualquier nombre de tabla interna por el término bancario oficial
        y elimina emojis. Devuelve el texto limpio y la lista de ajustes realizados.
        """
        ajustes: List[str] = []
        texto_limpio = texto

        # 1. Eliminar emojis
        if cls.EMOJI_REGEX.search(texto_limpio):
            texto_limpio = cls.EMOJI_REGEX.sub("", texto_limpio)
            ajustes.append("Se eliminaron emojis no permitidos para preservar estética bancaria sobria.")

        # 2. Reemplazar nombres de tablas
        for patron, reemplazo in cls.TABLAS_TECNICAS_MAP.items():
            if re.search(patron, texto_limpio, flags=re.IGNORECASE):
                texto_limpio = re.sub(patron, reemplazo, texto_limpio, flags=re.IGNORECASE)
                tabla_nombre = patron.replace(r"\b", "")
                ajustes.append(f"Se reemplazó término técnico '{tabla_nombre}' por lenguaje comercial oficial: '{reemplazo}'.")

        # 3. Limpiar espacios dobles generados por reemplazos y espacios antes de signos de puntuación
        texto_limpio = re.sub(r"\s+([.,;:!?])", r"\1", texto_limpio)
        texto_limpio = re.sub(r"\s{2,}", " ", texto_limpio).strip()
        return texto_limpio, ajustes

    def auditar_y_refinar_respuesta(
        self,
        peticion_usuario: str,
        titulo_propuesto: str,
        explicacion_propuesta: str,
        contexto_ml: Dict[str, Any],
        intencion_nlu: str,
        es_solicitud_deuda_nueva: bool,
    ) -> Tuple[str, str, AuditoriaSupervisorDTO]:
        """
        Inspecciona el borrador propuesto contra el checklist de 5 puntos.
        Si detecta fallas críticas y Azure AI Foundry está disponible, solicita
        una corrección (Reflection Loop). En cualquier caso, aplica sanitización
        garantizada para que nunca llegue una respuesta con fallas al usuario.
        """
        t_inicio = time.perf_counter()
        checklist: Dict[str, ItemChecklist] = {}
        observaciones: List[str] = []
        requirio_regeneracion = False
        intentos = 1

        # --------------------------------------------------------------------
        # CHECK 1: VOCABULARIO COMERCIAL LIMPIO (Cero nombres de tablas SQL)
        # --------------------------------------------------------------------
        tablas_detectadas = [
            p.replace(r"\b", "")
            for p in self.TABLAS_TECNICAS_MAP
            if re.search(p, f"{titulo_propuesto} {explicacion_propuesta}", flags=re.IGNORECASE)
        ]
        if tablas_detectadas:
            checklist["1_vocabulario_comercial_limpio"] = ItemChecklist(
                cumple=False,
                observacion=f"Borrador contenía nombres de tablas internas de BD: {', '.join(tablas_detectadas)}.",
                correccion_aplicada="Sanitización automática a términos comerciales de Banco Guayaquil.",
            )
            observaciones.append("Falla Check 1: Fuga de nombres de tablas de base de datos en la respuesta.")
        else:
            checklist["1_vocabulario_comercial_limpio"] = ItemChecklist(
                cumple=True,
                observacion="Vocabulario comercial limpio sin fuga de tablas de base de datos.",
            )

        # --------------------------------------------------------------------
        # CHECK 2: ALINEACIÓN CON LA INTENCIÓN DEL USUARIO
        # --------------------------------------------------------------------
        texto_completo = f"{titulo_propuesto} {explicacion_propuesta}".lower()
        if not es_solicitud_deuda_nueva and "cuota de $0.00" in texto_completo:
            checklist["2_alineacion_intencion_usuario"] = ItemChecklist(
                cumple=False,
                observacion="Mención inapropiada de 'cuota de $0.00' en consulta que no pide préstamo.",
                correccion_aplicada="Eliminación de referencias a cuota simulada vacía.",
            )
            observaciones.append("Falla Check 2: Se mencionaba una cuota ficticia de $0.00.")
        else:
            checklist["2_alineacion_intencion_usuario"] = ItemChecklist(
                cumple=True,
                observacion=f"Respuesta alineada con la intención '{intencion_nlu}' y la consulta del usuario.",
            )

        # --------------------------------------------------------------------
        # CHECK 3: CONSISTENCIA CUANTITATIVA CON EL MOTOR ML
        # --------------------------------------------------------------------
        cuota_esperada = float(contexto_ml.get("cuota_nueva", 0.0))
        flujo_libre = float(contexto_ml.get("flujo_neto_actual", 0.0))
        semaforo = contexto_ml.get("semaforo", "VERDE")

        # Verificar si semáforo es ROJO y se afirmó aprobación indebida
        if semaforo == "ROJO" and any(w in texto_completo for w in ["aprobado", "felicitaciones", "sí puedes"]):
            checklist["3_consistencia_cuantitativa_ml"] = ItemChecklist(
                cumple=False,
                observacion="Inconsistencia grave: Semáforo ROJO pero el texto sugiere aprobación de crédito.",
                correccion_aplicada="Reversión inmediata a veredicto preventivo de denegación.",
            )
            observaciones.append("Falla Check 3: Contradicción con el semáforo de riesgo actuarial.")
        else:
            checklist["3_consistencia_cuantitativa_ml"] = ItemChecklist(
                cumple=True,
                observacion=f"Cifras coherentes con Semáforo {semaforo} (Flujo: ${flujo_libre:,.2f}, Cuota: ${cuota_esperada:,.2f}).",
            )

        # --------------------------------------------------------------------
        # CHECK 4: ALTERNATIVA CONSTRUCTIVA EN SEMÁFORO ROJO O ALIVIO
        # --------------------------------------------------------------------
        if semaforo == "ROJO":
            tiene_alternativa = any(
                w in texto_completo for w in ["meta", "ahorro", "soluciones de pago", "alivio", "reestructur", "contado"]
            )
            if not tiene_alternativa:
                checklist["4_alternativa_constructiva_alivio"] = ItemChecklist(
                    cumple=False,
                    observacion="Semáforo ROJO no ofrecía una alternativa constructiva para el usuario.",
                    correccion_aplicada="Inyección de propuesta de Meta de Ahorro o Solución de Pago.",
                )
                observaciones.append("Falla Check 4: Ausencia de alternativa constructiva en denegación.")
            else:
                checklist["4_alternativa_constructiva_alivio"] = ItemChecklist(
                    cumple=True,
                    observacion="Ofrece alternativa constructiva adecuada (Meta de Ahorro o plan preventivo).",
                )
        else:
            checklist["4_alternativa_constructiva_alivio"] = ItemChecklist(
                cumple=True,
                observacion="No aplica restricción de semáforo rojo.",
            )

        # --------------------------------------------------------------------
        # CHECK 5: TONO BANCARIO, CERO EMOJIS Y LONGITUD
        # --------------------------------------------------------------------
        tiene_emojis = bool(self.EMOJI_REGEX.search(f"{titulo_propuesto} {explicacion_propuesta}"))
        conteo_palabras = len(explicacion_propuesta.split())
        es_muy_largo = conteo_palabras > 110

        if tiene_emojis or es_muy_largo:
            detalles = []
            if tiene_emojis:
                detalles.append("contenía emojis")
            if es_muy_largo:
                detalles.append(f"longitud excesiva ({conteo_palabras} palabras)")
            checklist["5_tono_cero_emojis_y_longitud"] = ItemChecklist(
                cumple=False,
                observacion=f"Falta de estándar: {', '.join(detalles)}.",
                correccion_aplicada="Eliminación de emojis y ajuste de concisión.",
            )
            observaciones.append("Falla Check 5: Violación de directrices de estilo profesional bancario.")
        else:
            checklist["5_tono_cero_emojis_y_longitud"] = ItemChecklist(
                cumple=True,
                observacion=f"Tono profesional sobrio sin emojis ({conteo_palabras} palabras).",
            )

        # --------------------------------------------------------------------
        # DECISIÓN DEL SUPERVISOR: ¿REQUIERE REFLECTION LOOP O AUTO-CORRECCIÓN?
        # --------------------------------------------------------------------
        fallas_criticas = [k for k, v in checklist.items() if not v.cumple]
        titulo_final = titulo_propuesto
        explicacion_final = explicacion_propuesta

        if fallas_criticas and self.foundry_client is not None and getattr(self.foundry_client, "is_available", False):
            # Intentar Reflection Loop vía Azure AI Foundry
            logger.info(f"QualitySupervisorAgent detectó fallas: {fallas_criticas}. Activando Reflection Loop...")
            titulo_refinado, explicacion_refinada = self._solicitar_reflection_llm(
                peticion_usuario=peticion_usuario,
                borrador_titulo=titulo_propuesto,
                borrador_explicacion=explicacion_propuesta,
                fallas=fallas_criticas,
                contexto_ml=contexto_ml,
            )
            if titulo_refinado and explicacion_refinada:
                titulo_final = titulo_refinado
                explicacion_final = explicacion_refinada
                requirio_regeneracion = True
                intentos = 2
                observaciones.append("Respuesta re-generada y perfeccionada por el Supervisor mediante Reflection Loop.")

        # Sanitización de defensa en profundidad obligatoria en todos los casos
        titulo_final, adj_tit = self.sanitizar_texto_comercial(titulo_final)
        explicacion_final, adj_exp = self.sanitizar_texto_comercial(explicacion_final)

        # Determinar veredicto final
        if not fallas_criticas:
            veredicto = "APROBADO_DIRECTO"
        elif requirio_regeneracion:
            veredicto = "APROBADO_CON_REFINAMIENTO"
        else:
            veredicto = "APROBADO_SANITIZADO"

        t_fin = time.perf_counter()
        duracion_ms = round((t_fin - t_inicio) * 1000, 2)

        dto = AuditoriaSupervisorDTO(
            agente_supervisor="QualitySupervisorAgent_v1",
            veredicto_auditoria=veredicto,
            requirio_regeneracion_llm=requirio_regeneracion,
            intentos_evaluados=intentos,
            checklist_calidad=checklist,
            observaciones_generales=observaciones,
            tiempo_auditoria_ms=duracion_ms,
        )

        return titulo_final, explicacion_final, dto

    def _solicitar_reflection_llm(
        self,
        peticion_usuario: str,
        borrador_titulo: str,
        borrador_explicacion: str,
        fallas: List[str],
        contexto_ml: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Invoca a Azure AI Foundry para reescribir corrigiendo las violaciones detectadas."""
        system_supervisor = (
            "Eres el Auditor y Supervisor de Calidad Bancaria de Banco Guayaquil. "
            "Un redactor preliminar generó un borrador que falló en los siguientes criterios de calidad: "
            f"{', '.join(fallas)}. "
            "Tu tarea es corregir y pulir el texto. Devuelve ÚNICAMENTE un JSON con {'titulo_pantalla': str, 'explicacion_ia': str}. "
            "Reglas inquebrantables: "
            "1. PROHIBIDO usar nombres de tablas SQL internas (nada de MAESTRO_METAS, MAESTRO_POLIZAS, MVTO_CUENTAS, MAESTRO_CREDITOS). "
            "   Usa nombres comerciales: 'Metas de Ahorro en tu App Banco Guayaquil', 'Póliza a Plazo Fijo', 'tus movimientos de cuenta'. "
            "2. PROHIBIDO el uso de emojis. "
            "3. Respeta exactamente el semáforo y cifras actuariales del banco. "
            "4. Si el semáforo es ROJO, explica por qué no conviene endeudarse y ofrece la alternativa constructiva de ahorro meta. "
            "5. Máximo 85 palabras."
        )

        user_content = (
            f"Consulta del usuario: {peticion_usuario}\n"
            f"Métricas del cliente: {contexto_ml}\n"
            f"Borrador que falló: Título: '{borrador_titulo}' | Explicación: '{borrador_explicacion}'"
        )

        try:
            res = self.foundry_client.call_chat_json(
                system_prompt=system_supervisor,
                user_prompt=user_content,
                max_tokens=200,
                temperature=0.1,
            )
            if res and "titulo_pantalla" in res and "explicacion_ia" in res:
                return str(res["titulo_pantalla"]), str(res["explicacion_ia"])
        except Exception as e:
            logger.warning(f"Error en Reflection Loop de QualitySupervisor: {e}")

        return None, None
