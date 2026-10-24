"""
Modulo de Guardrails de Seguridad, Filtro de Dominio y Optimizacion de Tokens
Proyecto: Copiloto de Salud Financiera - Banco Guayaquil

Responsabilidades:
1. Filtro rapido de Saludos y Despedidas simples (Responde en <1ms sin gastar tokens ni invocar ML).
2. Deteccion de Consultas Fuera de Dominio (Out of Scope / Absurdas / No Financieras).
3. Derivacion a Canales Oficiales para temas sensibles (Fraude, reclamos judiciales, reestructuracion profunda).
4. Deteccion basica de Idioma (Espanol, Ingles, Kichwa) para responder en la lengua del usuario.
5. Filtro de Inyeccion de Prompts / Jailbreaks de politicas bancarias.
6. Permite el paso transparente cuando hay intenciones financieras reales (ej. 'Hola, quiero un credito de $2,000...').
"""

from __future__ import annotations

import re
from typing import Any, Dict, Literal, Optional, Tuple
from pydantic import BaseModel, Field


class GuardrailCheckResult(BaseModel):
    es_valido_para_pipeline_financiero: bool = Field(
        ...,
        description="True si la peticion debe ejecutarse en el modelo ML y NLU financiero.",
    )
    tipo_evento: Literal[
        "FINANCIERO_VALIDO",
        "SALUDO_O_CORTESIA",
        "DESPEDIDA_O_AGRADECIMIENTO",
        "FUERA_DE_DOMINIO",
        "CASO_SENSIBLE_ASESOR_HUMANO",
        "INTENTO_JAILBREAK",
        "INTENTO_EXFILTRACION_O_SECRETOS",
    ] = Field(..., description="Clasificacion de la peticion del usuario.")
    idioma_detectado: Literal["es", "en", "qu"] = Field(
        default="es", description="Idioma detectado: 'es' (Espanol), 'en' (Ingles), 'qu' (Kichwa)."
    )
    respuesta_inmediata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Respuesta empaquetada lista para el cliente si no requiere inferencia ML.",
    )


class GuardrailsEngine:
    """Motor de guardrails ligero, determinista y seguro de alta velocidad."""

    # Canales oficiales de atencion humana de Banco Guayaquil
    CANALES_OFICIALES_BG = {
        "call_center": "04 3730100 (Guayaquil) / 1800 3730100 (Nacional)",
        "whatsapp_oficial": "+593 98 373 0100",
        "web": "https://www.bancoguayaquil.com",
        "horario": "Atencion 24/7 en canales digitales y telefonicos",
    }

    # Palabras clave financieras inequívocas
    KEYWORDS_FINANCIERAS = {
        "credito", "crédito", "prestamo", "préstamo", "cuota", "endeudamiento", "endeudar",
        "plazo", "tasa", "interes", "interés", "ahorro", "ahorrar", "meta", "poliza", "póliza",
        "inversion", "inversión", "plazo fijo", "saldo", "cuenta", "tarjeta", "diferir",
        "dinero", "dolares", "dólares", "sueldo", "ingreso", "gasto", "deuda", "equifax",
        "buro", "buró", "score", "alcanza", "comprar", "pagar", "financiar", "financiamiento",
        "moto", "auto", "carro", "casa", "terreno", "tv", "laptop", "computadora", "viaje",
        "sobreendeudado", "sobreendeudamiento", "endeudado", "reestructurar", "reestructuracion", "reestructuración",
        "renegociar", "renegociacion", "renegociación", "refinanciar", "refinanciamiento", "alivio", "moroso", "mora",
        "limite", "límite", "deudas", "cuello",
    }

    # Palabras clave de casos sensibles que requieren soporte humano urgente
    KEYWORDS_SENSIBLES = {
        "fraude", "estafa", "me robaron", "clonaron", "clonacion", "clonación", "bloquear tarjeta",
        "bloqueo de cuenta", "demanda", "juicio", "coactiva", "abogado", "quiebra", "bancarrota",
        "coactivo", "embargo", "fallecido", "herencia", "suplantacion", "suplantación",
    }

    # Patrones de Jailbreak / Prompt Injection comunes
    PATRONES_JAILBREAK = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"ignora\s+(todas\s+las\s+)?instrucciones\s+(anteriores|previas)",
        r"actua\s+como\s+dan",
        r"act\s+as\s+dan",
        r"you\s+are\s+now\s+an\s+unrestricted",
        r"system\s*prompt",
        r"dame\s+un\s+credito\s+con\s+tasa\s+0",
        r"aprueba\s+sin\s+revisar",
        r"bypass\s+rules",
    ]

    # Patrones estrictos de Exfiltración de Credenciales, Código, .env o PII de otros usuarios (OWASP LLM02)
    PATRONES_EXFILTRACION_Y_SECRETOS = [
        # Variables de entorno, .env, credenciales y API keys
        r"\.env",
        r"dotenv",
        r"(?:credenciales|credentials|secretos?|secrets?|api[_\s]?keys?|tokens?)",
        r"(?:contrase[ñn]a|passwords?|llave[s]?\s+privada[s]?|private[_\s]?key)",
        r"(?:azure[_\s]?openai|connection[_\s]?string|cadena\s+de\s+conexi[oó]n)",
        r"variables?\s+de\s+entorno",
        r"env\s+vars?",
        # Código fuente, arquitectura interna, scripts y archivos del sistema
        r"(?:c[oó]digo\s+(?:de\s+la\s+app|fuente|python|del\s+sistema))",
        r"(?:source[_\s]?code|mu[eé]strame\s+el\s+c[oó]digo|dame\s+el\s+c[oó]digo|ver\s+el\s+c[oó]digo)",
        r"(?:archivos?\s+del\s+proyecto|archivos?\s+locales|qu[eé]\s+archivos\s+tienes)",
        r"(?:app\.py|motor_ia_foundry\.py|guardrails\.py|requirements\.txt)",
        r"(?:c[oó]mo\s+est[aá]s\s+programad[oa]|qu[eé]\s+instrucciones\s+te\s+dieron|revela\s+tus\s+reglas)",
        # Datos de otros usuarios, PII y violación de privacidad (Anti-IDOR)
        r"(?:otros?\s+usuarios?|otros?\s+clientes?|otra\s+persona|otras\s+personas|otra\s+cuenta|otras\s+cuentas)",
        r"(?:datos|saldo|cuenta|tarjeta|c[eé]dula|informaci[oó]n|score)\s+de\s+(?:otro|otra|otros|otras|dem[aá]s)\s+(?:cliente|usuario|persona|cuentas?)",
        r"(?:base\s+de\s+datos\s+de\s+clientes|lista\s+de\s+clientes|quien\s+m[aá]s\s+tiene\s+cuenta)",
        r"(?:informaci[oó]n\s+confidencial\s+de\s+terceros|cu[aá]nto\s+dinero\s+tienen\s+los\s+dem[aá]s)",
    ]

    @classmethod
    def detectar_idioma(cls, texto: str) -> Literal["es", "en", "qu"]:
        t_low = texto.lower()
        # Deteccion basica de Kichwa
        palabras_kichwa = {"allillanchu", "imanalla", "yupaychani", "kullki", "mashna", "ari", "mana"}
        if any(w in t_low.split() for w in palabras_kichwa):
            return "qu"

        # Deteccion basica de Ingles
        palabras_en = {"hello", "hi", "credit", "loan", "how", "much", "can", "afford", "buy", "save", "what", "where", "thanks", "bye"}
        tokens = set(re.findall(r"\b\w+\b", t_low))
        en_matches = len(tokens.intersection(palabras_en))
        if en_matches >= 2 or (en_matches >= 1 and any(w in tokens for w in {"can", "how", "what", "where", "hello", "hi"})):
            return "en"

        return "es"

    @classmethod
    def evaluar_peticion(cls, peticion: str, cod_clte: int = 10045892) -> GuardrailCheckResult:
        """
        Analiza la peticion y decide si debe entrar al pipeline financiero o resolverse
        inmediatamente con 0 tokens de LLM.
        """
        texto = peticion.strip()
        t_low = texto.lower()
        idioma = cls.detectar_idioma(texto)

        # --------------------------------------------------------------------
        # 0. Filtro Anti-Exfiltración: Credenciales, .env, Código y Datos de Otros Usuarios (OWASP LLM02)
        # --------------------------------------------------------------------
        for patron in cls.PATRONES_EXFILTRACION_Y_SECRETOS:
            if re.search(patron, t_low):
                return GuardrailCheckResult(
                    es_valido_para_pipeline_financiero=False,
                    tipo_evento="INTENTO_EXFILTRACION_O_SECRETOS",
                    idioma_detectado=idioma,
                    respuesta_inmediata=cls._crear_respuesta_exfiltracion_seguridad(cod_clte, idioma),
                )

        # --------------------------------------------------------------------
        # 1. Filtro de Seguridad: Jailbreaks / Prompt Injections
        # --------------------------------------------------------------------
        for patron in cls.PATRONES_JAILBREAK:
            if re.search(patron, t_low):
                return GuardrailCheckResult(
                    es_valido_para_pipeline_financiero=False,
                    tipo_evento="INTENTO_JAILBREAK",
                    idioma_detectado=idioma,
                    respuesta_inmediata=cls._crear_respuesta_seguridad(cod_clte, idioma),
                )

        # --------------------------------------------------------------------
        # 2. Casos Sensibles que requieren Derivacion a Asesor Humano
        # --------------------------------------------------------------------
        if any(k in t_low for k in cls.KEYWORDS_SENSIBLES):
            return GuardrailCheckResult(
                es_valido_para_pipeline_financiero=False,
                tipo_evento="CASO_SENSIBLE_ASESOR_HUMANO",
                idioma_detectado=idioma,
                respuesta_inmediata=cls._crear_respuesta_sensible(cod_clte, idioma, t_low),
            )

        # --------------------------------------------------------------------
        # 3. Detectar si contiene contenido financiero real (montos, plazos o terminos bancarios)
        # --------------------------------------------------------------------
        tiene_cifras = bool(re.search(r"\$?\s*\d+(?:[.,]\d+)?", t_low))
        palabras_en_texto = set(re.findall(r"\b\w+\b", t_low))
        tiene_keywords_fin = bool(palabras_en_texto.intersection(cls.KEYWORDS_FINANCIERAS))

        # Si menciona una intencion financiera clara, pasa al pipeline bancario
        if tiene_keywords_fin or (tiene_cifras and any(w in t_low for w in ["cuota", "mes", "meses", "prestamo", "credito", "comprar"])):
            return GuardrailCheckResult(
                es_valido_para_pipeline_financiero=True,
                tipo_evento="FINANCIERO_VALIDO",
                idioma_detectado=idioma,
                respuesta_inmediata=None,
            )

        # --------------------------------------------------------------------
        # 4. Saludos y Cortesia simples (Costo 0 tokens)
        # --------------------------------------------------------------------
        saludos_es = {"hola", "buen dia", "buenos dias", "buenas tardes", "buenas noches", "que tal", "como estas", "cómo estás", "saludos"}
        saludos_en = {"hello", "hi", "good morning", "good afternoon", "good evening", "how are you"}
        saludos_qu = {"allillanchu", "imanalla"}

        es_saludo_puro = (
            any(t_low == s or t_low.startswith(s + " ") or t_low.endswith(" " + s) for s in saludos_es)
            or any(t_low == s or t_low.startswith(s + " ") for s in saludos_en)
            or any(w in t_low for w in saludos_qu)
        ) and len(palabras_en_texto) <= 6

        if es_saludo_puro:
            return GuardrailCheckResult(
                es_valido_para_pipeline_financiero=False,
                tipo_evento="SALUDO_O_CORTESIA",
                idioma_detectado=idioma,
                respuesta_inmediata=cls._crear_respuesta_saludo(cod_clte, idioma),
            )

        # --------------------------------------------------------------------
        # 5. Despedidas y Agradecimientos simples (Costo 0 tokens)
        # --------------------------------------------------------------------
        despedidas = {"gracias", "muchas gracias", "adios", "adiós", "chao", "hasta luego", "bye", "thank you", "thanks", "yupaychani"}
        if any(d in t_low for d in despedidas) and len(palabras_en_texto) <= 5:
            return GuardrailCheckResult(
                es_valido_para_pipeline_financiero=False,
                tipo_evento="DESPEDIDA_O_AGRADECIMIENTO",
                idioma_detectado=idioma,
                respuesta_inmediata=cls._crear_respuesta_despedida(cod_clte, idioma),
            )

        # --------------------------------------------------------------------
        # 6. Consultas Fuera de Dominio (Absurdas, Generales, Trivia, etc.)
        # --------------------------------------------------------------------
        return GuardrailCheckResult(
            es_valido_para_pipeline_financiero=False,
            tipo_evento="FUERA_DE_DOMINIO",
            idioma_detectado=idioma,
            respuesta_inmediata=cls._crear_respuesta_fuera_dominio(cod_clte, idioma, texto),
        )

    # ------------------------------------------------------------------------
    # GENERADORES DE RESPUESTAS ESTRUCTURADAS INMEDIATAS (CONTRATO JSON OFICIAL)
    # ------------------------------------------------------------------------
    @classmethod
    def _crear_respuesta_saludo(cls, cod_clte: int, idioma: str) -> Dict[str, Any]:
        if idioma == "en":
            titulo = "Hello! I am your Financial Health Copilot."
            explicacion = (
                "Welcome to Banco Guayaquil. I am here to evaluate your financial capacity, "
                "simulate loans with real risk modeling, and help you grow your savings with CD investments. "
                "How can I assist your financial goals today?"
            )
        elif idioma == "qu":
            titulo = "Allillanchu! Banco Guayaquil kullki yanapawukmi kani."
            explicacion = (
                "Alli shamushka kapay. Ñukaka kanpak kullkipi yanapanami kani, mañaykunata riksinkapak, "
                "chashnallatak kullkita wakaychinkapak. Imapitak yanapayta ushani?"
            )
        else:
            titulo = "¡Hola! Soy tu Copiloto de Salud Financiera de Banco Guayaquil."
            explicacion = (
                "Estoy aquí para ayudarte a tomar decisiones financieras informadas. "
                "Puedes consultarme si te alcanza para una compra (ej. una moto, un auto o una laptop), "
                "revisar tu capacidad de cuota mensual, o conocer cómo hacer crecer tus ahorros en una Póliza."
            )

        return cls._empaquetar_json_guardrail(
            cod_clte=cod_clte,
            titulo=titulo,
            explicacion=explicacion,
            tipo_guardrail="SALUDO",
            acciones=[
                {"tipo": "SIMULAR_COMPRA", "label": "Simular compra o crédito", "payload_tabla": "MAESTRO_CREDITOS"},
                {"tipo": "CONSULTAR_AHORRO", "label": "Rentabilizar ahorros", "payload_tabla": "MAESTRO_POLIZAS"},
            ],
        )

    @classmethod
    def _crear_respuesta_despedida(cls, cod_clte: int, idioma: str) -> Dict[str, Any]:
        if idioma == "en":
            titulo = "Thank you for trusting Banco Guayaquil."
            explicacion = "It was a pleasure assisting you. Remember you can consult me whenever you want to evaluate an expense or plan a goal."
        elif idioma == "qu":
            titulo = "Yupaychani Banco Guayaquilpi kashkamanta."
            explicacion = "Kushikunimi kanwan parlashpa. Maykan punchapash kullkipi tapunkapak shamupaylla."
        else:
            titulo = "¡Un gusto haberte ayudado!"
            explicacion = "Recuerda que estoy disponible 24/7 en tu banca móvil para cuidar tu salud financiera antes de asumir nuevos compromisos."

        return cls._empaquetar_json_guardrail(
            cod_clte=cod_clte,
            titulo=titulo,
            explicacion=explicacion,
            tipo_guardrail="DESPEDIDA",
            acciones=[],
        )

    @classmethod
    def _crear_respuesta_fuera_dominio(cls, cod_clte: int, idioma: str, peticion_orig: str) -> Dict[str, Any]:
        if idioma == "en":
            titulo = "Query out of financial scope."
            explicacion = (
                "As Banco Guayaquil's Financial Health Copilot, my specialty is analyzing credit capacity, "
                "preventing overindebtedness, and advising on savings and investment certificates. "
                "I cannot answer questions about general knowledge, geography, or non-banking topics. "
                "Would you like to simulate a personal loan or check your safe payment capacity?"
            )
        elif idioma == "qu":
            titulo = "Kay tapuyka mana kullkimantachu kan."
            explicacion = (
                "Ñukaka Banco Guayaquilpak kullki yanapawukmi kani. Kullkimanta, mañaykunamanta, "
                "wakaychinamantalla parlani. Mana shuktak tapuykunata kutichiyta ushanichu. "
                "Kullkimanta tapuyta munankichu?"
            )
        else:
            titulo = "Consulta fuera del ámbito financiero de Banco Guayaquil."
            explicacion = (
                "Como Copiloto de Salud Financiera de Banco Guayaquil, mi función exclusiva es evaluar tu capacidad "
                "de pago, prevenir el sobreendeudamiento y orientarte en créditos, pólizas y metas de ahorro. "
                "No puedo responder consultas de conocimiento general, entretenimiento o temas ajenos a tus finanzas. "
                "¿Deseas saber si te alcanza para una compra o conocer tu capacidad de cuota mensual?"
            )

        return cls._empaquetar_json_guardrail(
            cod_clte=cod_clte,
            titulo=titulo,
            explicacion=explicacion,
            tipo_guardrail="FUERA_DE_DOMINIO",
            acciones=[
                {"tipo": "EJEMPLO_SIMULACION", "label": "Consultar si me alcanza para una compra", "payload_tabla": "MAESTRO_CREDITOS"},
                {"tipo": "CONSULTAR_CAPACIDAD", "label": "Ver mi capacidad máxima de cuota", "payload_tabla": "MAESTRO_CUENTAS"},
            ],
        )

    @classmethod
    def _crear_respuesta_sensible(cls, cod_clte: int, idioma: str, detalle: str) -> Dict[str, Any]:
        tel = cls.CANALES_OFICIALES_BG["call_center"]
        ws = cls.CANALES_OFICIALES_BG["whatsapp_oficial"]

        if idioma == "en":
            titulo = "Urgent Support: Specialized Human Advisor Required."
            explicacion = (
                f"Due to the nature of your request, this case requires immediate verification with an official advisor. "
                f"Please contact our 24/7 Call Center at {tel} or our verified WhatsApp at {ws}. "
                f"You can also visit your nearest Banco Guayaquil branch for priority security assistance."
            )
        else:
            titulo = "Atención Especializada: Derivación a Asesor Humano de Banco Guayaquil."
            explicacion = (
                f"Por motivos de seguridad, confidencialidad y normativa bancaria, tu consulta no puede ser procesada "
                f"por este asistente automatizado. Por favor comunícate de inmediato con nuestros canales oficiales: "
                f"Call Center 24/7 al {tel}, WhatsApp verificado al {ws}, o acércate a la agencia Banco Guayaquil más cercana."
            )

        return cls._empaquetar_json_guardrail(
            cod_clte=cod_clte,
            titulo=titulo,
            explicacion=explicacion,
            tipo_guardrail="DERIVACION_ASESOR_HUMANO",
            acciones=[
                {"tipo": "LLAMAR_SOPORTE", "label": f"Llamar al Call Center ({tel[:10]})", "payload_tabla": "SISTEMA_SEGURIDAD_BG"},
                {"tipo": "WHATSAPP_OFICIAL", "label": "Abrir WhatsApp Oficial Banco Guayaquil", "payload_tabla": "SISTEMA_SEGURIDAD_BG"},
            ],
        )

    @classmethod
    def _crear_respuesta_exfiltracion_seguridad(cls, cod_clte: int, idioma: str) -> Dict[str, Any]:
        if idioma == "en":
            titulo = "Access Denied: Cybersecurity & Privacy Protection Protocol."
            explicacion = (
                "Due to strict banking cybersecurity standards, banking secrecy, and the Personal Data Protection Organic Law (LOPDP), "
                "this assistant has restricted access and cannot disclose API credentials, environment variables (.env), source code, "
                "or data belonging to other customers. Each banking session operates in an isolated and encrypted environment."
            )
        else:
            titulo = "Acceso Denegado: Protocolo de Ciberseguridad y Privacidad Bancaria."
            explicacion = (
                "Por estrictas políticas de ciberseguridad, secreto bancario y la Ley Orgánica de Protección de Datos Personales (LOPDP), "
                "este asistente tiene restringido el acceso a credenciales, variables de entorno (.env), código fuente de la app "
                "o información confidencial de otros clientes. Cada sesión opera en un entorno estrictamente aislado y auditado para tu seguridad."
            )

        return cls._empaquetar_json_guardrail(
            cod_clte=cod_clte,
            titulo=titulo,
            explicacion=explicacion,
            tipo_guardrail="SEGURIDAD_EXFILTRACION_BLOQUEADA",
            acciones=[
                {
                    "tipo": "CONSULTAR_MI_SALUD",
                    "label": "Consultar Mi Propia Salud Financiera",
                    "payload_tabla": "MAESTRO_CLIENTES",
                },
                {
                    "tipo": "VER_POLITICA_SEGURIDAD",
                    "label": "Ver Políticas de Ciberseguridad y LOPDP",
                    "payload_tabla": "SISTEMA_SEGURIDAD_BG",
                },
            ],
        )

    @classmethod
    def _crear_respuesta_seguridad(cls, cod_clte: int, idioma: str) -> Dict[str, Any]:
        titulo = "Instrucción no permitida por políticas de seguridad bancaria."
        explicacion = (
            "Esta solicitud infringe los protocolos de seguridad y gobernanza de Banco Guayaquil. "
            "Todas las evaluaciones crediticias se rigen estrictamente por nuestro modelo matemático "
            "y la normativa de la Junta de Política y Regulación Financiera del Ecuador."
        )
        return cls._empaquetar_json_guardrail(
            cod_clte=cod_clte,
            titulo=titulo,
            explicacion=explicacion,
            tipo_guardrail="SEGURIDAD_POLITICA",
            acciones=[],
        )

    @classmethod
    def _empaquetar_json_guardrail(
        cls,
        cod_clte: int,
        titulo: str,
        explicacion: str,
        tipo_guardrail: str,
        acciones: list,
    ) -> Dict[str, Any]:
        return {
            "transaccion_id": f"guardrail_bg_{tipo_guardrail.lower()}",
            "COD_CLTE": cod_clte,
            "evaluacion_riesgo_ml": {
                "score_salud_interna": 100 if tipo_guardrail == "SALUDO" else 0,
                "probabilidad_caer_en_vencido_mora": 0.0,
                "semaforo_salud_financiera": "INFORMATIVO",
                "flujo_neto_mensual_disponible": 0.0,
                "capacidad_maxima_nueva_cuota": 0.0,
                "endeudamiento_actual_pct": 0.0,
                "endeudamiento_con_simulacion_pct": 0.0,
                "motivo_guardrail": tipo_guardrail,
            },
            "dictamen_endeudamiento": {
                "se_puede_endeudar": False,
                "estado_resolucion": f"RESPUESTA_DIRECTA_{tipo_guardrail}",
                "monto_maximo_recomendado": 0.0,
                "cuota_mensual_segura": 0.0,
                "impacto_proyectado_equifax": "Sin afectacion a tu historial crediticio.",
                "factores_clave_modelo": ["Interaccion filtrada por Guardrails preventivos de Banco Guayaquil."],
            },
            "recomendaciones_productos_bg": {
                "1_oferta_maestro_creditos": {"habilitado": False},
                "2_oferta_maestro_polizas": {"habilitado": False},
                "3_oferta_maestro_metas": {"habilitado": False},
            },
            "agente_conversacional_foundry": {
                "titulo_pantalla": titulo,
                "explicacion_ia": explicacion,
                "origen_generacion": "GUARDRAILS_DETERMINISTAS_BG",
                "acciones_ui": acciones,
            },
        }

