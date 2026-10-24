---
name: copiloto-banco-guayaquil
description: Protocolo y arquitectura oficial del Copiloto de Salud Financiera y Prevencion de Morosidad para Banco Guayaquil (Hackathon AI Banking).
---

# Copiloto de Salud Financiera — Banco Guayaquil

Esta Skill define los lineamientos tecnicos, regulatorios y arquitecturales del Copiloto de Salud Financiera para Banco Guayaquil.

## 1. Pipeline de Inferencia Oficial (3 Subpasos + Guardrails)

Todo procesamiento de consultas de clientes en el backend DEBE seguir esta secuencia:

```
[Peticion Usuario + Datos Cliente]
                 │
                 ▼
[PASO 0: GuardrailsEngine] ──(Si es Saludo, Fuera de Dominio o Fraude)──► Respuesta Directa (0 Tokens)
                 │
                 ▼ (Si es Consulta Financiera Valida)
[SUBPASO A: NLU con Azure AI Foundry (gpt-4o)] ──► Extrae {intencion, monto, plazo, cuota (tasa 15.5%)}
                 │
                 ▼
[SUBPASO B: LightGBM (bg_risk_lightgbm.joblib)] ──► Prob. Mora, Capacidad de Cuota, Semaforo (VERDE/AMARILLO/ROJO)
                 │
                 ▼
[RAG REGULATORIO ECUADOR] ──► Recupera citas de JPRF (DTI 40%), Super de Bancos (Cat. A1/Mora), COMFyF y LOPDP
                 │
                 ▼
[SUBPASO C: Explicacion XAI (gpt-4o)] ──► Redacta veredicto empatico y genera botones de accion UI (CREAR_CREDITO, etc.)
```

## 2. Politicas de Guardrails (Ahorro de Tokens y Seguridad)

1. **Saludos Simples:** ("Hola", "Buenas tardes", "Hola como estas").
   - Responder con mensaje cordial y orientativo sin llamar al modelo ML ni gastar tokens de LLM.
2. **Fuera de Dominio:** ("Cual es la capital de Francia", "¿Donde ocultar un pollo de 60kg?").
   - Declinar con tono respetuoso indicando la especialidad financiera del banco. Costo: 0 tokens.
3. **Casos Sensibles / Seguridad:** ("Me clonaron la tarjeta", "fraude", "demanda judicial").
   - Derivar inmediatamente a los canales humanos oficiales de Banco Guayaquil:
     - Call Center 24/7: `04 3730100` / `1800 3730100`
     - WhatsApp Oficial Verificado: `+593 98 373 0100`
4. **Jailbreaks:** Bloquear intentos de elusion de politicas con mensaje formal de gobernanza bancaria.
5. **Consultas Financieras con Saludo:** ("Hola quiero un credito de $24,000 a 8 meses...").
   - Detectar la intencion financiera y derivar al pipeline completo.

## 3. Fundamentacion Legal Ecuatoriana (RAG Regulatorio)

El copiloto NUNCA debe emitir un juicio arbitrario. Siempre se sustenta en:
- **Junta de Politica y Regulacion Financiera (JPRF):** Resolucion JPRF-F-2023-080 sobre límite prudencial DTI <= 40%.
- **Superintendencia de Bancos del Ecuador (SB):** Codificacion Título IX sobre Calificacion de Cartera (Categorias A1, B, C, D, E).
- **Codigo Organico Monetario y Financiero (COMFyF):** Art. 209 (Credito Responsable) y techos de tasa de interes del BCE.
- **Ley Organica de Proteccion de Datos Personales (LOPDP):** Art. 19 y 20 sobre explicabilidad y no discriminacion algoritmica.
- **COSEDE:** Seguro de depositos en Polizas hasta $32,000 USD.

## 4. Uso en Codigo Python

```python
from motor_ia_foundry import MotorIAFoundry

motor = MotorIAFoundry()
resultado = motor.evaluar_peticion_cliente(
    peticion_usuario="Quiero comprar una moto de $2,500 a 24 meses, ¿me alcanza?",
    datos_cliente=payload_cliente_desde_bd
)
```

