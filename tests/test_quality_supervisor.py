"""
Tests para QualitySupervisorAgent (Agent-in-the-Middle y Checklist de 5 Puntos)
Verifica sanitización de jerga técnica SQL, stripping de emojis, cumplimiento del checklist
y la auditoría de calidad end-to-end con la consulta del usuario 'y puedo hacer un prestamo de 1000?'.
"""

import json
from pathlib import Path
import pytest

from src.inference.quality_supervisor import QualitySupervisorAgent
from motor_ia_foundry import MotorIAFoundry


def test_supervisor_sanitiza_tablas_en_borrador():
    texto_con_tablas = (
        "En su lugar, programa una Meta de Ahorro de $60/mes en MAESTRO_METAS para comprarlo al contado. "
        "Revisa tus transacciones en MVTO_CUENTAS e invierte tu excedente en MAESTRO_POLIZAS."
    )
    texto_limpio, ajustes = QualitySupervisorAgent.sanitizar_texto_comercial(texto_con_tablas)

    assert "MAESTRO_METAS" not in texto_limpio
    assert "MVTO_CUENTAS" not in texto_limpio
    assert "MAESTRO_POLIZAS" not in texto_limpio
    assert "Metas de Ahorro en tu App Banco Guayaquil" in texto_limpio
    assert "tus movimientos de cuenta" in texto_limpio
    assert "Póliza a Plazo Fijo en Banco Guayaquil" in texto_limpio
    assert len(ajustes) == 3


def test_supervisor_elimina_emojis():
    texto_con_emojis = "Tu solicitud ha sido evaluada exitosamente 🚀🏦. Tienes liquidez disponible 💰✨."
    texto_limpio, ajustes = QualitySupervisorAgent.sanitizar_texto_comercial(texto_con_emojis)

    assert "🚀" not in texto_limpio
    assert "🏦" not in texto_limpio
    assert "💰" not in texto_limpio
    assert "✨" not in texto_limpio
    assert "Tu solicitud ha sido evaluada exitosamente. Tienes liquidez disponible." == texto_limpio
    assert any("emojis" in a for a in ajustes)


def test_supervisor_aprueba_directo_si_cumple_checklist():
    supervisor = QualitySupervisorAgent(foundry_client=None)

    peticion = "y puedo hacer un prestamo de 1000?"
    titulo = "Préstamo de $1,000 viable en Banco Guayaquil"
    explicacion = (
        "Analicé tus cuentas de Banco Guayaquil y tu score Equifax de 785 puntos. "
        "Cuentas con $322.50 libres al mes y la nueva cuota de $90.49 encaja con holgura en tu presupuesto. "
        "Puedes gestionar la solicitud formal desde tu App Banco Guayaquil."
    )
    contexto_ml = {
        "semaforo": "VERDE",
        "cuota_nueva": 90.49,
        "flujo_neto_actual": 322.50,
    }

    tit_final, exp_final, dto = supervisor.auditar_y_refinar_respuesta(
        peticion_usuario=peticion,
        titulo_propuesto=titulo,
        explicacion_propuesta=explicacion,
        contexto_ml=contexto_ml,
        intencion_nlu="ENDEUDAMIENTO",
        es_solicitud_deuda_nueva=True,
    )

    assert dto.veredicto_auditoria == "APROBADO_DIRECTO"
    assert all(item.cumple for item in dto.checklist_calidad.values())
    assert "MAESTRO_METAS" not in exp_final


def test_supervisor_detecta_y_sanitiza_falla_en_checklist():
    supervisor = QualitySupervisorAgent(foundry_client=None)

    peticion = "y puedo hacer un prestamo de 1000?"
    # Borrador defectuoso como el que recibió el usuario originalmente
    titulo = "Alerta: No endeudarte ahora"
    explicacion_defectuosa = (
        "Te recomendamos que no hagas eso en este momento porque tu flujo neto es de $194.66/mes. "
        "En su lugar, programa una Meta de Ahorro de $60/mes en MAESTRO_METAS para comprarlo al contado sin intereses. 💡"
    )
    contexto_ml = {
        "semaforo": "ROJO",
        "cuota_nueva": 90.49,
        "flujo_neto_actual": 194.66,
    }

    tit_final, exp_final, dto = supervisor.auditar_y_refinar_respuesta(
        peticion_usuario=peticion,
        titulo_propuesto=titulo,
        explicacion_propuesta=explicacion_defectuosa,
        contexto_ml=contexto_ml,
        intencion_nlu="ENDEUDAMIENTO",
        es_solicitud_deuda_nueva=True,
    )

    # El veredicto debe ser APROBADO_SANITIZADO porque corrigió las fallas
    assert dto.veredicto_auditoria == "APROBADO_SANITIZADO"
    assert dto.checklist_calidad["1_vocabulario_comercial_limpio"].cumple is False
    assert dto.checklist_calidad["5_tono_cero_emojis_y_longitud"].cumple is False
    assert "MAESTRO_METAS" not in exp_final
    assert "Metas de Ahorro en tu App Banco Guayaquil" in exp_final
    assert "💡" not in exp_final


def test_consulta_usuario_prestamo_1000_end_to_end():
    """Prueba integral de la pregunta exacta del usuario: 'y puedo hacer un prestamo de 1000?'."""
    motor = MotorIAFoundry()
    with open(Path(__file__).resolve().parent.parent / "data" / "samples" / "sample_input.json", "r", encoding="utf-8") as f:
        cliente = json.load(f)

    # 1. Caso estándar (perfil solvente)
    res = motor.evaluar_peticion_cliente("y puedo hacer un prestamo de 1000?", cliente)
    conv = res["agente_conversacional_foundry"]
    exp = conv["explicacion_ia"]
    tit = conv["titulo_pantalla"]
    audit = res["auditoria_supervisor"]

    # Validaciones obligatorias de calidad
    assert "MAESTRO_METAS" not in exp
    assert "MAESTRO_METAS" not in tit
    assert "MAESTRO_POLIZAS" not in exp
    assert "MVTO_CUENTAS" not in exp
    assert audit is not None
    assert audit["checklist_calidad"]["1_vocabulario_comercial_limpio"]["cumple"] is True

    # 2. Caso ajustado / denegado (perfil con poco flujo libre que da semáforo ROJO)
    cliente_rojo = json.loads(json.dumps(cliente))
    cliente_rojo["resumen_cuentas_y_movimientos"]["agregado_mvto_cuentas_mensual"]["promedio_egresos_debito_mes"] = 1400.00
    res_rojo = motor.evaluar_peticion_cliente("y puedo hacer un prestamo de 1000?", cliente_rojo)
    exp_rojo = res_rojo["agente_conversacional_foundry"]["explicacion_ia"]
    tit_rojo = res_rojo["agente_conversacional_foundry"]["titulo_pantalla"]

    assert "MAESTRO_METAS" not in exp_rojo
    assert "MAESTRO_METAS" not in tit_rojo
    assert "MAESTRO_POLIZAS" not in exp_rojo
    assert "MVTO_CUENTAS" not in exp_rojo
    # Debe orientar a ahorro o solución de pago sin jerga de tablas
    assert any(term in exp_rojo.lower() for term in ["meta de ahorro", "ahorr", "soluciones de pago", "alivio", "reestructur"])
