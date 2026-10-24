"""
Generador de Estado de Cuenta Externo Realista en PDF para Pruebas
Proyecto: Copiloto de Salud Financiera - Banco Guayaquil
"""

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generar_estado_cuenta_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    # Estilos sobrios bancarios
    title_style = ParagraphStyle(
        "BankTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1A365D"),
    )
    subtitle_style = ParagraphStyle(
        "BankSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
    )
    header_style = ParagraphStyle(
        "HeaderStyle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
    )
    normal_style = ParagraphStyle(
        "NormalText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2D3748"),
    )
    bold_style = ParagraphStyle(
        "BoldText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1A202C"),
    )

    story = []

    # Encabezado institucional
    story.append(Paragraph("BANCO PICHINCHA C.A. — ESTADO DE CUENTA INTEGRAL", title_style))
    story.append(Paragraph("RUC: 1790010937001 | Matriz: Av. Amazonas y Pereira, Quito - Ecuador", subtitle_style))
    story.append(Paragraph("Canales de atención: 1700 800 800 | www.pichincha.com", subtitle_style))
    story.append(Spacer(1, 14))

    # Datos del cliente
    info_cliente = [
        [
            Paragraph("<b>Titular:</b> JUAN CARLOS RENGIFO", normal_style),
            Paragraph("<b>Identificación:</b> 0923456789", normal_style),
        ],
        [
            Paragraph("<b>Cuenta Corriente / Ahorros:</b> 2100492811", normal_style),
            Paragraph("<b>Período liquidado:</b> 01/10/2026 al 31/10/2026", normal_style),
        ],
        [
            Paragraph("<b>Dirección:</b> Guayaquil, Ecuador", normal_style),
            Paragraph("<b>Fecha de emisión:</b> 24/10/2026", normal_style),
        ],
    ]
    t_cliente = Table(info_cliente, colWidths=[3.5 * inch, 3.5 * inch])
    t_cliente.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EDF2F7")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_cliente)
    story.append(Spacer(1, 14))

    # Seccion 1: Resumen de Obligaciones y Creditos Activos en otra entidad
    story.append(Paragraph("DETALLE DE OBLIGACIONES CREDITICIAS ACTIVAS (PASIVOS EXTERNOS)", header_style))
    story.append(Spacer(1, 6))

    tabla_creditos = [
        [
            Paragraph("<b>Tipo de Operación</b>", bold_style),
            Paragraph("<b>No. Operación</b>", bold_style),
            Paragraph("<b>Saldo Capital</b>", bold_style),
            Paragraph("<b>Tasa Anual</b>", bold_style),
            Paragraph("<b>Cuota Mensual Fija</b>", bold_style),
            Paragraph("<b>Estado</b>", bold_style),
        ],
        [
            Paragraph("Préstamo Automotriz / Prendario", normal_style),
            Paragraph("OP-PRE-90214", normal_style),
            Paragraph("$6,400.00 USD", normal_style),
            Paragraph("15.20%", normal_style),
            Paragraph("<b>$320.00 USD</b>", bold_style),
            Paragraph("AL DÍA", normal_style),
        ],
    ]
    t_cred = Table(tabla_creditos, colWidths=[2.2 * inch, 1.1 * inch, 1.1 * inch, 0.9 * inch, 1.2 * inch, 0.9 * inch])
    t_cred.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("ALIGN", (2, 1), (4, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t_cred)
    story.append(Spacer(1, 14))

    # Seccion 2: Movimientos del Mes con Debito de la Cuota
    story.append(Paragraph("MOVIMIENTOS DE LA CUENTA (DÉBITOS Y CRÉDITOS DEL MES)", header_style))
    story.append(Spacer(1, 6))

    movimientos = [
        [
            Paragraph("<b>Fecha</b>", bold_style),
            Paragraph("<b>Concepto / Descripción</b>", bold_style),
            Paragraph("<b>Referencia</b>", bold_style),
            Paragraph("<b>Débito (USD)</b>", bold_style),
            Paragraph("<b>Crédito (USD)</b>", bold_style),
            Paragraph("<b>Saldo (USD)</b>", bold_style),
        ],
        [
            Paragraph("02/10/2026", normal_style),
            Paragraph("ABONO TRANSFERENCIA CLIENTE EXTERNO", normal_style),
            Paragraph("TRX-00129", normal_style),
            Paragraph("-", normal_style),
            Paragraph("$180.00", normal_style),
            Paragraph("$430.00", normal_style),
        ],
        [
            Paragraph("05/10/2026", normal_style),
            Paragraph("<b>DÉBITO AUTOMÁTICO CUOTA PRÉSTAMO VEHICULAR</b>", normal_style),
            Paragraph("DB-CUOTA-320", normal_style),
            Paragraph("<b>$320.00</b>", bold_style),
            Paragraph("-", normal_style),
            Paragraph("$110.00", normal_style),
        ],
        [
            Paragraph("15/10/2026", normal_style),
            Paragraph("RETIRO EN CAJERO AUTOMÁTICO BANRED", normal_style),
            Paragraph("ATM-99410", normal_style),
            Paragraph("$50.00", normal_style),
            Paragraph("-", normal_style),
            Paragraph("$60.00", normal_style),
        ],
    ]
    t_movs = Table(movimientos, colWidths=[0.9 * inch, 2.7 * inch, 1.0 * inch, 1.0 * inch, 0.9 * inch, 0.9 * inch])
    t_movs.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#FFF5F5")),  # Resaltar fila de cuota
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_movs)
    story.append(Spacer(1, 16))

    # Pie de pagina institucional
    story.append(
        Paragraph(
            "<b>Nota de Auditoría y Verificación:</b> Este documento certifica que el cliente mantiene una cuota crediticia externa "
            "fija mensual de $320.00 USD correspondiente a su préstamo prendario en Banco Pichincha, con 20 cuotas pendientes por amortizar. "
            "Certificación emitida conforme a las regulaciones de la Superintendencia de Bancos del Ecuador.",
            subtitle_style,
        )
    )

    doc.build(story)
    print(f"PDF generado exitosamente en: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "samples" / "estado_cuenta_externo_demo.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    generar_estado_cuenta_pdf(out)

