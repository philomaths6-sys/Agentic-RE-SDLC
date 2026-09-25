# pdf_generator.py
"""Automated PDF Generator for Financial Requirements Engineering & SDLC Analysis.
Built with ReportLab, producing a multi-page executive document with:
- Professional styling, headers, and footers
- Structured metadata tables
- 30-Point Categorized Requirements Matrix (Frontend, Backend, Auth, Security, Database, Resilience)
- Regulatory compliance mapping (RBI, NPCI, DPDP, UIDAI, SEBI)
- MoSCoW prioritization & Risk scoring
- Detailed architectural controls
- Traceability matrix & SDLC roadmap
"""

import os
from typing import Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
    PageBreak,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw total page numbers and headers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "AGENTIC RE-SDLC ADVISOR — COMPLETE SOFTWARE REQUIREMENTS SPECIFICATION")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer (all pages)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_str)
        self.drawString(54, 36, "Confidential — Indian Banking & Financial Systems Architecture")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)

        self.restoreState()


def generate_pdf_report(analysis_data: Dict[str, Any], output_path: str = "output/requirements_analysis.pdf") -> str:
    """Generate a high-density, multi-page professional PDF report for the 30 requirements."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    primary_color = colors.HexColor("#0F172A")
    secondary_color = colors.HexColor("#1E3A8A")
    accent_color = colors.HexColor("#0284C7")
    text_color = colors.HexColor("#1E293B")
    border_color = colors.HexColor("#CBD5E1")

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=primary_color,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=secondary_color,
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "Heading1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "Heading2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=secondary_color,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=text_color,
    )
    table_hdr_style = ParagraphStyle(
        "TableHdr",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=text_color,
    )
    risk_lvl = analysis_data.get("risk_level", "MEDIUM")
    risk_score = analysis_data.get("risk_score_pct", 70)
    risk_summary = analysis_data.get("risk_summary", "Multi-factor statutory and operational risk evaluation.")

    risk_colors_map = {
        "CRITICAL": colors.HexColor("#B91C1C"),
        "HIGH": colors.HexColor("#C2410C"),
        "MEDIUM": colors.HexColor("#B45309"),
        "LOW / MODERATE": colors.HexColor("#047857"),
    }
    risk_color = risk_colors_map.get(risk_lvl, colors.HexColor("#B91C1C"))

    dynamic_tag_style = ParagraphStyle(
        "DynamicTagStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=risk_color,
    )

    story = []

    # 1. Title Banner
    story.append(Paragraph(analysis_data.get("title", "Comprehensive Financial SRS Specification"), title_style))
    story.append(Paragraph(f"Domain: {analysis_data.get('domain', 'Indian Banking & Financial Services')} | Statutory Architecture", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceBefore=0, spaceAfter=10))

    # 2. Executive Metadata Summary Table
    total_req_count = len(analysis_data.get("requirements_list", []))
    meta_data = [
        [
            Paragraph("<b>Parameter</b>", table_hdr_style),
            Paragraph("<b>Evaluation Result</b>", table_hdr_style),
            Paragraph("<b>Governance & Statutory Impact</b>", table_hdr_style),
        ],
        [
            Paragraph("<b>MoSCoW Baseline</b>", table_cell_style),
            Paragraph(f"<b>{analysis_data.get('moscow_priority', 'Must Have')}</b>", table_cell_style),
            Paragraph(analysis_data.get("priority_justification", "Mandatory statutory compliance"), table_cell_style),
        ],
        [
            Paragraph("<b>Risk Classification</b>", table_cell_style),
            Paragraph(f"<b>{risk_lvl} ({risk_score}/100)</b>", dynamic_tag_style),
            Paragraph(risk_summary, table_cell_style),
        ],
        [
            Paragraph("<b>Total Requirements</b>", table_cell_style),
            Paragraph(f"<b>{total_req_count} Modular Requirements (Min 30)</b>", table_cell_style),
            Paragraph("Decomposed across Frontend, Backend, Auth, Security, Database Residency, and Resilience", table_cell_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[110, 140, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 3. Original Engineered Requirement Prompt
    story.append(Paragraph("1. System Scope & Stakeholder Requirement Statement", h1_style))
    import html
    raw_req = str(analysis_data.get('original_requirement', '')).strip()
    safe_req = html.escape(raw_req)

    # Cap callout table cell so it never exceeds frame height (prevents ReportLab LayoutError)
    max_callout_len = 450
    if len(safe_req) > max_callout_len:
        callout_text = safe_req[:max_callout_len] + "..."
    else:
        callout_text = safe_req

    req_box = Table([[Paragraph(f"<i>\"{callout_text}\"</i>", body_style)]], colWidths=[504])
    req_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LINELEFT', (0, 0), (0, -1), 3, accent_color),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
    ]))
    story.append(req_box)
    story.append(Spacer(1, 8))

    # If requirement text was truncated in the callout box, render full text as standard paragraphs (which paginate freely)
    if len(safe_req) > max_callout_len:
        story.append(Paragraph(f"<b>Extended Document Context:</b> {safe_req[:2500]}", body_style))
        story.append(Spacer(1, 8))

    # 4. Comprehensive 30-Point Requirements Matrix
    reqs_list = analysis_data.get("requirements_list", [])
    if reqs_list:
        story.append(Paragraph(f"2. Modular Software Requirements Specification ({len(reqs_list)} Detailed Points)", h1_style))

        categories = {}
        for r in reqs_list:
            cat = r.get("category", "General")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        for cat_name, cat_reqs in categories.items():
            story.append(Paragraph(f"<b>Pillar: {cat_name} ({len(cat_reqs)} Specifications)</b>", h2_style))

            cat_rows = [
                [
                    Paragraph("<b>Req ID</b>", table_hdr_style),
                    Paragraph("<b>Title & Detailed Specification</b>", table_hdr_style),
                    Paragraph("<b>Priority</b>", table_hdr_style),
                    Paragraph("<b>Statutory Citation & Acceptance Criteria</b>", table_hdr_style),
                ]
            ]
            for cr in cat_reqs:
                desc_text = f"<b>{cr.get('title', '')}</b><br/>{cr.get('description', '')}"
                rule_text = f"<b>Regulation:</b> {cr.get('regulation', '')}<br/><b>Acceptance:</b> <i>{cr.get('acceptance_criteria', '')}</i>"
                cat_rows.append([
                    Paragraph(f"<b>{cr.get('id', '')}</b>", table_cell_style),
                    Paragraph(desc_text, table_cell_style),
                    Paragraph(f"<b>{cr.get('priority', 'Must Have')}</b>", table_cell_style),
                    Paragraph(rule_text, table_cell_style),
                ])

            cat_table = Table(cat_rows, colWidths=[65, 175, 55, 209])
            cat_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), secondary_color),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, border_color),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
            ]))
            story.append(cat_table)
            story.append(Spacer(1, 8))

    story.append(PageBreak())

    # 5. Regulatory Grounding Table (RAG Verified)
    story.append(Paragraph("3. Regulatory & Statutory Grounding (Indian Banking Frameworks)", h1_style))
    reg_rows = [
        [
            Paragraph("<b>Framework</b>", table_hdr_style),
            Paragraph("<b>Enforcement Level</b>", table_hdr_style),
            Paragraph("<b>Statutory Directive / Citation</b>", table_hdr_style),
        ]
    ]
    for r in analysis_data.get("regulations", []):
        stat_cite = r.get("statutory_citation") or "RBI/NPCI Standard Directive"
        reg_rows.append([
            Paragraph(f"<b>{r.get('regulation', '')}</b>", table_cell_style),
            Paragraph(r.get("relevance", "Mandatory Statutory Enforcement"), table_cell_style),
            Paragraph(stat_cite, table_cell_style),
        ])
    reg_table = Table(reg_rows, colWidths=[140, 120, 244])
    reg_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
    ]))
    story.append(reg_table)
    story.append(Spacer(1, 10))

    # 6. Technical Controls Breakdown
    story.append(Paragraph("4. Mandatory Security & Architectural Controls", h1_style))
    ctrl_rows = [
        [
            Paragraph("<b>Control ID</b>", table_hdr_style),
            Paragraph("<b>Title & Architecture Specification</b>", table_hdr_style),
            Paragraph("<b>Automated Verification Method</b>", table_hdr_style),
        ]
    ]
    for c in analysis_data.get("technical_controls", []):
        spec_text = f"<b>{c.get('title', '')}</b><br/>{c.get('specification', '')}"
        ctrl_rows.append([
            Paragraph(f"<b>{c.get('control_id', '')}</b>", table_cell_style),
            Paragraph(spec_text, table_cell_style),
            Paragraph(c.get("verification", ""), table_cell_style),
        ])
    ctrl_table = Table(ctrl_rows, colWidths=[80, 240, 184])
    ctrl_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), secondary_color),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
    ]))
    story.append(ctrl_table)
    story.append(Spacer(1, 10))

    # 7. Traceability Matrix & SDLC Roadmap
    story.append(Paragraph("5. Traceability Matrix & Recommended SDLC Sprints", h1_style))
    trace_rows = [
        [
            Paragraph("<b>Req IDs</b>", table_hdr_style),
            Paragraph("<b>Target Sub-Module</b>", table_hdr_style),
            Paragraph("<b>SDLC Phase / Gate</b>", table_hdr_style),
            Paragraph("<b>Verification Deliverable</b>", table_hdr_style),
        ]
    ]
    for t in analysis_data.get("traceability_matrix", []):
        trace_rows.append([
            Paragraph(f"<b>{t.get('req_id', '')}</b>", table_cell_style),
            Paragraph(t.get("sub_module", ""), table_cell_style),
            Paragraph(t.get("sdlc_phase", ""), table_cell_style),
            Paragraph(f"{t.get('target_artefact', '')}<br/><i>{t.get('verification_method', '')}</i>", table_cell_style),
        ])
    trace_table = Table(trace_rows, colWidths=[80, 130, 120, 174])
    trace_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
    ]))
    story.append(trace_table)

    # Build document
    try:
        doc.build(story, canvasmaker=NumberedCanvas)
    except Exception as exc:
        print(f"[WARN] Error during PDF build: {exc}. Retrying fallback build...")
        doc.build(story)
    return output_path


if __name__ == "__main__":
    from requirements_analysis import analyze_requirement
    sample_text = (
        "Build a UPI AutoPay recurring payment application with biometric 2FA, "
        "Aadhaar Data Vault masking, and 100% RBI payment data localization on Indian soil."
    )
    analysis = analyze_requirement(sample_text)
    out_pdf = generate_pdf_report(analysis, "output/requirements_analysis.pdf")
    print(f"PDF generated successfully at: {out_pdf} (Size: {os.path.getsize(out_pdf)} bytes)")
