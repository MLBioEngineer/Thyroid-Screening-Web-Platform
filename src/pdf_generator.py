"""
Clinical PDF Report Generator for Multimodal Thyroid Screening System
Department of Biomedical Engineering, Jashore University of Science and Technology (JUST)
"""

import io
import math
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

def _safe_xml(val) -> str:
    """Safely escape text for inclusion in ReportLab XML Paragraphs."""
    if val is None:
        return ""
    return xml_escape(str(val))

def generate_patient_pdf(
    patient_id: str,
    patient_name: str,
    age: float,
    gender: str,
    weight: str,
    tsh_val: str,
    ft4_val: str,
    ref_status: str,
    pred_label: str,
    prob_dict: dict,
    model_name: str,
    sensor_feats: dict,
    researcher_name: str = "Authorized Researcher",
    researcher_phone: str = "N/A",
    researcher_gender: str = "N/A",
    clinical_notes: str = ""
) -> bytes:
    """
    Generates a professional clinical PDF screening report.
    Returns bytes of the compiled PDF.
    """
    import re
    pid_str = str(patient_id).strip()
    m_num = re.search(r'\d+', pid_str)
    norm_pid = f"TH{int(m_num.group()):03d}" if (m_num and 'th' in pid_str.lower()) else pid_str
    safe_pid = _safe_xml(norm_pid)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
        title=f"Thyroid Screening Report - {safe_pid}",
        author="JUST BME Multimodal Lab"
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    teal_dark = colors.HexColor("#0a3d36")
    teal_mid = colors.HexColor("#0f594e")
    teal_light = colors.HexColor("#e6f4f1")
    teal_accent = colors.HexColor("#10b981")
    gray_bg = colors.HexColor("#f8fafc")
    gray_border = colors.HexColor("#cbd5e1")
    text_dark = colors.HexColor("#0f172a")
    text_muted = colors.HexColor("#475569")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=teal_dark,
        alignment=TA_CENTER
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=text_muted,
        alignment=TA_CENTER
    )

    section_heading = ParagraphStyle(
        'SecHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=teal_mid,
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=text_dark
    )

    body_bold = ParagraphStyle(
        'BodyBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=text_dark
    )

    small_style = ParagraphStyle(
        'SmallMuted',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=text_muted
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>JASHORE UNIVERSITY OF SCIENCE AND TECHNOLOGY</b><br/><font size='8' color='#475569'>Department of Biomedical Engineering &bull; Multimodal Bio-Signal Screening Lab</font>", title_style)
        ],
        [
            Paragraph("<b>MULTIMODAL THYROID SCREENING & SENSOR REPORT</b>", ParagraphStyle('ReportTag', fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=teal_mid, alignment=TA_CENTER))
        ]
    ]
    header_table = Table(header_data, colWidths=[540])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), teal_light),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOX', (0, 0), (-1, -1), 1.5, teal_dark),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))

    # Metadata Strip (Report ID, Date, Researcher)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_r_name = _safe_xml(researcher_name)
    safe_r_phone = _safe_xml(researcher_phone)
    safe_r_gender = _safe_xml(researcher_gender)
    meta_text = (
        f"<b>Report ID:</b> REP-{safe_pid}-{datetime.now().strftime('%Y%m%d%H%M')} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Generated:</b> {now_str} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Clinician/Researcher:</b> {safe_r_name} ({safe_r_phone}, {safe_r_gender})"
    )
    story.append(Paragraph(meta_text, small_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=gray_border, spaceBefore=2, spaceAfter=8))

    # 2. Patient Demographics & Reference Laboratory Data (De-identified & Privacy-Protected)
    story.append(Paragraph("1. PATIENT DEMOGRAPHICS & CLINICAL LABORATORY REFERENCE", section_heading))

    def _clean_str(val, fallback="Not Recorded"):
        if val is None:
            return fallback
        s = str(val).strip()
        if not s or any(k in s.lower() for k in ['nan', 'none', 'null', 'n/a', 'pending', 'not recorded']):
            return fallback
        if s.lower().startswith('nan'):
            return fallback
        return s

    def _safe_float(val, default=0.0):
        try:
            f = float(val)
            return default if (math.isnan(f) or math.isinf(f)) else f
        except:
            return default

    clean_gender = _clean_str(gender, fallback="Not Recorded")
    if clean_gender != "Not Recorded":
        clean_gender = clean_gender.capitalize()
    clean_weight = _clean_str(weight, fallback="Not Recorded")
    clean_tsh = _clean_str(tsh_val, fallback="Pending Evaluation")
    clean_ft4 = _clean_str(ft4_val, fallback="Pending Evaluation")
    clean_ref = _clean_str(ref_status, fallback="Pending Evaluation")

    # Format TSH display
    if clean_tsh != "Pending Evaluation" and not any(k in clean_tsh.lower() for k in ['pending', 'nan', 'none', 'null', 'n/a', 'not recorded']):
        if "miu/l" not in clean_tsh.lower():
            tsh_display = f"{_safe_xml(clean_tsh)} mIU/L"
        else:
            tsh_display = _safe_xml(clean_tsh)
    else:
        tsh_display = "Pending Evaluation"

    # Format FT4 display
    if clean_ft4 != "Pending Evaluation" and not any(k in clean_ft4.lower() for k in ['pending', 'nan', 'none', 'null', 'n/a', 'not recorded']):
        if "ng/dl" not in clean_ft4.lower():
            ft4_display = f"{_safe_xml(clean_ft4)} ng/dL"
        else:
            ft4_display = _safe_xml(clean_ft4)
    else:
        ft4_display = "Pending Evaluation"

    # Format Weight display
    if clean_weight != "Not Recorded" and not any(k in clean_weight.lower() for k in ['not recorded', 'nan', 'none', 'null', 'n/a', 'pending']):
        if "kg" not in clean_weight.lower():
            weight_display = f"{_safe_xml(clean_weight)} kg"
        else:
            weight_display = _safe_xml(clean_weight)
    else:
        weight_display = "Not Recorded"

    # Format Age display
    if age is not None and str(age).strip().lower() not in ['nan', 'none', 'null', '']:
        try:
            age_f = float(age)
            age_display = f"{age_f:.0f} Years" if not (math.isnan(age_f) or math.isinf(age_f)) else "Not Recorded"
        except:
            age_display = "Not Recorded"
    else:
        age_display = "Not Recorded"

    # Ensure patient display strictly adheres to de-identified privacy format
    deid_patient = f"Research Participant {safe_pid} (De-identified Study Record - Privacy Protected)"

    demo_data = [
        [
            Paragraph("<b>Patient:</b>", body_style), Paragraph(_safe_xml(deid_patient), body_bold),
            Paragraph("<b>Study ID:</b>", body_style), Paragraph(safe_pid, body_bold)
        ],
        [
            Paragraph("<b>Age:</b>", body_style), Paragraph(age_display, body_style),
            Paragraph("<b>Gender:</b>", body_style), Paragraph(_safe_xml(clean_gender), body_style)
        ],
        [
            Paragraph("<b>Weight:</b>", body_style), Paragraph(weight_display, body_style),
            Paragraph("<b>Clinical TSH:</b>", body_style), Paragraph(tsh_display, body_bold)
        ],
        [
            Paragraph("<b>Serum FT4:</b>", body_style), Paragraph(ft4_display, body_style),
            Paragraph("<b>Lab Benchmark Status:</b>", body_style), Paragraph(f"<b>{_safe_xml(clean_ref)}</b>", body_bold)
        ]
    ]
    demo_table = Table(demo_data, colWidths=[75, 245, 90, 130])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), gray_bg),
        ('GRID', (0, 0), (-1, -1), 0.5, gray_border),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(demo_table)
    story.append(Spacer(1, 10))

    # 3. Multimodal Bio-Signal Sensor Findings
    story.append(Paragraph("2. MULTIMODAL SENSOR BIOMARKER ANALYSIS", section_heading))

    tremor_8_12 = _safe_float(sensor_feats.get('tremor_amag_ratio_8_12', 0.0)) * 100
    tremor_freq = _safe_float(sensor_feats.get('tremor_amag_dom_freq', 0.0))
    tremor_rms = _safe_float(sensor_feats.get('tremor_amag_rms', 0.0))
    neck_mean = _safe_float(sensor_feats.get('neck_mean_mm', 0.0))
    neck_std = _safe_float(sensor_feats.get('neck_std_mm', 0.0))

    sensor_data = [
        [
            Paragraph("<b>Biophysical Biomarker</b>", body_bold),
            Paragraph("<b>Sensor / Modality</b>", body_bold),
            Paragraph("<b>Measured Value</b>", body_bold),
            Paragraph("<b>Physiological Reference Range</b>", body_bold)
        ],
        [
            Paragraph("Fine Tremor Power Ratio (8–12 Hz)", body_style),
            Paragraph("MPU6050 6-Axis IMU (100 Hz)", small_style),
            Paragraph(f"<b>{tremor_8_12:.1f}%</b>", body_bold),
            Paragraph("&lt; 25% Normal | &gt; 35% Elevated (Thyrotoxic)", small_style)
        ],
        [
            Paragraph("Dominant Tremor Frequency", body_style),
            Paragraph("MPU6050 Accelerometer FFT", small_style),
            Paragraph(f"<b>{tremor_freq:.1f} Hz</b>", body_bold),
            Paragraph("8.0–12.0 Hz indicates fine physiological tremor", small_style)
        ],
        [
            Paragraph("Tremor Motion RMS Acceleration", body_style),
            Paragraph("MPU6050 Tri-axial Accel", small_style),
            Paragraph(f"<b>{tremor_rms:.2f} m/s²</b>", body_bold),
            Paragraph("Baseline postural acceleration intensity", small_style)
        ],
        [
            Paragraph("Anterior Neck Surface Distance", body_style),
            Paragraph("VL53L0X Laser ToF Ranging", small_style),
            Paragraph(f"<b>{neck_mean:.1f} mm</b>", body_bold),
            Paragraph("Nominal 40–60 mm optical collar distance", small_style)
        ],
        [
            Paragraph("Neck Distance Dispersion (Std Dev)", body_style),
            Paragraph("VL53L0X 10 Repeat Readings", small_style),
            Paragraph(f"<b>&plusmn;{neck_std:.2f} mm</b>", body_bold),
            Paragraph("Low dispersion (&lt; 2.0 mm) ensures stable reading", small_style)
        ]
    ]
    sensor_table = Table(sensor_data, colWidths=[170, 130, 90, 150])
    sensor_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), teal_light),
        ('GRID', (0, 0), (-1, -1), 0.5, gray_border),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(sensor_table)
    story.append(Spacer(1, 10))

    # 4. Machine Learning Screening Inference
    story.append(Paragraph("3. MACHINE LEARNING SCREENING DECISION", section_heading))

    pred_color = "#10b981" if pred_label == "Normal" else ("#8b5cf6" if pred_label == "Hyperthyroid" else "#f59e0b")
    prob_norm = prob_dict.get("Normal", 0.0) * 100
    prob_hyper = prob_dict.get("Hyperthyroid", 0.0) * 100
    prob_hypo = prob_dict.get("Hypothyroid", 0.0) * 100

    if pred_label == "Normal":
        triage_text = "LOW RISK &bull; Normal bio-signal and tremor profile. Routine follow-up recommended."
    elif pred_label == "Hyperthyroid":
        triage_text = "ELEVATED RISK &bull; Significant 8-12 Hz fine tremor detected. Recommend serum TSH/FT4 radioimmunoassay."
    else:
        triage_text = "ELEVATED RISK &bull; Anterior neck surface / hypothyroid profile detected. Recommend clinical endocrine consult."

    safe_model = _safe_xml(model_name)
    ml_data = [
        [
            Paragraph(f"<b>Classifier Model:</b> {safe_model}", body_style),
            Paragraph(f"<b>Prediction:</b> <font color='{pred_color}' size='11'><b>{pred_label.upper()}</b></font>", body_bold)
        ],
        [
            Paragraph(f"<b>Multimodal Probability Distribution:</b><br/>Normal: <b>{prob_norm:.1f}%</b> &nbsp;|&nbsp; Hyperthyroid: <b>{prob_hyper:.1f}%</b> &nbsp;|&nbsp; Hypothyroid: <b>{prob_hypo:.1f}%</b>", body_style),
            Paragraph(f"<b>Clinical Triage Action:</b><br/>{triage_text}", small_style)
        ]
    ]
    ml_table = Table(ml_data, colWidths=[270, 270])
    ml_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), gray_bg),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(pred_color)),
        ('GRID', (0, 0), (-1, -1), 0.5, gray_border),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(ml_table)
    story.append(Spacer(1, 10))

    # 5. Clinical Impression & Attestation
    story.append(Paragraph("4. RESEARCHER & CLINICIAN REVIEW", section_heading))

    raw_notes = clinical_notes if clinical_notes.strip() else "Sensor recordings and ML inference reviewed. Findings are consistent with preliminary non-invasive screening protocol."
    safe_notes = _safe_xml(raw_notes)

    review_data = [
        [
            Paragraph(f"<b>Researcher Impression / Clinical Notes:</b><br/>{safe_notes}", body_style)
        ]
    ]
    review_table = Table(review_data, colWidths=[540])
    review_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('BOX', (0, 0), (-1, -1), 0.5, gray_border),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(review_table)
    story.append(Spacer(1, 15))

    # Signatures Table
    sig_data = [
        [
            Paragraph("____________________________<br/><b>Lead Screening Bioengineer</b><br/><font size='7' color='#475569'>JUST BME Multimodal Lab</font>", small_style),
            Paragraph("____________________________<br/><b>Supervising Investigator</b><br/><font size='7' color='#475569'>Dr. Hassan Mohammad Al-Emran</font>", small_style),
            Paragraph("____________________________<br/><b>Attending Medical Officer</b><br/><font size='7' color='#475569'>Endocrinology / Clinical Review</font>", small_style)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[180, 180, 180])
    sig_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'BOTTOM'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(sig_table)
    story.append(Spacer(1, 8))

    # 6. Scope & Safety Disclaimer
    disclaimer_text = (
        "<b>CLINICAL SAFETY ADVISORY:</b> This report is generated by an experimental research IoT device developed at the Department of Biomedical Engineering, Jashore University of Science and Technology (JUST). It does NOT measure internal thyroid gland volume or replace definitive clinical pathology testing (serum TSH/FT4 radioimmunoassays). Prospective decision-support only."
    )
    story.append(HRFlowable(width="100%", thickness=0.5, color=gray_border, spaceBefore=4, spaceAfter=4))
    story.append(Paragraph(disclaimer_text, ParagraphStyle('Disc', fontName='Helvetica-Oblique', fontSize=7, leading=9, textColor=colors.HexColor("#881337"))))

    doc.build(story)
    return buf.getvalue()
