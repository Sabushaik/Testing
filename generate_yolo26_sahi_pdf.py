#!/usr/bin/env python3
"""
Generate a comprehensive PDF document about YOLO26+SAHI
with image comparisons and detailed explanations.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    Table, TableStyle, KeepTogether
)
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from PIL import Image as PILImage
import os


def create_yolo26_sahi_pdf():
    """Create the YOLO26+SAHI PDF document"""
    
    # Create PDF document
    pdf_filename = "YOLO26+SAHI.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#1a472a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=HexColor('#2e7d32'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=HexColor('#388e3c'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
        leading=14
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['BodyText'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=6,
        leading=13
    )
    
    # Title Page
    elements.append(Spacer(1, 1*inch))
    elements.append(Paragraph("📄 Detection Accuracy Enhancement Using", title_style))
    elements.append(Paragraph("YOLO26m + SAHI", title_style))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(
        "Manufacturing Surveillance Person Detection System",
        ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=14, 
                      alignment=TA_CENTER, textColor=colors.grey)
    ))
    elements.append(Spacer(1, 2*inch))
    
    # Objective Section
    elements.append(Paragraph("🎯 Objective", heading1_style))
    elements.append(Paragraph(
        "The goal of this task was to <b>improve person detection accuracy and reliability</b> "
        "in manufacturing surveillance streams.",
        body_style
    ))
    elements.append(Paragraph(
        "As part of continuous system optimization, we evaluated detection performance and "
        "upgraded our pipeline from <b>YOLOv8 (YOLOv8l)</b> to <b>YOLO26 (YOLO26m) + SAHI</b>.",
        body_style
    ))
    elements.append(Paragraph(
        "This migration resulted in <b>higher recall, better small-object detection, and more "
        "stable real-time inference</b>.",
        body_style
    ))
    elements.append(Spacer(1, 0.3*inch))
    
    elements.append(PageBreak())
    
    # Model Evolution Overview
    elements.append(Paragraph("🔁 Model Evolution Overview", heading1_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Earlier Approach - YOLOv8l
    elements.append(Paragraph("🔹 Earlier Approach — YOLOv8l", heading2_style))
    
    elements.append(Paragraph("<b>Observed Behavior:</b>", body_style))
    elements.append(Paragraph("• Good general detection performance", bullet_style))
    elements.append(Paragraph("• Accurate for medium/large persons", bullet_style))
    elements.append(Paragraph("• Real-time capable", bullet_style))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("<b>Gaps noticed during evaluation:</b>", body_style))
    elements.append(Paragraph("• Reduced detection of <b>far-distance/tiny persons</b>", bullet_style))
    elements.append(Paragraph("• Missed <b>partially occluded workers</b>", bullet_style))
    elements.append(Paragraph("• Lower recall in <b>crowded scenes</b>", bullet_style))
    elements.append(Paragraph("• Detail loss when processing high-resolution frames", bullet_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Upgraded Approach - YOLO26m + SAHI
    elements.append(Paragraph("🔹 Upgraded Approach — YOLO26m + SAHI", heading2_style))
    
    elements.append(Paragraph("<b>Outcome After Upgrade:</b>", body_style))
    elements.append(Paragraph("• More persons detected at long distances", bullet_style))
    elements.append(Paragraph("• Better recognition of tiny targets", bullet_style))
    elements.append(Paragraph("• Stable detection in dense environments", bullet_style))
    elements.append(Paragraph("• Higher overall recall", bullet_style))
    elements.append(Paragraph("• Consistent real-time inference", bullet_style))
    
    elements.append(PageBreak())
    
    # Image Comparisons Section
    elements.append(Paragraph("📊 Visual Performance Comparison", heading1_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Comparison 1: GEN2 Dataset
    elements.append(Paragraph("<b>Comparison 1: GEN2 Dataset</b>", heading2_style))
    elements.append(Paragraph(
        "The following images demonstrate the detection improvement on the GEN2 surveillance dataset. "
        "Notice the enhanced detection of distant and small objects with YOLO26m + SAHI.",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    # Add GEN2 comparison images
    try:
        img1_path = "GEN2_WITH_YOLOV8L.png"
        img2_path = "GEN2_with_YOLO26m_SAHI.png"
        
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            # Calculate image dimensions
            img_width = 4.5*inch
            
            # YOLOv8l image
            elements.append(Paragraph("YOLOv8l Detection:", 
                ParagraphStyle('ImageLabel', parent=styles['Normal'], 
                              fontSize=10, alignment=TA_CENTER, textColor=colors.grey)))
            img1 = Image(img1_path, width=img_width, height=img_width*0.75)
            elements.append(img1)
            elements.append(Spacer(1, 0.15*inch))
            
            # YOLO26m + SAHI image
            elements.append(Paragraph("YOLO26m + SAHI Detection:", 
                ParagraphStyle('ImageLabel', parent=styles['Normal'], 
                              fontSize=10, alignment=TA_CENTER, textColor=colors.grey)))
            img2 = Image(img2_path, width=img_width, height=img_width*0.75)
            elements.append(img2)
            elements.append(Spacer(1, 0.2*inch))
    except Exception as e:
        elements.append(Paragraph(f"[GEN2 comparison images would appear here]", body_style))
    
    elements.append(PageBreak())
    
    # Comparison 2: HP Dataset
    elements.append(Paragraph("<b>Comparison 2: HP Dataset</b>", heading2_style))
    elements.append(Paragraph(
        "This comparison showcases performance on the HP surveillance dataset. "
        "The YOLO26m + SAHI combination provides significantly better coverage and accuracy.",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    
    # Add HP comparison images
    try:
        img3_path = "HP_OVERVIEW_WITH_YOLOV8L.png"
        img4_path = "HP_WITH_SAHI_YOLO26m.jpg"
        
        if os.path.exists(img3_path) and os.path.exists(img4_path):
            # Calculate image dimensions
            img_width = 4.5*inch
            
            # YOLOv8l image
            elements.append(Paragraph("YOLOv8l Detection:", 
                ParagraphStyle('ImageLabel', parent=styles['Normal'], 
                              fontSize=10, alignment=TA_CENTER, textColor=colors.grey)))
            img3 = Image(img3_path, width=img_width, height=img_width*0.75)
            elements.append(img3)
            elements.append(Spacer(1, 0.15*inch))
            
            # YOLO26m + SAHI image
            elements.append(Paragraph("YOLO26m + SAHI Detection:", 
                ParagraphStyle('ImageLabel', parent=styles['Normal'], 
                              fontSize=10, alignment=TA_CENTER, textColor=colors.grey)))
            img4 = Image(img4_path, width=img_width, height=img_width*0.75)
            elements.append(img4)
            elements.append(Spacer(1, 0.2*inch))
    except Exception as e:
        elements.append(Paragraph(f"[HP comparison images would appear here]", body_style))
    
    elements.append(PageBreak())
    
    # YOLO26m Technical Capabilities
    elements.append(Paragraph("🧠 YOLO26m — Technical Capabilities Delivered", heading1_style))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("<b>1. Small-Object Focused Learning</b>", heading2_style))
    elements.append(Paragraph(
        "Improved label assignment and training strategy increases sensitivity to "
        "<b>tiny and distant persons</b>.",
        body_style
    ))
    
    elements.append(Paragraph("<b>2. End-to-End Detection (No NMS Dependency)</b>", heading2_style))
    elements.append(Paragraph(
        "Direct predictions reduce box suppression and overlapping miss detections. "
        "Result → <b>Better crowded-scene reliability</b>",
        body_style
    ))
    
    elements.append(Paragraph("<b>3. Faster Inference Pipeline</b>", heading2_style))
    elements.append(Paragraph(
        "Optimized architecture enables lower latency, higher FPS, and smooth CCTV processing.",
        body_style
    ))
    
    elements.append(Paragraph("<b>4. Robust Feature Extraction</b>", heading2_style))
    elements.append(Paragraph(
        "Enhanced backbone improves occlusion handling, complex background separation, "
        "and machinery/person differentiation.",
        body_style
    ))
    
    elements.append(Paragraph("<b>5. Efficient Compute Usage</b>", heading2_style))
    elements.append(Paragraph(
        "Medium variant provides near-large accuracy with lower GPU/CPU consumption. "
        "Ideal for edge/industrial deployments.",
        body_style
    ))
    
    elements.append(PageBreak())
    
    # SAHI Inference Strategy
    elements.append(Paragraph("✂️ SAHI — Inference Strategy Enhancements", heading1_style))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("<b>1. Image Slicing Mechanism</b>", heading2_style))
    elements.append(Paragraph(
        "Large frames are split into smaller tiles before detection.",
        body_style
    ))
    
    elements.append(Paragraph("<b>2. Enhanced Pixel Visibility</b>", heading2_style))
    elements.append(Paragraph(
        "Small persons appear larger inside each tile → easier detection.",
        body_style
    ))
    
    elements.append(Paragraph("<b>3. Higher Recall Without Retraining</b>", heading2_style))
    elements.append(Paragraph(
        "Improves detection using the same model weights.",
        body_style
    ))
    
    elements.append(Paragraph("<b>4. Scalable to High Resolution (1080p/4K)</b>", heading2_style))
    elements.append(Paragraph(
        "Maintains detail without downscaling.",
        body_style
    ))
    
    elements.append(Paragraph("<b>5. Plug-and-Play Integration</b>", heading2_style))
    elements.append(Paragraph(
        "Works directly with YOLO detectors without architectural changes.",
        body_style
    ))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Detection Pipeline
    elements.append(Paragraph("⚙️ How the Combined System Works", heading1_style))
    elements.append(Spacer(1, 0.1*inch))
    
    pipeline_text = """
    <b>Detection Pipeline:</b><br/><br/>
    Video Stream<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
    Frame Extraction<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
    SAHI → Slice into tiles<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
    YOLO26m inference per tile<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
    Box fusion &amp; merge<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
    Final detections<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
    Monitoring / Alerts / Analytics
    """
    
    elements.append(Paragraph(
        pipeline_text,
        ParagraphStyle('Pipeline', parent=styles['Code'], 
                      fontSize=10, leftIndent=20, 
                      backColor=HexColor('#f5f5f5'),
                      borderPadding=10)
    ))
    
    elements.append(PageBreak())
    
    # Performance Comparison Table
    elements.append(Paragraph("📊 Performance Comparison", heading1_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Create comparison table
    data = [
        ['Metric', 'YOLOv8l', 'YOLO26m + SAHI'],
        ['Small person detection', 'Medium', 'High'],
        ['Far-distance recall', 'Low–Medium', 'High'],
        ['Occlusion handling', 'Moderate', 'Strong'],
        ['Crowded scenes', 'Partial misses', 'Stable'],
        ['High-res support', 'Downscale needed', 'Native'],
        ['FPS', 'Good', 'Better'],
        ['Overall reliability', 'Good', 'Excellent']
    ]
    
    table = Table(data, colWidths=[2.5*inch, 1.75*inch, 1.75*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2e7d32')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f0f0f0')]),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Final Outcome
    elements.append(Paragraph("✅ Final Outcome", heading1_style))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("<b>Achievements:</b>", body_style))
    elements.append(Paragraph("• Increased person detection recall", bullet_style))
    elements.append(Paragraph("• Reduced missed detections", bullet_style))
    elements.append(Paragraph("• Improved small & distant object recognition", bullet_style))
    elements.append(Paragraph("• Stable performance in dense industrial scenes", bullet_style))
    elements.append(Paragraph("• Real-time edge deployment ready", bullet_style))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Summary Statement
    elements.append(Paragraph("📌 Summary Statement", heading1_style))
    elements.append(Spacer(1, 0.1*inch))
    
    summary_style = ParagraphStyle(
        'Summary',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        leftIndent=20,
        rightIndent=20,
        spaceAfter=12,
        leading=16,
        backColor=HexColor('#e8f5e9'),
        borderPadding=15,
        borderColor=HexColor('#2e7d32'),
        borderWidth=1
    )
    
    elements.append(Paragraph(
        "<i>As part of our detection optimization initiative, we migrated from YOLOv8l to "
        "YOLO26m integrated with SAHI sliced inference. This upgrade significantly improved "
        "small and distant person detection, enhanced robustness in crowded and occluded "
        "manufacturing environments, and delivered higher real-time accuracy without "
        "increasing computational cost.</i>",
        summary_style
    ))
    
    # Build PDF
    doc.build(elements)
    print(f"✅ PDF generated successfully: {pdf_filename}")
    return pdf_filename


if __name__ == "__main__":
    create_yolo26_sahi_pdf()
