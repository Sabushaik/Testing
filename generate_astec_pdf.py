"""
Script to generate a PDF with Astec video information in a tabular format.
"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def create_astec_pdf():
    """Create a professionally formatted PDF with Astec video information."""
    
    # Video data
    video_data = [
        ("Bay6_fisheye.mp4", "https://spectra-manufacturing-output.s3.amazonaws.com/astec/Bay6_fisheye_annotated.mp4?AWSAccessKeyId=AKIA467DTTIJZCFRBONJ&Signature=EGy6QaRfeLi2fGHn3Qp51S8HmO4%3D&Expires=1769523072"),
        ("Bay6.mp4", "https://spectra-manufacturing-output.s3.amazonaws.com/astec/Bay6_annotated.mp4?AWSAccessKeyId=AKIA467DTTIJZCFRBONJ&Signature=2YkE%2F7i54VWonxQk1GnshwFTHtA%3D&Expires=1769523296"),
        ("Break_Press_Video.mp4", "https://spectra-manufacturing-output.s3.amazonaws.com/astec/Break_Press_Video_annotated.mp4?AWSAccessKeyId=AKIA467DTTIJZCFRBONJ&Signature=FpRMON5fCaU2RKKaVWuW1G8RM6Q%3D&Expires=1769523511"),
        ("Break_Room_Exit.mp4", "https://spectra-manufacturing-output.s3.amazonaws.com/astec/Break_Room_Exit_annotated.mp4?AWSAccessKeyId=AKIA467DTTIJZCFRBONJ&Signature=e18dLe4HrenB1d45DwXJNpoyYyQ%3D&Expires=1769523702"),
        ("GEN_VIDEO_2(3).mp4", "https://spectra-manufacturing-output.s3.amazonaws.com/astec/GEN_VIDEO_2%283%29_annotated.mp4?AWSAccessKeyId=AKIA467DTTIJZCFRBONJ&Signature=CY3zfixMCxE1YcaEe991MQjnmIU%3D&Expires=1769522904"),
        ("GEN_VIDEO_2(2).mp4", "https://spectra-manufacturing-output.s3.amazonaws.com/astec/GEN_VIDEO_2%282%29_annotated.mp4?AWSAccessKeyId=AKIA467DTTIJZCFRBONJ&Signature=lDUGwFE45MkOyLGOsidwRnRm36g%3D&Expires=1769522504"),
        ("GEN_VIDEO_2.mp4", "https://spectra-manufacturing-output.s3.amazonaws.com/astec/GEN_VIDEO_2_annotated.mp4?AWSAccessKeyId=AKIA467DTTIJZCFRBONJ&Signature=368REz9PG3hZxVWUzTw56p1%2Bu00%3D&Expires=1769501630"),
        ("GEN_VIDEO_1.mp4", "https://spectra-manufacturing-output.s3.amazonaws.com/astec/GEN_VIDEO_1_annotated.mp4?AWSAccessKeyId=AKIA467DTTIJZCFRBONJ&Signature=ZY8jRYlnBHGRYgPEM%2BSp9gZncnQ%3D&Expires=1769524628"),
        ("HP_Overview_2.mp4", "https://spectra-manufacturing-output.s3.amazonaws.com/astec/HP_Overview_2_annotated.mp4?AWSAccessKeyId=AKIA467DTTIJZCFRBONJ&Signature=%2BeosvFQKg6b%2F4pHQeMm3bEqlmXc%3D&Expires=1769524331"),
        ("HP_Overview_1.mp4", "https://spectra-manufacturing-output.s3.amazonaws.com/astec/HP_Overview_1_annotated.mp4?AWSAccessKeyId=AKIA467DTTIJZCFRBONJ&Signature=nGNnkfcLcJKjBJlv6dCGEzBPH3g%3D&Expires=1769524025"),
    ]
    
    # Create PDF
    pdf_filename = "Astec Annotated.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1a5490'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    # Add title
    title = Paragraph("Astec videos on the Manufacturing Pipeline with Nova pro model", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2*inch))
    
    # Prepare table data
    table_data = [['Video Name', 'Annotated Presigned URI']]
    
    # Style for table cells
    cell_style = ParagraphStyle(
        'CellStyle',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.black,
        alignment=TA_LEFT
    )
    
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        textColor=colors.white,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    # Create header row with styled paragraphs
    table_data[0] = [
        Paragraph('Video Name', header_style),
        Paragraph('Annotated Presigned URI', header_style)
    ]
    
    # Add data rows with hyperlinks
    for video_name, url in video_data:
        # Create hyperlink with the video name as the link text
        link_text = f'<a href="{url}" color="blue"><u>{video_name}</u></a>'
        
        table_data.append([
            Paragraph(video_name, cell_style),
            Paragraph(link_text, cell_style)
        ])
    
    # Create table
    table = Table(table_data, colWidths=[2.5*inch, 4.5*inch])
    
    # Style the table
    table.setStyle(TableStyle([
        # Header row styling
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        
        # Data rows styling
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('ALIGN', (1, 1), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('LEFTPADDING', (0, 1), (-1, -1), 8),
        ('RIGHTPADDING', (0, 1), (-1, -1), 8),
        
        # Grid and borders
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#333333')),
        ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#1a5490')),
        
        # Alternating row colors for better readability
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        
        # Vertical alignment
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(table)
    
    # Build PDF
    doc.build(elements)
    print(f"PDF generated successfully: {pdf_filename}")


if __name__ == "__main__":
    create_astec_pdf()
