#!/usr/bin/env python3
"""
Script to generate a beautiful PDF from the Astec Transcripts.txt file.
Converts <tag> markers to formatted headings and maintains proper spacing.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import simpleSplit
import re
import html


def parse_transcript_file(file_path):
    """
    Parse the transcript file and structure the content.
    Returns a list of sections with their content.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections = []
    current_section = {
        'video_name': '',
        'content': []
    }
    
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this is a video name line
        if 'Name of the Video' in line or 'Name of the video' in line:
            # Save previous section if it has content
            if current_section['video_name'] or current_section['content']:
                sections.append(current_section)
            
            # Start new section
            current_section = {
                'video_name': line,
                'content': []
            }
            i += 1
            continue
        
        # Check if this is a transcript generated line
        if 'Transcript Generated' in line or 'Transcript generated' in line:
            current_section['content'].append(('subtitle', line))
            i += 1
            continue
        
        # Check for time range markers
        if line.startswith('[Start Time:'):
            current_section['content'].append(('time_range', line))
            i += 1
            continue
        
        # Check for tag markers like <general>, <machine_type>, etc.
        if line.startswith('<') and line.endswith('>'):
            # Extract tag name
            tag_name = line[1:-1]
            current_section['content'].append(('heading', tag_name))
            i += 1
            continue
        
        # Check for closing tags and skip them
        if line.startswith('</') and line.endswith('>'):
            i += 1
            continue
        
        # Check for alternative tag format (parentheses)
        if line.startswith('(') and line.endswith(')') and not line.startswith('(**'):
            # Extract tag name
            tag_name = line[1:-1]
            if tag_name in ['machine_type', 'worker_safety', 'operations', 'anomalies', 'operation', 'anomaly']:
                current_section['content'].append(('heading', tag_name))
                i += 1
                continue
        
        # Regular content line
        if line:
            current_section['content'].append(('text', line))
        else:
            # Empty line for spacing
            current_section['content'].append(('space', ''))
        
        i += 1
    
    # Add the last section
    if current_section['video_name'] or current_section['content']:
        sections.append(current_section)
    
    return sections


def create_pdf(input_file, output_file):
    """
    Generate a beautiful PDF from the transcript file.
    """
    # Create PDF document
    doc = SimpleDocTemplate(
        output_file,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor='#1a1a1a',
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    # Custom video name style
    video_name_style = ParagraphStyle(
        'VideoName',
        parent=styles['Heading1'],
        fontSize=16,
        textColor='#2c3e50',
        spaceAfter=12,
        spaceBefore=24,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    # Custom subtitle style
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor='#555555',
        spaceAfter=12,
        alignment=TA_LEFT,
        fontName='Helvetica-Oblique'
    )
    
    # Custom heading style (for tags)
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor='#34495e',
        spaceAfter=8,
        spaceBefore=12,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold',
        leftIndent=20
    )
    
    # Custom time range style
    time_range_style = ParagraphStyle(
        'TimeRange',
        parent=styles['Normal'],
        fontSize=11,
        textColor='#7f8c8d',
        spaceAfter=8,
        spaceBefore=8,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    # Custom body text style
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        textColor='#2c3e50',
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        leftIndent=40,
        leading=14
    )
    
    # Add main title
    story.append(Paragraph("Astec Transcripts", title_style))
    story.append(Spacer(1, 0.3 * inch))
    
    # Parse the transcript file
    sections = parse_transcript_file(input_file)
    
    # Process each section
    for section_idx, section in enumerate(sections):
        # Add video name
        if section['video_name']:
            escaped_name = html.escape(section['video_name'])
            story.append(Paragraph(escaped_name, video_name_style))
        
        # Add content
        for item_type, item_content in section['content']:
            if item_type == 'subtitle':
                escaped_content = html.escape(item_content)
                story.append(Paragraph(escaped_content, subtitle_style))
            
            elif item_type == 'time_range':
                escaped_content = html.escape(item_content)
                story.append(Paragraph(escaped_content, time_range_style))
            
            elif item_type == 'heading':
                # Format heading nicely (capitalize, replace underscores)
                formatted_heading = item_content.replace('_', ' ').title()
                escaped_heading = html.escape(formatted_heading)
                story.append(Paragraph(f"<b>{escaped_heading}</b>", heading_style))
            
            elif item_type == 'text':
                escaped_content = html.escape(item_content)
                story.append(Paragraph(escaped_content, body_style))
            
            elif item_type == 'space':
                story.append(Spacer(1, 0.1 * inch))
        
        # Add spacing between video sections (but not after the last one)
        if section_idx < len(sections) - 1:
            story.append(Spacer(1, 0.4 * inch))
    
    # Build PDF
    doc.build(story)
    print(f"PDF generated successfully: {output_file}")


def main():
    """Main function to generate the PDF."""
    input_file = "/home/runner/work/Testing/Testing/Astec Transcripts.txt"
    output_file = "/home/runner/work/Testing/Testing/Astec Transcripts.pdf"
    
    try:
        create_pdf(input_file, output_file)
        print(f"\n✅ Successfully generated PDF: {output_file}")
    except Exception as e:
        print(f"\n❌ Error generating PDF: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
