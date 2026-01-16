"""
Simple PDF generator for Sign Language Detector Documentation
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
import re

def create_pdf():
    """Create PDF from Project Documentation"""
    
    input_file = "Project_Documentation.md"
    output_file = "Sign_Language_Detector_Documentation.pdf"
    
    print("Creating PDF documentation...")
    
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
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold'
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=15,
        fontName='Helvetica-Bold'
    )
    
    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=9,
        fontName='Courier',
        textColor=colors.HexColor('#2c3e50'),
        backColor=colors.HexColor('#f4f4f4'),
        borderPadding=5,
        spaceAfter=10
    )
    
    # Read markdown file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # First line is title
    is_first_heading = True
    in_code_block = False
    code_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Handle code blocks
        if line_stripped.startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_lines = []
            else:
                # End of code block
                in_code_block = False
                if code_lines:
                    code_text = '<br/>'.join(code_lines)
                    story.append(Paragraph(code_text, code_style))
                    story.append(Spacer(1, 0.1*inch))
            continue
        
        if in_code_block:
            # Escape special characters
            safe_line = line.rstrip().replace('<', '&lt;').replace('>', '&gt;')
            code_lines.append(safe_line if safe_line else ' ')
            continue
        
        # Skip empty lines
        if not line_stripped:
            story.append(Spacer(1, 0.1*inch))
            continue
        
        # Skip horizontal rules and TOC markers
        if line_stripped.startswith('---') or line_stripped.startswith('## Table'):
            continue
        
        # Clean up markdown formatting
        text = line_stripped
        
        # Remove markdown link syntax but keep text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Bold
        text = re.sub(r'\*\*([^\*]+)\*\*', r'<b>\1</b>', text)
        
        # Italic
        text = re.sub(r'\*([^\*]+)\*', r'<i>\1</i>', text)
        
        # Inline code - escape it first
        def escape_code(match):
            return '<font face="Courier" color="#2c3e50">' + match.group(1).replace('<', '&lt;').replace('>', '&gt;') + '</font>'
        text = re.sub(r'`([^`]+)`', escape_code, text)
        
        # Process by markdown level
        if line_stripped.startswith('# '):
            # Main title
            text = line_stripped[2:]
            if is_first_heading:
                story.append(Paragraph(text, title_style))
                is_first_heading = False
            else:
                story.append(Paragraph(text, h1_style))
                
        elif line_stripped.startswith('## '):
            text = line_stripped[3:]
            story.append(Spacer(1, 0.15*inch))
            story.append(Paragraph(text, h1_style))
            
        elif line_stripped.startswith('### '):
            text = line_stripped[4:]
            story.append(Paragraph(text, h2_style))
            
        elif line_stripped.startswith('#### '):
            text = line_stripped[5:]
            story.append(Paragraph(text, h3_style))
            
        elif line_stripped.startswith(('- ', '* ', '+ ')):
            # List item
            text = '• ' + line_stripped[2:]
            story.append(Paragraph(text, normal_style))
            
        elif re.match(r'^\d+\.', line_stripped):
            # Numbered list
            story.append(Paragraph(text, normal_style))
            
        else:
            # Regular paragraph
            story.append(Paragraph(text, normal_style))
    
    # Build PDF
    try:
        doc.build(story)
        print(f"✅ SUCCESS! PDF created: {output_file}")
        print(f"\n📄 Location: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    create_pdf()
