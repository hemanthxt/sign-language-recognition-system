"""
Convert Project Documentation from Markdown to PDF
"""
import os
import sys

def convert_markdown_to_pdf():
    """Convert the markdown documentation to PDF"""
    
    input_file = "Project_Documentation.md"
    output_file = "Sign_Language_Detector_Documentation.pdf"
    
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found!")
        sys.exit(1)
    
    print("Attempting to convert markdown to PDF...")
    print("Trying different methods...\n")
    
    # Method 1: Try markdown-pdf
    try:
        import markdown
        from weasyprint import HTML, CSS
        
        print("Method 1: Using markdown + weasyprint")
        
        # Read markdown file
        with open(input_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'codehilite'])
        
        # Add CSS styling
        html_with_style = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 40px auto;
                    padding: 20px;
                    color: #333;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    border-bottom: 2px solid #95a5a6;
                    padding-bottom: 8px;
                    margin-top: 30px;
                }}
                h3 {{
                    color: #34495e;
                    margin-top: 20px;
                }}
                code {{
                    background-color: #f4f4f4;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }}
                pre {{
                    background-color: #f4f4f4;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                ul, ol {{
                    margin-left: 20px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                .warning {{
                    color: #e74c3c;
                }}
                .success {{
                    color: #27ae60;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Convert HTML to PDF
        HTML(string=html_with_style).write_pdf(output_file)
        print(f"✓ SUCCESS! PDF created: {output_file}")
        return True
        
    except ImportError as e:
        print(f"  ✗ weasyprint not available: {e}")
        print("  Install with: pip install markdown weasyprint")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Method 2: Try pdfkit
    try:
        import markdown
        import pdfkit
        
        print("\nMethod 2: Using pdfkit")
        
        # Read markdown file
        with open(input_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        # Convert to PDF
        pdfkit.from_string(html_content, output_file)
        print(f"✓ SUCCESS! PDF created: {output_file}")
        return True
        
    except ImportError:
        print("  ✗ pdfkit not available")
        print("  Install with: pip install pdfkit markdown")
        print("  Also requires wkhtmltopdf: https://wkhtmltopdf.org/downloads.html")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Method 3: Try reportlab (basic)
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        
        print("\nMethod 3: Using reportlab (basic text)")
        
        # Create PDF
        doc = SimpleDocTemplate(output_file, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Add custom styles
        styles.add(ParagraphStyle(name='CustomTitle', 
                                 parent=styles['Heading1'],
                                 fontSize=24,
                                 textColor='#2c3e50',
                                 spaceAfter=30))
        
        # Read markdown file
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.2*inch))
            elif line.startswith('# '):
                story.append(Paragraph(line[2:], styles['CustomTitle']))
            elif line.startswith('## '):
                story.append(Paragraph(line[3:], styles['Heading2']))
            elif line.startswith('### '):
                story.append(Paragraph(line[4:], styles['Heading3']))
            else:
                # Clean markdown formatting for basic display
                line = line.replace('**', '<b>').replace('**', '</b>')
                line = line.replace('`', '<code>').replace('`', '</code>')
                story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
        
        # Build PDF
        doc.build(story)
        print(f"✓ SUCCESS! PDF created: {output_file}")
        print("  Note: Basic formatting (no tables, limited styling)")
        return True
        
    except ImportError:
        print("  ✗ reportlab not available")
        print("  Install with: pip install reportlab")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # If all methods fail
    print("\n" + "="*60)
    print("Could not create PDF automatically.")
    print("="*60)
    print("\nAlternative methods:")
    print("1. Install required libraries:")
    print("   pip install markdown weasyprint")
    print("\n2. Use online converters:")
    print("   - https://www.markdowntopdf.com/")
    print("   - https://md2pdf.netlify.app/")
    print("\n3. Open Project_Documentation.md in:")
    print("   - VS Code: Use 'Markdown PDF' extension")
    print("   - Typora: File -> Export -> PDF")
    print("   - Pandoc: pandoc Project_Documentation.md -o output.pdf")
    print("\nThe markdown file is ready at: Project_Documentation.md")
    return False

if __name__ == "__main__":
    convert_markdown_to_pdf()
