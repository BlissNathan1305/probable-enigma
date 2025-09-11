#!/usr/bin/env python3
"""
Periodic Table PDF Generator
Creates a professional-looking periodic table in PDF format
"""

from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportpdf.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

class PeriodicTablePDF:
    def __init__(self, filename="periodic_table.pdf"):
        self.filename = filename
        self.doc = SimpleDocTemplate(
            filename,
            pagesize=landscape(A4),
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        self.styles = getSampleStyleSheet()
        self.elements = self._get_periodic_table_data()
        
    def _get_periodic_table_data(self):
        """Return a list of all elements with their properties"""
        return [
            # Format: [Atomic Number, Symbol, Name, Atomic Mass, Group, Period, Category]
            [1, "H", "Hydrogen", "1.008", 1, 1, "Nonmetal"],
            [2, "He", "Helium", "4.0026", 18, 1, "Noble Gas"],
            [3, "Li", "Lithium", "6.94", 1, 2, "Alkali Metal"],
            [4, "Be", "Beryllium", "9.0122", 2, 2, "Alkaline Earth Metal"],
            [5, "B", "Boron", "10.81", 13, 2, "Metalloid"],
            [6, "C", "Carbon", "12.011", 14, 2, "Nonmetal"],
            [7, "N", "Nitrogen", "14.007", 15, 2, "Nonmetal"],
            [8, "O", "Oxygen", "15.999", 16, 2, "Nonmetal"],
            [9, "F", "Fluorine", "18.998", 17, 2, "Halogen"],
            [10, "Ne", "Neon", "20.180", 18, 2, "Noble Gas"],
            [11, "Na", "Sodium", "22.990", 1, 3, "Alkali Metal"],
            [12, "Mg", "Magnesium", "24.305", 2, 3, "Alkaline Earth Metal"],
            [13, "Al", "Aluminum", "26.982", 13, 3, "Post-Transition Metal"],
            [14, "Si", "Silicon", "28.085", 14, 3, "Metalloid"],
            [15, "P", "Phosphorus", "30.974", 15, 3, "Nonmetal"],
            [16, "S", "Sulfur", "32.06", 16, 3, "Nonmetal"],
            [17, "Cl", "Chlorine", "35.45", 17, 3, "Halogen"],
            [18, "Ar", "Argon", "39.948", 18, 3, "Noble Gas"],
            [19, "K", "Potassium", "39.098", 1, 4, "Alkali Metal"],
            [20, "Ca", "Calcium", "40.078", 2, 4, "Alkaline Earth Metal"],
            # Add more elements as needed...
        ]
    
    def _get_category_color(self, category):
        """Return color based on element category"""
        color_map = {
            "Alkali Metal": colors.HexColor("#FF6666"),
            "Alkaline Earth Metal": colors.HexColor("#FFDEAD"),
            "Transition Metal": colors.HexColor("#FFC0CB"),
            "Post-Transition Metal": colors.HexColor("#CCCCCC"),
            "Metalloid": colors.HexColor("#99CC99"),
            "Nonmetal": colors.HexColor("#CCFFCC"),
            "Halogen": colors.HexColor("#FFFF99"),
            "Noble Gas": colors.HexColor("#CCFFFF"),
            "Lanthanide": colors.HexColor("#FFBFFF"),
            "Actinide": colors.HexColor("#FF99CC"),
        }
        return color_map.get(category, colors.white)
    
    def _create_element_cell(self, element):
        """Create a formatted cell for an element"""
        if not element:
            return [""]  # Empty cell
        
        atomic_number, symbol, name, mass, group, period, category = element
        
        cell_content = [
            [Paragraph(f"<b>{atomic_number}</b>", self.styles["Normal"])],
            [Paragraph(f"<b><font size=14>{symbol}</font></b>", self.styles["Normal"])],
            [Paragraph(f"<font size=6>{name}</font>", self.styles["Normal"])],
            [Paragraph(f"<font size=6>{mass}</font>", self.styles["Normal"])]
        ]
        
        return cell_content
    
    def create_periodic_table(self):
        """Create the periodic table layout"""
        # Create a 18x10 grid (groups x periods + lanthanides/actinides)
        table_data = [[None for _ in range(18)] for _ in range(10)]
        
        # Place elements in their correct positions
        for element in self.elements:
            atomic_number, symbol, name, mass, group, period, category = element
            if 1 <= period <= 7 and 1 <= group <= 18:
                table_data[period-1][group-1] = element
        
        # Convert to table format
        formatted_table = []
        for row_idx, row in enumerate(table_data):
            table_row = []
            for col_idx, element in enumerate(row):
                if element:
                    cell_data = self._create_element_cell(element)
                    category = element[6]
                    bg_color = self._get_category_color(category)
                    
                    # Create a mini-table for each element cell
                    element_table = Table(cell_data, colWidths=[0.6*cm], rowHeights=[0.3*cm, 0.5*cm, 0.3*cm, 0.3*cm])
                    element_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), bg_color),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('FONTSIZE', (0, 0), (-1, -1), 8),
                        ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
                        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                    ]))
                    table_row.append(element_table)
                else:
                    table_row.append("")
            formatted_table.append(table_row)
        
        return formatted_table
    
    def generate_pdf(self):
        """Generate the complete PDF document"""
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        story.append(Paragraph("Periodic Table of Elements", title_style))
        
        # Create the periodic table
        periodic_table_data = self.create_periodic_table()
        
        # Main table settings
        col_widths = [0.8*cm] * 18
        row_heights = [1.5*cm] * len(periodic_table_data)
        
        main_table = Table(periodic_table_data, colWidths=col_widths, rowHeights=row_heights)
        
        # Style the main table
        main_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        
        story.append(main_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Legend
        legend_style = ParagraphStyle(
            'LegendTitle',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=10,
            alignment=TA_CENTER
        )
        story.append(Paragraph("Element Categories Legend", legend_style))
        
        # Build the PDF
        self.doc.build(story)
        
        print(f"PDF generated successfully: {self.filename}")

def main():
    """Main function to generate the periodic table PDF"""
    try:
        # Create PDF generator instance
        pdf_generator = PeriodicTablePDF("periodic_table.pdf")
        
        # Generate the PDF
        pdf_generator.generate_pdf()
        
        # Open the PDF if possible
        if os.name == 'nt':  # Windows
            os.startfile("periodic_table.pdf")
        elif os.name == 'posix':  # macOS, Linux
            os.system("open periodic_table.pdf 2>/dev/null || xdg-open periodic_table.pdf")
            
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print("Make sure you have ReportLab installed: pip install reportlab")

if __name__ == "__main__":
    main()
