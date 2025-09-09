# 🚨 PATCH BEFORE IMPORTING REPORTLAB 🚨
import hashlib

# Save original md5
_real_md5 = hashlib.md5

# Wrap it to ignore 'usedforsecurity'
def md5(*args, **kwargs):
    if 'usedforsecurity' in kwargs:
        kwargs.pop('usedforsecurity')
    return _real_md5(*args, **kwargs)

# Replace in hashlib
hashlib.md5 = md5

# ✅ NOW import reportlab
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import calendar


def create_calendar_pdf(year, month, filename="calendar.pdf"):
    # Get month calendar as matrix
    cal = calendar.monthcalendar(year, month)
    month_name = calendar.month_name[month]

    # Create PDF
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2.0, height - 50, f"{month_name} {year}")

    # Table header + calendar days
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    table_data = [weekdays]

    # Add each week
    for week in cal:
        table_data.append([str(day) if day != 0 else "" for day in week])

    # Create table
    table = Table(table_data, colWidths=70, rowHeights=40)

    # Add style
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ])

    # Highlight weekends (Sat=5, Sun=6)
    for i, week in enumerate(cal, start=1):
        for j in [5, 6]:
            if week[j] != 0:
                style.add('BACKGROUND', (j, i), (j, i), colors.lightpink)

    table.setStyle(style)

    # Draw table
    table.wrapOn(c, width, height)
    table.drawOn(c, (width - 7*70) / 2, height - 300)

    # Save
    c.save()
    print(f"✅ Calendar saved as {filename}")


# Example usage
create_calendar_pdf(2025, 5, "may_2025_calendar.pdf")
